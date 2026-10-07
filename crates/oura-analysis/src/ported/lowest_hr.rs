//! The night's lowest heart rate and its time,
//! from the ring's 5-minute heart rates (one byte per slot, 0 for a missing slot).
//!
//! [`calculate_lowest_hr`] does not take the plain minimum:
//!
//! 1. Zeros become 255. Q1 and Q3 are the 25th and 75th percentiles of all slots (linear
//!    interpolation in float32) and the outlier fence is `trunc(Q3 + 2 (Q3 - Q1))`.
//! 2. The first three slots (in time order) above the fence (and at most 254) that lie before the
//!    first minimum of the whole night are set to 255.
//! 3. The lowest value is the minimum over slots 6.. (all slots when there are at most 6); `argmin`
//!    is its first slot.
//! 4. Candidates are slots `k` with `7 <= k < argmin` whose value is within 3 bpm of the lowest.
//!    For each candidate in order, the values before it (slots `0..k`) and between it and `argmin`
//!    (slots `k+1..argmin`) are summarized by their mean and sample standard deviation (255
//!    entries ignored; 255 when there are none or, for the deviation, fewer than two). The first
//!    candidate with `mean(before) > mean(after)` and `std(before) > std(after)` is the time slot.
//!    A candidate with an empty or all-255 `after` is kept and the scan continues; one with an
//!    all-255 `before` leaves the choice unchanged. Without a choice the time slot is `argmin`.
//! 5. The time is `slot * interval` seconds.
//!
//! Python port: `tools/lowest_hr.py`; vectors from `tools/gen_public_vectors_sleep.py`.

fn percentile(sorted: &[u8], p: f32) -> f32 {
    let rank = (p / 100.0) * (sorted.len() - 1) as f32;
    let lo = rank.floor();
    let hi = rank.ceil();
    let frac = rank - lo;
    (1.0 - frac).mul_add(
        sorted[lo as usize] as f32,
        frac * sorted[hi as usize] as f32,
    )
}

/// Mean and sample standard deviation of the entries below 255 (255.0 each when undefined).
fn summary(vals: &[u8]) -> (f32, f32) {
    let xs: Vec<f32> = vals
        .iter()
        .filter(|&&v| v < 255)
        .map(|&v| v as f32)
        .collect();
    let mut total = 0f32;
    for &x in &xs {
        total += x;
    }
    let mean = if xs.is_empty() {
        255.0
    } else {
        total / xs.len() as f32
    };
    let mut sq = 0f32;
    for &x in &xs {
        let d = x - mean;
        sq += d * d;
    }
    let std = if xs.len() > 1 {
        (sq / (xs.len() - 1) as f32).sqrt()
    } else {
        255.0
    };
    (mean, std)
}

/// Lowest heart rate (bpm) and its time (seconds from the first slot, `slot * interval`) of the
/// 5-minute heart rates `data`; `None` for an empty input.
pub fn calculate_lowest_hr(data: &[u8], interval: u32) -> Option<(u32, u32)> {
    let n = data.len();
    if n == 0 {
        return None;
    }
    let v1: Vec<u8> = data.iter().map(|&b| if b == 0 { 255 } else { b }).collect();
    let mut sorted = v1.clone();
    sorted.sort_unstable();
    let q1 = percentile(&sorted, 25.0);
    let q3 = percentile(&sorted, 75.0);
    let fence = (q3 + 2.0 * (q3 - q1)) as i32;
    let min1 = *v1.iter().min().unwrap();
    let first_min = v1.iter().position(|&v| v == min1).unwrap();
    let mut v2 = v1.clone();
    let mut masked = 0;
    for (j, &v) in v1.iter().enumerate() {
        if masked == 3 {
            break;
        }
        if j < first_min && v <= 254 && v as i32 > fence {
            v2[j] = 255;
            masked += 1;
        }
    }
    let start = if n > 6 { 6 } else { 0 };
    let lowest = *v2[start..].iter().min().unwrap();
    let argmin = start + v2[start..].iter().position(|&v| v == lowest).unwrap();
    let mut best: Option<usize> = None;
    for k in 7..argmin.min(n) {
        if v2[k] as u32 > lowest as u32 + 3 {
            continue;
        }
        let (before, after) = (&v2[..k], &v2[k + 1..argmin]);
        if after.is_empty() || after.iter().all(|&v| v == 255) {
            best = Some(k);
            continue;
        }
        if before.iter().all(|&v| v == 255) {
            continue;
        }
        let (ma, sa) = summary(before);
        let (mb, sb) = summary(after);
        if ma > mb && sa > sb {
            best = Some(k);
            break;
        }
    }
    Some((lowest as u32, best.unwrap_or(argmin) as u32 * interval))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp() {
        let d: Vec<u8> = (21..=80).rev().collect();
        assert_eq!(calculate_lowest_hr(&d, 300), Some((21, 16800)));
        assert_eq!(calculate_lowest_hr(&[], 300), None);
    }

    macro_rules! vector_tests {
        () => {
            #[test]
            fn matches_reference() {
                for (n, (data, want)) in lowest_hr_cases().into_iter().enumerate() {
                    assert_eq!(calculate_lowest_hr(&data, 300), Some(want), "case {n}");
                }
            }
        };
    }

    mod python_port {
        use super::*;
        include!("vectors/lowest_hr.rs");
        vector_tests!();
    }

}
