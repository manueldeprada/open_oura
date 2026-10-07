//! Night summary values that the sleep score does not use, and the resampling of ring events
//! that feeds them.
//!
//! * [`calculate_average_rmssd`]: average RMSSD, non-REM and REM averages.
//! * [`nightly_temperature`] (with a 7-point median): the highest temperature and, fed the
//!   digital temperature series, the highest digital temperature.
//! * [`hr_average_times8`]: average heart rate times 8 and the slot offset.
//! * [`grid`] and [`hrv_items`]: the resampling of ring events.
//!
//! Python port: `tools/sleep_summary.py`; the Rust tests use vectors from
//! `tools/gen_public_vectors_sleep.py`.

use super::sleep_inputs::epoch_count;

/// Average RMSSD (all slots, non-REM slots, REM slots).
pub type RmssdAverages = (u32, u32, u32);

fn div_round(total: u32, n: u32) -> u32 {
    (total + (n >> 1)) / n
}

/// Argument of [`calculate_average_rmssd`]: the end time of the HR/HRV series (first slot time in
/// ms / 1000 plus `n_slots * 300`) minus the bedtime start in seconds, wrapped to [-150, 150) s
/// (0 when not positive).
pub fn slot_offset(first_ms: i64, n_slots: usize, bed_start_s: i64) -> i32 {
    let d = first_ms.div_euclid(1000) + n_slots as i64 * 300 - bed_start_s;
    if d >= 1 {
        ((d + 150) % 300 - 150) as i32
    } else {
        0
    }
}

/// Stage per 30 s table row for [`calculate_average_rmssd`]: rows start as
/// awake (4) and the phases fill them in order ([`super::sleep_inputs::place_phases`]'s rule).
pub fn stage_rows(codes: &[u8], phase_first_ms: i64, start_ms: i64, end_ms: i64) -> Vec<u8> {
    let (start_s, end_s) = (start_ms.div_euclid(1000), end_ms.div_euclid(1000));
    let n = epoch_count(start_s, end_s);
    let t0 = start_s - start_s.rem_euclid(30);
    let p0 = phase_first_ms.div_euclid(1000);
    let mut rows = vec![4u8; n];
    let mut j = 0usize;
    for (k, &code) in codes.iter().enumerate() {
        if j >= n {
            break;
        }
        if (p0 + 30 * k as i64 + 15).div_euclid(30) * 30 >= t0 + 30 * j as i64 {
            rows[j] = code;
            j += 1;
        }
    }
    rows
}

/// `calculate_average_rmssd`.
///
/// `stages` has one entry per table row (1 deep, 2 light, 3 REM, 4 awake, 0 invalid), `rmssd` one
/// byte per 5-minute slot. Slot `i` looks at the nine epochs `10 i ..= 10 i + 8`, skipping those
/// whose index is at most `!v` with `v = 1 - (15 - offset) / 30` for `offset < 1` and
/// `(offset + 15) / 30` otherwise (unsigned division). A slot with an awake epoch or a zero RMSSD
/// is left out. Otherwise its RMSSD adds to the total and, with six or more deep or light epochs,
/// to the non-REM sum; else with six or more REM epochs, to the REM sum. Each average is
/// `(sum + n / 2) / n`, 0 without a slot. The scan ends after the slot in which the stage string
/// ends.
pub fn calculate_average_rmssd(
    stages: &[u8],
    rmssd: &[u8],
    n_slots: usize,
    offset: i32,
) -> RmssdAverages {
    let v: i32 = if offset < 1 {
        1i32.wrapping_sub(((15i32.wrapping_sub(offset)) as u32 / 30) as i32)
    } else {
        ((offset as u32 + 15) / 30) as i32
    };
    let limit = !(v as i64);
    let at = |k: i64| -> Option<u8> {
        if k >= 0 && (k as usize) < stages.len() {
            Some(stages[k as usize])
        } else {
            None
        }
    };
    let (mut tot, mut nrem, mut rem) = ((0u32, 0u32), (0u32, 0u32), (0u32, 0u32));
    let mut i = 0usize;
    while i != n_slots {
        let base = 10 * i as i64;
        let (mut asleep, mut rem_n, mut awake) = (0, 0, 0);
        let mut go = true;
        for j in 0..9 {
            let k = base + j;
            if k <= limit {
                go = true;
                continue;
            }
            match at(k) {
                Some(1) | Some(2) => asleep += 1,
                Some(3) => rem_n += 1,
                Some(4) => awake += 1,
                Some(_) => {}
                None => {
                    go = false;
                    break;
                }
            }
            go = at(k + 1).is_some();
            if !go {
                break;
            }
        }
        if awake == 0 {
            let r = rmssd.get(i).copied().unwrap_or(0) as u32;
            if r != 0 {
                if asleep >= 6 {
                    nrem = (nrem.0 + r, nrem.1 + 1);
                } else if rem_n >= 6 {
                    rem = (rem.0 + r, rem.1 + 1);
                }
                tot = (tot.0 + r, tot.1 + 1);
            }
        }
        i += 1;
        if !go {
            break;
        }
    }
    let f = |a: (u32, u32)| if a.1 > 0 { div_round(a.0, a.1) } else { 0 };
    (f(tot), f(nrem), f(rem))
}

/// Average of the non-zero 5-minute heart rates times 8, rounded half away from zero in float32
///; 0 without one.
pub fn hr_average_times8(hr: &[u8]) -> u32 {
    let xs: Vec<u32> = hr.iter().filter(|&&h| h != 0).map(|&h| h as u32).collect();
    if xs.is_empty() {
        return 0;
    }
    let q = (xs.iter().sum::<u32>() as f32 / xs.len() as f32) * 8.0f32;
    (q + 0.5) as u32
}

fn median7(v: &[u16; 7]) -> u16 {
    let mut s = *v;
    s.sort_unstable();
    s[3]
}

/// The night's highest temperature in centidegrees from
/// the u16 centidegrees of the 60 s windows (0 for a missing window).
///
/// Each sample enters a ring of seven and the median of the ring is the smoothed value. In every
/// block of 30 samples the largest non-zero smoothed value is kept when it exceeds the smallest
/// non-zero one by at most 249. With fewer than four kept values the result is 0, otherwise the
/// second largest kept value (duplicates count).
pub fn nightly_temperature(temps: &[u16]) -> u16 {
    let mut ring = [0u16; 7];
    let mut pos = 0usize;
    let (mut lo, mut hi) = (0xFFFFu16, 0u16);
    let mut kept: Vec<u16> = Vec::new();
    for (i, &t) in temps.iter().enumerate() {
        ring[pos] = t;
        let m = median7(&ring);
        if m != 0 {
            lo = lo.min(m);
            hi = hi.max(m);
        }
        pos = if pos == 6 { 0 } else { pos + 1 };
        if (i + 1) % 30 == 0 {
            if hi >= lo && hi - lo <= 249 && hi != 0 {
                kept.push(hi);
            }
            lo = 0xFFFF;
            hi = 0;
        }
    }
    if kept.len() < 4 {
        return 0;
    }
    let (mut top, mut second) = (0u16, 0u16);
    for &v in &kept {
        if v < top {
            if v >= second {
                second = v;
            }
        } else {
            second = top;
        }
        top = top.max(v);
    }
    second
}

/// Seconds in bed: the two times truncated to seconds, subtracted.
pub fn time_in_bed(start_ms: i64, end_ms: i64) -> i64 {
    end_ms.div_euclid(1000) - start_ms.div_euclid(1000)
}

/// Resampling: one slot per `width_s` seconds from the bedtime
/// start; a slot holds the earliest item `(time ms, value)` that falls into
/// `[slot start, min(slot end, bedtime end))`, `None` when there is none.
pub fn grid<T: Copy>(
    items: &[(i64, T)],
    start_ms: i64,
    end_ms: i64,
    width_s: i64,
) -> Vec<Option<(i64, T)>> {
    let w = width_s * 1000;
    let n = ((end_ms - start_ms + w - 1).div_euclid(w)).max(0) as usize;
    let mut out: Vec<Option<(i64, T)>> = vec![None; n];
    let mut sorted: Vec<(i64, T)> = items.to_vec();
    sorted.sort_by_key(|x| x.0);
    for (t, v) in sorted {
        if t >= start_ms && t < end_ms {
            let i = ((t - start_ms) / w) as usize;
            if i < n && out[i].is_none() {
                out[i] = Some((t, v));
            }
        }
    }
    out
}

/// Per-value `(time ms, (hr, rmssd))` items of the ring's HRV events `(event time s, hr, rmssd)`.
/// Value `i` of an event of `n` values is dated `T - (n - 1 - i) 300 s`; an event whose own time is
/// after the bedtime end is dropped whole.
pub fn hrv_items(events: &[(f64, Vec<u8>, Vec<u8>)], end_ms: i64) -> Vec<(i64, (u8, u8))> {
    let mut out = Vec::new();
    for (t, hr, rm) in events {
        if (*t * 1000.0) as i64 > end_ms {
            continue;
        }
        let n = hr.len();
        for i in 0..n {
            let ts = ((*t - ((n - 1 - i) * 300) as f64) * 1000.0).round() as i64;
            out.push((ts, (hr[i], rm[i])));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    macro_rules! vector_tests {
        () => {
        #[test]
        fn rmssd_matches_reference() {
            for (n, (stages, rmssd, offset, want)) in rmssd_cases().into_iter().enumerate() {
                assert_eq!(
                    calculate_average_rmssd(&stages, &rmssd, rmssd.len(), offset),
                    want,
                    "case {n}"
                );
            }
        }

        #[test]
        fn temperature_matches_reference() {
            for (n, (temps, want)) in temperature_cases().into_iter().enumerate() {
                assert_eq!(nightly_temperature(&temps), want, "case {n}");
            }
        }

        #[test]
        fn public_temperature_api_matches_reference() {
            // `temperature::nightly_temperature` is the signature open_health calls; it must
            // agree with the reference vectors (0 maps to None).
            for (n, (temps, want)) in temperature_cases().into_iter().enumerate() {
                let got = crate::ported::temperature::nightly_temperature(&temps);
                let want = (want != 0).then_some(want as i16);
                assert_eq!(got, want, "case {n}");
            }
        }

        #[test]
        fn night_matches_reference() {
            for (n, c) in night_cases().into_iter().enumerate() {
                let rows = stage_rows(&c.phases, c.phases_first_ms, c.start_ms, c.end_ms);
                let rm: Vec<u8> = c.hrhrv.iter().map(|p| p.1).collect();
                let hr: Vec<u8> = c.hrhrv.iter().map(|p| p.0).collect();
                let off = slot_offset(c.hrhrv_first_ms, hr.len(), c.start_ms.div_euclid(1000));
                assert_eq!(
                    calculate_average_rmssd(&rows, &rm, hr.len(), off),
                    c.rmssd,
                    "case {n}"
                );
                assert_eq!(hr_average_times8(&hr), c.hr8, "case {n}");
                assert_eq!(nightly_temperature(&c.temps), c.temp, "case {n}");
            }
        }
        };
    }

    mod python_port {
        use super::*;
        include!("vectors/sleep_summary.rs");
        vector_tests!();
    }


    #[test]
    fn grid_takes_first_item_per_slot() {
        let items = [
            (1_000i64, 1u8),
            (30_500, 2),
            (31_000, 3),
            (95_000, 4),
            (120_000, 5),
        ];
        let g = grid(&items, 0, 120_000, 30);
        assert_eq!(g.len(), 4);
        assert_eq!(g[0], Some((1_000, 1)));
        assert_eq!(g[1], Some((30_500, 2)));
        assert_eq!(g[2], None);
        assert_eq!(g[3], Some((95_000, 4)));
    }

    #[test]
    fn time_in_bed_truncates_each_end() {
        assert_eq!(time_in_bed(1_792_026_206_300, 1_792_053_226_600), 27020);
        assert_eq!(time_in_bed(1_791_501_060_500, 1_791_531_934_500), 30874);
    }
}
