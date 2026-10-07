//! Personal baseline — the asymmetric EMA of a rolling mean and abs-deviation
//! that ecore maintains per metric, with step size that anneals by baseline age.
//! Port of the item-dependent long-term update ([`update_lt_mean_and_dev`]).
//!
//! Values are stored as **real × 8** fixed-point (i16 in ecore; we keep i32 for
//! headroom). [`Baseline::update`] is the earlier, item-independent form of the EMA update:
//! it equals [`update_lt_mean_and_dev`] for `n < 4` (no clipping, no clamps) and differs
//! from `n >= 4` on because it has no outlier clipping and no per-item clamp tables.
//! [`update_lt_mean_and_dev`] is the item-dependent update (outlier clipping toward the mean, the
//! three age bands, and the per-item bound tables).

/// i16 "no data" sentinel used throughout ecore baselines.
pub const INVALID: i32 = 0x7eb6;

/// Saved-record selection before a long-term baseline update.
pub mod history;

/// Initial sleep-duration mean, in minutes multiplied by eight.
///
/// Version 7.26.0 of the reference implementation.
/// Feature 27 selects the newer age curve; the caller supplies the actual
/// feature setting. The native operation rounds minutes *before* scaling.
/// This does not initialize the other eight baseline items or a recovery state.
pub fn sleep_time_initial_mean_x8(age: u32, feature_27: bool) -> i32 {
    let minutes = if feature_27 {
        (age as f64).mul_add(-0.87, 468.0)
    } else {
        (age as f64).mul_add(-1.77, 486.0) + 20.0
    };
    (minutes.round() as i32).wrapping_shl(3)
}

/// Sleep-balance mean's lower limit, in minutes multiplied by eight.
///
/// Version 7.26.0 of the reference implementation.
/// Unlike the initial mean, this truncates *after* scaling. The native setter
/// stores the result as an `i16`; narrow explicitly when constructing that
/// table. This is the sleep-balance item (index 1), not the previous-night item.
pub fn sleep_time_acceptable_min_x8(age: u32, feature_27: bool) -> i32 {
    let minutes = if feature_27 {
        (age as f64).mul_add(-0.87, 408.0)
    } else {
        (age as f64).mul_add(-1.77, 486.0)
    };
    (minutes * 8.0) as i32
}

/// A rolling personal baseline: `mean` and abs-`deviation`, both ×8 fixed-point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Baseline {
    pub mean_x8: i32,
    pub dev_x8: i32,
}

/// Arithmetic shift right with round-toward-zero bias, matching the decompiled
/// `(t < 0) ? (t + (2^s - 1)) >> s : t >> s`.
fn ashr_round(t: i32, shift: u32) -> i32 {
    let adj = if t < 0 { t + ((1 << shift) - 1) } else { t };
    adj >> shift
}

impl Baseline {
    pub fn new() -> Self {
        Baseline::default()
    }

    /// Update with a new (unscaled) sample given the baseline age in days. The
    /// step size anneals across three age bands (<4, 4–14, >14 days): the warm-up
    /// is fast (mean gain 1/2) and settles to ~1/32 once the baseline is mature.
    pub fn update(&mut self, sample: i32, age_days: u32) {
        let sample_x8 = sample << 3;
        let delta = sample_x8 - self.mean_x8;

        // --- mean step ---
        if age_days > 14 {
            let bias = if delta != 0 && self.mean_x8 <= sample_x8 {
                16
            } else {
                -16
            };
            self.mean_x8 += ashr_round(delta + bias, 5);
        } else if age_days >= 4 {
            let bias = if delta > 0 { 4 } else { -4 };
            self.mean_x8 += ashr_round(delta + bias, 3);
        } else {
            let t = if delta > 0 { delta + 1 } else { delta - 1 };
            self.mean_x8 += ashr_round(t, 1);
        }

        // --- deviation step (target = |sample - new mean|) ---
        let absd = (sample_x8 - self.mean_x8).abs();
        let (mag, shift) = if age_days > 14 {
            (32, 6)
        } else if age_days >= 4 {
            (8, 4)
        } else {
            (4, 3)
        };
        let bias2 = if absd != self.dev_x8 && self.dev_x8 <= absd {
            mag
        } else {
            -mag
        };
        self.dev_x8 += ashr_round((absd - self.dev_x8) + bias2, shift);
    }

    /// Mean in real units.
    pub fn mean(&self) -> f64 {
        self.mean_x8 as f64 / 8.0
    }
    /// Abs-deviation in real units.
    pub fn deviation(&self) -> f64 {
        self.dev_x8 as f64 / 8.0
    }
    /// Normalized deviation of `sample` from the mean (readiness contributors are
    /// built on this). `None` if the deviation hasn't accumulated yet.
    pub fn z(&self, sample: f64) -> Option<f64> {
        if self.dev_x8 == 0 {
            None
        } else {
            Some((sample - self.mean()) / self.deviation())
        }
    }
}

/// Largest decoded 16-bit baseline value.
pub const M16: i32 = 0x7eb6;
/// Upper bound of the stored mean per readiness baseline item (8x scale). Items 0..8 are
/// sleep score, sleep time, resting HR, resting HR time, equivalent distance, vigorous
/// minutes, sedentary minutes, temperature, HRV.
pub const MEAN_MAX: [i32; 9] = [760, 4320, 720, 3360, 30000, 320, 4320, 29920, 6000];
/// Lower bound of the stored mean per item (8x scale). Entry 0 is set
/// by feature 9 and entry 1 by [`mean_min_table`] at every `recovery_run`.
pub const MEAN_MIN_DEFAULT: [i32; 9] = [680, 3250, 320, 2400, 4000, 80, 2400, 26400, 3040];
/// Lower bound of the stored deviation per item (8x scale).
pub const DEV_MIN: [i32; 9] = [40, 200, 24, 400, 2000, 16, 720, 240, 240];
/// Upper bound of the stored deviation per item (8x scale).
pub const DEV_MAX: [i32; 9] = [80, 360, 40, 560, 10000, 240, 960, 400, 480];

/// A stored 16-bit value of 0x8000 and above, or above
/// [`M16`], decodes to [`M16`].
pub fn decode16(v: u16) -> i32 {
    if v >= 0x8000 {
        M16
    } else {
        (v as i32).min(M16)
    }
}

/// The mean lower-bound table at a `recovery_run`. Entry 0 is `prev_night_min * 8` (feature 9,
/// 80 by default); entry 1 is `trunc(8 * (486 - 1.77 j))` as i16, from the sleep-time
/// baseline `j` in minutes.
pub fn mean_min_table(j: u32, prev_night_min: i32) -> [i32; 9] {
    let mut t = MEAN_MIN_DEFAULT;
    t[0] = prev_night_min * 8;
    t[1] = ((((j as f64).mul_add(-1.77, 486.0)) * 8.0) as i64) as i16 as i32;
    t
}

/// Move the stored
/// 16-bit long-term mean and deviation of baseline `item` (0..8) toward `val * 8` after `n`
/// days of baseline age.
///
/// For `n >= 4` the new value is first clipped toward the mean: beyond 2.5 deviations to one
/// deviation, between 1.5 and 2.5 deviations to 1.5 deviations (not for the athlete items 4
/// and 5 when `athletes` is set; those clamp to `0..=M16`). The mean then moves by
/// `(x - mean)/8` for `n < 15` (1/2 for `n < 4`) or `/32` for `n >= 15`; the deviation moves
/// by `(|x - mean'| - dev)/16`, `/8` or `/64` on the same schedule. Each step rounds away
/// from zero through a signed bias term and is truncated to i16. The results are clamped to
/// `mean_min`/[`MEAN_MAX`] and [`DEV_MIN`]/[`DEV_MAX`] (the deviation clamp is skipped for
/// `n < 4`), then to `0..=M16`.
pub fn update_lt_mean_and_dev(
    mean: &mut u16,
    dev: &mut u16,
    item: usize,
    val: i16,
    n: u32,
    athletes: bool,
    mean_min: &[i32; 9],
) {
    let m = decode16(*mean);
    let d = decode16(*dev);
    let mut x = (val as i32) << 3;
    let athl = athletes && (item == 4 || item == 5);
    if athl {
        x = if val < 0 { 0 } else { x.min(M16) };
    } else if n >= 4 {
        let diff = x - m;
        let s = if d != 0 { 1 } else { -1 };
        let t1 = 5 * d + s;
        let th1 = (t1 + (t1 < 0) as i32) >> 1;
        let ad = diff.abs();
        let mut clip = None;
        if ad > th1 {
            clip = Some(d);
        } else {
            let t2 = 3 * d + s;
            let th2 = (t2 + (t2 < 0) as i32) >> 1;
            if ad > th2 {
                clip = Some(th2);
            }
        }
        if let Some(c) = clip {
            x = m + if x <= m { -c } else { c };
        }
    }
    let diff = x - m;
    let i16w = |v: i32| v as i16 as i32;
    let (nm, dd) = if n >= 15 {
        let nm = m + i16w((diff + if diff > 0 { 16 } else { -16 }) / 32);
        let e = (x - nm).abs() - d;
        (nm, i16w((e + if e > 0 { 32 } else { -32 }) / 64))
    } else if n >= 4 {
        let nm = m + i16w((diff + if diff > 0 { 4 } else { -4 }) / 8);
        let e = (x - nm).abs() - d;
        (nm, i16w((e + if e > 0 { 8 } else { -8 }) / 16))
    } else {
        let nm = m + i16w((diff + if diff <= 0 { -1 } else { 1 }) / 2);
        let e = (x - nm).abs() - d;
        (nm, i16w((e + if e > 0 { 4 } else { -4 }) / 8))
    };
    let nd = d + dd;
    let dev_ub = if athl { M16 } else { DEV_MAX[item] };
    let mut t = if dev_ub < nd { dev_ub } else { nd };
    if nd < DEV_MIN[item] {
        t = DEV_MIN[item];
    }
    if n < 4 {
        t = nd;
    }
    let mean_ub = if athl { M16 } else { MEAN_MAX[item] };
    let mean_lb = if athl { 0 } else { mean_min[item] };
    let mut mm = if mean_ub < nm { mean_ub } else { nm };
    if nm < mean_lb {
        mm = mean_lb;
    }
    *mean = mm.clamp(0, M16) as u16;
    *dev = t.clamp(0, M16) as u16;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_matches_item_dependent_form_before_clipping() {
        // For n < 4 the item-dependent update does not clip or clamp the deviation, and with a zero mean
        // lower bound and the item-8 upper bound the mean is not clamped either.
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for _ in 0..500 {
            let mut b = Baseline { mean_x8: 80 * 8, dev_x8: 0 };
            let (mut m, mut d) = (80 * 8u16, 0u16);
            for n in 0..4u32 {
                let x = (40 + rnd() % 60) as i32;
                b.update(x, n);
                update_lt_mean_and_dev(&mut m, &mut d, 8, x as i16, n, false, &[0; 9]);
                assert_eq!((b.mean_x8, b.dev_x8), (m as i32, d as i32));
            }
        }
    }

    #[test]
    fn update_diverges_from_item_dependent_form_with_outliers() {
        // n >= 4: the item-dependent update clips a 5-sigma sample toward the mean; `update` does not.
        let mut b = Baseline { mean_x8: 80 * 8, dev_x8: 4 * 8 };
        let (mut m, mut d) = (80 * 8u16, 4 * 8u16);
        b.update(200, 10);
        update_lt_mean_and_dev(&mut m, &mut d, 8, 200, 10, false, &[0; 9]);
        assert!(b.mean_x8 > m as i32);
    }

    #[test]
    fn warm_up_then_settle() {
        let mut b = Baseline::new();
        b.update(100, 0); // delta 800 -> +400
        assert_eq!(b.mean_x8, 400);
        b.update(100, 0); // delta 400 -> +200
        assert_eq!(b.mean_x8, 600);
        for _ in 0..400 {
            b.update(100, 30); // mature: slow convergence toward 800 (=100*8)
        }
        // settles within the fixed-point deadband (~2 units) of the target
        assert!((b.mean() - 100.0).abs() < 2.5, "{}", b.mean());
    }

    #[test]
    fn current_android_age_curves_round_at_different_steps() {
        // Generated controls recorded from version 7.26.0, both feature states.
        // Columns: age, feature 27, initial mean times 8, acceptable minimum times 8.
        for (age, feature, mean, minimum) in [
            (0, false, 4048, 3888),
            (18, false, 3792, 3633),
            (30, false, 3624, 3463),
            (65, false, 3128, 2967),
            (255, false, 440, 277),
            (0, true, 3744, 3264),
            (18, true, 3616, 3138),
            (30, true, 3536, 3055),
            (65, true, 3288, 2811),
            (255, true, 1968, 1489),
            (1_206_797_259, false, 91_624_048, i32::MIN),
            (1_499_916_368, true, -1_849_479_584, i32::MIN),
        ] {
            assert_eq!(sleep_time_initial_mean_x8(age, feature), mean);
            assert_eq!(sleep_time_acceptable_min_x8(age, feature), minimum);
        }
    }
}
