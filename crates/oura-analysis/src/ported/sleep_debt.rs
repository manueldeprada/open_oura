//! Sleep debt. Ported from `sleep_debt_calculate` (7.27.0): a linear-decay
//! weighted sum of nightly shortfall (need - actual), clamped, then rounded to a step.
//! Index 0 = most recent night. The weight of night `d` is `1 - d * 0.75 / (n - 1)`, where
//! `n` is the window length including invalid slots: pass 14 slots and mark missing
//! nights `i32::MAX`. Reproduces the reference cumulative value on six logged days with
//! 5 or more valid nights.

/// Tunable config (defaults from `sleep_debt_get_default_config`).
#[derive(Clone, Copy, Debug)]
pub struct SleepDebtConfig {
    /// Fewer valid nights than this: cumulative debt is `i32::MAX`.
    pub min_valid_days: u32,
    pub max_debt_s: i32,
    pub rounding_step_s: i32,
    /// `valid` requires a rounded cumulative debt of at least this many seconds.
    pub min_debt_for_valid_s: i32,
    /// `valid` requires at least this many valid nights.
    pub min_valid_days_for_valid: u32,
    /// `valid` requires this percentage of valid nights with shortfall above `day_threshold_s`.
    pub min_short_night_pct: f32,
    pub day_threshold_s: i32,
}

impl Default for SleepDebtConfig {
    fn default() -> Self {
        Self {
            min_valid_days: 5,
            max_debt_s: 36_000,    // 10 h
            rounding_step_s: 600, // 10 min
            min_debt_for_valid_s: 14_400,
            min_valid_days_for_valid: 7,
            min_short_night_pct: 70.0,
            day_threshold_s: 2700,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct SleepDebt {
    pub debt_s: i32,
    pub recent_shortfall_s: i32,
    pub valid: bool,
}

const SENTINEL: i32 = i32::MAX;

/// Sleep debt from per-day actual and needed sleep (seconds), index 0 = most
/// recent. Days with actual/need of 0 or `i32::MAX` are treated as invalid.
/// `recent_shortfall_s` is need - actual of day 0.
pub fn sleep_debt(actual_s: &[i32], need_s: &[i32], cfg: &SleepDebtConfig) -> SleepDebt {
    let n = actual_s.len().min(need_s.len());
    let invalid = SleepDebt {
        debt_s: SENTINEL,
        recent_shortfall_s: SENTINEL,
        valid: false,
    };
    if n == 0 {
        return invalid;
    }
    let shortfall = |d: usize| -> Option<f64> {
        let (a, need) = (actual_s[d], need_s[d]);
        if a == 0 || a == SENTINEL || need == 0 || need == SENTINEL {
            None
        } else {
            Some(need as f64 - a as f64)
        }
    };
    let decay = if n == 1 { 0.0 } else { 0.75 / (n as f64 - 1.0) };
    let mut debt = 0.0;
    let mut valid = 0u32;
    let mut short_nights = 0u32;
    for d in 0..n {
        if let Some(sf) = shortfall(d) {
            valid += 1;
            debt += (1.0 - decay * d as f64) * sf;
            if sf > cfg.day_threshold_s as f64 {
                short_nights += 1;
            }
        }
    }
    let recent = shortfall(0).map_or(SENTINEL, |v| v.round() as i32);
    if valid < cfg.min_valid_days {
        return SleepDebt {
            debt_s: SENTINEL,
            recent_shortfall_s: recent,
            valid: false,
        };
    }
    let mut d = debt.max(0.0).min(cfg.max_debt_s as f64).round() as i32;
    let step = cfg.rounding_step_s;
    if step != 0 {
        d = (d + step / 2) / step * step;
    }
    let pct = short_nights as f32 / valid as f32 * 100.0;
    let ok = valid > 0
        && d != SENTINEL
        && d >= cfg.min_debt_for_valid_s
        && valid >= cfg.min_valid_days_for_valid
        && pct >= cfg.min_short_night_pct;
    SleepDebt {
        debt_s: d,
        recent_shortfall_s: recent,
        valid: ok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const S: i32 = i32::MAX;

    fn window(v: &[i32]) -> Vec<i32> {
        let mut w = v.to_vec();
        w.resize(14, S);
        w
    }

    /// App log 2026-10-07 and 2026-10-08 (`cumulativeSleepDebt` 13200 and 24600).
    #[test]
    fn matches_app_log() {
        let cfg = SleepDebtConfig::default();
        let need = window(&[26182, 26182, 26182, 26655, 26182]);
        let act = window(&[29250, 19620, 20340, 19770, 27090]);
        let r = sleep_debt(&act, &need, &cfg);
        assert_eq!((r.debt_s, r.recent_shortfall_s), (13200, -3068));
        let need = window(&[26572, 26182, 26182, 26182, 26655, 26182]);
        let act = window(&[14160, 29250, 19620, 20340, 19770, 27090]);
        let r = sleep_debt(&act, &need, &cfg);
        assert_eq!((r.debt_s, r.recent_shortfall_s), (24600, 12412));
    }

    #[test]
    fn fewer_than_five_nights_is_unset() {
        let need = window(&[26182, 26655, 26182]);
        let act = window(&[20340, 19770, 27090]);
        let r = sleep_debt(&act, &need, &SleepDebtConfig::default());
        assert_eq!(r.debt_s, S);
        assert!(!r.valid);
    }
}
