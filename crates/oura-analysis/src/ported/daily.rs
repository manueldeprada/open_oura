//! Per-day orchestration of the daily sleep and readiness scores: the loop of
//! `tools/score_daily_exact.py` that runs after the nights are scored (`tools/daily_scorer.py`).
//!
//! [`DailyScorer`] carries the state kept between days: the long-term baseline (nine means
//! and deviations, starting from [`DEFAULT_MEANS`]/[`DEFAULT_DEVS`] or caller-given values), the per-day history (total sleep seconds,
//! equivalent distance, RMSSD and wear percent; 14 entries, newest first, zero padded) and the
//! day counter. [`DailyScorer::step`] takes one day's [`SleepPeriod`] list and previous-day
//! words. The baseline of a day is [`baseline_calculate`] applied to the baseline of the day
//! before with the night and the previous-day statistics that day was scored with; the first day uses the starting baseline and the temperature item is the
//! digital baseline from the second day on. The day is then scored with
//! [`daily_outputs_stateless`] and pushed onto the history.
//!
//! The caller groups the ring's nights into days, sets `is_main` (the period with the most
//! sleep) and drops periods without sleep; the checks are in
//! [`check_inputs`](super::readiness_inputs::check_inputs).

use super::baseline::{decode16, mean_min_table, update_lt_mean_and_dev};
use super::readiness::Features;
use super::readiness_inputs::{
    daily_outputs_stateless, get_daily_sleep, DailyOutputs, DailySleep, DayData, History,
    SleepPeriod,
};
use super::sri::SriState;

/// Cold-start baseline means (sleep score, sleep time, resting HR, resting HR time, distance,
/// vigorous, sedentary, temperature, HRV; 8x scaled): the values of an empty recovery
/// state. They are generic initial values and do not depend on age or sex. The sleep-time mean starts at 0 and the first
/// updates move it toward the user's nights.
pub const DEFAULT_MEANS: [u16; 9] = [720, 0, 480, 2880, 8000, 160, 3840, 29360, 4400];
/// The matching cold-start baseline deviations.
pub const DEFAULT_DEVS: [u16; 9] = [80, 360, 160, 480, 8000, 64, 1600, 400, 2000];
/// Temperature item from the second day on: 36.3 and 0.5 degrees, times 800.
pub const TEMP_BASELINE: (u16, u16) = (29040, 400);
/// The `sri_calculate(true)` value the Python tool passes.
pub const SRI: u32 = 0x100 | 32;
/// The `j` of [`mean_min_table`] used for the sleep-time lower bound.
pub const MIN_SLEEP_J: u32 = 28;

/// Inputs of one [`baseline_calculate`] call: the night the previous day was scored with and
/// the statistics of the day before that night's day.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BaselineDay {
    pub sleep_score: u32,
    pub time_in_bed: u32,
    pub total_sleep: u32,
    pub lowest_hr: u32,
    pub lowest_hr_time: u32,
    pub eq_distance: u32,
    pub vigorous: u32,
    pub sedentary: u32,
    pub highest_temp: u32,
    pub rmssd: u32,
    pub light: u32,
    pub moderate: u32,
    pub resting: u32,
}

/// Percent of the non-sleep day the ring was worn, from the
/// class minutes (0 when the day is all resting).
pub fn wear_percent(sedentary: u32, light: u32, moderate: u32, vigorous: u32, resting: u32) -> u32 {
    if resting == 0x5A0 {
        return 0;
    }
    let sum = sedentary
        .wrapping_add(light)
        .wrapping_add(moderate)
        .wrapping_add(vigorous);
    let den = 0x5A0u32.wrapping_sub(resting);
    (((sum as f32 * 100.0) / den as f32) + 0.5) as u32
}

fn round_f32(x: f32) -> i32 {
    x.round() as i32 // ties away from zero
}

/// With the athlete flag on, one day's update of the nine
/// long-term baseline items. `day` is the day counter `n` of every item but the temperature,
/// which uses `ring_day`.
pub fn baseline_calculate(
    means: &[u16; 9],
    devs: &[u16; 9],
    day: u32,
    ring_day: u32,
    d: &BaselineDay,
) -> ([u16; 9], [u16; 9]) {
    let mut m = *means;
    let mut v = *devs;
    let mm = mean_min_table(MIN_SLEEP_J, 80);
    let wear = wear_percent(d.sedentary, d.light, d.moderate, d.vigorous, d.resting) as i32;
    fn up(m: &mut [u16; 9], v: &mut [u16; 9], mm: &[i32; 9], item: usize, val: i32, n: u32) {
        update_lt_mean_and_dev(&mut m[item], &mut v[item], item, val as i16, n, true, mm);
    }
    up(&mut m, &mut v, &mm, 0, d.sleep_score as i32, day);
    up(
        &mut m,
        &mut v,
        &mm,
        1,
        round_f32(d.total_sleep as f32 / 60.0),
        day,
    );
    if d.time_in_bed > d.lowest_hr_time && d.lowest_hr != 0 && d.lowest_hr_time != 0 {
        up(&mut m, &mut v, &mm, 2, d.lowest_hr as i32, day);
        up(
            &mut m,
            &mut v,
            &mm,
            3,
            round_f32((d.time_in_bed - d.lowest_hr_time) as f32 / 60.0),
            day,
        );
    }
    let eq8 = ((d.eq_distance >> 3) & 0xFFFF) as i32;
    let mut dist_today = 0i32;
    if eq8 != 0 {
        let mean_eq = decode16(m[4]);
        let tq = (100 - wear) * mean_eq + if mean_eq > 0 { 4 } else { -4 };
        let q = tq / 8;
        dist_today = eq8
            + if tq >= 8 {
                (q + 50) / 100
            } else {
                -((50 - q) / 100)
            };
    }
    up(&mut m, &mut v, &mm, 4, dist_today, day);
    if (dist_today & 0xFFFF) != 0 && wear as f64 / 100.0 >= 0.33 {
        up(&mut m, &mut v, &mm, 5, d.vigorous as i32, day);
        let mut sed_min = d.sedentary as i32;
        if (d.sedentary as i16) > 0 {
            let sm = m[6] as i16 as i32;
            let tq = (100 - wear) * sm + if sm > 0 { 4 } else { -4 };
            let q = tq / 8;
            sed_min += if tq >= 8 {
                (q + 50) / 100
            } else {
                -((50 - q) / 100)
            };
        }
        up(&mut m, &mut v, &mm, 6, sed_min, day);
    }
    let t_mean = m[7] as i16 as f32;
    if d.highest_temp > 0
        && d.highest_temp <= 0x10CB
        && (d.highest_temp as f32 + t_mean * -0.125).abs() < 200.0
    {
        up(&mut m, &mut v, &mm, 7, d.highest_temp as i32, ring_day);
    }
    if d.rmssd != 0 {
        let x = ((d.rmssd as f64).log2() as f32) * 100.0;
        if x > 0.0 {
            up(&mut m, &mut v, &mm, 8, round_f32(x), day);
        }
    }
    (m, v)
}

/// Result of one [`DailyScorer::step`].
#[derive(Clone, Debug)]
pub struct DailyResult {
    /// `get_daily_sleep` on the day's periods (the sleep score the Python tool prints).
    pub sleep: DailySleep,
    /// `None` when the input is rejected ([`check_inputs`](super::readiness_inputs::check_inputs)).
    pub outputs: Option<DailyOutputs>,
    pub readiness_score: Option<i32>,
    /// Activity balance, last day activity, last night sleep, resting HR, resting HR time,
    /// sleep balance, temperature, HRV balance, sleep regularity.
    pub contributors: Option<[i32; 9]>,
}

#[derive(Clone, Debug)]
pub struct DailyScorer {
    pub means: [u16; 9],
    pub devs: [u16; 9],
    /// Temperature item (mean, deviation) from the second day on.
    pub temp_baseline: (u16, u16),
    /// Per earlier day, newest first: (sleep s, distance, RMSSD, wear percent). At most 14.
    pub history: Vec<(u32, u32, u32, u8)>,
    /// Inputs of the baseline update that the previous day's data feeds.
    pub last: Option<BaselineDay>,
    pub day: u32,
    /// Sleep-regularity state; when set, each day's main period updates it and its value replaces
    /// the `sri` argument of `step` (see [`DailyScorer::track_sri`]).
    pub sri_state: Option<SriState>,
}

impl Default for DailyScorer {
    /// Starts from the cold-start baseline ([`DEFAULT_MEANS`], [`DEFAULT_DEVS`]).
    fn default() -> Self {
        Self::with_baseline(&DEFAULT_MEANS, &DEFAULT_DEVS)
    }
}

impl DailyScorer {
    /// Same as [`with_baseline`](Self::with_baseline).
    pub fn new(means: &[u16; 9], devs: &[u16; 9]) -> Self {
        Self::with_baseline(means, devs)
    }

    /// Start from a given day-0 baseline: nine means and nine deviations (8x scaled, in the item
    /// order of [`DEFAULT_MEANS`]), for example one stored from an earlier session.
    ///
    /// There is no separate bootstrap step. [`baseline_calculate`] moves each item by 1/2 of
    /// its gap to the day's value on days 0 to 3, by 1/8 until day 14 and by 1/32 afterwards, so
    /// stepping a scorer built with [`Default`] over a user's first nights adapts the baseline
    /// gradually. The first days still score against the cold-start values (day 0
    /// reads them unchanged); what shortens the transition would be a better day-0 baseline,
    /// which is not computed from the user's data.
    pub fn with_baseline(means: &[u16; 9], devs: &[u16; 9]) -> Self {
        DailyScorer {
            means: *means,
            devs: *devs,
            temp_baseline: TEMP_BASELINE,
            history: Vec::new(),
            last: None,
            day: 0,
            sri_state: None,
        }
    }

    /// Compute the sleep-regularity value from the nights scored so far (one update
    /// per main period) instead of taking the `sri` argument.
    pub fn track_sri(mut self) -> Self {
        self.sri_state = Some(SriState::new());
        self
    }

    /// One day. `periods` are the day's sleep periods (`is_main` set, `total_sleep` > 0),
    /// `prev_day` the 8 previous-day words (unused, distance, non-wear, sedentary, vigorous,
    /// light, moderate, resting), `day_utc` the sleep day's UTC midnight.
    pub fn step(
        &mut self,
        day_utc: i64,
        age: u32,
        periods: &[SleepPeriod],
        prev_day: &[u32; 8],
        chrono: Option<(u8, i32)>,
        sri: u32,
    ) -> DailyResult {
        self.step_with_history(day_utc, age, periods, prev_day, chrono, sri, None)
    }

    /// [`step`](Self::step) with a caller-built history. The history is indexed by date:
    /// entry `k` is the night of day D - 1 - k and the activity of day D - 2 - k, zero where
    /// there is no record. Entries are (sleep s, distance, RMSSD, wear percent), newest
    /// first.
    ///
    /// A day without a sleep record is not stepped. The day counter, the baseline chain and
    /// `last` then skip it: the day counter advances once per stepped day, whatever the
    /// gap between dates, and its history entries stay zero.
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_history(
        &mut self,
        day_utc: i64,
        age: u32,
        periods: &[SleepPeriod],
        prev_day: &[u32; 8],
        chrono: Option<(u8, i32)>,
        sri: u32,
        history: Option<&[(u32, u32, u32, u8)]>,
    ) -> DailyResult {
        // The period with the most sleep, the first on ties.
        let mut main = &periods[0];
        for p in &periods[1..] {
            if p.total_sleep > main.total_sleep {
                main = p;
            }
        }
        let sri = match self.sri_state.as_mut() {
            Some(st) => {
                let m = main;
                let (start, tz) = (m.start, m.tz);
                st.update_history(
                    (day_utc - 86400, 0),
                    (start, tz),
                    (start + m.time_in_bed as i64, tz),
                );
                st.value()
            }
            None => sri,
        };
        let ds = get_daily_sleep(age, 0, 0, periods);
        if let Some(last) = &self.last {
            let (m, v) = baseline_calculate(&self.means, &self.devs, self.day, 0, last);
            self.means = m;
            self.devs = v;
            self.means[7] = self.temp_baseline.0;
            self.devs[7] = self.temp_baseline.1;
        }
        let mut h = [[0u32; 4]; 14];
        for (slot, e) in h.iter_mut().zip(history.unwrap_or(&self.history).iter()) {
            *slot = [e.0, e.1, e.2, e.3 as u32];
        }
        let history = History {
            temp_dev_3: Some([-32768; 3]),
            hot_90: Some(vec![0; 90]),
            sleep_sec_14: Some(std::array::from_fn(|i| h[i][0])),
            distance_14: Some(std::array::from_fn(|i| h[i][1])),
            rmssd_14: Some(std::array::from_fn(|i| h[i][2])),
            wear_14: Some(std::array::from_fn(|i| h[i][3] as u8)),
        };
        let mut bl = [0u8; 0x38];
        for k in 0..9 {
            bl[0x10 + 4 * k..0x12 + 4 * k].copy_from_slice(&self.means[k].to_le_bytes());
            bl[0x12 + 4 * k..0x14 + 4 * k].copy_from_slice(&self.devs[k].to_le_bytes());
        }
        let data = DayData {
            prev_day: Some(*prev_day),
            user: Some((0, age)),
            history,
            baseline: Some(bl),
            day_number: self.day,
            cycle_day_type: 0,
            rest_input: Some((0, 0, 0, day_utc)),
        };
        let feats = Features::default();
        let out = daily_outputs_stateless(day_utc, 0, &data, periods, chrono, &feats, sri);
        // The day's final sleep score.
        let sleep_score = out.as_ref().map_or(ds.score, |o| o.sleep.score);
        let wear = wear_percent(
            prev_day[3],
            prev_day[5],
            prev_day[6],
            prev_day[4],
            prev_day[7],
        );
        self.last = Some(BaselineDay {
            sleep_score,
            time_in_bed: main.time_in_bed,
            total_sleep: main.total_sleep,
            lowest_hr: main.lowest_hr,
            lowest_hr_time: main.lowest_hr_time,
            eq_distance: prev_day[1],
            vigorous: prev_day[4],
            sedentary: prev_day[3],
            highest_temp: main.highest_temp,
            rmssd: main.rmssd,
            light: prev_day[5],
            moderate: prev_day[6],
            resting: prev_day[7],
        });
        self.history
            .insert(0, (main.total_sleep, prev_day[1], main.rmssd, wear as u8));
        self.history.truncate(14);
        self.day += 1;
        let (readiness_score, contributors) = match &out {
            Some(o) => {
                let r = &o.readiness;
                (
                    Some(r.score),
                    Some([
                        r.activity_balance,
                        r.last_day_activity,
                        r.last_night_sleep,
                        r.resting_hr,
                        r.resting_hr_time,
                        r.sleep_balance,
                        r.temperature,
                        r.hrv_balance,
                        r.sleep_regularity,
                    ]),
                )
            }
            None => (None, None),
        };
        DailyResult {
            sleep: ds,
            outputs: out,
            readiness_score,
            contributors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Generated by tools/gen_daily_vectors.py from tools/daily_scorer.py.
    include!("daily_vectors.rs");

    #[test]
    fn default_is_cold_start_baseline() {
        let sc = DailyScorer::default();
        assert_eq!(sc.means, DEFAULT_MEANS);
        assert_eq!(sc.devs, DEFAULT_DEVS);
        let (means, devs) = ([700u16; 9], [60u16; 9]);
        let sc = DailyScorer::with_baseline(&means, &devs);
        assert_eq!(sc.means, means);
        assert_eq!(sc.devs, devs);
    }

    #[test]
    fn explicit_history_equals_internal_history() {
        let seq = daily_sequences().remove(0);
        let mut a = DailyScorer::with_baseline(&seq.means, &seq.devs);
        let mut b = DailyScorer::with_baseline(&seq.means, &seq.devs);
        for day in &seq.days {
            let own: Vec<_> = b.history.clone();
            let ra = a.step(
                day.day_utc,
                day.age,
                &day.periods,
                &day.prev_day,
                day.chrono,
                day.sri,
            );
            let rb = b.step_with_history(
                day.day_utc,
                day.age,
                &day.periods,
                &day.prev_day,
                day.chrono,
                day.sri,
                Some(&own),
            );
            assert_eq!(ra.readiness_score, rb.readiness_score);
            assert_eq!(ra.contributors, rb.contributors);
        }
    }

    #[test]
    fn tracked_sri_equals_explicit_value() {
        let seq = daily_sequences().remove(0);
        let mut a = DailyScorer::with_baseline(&seq.means, &seq.devs).track_sri();
        let mut b = DailyScorer::with_baseline(&seq.means, &seq.devs);
        let mut st = SriState::new();
        for day in &seq.days {
            let m = day.periods.iter().fold(&day.periods[0], |m, p| {
                if p.total_sleep > m.total_sleep {
                    p
                } else {
                    m
                }
            });
            st.update_history(
                (day.day_utc - 86400, 0),
                (m.start, m.tz),
                (m.start + m.time_in_bed as i64, m.tz),
            );
            let ra = a.step(
                day.day_utc,
                day.age,
                &day.periods,
                &day.prev_day,
                day.chrono,
                0,
            );
            let rb = b.step(
                day.day_utc,
                day.age,
                &day.periods,
                &day.prev_day,
                day.chrono,
                st.value(),
            );
            assert_eq!(ra.readiness_score, rb.readiness_score);
            assert_eq!(ra.contributors, rb.contributors);
        }
    }

    #[test]
    fn sequences_match_python() {
        for (n, seq) in daily_sequences().into_iter().enumerate() {
            let mut sc = DailyScorer::with_baseline(&seq.means, &seq.devs);
            for (d, day) in seq.days.iter().enumerate() {
                let r = sc.step(
                    day.day_utc,
                    day.age,
                    &day.periods,
                    &day.prev_day,
                    day.chrono,
                    day.sri,
                );
                let (sleep, sleep_c, ready, contrib, final_sleep) = day.want;
                assert_eq!(r.sleep.score, sleep, "seq {n} day {d} sleep score");
                assert_eq!(
                    r.sleep.contributors, sleep_c,
                    "seq {n} day {d} sleep contributors"
                );
                assert_eq!(
                    r.readiness_score,
                    Some(ready),
                    "seq {n} day {d} readiness score"
                );
                assert_eq!(
                    r.contributors,
                    Some(contrib),
                    "seq {n} day {d} contributors"
                );
                assert_eq!(
                    r.outputs.unwrap().sleep.score,
                    final_sleep,
                    "seq {n} day {d} final sleep"
                );
            }
            assert_eq!(sc.day, seq.days.len() as u32);
        }
    }
}
