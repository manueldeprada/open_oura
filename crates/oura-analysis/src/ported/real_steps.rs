//! Step counts from raw ring events.
//!
//! Flow:
//! 1. Gait records come from `0x7e`/`0x7f` event pairs and are decoded into 10 s rows by
//!    [`steps_motion_decoder`](super::steps_motion_decoder); the columns are reordered with
//!    [`stepmotion_from_decoder`].
//! 2. Motion rows come from the 30 s `0x47` events ([`motion_row`]). The average axes are the
//!    event's signed byte times 8, and `regular_motion` is the `regularity` bit, bit 7 of byte 4.
//! 3. [`StepCounter::count`] merges both streams, counts steps per row, and splits the counts
//!    onto a grid of [`Config::aggregate_interval_ms`] (60 s by default).
//!
//! The calculation covers `[start, end]` with the step-motion query starting one aggregation
//! interval before `start` and the motion query a further [`Config::motion_lookback_ms`]
//! earlier. [`calculation_start`] picks `start` as the end of the newest stored count, not
//! earlier than [`Config::days_limit`] days before now. The step counter needs a weights file
//! you supply (see [`StepCounter::from_bytes`]).

use oura_protocol::events::{pair_real_steps, parse_motion_event, MotionEvent};

use super::step_counter::{
    stepmotion_from_decoder, StepCounter, MOTION_COLUMNS, STEPMOTION_COLUMNS,
};
use super::steps_motion_decoder;

const DAY_MS: i64 = 86_400_000;

/// Parameters of the calculation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Interval of the output counts, in milliseconds.
    pub aggregate_interval_ms: i64,
    /// Extra look-back of the motion query before the step-motion query.
    pub motion_lookback_ms: i64,
    /// Oldest start of a calculation, in days before now.
    pub days_limit: i64,
}

impl Default for Config {
    /// 60 s, 30 s, 7 days.
    fn default() -> Self {
        Config { aggregate_interval_ms: 60_000, motion_lookback_ms: 30_000, days_limit: 7 }
    }
}

/// One step count: `steps` over `[timestamp_ms, end_time_ms)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepCount {
    pub timestamp_ms: i64,
    pub end_time_ms: i64,
    pub steps: f64,
}

/// Start of a calculation: the end of the newest stored count (`day_start_ms - 1` if
/// none), not earlier than `days_limit` days before `now_ms`.
pub fn calculation_start(
    newest_end_ms: Option<i64>,
    day_start_ms: i64,
    now_ms: i64,
    days_limit: i64,
) -> i64 {
    newest_end_ms.unwrap_or(day_start_ms - 1).max(now_ms - days_limit * DAY_MS)
}

/// The step counter's motion input row for one `motion_event`. `regular_motion` is the
/// event's `regularity` bit; absent intensities are 0.
pub fn motion_row(e: &MotionEvent) -> [f32; MOTION_COLUMNS] {
    [
        e.orientation as f32,
        e.motion_seconds as f32,
        e.avg_x as f32,
        e.avg_y as f32,
        e.avg_z as f32,
        e.regularity as u8 as f32,
        e.low_intensity.unwrap_or(0) as f32,
        e.high_intensity.unwrap_or(0) as f32,
    ]
}

/// Count steps from ring events for the calculation range `[start_ms, end_ms]` (`None`: all
/// events).
///
/// `events` are `(tag, ring timestamp in deciseconds, body)`; tags other than `0x7e`, `0x7f`
/// and `0x47` are ignored. `to_unix_ms` maps a ring timestamp to unix milliseconds (ring clocks
/// restart on reboot, so the mapping needs the boot-epoch table). Returns no rows when either
/// the gait rows or the motion rows of the range are empty.
pub fn count_steps<'a, I, F>(
    counter: &StepCounter,
    events: I,
    to_unix_ms: F,
    range: Option<(i64, i64)>,
    cfg: &Config,
) -> Vec<StepCount>
where
    I: IntoIterator<Item = (u8, u32, &'a [u8])> + Clone,
    F: Fn(u32) -> i64,
{
    let mut gait: Vec<(i64, [u16; 27])> = pair_real_steps(events.clone())
        .into_iter()
        .map(|r| (to_unix_ms(r.timestamp), r.columns))
        .collect();
    gait.sort_by_key(|g| g.0);
    let (ts, records): (Vec<i64>, Vec<[u16; 27]>) = gait.into_iter().unzip();
    let in_range = |t: i64, lookback: i64| {
        range.is_none_or(|(start, end)| t >= start - cfg.aggregate_interval_ms - lookback && t <= end)
    };

    let (step_ts, stepmotion): (Vec<i64>, Vec<[f32; STEPMOTION_COLUMNS]>) =
        steps_motion_decoder::decode(&ts, &records)
            .into_iter()
            .filter(|r| in_range(r.timestamp, 0))
            .map(|r| (r.timestamp, stepmotion_from_decoder(&r.features)))
            .unzip();
    let mut motion: Vec<(i64, [f32; MOTION_COLUMNS])> = events
        .into_iter()
        .filter(|(tag, _, _)| *tag == 0x47)
        .filter_map(|(_, ts, body)| Some((to_unix_ms(ts), motion_row(&parse_motion_event(body)?))))
        .filter(|(t, _)| in_range(*t, cfg.motion_lookback_ms))
        .collect();
    motion.sort_by_key(|m| m.0);
    let (motion_ts, motion): (Vec<i64>, Vec<_>) = motion.into_iter().unzip();
    if step_ts.is_empty() || motion_ts.is_empty() {
        return vec![];
    }
    let out = counter.count(
        &step_ts,
        &stepmotion,
        &motion_ts,
        &motion,
        Some(cfg.aggregate_interval_ms as f64),
    );
    (0..out.steps.len())
        .map(|i| StepCount {
            timestamp_ms: out.start_ms[i],
            end_time_ms: out.end_ms[i],
            steps: out.steps[i] as f64,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::step_counter::test_support::counter;

    #[test]
    fn motion_row_fills_missing_fields() {
        let e = parse_motion_event(&[0b0110_0101, 3, 0xfe, 1]).unwrap();
        assert_eq!(motion_row(&e), [3.0, 5.0, 24.0, -16.0, 8.0, 0.0, 0.0, 0.0]);
        // byte 4: regularity (bit 7) and low intensity; byte 5: high intensity
        let e = parse_motion_event(&[0, 0, 0, 0, 0x80 | 7, 9]).unwrap();
        assert_eq!(motion_row(&e)[5..], [1.0, 7.0, 9.0]);
        assert!(parse_motion_event(&[0, 0, 0, 0, 0x40]).is_none());
    }

    #[test]
    fn calculation_start_is_clamped_to_the_day_limit() {
        let now = 10 * DAY_MS;
        assert_eq!(calculation_start(Some(now - 1000), 9 * DAY_MS, now, 7), now - 1000);
        assert_eq!(calculation_start(Some(2 * DAY_MS), 9 * DAY_MS, now, 7), now - 7 * DAY_MS);
        assert_eq!(calculation_start(None, 9 * DAY_MS, now, 7), 9 * DAY_MS - 1);
    }

    fn body(setup: impl Fn(&mut [u8; 14])) -> Vec<u8> {
        let mut b = [0u8; 14];
        setup(&mut b);
        b.to_vec()
    }

    /// A walking record in the first 10 s block: total amplitude column 400 (about 1118 mg),
    /// stride frequency column 154, stride amplitude column 256, gait amplitude column 64.
    fn walking_part1() -> Vec<u8> {
        body(|b| {
            b[0] = 200;
            b[1] = 77;
            b[2] = 128;
            b[5] = 64;
        })
    }

    fn events() -> Vec<(u8, u32, Vec<u8>)> {
        let silent = body(|_| {});
        vec![
            (0x7e, 6300, walking_part1()),
            (0x7f, 6301, silent.clone()),
            (0x7e, 6600, walking_part1()),
            (0x7f, 6601, silent),
            (0x47, 6300, vec![0, 0, 0, 0]),
            (0x47, 6600, vec![0, 0, 0, 0]),
        ]
    }

    fn run(range: Option<(i64, i64)>, e: f32) -> Vec<StepCount> {
        let events = events();
        let iter = events.iter().map(|(t, ds, b)| (*t, *ds, b.as_slice()));
        count_steps(&counter(e), iter, |ds| ds as i64 * 100, range, &Config::default())
    }

    #[test]
    fn counts_steps_per_minute_from_events() {
        // frequency column 154: 0.68 + 153 / 511 * 2.72 Hz, above the doubling range, 10 s per row
        let per_row = (0.68 + 153.0 / 511.0 * 2.72) * 10.0;
        let out = run(None, 0.95);
        // rows end at 610 to 660 s; the first four fill the 600 s cell, the last ends the 660 s cell
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].timestamp_ms, out[0].end_time_ms), (600_000, 660_000));
        assert!((out[0].steps - 2.0 * per_row).abs() < 1e-3, "{}", out[0].steps);
        assert_eq!((out[1].timestamp_ms, out[1].end_time_ms), (660_000, 660_000));
        assert_eq!(out[1].steps, 0.0);
    }

    #[test]
    fn low_eligibility_scales_the_counts_down() {
        let full: f64 = run(None, 0.95).iter().map(|c| c.steps).sum();
        let low: f64 = run(None, 0.3).iter().map(|c| c.steps).sum();
        assert!(low > 0.0 && low < full, "{low} vs {full}");
        assert_eq!(run(None, 0.1).iter().map(|c| c.steps).sum::<f64>(), 0.0);
    }

    #[test]
    fn range_outside_the_events_gives_no_rows() {
        assert!(run(Some((10_000_000, 11_000_000)), 0.95).is_empty());
    }

    #[test]
    fn missing_motion_or_gait_events_give_no_rows() {
        let counter = counter(0.95);
        let all = events();
        let cfg = Config::default();
        let only_motion = all.iter().filter(|e| e.0 == 0x47).map(|(t, ds, b)| (*t, *ds, b.as_slice()));
        assert!(count_steps(&counter, only_motion, |ds| ds as i64 * 100, None, &cfg).is_empty());
        let only_gait = all.iter().filter(|e| e.0 != 0x47).map(|(t, ds, b)| (*t, *ds, b.as_slice()));
        assert!(count_steps(&counter, only_gait, |ds| ds as i64 * 100, None, &cfg).is_empty());
    }
}
