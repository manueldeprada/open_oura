//! Inputs of bedtime detection (see [`super::bedtime`]) built from ring events. Port of the assembly
//! in `tools/bedtime_detect.py`; `docs/algorithms/metrics.md` summarizes the rules.
//!
//! Times are Unix milliseconds. The detector runs on a batch of events ending at `w_end` (the
//! newest motion or temperature event the sync delivered), over a window that starts at
//! [`window_start`]. The events it reads are motion (tag 0x47), temperature (0x46), state change
//! (0x45) and feature session (0x6c).

use super::bedtime::{detect_bedtime, Period, PreviousBedtime};
use super::sleepstaging::events::AppClock;
use serde_json::Value;

const MOTION_BIN_MS: i64 = 30_000;
const TEMP_BIN_MS: i64 = 60_000;
/// An event within this of the end of a bin goes to the next one.
const ROUND_UP_MS: i64 = 2_000;
/// Detections shorter than this are dropped.
const MIN_PERIOD_MS: i64 = 60_000;
const H_MS: i64 = 3_600_000;
const DAY_MS: i64 = 24 * H_MS;
const LOOKBACK_MAX_MS: i64 = 28 * DAY_MS;
const LOOKBACK_MIN_MS: i64 = 36 * H_MS;
/// A detected period of at least this length is a main sleep.
pub const MAJOR_MIN_MS: i64 = 3 * H_MS;
const RESTING_HR_FEATURE: i64 = 8;
const START_BY_SYSTEM: i64 = 1;
const STATE_NOT_IN_FINGER: i64 = 1;
const STATE_FINGER_DETECTION: i64 = 2;
const STATE_OUT_OF_POWER: i64 = 7;
const STATE_CHARGING: i64 = 8;
/// The function requires a nonzero value and does not read it.
const DEFAULT_AVG_TEMP_CENTI: u16 = 3500;

/// Events of the four types, each sorted by time and then value.
#[derive(Debug, Default, Clone)]
pub struct Events {
    /// (time, `motion_seconds`)
    pub motion: Vec<(i64, i64)>,
    /// (time, first temperature in degrees C)
    pub temp: Vec<(i64, f64)>,
    /// (time, state)
    pub state: Vec<(i64, i64)>,
    /// (time, feature id, session status)
    pub session: Vec<(i64, i64, i64)>,
}

impl Events {
    /// Collects the events from `(ring ds, tag, decoded json, captured_unix)`; `unix_ms` places
    /// an event on the calendar from its ring time and capture time.
    pub fn from_decoded(
        events: &[(i64, u8, String, i64)],
        unix_ms: &dyn Fn(i64, i64) -> i64,
    ) -> Self {
        let mut e = Events::default();
        for (ds, tag, json, cu) in events {
            if !matches!(*tag, 0x45 | 0x46 | 0x47 | 0x6c) {
                continue;
            }
            let Ok(v) = serde_json::from_str::<Value>(json) else { continue };
            let t = unix_ms(*ds, *cu);
            match *tag {
                0x47 => {
                    if let Some(s) = v["motion_seconds"].as_i64() {
                        e.motion.push((t, s));
                    }
                }
                0x46 => {
                    if let Some(x) = v["temps_c"].get(0).and_then(Value::as_f64) {
                        e.temp.push((t, x));
                    }
                }
                0x45 => {
                    if let Some(s) = v["state"].as_i64() {
                        e.state.push((t, s));
                    }
                }
                _ => {
                    if let (Some(f), Some(s)) = (v["feature_id"].as_i64(), v["session_status"].as_i64()) {
                        e.session.push((t, f, s));
                    }
                }
            }
        }
        e.motion.sort();
        e.temp.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        e.state.sort();
        e.session.sort();
        e
    }
}

/// The detector's arrays for one batch.
#[derive(Debug, Clone, PartialEq)]
pub struct Assembled {
    pub motion: Vec<u8>,
    pub motion_ts: Vec<u64>,
    pub temp: Vec<u16>,
    pub temp_ts: Vec<u64>,
    /// Resting-HR measurement periods.
    pub hr: Vec<(u64, u64)>,
    /// First and last timestamp of the span.
    pub span: (i64, i64),
}

/// The bin of `t`, rounded up when `t` is within
/// `round_ms` of the end of its bin.
fn bin_index(t: i64, start: i64, size: i64, round_ms: i64) -> i64 {
    let d = t - start;
    let i = d.div_euclid(size);
    if round_ms != 0 && d.rem_euclid(size) >= size - round_ms {
        i + 1
    } else {
        i
    }
}

/// `Math.round(temp1 * 100f)`: `temp1` is a float32 and the product is rounded in float32 before
/// the round-half-up.
fn centi(temp1: f64) -> i64 {
    let x = (temp1 as f32 * 100.0f32) as f64;
    (x + 0.5).floor() as i64
}

/// Resting-HR measurement periods: the session events from the window start to `last_event` and
/// the newest session event before the window. A start by system opens a period, any end status
/// (2 to 11) closes it, and a period still open at the span end closes there.
fn hr_periods(ev: &Events, win_start: i64, last_event: i64, span_end: i64) -> Vec<(u64, u64)> {
    let before = ev.session.iter().rev().find(|x| x.0 < win_start);
    let inside = ev.session.iter().filter(|x| win_start <= x.0 && x.0 <= last_event);
    let mut out = Vec::new();
    let mut start: Option<i64> = None;
    for &(t, fid, status) in before.into_iter().chain(inside) {
        if fid != RESTING_HR_FEATURE {
            continue;
        }
        if status == START_BY_SYSTEM {
            start = Some(t);
        } else if (2..=11).contains(&status) {
            if let Some(s) = start.take() {
                if t > s {
                    out.push((s as u64, t as u64));
                }
            }
        }
    }
    if let Some(s) = start {
        if span_end > s {
            out.push((s as u64, span_end as u64));
        }
    }
    out
}

/// The arrays for a batch ending at `w_end` with window start `win_start`, or `None` when detection is skipped
///  (no motion or no temperature event in the window).
pub fn assemble(ev: &Events, win_start: i64, w_end: i64, last_event: i64) -> Option<Assembled> {
    let mo: Vec<&(i64, i64)> = ev.motion.iter().filter(|x| win_start <= x.0 && x.0 <= w_end).collect();
    let te: Vec<&(i64, f64)> = ev.temp.iter().filter(|x| win_start <= x.0 && x.0 <= w_end).collect();
    if mo.is_empty() || te.is_empty() {
        return None;
    }
    let lo = mo[0].0.min(te[0].0);
    let hi = mo[mo.len() - 1].0.max(te[te.len() - 1].0);
    if hi <= lo {
        return None;
    }
    let n = ((hi - lo) / MOTION_BIN_MS).max(1) as usize;
    let mut first: Vec<Option<i64>> = vec![None; n];
    for &&(t, s) in &mo {
        let i = bin_index(t, lo, MOTION_BIN_MS, ROUND_UP_MS).clamp(0, n as i64 - 1) as usize;
        first[i].get_or_insert(s);
    }
    let before = ev.state.iter().rev().find(|x| x.0 <= hi && x.0 <= lo);
    let states: Vec<(i64, i64)> = before
        .into_iter()
        .copied()
        .chain(ev.state.iter().copied().filter(|x| x.0 <= hi && x.0 > lo))
        .collect();
    let (mut motion, mut motion_ts) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut k = 0usize;
    for (i, f) in first.iter().enumerate() {
        let tb = lo + i as i64 * MOTION_BIN_MS;
        while k + 1 < states.len() && states[k + 1].0 <= tb {
            k += 1;
        }
        let state = states.get(k).map(|s| s.1);
        let v = match (f, state) {
            (Some(v), _) => *v,
            (None, Some(STATE_CHARGING)) => 255,
            (None, Some(STATE_NOT_IN_FINGER | STATE_FINGER_DETECTION | STATE_OUT_OF_POWER)) => 31,
            _ => 0,
        };
        motion.push((v & 0xFF) as u8);
        motion_ts.push(tb as u64);
    }
    let n2 = ((hi - lo) / TEMP_BIN_MS).max(1) as usize;
    let mut bins: Vec<Option<f64>> = vec![None; n2];
    for &&(t, v) in &te {
        if v > 0.0 {
            let i = bin_index(t, lo, TEMP_BIN_MS, 0);
            if (0..n2 as i64).contains(&i) {
                bins[i as usize].get_or_insert(v);
            }
        }
    }
    // a temperature past the last bin is dropped, as the bin array has no slot for it
    let temp = bins.iter().map(|b| b.map_or(0, |v| centi(v).clamp(0, 0xFFFF) as u16)).collect();
    let temp_ts = (0..n2).map(|i| (lo + i as i64 * TEMP_BIN_MS) as u64).collect();
    Some(Assembled { motion, motion_ts, temp, temp_ts, hr: hr_periods(ev, win_start, last_event, hi), span: (lo, hi) })
}

/// Window start of the detection: `j = min(UTC midnight of start_ts, end_ts - 36 h)`.
///
/// Detected periods: `start < j` (strict), at least 3 h, then
/// the latest UTC day of the end first and the longest first; the first row wins and ties keep the
/// order given. Confirmed sleeps (the ring's `bedtime_period`):
/// `start < j` (strict), at least 3 h, the longest of each local day of the end, then the latest
/// start. Neither can be older than `end_ts - 28 d`, which is also the start when there is no
/// such period. `confirmed` is used only when `seed` is set.
pub fn window_start(
    detected: &[(i64, i64)],
    confirmed: &[(i64, i64)],
    seed: bool,
    tz_hours: f64,
    start_ts: i64,
    end_ts: i64,
) -> i64 {
    let floor = end_ts - LOOKBACK_MAX_MS;
    let j = (start_ts - start_ts.rem_euclid(DAY_MS)).min(end_ts - LOOKBACK_MIN_MS);
    let mut best: Option<&(i64, i64)> = None;
    for p in detected.iter().filter(|p| p.0 < j && p.1 - p.0 >= MAJOR_MIN_MS) {
        let key = |q: &(i64, i64)| (q.1.div_euclid(DAY_MS), q.1 - q.0);
        if best.map_or(true, |b| key(p) > key(b)) {
            best = Some(p);
        }
    }
    if let Some(p) = best {
        return p.0.max(floor);
    }
    if seed {
        let day = |p: &(i64, i64)| (p.1 + (tz_hours * H_MS as f64) as i64).div_euclid(DAY_MS);
        let mut per_day: std::collections::BTreeMap<i64, (i64, i64)> = Default::default();
        for p in confirmed.iter().filter(|p| p.0 < j && p.1 - p.0 >= MAJOR_MIN_MS) {
            let e = per_day.entry(day(p)).or_insert(*p);
            if p.1 - p.0 > e.1 - e.0 {
                *e = *p;
            }
        }
        if let Some(p) = per_day.values().max_by_key(|p| p.0) {
            return p.0.max(floor);
        }
    }
    floor
}

/// The previous-bedtime list (`c.d`): the detected periods overlapping the window; when there
/// are none and `seed` is set, the confirmed sleeps overlapping it. Major means at least 3 h.
pub fn previous(
    detected: &[(i64, i64)],
    confirmed: &[(i64, i64)],
    seed: bool,
    win_start: i64,
    w_end: i64,
) -> Vec<PreviousBedtime> {
    let pick = |src: &[(i64, i64)]| -> Vec<PreviousBedtime> {
        let mut v: Vec<(i64, i64)> = src.iter().copied().filter(|p| p.1 >= win_start && p.0 <= w_end).collect();
        v.sort();
        v.into_iter()
            .map(|(s, e)| PreviousBedtime { start_ms: s as u64, end_ms: e as u64, major: e - s >= MAJOR_MIN_MS })
            .collect()
    };
    let det = pick(detected);
    if !det.is_empty() || !seed {
        return det;
    }
    pick(confirmed)
}

/// Runs the detector on assembled inputs. Periods under a minute are dropped; `None` when the
/// arguments are rejected (no resting-HR period).
pub fn run(a: &Assembled, previous: &[PreviousBedtime]) -> Option<Vec<Period>> {
    let res = detect_bedtime(&a.motion, &a.motion_ts, &a.temp, &a.temp_ts, &a.hr, DEFAULT_AVG_TEMP_CENTI, previous, true)
        .ok()?;
    Some(res.into_iter().filter(|p| (p.end_ms - p.start_ms) as i64 >= MIN_PERIOD_MS).collect())
}

/// A detected window for a night the ring has not declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Detected {
    pub start_ms: i64,
    pub end_ms: i64,
    /// The same instants in ring time, through the latest ring-written sync before each.
    pub start_ds: i64,
    pub end_ds: i64,
}

/// Events captured longer than this before the newest `bedtime_period` cannot start the window.
const LOOKBACK_CAPTURE_S: i64 = 4 * 86_400;
/// Detected periods shorter than this are not a night (an evening doze).
const MIN_NIGHT_MS: i64 = 30 * 60_000;

/// The windows the detector gives for sleep the ring has not declared, from the events of a
/// database `(ring ds, tag, decoded json, captured_unix)`. This is the whole chain in one call:
///
/// 1. Ring time becomes Unix time with the ring-order clock ([`AppClock::ring_order`] with the
///    event's own ring time), and a `bedtime_period` event places its start and end with the
///    anchor of the event's own ring time. `fallback_unix_ms(ds, captured_unix)` places events
///    before the first ring-written sync.
/// 2. The confirmed sleeps are the ring's `bedtime_period` windows.
/// 3. The batch ends at the newest motion or temperature event; the window starts at
///    [`window_start`]; the arrays come from [`assemble`]; [`run`] gives the periods.
/// 4. Periods under 30 minutes and periods that overlap a confirmed sleep are dropped, so once the
///    ring declares a night the detector returns nothing for it.
///
/// `tz_hours` is the wearer's offset from UTC, which gives the local day of a confirmed sleep.
pub fn detect_open(
    events: &[(i64, u8, String, i64)],
    tz_hours: f64,
    fallback_unix_ms: &dyn Fn(i64, i64) -> i64,
) -> Vec<Detected> {
    let clock = AppClock::from_events(events);
    let unix_ms = |ds: i64, cu: i64| -> i64 {
        clock.ring_order(ds, ds).map_or_else(|| fallback_unix_ms(ds, cu), |s| (s * 1000.0).round() as i64)
    };
    let mut confirmed: Vec<(i64, i64)> = Vec::new();
    let mut newest_confirmed_cu = None;
    for (ds, tag, json, cu) in events {
        if *tag != 0x76 {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(json) else { continue };
        let (Some(s), Some(e)) = (v["bedtime_start_ds"].as_i64(), v["bedtime_end_ds"].as_i64()) else {
            continue;
        };
        let at = |x: i64| match clock.ring_order(x, *ds) {
            Some(u) => (u * 1000.0).round() as i64,
            None => fallback_unix_ms(x, *cu),
        };
        let w = (at(s), at(e));
        if !confirmed.contains(&w) {
            confirmed.push(w);
        }
        newest_confirmed_cu = newest_confirmed_cu.max(Some(*cu));
    }
    let Some(newest_confirmed_cu) = newest_confirmed_cu else { return vec![] };
    let recent: Vec<(i64, u8, String, i64)> =
        events.iter().filter(|e| e.3 >= newest_confirmed_cu - LOOKBACK_CAPTURE_S).cloned().collect();
    let ev = Events::from_decoded(&recent, &unix_ms);
    let Some(w_end) = ev.motion.iter().map(|x| x.0).chain(ev.temp.iter().map(|x| x.0)).max() else {
        return vec![];
    };
    let win_start = window_start(&[], &confirmed, true, tz_hours, w_end - 30 * 60_000, w_end);
    let prev = previous(&[], &confirmed, true, win_start, w_end);
    let Some(periods) = assemble(&ev, win_start, w_end, w_end).and_then(|a| run(&a, &prev)) else {
        return vec![];
    };
    periods
        .into_iter()
        .map(|p| (p.start_ms as i64, p.end_ms as i64))
        .filter(|&(s, e)| e - s >= MIN_NIGHT_MS)
        .filter(|&(s, e)| !confirmed.iter().any(|&(cs, ce)| s < ce && cs < e))
        .filter_map(|(s, e)| {
            let ds = |ms: i64| clock.ring_order_ds(ms as f64 / 1000.0);
            Some(Detected { start_ms: s, end_ms: e, start_ds: ds(s)?, end_ds: ds(e)? })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_round_up_near_the_end() {
        assert_eq!(bin_index(27_999, 0, 30_000, 2_000), 0);
        assert_eq!(bin_index(28_000, 0, 30_000, 2_000), 1);
        assert_eq!(bin_index(28_000, 0, 30_000, 0), 0);
    }

    #[test]
    fn centi_rounds_in_f32() {
        assert_eq!(centi(33.5), 3350);
        assert_eq!(centi(33.455), 3346);
    }

    #[test]
    fn empty_bins_follow_the_state() {
        let ev = Events {
            motion: vec![(0, 5), (150_000, 7)],
            temp: vec![(0, 33.0), (60_000, 33.1)],
            state: vec![(-10_000, STATE_CHARGING), (59_000, STATE_NOT_IN_FINGER), (100_000, 0)],
            session: vec![],
        };
        let a = assemble(&ev, -1_000_000, 200_000, 200_000).unwrap();
        assert_eq!(a.span, (0, 150_000));
        // bins at 0, 30, 60, 90, 120 s: motion, charging, not in finger, not in finger, and the
        // last motion event (at 150 s, past the final bin) clamped into the last bin
        assert_eq!(a.motion, vec![5, 255, 31, 31, 7]);
        assert_eq!(a.temp, vec![3300, 3310]);
    }

    #[test]
    fn hr_periods_follow_session_events() {
        let ev = Events {
            session: vec![(0, 8, 2), (10, 8, 1), (20, 9, 1), (30, 8, 3), (40, 8, 1)],
            ..Default::default()
        };
        // the event before the window (status 2, no open period) opens nothing; one period is
        // closed, the last stays open to the span end
        assert_eq!(hr_periods(&ev, 5, 50, 100), vec![(10, 30), (40, 100)]);
    }

    #[test]
    fn window_starts_from_the_confirmed_sleep() {
        let day = DAY_MS;
        let confirmed = [(10 * day + 2 * H_MS, 10 * day + 9 * H_MS)];
        let end = 12 * day + 6 * H_MS;
        assert_eq!(window_start(&[], &confirmed, true, 0.0, 11 * day + 5 * H_MS, end), 10 * day + 2 * H_MS);
        assert_eq!(window_start(&[], &confirmed, false, 0.0, 11 * day + 5 * H_MS, end), end - LOOKBACK_MAX_MS);
        // a sleep under 3 h is not a candidate, and the longest sleep of a day wins
        let short = [(10 * day + 2 * H_MS, 10 * day + 4 * H_MS)];
        assert_eq!(window_start(&[], &short, true, 0.0, 11 * day + 5 * H_MS, end), end - LOOKBACK_MAX_MS);
        let two = [(10 * day + 2 * H_MS, 10 * day + 9 * H_MS), (10 * day + 12 * H_MS, 10 * day + 16 * H_MS)];
        assert_eq!(window_start(&[], &two, true, 0.0, 11 * day + 5 * H_MS, end), 10 * day + 2 * H_MS);
        assert_eq!(previous(&[], &confirmed, true, 10 * day, end).len(), 1);
        assert!(previous(&[], &confirmed, false, 10 * day, end).is_empty());
    }
}
