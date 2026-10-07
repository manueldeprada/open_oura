//! Per-day activity statistics and score from the ring's per-minute MET bytes: the
//! day bookkeeping around the ports of the `actinfo` module ([`super::actinfo`]) and
//! activity score ([`super::activity_score`]).
//!
//! The activity day starts at 04:00 local time. An `activity_information` event is stamped
//! when the ring writes it, so its MET bytes cover the minutes before the stamp. Bytes that
//! precede the day boundary go to the previous day, and a day's target scales with the readiness
//! score of its date. Days are keyed by the `YYYY-MM-DD` date they start on.

use super::actinfo::{sleep_minutes, ActInfo, Stats, User, DAY_MIN};
use super::activity_score::{self, Day};
use std::collections::BTreeMap;

const DAY_START_HOUR: i64 = 4;
const DAY_S: i64 = 86_400;
/// Readiness percent assumed for the daily target when no readiness is known.
const DEFAULT_READINESS_PCT: u32 = 80;

/// One `activity_information` event: wall-clock stamp, step count, MET bytes.
pub struct ActEvent {
    pub unix: f64,
    pub steps: u8,
    pub met: Vec<u8>,
}

pub struct Inputs<'a> {
    pub events: &'a [ActEvent],
    /// `alert_event` stamps with the inactivity alert type.
    pub sit_alerts: &'a [f64],
    /// Sleep periods as wall-clock `(start, end)` seconds.
    pub sleeps: &'a [(f64, f64)],
    pub user: User,
    /// Hours east of UTC.
    pub tz: f64,
    /// Readiness score by activity-day date; it scales the day's target (80 when absent).
    pub readiness: &'a BTreeMap<String, u32>,
}

/// The ring byte for a decoded MET value (`b < 128`: 0.1 MET steps, above: `12.8 + 0.2 (b - 128)`).
pub fn met_byte(met: f64) -> u8 {
    if met < 12.8 {
        (met * 10.0).round().clamp(0.0, 127.0) as u8
    } else {
        (128.0 + (met - 12.8) / 0.2).round().clamp(128.0, 255.0) as u8
    }
}

pub struct DayResult {
    pub stats: Stats,
    pub score: u32,
    pub contributors: [f32; 7],
}

/// Gregorian (year, month, day) of a day count since 1970-01-01.
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (y + i64::from(m <= 2), m, d)
}

/// Local-time start of the activity day containing `unix`, in wall-clock seconds.
fn day_start(unix: f64, tz: f64) -> i64 {
    let off = (tz * 3600.0).round() as i64;
    let local = unix.floor() as i64 + off - DAY_START_HOUR * 3600;
    local.div_euclid(DAY_S) * DAY_S + DAY_START_HOUR * 3600 - off
}

fn date_key(day_start_unix: i64, tz: f64) -> String {
    let local = day_start_unix + (tz * 3600.0).round() as i64;
    let (y, m, d) = civil(local.div_euclid(DAY_S));
    format!("{y:04}-{m:02}-{d:02}")
}

/// The day's state, created with its sleep period (the night that ended inside the day).
fn state<'a>(
    days: &'a mut BTreeMap<i64, ActInfo>,
    inp: &Inputs,
    sleeps: &[(f64, f64)],
    start: i64,
) -> &'a mut ActInfo {
    days.entry(start).or_insert_with(|| {
        let mut a = ActInfo::new(inp.user, true);
        let pct = inp.readiness.get(&date_key(start, inp.tz)).copied().filter(|&p| p > 0);
        a.set_target(pct.unwrap_or(DEFAULT_READINESS_PCT));
        let end = start + DAY_S;
        if let Some(&(s, e)) =
            sleeps.iter().rev().find(|(_, e)| *e > start as f64 && (*e as i64) < end)
        {
            a.sleep = (sleep_minutes(s as i64, start), sleep_minutes(e as i64, start));
        }
        a
    })
}

/// Statistics and score for every day with activity events, keyed by the date the day
/// starts on.
pub fn compute(inp: &Inputs) -> BTreeMap<String, DayResult> {
    let mut days: BTreeMap<i64, ActInfo> = BTreeMap::new();
    let mut sleeps: Vec<(f64, f64)> = inp.sleeps.to_vec();
    sleeps.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut events: Vec<&ActEvent> = inp.events.iter().collect();
    events.sort_by(|a, b| a.unix.total_cmp(&b.unix));
    for e in events {
        let start = day_start(e.unix, inp.tz);
        let last = ((e.unix as i64 - start) / 60) as u32;
        state(&mut days, inp, &sleeps, start).add_activity(last, e.steps as u32, &e.met);
        // bytes that precede the day boundary belong to the previous day
        if (last as usize) < e.met.len() {
            state(&mut days, inp, &sleeps, start - DAY_S).add_activity(last + DAY_MIN as u32, 0, &e.met);
        }
    }
    for &t in inp.sit_alerts {
        state(&mut days, inp, &sleeps, day_start(t, inp.tz)).add_sit_alert();
    }
    let stats: Vec<(i64, Stats)> = days.iter().map(|(s, a)| (*s, a.statistics(0))).collect();
    let first_day = stats.first().map(|(s, _)| *s).unwrap_or(0);
    let mut history: Vec<Day> = Vec::new();
    let mut out = BTreeMap::new();
    for (start, st) in stats {
        // a day reached by the boundary spill alone carries no steps or events of its own
        if st.classes.iter().sum::<u32>() == 0 && st.steps == 0 {
            continue;
        }
        history.push(st.day);
        let ring_day = ((start - first_day) / DAY_S) as u32;
        let tail = &history[history.len().saturating_sub(10)..];
        out.insert(
            date_key(start, inp.tz),
            DayResult {
                score: activity_score::score(tail, ring_day),
                contributors: activity_score::contributors(tail, ring_day),
                stats: st,
            },
        );
    }
    out
}

// ---- the stateless day -------------------------------------------------------------------------

/// A `time_sync` event written by the ring: its ring time (deciseconds) and the Unix time it carries.
/// Phone-written anchors are not passed and are not used.
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    pub ds: i64,
    pub unix: i64,
}

/// An `activity_information` event by ring time: the first body byte and the MET bytes (oldest first,
/// the last one the minute before the event time).
#[derive(Clone, Debug)]
pub struct RingEvent {
    pub ds: i64,
    pub steps: u8,
    pub met: Vec<u8>,
}

/// A `wear_event` (tag 0x53): ring time and state. States 3 to 6 are "on the finger"; every other state
/// (not in finger 1, detection 2, out of power 7, charging 8, ...) is non-wear.
#[derive(Clone, Copy, Debug)]
pub struct WearEvent {
    pub ds: i64,
    pub state: u8,
}

/// An `alert_event` (tag 0x56); `sit` marks the inactivity alert type (1).
#[derive(Clone, Copy, Debug)]
pub struct AlertEvent {
    pub ds: i64,
    pub sit: bool,
}

pub struct StatelessInputs<'a> {
    pub anchors: &'a [Anchor],
    pub events: &'a [RingEvent],
    pub wear: &'a [WearEvent],
    pub alerts: &'a [AlertEvent],
    /// Ring time of the newest event of any kind (time syncs included): the minutes of
    /// the last day are counted up to it. `None` takes the newest activity, wear or alert event.
    pub newest_ds: Option<i64>,
    /// Sleep periods as Unix `(start, end)` seconds; a period belongs to the days its start and its end fall in.
    pub sleeps: &'a [(f64, f64)],
    pub user: User,
    /// Phone zone, hours east of UTC.
    pub tz: f64,
    /// Zone of the days stored in another frame, by `YYYY-MM-DD` date (a day keeps the zone of
    /// its first record, see `docs/algorithms/metrics.md`).
    pub day_tz: &'a BTreeMap<String, f64>,
    /// Readiness score of the date (80 when absent); it scales the day's target.
    pub readiness: &'a BTreeMap<String, u32>,
    /// Fill the nine-day history before the first ring day with two placeholder days.
    pub pad_history: bool,
}

/// The history placeholders: A (10000 m, 2 alerts) and B (5000 m, 9 alerts), repeating A, A, B back
/// from the newest of the nine days.
pub fn pad_days(age: u8, gender: u32) -> Vec<Day> {
    let a = Day {
        eq_distance: 10000,
        eq_target: 10000,
        sedentary_min: 480,
        moderate_met_min: 250,
        vigorous_met_min: 150,
        sit_alerts: 2,
        age,
        gender,
        moderate_min: 120,
        vigorous_min: 30,
        target_reached: 0,
    };
    let b = Day {
        eq_distance: 5000,
        eq_target: 10000,
        sedentary_min: 999,
        moderate_met_min: 30,
        vigorous_met_min: 15,
        sit_alerts: 9,
        age,
        gender,
        moderate_min: 30,
        vigorous_min: 10,
        target_reached: 0,
    };
    (0..9).rev().map(|k| if k % 3 == 2 { b } else { a }).collect()
}

/// Wear states that mean "on the finger".
const FINGER_STATES: [u8; 4] = [3, 4, 5, 6];
/// Events with a stored time up to this long after the day's end still belong to its batch.
const BATCH_TAIL_S: f64 = 900.0;

/// The stored time of an event: the latest ring-written `time_sync` at or before it in ring order,
/// `unix + (ds - sync_ds) / 10` (the first anchor extrapolates backwards).
fn stream_time(anchors: &[Anchor], ds: i64) -> Option<f64> {
    let i = anchors.partition_point(|a| a.ds <= ds);
    let a = anchors.get(i.checked_sub(1).unwrap_or(0))?;
    Some(a.unix as f64 + (ds - a.ds) as f64 / 10.0)
}

/// The time of an event in a batch parsed with its bracket of time syncs: the syncs from the latest one
/// whose own time is at or before the batch's first stored time to the earliest at or after its last, and
/// for each event the latest of them at or before the event's stored time. (The stored time is the
/// stream-order time, so an event just before a sync in ring order but after it in stored time, as happens
/// when the ring clock has drifted, is dated from that sync.)
struct BatchClock {
    usable: Vec<Anchor>,
}

impl BatchClock {
    fn new(anchors: &[Anchor], first_stored: f64, last_stored: f64) -> Self {
        let mut by_unix: Vec<Anchor> = anchors.to_vec();
        by_unix.sort_by_key(|a| (a.unix, a.ds));
        by_unix.dedup_by_key(|a| (a.unix, a.ds));
        if by_unix.is_empty() {
            return BatchClock { usable: by_unix };
        }
        let lo = by_unix.partition_point(|a| (a.unix as f64) <= first_stored).saturating_sub(1);
        let hi = by_unix.partition_point(|a| (a.unix as f64) < last_stored).min(by_unix.len() - 1);
        BatchClock { usable: by_unix[lo..=hi.max(lo)].to_vec() }
    }

    fn time(&self, ds: i64, stored: f64) -> f64 {
        let k = self.usable.partition_point(|a| (a.unix as f64) <= stored);
        match k.checked_sub(1).and_then(|k| self.usable.get(k)) {
            Some(a) => a.unix as f64 + (ds - a.ds) as f64 / 10.0,
            None => stored,
        }
    }
}

/// One day of the stateless path before the score: the 1440 bytes (9 where no event wrote, 1 for
/// non-wear), the step sum, and the number of minutes counted.
struct DayArray {
    buf: Vec<u8>,
    steps: u32,
    /// `None`: nothing is counted yet (the day starts after the newest event). `Some(true)`: completed.
    state: Option<bool>,
    counted: usize,
}

struct Stored<'a, T> {
    t: f64,
    ev: &'a T,
}

fn floor_div(x: f64, d: f64) -> i64 {
    (x / d).floor() as i64
}

/// Statistics and score of every activity day by the stateless path, keyed by `YYYY-MM-DD`.
///
/// Rules:
///
/// - An event's time is the latest ring-written sync at or before it in ring order; the events of a
///   day's batch ([day start, day start + 24 h + 15 min] by that time) are dated again with the syncs
///   bracketing the batch ([`BatchClock`]). The minute an event ends in is `floor((t - day start) / 60)`
///   and its bytes fill the minutes before it, later events overwriting earlier ones; bytes outside the
///   day are dropped.
/// - Every minute starts at byte 9 (0.9 MET). Wear events mark non-wear (byte 1) from the first non-finger
///   state to the next finger state, minutes `floor(start)` to `floor(end) - 1`, without a minimum
///   duration; an interval still open at the end runs to the newest event.
/// - A completed day counts all 1440 minutes (minute 1439 only as a non-wear flag); the last day counts the
///   minutes up to the newest event. The 24 hourly sedentary counters start from the previous date's array.
/// - Every sleep period of the day rests its sedentary minutes. The day's own zone is `day_tz` or `tz`.
/// - The history is the nine days before (placeholders before the first ring day when `pad_history`)
///   and the day.
pub fn compute_stateless(inp: &StatelessInputs) -> BTreeMap<String, DayResult> {
    let mut anchors: Vec<Anchor> = inp.anchors.to_vec();
    anchors.sort_by_key(|a| a.ds);
    let stored = |ds: i64| stream_time(&anchors, ds);
    let mut events: Vec<&RingEvent> = inp.events.iter().collect();
    events.sort_by_key(|e| e.ds);
    let mut wear: Vec<&WearEvent> = inp.wear.iter().collect();
    wear.sort_by_key(|w| w.ds);
    let all_alerts: Vec<&AlertEvent> = {
        let mut v: Vec<&AlertEvent> = inp.alerts.iter().collect();
        v.sort_by_key(|a| a.ds);
        v
    };
    let newest_ds = inp
        .newest_ds
        .or_else(|| {
            events.iter().map(|e| e.ds).chain(wear.iter().map(|w| w.ds)).chain(all_alerts.iter().map(|a| a.ds)).max()
        });
    let newest = newest_ds.and_then(stored);

    let ev_t: Vec<Stored<RingEvent>> =
        events.iter().filter_map(|e| stored(e.ds).map(|t| Stored { t, ev: *e })).collect();
    let wear_t: Vec<Stored<WearEvent>> =
        wear.iter().filter_map(|w| stored(w.ds).map(|t| Stored { t, ev: *w })).collect();
    let alert_t: Vec<Stored<AlertEvent>> =
        all_alerts.iter().filter_map(|a| stored(a.ds).map(|t| Stored { t, ev: *a })).collect();

    // The days: each date takes the zone it was stored with; a day exists when an event ends in it.
    let zone_of = |date: &str| inp.day_tz.get(date).copied().unwrap_or(inp.tz);
    let mut zones: Vec<f64> = vec![inp.tz];
    zones.extend(inp.day_tz.values().copied());
    zones.sort_by(|a, b| a.total_cmp(b));
    zones.dedup();
    let mut days: BTreeMap<String, (i64, f64)> = BTreeMap::new();
    for &z in &zones {
        for e in &ev_t {
            let d0 = day_start(e.t, z);
            let date = date_key(d0, z);
            if zone_of(&date) == z {
                days.entry(date).or_insert((d0, z));
            }
        }
    }

    // Pass 1: each day's array.
    let mut arrays: BTreeMap<String, DayArray> = BTreeMap::new();
    for (date, &(d0, _)) in &days {
        let d0f = d0 as f64;
        let in_batch = |t: f64| t >= d0f && t <= d0f + DAY_S as f64 + BATCH_TAIL_S;
        let ev_b: Vec<&Stored<RingEvent>> = ev_t.iter().filter(|e| in_batch(e.t)).collect();
        let wear_b: Vec<&Stored<WearEvent>> = wear_t.iter().filter(|w| in_batch(w.t)).collect();
        let alert_b: Vec<&Stored<AlertEvent>> = alert_t.iter().filter(|a| in_batch(a.t)).collect();
        let mut buf = vec![9u8; DAY_MIN];
        let mut steps = 0u32;
        let mut marks: Vec<(i64, i64)> = Vec::new();
        let mut open: Option<(f64, Option<f64>)> = None;
        if !ev_b.is_empty() || !wear_b.is_empty() || !alert_b.is_empty() {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for t in ev_b.iter().map(|e| e.t).chain(wear_b.iter().map(|w| w.t)).chain(alert_b.iter().map(|a| a.t)) {
                lo = lo.min(t);
                hi = hi.max(t);
            }
            let clock = BatchClock::new(&anchors, lo, hi);
            // the latest wear event before the day says whether the day starts in non-wear
            let before_state = wear_t.iter().filter(|w| w.t < d0f).last().map(|w| w.ev.state);
            let mut t_open: Option<f64> = match before_state {
                Some(s) if !FINGER_STATES.contains(&s) => Some(d0f),
                _ => None,
            };
            let mut last_wear = None;
            // the batch in ring order: events and wear events interleave only through their own rules
            let mut items: Vec<(i64, u8, usize)> = Vec::new();
            for (i, e) in ev_b.iter().enumerate() {
                items.push((e.ev.ds, 0, i));
            }
            for (i, w) in wear_b.iter().enumerate() {
                items.push((w.ev.ds, 1, i));
            }
            items.sort();
            let mut placed: Vec<(i64, &RingEvent)> = Vec::new();
            for (_, kind, i) in items {
                if kind == 0 {
                    let e = ev_b[i];
                    let t = clock.time(e.ev.ds, e.t);
                    placed.push((floor_div(t - d0f, 60.0), e.ev));
                } else {
                    let w = wear_b[i];
                    let t = clock.time(w.ev.ds, w.t);
                    last_wear = Some(t);
                    if FINGER_STATES.contains(&w.ev.state) {
                        if let Some(a) = t_open {
                            if a < t {
                                marks.push((floor_div(a - d0f, 60.0), floor_div(t - d0f, 60.0) - 1));
                            }
                        }
                        t_open = None;
                    } else if t_open.is_none() {
                        t_open = Some(t);
                    }
                }
            }
            if let Some(a) = t_open {
                open = Some((a, last_wear));
            }
            placed.sort_by_key(|p| p.0);
            for (end, ev) in placed {
                steps += ev.steps as u32;
                let n = ev.met.len() as i64;
                for (j, &v) in ev.met.iter().enumerate() {
                    let i = end - n + j as i64;
                    if (0..DAY_MIN as i64).contains(&i) {
                        buf[i as usize] = v;
                    }
                }
            }
        }
        if let Some((a, last_wear)) = open {
            let end = newest.unwrap_or(a).max(last_wear.unwrap_or(a).min(d0f + DAY_S as f64));
            marks.push((floor_div(a - d0f, 60.0), floor_div(end - d0f, 60.0) - 1));
        }
        for (lo, hi) in marks {
            for i in lo.max(0)..=hi.min(DAY_MIN as i64 - 1) {
                buf[i as usize] = 1;
            }
        }
        let (state, counted) = match newest {
            Some(t) if d0f + DAY_S as f64 <= t => (Some(true), DAY_MIN),
            Some(t) if d0f <= t => (Some(false), (floor_div(t - d0f, 60.0).max(0) as usize).min(DAY_MIN - 1)),
            _ => (None, 0),
        };
        arrays.insert(date.clone(), DayArray { buf, steps, state, counted });
    }

    // Pass 2: statistics, seeded from the previous date's finished array.
    let mut finished: BTreeMap<String, [u8; DAY_MIN]> = BTreeMap::new();
    let mut records: Vec<(String, Stats)> = Vec::new();
    for (date, &(d0, z)) in &days {
        let arr = &arrays[date];
        let mut a = ActInfo::new(inp.user, true);
        a.stateless = true;
        let prev_date = prev_date_key(date);
        if let Some(pb) = finished.get(&prev_date) {
            a.seed_hourly(pb);
        }
        a.set_target(inp.readiness.get(date).copied().unwrap_or(DEFAULT_READINESS_PCT));
        let mut ranges: Vec<(i32, i32)> = Vec::new();
        for &(s, e) in inp.sleeps {
            if day_start(s, z) == d0 || day_start(e, z) == d0 {
                let r = (sleep_minutes(s as i64, d0), sleep_minutes(e as i64, d0));
                if !ranges.contains(&r) {
                    ranges.push(r);
                }
            }
        }
        if let Some(&longest) = ranges.iter().filter(|r| r.0 >= 0 || r.1 >= 0).max_by_key(|r| r.1 - r.0) {
            a.sleep = longest;
        }
        a.ranges = Some(ranges);
        match arr.state {
            Some(true) => {
                a.add_activity(DAY_MIN as u32 - 1, arr.steps, &arr.buf[..DAY_MIN - 1]);
                a.finish_day(arr.buf[DAY_MIN - 1]);
            }
            Some(false) => a.add_activity(arr.counted as u32, arr.steps, &arr.buf[..arr.counted]),
            None => {}
        }
        for _ in alert_t.iter().filter(|al| al.ev.sit && day_start(al.t, z) == d0) {
            a.add_sit_alert();
        }
        finished.insert(date.clone(), a.buf);
        records.push((date.clone(), a.statistics(0)));
    }

    // Scores.
    let mut out = BTreeMap::new();
    let mut history: Vec<Day> = if inp.pad_history { pad_days(inp.user.age as u8, inp.user.gender) } else { Vec::new() };
    let first = records.first().map(|(d, _)| date_ordinal(d)).unwrap_or(0);
    for (date, st) in records {
        history.push(st.day);
        let tail = &history[history.len().saturating_sub(10)..];
        let ring_day = (date_ordinal(&date) - first + 1) as u32;
        out.insert(
            date,
            DayResult {
                score: activity_score::score(tail, ring_day),
                contributors: activity_score::contributors(tail, ring_day),
                stats: st,
            },
        );
    }
    out
}

/// Days since 1970-01-01 of a `YYYY-MM-DD` key.
fn date_ordinal(date: &str) -> i64 {
    let p: Vec<i64> = date.split('-').filter_map(|x| x.parse().ok()).collect();
    let (y, m, d) = (p[0], p[1], p[2]);
    let y = y - i64::from(m <= 2);
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn prev_date_key(date: &str) -> String {
    let (y, m, d) = civil(date_ordinal(date) - 1);
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user() -> User {
        User { bmr: 1700, age: 35, gender: 1, weight_kg: 75, target_mult: 100, target_type: 0 }
    }

    #[test]
    fn met_byte_inverts_the_decoder() {
        for b in 0u8..=255 {
            let m = if b < 128 { b as f64 * 0.1 } else { 12.8 + (b as f64 - 128.0) * 0.2 };
            let m = (m * 100.0).round() / 100.0;
            assert_eq!(met_byte(m), b);
        }
    }

    #[test]
    fn day_starts_at_four_local() {
        // 2026-01-02 03:00 UTC at tz +0 belongs to the day that began 2026-01-01 04:00
        let t = 1_767_322_800.0; // 2026-01-02 03:00:00 UTC
        assert_eq!(day_start(t, 0.0), 1_767_240_000);
        assert_eq!(day_start(t + 3600.0, 0.0), 1_767_326_400);
        assert_eq!(day_start(t, 1.0), 1_767_322_800);
    }

    #[test]
    fn walking_hour_counts_steps_and_active_minutes() {
        let start = 1_767_326_400.0; // 2026-01-02 04:00 UTC
        let events: Vec<ActEvent> = (0..24)
            .map(|h| ActEvent {
                unix: start + (h * 3600 + 3600) as f64 - 60.0,
                steps: 100,
                met: vec![if h == 10 { 40 } else { 12 }; 60],
            })
            .collect();
        let r = compute(&Inputs { events: &events, sit_alerts: &[], sleeps: &[], user: user(), tz: 0.0, readiness: &BTreeMap::new() });
        let d = &r["2026-01-02"];
        assert_eq!(d.stats.steps, 2400);
        assert!(d.stats.classes[3] + d.stats.classes[4] > 0);
        assert!(d.score <= 100);
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    /// Synthetic ring events with the expected values of the Python reference (tools/score_activity.py).
    mod synthetic {
        use super::*;
        include!("vectors/activity_day.rs");

        fn run(sc: &Scenario, use_day_tz: bool) -> BTreeMap<String, DayResult> {
            let anchors: Vec<Anchor> = sc.anchors.iter().map(|&(ds, unix)| Anchor { ds, unix }).collect();
            let events: Vec<RingEvent> =
                sc.events.iter().map(|&(ds, steps, met)| RingEvent { ds, steps, met: hex(met) }).collect();
            let wear: Vec<WearEvent> = sc.wear.iter().map(|&(ds, state)| WearEvent { ds, state }).collect();
            let alerts: Vec<AlertEvent> = sc.alerts.iter().map(|&(ds, sit)| AlertEvent { ds, sit }).collect();
            let day_tz: BTreeMap<String, f64> = if use_day_tz {
                sc.day_tz.iter().map(|&(d, z)| (d.to_string(), z)).collect()
            } else {
                BTreeMap::new()
            };
            let readiness: BTreeMap<String, u32> = sc.readiness.iter().map(|&(d, r)| (d.to_string(), r)).collect();
            let (age, gender, weight_kg, bmr) = sc.user;
            compute_stateless(&StatelessInputs {
                anchors: &anchors,
                events: &events,
                wear: &wear,
                alerts: &alerts,
                newest_ds: Some(sc.newest_ds),
                sleeps: sc.sleeps,
                user: User { bmr, age, gender, weight_kg, target_mult: 100, target_type: 0 },
                tz: sc.tz,
                day_tz: &day_tz,
                readiness: &readiness,
                pad_history: true,
            })
        }

        #[test]
        fn stateless_days_equal_the_python_reference() {
            let mut days = 0;
            for (n, sc) in SCENARIOS.iter().enumerate() {
                let got = run(sc, true);
                for e in sc.expect.iter() {
                    let r = got.get(e.date).unwrap_or_else(|| panic!("scenario {n}: {} missing", e.date));
                    let s = &r.stats;
                    assert_eq!(s.classes, e.classes, "scenario {n} {} classes", e.date);
                    assert_eq!(s.steps, e.steps, "scenario {n} {} steps", e.date);
                    assert_eq!(s.eq_distance, e.eq_distance, "scenario {n} {} eq distance", e.date);
                    assert_eq!(s.active_cal, e.active_cal, "scenario {n} {} active kcal", e.date);
                    assert_eq!(s.total_cal, e.total_cal, "scenario {n} {} total kcal", e.date);
                    assert_eq!(s.non_wear, e.non_wear, "scenario {n} {} non-wear", e.date);
                    assert_eq!(s.day.sedentary_min as u32, e.sedentary_min, "scenario {n} {} hourly sedentary", e.date);
                    assert_eq!(r.score, e.score, "scenario {n} {} score", e.date);
                    for k in 0..7 {
                        assert!(
                            (r.contributors[k] - e.contributors[k]).abs() < 1e-3,
                            "scenario {n} {} contributor {k}",
                            e.date
                        );
                    }
                    days += 1;
                }
            }
            assert!(days >= 10);
        }

        #[test]
        fn day_zone_moves_the_frame() {
            // the same events with the stored zone of one day dropped give a different set of minutes
            let sc = SCENARIOS.iter().find(|s| !s.day_tz.is_empty()).unwrap();
            let (with, without) = (run(sc, true), run(sc, false));
            let date = sc.day_tz[0].0;
            let (a, b) = (&with[date].stats, &without[date].stats);
            assert!(a.eq_distance != b.eq_distance || a.classes != b.classes);
        }
    }


    #[test]
    fn placeholder_history_is_a_a_b() {
        let p = pad_days(28, 2);
        assert_eq!(p.len(), 9);
        assert_eq!((p[0].eq_distance, p[1].eq_distance, p[2].eq_distance, p[3].eq_distance), (5000, 10000, 10000, 5000));
        assert_eq!(p[8].eq_distance, 10000);
    }
}
