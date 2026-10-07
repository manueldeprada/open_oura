//! The ring's heart-rate, MET and motion records inside a time window, for plotting an activity
//! or a workout afterwards. Read from decoded events `(ring ds, tag, json, captured_unix)` with the
//! caller's clock; nothing here talks to the ring.
//!
//! * `0x80` heart-rate events hold `hr_bpm` beats; they are averaged per minute.
//! * `0x50` activity events hold `met` values, one per minute, the first at the event's minute.
//! * `0x47` motion events hold `motion_seconds`.

use serde_json::Value;
use std::collections::BTreeMap;

const TAG_MOTION: u8 = 0x47;
const TAG_ACTIVITY: u8 = 0x50;
const TAG_HR: u8 = 0x80;
/// Events up to this many seconds outside the window are read (an event is stamped when written).
const MARGIN_S: f64 = 120.0;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct WindowSeries {
    /// `(unix seconds of the minute's middle, average bpm)`
    pub hr: Vec<(f64, f64)>,
    /// `(unix seconds of the minute, MET)`
    pub met: Vec<(f64, f64)>,
    /// `(unix seconds, motion seconds)`
    pub motion: Vec<(f64, f64)>,
}

pub fn window_series(
    events: &[(i64, u8, String, i64)],
    unix_s: &dyn Fn(i64, i64) -> Option<f64>,
    start: f64,
    end: f64,
) -> WindowSeries {
    let mut hr_minutes: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    let mut out = WindowSeries::default();
    for (ds, tag, json, cu) in events {
        if !matches!(*tag, TAG_MOTION | TAG_ACTIVITY | TAG_HR) {
            continue;
        }
        let Some(t) = unix_s(*ds, *cu) else { continue };
        if t < start - MARGIN_S || t > end + MARGIN_S {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(json) else { continue };
        match *tag {
            TAG_HR => {
                if let Some(a) = v["hr_bpm"].as_array() {
                    let good = a.iter().filter_map(Value::as_f64).filter(|b| *b > 20.0 && *b < 250.0);
                    hr_minutes.entry((t / 60.0).floor() as i64).or_default().extend(good);
                }
            }
            TAG_ACTIVITY => {
                if let Some(a) = v["met"].as_array() {
                    let base = (t / 60.0).round();
                    for (i, m) in a.iter().enumerate() {
                        let mt = (base + i as f64) * 60.0;
                        if let (true, Some(m)) = (mt >= start && mt <= end, m.as_f64()) {
                            out.met.push((mt, m));
                        }
                    }
                }
            }
            _ => {
                if let (true, Some(s)) = (t >= start && t <= end, v["motion_seconds"].as_f64()) {
                    out.motion.push((t, s));
                }
            }
        }
    }
    for (k, vals) in hr_minutes {
        let t = k as f64 * 60.0 + 30.0;
        if t >= start && t <= end && !vals.is_empty() {
            out.hr.push((t, vals.iter().sum::<f64>() / vals.len() as f64));
        }
    }
    out.met.sort_by(|a, b| a.0.total_cmp(&b.0));
    out.motion.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_three_series_inside_the_window() {
        let ev = |ds, tag, j: &str| (ds, tag, j.to_string(), 0);
        let events = vec![
            ev(600, TAG_HR, r#"{"hr_bpm":[60,70,0,300]}"#),
            ev(610, TAG_HR, r#"{"hr_bpm":[80]}"#),
            ev(1200, TAG_ACTIVITY, r#"{"met":[1.5,2.0,2.5]}"#),
            ev(1250, TAG_MOTION, r#"{"motion_seconds":9}"#),
            ev(90_000, TAG_MOTION, r#"{"motion_seconds":1}"#),
        ];
        let s = window_series(&events, &|ds, _| Some(ds as f64 / 10.0), 30.0, 200.0);
        assert_eq!(s.hr, vec![(90.0, 70.0)]);
        assert_eq!(s.met, vec![(120.0, 1.5), (180.0, 2.0)]);
        assert_eq!(s.motion, vec![(125.0, 9.0)]);
    }
}
