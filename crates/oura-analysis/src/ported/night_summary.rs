//! The night summary the daily functions read from a bedtime period: the lowest heart rate and its
//! time, the average RMSSD, the highest temperature and the time in bed. Built from the ring's own
//! event series: motion, temperature
//! and HR/HRV events resampled to 30 s, 60 s and 300 s slots from the bedtime start
//! ([`super::sleep_summary::grid`]), then [`super::lowest_hr::calculate_lowest_hr`],
//! [`super::sleep_summary::calculate_average_rmssd`] and [`super::sleep_summary::nightly_temperature`].
//!
//! Python reference: `tools/score_daily_exact.py` (`night_summary`) and `tools/sleep_summary.py`
//! (`night_inputs`); the tests use vectors from `tools/gen_night_summary_vectors.py`.

use super::lowest_hr::calculate_lowest_hr;
use super::sleep_summary::{
    calculate_average_rmssd, grid, hrv_items, nightly_temperature, slot_offset, stage_rows, time_in_bed,
};
use serde_json::Value;

/// Python's `round` on a float: half to even.
fn round_even(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
        (r - x.signum()) as i64
    } else {
        r as i64
    }
}

/// The ring events of `motion_event`, `temp_event` and `hrv_event` with Unix times in seconds.
#[derive(Debug, Default, Clone)]
pub struct RingSeries {
    /// `(time, motion_seconds)`
    pub motion: Vec<(f64, i64)>,
    /// `(time, first of the temperatures in degrees C)`
    pub temp: Vec<(f64, f64)>,
    /// `(time, heart rates, RMSSD values)`, one value per 5 minutes ending at the event time
    pub hrv: Vec<(f64, Vec<u8>, Vec<u8>)>,
}

impl RingSeries {
    /// Collects the series from decoded events `(ring ds, tag, json, captured_unix)`; `unix_s` places
    /// an event on the calendar (`None` drops it).
    pub fn from_decoded(events: &[(i64, u8, String, i64)], unix_s: &dyn Fn(i64, i64) -> Option<f64>) -> Self {
        let mut s = RingSeries::default();
        for (ds, tag, json, cu) in events {
            if !matches!(*tag, 0x46 | 0x47 | 0x5d) {
                continue;
            }
            let Some(t) = unix_s(*ds, *cu) else { continue };
            let Ok(v) = serde_json::from_str::<Value>(json) else { continue };
            match *tag {
                0x47 => s.motion.push((t, v["motion_seconds"].as_i64().unwrap_or(0))),
                0x46 => {
                    if let Some(x) = v["temps_c"].get(0).and_then(Value::as_f64) {
                        s.temp.push((t, x));
                    }
                }
                _ => {
                    let bytes = |k: &str| -> Option<Vec<u8>> {
                        v[k].as_array()?.iter().map(|x| x.as_u64().map(|y| y.min(255) as u8)).collect()
                    };
                    if let (Some(hr), Some(rm)) = (bytes("hr_bpm"), bytes("rmssd_ms")) {
                        if hr.len() == rm.len() {
                            s.hrv.push((t, hr, rm));
                        }
                    }
                }
            }
        }
        s
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NightSummary {
    /// Lowest heart rate (0 without data) and its time in seconds from the bedtime start.
    pub lowest_hr: u32,
    pub lowest_hr_time: u32,
    /// Average RMSSD over the slots with a stage string (0 without one).
    pub rmssd: u32,
    /// Highest temperature in centidegrees.
    pub highest_temp: u16,
    pub time_in_bed: i64,
    /// Motion seconds per 30 s slot (0 for an empty slot), capped at 255.
    pub motion: Vec<u8>,
}

/// `stages` are the night's stages per 30 s epoch from the bedtime start (1 deep, 2 light, 3 REM,
/// 4 awake); with `None` the RMSSD is 0.
pub fn night_summary(series: &RingSeries, start_ms: i64, end_ms: i64, stages: Option<&[u8]>) -> NightSummary {
    let ms = |t: f64| round_even(t * 1000.0);
    let motion_items: Vec<(i64, u8)> =
        series.motion.iter().map(|&(t, v)| (ms(t), v.clamp(0, 255) as u8)).collect();
    let motion: Vec<u8> = grid(&motion_items, start_ms, end_ms, 30).iter().map(|x| x.map_or(0, |y| y.1)).collect();
    let temp_items: Vec<(i64, u16)> = series
        .temp
        .iter()
        .map(|&(t, v)| (ms(t), round_even(v * 100.0).clamp(0, 0xFFFF) as u16))
        .collect();
    let temps: Vec<u16> = grid(&temp_items, start_ms, end_ms, 60).iter().map(|x| x.map_or(0, |y| y.1)).collect();
    let hv = grid(&hrv_items(&series.hrv, end_ms), start_ms, end_ms, 300);
    let first = hv.first().and_then(|x| x.map(|y| y.0)).unwrap_or(start_ms);
    let slots: Vec<(u8, u8)> = hv.iter().map(|x| x.map_or((0, 0), |y| y.1)).collect();
    let hr: Vec<u8> = slots.iter().map(|p| p.0).collect();
    let lowest = if hr.is_empty() { None } else { calculate_lowest_hr(&hr, 300) };
    let rmssd = match stages {
        Some(codes) if !hr.is_empty() => {
            let rows = stage_rows(codes, start_ms, start_ms, end_ms);
            let rm: Vec<u8> = slots.iter().map(|p| p.1).collect();
            calculate_average_rmssd(&rows, &rm, rm.len(), slot_offset(first, rm.len(), start_ms.div_euclid(1000))).0
        }
        _ => 0,
    };
    NightSummary {
        lowest_hr: lowest.map_or(0, |l| l.0),
        lowest_hr_time: lowest.map_or(0, |l| l.1),
        rmssd,
        highest_temp: nightly_temperature(&temps),
        time_in_bed: time_in_bed(start_ms, end_ms),
        motion,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Generated by tools/gen_night_summary_vectors.py from the Python reference.
    include!("night_summary_vectors.rs");

    #[test]
    fn matches_python_reference() {
        assert!(!NIGHT_CASES.is_empty());
        for (i, c) in NIGHT_CASES.iter().enumerate() {
            let series = RingSeries {
                motion: c.motion.to_vec(),
                temp: c.temp.to_vec(),
                hrv: c.hrv.iter().map(|(t, h, m)| (*t, h.to_vec(), m.to_vec())).collect(),
            };
            let got = night_summary(&series, c.start_ms, c.end_ms, c.stages);
            assert_eq!(
                (got.lowest_hr, got.lowest_hr_time, got.rmssd, got.highest_temp, got.time_in_bed),
                (c.lowest_hr, c.lowest_hr_time, c.rmssd, c.highest_temp, c.time_in_bed),
                "case {i}"
            );
        }
    }

    #[test]
    fn collects_series_from_events() {
        let ev = |ds, tag, j: &str| (ds, tag, j.to_string(), 0);
        let events = vec![
            ev(10, 0x47, r#"{"motion_seconds":8}"#),
            ev(20, 0x46, r#"{"temps_c":[31.5,30.0]}"#),
            ev(30, 0x5d, r#"{"hr_bpm":[60,61],"rmssd_ms":[30,40]}"#),
        ];
        let s = RingSeries::from_decoded(&events, &|ds, _| Some(1000.0 + ds as f64 / 10.0));
        assert_eq!(s.motion, vec![(1001.0, 8)]);
        assert_eq!(s.temp, vec![(1002.0, 31.5)]);
        assert_eq!(s.hrv, vec![(1003.0, vec![60, 61], vec![30, 40])]);
    }
}
