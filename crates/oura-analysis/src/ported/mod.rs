//! **Ported algorithms — reverse-engineered from Oura's own software.**
//!
//! Every module here reproduces a computation that Oura performs, ported from the
//! decompiled on-device `ecore` engine (`libappecore.so`). Each function's
//! doc-comment cites its source as `<ecore function> @ <address>`. These are *not*
//! our inventions: the intent is bit-for-bit (or as close as the recovered
//! constants allow) fidelity to Oura's results. Anything we could not fully recover
//! is flagged in the module docs and in `docs/algorithms/`.

pub mod actinfo;
pub mod activity_day;
pub mod activity_score;
pub mod baseline;
pub mod bedtime;
pub mod bedtime_inputs;
pub mod breathing_rate;
pub mod daily;
pub mod hrv;
pub mod ibi_correction;
pub mod lowest_hr;
pub mod metabolic;
pub mod night_summary;
pub mod readiness;
pub mod readiness_inputs;
pub mod real_steps;
pub mod rest_recovery;
pub mod readiness_history;
pub mod sleep;
pub mod sleep_classifier;
pub mod sleep_debt;
pub mod sleep_inputs;
pub mod sleep_score;
pub mod sleep_summary;
pub mod sleepstaging;
pub mod spo2;
pub mod sri;
pub mod step_counter;
pub mod steps_motion_decoder;
pub mod temperature;
pub mod window_series;
