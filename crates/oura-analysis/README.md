# oura-analysis

Metric algorithms over the events a ring writes: HRV, SpO2, baselines, sleep summary, sleep, readiness and
activity scores with their input assembly, breathing rate, IBI correction, bedtime detection, sleep regularity,
rest mode, and the sleep-staging pipeline. All functions are pure (no I/O, no global state) and take
slices; the caller reads stored events (for example with `oura-store`) and supplies the arrays. The crate
depends on `oura-protocol` only in `ported/real_steps.rs`, which decodes step events with its parsers.

Algorithm descriptions are in [`docs/algorithms/`](../../docs/algorithms/README.md). Paths below are relative to
`src/ported/`. All public items live under `oura_analysis::ported`.

## Call sequence for one day

| Step | Item | Input | Output |
| --- | --- | --- | --- |
| 1 | `bedtime::Detector::detect_bedtime` | 30 s motion bytes and timestamps, 60 s temperatures, resting-HR periods, previous bedtimes | `Vec<Period>` (start, end, major) |
| 2 | `sleepstaging::pipeline::correct_beats`, then `stage_night_default` (or `stage_night_with`) | beats of the bedtime window, motion rows, temperatures, `[age, BMI, sex]`, bedtime | `Output` (hypnogram, probabilities) |
| 3 | `sleep_inputs::night_from_stages` (or `scan`), `period_from_scan`, `app_sleep_score` | stage codes and motion placed on the 30 s grid | `NightScore`, `Scan`, `SleepPeriod`, `DailySleep` (score, 7 contributors) |
| 4 | `activity_day::compute_stateless` (or `actinfo::ActInfo` step by step), `activity_score::score` | `time_sync` anchors, `activity_information`, `wear_event` and `alert_event` rows, sleep periods | `DayResult` per date: `Stats` with the day record, activity score, contributors |
| 5 | `daily::DailyScorer::step` | the day's `SleepPeriod` list, the 8 previous-day words | `DailyResult` (sleep and readiness scores, contributors) |
| 6 | `sri::SriState` (or `DailyScorer::track_sri`) | each scored main period | sleep regularity value for step 5 |

`oura-cli` does not depend on this crate. A caller that needs the whole sequence assembles the arrays itself:
assigning nights to days, choosing the main period, and converting decoded events to the arrays above is not
part of the crate.

## Sleep staging

```rust
pub fn stage_night_with(
    clf: &dyn StageClassifier,
    inp: &Inputs,
    rerun_after_long_wake: bool,
    false_nap_check: bool,
) -> Result<Output, StagingError>
```

Both flags are normally `true`. `stage_night_default(inp, rerun, nap)` stages with the classifier bundled in the
crate and needs no file. `stage_night(clf: &SleepClassifier, ...)` has the same behavior as `stage_night_with` for
the blob classifier.

- `Inputs` borrows its arrays: `acm_values: &[[f32; 6]]` with `acm_timestamps: &[i64]` (MAD iqr, max, mean, then
  dANGLEZ iqr, max, mean per 30 s epoch), `ibi_values: &[[f32; 3]]` (`[ibi_ms, amplitude, validity]`) with
  `ibi_timestamps`, `temp_values: &[f32]` with `temp_timestamps`, `scalars: [f32; 3]` (`[age, BMI, sex]`; sex 2
  male, 1 female, -1 unknown; `events::profile_scalars` builds it), and `bedtime: [i64; 2]`. All timestamps are
  Unix milliseconds.
- `correct_beats(&[(ts_ms, ibi_ms, amplitude)]) -> (Vec<i64>, Vec<[f32; 3]>)` runs IBI correction and returns the
  timestamps and rows for `ibi_values`.
- `Output.stages` has rows `[label, p_deep, p_light, p_rem, p_wake]` (labels 1 deep, 2 light, 3 REM, 4 wake) at
  `timestamps_ms`. `five_minute_stages(&out)` gives the 5 minute aggregate.
- `events::collect` builds the window's `StagerInputs` from stored events `(ring ds, tag, decoded json,
  captured_unix)`, converting ring time with the latest ring-written `time_sync` captured at or before each event.

### Bundled classifier

The crate contains a classifier, `sleepstaging::bundled::bundled_classifier()`, which `stage_night_default` uses.
It is a gradient-boosted tree ensemble (4 classes, 800 rounds of trees with up to 32 leaves, 99,200 nodes, 1,846,470
bytes) in `data/sleep_stage_gbt.bin`, trained by the maintainer on simulated nights only. On 14 nights from one user,
it agrees with the official app's per-30-second stages on 92.6% of epochs (pooled; Cohen's kappa 0.89). These figures measure
agreement with the official app's stages for one user; they are not accuracy against polysomnography and do not cover other users.

### Supplying another classifier

The pipeline computes 182 features per 30 s epoch and passes each row to a `StageClassifier`
(`sleepstaging::classifier`). To use a model other than the bundled one, either implement the trait:

```rust
pub trait StageClassifier {
    fn n_features(&self) -> usize { 182 }
    fn n_classes(&self) -> usize { 4 }
    fn predict_proba(&self, features: &[f32; 182]) -> [f32; 4];   // deep, light, REM, wake
    fn margins(&self, features: &[f32; 182]) -> Option<[f64; 4]> { None }
    fn predict_proba_rows(&self, rows: &[f32]) -> Vec<[f32; 4]>;  // default: row by row
}
```

or load a file in the blob format with `classifier::load_classifier(path)` or `load_classifier_bytes(bytes)`, which
return `Box<dyn StageClassifier + Send + Sync>` and reject malformed input with a `ClassifierError`. The format and
the entry points that work without a classifier (`stage_features`, `stage_night_or_features`,
`stage_night_required`) are in [`docs/algorithms/staging-classifier.md`](../../docs/algorithms/staging-classifier.md).
Pass the result to `stage_night_with`. Construct a classifier once per process and share it.

## Sleep score

`sleep_inputs::night_to_score(&Night) -> Option<NightScore>` scores a night given packed stage nibbles and motion
bytes; `None` means fewer than 10 epochs were scanned. `night_from_stages(age, bed_start_s, bed_end_s, tz_min,
stages, motion, chronotype)` takes one stage per 30 s epoch and motion as `(unix seconds, motion seconds)` points,
which is the stager's output form. `scan` returns the `Scan` counters. `period_from_scan(&Scan, bed_start_s, tz_min,
is_main)` builds the `readiness_inputs::SleepPeriod` that step 5 takes, and `app_sleep_score(&Scan, age,
bed_start_s, tz_min, chrono)` returns the daily `DailySleep`. `chrono` is `(type, ideal midpoint in seconds)`;
`None` scores timing against a fixed reference curve. Fields of `SleepPeriod` that the scan does not give
(`highest_temp`, `lowest_hr`, `lowest_hr_time`, `rmssd`) are filled by the caller; `lowest_hr::calculate_lowest_hr`
gives `lowest_hr` and `lowest_hr_time`.

## Activity, stateless day

`activity_day::compute_stateless(&StatelessInputs) -> BTreeMap<String, DayResult>` is keyed by `YYYY-MM-DD`:

```rust
pub struct StatelessInputs<'a> {
    pub anchors: &'a [Anchor],       // ring-written time_sync events: { ds, unix } (not phone anchors)
    pub events: &'a [RingEvent],     // activity_information: { ds, steps (first body byte), met (the MET bytes) }
    pub wear: &'a [WearEvent],       // wear_event: { ds, state }
    pub alerts: &'a [AlertEvent],    // alert_event: { ds, sit (alert type 1) }
    pub newest_ds: Option<i64>,      // ring time of the newest event of any kind; None: the newest of the above
    pub sleeps: &'a [(f64, f64)],    // Unix (start, end); a period rests the days its start and its end fall in
    pub user: User,                  // gender: 2 male, 1 female, 3 other, 0 unknown
    pub tz: f64,                     // zone, hours east of UTC
    pub day_tz: &'a BTreeMap<String, f64>,   // dates whose day frame uses another zone
    pub readiness: &'a BTreeMap<String, u32>, // readiness score by date (80 when absent): scales the target
    pub pad_history: bool,           // two placeholder days before the first ring day
}
```

It dates each event with the latest ring `time_sync` at or before it in ring order, places the MET bytes by
absolute minute over a day that starts at byte 9, writes non-wear (byte 1) from the wear events, seeds the hourly
sedentary counters from the previous date, rests every sleep period of the day, and scores each day against the
nine before it. A caller that has already resolved times can use `ActInfo` directly:

```rust
let mut a = ActInfo::new(User { bmr, age, gender, weight_kg, target_mult: 100, target_type: 0 }, true);
a.stateless = true;
a.seed_hourly(&prev_day_met_bytes);   // previous day's per-minute bytes; no-op when the first byte is 0
a.set_target(readiness_pct);          // percent; 80 is used for 0
a.ranges = Some(sleep_ranges);        // every sleep period in minutes from the day start (None: the last one)
a.add_activity(n as u32, steps, &cur[..n]);   // n = min(minutes so far, 1439); cur starts at 9, non-wear 1
if day_complete { a.finish_day(cur[1439]); }
let stats = a.statistics(0);          // stats.day is the activity_score::Day for today
let score = activity_score::score(&days_oldest_to_newest, ring_day);
```

`User.bmr` comes from `metabolic::bmr_kcal`. The event-driven `activity_day::compute(&Inputs)` keeps its earlier
signature; it has no previous-day seeding, no minimum byte, one sleep period per day, and no per-day zone.

## Bedtime

`Detector::new()` then `detect_bedtime(motion, motion_ts, temp, temp_ts, hr_periods, avg_temp, previous,
report_naps)` returns every detected `Period`; `Err(InvalidArguments)` means no motion, no resting-HR period, a
zero `avg_temp`, or mismatched lengths. `avg_temp` must be nonzero but is otherwise unused. After a call,
`clear_minor_periods_ms(a, b)` and `candidates()` act on the state the call left, so keep one `Detector` per sync if
they are needed. The free function `bedtime::detect_bedtime` uses a fresh detector, and `merge_periods` merges
period records. `bedtime_inputs::detect_open` runs the whole chain for a night the ring has not declared;
`Events::from_decoded`, `assemble`, `window_start`, `previous` and `run` are its steps.

## Daily scores

`DailyScorer::new(&means, &devs)` (also `Default`, which uses `DEFAULT_MEANS` and `DEFAULT_DEVS`) keeps the
baseline, the 14 day history and the day counter between calls. `step(day_utc, age, &periods, &prev_day, chrono,
sri)` returns a `DailyResult` whose `outputs` is `None` when `readiness_inputs::check_inputs` rejects the data (no
period, a period without sleep, no main period, or previous-day words out of range). `step_with_history` takes a
caller-built history indexed by date. `track_sri()` makes the scorer update an `SriState` with each main period and
use its value; otherwise the `sri` argument applies, and `daily::SRI` is the constant used when the feature is
off. `with_baseline` starts from another day-0 baseline.

## Metabolic helpers

`metabolic::bmr_kcal(age, weight_kg, sex)` (`sex` 0 male, 1 female, other the mean of both) returns whole kcal.
`bmr_schofield(age: f64, sex: u8, kg: f64) -> f64` has the same value; `bmr_schofield_band` is the unblended band
version. `vo2max_jackson(age_years, female, weight_kg)` and `steps_to_meters(steps)` are the other helpers.

## Smaller items

| Item | Purpose |
| --- | --- |
| `hrv::rmssd(&[u16]) -> Option<f64>` | RMSSD of an IBI series in ms; `None` for fewer than 2 intervals |
| `spo2::spo2_simple(r, a, b, c)` | `a + b*r + c*r^2` clamped to 0..120; the crate holds no constants (see [`docs/spo2-calibration.md`](../../docs/spo2-calibration.md)) |
| `breathing_rate::breathing_rate(bedtime_ms, &ibi, &ibi_ts_ms, &motion) -> (f32, f32, f32)` | night breathing rate, its variability (breaths/min) and heart rate (beats/min); zeros under 361 epochs |
| `ibi_correction::correct_ibis(&[(ts, ibi, amp)]) -> Vec<IbiEvent>` | IBI cleaning; `IbiCorrection::feed` is the streaming form, `correct_stateless` the batch form |
| `lowest_hr::calculate_lowest_hr(&[u8], interval) -> Option<(u32, u32)>` | lowest HR and its time from 5 minute heart rates |
| `steps_motion_decoder::decode(&timestamps, &records) -> Vec<GaitRow>` | 27 quantized gait columns per `0x7e`/`0x7f` record to three 10 s rows of 11 features; no file needed |
| `step_counter::StepCounter::{from_file, count}` | steps per 10 s row or per resampled interval from gait rows and `0x47` motion rows; `from_file` reads a weights file you supply ([`docs/models-you-supply.md`](../../docs/models-you-supply.md)) |
| `real_steps::count_steps(&counter, events, to_unix_ms, range, &Config)` | step counts per minute from raw `0x7e`, `0x7f` and `0x47` events |
| `sleep_summary` | night summary values from ring events: RMSSD averages, nightly temperature, HR average, slot grids |
| `temperature::nightly_temperature(&[u16]) -> Option<i16>` | nightly temperature in centi-degrees |
| `baseline::Baseline`, `baseline::update_lt_mean_and_dev` | personal baseline update |
| `sleep_debt::sleep_debt(&actual_s, &need_s, &SleepDebtConfig)` | sleep debt in seconds |
| `rest_recovery` | rest-mode state from a period, the ring-event state machine, and the 30-period history |
| `high_low_hr::run`, `daily_medians::run`, `cva::Head::run` | pure functions with the input validators of the matching model runners; see [`docs/model-runners.md`](../../docs/model-runners.md) |

## Tests

`cargo test -p oura-analysis` runs unit tests and the generated `*_vectors.rs` tables beside the sources. Tests
that need an optional classifier or vector file print `skip` and pass when it is absent.
