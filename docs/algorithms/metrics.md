# Metric method notes

What each metric in `crates/oura-analysis` computes, its inputs and outputs, and its parameters. The code in
`crates/oura-analysis/src/ported/` is the reference; this page describes the behavior of that implementation.
Status per metric is in [README.md](README.md); call signatures are in
[`crates/oura-analysis/README.md`](../../crates/oura-analysis/README.md).

**Conventions.** Times are Unix seconds or milliseconds as stated; ring times are deciseconds. Sleep stages use
codes 1 deep, 2 light, 3 REM, 4 awake (0 is missing and reads as awake). A "30 s epoch" is one stage or one motion
byte. Contributor values are integers in 1..100 unless stated.

## Sleep score

`sleep_score` combines nine inputs into a total in 0..100 and seven reported contributors. `sleep_inputs` derives
the inputs from a night's stage and motion series.

**Combiner.** `total = round(sum(w_i * c_i) / 100)` over nine curve values `c_i` in 0..100, with the sum taken
before the restfulness rescaling below.

| i | weight | input | curve X knots |
| --- | --- | --- | --- |
| 0 | 35 | total sleep (minutes) | age dependent |
| 1 | 10 | deep sleep (minutes) | age dependent |
| 2 | 10 | REM sleep (minutes) | age dependent |
| 3 | 10 | efficiency (percent) | 30, 60, 90, 95 |
| 4 | 10 | latency, `abs(minutes - 15)` | 60, 28, 2, 0 |
| 5 | 5 | wake-ups per hour of sleep | 240, 128, 16, 0 |
| 6 | 5 | motion epochs, percent of sleep | 80, 50, 25, 15 |
| 7 | 5 | motion bursts per hour of sleep | 96, 48, 16, 0 |
| 8 | 10 | timing | 390, 240, 100, 90, or the timing function |

Every curve maps its four knots to `[0, 25, 95, 100]` by linear interpolation in integer arithmetic (ascending
segments round half up, descending segments round half down). Ascending knots give 0 at or below the first knot and
clamp to 100 above the last; descending knots mirror this. `limits_v2(age)` computes the knots of inputs 0 to 2 from
the age in years (total sleep, deep, and REM knots shrink with age).

**Reported contributors.** Total sleep, deep, REM, efficiency, latency, restfulness (inputs 5 to 7, scaled by about
`c / 3` each and summed) and timing. Each is forced into 1..100.

**Timing.** Without a chronotype the input is `abs(mid - 90)` with `mid` the sleep midpoint in minutes on a signed
16-bit clock. With a chronotype `(type, ideal midpoint seconds)` the contributor is a piecewise-linear function of
the midpoint around the ideal one (100 inside a plateau, falling to 25 at 150 minutes outside it and to 0 at 280
minutes outside it; `timing_score` has the breakpoints). The chronotype is not available from the ring.

**Input derivation.** `scan` visits the 30 s epochs of the bedtime window (at most 0x707 rows; no score if fewer
than 10 are scanned):

| quantity | rule |
| --- | --- |
| time in bed | rows visited |
| deep, light, REM | counts of stages 1, 2, 3 |
| latency | index of the 10th consecutive asleep epoch minus 9 (1 if the index is under 10, 0xff if over 264) |
| wake-ups | awake for 2 consecutive epochs after sleep onset, minus 1 if the scan ends awake |
| motion epochs | epochs with nonzero motion and a sleep stage |
| motion bursts | rises of 24 or more in the motion byte over the larger of the bytes 2 and 3 epochs earlier, with a calm-down counter that re-arms the detector |
| midpoint | mean epoch index over asleep epochs, added to the local clock minute of the bedtime start |

The daily value (`app_sleep_score`) scores the period built by `period_from_scan`: durations in seconds, time in
bed one epoch less than the row count, and latency and midpoint one epoch more. With several periods in a day, the
running sums are scored again and the timing contributor is the sleep-weighted mean of the periods' timing values.
Chronotype `None` selects the fixed reference curve, which can differ from a chronotype-based value by tens of
points.

## Activity score and day statistics

`activity_score::score(days, ring_day)` combines seven contributors from up to nine earlier day records plus
today's; `actinfo` computes a day's statistics from per-minute MET bytes; `activity_day` handles day bookkeeping.

**Per-minute input.** One byte per minute, 1440 per day. A byte `v` is MET x 10 for `v < 128` and `12.8 + 0.2
(v - 128)` MET otherwise; byte 1 marks non-wear. `activity_information` (tag `0x50`) events carry bytes for the
minutes before the event time. The activity day starts at 04:00 local time.

**Minute classes.** For minute `i` with MET `m_i` and the five-minute window sum `W_i`, the class is vigorous if
`m_i > vig` or `W_i > 5 vig`, moderate if `m_i > mod` or `W_i > 5 mod`, light and sedentary use fixed lower limits on `m_i` and `W_i` (`met_to_class`), else rest. `mod` and `vig` depend linearly on
`abs(age - 25)` and on the gender code (`actinfo::met_limits`; defaults 4.0 and 7.0 MET). A sedentary minute inside
a sleep period is rest. Hourly sedentary counters (24 per day) are written as the day progresses.

**Day statistics (`Stats`).**

| output | definition |
| --- | --- |
| `eq_distance` | summed equivalent meters per minute: 0 below 1.9 MET, linear in MET above |
| classes | minutes in rest, sedentary, light, moderate, vigorous; `non_wear` counted separately |
| `met_min` | MET-minutes above 1.5 MET, summed per class |
| `active_cal`, `total_cal` | from the minute MET values and the BMR (Schofield) |
| `steps` | first body byte of the events, summed |
| targets | daily target in equivalent meters from age, gender and the readiness percent; converted to calories and steps |

The daily target scales with the readiness score of the date: `pct / 100` up to 69, 1 from 70 to 84, `2 pct / 100
- 0.5` from 85 (80 when no score is known). `target_reached` is 0 when the target is 0, 1 when met, else 2.

**Stateless day path.** The reference computation for a day uses three rules that differ from feeding events one
by one: every byte starts at 9 (0.9 MET) except non-wear; the 24 hourly sedentary counters are seeded from the
previous day's bytes (so a partial day keeps the previous day's counts for hours it has not reached); and a
completed day treats minute 1439 only as a non-wear flag. Event times come from the latest ring-written
`time_sync` at or before the event, and events of one day's batch are dated again with the syncs that bracket the
batch. Wear events write byte 1 from the first non-finger state (anything but states 3 to 6) to the next finger
state.

**Combiner.** `round(sum(w_i * c_i) / 100)`.

| i | weight | contributor | driver |
| --- | --- | --- | --- |
| 0 | 25 | meet daily targets | days that reached the target, over the last 7 days |
| 1 | 15 | stay active | today's sedentary minutes: 300 maps to 100, 1020 or more to 0 |
| 2 | 10 | move every hour | today's inactivity alerts: 0 maps to 100, 10 or more to 0 |
| 3 | 10 | training frequency | days with a hard session, over the last 7 days |
| 4 | 15 | training volume | recency-weighted moderate plus vigorous MET-minutes over the last 7 days |
| 5 | 10 | recovery time, load | hard days among the last 7 |
| 6 | 15 | recovery time, run | consecutive hard days ending yesterday |

Curves interpolate linearly between knots with `Y = [0, 25, 95, 100]` unless stated. A day is hard for contributor
3 when its vigorous MET-minutes or moderate plus vigorous MET-minutes exceed thresholds derived from the age and
gender MET values, and for contributors 5 and 6 when its vigorous minutes or equivalent distance exceed fixed
limits. A user with fewer than 7 days of data uses the new-user rule: today counts as having reached the target when
no day has. Gender codes: 2 male, 1 female, 3 other, 0 unknown. The reported "recovery time" contributor combines
contributors 5 and 6 with weights 10 and 15 in this reading; the display rule is not confirmed.

## Readiness

`readiness::readiness_calculate` produces the classic eight-contributor score; `readiness_v2` produces a second,
grouped score. `readiness_inputs` assembles both from a day's sleep periods, the previous day, 14-day histories,
the baseline, and the rest-mode state. `daily::DailyScorer` chains days.

**Classic contributors.** Each is a piecewise-linear function of a z-score against a personal baseline (mean and
deviation, stored times 8), floored at 1 and capped at 100, and 0 when its inputs are unavailable.

| contributor | driver |
| --- | --- |
| last night | the day's sleep score against its baseline |
| sleep balance | recency-weighted mean of 14 nightly sleep durations |
| last day activity | previous day's distance, vigorous minutes and sedentary minutes, with wear correction |
| activity balance | recency-weighted mean of 14 daily distances against the distance baseline |
| resting HR | lowest night heart rate (1 to 179 bpm) against its baseline |
| resting HR time | minutes from the lowest heart rate to the end of the bedtime window |
| temperature | highest skin temperature against the baseline (mean and deviation in degrees), lowered by 0.20 degrees on cycle day type 2 (an input supplied by the caller) |
| HRV balance | recency-weighted mean of `100 log2(RMSSD)` over 14 days against its baseline |
| sleep regularity | `sri_calculate`, off when the feature is off |

The 14 recency weights, oldest first, run from 0.02 to 0.72. Balance contributors need data on at least 3 of the last
4 days; HRV balance also needs a baseline age of at least 8 days.

**Score.** `round(sum(w_i c_i) / sum(w_i))` over contributors with `c_i >= 1`. There are two weight sets, A without
a temperature weight and B with one; B also caps the result when the temperature contributor is 74 or lower. In rest
mode, B is compressed and, while recovering, the score blends A and B by the share of the recovery period passed.
The weights are the constants in `readiness.rs`.

**Baselines.** Nine baselines (sleep score, sleep time, resting HR, resting HR time, equivalent distance, vigorous
minutes, sedentary minutes, temperature, HRV) each hold a mean and a deviation in 8x units.
`baseline::update_lt_mean_and_dev` clips the day's value toward the mean (beyond 2.5 deviations it becomes
`m +- d`, between 1.5 and 2.5 deviations `m +- 1.5 d`), moves the mean by `(x - m) / 2` for baseline age under 4
days, `/ 8` under 15, `/ 32` afterward, moves the deviation with divisors 8, 16 and 64, and clamps both to per-item
bounds (`MEAN_MAX`, `MEAN_MIN_DEFAULT`, `DEV_MIN`, `DEV_MAX`). `DEFAULT_MEANS` and `DEFAULT_DEVS` in `daily.rs`
are the cold-start baseline. `daily::baseline_calculate` applies the update once per day.

**Night values.** `sleep_summary` derives the night's RMSSD averages (all, non-REM, REM), highest temperature and
average heart rate from 30 s motion, 60 s temperature and 5 minute HR/HRV series cut to the bedtime window.
`lowest_hr::calculate_lowest_hr` takes the 5 minute heart rates: zeros count as 255, values above `Q3 + 2 (Q3 - Q1)`
are masked, the lowest value is the minimum over slots 6 onward, and its time is the first candidate slot within 3
bpm of the minimum that separates the earlier higher segment from the later lower one. Nightly temperature is the
second-largest of the per-block maxima of a 7-sample median-smoothed series (blocks of 30 samples, kept when their
range is at most 2.49 degrees; at least 4 blocks).

**Night summary from event series.** `night_summary::night_summary(&RingSeries, start_ms, end_ms, Some(&stages))`
returns `lowest_hr`, `lowest_hr_time` (seconds from the bedtime start), `rmssd` (average over the slots with a stage
string; 0 without stages), `highest_temp` (centidegrees), `time_in_bed` and the motion seconds per 30 s slot.
`RingSeries::from_decoded(events, unix_s)` collects the `motion_event` (0x47), `temp_event` (0x46, first
temperature) and `hrv_event` (0x5d) series from decoded events with the caller's clock. The Rust tests compare 10
random nights with `tools/score_daily_exact.py` (`night_summary`); `tools/gen_night_summary_vectors.py` writes them.
The results go into `readiness_inputs::SleepPeriod` (`lowest_hr`, `lowest_hr_time`, `rmssd`, `highest_temp`).
`window_series::window_series` reads the heart-rate, MET and motion records of a time window from stored events, for
plots of an activity or a workout.

**Temperature deviation.** `temperature_deviation` is the highest temperature minus the baseline, valid for
`T <= 4200` centidegrees and at most 200 below the baseline; the trend is the weighted mean (2, 4, 4) of the last
three deviations measured against the 30th percentile of the last 90 highest temperatures.

**Readiness v2.** Inputs are seven scores (total sleep, deep, REM, HRV balance, resting HR, recovery index, body
temperature; missing = -1), up to 255 sleep sessions, 90-day histories of sleep fragmentation index (SFI) and
breathing rate, and the age.

- SFI = fragmenting transitions (sleep to wake, deep to light, REM to light) per hour asleep; WASO is the mean wake
  time after sleep onset over long sleeps, in minutes; breathing rate is the mean over sessions.
- Sub-scores: WASO is logistic in `WASO - (age + 40)` minutes; SFI is linear between the 25th and 75th percentile
  of the last 90 values; breathing rate is scored by z-score against 90 values (at least 14 needed); the breathing
  disturbance index (BDI) is scored against an age-dependent allowance; restorative sleep is the mean of deep and
  REM.
- Combiner: weighted power mean with exponent 0.6 over the available values. Groups: sleep recovery (total sleep
  0.30, deep 0.20, REM 0.20, WASO 0.15, SFI 0.15), nervous system recovery (HRV balance 0.35, resting HR 0.35,
  recovery index 0.30), body strain signal (temperature 0.40, breathing rate 0.40, BDI 0.20). Readiness is the
  group mean with weights 0.40, 0.30, 0.30.

**Daily input assembly.** `readiness_inputs::daily_outputs_stateless` scores the first `k` periods for
`k = 1..n`, which gives per-prefix sleep and readiness scores and deltas. It returns `None` (`check_inputs`) when
there is no period, a period has no sleep, no period is main, or the previous-day distance exceeds 5,000,000 m or a
minute field exceeds 1500. Periods count only when their time lies between 2012-12-06 and 2049.

## Rest mode

`rest_recovery` implements a user-declared rest period. `get_state` maps `(start, end, current UTC midnight)` to a
state: 0 none, 1 starts today, 2 resting, 3 last day of a finished period, 4 recovering, 5 invalid input. With `D`
the number of days between start and end midnights (1 when equal), the recovery phase starts the day after the end
and lasts `min(D, 7)` days; the state returns to 0 when it ends. `Mode::process_event` is a small state machine
over the ring's rest-mode events, and `History` stores up to 30 periods and answers the state at a time.

## Sleep regularity

`sri::SriState` keeps up to 15 day rows of 144 ten-minute slots (0 awake, 1 asleep, 2 missing) filed by sleep date
(`floor((utc + 60 tz) / 86400)`). `update_history(date, start, end)` writes a main sleep period into the rows
(the night as asleep, the rest awake, skipped dates as missing; a longer period for an existing date rewrites it).
`value()` is `sri_calculate(rows, true)`: each slot is compared across days with weights `1..n` (recent days weigh
more), the mean state minus 0.5 is squared and summed over the 144 slots, and the sum is scaled by `100 / 36`.
Every slot needs at least 5 non-missing days. The result is the index in 1..100 with bit 8 set, or 0 when invalid.

## Breathing rate

`breathing_rate(bedtime_ms, ibi, ibi_ts_ms, motion)` returns the night's breathing rate (breaths/min), its
variability (breaths/min) and heart rate (beats/min), each an integer divided by 8. A night under 361 epochs (3 h)
returns zeros. Inputs: IBI values in ms (u16), their timestamps in ms (ascending), and one motion byte per 30 s
epoch starting at `bedtime_ms`.

- **Epoch loop.** For epoch `i` the window is `[T_i - 30 s, T_i]`. IBIs above 1999 ms are dropped. The four most
  recent epochs are pooled (a 2 minute window).
- **Resampling.** Each epoch's IBIs are resampled to 4 Hz by linear interpolation in integer arithmetic.
- **Filter.** The 4 Hz series passes a 12th-order IIR filter, then a peak picker records the interval
  between peaks (capped at 255 samples) and half the peak value.
- **Estimates.** An epoch yields a rate when at least 41 new samples (10.25 s) were produced: over consecutive peak
  pairs with a positive peak and an interval of at least 9 samples, `rate = 240 n / s` breaths/min with `n` pairs
  and interval sum `s`. Otherwise it uses the defaults 13.25 breaths/min and 5.25 variability. Variability is the
  standard deviation of `240 / interval` over the pairs. Heart rate is the mean IBI of the pooled epochs, floored at
  35 bpm.
- **Record flag.** An epoch with at most one IBI in its own range has flag 2 and is skipped. Motion bytes of the
  last four epochs decide flag 1 (usable) or 0 (rejected).
- **Night average.** Up to 1800 records are kept. For `n >= 601` epochs the records from epoch 120 to `n - 120` are
  averaged; for `361 <= n <= 600` the 360 records around the middle.

## IBI correction

`ibi_correction::IbiCorrection` cleans a stream of beats `(timestamp ms, ibi ms, amplitude)` and tags each output
with a validity indicator. `correct_ibis` runs a night through one fresh state; `correct_stateless` restarts the
state when a state-change or wear event requires it.

**Validity values.** 0 first beat (passed through), 1 valid and unchanged, 2 invalid (rejected, amplitude 0),
3 interpolated (replaces a rejected run, amplitude 0). A run of rejected beats that is interpolated can be promoted
to 1 if the replacement fits the neighboring beats.

**Parameters.** Defaults: at most 4 output events per call, repair limit `gap_ms` 5000, upper bound 150 bpm,
two 64-entry beat rings, a 7-beat output window. An output event lags the input by three beats, so the last three
beats of a night are never delivered.

**Beat test.** Each beat is judged once three later beats exist. It is rejected when its rate is at or below 30 bpm
or above 150 bpm, and by tests against the previous accepted beat (a rate below 0.625 times it, or above it by a
fixed margin, or below it by a fixed margin). Once the ring holds more than 6 beats, tests against the median of the
7 surrounding IBIs (`abs(b - median rate) > 14` bpm) and a variance test (`d^2 > 1.5 var`, with `d` the difference
between the beat and the 7-beat mean rate) apply. A run of more than 60 rejected beats is flushed as invalid.

**Gap repair.** When a good beat follows rejected ones, the run between the last good beat and this one is replaced
by `n_est = max(1, round(S / mean(ibi_P, ibi_X)))` interpolated beats if the run's total duration `S` is at most
`gap_ms` and fits the output limit; otherwise the run is marked invalid.

**Output stage.** A 7-beat window computes a median-based error test on the centre beat (`abs(b - median rate) >
18` bpm, relative jumps against the previous centre beat, and a variance test) and downgrades failing valid beats
to indicator 2.

## Bedtime detection

`bedtime::Detector::detect_bedtime` finds sleep periods from 30 s motion bytes and 60 s skin temperatures, and
returns `(start ms, end ms, major)` periods. `bedtime_inputs` builds the arrays from decoded ring events.

**Inputs.**

| argument | content |
| --- | --- |
| `motion`, `motion_ts` | one byte per 30 s bin: seconds of motion 0..30; 31 for a bin with no data while the ring is off the finger or out of power; 255 for charging |
| `temp`, `temp_ts` | one u16 per 60 s bin: centi-degrees, 0 for none |
| `hr_periods` | start and end in ms of resting-HR measurement periods (feature 8 sessions); at least one is required |
| `previous` | earlier detected bedtimes (start, end, major); a new period must start at least 15 minutes after the latest end |
| `avg_temp` | must be nonzero and is otherwise unused |
| `report_naps` | true in the ring-event assembly |

**Algorithm.** The detector scans minute by minute (two 30 s bins). It classifies a minute as charging (0), quiet
(1), moderate (2) or active (3) from the summed motion bytes against thresholds that shift with the temperature
(`thr_a` between 100 and 700, `thr_b` between 1000 and 2800), where the temperature is compared against a running
mean of the nonzero temperatures (`lo = mean - 2.5` degrees, `hi = mean + 2.5` degrees after 1440 samples; fixed
32 and 34.5 degrees before). Fourteen saturating counters over the last 240 minutes (quiet minutes in nested old and
new ranges, extreme minutes, and a warm-temperature balance) trigger a sleep bout when fixed thresholds are met (100
quiet minutes in the oldest 180, 43 in the oldest 60, 11 in the oldest 15, 4 in the oldest 5, at most 10 extreme
minutes in the oldest 60). A bout is recognized about 4 hours after its start, so the start is placed by walking
back over the counters. A bout ends when the minute 240 minutes back is charging, or after more than 7 hours when
the oldest minutes are mostly not quiet. Resting-HR periods produce nap candidates (up to five are kept). After the
last bin, minor periods overlapping a major one are removed, the rest are sorted and merged, and a period longer
than 15 hours keeps only its last 15 hours.

**Window assembly (`bedtime_inputs`).** The batch ends at the newest motion or temperature event. The window starts
at the later of 28 days before that end and the start of a recent long sleep (at least 3 h, starting before the
UTC midnight of the batch's first event or 36 hours before the end). Motion bins take the `motion_seconds` of the
first event in the bin (an event within 2 s of a bin's end goes to the next bin); empty bins take the last
`state_change` at or before the bin start. `detect_open` returns only periods that overlap no confirmed
`bedtime_period` and last at least 30 minutes.

## Score weights

The combiner weights are constants in `sleep_score.rs` (`WEIGHTS`), `activity_score.rs` (`WEIGHTS`) and
`readiness.rs`.

## Smaller metrics

| metric | definition |
| --- | --- |
| HRV (`hrv::rmssd`) | `sqrt(mean(diff(ibi)^2))` over an IBI series in ms; needs at least 2 intervals |
| SpO2 (`spo2::spo2_simple`) | quadratic in the R-ratio; see [spo2-calibration.md](../spo2-calibration.md) |
| Sleep debt (`sleep_debt::sleep_debt`) | linear-decay weighted sum of nightly shortfall (need minus actual), capped at 10 h and rounded to 45 minutes; needs at least 5 valid days |
| VO2max, BMR, steps to meters (`metabolic`) | Jackson VO2max, Schofield BMR (rounded kcal), `steps * 0.762` meters |
