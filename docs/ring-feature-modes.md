# Ring feature modes: workouts, real steps, and PPG sampler

Packets and events for three ring features: EXERCISE_HR (`0x03`), REAL_STEPS (`0x0b`) and CVA_PPG_SAMPLER
(`0x0d`). [ring-features.md](ring-features.md) covers the other features and the mode values.

**Labels.** **E** established: the behavior the code and its tests rely on. **I** inferred from E items.
**U** unknown. Statements apply to a Ring 3 unless a Ring 5 is named.

## Summary

| Question | Answer | Label |
| --- | --- | --- |
| How does a user-started workout begin? | Write the workout parameters, then set EXERCISE_HR (`0x03`) to REQUESTED (2). End it by writing OFF. There is no dedicated start opcode. | E |
| What marks the session? | A pair of `feature_session` events (tag `0x6c`, feature 3): status 0 at the start, status 2 at an end by request. | E |
| Does the ring send a bpm value during a workout? | No. The ring writes `0x73` (spectral traces) and `0x74` (motion intensity) into the history buffer; heart rate is computed from them after a normal sync. | E |
| Is REAL_STEPS a prerequisite for EXERCISE_HR? | For AUTOMATIC mode, REAL_STEPS is set first. For REQUESTED (manual workout) it is not touched. | E |
| What do `SetFeatureMode` writes to `0x0b` and `0x03` produce? | `0x0b` AUTOMATIC produces a `feature_session` and `0x7e`/`0x7f` events. `0x03` REQUESTED starts a status 0 session and OFF ends it with status 2. | E |
| Does a Ring 3 accept `SetFeatureMode` for `0x0d`? | Not established. | U |
| Volume | `0x7e`/`0x7f`: one record every 30 s, 4.8 kB per hour. `0x73`/`0x74`: about 19 kB per exercise hour. | E |

## Feature ids and modes

| id | feature | produces |
| --- | --- | --- |
| `0x02` | DAYTIME_HR | `0x60`, `0x80` |
| `0x03` | EXERCISE_HR | `0x73`, `0x74` |
| `0x04` | SPO2 | `0x8b` (Ring 5), `0x6f` (Ring 3) |
| `0x08` | RESTING_HR | resting-HR `feature_session` events |
| `0x0b` | REAL_STEPS | `0x7e`, `0x7f` |
| `0x0c` | EXPERIMENTAL | nothing readable |
| `0x0d` | CVA_PPG_SAMPLER | `0x81` |

Modes: `OFF` 0, `AUTOMATIC` 1, `REQUESTED` 2, `CONNECTED_LIVE` 3. Feature modes can only be set on Gen3 and later
rings; older rings return `NOT_SUPPORTED`.

## Packets

| Request | Bytes | Reply |
| --- | --- | --- |
| Feature status | `2f 02 20 <id>` | `2f 06 21 <id> ...`; the mode is in the payload |
| Set mode | `2f 03 22 <id> <mode>` | `2f 03 23 <id> <result>`; result 0 success |
| Set parameters | `2f <n+2> 29 <id> <data...>` | `2f 03 2a <id> <result>`; result 0 success |
| Enable history bundling | `2f 02 03 01` | `2f 02 04 01` (Horizon) |
| Data flush and sync | `28 01 00`, then `GetEvent` from the cursor | |

`SetFeatureMode` results that the ring returns: `SUCCESS`, `ALREADY_SET`, `NOT_IN_FINGER`, `LOW_BATTERY`,
`NOT_SUPPORTED`, `NOT_AVAILABLE`. `oura-protocol` builds these with `req_set_feature_mode`,
`req_set_feature_parameters` and `req_set_workout_hr_parameters`.

### User-started workout

1. Optionally write `2f 03 22 03 00` first (OFF), so the session starts from a known state.
2. Write the workout parameters: `2f 07 29 03 <activity u16 LE> <resting HR u8> 32 <max HR u8>`. The third data
   byte is the constant `0x32` (50) and the last is the maximum heart rate (the client uses `230 - age`, a common estimate). Activity ids: walking 14, running 12, cycling 5, strength training 8, pilates 10, yoga 15, other 65535. What the
   ring does with each parameter byte is U.
3. Start: `2f 03 22 03 02`.
4. End: `2f 03 22 03 00`, retried up to 3 times if the reply is not success.
5. Sync normally; the `0x73`/`0x74` events and the session markers arrive in the history stream.

The ring keeps EXERCISE_HR in REQUESTED until one of the end statuses below or a later write. A client that starts
a workout writes OFF on every exit path, including a reconnect after a crash (read the status first). Whether OFF
restores the mode held before the workout is U.

### Enable sequence

`apply_app_capture_config` sets the notification flags to `0xbf`, then reads the status of features `0x03`,
`0x04`, `0x0b` and `0x0d` (`2f 02 20 <id>`) and writes AUTOMATIC when the mode is OFF. The writes per feature:

| Operation | Writes |
| --- | --- |
| DAYTIME_HR | status read; no write when already AUTOMATIC |
| SPO2 | `2f 02 20 04`, `2f 03 22 04 01` |
| Bundling | `2f 02 03 01` |
| REAL_STEPS | `2f 02 20 0b`, `2f 03 22 0b 01` |
| PPG sampler | `2f 02 20 0d`, `2f 03 22 0d 01` |
| AWHR (EXERCISE_HR AUTOMATIC) | `2f 02 20 03`, `2f 02 20 0b`, `2f 03 22 0b 01` if `0x0b` is not AUTOMATIC, then `2f 03 22 03 01` |

No parameters exist for `0x0b` or `0x0d`: parameter writes are defined for the charging profile, daytime-HR
meditation, SpO2, and workout HR only, so the ring-side schedule of the two features is fixed in firmware. Writing
OFF to `0x0b` and `0x0d` at every connect would undo a mode set earlier.

## Session markers: `feature_session` (`0x6c`)

Body: feature id, status, then a payload whose length depends on the feature (2 bytes for feature 2, 5 for feature
3, 1 for features 4, 8 and `0x0b`; the meaning is U).

| Status | Name |
| --- | --- |
| 0 | start by user |
| 1 | start by system |
| 2 | end by user request |
| 3 | end of normal measurement period |
| 4 | end, too much movement |
| 5 | end, low battery |
| 6 | end, firmware update |
| 7 | end, unexpected error |
| 8 | end, ring charging |
| 9 | end, resting HR detection |
| 10 | end, no movement |
| 11 | end, ring not on finger |
| 12 | start by ring AWHR feature |

Pairings: feature 2 pairs 1 with 3, SpO2 and resting HR end with 8, 9 or 11, a manual workout writes 0
then 2, and REAL_STEPS writes 1 when enabled. Feature ids above 7 occur (feature 8). A session whose end event is
missing can be closed at the last `0x73` or `0x74` event.

## Events

### REAL_STEPS: `0x7e` and `0x7f`

`real_step_event_feature_1` (`0x7e`) and `_2` (`0x7f`) have 14-byte bodies. One gait record is two events, and the
`0x7f` is stamped one decisecond after its `0x7e` (`pair_real_steps` pairs by `ts + 1`). `unpack_real_steps` merges
both bodies into 27 quantized columns: three globals, then three blocks of eight (`COLUMN_BITS` in
`tools/feature_events.py`). `part2[13]` carries the ninth bit of six `<<1` fields and the low two bits of the first
global. Every record fits the column bit widths. The meaning of each column is U.

Records are written every 30 s: 240 events and 4.8 kB per hour (20 bytes
per event, 40 bytes per record). Behavior asleep, on the
charger and off the finger is U.

### EXERCISE_HR: `0x73` and `0x74`

`0x73` `ehr_trace_event` (layout I): `counter u8` (+1 per event), `trace_type u8` (0 ACM, 1 PPG), then triples
`(freq1 u8, freq2 u8, pov_x10 u8)`. A 14-byte body holds four triples and a 5-byte body one, so a trace of five
triples spans two events. Events come in groups of four every 12 s: two of type 0 then two of type 1. `pov_x10` is
non-increasing within a trace. Frequencies fit per-minute rates; units and the meaning of
`freq2` are U.

`0x74` `ehr_acm_intensity_event` (width E): up to seven `u16` LE intensity values per event, one value per about
2 s (I); a closing event can hold fewer.

Volume: about 19 kB per exercise hour.

### CVA_PPG_SAMPLER: `0x81`

`0x81` `cva_raw_ppg_data` is a delta stream: `0x80` followed by a signed 24-bit absolute sample, other bytes are
signed 8-bit deltas (see [cva-cardiovascular-age.md](cva-cardiovascular-age.md)). Data is meaningful only between
the sampler's start and end `feature_session` markers (feature `0x0d`); a 2 s gap heuristic stands in for them. A
measurement holds about 1500 samples, at about one byte per sample plus 4 bytes per absolute marker (I). Whether
a Ring 3 writes this event is U.

## Decoder coverage in `crates/oura-protocol`

| Tag | Name | Body decoder | Gap |
| --- | --- | --- | --- |
| `0x6c` | `feature_session` | feature, status, `status_name`, optional u16, `payload` | no pairing into sessions; payload meaning U |
| `0x73` | `ehr_trace_event` | yes (inferred layout) | frequency units |
| `0x74` | `ehr_acm_intensity_event` | u16 list | sample interval (about 2 s per value, I) |
| `0x7e`, `0x7f` | `real_step_event_feature_1/2` | raw fields per event; `unpack_real_steps` and `pair_real_steps` build the 27 columns | meaning of the columns |
| `0x81` | `cva_raw_ppg_data` | stateless delta decoder | needs state across records and session markers |

Raising `DECODER_VERSION` in `oura-store` makes a store re-decode rows written by an older decoder.

## Cost and safe use

- **Battery.** Not measured for any of the three features. The firmware guards against low battery: `end, low battery` (status 5) exists and `SetFeatureMode` can return
  `LOW_BATTERY`. The PPG sampler uses the LEDs, so `0x0d` is expected to cost the most of the three (I).
- **Sync time.** Continuous `0x7e`/`0x7f` adds 5,760 events per day (I), about 23 batches of 255 events
  ([sync-orchestration.md](sync-orchestration.md)).
- **Feature-write bursts.** Sending `2f 03 22 02 03` or `2f 03 26 02 02` switches on dense HR and
  CVA-related output ([ring-5-observations.md](ring-5-observations.md)). Write one feature at a time and check
  `oura feature-status` after each.
- **Persisting REQUESTED.** See "User-started workout".
- **Other clients.** Re-asserting modes at connect (including OFF for `0x0b` and `0x0d`) overrides these writes.
- **AUTOMATIC `0x03` without `0x0b`.** Whether a ring accepts it is U.

## Open questions

Behavior of `0x0b` asleep, on the charger and across reboots; the battery cost of `0x0b` and of AWHR over hours;
payload bytes of feature 3 sessions; the unit of `0x73` frequencies; what the 27 columns measure; every `0x0d`
question on a Ring 3; whether history bundling changes how the ring buffers `0x7e`/`0x7f`.
