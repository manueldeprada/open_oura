# Sync Orchestration

How the Oura Android app decides *when* to use each ring data channel, from the
decompiled state machine (`com/ouraring/oura/data/device/ring/`) plus the
Ring 4 `open_ring` capture notes that were re-tested against our Ring 5 on
2026-06-30. This is the behavior an independent client should replicate.

## Channels and when each fires

| Channel | Wire | When the app uses it | Needed to mirror app data? |
| --- | --- | --- | --- |
| History events (NORMAL buffer) | `GetEvent` 0x10 / `ExtGetEvent` 0x2f, bufferId 0 | every sync (connect, foreground, background) | yes -- this is the whole game |
| Live / realtime | `SetFeatureMode CONNECTED_LIVE`, realtime 0x06 | only while a specific UI screen is open | only for a live HR readout |
| RData bulk raw | `RDataStart`/`GetPage` 0x03, RAW_DATA buffer | never by default (`r_data_autosync`=false) | no, unless you want raw waveforms |

Everything the user sees (sleep stages, last-night HR/HRV/SpO2, MET/steps) flows
through the history-event drain. RData is a research/diagnostics opt-in.

## Connect handshake (ordered)

Driven by `RingStateMachine` / `DefaultRingStateMachine$Operations`
(states: CONNECTING -> AUTHENTICATING -> CHECK_CAPABILITIES ->
APP_LEVEL_AUTHENTICATING -> FOREGROUND_SYNC | BACKGROUND_SYNC):

1. CONNECT (BLE connect + bond)
2. AUTHENTICATE (nonce -> AES -> `Authenticate`)
3. GET_CAPABILITIES (records whether the ring supports extended event sync)
4. APP_LEVEL_AUTHENTICATE
5. STREAM_REGISTER (`16 01 02`) and per-category event subscriptions (`0x18`)
6. PARAM_SWEEP (`2f 02 20 ...` reads plus app setup writes)
7. SYNC_TIMESTAMPS (`SyncTime`, write phone UTC)
8. ENABLE_NOTIFICATION / state poll (`SetNotification` / `1c 01 bf`, depending
   on generation and app path)
9. BATTERY_LEVEL / PRODUCT_INFO / RING_VERSION (metadata)
10. feature ENABLE_* toggles (conditional on capabilities + user flags)
11. SYNC_EVENTS (main drain)
12. SYNC_R_DATA (only if `r_data_autosync`)

Load-bearing for a client: app-auth, stream registration, time-sync, and
capability/parameter reads all precede event sync. Ring 5 accepts the app-style
registration and sweep; `oura sync` now performs them before draining history.

## App-style setup commands confirmed on Ring 5

These are safe registration/read shapes, not destructive device-management
commands:

- `16 01 02` - enable/register the event stream.
- Event-category subscriptions:
  `18 03 14 00 10`, `18 03 18 00 10`, `18 03 28 00 09`,
  `18 03 34 00 04`, `18 03 04 00 10`, `18 03 08 00 10`.
- Parameter sweep:
  `2f022002`, `2f022004`, `2f020301`, `2f02200b`, `2f02200d`,
  `2f022003`, `2f02200b`, `2f022010`.
- App-style time sync:
  `12 09 <token> <unix/256:u24 LE> 00 00 00 00 f6`.

## Event-drain loop

```
cursor = store.get_next_event_to_sync()        # deciseconds (100 ms units)
loop:
    send DataFlush 28 01 00                    # releases buffered events
    if extended_supported:
        summary = ExtGetEvent(start_ms = cursor*100, max_events = 65535, buffer = NORMAL)
    else:
        summary = GetEvent(start_deciseconds = cursor, max_events = 255, flags = -1)
    decode + persist every event frame from every notification
    if any event was observed:
        cursor = max(event.timestamp) + 1
        store.set_next_event_to_sync(cursor)    # incremental-sync bookmark
        # The official app then sends GetEvent(cursor, max_events=0) ("ack-fetch").
        # oura-link does not: GetEvent is already cursor-addressed, and on
        # Horizon 3.4.3 that ack-fetch holds the next 255 events for ~2 minutes
        # (then we used to throw the stream away and fetch it again).
    if summary.bytes_left > 0:                   # ring has more data
        repeat
    else:
        done
```

The ring reports `bytes_left` in the `0x11` / `0x42` confirmation packet; loop
until it reaches 0. Ring 5 can concatenate multiple `tag|len|payload` event
frames into one BLE notification on legacy sync, and returns length-prefixed
bundled events on extended sync. The parser must walk the whole notification or
bundle. The persisted cursor (`nextEventToSync`) makes sync incremental.
`sleepAnalysisProgress` is surfaced as progress only, not a block.

Four places where `oura-link` deliberately deviates from the app's literal
behavior:

- **No ack-fetch.** The app's `GetEvent(cursor, max_events=0)` after a
  successful batch is skipped. Horizon 3.4.3 (and Ring 5) treat it as another
  fetch; waiting for its `0x11` stalled a Windows sync at ~2 minutes per 255
  events (2026-09-18). The next loop's `DataFlush` + cursor-addressed `GetEvent`
  is enough.

- **Batch termination.** The summary packet is the ring's explicit batch
  terminator, so the client returns from a request the moment it (or a
  command's known single response) arrives, instead of waiting out a quiet
  window after the last frame. The quiet window (1.5 s) remains only as the
  fallback for errors/dead links. A batch that ends *without* a summary is a
  hard error (link lost mid-batch), never treated as "drained".
- **Replayed tail.** Horizon 3.4.3 answers a cursor past its newest event with
  its last few events again instead of an empty batch. Events older than the
  requested cursor are dropped as replays and do not advance it, so that pass
  counts as empty and ends the drain (it used to loop forever, 2026-09-24).
- **Live tail.** The same ring can also keep writing a few *new* events on every
  pass (7 per pass at bytes_left=0), so an empty pass never comes even with
  replays filtered (still looping, 2026-09-28). After bytes_left=0, the drain
  accepts at most `MAX_TAIL_PASSES` (3) small batches (< 32 events) and then
  stops. Nothing is lost: the cursor is persisted and the next sync resumes.
  A real backlog arrives in full batches and is not affected.
- **Batch size.** The app requests `max_events = 65535` — effectively the whole
  backlog as one batch. Since the cursor can only be checkpointed at batch
  boundaries, one giant batch means a dropped link forfeits all progress and
  gives no progress feedback. `oura-link` uses 4096-event batches
  (`EXT_BATCH_MAX_EVENTS`): per-batch overhead is a couple of round-trips
  (negligible), and every batch yields a cursor checkpoint + a `bytes_left`
  progress report.

- **Stream registration on every ring.** `16 01 02` is the app's first command after
  auth. On Horizon (fw 3.4.3) the ring answers it by asking for a 15 ms connection
  interval instead of ~30 ms, so a 255-event legacy batch (one 20-byte notification per
  event) arrives in ~420 ms instead of ~850 ms. The mode lasts one BLE connection. BLE
  mode `1` (fast HR) changes nothing.
- **DataFlush once, then at the tail.** One flush before the first read covers the ring's
  buffered events; mid-backlog batches skip it (~45 ms each) and it is repeated once
  `bytes_left` reaches 0.

### Extended event sync status

Ring 5 responds to Android's newer `ExtGetEvent` request:

```
2f 0c 41 <bufferId> <start_ms:u64 LE> <max_events:u16 LE>
```

The `0x42` confirmation layout is confirmed:

```
events_received:u16, sleep_progress:u8, bytes_left:u32, buffer_id:u8, result:u8
```

The `0x43` data path is confirmed on Ring 5. Android accumulates length-prefixed
envelopes into `GetEventSummary.Extended.rawBuffer`; each completed envelope is
a bundle of:

```
control:u8, tag:u8, ext_len:u8, timestamp_varint, body...
```

The first timestamp varint in a bundle is absolute milliseconds; later varints
are millisecond deltas. The body bytes match the corresponding legacy `GetEvent`
event body. The Rust decoder expands these bundled events into normal
`tag|length|timestamp|body` packets before storing them.

Validation on 2026-07-01:

- `ExtGetEvent(max_events=10)` returned the same 10 events as legacy `GetEvent`
  from cursor `12214277`, in the same order.
- Full fast sync inserted `40063` new rows into `oura.db`, advancing the cursor
  to `13151168`, with no impossible timestamps.

## Scheduling

- On connect: full handshake then SYNC_EVENTS automatically.
- Foreground: user-triggered `triggerForegroundSync()`.
- Background: on app backgrounded.
- A small periodic worker refreshes battery only; the real data sync is
  connection-/lifecycle-triggered, not a fixed timer.

## Gating

- Skips if a sync is already ongoing.
- Routes around *_SYNC during onboarding.
- No hard low-battery block on the event path; RData (heavier) is the gated one.

## Minimal client sync recipe

1. Connect + bond; subscribe to the notify characteristic.
2. Authenticate with the stored 16-byte key.
3. Register the app-style stream (`16 01 02`), event categories (`0x18`), and
   parameter sweep.
4. GetCapabilities -> choose extended vs legacy event path.
5. SyncTime using the app-style counter packet (`12 09 ... f6`), falling back to the
   legacy `12` form. `oura sync` does this by default (`--no-sync-time` to skip): the
   `time_sync` event it writes is what dates the ring's history.
6. (optional) firmware / product / battery for metadata.
7. Trigger the ring's sleep postprocessing (`check_sleep_analysis(false)`), so a
   finished night's `bedtime_period` lands in this sync.
8. DataFlush, then drain history events from the persisted cursor; persist each
   batch with its cursor and advance; stop on an empty pass at `bytes_left == 0`,
   or after `MAX_TAIL_PASSES` small batches once the ring reports 0 (see above).
   Do not send the `GetEvent(max_events=0)` ack-fetch (see above).

Do not issue any RData (0x03) for a normal pull.
