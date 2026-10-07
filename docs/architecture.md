# Architecture

The Rust client is split by concern into three layers - **fetch → interpret →
apply** - across a small workspace of focused crates. The rule of thumb: each
crate owns one job and depends only "downward".

```
                ┌───────────────────────────────────────────────┐
   FETCH        │ oura-link   BLE transport, connect+auth, sync   │
                │             drain, live HR/ACM, features, RData │
                └───────────────┬───────────────────────────────┘
                                │ raw frames / events
                ┌───────────────▼───────────────────────────────┐
   INTERPRET    │ oura-protocol  framing · commands · auth crypto │  (low level)
   (decode)     │                device parsers · event decoders  │
                │                bytes → typed samples            │
                └───────────────┬───────────────────────────────┘
                                │ typed samples
                ┌───────────────▼───────────────────────────────┐
   INTERPRET    │ oura-analysis  metric algorithms:               │  (high level)
   (compute)    │                HRV, SpO2, baselines, sleep      │
                │                summary, sleep/readiness/activity │
                │                scores, breathing rate,          │
                │                sleep staging                     │
                └────────────────────────────────────────────────┘
                (pure functions on slices; the caller reads stored
                 events and supplies the inputs)

                ┌───────────────────────────────────────────────┐
   APPLY        │ oura-store   SQLite: raw events (lossless),     │
                │              decoded_json, readings, device,    │
                │              sync cursor                        │
                └────────────────────────────────────────────────┘

   oura-cli  orchestrates protocol/debug flows: fetch → store(raw) → decode
             (it does not depend on oura-analysis)
```

## Crates and what goes where

| Crate | Layer | Owns | Depends on | I/O? |
| --- | --- | --- | --- | --- |
| `oura-protocol` | interpret (decode) | packet framing, request builders, app-auth AES, `device` parsers, `events` decoders + typed sample/event structs | - | none (pure) |
| `oura-link` | fetch | `Transport` trait, `btleplug` `BleTransport`, `OuraClient` (firmware/battery/auth/sync/live/features/rdata) | oura-protocol | BLE, async |
| `oura-analysis` | interpret (compute) | metric algorithms in `ported::` (`sleepstaging` is a directory of submodules) and their generated `*_vectors.rs` test tables | oura-protocol (only `ported/real_steps.rs` imports it) | none (pure compute) |
| `oura-store` | apply | SQLite schema + read/write, sync cursor, `redecode` | oura-protocol | SQLite |
| `oura-cli` | wiring | protocol/debug command dispatch | oura-protocol, oura-link, oura-store | BLE, SQLite |

The store schema has five tables (`store_meta`, `device`, `sync_state`, `events`, `readings`); there are no daily-metric tables. `oura-cli` has no score command; scores are computed by callers of `oura-analysis`.

**Where to add things**
- A new **wire decoder** (new event body) → `oura-protocol::events::decode_body`
  (+ a test with captured bytes). Never needs a re-sync; run `oura redecode`.
- A new **BLE command / capability** → `oura-protocol::protocol` (request builder)
  + a method on `OuraClient` in `oura-link`.
- A new **metric/algorithm** (score, sleep, baseline…) → a module in
  `oura-analysis`, with a doc under `docs/algorithms/` (see below).
- A new **persisted table / query** → `oura-store::storage`.
- A new **protocol/debug command** → `oura-cli`.
- Product UI, dashboard APIs, DNA/blood, and iOS app work live in
  [`open_health`](https://github.com/Th0rgal/open_health).

## Errors

Each layer carries its own error type (no shared god-enum): `oura-link::Error`
(BLE/auth/protocol), `oura-store::Error` (SQLite). `oura-protocol` decoders are
infallible (return `Option`). `oura-cli` collapses everything to `anyhow`.

## Data flow (a sync)

1. `oura-link` connects, authenticates, and drains history events (raw frames).
2. `oura-protocol` decodes each event body into typed samples (or leaves it raw).
3. `oura-store` persists raw events + a per-device cursor (idempotent, incremental).
4. A caller reads the stored events back (`decoded_events`, `open_read_only`) and passes
   them to `oura-analysis` functions to compute daily metrics (HRV, scores, …); see
   [`crates/oura-analysis/README.md`](../crates/oura-analysis/README.md).
5. Nothing is written back: daily metrics are recomputed on demand and are not persisted
   in `oura.db`.

## Maintaining algorithm docs

Every ported algorithm in `oura-analysis` is documented one-file-per-metric under
`docs/algorithms/`, each recording: **source** (ecore function `@ addr` or model
file + version), **inputs**, the **formula/constants**, the **Rust impl location**,
and a **status/confidence + validation** note. The module's doc-comment links to
its markdown. `docs/algorithms/README.md` holds the index + status table. Update
the status when an algorithm moves from best-effort → validated (e.g. against the
trends CSV or captured data).
