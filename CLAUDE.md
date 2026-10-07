# open_oura — repo guide for agents

Reverse-engineering of the Oura ring (Ring 3/4/5) and an independent, cloud-free Rust
client: BLE protocol, auth, sync, event decoders, and the metric algorithms ported from
Oura's `ecore` engine. See `README.md` and `docs/` for the reverse-engineering details.

The user-facing apps (native iOS app + web dashboard) are **not** here: they live in
[open_health](https://github.com/Th0rgal/open_health), which depends on these crates by
git `rev`. A change here only reaches the apps once open_health bumps that `rev`.

## Layout

- `crates/oura-protocol` — framing, app-auth (AES), request builders, event-body
  decoders. Pure, no I/O; unit-tested against real captured packets.
- `crates/oura-link` — `Transport` trait + `btleplug` BLE, handshake, history-event
  drain, live HR/ACM, features, RData (`OuraClient`).
- `crates/oura-analysis`: metric functions (HRV, SpO2, baselines, sleep summary, sleep,
  readiness and activity scores with input assembly, breathing rate, IBI correction)
  and the sleep staging pipeline. Pure; the `oura` binary does not use it.
  API guide: `crates/oura-analysis/README.md`.
- `crates/oura-store` — SQLite persistence (raw events, readings, sync cursor).
- `crates/oura-cli` — the `oura` protocol/debug binary.
- `tools/` — Python research bench (`oura_protocol.py`, `oura_realtime_listener.py`).

Architecture and module boundaries: `crates/README.md` and `docs/architecture.md`. Topic
index: `docs/README.md`.

## Building / testing

- `cargo build --release` (binary at `target/release/oura`), `cargo test`.
- Keep the crate APIs used by open_health stable, or update open_health alongside.

## Safety

- Destructive commands (`factory-reset`) require `--yes`; never run them against a ring
  without explicit permission.
- Oura's proprietary models (`notes/models/`), APKs/decompiled app under `reverse/`,
  `captures/`, `*.db`, and auth keys are gitignored — never commit them.
