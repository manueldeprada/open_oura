# Documentation

Index of the protocol and algorithm documentation. The crates are described in
[`crates/README.md`](../crates/README.md) and [`architecture.md`](architecture.md).

## Layering

The Rust client is split into three layers: fetch (`oura-link`), interpret (`oura-protocol` for byte decoding,
`oura-analysis` for metrics), and apply (`oura-store`). `oura-cli` wires `oura-protocol`, `oura-link` and
`oura-store` together and does not depend on `oura-analysis`. Raw event bodies are stored losslessly, so a decoder
added later applies to stored events with `oura redecode`.

## Topics

| Topic | Page |
| --- | --- |
| Protocol command reference (Ring 3): requests, responses, auth, features | [horizon-ring3-protocol-cheatsheet.md](horizon-ring3-protocol-cheatsheet.md) |
| Ring 5 BLE surface and first-contact findings | [ring-5-observations.md](ring-5-observations.md) |
| Sync sequence, event drain, time sync, history plausibility check, phone anchors | [sync-orchestration.md](sync-orchestration.md) |
| What the ring emits, what the phone computes, what runs in the cloud | [data-recovery-map.md](data-recovery-map.md) |
| Event-body decoders and how to add one | [native-decoder.md](native-decoder.md) |
| Decoder validation status per event | [algorithms/unvalidated-events.md](algorithms/unvalidated-events.md) |
| Feature capabilities, modes, defaults, and the event each feature produces | [ring-features.md](ring-features.md) |
| Workout, REAL_STEPS and PPG sampler packets, event layouts, byte rates | [ring-feature-modes.md](ring-feature-modes.md) |
| Raw PPG decoding for cardiovascular age (`0x81`) | [cva-cardiovascular-age.md](cva-cardiovascular-age.md) |
| SpO2 R-ratio to percentage | [spo2-calibration.md](spo2-calibration.md) |
| RData bulk sampler probe | [rdata-capacity-probe.md](rdata-capacity-probe.md) |
| Retired activity heuristic | [original-heuristics.md](original-heuristics.md) |
| Factory reset procedures | [factory-reset.md](factory-reset.md) |
| Firmware update opcodes and encryption status | [firmware-update.md](firmware-update.md) |
| Findings on model and firmware encryption | [security-observations.md](security-observations.md) |
| Metric algorithms: status table | [algorithms/README.md](algorithms/README.md) |
| Metric algorithms: method notes | [algorithms/metrics.md](algorithms/metrics.md) |
| `oura-analysis` API and call sequence | [`crates/oura-analysis/README.md`](../crates/oura-analysis/README.md) |
| Sleep-stage classifier interface and blob format | [algorithms/staging-classifier.md](algorithms/staging-classifier.md) |
| Machine-learning model runners (user-supplied model files) | [model-runners.md](model-runners.md) |
| Which model files you supply, and the step counter weights layout | [models-you-supply.md](models-you-supply.md) |

## Which source decides

When two documents disagree, check in this order: known-good packets, then the code and its unit tests,
then the prose. The cheatsheet and observation pages describe specific ring models; other firmware can differ.

## What is tested

`oura-protocol` has unit tests for framing, auth, request builders and event decoders against known-good packets.
`oura-analysis` has unit tests with fixed input and output vectors for each metric
([algorithms/README.md](algorithms/README.md)). `oura-store` has tests for the schema, the sync cursor and
re-decoding. Behavior that depends on a particular ring or on a model file is not covered by `cargo test`.
