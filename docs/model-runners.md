# Model runners

Some metrics come from machine-learning models: sleep staging,
apnea events, daytime stress, resilience, cardiovascular age, and workout detection.
Those model files are proprietary and **are not distributed with this repository**.

The runner scripts in `tools/` (`run_sleep_model.py`,
`run_stress_model.py`, `run_cva_model.py`, `run_activity_model.py`, `run_models.py`) read
model files that the user supplies from a local directory and apply them to events from a
local `oura.db`. Nothing model-related is committed. Without a model file a runner exits
with an error, and the rest of the repository works unchanged.

Metrics that need no model file are in `crates/oura-analysis` (see
[`algorithms/README.md`](algorithms/README.md)). The sleep stager accepts any classifier
that implements the `StageClassifier` trait
([`algorithms/staging-classifier.md`](algorithms/staging-classifier.md)).

The step counter reads a weights file in a documented layout through `StepCounter::from_file`
([models-you-supply.md](models-you-supply.md)); `tools/export_step_counter_weights.py` creates
the file from a model archive you supply.
