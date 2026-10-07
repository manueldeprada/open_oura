# Models you supply

This repository contains no neural-network weights from the ring vendor. Code that uses such a network reads a file you provide. Everything else runs from the repository alone.

## What runs without any file

| Function | Where | Note |
| --- | --- | --- |
| Sleep staging | `oura-analysis::sleepstaging` | A tree ensemble trained by the maintainer is bundled; see [algorithms/staging-classifier.md](algorithms/staging-classifier.md). A caller can pass a different classifier. |
| Gait feature decoding | `oura-analysis::steps_motion_decoder` | A quantization table, no learned parameters. Needs REAL_STEPS enabled on the ring ([ring-feature-modes.md](ring-feature-modes.md)). |
| Motion rows, merge, stride rules, step multiplier, resampling | `oura-analysis::step_counter`, `real_steps` | Everything except the eligibility network. |
| Scores, baselines, HRV, SpO2, breathing rate, temperature, bedtime, activity day | `oura-analysis` | [algorithms/README.md](algorithms/README.md). |

## What needs a file you supply

| Function | Role of the file | How the repository reads it | Without it |
| --- | --- | --- | --- |
| Step counting from gait records | Eligibility network of the step counter, in the layout below | `StepCounter::from_file(path)` or `StepCounter::from_bytes(bytes)`; the caller chooses the path | `count_steps` cannot run. A client falls back to the step count that the ring reports in `activity_information`. |
| Automatic activity detection | Activity detection network | No Rust loader. The Python runner reads a local model directory ([model-runners.md](model-runners.md)); a client can run the model with its own runtime. | No detected activities. Step motion rows still decode. |
| Illness detection | Illness detection network | Same as activity detection. | No illness score from the network. |
| Cardiovascular age | Cardiovascular age network | Same as activity detection. | No vascular age. |

The files are vendor property and stay on your machine. Do not commit them.

## Step counter weights file

The eligibility network is a stack of batch-norm MLPs followed by a sigmoid. The file holds its weights as little-endian float32 values with no padding:

1. The magic bytes `STP1`.
2. The first MLP, layer widths `[19, 32, 32, 32, 32, 32, 32, 32, 6]`.
3. The second MLP, layer widths `[6, 4, 4, 4, 4]`.
4. The head: 4 weights, then 1 bias.
5. `beta`, then `tau`.

An MLP is an input batch norm over `widths[0]` values, then one stage per consecutive pair of widths `(n_in, n_out)`: the linear weights as `n_out` rows of `n_in` values, the `n_out` biases, and a batch norm over `n_out` values. A batch norm of width `n` is stored as `n` scale values, `n` shifts, `n` running means, and `n` running variances (epsilon 1e-5). Each stage ends in a leaky ReLU with slope 0.01. The file holds 8353 values, 33,416 bytes with the magic. `StepCounter::from_bytes` returns `None` for any other length or magic.

`beta` and `tau` set the multiplier that scales the step count by the eligibility: the multiplier is a sigmoid of `exp(beta) * (e - tau)` for eligibility `e`, normalized so that it is 0 at `e = 0.17` and 1 at `e = 0.90`.

**Creating the file.** `tools/export_step_counter_weights.py MODEL_FILE OUT_FILE` reads a serialized model archive that you supply and writes this layout. It matches tensors by shape and order, so it needs no layer names. It requires `torch`, as the other model runners do.

**Checking the file.** Set `OURA_STEP_COUNTER_WEIGHTS` to the absolute path of the file and run `cargo test -p oura-analysis user_weights`. The test loads the file and counts steps for one synthetic row. It is skipped when the variable is unset.

## Using the counter from a client

1. Decode the stored `0x7e`/`0x7f` and `0x47` events with `real_steps::count_steps`, passing a `StepCounter` and a function from ring timestamps to Unix milliseconds.
2. Choose where the day's step total comes from. The ring's counter is always available. The network-based counter replaces it only when a weights file was loaded and the ring has REAL_STEPS enabled; a day without gait records has no network-based count.
