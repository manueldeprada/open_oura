# Sleep-stage classifier interface

The 30 s sleep stager scores one feature row per epoch with a four-class classifier. The feature
pipeline and the post-processing are in `crates/oura-analysis`
(`ported/sleepstaging`). The classifier is any type that implements `StageClassifier`, or a file in
the blob format below. The crate bundles a classifier, so the stager runs without a user-supplied
file (next section).

## Bundled classifier

`stage_night_default(&Inputs, rerun, nap)` stages a night with the classifier in
`ported::sleepstaging::bundled`. `bundled_classifier()` returns it as a `&'static SleepClassifier`,
parsed once on first use from `crates/oura-analysis/data/sleep_stage_gbt.bin`, which is compiled into
the crate with `include_bytes!`. The blob is a gradient-boosted tree ensemble (4 classes, 800 rounds,
3,200 trees with up to 32 leaves, 99,200 nodes, 102,400 leaves, 1,846,470 bytes) trained by the
maintainer on simulated nights only, so no recorded night is in its training data.

Evaluation: on 14 nights from one user, it agrees with the official app's per-30-second stages on 92.6% of epochs
(pooled; 95% bootstrap interval over nights 91.6% to 93.6%; Cohen's kappa 0.89; lowest night 89.1%). The
previous bundled model (174,052 bytes), trained on 8 of those nights, agrees on 92.4% of the 14 and 90.5% of
the 6 it was not trained on. These figures measure agreement with the official app's stages for one user. They are
not accuracy against polysomnography, and users other than that one are not covered.

Classifier time grows with the number of trees. On one Apple M3 core the bundled blob takes about 0.18 s
per night for one classifier pass, and the full stager about 1 s per night.

To use another model, implement `StageClassifier` or load a blob with `classifier::load_classifier`
(format below) and call `stage_night_with`. The other entry points keep their signatures.

## The trait

```rust
pub trait StageClassifier {
    fn n_features(&self) -> usize { 182 }
    fn n_classes(&self) -> usize { 4 }
    fn predict_proba(&self, features: &[f32; 182]) -> [f32; 4];
    fn margins(&self, features: &[f32; 182]) -> Option<[f64; 4]> { None }
    fn predict_proba_rows(&self, rows: &[f32]) -> Vec<[f32; 4]>; // default: row by row
}
```

- A row is 182 float32 values in the order of `columns::KEEP_COLUMNS` without its first (timestamp)
  entry. NaN is a missing value.
- The output order is deep, light, REM, wake. The stage label is the first most probable class
  plus one (1 deep, 2 light, 3 REM, 4 wake).
- `n_features` and `n_classes` are checked against 182 and 4 before staging; a mismatch returns
  `StagingError::ClassifierShape`.
- `SleepClassifier` (the blob reader, `ported/sleep_classifier.rs`) implements the trait.

## Entry points (`ported::sleepstaging::pipeline`)

| Function | Use |
| --- | --- |
| `stage_night_default(&Inputs, rerun, nap)` | Full pipeline with the bundled classifier; needs no file. |
| `stage_night(&SleepClassifier, &Inputs, rerun, nap)` | Unchanged signature. Calls `stage_night_with`. |
| `stage_night_with(&dyn StageClassifier, &Inputs, rerun, nap)` | Full pipeline with any classifier. |
| `stage_night_required(Option<&dyn StageClassifier>, ...)` | `None` returns `StagingError::NoClassifier`. |
| `stage_features(&Inputs)` | Features of the whole bedtime window, no classifier. |
| `stage_night_or_features(Option<&dyn StageClassifier>, ...)` | `Staged::Stages(Output)` with a classifier, `Staged::FeaturesOnly(NightFeatures)` without. |

`stage_features` skips the wake-trimming rerun and the post-processing, which both depend on
stage labels. A caller without a classifier can show the features, classify them with its own
model, or supply an implementation of the trait and call `stage_night_with`.

## Loading a blob

`classifier::load_classifier(path)` and `load_classifier_bytes(bytes)` return
`Box<dyn StageClassifier + Send + Sync>`. They check the magic, version, feature count (182) and
class count (4) against the pipeline's column list, then validate every section. Malformed input
returns a `ClassifierError`; no input panics. `classifier::inspect_blob` reads the header only.

## Blob format

Little endian, no padding between sections. Trees are numbered `class * n_rounds + round`.

Header, 64 bytes: magic `OSCF`; u32 version (1); u32 `n_features`; u32 `n_classes`;
u32 `n_rounds`; u32 `n_nodes`; u32 `n_leaves`; u32 `n_prog`; u32 `n_side`; f64 `base` (at offset 36); zero
padding.

Sections, in order:

| Section | Type | Length |
| --- | --- | --- |
| `prog` | u16 | `n_prog` |
| `roots` | u16 | `n_classes * n_rounds` |
| `node_off` | u32 | trees + 1 |
| `leaf_off` | u32 | trees + 1 |
| `feat` | u8 | `n_nodes` |
| `miss` | u8 | `n_nodes` |
| `thr` | f32 | `n_nodes` |
| `le` | u16 | `n_nodes` |
| `gt` | u16 | `n_nodes` |
| `side_node` | u32 | `n_side` |
| `side_child` | u16 | `n_side` |
| `leaf` | f64 | `n_leaves` |

**Nodes.** Node `i` of a tree is entry `node_off[tree] + i` of the node arrays. It tests
feature `feat` against the float32 threshold `thr`: `x <= thr` goes to child `le`, `x > thr` to
child `gt`. A NaN value goes to `le` when `miss` is 0, to `gt` when 1, and to the child stored in
the side table for this node when 2 (`side_node` holds global node indices in increasing order).

**References.** A child or root reference is a u16 local to its tree. With bit 15 set it names leaf
`ref & 0x7fff` of the tree's leaves (entry `leaf_off[tree] + ...` of `leaf`); otherwise it names a
node of the tree. A child index is larger than its parent's, so every walk ends.

**Addition program.** The blob stores a postfix program `prog` that fixes the order in which a class's
leaf values are added; the same program applies to all four classes. A token below `n_rounds` pushes the
leaf value of that round for the class. The token `0xFFFF` pops two values and pushes their float64 sum
(second popped plus first popped). The loader requires that each round is pushed exactly once and that
one value remains at the end. Any association is accepted. Float64 addition is not associative, so the
program determines the margins.

**Margins.** The margin of a class is `base` plus the value that the addition program leaves for that
class.

**Probabilities.** `e_c = exp(m_c - max m)`, `total = e_3 + (e_2 + (e_0 + e_1))`,
`p_c = float32(e_c / total)`. The format defines this order for the sum of the four exponentials.

## Supplying a classifier

1. Write a blob with any tool that produces the documented layout, or implement `StageClassifier` for any model.
2. Load the blob with `load_classifier`, or construct the implementation.
3. Call `stage_night_with(&*clf, &inputs, true, true)`.

With a blob in an application, load it once and keep it behind an `Arc`; pass `&*clf` to
`stage_night_with`, or `&clf` to `stage_night` when it is a `SleepClassifier`.
