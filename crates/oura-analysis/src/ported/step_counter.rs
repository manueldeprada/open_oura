//! Step counter: counts steps per 10 s row from gait features and 30 s motion events.
//!
//! Inputs are the gait features of `steps_motion_decoder` (one row per 10 s, see
//! `steps_motion_decoder.rs`) and the 8 columns of the 30 s `motion_event` (tag `0x47`).
//! Stages:
//! 1. **Merge.** Each gait row takes the first motion row, searching forward from the last
//!    match, whose timestamp after the shifts (gait `-10 s`, motion `-30 s`) is within 30 s of
//!    the gait row's. Rows without a match keep NaN motion columns.
//! 2. **Defaults.** NaN and infinite entries become per-column defaults ([`DEFAULTS`]) before the
//!    later stages, which see the replaced values.
//! 3. **Step eligibility.** A batch-norm MLP (layer widths 19, 32 seven times, 6; leaky ReLU
//!    0.01), a second MLP (widths 6, 4 four times) and a linear unit with a sigmoid give the
//!    probability that the row contains walking or running.
//! 4. **Stride rules.** Steps per row are `stride_frequency * 10 s`. The frequency doubles
//!    when it lies in an L-shaped region of low frequency and low gait-to-stride amplitude
//!    ratio, the amplitude is nonzero and the eligibility is at least 0.55. Rows with `stride + gait` amplitude fraction below
//!    0.03 or amplitude below 10 mg get 0 steps, as do results below 0.39 steps.
//! 5. **Multiplier.** The steps are scaled by a normalized sigmoid of the eligibility
//!    (`beta`, `tau` from the weights; 0 at eligibility 0.17 and 1 at 0.90).
//! 6. Optionally, the 10 s counts are split proportionally onto a coarser grid
//!    ([`resample`]).
//!
//! The stepmotion columns are in the network's order, which differs from the decoder's output order;
//! [`stepmotion_from_decoder`] converts.
//!
//! **Weights.** The network weights are not part of this repository. [`StepCounter::from_file`]
//! and [`StepCounter::from_bytes`] read a file you supply; the layout is documented on
//! [`StepCounter::from_bytes`], and `tools/export_step_counter_weights.py` writes it from a
//! model archive you supply. Everything else in this module (merge, defaults, stride rules,
//! multiplier, resampling) needs no weights file. Network arithmetic is float32; resampling is
//! float64.

use super::steps_motion_decoder::OUTPUT_FEATURES;

pub const STEPMOTION_COLUMNS: usize = 11;
pub const MOTION_COLUMNS: usize = 8;
const N_FEATURES: usize = STEPMOTION_COLUMNS + MOTION_COLUMNS;
const MAGIC: &[u8; 4] = b"STP1";
const DIMS_A: [usize; 9] = [19, 32, 32, 32, 32, 32, 32, 32, 6];
const DIMS_B: [usize; 5] = [6, 4, 4, 4, 4];
const BN_EPS: f32 = 1e-5;
const LEAKY_SLOPE: f32 = 0.01;

/// Column order of the 11 stepmotion columns the network expects.
pub const STEPMOTION_NAMES: [&str; STEPMOTION_COLUMNS] = [
    "first_non_locomotor_frequency",
    "first_non_locomotor_amplitude_frac",
    "frequency_bin_high_frac",
    "frequency_bin_mid_frac",
    "gait_amplitude_frac",
    "stride_amplitude_frac",
    "stride_frequency",
    "sum_accel_mg_std",
    "total_amplitude_mg",
    "y_accel_std_ratio",
    "z_accel_std_ratio",
];

/// Motion columns of the 8-wide motion input, in order.
pub const MOTION_NAMES: [&str; MOTION_COLUMNS] = [
    "orientation",
    "motion_seconds",
    "acm_average_x",
    "acm_average_y",
    "acm_average_z",
    "regular_motion",
    "low_intensity_motion_count",
    "high_intensity_motion_count",
];

/// Replacement for NaN or infinite input, per column of the 19 merged columns.
const DEFAULTS: [f32; N_FEATURES] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.34, 0.32, 2.55, 0.0, -15.8, -13.81, -8.01,
    0.0, 0.0, 0.0,
];

const IDX_STRIDE_FREQUENCY: usize = 6;
const IDX_STRIDE_AMPLITUDE_FRAC: usize = 5;
const IDX_GAIT_AMPLITUDE_FRAC: usize = 4;
const IDX_TOTAL_AMPLITUDE_MG: usize = 8;
const SECONDS_PER_BATCH: f32 = 10.0;
const MAX_STRIDE_FREQUENCY: f32 = 1.25;
const MAX_GAIT_STRIDE_RATIO: f32 = 0.11;
const MAX_START_STRIDE_FREQUENCY: f32 = 0.8;
const MAX_START_GAIT_STRIDE_RATIO: f32 = 1.2;
const MIN_ELIGIBILITY_TO_CORRECT: f32 = 0.55;
const MIN_FRAC_TO_CORRECT: f32 = 0.0;
const MIN_MG_TO_CORRECT: f32 = 0.0;
const MIN_FRAC: f32 = 0.03;
const MIN_MG: f32 = 10.0;
const MIN_STEPS: f32 = 0.39;
const MULTIPLIER_X0: f32 = 0.17;
const MULTIPLIER_X1: f32 = 0.9;

const SHIFT_STEPMOTION_MS: i64 = -10_000;
const SHIFT_MOTION_MS: i64 = -30_000;
const MAX_DELTA_MS: i64 = 30_000;
const BATCH_MS: i64 = 10_000;

/// Reorder one `steps_motion_decoder` output row (`FEATURE_NAMES` order) into the network's
/// stepmotion column order ([`STEPMOTION_NAMES`]).
pub fn stepmotion_from_decoder(row: &[f32; OUTPUT_FEATURES]) -> [f32; STEPMOTION_COLUMNS] {
    // decoder index of each network column: first_non_locomotor_frequency 6, its amplitude 7,
    // bins high/mid 9/10, gait 8, stride amplitude 5, stride frequency 4, sum_accel 0,
    // total amplitude 3, y/z ratios 1/2
    const FROM: [usize; STEPMOTION_COLUMNS] = [6, 7, 9, 10, 8, 5, 4, 0, 3, 1, 2];
    FROM.map(|i| row[i])
}

struct BatchNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    mean: Vec<f32>,
    var: Vec<f32>,
}

impl BatchNorm {
    fn apply(&self, x: &mut [f32]) {
        for (i, v) in x.iter_mut().enumerate() {
            *v = (*v - self.mean[i]) / (self.var[i] + BN_EPS).sqrt() * self.weight[i] + self.bias[i];
        }
    }
}

struct Linear {
    weight: Vec<f32>, // [out][in]
    bias: Vec<f32>,
    n_in: usize,
}

impl Linear {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        self.bias
            .iter()
            .enumerate()
            .map(|(o, &b)| {
                let row = &self.weight[o * self.n_in..(o + 1) * self.n_in];
                row.iter().zip(x).fold(b, |acc, (w, v)| acc + w * v)
            })
            .collect()
    }
}

/// Input normalization, then `k` blocks of a linear layer, a normalization and a leaky ReLU
/// (dropout is the identity at inference).
struct Mlp {
    input_norm: BatchNorm,
    stages: Vec<(Linear, BatchNorm)>,
}

impl Mlp {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        self.input_norm.apply(&mut h);
        for (lin, bn) in &self.stages {
            h = lin.apply(&h);
            bn.apply(&mut h);
            for v in h.iter_mut() {
                if *v < 0.0 {
                    *v *= LEAKY_SLOPE;
                }
            }
        }
        h
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The step counter network and its output scaling.
pub struct StepCounter {
    core: Mlp,
    second: Mlp,
    head: Linear,
    beta: f32,
    tau: f32,
}

/// Output rows: `[start_ms, end_ms)` intervals with the step count of each.
#[derive(Clone, Debug, PartialEq)]
pub struct Steps {
    pub start_ms: Vec<i64>,
    pub end_ms: Vec<i64>,
    pub steps: Vec<f32>,
}

impl StepCounter {
    /// Read the weights file at `path` (see [`from_bytes`](Self::from_bytes) for the layout).
    /// Returns `None` if the file cannot be read or does not have the layout.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Option<StepCounter> {
        StepCounter::from_bytes(&std::fs::read(path).ok()?)
    }

    /// Parse a weights file: the magic bytes `STP1`, then little-endian float32 values with no
    /// padding, in this order.
    ///
    /// 1. The first MLP, widths `[19, 32, 32, 32, 32, 32, 32, 32, 6]`.
    /// 2. The second MLP, widths `[6, 4, 4, 4, 4]`.
    /// 3. The head: 4 weights, then 1 bias.
    /// 4. `beta`, then `tau`.
    ///
    /// An MLP is an input batch norm over `widths[0]` values, then one stage per consecutive
    /// pair of widths `(n_in, n_out)`: the linear weights as `n_out` rows of `n_in` values, the
    /// `n_out` biases, and a batch norm over `n_out` values. A batch norm of width `n` is
    /// stored as `n` scale values, `n` shifts, `n` running means and `n` running variances
    /// (epsilon 1e-5). Each stage ends in a leaky ReLU with slope 0.01. The file holds 8353
    /// values, 33,416 bytes with the magic.
    pub fn from_bytes(bytes: &[u8]) -> Option<StepCounter> {
        let n_mlp = |d: &[usize]| {
            4 * d[0] + d.windows(2).map(|w| w[0] * w[1] + w[1] + 4 * w[1]).sum::<usize>()
        };
        let floats = n_mlp(&DIMS_A) + n_mlp(&DIMS_B) + 5 + 2;
        if bytes.len() != 4 + 4 * floats || &bytes[..4] != MAGIC {
            return None;
        }
        let mut r = Reader {
            f: bytes[4..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            at: 0,
        };
        let core = r.mlp(&DIMS_A);
        let second = r.mlp(&DIMS_B);
        let head = r.linear(4, 1);
        let (beta, tau) = (r.take(1)[0], r.take(1)[0]);
        Some(StepCounter { core, second, head, beta, tau })
    }

    /// Eligibility probability of one merged, default-filled row.
    fn eligibility(&self, x: &[f32; N_FEATURES]) -> f32 {
        let h = self.second.apply(&self.core.apply(x));
        sigmoid(self.head.apply(&h)[0])
    }

    fn multiplier_sigmoid(&self, x: f32) -> f32 {
        sigmoid(self.beta.exp() * (x - self.tau))
    }

    fn multiplier(&self, eligibility: f32) -> f32 {
        let f0 = self.multiplier_sigmoid(MULTIPLIER_X0);
        let f1 = self.multiplier_sigmoid(MULTIPLIER_X1);
        ((self.multiplier_sigmoid(eligibility) - f0) / (f1 - f0)).clamp(0.0, 1.0)
    }

    /// Steps of one row: stride rules, then the multiplier.
    fn row_steps(&self, x: &[f32; N_FEATURES]) -> f32 {
        let eligibility = self.eligibility(x);
        let mut freq = x[IDX_STRIDE_FREQUENCY];
        let stride_frac = x[IDX_STRIDE_AMPLITUDE_FRAC];
        let gait_frac = x[IDX_GAIT_AMPLITUDE_FRAC];
        let stride_gait_frac = stride_frac + gait_frac;
        let stride_gait_mg = x[IDX_TOTAL_AMPLITUDE_MG] * stride_gait_frac;
        let ratio = gait_frac / stride_frac.max(1e-6);
        let (f_high, r_high) = (MAX_STRIDE_FREQUENCY, MAX_GAIT_STRIDE_RATIO);
        let (f_low, r_low) = (MAX_START_STRIDE_FREQUENCY, MAX_START_GAIT_STRIDE_RATIO);
        let slope = (r_high - r_low) / (f_high - f_low).max(1e-9);
        let r_line = r_low + slope * (freq - f_low);
        let case1 = freq <= f_low && ratio <= r_low;
        let case3 = freq > f_low && freq <= f_high && ratio <= r_line;
        let amplitude_ok = stride_gait_frac > MIN_FRAC_TO_CORRECT && stride_gait_mg > MIN_MG_TO_CORRECT;
        if (case1 || case3) && amplitude_ok && eligibility >= MIN_ELIGIBILITY_TO_CORRECT {
            freq *= 2.0;
        }
        if stride_gait_frac < MIN_FRAC || stride_gait_mg < MIN_MG {
            freq = 0.0;
        }
        let mut steps = freq * SECONDS_PER_BATCH;
        if steps < MIN_STEPS {
            steps = 0.0;
        }
        steps * self.multiplier(eligibility)
    }

    /// Count steps. Timestamps are milliseconds, rows ascending. `resample_ms` splits the
    /// 10 s counts onto a grid of that interval ([`resample`]); `None` returns one row per
    /// stepmotion row, ending at its timestamp.
    pub fn count(
        &self,
        stepmotion_ts: &[i64],
        stepmotion: &[[f32; STEPMOTION_COLUMNS]],
        motion_ts: &[i64],
        motion: &[[f32; MOTION_COLUMNS]],
        resample_ms: Option<f64>,
    ) -> Steps {
        assert_eq!(stepmotion_ts.len(), stepmotion.len());
        assert_eq!(motion_ts.len(), motion.len());
        let n = stepmotion_ts.len();
        let merged = merge(stepmotion_ts, stepmotion, motion_ts, motion);
        let steps: Vec<f32> = merged
            .iter()
            .map(|row| {
                let mut x = *row;
                for (v, d) in x.iter_mut().zip(DEFAULTS) {
                    if !v.is_finite() {
                        *v = d;
                    }
                }
                self.row_steps(&x)
            })
            .collect();
        let end: Vec<i64> = stepmotion_ts.to_vec();
        let start = start_timestamps(&end);
        match resample_ms {
            Some(interval) if n > 0 => resample(&start, &end, &steps, interval),
            _ => Steps { start_ms: start, end_ms: end, steps },
        }
    }
}

/// Sequential reader over the weight blob.
struct Reader {
    f: Vec<f32>,
    at: usize,
}

impl Reader {
    fn take(&mut self, n: usize) -> Vec<f32> {
        let s = self.f[self.at..self.at + n].to_vec();
        self.at += n;
        s
    }

    fn batch_norm(&mut self, n: usize) -> BatchNorm {
        BatchNorm { weight: self.take(n), bias: self.take(n), mean: self.take(n), var: self.take(n) }
    }

    fn linear(&mut self, n_in: usize, n_out: usize) -> Linear {
        let weight = self.take(n_in * n_out);
        Linear { weight, bias: self.take(n_out), n_in }
    }

    fn mlp(&mut self, dims: &[usize]) -> Mlp {
        let input_norm = self.batch_norm(dims[0]);
        let stages = dims
            .windows(2)
            .map(|w| {
                let lin = self.linear(w[0], w[1]);
                (lin, self.batch_norm(w[1]))
            })
            .collect();
        Mlp { input_norm, stages }
    }
}

/// Merge: the stepmotion columns, then the matched motion row (NaN if none).
fn merge(
    sm_ts: &[i64],
    sm: &[[f32; STEPMOTION_COLUMNS]],
    m_ts: &[i64],
    m: &[[f32; MOTION_COLUMNS]],
) -> Vec<[f32; N_FEATURES]> {
    let mut out = vec![[f32::NAN; N_FEATURES]; sm.len()];
    let mut current = 0usize;
    for (i, row) in out.iter_mut().enumerate() {
        row[..STEPMOTION_COLUMNS].copy_from_slice(&sm[i]);
        let left = sm_ts[i] + SHIFT_STEPMOTION_MS;
        // first index of the minimal |delta| over the remaining motion rows
        let best = m_ts[current.min(m_ts.len())..]
            .iter()
            .enumerate()
            .map(|(k, &t)| (k, (left - (t + SHIFT_MOTION_MS)).abs()))
            .fold(None, |best: Option<(usize, i64)>, c| match best {
                Some(b) if b.1 <= c.1 => Some(b),
                _ => Some(c),
            });
        if let Some((k, d)) = best {
            if d <= MAX_DELTA_MS {
                current += k;
                row[STEPMOTION_COLUMNS..].copy_from_slice(&m[current]);
            }
        }
    }
    out
}

/// Start of each 10 s interval. The previous end is used when
/// the gap is within 10 % of 10 s; otherwise the start is the end minus the mean valid gap
/// (10 s if no gap is valid). The first row compares against the last (wrap-around).
fn start_timestamps(end: &[i64]) -> Vec<i64> {
    let n = end.len();
    if n == 0 {
        return vec![];
    }
    let interval = BATCH_MS as f32;
    let upper = (interval * (1.0f32 + 0.1)) as i64;
    let lower = (interval * (1.0f32 - 0.1)) as i64;
    let prev = |i: usize| end[(i + n - 1) % n];
    let bad = |i: usize| {
        let d = end[i] - prev(i);
        d > upper || d < lower
    };
    let good: Vec<f32> = (0..n).filter(|&i| !bad(i)).map(|i| (end[i] - prev(i)) as f32).collect();
    let avg = if good.is_empty() {
        BATCH_MS
    } else {
        (good.iter().sum::<f32>() / good.len() as f32) as i64
    };
    (0..n).map(|i| if bad(i) { end[i] - avg } else { prev(i) }).collect()
}

/// Resample onto a grid: each interval's steps are split between the grid cells holding its start and its
/// end, in proportion to the time in each. The first cell is dropped when it holds less than
/// 95 % of the intervals a full cell would.
pub fn resample(start: &[i64], end: &[i64], steps: &[f32], interval_ms: f64) -> Steps {
    let n = start.len();
    let start: Vec<f64> = start.iter().map(|&v| v as f64).collect();
    let end_f: Vec<f64> = end.iter().map(|&v| v as f64).collect();
    let cell = |t: f64| interval_ms * (t / interval_ms).floor();
    let left: Vec<f64> = start.iter().map(|&t| cell(t)).collect();
    let right: Vec<f64> = end_f.iter().map(|&t| cell(t)).collect();
    let pct_left: Vec<f64> = (0..n)
        .map(|i| if left[i] != right[i] { (right[i] - start[i]) / (end_f[i] - start[i]) } else { 0.5 })
        .collect();
    let pct_right: Vec<f64> = pct_left.iter().map(|p| 1.0 - p).collect();
    let mut times: Vec<f64> = left.iter().chain(&right).copied().collect();
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup();
    let mut out_steps = vec![0.0f64; times.len()];
    let mut n_original = vec![0.0f64; times.len()];
    for (k, &t) in times.iter().enumerate() {
        for i in 0..n {
            if left[i] == t {
                out_steps[k] += steps[i] as f64 * pct_left[i];
                n_original[k] += pct_left[i];
            }
        }
        for i in 0..n {
            if right[i] == t {
                out_steps[k] += steps[i] as f64 * pct_right[i];
                n_original[k] += pct_right[i];
            }
        }
    }
    let mut ends: Vec<f64> = times.iter().map(|t| t + interval_ms).collect();
    *ends.last_mut().unwrap() = *end_f.last().unwrap();
    let original_sampling = (0..n).map(|i| end_f[i] - start[i]).sum::<f64>() / n as f64;
    let expected = interval_ms / original_sampling;
    let skip = usize::from(n_original[0] < expected * 0.95);
    Steps {
        start_ms: times[skip..].iter().map(|&t| t as i64).collect(),
        end_ms: ends[skip..].iter().map(|&t| t as i64).collect(),
        steps: out_steps[skip..].iter().map(|&s| s as f32).collect(),
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Weights file with identity batch norms and zero linear weights, so every row gets the
    /// same eligibility `sigmoid(head_bias)`. `last_bias` is the bias of the last stage of the
    /// second MLP (it reaches the head through a leaky ReLU), `head_w` the head weights.
    pub(crate) fn blob(last_bias: f32, head_w: [f32; 4], head_bias: f32, beta: f32, tau: f32) -> Vec<u8> {
        let mut f: Vec<f32> = Vec::new();
        let bn = |f: &mut Vec<f32>, n: usize| {
            f.extend(vec![1.0; n]); // scale
            f.extend(vec![0.0; n]); // shift
            f.extend(vec![0.0; n]); // running mean
            f.extend(vec![1.0; n]); // running variance
        };
        for (dims, bias) in [(&DIMS_A[..], 0.0), (&DIMS_B[..], last_bias)] {
            bn(&mut f, dims[0]);
            for (k, w) in dims.windows(2).enumerate() {
                f.extend(vec![0.0; w[0] * w[1]]);
                let b = if k + 2 == dims.len() { bias } else { 0.0 };
                f.extend(vec![b; w[1]]);
                bn(&mut f, w[1]);
            }
        }
        f.extend(head_w);
        f.extend([head_bias, beta, tau]);
        let mut out = MAGIC.to_vec();
        out.extend(f.iter().flat_map(|v| v.to_le_bytes()));
        out
    }

    /// A counter whose eligibility is `e` for every row; the multiplier uses beta 0, tau 0.5.
    pub(crate) fn counter(e: f32) -> StepCounter {
        StepCounter::from_bytes(&blob(0.0, [0.0; 4], (e / (1.0 - e)).ln(), 0.0, 0.5)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{blob, counter};
    use super::*;

    /// Multiplier for eligibility `e` with beta 0, tau 0.5, computed independently.
    fn expected_multiplier(e: f64) -> f64 {
        let f = |x: f64| 1.0 / (1.0 + (-(x - 0.5)).exp());
        ((f(e) - f(0.17)) / (f(0.9) - f(0.17))).clamp(0.0, 1.0)
    }

    /// Network-order row: stride frequency, gait and stride amplitude fractions, total mg.
    fn row(freq: f32, gait: f32, stride: f32, total_mg: f32) -> [f32; STEPMOTION_COLUMNS] {
        let mut r = [0.1f32; STEPMOTION_COLUMNS];
        r[6] = freq;
        r[4] = gait;
        r[5] = stride;
        r[8] = total_mg;
        r
    }

    /// Steps of one row, merged with a motion row at the same time.
    fn steps_of(c: &StepCounter, r: [f32; STEPMOTION_COLUMNS]) -> f32 {
        let out = c.count(&[100_000], &[r], &[130_000], &[[0.0; MOTION_COLUMNS]], None);
        assert_eq!(out.steps.len(), 1);
        out.steps[0]
    }

    fn near(got: f32, want: f64) {
        assert!((got as f64 - want).abs() < 1e-3, "got {got}, want {want}");
    }

    #[test]
    fn decoder_order_maps_to_network_order() {
        let mut row = [0f32; OUTPUT_FEATURES];
        for (i, v) in row.iter_mut().enumerate() {
            *v = i as f32;
        }
        let m = stepmotion_from_decoder(&row);
        for name in STEPMOTION_NAMES {
            let dec = super::super::steps_motion_decoder::FEATURE_NAMES
                .iter()
                .position(|n| *n == name)
                .unwrap();
            let model = STEPMOTION_NAMES.iter().position(|n| *n == name).unwrap();
            assert_eq!(m[model], dec as f32, "{name}");
        }
    }

    #[test]
    fn merge_matches_forward_within_30_s() {
        let sm = [[1.0; STEPMOTION_COLUMNS]; 4];
        // shifted gait times: 90, 120, 150, 390 s; shifted motion times: 95 and 170 s
        let m = [[2.0; MOTION_COLUMNS], [3.0; MOTION_COLUMNS]];
        let out = merge(&[100_000, 130_000, 160_000, 400_000], &sm, &[125_000, 200_000], &m);
        assert_eq!(out[0][STEPMOTION_COLUMNS], 2.0);
        assert_eq!(out[1][STEPMOTION_COLUMNS], 2.0, "a motion row can serve two gait rows");
        assert_eq!(out[2][STEPMOTION_COLUMNS], 3.0);
        assert!(out[3][STEPMOTION_COLUMNS..].iter().all(|v| v.is_nan()), "220 s apart: no match");
        assert_eq!(out[0][0], 1.0);
    }

    #[test]
    fn non_finite_inputs_take_the_column_default() {
        let c = counter(0.95);
        let walking = row(1.0, 0.1, 0.5, 500.0);
        near(steps_of(&c, walking), 20.0); // doubled, see stride_rules
        // total amplitude NaN -> default 0 mg -> below the 10 mg floor -> 0 steps
        let mut r = walking;
        r[8] = f32::NAN;
        assert_eq!(steps_of(&c, r), 0.0);
        r[8] = f32::INFINITY;
        assert_eq!(steps_of(&c, r), 0.0);
        // the default is also seen by the stride rules: NaN stride frequency -> 0 steps
        let mut r = walking;
        r[6] = f32::NAN;
        assert_eq!(steps_of(&c, r), 0.0);
    }

    #[test]
    fn stride_rules() {
        let c = counter(0.95); // multiplier 1
        // above the maximum stride frequency 1.25: not doubled, 1.5 * 10 s
        near(steps_of(&c, row(1.5, 0.1, 0.5, 500.0)), 15.0);
        // frequency <= 0.8 with gait/stride ratio <= 1.2: doubled
        near(steps_of(&c, row(0.7, 0.1, 0.5, 500.0)), 14.0);
        // frequency 1.0: boundary line is 1.2 + (0.11 - 1.2) / 0.45 * 0.2 = 0.7156
        near(steps_of(&c, row(1.0, 0.1, 0.5, 500.0)), 20.0); // ratio 0.2 below the line
        near(steps_of(&c, row(1.0, 0.4, 0.5, 500.0)), 10.0); // ratio 0.8 above the line
        // amplitude floors: fraction sum < 0.03, or fraction sum * total mg < 10
        assert_eq!(steps_of(&c, row(1.5, 0.01, 0.01, 5000.0)), 0.0);
        assert_eq!(steps_of(&c, row(1.5, 0.1, 0.5, 10.0)), 0.0);
        near(steps_of(&c, row(1.5, 0.1, 0.5, 20.0)), 15.0);
        // fewer than 0.39 steps become 0: 0.03 doubled is 0.06 * 10 = 0.6, kept
        near(steps_of(&c, row(0.03, 0.1, 0.5, 500.0)), 0.6);
    }

    #[test]
    fn low_eligibility_disables_doubling_and_scales_steps() {
        let e = 0.3;
        let c = counter(e);
        // eligibility below 0.55: frequency 0.7 is not doubled; steps = 7 * multiplier
        near(steps_of(&c, row(0.7, 0.1, 0.5, 500.0)), 7.0 * expected_multiplier(e as f64));
        // 0.03 * 10 = 0.3 < 0.39 -> 0, 0.04 * 10 = 0.4 is kept and scaled
        assert_eq!(steps_of(&c, row(0.03, 0.1, 0.5, 500.0)), 0.0);
        near(steps_of(&c, row(0.04, 0.1, 0.5, 500.0)), 0.4 * expected_multiplier(e as f64));
    }

    #[test]
    fn multiplier_is_zero_at_0_17_and_one_at_0_9() {
        near(steps_of(&counter(0.17), row(1.5, 0.1, 0.5, 500.0)), 0.0);
        near(steps_of(&counter(0.9), row(1.5, 0.1, 0.5, 500.0)), 15.0);
        near(steps_of(&counter(0.99), row(1.5, 0.1, 0.5, 500.0)), 15.0);
        near(steps_of(&counter(0.1), row(1.5, 0.1, 0.5, 500.0)), 0.0);
        let mid = steps_of(&counter(0.5), row(1.5, 0.1, 0.5, 500.0));
        near(mid, 15.0 * expected_multiplier(0.5));
        assert!(mid > 0.0 && mid < 15.0);
    }

    #[test]
    fn network_output_reaches_the_eligibility() {
        // last stage bias 2 -> leaky ReLU keeps 2; head 1 * 2 - 1 = logit 1 -> sigmoid(1)
        let c = StepCounter::from_bytes(&blob(2.0, [1.0, 0.0, 0.0, 0.0], -1.0, 0.0, 0.5)).unwrap();
        let x = [0.0f32; N_FEATURES];
        assert!((c.eligibility(&x) - 0.731_058_6).abs() < 1e-5);
        // a negative activation is scaled by the leaky slope: 0.01 * -2 = -0.02
        let c = StepCounter::from_bytes(&blob(-2.0, [1.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.5)).unwrap();
        let want = 1.0 / (1.0 + (0.02f32).exp());
        assert!((c.eligibility(&x) - want).abs() < 1e-5);
    }

    #[test]
    fn start_timestamps_use_previous_end_when_regular() {
        let end = [100_000, 110_000, 120_500, 150_000];
        // gaps: first vs last (wrap-around) is bad, 10000 good, 10500 good, 29500 bad; mean good = 10250
        assert_eq!(start_timestamps(&end), vec![100_000 - 10_250, 100_000, 110_000, 150_000 - 10_250]);
        assert!(start_timestamps(&[]).is_empty());
    }

    #[test]
    fn count_without_resampling_ends_at_the_row_timestamps() {
        let c = counter(0.95);
        let r = row(1.5, 0.1, 0.5, 500.0);
        let out = c.count(
            &[100_000, 110_000],
            &[r, r],
            &[120_000],
            &[[0.0; MOTION_COLUMNS]],
            None,
        );
        assert_eq!(out.end_ms, vec![100_000, 110_000]);
        assert_eq!(out.start_ms[1], 100_000);
        assert_eq!(out.steps.len(), 2);
    }

    #[test]
    fn resample_splits_across_cells() {
        // one 10 s interval 55..65 s on a 60 s grid: 50 % in each cell, first cell dropped
        // only if under 95 % of the expected intervals (6 per 60 s), which holds here.
        let r = resample(&[55_000], &[65_000], &[10.0], 60_000.0);
        assert_eq!(r.start_ms, vec![60_000]);
        assert_eq!(r.end_ms, vec![65_000]);
        assert_eq!(r.steps, vec![5.0]);
    }

    #[test]
    fn resample_conserves_steps_and_drops_the_partial_first_cell() {
        // three 10 s intervals, 55..65, 65..75, 75..85 s, 10 steps each, 60 s grid.
        // Cell 0 gets 5 (half of the first interval), cell 60 s gets 5 + 10 + 10.
        let r = resample(&[55_000, 65_000, 75_000], &[65_000, 75_000, 85_000], &[10.0; 3], 60_000.0);
        assert_eq!(r.start_ms, vec![60_000]);
        assert_eq!(r.end_ms, vec![85_000]);
        assert_eq!(r.steps, vec![25.0]);
    }

    #[test]
    fn rejects_malformed_blob() {
        assert!(StepCounter::from_bytes(b"STP1").is_none());
        assert!(StepCounter::from_bytes(&[0u8; 100]).is_none());
        let mut bad = blob(0.0, [0.0; 4], 0.0, 0.0, 0.5);
        bad[0] = b'X';
        assert!(StepCounter::from_bytes(&bad).is_none());
        let good = blob(0.0, [0.0; 4], 0.0, 0.0, 0.5);
        assert_eq!(good.len(), 4 + 4 * 8353);
        assert!(StepCounter::from_bytes(&good[..good.len() - 4]).is_none());
        assert!(StepCounter::from_file("/nonexistent/step_counter_weights").is_none());
    }

    #[test]
    fn from_file_reads_the_layout() {
        let path = std::env::temp_dir().join(format!("step_counter_test_{}.bin", std::process::id()));
        std::fs::write(&path, blob(0.0, [0.0; 4], 0.0, 0.0, 0.5)).unwrap();
        let loaded = StepCounter::from_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(loaded.is_some());
    }

    /// With a weights file you supply: `OURA_STEP_COUNTER_WEIGHTS=/path/to/file cargo test`.
    /// Checks only that the file loads and gives finite, non-negative counts.
    #[test]
    fn user_weights_load_when_configured() {
        let Some(path) = std::env::var_os("OURA_STEP_COUNTER_WEIGHTS") else {
            eprintln!("skipped: OURA_STEP_COUNTER_WEIGHTS is not set");
            return;
        };
        let c = StepCounter::from_file(path).expect("weights file with the documented layout");
        let steps = steps_of(&c, row(1.8, 0.1, 0.5, 500.0));
        assert!(steps.is_finite() && steps >= 0.0);
    }

}
