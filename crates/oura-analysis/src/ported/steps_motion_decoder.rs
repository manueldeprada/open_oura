//! Gait-feature decoder: dequantizes the 27 integer columns of a real-steps feature record
//! and splits each 30 s record into three 10 s rows. There are no learned parameters: a table
//! of `(low, high, bits, encode_zero, transform)` settings per feature defines the mapping.
//!
//! Input: one record per `0x7e`/`0x7f` event pair, 27 columns (`oura_protocol::events::
//! unpack_real_steps`): three globals, then three blocks of eight features, one block per 10 s
//! sub-window. Output: three rows per record with 11 features each (the three globals repeated
//! in all rows, then the block of the row's sub-window).
//!
//! Dequantization of column value `q` with settings `(low, high, bits)` is
//! `inverse(q / 2^bits * (T(high) - T(low)) + T(low))`, where `T` is the forward transform
//! (`log10(x + 1)` or `sqrt`, identity if none) and `inverse` its inverse (`10^x - 1` or
//! `x^2`). With `encode_zero`, `q = 0` decodes to 0 and `q >= 1` uses `(q - 1) / (2^bits - 1)`.
//! The arithmetic is float32.

/// Quantized columns per record.
pub const INPUT_COLUMNS: usize = 27;
/// Features per output row.
pub const OUTPUT_FEATURES: usize = 11;
/// Output feature names, in column order (without `timestamp`).
pub const FEATURE_NAMES: [&str; OUTPUT_FEATURES] = [
    "sum_accel_mg_std",
    "y_accel_std_ratio",
    "z_accel_std_ratio",
    "total_amplitude_mg",
    "stride_frequency",
    "stride_amplitude_frac",
    "first_non_locomotor_frequency",
    "first_non_locomotor_amplitude_frac",
    "gait_amplitude_frac",
    "frequency_bin_high_frac",
    "frequency_bin_mid_frac",
];

#[derive(Clone, Copy)]
enum Transform {
    None,
    Log,
    Sqrt,
}

impl Transform {
    fn apply(self, x: f32) -> f32 {
        match self {
            Transform::None => x,
            Transform::Log => (x + 1.0).log10(),
            Transform::Sqrt => x.sqrt(),
        }
    }

    fn inverse(self, x: f32) -> f32 {
        match self {
            Transform::None => x,
            Transform::Log => 10f32.powf(x) - 1.0,
            Transform::Sqrt => x * x,
        }
    }
}

struct Setting {
    low: f32,
    high: f32,
    bits: i32,
    encode_zero: bool,
    transform: Transform,
}

const fn s(low: f32, high: f32, bits: i32, encode_zero: bool, transform: Transform) -> Setting {
    Setting { low, high, bits, encode_zero, transform }
}

/// Quantizer settings, indexed like [`FEATURE_NAMES`].
const SETTINGS: [Setting; OUTPUT_FEATURES] = [
    s(0.0, 8000.0, 10, false, Transform::Log),
    s(0.0, 0.85, 8, false, Transform::None),
    s(0.0, 0.85, 8, false, Transform::None),
    s(0.0, 8000.0, 9, false, Transform::Log),
    s(0.68, 3.4, 9, true, Transform::None),
    s(0.0, 1.0, 9, false, Transform::Sqrt),
    s(0.0, 24.804_688, 7, false, Transform::None),
    s(0.0, 0.65, 8, false, Transform::Sqrt),
    s(0.0, 0.75, 8, false, Transform::Sqrt),
    s(0.0, 1.0, 8, false, Transform::None),
    s(0.0, 1.0, 8, false, Transform::None),
];

/// Output feature of input column `col` (the three sub-window blocks share one set of settings).
fn feature_of(col: usize) -> usize {
    if col < 3 { col } else { 3 + (col - 3) % 8 }
}

/// One 10 s row of decoded gait features.
#[derive(Clone, Debug, PartialEq)]
pub struct GaitRow {
    /// Row timestamp, in the unit of the input timestamps (unix milliseconds in the usual input).
    pub timestamp: i64,
    /// Features in [`FEATURE_NAMES`] order.
    pub features: [f32; OUTPUT_FEATURES],
}

/// Dequantize `q` (all values of one input column).
fn decode_column(q: &[f32], st: &Setting) -> Vec<f32> {
    let avail = 2f32.powi(st.bits);
    let (x, avail): (Vec<f32>, f32) = if st.encode_zero {
        let x1: Vec<f32> = q.iter().map(|v| v - 1.0).collect();
        let max = x1.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // clamp to [0, max(x1)] as min(max(x, 0), max); with max < 0 the result is max.
        (x1.iter().map(|v| v.max(0.0).min(max)).collect(), avail - 1.0)
    } else {
        (q.to_vec(), avail)
    };
    let low = st.transform.apply(st.low);
    let high = st.transform.apply(st.high);
    x.iter()
        .zip(q)
        .map(|(&v, &orig)| {
            let y = st.transform.inverse(v / avail * (high - low) + low);
            if st.encode_zero {
                y * if orig > 0.0 { 1.0 } else { 0.0 }
            } else {
                y
            }
        })
        .collect()
}

/// Decode gait records. `timestamps[i]` is the timestamp of `records[i]` in milliseconds and
/// both slices have the same length; the result has three rows per record, ordered
/// `t - 2d`, `t - d`, `t`, where `d = floor(interval / 3)` and `interval` is the gap to the
/// previous record (the next record's gap for the first one, 30000 if it is the only one or
/// if the gap is outside 25500..=34500).
pub fn decode(timestamps: &[i64], records: &[[u16; INPUT_COLUMNS]]) -> Vec<GaitRow> {
    assert_eq!(timestamps.len(), records.len(), "timestamps and records differ in length");
    let n = records.len();
    let mut decoded = vec![[0f32; INPUT_COLUMNS]; n];
    for col in 0..INPUT_COLUMNS {
        let q: Vec<f32> = records.iter().map(|r| r[col] as f32).collect();
        for (row, v) in decoded.iter_mut().zip(decode_column(&q, &SETTINGS[feature_of(col)])) {
            row[col] = v;
        }
    }
    let mut out = Vec::with_capacity(3 * n);
    for i in 0..n {
        let raw = if n > 1 {
            if i == 0 { timestamps[1] - timestamps[0] } else { timestamps[i] - timestamps[i - 1] }
        } else {
            30_000
        };
        let interval = if !(25_500..=34_500).contains(&raw) { 30_000 } else { raw };
        let delta = interval.div_euclid(3);
        for sub in 0..3 {
            let mut features = [0f32; OUTPUT_FEATURES];
            features[..3].copy_from_slice(&decoded[i][..3]);
            features[3..].copy_from_slice(&decoded[i][3 + sub * 8..3 + (sub + 1) * 8]);
            out.push(GaitRow { timestamp: timestamps[i] - delta * (2 - sub as i64), features });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_record_decodes_to_lower_bounds() {
        let rows = decode(&[1_000_000], &[[0; INPUT_COLUMNS]]);
        assert_eq!(rows.len(), 3);
        assert_eq!(
            rows.iter().map(|r| r.timestamp).collect::<Vec<_>>(),
            [1_000_000 - 20_000, 1_000_000 - 10_000, 1_000_000]
        );
        for r in &rows {
            // stride_frequency has encode_zero: q = 0 stays 0 instead of the 0.68 lower bound.
            assert!(r.features.iter().all(|&v| v == 0.0), "{:?}", r.features);
        }
    }

    #[test]
    fn full_scale_hits_upper_bounds() {
        let mut rec = [0u16; INPUT_COLUMNS];
        rec[0] = 1024; // sum_accel_mg_std, bits 10: q / 2^10 = 1
        rec[3] = 512; // total_amplitude_mg_1, bits 9
        rec[4] = 512; // stride_frequency_1: (512 - 1) / 511 = 1
        let rows = decode(&[0], &[rec]);
        let f = rows[0].features;
        assert!((f[0] - 8000.0).abs() < 0.5, "{}", f[0]);
        assert!((f[3] - 8000.0).abs() < 0.5, "{}", f[3]);
        assert!((f[4] - 3.4).abs() < 1e-5, "{}", f[4]);
        // Blocks 2 and 3 are zero and land in rows 1 and 2.
        assert_eq!(rows[1].features[3], 0.0);
    }

    #[test]
    fn timestamp_gaps_outside_tolerance_use_30s() {
        let rec = [[0u16; INPUT_COLUMNS]; 3];
        // gaps: first row uses ts[1]-ts[0] = 40000 (invalid), second 30000, third 31500 -> d = 10500
        let rows = decode(&[0, 40_000, 70_000, 101_500], &[rec[0], rec[1], rec[2], rec[0]]);
        let ts: Vec<i64> = rows.iter().map(|r| r.timestamp).collect();
        assert_eq!(&ts[0..3], &[-20_000, -10_000, 0]);
        assert_eq!(&ts[3..6], &[20_000, 30_000, 40_000]);
        assert_eq!(&ts[6..9], &[50_000, 60_000, 70_000]);
        assert_eq!(&ts[9..12], &[101_500 - 21_000, 101_500 - 10_500, 101_500]);
    }

    /// Reference value from the documented formula, in float64.
    fn dequantize(q: f64, low: f64, high: f64, bits: i32, t: fn(f64) -> f64, inv: fn(f64) -> f64) -> f64 {
        inv(q / 2f64.powi(bits) * (t(high) - t(low)) + t(low))
    }

    fn log_t(x: f64) -> f64 {
        (x + 1.0).log10()
    }
    fn log_inv(x: f64) -> f64 {
        10f64.powf(x) - 1.0
    }
    fn id(x: f64) -> f64 {
        x
    }
    fn sq(x: f64) -> f64 {
        x * x
    }

    fn close(got: f32, want: f64, name: &str) {
        let tol = 2e-5 * want.abs().max(1.0);
        assert!((got as f64 - want).abs() <= tol, "{name}: got {got}, want {want}");
    }

    #[test]
    fn each_feature_follows_its_setting() {
        // Columns 0 to 2 are the globals; the first sub-window block starts at column 3.
        let mut rec = [0u16; INPUT_COLUMNS];
        rec[0] = 512; // sum_accel_mg_std: 10 bits, log
        rec[1] = 128; // y_accel_std_ratio: 8 bits
        rec[2] = 64; // z_accel_std_ratio: 8 bits
        rec[3] = 256; // total_amplitude_mg: 9 bits, log
        rec[4] = 256; // stride_frequency: 9 bits, zero encoded
        rec[5] = 256; // stride_amplitude_frac: 9 bits, sqrt
        rec[6] = 128; // first_non_locomotor_frequency: 7 bits
        rec[7] = 128; // first_non_locomotor_amplitude_frac: 8 bits, sqrt
        rec[8] = 64; // gait_amplitude_frac: 8 bits, sqrt
        let rows = decode(&[0], &[rec]);
        let f = rows[0].features;
        close(f[0], dequantize(512.0, 0.0, 8000.0, 10, log_t, log_inv), "sum_accel_mg_std");
        close(f[1], dequantize(128.0, 0.0, 0.85, 8, id, id), "y_accel_std_ratio");
        close(f[2], dequantize(64.0, 0.0, 0.85, 8, id, id), "z_accel_std_ratio");
        close(f[3], dequantize(256.0, 0.0, 8000.0, 9, log_t, log_inv), "total_amplitude_mg");
        // encode_zero: (256 - 1) / 511 of the 0.68 to 3.4 range
        close(f[4], 0.68 + 255.0 / 511.0 * (3.4 - 0.68), "stride_frequency");
        close(f[5], dequantize(256.0, 0.0, 1.0, 9, f64::sqrt, sq), "stride_amplitude_frac");
        close(f[6], dequantize(128.0, 0.0, 24.804_688, 7, id, id), "first_non_locomotor_frequency");
        close(f[7], dequantize(128.0, 0.0, 0.65, 8, f64::sqrt, sq), "first_non_locomotor_amplitude_frac");
        close(f[8], dequantize(64.0, 0.0, 0.75, 8, f64::sqrt, sq), "gait_amplitude_frac");
    }

    #[test]
    fn stride_frequency_of_one_is_the_lower_bound() {
        let mut rec = [0u16; INPUT_COLUMNS];
        rec[4] = 1;
        let rows = decode(&[0], &[rec]);
        close(rows[0].features[4], 0.68, "q = 1");
        // q = 0 stays 0 in the same column when another record has a nonzero value
        let mut other = [0u16; INPUT_COLUMNS];
        other[4] = 300;
        let rows = decode(&[0, 30_000], &[[0; INPUT_COLUMNS], other]);
        assert_eq!(rows[0].features[4], 0.0);
        close(rows[3].features[4], 0.68 + 299.0 / 511.0 * 2.72, "q = 300");
    }

    #[test]
    fn globals_repeat_and_blocks_split_by_sub_window() {
        let mut rec = [0u16; INPUT_COLUMNS];
        rec[1] = 128; // y ratio: global, all three rows
        rec[3 + 7] = 128; // block 1, column 7 (bin_mid): row 0
        rec[3 + 8 + 7] = 64; // block 2: row 1
        rec[3 + 16 + 7] = 192; // block 3: row 2
        let rows = decode(&[90_000], &[rec]);
        for r in &rows {
            close(r.features[1], 0.425, "global y ratio");
        }
        close(rows[0].features[10], 0.5, "block 1");
        close(rows[1].features[10], 0.25, "block 2");
        close(rows[2].features[10], 0.75, "block 3");
        assert_eq!(rows.iter().map(|r| r.timestamp).collect::<Vec<_>>(), [70_000, 80_000, 90_000]);
    }

    #[test]
    fn gap_between_records_sets_the_row_spacing() {
        let rec = [0u16; INPUT_COLUMNS];
        // gap 31500 is within 25500..=34500, d = 10500; gaps of 25499 and 34501 fall back to 30000
        let rows = decode(&[0, 31_500, 57_000], &[rec, rec, rec]);
        let ts: Vec<i64> = rows.iter().map(|r| r.timestamp).collect();
        assert_eq!(&ts[3..6], &[31_500 - 21_000, 31_500 - 10_500, 31_500]);
        assert_eq!(&ts[6..9], &[57_000 - 17_000, 57_000 - 8_500, 57_000], "25500 gap: d = 8500");
        let rows = decode(&[0, 25_499], &[rec, rec]);
        assert_eq!(rows[3].timestamp, 25_499 - 20_000);
        assert_eq!(rows[5].timestamp, 25_499);
    }

    #[test]
    fn empty_input_gives_no_rows() {
        assert!(decode(&[], &[]).is_empty());
    }

}
