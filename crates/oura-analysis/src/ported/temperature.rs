//! Nightly skin temperature + deviation. Ported from
//! A 7-sample sliding
//! median, then 30-sample windows; each window contributes its max when its
//! range < 2.50 °C. The nightly value is the **second-highest window maximum,
//! counting ties** (needs >= 4 valid windows). Units: i16 centi-°C (value/100 = °C).
//! The computation lives in `sleep_summary::nightly_temperature`; this module keeps the
//! `Option<i16>` signature and the deviation helper.

/// Nightly temperature (centi-°C) from per-sample temps (centi-°C; 0 = invalid).
/// `None` if fewer than 4 valid 30-sample windows passed the range gate.
///
/// Delegates to [`super::sleep_summary::nightly_temperature`], the port used here. This signature
/// (`Option<i16>`, centi-°C) is what open_health calls.
pub fn nightly_temperature(samples: &[u16]) -> Option<i16> {
    match super::sleep_summary::nightly_temperature(samples) {
        0 => None,
        v => Some(v.min(i16::MAX as u16) as i16),
    }
}

/// Temperature deviation (centi-°C) = nightly temperature − personal baseline mean.
pub fn temperature_deviation(nightly_centi: i16, baseline_mean_centi: f64) -> f64 {
    nightly_centi as f64 - baseline_mean_centi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_night() {
        // 150 steady samples at 35.00 °C -> 5 windows, all max 3500 -> 3500
        let s = vec![3500u16; 150];
        assert_eq!(nightly_temperature(&s), Some(3500));
    }

    #[test]
    fn too_few_windows() {
        // 60 samples -> 2 windows, 50 -> 1 window; both < 4 required -> None
        assert_eq!(nightly_temperature(&[3500u16; 60]), None);
        assert_eq!(nightly_temperature(&[3500u16; 50]), None);
    }
}
