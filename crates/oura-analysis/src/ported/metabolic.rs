//! Metabolic helpers for the activity and calorie path: standard published formulas.
//! `bmr_*` is computed in integer-rounded kcal/day (see `actinfo::bmr`).

/// Anthropometric VO2max estimate (Jackson NES):
/// `79.9 − 0.39·age − 13.7·female − 0.127·(weight_kg·2.20462)`, floored at 0.
pub fn vo2max_jackson(age_years: f64, female: bool, weight_kg: f64) -> f64 {
    let weight_lb = weight_kg * 2.20462;
    (79.9 - 0.39 * age_years - 13.7 * (female as i32 as f64) - 0.127 * weight_lb).max(0.0)
}

/// VO2max from a 6-minute walk, the estimate behind Cardio Capacity. `gender`: 1 female,
/// 2 or 3 male, otherwise 0. `distance_m` is the GPS route length of the 6-minute window,
/// `hr_bpm` the median workout heart rate of the final 70 s of it. Floored at 0. Callers
/// round the result to an integer.
///
/// The female branch ignores height and heart rate.
pub fn vo2max_ukk_6min(
    age_years: f64,
    weight_kg: f64,
    gender: u8,
    distance_m: f64,
    height_cm: f64,
    hr_bpm: f64,
) -> f64 {
    let v = match gender {
        1 => 22.506 - 0.271 * weight_kg + 0.051 * distance_m - 0.065 * age_years,
        2 | 3 => {
            let bmi = weight_kg / (height_cm / 100.0).powi(2);
            110.546 + 0.063 * distance_m - 0.25 * age_years - 0.486 * bmi - 0.42 * height_cm
                - 0.109 * hr_bpm
        }
        _ => 0.0,
    };
    v.max(0.0)
}

/// Schofield basal metabolic rate (kcal/day) by age band and sex, without the blending
/// between bands that [`bmr_kcal`] applies. `sex`: 0=male, 1=female; anything else returns
/// the male/female average.
///
/// Kept for callers that need the pre-fix values. [`bmr_schofield`] is the exact version.
pub fn bmr_schofield_band(age_years: f64, sex: u8, weight_kg: f64) -> f64 {
    fn band(coeffs: &[(f64, f64, f64)], age: f64, kg: f64) -> f64 {
        let &(_, s, i) = coeffs
            .iter()
            .find(|&&(hi, _, _)| age < hi)
            .unwrap_or(coeffs.last().unwrap());
        s * kg + i
    }
    // (upper-age-bound, slope, intercept)
    const MALE: [(f64, f64, f64); 6] = [
        (3.0, 59.512, -30.4),
        (10.0, 22.706, 504.3),
        (18.0, 17.686, 658.2),
        (30.0, 15.057, 692.2),
        (60.0, 11.472, 873.1),
        (f64::INFINITY, 11.711, 587.7),
    ];
    const FEMALE: [(f64, f64, f64); 6] = [
        (3.0, 58.317, -31.1),
        (10.0, 20.315, 485.9),
        (18.0, 13.384, 692.6),
        (30.0, 14.818, 486.6),
        (60.0, 8.126, 845.6),
        (f64::INFINITY, 9.082, 658.5),
    ];
    match sex {
        0 => band(&MALE, age_years, weight_kg),
        1 => band(&FEMALE, age_years, weight_kg),
        _ => (band(&MALE, age_years, weight_kg) + band(&FEMALE, age_years, weight_kg)) / 2.0,
    }
}

/// Basal metabolic rate in whole kcal/day: the band formulas blended linearly in age between band
/// centers, rounded to the nearest integer. `sex`: 0=male, 1=female, anything else the
/// mean of both. Age and weight are whole years and kilograms taken modulo 256 (see `actinfo::bmr`).
pub fn bmr_kcal(age_years: u32, weight_kg: u32, sex: u8) -> u32 {
    let gender_code = match sex {
        0 => 2,
        1 => 1,
        _ => 3,
    };
    super::actinfo::bmr(age_years, weight_kg, gender_code)
}

/// Basal metabolic rate (kcal/day) with the signature open_health calls, now exact: age is
/// truncated and weight rounded to whole units, then [`bmr_kcal`] applies. The result is a
/// whole number of kcal and differs from [`bmr_schofield_band`] by up to a few percent near
/// band edges. `sex`: 0=male, 1=female, anything else the mean of both.
pub fn bmr_schofield(age_years: f64, sex: u8, weight_kg: f64) -> f64 {
    bmr_kcal(
        age_years.max(0.0) as u32,
        weight_kg.max(0.0).round() as u32,
        sex,
    ) as f64
}

/// Walking distance from step count: `0.762 m` per step.
pub fn steps_to_meters(steps: u32) -> u32 {
    (steps as f64 * 0.762) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vo2max_plausible() {
        // 30yo male, 75 kg -> ~47
        let v = vo2max_jackson(30.0, false, 75.0);
        assert!((44.0..50.0).contains(&v), "{v}");
    }

    #[test]
    fn ukk_6min_reference_values() {
        // Reference values for the estimate.
        let cases = [
            ((28.0, 85.0, 2, 520.0, 180.0, 125.0), 34.331),
            ((35.0, 62.0, 1, 480.0, 168.0, 130.0), 27.909),
            ((60.0, 70.0, 2, 300.0, 175.0, 150.0), 13.487428571428579),
            ((45.0, 75.0, 1, 650.0, 165.0, 95.0), 32.406),
            ((28.0, 85.0, 0, 520.0, 180.0, 125.0), 0.0),
        ];
        for ((a, w, g, d, h, hr), want) in cases {
            let got = vo2max_ukk_6min(a, w, g, d, h, hr);
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }

    #[test]
    fn bmr_band_adult_male() {
        // 30-60 band: 11.472*80 + 873.1 = 1790.86
        assert!((bmr_schofield_band(40.0, 0, 80.0) - 1790.86).abs() < 0.01);
    }

    #[test]
    fn bmr_exact_matches_actinfo_port() {
        // Ages inside a band and between band centers.
        for (age, kg) in [(29u32, 82u32), (40, 80), (1, 10), (95, 60), (13, 45)] {
            for (sex, jg) in [(0u8, 2u32), (1, 1), (2, 3)] {
                assert_eq!(
                    bmr_kcal(age, kg, sex),
                    super::super::actinfo::bmr(age, kg, jg)
                );
                assert_eq!(
                    bmr_schofield(age as f64, sex, kg as f64),
                    bmr_kcal(age, kg, sex) as f64
                );
            }
        }
        // At a band center the blend equals that band's formula (age 24 = 48 half-years).
        let band = bmr_schofield_band(24.0, 0, 80.0); // 18-30 band: 15.057*80+692.2
        assert_eq!(bmr_kcal(24, 80, 0), band.round() as u32);
        // Between centers the blend differs from the band value.
        assert_ne!(
            bmr_kcal(40, 80, 0) as f64,
            bmr_schofield_band(40.0, 0, 80.0).round()
        );
    }

    #[test]
    fn distance() {
        assert_eq!(steps_to_meters(10000), 7620);
    }
}
