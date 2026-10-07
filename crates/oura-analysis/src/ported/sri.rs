//! Sleep-regularity state: the day rows of sleep and awake slots, updated from sleep periods.
//!
//! The state keeps up to 15 day rows of 144 ten-minute slots (0 awake, 1 asleep, 2 missing),
//! a row count and the sleep date of the last update. It is filled by
//! [`SriState::init_sleeps`] (one `update_history` per stored sleep, oldest first), and
//! each ring sleep summary processed afterwards calls `update_history` again (with the time zone of the
//! ring's `time_sync` event). [`SriState::value`] is
//! `sri_calculate(true)`. Times are `(utc_seconds, tz_minutes)`.

use super::readiness::{sri_calculate, SRI_COLUMNS};

const DAY: i64 = 86400;
const ROWS: usize = 15;

#[derive(Clone, Copy, Debug)]
struct Row {
    date: i64,
    dur: u32,
    slots: [u8; SRI_COLUMNS],
}

impl Row {
    fn new() -> Row {
        Row {
            date: 0,
            dur: 0,
            slots: [2; SRI_COLUMNS],
        }
    }
}

/// `(utc seconds, tz minutes)`.
pub type Time = (i64, i32);

/// UTC midnight of the local calendar day.
pub fn sleep_date(t: Time) -> i64 {
    (t.0 + 60 * t.1 as i64).div_euclid(DAY) * DAY
}

/// Ten-minute slot of the local time of day.
pub fn slot(t: Time) -> usize {
    let s = (t.0 + 60 * t.1 as i64).rem_euclid(DAY) as usize;
    (s / 3600) * 6 + (s % 3600) / 600
}

#[derive(Clone, Debug)]
pub struct SriState {
    rows: [Row; ROWS],
    count: usize,
    last: i64,
}

impl Default for SriState {
    fn default() -> Self {
        SriState {
            rows: [Row::new(); ROWS],
            count: 0,
            last: 0,
        }
    }
}

impl SriState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stored day rows, oldest first.
    pub fn rows(&self) -> Vec<[u8; SRI_COLUMNS]> {
        self.rows[..self.count].iter().map(|r| r.slots).collect()
    }

    /// `sri_calculate(true)`: bit 8 set when valid, low byte the index.
    pub fn value(&self) -> u32 {
        sri_calculate(&self.rows(), true)
    }

    fn idx(&self, date: i64) -> Option<usize> {
        self.rows[..self.count].iter().position(|r| r.date == date)
    }

    fn add_new_day(&mut self, t: Time) {
        let d = sleep_date(t);
        if self.count >= ROWS {
            for i in 0..ROWS - 1 {
                self.rows[i] = self.rows[i + 1];
            }
            let r = &mut self.rows[ROWS - 1];
            r.slots = [2; SRI_COLUMNS];
            r.date = d;
        } else {
            let r = &mut self.rows[self.count];
            r.slots = [2; SRI_COLUMNS];
            r.date = d;
            self.count += 1;
        }
    }

    fn ensure(&mut self, t: Time) -> Option<usize> {
        if self.idx(sleep_date(t)).is_none() {
            self.add_new_day(t);
        }
        self.idx(sleep_date(t))
    }

    fn write_sleep_day(&mut self, t: Time, typ: u8) {
        if let Some(i) = self.ensure(t) {
            self.rows[i].slots[108..].fill(typ);
        }
        if let Some(j) = self.ensure((t.0 + DAY, t.1)) {
            self.rows[j].slots[..108].fill(typ);
        }
    }

    fn fill(&mut self, i: Option<usize>, a: usize, len: usize, v: u8) {
        if let Some(i) = i {
            let end = (a + len).min(SRI_COLUMNS);
            if a < end {
                self.rows[i].slots[a..end].fill(v);
            }
        }
    }

    fn write_new_sleep_period(&mut self, d: Time, s: Time, e: Time) {
        let (ss, se) = (slot(s), slot(e));
        if let Some(i) = self.ensure(d) {
            self.rows[i].dur = (e.0 - s.0) as u32;
        }
        let ds = sleep_date(s);
        if self.idx(ds).is_none() {
            self.add_new_day((ds, 0));
        }
        let de = sleep_date(e);
        if self.idx(de).is_none() {
            self.add_new_day((de, 0));
        }
        let i_s = self.idx(ds);
        if ss > se {
            if ss == 0 || ss >= 109 {
                self.fill(i_s, 108, ss.max(109) - 108, 0);
            }
            self.fill(i_s, ss, 0x90 - ss, 1);
            let i_e = self.idx(de);
            self.fill(i_e, 0, se + 1, 1);
            if se + 1 <= 0x6B {
                self.fill(i_e, se + 1, 107 - se, 0);
            }
        } else {
            if let Some(i) = i_s {
                if i >= 1 {
                    self.rows[i - 1].slots[108..].fill(0);
                }
            }
            if ss >= 1 {
                self.fill(i_s, 0, ss, 0);
            }
            self.fill(i_s, ss, se - ss + 1, 1);
            if se + 1 <= 0x6B {
                self.fill(i_s, se + 1, 107 - se, 0);
            }
        }
    }

    ///.
    pub fn update_history(&mut self, d: Time, s: Time, e: Time) {
        let i = self.idx(sleep_date(d));
        if d.0 < self.rows[0].date {
            return;
        }
        if let Some(i) = i {
            if (e.0 - s.0) <= self.rows[i].dur as i64 {
                self.last = d.0;
                return;
            }
            self.write_sleep_day(d, 0);
        } else if self.last != 0 && d.0 >= self.last {
            let diff = ((d.0 - self.last) / DAY) & 0xFFFF;
            if diff & 0xFFFE != 0 {
                for k in 1..diff {
                    self.write_sleep_day((self.last + k * DAY, d.1), 2);
                }
            }
        }
        self.write_new_sleep_period(d, s, e);
        self.last = d.0;
    }

    /// Clear, then update once per sleep, oldest first.
    pub fn init_sleeps(&mut self, sleeps: &[(Time, Time, Time)]) {
        *self = Self::default();
        for &(d, s, e) in sleeps {
            self.update_history(d, s, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! vector_tests {
        () => {
            #[test]
            fn matches_reference_sequences() {
                for (n, (ops, want)) in sri_sequences().into_iter().enumerate() {
                    let mut st = SriState::new();
                    for (d, s, e) in ops {
                        st.update_history(d, s, e);
                    }
                    assert_eq!(st.value(), want, "sequence {n}");
                }
            }
        };
    }

    mod python_port {
        use super::*;
        include!("vectors/sri.rs");
        vector_tests!();
    }

}
