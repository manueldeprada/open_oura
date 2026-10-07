"""Per-day readiness orchestration of tools/score_daily_exact.py, as a reusable class.

`DailyScorer` carries the state kept between days: the long-term baseline (nine means and
deviations), the per-day history (total sleep seconds, equivalent distance, RMSSD, wear percent;
14 entries, newest first, zero padded) and the day counter. `step` takes one day's sleep periods
and previous-day statistics and returns the sleep and readiness outputs. The baseline of day D is
`baseline_calc.baseline_calculate` applied to the baseline of day D - 1 with the night and the
previous-day statistics that day D - 1 was scored with; the first day uses the baseline given to the constructor (the
cold-start baseline by default, `DEFAULT_MEANS`/`DEFAULT_DEVS`) and the temperature item is the
digital baseline (36.3 and 0.5 degrees) from the second day on.

Baseline. `DailyScorer()` starts from the cold-start baseline of an empty
recovery state. `DailyScorer(means, devs)` starts from caller-given values, for example a baseline
stored from an earlier session. There is no separate bootstrap step: `baseline_calc.baseline_calculate` moves each
item by 1/2 of the gap to the day's value on days 0 to 3, 1/8 up to day 14 and 1/32 afterwards (see
`baseline_calc.baseline_calculate`), so running the scorer over a user's first nights from the default
baseline adapts it gradually. The Rust mirror is
crates/oura-analysis/src/ported/daily.rs.
"""
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_calc as BC
import readiness_inputs as RI
import readiness_score as RS
import sri as SRI_STATE

SRI = 0x100 | 32
# Cold-start baseline of an empty recovery state. Items: sleep score, sleep time, resting HR, resting HR time,
# distance, vigorous minutes, sedentary minutes, temperature, HRV (8x scaled; HRV is 100 log2 RMSSD).
# These are the model's generic initial values, not a function of age or sex.
DEFAULT_MEANS = (720, 0, 480, 2880, 8000, 160, 3840, 29360, 4400)
DEFAULT_DEVS = (80, 360, 160, 480, 8000, 64, 1600, 400, 2000)
# Temperature item from day 1 on: 36.3 degrees and 0.5 degrees, times 800 (male, Gen3 or later).
TEMP_BASELINE = (29040, 400)

CONTRIBUTORS = ("activity_balance", "last_day_activity", "last_night_sleep", "resting_hr",
                "resting_hr_time", "sleep_balance", "temperature", "hrv_balance", "sleep_regularity")


@dataclass
class DailyResult:
    sleep: RI.DailySleep                      # the sleep output for the day's periods (the printed score)
    outputs: Optional[RI.DailyOutputs]        # the daily outputs, None when the input is rejected
    readiness_score: Optional[int]
    contributors: Optional[tuple]             # the nine readiness contributors, in CONTRIBUTORS order


class DailyScorer:
    @classmethod
    def with_baseline(cls, means, devs, **kw):
        """Start from the given nine means and deviations (8x scaled, in the item order above)."""
        return cls(means, devs, **kw)

    def __init__(self, means=DEFAULT_MEANS, devs=DEFAULT_DEVS, temp_baseline=TEMP_BASELINE, track_sri=False):
        self.sri_state = SRI_STATE.SriState() if track_sri else None   # None: use the `sri` argument of step
        self.means, self.devs = list(means), list(devs)
        self.temp_baseline = temp_baseline
        self.hist = []      # per earlier day: (sleep s, distance, rmssd, wear %), oldest first
        self.last = None    # arguments of the baseline update that the previous day's data feeds
        self.day = 0

    def step(self, day_utc, age, periods, prev_day, chrono=(0, 0), sri=SRI, history=None):
        """One day. `periods` are the day's SleepPeriod list (is_main set, total_sleep > 0),
        `prev_day` the 8 previous-day words (unused, distance, non-wear, sedentary, vigorous,
        light, moderate, resting), `day_utc` the sleep day's UTC midnight. `history` replaces the
        internal 14-day history by calendar days (indexed by date: entry k is the night
        of day D - 1 - k and the activity of day D - 2 - k, zero where there is no record); it is
        a list of 14 (sleep s, distance, RMSSD, wear %) tuples, newest first."""
        period = max(periods, key=lambda p: p.total_sleep)
        if self.sri_state is not None:
            self.sri_state.update_history((day_utc - 86400, 0), (period.start, period.tz),
                                          (period.start + period.time_in_bed, period.tz))
            sri = RS.sri_calculate(self.sri_state.rows(), True)
        ds = RI.get_daily_sleep(age, 0, 0, periods)
        if self.last is not None:
            self.means, self.devs = BC.baseline_calculate(self.means, self.devs, self.day, 0, *self.last)
            self.means[7], self.devs[7] = self.temp_baseline
        h = (list(history) if history is not None else self.hist[::-1][:14]) + [None] * 14
        pad = lambda k, z: [x[k] if x else z for x in h[:14]]
        history = RI.History([-32768] * 3, [0] * 90, pad(0, 0), pad(1, 0), pad(2, 0), pad(3, 0))
        bl = bytearray(0x38)
        for k in range(9):
            struct.pack_into("<HH", bl, 0x10 + 4 * k, self.means[k], self.devs[k])
        out = RI.daily_outputs_stateless(day_utc, 0, self.day, periods, prev_day, (0, age), history,
                                         bytes(bl), chrono, (0, 0, 0, day_utc), 0, None, sri)
        sleep_score = out.sleep.score if out else ds.score  # the day's final sleep score
        wear = BC.wear_percent(prev_day[3], prev_day[5], prev_day[6], prev_day[4], prev_day[7])
        self.last = (sleep_score, period.time_in_bed, period.total_sleep, period.lowest_hr, period.lowest_hr_time,
                     prev_day[1], prev_day[4], prev_day[3], period.highest_temp, period.rmssd,
                     prev_day[5], prev_day[6], prev_day[7])
        self.hist.append((period.total_sleep, prev_day[1], period.rmssd, wear))
        self.day += 1
        if out is None:
            return DailyResult(ds, None, None, None)
        ro = out.readiness
        return DailyResult(ds, out, ro.score, tuple(getattr(ro, n) for n in CONTRIBUTORS))
