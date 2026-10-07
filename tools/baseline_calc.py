#!/usr/bin/env python3
"""The per-day update of the long-term baseline.

The update runs once per day with the
baseline of day D - 1, the night of day D - 1 (its sleep summary and final sleep score) and the
activity statistics that day D - 1's readiness input carried as "previous day" (day D - 2). The
result is the baseline of day D, which that day's readiness reads. The day-0 baseline is the
default baseline (`daily_scorer.SEED_MEANS`) and is not updated.
The temperature item is replaced by the digital baseline (`daily_scorer.TEMP_BASELINE`) before
it is used or carried on. The athlete-fix feature flag is on (blend of the previous baseline
into the distance and sedentary values by the wear percent); the mean lower bounds come from
`readiness_score.mean_min_table(28)`.

Item order and values (each goes through `readiness_score.baseline_update` with n = day number):
0 sleep score; 1 round(total sleep / 60 s); 2 lowest HR and 3 round((time in bed - lowest HR time)
/ 60 s), both only when time in bed exceeds the lowest HR time and both are non-zero; 4 equivalent
distance blended with the previous baseline by the previous day's wear percent; 5 vigorous minutes
and 6 sedentary minutes, both only when distance is non-zero and wear is at least 33 percent;
7 highest temperature (centidegrees) when within 200 of the baseline mean, with n = ring day;
8 100 log2(RMSSD).
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readiness_score as R

MIN_SLEEP_J = 28  # j of the sleep-time minimum in the mean-min table
NAMES = ("sleepScore", "sleepTime", "restingHr", "restingHrTime", "activityDistance", "vigorousActivity",
         "sedentaryActivity", "temperature", "hrv")


def wear_percent(sedentary, light, moderate, vigorous, resting):
    """Percent of the non-sleep day the ring was worn, from the
    class minutes (0 when the day is all resting)."""
    if resting == 0x5A0:
        return 0
    return int(R.f32(R.f32(R.f32(float(sedentary + light + moderate + vigorous)) * 100.0) / float(0x5A0 - resting) + 0.5))


def baseline_calculate(means, devs, day, ring_day, sleep_score, time_in_bed, total_sleep, lowest_hr, lowest_hr_time,
                       eq_distance, vigorous, sedentary, highest_temp, rmssd, light, moderate, resting, feats=None, j=None):
    """Return (means, devs) of the nine items after one day's update. Inputs are the previous baseline and the day's statistics."""
    feats = feats or R.Features()  # athlete-adjusted contributors on
    S = R.State()
    for k in range(9):
        S.set16(0x40 + 2 * k, means[k])
        S.set16(0x52 + 2 * k, devs[k])
    wear = wear_percent(sedentary, light, moderate, vigorous, resting)
    n = day
    mm = R.mean_min_table(MIN_SLEEP_J if j is None else j)
    up = lambda item, val, nn=n: R.baseline_update(S, item, val, nn, feats, mm)
    up(0, sleep_score)
    up(1, R.round_half_away_i32(R.f32(R.f32(float(total_sleep)) / 60.0)))
    if time_in_bed > lowest_hr_time and lowest_hr and lowest_hr_time:
        up(2, lowest_hr)
        up(3, R.round_half_away_i32(R.f32(R.f32(float(time_in_bed - lowest_hr_time)) / 60.0)))
    eq8 = (eq_distance >> 3) & 0xFFFF
    dist_today = 0
    if eq8:
        mean_eq = R.decode16(S.u16(0x48))
        if feats.athletes:
            tq = (100 - wear) * mean_eq + (4 if mean_eq > 0 else -4)
            q = R.tdiv(tq, 8)
            dist_today = eq8 + (R.div100_round(q) if tq >= 8 else -((50 - q) // 100))
        else:
            dv = S.i16_at(0x5A)
            tq = (100 - wear) * dv + (4 if dv > 0 else -4)
            dist_today = R.div100_round(wear * eq8 + R.tdiv(tq, 8))
    up(4, dist_today)
    if (dist_today & 0xFFFF) != 0 and wear / 100.0 >= 0.33:
        up(5, vigorous)
        sed_min = sedentary
        if feats.athletes and R.sx16(sedentary) > 0:
            sm = S.i16_at(0x4C)
            tq = (100 - wear) * sm + (4 if sm > 0 else -4)
            q = R.tdiv(tq, 8)
            sed_min += (q + 50) // 100 if tq >= 8 else -((50 - q) // 100)
        up(6, sed_min)
    if 0 < highest_temp <= 0x10CB and abs(R.f32(float(highest_temp) + R.f32(float(S.i16_at(0x4E)) * -0.125))) < 200.0:
        up(7, highest_temp, ring_day)
    if rmssd:
        v = R.f32(R.f32(R.math.log2(float(rmssd))) * 100.0)
        if v > 0:
            up(8, R.round_half_away_i32(v))
    return [S.u16(0x40 + 2 * k) for k in range(9)], [S.u16(0x52 + 2 * k) for k in range(9)]



