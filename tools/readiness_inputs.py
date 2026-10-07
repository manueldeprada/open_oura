#!/usr/bin/env python3
"""Input assembly of the Oura daily sleep and readiness scores.

`tools/readiness_score.py` holds the arithmetic (`readiness_calculate`, `recovery_run`) and
`tools/sleep_score.py` the sleep score combiner; this module builds their arguments from the
day's data:

    sleep periods (records), user info, previous-day activity, baseline,
    14/90-day histories, rest-mode input, chronotype, cycle day type
        -> DailyOutputs(sleep, readiness, per-prefix scores and deltas, readiness v2 inputs)

The day is processed prefix by prefix: for k = 1..n the first k periods are scored
(`get_daily_sleep`, then `daily_readiness_stateless`), which gives the score history
(the per-prefix deltas). The last prefix supplies every output field except the sleep
score, which is re-derived from the prefix scores (`final_sleep_score`).

Sleep period record: is_main, bedtime start (Unix s), time zone offset (minutes), time in bed, total sleep,
REM, deep (s), efficiency (percent), latency (s), wake-up count, restless periods, got-up count, sleep
midpoint (30 s epochs after the start), highest temperature (centidegrees), lowest HR, lowest-HR time (s),
RMSSD.

get_daily_sleep. Each period whose midpoint (start + tz * 60 + midpoint * 30) lies in
[1354757888, 2522367489) is scored alone with the minutes-based sleep score (durations in
rounded minutes, latency as |latency - 900 s|, ratios per 60 s of sleep). The period midpoint
as minutes after local midnight, wrapped to [-14 h, 10 h), is the timing input; for periods that
are not the main one it is moved by +-12 h toward the nearest of three reference times
(5400 s, 48600 s, -37800 s). One valid period gives its own result. With several valid periods,
the results are recomputed on the running sums (total, deep, REM, counters weighted by sleep
time, the main period's latency, midpoint 180 min), and the timing contributor is replaced
by the sleep-time-weighted mean of the periods' timing contributors (the score is corrected by
(mean // 10) - (contributor * 205 >> 11)).

daily_readiness_stateless. The "main" period is the one with the largest total sleep among
those with is_main != 0 (first on ties). Periods with a valid start (as above, without tz or
midpoint) are summed (time in bed, total, REM, deep); the lowest HR, highest temperature and
highest RMSSD come from the periods from the main one on; the lowest-HR time is that of the
longest period seen so far. The highest temperature is the maximum when it exceeds the main
period's temperature by more than 50, otherwise the main period's. The record is passed to
`readiness_calculate` with the sleep score of `get_daily_sleep`, the first user-info words, the
previous day, histories, baseline, cycle day type and rest-mode input unchanged.
"""
import math
import os
import sys
import time as _time
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readiness_score as R
import sleep_score as S

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
TIME_LO = -0x50B7F700           # periods before 1354757888 (2012-12-06) are ignored
TIME_SPAN = 0x45984F01          # and so are those 1167609601 s (to 2049) after it
PERIOD_SIZE = 0x58


@dataclass
class SleepPeriod:
    """the sleep period record (fields read by the daily functions)."""
    is_main: int = 0
    start: int = 0
    tz: int = 0
    time_in_bed: int = 0
    total_sleep: int = 0
    rem: int = 0
    deep: int = 0
    efficiency: int = 0
    latency: int = 0
    wake_ups: int = 0
    restless: int = 0
    got_up: int = 0
    midpoint: int = 0
    highest_temp: int = 0
    lowest_hr: int = 0
    lowest_hr_time: int = 0
    rmssd: int = 0

    def pack(self):
        import struct
        b = bytearray(PERIOD_SIZE)
        struct.pack_into("<I", b, 0x08, self.is_main & M32)
        struct.pack_into("<q", b, 0x10, self.start)
        struct.pack_into("<i", b, 0x18, self.tz)
        struct.pack_into("<14I", b, 0x20, *[v & M32 for v in (
            self.time_in_bed, self.total_sleep, self.rem, self.deep, self.efficiency, self.latency,
            self.wake_ups, self.restless, self.got_up, self.midpoint, self.highest_temp, self.lowest_hr,
            self.lowest_hr_time, self.rmssd)])
        return bytes(b)


@dataclass
class DailySleep:
    """the daily sleep output record: score, seven
    contributors (total sleep, deep, REM, efficiency, latency, disturbances, circadian
    alignment), total sleep sum (s), highest temperature, lowest HR, lowest-HR time, RMSSD (the last four are written by the readiness step)."""
    score: int = 0
    contributors: tuple = (0,) * 7
    total_sleep_sum: int = 0
    highest_temp: int = 0
    lowest_hr: int = 0
    lowest_hr_time: int = 0
    rmssd: int = 0

    def words(self):
        return [self.score, *self.contributors, self.total_sleep_sum, self.highest_temp,
                self.lowest_hr, self.lowest_hr_time, self.rmssd]


def valid_time(t):
    return ((t + TIME_LO) & M64) < TIME_SPAN


def _tdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _s32(v):
    v &= M32
    return v - (1 << 32) if v & 0x80000000 else v


def _div60r(x):
    """(x + 30) / 60 on u32 (a multiply-shift reciprocal)."""
    return ((x + 30) & M32) // 60


def get_daily_sleep(age, chrono_type, chrono_mid, periods, limits=S.limits_v1):
    """Daily sleep output; `age` is
    the user-info u32 at +4. `limits` is the version 1 knot table. Returns DailySleep (all zero when no period scores)."""
    n = len(periods)
    s_total = 0
    main_idx = 0
    inv = 0
    day_score = 0
    weighted = 0
    acc = None
    summary = [0] * 7
    contrib6 = 0
    for i, p in enumerate(periods):
        t = (p.start + _s32(p.tz) * 60 + ((p.midpoint * 30) & M32)) & M64
        if not valid_time(t):
            inv += 1
            continue
        tm = _time.gmtime(t - (1 << 64) if t >> 63 else t)
        hour = tm.tm_hour
        mins = hour * 60 if hour < 10 else -(24 - hour) * 60
        sod = _s32((mins + tm.tm_min) * 60 + tm.tm_sec)
        lat = _s32(p.latency - 900)
        lat = lat if lat >= 0 else _s32(900 - p.latency)
        if p.is_main == 1:
            main_idx = i
            mid_s = _s32(sod + 30)
        else:
            mid_s = _s32(sod + 30)
            a, b = abs(_s32(sod - 5400)), abs(_s32(sod - 48600))
            c = _s32(sod + 37800)
            c = c if c >= 0 else _s32(-37800 - sod)
            if not (a < c and a < b):
                mid_s = _s32(sod + (0xA8DE if c < b else -0xA8A2))
        tot, rem, deep, eff = p.total_sleep & M32, p.rem & M32, p.deep & M32, p.efficiency & M32
        wake_s = (p.wake_ups * 0x1A40) & M32
        got_up_s = (p.got_up * 0x1A40) & M32
        tot_min = tot // 60
        eff_sum = (eff * tot_min) & M32
        restless_sum = (p.restless * tot_min) & M32
        if tot >= 60:
            eff_avg, wake_avg, restless_avg, ext = eff_sum // tot_min, wake_s // tot_min, restless_sum // tot_min, got_up_s // tot_min
        else:
            eff_avg = wake_avg = restless_avg = ext = 0
        s_total = (s_total + tot) & M32
        mid = _tdiv(mid_s, 60)
        summary, sc = S.sleep_score_minutes(
            age, _div60r(tot), _div60r(deep), _div60r(rem), eff_avg, _div60r(lat), wake_avg, restless_avg, ext & 0xFF, mid,
            chrono_type & 0xFF, chrono_mid, limits)
        sc &= 0xFF
        contrib6 = summary[6]
        if (n - (inv & 0xFF)) & M32 == 1:
            day_score = sc
            break
        weighted = (weighted + tot * contrib6) & M32
        if i == (inv & 0xFF):
            acc = dict(tot=tot, deep=deep, rem=rem, ext=got_up_s, eff=eff_sum, wake=wake_s, tosses=restless_sum)
            day_score = sc
            continue
        acc["tot"] = (acc["tot"] + tot) & M32
        acc["rem"] = (acc["rem"] + rem) & M32
        acc["eff"] = (acc["eff"] + (((eff * tot) & M32) // 60)) & M32
        wt = acc["tot"] // 60
        div_trunc = (lambda a_, b_: a_ // b_ if b_ else 0)
        eff_avg = div_trunc(acc["eff"], wt)
        acc["wake"] = (acc["wake"] + p.wake_ups * 0x1A40) & M32
        wake_avg = div_trunc(acc["wake"], wt)
        acc["tosses"] = (acc["tosses"] + ((((p.restless * tot) & M32)) // 60)) & M32
        restless_avg = div_trunc(acc["tosses"], wt)
        acc["ext"] = (acc["ext"] + p.got_up * 0x1A40) & M32
        acc["deep"] = (acc["deep"] + deep) & M32
        main_lat = _s32(periods[main_idx].latency - 900)
        main_lat = main_lat if main_lat >= 0 else _s32(900 - periods[main_idx].latency)
        ext = div_trunc(acc["ext"], wt)
        summary, sc = S.sleep_score_minutes(
            age, _div60r(acc["tot"]), _div60r(acc["deep"]), _div60r(acc["rem"]), eff_avg, _div60r(main_lat),
            wake_avg, restless_avg, ext & 0xFF, 0xB4, chrono_type & 0xFF, chrono_mid, limits)
        contrib6 = weighted // s_total if s_total else 0
        old = summary[6]
        summary = list(summary)
        summary[6] = contrib6 & 0xFF
        day_score = ((((10 * contrib6) & M32) * 0x51EB851F) >> 37) - (((old * 0xCD) & M32) >> 11) + (sc & 0xFF)
        day_score &= M32
    if day_score == 0:
        return DailySleep()
    return DailySleep(score=day_score, contributors=tuple(summary[:6]) + (contrib6 & 0xFF,), total_sleep_sum=s_total)


@dataclass
class History:
    """the readiness history record; lists newest first."""
    temp_dev_3: Optional[List[int]] = None
    hot_90: Optional[List[int]] = None
    sleep_sec_14: Optional[List[int]] = None
    distance_14: Optional[List[int]] = None
    rmssd_14: Optional[List[int]] = None
    wear_14: Optional[List[int]] = None


def aggregate_periods(periods, rmssd_init=0):
    """First half of `daily_readiness_stateless`: the night record from the periods.
    Returns None when no period has a valid start, else
    (sums[time in bed, total, REM, deep], temp, lowest_hr, lowest_hr_time, rmssd). The stateful
    path is the same scan with the highest RMSSD starting at 1."""
    n = len(periods)
    found, best, idx = False, 0, M32
    for i, p in enumerate(periods):
        if p.is_main != 0:
            found = True
            if p.total_sleep & M32 > best:
                best, idx = p.total_sleep & M32, i
    v1 = [0] * 4
    lh_time = main_temp = max_temp = inv = best_tot = before_main = seen_main = 0
    max_rmssd = rmssd_init
    min_hr = M32
    for i, p in enumerate(periods):
        if found:
            ismain = int((idx - i) & M64 == 0 and p.is_main != 0)
            before_main = int((not seen_main or before_main) and not ismain)
            seen_main |= ismain
        if not valid_time(p.start):
            inv += 1
            continue
        fields = [p.time_in_bed & M32, p.total_sleep & M32, p.rem & M32, p.deep & M32]
        v1 = [(a + b) & M32 for a, b in zip(v1, fields)]
        tot = fields[1]
        if tot > best_tot:
            lh_time = p.lowest_hr_time & M32
            if before_main:
                best_tot = tot
                continue
            best_tot = tot
            main_temp = p.highest_temp & M32
        temp_p, hr, rm = p.highest_temp & M32, p.lowest_hr & M32, p.rmssd & M32
        if not (hr >= min_hr or before_main):
            min_hr = hr
        if not (temp_p <= max_temp or before_main):
            max_temp = temp_p
        if not (rm <= max_rmssd or before_main):
            max_rmssd = rm
    if (inv & 0xFF) >= n:
        return None
    temp = max_temp if max_temp > ((main_temp + 0x32) & M32) else main_temp
    return v1, temp, min_hr, lh_time, max_rmssd


def daily_readiness_stateless(sleep: DailySleep, prev_day, periods, user, history: History, baseline,
                              day_number, time0, cycle_day_type, rest_input, feats=None, sri=0):
    """Returns (ReadinessOutput, DailySleep) with the sleep
    output updated. `user` is (u32 at +0, age); a missing pointer is
    None. The ReadinessOutput is all zero when no period is usable."""
    agg = aggregate_periods(periods)
    if agg is None:
        return R.ReadinessOutput(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), sleep
    v1, temp, hr, hr_time, rmssd = agg
    x = R.ReadinessInput()
    x.utc = time0
    x.day_number = day_number
    x.sleep_score = sleep.score
    x.time_in_bed, x.total_sleep, x.rem, x.deep = v1
    x.latency = x.wake_ups = 0
    x.highest_temp, x.lowest_hr, x.lowest_hr_time, x.rmssd = temp, hr, hr_time, rmssd
    x.p40 = user
    x.prev_day = prev_day
    x.sleep_sec_14, x.distance_14, x.rmssd_14 = history.sleep_sec_14, history.distance_14, history.rmssd_14
    x.wear_14, x.temp_dev_3, x.hot_90 = history.wear_14, history.temp_dev_3, history.hot_90
    x.baseline = baseline
    x.cycle_day_type = cycle_day_type
    x.rest_input = rest_input
    out = R.readiness_calculate(x, feats, sri=sri)
    if out is None:
        out = R.ReadinessOutput(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    sleep = DailySleep(sleep.score, sleep.contributors, v1[1], temp, hr, hr_time, rmssd)
    return out, sleep


def daily_readiness_stateful(sleep: DailySleep, prev_day, periods, S: R.State, rm: R.RestMode, j, j1, j2, upd,
                             time0, feats=None, sri=0):
    """The stateful (V1) path: the same night record,
    but `recovery_run` runs on the persistent state `S` (modified in place, with baseline
    updates when `upd`) and rest-mode block `rm`. Returns (ReadinessOutput, DailySleep); the
    output's temperature deviation and trend are not computed (0 here). `j` is
    the baseline sleep-time input of `recovery_run`, `j1` the state day word, `j2` the baseline
    age used by the temperature update."""
    agg = aggregate_periods(periods, 1)
    if agg is None:
        return R.ReadinessOutput(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), sleep
    (tib, total, rem, deep), temp, hr, hr_time, rmssd = agg
    light = (total - (deep + rem)) // 30 if total >= (deep + rem) & M32 else 0
    awake = (tib - total) // 30 if tib >= total else 0
    ss = R.SleepSummary(awake=awake, rem=rem // 30, light=light, deep=deep // 30, extra=hr_time // 30,
                        b0a=hr & 0xFF, sleep_score=sleep.score & 0xFF)
    hours = int(R.f32(R.f32(R.f32(float(tib)) + 1800.0) / 3600.0))
    S.b[0x65] = prev_day[4] & 0xFF
    S.set16(0x20, prev_day[1] >> 3)
    S.set16(0x3E, prev_day[3])
    f = R.f32(math.log2(float(rmssd)))
    summary, _ = R.recovery_run(S, S, ss, rm, temp, hours, f, j, j1, j2, upd, (hr, hr_time), time0, feats, sri)
    out = R.ReadinessOutput(
        score=R._sb(summary[0]), activity_balance=R._sb(summary[4]), last_day_activity=R._sb(summary[3]),
        last_night_sleep=R._sb(summary[1]), resting_hr=R._sb(summary[5]), resting_hr_time=R._sb(summary[7]),
        sleep_balance=R._sb(summary[2]), temperature=R._sb(summary[6]), hrv_balance=R._sb(summary[8]),
        sleep_regularity=-1 if summary[9] == 0x80 else R._sb(summary[9]), temperature_deviation=0,
        temperature_trend_deviation=0)
    return out, DailySleep(sleep.score, sleep.contributors, total, temp, hr, hr_time, rmssd)


def final_sleep_score(prefix_scores, periods):
    """Tail of `daily_outputs_stateless`: the running maximum of the prefix sleep
    scores, lowered to a later score when that prefix ends in a main period (is_main == 1)."""
    cur = 0
    for s, p in zip(prefix_scores, periods):
        nxt = max(s, cur)
        if cur > s and p.is_main == 1:
            nxt = s
        cur = nxt
    return cur


@dataclass
class DailyOutputs:
    sleep: DailySleep
    readiness: R.ReadinessOutput
    sleep_scores: List[int]        # per prefix
    readiness_scores: List[int]
    sleep_deltas: List[int]        # score minus the previous prefix's (0 for the first)
    readiness_deltas: List[int]
    v2_inputs: List[tuple]         # per prefix: total, deep, REM, HRV balance, resting HR, recovery index, body temp
    time: tuple = (0, 0)           # (sleep day UTC, second time word), copied to the first 16 bytes of both outputs


def check_inputs(prev_day, user, periods):
    """The input validation of `daily_outputs_stateless` (returns 0 when the input is
    accepted, 13 when it is rejected)."""
    if prev_day is None or user is None or not periods:
        return 13
    if any(p.total_sleep & M32 == 0 for p in periods) or not any(p.is_main == 1 for p in periods):
        return 13
    if prev_day[1] > 5_000_000 or prev_day[2] > 1500 or prev_day[3] > 1500 or prev_day[4] > 1500:
        return 13
    return 0


def daily_outputs_stateless(sleep_day_utc, tz_word, day_number, periods, prev_day, user, history, baseline,
                            chrono, rest_input, cycle_day_type, feats=None, sri=0, limits=S.limits_v1):
    """Daily outputs from the day's periods. `chrono` is (type, ideal midpoint in seconds) or
    None; `tz_word` is the second time word. Returns DailyOutputs, or None when the input
    is rejected (see `check_inputs`)."""
    if check_inputs(prev_day, user, periods):
        return None
    t0 = sleep_day_utc - 86400 if sleep_day_utc > 86400 else sleep_day_utc
    ctype, cmid = (chrono[0] & 0xFF, _tdiv(_s32(chrono[1]), 60)) if chrono is not None else (0xFF, 0)
    res = DailyOutputs(None, None, [], [], [], [], [])
    sl = ro = None
    prev_s = prev_r = 0
    for k in range(1, len(periods) + 1):
        sl = get_daily_sleep(user[1], ctype, cmid, periods[:k], limits)
        ro, sl = daily_readiness_stateless(sl, prev_day, periods[:k], user, history, baseline, day_number,
                                           t0, cycle_day_type, rest_input, feats, sri)
        res.sleep_scores.append(sl.score)
        res.readiness_scores.append(ro.score)
        res.sleep_deltas.append((sl.score - prev_s) & M32 if k > 1 else 0)
        res.readiness_deltas.append((ro.score - prev_r) & M32 if k > 1 else 0)
        res.v2_inputs.append((sl.contributors[0], sl.contributors[1], sl.contributors[2], ro.hrv_balance,
                              ro.resting_hr, ro.resting_hr_time, ro.temperature))
        prev_s, prev_r = sl.score, ro.score
    sl.score = final_sleep_score(res.sleep_scores, periods)
    res.sleep, res.readiness = sl, ro
    res.time = (sleep_day_utc, tz_word)
    return res


def v2_input(prefix, sessions, sfi_history, br_history, age):
    """Assembly of the readiness v2 input from the prefix scores: the
    seven prefix scores of `DailyOutputs.v2_inputs` (total sleep, deep, REM, HRV balance, resting
    HR, recovery index, body temperature) become the v2 scores, with 0 or less mapped to -1
    (missing). It is called with the first i+1 sessions of the day and the prefix i, and
    separately with only session i and the contributors stored on that sleep. The sessions'
    hypnograms, the 90-day SFI and breathing-rate histories and the age are supplied by the caller."""
    scores = [v if v > 0 else R.MISSING for v in prefix]
    return R.ReadinessV2Input(*scores, sessions=list(sessions), sfi_history=sfi_history, br_history=br_history,
                              age=age)
