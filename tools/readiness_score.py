#!/usr/bin/env python3
"""Oura readiness scores.

Two independent calculations are implemented here:

* Readiness v2 (the Readiness 2
  screens, active from a start day set per user). Inputs are seven 0..100 scores from
  the sleep and recovery pipeline, the night's 30-second hypnograms, 90-day histories of sleep
  fragmentation and breathing rate, and the age. `readiness_v2()`.
* Readiness (`readiness_calculate`, which runs `recovery_run`; the daily readiness score). Nine contributors against personal baselines, combined with a
  weighted mean and capped by temperature and rest-mode rules. `readiness_calculate()`; the
  contributor logic is `recovery_core()`, the baseline update `baseline_update()`. Rest mode
  itself is in `tools/rest_recovery.py`.

Readiness v2 combines groups of contributors with a weighted power mean (exponent 0.6):

    M(v, w) = (sum w_i * v_i^0.6 / sum w_i)^(1/0.6)        v_i in 1..100, -1 means missing

    sleep recovery   = M(total sleep, deep sleep, REM sleep, WASO, SFI ; .30 .20 .20 .15 .15)
    nervous system   = M(HRV balance, resting HR, recovery index ; .35 .35 .30)
    body strain      = M(body temperature, breathing rate, BDI ; .40 .40 .20)
    readiness        = M(sleep recovery, nervous system, body strain ; .40 .30 .30)

Arithmetic mixes float32 and float64; `f32()` rounds to single precision
after every single-precision operation and `fma32()` is a fused multiply-add. The libm calls
(powf, expf, log2f, exp2) use Python's correctly rounded results, so another libm that is off
by one ulp could differ on a rounding boundary.
"""
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NAN = float("nan")
INF = float("inf")


# --------------------------------------------------------------------------- float32 helpers
def f32(x):
    """Round a Python float to the nearest float32 (returned as a Python float)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(INF, x)


def fma32(a, b, c):
    """Single-precision fused multiply-add a*b + c."""
    try:
        return f32(math.fma(a, b, c))
    except (ValueError, OverflowError):
        return NAN


def isfin(x):
    return not (math.isnan(x) or math.isinf(x))


def round_half_away_i32(x):
    """Round to nearest, ties away from zero (NaN gives 0)."""
    if math.isnan(x):
        return 0
    if math.isinf(x):
        return 2**31 - 1 if x > 0 else -2**31
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def powf(a, b):
    try:
        return f32(math.pow(a, b))
    except (ValueError, OverflowError):
        return NAN


def expf(a):
    try:
        return f32(math.exp(a))
    except OverflowError:
        return INF


C06 = struct.unpack("<f", struct.pack("<I", 0x3F19999A))[0]    # 0.6 as float32
C1_06 = struct.unpack("<f", struct.pack("<I", 0x3FD55555))[0]  # 1/0.6 as float32
MISSING = 0xFFFFFFFF

W_SLEEP = (0.3, 0.2, 0.2, 0.15, 0.15)
W_NERVOUS = (0.35, 0.35, 0.3)
W_STRAIN = (0.4, 0.4, 0.2)
W_TOP = (0.4, 0.3, 0.3)


def _w32(ws):
    return [f32(w) for w in ws]


def weighted_power_mean(vals, weights):
    """Weighted power mean: values equal to -1 (0xffffffff) are skipped, 0 counts
    as 1, values of 100 and above as 100. Returns -1 when nothing is left."""
    sw, acc = 0.0, 0.0
    for v, w in zip(vals, _w32(weights)):
        v &= 0xFFFFFFFF
        if v == MISSING:
            continue
        base = 1.0 if v == 0 else (float(v) if v <= 99 else 100.0)
        sw = f32(sw + w)
        acc = fma32(w, powf(base, C06), acc)
    if not sw > 0:
        return -1
    x = powf(f32(acc / sw), C1_06)
    x = 0.0 if x < 0 else x
    return round_half_away_i32(min(x, 100.0))


# --------------------------------------------------------------------------- sub-scores
def waso_score(waso_min, age):
    """Wake after sleep onset (minutes) to 1..100. The allowed WASO is 60 min at age 20 and
    below, rising linearly to 120 min at age 80 and above; the score is a logistic
    100 / (1 + exp(0.055 * (waso - allowed)))."""
    if not isfin(waso_min):
        return -1
    a = min(max(age, 20), 80)
    t = f32(float(a - 20) / 60.0)
    thr = 120.0 if t == 1.0 else min(fma32(t, 60.0, 60.0), 120.0)
    z = f32(f32(waso_min - thr) * f32(0.055))
    s = f32(100.0 / f32(expf(z) + 1.0))
    s = 1.0 if s < 1.0 else s
    return round_half_away_i32(min(s, 100.0))


def sfi_score(sfi, p25, p75):
    """Sleep fragmentation index against the user's 90-day 25th/75th percentiles: 90 at p25,
    60 at p75 (slope -30 / (p75 - p25)), clamped to 1..100."""
    if not p75 > p25:
        return -1
    d = f32(p75 - p25)
    slope = f32(-30.0 / d)
    s = fma32(slope, f32(sfi - p25), 90.0)
    s = 1.0 if s < 1.0 else s
    return round_half_away_i32(min(s, 100.0))


def br_score(x, mean, std, min_std=0.5):
    """Breathing rate score from z = (x - mean) / max(std, min_std). 100 for z in [-1, -0.5],
    100*(1 - (z + 0.5)/3.5) above that (0 at z = 3), 50 + 50*(z + 4)/3 below -1 (50 at z = -4)."""
    std = min_std if std < min_std else std
    if not (isfin(std) and std > 0):
        return -1
    z = f32(f32(x - mean) / std)
    if z <= -1.0:
        t = f32(f32(z + 4.0) / 3.0)
        t = 0.0 if t < 0 else t
        t = min(t, 1.0)
        s = 100.0 if t == 1.0 else min(fma32(t, 50.0, 50.0), 100.0)
    elif z <= -0.5:
        s = 100.0
    else:
        t = f32(f32(z + 0.5) / 3.5)
        t = 0.0 if t < 0 else t
        t = min(t, 1.0)
        s = fma32(t, 0.0, f32(f32(1.0 - t) * 100.0))
    s = 1.0 if s < 1.0 else s
    return round_half_away_i32(min(s, 100.0))


def bdi_score(bdi, age):
    """Breathing disturbance index to 1..100: full score up to an allowed BDI of
    1 + 0.15 * (min(age, 80) - 20) (age floor 20), then 100 - 4 per index point above it;
    capped at 30 when the index exceeds 30."""
    if not isfin(bdi):
        return -1
    a = max(float(age), 20.0)
    y = f32(a - 20.0) if a < 80.0 else 60.0
    thr = fma32(y, f32(0.15), 1.0)
    d = f32(bdi - thr)
    lin = fma32(d, -4.0, 100.0)
    floored = 1.0 if lin < 1.0 else lin
    gated = floored if d > 0 else 100.0
    capped = min(gated, 30.0) if bdi > 30.0 else gated
    return round_half_away_i32(capped)


def restorative_sleep_score(deep, rem):
    """Mean of the deep and REM scores (each clamped to 1..100), rounded down; the available
    one when the other is missing (-1)."""
    deep &= 0xFFFFFFFF
    rem &= 0xFFFFFFFF
    deep_c = min(max(float(deep), 1.0), 100.0)
    rem_c = min(max(float(rem), 1.0), 100.0)
    if deep != MISSING and rem != MISSING:
        return int(math.floor(f32(f32(deep_c + rem_c) * 0.5)))
    r = int(rem_c) if rem != MISSING else MISSING
    return r if deep == MISSING else int(deep_c)


# --------------------------------------------------------------------------- hypnograms
# Stage codes in the 30-second hypnogram: 1 deep, 2 light, 3 REM, 4 wake (0 unknown).
def hypnogram_tst_sec(ph):
    return 30 * sum(1 for s in ph if 1 <= s <= 3)


def hypnogram_waso_sec(ph):
    """Seconds awake (stage 4) from the first sleep epoch to the end of the record; -1 when
    the record has no sleep epoch."""
    for i, s in enumerate(ph):
        if 1 <= s <= 3:
            if i == 0xFFFF:
                return -1
            return 30 * sum(1 for t in ph[i:] if t == 4)
    return -1


def hypnogram_transitions(ph):
    """Number of fragmenting transitions: sleep to wake, deep to light, REM to light.
    None when the record has no sleep epoch."""
    if not any(1 <= s <= 3 for s in ph):
        return None
    n = 0
    for a, b in zip(ph, ph[1:]):
        if (1 <= a <= 3 and b == 4) or (a == 1 and b == 2) or (a == 3 and b == 2):
            n += 1
    return n


@dataclass
class Session:
    """One readiness v2 session: type (0 long sleep, 1 sleep,
    2 late nap), hypnogram, length, breathing rate and breathing disturbance index."""
    type: int = 0
    phases: bytes = b""
    breathing_rate: float = NAN
    bdi: float = NAN


@dataclass
class ReadinessV2Input:
    """The readiness v2 input: seven u32 scores (total, deep, REM, HRV balance,
    resting HR, recovery index, body temperature; 0xffffffff = missing), the sessions, the SFI history,
    the breathing rate history and the age."""
    total_sleep: int = MISSING
    deep_sleep: int = MISSING
    rem_sleep: int = MISSING
    hrv_balance: int = MISSING
    resting_hr: int = MISSING
    recovery_index: int = MISSING
    body_temp: int = MISSING
    sessions: List[Session] = field(default_factory=list)
    sfi_history: Optional[List[float]] = None
    br_history: Optional[List[float]] = None
    age: int = 0


@dataclass
class ReadinessV2Output:
    score: int = 0
    sleep_recovery: int = 0
    nervous_system: int = 0
    body_strain: int = 0
    waso_score: int = 0
    sfi_score: int = 0
    br_score: int = 0
    bdi_score: int = 0
    restorative_sleep: int = 0
    sfi: float = NAN
    waso_min: float = NAN
    br_avg: float = NAN
    bdi: float = NAN
    sfi_p25: float = NAN
    sfi_p75: float = NAN
    br_mean: float = NAN
    br_std: float = NAN


def aggregate_sessions(sessions):
    """Night summary over the sessions. Returns (SFI per hour of
    sleep, WASO minutes, mean breathing rate, TST-weighted BDI); NaN where undefined.
    WASO averages the type-0 (long sleep) sessions only."""
    trans_sum = tst_sum = waso_n = br_n = 0
    waso_acc = br_acc = w_acc = wsum = 0.0
    for s in sessions[:255]:
        ph = s.phases
        tst = waso = 0
        tr = 0
        waso_ok = False
        if ph:
            tst = hypnogram_tst_sec(ph)
            waso = hypnogram_waso_sec(ph)
            waso_ok = waso != -1
            t = hypnogram_transitions(ph)
            if t is not None:
                tst_sum += tst
                tr = 0 if tst == 0 else t
            else:
                tst = 0
        else:
            waso = -1
            waso_ok = False
        trans_sum += tr
        if s.type == 0 and waso_ok:
            waso_n += 1
            waso_acc += float(waso)
        if isfin(s.breathing_rate):
            br_n += 1
            br_acc += s.breathing_rate
        if isfin(s.bdi) and tst != 0:
            w_acc = math.fma(s.bdi, float(tst), w_acc)
            wsum += float(tst)
    sfi = NAN if tst_sum == 0 else f32(float(trans_sum) / f32(float(tst_sum) / 3600.0))
    waso_min = NAN if waso_n == 0 else f32(waso_acc / waso_n / 60.0)
    br_avg = NAN if br_n == 0 else f32(br_acc / br_n)
    bdi = f32(w_acc / wsum) if wsum > 0 else NAN
    return sfi, waso_min, br_avg, bdi


def sfi_percentiles(hist):
    """25th and 75th percentile (linear interpolation, float32) of the finite values among the
    first 90 entries; None with fewer than 14 finite values."""
    v = [f32(x) for x in hist[:90] if isfin(x)]
    if len(v) < 14:
        return None
    v.sort()
    n = len(v)
    res = []
    for q in (0.25, 0.75):
        pos = f32(float(n - 1) * q)
        i = int(math.floor(pos))
        frac = f32(pos - float(i))
        lo, hi = v[i], v[min(i + 1, n - 1)]
        res.append(fma32(f32(1.0 - frac), lo, f32(frac * hi)))
    return res


def br_baseline(hist):
    """Mean (float32) and sample standard deviation (float32) of the finite breathing-rate
    history entries among the first 90; None with fewer than 14."""
    n, s, ss = 0, 0.0, 0.0
    for x in hist[:90]:
        if isfin(x):
            n += 1
            s += x
            ss = math.fma(x, x, ss)
    if n < 14:
        return None
    mean = s / n
    d2 = -(mean * n)
    var = math.fma(d2, mean, ss) / (n - 1)
    var = 0.0 if var < 0 else var
    return f32(mean), f32(math.sqrt(var))


def readiness_v2(inp: ReadinessV2Input) -> ReadinessV2Output:
    sfi, waso_min, br_avg, bdi = aggregate_sessions(inp.sessions)
    o = ReadinessV2Output(sfi=sfi, waso_min=waso_min, br_avg=br_avg, bdi=bdi)
    waso_s = waso_score(waso_min, inp.age) if isfin(waso_min) else -1

    sfi_s = -1
    if inp.sfi_history:
        pc = sfi_percentiles(inp.sfi_history)
        if pc is not None and isfin(pc[0]):
            if isfin(pc[1]):
                o.sfi_p25, o.sfi_p75 = pc
                if isfin(sfi):
                    sfi_s = sfi_score(sfi, pc[0], pc[1])

    br_s = -1
    if inp.br_history:
        bl = br_baseline(inp.br_history)
        if bl is not None:
            mean, std = bl
            if isfin(mean):
                o.br_mean = mean
                if isfin(std):
                    o.br_std = std
                    if isfin(br_avg):
                        br_s = br_score(br_avg, mean, std, 0.5)

    bdi_s = bdi_score(bdi, inp.age) if isfin(bdi) else -1
    sleep = weighted_power_mean([inp.total_sleep, inp.deep_sleep, inp.rem_sleep, waso_s, sfi_s], W_SLEEP)
    resto = restorative_sleep_score(inp.deep_sleep, inp.rem_sleep)
    nerv = weighted_power_mean([inp.hrv_balance, inp.resting_hr, inp.recovery_index], W_NERVOUS)
    strain = weighted_power_mean([inp.body_temp, br_s, bdi_s], W_STRAIN)
    total = weighted_power_mean([sleep, nerv, strain], W_TOP)
    neg = lambda v: 0 if v == -1 or v == MISSING else v  # missing scores are stored as 0
    o.score, o.sleep_recovery, o.nervous_system, o.body_strain = (neg(total), neg(sleep), neg(nerv), neg(strain))
    o.waso_score, o.sfi_score, o.br_score, o.bdi_score = (neg(waso_s), neg(sfi_s), neg(br_s), neg(bdi_s))
    o.restorative_sleep = neg(resto)
    return o


# =========================================================================== legacy readiness
# `readiness_calculate` builds a recovery state from the baseline, the 14-day
# histories and the previous day, then `recovery_run` (the version >= 22 path,
# default) turns the state into nine contributors and the
# score. Contributors are bytes in 1..100 (0 = not available).

#   byte  contributor            output field            driver
#    1    last night             lastNightSleep          sleep score against its baseline
#    2    sleep balance          sleepBalance            14-day weighted sleep time
#    3    last day activity      lastDayActivity         previous day distance, vigorous and sedentary time
#    4    activity balance       activityBalance         14-day weighted equivalent walking distance
#    5    resting HR             restingHr               lowest HR of the night
#    6    temperature            temperature             highest skin temperature
#    7    resting HR time        restingHrTime           time from the lowest HR to the end of the sleep
#    8    HRV balance            hrvBalance              14-day weighted log2(RMSSD)
#    9    sleep regularity       sleepRegularity         `sri_calculate` (external state)
#    0    readiness score        score                   weighted mean of the above
M16 = 0x7EB6                      # largest stored baseline value (decoded 16-bit field)
MEAN_MAX = (760, 4320, 720, 3360, 30000, 320, 4320, 29920, 6000)   # per baseline item, 8x scale
MEAN_MIN_DEFAULT = (680, 3250, 320, 2400, 4000, 80, 2400, 26400, 3040)   # set at run time
DEV_MIN = (40, 200, 24, 400, 2000, 16, 720, 240, 240)
DEV_MAX = (80, 360, 40, 560, 10000, 240, 960, 400, 480)
# 14-day recency weights (oldest first) for sleep time, equivalent distance and HRV
W14 = [f32(x) for x in (0.02, 0.05, 0.08, 0.11, 0.15, 0.18, 0.22, 0.30, 0.38, 0.48, 0.55, 0.63, 0.71, 0.72)]
WEIGHTS_A = (0.15, 0.25, 0.10, 0.10, 0.15, 0.10, 0.0, 0.15, 0.0)
WEIGHTS_B = (0.12, 0.20, 0.07, 0.07, 0.14, 0.04, 0.22, 0.14, 0.0)
WEIGHTS_B_REST = (0.15, 0.10, 0.025, 0.025, 0.25, 0.10, 0.25, 0.10, 0.0)
WEIGHTS_A_SRI = (0.15, 0.15, 0.10, 0.10, 0.15, 0.10, 0.0, 0.15, 0.10)
WEIGHTS_B_SRI = (0.10, 0.08, 0.025, 0.025, 0.25, 0.10, 0.25, 0.10, 0.07)
# order in which the combiner walks the summary bytes against the nine weights
_COMBINE_ORDER = (1, 2, 3, 4, 5, 7, 6, 8, 9)
# default baseline written into 0x40..0x64 (means, then deviations), little-endian u16
_INIT_MEANS = (720, 0, 480, 2880, 8000, 160, 3840, 29360, 4400)
_INIT_DEVS = (80, 360, 160, 480, 8000, 64, 1600, 400, 2000)


@dataclass
class Features:
    """Switches that change the readiness arithmetic. By default the HRV, rest mode, sleep regularity and
    athlete-adjusted contributors are on and `freeze_baseline` is off."""
    hrv_in_readiness: bool = True
    rest_mode: bool = True
    sleep_regularity: bool = True
    freeze_baseline: bool = False   # when on, no baseline update is run
    athletes: bool = True


def div32(a, b):
    if b == 0:
        if a == 0 or math.isnan(a):
            return NAN
        return math.copysign(INF, a) * math.copysign(1.0, b)
    return f32(a / b)


def round_half_away(x):
    if not isfin(x):
        return x
    return math.copysign(math.floor(abs(x) + 0.5), x)


def sx16(v):
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


def sx13(v):
    v &= 0x1FFF
    return v - 0x2000 if v & 0x1000 else v


def tdiv(a, b):
    """Integer division truncating toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _sb(v):
    return v - 256 if v > 127 else v


def decode16(v):
    """Clamp a stored 16-bit baseline value to 0x7eb6."""
    v &= 0xFFFF
    return M16 if v >= 0x8000 else min(v, M16)


def div100_round(q):
    """(q + 50) / 100 for q >= 1, else -((50 - q) / 100): rounds q/100 half away from zero."""
    return (q + 50) // 100 if q >= 1 else -((50 - q) // 100)


class State:
    """The recovery state `recovery_run` reads: 14 days of sleep minutes (the last slot is the night being
    scored), the previous-day and 14-day equivalent distance, sedentary and vigorous minutes, the baseline
    means and deviations (8x scaled, in the order sleep score, sleep time, resting HR, resting HR time,
    distance, vigorous, sedentary, temperature, HRV), baseline age, wear percent, the 14-day RMSSD history,
    the HRV baseline age and the time."""
    SIZE = 0x8C

    def __init__(self, b=None):
        self.b = bytearray(b) if b is not None else bytearray(self.SIZE)

    def u8(self, o):
        return self.b[o]

    def u16(self, o):
        return self.b[o] | self.b[o + 1] << 8

    def i16_at(self, o):
        return sx16(self.u16(o))

    def u32(self, o):
        return int.from_bytes(self.b[o:o + 4], "little")

    def set16(self, o, v):
        self.b[o:o + 2] = (v & 0xFFFF).to_bytes(2, "little")

    def set32(self, o, v):
        self.b[o:o + 4] = (v & 0xFFFFFFFF).to_bytes(4, "little")

    def shift_days(self):
        """Move the three 14-day histories one day back and clear today's slot."""
        for lo, hi in ((0x04, 0x1E), (0x22, 0x3C), (0x68, 0x82)):
            self.b[lo:hi] = self.b[lo + 2:hi + 2]
            self.set16(hi, 0)

    def init_defaults(self):
        """Restore the default baseline and clear the histories."""
        b = self.b
        b[0:4] = bytes(4)
        b[0x04:0x20] = bytes(0x1C)
        b[0x20:0x22] = bytes(2)
        b[0x22:0x3E] = bytes(0x1C)
        b[0x3E:0x40] = bytes(2)
        for i, v in enumerate(_INIT_MEANS):
            self.set16(0x40 + 2 * i, v)
        for i, v in enumerate(_INIT_DEVS):
            self.set16(0x52 + 2 * i, v)
        b[0x64:0x66] = bytes(2)
        b[0x68:0x84] = bytes(0x1C)
        b[0x84] = 0
        b[0x88:0x8C] = bytes(4)


def baseline_update(S, item, val, n, feats=None, mean_min=None):
    """Move the long-term mean and
    deviation of baseline `item` (0..8) toward the 8x scaled value `val` after `n` days.

    The new value is first clipped toward the mean (beyond 2.5 deviations to one deviation,
    between 1.5 and 2.5 deviations to 1.5 deviations; not for the athlete items 4 and 5). The
    mean then moves by (x - mean)/8 for n < 15 (1/2 for n < 4) or /32 for n >= 15; the deviation
    moves by (|x - mean'| - dev)/16, /8 or /64 on the same schedule. Results are clamped to
    per-item bounds."""
    feats = feats or Features()
    mean_min = mean_min or MEAN_MIN_DEFAULT
    mo, do = 0x40 + 2 * item, 0x52 + 2 * item
    m, d = decode16(S.u16(mo)), decode16(S.u16(do))
    x = sx16(val) << 3
    athl = feats.athletes and item in (4, 5)
    if athl:
        x = 0 if sx16(val) < 0 else min(x, M16)
    elif n >= 4:
        diff = x - m
        s = 1 if d != 0 else -1
        t1 = 5 * d + s
        th1 = (t1 + (1 if t1 < 0 else 0)) >> 1
        ad = abs(diff)
        clip = None
        if ad > th1:
            clip = d
        else:
            t2 = 3 * d + s
            th2 = (t2 + (1 if t2 < 0 else 0)) >> 1
            if ad > th2:
                clip = th2
        if clip is not None:
            x = m + (-clip if x <= m else clip)
    diff = x - m
    if n >= 15:
        dm = tdiv(diff + (16 if diff > 0 else -16), 32)
        nm = m + sx16(dm)
        e = abs(x - nm) - d
        dd = tdiv(e + (32 if e > 0 else -32), 64)
    elif n >= 4:
        dm = tdiv(diff + (4 if diff > 0 else -4), 8)
        nm = m + sx16(dm)
        e = abs(x - nm) - d
        dd = tdiv(e + (8 if e > 0 else -8), 16)
    else:
        dm = tdiv(diff + (-1 if diff <= 0 else 1), 2)
        nm = m + sx16(dm)
        e = abs(x - nm) - d
        dd = tdiv(e + (4 if e > 0 else -4), 8)
    nd = d + sx16(dd)
    dev_ub = M16 if athl else DEV_MAX[item]
    t = dev_ub if dev_ub < nd else nd
    if nd < DEV_MIN[item]:
        t = DEV_MIN[item]
    if n < 4:
        t = nd
    mean_ub = M16 if athl else MEAN_MAX[item]
    mean_lb = 0 if athl else mean_min[item]
    mm = mean_ub if mean_ub < nm else nm
    if nm < mean_lb:
        mm = mean_lb
    S.set16(mo, 0 if mm < 0 else min(mm, M16))
    S.set16(do, 0 if t < 0 else min(t, M16))


def combine_contributors(summary, weights):
    """summary[0] = round(sum(w_i * c_i) / sum(w_i)) over the contributors c_i >= 1 (float32)."""
    acc = wsum = 0.0
    for idx, w in zip(_COMBINE_ORDER, _w32(weights)):
        v = _sb(summary[idx])
        if v >= 1:
            acc = fma32(float(v), w, acc)
            wsum = f32(w + wsum)
    res = round_half_away_i32(div32(acc, wsum)) if wsum > 0 else 0
    summary[0] = min(res, 100) & 0xFF


def _clip(x, lo=None):
    """Round half away from zero, optional lower clamp, cap at 100, truncate to integer."""
    r = round_half_away(x)
    if lo is not None and r < lo:
        r = float(lo)
    return int(min(r, 100.0))


def temperature_contributor(S, t, n, upd, feats, mean_min=None):
    """Temperature contributor from the highest temperature `t` (hundredths of a degree).
    z = (t/100 - mean) / deviation: 100 for z in [-0.5, 0.15], 105 - 25 z above, 105 + 10 z below."""
    dev = f32(f32(float(S.i16_at(0x60)) / 8.0) / 100.0)
    if dev == 0:
        return 0
    tc = f32(float(t & 0xFFFF) / 100.0)
    if tc == 0 or not tc < 43.0:
        return 0
    mean = f32(f32(float(S.i16_at(0x4E)) * -0.125) / 100.0)
    dd = f32(tc + mean)
    z = div32(dd, dev)
    if z > 0.15:
        s = fma32(z, -25.0, 105.0)
    elif z >= -0.5:
        s = 100.0
    else:
        s = fma32(z, 10.0, 105.0)
    if abs(dd) < 2.0 and upd:
        baseline_update(S, 7, t, n, feats, mean_min)
    return _clip(s, 1)


@dataclass
class RestMode:
    """The rest-mode block `recovery_run` reads (the rest mode state): state byte (0 off, 2/3 resting, 4 recovering), days
    total and days remaining."""
    state: int = 0
    total: int = 0
    remaining: int = 0


@dataclass
class SleepSummary:
    """the sleep summary record: five u16 durations in 30-second units (awake, three sleep durations
    that sum to total sleep, and `extra`) and byte fields."""
    awake: int = 0
    rem: int = 0
    light: int = 0
    deep: int = 0
    extra: int = 0
    b0a: int = 0
    sleep_score: int = 0
    b0c: int = 0
    b0d: int = 0

    def u16(self, o):
        return (self.awake, self.rem, self.light, self.deep, self.extra)[o // 2]


def recovery_core(summary, S, ss, rm, h, t, flag, n2, upd, lowest, f, feats, sri, mean_min=None):
    """The body of `recovery_run` for version >= 22. `summary` is a 10-entry list that is
    filled in; `S` is the State, modified in place. `ss` is a SleepSummary or None, `rm` a
    RestMode, `h` whole hours slept (used when `ss` is None), `t` the highest temperature,
    `flag` set when the day rolled over, `n2` the baseline age used by the temperature update,
    `upd` allows baseline updates, `lowest` is (HR, seconds from bedtime) or None, `f` is
    log2(RMSSD) and `sri` the value of `sri_calculate(true)`."""
    if flag and S.u8(0x64) != 0xFF:
        S.b[0x64] += 1
    rest_flag = rm_state4 = 0
    if feats.rest_mode:
        st = rm.state
        rest_flag = 1 if (st & 0xFE) == 2 else 0
        rm_state4 = 1 if st == 4 else 0
        if rest_flag:
            upd = 0
    # a stored baseline outside its valid range is replaced by the defaults
    ok = (((S.u16(0x40) - 0x328) & 0xFFFF) > 0xFCD0 and ((S.u16(0x42) - 0x2D08) & 0xFFFF) > 0xD2F0
          and ((S.u16(0x44) - 0x378) & 0xFFFF) > 0xFD87 and ((S.u16(0x46) - 0x2D08) & 0xFFFF) > 0xD2F0
          and ((S.u16(0x4A) - 0x2D08) & 0xFFFF) > 0xD2F0 and ((S.u16(0x4C) - 0x2D08) & 0xFFFF) > 0xD2F0
          and S.i16_at(0x4E) > 0x63FF)
    if ok:
        hm = S.i16_at(0x50)
        hm = sx13((hm + (7 if hm < 0 else 0)) >> 3)
        ok = int(2.0 ** f32(hm / 100.0)) <= 0xFF
    if ok:
        ok = (all(S.u16(o) <= 0x5A0 for o in range(0x04, 0x20, 2))
              and all(S.i16_at(o) >= 0 for o in range(0x22, 0x3E, 2))
              and all(S.i16_at(o) >= 0 for o in range(0x68, 0x84, 2)))
    if not ok:
        S.init_defaults()
    nb = S.u8(0x64)
    upd = bool(upd) and not feats.freeze_baseline
    mm = mean_min

    # 1: last night's sleep score
    last_score = float(ss.sleep_score) if ss else float(min(10 * (h & 0xFF), 85))
    dev8 = f32(float(S.i16_at(0x52)) / 8.0)
    summary[1] = 0
    if dev8 != 0:
        delta = f32(last_score + f32(float(S.i16_at(0x40)) * -0.125))
        contrib = fma32(div32(delta, dev8), 12.5, 85.0)
        contrib = 1.0 if contrib < 1.0 else contrib
        if upd:
            baseline_update(S, 0, round_half_away_i32(last_score), nb, feats, mm)
        summary[1] = _clip(contrib)

    # 2: sleep balance, 14-day recency-weighted sleep time
    if ss:
        today_min = f32(float(ss.rem + ss.light + ss.deep) / 2.0)
        today = round_half_away_i32(today_min)
    else:
        today = round_half_away_i32(float(min((h & 0xFF) * 55, 480)))
        today_min = float(today)
    if flag:
        S.b[0x04:0x1E] = S.b[0x06:0x20]
    S.set16(0x1E, today)
    dev8 = f32(float(S.i16_at(0x54)) / 8.0)
    summary[2] = 0
    if dev8 != 0:
        mean8 = f32(float(S.i16_at(0x42)) / 8.0)
        acc = wsum = 0.0
        vals = [S.i16_at(o) for o in range(0x04, 0x1E, 2)] + [sx16(today)]
        for v, w in zip(vals, W14):
            w = 0.0 if v < 1 else w
            wsum = f32(wsum + w)
            acc = fma32(w, float(v), acc)
        avg = div32(acc, wsum) if wsum != 0 else mean8
        z = div32(f32(avg - mean8), dev8)
        cnt = sum(1 for o in (0x16, 0x18, 0x1A, 0x1C) if S.i16_at(o) > 0) + (1 if sx16(today) > 0 else 0)
        raw = fma32(z, 15.0, 85.0)
        raw = 1.0 if raw < 1.0 else raw
        gated = 0.0 if cnt < 3 else raw
        if upd and ss:
            baseline_update(S, 1, round_half_away_i32(today_min), nb, feats, mm)
        summary[2] = _clip(gated)

    # 3: previous-day activity against the distance, vigorous and sedentary baselines
    summary[3] = 0
    if S.u16(0x20) != 0:
        wear = f32(float(S.u8(0x66)) / 100.0)
        if not wear < 0.33:
            d_eq = f32(float(decode16(S.u16(0x5A))) / 8.0)
            d_vig = f32(float(S.i16_at(0x5C)) / 8.0)
            d_sed = f32(float(S.i16_at(0x5E)) / 8.0)
            if d_eq != 0 and d_vig != 0 and d_sed != 0:
                eq_m8 = f32(float(decode16(S.u16(0x48))) / 8.0)
                sed_m8 = f32(float(S.i16_at(0x4C)) / 8.0)
                vig_m8 = f32(float(S.i16_at(0x4A)) / 8.0)
                one_w = f32(1.0 - wear)
                if feats.athletes:
                    a0 = fma32(eq_m8, one_w, float(S.u16(0x20)))
                    a1 = float(S.u8(0x65))
                    a2 = fma32(sed_m8, one_w, float(S.u16(0x3E)))
                else:
                    a0 = fma32(float(S.u16(0x20)), wear, f32(one_w * eq_m8))
                    a1 = fma32(float(S.u8(0x65)), wear, f32(one_w * vig_m8))
                    a2 = fma32(float(S.u16(0x3E)), wear, f32(one_w * sed_m8))
                r0 = div32(f32(a0 - eq_m8), d_eq)
                r1 = div32(f32(a1 - vig_m8), d_vig)
                r2 = div32(f32(a2 - sed_m8), d_sed)
                zsum = f32(f32(r0 + r1) + r2)
                contrib = fma32(zsum, -6.0, 85.0)
                contrib = contrib if contrib > 0 else 1.0
                if upd:
                    baseline_update(S, 5, S.u8(0x65), nb, feats, mm)
                    sed_min = S.u16(0x3E)
                    if feats.athletes and sed_min != 0:
                        sm = S.i16_at(0x4C)
                        tq = (100 - S.u8(0x66)) * sm + (4 if sm > 0 else -4)
                        q = tdiv(tq, 8)
                        sed_min += (q + 50) // 100 if tq >= 8 else -((50 - q) // 100)
                    baseline_update(S, 6, sed_min, nb, feats, mm)
                summary[3] = _clip(contrib, 1)

    # 4: activity balance, 14-day recency-weighted equivalent distance
    mean_eq = decode16(S.u16(0x48))
    mean_eq_f = f32(float(mean_eq) / 1000.0)
    summary[4] = 0
    if mean_eq_f != 0:
        dist_today = 0
        if S.u16(0x20) != 0:
            if feats.athletes:
                tq = (100 - S.u8(0x66)) * mean_eq + (4 if mean_eq > 0 else -4)
                q = tdiv(tq, 8)
                dist_today = S.u16(0x20) + ((q + 50) // 100 if tq >= 8 else -((50 - q) // 100))
            else:
                dv = S.i16_at(0x5A)
                tq = (100 - S.u8(0x66)) * dv + (4 if dv > 0 else -4)
                dist_today = div100_round(S.u8(0x66) * S.u16(0x20) + tdiv(tq, 8))
        if flag:
            S.b[0x22:0x3C] = S.b[0x24:0x3E]
        S.set16(0x3C, dist_today)
        acc = wsum = 0.0
        for i in range(14):
            v = S.i16_at(0x22 + 2 * i)
            w = 0.0 if v < 1 else W14[i]
            sc = div32(f32(float(v) * 8.0), 1000.0)
            wsum = f32(wsum + w)
            acc = fma32(w, sc, acc)
        avg = div32(acc, wsum) if wsum != 0 else mean_eq_f
        d = div32(f32(avg - mean_eq_f), mean_eq_f)
        cnt = sum(1 for o in (0x34, 0x36, 0x38, 0x3A) if S.i16_at(o) > 0) + (1 if sx16(dist_today) > 0 else 0)
        bal = f32(d + 0.25)
        bal = fma32(abs(bal), -60.0, 100.0)
        bal = 1.0 if bal < 1.0 else bal
        gated = 0.0 if cnt < 3 else bal
        if upd:
            baseline_update(S, 4, dist_today, nb, feats, mm)
        summary[4] = _clip(gated)

    # 5: resting HR; 7: time of the lowest HR relative to the end of the sleep
    summary[5] = summary[7] = 0
    if lowest is not None:
        hr, hr_time = lowest
        dev8 = f32(float(S.i16_at(0x56)) / 8.0)
        if 1 <= hr <= 179 and dev8 != 0:
            delta = f32(float(hr) + f32(float(S.i16_at(0x44)) * -0.125))
            z = div32(delta, dev8)
            raw = fma32(z, -30.0, 87.5) if z > -1.0 else fma32(z, 16.0, 116.0)
            contrib = raw if raw > 0 else 1.0
            if upd:
                baseline_update(S, 2, hr, nb, feats, mm)
            summary[5] = _clip(contrib, 1)
        dev8 = f32(float(S.i16_at(0x58)) / 8.0)
        if ss and hr <= 179 and dev8 != 0:
            t30 = hr_time // 30
            tot = (ss.u16(6) + ss.u16(4)) + (ss.u16(2) + ss.u16(0))
            sleep_time = f32(float((tot - t30) & 0xFFFFFFFF) / 2.0)
            delta = f32(sleep_time + f32(float(S.i16_at(0x46)) * -0.125))
            time_score = fma32(div32(delta, dev8), 15.0, 85.0)
            if upd:
                baseline_update(S, 3, round_half_away_i32(sleep_time), nb, feats, mm)
            summary[7] = _clip(time_score, 1)

    # 6: temperature
    summary[6] = temperature_contributor(S, t, n2, upd, feats, mm)
    if S.u8(0x84) != 0xFF:
        S.b[0x84] += 1

    # 8: HRV balance, 14-day recency-weighted 100*log2(RMSSD)
    summary[8] = 0
    dev8 = f32(float(S.i16_at(0x62)) / 8.0)
    if dev8 != 0:
        mean8 = f32(float(S.i16_at(0x50)) / 8.0)
        log_h = f32(f * 100.0)
        today_h = round_half_away_i32(log_h)
        if flag:
            S.b[0x68:0x82] = S.b[0x6A:0x84]
        nh = S.u8(0x84)
        S.set16(0x82, today_h)
        acc = wsum = 0.0
        for i in range(14):
            v = S.i16_at(0x68 + 2 * i)
            w = 0.0 if v < 1 else W14[i]
            acc = fma32(w, float(v), acc)
            wsum = f32(wsum + w)
        avg = div32(acc, wsum) if wsum != 0 else mean8
        z = div32(f32(avg - mean8), dev8)
        cnt = sum(1 for o in (0x7A, 0x7C, 0x7E, 0x80) if S.i16_at(o) > 0) + (1 if sx16(today_h) > 0 else 0)
        raw = fma32(z, -16.0, 117.0) if z > 1.0 else fma32(z, 16.0, 85.0)
        if not z > 0:
            raw = fma32(z, 42.5, 85.0)
        floored = 1.0 if raw < 1.0 else raw
        contrib = floored if (feats.hrv_in_readiness and cnt >= 3 and nh >= 8) else 0.0
        if log_h > 0 and upd:
            baseline_update(S, 8, round_half_away_i32(log_h), nh, feats, mm)
        summary[8] = _clip(contrib)

    # 9: sleep regularity
    summary[9] = 0x80
    if feats.sleep_regularity:
        summary[9] = (sri & 0xFF) if (sri & 0x100) else 0

    # score
    wa, wb = WEIGHTS_A, WEIGHTS_B
    if feats.rest_mode:
        wb = WEIGHTS_B_REST
    if feats.sleep_regularity:
        wa, wb = WEIGHTS_A_SRI, WEIGHTS_B_SRI
    combine_contributors(summary, wa)
    if feats.rest_mode:
        temp_contrib = summary[6]
        flag2 = rest_flag if ((temp_contrib - 1) & 0xFFFFFFFF) >= 0x4A else 1
        if (flag2 | rm_state4) != 1:
            return
        if rest_flag:
            summary[3] = summary[4] = 0
        old = summary[0]
        combine_contributors(summary, wb)
        new = _sb(summary[0])
        if new >= 0x51:
            new = round_half_away_i32(f32(f32(float(new - 0x50) / 5.0) + 80.0))
        if new <= 0x13:
            new = round_half_away_i32(f32(f32(f32(20.0 - float(new)) * -0.5) + 20.0))
        oldsb = _sb(old)
        summary[0] = (min(new, oldsb) if flag2 else old) & 0xFF
        if not rm_state4:
            return
        cur_f, new_f = float(oldsb), float(new)
        if not new_f >= cur_f:
            p = min(div32(float(rm.total - rm.remaining + 1), float(rm.total + 1)), 1.0)
            cur_f = fma32(cur_f, p, f32(f32(1.0 - p) * new_f))
        summary[0] = round_half_away_i32(cur_f) & 0xFF
    else:
        if _sb(summary[6]) > 0x4A:
            return
        old = _sb(summary[0])
        combine_contributors(summary, wb)
        if not old >= _sb(summary[0]):
            summary[0] = old & 0xFF


def mean_min_table(j, prev_night_min=80):
    """Lower bounds for the baseline means, 8x scale. Entry 0 is set by
    feature 9 (previous-night contributor minimum, 80) and entry 1 at every `recovery_run`, from the
    sleep-time baseline `j` (minutes): trunc(8 * (486 - 1.77 j))."""
    t = list(MEAN_MIN_DEFAULT)
    t[0] = prev_night_min * 8
    t[1] = sx16(int(math.fma(float(j & 0xFFFFFFFF), struct.unpack("<d", bytes.fromhex("52b81e85eb51fcbf"))[0], 486.0) * 8))
    return t


def recovery_run(S, prev, ss, rm, t, h, f, j, j1, j2, upd, lowest, utc, feats=None, sri=0, mean_min=None):
    """`recovery_run` (version >= 22 path). `S` is the state to update and `prev` the previous
    state (or S itself). `utc` is the time in seconds. Returns (summary, returned flag)."""
    feats = feats or Features()
    mean_min = mean_min or mean_min_table(j)
    summary = [0] * 10
    s88 = S.u32(0x88)
    c1 = s88 != 0 and utc <= ((s88 + 43200) & 0xFFFFFFFF)
    new_day = (not c1) or (utc + 129600 < s88)
    flag = 1 if new_day else 0
    if not new_day:
        p88 = prev.u32(0x88)
        if p88 != 0 and utc > ((p88 + 3600) & 0xFFFFFFFF):
            S.b[0:0x90] = prev.b[0:0x90]
            s88 = S.u32(0x88)
            flag = 1
    if s88 != 0:
        k = 0
        while utc > ((s88 + 129600) & 0xFFFFFFFF) and k < 14:
            S.shift_days()
            s88 = (s88 + 86400) & 0xFFFFFFFF
            k += 1
    S.set32(0, j1)
    S.set32(0x88, utc)
    recovery_core(summary, S, ss, rm, h, t, flag, j2, upd, lowest, f, feats, sri, mean_min)
    if ss is None:   # the first nine bytes are cleared when there is no sleep summary
        summary = [0] * 9 + [summary[9]]
    return summary, int(new_day)


# --------------------------------------------------------------------------- sleep regularity
SRI_COLUMNS = 144   # 10-minute slots of a day


def sri_calculate(rows, weighted=True):
    """`sri_calculate(bool)`: sleep regularity index over the stored days. `rows` holds one list of
    144 slot states per stored day, oldest first (0 asleep, 1 awake, 2 missing). Each slot is
    compared across days with weights 1..n (recent days weigh more; all 1 when `weighted` is
    false); the mean of the weighted state minus 0.5, squared, summed over the slots and scaled
    by 100/36 gives the index. Every slot needs at least 5 non-missing days. Returns a u16: bit 8 set when valid, low byte the index (1..100), 0 when invalid."""
    if not rows:
        return 0
    total = 0.0
    for j in range(SRI_COLUMNS):
        num = den = valid = 0
        for i, row in enumerate(rows):
            v = row[j]
            w = i + 1 if weighted else 1
            if v != 2:
                num += w * v
                den += w
                valid += 1
        if valid & 0xFF < 5:
            return 0
        x = f32(div32(float(num & 0xFF), float(den & 0xFF)) - 0.5)
        total = fma32(x, x, total)
    r = round_half_away_i32(f32(f32(total / 36.0) * 100.0))
    low = r & 0xFF
    val = 1 if low == 0 else (100 if low > 100 else r)
    return (val & 0xFF) | 0x100


# --------------------------------------------------------------------------- temperature deviation
def temperature_deviation(temp, base, hot90, dev3):
    """(deviation, trend). `temp` is today's highest temperature
    (centidegrees), `base` the baseline temperature (centidegrees, 13-bit field), `hot90` the 90
    previous highest temperatures (newest first), `dev3` the three previous deviations (newest
    first). A deviation is temp minus the 30th percentile of the last 90 highest temperatures;
    the trend is the weighted mean (2, 4, 4) of the last three. -0x8000 marks missing."""
    NA = -0x8000
    dev_base = (temp - base) & 0xFFFFFFFF
    low = dev_base - (1 << 32) if dev_base & 0x80000000 else dev_base
    h0, h4 = dev3[1], dev3[0]
    new = NA
    if temp <= 0x1068 and ((base - 200) & 0xFFFFFFFF) <= (temp & 0xFFFFFFFF):
        vals = sorted([v & 0xFFFFFFFF for v in hot90[:89]] + [temp & 0xFFFFFFFF])
        nz = next((i for i, v in enumerate(vals) if v), 90)
        k = nz + (30 * (90 - nz)) // 100
        new = (temp - (vals[k] if k < 90 else 0))
    else:
        low = NA
    w = (2 if h0 != NA else 0) + (4 if h4 != NA else 0) + (4 if new != NA else 0)
    if w < 6:
        return low, NA
    num = (2 * h0 if h0 != NA else 0) + (4 * h4 if h4 != NA else 0) + (4 * new if new != NA else 0)
    return low, tdiv(num + (w >> 1), w)


# --------------------------------------------------------------------------- readiness_calculate
@dataclass
class ReadinessInput:
    """The readiness input record (the fields `readiness_calculate` reads).

    Fields: utc, day number (baseline age), sleep score, time in bed, total sleep, REM, deep, latency (s),
    wake-up count, highest temperature (centidegrees), lowest HR, lowest-HR time (s from bedtime), RMSSD (ms),
    the sleep-time baseline, the previous day (8 u32: [1] equivalent distance m, [3] sedentary min,
    [4] vigorous min, [3..6] summed for the wear estimate, [7] non-wear min), the histories, the baseline,
    the cycle day type and the rest-mode input."""
    utc: int = 0
    day_number: int = 0
    sleep_score: int = 0
    time_in_bed: int = 0
    total_sleep: int = 0
    rem: int = 0
    deep: int = 0
    latency: int = 0
    wake_ups: int = 0
    highest_temp: int = 0
    lowest_hr: int = 0
    lowest_hr_time: int = 0
    rmssd: int = 0
    p40: tuple = (0, 0)
    prev_day: Optional[List[int]] = None
    sleep_sec_14: Optional[List[int]] = None     # newest first
    distance_14: Optional[List[int]] = None      # meters, newest first
    rmssd_14: Optional[List[int]] = None
    wear_14: Optional[List[int]] = None          # percent
    temp_dev_3: Optional[List[int]] = None
    hot_90: Optional[List[int]] = None
    baseline: Optional[bytes] = None             # nine means and nine deviations
    cycle_day_type: int = 0
    rest_input: Optional[tuple] = None           # (start, end, unused, current midnight) for rest mode


@dataclass
class ReadinessOutput:
    score: int = 0
    activity_balance: int = 0
    last_day_activity: int = 0
    last_night_sleep: int = 0
    resting_hr: int = 0
    resting_hr_time: int = 0
    sleep_balance: int = 0
    temperature: int = 0
    hrv_balance: int = 0
    sleep_regularity: int = -1
    temperature_deviation: int = -0x8000
    temperature_trend_deviation: int = -0x8000


def readiness_calculate(x: ReadinessInput, feats=None, sri=0, mean_min=None):
    """`readiness_calculate`: returns a ReadinessOutput, or None when a required input block
    (p40, baseline, history) is missing. `sri` is the value `sri_calculate(true)` returns from
    the sleep-regularity state, which the caller keeps."""
    feats = feats or Features()
    if x.baseline is None or x.p40 is None:
        return None
    S = State()
    age = min(x.day_number, 255)
    base = x.baseline
    for k in range(9):
        S.set16(0x40 + 2 * k, int.from_bytes(base[0x10 + 4 * k:0x12 + 4 * k], "little"))
        S.set16(0x52 + 2 * k, int.from_bytes(base[0x12 + 4 * k:0x14 + 4 * k], "little"))
    S.b[0x64] = S.b[0x84] = age
    S.set32(0x88, x.utc)
    pd = x.prev_day
    if pd is not None:
        S.set16(0x20, pd[1] >> 3)
        S.set16(0x3E, pd[3])
        S.b[0x65] = min(pd[4], 255)
        if pd[7] != 0x5A0:
            tot = f32(float((pd[4] + pd[3]) + (pd[6] + pd[5]) & 0xFFFFFFFF))
            den = float((0x5A0 - pd[7]) & 0xFFFFFFFF)
            S.b[0x66] = int(f32(f32(f32(tot * 100.0) / den) + 0.5)) & 0xFF
    sleep_sec, dist, rmssd, wear = x.sleep_sec_14, x.distance_14, x.rmssd_14, x.wear_14
    mean_eq_dec = decode16(S.u16(0x48))
    for i in range(14):
        if sleep_sec is not None:
            S.set16(0x1E - 2 * i, round_half_away_i32(f32(float(sleep_sec[i]) / 60.0)))
        if dist is not None:
            S.set16(0x3C - 2 * i, round_half_away_i32(f32(float(dist[i]) * 0.125)))
            if wear is not None:
                wp = wear[i]
                cur = S.i16_at(0x3C - 2 * i)
                if feats.athletes:
                    tq = (100 - wp) * mean_eq_dec + (4 if mean_eq_dec > 0 else -4)
                    v = div100_round_q(tq)
                    S.set16(0x3C - 2 * i, cur + v)
                else:
                    g = S.i16_at(0x5A)
                    tq = (100 - wp) * g + (4 if g > 0 else -4)
                    v = cur * wp + tdiv(tq, 8)
                    S.set16(0x3C - 2 * i, div100_round(v))
        if rmssd is not None:
            v = rmssd[i] & 0xFFFFFFFF
            lg = 0.0 if v == 0 else f32(math.log2(float(v)))
            S.set16(0x82 - 2 * i, round_half_away_i32(f32(lg * 100.0)))
    rm = RestMode()
    if x.rest_input is not None:
        from rest_recovery import RestInput, get_state
        r = get_state(RestInput(*x.rest_input))
        rm = RestMode(r.state, r.b, r.c)
    sum_s = (x.time_in_bed - x.total_sleep) & 0xFFFFFFFF
    ss = SleepSummary(
        awake=round_half_away_i32(f32(float(sum_s) / 30.0)) if x.time_in_bed > x.total_sleep else 0,
        rem=round_half_away_i32(f32(float(x.rem) / 30.0)),
        light=round_half_away_i32(f32(float(((x.total_sleep - (x.deep + x.rem)) & 0xFFFFFFFF)) / 30.0)),
        deep=round_half_away_i32(f32(float(x.deep) / 30.0)),
        extra=round_half_away_i32(f32(float(x.lowest_hr_time) / 30.0)),
        b0a=x.lowest_hr & 0xFF, sleep_score=x.sleep_score & 0xFF, b0c=x.wake_ups & 0xFF,
        b0d=round_half_away_i32(f32(float(x.latency) / 30.0)) & 0xFF)
    t = x.highest_temp & 0xFFFF
    if x.cycle_day_type == 2:
        t = (t - 20) & 0xFFFF
    h = round_half_away_i32(f32(float(x.time_in_bed) / 3600.0))
    f = 0.0 if x.rmssd == 0 else f32(math.log2(float(x.rmssd)))
    summary, _ = recovery_run(S, S, ss, rm, t, h, f, x.p40[1], 0, age, False, (x.lowest_hr, x.lowest_hr_time),
                              x.utc, feats, sri, mean_min)
    out = ReadinessOutput(
        score=_sb(summary[0]), activity_balance=_sb(summary[4]), last_day_activity=_sb(summary[3]),
        last_night_sleep=_sb(summary[1]), resting_hr=_sb(summary[5]), resting_hr_time=_sb(summary[7]),
        sleep_balance=_sb(summary[2]), temperature=_sb(summary[6]), hrv_balance=_sb(summary[8]),
        sleep_regularity=-1 if summary[9] == 0x80 else _sb(summary[9]))
    if x.temp_dev_3 is not None and x.hot_90 is not None:
        bm = int.from_bytes(base[0x2C:0x2E], "little")
        bm = sx16(bm)
        bt = sx13((bm + (7 if bm < 0 else 0)) >> 3)
        out.temperature_deviation, out.temperature_trend_deviation = temperature_deviation(
            x.highest_temp, bt & 0xFFFFFFFF, x.hot_90, x.temp_dev_3)
    return out


def div100_round_q(tq):
    """round(trunc(tq / 8) / 100) half away from zero; tq is the offset product."""
    q = tdiv(tq, 8)
    return (q + 50) // 100 if tq >= 8 else -((50 - q) // 100)
