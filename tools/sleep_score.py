#!/usr/bin/env python3
"""Oura sleep score from stage counts (`sleep_score`) or minutes (`sleep_score_minutes`), with
age-dependent limits and a timing contributor.

Nine inputs are mapped to contributors in 0..100 by a piecewise-linear curve (shared
Y values 0, 25, 95, 100, four X knots per input) and combined as
round(sum(w_i * c_i) / 100):

    i  input / contributor            weight  knots (X)
    0  total sleep, minutes               35  age dependent (`limits_v2`)
    1  deep sleep, minutes                10  age dependent
    2  REM sleep, minutes                 10  age dependent
    3  efficiency, percent                10  30 60 90 95
    4  latency, |minutes - 15|            10  60 28 2 0 (descending)
    5  wake-ups A (u8)                     5  240 128 16 0 (descending); input = A * 6720 / max(T, 60)
    6  motion epochs B (u16)               5  80 50 25 15 (descending); input = B * 50 / max(T, 60)
    7  motion bursts C (u8)                5  96 48 16 0 (descending); input = C * 6720 / max(T, 60)
    8  timing                              10  390 240 100 90 (descending) on |midpoint - 90|, or
                                              the weekday timing score when a weekday (0..254) is given

T is total sleep in minutes. Rows 5-7 are the restfulness group; `tools/sleep_score_inputs.py`
derives A, B and C (and the other arguments) from a night's data.
`limits_v2(age)` fills the age-dependent knots.

There are seven contributors: total sleep (0), deep (1), REM (2), efficiency (3),
latency (4), restfulness (5 + 6 + 7, each scaled to about a third by (2c + 3) / 6) and timing (8).
"""
from fractions import Fraction

import numpy as np

Y = (0, 25, 95, 100)
WEIGHTS = (35, 10, 10, 10, 10, 5, 5, 5, 10)
# Rows 3..8 are constants; rows 0..2 come from the age limits.
_STATIC = (
    (30, 60, 90, 95),
    (60, 28, 2, 0),
    (240, 128, 16, 0),
    (80, 50, 25, 15),
    (96, 48, 16, 0),
    (390, 240, 100, 90),
)
_F32 = np.float32


def _round_away(x):
    """Round to nearest, ties away from zero."""
    x = float(x)
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def _f32_fma(a, b, c):
    """float32 fused multiply-add: a * b + c with a single rounding."""
    return _F32(float(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))))


def limits_v2(age):
    """Knots of rows 0..2 (total, deep, REM) for an age in years (low byte used)."""
    age &= 0xFF
    row_peak = 0x1E6 - ((age * 0x1C5) >> 8)
    fit = _f32_fma(_F32(age), _F32(np.uint32(0xBF9126E9).view(np.float32)),
                  _F32(np.uint32(0x42E747AE).view(np.float32)))
    m = max(((_round_away(fit) + 0x8000) & 0xFFFF) - 0x8000, 20)
    r = max(0x72 - ((age * 0x155) >> 9), 60)
    top = (row_peak * 5 + 2) >> 2 if age < 31 else 0x21C
    row0 = (0, (row_peak + 1) >> 1, (row_peak * 0x12E + 0x80) >> 8, top)
    row1 = (0, _round_away(_F32(m) / _F32(4)), m, _round_away(_F32(m) * _F32(1.25)))
    row2 = (0, _round_away(_F32(r) / _F32(4)), r, _round_away(_F32(r) * _F32(1.25)))
    return row0, row1, row2


def limits_v1(age):
    """Older limits, kept for reference."""
    age &= 0xFF
    row_peak = 0x1E6 - ((age * 0x1C5) >> 8)
    r = max(0x72 - ((age * 0x155) >> 9), 60)
    m = max(0x6D - ((age * 0x117) >> 8), 20)
    top = (row_peak * 5 + 2) >> 2 if age < 31 else 0x21C
    row0 = (0, (row_peak + 1) >> 1, (row_peak * 0x12E + 0x80) >> 8, top)
    row1 = (0, (m * 0x54 + 0x80) >> 8, (m * 0x12E + 0x80) >> 8, 150)
    row2 = (0, (r * 0x54 + 0x80) >> 8, (r * 0x131 + 0x80) >> 8, 150)
    return row0, row1, row2


def _sdiv(a, b):
    """Integer division: truncates toward zero; division by zero gives 0."""
    if b == 0:
        return 0
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _clamp100(v):
    return max(0, min(100, v))


def curve(v, k):
    """Contributor value (0..100) for input v on knots k; uses integer
    arithmetic (round half up for ascending X, half down for descending X)."""
    if k[1] > k[0]:  # ascending
        if v <= k[0]:
            return 0
        i = 1 if v <= k[1] else (3 if v > k[2] else 2)
    else:  # descending
        if v >= k[0]:
            return 0
        i = 1 if v >= k[1] else (3 if v < k[2] else 2)
    lo, hi = k[i - 1], k[i]
    d = hi - lo
    if d == 0:
        return 0
    n = (v - lo) * (Y[i] - Y[i - 1])
    n += (d >> 1) if d > 0 else ((-d) & 0xFFFE) >> 1
    return _clamp100(_sdiv(n, d) + Y[i - 1])


def timing_score(day, ref, x):
    """ as float32 arithmetic. `day` 1..6
    selects the narrow windows; other values use a reference of 180 and wider windows.
    Returns a float in 0..100."""
    f = _F32
    inside = 1 <= (day & 0xFF) <= 6
    a = f(-30) if inside else f(-45)
    p = f(ref) if inside else f(180)
    q = f(30) if inside else f(45)
    x = f(x)
    t = f(a + f(-150))
    b1 = f(p + t)
    b0 = f(p + f(t + f(-130)))
    c25, c0 = f(25), f(0)

    def interp(lo, hi, ylo, yhi):
        if hi == lo:
            return 0.0
        slope = f(f(yhi) - f(ylo)) / f(hi - lo)
        r = _f32_fma(f(x - lo), slope, f(ylo))
        return float(min(max(r, f(0)), f(100)))

    if b1 >= x and b0 <= x:
        return interp(b0, b1, 0, 25)
    b2 = f(p + a)
    if b2 >= x and b1 <= x:
        return interp(b1, b2, 25, 100)
    b3 = f(p + q)
    if not (b2 > x or b3 < x):  # b2 <= x <= b3
        return interp(b2, b3, 100, 100)
    ext = f(q + f(150))
    b4 = f(p + ext)
    if not (b3 > x or b4 < x):  # b3 <= x <= b4
        return interp(b3, b4, 100, 25)
    if b4 > x:
        return 0.0
    b5 = f(p + f(ext + f(130)))
    if b5 >= x:
        return interp(b4, b5, 25, 0)
    return 0.0


def score_from_inputs(inputs, age, timing_day=0xFF, timing_mid=0, timing_x=0, limits=limits_v2):
    """Inner combiner. `inputs` is the 9 u16 values (see the table above). Returns
    (contributors[7], total) with the seven contributors."""
    rows = list(limits(age)) + list(_STATIC)
    acc = 0
    out = []
    for i in range(9):
        v = inputs[i] & 0xFFFF
        if i == 8 and timing_day != 0xFF:
            c = _round_away(_F32(timing_score(timing_day, _F32(timing_mid), _F32(timing_x))))
        else:
            c = curve(v, rows[i])
        acc += c * WEIGHTS[i]
        c = _clamp100(c)
        if 5 <= i <= 7:
            c = (((2 * c + 3) & 0xFF) * 171) >> 10
        out.append(c)
    total = ((acc + 50) * 0x51EB851F) >> 37 if acc + 50 < (1 << 32) else (acc + 50) // 100
    return out, total


def _fix(c):
    return 1 if c == 0 else (100 if c >= 101 else c)


def _contributors(out):
    c = [out[0], out[1], out[2], out[3], out[4], out[5] + out[6] + out[7], out[8]]
    return [_fix(v) for v in c]


def _udiv(a, b):
    return 0 if b == 0 else a // b


def sleep_score(age, time_in_bed, deep, light, rem, latency, restless, tosses, ext,
                mid=0, timing_day=0xFF, timing_ref=0, limits=limits_v2):
    """Sleep score from stage counts. Sleep stages and `time_in_bed` are in
    30-second epochs (u16); `latency` is in epochs (u8), `restless`, `ext` are u8
    counters, `tosses` is u16, `mid` is a signed 16-bit midpoint, `timing_day` a u8
    (0xFF disables the timing function), `timing_ref` an int32 passed to it.
    Returns (7 contributors, total score)."""
    a, b, c, d = time_in_bed & 0xFFFF, deep & 0xFFFF, light & 0xFFFF, rem & 0xFFFF
    b_plus = b + 1
    t = ((b_plus + c + d) >> 1) & 0xFFFF
    denom = t if t > 0x3C else 0x3C
    h = denom >> 1
    e = ((latency & 0xFF) + 1) >> 1
    mid = ((mid + 0x8000) & 0xFFFF) - 0x8000
    inputs = [
        t,
        (b_plus >> 1) & 0xFFFF,
        ((d + 1) >> 1) & 0xFFFF,
        _udiv(t * 100 + ((a >> 2)), a >> 1) & 0xFFFF,
        abs(e - 15),
        _udiv((restless & 0xFF) * 0x1A40 + h, denom) & 0xFFFF,
        _udiv((tosses & 0xFFFF) * 50 + h, denom) & 0xFFFF,
        _udiv((ext & 0xFF) * 0x1A40 + h, denom) & 0xFFFF,
        abs(mid - 90),
    ]
    ref = ((timing_ref + 0x80000000) & 0xFFFFFFFF) - 0x80000000
    out, total = score_from_inputs(inputs, age, timing_day, ref, mid, limits)
    return _contributors(out), total


def sleep_score_minutes(age, total, deep, rem, efficiency, latency, restless, tosses,
                        ext, mid=0, timing_day=0xFF, timing_ref=0, limits=limits_v2):
    """Sleep score from minutes: the same combiner with durations
    already in minutes and the ratios precomputed (`efficiency` percent u8, `latency`
    u8 used as is, `restless` u8, `tosses` u16, `ext` u8)."""
    a = total & 0xFFFF
    mid = ((mid + 0x8000) & 0xFFFF) - 0x8000
    inputs = [
        a, deep & 0xFFFF, rem & 0xFFFF, efficiency & 0xFF, latency & 0xFF, restless & 0xFF,
        _udiv((tosses & 0xFFFF) * 50 + (a >> 1), a) & 0xFFFF, ext & 0xFF, abs(mid - 90),
    ]
    ref = ((timing_ref + 0x80000000) & 0xFFFFFFFF) - 0x80000000
    out, tot = score_from_inputs(inputs, age, timing_day, ref, mid, limits)
    return _contributors(out), tot
