#!/usr/bin/env python3
"""Inputs of the Oura sleep score: the motion-seconds handling and the epoch scan.

`tools/sleep_score.py` holds the arithmetic of the score; this module
derives its arguments from a night's data, so `night_to_score` maps raw night data to the
total score and the seven app contributors. `tools/sleep_inputs.py` (used by
`score_sleep.py`) is an older heuristic derivation from decoded ring events; this one
applies the scoring scan.

Night data (the fields that reach the score):

    phases        packed stage nibbles (low nibble first), 1 deep, 2 light, 3 REM, 4 awake
    motion        one byte per 30 s ("motion seconds"), starting at motion_start
    bedtime       start/end in Unix seconds and the time zone offset in minutes
    age           user age in years (selects the total, deep and REM sleep knots)
    chronotype    (type 0..255, ideal sleep midpoint in seconds) or None

Table. The night is laid on a grid of 30-second epochs aligned down to a
multiple of 30 s from the bedtime start: `n = end // 30 - start // 30` rows, at most
0x708. Row `i` takes stage `i` from the phase list (the first phases whose time, rounded
to the nearest 30 s, precedes the row time are skipped), and a trailing four-sample sum
of the motion bytes (`motion_row`). The per-epoch heart rate, breathing rate and quality
features are also rows of the table, but they do not enter the score.

Scan. Rows 0..min(n, 0x707) are visited in order (the row after the last night epoch is
read too and counts as awake when the table is empty there). `eff` is the stage with 0
read as awake (4). The scan yields

    deep, light, REM   number of epochs of stage 1, 2, 3
    time in bed        rows visited (min(n, 0x707) + 1), the argument `a`
    latency            u8 epochs: index of the first epoch of the first run of 10
                       epochs that are all asleep (stage 1..3); 1 when that run starts
                       before epoch 10 (a start at 0 is stored as 1); 0xff when it
                       starts after epoch 264; 0 if there is no such run
    wake-ups  (f)      times the sleeper is awake for 2 or more consecutive epochs after
                       sleep onset; the final wake-up is not counted
    restless  (g)      epochs with a nonzero motion row while not awake
    got-up    (x)      "motion bursts": a motion row at least 24 above the larger of the
                       rows 2 and 3 epochs earlier, when that larger value is itself at
                       least 24, after the detector re-armed (see `scan`)
    midpoint           (sum of (i + 1) over asleep epochs + count / 2) / count, in epochs

Sleep onset needs min(i + 1, 20) consecutive asleep epochs. The counters are the wake-up count for f, the restless periods for g and the got-up count for x.
The Restfulness contributor (15 points in the weighting) is the sum of the three
scaled counters (see docs/algorithms/metrics.md).

Timing midpoint: `mid = int16(clock + (midpoint + 1) // 2)` with `clock` the local time of
the bedtime start in minutes, folded to -720..719 (hours 12..23 minus 1440). The weekday
argument of the timing score is the chronotype type (0xff without a
chronotype) and its reference is `trunc(ideal_midpoint_seconds / 60)`.

Not ported: the heart-rate, breathing-rate and temperature outputs of the same loop (they
do not affect the score), and the entry that reads the same table from encoded event bytes.
"""
from dataclasses import dataclass

import sleep_score as S

ROWS = 0x708          # table capacity (1800 epochs = 15 h)
SCAN_MAX = 0x707      # last row the scan visits


def _trunc_div(a, b):
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def epoch_count(start_s, end_s):
    """Rows of the table for a bedtime period, at most 0x708."""
    n = (end_s - (end_s % 30) - (start_s - start_s % 30)) // 30
    return ROWS if ((n >> 0) & 0xFFFF) > ROWS else n & 0xFFFF


def unpack_phases(data):
    """Packed stage nibbles to a per-epoch list. A byte below 0x10 holds one last epoch;
    a nibble above 4 reads as 0."""
    out = []
    for b in data:
        low = b & 0xF
        out.append(0 if low > 4 else low)
        if b < 0x10:
            break
        out.append(0 if b > 0x4F else b >> 4)
    return out


def place_phases(codes, phase_start_s, bed_start_s):
    """Stage per table row. Phase `k` has time `phase_start_s + 30 k`, rounded to the
    nearest 30 s; phases earlier than the next row are skipped, the others fill the
    rows in order (a later start is not padded)."""
    rows = [0] * ROWS
    t0 = bed_start_s - bed_start_s % 30
    j = 0
    for k, code in enumerate(codes):
        if j == ROWS:
            break
        r = ((phase_start_s + 30 * k + 15) // 30) * 30
        if r >= t0 + 30 * j:
            rows[j] = code
            j += 1
    return rows


def motion_rows(motion, motion_start_s, bed_start_s):
    """Motion field per table row. Sample `q` of the
    kept samples contributes the sum of the last four kept samples (fewer at the start),
    modulo 256, to row `first + q`."""
    rows = [0] * ROWS
    t0 = bed_start_s - bed_start_s % 30
    count = len(motion)
    grid_start = motion_start_s - motion_start_s % 30
    first = skip = 0
    if grid_start > t0:
        first = min((grid_start - t0 + 29) // 30, SCAN_MAX)
    else:
        while grid_start < t0 and skip < count - 1:
            grid_start += 30
            skip += 1
    keep = count - skip
    if keep >= 0x709:
        keep = 0x708
    if keep <= 0:
        return rows
    m = motion[skip:skip + keep]
    ncopy = min(max(keep, 1) - 1, SCAN_MAX - first) + 1
    for k in range(ncopy):
        rows[first + k] = sum(m[max(0, k - 3):k + 1]) & 0xFF
    return rows


@dataclass
class Scan:
    time_in_bed: int      # u16 epochs, `a`
    deep: int             # `b`
    light: int            # `c`
    rem: int              # `d`
    latency: int          # u8 epochs, `e`
    wake_ups: int         # `f`
    restless: int         # `g`
    got_up: int           # `x`
    midpoint: int         # mean asleep epoch index + 1
    visited: int          # min(n, 0x707): the score runs only from 10 on


def scan(phase, motion, n):
    """Port of the epoch loop. `phase` and `motion` are the table rows."""
    last = min(n, SCAN_MAX)
    cnt = {1: 0, 2: 0, 3: 0, 4: 0}
    run = 0                 # signed byte: +asleep run, -awake run (saturating at +-127)
    onset = False
    asleep_flag = 0
    wakes = 0
    k = 0                   # consecutive asleep epochs, saturating at 255
    latency = 0
    armed = True
    calm = 0
    got_up = 0
    restless = 0
    idx_sum = 0
    hist = [0, 0, 0]               # m[i-1], m[i-2], m[i-3]
    for i in range(last + 1):
        ph = phase[i]
        mi = motion[i]
        eff = 4 if ph == 0 else ph
        if eff > 4:
            break
        cnt[eff] += 1
        if eff == 4:
            r = run if run < 128 else run - 256
            r = min(r, 0) - (1 if r > -127 else 0)
            run = r & 0xFF
            asleep = 0
            if onset and r <= -2:
                onset = False
                wakes += 1
                asleep_flag = 0
            knext = 0
        else:
            r = run if run < 128 else run - 256
            run = 0x7F if run == 0x7F else (max(r, 0) + 1) & 0xFF
            rr = run if run < 128 else run - 256
            if not onset and min(i + 1, 20) <= (rr & 0xFFFF):
                onset = True
                asleep_flag = 1
            asleep = 1
            if k == 0xFF:
                knext = 0xFF
            else:
                knext = k + 1
                if k >= 9 and latency == 0:
                    if i < 10:
                        latency = 1
                    elif i > 0x108:
                        latency = 0xFF
                    else:
                        latency = i + 1 - 10
        m3 = max(hist[2], hist[1])
        d = max(mi - m3, 0)
        if armed and m3 >= 24 and d >= 24:
            got_up += 1
            armed = False
            calm = 0
        elif d >= 24:
            calm = 0
        elif calm != 0xFF:
            calm += 1
        if not armed and min(i + 1, 20) <= calm:
            armed = True
        if mi != 0 and eff != 4:
            restless += 1
        if asleep:
            idx_sum += i + 1
        k = knext
        hist = [mi, hist[0], hist[1]]
    deep, light, rem = cnt[1], cnt[2], cnt[3]
    if wakes & 0xFF and not asleep_flag:
        wakes -= 1
    total = (light + deep + rem) & 0xFFFF
    mean = 0 if total == 0 else (idx_sum + (total >> 1)) // total
    return Scan(last + 1, deep, light, rem, latency, wakes, restless, got_up, mean, last)


def local_clock_minutes(start_s, tz_min):
    """Local time of the bedtime start in minutes, folded to -720..719."""
    local = start_s + tz_min * 60
    h, m = (local // 3600) % 24, (local // 60) % 60
    return h * 60 + m - (1440 if h > 11 else 0)


def _s16(v):
    return ((v + 0x8000) & 0xFFFF) - 0x8000


def score_args(sc, clock, chronotype=None):
    """Arguments of `sleep_score.sleep_score` (after `age`), as the keyword names of
    `sleep_score.sleep_score`."""
    day, ref = 0xFF, 0
    if chronotype is not None:
        day = chronotype[0] & 0xFF
        ref = _trunc_div(_s32(chronotype[1]), 60)
    return dict(time_in_bed=sc.time_in_bed, deep=sc.deep, light=sc.light, rem=sc.rem,
                latency=sc.latency, restless=sc.wake_ups, tosses=sc.restless, ext=sc.got_up,
                mid=_s16(clock + ((sc.midpoint + 1) >> 1)), timing_day=day, timing_ref=ref)


def _s32(v):
    return ((v + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def night_to_score(age, bed_start_s, bed_end_s, tz_min, phases, phase_start_ms,
                   motion=b"", motion_start_ms=0, chronotype=None):
    """Total score and the seven app contributors from a night's data.

    Returns None when fewer than 10 epochs were scanned (no score is written),
    otherwise a dict with `total`, `contributors` (total sleep, deep, REM, efficiency,
    latency, restfulness, timing), `args` (the arguments of `sleep_score.sleep_score`)
    and `scan`."""
    n = epoch_count(bed_start_s, bed_end_s)
    pr = place_phases(unpack_phases(phases), _trunc_div(phase_start_ms, 1000), bed_start_s)
    # fewer motion samples than epochs: the motion is skipped
    mr = motion_rows(motion if len(motion) >= n else b"", _trunc_div(motion_start_ms, 1000), bed_start_s)
    sc = scan(pr, mr, n)
    if sc.visited < 10:
        return None
    args = score_args(sc, local_clock_minutes(bed_start_s, tz_min), chronotype)
    contributors, total = S.sleep_score(age, **args)
    return dict(total=total, contributors=contributors, args=args, scan=sc)


def period_from_scan(sc, bed_start_s, tz_min, is_main=1):
    """The `SleepPeriodInput` built from a night's scan: epochs become seconds, time in bed is one epoch less than the scan's row
    count (the scan counts one extra row), efficiency is the rounded percent of sleep in bed.
    Returns a `readiness_inputs.SleepPeriod`."""
    import readiness_inputs as RI
    tib = sc.time_in_bed - 1
    total = sc.deep + sc.light + sc.rem
    return RI.SleepPeriod(is_main=is_main, start=bed_start_s, tz=tz_min, time_in_bed=tib * 30, total_sleep=total * 30,
                          rem=sc.rem * 30, deep=sc.deep * 30, efficiency=(200 * total + tib) // (2 * tib) if tib else 0,
                          latency=sc.latency * 30, wake_ups=sc.wake_ups, restless=sc.restless, got_up=sc.got_up,
                          midpoint=sc.midpoint)


def app_sleep_score(sc, age, bed_start_s, tz_min, chrono=None):
    """Daily sleep score and contributors: the daily sleep output (minutes, limits v1)
    on the period built by `period_from_scan`. `chrono` is (type, ideal midpoint in seconds) or
    None for the default (0, 0). Returns the `DailySleep` (contributors in the order
    total sleep, deep, REM, efficiency, latency, restfulness, timing)."""
    import readiness_inputs as RI
    ctype, cmid = chrono if chrono is not None else (0, 0)
    return RI.get_daily_sleep(age, ctype & 0xFF, RI._tdiv(RI._s32(cmid), 60), [period_from_scan(sc, bed_start_s, tz_min)])
