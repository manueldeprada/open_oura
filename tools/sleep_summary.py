#!/usr/bin/env python3
"""Night summary values that
the sleep score does not use: average RMSSD (total, non-REM, REM), highest temperature, average
heart rate, and the resampling of ring events onto the grids they are computed on.


Functions:

* `calculate_average_rmssd`
* `nightly_temperature` (with a 7-point median), the night's highest temperature and, fed the
  digital temperature series, the highest digital temperature
* `hr_average_times8` (average heart rate times 8 and the offset argument)
* `grid`, the resampling of ring events onto the grids
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sleep_score_inputs as SI
from readiness_score import f32


def _div_round(total, n):
    return (total + (n >> 1)) // n


def slot_offset(first_ms, n_slots, bed_start_s):
    """Argument of `calculate_average_rmssd`: the event end time (first slot time + n * 300 s) minus
    the bedtime start, wrapped to [-150, 150) s."""
    end = first_ms // 1000 + n_slots * 300
    d = end - bed_start_s
    return ((d + 150) % 300) - 150 if d >= 1 else 0


def calculate_average_rmssd(stages, rmssd, n_slots, offset):
    """(average RMSSD, non-REM average, REM average) as `calculate_average_rmssd` returns them.

    `stages` is the night's per-30-second stage list, one entry per table row (1 deep, 2 light, 3 REM,
    4 awake, 0 invalid; they are held as the ASCII characters '0'..'4' and the string ends
    after the last row), `rmssd` the RMSSD byte of each 5-minute slot and `offset` the argument of
    `slot_offset`.

    Slot i looks at the nine epochs 10 i .. 10 i + 8 and skips those whose index is at most
    ~v, v = 1 - (15 - offset) / 30 for offset < 1 and (offset + 15) / 30 otherwise (unsigned
    division). A slot with any awake epoch or a zero RMSSD is left out. Otherwise its RMSSD adds
    to the total and, with six or more deep or light epochs, to the non-REM sum; else with six or
    more REM epochs, to the REM sum. Each average is (sum + n / 2) / n, 0 without a slot. The scan
    ends after the slot in which the stage string ends."""
    if offset < 1:
        v = (1 - ((15 - offset) & 0xFFFFFFFF) // 30)
    else:
        v = ((offset + 15) & 0xFFFFFFFF) // 30
    v = ((v + 0x80000000) & 0xFFFFFFFF) - 0x80000000
    limit = ~v
    data = list(stages)
    at = lambda k: data[k] if 0 <= k < len(data) else None
    tot, nrem, rem = [0, 0], [0, 0], [0, 0]
    i = 0
    go = True
    while i != n_slots:
        base = 10 * i
        asleep = rem_n = awake = 0
        go = True
        for j in range(9):
            k = base + j
            if k <= limit:
                go = True
                continue
            c = at(k)
            if c == 1 or c == 2:
                asleep += 1
            elif c == 3:
                rem_n += 1
            elif c == 4:
                awake += 1
            elif c is None:
                go = False
                break
            go = at(k + 1) is not None
            if not go:
                break
        if awake == 0:
            r = rmssd[i] if i < len(rmssd) else 0
            if r:
                if asleep >= 6:
                    nrem[0] += r
                    nrem[1] += 1
                elif rem_n >= 6:
                    rem[0] += r
                    rem[1] += 1
                tot[0] += r
                tot[1] += 1
        i += 1
        if not go:
            break
    f = lambda a: _div_round(a[0], a[1]) if a[1] else 0
    return f(tot), f(nrem), f(rem)


def median7(v):
    """`median7_u16`: the median of seven u16 values."""
    return sorted(v)[3]


def nightly_temperature(temps):
    """The night's highest temperature in centidegrees.

    `temps` are the u16 centidegrees of the 60 s windows (0 for a missing window). Each sample enters a
    ring of seven and the median of the ring is the smoothed value (the ring starts as zeros). In every
    block of 30 samples the largest non-zero smoothed value is kept when it exceeds the smallest
    non-zero one by at most 249 (0 when the block has no non-zero value or the spread is larger).
    With fewer than four kept maxima the result is 0, otherwise the second largest maximum
    (duplicates count)."""
    ring = [0] * 7
    pos = 0
    lo, hi = 0xFFFF, 0
    kept = []
    for k, t in enumerate(temps, 1):
        ring[pos] = t & 0xFFFF
        m = median7(ring)
        if m != 0:
            lo = min(lo, m)
            hi = max(hi, m)
        pos = 0 if pos == 6 else pos + 1
        if k % 30 == 0:
            if hi >= lo and hi - lo <= 249 and hi != 0:
                kept.append(hi)
            lo, hi = 0xFFFF, 0
    if len(kept) < 4:
        return 0
    top = second = 0
    for v in kept:
        if v < top:
            if v >= second:
                second = v
        else:
            second = top
        top = max(top, v)
    return second


def hr_average_times8(hr):
    """Average of the non-zero 5-minute heart rates times 8, rounded half away from zero in float32
, 0 without one."""
    xs = [h for h in hr if h]
    if not xs:
        return 0
    q = f32(f32(f32(float(sum(xs))) / f32(float(len(xs)))) * f32(8.0))
    return int(q + 0.5) if q >= 0 else -int(-q + 0.5)


def grid(items, start_ms, end_ms, width_s):
    """Resampling: one slot per `width_s` seconds from the bedtime
    start; a slot holds the earliest item of `items` = [(time ms, value)] that falls into
    [slot start, min(slot end, bedtime end)), None when there is none."""
    w = width_s * 1000
    n = max(0, -(-(end_ms - start_ms) // w))
    out = [None] * n
    for t, v in sorted(items, key=lambda x: x[0]):
        if start_ms <= t < end_ms:
            i = (t - start_ms) // w
            if i < n and out[i] is None:
                out[i] = (t, v)
    return out


def hrv_items(events, end_ms):
    """Per-value (time ms, (hr, rmssd)) items of the ring's HRV events `events` = [(event time s,
    hr list, rmssd list)]. Value i of an event of n values is dated T - (n - 1 - i) 300 s; an event
    whose own time T is after the bedtime end is dropped whole."""
    out = []
    for t, hr, rm in events:
        if t * 1000 > end_ms:
            continue
        n = len(hr)
        for i in range(n):
            out.append((int(round((t - (n - 1 - i) * 300) * 1000)), (hr[i], rm[i])))
    return out


def stage_rows(codes, phase_first_ms, start_ms, end_ms):
    """The stage string for `calculate_average_rmssd`: one stage per 30 s table row
    of the bedtime period. Rows start as awake (4); the phases fill the rows in order
    (`sleep_score_inputs.place_phases`), so rows after the last phase stay awake."""
    start_s, end_s = start_ms // 1000, end_ms // 1000
    n = SI.epoch_count(start_s, end_s)
    t0 = start_s - start_s % 30
    p0 = phase_first_ms // 1000
    rows = [4] * n
    j = 0
    for k, code in enumerate(codes):
        if j >= n:
            break
        if ((p0 + 30 * k + 15) // 30) * 30 >= t0 + 30 * j:
            rows[j] = code
            j += 1
    return rows


def time_in_bed(start_ms, end_ms):
    """Seconds in bed: the difference of the two times truncated to seconds."""
    return end_ms // 1000 - start_ms // 1000


def load_ring_series(con, clock, unix_s):
    """Ring events of `motion_event`, `temp_event` and `hrv_event` with Unix times in seconds (the
    ring-sync clock `app_clock.AppClock`, then `epoch_time` where no ring sync precedes the event).
    Returns three lists: motion [(t, motion_seconds)], temp [(t, temp1)], hrv [(t, hr list, rmssd list)]."""
    import json

    def rows(name):
        for ds, cu, js in con.execute("SELECT ring_timestamp, captured_unix, decoded_json FROM events "
                                      "WHERE name=? AND decoded_json IS NOT NULL ORDER BY ring_timestamp", (name,)):
            t = clock.unix_s(ds, cu) if clock else None
            if t is None:
                t = unix_s(ds, cu)
            if t is not None:
                yield t, json.loads(js)

    motion = [(t, p.get("motion_seconds", 0)) for t, p in rows("motion_event")]
    temp = [(t, p["temps_c"][0]) for t, p in rows("temp_event")]
    hrv = [(t, p["hr_bpm"], p["rmssd_ms"]) for t, p in rows("hrv_event")]
    return motion, temp, hrv


def night_inputs(series, start_ms, end_ms):
    """The three event series of one bedtime period:
    (motion bytes per 30 s, temperature centidegrees per 60 s, [(hr, rmssd)] per 300 s, ms time of the
    first HR/HRV slot). A slot without an event holds zero; the HR/HRV series is dated by its first
    non-empty slot, else by the bedtime start."""
    motion, temp, hrv = series
    m = grid([(int(round(t * 1000)), min(255, v)) for t, v in motion], start_ms, end_ms, 30)
    tp = grid([(int(round(t * 1000)), int(round(v * 100))) for t, v in temp], start_ms, end_ms, 60)
    hv = grid(hrv_items(hrv, end_ms), start_ms, end_ms, 300)
    first = hv[0][0] if hv and hv[0] else start_ms
    return ([x[1] if x else 0 for x in m], [x[1] if x else 0 for x in tp],
            [x[1] if x else (0, 0) for x in hv], first)
