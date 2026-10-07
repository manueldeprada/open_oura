#!/usr/bin/env python3
"""Checks for bedtime_detect.py: input assembly, the window-start rule and the sync chain on a synthetic
night.

    python tools/test_bedtime_detect.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bedtime_detect as B  # noqa: E402

T0 = 1_790_000_000_000
H = 3_600_000
MIN = 60_000


def synthetic_night():
    """Awake 0-1 h, still 1-8 h, awake 8-9 h; temperature 1 degree warmer while asleep."""
    rnd = random.Random(1)
    ev = {"motion": [], "temp": [], "state": [], "session": [(T0, 8, 1), (T0 + 9 * H, 8, 3)]}
    for k in range(9 * 120):
        h = k / 120
        s = rnd.randint(8, 25) if (h < 1 or h >= 8) else (rnd.randint(1, 3) if rnd.random() < 0.05 else 0)
        if s:
            ev["motion"].append((T0 + k * 30_000, s))
    for k in range(9 * 60):
        ev["temp"].append((T0 + k * 60_000 + 3000, 34.0 + (1.0 if 1 <= k / 60 < 8 else 0)))
    return ev


def test_assembly():
    ev = {"motion": [(T0, 7), (T0 + 29_000, 9), (T0 + 90_000, 4)],
          "temp": [(T0 + 1000, 34.567), (T0 + 61_000, 0.0), (T0 + 125_000, 35.0)],
          "state": [(T0 - 5000, 8), (T0 + 62_000, 1)], "session": []}
    motion, motion_ts, temp, temp_ts, hr, lo, hi = B.assemble(ev, T0 - 10_000, T0 + 200_000)
    assert (lo, hi) == (T0, T0 + 125_000)
    assert motion_ts == [T0, T0 + 30_000, T0 + 60_000, T0 + 90_000]
    # the 29 s event is within 2 s of the bin end, so it goes to bin 1; bin 2 is empty and the last state
    # at its start (charging, 8) gives 255 until the state at +62 s (not in finger) gives 31
    assert motion == [7, 9, 255, 4], motion
    assert temp == [3457, 0], temp  # the 0.0 sample is ignored; 35.0 at +125 s is past the last 60 s bin
    assert temp_ts == [T0, T0 + 60_000]
    assert hr == []
    assert B.centi(34.567) == 3457 and B.centi(36.125) == 3613 and B.centi(0.005) == 1


def test_hr_periods():
    s = [(T0 - 50, 8, 1), (T0 - 10, 8, 4), (T0, 8, 1), (T0 + 5, 4, 1), (T0 + 10, 8, 3), (T0 + 20, 8, 1), (T0 + 30, 8, 12)]
    ev = {"session": s}
    # window starts at T0: the newest session event before it (T0-10, an end) does not open a period
    assert B.hr_periods(ev, T0, T0 + 100, T0 + 90) == [(T0, T0 + 10), (T0 + 20, T0 + 90)]
    # a period whose end event is after the span end but within lastEventTimestamp closes there
    assert B.hr_periods(ev, T0, T0 + 10, T0 + 5) == [(T0, T0 + 10)]
    # the newest session event before the window opens a period
    ev = {"session": [(T0 - 10, 8, 1), (T0 + 40, 8, 4)]}
    assert B.hr_periods(ev, T0, T0 + 100, T0 + 90) == [(T0 - 10, T0 + 40)]


def test_window_start():
    day = 24 * H
    base = (T0 // day) * day
    end = base + 12 * H   # a batch ending at noon UTC
    app = B.App({"motion": [], "temp": [], "state": [], "session": []},
                sleeps=[(base - 2 * day, base - 2 * day + 8 * H, 0), (base - day, base - day + 8 * H, 0)], seed=True)
    # j = min(UTC midnight of the batch start, end - 36 h) = base - 24 h; the sleep that starts before j is two days back
    assert app.window_start(end - 10 * MIN, end, end) == base - 2 * day
    # no seed and nothing stored: the 28 day look-back
    app.seed = False
    assert app.window_start(end - 10 * MIN, end, end) == end - 28 * day
    # a stored detected period of 3 h or more wins over the confirmed sleeps
    app.seed = True
    app.detected = [(base - 40 * H, base - 30 * H), (base - 50 * H, base - 48 * H)]
    assert app.window_start(end - 10 * MIN, end, end) == base - 40 * H
    # among detected periods the one whose end falls on the latest UTC day wins, then the longer one,
    # whatever the starts; equal keys keep the stored order
    d = [(base - 40 * H, base - 30 * H), (base - 45 * H, base - 30 * H + 1), (base - 44 * H, base - 33 * H)]
    app.detected = d
    assert app.window_start(end - 10 * MIN, end, end) == base - 45 * H
    app.detected = [(base - 40 * H, base - 30 * H), (base - 41 * H, base - 31 * H)]
    assert app.window_start(end - 10 * MIN, end, end) == base - 40 * H
    # a detected period must start strictly before j = base - 24 h
    app.detected = [(base - 24 * H, base - 10 * H)]
    app.sleeps = [(base - 2 * day, base - 2 * day + 8 * H, 0)]
    assert app.window_start(end - 10 * MIN, end, end) == base - 2 * day
    # confirmed sleeps: only 3 h or longer, the longest of each local day, then the latest start; a nap is skipped
    app.detected = []
    app.sleeps = [(base - 2 * day, base - 2 * day + 8 * H, 0), (base - 2 * day + 10 * H, base - 2 * day + 12 * H, 0)]
    assert app.window_start(end - 10 * MIN, end, end) == base - 2 * day
    app.sleeps = [(base - 2 * day, base - 2 * day + 8 * H, 0), (base - 2 * day + 9 * H, base - 2 * day + 13 * H, 0)]
    assert app.window_start(end - 10 * MIN, end, end) == base - 2 * day   # same local day: the 8 h sleep is longer
    # strict comparison with j
    app.sleeps = [(base - day, base - day + 8 * H, 0)]
    assert app.window_start(end - 10 * MIN, end, end) == end - 28 * day


def test_chain():
    ev = synthetic_night()
    app = B.App(ev, [], seed=True)
    res = app.sync(T0 - H, T0 + 9 * H)
    win_start, prev, periods = res
    assert prev == [] and periods and len(periods) == 1, periods
    s, e, major = periods[0]
    assert major == 1 and abs(s - (T0 + H)) < 300_000 and abs(e - (T0 + 8 * H)) < 300_000, periods
    assert app.detected == [(s, e)]
    # a sync delivering only the first five hours: the end follows the newest event by under 3 minutes
    app = B.App(ev, [], seed=True)
    win_start, prev, periods = app.sync(T0 - H, T0 + 5 * H)
    assert 0 < T0 + 5 * H - periods[-1][1] < 180_000, periods
    # no events in the batch: no detection
    assert app.sync(T0 + 20 * H, T0 + 21 * H) is None


if __name__ == "__main__":
    test_assembly()
    test_hr_periods()
    test_window_start()
    test_chain()
    print("ok")
