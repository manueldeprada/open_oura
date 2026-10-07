#!/usr/bin/env python3
"""Bedtime detection pipeline on ring events, ending in the detector.

    bedtime_detect.py [DB] --nights [--lags 10,20,40,60]       # whole nights against the ring's bedtime_period
    bedtime_detect.py [DB] --partial 'YYYY-MM-DD HH:MM' [--to 'YYYY-MM-DD HH:MM'] [--step MIN]
    bedtime_detect.py [DB] --until 'YYYY-MM-DD HH:MM[:SS]' [--window seeded|28d|HOURS]

The detector is the Python port in `bedtime_find.py`.
Times are local (`--tz`, hours east of UTC, default 0).

A detection runs after a ring sync. This module assembles its inputs:

- Batch. The events the sync delivered, of the types feature session, state change, motion, temperature,
  time sync and alert. `start_ts` and `end_ts` are the oldest and newest motion or temperature
  event in the batch; `last_event_ts` is the newest event of any of those types. There is no
  look-ahead and no margin relative to the wall clock: the window end is the newest event the ring
  delivered.
- Window start: `j = min(UTC midnight of start_ts, end_ts - 36 h)`. The start of the first stored detected period (`bedtime_start < j`, at least 3 h,
  latest end UTC day first, then longest) wins; otherwise, when `seed` is set, the start of
  the latest confirmed long sleep before `j` (strict `start < j`, longest of each day, then
  latest start); otherwise `end_ts - 28 d`, which is also the lower bound. See `App.window_start`.
- Clock: `--clock ring` (default) converts ring time with the latest ring-written `time_sync` at or before
  the event in ring time; `capture` uses `app_clock.py`'s capture order; `raw` one constant offset.
- Events used: motion and temperature events with `window_start <= t <= end_ts`, state
  changes up to the last of those (plus the newest one before the window), feature sessions of the
  resting-HR capability from `window_start` to `last_event_ts` (plus the newest session event before
  the window).
- Motion: one byte per 30 s bin from the span start, `bin = (t - start) // 30 s`, rounded up to the next
  bin when `t` is within 2 s of its end; the value is `motion_seconds` of the first `motion_event` (tag
  0x47) in the bin. An empty bin takes the last `state_change` (tag 0x45) at or before the bin start:
  charging (8) gives 255, not in finger (1), finger detection (2) and out of power (7) give 31, any other
  state or none gives 0.
- Temperature: one u16 per 60 s bin, `round(100 * temp1)` of the first `temp_event` (tag 0x46) sample
  with `temp1 > 0` in the bin, 0 for an empty bin. `temp1` is the first of the seven centi-degree
  values in the event body, which is `temps_c[0]` of the decoded events.
- Heart-rate measurement periods: a start by system (status 1) opens a period, any end status (2 to 11)
  closes it, and a period still open at the span end closes there. The detector returns status 13
  without at least one period.
- Previous bedtimes: the stored detected periods overlapping the window; if there are none and
  `seed` is set, the confirmed sleeps overlapping it. Major means at least 3 h for detected periods and
  a long sleep for confirmed ones.
- Average temperature: the long-term baseline temperature in centi-degrees, or a default by sex. The
  detector requires it to be nonzero and does not read it.

Periods shorter than 1 minute are dropped. See docs/algorithms/metrics.md.
"""
import argparse
import datetime
import json
import math
import sqlite3
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import resolve_db  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MOTION_BIN_MS = 30_000
TEMP_BIN_MS = 60_000
ROUND_UP_MS = 2_000
MIN_PERIOD_MS = 60_000      # shorter detections are dropped
H_MS = 3_600_000
DAY_MS = 24 * H_MS
LOOKBACK_MAX_MS = 28 * DAY_MS
LOOKBACK_MIN_MS = 36 * H_MS
MAJOR_MIN_MS = 3 * H_MS
RESTING_HR_FEATURE = 8
START_BY_SYSTEM = 1
END_STATUSES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
STATE_CHARGING, STATE_NOT_IN_FINGER, STATE_FINGER_DETECTION, STATE_OUT_OF_POWER = 8, 1, 2, 7
DEFAULT_AVG_TEMP_CENTI = 3500
MAX_OUT = 200



class RingOrderClock:
    """Ring time to Unix ms with the latest ring-sourced `time_sync` whose own ring time is at or before the
    event's ring time (the stream order of a sync). It differs from the capture-order rule of `app_clock.py`
    when an event's capture time is later than a `time_sync` that follows the event in ring time.
    `fallback(ds, cu)` converts events before the first anchor."""

    def __init__(self, con, fallback):
        self.fallback = fallback
        a = []
        for ds, js in con.execute("SELECT ring_timestamp, decoded_json FROM events "
                                  "WHERE tag = 66 AND decoded_json IS NOT NULL ORDER BY ring_timestamp, id"):
            v = json.loads(js)
            if v.get("source") != "phone" and v.get("unix_time") is not None:
                a.append((ds, int(v["unix_time"])))
        self.ds = [x[0] for x in a]
        self.syncs = a

    def __call__(self, ds, cu=None, anchor_ds=None):
        import bisect
        i = bisect.bisect_right(self.ds, ds if anchor_ds is None else anchor_ds) - 1
        if i < 0:
            return self.fallback(ds, cu)
        sd, su = self.syncs[i]
        return int(round((su + (ds - sd) / 10.0) * 1000))


def load_events(con, unix_ms):
    """Motion, temperature, state-change and feature-session events as (Unix ms, ...) tuples, sorted."""
    ev = {"motion": [], "temp": [], "state": [], "session": []}
    for ds, tag, j, cu in con.execute("SELECT ring_timestamp, tag, decoded_json, captured_unix FROM events "
                                      "WHERE tag IN (69, 70, 71, 108) AND decoded_json IS NOT NULL ORDER BY captured_unix, id"):
        d = json.loads(j)
        t = unix_ms(ds, cu)
        if tag == 71:
            ev["motion"].append((t, d["motion_seconds"]))
        elif tag == 70:
            ev["temp"].append((t, d["temps_c"][0]))
        elif tag == 69:
            ev["state"].append((t, d["state"]))
        else:
            ev["session"].append((t, d["feature_id"], d["session_status"]))
    for v in ev.values():
        v.sort()
    return ev


def load_bedtimes(con, unix_ms):
    """The ring's `bedtime_period` events as (start ms, end ms, available ms); available is the event's own time."""
    out = []
    for ds, cu, j in con.execute("SELECT ring_timestamp, captured_unix, decoded_json FROM events WHERE tag = 118 "
                                 "AND decoded_json IS NOT NULL ORDER BY id"):
        b = json.loads(j)
        # the anchor is chosen by the event's own ring time, not by the start or end it carries
        out.append((unix_ms(b["bedtime_start_ds"], cu, ds), unix_ms(b["bedtime_end_ds"], cu, ds), unix_ms(ds, cu)))
    return out


def bin_index(t, start, size, round_ms):
    """Index of the bin that holds time `t`; rounds up to the next bin when `t` is within `round_ms` of its end."""
    d = t - start
    i = d // size
    return i + 1 if round_ms and d % size >= size - round_ms else i


def centi(temp1):
    """`Math.round(temp1 * 100f)`: `temp1` is a float32 and the product is rounded in float32 before the
    round-half-up."""
    f = struct.unpack("<f", struct.pack("<f", temp1))[0]
    x = struct.unpack("<f", struct.pack("<f", f * 100.0))[0]
    return math.floor(x + 0.5)


def hr_periods(ev, win_start, last_event, span_end):
    """Resting-HR measurement periods as `b.a` builds them: the session events from the window start to
    `last_event` and the newest session event before the window, in time order."""
    before = [x for x in ev["session"] if x[0] < win_start]
    inside = [x for x in ev["session"] if win_start <= x[0] <= last_event]
    sess = ([before[-1]] if before else []) + inside
    out, start = [], None
    for t, fid, status in sess:
        if fid != RESTING_HR_FEATURE:
            continue
        if status == START_BY_SYSTEM:
            start = t
        elif status in END_STATUSES and start is not None:
            if t > start:
                out.append((start, t))
            start = None
    if start is not None and span_end > start:
        out.append((start, span_end))
    return out


def assemble(ev, win_start, w_end, last_event=None):
    """The detector's arrays for a batch ending at `w_end` with window start `win_start`:
    (motion, motion_ts, temp, temp_ts, hr, span_start, span_end), or None when detection is skipped."""
    last_event = w_end if last_event is None else last_event
    mo = [(t, s) for t, s in ev["motion"] if win_start <= t <= w_end]
    te = [(t, x) for t, x in ev["temp"] if win_start <= t <= w_end]
    if not mo or not te:
        return None
    lo = min(mo[0][0], te[0][0])
    hi = max(mo[-1][0], te[-1][0])
    if hi <= lo:
        return None
    st = [(t, s) for t, s in ev["state"] if t <= hi]
    before = [x for x in st if x[0] <= lo]
    states = ([before[-1]] if before else []) + [x for x in st if x[0] > lo]
    n = max(1, (hi - lo) // MOTION_BIN_MS)
    first = {}
    for t, s in mo:
        i = min(max(bin_index(t, lo, MOTION_BIN_MS, ROUND_UP_MS), 0), n - 1)
        first.setdefault(i, s)
    motion, motion_ts = [], []
    k = 0
    for i in range(n):
        tb = lo + i * MOTION_BIN_MS
        while k + 1 < len(states) and states[k + 1][0] <= tb:
            k += 1
        state = states[k][1] if states else None
        if i in first:
            v = first[i]
        elif state == STATE_CHARGING:
            v = 255
        elif state in (STATE_NOT_IN_FINGER, STATE_FINGER_DETECTION, STATE_OUT_OF_POWER):
            v = 31
        else:
            v = 0
        motion.append(v & 0xFF)
        motion_ts.append(tb)
    n2 = max(1, (hi - lo) // TEMP_BIN_MS)
    tb2 = {}
    for t, v in te:
        if v > 0:
            tb2.setdefault(bin_index(t, lo, TEMP_BIN_MS, 0), v)
    temp = [min(0xFFFF, max(0, centi(tb2[i]))) if i in tb2 else 0 for i in range(n2)]
    temp_ts = [lo + i * TEMP_BIN_MS for i in range(n2)]
    return motion, motion_ts, temp, temp_ts, hr_periods(ev, win_start, last_event, hi), lo, hi




def run_detector(a, avg_temp_centi=DEFAULT_AVG_TEMP_CENTI, previous=(), native=False):
    """Detected periods [(start_ms, end_ms, major)] of assembled inputs, 1 minute filter applied; None when the
    detector rejects them. `native` is accepted and ignored."""
    motion, motion_ts, temp, temp_ts, hr, _, _ = a
    import bedtime_find as F
    rc, res = F.detect_bedtime(motion, motion_ts, temp, temp_ts, hr, avg_temp_centi, previous, True)
    res = None if rc else res
    return None if res is None else [p for p in res if p[1] - p[0] >= MIN_PERIOD_MS]


def utc_midnight(ms):
    return ms - ms % DAY_MS


def epoch_day(ms):
    return ms // DAY_MS


class App:
    """The ring-sync to detection chain over a list of events.

    `sleeps`: confirmed sleeps as (start, end, available), known once `available` has passed.
    `seed`: with it the confirmed sleeps seed the window start and the previous-bedtime list; without
    it only detected periods are used.
    """

    def __init__(self, ev, sleeps=(), seed=True, native=False, lookback_cap=LOOKBACK_MAX_MS, tz=0):
        self.tz = tz
        self.ev, self.sleeps, self.seed, self.native = ev, list(sleeps), seed, native
        self.cap = lookback_cap
        self.detected = []   # stored detected periods: (start, end)

    def known_sleeps(self, now):
        return [(s, e) for s, e, av in self.sleeps if av <= now]

    def window_start(self, start_ts, end_ts, now):
        """Start of the detection window.

        Detected periods: `start < j` (strict), then `end - start >= 3 h`, then a stable sort by the UTC
        day of the end, latest first, and by duration, longest first; the first row wins and ties keep the
        insertion order.
        Confirmed sleeps: `start < j` (strict) and either a long sleep or a period of at least 3 h without
        a type; the longest of each day, then the latest start. The period length is the sleep duration."""
        floor = end_ts - self.cap
        j = min(utc_midnight(start_ts), end_ts - LOOKBACK_MIN_MS)
        cand = [p for p in self.detected if p[0] < j and p[1] - p[0] >= MAJOR_MIN_MS]
        if cand:
            best = max(cand, key=lambda p: (epoch_day(p[1]), p[1] - p[0]))   # max keeps the first of equal keys
            return max(best[0], floor)
        if self.seed:
            cand = [p for p in self.known_sleeps(now) if p[0] < j and p[1] - p[0] >= MAJOR_MIN_MS]
            per_day = {}
            for p in cand:
                d = self.sleep_day(p)
                if d not in per_day or p[1] - p[0] > per_day[d][1] - per_day[d][0]:
                    per_day[d] = p
            if per_day:
                return max(max(per_day.values(), key=lambda p: p[0])[0], floor)
        return floor

    def sleep_day(self, p):
        """The local date of the end of the sleep."""
        return (p[1] + int(self.tz * H_MS)) // DAY_MS

    def previous(self, win_start, w_end, now):
        det = [p for p in self.detected if p[1] >= win_start and p[0] <= w_end]
        if det:
            return [(s, e, e - s >= MAJOR_MIN_MS) for s, e in sorted(det)]
        if not self.seed:
            return []
        return [(s, e, e - s >= MAJOR_MIN_MS) for s, e in self.known_sleeps(now) if e >= win_start and s <= w_end]

    def run(self, batch_start, batch_end, last_event, now, win_start=None):
        """One detection for a batch; returns (window_start, previous, periods)."""
        if win_start is None:
            win_start = self.window_start(batch_start, batch_end, now)
        a = assemble(self.ev, win_start, batch_end, last_event)
        if a is None:
            return win_start, [], None
        prev = self.previous(win_start, batch_end, now)
        res = run_detector(a, DEFAULT_AVG_TEMP_CENTI, prev, self.native)
        return win_start, prev, res

    def sync(self, t_prev, t_now):
        """A ring sync that delivered the events in (t_prev, t_now]; stores the output when seeded."""
        ev = self.ev
        mt = [t for t, _ in ev["motion"] if t_prev < t <= t_now] + [t for t, _ in ev["temp"] if t_prev < t <= t_now]
        if not mt:
            return None
        last = max(mt + [t for t, *_ in ev["state"] + ev["session"] if t_prev < t <= t_now])
        win_start, prev, res = self.run(min(mt), max(mt), last, t_now)
        if res and self.seed:
            self.detected += [(s, e) for s, e, _ in res]
        return win_start, prev, res


def local(ms, tz):
    return datetime.datetime.fromtimestamp(ms / 1000 + tz * 3600, datetime.timezone.utc).strftime("%m-%d %H:%M:%S")


def parse_local(text, tz):
    fmt = "%Y-%m-%d %H:%M:%S" if text.count(":") == 2 else "%Y-%m-%d %H:%M"
    return int(datetime.datetime.strptime(text, fmt).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000 - tz * H_MS)


def overlap_best(res, s, e):
    """The detected period that overlaps [s, e] the most."""
    best = max(res or [], key=lambda p: min(p[1], e) - max(p[0], s), default=None)
    if best is None or min(best[1], e) <= max(best[0], s):
        return None
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("db", nargs="?")
    ap.add_argument("--tz", type=float, default=0, help="hours east of UTC for input and output times")
    ap.add_argument("--clock", choices=("ring", "capture", "raw"), default="ring",
                    help="ring: latest time_sync at or before the event in ring time (default); "
                         "capture: latest time_sync captured at or before the event's capture (app_clock.py); "
                         "raw: one constant offset for every event, so detector and ring period share a clock")
    ap.add_argument("--raw-offset", type=float, default=0.0, help="seconds added to every event with --clock raw")
    ap.add_argument("--nights", action="store_true", help="every bedtime_period against the detector run a lag after its end")
    ap.add_argument("--lags", default="10,20,40,60", help="minutes of events after the ring's bedtime end (--nights)")
    ap.add_argument("--no-seed", action="store_true", help="no confirmed sleeps seed the window or the previous list")
    ap.add_argument("--partial", help="local time; sweep the newest event from this time (with --to and --step)")
    ap.add_argument("--to", help="end of the --partial sweep (default 20 minutes after --partial)")
    ap.add_argument("--step", type=float, default=1.0, help="minutes between --partial rows")
    ap.add_argument("--until", help="local time of the newest event of one detection")
    ap.add_argument("--window", default="seeded", help="window start: 'seeded' (confirmed sleeps seed the start), '28d', or a number of hours")
    a = ap.parse_args()
    db = resolve_db(a.db, REPO)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    import events_io as RS
    rows, epochs, unix_s = RS.load_events(con)
    unix_ms = RS.make_unix_ms(con, unix_s, "app")
    if a.clock == "raw":
        unix_ms = lambda ds, cu=None, anchor_ds=None: int(1_790_933_200_000 + a.raw_offset * 1000) + ds * 100  # noqa: E731
    elif a.clock == "ring":
        unix_ms = RingOrderClock(con, unix_ms)
    else:
        capture_ms = unix_ms
        unix_ms = lambda ds, cu=None, anchor_ds=None: capture_ms(ds, cu)  # noqa: E731
    ev = load_events(con, unix_ms)
    beds = load_bedtimes(con, unix_ms)
    app = App(ev, beds, seed=not a.no_seed, native=getattr(a, "native", False))
    tz = a.tz

    def window_arg(end):
        if a.window in ("seeded", "28d"):
            return None if a.window == "seeded" else end - LOOKBACK_MAX_MS
        return end - int(float(a.window) * H_MS)

    if a.nights:
        print("ring bedtime_period                  lag  window start        detected (major)                 start diff  end diff (s)")
        for lag in (float(x) for x in a.lags.split(",")):
            for s, e, av in beds:
                end = e + int(lag * 60_000)
                if app.seed:
                    app.sleeps = [b for b in beds if b[1] <= s - 60_000 and b[2] <= end]
                else:
                    app.sleeps = []
                win_start, prev, res = app.run(end - 30 * 60_000, end, end, end, window_arg(end))
                best = overlap_best(res, s, e)
                ring = f"{local(s, tz)} - {local(e, tz)[6:]}"
                if best is None:
                    print(f"{ring}  {lag:4.0f}  {local(win_start, tz)}  none")
                    continue
                print(f"{ring}  {lag:4.0f}  {local(win_start, tz)}  {local(best[0], tz)} - {local(best[1], tz)[6:]} ({best[2]})"
                      f"  {(best[0] - s) / 1000:+8.0f} {(best[1] - e) / 1000:+8.0f}")
        return
    if a.partial:
        t0 = parse_local(a.partial, tz)
        t1 = parse_local(a.to, tz) if a.to else t0 + 20 * 60_000
        times = sorted({t for t, _ in ev["motion"] + ev["temp"] if t0 <= t <= t1})
        step = int(a.step * 60_000)
        last_shown = None
        print("newest event        window start         detected                            end - newest (min)")
        for t in times:
            if last_shown is not None and t - last_shown < step:
                continue
            last_shown = t
            win_start, prev, res = app.run(t - 30 * 60_000, t, t, t, window_arg(t))
            res = res or []
            main_p = max(res, key=lambda p: p[1], default=None)
            if main_p is None:
                print(f"{local(t, tz)}  {local(win_start, tz)}  none")
            else:
                print(f"{local(t, tz)}  {local(win_start, tz)}  {local(main_p[0], tz)} - {local(main_p[1], tz)[6:]} ({main_p[2]})"
                      f"   {(t - main_p[1]) / 60000:5.2f}")
        return
    end = parse_local(a.until, tz) if a.until else max(t for t, _ in ev["motion"] + ev["temp"])
    win_start, prev, res = app.run(end - 30 * 60_000, end, end, end, window_arg(end))
    if res is None:
        sys.exit("no detection: no motion or temperature event in the window, or no resting-HR period")
    print(f"window {local(win_start, tz)} - {local(end, tz)}, {len(prev)} previous bedtimes")
    for s, e, m in res:
        print(f"{local(s, tz)} - {local(e, tz)}  {(e - s) / 3600_000:.2f} h  major={m}")


if __name__ == "__main__":
    main()
