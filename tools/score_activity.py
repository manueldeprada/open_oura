#!/usr/bin/env python3
"""Daily activity statistics and the activity score from the ring events in oura.db.

    python tools/score_activity.py [DB] [--tz H] [--day-tz DATE=H ...] [--days N] [--readiness P] [--weight KG]
        [--until ISO] [--clock window|stream|epoch] [--stateless 0|1]

Per activity day (04:00 to 04:00 local, in the zone the day was stored with) the day's MET array is built
and fed to `actinfo.ActInfo`, which yields class minutes, steps, calories and the day record that
`activity_score` combines with the previous nine days (see docs/algorithms/metrics.md). Sex, age and weight come from profile.json. Sleep periods come from
`bedtime_period`, inactivity alerts from `alert_event` type 1. The target scaling uses `--readiness`
(default 80). Rust port: `ported::activity_day::compute_stateless`.
"""
import argparse
import bisect
import datetime
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import activity_score as AS
import actinfo as AI
from _common import resolve_db
from epoch_time import build_epochs, make_unix_s

REPO = Path(__file__).resolve().parent.parent
DAY_START_HOUR = 4
DAY_END_MIN = 1440
OFF_FINGER_MIN_S = 600
ALL_PERIODS = True  # the stateless path rests every sleep period overlapping the day
STATELESS_ZERO_AS_REST = True   # a 0 byte counts as 0.9 MET
SEED_PREVIOUS_DAY = True  # hourly sedentary counters start from the previous day
SLEEP_PICK = "longest"  # which of the day's bedtime periods actinfo gets




LAST_BUFS = {}  # local date -> per-minute buffer of the last compute() call


def pad_days(age, gender):
    """Fill the nine-day history before the first ring days with two placeholder days,
    A (10000 m, 2 alerts) and B (5000 m, 9 alerts), repeating A, A, B back from the newest."""
    A = AS.Day(eq_distance=10000, eq_target=10000, sedentary_min=480, moderate_met_min=250, vigorous_met_min=150,
               sit_alerts=2, age=age, gender=gender, moderate_min=120, vigorous_min=30, target_reached=0)
    B = AS.Day(eq_distance=5000, eq_target=10000, sedentary_min=999, moderate_met_min=30, vigorous_met_min=15,
               sit_alerts=9, age=age, gender=gender, moderate_min=30, vigorous_min=10, target_reached=0)
    return [B if k % 3 == 2 else A for k in range(8, -1, -1)]




def stream_clock(con, fallback):
    """The stored event time: the latest ring-written `time_sync` at or before the event in ring order,
    `unix + (ds - sync_ds) / 10`. `fallback` serves
    events before the first sync."""
    import app_clock
    ac = app_clock.AppClock(con)

    def unix_s(ds, captured_unix=None):
        t = ac.ring_order(ds, ds)
        return fallback(ds, captured_unix) if t is None else t
    return unix_s


FINGER_STATES = (3, 4, 5, 6)  # wear_event states that mean "on the finger"; all others are non-wear


def app_assemble(a, day, d0, t_latest):
    """The day's 1440-byte array, fed to `a`: bytes start at 9 (0.9 MET), each event overwrites its minutes (in time order), non-wear
    intervals from the wear events write 1 over them, then `a` counts the minutes up to the newest event
    (a completed day: all 1440, minute 1439 only as a non-wear flag)."""
    buf, steps = [9] * DAY_END_MIN, 0
    for end, st, data in sorted(day["ev"], key=lambda e: e[0]):
        steps += st
        for j, v in enumerate(data):
            i = end - len(data) + j
            if 0 <= i < DAY_END_MIN:
                buf[i] = v
    marks = list(day["marks"])
    if day.get("open_wear"):
        t_open, last_wear = day["open_wear"]
        t_end = max(t_latest, min(last_wear or t_open, d0 + 86400))
        marks.append((int((t_open - d0) // 60), int((t_end - d0) // 60) - 1))
    for lo, hi in marks:
        for i in range(max(lo, 0), min(hi, DAY_END_MIN - 1) + 1):
            buf[i] = 1
    if d0 + 86400 <= t_latest:
        a.add_activity(DAY_END_MIN - 1, steps, buf[:DAY_END_MIN - 1])
        a.complete_day(buf[DAY_END_MIN - 1])
    elif d0 <= t_latest:
        n = min(int((t_latest - d0) // 60), DAY_END_MIN - 1)
        a.add_activity(n, steps, buf[:n])


def ts_order_clock(ac, batch):
    """Time of an event in a parsed batch. The ring time_sync events that bracket the batch by
    stored time (latest at or before the first event's stored time, earliest at or after the last one's)
    are fetched, and each
    event takes the latest of them whose own time is at or before the event's stored time: `unix + (ds - sync_ds) / 10`.
    The stored time is the stream-order time (`stream_clock`). Where the ring clock has drifted, an event just
    before a sync in ring order sits after it in stored time and is dated from that sync."""
    stored = [e[3] for e in batch]
    lo, hi = min(stored), max(stored)
    syncs = sorted((u, ds) for ds, u, ring in ac.all_syncs if ring)  # (unix_time, ring ds)
    unix = [u for u, _ in syncs]
    i = max(bisect.bisect_right(unix, lo) - 1, 0)
    j = min(bisect.bisect_left(unix, hi), len(syncs) - 1)
    usable = syncs[i:j + 1]
    keys = [u for u, _ in usable]

    def convert(ds, stored_t):
        k = bisect.bisect_right(keys, stored_t) - 1
        if k < 0:
            return None
        u, sds = usable[k]
        return u + (ds - sds) / 10.0
    return convert


def _window_events(con, days, unix_s, until, shift_s):
    """Each day's events and non-wear minutes.

    The raw activity, wear and alert events with a stored time in [day start, day start + 24 h + 15 min] form
    one batch, parsed with the anchors `app_clock.window(first_ds, last_ds)`; the latest wear event before the
    day start joins it. Activity events are placed by absolute minute, wear events mark non-wear (byte 1, over
    the minutes `floor(start)` to `floor(end) - 1`, whatever MET they replace; no minimum duration) from the first non-finger
    state to the next finger state, and an open interval runs to the newest event."""
    import app_clock
    ac = app_clock.AppClock(con)
    raw = [(ds, tag, body, unix_s(ds, cu) + shift_s) for ds, tag, body, cu in con.execute(
        "SELECT ring_timestamp, tag, body, captured_unix FROM events WHERE tag IN (80, 83, 86) ORDER BY ring_timestamp")]
    state = {ds: json.loads(j).get("state") for ds, j in con.execute(
        "SELECT ring_timestamp, decoded_json FROM events WHERE tag = 83 AND decoded_json IS NOT NULL")}
    wear_all = [e for e in raw if e[1] == 83]
    for d0, d in days.items():
        batch = [e for e in raw if d0 <= e[3] <= d0 + 86400 + 900 and (until is None or e[3] <= until)]
        d["ev"], d["marks"] = [], []
        if not batch:
            continue
        conv = ts_order_clock(ac, batch)
        when = lambda e: e[3] if conv(e[0], e[3]) is None else conv(e[0], e[3]) + shift_s
        before = [e for e in wear_all if e[3] < d0]
        t_open = d0 if before and state.get(before[-1][0]) not in FINGER_STATES else None
        last_wear = None

        def mark(a, b):
            lo, hi = int((a - d0) // 60), int((b - d0) // 60) - 1
            d["marks"].append((lo, hi))
        for e in batch:
            t = when(e)
            if e[1] == 80:
                d["ev"].append((int((t - d0) // 60), e[2][0], list(e[2][1:])))
            elif e[1] == 83:
                last_wear = t
                if state.get(e[0]) in FINGER_STATES:
                    if t_open is not None and t_open < t:
                        mark(t_open, t)
                    t_open = None
                elif t_open is None:
                    t_open = t
        d["open_wear"] = (t_open, last_wear) if t_open is not None else None


def load_profile(path=None):
    p = Path(path) if path else REPO / "profile.json"
    prof = json.loads(p.read_text()) if p.exists() else {}
    # Gender codes: 2 male, 1 female.
    return (int(prof.get("age", 35)), 2 if prof.get("sex", "M") == "M" else 1, int(round(prof.get("weight_kg", 75))))


def compute(args):
    """With `args.day_tz` (date -> hours east of UTC) days are built in their own zone, see `compute_mixed`.
    Per activity day of the events DB: (list of (day_start, Stats), day records, first day start,
    UTC offset in seconds, age, gender code). `args` carries db, tz, readiness, profile, weight,
    no_pad (see `main`)."""
    con = sqlite3.connect(f"file:{resolve_db(args.db, REPO)}?mode=ro", uri=True)

    rows = con.execute("SELECT ring_timestamp, tag, decoded_json, captured_unix FROM events "
                       "WHERE decoded_json IS NOT NULL ORDER BY ring_timestamp").fetchall()
    unix_s = make_unix_s(build_epochs(rows))
    if getattr(args, "clock", "window") in ("stream", "window"):
        unix_s = stream_clock(con, unix_s)
    off = int(args.tz * 3600)
    until = getattr(args, "until", None)  # epoch seconds; events after it are ignored

    def local_day_start(t):
        """Unix seconds of the 04:00 local boundary at or before t."""
        s = t + off - DAY_START_HOUR * 3600
        return (s // 86400) * 86400 + DAY_START_HOUR * 3600 - off

    spill = []
    days = {}   # day_start -> {"ev": [(minute_end, steps, bytes)], "alerts": n, "sleep": [(s_min, e_min)]}
    for ds, body, cu in con.execute("SELECT ring_timestamp, body, captured_unix FROM events WHERE name='activity_information' ORDER BY ring_timestamp"):
        t = unix_s(ds, cu) + getattr(args, "shift_s", 0)
        if until is not None and t > until:
            continue
        d0 = local_day_start(t)
        minute = int((t - d0) // 60)
        days.setdefault(d0, {"ev": [], "alerts": 0, "sleep": []})["ev"].append((minute, body[0], list(body[1:])))
        if minute < len(body) - 1:  # the event also covers the last minutes of the previous day
            spill.append((d0 - 86400, (minute + DAY_END_MIN, 0, list(body[1:]))))
    for d0, e in spill:
        if d0 in days:
            days[d0]["ev"].append(e)
    if getattr(args, "clock", "window") == "window":
        _window_events(con, days, unix_s, until, getattr(args, "shift_s", 0))
    # Non-wear: wear_event state 8 ("chg. detected", on the charger) until the next state 3, and
    # state 1 until state 3 when it lasts at least OFF_FINGER_MIN_S (shorter ones are not counted).
    charge_from = None
    for ds, j, cu in con.execute("SELECT ring_timestamp, decoded_json, captured_unix FROM events WHERE name='wear_event' "
                                 "AND decoded_json IS NOT NULL ORDER BY ring_timestamp"):
        st_, t = json.loads(j).get("state"), unix_s(ds, cu)
        if st_ in (1, 8) and charge_from is None:
            charge_from = (t, st_)
        elif st_ == 3 and charge_from is not None:
            (t0, kind), charge_from = charge_from, None
            if until is not None:
                if t0 > until:
                    continue
                t = min(t, until)
            if kind == 1 and t - t0 < OFF_FINGER_MIN_S:
                continue
            for d0 in {local_day_start(t0), local_day_start(t)}:
                if d0 in days:
                    lo, hi = max(t0, d0), min(t, d0 + 86400)
                    if hi > lo:
                        days[d0].setdefault("charging", []).append((int((lo - d0) // 60), int((hi - d0) // 60)))
    for ds, j, cu in con.execute("SELECT ring_timestamp, decoded_json, captured_unix FROM events WHERE name='alert_event' AND decoded_json IS NOT NULL"):
        if json.loads(j).get("alert_type") == 1:
            d0 = local_day_start(unix_s(ds, cu))
            if d0 in days:
                days[d0]["alerts"] += 1
    for ds, j, cu in con.execute("SELECT ring_timestamp, decoded_json, captured_unix FROM events WHERE name='bedtime_period' AND decoded_json IS NOT NULL"):
        b = json.loads(j)
        s, e = int(unix_s(b["bedtime_start_ds"], cu)), int(unix_s(b["bedtime_end_ds"], cu))
        for d0 in (local_day_start(s), local_day_start(e)):
            if d0 in days and (s_e := (AI.sleep_minutes(s, d0), AI.sleep_minutes(e, d0))) not in days[d0]["sleep"]:
                days[d0]["sleep"].append(s_e)

    readiness_by_day = dict(getattr(args, "readiness_map", None) or {})
    age, gender, kg = load_profile(args.profile)
    kg = args.weight or kg
    user = AI.User(bmr=AI.bmr(age, kg, gender), age=age, gender=gender, weight_kg=kg)
    first_day = min(days) if days else 0
    last_day = max(days) if days else 0
    records, stats = [], []
    alt_prev = dict(getattr(args, "seed_bufs", None) or {})
    t_latest = unix_s(rows[-1][0], rows[-1][3]) if rows else 0
    if until is not None:
        t_latest = min(t_latest, until)  # newest event of any kind: the MET count ends there
    prev_buf = {}
    for d0 in sorted(days):
        a = AI.ActInfo(user)
        a.stateless = bool(getattr(args, "stateless", STATELESS_ZERO_AS_REST))
        date = datetime.datetime.fromtimestamp(d0 + off, datetime.timezone.utc).strftime("%Y-%m-%d")
        prev_date = (datetime.date.fromisoformat(date) - datetime.timedelta(days=1)).isoformat()
        if SEED_PREVIOUS_DAY:
            pb = alt_prev.get(prev_date) if alt_prev else prev_buf.get(d0 - 86400)
            if pb is not None:
                a.seed_hourly(pb)
        a.set_target(readiness_by_day.get(datetime.datetime.fromtimestamp(d0 + off, datetime.timezone.utc).strftime("%Y-%m-%d"), args.readiness))
        sleeps = [s for s in days[d0]["sleep"] if s[0] >= 0 or s[1] >= 0]
        if ALL_PERIODS:
            a.ranges = list(days[d0]["sleep"])
        if sleeps:
            a.sleep = {"longest": max(sleeps, key=lambda s: s[1] - s[0]), "last": max(sleeps, key=lambda s: s[1]),
                       "first": min(sleeps, key=lambda s: s[0])}[SLEEP_PICK]
        app_day = "marks" in days[d0]
        if app_day:
            app_assemble(a, days[d0], d0, t_latest)
        else:
            for minute, steps, data in days[d0]["ev"]:
                a.add_minutes(minute, steps, data)
        for _ in range(days[d0]["alerts"]):
            a.add_sit_alert()
        if app_day:
            pass
        elif d0 + 86400 <= t_latest and a.last < DAY_END_MIN:
            a.complete_day(a.buf[DAY_END_MIN - 1])  # a completed day covers all 1440 minutes (zero-filled where events are missing)
        elif d0 <= t_latest < d0 + 86400:
            # Partial day: count the minutes up to the newest event and fill minutes without
            # events with 0.9 MET (byte 9).
            n = min(int((t_latest - d0) // 60), DAY_END_MIN - 1)
            for i in range(a.last, n):
                a.buf[i] = max(a.buf[i], 9)
            a.last = max(a.last, n)
            a._advance_hours()
        for lo, hi in ([] if app_day else days[d0].get("charging", [])):
            a.add_non_wear(hi, hi - lo)
        if a.last >= DAY_END_MIN:
            a.absorb_tail_non_wear()
        prev_buf[d0] = list(a.buf)
        LAST_BUFS[date] = list(a.buf)
        st = a.statistics()
        records.append(st.day)
        stats.append((d0, st))

    return stats, records, first_day, off, age, gender


def compute_mixed(args):
    """Days in the zone they were stored with. Most
    days use the phone zone (`args.tz`) but some days can be stored at another zone, for example UTC-5
    (zone byte `f6` of the ring's time_sync events), whose 04:00 day start is 09:00 UTC. `args.day_tz` maps
    "YYYY-MM-DD" to hours east of UTC for those days. Each day's array uses its own zone and the hourly
    counters are seeded from the previous date's array in that date's zone."""
    import copy
    day_tz = dict(getattr(args, "day_tz", None) or {})
    if not day_tz:
        return compute(args)
    zones = sorted({args.tz, *day_tz.values()})
    zone_of = lambda date: day_tz.get(date, args.tz)
    bufs = {}
    for z in zones:
        a = copy.copy(args)
        a.tz, a.day_tz, a.seed_bufs, a.prev_tz = z, None, None, None
        LAST_BUFS.clear()
        compute(a)
        bufs[z] = dict(LAST_BUFS)
    chosen = {d: b for z in zones for d, b in bufs[z].items() if zone_of(d) == z}
    rows, meta = [], None
    for z in zones:
        a = copy.copy(args)
        a.tz, a.day_tz, a.seed_bufs, a.prev_tz = z, None, chosen, None
        stats, records, _, off, age, gender = compute(a)
        for (d0, st), rec in zip(stats, records):
            date = datetime.datetime.fromtimestamp(d0 + off, datetime.timezone.utc).strftime("%Y-%m-%d")
            if zone_of(date) == z:
                rows.append((date, st, rec))
        meta = meta or (age, gender)
    rows.sort(key=lambda r: r[0])
    LAST_BUFS.clear()
    LAST_BUFS.update(chosen)
    base = int(args.tz * 3600)

    def norm(date):  # the day start in the base zone, so callers can keep one frame
        return int(datetime.datetime.fromisoformat(date).replace(tzinfo=datetime.timezone.utc).timestamp()) + DAY_START_HOUR * 3600 - base

    stats = [(norm(d), st) for d, st, _ in rows]
    return stats, [r for _, _, r in rows], stats[0][0], base, meta[0], meta[1]


def score_days(stats, records, first_day, off, age, gender, no_pad=False):
    """[(date, Stats, score, contributors)] per day: the day and the nine records before it (two
    placeholder days fill the history before the first ring day), ring day 1 on the first."""
    out = []
    for i, (d0, st) in enumerate(stats):
        ring_day = (d0 - first_day) // 86400 + 1
        window = ((records[:i] if no_pad else pad_days(age, gender) + records[:i])[-9:]) + [records[i]]
        score, c, _ = AS.score(window, ring_day)
        out.append((datetime.datetime.fromtimestamp(d0 + off, datetime.timezone.utc).strftime("%Y-%m-%d"), st, score, c))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("db", nargs="?")
    ap.add_argument("--tz", type=float, default=1, help="hours east of UTC")
    ap.add_argument("--days", type=int, default=14, help="most recent days to list")
    ap.add_argument("--readiness", type=int, default=80)
    ap.add_argument("--profile", help="profile.json with age, sex and weight_kg (default: the repo's)")
    ap.add_argument("--no-pad", action="store_true", help="do not pad the history to nine days with placeholder days")
    ap.add_argument("--prev-tz", type=float, help="hours east of UTC used to build the previous day's array that seeds the hourly counters (by default the zone stored on that day)")
    ap.add_argument("--day-tz", action="append", default=[], metavar="DATE=H",
                    help="zone (hours east of UTC) a day was stored with, e.g. 2025-03-09=-5 (repeatable)")
    ap.add_argument("--clock", choices=("window", "stream", "epoch"), default="window",
                    help="window: stored-time anchor bracket, wear intervals, bytes start at 9 (default); stream: stream-order times with the older assembly; epoch: the older nearest-anchor times")
    ap.add_argument("--stateless", type=int, choices=(0, 1), default=int(STATELESS_ZERO_AS_REST),
                    help="1: bytes below 9 other than 1 count as 0.9 MET (the stateless path)")
    ap.add_argument("--shift-s", type=float, default=0, help="seconds added to every event time (checks the anchor offset)")
    ap.add_argument("--until", help="ISO local time (--tz zone); ignore later events")
    ap.add_argument("--weight", type=int, help="override the profile weight (kg)")
    args = ap.parse_args()
    if args.until:
        args.until = int(datetime.datetime.fromisoformat(args.until).replace(tzinfo=datetime.timezone.utc).timestamp() - args.tz * 3600)
    args.day_tz = dict((k, float(v)) for k, v in (x.split('=') for x in args.day_tz))
    stats, records, first_day, off, age, gender = compute_mixed(args)
    names = ("targets", "stay", "hourly", "freq", "volume", "rec.load", "rec.run")
    print(f"{'day':10} {'rest':>5} {'sed':>5} {'light':>5} {'mod':>4} {'vig':>4} {'steps':>6} {'kcal':>5} {'eq m':>6} {'score':>5} {'nw':>4}  " + " ".join(f"{n:>8}" for n in names))
    for day, st, score, c in score_days(stats, records, first_day, off, age, gender, args.no_pad)[-args.days:]:
        print(f"{day:10} {st.classes[0]:5} {st.classes[1]:5} {st.classes[2]:5} {st.classes[3]:4} {st.classes[4]:4} {st.steps:6} {st.active_cal:5} {st.eq_distance:6} {score:5} {st.non_wear:4}  " + " ".join(f"{v:8.1f}" for v in c))

if __name__ == "__main__":
    main()
