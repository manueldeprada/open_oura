#!/usr/bin/env python3
"""Write reference vectors for `ported::activity_day::compute_stateless` from the Python reference
(tools/score_activity.py) on synthetic ring events.

    python tools/gen_public_vectors_activity_day.py

Each scenario draws a user, a phone time zone, optional per-day zones, a ring clock with drift and
time-sync anchors, per-minute MET series cut into `activity_information` events, charging gaps with
`wear_event`s, inactivity `alert_event`s and sleep periods. The events go into a temporary SQLite
database in the layout `score_activity.py` reads, which computes the expected per-day statistics and
scores. Output: crates/oura-analysis/src/ported/vectors/activity_day.rs. Seeds are fixed.
"""
import argparse
import datetime
import json
import random
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import actinfo as AI
import score_activity as SA
from public_vectors_common import write

SEED = 20261018 + 100

# tz: phone zone (hours east of UTC); day_tz: zone a day was stored with, by index of the day; gender: code
# (2 male, 1 female); drift: ring counter rate error; cut_h: hours after local midnight of the last
# day at which the events end; spacing: minutes between activity events.
SCENARIOS = [
    dict(tz=3.0, day_tz={}, age=34, gender=1, kg=62, days=3, drift=1e-4, cut_h=14.5, spacing=10),
    dict(tz=1.0, day_tz={1: -5.0}, age=47, gender=2, kg=88, days=3, drift=-2e-4, cut_h=9.0, spacing=10),
    dict(tz=5.5, day_tz={}, age=22, gender=2, kg=70, days=3, drift=0.0, cut_h=26.5, spacing=15),
    dict(tz=0.0, day_tz={0: 3.0}, age=61, gender=1, kg=55, days=3, drift=2e-3, cut_h=19.0, spacing=10),
]


def met_minutes(r, start_min, n, tz):
    """n minutes of MET bytes starting at absolute minute `start_min`, shaped by the local hour: low
    values at night, desk, walking and workout stretches by day, rare zero bytes."""
    out = []
    while len(out) < n:
        m = start_min + len(out)
        hour = ((m * 60 + int(tz * 3600)) // 3600) % 24
        if hour >= 23 or hour < 6:
            kind = "sleep"
        else:
            kind = r.choices(["desk", "walk", "work", "idle", "zero"], [6, 3, 1, 2, 1])[0]
        ln = r.randint(8, 70)
        lo, hi = {"sleep": (8, 10), "desk": (10, 16), "walk": (18, 40), "work": (50, 120),
                  "idle": (0, 12), "zero": (0, 0)}[kind]
        while ln > 0 and len(out) < n:
            rep = min(ln, r.randint(2, 9))
            v = r.randint(lo, hi)
            if kind == "work" and r.random() < 0.15:
                v = r.randint(128, 150)  # values above 127 encode 12.8 + 0.2 (b - 128) MET
            out += [v] * min(rep, n - len(out))
            ln -= rep
    return out


def build_scenario(sc, seed):
    r = random.Random(seed)
    tz = sc["tz"]
    day0 = datetime.date(2025, 3, 2) + datetime.timedelta(days=r.randint(0, 30))
    midnight = int(datetime.datetime(day0.year, day0.month, day0.day, tzinfo=datetime.timezone.utc).timestamp()
                   - tz * 3600)
    t_first = midnight + int((4 + r.random() * 5) * 3600)  # the first event, local 04:00 to 09:00
    t_last = midnight + int(sc["cut_h"] * 3600) + 86400 * (sc["days"] - 1)
    ds0 = r.randint(200_000, 5_000_000)
    drift = sc["drift"]

    def ds_of(t):
        return ds0 + int(round((t - t_first) * 10 * (1 + drift)))

    # time-sync anchors every 5 to 9 hours; the ring's own unix value is off by up to 2 s
    anchors, t = [], t_first - 60
    while t < t_last + 3600:
        anchors.append((ds_of(t), t + r.randint(-2, 2)))
        t += r.randint(5, 9) * 3600
    # activity events
    spacing = sc["spacing"] * 60
    events, prev_min, t = [], None, t_first
    first_min = t_first // 60
    mins = met_minutes(r, first_min - 12, (t_last - t_first) // 60 + 40, tz)  # minute m at index m - (first_min - 12)
    base_min = first_min - 12
    charging = []  # (start, end) unix
    for _ in range(sc["days"]):
        if r.random() < 0.8:
            a = midnight + r.randint(0, 86400 * sc["days"])
            charging.append((a, a + r.randint(30, 150) * 60))
    shorts = [(a := midnight + r.randint(0, 86400 * sc["days"]), a + r.randint(2, 20) * 60) for _ in range(2)]
    wear = []
    for a, b in charging:
        wear += [(a, 8), (b, 3)]
    for a, b in shorts:
        wear += [(a, 1), (b, 3)]
    off_windows = sorted(charging + shorts)
    while t <= t_last:
        t_ev = t + r.randint(-20, 20)
        m_ev = t_ev // 60
        if prev_min is not None and not any(a <= t_ev <= b for a, b in off_windows):
            n = m_ev - prev_min
            if 1 <= n <= 30:
                body = mins[prev_min - base_min:m_ev - base_min]
                steps = min(255, sum(r.randint(60, 110) for v in body if 18 <= v < 60))
                events.append((ds_of(t_ev), steps, bytes(body)))
        prev_min = m_ev
        t += spacing
    # the events before the first wear window keep ring order; wear and alert events by time
    wear_ev = [(ds_of(tt), s) for tt, s in sorted(wear) if t_first < tt < t_last]
    alerts = []
    for _ in range(r.randint(3, 4 * sc["days"])):
        tt = r.randint(t_first + 3600, t_last)
        alerts.append((ds_of(tt), r.random() < 0.8))
    alerts.sort()
    newest_ds = max(events[-1][0], wear_ev[-1][0] if wear_ev else 0) + r.randint(0, 2000)
    # sleeps: one night per day (local 22:00 to 07:00 with jitter) and a nap
    sleeps_ds = []
    for k in range(sc["days"] + 1):
        s = midnight + k * 86400 - 2 * 3600 + r.randint(-3000, 3000)
        e = s + 9 * 3600 + r.randint(-3000, 3000)
        if s > t_first and e < t_last:
            sleeps_ds.append((ds_of(s), ds_of(e)))
    if r.random() < 0.7:
        s = midnight + r.randint(0, sc["days"] - 1) * 86400 + 14 * 3600 + 1800
        if t_first < s and s + 2400 < t_last:
            sleeps_ds.append((ds_of(s), ds_of(s + 2400)))
    readiness = {}
    for k in range(sc["days"] + 1):
        d = day0 + datetime.timedelta(days=k)
        if r.random() < 0.75:
            readiness[d.isoformat()] = r.randint(40, 95)
    return dict(anchors=anchors, events=events, wear=wear_ev, alerts=alerts, newest_ds=newest_ds,
                sleeps_ds=sleeps_ds, readiness=readiness)


def reference(sc, data):
    """Run tools/score_activity.py on the scenario's events."""
    tmp = Path(tempfile.mkdtemp())
    db = tmp / "events.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, serial TEXT, tag INTEGER, name TEXT, "
                "ring_timestamp INTEGER, body BLOB, decoded_json TEXT, captured_unix INTEGER)")
    ins = ("INSERT INTO events (serial, tag, name, ring_timestamp, body, decoded_json, captured_unix) "
           "VALUES ('x', ?, ?, ?, ?, ?, ?)")
    for ds, u in data["anchors"]:
        con.execute(ins, (66, "time_sync", ds, b"", json.dumps({"unix_time": u}), u))
    for ds, steps, met in data["events"]:
        con.execute(ins, (80, "activity_information", ds, bytes([steps]) + met, "{}", 0))
    for ds, st in data["wear"]:
        con.execute(ins, (83, "wear_event", ds, b"", json.dumps({"state": st}), 0))
    for ds, sit in data["alerts"]:
        con.execute(ins, (86, "alert_event", ds, b"", json.dumps({"alert_type": 1 if sit else 2}), 0))
    for s, e in data["sleeps_ds"]:
        con.execute(ins, (0x6C, "bedtime_period", e + 10, b"",
                          json.dumps({"bedtime_start_ds": s, "bedtime_end_ds": e}), 0))
    con.execute(ins, (67, "debug_event", data["newest_ds"], b"", "{}", 0))
    con.commit()
    con.close()
    profile = tmp / "profile.json"
    profile.write_text(json.dumps({"age": sc["age"], "sex": "M" if sc["gender"] == 2 else "F", "weight_kg": sc["kg"]}))
    args = argparse.Namespace(db=str(db), tz=sc["tz"], readiness=80, profile=str(profile), weight=sc["kg"],
                              app_log=None, no_pad=False, day_tz=data["day_tz"], activity_log=None, until=None,
                              shift_s=0, clock="window", stateless=1, readiness_map=data["readiness"])
    stats, records, first_day, off, age, gender = SA.compute_mixed(args)
    ref = SA.score_days(stats, records, first_day, off, age, gender)
    # sleep periods as Unix seconds, converted the way the reference converts them (latest sync at or before)
    con = sqlite3.connect(db)
    from app_clock import AppClock
    ac = AppClock(con)
    sleeps = [(int(ac.ring_order(s, s)), int(ac.ring_order(e, e))) for s, e in data["sleeps_ds"]]
    con.close()
    user = AI.User(bmr=AI.bmr(age, sc["kg"], gender), age=age, gender=gender, weight_kg=sc["kg"])
    return ref, sleeps, user


def main():
    out = ["// Reference vectors for `compute_stateless` from tools/score_activity.py on synthetic ring events.",
           "// Generated by tools/gen_public_vectors_activity_day.py; do not edit.",
           "",
           "struct DayExpect {",
           "    date: &'static str,",
           "    classes: [u32; 5],",
           "    steps: u32,",
           "    eq_distance: u32,",
           "    active_cal: u32,",
           "    total_cal: u32,",
           "    non_wear: u32,",
           "    sedentary_min: u32,",
           "    score: u32,",
           "    contributors: [f32; 7],",
           "}",
           "",
           "struct Scenario {",
           "    /// age, gender code, kg, bmr",
           "    user: (u32, u32, u32, u32),",
           "    tz: f64,",
           "    day_tz: &'static [(&'static str, f64)],",
           "    newest_ds: i64,",
           "    anchors: &'static [(i64, i64)],",
           "    sleeps: &'static [(f64, f64)],",
           "    readiness: &'static [(&'static str, u32)],",
           "    wear: &'static [(i64, u8)],",
           "    alerts: &'static [(i64, bool)],",
           "    /// (ring time, steps, MET bytes as hex)",
           "    events: &'static [(i64, u8, &'static str)],",
           "    expect: &'static [DayExpect],",
           "}",
           "",
           f"const SCENARIOS: [Scenario; {len(SCENARIOS)}] = ["]
    for i, sc in enumerate(SCENARIOS):
        data = build_scenario(sc, SEED + i)
        # the zone overrides are given by day index; map them onto the dates the reference finds
        data["day_tz"] = {}
        ref0, _, _ = reference(sc, dict(data, day_tz={}))
        dates = [d for d, *_ in ref0]
        for k, z in sc["day_tz"].items():
            if k < len(dates):
                data["day_tz"][dates[k]] = z
        ref, sleeps, user = reference(sc, data)
        print(f"scenario {i}: {len(data['events'])} events, days {[d for d, *_ in ref]}")
        L = [f"    Scenario {{",
             f"        user: ({user.age}, {user.gender}, {user.weight_kg}, {user.bmr}),",
             f"        tz: {float(sc['tz'])!r},",
             "        day_tz: &[" + ", ".join(f'("{d}", {float(z)!r})' for d, z in data["day_tz"].items()) + "],",
             f"        newest_ds: {data['newest_ds']},",
             "        anchors: &[" + ", ".join(f"({d}, {u})" for d, u in data["anchors"]) + "],",
             "        sleeps: &[" + ", ".join(f"({s}.0, {e}.0)" for s, e in sleeps) + "],",
             "        readiness: &[" + ", ".join(f'("{d}", {v})' for d, v in data["readiness"].items()) + "],",
             "        wear: &[" + ", ".join(f"({d}, {s})" for d, s in data["wear"]) + "],",
             "        alerts: &[" + ", ".join(f"({d}, {str(s).lower()})" for d, s in data["alerts"]) + "],",
             "        events: &["]
        for ds, steps, met in data["events"]:
            L.append(f'            ({ds}, {steps}, "{met.hex()}"),')
        L.append("        ],")
        L.append("        expect: &[")
        for day, st, score, c in ref:
            L.append(f'            DayExpect {{ date: "{day}", classes: {list(st.classes)}, steps: {st.steps}, '
                     f"eq_distance: {st.eq_distance}, active_cal: {st.active_cal}, total_cal: {st.total_cal}, "
                     f"non_wear: {st.non_wear}, sedentary_min: {st.day.sedentary_min}, score: {score}, "
                     f"contributors: [{', '.join(f'{v:.4f}' for v in c)}] }},")
        L.append("        ],")
        L.append("    },")
        out += L
    out.append("];")
    write("activity_day", "\n".join(out) + "\n")


if __name__ == "__main__":
    main()
