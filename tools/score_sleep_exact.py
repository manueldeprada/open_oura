#!/usr/bin/env python3
"""Sleep score from ring data.

    python tools/score_sleep_exact.py [DB] [--start DS --end DS] [--tz H]
                                      [--chronotype NAME] [--ideal-midpoint HH:MM]

The hypnogram comes from an earlier stage model (`tools/run_sleep_model.py`), the 30 s motion bytes from
the ring's `motion_event` stream (resampled by `sleep_summary.night_inputs`), then `sleep_score_inputs.night_to_score` derives the
score arguments.

Timing. The timing contributor compares the night's sleep midpoint with an ideal midpoint
given by a chronotype (`chronotype` -3..3) and `ideal_midpoint_of_sleep` (seconds from midnight).
The ring does not hold them. Without them the timing is scored on a fixed reference curve.
Set the two values (chronotype and ideal sleep midpoint) in profile.json:

    "chronotype": {"type": "late morning", "ideal_midpoint": "03:30"}

or pass the flags above. Types: early morning, morning, late morning, early evening,
evening, late evening.
"""
import argparse
import os
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sleep_score_inputs as SI
import sleep_summary as SS
from _common import resolve_db
from epoch_time import build_epochs, make_unix_s

REPO = Path(__file__).resolve().parent.parent
CHRONOTYPES = {"early morning": 1, "morning": 2, "late morning": 3,
               "early evening": 4, "evening": 5, "late evening": 6}
LABELS = ("deep sleep total", "deep", "REM", "efficiency", "latency", "restfulness", "timing")
LABELS = ("total sleep", "deep sleep", "REM sleep", "efficiency", "latency", "restfulness", "timing")


def parse_midpoint(text):
    """'HH:MM' (24 h, after-midnight hours positive; 22:30 or -01:30 before midnight) or seconds."""
    if ":" in str(text):
        neg = str(text).startswith("-")
        h, m = str(text).lstrip("-").split(":")
        s = int(h) * 3600 + int(m) * 60
        if neg or s >= 12 * 3600:
            s = -s if neg else s - 86400
        return s
    return int(text)


def chronotype_from(args, profile):
    name, mid = args.chronotype, args.ideal_midpoint
    prof = profile.get("chronotype")
    if prof:
        name, mid = name or prof.get("type"), mid or prof.get("ideal_midpoint")
    if name is None or mid is None:
        return None
    code = CHRONOTYPES.get(str(name).lower()) if not str(name).isdigit() else int(name)
    if code is None:
        sys.exit(f"unknown chronotype {name!r}; use one of {', '.join(CHRONOTYPES)}")
    return code, parse_midpoint(mid)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("db", nargs="?")
    ap.add_argument("--start", type=int, help="bedtime start (deciseconds)")
    ap.add_argument("--end", type=int, help="bedtime end (deciseconds)")
    ap.add_argument("--tz", type=float, default=1, help="hours east of UTC")
    ap.add_argument("--chronotype")
    ap.add_argument("--ideal-midpoint")
    ap.add_argument("--profile", help="profile.json (age, chronotype); default: the repo's")
    ap.add_argument("--json", action="store_true", help="print total, contributors and scan fields as JSON")
    args = ap.parse_args()
    db = resolve_db(args.db, REPO)
    pf = Path(args.profile) if args.profile else REPO / "profile.json"
    profile = json.loads(pf.read_text()) if pf.exists() else {}
    age = int(profile.get("age", 35))
    chrono = chronotype_from(args, profile)

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    if args.start is None:
        row = con.execute("SELECT decoded_json, captured_unix FROM events WHERE tag=118 "
                          "ORDER BY captured_unix DESC, id DESC").fetchone()
        if row is None:
            sys.exit("no bedtime_period in the DB; pass --start and --end")
        b = json.loads(row[0])
        args.start, args.end, cu = b["bedtime_start_ds"], b["bedtime_end_ds"], row[1]
    else:
        cu = None
    script = "run_sleep_model.py"
    env = dict(os.environ)
    run = subprocess.run([sys.executable, str(REPO / "tools" / script), str(args.start), str(args.end),
                          str(db), str(args.tz), "--json"], capture_output=True, text=True, env=env)
    if run.returncode != 0 or not run.stdout.strip().startswith("{"):
        sys.exit(run.stdout.strip() or run.stderr.strip()[-400:])
    night = json.loads(run.stdout)
    stages = night["stages"]
    cu = night.get("captured_unix") or cu

    rows = con.execute("SELECT ring_timestamp, tag, decoded_json, captured_unix FROM events "
                       "WHERE decoded_json IS NOT NULL ORDER BY ring_timestamp").fetchall()
    unix_s = make_unix_s(build_epochs(rows))
    clock = None

    ev = con.execute("SELECT ring_timestamp FROM events WHERE tag = 118 AND json_extract(decoded_json, '$.bedtime_start_ds') = ? "
                     "ORDER BY id DESC", (args.start,)).fetchone()

    def to_ms(ds):
        # the period's start and end convert with the sync in force at the bedtime_period event's ring time
        t = (clock.ring_order(ds, ev[0]) if ev else None) if clock else None
        if t is None and clock:
            t = clock.unix_s(ds, cu)
        return int(round((t if t is not None else unix_s(ds, cu)) * 1000))

    start_ms, end_ms = to_ms(args.start), to_ms(args.end)
    bed_start, bed_end = start_ms // 1000, end_ms // 1000

    packed = bytearray((len(stages) + 1) // 2)
    for i, s in enumerate(stages):
        packed[i // 2] |= (s & 0xF) << (4 * (i % 2))
    n = SI.epoch_count(bed_start, bed_end)
    # the 30 s motion series resampled from the ring's events (sleep_summary.night_inputs)
    series = SS.load_ring_series(con, clock, unix_s)
    motion = bytes(SS.night_inputs(series, start_ms, end_ms)[0])

    def score(chrono_value):
        return SI.night_to_score(age, bed_start, bed_end, int(args.tz * 60), bytes(packed), bed_start * 1000,
                                 bytes(motion), bed_start * 1000, chrono_value)

    res = score(chrono)
    if res is None:
        sys.exit("fewer than 10 epochs: no score")
    sc = res["scan"]
    if args.json:
        import dataclasses
        ds = SI.app_sleep_score(sc, age, bed_start, int(args.tz * 60), chrono)
        print(json.dumps(dict(total=ds.score, contributors=list(ds.contributors), night_total=res["total"], args=list(res["args"]),
                              scan=dataclasses.asdict(sc), epochs=n, bed_start=bed_start, bed_end=bed_end, stages=stages)))
        return
    import datetime
    local = lambda t: datetime.datetime.utcfromtimestamp(t + int(args.tz * 3600)).strftime("%a %H:%M")
    ds = SI.app_sleep_score(sc, age, bed_start, int(args.tz * 60), chrono)
    print(f"night {local(bed_start)} to {local(bed_end)}, age {age}, {n} epochs")
    print(f"sleep score {ds.score}   (per-night path: {res['total']})")
    for name, v in zip(LABELS, ds.contributors):
        print(f"  {name:13} {v}")
    if chrono is None:
        print("\nNo chronotype set: timing uses the default without a chronotype (type 0, midpoint 00:00).")
        print("Timing for ideal midpoints (set the one for your chronotype):")
        for hh in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
            alt = SI.app_sleep_score(sc, age, bed_start, int(args.tz * 60), (3, int(hh * 3600)))
            print(f"  ideal midpoint {int(hh):02d}:{int(hh % 1 * 60):02d}  timing {alt.contributors[6]}  score {alt.score}")


if __name__ == "__main__":
    main()
