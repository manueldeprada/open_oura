#!/usr/bin/env python3
"""Daily activity, sleep and readiness scores from the ring events in oura.db, through the ports in this directory.

    python tools/score_daily_exact.py [DB] [--tz H] [--profile profile.json] [--weight KG]

Per night (bedtime_period): staging by an earlier stage model and the ring's motion -> scan -> SleepPeriodInput ->
the daily sleep output (sleep score). Per activity day: `score_activity` statistics. Readiness:
`daily_outputs_stateless` with the previous day's statistics, 14-day histories indexed by calendar
date and the baseline of `baseline_calc.baseline_calculate` carried from the day before
(`daily_scorer.DailyScorer`).

The night's lowest heart rate and its time are computed by `lowest_hr.py` on the `hrv_event` heart rates. RMSSD (mean of
`hrv_event` RMSSD) and the highest temperature (95th percentile of `sleep_temp_event`) are
approximations of the night's summary values. See docs/algorithms/metrics.md.
"""
import argparse
import datetime
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import activity_score as AS
import daily_scorer as DS
import baseline_calc as BC
import lowest_hr as LH
import readiness_inputs as RI
import readiness_score as R
import score_activity as SA
import sleep_score_inputs as SI
import sleep_summary as SS
from _common import resolve_db
from app_clock import AppClock
from epoch_time import build_epochs, make_unix_s

REPO = Path(__file__).resolve().parent.parent
SRI = DS.SRI


def night_scan(db, start_ds, end_ds, tz, profile=None, cache=None):
    key = f"{start_ds}-{end_ds}-{tz}"
    if cache is not None and key in cache:
        r = dict(cache[key])
        r["scan"] = SI.Scan(**r["scan"])
        return r
    cmd = [sys.executable, str(REPO / "tools/score_sleep_exact.py"), str(db), "--start", str(start_ds),
           "--end", str(end_ds), "--tz", str(tz), "--json"] + (["--profile", profile] if profile else [])
    run = subprocess.run(cmd, capture_output=True, text=True)
    if not run.stdout.startswith("{"):
        return None
    r = json.loads(run.stdout)
    if cache is not None:
        cache[key] = json.loads(run.stdout)
    r["scan"] = SI.Scan(**r["scan"])
    return r




def night_summary(series, start_ms, end_ms, stages):
    """(lowest HR, its time in s from bedtime, average RMSSD, highest temperature in centidegrees, time in bed in s).

    The ring's motion, temperature and HR/HRV events are resampled
    (`sleep_summary.night_inputs`), then these functions are applied: `calculate_lowest_hr` on the 5-minute heart rates (`lowest_hr.py`),
    `calculate_average_rmssd` on the stage string and RMSSD bytes, `nightly_temperature`
    on the 60 s temperatures (`sleep_summary.py`). `stages` are the
    night's per-30 s stages; with None the RMSSD is 0. The sixth value is the 30 s motion series
    read (the scan's restless and got-up counts depend on it)."""
    motion, temps, hrhrv, first = SS.night_inputs(series, start_ms, end_ms)
    hr = [p[0] for p in hrhrv]
    lo = LH.calculate_lowest_hr(hr) if hr else None
    rmssd = 0
    if stages is not None and hr:
        rows = SS.stage_rows(stages, start_ms, start_ms, end_ms)
        rmssd = SS.calculate_average_rmssd(rows, [p[1] for p in hrhrv], len(hrhrv),
                                           SS.slot_offset(first, len(hrhrv), start_ms // 1000))[0]
    return ((lo[0] if lo else 0), (lo[1] if lo else 0), rmssd, SS.nightly_temperature(temps),
            SS.time_in_bed(start_ms, end_ms), motion)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("db", nargs="?")
    ap.add_argument("--tz", type=float, default=1)
    ap.add_argument("--profile")
    ap.add_argument("--weight", type=int)
    ap.add_argument("--readiness", type=int, default=80)
    ap.add_argument("--scan-cache", help="JSON file that keeps the nights' staging results between runs")
    ap.add_argument("--skip-days", help="comma-separated nights (YYYY-MM-DD) to leave out, as days without a record: "
                    "the day counter, the baseline chain and the 14-day histories then skip them")
    ap.add_argument("--sri", type=int, help="sleep-regularity contributor to pass (default: the daily scorer's)")
    ap.add_argument("--dump", help="write each day's derived readiness inputs to this JSON file")
    args = ap.parse_args()
    db = resolve_db(args.db, REPO)
    age, gender, kg = SA.load_profile(args.profile)
    kg = args.weight or kg
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("SELECT ring_timestamp, tag, decoded_json, captured_unix FROM events "
                       "WHERE decoded_json IS NOT NULL ORDER BY ring_timestamp").fetchall()
    unix_s = make_unix_s(build_epochs(rows))
    clock = AppClock(con)
    off = int(args.tz * 3600)
    day_name = lambda t: datetime.datetime.fromtimestamp(t + off, datetime.timezone.utc).strftime("%Y-%m-%d")

    nights = {}
    for ds, cu, j in con.execute("SELECT ring_timestamp, captured_unix, decoded_json FROM events WHERE tag=118 AND decoded_json IS NOT NULL"):
        b = json.loads(j)
        nights[(b["bedtime_start_ds"], b["bedtime_end_ds"])] = (cu, ds)
    cache = json.loads(Path(args.scan_cache).read_text()) if args.scan_cache and Path(args.scan_cache).exists() else None
    if cache is None and args.scan_cache:
        cache = {}
    series = SS.load_ring_series(con, clock, unix_s)
    def to_ms(ds, ev):
        """Bedtime start or end in ms: the latest ring-written sync at or before the `bedtime_period` event's ring time."""
        cu, event_ds = ev
        t = clock.ring_order(ds, event_ds)
        if t is None:
            t = clock.unix_s(ds, cu)
        if t is None:
            t = unix_s(ds, cu)
        return int(round(t * 1000))
    scored = []
    for (s, e), ev in sorted(nights.items(), key=lambda kv: kv[0][0]):
        r = night_scan(db, s, e, args.tz, args.profile, cache)
        if r is None:
            continue
        bs, be = r["bed_start"], r["bed_end"]
        stages = r.get("stages")
        summ = night_summary(series, to_ms(s, ev), to_ms(e, ev), stages)
        if stages:  # the scan again with the motion series (restless and got-up counts)
            packed = bytearray((len(stages) + 1) // 2)
            for i, x in enumerate(stages):
                packed[i // 2] |= (x & 15) << (4 * (i % 2))
            res = SI.night_to_score(age, bs, be, int(args.tz * 60), bytes(packed), bs * 1000, bytes(summ[5]), bs * 1000, None)
            if res is not None:
                r["scan"] = res["scan"]
        scored.append((be, bs, r, summ[:5]))
    if args.scan_cache:
        Path(args.scan_cache).write_text(json.dumps(cache))
    scored.sort(key=lambda x: x[0])

    sac = argparse.Namespace(db=str(db), tz=args.tz, readiness=args.readiness, profile=args.profile, weight=args.weight,
                             app_log=None, no_pad=False, readiness_map={})
    scorer = DS.DailyScorer()
    readiness_map, results = {}, []
    print(f"{'night':10} {'sleep':>5} {'ready':>5} {'activity':>8}   sleep contributors")
    by_night = {}
    for item in scored:
        by_night.setdefault(day_name(item[0]), []).append(item)
    app_days, app_night = None, {}
    skip_days = set(args.skip_days.split(",")) if args.skip_days else set()
    dump = {}
    night_by_day = {}
    for day, items in sorted(by_night.items()):
        if app_days is not None and day not in app_days:
            continue
        if day in skip_days:
            continue
        be = max(i[0] for i in items)
        periods = []
        for ibe, ibs, ir, (lhr, lhr_t, rmssd, temp, tib) in sorted(items, key=lambda i: i[1]):
            p = SI.period_from_scan(ir["scan"], ibs, int(args.tz * 60), is_main=0)
            p.lowest_hr, p.lowest_hr_time, p.rmssd, p.highest_temp = lhr, lhr_t, rmssd, temp
            if p.total_sleep and tib:
                p.time_in_bed = tib
                p.efficiency = (200 * p.total_sleep + tib) // (2 * tib)
            if p.total_sleep:
                periods.append(p)
        if not periods:
            continue
        # Naps are scored separately, so the daily call gets the longest period only.
        periods = [max(periods, key=lambda p: p.total_sleep)]
        periods[0].is_main = 1
        if os.environ.get("OURA_DEBUG"):
            print("  periods", day, [(p.start, p.time_in_bed // 30, p.total_sleep // 30, p.is_main) for p in periods], file=sys.stderr)
        sac.readiness_map = dict(readiness_map)
        stats, records, first_day, _, _, _ = SA.compute(sac)
        by_day = {day_name(d0): (st, i) for i, (d0, st) in enumerate(stats)}
        prev_name = day_name(be - 86400)
        prev = by_day.get(prev_name)
        if prev is None:
            prev_arr = [0] * 8
        else:
            st = prev[0]
            prev_arr = [0, st.eq_distance, st.non_wear, st.classes[1], st.classes[4], st.classes[2], st.classes[3], st.classes[0]]
        def act_entry(name):
            a = by_day.get(name)
            if a is None:
                return 0, 0
            c = a[0].classes
            return a[0].eq_distance, BC.wear_percent(c[1], c[2], c[3], c[4], c[0])
        hist_cal = []
        for k in range(14):
            sl = night_by_day.get(day_name(be - 86400 * (k + 1)), (0, 0))
            dist, wear = act_entry(day_name(be - 86400 * (k + 2)))
            hist_cal.append((sl[0], dist, sl[1], wear))
        hist_before = hist_cal
        res = scorer.step(be // 86400 * 86400, age, periods, prev_arr, (0, 0), (0x100 | args.sri) if args.sri else SRI, history=hist_cal)
        night_by_day[day] = (periods[0].total_sleep, periods[0].rmssd)
        if args.dump:
            p0 = periods[0]
            dump[day] = {"prev": prev_arr, "baseline": [list(scorer.means), list(scorer.devs)], "hist": hist_before,
                         "period": {k: getattr(p0, k) for k in ("time_in_bed", "total_sleep", "rem", "deep", "efficiency", "latency",
                                    "wake_ups", "restless", "got_up", "midpoint", "highest_temp", "lowest_hr", "lowest_hr_time", "rmssd")},
                         "sleep_score": res.outputs.sleep.score if res.outputs else res.sleep.score,
                         "readiness": res.readiness_score, "contributors": list(res.contributors) if res.contributors else None}
        ds, rscore = res.sleep, res.readiness_score
        sscore = res.outputs.sleep.score if res.outputs else ds.score  # the day's final sleep score
        readiness_map[day] = rscore or args.readiness
        acts = by_day.get(day)
        ascore = None
        if acts:
            st, i = acts
            window = (SA.pad_days(age, gender) + records[:i])[-9:] + [records[i]]
            ascore = AS.score(window, i)[0]
        results.append((day, sscore, rscore, ascore))
        if args.dump:
            dump[day]["activity"] = ascore
        print(f"{day:10} {sscore:5} {rscore if rscore is not None else '-':>5} {ascore if ascore is not None else '-':>8}   {list(ds.contributors)}")
    if args.dump:
        Path(args.dump).write_text(json.dumps(dump, indent=1))


if __name__ == "__main__":
    main()
