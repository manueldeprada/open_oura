#!/usr/bin/env python3
"""Write crates/oura-analysis/src/ported/daily_vectors.rs: three synthetic 10-day sequences
through `DailyScorer` (tools/daily_scorer.py), the Python orchestration that
crates/oura-analysis/src/ported/daily.rs mirrors.

    python tools/gen_daily_vectors.py

Sequence 1 starts from the cold-start baseline (`DailyScorer()`); the others from that baseline
with each item perturbed by up to 10 percent (`DailyScorer.with_baseline`). Each sequence has
its own random seed and age. Days have one main period and
sometimes a second one; every period has total_sleep > 0 and the previous-day words pass
`check_inputs` of readiness_inputs.py. Pure Python: the expected values are
the Python port's.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_scorer as DS
import readiness_inputs as RI

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "crates/oura-analysis/src/ported/daily_vectors.rs"
SEEDS = (1, 2, 3)
DAYS = 10
DAY0 = 1759968000   # 2025-10-09 00:00 UTC

HEADER = """struct DailyDay {
    day_utc: i64,
    age: u32,
    periods: Vec<SleepPeriod>,
    prev_day: [u32; 8],
    chrono: Option<(u8, i32)>,
    sri: u32,
    /// sleep score (`get_daily_sleep`), its seven contributors, readiness score, the nine
    /// readiness contributors (activity balance, last day activity, last night sleep, resting
    /// HR, resting HR time, sleep balance, temperature, HRV balance, sleep regularity), and the
    /// score of `daily_outputs_stateless` (running maximum over the day's prefixes)
    want: (u32, [u32; 7], i32, [i32; 9], u32),
}

struct DailySeq {
    means: [u16; 9],
    devs: [u16; 9],
    days: Vec<DailyDay>,
}
"""


def rand_period(r, day_utc, main):
    tib = r.randint(14000, 36000) if main else r.randint(1200, 9000)
    tot = r.randint(tib * 6 // 10, tib * 95 // 100)
    rem = r.randint(tot // 12, tot // 3)
    deep = r.randint(tot // 20, tot // 4)
    start = day_utc - r.randint(3, 9) * 3600 if main else day_utc + r.randint(10, 15) * 3600
    return RI.SleepPeriod(
        is_main=1 if main else 0, start=start, tz=r.choice((0, 60, 120, 330, -300)), time_in_bed=tib,
        total_sleep=tot, rem=rem, deep=deep, efficiency=r.randint(60, 99), latency=r.choice((0, 300, 600, 900, 1500)),
        wake_ups=r.randint(0, 6), restless=r.randint(0, 400), got_up=r.randint(0, 4), midpoint=r.randint(300, 900),
        highest_temp=r.randint(3300, 3750), lowest_hr=r.randint(42, 70), lowest_hr_time=r.randint(0, tib) // 300 * 300,
        rmssd=r.randint(20, 120))


def rand_prev_day(r):
    return [0, r.randint(2000, 14000), r.randint(0, 300), r.randint(0, 120), r.randint(0, 240), r.randint(0, 60),
            r.randint(300, 900), r.randint(0, 600)]


def lit(p):
    return (f"SleepPeriod {{ is_main: {p.is_main}, start: {p.start}, tz: {p.tz}, time_in_bed: {p.time_in_bed}, "
            f"total_sleep: {p.total_sleep}, rem: {p.rem}, deep: {p.deep}, efficiency: {p.efficiency}, "
            f"latency: {p.latency}, wake_ups: {p.wake_ups}, restless: {p.restless}, got_up: {p.got_up}, "
            f"midpoint: {p.midpoint}, highest_temp: {p.highest_temp}, lowest_hr: {p.lowest_hr}, "
            f"lowest_hr_time: {p.lowest_hr_time}, rmssd: {p.rmssd} }}")


def gen_seq(seed):
    r = random.Random(seed)
    means = tuple(m + r.randint(-m // 10, m // 10) for m in DS.DEFAULT_MEANS) if seed > 1 else DS.DEFAULT_MEANS
    devs = tuple(max(1, d + r.randint(-d // 10, d // 10)) for d in DS.DEFAULT_DEVS) if seed > 1 else DS.DEFAULT_DEVS
    age = r.choice((31, 35, 52))
    scorer = DS.DailyScorer.with_baseline(means, devs)
    rows = []
    for d in range(DAYS):
        day_utc = DAY0 + d * 86400
        periods = [rand_period(r, day_utc, True)]
        if d in (2, 5, 8) or r.random() < 0.15:
            periods.append(rand_period(r, day_utc, False))
            r.shuffle(periods)
        prev = rand_prev_day(r)
        chrono = r.choice(((0, 0), (0, 0), None, (2, r.randint(-3600, 6 * 3600))))
        sri = r.choice((DS.SRI, DS.SRI, r.randint(0, 0x1FF)))
        assert RI.check_inputs(prev, (0, age), periods) == 0
        res = scorer.step(day_utc, age, periods, prev, chrono, sri)
        assert res.outputs is not None
        want = (res.sleep.score, list(res.sleep.contributors), res.readiness_score, list(res.contributors),
                res.outputs.sleep.score)
        rows.append((day_utc, age, periods, prev, chrono, sri, want))
    return means, devs, rows


def render(seq):
    means, devs, rows = seq
    out = [f"        DailySeq {{ means: {list(means)}, devs: {list(devs)}, days: vec!["]
    for day_utc, age, periods, prev, chrono, sri, w in rows:
        ch = "None" if chrono is None else f"Some(({chrono[0]}, {chrono[1]}))"
        ps = ", ".join(lit(p) for p in periods)
        out.append(f"            DailyDay {{ day_utc: {day_utc}, age: {age}, periods: vec![{ps}], prev_day: {prev}, "
                   f"chrono: {ch}, sri: {sri}, want: ({w[0]}, {w[1]}, {w[2]}, {w[3]}, {w[4]}) }},")
    out.append("        ] },")
    return "\n".join(out)


def main():
    parts = [HEADER, "fn daily_sequences() -> Vec<DailySeq> {\n    vec![", *[render(gen_seq(s)) for s in SEEDS], "    ]\n}"]
    OUT.write_text("\n".join(parts) + "\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
