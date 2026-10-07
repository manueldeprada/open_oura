#!/usr/bin/env python3
"""Write reference vectors for the readiness-side ported modules from the Python ports on random
synthetic inputs: readiness (v2, legacy, recovery, baseline update), readiness_inputs, rest_recovery.

    python tools/gen_public_vectors_readiness.py [SECTION ...]

Outputs go to crates/oura-analysis/src/ported/vectors/<module>.rs. Seeds are fixed.
"""
import math
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readiness_inputs as RI
import readiness_score as R
import rest_recovery as RR
from public_vectors_common import nums, write

SEED = 20261018 + 200
BASE_T = 1_760_000_000 // 86400 * 86400


def fl(x):
    """Rust f32 literal for a float32 value."""
    if math.isnan(x):
        return "f32::NAN"
    if math.isinf(x):
        return "f32::INFINITY" if x > 0 else "f32::NEG_INFINITY"
    s = repr(float(__import__("numpy").float32(x)))
    return s if ("." in s or "e" in s) else s + ".0"


def arr(vals):
    return "[" + ", ".join(str(v) for v in vals) + "]"


def opt(v, f=str):
    return "None" if v is None else f"Some({f(v)})"


def rle_phases(ph):
    runs = []
    for s in ph:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    return "&[" + ", ".join(f"({s}, {n})" for s, n in runs) + "]"


# ---- random inputs (the generators of the ports' fuzz tests) ----------------------

def rand_score(r):
    return r.choice([R.MISSING, 0, 1, 20, 50, 70, 85, 99, 100, 100, 100, r.randint(0, 100), r.randint(0, 100)])


def rand_phases(r):
    if r.random() < 0.08:
        return b""
    n = r.choice([0, 1, 2, 5, 40, 300, 700, 960, 1100])
    if r.random() < 0.1:
        return bytes(r.choice([0, 4]) for _ in range(n))
    out, st = bytearray(), r.choice([4, 2])
    while len(out) < n:
        out += bytes([st]) * r.randint(1, 60)
        st = r.choice([1, 2, 2, 3, 4])
    return bytes(out[:n])


def rand_v2(r):
    x = R.ReadinessV2Input(*[rand_score(r) for _ in range(7)])
    for _ in range(r.choice([0, 1, 1, 1, 2, 3])):
        x.sessions.append(R.Session(r.choice([0, 0, 1, 2]), rand_phases(r),
                                    r.choice([r.uniform(11, 20), r.uniform(11, 20), R.NAN, R.INF]),
                                    r.choice([r.uniform(0, 12), r.uniform(0, 40), R.NAN, 0.0])))
    if r.random() < 0.8:
        n = r.choice([0, 5, 13, 14, 30, 90, 120])
        x.sfi_history = [r.choice([r.uniform(0, 30), r.uniform(5, 15), R.NAN, R.INF]) if r.random() < 0.05
                         else r.uniform(2, 25) for _ in range(n)]
    if r.random() < 0.8:
        n = r.choice([0, 5, 13, 14, 30, 90, 120])
        base = r.uniform(12, 18)
        x.br_history = [R.NAN if r.random() < 0.03 else base + r.gauss(0, r.choice([0.1, 0.5, 1.0])) for _ in range(n)]
    x.age = r.choice([0, 15, 20, 25, 33, 47, 62, 80, 95])
    return x


def pack_baseline(means, devs, head=bytes(16)):
    b = bytearray(0x38)
    b[:16] = head
    for k in range(9):
        struct.pack_into("<HH", b, 0x10 + 4 * k, means[k], devs[k])
    return bytes(b)


def rand_baseline(r):
    if r.random() < 0.05:   # out-of-range values restore the defaults
        means = [r.randint(0, 0xFFFF) for _ in range(9)]
    else:
        means = [r.randint(300, 790), r.randint(2800, 4300), r.randint(300, 700), r.randint(1500, 3300),
                 r.randint(3000, 28000), r.randint(40, 300), r.randint(2400, 4300), r.randint(25600, 29900),
                 r.randint(2400, 6000)]
    devs = [r.choice([0, 0, r.randint(20, 400)]) if r.random() < .1 else r.randint(20, 400) for _ in range(9)]
    devs[4] = r.randint(1000, 9000) if r.random() > .1 else devs[4]
    return pack_baseline(means, devs)


def rand_rest_input(r, utc):
    kind = r.choice(["none", "none", "open", "done", "done", "today", "future", "zero"])
    cur = utc // 86400 * 86400 if utc else 1_760_000_000 // 86400 * 86400
    if kind == "none":
        return None
    start = cur - r.randint(0, 14) * 86400 + r.randint(0, 86399)
    if kind == "open":
        return (start, 0, 0, cur)
    if kind == "done":
        return (start, start + r.randint(1, 9) * 86400 + r.randint(-80000, 80000), 0, cur)
    if kind == "today":
        return (cur + r.randint(0, 86399), 0, 0, cur)
    if kind == "future":
        return (cur + 2 * 86400, cur + 3 * 86400, 0, cur)
    return (0, 0, 0, cur)


def rand_legacy(r):
    x = R.ReadinessInput()
    x.utc = r.choice([1_760_000_000 + r.randint(0, 10**6), 1_700_000_000, 0])
    x.day_number = r.choice([0, 1, 3, 4, 8, 14, 15, 30, 200, 300])
    x.sleep_score = r.randint(0, 100)
    tib = r.randint(4 * 3600, 11 * 3600)
    tst = int(tib * r.uniform(0.7, 0.95))
    x.time_in_bed, x.total_sleep = tib, tst
    x.rem = int(tst * r.uniform(0.1, 0.3))
    x.deep = int(tst * r.uniform(0.05, 0.25))
    x.latency = r.randint(0, 3000)
    x.wake_ups = r.randint(0, 9)
    x.highest_temp = r.choice([r.randint(3300, 3750), r.randint(3300, 3750), 0, 4300, 5000])
    x.lowest_hr = r.choice([r.randint(40, 90), r.randint(40, 90), 0, 180, 255])
    x.lowest_hr_time = r.randint(0, tib)
    x.rmssd = r.choice([0, r.randint(10, 150), r.randint(10, 150)])
    x.p40 = (r.randint(0, 5), r.randint(0, 600))
    if r.random() > 0.05:
        x.prev_day = [r.randint(0, 2000), r.choice([0, r.randint(1000, 25000)]), r.randint(0, 600),
                      r.randint(100, 900), r.choice([0, r.randint(0, 120), 300]), r.randint(0, 300),
                      r.randint(0, 300), r.choice([0, 100, 400, 1440])]
    if r.random() > 0.05:
        base_sleep = r.randint(300, 540) * 60
        x.sleep_sec_14 = [max(0, int(base_sleep * r.uniform(0.6, 1.3))) if r.random() > .1 else 0 for _ in range(14)]
        x.distance_14 = [r.choice([0, r.randint(1000, 20000)]) for _ in range(14)]
        x.rmssd_14 = [r.choice([0, r.randint(15, 140)]) for _ in range(14)]
        x.wear_14 = [r.choice([100, 100, r.randint(20, 100)]) for _ in range(14)]
        x.temp_dev_3 = [r.choice([-0x8000, r.randint(-100, 100)]) for _ in range(3)]
        x.hot_90 = [r.choice([0, r.randint(3300, 3750)]) for _ in range(90)]
    x.baseline = rand_baseline(r)
    x.cycle_day_type = r.choice([0, 1, 2, 3])
    x.rest_input = rand_rest_input(r, x.utc)
    return x


def rand_state(r, utc):
    S = R.State()
    means = [r.randint(300, 790), r.randint(2800, 4300), r.randint(300, 700), r.randint(1500, 3300),
             r.randint(3000, 28000), r.randint(40, 300), r.randint(2400, 4300), r.randint(25600, 29900),
             r.randint(2400, 6000)]
    devs = [r.randint(20, 400) for _ in range(9)]
    if r.random() < 0.1:
        means[r.randrange(9)] = r.randint(0, 0xFFFF)
    for k in range(9):
        S.set16(0x40 + 2 * k, means[k])
        S.set16(0x52 + 2 * k, devs[k] if r.random() > 0.05 else 0)
    for o in range(0x04, 0x20, 2):
        S.set16(o, r.choice([0, r.randint(200, 600)]))
    for o in range(0x22, 0x3E, 2):
        S.set16(o, r.choice([0, r.randint(100, 3000)]))
    for o in range(0x68, 0x84, 2):
        S.set16(o, r.choice([0, r.randint(300, 700)]))
    S.set16(0x20, r.choice([0, r.randint(100, 3000)]))
    S.set16(0x3E, r.randint(100, 900))
    S.b[0x64] = r.choice([0, 2, 4, 9, 15, 40])
    S.b[0x65] = r.randint(0, 120)
    S.b[0x66] = r.choice([100, r.randint(20, 100)])
    S.b[0x84] = r.choice([0, 3, 8, 20])
    S.set32(0x88, (utc - r.choice([0, 0, 600, 3600, 5000, 50000, 90000, 200000, 700000, -300000])) & 0xFFFFFFFF)
    return S


def rand_sri(r):
    """A sleep-regularity state (rows of 144 slots) and its index, or none."""
    if r.random() < 0.5:
        nrows = r.randint(5, 20)
        rows = [[r.choice([0, 1]) if r.random() > 0.08 else 2 for _ in range(R.SRI_COLUMNS)] for _ in range(nrows)]
        return R.sri_calculate(rows, True)
    return 0


# ---- readiness ---------------------------------------------------------------------------------------

def same_nan(a):
    return a


def gen_v2(n, r):
    rows = []
    for _ in range(n):
        x = rand_v2(r)
        for s in x.sessions:   # the model reads float32
            s.breathing_rate, s.bdi = R.f32(s.breathing_rate), R.f32(s.bdi)
        if x.sfi_history is not None:
            x.sfi_history = [R.f32(v) for v in x.sfi_history]
        if x.br_history is not None:
            x.br_history = [R.f32(v) for v in x.br_history]
        o = R.readiness_v2(x)
        ints = (o.score, o.sleep_recovery, o.nervous_system, o.body_strain, o.waso_score, o.sfi_score,
                o.br_score, o.bdi_score, o.restorative_sleep)
        floats = (o.sfi, o.waso_min, o.br_avg, o.bdi, o.sfi_p25, o.sfi_p75, o.br_mean, o.br_std)
        scores = [x.total_sleep, x.deep_sleep, x.rem_sleep, x.hrv_balance, x.resting_hr, x.recovery_index, x.body_temp]
        sess = ", ".join(f"({s.type}, {rle_phases(s.phases)}, {fl(s.breathing_rate)}, {fl(s.bdi)})" for s in x.sessions)
        hist = lambda h: opt(h, lambda v: "&[" + ", ".join(fl(a) for a in v) + "]")
        rows.append(f"    V2Case {{ scores: {arr(scores)}, sessions: &[{sess}], sfi_history: {hist(x.sfi_history)}, "
                    f"br_history: {hist(x.br_history)}, age: {x.age}, ints: {arr(ints)}, "
                    f"floats: [{', '.join(fl(v) for v in floats)}] }},")
    return ("const V2_VECTORS: [V2Case; %d] = [\n" % n) + "\n".join(rows) + "\n];"


def pack_row(row):
    b = bytearray(len(row) // 4)
    for k, v in enumerate(row):
        b[k // 4] |= v << (2 * (k % 4))
    return b.hex()


def rust_input(x):
    a = lambda v: opt(v, arr)
    return (f"ReadinessInput {{ utc: {x.utc}, day_number: {x.day_number}, sleep_score: {x.sleep_score}, "
            f"time_in_bed: {x.time_in_bed}, total_sleep: {x.total_sleep}, rem: {x.rem}, deep: {x.deep}, "
            f"latency: {x.latency}, wake_ups: {x.wake_ups}, highest_temp: {x.highest_temp}, "
            f"lowest_hr: {x.lowest_hr}, lowest_hr_time: {x.lowest_hr_time}, rmssd: {x.rmssd}, "
            f"p40: {opt(x.p40, lambda v: str(tuple(v)))}, prev_day: {a(x.prev_day)}, "
            f"sleep_sec_14: {a(x.sleep_sec_14)}, distance_14: {a(x.distance_14)}, rmssd_14: {a(x.rmssd_14)}, "
            f"wear_14: {a(x.wear_14)}, temp_dev_3: {a(x.temp_dev_3)}, "
            f"hot_90: {opt(x.hot_90, lambda v: 'vec!' + arr(v))}, "
            f"baseline: {opt(x.baseline, lambda b: 'hex(\"' + bytes(b).hex() + '\")')}, "
            f"cycle_day_type: {x.cycle_day_type}, rest_input: {opt(x.rest_input, lambda v: str(tuple(v)))} }}")


def gen_legacy(n, r):
    out = []
    for _ in range(n):
        x = rand_legacy(r)
        feats = R.Features(*[r.random() > 0.3 for _ in range(4)], athletes=r.random() > 0.3)
        feats.freeze_baseline = False
        sri, rows = 0, []
        if r.random() < 0.5:
            nrows = r.randint(5, 12)
            rows = [[r.choice([0, 1]) if r.random() > 0.08 else 2 for _ in range(R.SRI_COLUMNS)] for _ in range(nrows)]
            sri = R.sri_calculate(rows, True)
        o = R.readiness_calculate(x, feats, sri=sri)
        v = None if o is None else (o.score, o.activity_balance, o.last_day_activity, o.last_night_sleep,
                                    o.resting_hr, o.resting_hr_time, o.sleep_balance, o.temperature,
                                    o.hrv_balance, o.sleep_regularity, o.temperature_deviation,
                                    o.temperature_trend_deviation)
        ft = arr(str(b).lower() for b in (feats.hrv_in_readiness, feats.rest_mode, feats.sleep_regularity, feats.athletes))
        rs = "vec![" + ", ".join(f'"{pack_row(row)}"' for row in rows) + "]"
        want = f"Some({arr(v)})" if v is not None else "None"
        out.append(f"        ({rust_input(x)}, {ft}, {rs}, {want}),")
    return ("fn legacy_vectors() -> Vec<(ReadinessInput, [bool; 4], Vec<&'static str>, Option<[i32; 12]>)> {\n"
            "    vec![\n" + "\n".join(out) + "\n    ]\n}")


def gen_recovery(n, r):
    rows = []
    for _ in range(n):
        utc = r.choice([1_760_000_000, 1_761_234_567, 0])
        S = rand_state(r, utc)
        prev = rand_state(r, utc) if r.random() < 0.5 else R.State(S.b)
        ss = None
        if r.random() > 0.2:
            ss = R.SleepSummary(r.randint(0, 120), r.randint(0, 200), r.randint(100, 700), r.randint(0, 200),
                                r.randint(0, 900), 0, r.randint(0, 100), 0, 0)
        rm = R.RestMode(r.choice([0, 0, 1, 2, 3, 4, 4]), r.randint(0, 7), r.randint(0, 7))
        feats = R.Features(*[r.random() > 0.3 for _ in range(4)], athletes=r.random() > 0.3)
        feats.freeze_baseline = r.random() < 0.15
        tmp = r.choice([r.randint(3300, 3750), 0, 4300])
        h = r.randint(0, 12)
        f = R.f32(r.choice([0.0, r.uniform(3.0, 7.0)]))
        j, j1, j2 = r.randint(0, 600), r.randint(0, 9), r.choice([0, 3, 9, 30])
        upd = r.random() < 0.6
        lowest = r.choice([None, (r.randint(40, 90), r.randint(0, 40000)), (0, 100), (200, 5)])
        Sm = R.State(S.b)
        summary, ret = R.recovery_run(Sm, R.State(prev.b), ss, rm, tmp, h, f, j, j1, j2, upd, lowest, utc, feats, sri=0)
        ssv = opt(ss, lambda v: arr([v.awake, v.rem, v.light, v.deep, v.extra, v.sleep_score]))
        ft = arr(str(b).lower() for b in (feats.hrv_in_readiness, feats.rest_mode, feats.sleep_regularity,
                                          feats.freeze_baseline, feats.athletes))
        rows.append(f'    RecoveryCase {{ state: "{bytes(S.b).hex()}", prev: "{bytes(prev.b).hex()}", '
                    f'sleep_summary: {ssv}, rest: {arr([rm.state, rm.total, rm.remaining])}, t: {tmp}, h: {h}, '
                    f'f: {fl(f)}, j: {j}, j1: {j1}, j2: {j2}, upd: {str(upd).lower()}, '
                    f'lowest: {opt(lowest, lambda v: str(tuple(v)))}, utc: {utc}, feats: {ft}, '
                    f'summary: {arr(summary)}, out_state: "{bytes(Sm.b).hex()}", ret: {str(bool(ret)).lower()} }},')
    return ("const RECOVERY_VECTORS: [RecoveryCase; %d] = [\n" % n) + "\n".join(rows) + "\n];"


def gen_baseline(n, r):
    rows = []
    for _ in range(n):
        feats = R.Features(athletes=r.random() > 0.3)
        item = r.randrange(9)
        m = r.choice([r.randint(0, 0x7EB6), r.randint(0, 0xFFFF), 0])
        d = r.choice([r.randint(0, 600), r.randint(0, 0xFFFF), 0])
        val = r.choice([r.randint(-50, 1200), r.randint(0, 0xFFFF), r.randint(2000, 5000)])
        cnt = r.choice([0, 1, 3, 4, 9, 14, 15, 40, 255])
        S = R.State()
        S.set16(0x40 + 2 * item, m)
        S.set16(0x52 + 2 * item, d)
        R.baseline_update(S, item, val, cnt, feats, [640] + list(R.MEAN_MIN_DEFAULT[1:]))
        want = (S.u16(0x40 + 2 * item), S.u16(0x52 + 2 * item))
        rows.append(f"    ({item}, {m}, {d}, {R.sx16(val)}, {cnt}, {str(feats.athletes).lower()}, {want[0]}, {want[1]}),")
    return ("const BASELINE_VECTORS: [(usize, u16, u16, i16, u32, bool, u16, u16); %d] = [\n" % n
            + "\n".join(rows) + "\n];")


READINESS_HEADER = """// Reference vectors from the Python port tools/readiness_score.py on random synthetic inputs.
// Generated by tools/gen_public_vectors_readiness.py; do not edit.
struct V2Case {
    scores: [u32; 7],
    /// (type, hypnogram as (stage, run length) pairs, breathing rate, BDI)
    sessions: &'static [(u32, &'static [(u8, u16)], f32, f32)],
    sfi_history: Option<&'static [f32]>,
    br_history: Option<&'static [f32]>,
    age: u32,
    ints: [i32; 9],
    floats: [f32; 8],
}

struct RecoveryCase {
    /// State bytes as hex.
    state: &'static str,
    prev: &'static str,
    /// (awake, rem, light, deep, extra, sleep score)
    sleep_summary: Option<[i64; 6]>,
    /// (state, total, remaining)
    rest: [u32; 3],
    t: u32,
    h: i32,
    f: f32,
    j: u32,
    j1: u32,
    j2: u32,
    upd: bool,
    lowest: Option<(u32, u32)>,
    utc: i64,
    /// hrv_in_readiness, rest_mode, sleep_regularity, freeze_baseline, athletes
    feats: [bool; 5],
    summary: [u8; 10],
    out_state: &'static str,
    ret: bool,
}
"""


def gen_readiness():
    r = random.Random(SEED + 1)
    parts = [READINESS_HEADER, gen_v2(24, r), gen_legacy(20, r), gen_recovery(20, r), gen_baseline(60, r)]
    write("readiness", "\n".join(parts) + "\n")


# ---- readiness inputs ------------------------------------------------------------------------------

INPUTS_HEADER = """// Reference vectors from the Python port tools/readiness_inputs.py on random synthetic days.
// Generated by tools/gen_public_vectors_readiness.py; do not edit.
struct DayCase {
    sleep_score: u32,
    day: DayData,
    periods: Vec<SleepPeriod>,
    time0: i64,
    /// hrv_in_readiness, rest_mode, sleep_regularity, freeze_baseline, athletes
    feats: [bool; 5],
    sri: u32,
    /// (readiness output, [total sleep sum, highest temp, lowest HR, lowest-HR time, RMSSD])
    want: Option<([i32; 12], [u32; 5])>,
}

struct StatefulCase {
    state: &'static str,
    sleep_score: u32,
    prev_day: [u32; 8],
    periods: Vec<SleepPeriod>,
    /// (state, total, remaining)
    rest: [u32; 3],
    j: u32,
    j1: u32,
    j2: u32,
    upd: bool,
    time0: i64,
    feats: [bool; 5],
    sri: u32,
    /// (first ten readiness outputs, night values as above, state after the call)
    want: Option<([i32; 10], [u32; 5], &'static str)>,
}

struct DailyWant {
    readiness: [i32; 12],
    sleep: [u32; 13],
    sleep_scores: Vec<u32>,
    readiness_scores: Vec<i32>,
    sleep_deltas: Vec<u32>,
    readiness_deltas: Vec<u32>,
    v2_inputs: Vec<[u32; 7]>,
}

struct DailyCase {
    day_utc: i64,
    tz_word: i64,
    day: DayData,
    periods: Vec<SleepPeriod>,
    chrono: Option<(u8, i32)>,
    feats: [bool; 5],
    sri: u32,
    want: Option<DailyWant>,
}
"""

S32 = lambda v: struct.unpack("<i", struct.pack("<I", v & 0xFFFFFFFF))[0]


def rand_period(r, day, main):
    dur = r.randint(1, 11) * 3600 + r.randint(0, 3599)
    start = day - r.randint(0, 40) * 3600 - r.randint(0, 3599)
    if r.random() < 0.06:
        start = r.choice([0, 1_300_000_000, 2_600_000_000, -5, 4_000_000_000])
    tib = dur if r.random() > 0.05 else r.randint(0, 100)
    tst = int(tib * r.uniform(0.6, 0.97)) if tib > 0 else 0
    if r.random() < 0.05:
        tst = r.randint(1, 80)
    return RI.SleepPeriod(
        is_main=main, start=start, tz=r.choice([0, 60, 120, -300, 330, 540, -720, r.randint(-720, 840)]),
        time_in_bed=tib, total_sleep=tst, rem=int(tst * r.uniform(0.05, 0.3)), deep=int(tst * r.uniform(0.02, 0.3)),
        efficiency=r.randint(40, 100), latency=r.choice([0, 300, 899, 900, 901, r.randint(0, 4000)]),
        wake_ups=r.choice([0, 1, 3, r.randint(0, 30)]), restless=r.choice([0, r.randint(0, 600)]),
        got_up=r.choice([0, 1, 2, r.randint(0, 10)]), midpoint=r.randint(0, max(1, tib // 30)),
        highest_temp=r.choice([r.randint(3300, 3750), r.randint(3300, 3750), 0, 4300]),
        lowest_hr=r.choice([r.randint(38, 90), r.randint(38, 90), 0, 255]),
        lowest_hr_time=r.randint(0, max(1, tib)), rmssd=r.choice([0, r.randint(8, 160), r.randint(8, 160)]))


def rand_periods(r, day):
    n = r.choice([1, 1, 1, 2, 2, 3, 4])
    mains = [r.choice([1, 1, 0, 2]) if r.random() < 0.7 else 0 for _ in range(n)]
    if r.random() < 0.85 and 1 not in mains:
        mains[r.randrange(n)] = 1
    ps = [rand_period(r, day, m) for m in mains]
    for p in ps:   # spread the midpoints over the three timing regions of the non-main adjustment
        if p.is_main != 1 and r.random() < 0.5:
            tod = r.randint(-50400, 35999) % 86400
            p.start += (tod - (p.start + p.tz * 60 + p.midpoint * 30)) % 86400
    if r.random() < 0.85:   # keep the quiet cases (one valid period) from dominating
        for p in ps:
            p.start = max(p.start, 1_354_757_888 + 10_000)
    return ps


def rand_case(r):
    x = rand_legacy(r)
    day = r.choice([1_760_000_000, 1_740_000_000 + r.randint(0, 10**6), 1_800_000_000]) // 86400 * 86400
    periods = rand_periods(r, day)
    if r.random() < 0.97:
        for p in periods:
            p.total_sleep = max(p.total_sleep, 1)
    user = (r.choice([0, 1, 2, 3]), r.choice([r.randint(18, 80), r.randint(18, 80), 0, 255, 300]))
    hist = RI.History(x.temp_dev_3, x.hot_90, x.sleep_sec_14, x.distance_14, x.rmssd_14, x.wear_14)
    if r.random() < 0.97 and x.prev_day is None:
        x.prev_day = [0] * 8
    chrono = None if r.random() < 0.4 else (r.choice([0, 1, 2, 3, 4, 5, 6, 255, r.randint(0, 255)]),
                                            r.choice([r.randint(0, 86399), r.randint(-3600, 90000)]))
    feats = R.Features(*[r.random() > 0.3 for _ in range(4)], athletes=r.random() > 0.3)
    feats.freeze_baseline = False
    feats.rest_mode = True if r.random() < 0.7 else feats.rest_mode
    return dict(day=day, periods=periods, user=user, hist=hist, x=x, chrono=chrono, feats=feats)


def ro_tuple(o):
    return (o.score, o.last_night_sleep, o.sleep_balance, o.last_day_activity, o.activity_balance, o.resting_hr,
            o.resting_hr_time, o.temperature, o.hrv_balance, o.sleep_regularity, o.temperature_deviation,
            o.temperature_trend_deviation)


def period(p):
    return (f"SleepPeriod {{ is_main: {p.is_main}, start: {p.start}, tz: {p.tz}, time_in_bed: {p.time_in_bed}, "
            f"total_sleep: {p.total_sleep}, rem: {p.rem}, deep: {p.deep}, efficiency: {p.efficiency}, "
            f"latency: {p.latency}, wake_ups: {p.wake_ups}, restless: {p.restless}, got_up: {p.got_up}, "
            f"midpoint: {p.midpoint}, highest_temp: {p.highest_temp}, lowest_hr: {p.lowest_hr}, "
            f"lowest_hr_time: {p.lowest_hr_time}, rmssd: {p.rmssd} }}")


def periods_lit(ps):
    return "vec![" + ", ".join(period(p) for p in ps) + "]"


def day_data(c, x):
    h = c["hist"]
    return (f"DayData {{ prev_day: {opt(x.prev_day, arr)}, user: {opt(c['user'], lambda v: str(tuple(v)))}, "
            f"history: History {{ temp_dev_3: {opt(h.temp_dev_3, arr)}, hot_90: {opt(h.hot_90, lambda v: 'vec!' + arr(v))}, "
            f"sleep_sec_14: {opt(h.sleep_sec_14, arr)}, distance_14: {opt(h.distance_14, arr)}, "
            f"rmssd_14: {opt(h.rmssd_14, arr)}, wear_14: {opt(h.wear_14, arr)} }}, "
            f"baseline: {opt(x.baseline, lambda b: 'hex(\"' + bytes(b).hex() + '\")')}, "
            f"day_number: {x.day_number}, cycle_day_type: {x.cycle_day_type}, "
            f"rest_input: {opt(x.rest_input, lambda v: str(tuple(v)))} }}")


def feats_lit(f):
    return arr(str(b).lower() for b in (f.hrv_in_readiness, f.rest_mode, f.sleep_regularity, f.freeze_baseline,
                                        f.athletes))


def gen_ri_sleep(n, r):
    rows = []
    for _ in range(n):
        c = rand_case(r)
        age = c["user"][1]
        ct, cm = (c["chrono"][0], RI._tdiv(S32(c["chrono"][1]), 60)) if c["chrono"] else (0xFF, 0)
        got = RI.get_daily_sleep(age, ct, cm, c["periods"])
        rows.append(f"        ({age}, {ct}, {cm}, {periods_lit(c['periods'])}, {arr(got.words()[:9])}),")
    return ("fn sleep_vectors() -> Vec<(u32, u8, i32, Vec<SleepPeriod>, [u32; 9])> {\n    vec![\n"
            + "\n".join(rows) + "\n    ]\n}")


def gen_ri_stateless(n, r):
    rows = []
    for _ in range(n):
        c = rand_case(r)
        x, ps = c["x"], c["periods"]
        sri = rand_sri(r)
        score, t0 = r.randint(0, 100), c["day"] - 86400
        gro, gso = RI.daily_readiness_stateless(RI.DailySleep(score=score), x.prev_day, ps, c["user"], c["hist"],
                                                x.baseline, x.day_number, t0, x.cycle_day_type, x.rest_input,
                                                c["feats"], sri)
        night = (gso.total_sleep_sum, gso.highest_temp, gso.lowest_hr, gso.lowest_hr_time, gso.rmssd)
        want = None
        if RI.aggregate_periods(ps) is not None:
            want = f"Some(({arr(ro_tuple(gro))}, {arr(night)}))"
        rows.append(f"        DayCase {{ sleep_score: {score}, day: {day_data(c, x)}, periods: {periods_lit(ps)}, "
                    f"time0: {t0}, feats: {feats_lit(c['feats'])}, sri: {sri}, want: {want or 'None'} }},")
    return "fn stateless_vectors() -> Vec<DayCase> {\n    vec![\n" + "\n".join(rows) + "\n    ]\n}"


def gen_ri_stateful(n, r):
    rows = []
    for _ in range(n):
        c = rand_case(r)
        ps = c["periods"]
        utc = c["day"] - 86400
        sri = rand_sri(r)
        S0 = rand_state(r, utc)
        rm = R.RestMode(r.choice([0, 0, 1, 2, 3, 4, 4]), r.randint(0, 7), r.randint(0, 7))
        score, j, j1, j2, upd = r.randint(0, 100), r.randint(0, 600), r.randint(0, 9), r.choice([0, 3, 9, 30]), r.random() < 0.6
        c["feats"].freeze_baseline = r.random() < 0.15
        prev = [r.randint(0, 2000), r.choice([0, r.randint(1000, 25000)]), r.randint(0, 600), r.randint(100, 900),
                r.choice([0, r.randint(0, 120), 300]), r.randint(0, 300), r.randint(0, 300), r.choice([0, 100, 400, 1440])]
        Sm = R.State(S0.b)
        gro, gso = RI.daily_readiness_stateful(RI.DailySleep(score=score), prev, ps, Sm, rm, j, j1, j2, upd, utc,
                                               c["feats"], sri)
        want = None
        if RI.aggregate_periods(ps, 1) is not None:
            night = (gso.total_sleep_sum, gso.highest_temp, gso.lowest_hr, gso.lowest_hr_time, gso.rmssd)
            want = f'Some(({arr(ro_tuple(gro)[:10])}, {arr(night)}, "{bytes(Sm.b).hex()}"))'
        rows.append(f'        StatefulCase {{ state: "{bytes(S0.b).hex()}", sleep_score: {score}, prev_day: {arr(prev)}, '
                    f"periods: {periods_lit(ps)}, rest: {arr([rm.state, rm.total, rm.remaining])}, j: {j}, j1: {j1}, "
                    f"j2: {j2}, upd: {str(upd).lower()}, time0: {utc}, feats: {feats_lit(c['feats'])}, sri: {sri}, "
                    f"want: {want or 'None'} }},")
    return "fn stateful_vectors() -> Vec<StatefulCase> {\n    vec![\n" + "\n".join(rows) + "\n    ]\n}"


def gen_ri_daily(n, r):
    rows = []
    for _ in range(n):
        c = rand_case(r)
        x, ps = c["x"], c["periods"]
        sri = rand_sri(r)
        tzw = r.choice([0, 3600, -18000, 7200])
        args = (c["day"], tzw, x.day_number, ps, x.prev_day, c["user"], c["hist"], x.baseline, c["chrono"],
                x.rest_input, x.cycle_day_type, c["feats"])
        got = RI.daily_outputs_stateless(*args[:11], c["feats"], sri)
        want = "None"
        if got is not None:
            want = (f"Some(DailyWant {{ readiness: {arr(ro_tuple(got.readiness))}, "
                    f"sleep: {arr(got.sleep.words()[:13])}, "
                    f"sleep_scores: vec!{arr(got.sleep_scores)}, readiness_scores: vec!{arr(got.readiness_scores)}, "
                    f"sleep_deltas: vec!{arr(got.sleep_deltas)}, readiness_deltas: vec!{arr(got.readiness_deltas)}, "
                    f"v2_inputs: vec![{', '.join(arr(v) for v in got.v2_inputs)}] }})")
        chrono = None if c["chrono"] is None else (c["chrono"][0], S32(c["chrono"][1]))
        rows.append(f"        DailyCase {{ day_utc: {c['day']}, tz_word: {tzw}, day: {day_data(c, x)}, "
                    f"periods: {periods_lit(ps)}, chrono: {opt(chrono, lambda v: str(tuple(v)))}, "
                    f"feats: {feats_lit(c['feats'])}, sri: {sri}, want: {want} }},")
    return "fn daily_vectors() -> Vec<DailyCase> {\n    vec![\n" + "\n".join(rows) + "\n    ]\n}"


def gen_readiness_inputs(n=14):
    r = random.Random(SEED + 2)
    parts = [INPUTS_HEADER, gen_ri_sleep(n, r), gen_ri_stateless(n, r), gen_ri_stateful(n, r), gen_ri_daily(n, r)]
    write("readiness_inputs", "\n".join(parts) + "\n")


# ---- rest recovery -----------------------------------------------------------------------------------

def rand_times(r):
    cur = BASE_T + r.randint(0, 400) * 3600 * r.choice([1, 24])
    cur = cur // 86400 * 86400
    kind = r.choice(["open", "done", "done", "done", "today", "future", "zero", "equal", "bad"])
    start = cur - r.randint(0, 20) * 86400 + r.randint(0, 86399)
    if kind == "open":
        return start, 0, cur
    if kind == "done":
        return start, start + r.randint(0, 12) * 86400 + r.randint(-70000, 70000), cur
    if kind == "today":
        return cur + r.randint(0, 86399), 0, cur
    if kind == "future":
        return cur + r.randint(1, 3) * 86400, cur + 5 * 86400, cur
    if kind == "zero":
        return 0, 0, cur
    if kind == "equal":
        return start, start, cur
    return start, r.choice([-5, 0, 1]), r.choice([0, cur])


def rand_event(r):
    tag = r.choice([0xFE20, 0xFE21, 0xFE21, 0xFE22, 0x1234])
    ln = r.choice([0, 7, 8, 9, 10])
    body = bytes(r.randint(0, 255) for _ in range(max(ln, 12)))
    return bytes([tag >> 8, tag & 0xFF]) + struct.pack("<H", ln) + body


def gen_rr_state(n, r):
    rows = []
    for _ in range(n):
        s, en, cur = rand_times(r)
        g = RR.get_state(RR.RestInput(s, en, 0, cur))
        rows.append(f"    ({s}, {en}, {cur}, {arr((g.state, g.a, g.b, g.c))}),")
    return "STATE_VECTORS", f"[(i64, i64, i64, [u32; 4]); {n}]", rows


def gen_rr_times(n, r):
    rows = []
    for _ in range(n):
        s = r.choice([0, 1, BASE_T + r.randint(-30, 30) * 86400 + r.randint(0, 86399)])
        en = r.choice([0, s, s + r.randint(0, 12) * 86400 + r.randint(-70000, 70000)])
        p = RR.calculate_all_times(s, en)
        want_p = (p.start, p.start_mid, p.start_next, p.end, p.end_mid, p.end_next, p.recovery_end, p.days,
                  p.recovery_days)
        now = r.choice([0, s + r.randint(-3, 25) * 86400 + r.randint(0, 86399), s])
        g = RR.past_state(p, now)
        want_d = tuple(RR.durations(p))
        rows.append(f"    ({s}, {en}, {arr(want_p)}, {now}, {arr((g.state, g.a, g.b, g.c))}, {want_d}),")
    return "TIMES_VECTORS", f"[(i64, i64, [i64; 9], i64, [u32; 4], (i64, i64)); {n}]", rows


def gen_rr_history(n, r):
    rows = []
    for _ in range(n):
        cnt = r.choice([0, 1, 2, 3, 5, 12, 30, 31])
        periods, t = [], BASE_T - 200 * 86400
        for _ in range(cnt):
            t += r.randint(1, 20) * 86400 + r.randint(0, 80000)
            end = t + r.randint(0, 9) * 86400 + r.randint(0, 80000)
            periods.append((t, r.choice([end, end, 0])))
            t = end
        use_null = r.random() < 0.05
        h = RR.History()
        h.set(None if use_null else periods, cnt)
        qs = []
        for _ in range(6):
            q = r.choice([periods[r.randrange(len(periods))][0] if periods else BASE_T,
                          BASE_T + r.randint(-100, 100) * 86400 + r.randint(0, 86399), 0])
            q += r.choice([0, 0, 86400, -86400, 3 * 86400, 8 * 86400, 17 * 86400])
            g = h.status_for_activity(q)
            qs.append(f"({q}, {arr((g.state, g.a, g.b, g.c))})")
        ps = ", ".join(f"({a}, {b})" for a, b in periods)
        rows.append(f"    (&[{ps}], {str(use_null).lower()}, {cnt}, &[{', '.join(qs)}]),")
    return "HISTORY_VECTORS", f"[(&[(i64, i64)], bool, u32, &[(i64, [u32; 4])]); {n}]", rows


def gen_rr_machine(n, r):
    rows = []
    for _ in range(n):
        m = RR.init_mode()
        steps = []
        for _ in range(r.randint(1, 12)):
            if r.random() < 0.6:
                data = rand_event(r)
                ret = RR.process_event(m, data)
                k, evs = 0, "Some(&[" + ", ".join(str(b) for b in data) + "])"
            else:
                k = r.randint(0, 4)
                RR.major_sleep_period(m, k)
                ret, evs = 0, "None"
            steps.append(f"({evs}, {k}, {ret}, {(m.state, m.n, m.total, m.remaining)})")
        rows.append(f"    (&[{', '.join(steps)}], {str(bool(m.sane())).lower()}),")
    return "MACHINE_VECTORS", f"[(&[(Option<&[u8]>, u32, i32, (u8, u32, u32, u32))], bool); {n}]", rows


def gen_rest_recovery(n=40):
    r = random.Random(SEED + 3)
    parts = ["// Reference vectors from the Python port tools/rest_recovery.py on random synthetic inputs.",
             "// Generated by tools/gen_public_vectors_readiness.py; do not edit."]
    for gen, count in ((gen_rr_state, 3 * n), (gen_rr_times, n), (gen_rr_history, n // 3), (gen_rr_machine, n)):
        name, ty, rows = gen(count, r)
        parts.append(f"const {name}: {ty} = [\n" + "\n".join(rows) + "\n];")
    write("rest_recovery", "\n".join(parts) + "\n")




SECTIONS = dict(readiness=gen_readiness, readiness_inputs=gen_readiness_inputs, rest_recovery=gen_rest_recovery)

if __name__ == "__main__":
    for name in sys.argv[1:] or SECTIONS:
        SECTIONS[name]()
