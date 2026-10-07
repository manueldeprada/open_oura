#!/usr/bin/env python3
"""The night's lowest heart rate and its time.

    python tools/lowest_hr.py            # self-check on a ramp

The ring's 5-minute heart rates of the night (one byte per slot, 0 for a missing slot, slot k covers
the five minutes that end at bedtime start + 300 k + something below 300) become the sleep record's
`LowestHeartRate` and `LowestHeartRateTimeOffset`. The function does not take the plain minimum:

1. Zeros become 255. Q1 and Q3 are the 25th and 75th percentiles of all slots (linear
   interpolation, float32), and the outlier fence is trunc(Q3 + 2 (Q3 - Q1)).
2. The first three slots (in time order) that are above the fence (and at most 254) and lie
   before the first minimum of the whole night are set to 255.
3. The lowest value is the minimum over slots 6.. (all slots when there are at most 6) and its
   first slot is `argmin`.
4. Candidates are slots k with 7 <= k < argmin whose value is within 3 bpm of the lowest value.
   For each candidate in order, the values before it (A, slots 0..k-1) and between it and argmin
   (B, slots k+1..argmin-1) are summarized by their mean and sample standard deviation (255
   entries ignored, 255 when there are none or, for the deviation, fewer than two). The first
   candidate with mean(A) > mean(B) and std(A) > std(B) is the time slot. A candidate with an
   empty or all-255 B is kept and the scan continues; one with an all-255 A leaves the choice
   unchanged. With no choice the time slot is `argmin`.
5. The time is slot * interval seconds; the heart rate is the lowest value of step 3.

"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readiness_score import f32, fma32


def _percentile(vals, p):
    s = sorted(vals)
    rank = f32(f32(f32(p) / f32(100.0)) * f32(float(len(s) - 1)))
    lo, hi = math.floor(rank), math.ceil(rank)
    frac = f32(rank - float(lo))
    return fma32(f32(1.0 - frac), f32(float(s[lo])), f32(frac * f32(float(s[hi]))))


def _summary(vals):
    """(mean, sample standard deviation) of the entries below 255, 255.0 for each when undefined."""
    xs = [f32(float(v)) for v in vals if v < 255]
    total = f32(0.0)
    for x in xs:
        total = f32(total + x)
    mean = f32(total / f32(float(len(xs)))) if xs else f32(255.0)
    sq = f32(0.0)
    for x in xs:
        d = f32(x - mean)
        sq = f32(sq + f32(d * d))
    std = f32(math.sqrt(f32(sq / f32(float(len(xs) - 1))))) if len(xs) > 1 else f32(255.0)
    return mean, std


def calculate_lowest_hr(data, interval=300):
    """Return (lowest heart rate, time in seconds) for the 5-minute heart rates `data` (ints,
    0 = missing), or None for an empty input."""
    n = len(data)
    if n == 0:
        return None
    v1 = [255 if b == 0 else b for b in data]
    q1, q3 = _percentile(v1, 25.0), _percentile(v1, 75.0)
    fence = int(f32(q3 + f32(2.0 * f32(q3 - q1))))
    first_min = v1.index(min(v1))
    v2 = list(v1)
    outliers = [j for j, v in enumerate(v1) if j < first_min and 254 >= v > fence]
    for j in outliers[:3]:
        v2[j] = 255
    start = 6 if n > 6 else 0
    lowest = min(v2[start:])
    argmin = start + v2[start:].index(lowest)
    best = -1
    for k in range(7, min(argmin, n)):
        if v2[k] > lowest + 3:
            continue
        before, after = v2[:k], v2[k + 1:argmin]
        if not after or all(v == 255 for v in after):
            best = k  # nothing to compare with: the candidate is kept and the scan goes on
            continue
        if all(v == 255 for v in before):
            continue  # keeps the previous choice
        ma, sa = _summary(before)
        mb, sb = _summary(after)
        if ma > mb and sa > sb:
            best = k
            break
    return lowest, (argmin if best == -1 else best) * interval


if __name__ == "__main__":
    print(calculate_lowest_hr(list(range(80, 20, -1))))  # (21, 16800)
