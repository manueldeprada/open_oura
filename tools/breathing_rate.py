#!/usr/bin/env python3
"""Oura nightly breathing rate: IBI resampling, an IIR band-pass, peak picking and
per-epoch records averaged over the night.

Pipeline, per 30 s epoch of the night:

    IBI (ms) -> linear resample at 4 Hz (integer arithmetic)
             -> 12th-order IIR band-pass (float-float
                arithmetic, rounded to integer Q16.16 counts)
             -> peak picker: the interval between successive
                peaks (in 4 Hz samples) and half the preceding sample
             -> per-epoch breathing rate over the last four epochs (2 min):
                240 * (number of peak intervals) / (sum of intervals)  [breaths/min],
                its spread, and the mean heart rate from the mean IBI
             -> epoch record {rate, spread, heart rate, motion, valid flag}
             -> night average over the records that are valid, trimmed by 120 epochs
                at both ends (or 180 around the middle for short nights).

`breathing_rate(bedtime_ms, ibi, ibi_ts_ms, motion)` returns `(rate, variability, hr)`
in breaths/min, breaths/min, beats/min. `ibi` are u16 ms values, `ibi_ts_ms` their timestamps (ascending),
`motion` one byte per 30 s epoch starting at `bedtime_ms`. Fewer than 361 epochs return
zeros.

Everything is integer except the IIR, which uses pairs of float32 (hi, lo) with
fused multiply-add in a fixed operation order. The Q16.16
fixed-point helpers round to nearest; overflow returns 0x80000000.
"""
import bisect
import math
import struct

M32 = 0xFFFFFFFF
FIX_MIN = -0x80000000


def _s32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


# --- Q16.16 fixed-point helpers (rounding variant) -------------------------------------------------

def q16_add(a, b):
    r = _s32(a + b)
    return r if not ((a ^ b) >= 0 and (r ^ a) < 0) else FIX_MIN


def q16_sub(a, b):
    r = _s32(a - b)
    return r if not ((a ^ b) < 0 and (r ^ a) < 0) else FIX_MIN


def q16_mul(a, b):
    p = a * b
    hi = p >> 47
    if hi != (0 if p >= 0 else -1):
        return FIX_MIN
    if p < 0:
        p -= 1
    return _s32(((p >> 16) + ((p >> 15) & 1)))


def q16_div(a, b):
    if b == 0:
        return FIX_MIN
    rem, div, quo, bit = abs(a), abs(b), 0, 17
    if div >= 0x100000:
        sd = (div >> 17) + 1
        quo = rem // sd
        rem -= (quo * div) >> 17
    else:  # divisor below 16.0: strip trailing zero nibbles
        while not (div & 0xF) and bit >= 4:
            div >>= 4
            bit -= 4
    while rem and bit >= 0:
        shift = min(32 - rem.bit_length(), bit)
        rem = (rem << shift) & M32
        bit -= shift
        d = rem // div
        rem %= div
        quo = (quo + (d << bit)) & M32
        if d > (M32 >> bit):
            return FIX_MIN
        rem = (rem << 1) & M32
        bit -= 1
    r = ((quo + 1) & M32) >> 1
    return _s32(-r if (a ^ b) < 0 else r)


def q16_sqrt(x):
    num = abs(x) & M32
    res = 0
    bit = 1 << 30 if num >= 0x100000 else 1 << 18
    while bit > num:
        bit >>= 2
    for n in range(2):
        while bit:
            if num >= res + bit:
                num -= res + bit
                res = (res >> 1) + bit
            else:
                res >>= 1
            bit >>= 2
        if n == 0:
            if num > 65535:
                num -= res
                num = ((num << 16) - 0x8000) & M32
                res = ((res << 16) + 0x8000) & M32
            else:
                num = (num << 16) & M32
                res = (res << 16) & M32
            bit = 1 << 14
    if num > res:
        res += 1
    return _s32(-res if x < 0 else res)


# --- float-float arithmetic (two float32 per value) ----------------------------------

def _f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _fma(a, b, c):
    """float32 fused multiply-add: a*b + c with a single rounding."""
    r = math.fma(a, b, c)
    bits = struct.unpack("<Q", struct.pack("<d", r))[0]
    if (bits & 0x1FFFFFFF) == 0x10000000 and math.isfinite(r):  # double result on a float32 tie
        from fractions import Fraction
        ex = Fraction(a) * Fraction(b) + Fraction(c)
        # choose the float32 nearest to the exact value, ties to even
        base = _f32(float(ex))
        cands = {base, struct.unpack("<f", struct.pack("<I", (struct.unpack("<I", struct.pack("<f", base))[0] + 1) & M32))[0],
                 struct.unpack("<f", struct.pack("<I", (struct.unpack("<I", struct.pack("<f", base))[0] - 1) & M32))[0]}
        best = min(cands, key=lambda v: (abs(Fraction(v) - ex),
                                         struct.unpack("<I", struct.pack("<f", v))[0] & 1))
        return best
    return _f32(r)


def _ff_add(a, b):
    ah, al = a
    bh, bl = b
    lo = _f32(al + bl)
    hs = _f32(ah + bh)
    bv = _f32(hs - ah)
    av = _f32(hs - bv)
    eb = _f32(bh - bv)
    ea = _f32(ah - av)
    ea = _f32(eb + ea)
    lo = _f32(lo + ea)
    h = _f32(hs + lo)
    eb = _f32(h - hs)
    return h, _f32(lo - eb)


def _ff_mul(a, b):
    ah, al = a
    bh, bl = b
    lo = _f32(al * bh)
    prod = _f32(ah * bh)
    lo = _fma(ah, bl, lo)
    perr = _fma(ah, bh, -prod)
    lo = _f32(perr + lo)
    h = _f32(prod + lo)
    resid = _f32(h - prod)
    return h, _f32(lo - resid)


def _ff_mul_f(a, c):
    ah, al = a
    prod = _f32(ah * c)
    perr = _fma(ah, c, -prod)
    lo = _fma(al, c, perr)
    h = _f32(prod + lo)
    resid = _f32(h - prod)
    return h, _f32(lo - resid)


def _bits(h, l):
    return (struct.unpack("<f", struct.pack("<I", h))[0], struct.unpack("<f", struct.pack("<I", l))[0])


# (hi, lo) float32 bit patterns of the feedback taps a_1..a_12 (y[n-k]) and the
# feed-forward taps b_2, b_4, ..., b_12 (x[n-k]); b_0 = b_12 (b_odd = 0).
_A = [_bits(h, l) for h, l in (
    (0x411513BC, 0xB27A9F2A), (0xC220C39F, 0x35FF51C0), (0x42D46A02, 0x365187CB), (0xC33F97FD, 0x36E959D5),
    (0x4378A8E2, 0xB6DED1E6), (0xC36E2144, 0x3697D4E4), (0x4329912B, 0x35B814A2), (0xC2B23955, 0xB4A822F6),
    (0x4206D43B, 0xB5F523C9), (0xC10B6168, 0xB4A840C0), (0x3FB0C99B, 0xB2DDA4E7), (0xBDD0145D, 0xB0822E1E))]
_B = [_bits(h, l) for h, l in (
    (0xBAB35876, 0xAE250CF3), (0x3B602E90, 0xAEC8B5C4), (0xBB957460, 0x2F05CE83),
    (0x3B602E90, 0xAEC8B5C4), (0xBAB35876, 0xAE250CF3), (0x396F20B5, 0xACA92451))]
_B0 = _bits(0x396F20B5, 0xACA92451)


class Iir:
    """Input an integer sample, output round(y * 65536) as int32."""

    def __init__(self):
        self.n = 0
        self.x = [0] * 16
        self.y = [(0.0, 0.0)] * 16

    def run(self, sample):
        n = self.n
        s = ((sample & 0xFFFF) ^ 0x8000) - 0x8000
        self.x[n] = s
        acc = _ff_mul_f(_B0, float(self.x[n]))
        for m in range(6):
            acc = _ff_add(acc, _ff_mul(self.y[(n - 1 - 2 * m) & 15], _A[2 * m]))
            acc = _ff_add(acc, _ff_mul(self.y[(n - 2 - 2 * m) & 15], _A[2 * m + 1]))
            acc = _ff_add(acc, _ff_mul_f(_B[m], float(self.x[(n - 2 - 2 * m) & 15])))
        self.y[n] = acc
        self.n = (n + 1) & 15
        v = _fma(acc[0], 65536.0, 0.5)
        if v != v:
            return 0
        if v >= 2147483648.0:
            return 0x7FFFFFFF
        if v <= -2147483648.0:
            return FIX_MIN
        return int(v)


# --- the epoch state machine ---------------------------------------------------------

EPOCH_MS = 30000
MAX_RECORDS = 1800


def _round_half_away(v):
    """Round a Q16.16 value to an integer: half away from zero."""
    return ((v + 0x8000) & M32) >> 16 if v >= 0 else -(((0x8000 - v) & M32) >> 16)


class _State:
    def __init__(self, bedtime_ms):
        self.iir = Iir()
        self.cur = 0                     # index of the current epoch slot (0..3)
        self.rising = 0                  # the integer waveform was last seen rising
        self.idx7 = 0                    # write position in the 16-slot motion ring
        self.qidx = 0
        self.prev = 0x7FFF               # previous integer sample of the waveform
        self.epochs = 0                  # u16, epochs advanced minus one
        self.samples = 0                 # resampled samples fed to the picker
        self.last_peak = 0
        self.t = bedtime_ms
        self.S = [[0, 0, 0, 0] for _ in range(4)]   # per slot: ibi range [a,b), peak range [c,d)
        self.ibiring = [0] * 256
        self.motion = [0] * 16
        self.quality = [0] * 4
        self.peaks = [[0, 0] for _ in range(64)]    # (interval in samples u8, half peak value)
        self.records = []
        # resampler
        self.rs_prev = 0
        self.rs_t = 0
        self.rs_next = 0

    # IBI resampling and peak handler ------------------------------------------------------
    def resample(self, ibi):
        if self.rs_prev:
            prev, a, b = self.rs_prev, self.rs_t, self.rs_next
            end = (a + ibi) & M32
            if b < end:
                slope = _s32(((ibi >> 1) & 0x7FFF) | (((ibi - prev) << 15) & M32))
                slope = (abs(slope) // ibi) * (-1 if slope < 0 else 1) if ibi else 0   # integer division truncating toward zero
                span = b - a
                while True:
                    prod_fix = _s32(span * slope)
                    v = -(((0x4000 - prod_fix) & M32) >> 15) if prod_fix < 1 else ((prod_fix + 0x4000) & M32) >> 15
                    self.handle((prev + v) & M32)
                    span = (span + 250) & M32
                    b = (b + 250) & M32
                    if not b < end:
                        break
            self.rs_next = b
            self.rs_t = end
        self.rs_prev = ibi

    def handle(self, x):
        y = self.iir.run(x)
        cur_v = (y + 0xFFFF if y < 0 else y) >> 16
        cur_v = _s32(cur_v)
        prev = self.prev
        if cur_v < prev:
            if self.rising:
                S = self.S
                d = S[self.cur][3]
                nd = (d + 1) & 0x3F
                if nd != S[(self.cur + 1) & 3][2]:
                    interval = (self.samples - self.last_peak) & M32
                    self.peaks[d] = [min(interval, 255), max(-128, min(127, int(prev / 2)))]
                    S[self.cur][3] = nd
                    self.last_peak = self.samples
            self.rising = 0
        if cur_v > prev:
            self.rising = 1
        self.prev = cur_v
        self.samples = (self.samples + 1) & M32

    def advance(self):
        c = self.cur
        n = (c + 1) & 3
        b, d = self.S[c][1], self.S[c][3]
        self.S[n] = [b, b, d, d]
        self.cur = n
        self.idx7 = (self.idx7 + 1) & 15
        self.motion[self.idx7] = 0
        self.epochs = (self.epochs + 1) & 0xFFFF

    # peak-pair scan over the four slots, oldest first ------------------------------
    def pair_scan(self, visit):
        cum = 0
        for k in range(4):
            c, d = self.S[(self.cur + 1 + k) & 3][2:]
            nxt = (c + 1) & 0x3F
            if nxt == d:
                continue
            i = c
            while True:
                e = self.peaks[i & 0x3F]
                cum += e[0]
                if cum >= 41 and e[1] >= 1:
                    f = self.peaks[nxt]
                    if f[1] >= 1 and f[0] >= 9:
                        visit(f[0])
                i = nxt
                nxt = (i + 1) & 0x3F
                if nxt == d:
                    break

    def motion_sum(self, offs):
        return sum(self.motion[(self.idx7 + o) & 15] for o in offs)


def _epoch(st, T, prev_samples):
    """Body of one epoch after the IBIs of the window were pushed; returns the new
    sample counter and appends a record when the epoch produces one."""
    cur = st.cur
    S = st.S
    # mean IBI of the last four epochs -> heart rate
    cnt = sm = 0
    for k in range(4):
        a, b = S[(cur + 1 + k) & 3][:2]
        while a != b:
            sm += st.ibiring[a]
            cnt += 1
            a = (a + 1) & 0xFF
    q10 = None
    if cnt:
        t = (((sm << 10) & M32) + (cnt >> 1)) & M32
        if t >= cnt:
            q10 = t // cnt
    if q10 is None:
        q10 = 0xB0800
    period_fix = (((q10 << 6) & M32) + 0x8000) & 0xFFFF0000
    hr = q16_mul(q16_div(0x75300000, _s32(period_fix)), 0x20000)
    if not hr > 0x230000:
        hr = 0x230000
    hr_half = min(_round_half_away(q16_mul(hr, 0x20000)), 255)

    a, b = S[cur][0], S[cur][1]
    samples = st.samples
    if a != b:
        i = a
        while True:
            st.resample(st.ibiring[i])
            i = (i + 1) & 0xFF
            if i == S[cur][1]:
                break
        samples = st.samples
    else:
        samples = prev_samples
    br_b, var_b = 106, 42
    if samples - prev_samples >= 41:
        n_pk = [0]
        tot = [0]

        def acc(v):
            n_pk[0] += 1
            tot[0] += v
        st.pair_scan(acc)
        n, s = n_pk[0], tot[0]
        if n and s:
            q8 = (n * 0xF000 + (s & ~1)) & M32
            if s <= q8:
                q8 //= s
                mean_br = q16_div(_s32(q8 << 16), 0x1000000)
                sq = [0, 0]

                def var(v):
                    d = q16_sub(q16_div(0xF00000, v << 16), mean_br)
                    sq[0] = q16_add(sq[0], q16_mul(d, d))
                    sq[1] += 1
                st.pair_scan(var)
                var_fix = 0
                if sq[1] >= 2 and sq[0] >= 1:
                    var_fix = q16_div(sq[0], sq[1] << 16)
                sd = q16_sqrt(var_fix)
                r = _round_half_away(q16_mul(sd, 0x80000))
                var_b = max(0, min(r, 255))
                br_b = ((q8 + 16) >> 5) & 0xFF
    # record
    a, b = S[cur][0], S[cur][1]
    if b == a or b == ((a + 1) & 0xFF):
        rec = (170, 80, 16, 15, 2)
    else:
        if st.epochs >= 10:
            c = 0
            for k in range(12):
                s = min(st.motion_sum((-k, -k - 1, -k - 2, -k - 3)), 255)
                c += s > 2
            ok = c < 4 or not (any(q > 27 for q in st.quality))
        else:
            ok = min(st.motion_sum((-2, -3, -4, -5)), 255) <= 6
        flag = 1 if ok else 0
        msum = min(st.motion_sum((0, -1, -2, -3)), 255)
        st.quality[st.qidx] = var_b
        st.qidx = (st.qidx + 1) & 3
        rec = (max(hr_half, 0), br_b, var_b, msum, flag)
    if len(st.records) < MAX_RECORDS:
        st.records.append((T - EPOCH_MS,) + rec)
    return samples


def epoch_records(bedtime_ms, ibi, ibi_ts_ms, motion):
    """Run the epoch state machine; returns the record list, one tuple per epoch after
    the first: (start_ms, heart_rate_half_bpm, rate_eighth_bpm, spread_eighth_bpm,
    motion_sum, flag) with flag 1 = usable, 0 = rejected for motion/quality, 2 = no data."""
    st = _State(bedtime_ms)
    n_ibi, n_mot = len(ibi), len(motion)
    ordered = all(ibi_ts_ms[k] <= ibi_ts_ms[k + 1] for k in range(n_ibi - 1))
    prev_samples = 0
    st.advance()
    st.epochs = 0
    T = bedtime_ms
    for i in range(n_mot):
        st.motion[st.idx7] = motion[i]
        pushed = 0
        lo = T - EPOCH_MS
        # the scan starts at the first timestamp and stops at the first one after T
        j = bisect.bisect_left(ibi_ts_ms, lo) if ordered else 0
        while j < n_ibi:
            t = ibi_ts_ms[j]
            j += 1
            if t > T:
                break
            if t < lo:
                continue
            if ibi[j - 1] <= 1999:
                pushed += 1
                s = st.S[st.cur]
                nb = (s[1] + 1) & 0xFF
                if nb != st.S[(st.cur + 1) & 3][0]:
                    st.ibiring[s[1]] = ibi[j - 1]
                    s[1] = nb
        if not pushed:
            st.motion[st.idx7] = 0
        if st.epochs:
            prev_samples = _epoch(st, T, prev_samples)
        st.advance()
        T += EPOCH_MS
    return st.records


def breathing_rate(bedtime_ms, ibi, ibi_ts_ms, motion):
    """Night averages `(breathing_rate, variability, heart_rate)`; zeros when the night
    has fewer than 361 epochs or no valid record."""
    return _summarize(epoch_records(bedtime_ms, ibi, ibi_ts_ms, motion), bedtime_ms, len(motion))


def _summarize(records, bedtime_ms, n):
    if n < 361:
        return 0.0, 0.0, 0.0
    first = 0
    for k, r in enumerate(records):
        if r[0] >= bedtime_ms - EPOCH_MS:
            first = k
            break
    if n >= 601:
        lo, hi = first + 120, n + first - 120
    else:
        lo, hi = n // 2 - 180, n // 2 + 180
    hi = min(hi, MAX_RECORDS)
    if hi <= lo:
        return 0.0, 0.0, 0.0
    c = sum_hr = sum_br = sum_var = 0
    for k in range(lo, hi):
        if k < len(records) and records[k][5] != 2:
            hr, br, var = records[k][1:4]
            c += 1
            sum_hr += hr
            sum_br += br
            sum_var += var
    if not c:
        return 0.0, 0.0, 0.0
    h = c >> 1
    return ((((h + sum_br) // c) & 0xFF) / 8.0, (((h + sum_var) // c) & 0xFF) / 8.0,
            (((h + ((sum_hr << 2) & M32)) // c) & 0xFFFF) / 8.0)
