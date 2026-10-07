#!/usr/bin/env python3
"""Bedtime detector.

    detect_bedtime(motion, motion_ts, temp, hr_periods, previous=(), report_naps=True)

The detector is a streaming scan over 30 s bins with a state
that is re-initialized on every call. Its parts:

- argument checks and glue (`detect_bedtime`).
- normalization: relabels quiet bins that have no temperature nearby.
- the scan (`find`).
- `update_counters`: sliding-window counters over the last 240 minutes.
- `estimate_start`, `estimate_end`: look-back and look-ahead over the same ring.
- candidate insertion into the output vector.

Inputs: `motion` one signed byte per 30 s bin
(0..30 seconds of motion, 31 = filler, 255 = charging), `motion_ts` the bin start in Unix ms,
`temp` one u16 per 60 s bin in centi-degrees (0 = none), `hr_periods` a list of (start, end) ms,
`previous` a list of (start, end, major). `avg_temp` is only checked to be nonzero. Output: list of (start_ms, end_ms, major).
"""
import math
import struct

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
MIN_MS = 60_000
H_MS = 3_600_000
ERR_ARGS = 13

# The fourteen byte counters of the module state, saturating at 0 and 255. Minute
# classes: 0 charging, 1 quiet, 2 moderate, 3 active. "Extreme" is class 0 or 3. Depth 0 is the newest
# minute and the window holds depths 0..239. Each counter equals the number of minutes of its class
# in a range of depths (checked at every minute on 3599 steps of a 60 h real window, one
# matching definition per counter, no other (class set, range) pair fits).

#   index  name            counts                                   depths
#     0    QUIET_240       quiet minutes                             0..239
#     1    QUIET_OLD180    quiet minutes in the oldest 180           60..239
#     2    QUIET_OLD90     quiet minutes in the oldest 90            150..239
#     3    QUIET_OLD60     quiet minutes in the oldest 60            180..239
#     4    QUIET_OLD15     quiet minutes in the oldest 15            225..239
#     5    QUIET_OLD5      quiet minutes in the oldest 5             235..239
#     6    EXTREME_OLD60   extreme minutes in the oldest 60          180..239
#     7    QUIET_NEW181    quiet minutes in the newest 181           0..180
#     8    QUIET_NEW91     quiet minutes in the newest 91            0..90
#     9    QUIET_NEW16     quiet minutes in the newest 16            0..15
#    10    QUIET_NEW6      quiet minutes in the newest 6             0..5
#    11    EXTREME_NEW6    extreme minutes in the newest 6           0..5
#    12    EXTREME_NEW61   extreme minutes in the newest 61          0..60
#    13    WARM            minutes entering with T > hi, minus those leaving with T > (lo + hi) / 2
#                          (the two tests differ, so this is not a count over a fixed range)

# The oldest minute is the candidate onset: a bout starts when, after it, 100 of 180 minutes are
# quiet, 43 of 60, 11 of 15 and 4 of 5, with at most 10 extreme minutes in the oldest 60. That is why
# a bout is recognized about 4 h after its start. The newest-side counters (7..12) drive
# `estimate_start`'s walk back (it reads QUIET_240, QUIET_NEW91, QUIET_NEW16, QUIET_NEW6 and WARM) and the ring's 4 h look-ahead.
(QUIET_240, QUIET_OLD180, QUIET_OLD90, QUIET_OLD60, QUIET_OLD15, QUIET_OLD5, EXTREME_OLD60, QUIET_NEW181,
 QUIET_NEW91, QUIET_NEW16, QUIET_NEW6, EXTREME_NEW6, EXTREME_NEW61, WARM) = range(14)


def sign_extend_16(x):
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def as_i8(x):
    x &= 0xFF
    return x - 256 if x & 0x80 else x


def as_i32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class Cand:
    """One 32-byte candidate entry: start, end, mark, state counter."""
    __slots__ = ("f0", "f8", "f10", "f18")

    def __init__(self):
        self.clear()

    def clear(self):
        self.f0 = self.f8 = self.f10 = self.f18 = 0

    def copy_from(self, o):
        self.f0, self.f8, self.f10, self.f18 = o.f0, o.f8, o.f10, o.f18


class State:
    """Detector state, reset at the top of `bedtime_find`."""

    def __init__(self):
        self.in_bout = 0
        self.cur = 0              # motion sum of the current minute (signed byte)
        self.prev = 0             # motion sum of the previous minute
        self.idx = 0              # ring head
        self.temp = 0             # current temperature
        self.f08 = 0              # bout start estimate
        self.f10 = 0              # timestamp at bout start
        self.f18 = 0              # bout end estimate
        self.f20 = 0              # timestamp of the first resting-HR sample of a pending candidate
        self.n_cand = 0
        self.samp = 0             # sample counter; odd samples close a minute
        self.cnt = [0] * 14       # window counters
        self.cnt[EXTREME_OLD60] = 60
        self.cnt[EXTREME_NEW6] = 5
        self.cnt[EXTREME_NEW61] = 60
        self.cand = [Cand() for _ in range(5)]
        self.ring2 = bytearray([0xFF] * 64)   # 2-bit minute class, 256 slots
        self.ring1 = bytearray(32)            # 1-bit warm flag, 256 slots
        self.avg = 0.0            # running mean temperature (single precision)
        self.n_t = 0              # its count


def state_bytes(st):
    """The state serialized as a 0x148-byte block."""
    b = bytearray(0x148)
    b[0] = st.in_bout
    b[2] = st.cur & 0xFF
    b[3] = st.prev & 0xFF
    struct.pack_into("<HH", b, 4, st.idx, st.temp)
    struct.pack_into("<QQQQ", b, 8, st.f08, st.f10, st.f18, st.f20)
    struct.pack_into("<II", b, 0x28, st.n_cand, st.samp)
    b[0x30:0x3E] = bytes(st.cnt)
    for k, e in enumerate(st.cand):
        struct.pack_into("<QQQI", b, 0x40 + 32 * k, e.f0, e.f8, e.f10, e.f18)
    b[0xE0:0x120] = st.ring2
    b[0x120:0x140] = st.ring1
    struct.pack_into("<fI", b, 0x140, st.avg, st.n_t)
    return bytes(b)


def cls_at(st, p):
    p &= M32
    return (st.ring2[(p >> 2) & 0x3F] >> ((p & 3) * 2)) & 3


def bit_at(st, p):
    p &= M32
    return (st.ring1[(p >> 3) & 0x1F] >> (p & 7)) & 1


def update_counters(st, cnt, add_cls, add_hot, off, rem_cls, rem_hot, d):
    """Update the counters. With d=1, the sample (add_cls, add_hot) enters the window
    and (rem_cls, rem_hot) leaves it; d=0 is the inverse. `off` places the lookahead patterns."""
    d &= 1
    add_hot &= 1
    rem_hot &= 1

    def up(k):   # moves with the window entering sample
        c = cnt[k]
        if d:
            if c != 255:
                cnt[k] = c + 1
        elif c != 0:
            cnt[k] = c - 1

    def down(k):
        c = cnt[k]
        if d:
            if c != 0:
                cnt[k] = c - 1
        elif c != 255:
            cnt[k] = c + 1

    off &= M32
    idx = st.idx

    def inr(k):
        return ((off + k) & 0xFFFF) <= 0xFF

    def at(k):
        return cls_at(st, idx + off + k)

    if rem_hot:
        down(13)
    if rem_cls == 0 or rem_cls == 3:
        down(6)
    elif rem_cls == 1:
        for k in range(6):
            down(k)
    if add_hot:
        up(13)
    one = add_cls == 1
    if one:
        up(0)
    if inr(0x104) and at(0x105) == 1:
        up(5)
    if inr(0x10E) and at(0x10F) == 1:
        up(4)
    if inr(0x13B):
        v = at(0x13C)
        if v == 0 or v == 3:
            up(6)
        if v == 1:
            up(3)
    else:
        up(6)
    if inr(0x159) and at(0x15A) == 1:
        up(2)
    if inr(0x1B3) and at(0x1B4) == 1:
        up(1)
    if inr(0x1E9):
        v = at(0x1EA)
        if v == 0 or v == 3:
            down(11)
    else:
        down(11)
    if ((off + 0x1B2) & 0xFFFF) <= 0xFF:
        v = cls_at(st, idx + off + 0x1B3)
        if v == 0 or v == 3:
            down(12)
    else:
        down(12)
    if inr(0x1E9) and at(0x1EA) == 1:
        down(10)
    if inr(0x1DF) and at(0x1E0) == 1:
        down(9)
    if inr(0x194) and at(0x195) == 1:
        down(8)
    if inr(0x13A) and cls_at(st, off + 0x13B + idx) == 1:
        down(7)
    if one:
        up(10)
        up(9)
        up(8)
        up(7)
    elif add_cls == 0 or add_cls == 3:
        up(11)
        up(12)


def estimate_start(st, ts, flag):
    """Walk back from the newest minute until the window
    statistics (on a copy of the counters) look like the start of a sleep bout."""
    cnt = list(st.cnt)
    warm_pen = min(cnt[WARM] * 2, 200)
    t1 = warm_pen + 0x14758
    t8 = warm_pen + 0x7A44
    t9 = warm_pen + 0x10CC
    t10 = warm_pen + 0xDDE
    idx = st.idx
    j = 0xFFFF
    found = False
    while True:
        if ((j + 0xE) & 0xFFFF) > 0xFF:
            cls_old, hot_old = 3, 1
        else:
            cls_old = cls_at(st, idx + j + 0xF)
            hot_old = bit_at(st, idx + j + 0xF)
        hot_new = bit_at(st, idx + j + 0x100)
        cls_new = cls_at(st, idx + j + 0x100)
        update_counters(st, cnt, cls_new, hot_new, (j - 0xF1) & M32, cls_old, hot_old, 0)
        ca = cls_at(st, idx + j + 0x100)
        cb = cls_at(st, idx + j + 0xFF)
        skip = (j & 0xFFFF) >= 0xFF4D and (flag & 1)
        if not skip:
            ok = (cnt[QUIET_240] * 1000 <= t1 and cnt[QUIET_NEW91] * 1000 <= t8 and cnt[QUIET_NEW16] * 1000 <= t9)
            f2 = ca != 1 and cb != 1 and cnt[QUIET_NEW6] * 1000 <= t10
            if ca == 0 or (ok and f2):
                found = True
                break
        j = (j - 1) & M32
        if 0xFF01 >= (j & 0xFFFF):
            break
    if not found:
        est = 0
    else:
        m = 1
        sel = 0
        while True:
            back = (j - m) & M32
            if ((back + 0xFF) & 0xFFFF) > 0xFF:
                sel = m
                break
            c = cls_at(st, idx + 0x100 + back)
            if c == 0 or c == 3:
                sel = m
                break
            m16 = m & 0xFFFF
            m += 1
            if not m16 < 0x59:
                sel = 0
                break
        est = (ts + (sign_extend_16(j + 1) - (sel & 0xFFFF)) * MIN_MS) & M64
    fallback = MIN_MS if ts < 14_400_000 else (ts - 14_400_000) & M64
    return est if ((est - 1) & M64) < ts else fallback


def estimate_end(st, ts, hint):
    """Look ahead up to 4 h by feeding class 3 samples."""
    res = st.f18
    if not st.in_bout:
        if st.cand[0].f0 != 0:
            cend = st.cand[0].f8
            res = ts if cend == 0 else cend
        return res
    cnt = list(st.cnt)
    if cnt[QUIET_240] != 255:
        cnt[QUIET_240] += 1
    idx = st.idx
    back = -241
    lim = (ts + 14_400_000) & M64
    remain = 241
    while True:
        step = back + 1
        p = idx + step + 0x100
        update_counters(st, cnt, 3, 0, step & M32, cls_at(st, p), bit_at(st, p), 1)
        cont = lim < hint
        if not cont:
            hi_byte = ((step & M32) >> 8) & 0xFF
            cb = 3 if hi_byte < 0xFF else cls_at(st, idx + step + 0x101)
            ca = cls_at(st, idx + step + 0x100)
            if ca != 0:
                if (((st.f08 + 10_800_000) & M64) >= lim or cnt[QUIET_240] > 0x54 or cnt[QUIET_OLD90] > 0x1F or cnt[QUIET_OLD15] > 4
                        or cb == 1 or ca == 1 or cnt[QUIET_OLD5] > 1):
                    cont = True
        if not cont:
            best, last1, k = 0, 0, 0
            found_k = 0
            base = idx + 0x100
            while True:
                found_k = (step + (k & 0xFF)) & M32
                if ((found_k + 0xFF) & 0xFFFF) > 0xFF:
                    found_k = remain
                    break
                c = cls_at(st, base + found_k)
                found_k = k
                if c == 0 or c == 3:
                    break
                if c == 1:
                    last1 = k
                    found_k = best
                else:
                    found_k = best
                prev_k = k & 0xFF
                k += 1
                if prev_k >= 0x3B:
                    break
            run_len = found_k if (found_k & 0xFF) else last1 + 1
            res = (ts + (back + (run_len & 0xFF) + 1) * MIN_MS) & M64
            break
        if back == -1:
            break
        back += 1
        remain -= 1
        lim = (lim - MIN_MS) & M64
    return res if ((res - 1) & M64) < ts else ts


class Out:
    """Output vector of 24-byte entries plus the push filter state."""

    def __init__(self, previous):
        mx, fl = 0, 0
        for _, e, _ in previous:
            if e > mx:
                mx, fl = e, 1
        self.st_end, self.st_has = mx, fl
        self.v = []

    def push(self, start, end, flag):
        if start == 0 or end == 0:
            return
        if self.st_has and ((self.st_end + 900_000) & M64) > end:
            return
        s_adj = (end - 54_000_000) & M64 if ((start + 54_000_000) & M64) < end else start
        self.v.append([s_adj, end, flag & 1])
        self.st_end = end
        self.st_has = 1


def find(M, T, temp, hr, previous, naps, state_out=None):
    """`bedtime_find`. M: signed-byte motion list as 0..255, T: ms timestamps."""
    st = State()
    out = Out(previous)
    temp2 = []
    for t in temp:
        temp2 += [t, t]
    ts = 0
    n = len(M)
    for i in range(n):
        ts = T[i]
        mu = M[i] & 0xFF
        ms = as_i8(mu)
        if ms != 31 and ms >= 0:
            if i < len(temp2):
                st.temp = temp2[i]
        else:
            st.temp = 0
        in_hr = 0
        for s, e in hr:
            if s < ts and e >= ts:
                in_hr = 1
                break
        temp_v = st.temp
        if temp_v:
            n_old = st.n_t
            st.n_t = (n_old + 1) & M32
            st.avg = f32(f32(math.fma(st.avg, f32(n_old), f32(temp_v))) / f32(st.n_t))
        a = 0 if st.avg < 0 else min(int(st.avg), M32)
        if st.n_t > 0x5A0:
            hi, lo = (a + 250) & M32, (a - 250) & M32
        else:
            hi, lo = 0xD7A, 0xC80
        if mu < 128:
            cur = as_i8(st.cur)
            if cur < 0:
                mot = cur & 0xFF
            else:
                mot = cur + mu
                st.cur = mot & 0xFF
        else:
            mot = 0xFF
            st.cur = 0xFF
        c = st.samp
        st.samp = (c + 1) & M32
        closed = False
        if c & 1:
            if mot & 0x80:
                cn = 0
            else:
                d = (temp_v - lo) & M32
                sd = sign_extend_16(d)
                sd3 = (3 * sd) & M32
                t = (sd3 + 2) & M32 if as_i32(sd3) + 1 < 0 else (sd3 + 1) & M32
                a2 = sign_extend_16(((t >> 1) + 100) & M32)
                thr_a = a2 if a2 > 100 else 100
                thr_a = thr_a if thr_a < 700 else 700
                c4 = sign_extend_16((4 * d + 0x960) & M32)
                thr_b = c4 if c4 > 1000 else 1000
                thr_b = thr_b if thr_b < 2800 else 2800
                s100 = (mot * 100) & M32
                p100 = as_i8(st.prev) * 100
                if s100 <= thr_a and p100 <= thr_a:
                    cn = 1
                else:
                    cn = 3 if s100 >= thr_b else 2
            idx = st.idx
            shift = (idx & 3) * 2
            b2 = (idx >> 2) & 0x3F
            st.ring2[b2] = (st.ring2[b2] & ~(3 << shift) & 0xFF) | (cn << shift)
            b1 = (idx >> 3) & 0x1F
            if temp_v > ((lo + hi) & M32) >> 1:
                st.ring1[b1] |= 1 << (idx & 7)
            else:
                st.ring1[b1] &= ~(1 << (idx & 7)) & 0xFF
            hot = 1 if hi < temp_v else 0
            oc = cls_at(st, idx + 0x10)
            ow = bit_at(st, idx + 0x10)
            st.idx = (idx + 1) & 0xFFFF
            update_counters(st, st.cnt, cn, hot, 0xFF0F, oc, ow, 1)
            st.prev = st.cur
            st.cur = 0
            cnt = st.cnt
            if not (mu & 0x80) and not st.in_bout:
                if cnt[QUIET_OLD180] >= 100 and cnt[QUIET_OLD60] >= 43 and cnt[EXTREME_OLD60] <= 10 and cnt[QUIET_OLD15] >= 11 and cnt[QUIET_OLD5] >= 4:
                    st.in_bout = 1
                    st.f10 = ts
                    st.f08 = estimate_start(st, ts, 1)
                    st.f18 = 0
                    st.n_cand = 0
                    for e in st.cand:
                        e.clear()
            if st.in_bout:
                idx = st.idx
                c15 = cls_at(st, idx + 0xF)
                go = True
                if c15 != 0:
                    if not (((st.f08 + 10_800_000) & M64) < ((ts - 14_400_000) & M64)):
                        go = False
                    elif cnt[QUIET_240] > 0x54 or cnt[QUIET_OLD90] > 0x1F or cnt[QUIET_OLD15] > 4:
                        go = False
                    elif cls_at(st, idx + 0x10) == 1 or c15 == 1 or cnt[QUIET_OLD5] > 1:
                        go = False
                if go:
                    best, last1, k = 0, 0, 0
                    end_k = 0
                    while True:
                        c = cls_at(st, idx + ((k + 16) & 0xFF))
                        end_k = k
                        if c == 0 or c == 3:
                            break
                        if c == 1:
                            last1 = k
                        end_k = best
                        prev_k = k & 0xFF
                        k += 1
                        if prev_k >= 0x3B:
                            break
                    run_end = end_k if (end_k & 0xFF) else last1 + 1
                    st.in_bout = 0
                    end_ts = (ts - 14_400_000 + (run_end & 0xFF) * MIN_MS) & M64
                    st.f18 = end_ts
                    if end_ts > st.f08:
                        finalize(st, out, ts, 1, st.f08)
                        closed = True
        if closed:
            continue
        has_hr = 0 if mu == 0xFF else in_hr
        ci = st.n_cand
        e = st.cand[ci]
        if e.f8 == 0:
            if st.f20 != 0:
                idle = ((st.f20 + 600_000) & M64) <= ts and not st.in_bout
                if has_hr & 1:
                    if idle and e.f0 == 0:
                        st.cand[st.n_cand].f0 = estimate_start(st, ts, 0)
                    ci = st.n_cand
                elif not idle:
                    e.clear()
                    st.f20 = 0
                else:
                    idx = st.idx
                    k = 0
                    while k < 6 and cls_at(st, idx + 0xFB + k) not in (0, 3):
                        k += 1
                    v = max(k, 1) if k < 6 else 6
                    pre = (300_000 - v * MIN_MS) & M64
                    ahead = (ts + v * MIN_MS) & M64
                    lo_c = pre > ts and v < 5
                    cend = 0 if lo_c else (ahead - 300_000) & M64
                    f0 = e.f0
                    e.f8 = cend
                    if cend >= f0 and ci > 3:
                        st.f20 = 0
                    elif cend >= f0 and (ci == 0 or f0 > st.cand[ci - 1].f8):
                        ci += 1
                        st.n_cand = ci
                        st.f20 = 0
                    else:
                        e.clear()
                        st.f20 = 0
            elif in_hr:
                st.f20 = ts
        if ci == 0:
            continue
        c0 = st.cand[0]
        first_end = c0.f8
        if first_end > ts:
            continue
        if ((first_end + 18_000_000) & M64) > ts and not (naps & 1):
            continue
        cc = c0.f18
        c0.f18 = (cc + 1) & M32
        if cc == 0:
            c0.f10 = ts
        elif cc == 3:
            for k in range(4):
                st.cand[k].copy_from(st.cand[k + 1])
            if ci < 4 or (ci == 4 and st.cand[4].f8 == 0):
                ci -= 1
                st.n_cand = ci
            for k in range(min(ci, 4), 5):
                st.cand[k].clear()
        elif cc == 1:
            finalize(st, out, ts, 0, st.f08)
    if state_out is not None:
        state_out.append(st)
    # after the last bin
    v = out.v
    if len(v) == 0 or st.in_bout:
        c0 = st.cand[0]
        open_end = st.f08 if c0.f0 == 0 else c0.f0
        if open_end != 0 and not any(open_end <= y[1] and y[2] for y in v):
            close_end = estimate_end(st, ts, open_end)
            if ((close_end - open_end) & M64) > 899_999 or c0.f0 == 0:
                out.push(open_end, close_end, 1 if c0.f0 == 0 else 0)
    keep = []
    for x in out.v:
        if x[2] or not any(y[2] and x[0] <= y[1] and y[0] <= x[1] for y in out.v):
            keep.append(x)
    if not keep:
        return []
    keep.sort(key=lambda e: e[0])
    res = []
    cur = list(keep[0])
    for e in keep[1:]:
        if e[0] > cur[1]:
            res.append(cur)
            cur = list(e)
        else:
            if e[1] > cur[1]:
                cur[1] = e[1]
            cur[2] = 1 if (cur[2] | e[2]) else 0
    res.append(cur)
    for e in res:
        if ((e[0] + 54_000_000) & M64) < e[1]:
            e[0] = (e[1] - 54_000_000) & M64
    return [tuple(e) for e in res]


def finalize(st, out, ts, major, fallback):
    """Close a candidate period and push it."""
    f0 = st.cand[0].f0
    start = fallback if f0 == 0 else f0
    end = estimate_end(st, ts, start)
    if not (major & 1) and ((end - start) & M64) <= 899_999:
        return
    out.push(start, end, major)


def normalize(M, T, temp, temp_ts):
    """Normalize the inputs. Returns (rc, M, T)."""
    if len(M) != len(T):
        return ERR_ARGS, [], []
    if not any(1 <= b <= 30 for b in M) and not any(temp):
        return 0, [], []
    n = min(len(temp), len(temp_ts)) & M32
    q = sorted(temp_ts[i] for i in range(n) if temp[i] != 0)
    last = 0
    for i, b in enumerate(M):
        if 1 <= b <= 30 and T[i] > last:
            last = T[i]
    for i in range(n):
        if temp[i] != 0 and temp_ts[i] > last:
            last = temp_ts[i]
    M, T = list(M), list(T)
    if last:
        while T and T[-1] > last:
            T.pop()
            M.pop()
    if not M:
        return 0, M, T
    import bisect
    for i in range(len(M)):
        if M[i] == 0:
            t = T[i]
            lo = t - 300_000 if t >= 300_000 else 0
            j = bisect.bisect_left(q, lo)
            if j == len(q) or q[j] > (t + 330_000) & M64:
                M[i] = 31
    return 0, M, T


def detect_bedtime(motion, motion_ts, temp, temp_ts, hr_periods, avg_temp=1, previous=(), report_naps=True,
                   capacity=200):
    """Bedtime detection. Returns (rc, periods); rc 13 means rejected arguments."""
    if not motion or avg_temp & 0xFFFF == 0 or not hr_periods:
        return ERR_ARGS, []
    rc, M, T = normalize([m & 0xFF for m in motion], list(motion_ts), temp, temp_ts)
    if rc:
        return rc, []
    if not M:
        return 0, []
    res = find(M, T, temp, hr_periods, previous, 1 if report_naps else 0)
    return 0, res[:capacity]


def sleep_period_check_overlap(a, b):
    """`sleep_period_check_overlap` on two (start, start_tz, end, end_tz) records; signed
    compares of fields 0 and 2."""
    a0, a2, b0, b2 = a[0], a[2], b[0], b[2]
    if not a0 < b0 and a0 < b2:
        return True
    if not a2 < b0 and a2 < b2:
        return True
    if not b0 < a0 and b0 < a2:
        return True
    return b2 >= a0 and b2 < a2


def merge_periods(periods, capacity):
    """Sort 32-byte period records by field 0 (signed) and merge the
    overlapping ones. The scan starts at the first record again, so a record that does not overlap
    itself (start >= end) is output twice. Records with
    equal starts are ordered by input position. Needs at least one record."""
    p = sorted((list(x) for x in periods), key=lambda x: x[0])
    out = []
    cur = list(p[0])
    for nxt in p:
        if sleep_period_check_overlap(cur, nxt):
            if nxt[0] <= cur[0]:
                cur[0], cur[1] = nxt[0], nxt[1]
            if nxt[2] >= cur[2]:
                cur[2], cur[3] = nxt[2], nxt[3]
        else:
            out.append(cur)
            cur = list(nxt)
    out.append(cur)
    return [tuple(x) for x in out[:capacity]]


def clear_minor_periods_ms(st, a, b):
    """Drop the candidate entries of the module state that
    overlap [a, b] (ms). Acts on the state left by the last `find` call."""
    if a == 0 or b <= a:
        return
    i = 0
    while i < 5:
        e = st.cand[i]
        f0, f8 = e.f0, e.f8
        remove = False
        if f0 <= a and f8 >= a:
            remove = True
        else:
            apart_hi = 1 if f8 < b else (1 if f0 > b else 0)
            apart_lo = 1 if f8 > b else (1 if f8 < a else 0)
            if f0 <= b and f0 >= a:
                remove = True
            elif apart_hi == 0:
                remove = True
            elif not (apart_lo & 1):
                remove = True
        if not remove:
            i += 1
            continue
        if i != 4:
            for j in range(i, 4):
                st.cand[j].copy_from(st.cand[j + 1])
        n = st.n_cand
        dec = False
        if n > i:
            if n < 4:
                dec = True
            elif n == 4 and st.cand[4].f8 == 0:
                dec = True
        elif n == 4 and st.cand[4].f8 == 0:
            dec = True
        if dec:
            n -= 1
            st.n_cand = n
        for k in range(min(n, 4), 5):
            st.cand[k].clear()


if __name__ == "__main__":
    print(__doc__)
