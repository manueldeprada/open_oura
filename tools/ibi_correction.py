#!/usr/bin/env python3
"""IBI correction.

Every ring IBI `(timestamp_ms, ibi_ms, amplitude)` is fed to the corrector, which emits
corrected events `(timestamp_ms, ibi_ms, amplitude, validity)`, three beats late. See docs/algorithms/metrics.md for the algorithm in words.

    c = IbiCorrection()
    events = c.feed(ts, ibi, amp)          # events emitted by this beat
    out = correct_ibis(beats)              # a whole night, one fresh state
    out = correct_stateless(ts, ibi, amp, sc_ts, sc_states, wear_ts, wear_states)

Everything is integer except the bpm tests, which use float32 arithmetic (`numpy.float32`;
one IEEE operation per source operation, no fused multiply-add). Comparisons with NaN
operands are defined by the functions `greater`, `at_most`, `less` and their negations below,
because zeroed window entries make the 7-beat variance produce inf and NaN.
"""
import math
import struct

import numpy as np

M16, M32, M64 = 0xFFFF, 0xFFFFFFFF, (1 << 64) - 1
STATE_SIZE = 0x10F8
RING_CAP = 64

F = np.float32
K60000 = F(60000.0)
K7, K6 = F(7.0), F(6.0)


class Abort(Exception):
    """Ring buffer index out of range or empty read."""


def _s32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


def _s16(x):
    x &= M16
    return x - (1 << 16) if x & 0x8000 else x


def _s8(x):
    x &= 0xFF
    return x - 256 if x & 0x80 else x


# ---- float32 helpers ---------------------------------------------------------------

def div32(a, b):
    with np.errstate(all="ignore"):
        return F(a) / F(b)


def add32(a, b):
    with np.errstate(all="ignore"):
        return F(a) + F(b)


def sub32(a, b):
    with np.errstate(all="ignore"):
        return F(a) - F(b)


def mul32(a, b):
    with np.errstate(all="ignore"):
        return F(a) * F(b)


def abs_diff(a, b):
    with np.errstate(all="ignore"):
        return abs(F(a) - F(b))


# Float comparisons. A NaN operand makes every ordered comparison (greater, at_most, less, equal)
# false and every negated one (not_greater, not_less) true.

def greater(a, b):
    return float(a) > float(b)


def not_greater(a, b):
    return not float(a) > float(b)


def at_most(a, b):
    return float(a) <= float(b)


def less(a, b):
    return float(a) < float(b)


def not_less(a, b):
    return not float(a) < float(b)


def equal(a, b):
    return float(a) == float(b)


def _sat(x, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return max(lo, min(hi, x))


def round_away(v, bits=32):
    """Round to nearest, ties away from zero, saturating; NaN gives 0."""
    x = float(v)
    if x != x:
        return 0
    if math.isinf(x):
        return _sat(1 << 70 if x > 0 else -(1 << 70), bits)
    r = int(math.floor(abs(x) + 0.5))
    return _sat(-r if x < 0 else r, bits)


def to_int(v, bits=32):
    x = float(v)
    if x != x:
        return 0
    if math.isinf(x):
        return _sat(1 << 70 if x > 0 else -(1 << 70), bits)
    return _sat(int(x), bits)


def bpm(ibi):
    return div32(K60000, F(ibi))


# ---- ring buffer entry and ring -------------------------------------------------

class Entry:
    """32-byte ring entry. ts0 is the input timestamp, ts1 the corrected one; `ind` is the
    validity indicator reported to callers; `flag` marks a beat that passed the beat tests."""
    __slots__ = ("ts0", "ts1", "ibi", "amp", "ibi_c", "amp_c", "ind", "flag", "vin")

    def __init__(self):
        self.ts0 = self.ts1 = 0
        self.ibi = self.amp = self.ibi_c = self.amp_c = 0
        self.ind = self.flag = self.vin = 0

    def copy(self):
        e = Entry()
        for k in Entry.__slots__:
            setattr(e, k, getattr(self, k))
        return e



class Ring:
    """64 entries; push over a full ring drops the oldest."""

    def __init__(self):
        self.data = [Entry() for _ in range(RING_CAP)]
        self.head = self.tail = 0
        self.full = 0

    def is_empty(self):
        return not self.full and self.head == self.tail

    def count(self):
        return RING_CAP if self.full else (self.head + RING_CAP - self.tail) % RING_CAP

    def push(self, e):
        self.data[self.head] = e.copy()
        if self.full:
            self.tail = 0 if self.tail + 1 == RING_CAP else self.tail + 1
        self.head = 0 if self.head + 1 == RING_CAP else self.head + 1
        self.full = 1 if self.head == self.tail else 0

    def pop(self):
        if self.is_empty():
            return None
        e = self.data[self.tail]
        self.full = 0
        self.tail = 0 if self.tail + 1 == RING_CAP else self.tail + 1
        return e.copy()

    def rfh(self, idx):
        """Read from the head: idx 0 is the newest entry, -1 the one before."""
        if self.is_empty():
            raise Abort("read from empty ring")
        i = _s8(idx)
        if abs(i) >= RING_CAP:
            raise Abort("ring index")
        j = i + RING_CAP + self.head - 1
        if j < 0:
            raise Abort("ring index")
        return self.data[j % RING_CAP]



def _median7(vals):
    return sorted(vals)[3]


def _variance_u16(a):
    """bpm variance of the 7 window IBIs; a zero IBI counts as bpm 0."""
    b = [bpm(x) if x else F(0.0) for x in a]
    s = b[0]
    for x in b[1:]:
        if x != F(0.0):
            s = add32(s, x)
    mean = div32(s, K7)
    acc = None
    for x in b:
        d = sub32(x, mean)
        t = mul32(d, d)
        acc = t if acc is None else add32(acc, t)
    return div32(acc, K6)


def _variance_ring(buf):
    """Same over ring entries -6..0, without the zero guard."""
    b = [bpm(buf.rfh(i).ibi) for i in range(-6, 1)]
    s = b[0]
    for x in b[1:]:
        s = add32(s, x)
    mean = div32(s, K7)
    acc = None
    for x in b:
        d = sub32(x, mean)
        t = mul32(d, d)
        acc = t if acc is None else add32(acc, t)
    return div32(acc, K6)


# ---- the corrector -----------------------------------------------------------------

class IbiCorrection:
    """State of the IBI corrector."""

    def __init__(self, max_out=4, gap_ms=5000, flag_a=0, flag_b=1, max_bpm=150.0):
        self.buf0 = Ring()          # input beats, 64 deep
        self.buf1 = Ring()          # corrected beats waiting to be emitted
        self.count = 0              # beats fed (i32)
        self.max_out = max_out & M16        # events emitted per call
        self.gap_ms = gap_ms & M16          # longest run of rejected beats that is interpolated
        self.prev_ibi = 1000
        self.n_out = 0              # u16 counters, not used by the output
        self.n_valid = 0
        self.flag_a = flag_a & 1
        self.flag_b = flag_b & 1
        self.prev_bpm = F(60.0)
        self.max_bpm = F(max_bpm)
        self.rej = 0                # u8: beats rejected in the current run
        self.rej2 = 0               # u8: length of the previous rejected run
        self.A = [0] * 7            # 7-beat output window: ibi_c
        self.B = [0] * 7            #   amplitude
        self.C = [0] * 7            #   corrected amplitude
        self.D = [0] * 7            #   validity indicator
        self.Fts = [0] * 7          #   corrected timestamp
        self.G = [0] * 7            #   input validity byte
        self.me_f0 = F(0.0)         # median_error state
        self.me_flag = self.flag_b
        self.me_prev_ibi = 1000
        self.ts = 0                 # timestamp of the beat being fed
        self.ts_epoch = 1           # cleared when the first timestamp is below 1e11
        self.out = []

    # -- public ---------------------------------------------------------------------
    def feed(self, ts, ibi, amp, vin=0):
        """Feed one beat; returns the events emitted by this call as
        `(timestamp, ibi, amplitude, validity)`."""
        self.ts = ts & M64
        self.out = []
        self._run(ibi & M16, amp & M16, vin & 0xFF)
        return self.out

    # -- the correction step -------------------------------------------------------
    def _run(self, ibi, amp, vin):
        if ibi:
            self.count = (self.count + 1) & M32
            cnt = _s32(self.count)
            e = Entry()
            e.ts0, e.ibi, e.amp, e.flag, e.vin = self.ts, ibi, amp, 1, vin
            self.buf0.push(e)
            if cnt == 1:
                e.ts1, e.ibi_c, e.amp_c, e.ind = self.ts, ibi, amp, 0
                if (self.ts - 1) & M64 <= 99_999_999_998:
                    self.ts_epoch = 0
                self.buf1.push(e)
            elif cnt >= 5:
                self._classify()
        self._drain()

    def _classify(self):
        """Decide whether the beat at -3 is plausible (flag), then emit or interpolate."""
        b0 = self.buf0
        x = b0.rfh(-3)
        ibx = x.ibi
        bpm_cur = bpm(ibx) if ibx else K60000
        n = b0.count()
        fb = self.flag_b
        mx = self.max_bpm
        prev = self.prev_bpm
        if n <= 6:
            if fb:
                reject = (at_most(bpm_cur, F(30.0)) or greater(bpm_cur, mx)
                          or less(bpm_cur, mul32(prev, F(0.625))) or greater(bpm_cur, add32(prev, F(18.0)))
                          or not not_less(bpm_cur, add32(prev, F(-25.0))))
            else:
                p = self.prev_ibi
                reject = (at_most(bpm_cur, F(30.0)) or greater(bpm_cur, mx)
                          or abs(ibx - p) * 100 > (p * 5) << 2 or x.vin >= 7)
            if reject:
                x.flag = 0
                self.rej = (self.rej + 1) & 0xFF
                self._set_prev(bpm_cur, x)
            else:
                self._accept(bpm_cur, x)
        else:
            m = _median7([b0.rfh(i).ibi for i in range(-6, 1)])
            bpm_med = bpm(m if m & 0xFFFE else 1)
            tot = sum(b0.rfh(i).ibi for i in range(-6, 1))
            mean_ibi = div32(F(tot), K7)
            bpm_mean = div32(K60000, mean_ibi)
            bpm_diff = sub32(bpm_mean if float(mean_ibi) != 0.0 else F(0.0), bpm_cur)
            if fb:
                reject = (at_most(bpm_cur, F(30.0)) or greater(bpm_cur, mx)
                          or greater(abs_diff(bpm_cur, bpm_med), F(14.0))
                          or less(bpm_cur, mul32(prev, F(0.625)))
                          or greater(bpm_cur, add32(prev, F(23.0)))
                          or less(bpm_cur, add32(prev, F(-16.0))))
                if not reject:
                    reject = greater(mul32(bpm_diff, bpm_diff), mul32(_variance_ring(b0), F(1.5)))
            else:
                p = self.prev_ibi
                reject = (at_most(bpm_cur, F(30.0)) or greater(bpm_cur, mx)
                          or greater(abs_diff(bpm_cur, bpm_med), F(14.0))
                          or abs(x.ibi - p) * 100 > (p * 5) << 2)
                if not reject:
                    reject = greater(mul32(bpm_diff, bpm_diff), mul32(_variance_ring(b0), F(1.5)))
                if not reject:
                    reject = x.vin >= 7
            if reject:
                x.flag = 0
                self.rej = (self.rej + 1) & 0xFF
                if self.rej >= 4:       # the first three rejections leave the reference bpm alone
                    self._set_prev(bpm_cur, x)
            else:
                self._accept(bpm_cur, x)
        self._tail()

    def _set_prev(self, bpm_cur, x):
        self.prev_bpm = bpm_cur
        self.prev_ibi = x.ibi

    def _accept(self, bpm_cur, x):
        x.flag = 1
        self.rej2 = self.rej
        self.rej = 0
        self._set_prev(bpm_cur, x)

    def _tail(self):
        """After the beat at -3 is classified."""
        b0 = self.buf0
        cur = b0.rfh(-3)
        before = b0.rfh(-4)
        if cur.flag == 1 and before.flag == 0:
            self._gap()
        else:
            self._after(0)

    def _gap(self):
        """The beat at -3 is good and the one before it is not. Find the previous
        good beat and fix up the run in between."""
        b0 = self.buf0
        skip = 0
        while True:
            prev_good = skip - 4
            if b0.rfh(prev_good).flag != 0:
                break
            next_back = skip - 5
            skip -= 1
            if not next_back > -64:
                if self.rej > 0x3C:
                    return self._flush(0xC1, self.rej, 1)
                return self._flush_pending_rejects()
        run_start = skip - 3
        prev_e = b0.rfh(prev_good)
        prev_ibi_v = prev_e.ibi
        ts_prev = prev_e.ts0
        cur_e = b0.rfh(-3)
        cur_ibi_v = cur_e.ibi
        ts_cur = cur_e.ts0
        sum_rej = 0
        if not prev_good > -5:
            idx = run_start
            while True:
                sum_rej += b0.rfh(idx).ibi
                idx += 1
                if idx == -3:
                    break
        ibi_sum = cur_ibi_v + prev_ibi_v
        n_est = max(round_away(div32(F(sum_rej), F(ibi_sum >> 1))), 1)
        eq = 1 if (n_est + skip) == 0 else 0
        if sum_rej > self.gap_ms or not (1 - skip) <= self.max_out:
            return self._too_long(run_start, prev_good)
        self._interpolate(cur_ibi_v, skip, prev_good, prev_ibi_v, ibi_sum, n_est, eq, run_start, sum_rej, ts_prev, ts_cur)

    def _too_long(self, run_start, prev_good):
        """Too many rejected beats to interpolate; emit them as invalid (2)."""
        b0 = self.buf0
        if b0.rfh(run_start).flag != 0:
            ts_last = 0
        else:
            while True:
                e = b0.rfh(run_start)
                l = e.copy()
                l.amp_c, l.ind, l.ibi_c, l.ts1 = 0, 2, e.ibi, e.ts0
                self.buf1.push(l)
                run_start += 1
                ts_last = e.ts0
                if b0.rfh(run_start).flag != 0:
                    break
        if self.gap_ms != 0:
            m = Entry()
            m.ts1, m.ind = ts_last, 2
            self.buf1.push(m)
        self._after(prev_good)

    def _interpolate(self, cur_ibi_v, skip, prev_good, prev_ibi_v, ibi_sum, n_est, eq, run_start, sum_rej, ts_prev, ts_cur):
        """Replace the rejected run by n_est interpolated beats (validity 3)."""
        b0, b1 = self.buf0, self.buf1
        ibi_prev = prev_ibi_v
        neg_skip = -skip
        end_idx = n_est + skip - 4
        ts_epoch = self.ts_epoch
        last_ts = 0
        sum_interp = 0
        step = round_away(div32(F(cur_ibi_v - ibi_prev), F(n_est + 1)))
        pos = prev_good
        cur_val = (step + ibi_prev) & M32
        while True:
            idx = pos + 1
            e = b0.rfh(idx)
            if neg_skip >= n_est or idx < -3:
                l = e.copy()
                last_ts = e.ts0
            else:
                l = Entry()
                l.ts0 = last_ts
            l.ibi_c = cur_val & M16
            l.amp_c = 0
            l.ind = 3
            b1.push(l)
            pos += 1
            sum_interp = (sum_interp + cur_val) & M32
            cur_val = (cur_val + step) & M32
            if _s32(pos) >= end_idx:
                break
        sum_interp = _s32(sum_interp)
        if self.flag_a and ibi_sum >= 2:
            return self._amp_check(eq, end_idx, n_est, prev_good, run_start)
        # remainder distribution
        resid = _s32(sum_rej - sum_interp)
        quot_f = div32(F(resid), F(n_est))
        q = to_int(quot_f)
        ts_div = F(1000.0) if ts_epoch == 0 else F(1.0)
        rem = _s32(resid - n_est * q)
        out_idx = 1 - n_est
        keep_ok = eq
        cnt = prev_good
        acc = 0
        while True:
            e2 = b1.rfh(out_idx)
            if rem == 0:
                v = e2.ibi_c + q
            elif rem > 0:
                rem -= 1
                v = (q + 1) + e2.ibi_c
            else:
                rem += 1
                v = (q - 1) + e2.ibi_c
            e2.ibi_c = v & M16
            acc += v & M16
            ts = (round_away(div32(F(acc), ts_div), 64) + ts_prev) & M64
            e2.ts1 = ts_cur if ts > ts_cur else ts
            if keep_ok & 1:
                keep_ok = 1 if abs(b0.rfh(end_idx).ibi - e2.ibi_c) < 0x7E else 0
            else:
                keep_ok = 0
            cnt += 1
            out_idx += 1
            if cnt >= end_idx:
                break
        self._amp_check(keep_ok, end_idx, n_est, prev_good, run_start)

    def _amp_check(self, ok, end_idx, n_est, prev_good, run_start):
        """With the amplitude flag or an equal-count run, copy the real IBIs into the
        interpolated slots and decide whether the run is valid (1) or not (2)."""
        b0, b1 = self.buf0, self.buf1
        if ok == 0:
            return self._after(prev_good)
        out_idx = 1 - n_est
        src_idx = prev_good
        dst_idx = out_idx
        while True:
            src_idx += 1
            ibi = b0.rfh(src_idx).ibi
            b1.rfh(dst_idx).ibi_c = ibi
            dst_idx += 1
            if not src_idx < end_idx:
                break
        cur_ibi_v = b0.rfh(-3).ibi
        prev_good_ibi = b0.rfh(prev_good).ibi
        bpm_lo = add32(div32(K60000, F(min(cur_ibi_v, prev_good_ibi))), F(34.5))
        mn = 0xFFFF
        if not run_start > -4:
            for i in range(run_start, -3):
                v = b0.rfh(i).ibi
                if not mn < v:
                    mn = v
        bpm_min = div32(K60000, F(mn))
        mxr = 0
        if not run_start > -4:
            for i in range(run_start, -3):
                v = b0.rfh(i).ibi
                if not mxr >= v:
                    mxr = v
        if not_less(bpm_min, bpm_lo):
            keep, ind = 0, 2
        else:
            bpm_hi = div32(K60000, F(cur_ibi_v if cur_ibi_v > prev_good_ibi else prev_good_ibi))
            bpm_max = div32(K60000, F(mxr))
            bpm_hi = add32(bpm_hi, F(-24.0))
            bpm_cap = F(np.uint32(0x41F0CCCD).view(np.float32))
            if less(bpm_hi, bpm_cap):
                bpm_hi = bpm_cap
            if not_greater(bpm_max, bpm_hi):
                keep, ind = 0, 2
            else:
                keep, ind = 1, 1
        src_idx = prev_good
        while True:
            e = b1.rfh(out_idx)
            e.ind = ind
            if keep:
                e.amp_c = e.amp
                e.ts1 = e.ts0
            src_idx += 1
            out_idx += 1
            if not src_idx < end_idx:
                break
        self._after(prev_good)

    def _after(self, prev_good):
        """Too many rejected beats pile up, flush them as invalid (2)."""
        if self.rej > 0x3C:
            return self._flush(0xC1, self.rej, 1)
        if _s32(prev_good) > -0x40:
            return self._finish()
        self._flush_pending_rejects()

    def _flush_pending_rejects(self):
        n_rej = self.rej2
        if n_rej == 0:
            return self._finish()
        self._flush(-3 - n_rej, n_rej, 0)

    def _flush(self, idx, remaining, dec):
        b0 = self.buf0
        while True:
            e = b0.rfh(idx)
            l = e.copy()
            l.amp_c, l.ind, l.ibi_c, l.ts1 = 0, 2, e.ibi, e.ts0
            self.buf1.push(l)
            e.ibi = 0
            e.amp = 0
            e.ind = 0
            b0.pop()
            if dec:
                self.rej = (self.rej - 1) & 0xFF
            remaining -= 1
            idx += 1
            if remaining & 0xFF == 0:
                break
        self._finish()

    def _finish(self):
        cur = self.buf0.rfh(-3)
        if cur.flag == 1:
            l = cur.copy()
            l.ibi_c, l.amp_c, l.ind, l.ts1 = cur.ibi, cur.amp, 1, cur.ts0
            self.buf1.push(l)

    # -- output stage ---------------------------------------------------------------
    def _drain(self):
        b1 = self.buf1
        if b1.is_empty():
            return
        n = 0
        while True:
            if n >= self.max_out:
                return
            e = b1.pop()
            if e.ibi_c != 0 and _s32(self.count) >= 0:
                self.A = self.A[1:] + [e.ibi_c]
                self.B = self.B[1:] + [e.amp]
                self.C = self.C[1:] + [e.amp_c]
                self.D = self.D[1:] + [e.ind]
                self.Fts = self.Fts[1:] + [e.ts1]
                self.G = self.G[1:] + [e.vin]
                if self.A[0] != 0:
                    self._median_error()
                if self.A[3] != 0:
                    self.n_out = (self.n_out + 1) & M16
                    if not self.G[3] > 1 and self.D[3] == 1:
                        self.n_valid = (self.n_valid + 1) & M16
                    self.out.append((self.Fts[3], self.A[3], self.C[3], self.D[3] & 0xFF))
                    n += 1
            if b1.is_empty():
                return

    def _median_error(self):
        """Demote the centre beat (indicator 1 to 2) when it departs from
        the 7-beat median and mean."""
        if self.D[3] != 1:
            return
        a = self.A
        ibi_c = a[3]
        bpm_c = bpm(ibi_c)
        if equal(self.me_f0, F(0.0)):
            self.me_f0 = bpm_c
        med = _median7([_s16(x) for x in a])
        mean = div32(F(sum(_s16(x) for x in a)), K7)
        bpm_med = div32(K60000, F(med))
        bpm_mean = div32(K60000, F(to_int(mean)))
        f0 = self.me_f0
        mx = self.max_bpm
        if self.me_flag:
            fail = (at_most(bpm_c, F(30.0)) or greater(bpm_c, mx)
                    or greater(abs_diff(bpm_c, bpm_med), F(18.0))
                    or greater(bpm_c, add32(add32(f0, F(23.0)), F(4.0)))
                    or less(bpm_c, mul32(f0, F(0.625)))
                    or less(bpm_c, add32(add32(f0, F(-16.0)), F(-4.0))))
            if not fail:
                bpm_mean = sub32(bpm_mean, bpm_c)
                fail = greater(mul32(bpm_mean, bpm_mean), mul32(_variance_u16(a), F(5.0)))
        else:
            p = self.me_prev_ibi
            fail = (at_most(bpm_c, F(30.0)) or greater(bpm_c, mx)
                    or greater(abs_diff(bpm_c, bpm_med), F(18.0))
                    or abs(ibi_c - p) * 100 > p * 25)
            if not fail:
                bpm_mean = sub32(bpm_mean, bpm_c)
                fail = greater(mul32(bpm_mean, bpm_mean), mul32(_variance_u16(a), F(5.0)))
            if not fail:
                fail = not self.G[3] < 7
        if fail:
            self.D[3] = 2
        self.me_f0 = bpm_c
        self.me_prev_ibi = ibi_c



def correct_ibis(beats, **kw):
    """Run a night through one freshly initialized state.
    beats: iterable of (timestamp_ms, ibi_ms, amplitude); returns the corrected events
    `(timestamp_ms, ibi_ms, amplitude, validity)`. Validity 1 is a valid beat."""
    c = IbiCorrection(**kw)
    out = []
    for ts, ibi, amp in beats:
        out += c.feed(ts, ibi, amp)
    return out


# ---- stateless variant -------------------------------

def merge_data(a_ts, a_st, b_ts, b_st):
    """Merge two (timestamp, state) lists by timestamp; on equal
    timestamps the entry of the second list goes first."""
    ts, st, i, j = [], [], 0, 0
    while i < len(a_ts) and j < len(b_ts):
        if a_ts[i] < b_ts[j]:
            ts.append(a_ts[i])
            st.append(a_st[i])
            i += 1
        else:
            ts.append(b_ts[j])
            st.append(b_st[j])
            j += 1
    ts += a_ts[i:] + b_ts[j:]
    st += a_st[i:] + b_st[j:]
    return ts, st


def next_reset_timestamp(after, ts, st):
    """The first state-change timestamp above `after`
    that is not a 5/6 or 6/5 transition (index 0 never counts); 2^64-1 if none."""
    for i in range(1, len(ts)):
        if ts[i] <= after:
            continue
        pair = (st[i - 1], st[i])
        if pair in ((6, 5), (5, 6)):
            continue
        return ts[i]
    return M64


def correct_stateless(ts, ibi, amp, sc_ts=(), sc_states=(), wear_ts=(), wear_states=(), **kw):
    """The batch variant: a fresh state per run of beats. The state is dropped (and its
    undelivered beats lost) when a beat's timestamp passes the next reset timestamp, which
    comes from the merged state-change and wear lists. At most one reset per beat. Returns
    the events in `correct_ibis` form."""
    m_ts, m_st = merge_data(list(sc_ts), list(sc_states), list(wear_ts), list(wear_states))
    reset = next_reset_timestamp(0, m_ts, m_st)
    c = IbiCorrection(**kw)
    out = []
    for t, i, a in zip(ts, ibi, amp):
        t &= M64
        if t > reset:
            c = IbiCorrection(**kw)
            reset = next_reset_timestamp(reset, m_ts, m_st)
        out += c.feed(t, i, a)
    return out


if __name__ == "__main__":
    import random
    r = random.Random(1)
    t, beats = 1_700_000_000_000, []
    for _ in range(40):
        v = r.randint(700, 900)
        t += v
        beats.append((t, v, r.randint(100, 900)))
    for e in correct_ibis(beats):
        print(e)
