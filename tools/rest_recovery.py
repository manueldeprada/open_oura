#!/usr/bin/env python3
"""Oura rest mode and rest recovery.

Rest mode is a user-declared break (illness, travel). Given the start and end
of the rest period, the state for the current day is reported and, after the break ends, a
recovery phase of up to 7 days during which the readiness score is capped (see
`readiness_score.recovery_core`). Two pieces are ported:

* The stateless query used by the readiness calculation:
  `get_state(RestInput)`. A period is the interval [start, end] in seconds (end 0 while the
  period is open); `cur_midnight` is the UTC midnight of the day being scored.
* The ring-event state machine and the 30-entry period history.

State codes of `RestOutput.state`: 0 normal, 1 rest period starts today, 2 resting (a = whole days
since the first day), 3 last day of a finished period (a = days of rest, b = recovery days,
c = recovery days remaining), 4 recovering (same fields), 5 invalid input.

"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

DAY = 86400
MAX_RECOVERY_DAYS = 7
HISTORY_SIZE = 30
MODE_MAGIC0 = 0x47C29A20
MODE_MAGIC1 = 0xAB4D3D6E


def midnight(t):
    """UTC midnight at or before `t` (unsigned division)."""
    return (t & 0xFFFFFFFFFFFFFFFF) // DAY * DAY


def _tdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


@dataclass
class RestInput:
    """the rest mode input record: i64 start, i64 end, i64 (not
    read), i64 current UTC midnight."""
    start: int = 0
    end: int = 0
    unused: int = 0
    cur_midnight: int = 0


@dataclass
class RestOutput:
    """the rest mode output record without the trailing byte: state, then a, b, c."""
    state: int = 0
    a: int = 0
    b: int = 0
    c: int = 0


def _sx32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def recovery_status(inp: Optional[RestInput], start_mid: int, end_mid: int):
    """(rest days D, days since the end K, recovery days
    min(D, 7), recovery days remaining, valid). D is the whole days between the start and end
    midnights (1 when they coincide); K = (current midnight - end midnight - 1 day) / 1 day,
    truncated, in 32-bit arithmetic."""
    if inp is None or end_mid == 0 or start_mid == 0 or end_mid < start_mid:
        return (0, 0, 0, 0, 0)
    d = 1 if end_mid == start_mid else ((end_mid - start_mid) & 0xFFFFFFFF) // DAY
    k = _tdiv(_sx32(inp.cur_midnight - end_mid - DAY), DAY)
    d7 = min(d, MAX_RECOVERY_DAYS)
    remaining = 0 if k > d7 else d7 - max(k, 0)
    return (d, k, d7, remaining, 1)


def get_state(inp: Optional[RestInput]) -> RestOutput:
    out = RestOutput()
    if inp is None:
        out.state = 5
        return out
    out.state = 5
    cur = inp.cur_midnight
    if cur == 0:
        return out
    start, end = inp.start, inp.end
    start_mid, end_mid = midnight(start), midnight(end)
    if end_mid > cur:
        end = 0
    if start_mid > cur:
        return out
    if start < 1:
        out.state = 0
        return out
    if end > start:
        end_mid = midnight(end)
        st = recovery_status(inp, start_mid, end_mid)
        if cur >= end_mid + DAY:
            if st[1] >= st[2]:
                out.state = 0
            else:
                out.state, out.a, out.b, out.c = 4, st[0], st[2], st[3]
        else:
            out.state, out.a, out.b, out.c = 3, st[0], st[2], st[3]
    elif start > end:
        if cur >= start_mid + DAY:
            out.state, out.a = 2, ((cur - (start_mid + DAY)) & 0xFFFFFFFF) // DAY
        else:
            out.state = 1
    return out


def current_recovery_day(end_mid: int, cur_midnight: int) -> int:
    """Whole days after the end day, 0 when either
    time is not positive or the gap is shorter than a day."""
    if end_mid < 1 or cur_midnight < 1:
        return 0
    d = cur_midnight - end_mid
    return d // DAY if d > 0x1517F else 0


def remaining_recovery_days(day: int, total: int) -> int:
    """Remaining recovery days (unsigned, floored at 0)."""
    return max(total - day, 0)


def durations(period) -> Tuple[int, int]:
    """Durations of a computed period: (rest days, recovery days)."""
    start_mid, end_mid = period.start_mid, period.end_mid
    days = max(_tdiv(end_mid - start_mid, DAY), 0)
    d = days if end_mid != start_mid else 1
    return d, min(d, MAX_RECOVERY_DAYS)


@dataclass
class Period:
    """The rest recovery period record: start, start midnight, start midnight + 1 day, end,
    end midnight, end midnight + 1 day, end of recovery, rest days (i32), recovery days (i32)."""
    start: int = 0
    start_mid: int = 0
    start_next: int = 0
    end: int = 0
    end_mid: int = 0
    end_next: int = 0
    recovery_end: int = 0
    days: int = 0
    recovery_days: int = 0


def calculate_all_times(start: int, end: int) -> Period:
    p = Period()
    if start < 1:
        return p
    p.start = start
    p.start_mid = midnight(start)
    p.start_next = p.start_mid + DAY
    p.end = end
    if end < 1:
        return p
    p.end_mid = midnight(end)
    p.end_next = p.end_mid + DAY
    d = max(_tdiv(p.end_mid - p.start_mid, DAY), 0)
    if p.end_mid == p.start_mid:
        d = 1
    d7 = min(d, MAX_RECOVERY_DAYS)
    p.days, p.recovery_days = d, d7
    p.recovery_end = p.end_next + d7 * DAY
    return p


def past_state(period: Optional[Period], now: int) -> RestOutput:
    """The state of a stored period at time `now`."""
    out = RestOutput()
    out.state = 5
    if period is None or period.start < 1:
        return out
    if now < 1 or period.end < 1:
        return out
    if period.recovery_end < now:
        out.state = 0
        return out
    now_mid = midnight(now)
    if period.end_mid >= now:
        if period.start_next >= now:
            out.state = 0 if period.start_mid >= now else 1
        else:
            out.state = 2
            out.a = ((now_mid - period.start_next) & 0xFFFFFFFF) // DAY
        return out
    out.state = 4 if period.end_next < now else 3
    out.a, out.b = period.days, period.recovery_days
    gap = now_mid - period.end_next
    k = gap // DAY if gap > 0x1517F else 0
    out.c = max(period.recovery_days - k, 0)
    return out


@dataclass
class History:
    """The 30-entry period history: raw (start, end) pairs and the
    computed `Period` for each."""
    valid: bool = False
    periods: List[Period] = field(default_factory=list)

    def set(self, raw: Optional[List[Tuple[int, int]]], count: int):
        self.valid = False
        self.periods = []
        if raw is None:
            return
        self.valid = True
        for s, e in raw[:min(count & 0xFFFF, HISTORY_SIZE)]:
            self.periods.append(calculate_all_times(s, e))

    def find(self, t: int):
        """The newest period that contains `t`, from its start
        to the end of its recovery (an open period contains every later time)."""
        for p in reversed(self.periods):
            if p.end == 0:
                if p.start <= t:
                    return p
                continue
            if p.start > t:
                continue
            if p.recovery_end < t:
                continue
            return p
        return None

    def status_for_activity(self, t: int) -> RestOutput:
        out = RestOutput()
        if t < 1 or not self.valid:
            out.state = 5
            return out
        p = self.find(t)
        if p is None:
            return out
        if p.end != 0:
            return past_state(p, t)
        return get_state(RestInput(p.start, p.end, 0, midnight(t)))


# --------------------------------------------------------------------------- event state machine
@dataclass
class Mode:
    """the rest mode state: u32 magic, u8 state, u32 rest-day count, u32 `total`, u32 `remaining`, u32 magic."""
    magic0: int = MODE_MAGIC0
    state: int = 0
    n: int = 0
    total: int = 0
    remaining: int = 0
    magic1: int = MODE_MAGIC1

    def sane(self) -> bool:
        return (self.magic0 == MODE_MAGIC0 and self.magic1 == MODE_MAGIC1 and self.n < 0x1C85
                and self.total < 8 and self.remaining < 8 and self.state < 5)


def init_mode() -> Mode:
    return Mode()


def process_event(m: Mode, ev: bytes) -> int:
    """Apply a ring rest-mode event (header 0xfe 0x20 or 0xfe
    0x21, u16 LE length at 2, payload from 4). 0xfe20 starts a rest period (state 1) from state
    0, 3 or 4; 0xfe21 ends it (byte 12 set keeps the state at 3, i.e. recovering). Returns the
    status code (0 ok, 4 unknown event, 9 short event)."""
    tag = (ev[0] << 8 | ev[1]) if len(ev) >= 2 else 0
    ln = ev[2] | ev[3] << 8 if len(ev) >= 4 else 0
    if tag == 0xFE21:
        if ln < 9:
            return 9
        flag = ev[12] if 12 < ln + 4 and len(ev) > 12 else 0
        if m.state == 2:
            m.total = m.remaining = min(m.n, 7)
            m.state = 3
        elif m.state == 1:
            m.n = m.total = m.remaining = 0
            m.state = 3 if flag else 0
        elif m.state == 0:
            m.n = m.total = m.remaining = 0
        return 0
    if tag == 0xFE20:
        if ln < 8:
            return 9
        if m.state <= 4 and (1 << m.state) & 0x19:
            m.n = m.total = m.remaining = 0
            m.state = 1
        return 0
    return 4


def major_sleep_period(m: Mode, n: int):
    """A main sleep ended. While resting the rest-day count
    grows by `n` (at least 1); while recovering the remaining days drop by `n` (at least 1)."""
    step = n if n > 1 else 1
    st = m.state
    if st == 1:
        m.state = 2
    elif st == 2:
        m.n += step
        m.state = 2
    elif st == 3:
        m.state = 4
    elif st == 4 and m.remaining > step:
        m.remaining -= step
        m.state = 4
    else:
        m.n = m.total = m.remaining = 0
        m.state = 0
