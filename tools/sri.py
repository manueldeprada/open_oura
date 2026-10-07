#!/usr/bin/env python3
"""Sleep regularity state over an explicit state.

The state keeps up to 15 day rows of 144 ten-minute slots (0 awake, 1 asleep, 2 missing), a row
count, and the sleep date of the last update. `SriState.update_history` runs for each ring
sleep summary when feature 20 is on, with the time zone of the ring's time_sync event, and once per
stored sleep, oldest first, in `init_sleeps`. `sri_calculate` in readiness_score.py reads `SriState.rows()`.

Times are (utc_seconds, tz_minutes) pairs.
"""
DAY = 86400
ROWS = 15
SLOTS = 144


def sleep_date(t):
    """UTC midnight of the local calendar day."""
    return ((t[0] + 60 * t[1]) // DAY) * DAY


def slot(t):
    """Ten-minute slot of the local time of day."""
    s = (t[0] + 60 * t[1]) % DAY
    return (s // 3600) * 6 + (s % 3600) // 60 // 10


class Row:
    __slots__ = ("date", "start", "start_tz", "end", "end_tz", "dur", "slots")

    def __init__(self):
        self.date = 0
        self.start = self.start_tz = self.end = self.end_tz = self.dur = 0
        self.slots = [2] * SLOTS

    def copy_from(self, o):
        self.date, self.start, self.start_tz, self.end, self.end_tz, self.dur = (
            o.date, o.start, o.start_tz, o.end, o.end_tz, o.dur)
        self.slots = list(o.slots)


class SriState:
    def __init__(self):
        self.clear()

    def clear(self):
        """Empty the sleep-regularity history."""
        self.r = [Row() for _ in range(ROWS)]
        self.count = 0
        self.last = (0, 0)

    def rows(self):
        return [list(x.slots) for x in self.r[:min(self.count, ROWS)]]

    def idx(self, date):
        for i in range(min(self.count, ROWS)):
            if self.r[i].date == date:
                return i
        return 0xFF

    def add_new_day(self, t):
        d = sleep_date(t)
        if self.count >= ROWS:
            for i in range(ROWS - 1):
                self.r[i].copy_from(self.r[i + 1])
            row = self.r[ROWS - 1]       # header stays from the old last row
            row.slots = [2] * SLOTS
            row.date = d
            self.count = ROWS
        else:
            row = self.r[self.count]
            row.slots = [2] * SLOTS
            row.date = d
            self.count += 1

    def _ensure(self, t):
        if self.idx(sleep_date(t)) == 0xFF:
            self.add_new_day(t)
        return self.idx(sleep_date(t))

    def write_sleep_day(self, t, typ):
        """Night of sleep date `t`: evening slots 108..143 of that day and 0..107 of the next."""
        i = self._ensure(t)
        if i != 0xFF and typ <= 2:
            self.r[i].slots[108:144] = [typ] * 36
        t2 = (t[0] + DAY, t[1])
        j = self._ensure(t2)
        if j != 0xFF and typ <= 2:
            self.r[j].slots[0:108] = [typ] * 108

    def write_new_sleep_period(self, D, S, E):
        s, e = slot(S), slot(E)
        iD = self._ensure(D)
        if iD != 0xFF:
            row = self.r[iD]
            row.start, row.start_tz, row.end, row.end_tz = S[0], S[1], E[0], E[1]
            row.dur = (E[0] - S[0]) & 0xFFFFFFFF
        dS = (sleep_date(S), 0)
        if self.idx(dS[0]) == 0xFF:
            self.add_new_day(dS)
        dE = (sleep_date(E), 0)
        if self.idx(dE[0]) == 0xFF:
            self.add_new_day(dE)
        iS = self.idx(dS[0])
        n = self.count

        def fill(i, a, length, v):
            if i != 0xFF and n > i and length > 0:
                for k in range(a, min(a + length, SLOTS)):
                    self.r[i].slots[k] = v

        if s > e:                                       # crosses midnight
            if s == 0 or s >= 109:
                fill(iS, 108, max(s, 109) - 108, 0)
            if s <= 0x8F:
                fill(iS, s, 0x90 - s, 1)
            iE = self.idx(dE[0])
            if e <= 0x8F:
                fill(iE, 0, e + 1, 1)
            if e + 1 <= 0x6B:
                fill(iE, e + 1, 107 - e, 0)
        else:
            if iS != 0xFF and iS >= 1 and n > iS - 1:
                self.r[iS - 1].slots[108:144] = [0] * 36
            if 1 <= s <= 0x90:
                fill(iS, 0, s, 0)
            fill(iS, s, e - s + 1, 1)
            if e + 1 <= 0x6B:
                fill(iS, e + 1, 107 - e, 0)

    def update_history(self, D, S, E):
        """."""
        i = self.idx(sleep_date(D))
        if D[0] < self.r[0].date:
            return
        if i != 0xFF:
            if (E[0] - S[0]) <= self.r[i].dur:
                self.last = D
                return
            self.write_sleep_day(D, 0)
        elif self.last[0] != 0 and D[0] >= self.last[0]:
            diff = ((D[0] - self.last[0]) // DAY) & 0xFFFF
            if diff & 0xFFFE:
                for k in range(1, diff):
                    self.write_sleep_day((self.last[0] + k * DAY, D[1]), 2)
        self.write_new_sleep_period(D, S, E)
        self.last = D

    def init_sleeps(self, sleeps):
        """`sleeps` is newest first in the array; the array is walked from its end."""
        self.clear()
        for D, S, E in reversed(sleeps):
            self.update_history(D, S, E)
