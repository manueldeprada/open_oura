#!/usr/bin/env python3
"""Daily activity statistics:
per-minute MET bytes in, class minutes, MET-minutes,
equivalent distance, calories, steps, daily target and the activity-score day record out.

Inputs. The ring's `activity_information` events (tag 0x50) carry 5 per-minute MET*10
bytes (value 1 marks non-wear; values >= 128 are stored as 2*v-128 in MET*10) plus a
step count. `ActInfo.add_activity` folds the events into the per-minute buffer, `ActInfo.statistics`
computes the day's values:

    a = ActInfo(User(bmr=1700, age=35, gender=1, weight_kg=75))
    a.set_target(readiness_pct=80)          # daily target
    for last_minute, steps, five_bytes in events:
        a.add_activity(last_minute, steps, five_bytes)
    out = a.statistics()                    # out.classes, out.met_min, out.day (activity_score.Day)

Pipeline (all integer except the target, which is float32):

* A minute's MET*32 `m` is `met32_from_u8(v)` (non-wear 1 counts as 12 = 1.2 MET). Each
  minute is classified from `m` and the sum `W` of `m` over the last 5 minutes
  (5*sum/(i+1) for the first four minutes) with `met_to_class(2*m, 2*W)`:
  vigorous (4) if m > vig or W > 5*vig; moderate (3) if m > mod or W > 5*mod;
  light (2) if m > 57 or W > 288; sedentary (1) if W > 167; else rest (0). `mod`, `vig` default
  to 128 and 224 (4.0 and 7.0 MET) and `met_limits(age, gender)` replaces them
  with age/sex values.
* Sedentary minutes inside a sleep period `(s, e)` (s < i < e) are rest. Class minutes
  are counted over minutes `i < last` with byte != 1 (non-wear is excluded).
* `sedentary_min` of the day record is the sum of 24 hourly counters, each the number of
  sedentary minutes in one clock hour. Counter h is written by `add_activity` the first
  time the cursor reaches minute 60*(h+1) strictly before `last`, so the final partial
  hour is never counted (a fully sedentary 1439-minute day gives 1380).
* MET-minutes per class: sum over minutes of that class with m > 48 (1.5 MET) of
  `m - 48`, rounded as (x + 16) >> 5. Equivalent distance: sum of
  `met10_to_eq_meters(MET*10)`; the day record holds half of it.
* Active calories: sum over minutes with m >= 49 of (m - 48) * bmr / 46080 (kcal;
  `act_calories` returns it x1000 and the day total is rounded as (x + 500) / 1000).
  Total calories: bmr * (1 + sum of (m - 34) / 46080 over minutes with m >= 35).
* Target: `activity_target(pct, age, gender)` is the daily equivalent distance in meters
  (180*eq(a) + 45*eq(b) + 5*eq(c) for three age/sex MET levels, scaled by the readiness
  percent); the stored base is half of it. Target calories, steps, distance and the
  remaining distance follow from base * multiplier / 100.

"""
from dataclasses import dataclass, field

import numpy as np

import activity_score as AS

f32 = np.float32
M32 = 0xFFFFFFFF
DAY_MIN = 1440


def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def sign_extend_16(x):
    x &= 0xFFFF
    return x - 0x10000 if x >> 15 else x


def F(bits):
    return np.frombuffer(np.uint32(bits).tobytes(), dtype=np.float32)[0]


def fma32(a, b, c):
    """Fused float32 multiply-add (one rounding)."""
    return f32(np.float64(a) * np.float64(b) + np.float64(c))


def to_int(x):
    x = float(x)
    if x != x:
        return 0
    return max(-(1 << 31), min((1 << 31) - 1, int(x)))


def to_uint(x):
    x = float(x)
    if x != x:
        return 0
    return max(0, min(M32, int(x)))


def div_by_magic(w, magic, shift):
    """Signed division of a 32-bit value by multiplying with `magic`: the 64-bit product is shifted
    right and 1 is added when the product is negative."""
    p = s32(w) * s32(magic)
    return s32((p >> shift) + (1 if p < 0 else 0))


# ---- MET byte codecs -------------------------------------------------------------

def met10_from_u8(v):
    v &= 0xFF
    return (2 * v - 128) & M32 if v & 0x80 else v


def met32_from_u8(v):
    v &= 0xFF
    if v & 0x80:
        return ((((v << 6) - 0x1FFB) & 0xFFFF) * 0xCCCD >> 19) + 0x19A
    return ((4 | v << 5) * 0x199A & M32) >> 16


def met32_to_u8(x):
    x &= 0xFFFF
    if x <= 0x19A:
        return ((x * 10 + 16) >> 5) & 0x7FF
    if x <= 0x4C5:
        return ((((x * 5 + 0x180E) & M32) >> 5) & 0x7FF) ^ 0xFFFFFF80
    return 0xFF


def act_calories(v, bmr):
    """Calories (x1000 scale folded in) of one minute."""
    v &= 0xFF
    if v & 0x80:
        m = ((((v << 6) - 0x1FFB) & 0xFFFF) * 0xCCCD >> 19) + 0x19A
    else:
        x = 4 | v << 5
        if x < 0x1EA:
            return 0
        m = (x * 0x199A & M32) >> 16
    w = (m * 1000 - 0xBB80) & M32
    w = (w * (bmr & M32) + 0x5A00) & M32
    return ((w >> 10) * 0x05B05B06) >> 32


def met10_to_eq_meters(x):
    """Equivalent walking meters per minute at MET*10 `x`."""
    x &= 0xFFFF
    if x < 0x13:
        return 0
    a, b = (0x898, -0x9984) if x < 0x21 else (0x3E8, 0x12C)
    return div_by_magic((a * x + b) & M32, 0x1B4E81B5, 38)


def met_to_eq_meters(x):
    """Equivalent meters per minute at `x` MET (float32)."""
    x = f32(x)
    if x <= F(0x3FE66666):
        speed = f32(0)
    else:
        if not (x >= F(0x40533333)):
            x = fma32(x, F(0x400CCCCD), F(0xC07D70A4))
        speed = f32(x / f32(60.0))
    return f32(speed * f32(1000.0))


def met_limits(age, gender, current=(128, 224)):
    """(moderate, vigorous) limits in MET*32 for a user; unchanged
    for gender codes other than 1, 2, 3. The branch names are swapped relative to the codes:
    the caller passes 2 for a male user and 1 for a female user (see `bmr`), so `male_*` below is
    the table of code 1."""
    age &= M32
    age_diff = (age - 0x19) & M32 if (age & 0xFF) > 0x18 else (0x19 - age) & M32
    d = (age_diff & 0xFF) - 256 if age_diff & 0x80 else age_diff & 0xFF
    male_vig = ((d * 0x1FFFCD + 0x1EF1) & M32) >> 5
    male_mod = ((d * 0x1FFFEC + 0x1081) & M32) >> 5
    fem_mod = ((d * 0x1FFFEF + 0x11AA) & M32) >> 5
    fem_vig = ((d * 0x1FFFD6 + 0x2224) & M32) >> 5
    if gender == 1:
        return male_mod & 0xFFFF, male_vig & 0xFFFF
    if gender == 2:
        return fem_mod & 0xFFFF, fem_vig & 0xFFFF
    if gender == 3:
        mod = (((fem_mod & 0xFFFF) + (male_mod & 0xFFFF) + 1) & M32) >> 1
        vig = (((fem_vig & 0xFFFF) + (male_vig & 0xFFFF) + 1) & M32) >> 1
        return mod & 0xFFFF, vig & 0xFFFF
    return current


def met_to_class(a, b, mod=128, vig=224):
    """met_to_class(2*m, 2*W) -> 0 rest, 1 sedentary, 2 light, 3 moderate, 4 vigorous."""
    a &= 0xFFFF
    b &= 0xFFFF
    if vig * 10 < b or a > vig * 2:
        return 4
    if not (mod * 10 >= b and mod * 2 >= a):
        return 3
    if a > 0x73 or b > 0x240:
        return 2
    return 1 if b > 0x14F else 0


def sleep_minutes(t, day_start):
    """Minutes from the day start to epoch second `t`, as the signed 16-bit value stored
    in state[0x676/0x678]: -1 when `t` is at or before the day start or 24 h or more after."""
    x = (max(t, day_start) - day_start) & M32
    return sign_extend_16(x * 0x88888889 >> 37) if ((x - 86400) & M32) >= 0xFFFEAE81 else -1


# ---- basal metabolic rate ------------------------------

# Schofield equations (kcal/day = slope * kg + intercept) for the age bands 0-3, 3-10, 10-18,
# 18-30, 30-60, 60+ (`schofield`).
_SCHOFIELD = {
    "male": ((59.512, -30.4), (22.706, 504.3), (17.686, 658.2), (15.057, 692.2), (11.472, 873.1), (11.711, 587.7)),
    "female": ((58.317, -31.1), (20.315, 485.9), (13.384, 692.6), (14.818, 486.6), (8.126, 845.6), (9.082, 658.5)),
}
_BAND_CENTER = (0, 3, 13, 28, 48, 90, 1000)  # in half-years, indexed by band number 1..6


def _fma(a, b, c):
    import math
    return math.fma(a, b, c) if hasattr(math, "fma") else a * b + c


def _schofield(age, kg, sex):
    """schofield(age, kg, 0 male | 1 female): the band formulas, blended linearly in age between
    the centers of adjacent bands."""
    age, kg = age & 0xFF, kg & 0xFF
    x = age * 2
    if age < 2:
        lo = hi = 1
    else:
        hi = 2 if x < 14 else 3 if x < 29 else 4 if x < 49 else 5 if x <= 90 else 6
        lo = hi - 1
    f = lambda band: _fma(float(kg), _SCHOFIELD[sex][band - 1][0], _SCHOFIELD[sex][band - 1][1])
    d0 = f(lo)
    if hi <= lo:
        return d0
    d1 = f(hi)
    return d0 + (d1 - d0) * (x - _BAND_CENTER[lo]) / (_BAND_CENTER[hi] - _BAND_CENTER[lo])


def bmr(age, kg, gender_code):
    """Basal metabolic rate: gender 2 is male, 1 female, 3 (or 0) the mean of both;
    the value is rounded to the nearest integer."""
    m, f = _schofield(age, kg, "male"), _schofield(age, kg, "female")
    v = {2: m, 1: f}.get(gender_code, (m + f) * 0.5)
    return int(v + 0.5)


# ---- daily target ------------------------------------------------------------------

# float32 pairs: category MET levels (lane 0, lane 1) = base + slope * |age - 25|
_MALE = ((F(0x40F70A3D), F(0x4083851F)), (F(0xBD4CCCCD), F(0xBCA3D70A)))
_FEMALE = ((F(0x410851EC), F(0x408CCCCD)), (F(0xBD27EF9E), F(0xBC8B4396)))


def activity_target(pct, age, gender):
    """Daily target in equivalent meters (float32).
    `pct` is the readiness percent (80 when unknown); gender is the gender code of `User`."""
    age_dev = f32(abs(f32(np.float64(age & 0xFF) - 25.0)))
    male = tuple(fma32(age_dev, _MALE[1][k], _MALE[0][k]) for k in (0, 1))
    female = tuple(fma32(age_dev, _FEMALE[1][k], _FEMALE[0][k]) for k in (0, 1))
    if gender == 1:
        lanes = male
    elif gender == 3:
        lanes = tuple(f32(f32(male[k] + female[k]) * f32(0.5)) for k in (0, 1))
    elif gender == 2:
        lanes = female
    else:
        lanes = None
    if lanes is None:
        lane_a = lane_b = lane_c = f32(0)
    else:
        l0, l1 = lanes
        k = F(0xBFE66666)
        l1_k, l0_k = f32(l1 + k), f32(l0 + k)
        l_diff = f32(l0 - l1)
        lane_a = fma32(l1_k, F(0x3E8CCCCD), F(0x3FE66666))
        lane_c = fma32(l0_k, f32(1.25), F(0x3FE66666))
        lane_b = fma32(l_diff, F(0x3E4CCCCD), l1)
    sum_a = fma32(met_to_eq_meters(lane_a), f32(180.0), f32(0))
    sum_b = fma32(met_to_eq_meters(lane_b), f32(45.0), sum_a)
    total = fma32(met_to_eq_meters(lane_c), f32(5.0), sum_b)
    p = pct & 0xFFFF
    f = f32(p)
    if p <= 0x45:
        f = f32(f / f32(100.0))
    elif p >= 0x55:
        f = f32(f32(f + f) / f32(100.0) + f32(-0.5))
    else:
        return total
    return f32(f * total)


def target_to_cal_base(target, bmr):
    steps_scale = f32(f32(f32(target) * f32(60.0)) / f32(5000.0))
    return f32(steps_scale * f32(act_calories(met32_to_u8(160), bmr)))


def target_to_steps(target2, bmr, age, gender, weight):
    """Steps for the daily target (target2 = 2*base)."""
    steps_scale = f32(f32(f32(target2) * f32(60.0)) / f32(5000.0))
    kcal = f32(steps_scale * f32(act_calories(met32_to_u8(160), bmr)))
    x = f32(kcal / f32(1000.0))
    male = f32(1.0) if gender == 1 else f32(0.0)
    other = f32(1.0) if gender == 3 else f32(0.0)
    xd = float(x)
    if xd > 508.84764358745565:
        hi = xd > 776.8511438242949
        pick_hi = f32(f32(x * F(0x40AE90CB)) + F(0x464D0B5B))
        tmp_b = f32(f32(x * F(0x4159D1F8)) + F(0x45D32590))
        pick_hi = f32(f32(male * F(0x448D36F2)) + pick_hi)
        tmp_b = f32(f32(male * F(0x4492B792)) + tmp_b)
        pick_hi = f32(f32(other * F(0x4305F878)) + pick_hi)
        pick_lo = f32(tmp_b + f32(other * F(0xC257CD1B)))
        age_c, w_c = (F(0x41E447C0), F(0x42631479)) if hi else (F(0x41DEAB0A), F(0x42635FB2))
        pick = pick_hi if hi else pick_lo
    else:
        hi = xd > 219.8440342176652
        pick_hi = f32(f32(x * F(0x41A24587)) + F(0x4535557A))
        tmp_b = f32(f32(x * F(0x41BAC1D5)) + F(0x4461DB80))
        pick_hi = f32(f32(male * F(0x448E7F9D)) + pick_hi)
        tmp_c = f32(f32(male * F(0x4401CC68)) + tmp_b)
        pick_hi = f32(f32(other * F(0x41B5F2C3)) + pick_hi)
        pick_lo = f32(f32(other * F(0x42036E77)) + tmp_c)
        age_c, w_c = (F(0x41879A6E), F(0x422A897A)) if hi else (F(0x410480B3), F(0x4193D0CE))
        pick = pick_hi if hi else pick_lo
    r = f32(f32(age_c * f32(age & M32)) + pick)
    r = f32(r - f32(w_c * f32(weight & M32)))
    if r < 0:
        r = f32(0)
    q = f32(r / f32(500.0))
    n = to_uint(np.float64(q) + 0.5)
    return (n * 500) & M32


# ---- state ---------------------------------------------------------------------------

@dataclass
class User:
    """User profile."""
    bmr: int = 1700
    age: int = 35
    gender: int = 1          # gender code: 1 male, 2 female, 3 other, 0 unknown
    weight_kg: int = 75
    target_mult: int = 100   # percent of the base daily target, applied for 25..200
    target_type: int = 0     # 0 calories, 1 steps, other: calories-style distance


@dataclass
class Stats:
    """Output fields written by the statistics update."""
    eq_distance: int = 0       # summed equivalent meters
    non_wear: int = 0          # minutes
    classes: list = field(default_factory=lambda: [0] * 5)  # rest, sedentary, light, moderate, vigorous
    avg_met32: int = 0
    steps: int = 0
    active_cal: int = 0
    total_cal: int = 0
    met_min: list = field(default_factory=lambda: [0] * 4)  # sedentary, light, moderate, vigorous
    sit_alerts: int = 0
    target_distance_m: int = 0
    target_miles_x10: int = 0
    target_type: int = 0
    target_mult: int = 0
    target_cal: int = 0
    target_steps: int = 0
    meters_to_target: int = 0
    miles_to_target_x10: int = 0
    day: AS.Day = field(default_factory=AS.Day)


class ActInfo:
    """One day of activity state (only the fields the statistics read).

    `buf` is the per-minute maximum buffer. `last` is state[0xe]: minutes
    0..last-1 are the ones counted, so a day of N minutes needs last = N + 1 to include
    the final minute."""

    def __init__(self, user=None, user_met_limits=True):
        self.user = user or User()
        self.buf = [0] * DAY_MIN
        self.last = 0
        self.steps = 0
        self.non_wear_base = 0     # state[0xc], minutes added outside the buffer
        self.sit_alerts = 0        # state[0x10], inactivity alert events
        self.hourly = [0] * 24     # state[0x65c..0x673]
        self.cursor = 0            # state[0x674]
        self.stateless = False     # stateless mode: bytes below 9 other than 1 (non-wear) count as 0.9 MET (byte 9)
        self.ranges = None         # all sleep periods of the day (stateless path); default: [sleep]
        self.sleep = (0, 0)        # state[0x676], state[0x678], minutes of the last sleep period
        self.base_target = 0       # state[0x65a]
        self.mod, self.vig = 128, 224
        if user_met_limits:
            self.mod, self.vig = met_limits(self.user.age, self.user.gender)

    # -- classification --
    def _m(self, i):
        v = self.buf[i]
        if v < 9 and v != 1 and self.stateless:
            v = 9
        return met32_from_u8(12 if v == 1 else v)

    def classes_series(self, n):
        """Class per minute for minutes 0..n-1 before the sleep adjustment."""
        out, total = [], 0
        for i in range(n):
            m = self._m(i)
            total += m
            if i >= 5:
                total -= self._m(i - 5)
                w = total
            else:
                t = total & 0xFFFF
                w = (t + t * 4) // (i + 1)
            out.append(met_to_class(2 * m, 2 * w, self.mod, self.vig))
        return out

    def class_minutes(self, cls, start=0, count=None, exclude_non_wear=True, ranges=None):
        """Minutes i in [start, count) of class
        `cls`; with exclude_non_wear only minutes i < last whose byte is not 1. A
        sedentary minute inside any sleep range (s, e), s < i < e, is rest. `ranges`
        defaults to the state's last sleep period."""
        count = self.last if count is None else count
        ranges = ([self.sleep] if self.ranges is None else self.ranges) if ranges is None else ranges
        n = 0
        for i, c in enumerate(self.classes_series(count)):
            if c == 1 and any(s < i < e for s, e in ranges):
                c = 0
            if i >= start and c == cls:
                if not exclude_non_wear or (i < self.last and self.buf[i] != 1):
                    n += 1
        return n

    def class_5min(self, k):
        """Class of 5-minute block k: block k covers minutes 5k..5k+4. Each minute is
        classified with `met_to_class` from its own MET and the sum of the five minutes starting
        at it; the block takes the second largest of the five classes (0 rest to 4 vigorous).
        Minutes past the day count as MET 0."""
        m = lambda i: self._m(i) if i < DAY_MIN else met32_from_u8(0)
        cs = sorted(met_to_class(2 * m(i), 2 * sum(m(j) for j in range(i, i + 5)), self.mod, self.vig)
                    for i in range(5 * k, 5 * k + 5))
        return cs[3]

    def class_minutes_periods(self, cls, start, count, periods, day_start):
        """The stateless variant: sleep periods as (start_s, end_s) epoch seconds relative
        to the day start. Also stores the last period's minutes as the state's range."""
        ranges = [(sleep_minutes(a, day_start), sleep_minutes(b, day_start)) for a, b in periods]
        if ranges and 1 in self.classes_series(self.last if count is None else count):
            self.sleep = ranges[-1]  # stored only when a sedentary minute reaches the check
        return self.class_minutes(cls, start, count, ranges=ranges)

    def met_minutes_times32(self, cls, n):
        total = 0
        for i, c in enumerate(self.classes_series(n)):
            m = self._m(i)
            if c == cls and m > 0x30:
                total += m - 0x30
        return total

    # -- inputs --
    def add_activity(self, last, steps, data, count=None):
        """Add one event (state, last, steps, count, data). The first `count + min(last - count, 0)` bytes of `data` to minutes
        max(last - count, 0).. with a per-minute maximum (when `last` is 1440 or more,
        only `last - 1439` bytes are copied), then advances the hourly sedentary
        counters."""
        count = len(data) if count is None else count
        end = last if (last & 0xFFFF) < DAY_MIN else DAY_MIN - 1
        diff = sign_extend_16(last - count)
        n_len = count if (last & 0xFFFF) < DAY_MIN else count - max(DAY_MIN - 1 - diff, 0)
        n_copy = (n_len + min(diff, 0)) & 0xFFFF
        start = max(diff, 0)
        for k in range(n_copy):
            if start + k < DAY_MIN:
                self.buf[start + k] = max(self.buf[start + k], data[k])
        if end > self.last:
            self.last = end
        self.steps = min(self.steps + steps, 0xFFFFFF)
        self._advance_hours()

    def add_minutes(self, end, steps, data):
        """The bytes of an `activity_information` event placed by absolute minute: `data[j]` is minute
        `end - len(data) + j` of the day, and only minutes 0..1439 are kept (maximum with what is there).
        The stateless assembly does this for each event overlapping the day,
        including events that end after the day boundary; `add_activity` copies the leading bytes of such
        events instead."""
        start = end - len(data)
        for j, v in enumerate(data):
            i = start + j
            if 0 <= i < DAY_MIN:
                self.buf[i] = max(self.buf[i], v)
        self.last = max(self.last, min(end, DAY_MIN - 1))
        self.steps = min(self.steps + steps, 0xFFFFFF)
        self._advance_hours()

    def complete_day(self, last_byte=0):
        """The stateless path of a completed day (1440 minutes). The whole 1440-byte array is read
        but minute 1439 counts only as a non-wear flag: a byte of 1 there is excluded from the class minutes,
        any other value behaves as the minimum byte (the minute adds nothing to the distance or MET-minutes)."""
        if last_byte == 1:
            self.buf[DAY_MIN - 1] = 1
        self.last = DAY_MIN
        self._advance_hours()

    def absorb_tail_non_wear(self):
        """End of the array in the stateless path: when the run of 0 and 1 bytes that ends at the last counted
        minute contains a non-wear byte (1), the whole run becomes non-wear. A tail of zeros only is left alone.
        Zeros next to a non-wear run elsewhere in the day stay minutes (rest or sedentary)."""
        end = min(self.last, DAY_MIN)
        i = end
        while i > 0 and self.buf[i - 1] in (0, 1):
            i -= 1
        if any(b == 1 for b in self.buf[i:end]):
            for k in range(i, end):
                self.buf[k] = 1
            self._advance_hours()

    def add_non_wear(self, end, length=0):
        """Mark non-wear (state, end, length): minutes end-length..end become
        non-wear (byte 1) unless they already hold a value above 9."""
        end &= 0xFFFF
        excess = max(end - 0x59F, 0)
        diff = sign_extend_16(end - length)
        ln = (length - excess + min(diff, 0)) & 0xFFFF
        start = max(diff, 0)
        for k in range(ln + 1):
            if start + k < DAY_MIN and self.buf[start + k] <= 9:
                self.buf[start + k] = 1
        self.last = max(self.last, min(end, 0x59F))
        self._advance_hours()

    def _advance_hours(self):
        c = self.cursor
        while c <= 0x5A0 and c != self.last:
            if c >= 60 and c % 60 == 0:
                self.hourly[c // 60 - 1] = self.class_minutes(1, c - 60, c) & 0xFF
            c += 1
            self.cursor = c

    def seed_hourly(self, prev_buf):
        """A completed day seeds the 24 hourly sedentary counters (state[0x65c..0x673]) from the
        previous day's per-minute MET bytes before it adds the day's own data, when the first byte of
        that array is nonzero. A scratch state with 1439 counted minutes and no sleep periods runs the
        hourly counter loop, so hours 0 to 22 get the previous day's sedentary minutes and hour 23 is 0.
        The day's own events overwrite hours as their cursor passes, so hours not yet elapsed keep the
        previous day's counts (the rule for a partial day)."""
        if not prev_buf or prev_buf[0] == 0:
            return
        self.hourly = [0] * 24
        t = ActInfo(self.user, user_met_limits=False)
        t.stateless = False  # the scratch state copies the array without add_activity: raw bytes stay
        t.mod, t.vig = self.mod, self.vig
        t.buf = list(prev_buf)[:DAY_MIN] + [0] * max(0, DAY_MIN - len(prev_buf))
        t.last = 0x59F
        t.ranges = []
        for c in range(60, 0x59F, 60):
            self.hourly[c // 60 - 1] = t.class_minutes(1, c - 60, c) & 0xFF

    def add_sit_alert(self):
        self.sit_alerts = (self.sit_alerts + 1) & 0xFF

    def set_target(self, readiness_pct=80):
        """Base target = trunc(0.5 * target). The
        event path substitutes 80 for a readiness percent of 0;
        this call does not."""
        u = self.user
        t = activity_target(readiness_pct, u.age, u.gender)
        self.base_target = to_int(f32(t * f32(0.5))) & 0xFFFF

    # -- outputs --
    def statistics(self, buf_count=0):
        u = self.user
        n = min(max(self.last, min(buf_count, 0x5A0)), 0x5A0)
        o = Stats()
        vals = [12 if b == 1 else (9 if b < 9 and self.stateless else b) for b in self.buf[:n]]
        o.eq_distance = sum(met10_to_eq_meters(met10_from_u8(v)) & 0xFFFF for v in vals) & M32
        o.non_wear = (self.non_wear_base + sum(1 for b in self.buf[:n] if b == 1)) & 0xFFFF
        o.classes = [self.class_minutes(c, 0, self.last) for c in range(5)]
        m32 = [met32_from_u8(v) & 0xFFFF for v in vals]
        o.steps = self.steps
        bmr16 = u.bmr & 0xFFFF
        if n:
            o.avg_met32 = ((sum(m32) + (n >> 1)) // n) & 0xFFFF
            acc = sum(act_calories(v, bmr16) for v, m in zip(vals, m32) if m >= 0x31) & M32
            o.active_cal = (((acc + 0x1F4) & M32) * 0x10624DD3 >> 38) & 0xFFFF
        bmr_shift = bmr16 << 8
        acc = bmr_shift
        for m in m32:
            if m >= 0x23:
                t = ((m - 0x22) * bmr_shift + 0x5A00) & M32
                p = s32(t) * s32(0xB60B60B7)
                w = (((p & 0xFFFFFFFFFFFFFFFF) >> 32) + t) & M32
                acc = (acc + s32((s32(w) >> 15) + (w >> 31))) & M32
        o.total_cal = (((acc + 0x80) & M32) >> 8) & 0xFFFF
        o.met_min = [(self.met_minutes_times32(c, n) + 16) >> 5 for c in (1, 2, 3, 4)]
        o.sit_alerts = self.sit_alerts

        mult, ttype = u.target_mult, u.target_type
        mult_frac = f32(1.0)
        if (mult - 25) & M32 <= 0xAF:
            mult_frac = f32(f32(mult) / f32(100.0))
        T = to_int(f32(mult_frac * f32(self.base_target))) & 0xFFFF
        t20 = (T * 20 + 0x324) & M32
        t2 = (T << 1) & M32
        steps_f = f32(f32(120 * T) / f32(5000.0))
        miles10 = (((t20 * 0x28BB1C) >> 32) + 4) & M32
        div_t = ((miles10 * 0x3334) & M32) >> 16
        div_t = (div_t + (div_t << 2)) & M32
        miles10 = (miles10 - ((miles10 - div_t) & 0xFFFF)) & M32
        o.target_distance_m = ((((t2 + 0x3E7) * 0x418938) >> 32) * 1000) & M32
        o.target_miles_x10 = miles10
        o.target_type, o.target_mult = ttype, mult
        cal5 = act_calories(met32_to_u8(160), u.bmr)
        v = f32(f32(steps_f * f32(cal5)) / f32(50000.0))
        o.target_cal = (((to_uint(np.float64(v) + 0.5) * 50000) & M32) * 0x10624DD3 >> 38) & M32
        o.target_steps = target_to_steps(t2, u.bmr, u.age, u.gender, u.weight_kg)
        if ttype == 1:
            d = s32(o.target_steps - o.steps)
            r = to_int(f32(f32(d) * F(0x3F43126F)))
            o.meters_to_target = (div_by_magic(r + 99, 0x51EB851F, 37) * 100) & M32
            o.miles_to_target_x10 = div_by_magic(r * 10 + 0x324, 0x5176370D, 41) & M32
            flag = 0 if o.target_steps == 0 else (2 if o.target_steps > o.steps else 1)
        else:
            if u.bmr:
                rem = s32(o.target_cal - o.active_cal)
                rem_scaled = f32(rem * 1000)
                v = f32(f32(rem_scaled / f32(cal5)) * f32(5.0))
                v = f32(f32(v * f32(1000.0)) / f32(60.0))
                r = to_int(np.float64(v) + 0.5)
                o.meters_to_target = (div_by_magic(r + 99, 0x51EB851F, 37) * 100) & M32
                o.miles_to_target_x10 = div_by_magic(r * 10 + 0x324, 0x5176370D, 41) & M32
            flag = 0 if o.target_cal == 0 else (2 if o.active_cal < o.target_cal else 1)

        eqd = (o.eq_distance >> 1) & 0xFFFF
        m = s32(o.meters_to_target)
        half = (m + 2 if m < -1 else m + 1) >> 1
        o.day = AS.Day(
            eq_distance=eqd,
            eq_target=max(0, min(0xFFFF, half + eqd)),
            sedentary_min=sum(self.hourly) & 0xFFFF,
            moderate_met_min=o.met_min[2] & 0xFFFF,
            vigorous_met_min=o.met_min[3] & 0xFFFF,
            sit_alerts=o.sit_alerts,
            age=u.age & 0xFF,
            gender=u.gender,
            moderate_min=min(o.classes[3], 0xFFFF),
            vigorous_min=min(o.classes[4], 0xFFFF),
            target_reached=flag,
        )
        return o


def day_from_minutes(met_bytes, user=None, steps=0, sit_alerts=0, readiness_pct=80, sleep=(0, 0)):
    """Convenience: feed one day of per-minute MET bytes in 5-minute events and return
    the statistics. `met_bytes` is padded with zeros to a whole number of events; the
    last minute handed over is excluded, so pass one extra minute."""
    a = ActInfo(user)
    a.sleep = sleep
    a.set_target(readiness_pct)
    for k in range(0, len(met_bytes), 5):
        chunk = met_bytes[k:k + 5]
        a.add_activity(k + len(chunk), steps if k == 0 else 0, chunk)
    for _ in range(sit_alerts):
        a.add_sit_alert()
    return a.statistics()
