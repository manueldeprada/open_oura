#!/usr/bin/env python3
"""Oura activity score: seven raw contributors and the weighted score.

Seven raw contributors in 0..100 are combined as round(sum(w_i * c_i) / 100):

    i  contributor          weight  driven by
    0  meet daily targets      25   share of the last 7 days that reached the daily target
    1  stay active             15   today's sedentary minutes
    2  move every hour         10   today's inactivity alerts (sit alerts)
    3  training frequency      10   share of days with a hard session (age/sex MET thresholds)
    4  training volume         15   recency-weighted MET-minutes over the window
    5  recovery, 7-day load    10   hard days (>45 vigorous minutes) among the last 7
    6  recovery, run length    15   consecutive hard days ending yesterday

Recovery time is (10*c5 + 15*c6) / 25. The recovery path uses the corrected recovery time and
the athlete-adjusted contributor.
"""
from dataclasses import dataclass, field

WEIGHTS = (25, 15, 10, 10, 15, 10, 15)

# gender code: 1 male, 2 female, 3 other (mean of the two), 0 unknown (all zero)
_MET = {  # category -> (peak at age 25, decrease per year of |age - 25|)
    1: ((2.4353, 0.005504), (4.832, 0.026), (9.2, 0.0625)),
    2: ((2.515, 0.004675), (5.224, 0.0218), (10.2, 0.05125)),
}


@dataclass
class Day:
    """One day record. Field names follow the
    byte offsets; the meaning is inferred from the score's use of each field."""
    eq_distance: int = 0       # equivalent meters walked
    eq_target: int = 0         # daily target in the same unit
    sedentary_min: int = 0
    moderate_met_min: int = 0
    vigorous_met_min: int = 0
    sit_alerts: int = 0        # today only
    age: int = 0               # today only
    gender: int = 0            # today only
    moderate_min: int = 0
    vigorous_min: int = 0
    target_reached: int = 0    # 1 reached, 0 compare eq_distance with eq_target, else not

    def has_data(self):
        return (self.eq_distance + self.sedentary_min + self.moderate_met_min
                + self.vigorous_met_min + self.moderate_min + self.vigorous_min) != 0

    def met_target(self):
        return self.target_reached == 1 or (self.target_reached == 0 and self.eq_distance >= self.eq_target)


def _interp(x, xs, ys):
    """Piecewise-linear interpolation clamped to the end points; xs ascending."""
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[-1]


def met_thresholds(age, gender):
    """(moderate, vigorous) MET category values for a user; zeros for unknown gender."""
    if gender == 0:
        return 0.0, 0.0
    cols = ([_MET[1], _MET[2]] if gender == 3 else [_MET[gender]])
    out = []
    for k in (1, 2):
        vals = [peak - slope * abs(age - 25) for peak, slope in (c[k] for c in cols)]
        out.append(sum(vals) / len(vals))
    return out[0], out[1]


_Y = (0, 25, 95, 100)


# Recovery defaults, used when no override is set.
DISTANCE_REF = 2000.0
VIGOROUS_REF = 160.0


def contributors(days, ring_day=100):
    """days: oldest to newest, last item is today; at most 9 history days are read.
    ring_day: days since the ring was first used (`j`); the recovery
    contributors use a looser definition of a hard day from day 10 on.
    Returns the seven raw contributors in 0..100."""
    today = days[-1]
    hist = days[:-1][-9:]
    week = (hist[-6:] + [today])
    n_slots = len(week)

    # c0: meet daily targets
    met = days_with_data = 0.0
    for d in week[:-1]:
        if d.has_data():
            days_with_data += 1
            met += d.met_target()
    if today.has_data():
        if today.met_target():
            met += 1
            days_with_data += 1
        elif days_with_data > 1 or met != 0:
            days_with_data += 1
        else:  # first days of use: an unmet target still counts as met
            met += 1
            days_with_data += 2
    frac = min(max(days_with_data / 7.0, 0.0), 1.0)
    x = met / frac if frac > 0 else met
    c0 = 0.0 if x <= 0 else min(100.0, _interp(x, (0, 1, 5, 6), _Y))

    # c1, c2
    c1 = _interp(today.sedentary_min, (300, 400, 955, 1020), (100, 95, 25, 0)) if today.sedentary_min <= 1019 else 0.0
    c2 = _interp(today.sit_alerts, (0, 1, 5, 10), (100, 95, 25, 0)) if today.sit_alerts <= 9 else 0.0

    # c3: training frequency
    mod_thr, vig_thr = met_thresholds(today.age, today.gender)
    thr_v, thr_m = (vig_thr - 1.5) * 10, (mod_thr - 1.5) * 45
    hard = n = 0.0
    for d in week[:-1]:
        if d.has_data():
            n += 1
            hard += (d.vigorous_met_min > thr_v) or (d.moderate_met_min + d.vigorous_met_min > thr_m)
    if today.has_data():
        if (today.vigorous_met_min > thr_v) or (today.moderate_met_min + today.vigorous_met_min > thr_m):
            hard += 1
            n += 1
        else:
            if n <= 1 and hard == 0:
                hard += 1
                n += 1
            n += 1
    frac = min(max(n / 7.0, 0.0), 1.0)
    x = hard / frac if frac > 0 else hard
    c3 = 0.0 if x <= 0 else min(100.0, _interp(x, (0, 0.5, 2.8, 4.0), _Y))

    # c4: training volume
    present = [d for d in week if d.has_data()]
    if present:
        w = (0.19, 0.42, 0.77, 1.09, 1.34, 1.53, 1.66)
        vol = 7.0 / len(present) * sum(w[k] * (d.moderate_met_min + d.vigorous_met_min) for k, d in enumerate(present))
        c4 = _interp(vol, (0, 150, 850, 2000), _Y)
    else:
        c4 = 0.0

    # c5, c6: recovery
    if ring_day <= 9:
        def is_hard(d):
            return d.vigorous_min > 45
    else:
        dist_thr = max(DISTANCE_REF * 1.01, 4000.0)
        vig_thr = max(VIGOROUS_REF * 0.75 * 0.125, 10.0)

        def is_hard(d):
            return dist_thr < d.eq_distance or vig_thr < d.vigorous_min
    count = sum(is_hard(d) for d in week)
    c5 = {6: 95.0, 7: 0.0}.get(count, 100.0 if count < 6 else 0.0)
    run = 0
    for d in reversed(hist):
        if not is_hard(d):
            break
        run += 1
    c6 = _interp(run + 1, (2, 3, 7, 9), (100, 95, 25, 0)) if run + 1 >= 2 else 100.0
    return [c0, c1, c2, c3, c4, c5, c6]


def score(days, ring_day=100):
    """(score 0..100, contributors, recovery_time_contributor)."""
    c = contributors(days, ring_day)
    total = sum(w * v for w, v in zip(WEIGHTS, c)) / 100.0
    return int(total + 0.5), c, (10 * c[5] + 15 * c[6]) / 25
