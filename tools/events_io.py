#!/usr/bin/env python3
"""Input assembly for the night scorers: read ring events from the SQLite log, convert ring time
to Unix time, and collect the IBI, motion and temperature samples of one bedtime window.

Timestamps are Unix milliseconds unless a name ends in `_ds` (ring deciseconds). Uses only the
standard library, `app_clock.py` and `epoch_time.py`.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_clock import AppClock  # noqa: E402
from epoch_time import build_epochs, is_dated, make_unix_s  # noqa: E402

EPOCH_MS = 30000
MAX_SKEW_MS = 600000  # events whose clock disagrees with the bedtime by more than this belong to another boot
SEX_CODE = {"M": 2.0, "MALE": 2.0, "F": 1.0, "FEMALE": 1.0}


def make_unix_ms(con, unix_s, clock="app"):
    """ds -> Unix ms. `clock="app"` converts with the ring-written syncs (`app_clock.py`) and falls back to
    `epoch_time` for events before the first ring sync; `"epoch"` always uses `epoch_time`."""
    app = AppClock(con) if clock == "app" else None

    def unix_ms(ds, cu=None):
        u = app.unix_s(ds, cu) if app else None
        if u is None:
            u = unix_s(ds, cu)
        return int(round(u * 1000)) if u is not None else int(ds * 100)

    def window(first_ds, last_ds):
        """Converter for one batch of events (app_clock.py)."""
        f = app.window(first_ds, last_ds) if app else None
        return (lambda ds, cu=None: int(round(f(ds) * 1000))) if f else unix_ms
    def bedtime(event_ds):
        """Converter for the start and end of the `bedtime_period` event at ring time `event_ds`."""
        return lambda ds, cu=None: (int(round(app.ring_order(ds, event_ds) * 1000))
                                    if app and app.ring_order(ds, event_ds) is not None else unix_ms(ds, cu))
    unix_ms.window = window
    unix_ms.bedtime = bedtime
    return unix_ms


def load_events(con):
    rows = con.execute("SELECT ring_timestamp, tag, COALESCE(decoded_json, '{}'), captured_unix FROM events "
                       "WHERE decoded_json IS NOT NULL OR (tag = 65 AND LENGTH(body) >= 14) "
                       "ORDER BY captured_unix, id").fetchall()
    epochs = build_epochs(rows)
    return rows, epochs, make_unix_s(epochs)


def profile_scalars(profile):
    """[age, BMI, sex] as the model takes them; unknown values use the model's own sentinels."""
    age = float(profile.get("age", -1))
    h, w = profile.get("height_m"), profile.get("weight_kg")
    bmi = float(w) / float(h) ** 2 if h and w else -1.0
    sex = SEX_CODE.get(str(profile.get("sex", "")).upper(), -1.0)
    return [age, bmi, sex]


def collect_night(rows, unix_ms, start_ds, end_ds, captured_unix):
    """Beats, motion rows and temperature rows of one bedtime window.

    Every sample is kept whose own timestamp lies in [bedtime start, bedtime end]."""
    bed_ds = [ds for ds, tag, js, cu in rows if tag == 118 and cu == captured_unix and json.loads(js).get("bedtime_start_ds") == start_ds]
    bed_ms = unix_ms.bedtime(bed_ds[-1]) if bed_ds and hasattr(unix_ms, "bedtime") else unix_ms
    start_ms, end_ms = bed_ms(start_ds, captured_unix), bed_ms(end_ds, captured_unix)
    # the events of the night are parsed as one batch with the syncs that bracket it
    in_night = [ds for ds, tag, _, _ in rows if tag not in (65, 66) and start_ds <= ds <= end_ds]
    event_ms = unix_ms.window(min(in_night), max(in_night)) if in_night and hasattr(unix_ms, "window") else unix_ms
    beats, acm, temp = [], [], []
    for ds, tag, js, cu in rows:
        if tag not in (0x60, 0x72, 0x75) or not start_ds - 6000 <= ds <= end_ds + 6000:
            continue
        t = event_ms(ds, cu)
        if abs(t - (start_ms + (ds - start_ds) * 100)) > MAX_SKEW_MS:
            continue
        v = json.loads(js)
        if tag == 0x60:
            ibis, amps = v.get("ibi_ms", []), v.get("amplitude", [])
            for i, ibi in enumerate(ibis):
                ts = t - sum(ibis[i:])
                if start_ms <= ts <= end_ms:
                    beats.append((ts, ibi, amps[i] if i < len(amps) else 0))
        elif tag == 0x72:
            m = v["acm_mad"]
            if start_ms <= t <= end_ms:
                acm.append((t, [m[5], m[4], m[3], m[2], m[1], m[0]]))
        elif v.get("temps_c"):
            n = len(v["temps_c"])
            temp += [(ts, float(x)) for i, x in enumerate(v["temps_c"])
                     if start_ms <= (ts := t - EPOCH_MS * (n - 1 - i)) <= end_ms]
    # beats stay in stream order: the corrector is stateful and drops beats that go back in time
    return start_ms, end_ms, beats, sorted(acm), sorted(temp)


def last_bedtime(con):
    r = con.execute("SELECT decoded_json, captured_unix FROM events WHERE tag = 118 ORDER BY captured_unix DESC, id DESC").fetchone()
    if r is None:
        sys.exit("no bedtime_period (tag 0x76) in DB; pass start/end deciseconds or sync overnight data first")
    v = json.loads(r[0])
    return v["bedtime_start_ds"], v["bedtime_end_ds"], r[1]


def find_profile(db, explicit):
    for p in ([Path(explicit)] if explicit else []) + [Path(db).parent / "profile.json"]:
        if p.is_file():
            return json.loads(p.read_text())
    return {}
