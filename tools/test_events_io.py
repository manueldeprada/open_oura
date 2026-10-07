#!/usr/bin/env python3
"""Tests of the input assembly in tools/events_io.py and tools/app_clock.py (no model needed).

    python tools/test_events_io.py
"""
import json
import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import events_io as RS
from app_clock import AppClock


def db(events):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ring_timestamp INT, tag INT, decoded_json TEXT, captured_unix INT)")
    for ds, tag, v, cu in events:
        con.execute("INSERT INTO events (ring_timestamp, tag, decoded_json, captured_unix) VALUES (?, ?, ?, ?)",
                    (ds, tag, json.dumps(v), cu))
    return con


class AppClockRule(unittest.TestCase):
    def test_latest_ring_sync_captured_at_or_before_the_event(self):
        con = db([
            (1000, 66, {"unix_time": 5000}, 100),
            (2000, 66, {"unix_time": 5100}, 200),                       # later ring sync
            (3000, 66, {"unix_time": 9999, "source": "phone"}, 150),    # phone syncs are ignored
            (4000, 66, {"unix_time": 5200}, 300),                       # captured after the events below
        ])
        clock = AppClock(con)
        self.assertEqual(clock.unix_s(1500, 150), 5000 + 50.0)      # only the first sync is captured by 150
        self.assertEqual(clock.unix_s(2600, 200), 5100 + 60.0)      # ties go to the sync captured at 200
        self.assertEqual(clock.unix_s(2600, 299), 5100 + 60.0)      # the sync captured at 300 is not used
        self.assertIsNone(clock.unix_s(1500, 99))                   # no sync yet

    def test_a_sync_after_the_event_in_ring_time_still_applies(self):
        con = db([(5000, 66, {"unix_time": 7000}, 10)])
        self.assertEqual(AppClock(con).unix_s(4000, 20), 7000 - 100.0)


class AppClockWindow(unittest.TestCase):
    """Window rule: ring syncs between the sync before the first event and the sync after the last."""

    def clock(self):
        return AppClock(db([
            (1000, 66, {"unix_time": 5000}, 1),                         # ring
            (2000, 66, {"unix_time": 6000, "source": "phone"}, 2),      # phone: bounds the window, never used
            (4000, 66, {"unix_time": 7000}, 3),                         # ring
            (6000, 66, {"unix_time": 8000, "source": "phone"}, 4),
        ]))

    def test_latest_usable_sync_before_the_event(self):
        f = self.clock().window(1100, 1500)    # window ds 1000..2000: the ring sync at 1000
        self.assertEqual(f(1200), 5000 + 20.0)

    def test_phone_anchor_before_the_batch_excludes_earlier_ring_syncs(self):
        # the batch starts after the phone sync at ds 2000, so the ring sync at 1000 is outside the window
        f = self.clock().window(2500, 3500)
        self.assertEqual(f(3000), 7000 - 100.0)         # only the sync at 4000 is usable; it follows the events

    def test_ring_sync_inside_the_window_is_preferred_to_the_one_after(self):
        f = self.clock().window(1500, 4500)             # window ds 1000..6000: ring syncs 1000 and 4000
        self.assertEqual(f(3500), 5000 + 250.0)
        self.assertEqual(f(4200), 7000 + 20.0)

    def test_no_ring_sync_in_the_window(self):
        self.assertIsNone(self.clock().window(2000, 2000))   # bracketed by the phone sync at 2000 alone


class CollectNight(unittest.TestCase):
    def setUp(self):
        sync = (0, 66, {"unix_time": 1_800_000_000}, 1)
        self.events = [sync,
                       (100, 96, {"ibi_ms": [1000, 1000, 1000], "amplitude": [10, 20, 30]}, 2),   # beats end at 10 s
                       (3100, 96, {"ibi_ms": [1000, 1000, 1000], "amplitude": [1, 2, 3]}, 2),
                       (500, 114, {"acm_mad": [1, 2, 3, 4, 5, 6]}, 2),
                       (1500, 117, {"temps_c": [35.0, 36.0, 37.0]}, 2),
                       (9000, 114, {"acm_mad": [9, 9, 9, 9, 9, 9]}, 2)]
        self.con = db(self.events)
        rows = self.con.execute("SELECT ring_timestamp, tag, decoded_json, captured_unix FROM events ORDER BY id").fetchall()
        self.rows = rows
        self.unix_ms = RS.make_unix_ms(self.con, lambda ds, cu=None: 0.0, "app")

    def test_selection_and_stamps(self):
        start_ms, end_ms, beats, acm, temp = RS.collect_night(self.rows, self.unix_ms, 200, 3200, 2)
        base = 1_800_000_000_000
        self.assertEqual((start_ms, end_ms), (base + 20_000, base + 320_000))
        # beat i of an event is stamped event time minus the sum of ibi[i:]: the start of its interval
        self.assertEqual([b[0] - base for b in beats], [307_000, 308_000, 309_000])  # the first event ends before the window
        self.assertEqual([a[0] - base for a in acm], [50_000])
        self.assertEqual(acm[0][1], [6, 5, 4, 3, 2, 1])  # [MAD iqr, max, mean, dANGLEZ iqr, max, mean]
        # the last temperature sample sits at the event time, the others 30 s apart before it
        self.assertEqual([(t - base, v) for t, v in temp], [(90_000, 35.0), (120_000, 36.0), (150_000, 37.0)])

    def test_beats_before_the_window_start_are_dropped(self):
        # window starts 1 s into the first event: its first beat (stamped 8 s) is outside
        start_ds = 100 + 20
        _, _, beats, _, _ = RS.collect_night(self.rows, self.unix_ms, start_ds, 3200, 2)
        base = 1_800_000_000_000
        self.assertTrue(all(b[0] >= base + start_ds * 100 for b in beats))


class Scalars(unittest.TestCase):
    def test_profile(self):
        age, bmi, sex = RS.profile_scalars({"age": 41, "height_m": 1.80, "weight_kg": 78, "sex": "M"})
        self.assertEqual((age, sex), (41.0, 2.0))
        self.assertAlmostEqual(bmi, 78 / 1.80 ** 2)
        self.assertEqual(RS.profile_scalars({"sex": "F"})[2], 1.0)
        self.assertEqual(RS.profile_scalars({}), [-1.0, -1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
