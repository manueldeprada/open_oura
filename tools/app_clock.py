#!/usr/bin/env python3
"""Ring time to Unix time from the ring-written `time_sync` events.

Every ring event carries a counter in deciseconds. The ring also writes `time_sync` events
(tag 0x42) holding its own idea of the Unix time; the client adds `"source": "phone"` events
of its own, which this conversion does not use. An event converts with the latest ring-sourced
`time_sync` captured at or before the event's own capture:

    unix = sync.unix_time + (event_ds - sync.ds) / 10

The nearest-anchor rule in `epoch_time.py` differs from this by up to a few minutes, which
moves every 30 s epoch of a night. To place the events of a night, parse a batch of raw
events together with the time events that bracket it, and use only the ring-written ones
among them.

    before = the latest time_sync of either source at or before the batch's first event
    after  = the earliest time_sync of either source at or after the batch's last event
    usable = ring-written time_syncs from `before` through `after`

An event takes the latest usable sync at or before its ring time, or the earliest usable one
after it when none precedes. Phone-written anchors (`"source": "phone"`, inserted by the client
after each drain) bound the window but are never used. `window(first_ds, last_ds)` returns this
converter.
"""
import bisect
import json


class AppClock:
    def __init__(self, con):
        self.syncs = []  # (captured_unix, ring_ds, unix_time) in capture order
        for ds, cu, js in con.execute("SELECT ring_timestamp, captured_unix, decoded_json FROM events "
                                      "WHERE tag = 66 AND decoded_json IS NOT NULL ORDER BY captured_unix, id"):
            v = json.loads(js)
            if v.get("source") != "phone" and v.get("unix_time") is not None:
                self.syncs.append((cu, ds, int(v["unix_time"])))
        self.captured = [s[0] for s in self.syncs]
        self._ring_by_ds = sorted(self.syncs, key=lambda s: (s[1], s[0]))
        # every time_sync, ring or phone: (ring_ds, unix_time, is_ring), by ring time
        self.all_syncs = []
        for ds, js in con.execute("SELECT ring_timestamp, decoded_json FROM events WHERE tag = 66 "
                                  "AND decoded_json IS NOT NULL ORDER BY ring_timestamp, id"):
            v = json.loads(js)
            if v.get("unix_time") is not None:
                self.all_syncs.append((ds, int(v["unix_time"]), v.get("source") != "phone"))
        self._all_ds = [s[0] for s in self.all_syncs]

    def ring_order(self, ds, anchor_ds):
        """Unix seconds for ring time `ds` with the latest ring-written sync whose ring time is at or before
        `anchor_ds` (the ring time of the event that carries `ds`, e.g. the `bedtime_period` event for its
        start and end), or None without one. This is the stream order of a sync, as
        `bedtime_detect.RingOrderClock` uses it."""
        i = bisect.bisect_right([s[1] for s in self._ring_by_ds], anchor_ds) - 1
        if i < 0:
            return None
        sync_ds, unix = self._ring_by_ds[i][1:]
        return unix + (ds - sync_ds) / 10.0

    def window(self, first_ds, last_ds):
        """Converter `f(ds) -> unix seconds` for a batch of events from `first_ds` to `last_ds`
        (see the module docstring), or None when no ring-written sync lies in the bracket."""
        i = bisect.bisect_right(self._all_ds, first_ds) - 1
        j = bisect.bisect_left(self._all_ds, last_ds)
        lo = max(i, 0)
        hi = min(j, len(self.all_syncs) - 1)
        usable = sorted({(d, u) for d, u, ring in self.all_syncs[lo:hi + 1] if ring})
        if not usable:
            return None
        ds_list = [d for d, _ in usable]

        def convert(ds):
            k = max(bisect.bisect_right(ds_list, ds) - 1, 0)
            d, u = usable[k]
            return u + (ds - d) / 10.0
        return convert

    def unix_s(self, ds, captured_unix):
        """Unix seconds for ring time `ds` captured at `captured_unix`, or None without a ring sync before it."""
        if captured_unix is None:
            return None
        i = bisect.bisect_right(self.captured, captured_unix) - 1
        if i < 0:
            return None
        _, sync_ds, unix = self.syncs[i]
        return unix + (ds - sync_ds) / 10.0
