"""Decoders for the events written by ring features REAL_STEPS (0x0b), EXERCISE_HR (0x03) and
feature sessions (0x6c). Mirrors crates/oura-protocol/src/events.rs; no torch needed.

Layouts: the 0x7e/0x7f pairing and the 27-column unpack; 0x73 is inferred (I); 0x74 is u16 LE.
See docs/ring-feature-modes.md.
"""

SESSION_STATUS = {
    0: "start_by_user", 1: "start_by_system", 2: "end_user_request", 3: "end_normal_period",
    4: "end_too_much_movement", 5: "end_low_battery", 6: "end_firmware_update",
    7: "end_unexpected_error", 8: "end_ring_charging", 9: "end_resting_hr_detection",
    10: "end_no_movement", 11: "end_ring_not_in_finger", 12: "start_by_ring_awhr_feature",
}

# Bit width of each of the 27 columns (globals, then three blocks of eight).
COLUMN_BITS = [10, 8, 8] + ([9, 9, 9, 7, 8, 8, 8, 8] * 3)


def unpack_real_steps(p1, p2):
    """27 quantized columns from the 14-byte bodies of 0x7e (p1) and 0x7f (p2)."""
    if len(p1) != 14 or len(p2) != 14:
        raise ValueError("real_step bodies are 14 bytes")
    c = p2[13]
    return [
        (p2[10] << 2) | (c & 0x3), p2[11], p2[12],
        (p1[0] << 1) | (p1[3] >> 7), (p1[1] << 1) | ((c >> 7) & 1), (p1[2] << 1) | ((c >> 6) & 1),
        p1[3] & 0x7F, p1[4], p1[5], p1[6], p1[7],
        (p1[8] << 1) | (p1[11] >> 7), (p1[9] << 1) | ((c >> 5) & 1), (p1[10] << 1) | ((c >> 4) & 1),
        p1[11] & 0x7F, p1[12], p1[13], p2[0], p2[1],
        (p2[2] << 1) | (p2[5] >> 7), (p2[3] << 1) | ((c >> 3) & 1), (p2[4] << 1) | ((c >> 2) & 1),
        p2[5] & 0x7F, p2[6], p2[7], p2[8], p2[9],
    ]


def pair_real_steps(events):
    """events: iterable of (ring_timestamp, tag, body). Returns [(timestamp, 27 columns)] for every
    0x7e whose 0x7f follows one decisecond later."""
    first, second = [], {}
    for ts, tag, body in events:
        if tag == 0x7E and len(body) == 14:
            first.append((ts, body))
        elif tag == 0x7F and len(body) == 14:
            second[ts] = body
    return [(ts, unpack_real_steps(b, second[ts + 1])) for ts, b in first if ts + 1 in second]


def decode_ehr_trace(body):
    """0x73: counter, trace type (0 ACM, 1 PPG), then (freq1, freq2, pov_x10) triples. Inferred."""
    if len(body) < 5 or (len(body) - 2) % 3:
        return None
    peaks = [{"freq1": body[i], "freq2": body[i + 1], "pov_x10": body[i + 2]}
             for i in range(2, len(body), 3)]
    return {"counter": body[0], "trace_type": body[1],
            "trace": {0: "acm", 1: "ppg"}.get(body[1], "unknown"), "peaks": peaks}


def decode_ehr_acm_intensity(body):
    """0x74: up to seven u16 LE intensity values."""
    if not body or len(body) % 2:
        return None
    return {"intensity": [int.from_bytes(body[i:i + 2], "little") for i in range(0, len(body), 2)]}


def decode_feature_session(body):
    """0x6c: feature id, status, then a feature-specific payload."""
    if len(body) < 2:
        return None
    v = {"feature_id": body[0], "session_status": body[1]}
    if body[1] in SESSION_STATUS:
        v["status_name"] = SESSION_STATUS[body[1]]
    if len(body) >= 4:
        v["value"] = int.from_bytes(body[2:4], "little")
    if len(body) > 2:
        v["payload"] = body[2:].hex()
    return v
