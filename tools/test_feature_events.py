import unittest

from feature_events import (COLUMN_BITS, decode_ehr_acm_intensity, decode_ehr_trace,
                            decode_feature_session, pair_real_steps, unpack_real_steps)

H = bytes.fromhex


class FeatureEvents(unittest.TestCase):
    def test_unpack_real_steps(self):
        # synthetic record pair
        cols = unpack_real_steps(H("644B5A9E3C142D4669507D8C2133"), H("1219963246B70800406396285A56"))
        self.assertEqual(cols, [602, 40, 90, 201, 150, 181, 30, 60, 20, 45, 70, 211, 160, 251,
                                12, 33, 51, 18, 25, 301, 100, 141, 55, 8, 0, 64, 99])
        self.assertTrue(all(0 <= c < (1 << b) for c, b in zip(cols, COLUMN_BITS)))

    def test_pair_requires_next_decisecond(self):
        p1, p2 = H("644B5A9E3C142D4669507D8C2133"), H("1219963246B70800406396285A56")
        self.assertEqual(len(pair_real_steps([(10, 0x7E, p1), (11, 0x7F, p2)])), 1)
        self.assertEqual(pair_real_steps([(10, 0x7E, p1), (12, 0x7F, p2)]), [])

    def test_ehr_trace(self):
        v = decode_ehr_trace(H("02013C42054B5004787D03505C01"))
        self.assertEqual((v["counter"], v["trace"], len(v["peaks"])), (2, "ppg", 4))
        self.assertEqual(v["peaks"][0], {"freq1": 60, "freq2": 66, "pov_x10": 5})
        self.assertEqual(len(decode_ehr_trace(H("0100A0A502"))["peaks"]), 1)
        self.assertIsNone(decode_ehr_trace(bytes([0, 0, 1, 2])))

    def test_ehr_acm_intensity(self):
        self.assertEqual(decode_ehr_acm_intensity(H("0C000F0013001A001F0028002C00"))["intensity"],
                         [12, 15, 19, 26, 31, 40, 44])
        self.assertEqual(decode_ehr_acm_intensity(H("3C00340037003D00"))["intensity"], [60, 52, 55, 61])

    def test_feature_session(self):
        self.assertEqual(decode_feature_session(H("0B0100"))["status_name"], "start_by_system")
        v = decode_feature_session(H("03020300000000"))
        self.assertEqual((v["feature_id"], v["status_name"], v["value"]), (3, "end_user_request", 3))


if __name__ == "__main__":
    unittest.main()
