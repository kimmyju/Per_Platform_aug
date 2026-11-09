"""Tests for embedded empirical profiles. No measurement CSV needed."""
import importlib.util
import random
from pathlib import Path
import unittest

path = Path(__file__).resolve().parent / "platform_aware_augment_standalone.py"
spec = importlib.util.spec_from_file_location("platform_aug", path)
aug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aug)

class EmpiricalPolicyTests(unittest.TestCase):
    def test_cardinality(self):
        self.assertEqual(len(aug.EMPIRICAL_POLICY), 12)
        self.assertEqual(sum(len(rows) for cells in aug.EMPIRICAL_POLICY.values()
                             for rows in cells.values()), 360)
        self.assertEqual(len(aug.EMBEDDED_SOURCE_SHA256), 12)

    def test_joint_values_and_seed(self):
        policy = aug.MeasurementProfiles()
        def choose(seed):
            return policy.choose("facebook", "image", "low", "le2048",
                                 1600, 2048, random.Random(seed))[0]
        self.assertEqual(choose(42), choose(42))
        self.assertGreater(len({choose(i)["reference_profile_row_ordinal"]
                                for i in range(100)}), 1)
        selected = choose(42)
        matched = next(row for row in policy.data["facebook|image|low"]["le2048/port"]
                       if row["n"] == selected["reference_profile_row_ordinal"])
        self.assertEqual(float(selected["brightness_offset_b"]), matched["b"])
        self.assertEqual(float(selected["contrast_gain_a"]), matched["a"])
        self.assertEqual(float(selected["down_estimated_jpeg_qf"]), matched["qf"])

    def test_unsupported_geometry_uses_nearest_empirical_row(self):
        policy = aug.MeasurementProfiles()
        first, _ = policy.choose("facebook", "image", "low", "le2048",
                                 2000, 2000, random.Random(1))
        repeat, _ = policy.choose("facebook", "image", "low", "le2048",
                                  2000, 2000, random.Random(1))
        self.assertEqual(first, repeat)
        self.assertTrue(first["is_extrapolated"])
        self.assertEqual(first["selection_method"], "nearest_same_band")
        self.assertTrue(first["selected_condition"].startswith("le2048/"))
        self.assertEqual(first["requested_condition"], "le2048/square")
        self.assertEqual(first["target_id"], "facebook|image|low")

    def test_strict_mode_preserves_rejection(self):
        with self.assertRaisesRegex(ValueError, "No measured profile"):
            aug.MeasurementProfiles().choose("facebook", "image", "low", "le512",
                                             640, 480, random.Random(1),
                                             fallback_policy="strict")

    def test_landscape_le512_kakao_fallback(self):
        policy = aug.MeasurementProfiles()
        first, _ = policy.choose("kakao", "image", "low", "le512",
                                 640, 480, random.Random(42))
        second, _ = policy.choose("kakao", "image", "low", "le512",
                                  640, 480, random.Random(42))
        self.assertEqual(first, second)
        self.assertEqual(first["requested_condition"], "le512/land")
        self.assertEqual(first["selected_condition"], "le512/port")
        self.assertEqual(first["selection_method"], "nearest_same_band")
        self.assertEqual(first["target_id"], "kakao|image|low")
        self.assertTrue(first["is_extrapolated"])

    def test_exact_same_row_and_random_draw(self):
        policy = aug.MeasurementProfiles()
        a, _ = policy.choose("instagram", "image", "shared", "le512",
                             544, 704, random.Random(42))
        b, _ = policy.choose("instagram", "image", "shared", "le512",
                             544, 704, random.Random(42),
                             fallback_policy="strict")
        self.assertEqual(a, b)
        self.assertFalse(a["is_extrapolated"])
        self.assertEqual(a["selection_method"], "exact")

    def test_missing_band_fallback_stays_in_platform_quality(self):
        policy = aug.MeasurementProfiles()
        row, _ = policy.choose("kakao", "image", "normal", "unmeasured",
                               1000, 1000, random.Random(42))
        self.assertEqual(row["target_id"], "kakao|image|normal")
        self.assertEqual(row["selection_method"], "nearest_other_band")
        self.assertTrue(row["is_extrapolated"])

    def test_instagram_collapsed(self):
        self.assertEqual(aug.pick_quality("instagram","image",random.Random(1)),"shared")
        self.assertEqual(aug.pick_quality("instagram","video",random.Random(1)),"shared")

if __name__ == "__main__":
    unittest.main()
