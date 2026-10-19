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

    def test_unsupported_geometry_rejected(self):
        with self.assertRaises(ValueError):
            aug.MeasurementProfiles().choose("facebook", "image", "low", "le2048",
                                             2000, 2000, random.Random(1))

    def test_instagram_collapsed(self):
        self.assertEqual(aug.pick_quality("instagram","image",random.Random(1)),"shared")
        self.assertEqual(aug.pick_quality("instagram","video",random.Random(1)),"shared")

if __name__ == "__main__":
    unittest.main()
