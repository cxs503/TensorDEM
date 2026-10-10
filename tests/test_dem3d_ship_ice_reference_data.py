from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks" / "dem3d" / "ship_ice_validation" / "mt_uikku_level_ice.csv"


class MTUikkuReferenceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with REFERENCE.open(newline="", encoding="utf-8") as stream:
            cls.rows = list(csv.DictReader(stream))

    def test_published_level_ice_cases_are_present(self):
        self.assertEqual(
            [row["test_id"] for row in self.rows],
            ["103", "104", "205", "206", "301", "302", "303", "401", "402", "403"],
        )
        self.assertTrue(all(row["ice_type"] == "level" for row in self.rows))
        self.assertTrue(all(float(row["ice_thickness_m"]) > 0 for row in self.rows))
        self.assertTrue(all(float(row["experimental_reference_table8_kN"]) > 0 for row in self.rows))

    def test_published_force_statistics_and_formula_baselines_are_transcribed(self):
        by_id = {row["test_id"]: row for row in self.rows}
        self.assertEqual(float(by_id["103"]["measured_mean_resistance_table3_kN"]), 480.0)
        self.assertEqual(float(by_id["103"]["experimental_reference_table8_kN"]), 470.0)
        self.assertEqual(float(by_id["103"]["lindqvist_prediction_kN"]), 320.0)
        self.assertEqual(float(by_id["303"]["jeong_prediction_kN"]), 961.0)
        self.assertEqual(float(by_id["402"]["keinonen_prediction_kN"]), 526.0)

    def test_unreported_statistics_remain_missing(self):
        by_id = {row["test_id"]: row for row in self.rows}
        for test_id in ("401", "402", "403"):
            self.assertEqual(by_id[test_id]["measured_std_kN"], "")
            self.assertEqual(by_id[test_id]["measured_max_kN"], "")
            self.assertEqual(by_id[test_id]["measured_min_kN"], "")


if __name__ == "__main__":
    unittest.main()
