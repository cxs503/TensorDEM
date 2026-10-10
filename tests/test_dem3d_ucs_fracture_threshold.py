from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dem3d_ucs_fracture_threshold import run_fracture_threshold_screen


class DEM3DUCSFractureThresholdTests(unittest.TestCase):
    def test_threshold_screen_is_repeatable_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "screen"
            report = run_fracture_threshold_screen(out, target_strain=0.002)
            self.assertEqual(report["verdict"], "PASS", report["checks"])
            self.assertEqual(report["case_count"], 3)
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(
                [x["breaking_strain"] for x in report["cases"]],
                [0.01, 0.015, 0.02],
            )
            for name in (
                "ucs_fracture_threshold_report.json",
                "ucs_fracture_threshold_summary.csv",
                "ucs_fracture_threshold_histories.csv",
            ):
                self.assertTrue((out / name).is_file(), name)
            saved = json.loads(
                (out / "ucs_fracture_threshold_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["protocol"], "tensordem-dem3d-ucs-fracture-threshold-screen-v1")

    def test_invalid_thresholds_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_fracture_threshold_screen(Path(tmp), thresholds=(0.01,))
            with self.assertRaises(ValueError):
                run_fracture_threshold_screen(Path(tmp), thresholds=(0.01, 0.01))
            with self.assertRaises(ValueError):
                run_fracture_threshold_screen(Path(tmp), target_strain=0.0)


if __name__ == "__main__":
    unittest.main()
