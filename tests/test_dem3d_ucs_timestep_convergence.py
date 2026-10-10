from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dem3d_ucs_timestep_convergence import run_convergence_campaign


class DEM3DUCSTimestepConvergenceTests(unittest.TestCase):
    def test_campaign_reports_repeatability_and_delta_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_convergence_campaign(
                Path(tmp), base_steps=35, factors=(1.0, 0.5, 0.25), relative_tolerance=0.20
            )
            self.assertEqual(report["verdict"], "PASS", report)
            self.assertEqual(len(report["cases"]), 3)
            self.assertEqual(report["reference"], "finest timestep in this campaign")
            for case in report["cases"]:
                self.assertTrue(case["finite"])
                self.assertTrue(case["repeatable"])
                self.assertGreaterEqual(case["peak_stress_relative_delta_vs_fine"], 0.0)
                self.assertIn(case["convergence_status"], ("PASS", "WARN"))
            self.assertTrue((Path(tmp) / "ucs_timestep_convergence_report.json").is_file())
            self.assertTrue((Path(tmp) / "ucs_timestep_convergence_summary.csv").is_file())
            self.assertTrue((Path(tmp) / "ucs_timestep_convergence_history.csv").is_file())

    def test_invalid_tolerance_and_factors_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_convergence_campaign(Path(tmp), factors=(1.0, 0.0), base_steps=4)
            with self.assertRaises(ValueError):
                run_convergence_campaign(Path(tmp), relative_tolerance=-0.1, base_steps=4)

    def test_base_steps_must_be_positive_integer(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_convergence_campaign(Path(tmp), base_steps=0)
            with self.assertRaises(ValueError):
                run_convergence_campaign(Path(tmp), base_steps=2.5)


if __name__ == "__main__":
    unittest.main()
