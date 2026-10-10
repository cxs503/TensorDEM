"""Tests for energy-audit timestep sensitivity outputs and invariants."""
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dem3d_energy_timestep import run_campaign


class DEM3DEnergyTimestepTests(unittest.TestCase):
    def test_campaign_is_finite_repeatable_and_persists_all_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_campaign(Path(tmp), factors=(1.0, 0.5), base_steps=8)
            self.assertEqual(report["verdict"], "PASS", report)
            self.assertEqual(len(report["cases"]), 2)
            for case in report["cases"]:
                self.assertTrue(case["finite"])
                self.assertTrue(case["repeat_verified"])
                self.assertTrue(case["passed"])
                self.assertGreater(case["steps"], 0)
            self.assertTrue((Path(tmp) / "dem3d_energy_timestep_report.json").is_file())
            self.assertTrue((Path(tmp) / "dem3d_energy_timestep_summary.csv").is_file())
            self.assertTrue((Path(tmp) / "dem3d_energy_timestep_history.csv").is_file())

    def test_invalid_timestep_factors_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_campaign(Path(tmp), factors=(1.0, 0.0), base_steps=4)
            with self.assertRaises(ValueError):
                run_campaign(Path(tmp), factors=(1.0, 1.0), base_steps=4)

    def test_base_steps_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_campaign(Path(tmp), factors=(1.0,), base_steps=0)


if __name__ == "__main__":
    unittest.main()
