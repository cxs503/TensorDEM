from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dem3d_wedge_resistance_sensitivity import run_sensitivity_campaign


class WedgeResistanceSensitivityTests(unittest.TestCase):
    def test_campaign_is_repeatable_and_persists_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_sensitivity_campaign(
                Path(tmp), base_steps=8,
                speeds_m_s=(0.05, 0.1), angles_deg=(30.0, 45.0),
                dt_factors=(1.0, 0.5, 0.25),
            )
            self.assertEqual(report["verdict"], "PASS", report)
            self.assertEqual(report["case_count"], 10)
            self.assertTrue(all(report["checks"].values()))
            self.assertTrue(all(case["finite"] and case["repeatable"] for case in report["cases"]))
            timestep_cases = [c for c in report["cases"] if c["sweep"] == "timestep"]
            self.assertEqual([c["steps"] for c in timestep_cases], [8, 16, 32])
            width_cases = [c for c in report["cases"] if c["sweep"] == "bow_half_width"]
            self.assertEqual([c["value"] for c in width_cases], [0.02, 0.03, 0.04])
            self.assertTrue(all(c["bow_half_width_m"] == c["value"] for c in width_cases))
            for filename in (
                "wedge_resistance_sensitivity_report.json",
                "wedge_resistance_sensitivity_summary.csv",
                "wedge_resistance_sensitivity_history.csv",
            ):
                self.assertTrue((Path(tmp) / filename).is_file())

    def test_invalid_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_sensitivity_campaign(Path(tmp), base_steps=0)
            with self.assertRaises(ValueError):
                run_sensitivity_campaign(Path(tmp), dt_factors=(1.0, 1.0))
            with self.assertRaises(ValueError):
                run_sensitivity_campaign(Path(tmp), speeds_m_s=(0.0,))
            with self.assertRaises(ValueError):
                run_sensitivity_campaign(Path(tmp), widths_m=(0.0,))


if __name__ == "__main__":
    unittest.main()
