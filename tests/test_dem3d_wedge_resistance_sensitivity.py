from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_repeat_run_uses_identical_geometry_for_every_width(self):
        calls = []

        def fake_run_once(steps, **kwargs):
            calls.append((steps, kwargs.copy()))
            speed = kwargs["bow_speed"]
            rows = [
                {"step": 0, "time_s": 0.0, "bow_tip_x_m": 0.0,
                 "ice_resistance_N": 0.0, "broken_bonds": 0},
                {"step": 1, "time_s": 0.001, "bow_tip_x_m": speed * 0.001,
                 "ice_resistance_N": 1.0, "broken_bonds": 0},
            ]
            metrics = {
                "particle_count": 1, "initial_bond_count": 1,
                "final_broken_bonds": 0, "broken_bond_fraction": 0.0,
                "peak_ice_resistance_N": 1.0, "mean_ice_resistance_N": 0.5,
                "integrated_resistance_work_J": speed * 0.001,
                "first_fracture_time_s": None, "bow_travel_m": speed * 0.001,
            }
            return rows, "stable-signature", metrics

        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.benchmark_dem3d_wedge_resistance_sensitivity._run_once",
            side_effect=fake_run_once,
        ):
            report = run_sensitivity_campaign(
                Path(tmp), base_steps=2,
                speeds_m_s=(0.1,), angles_deg=(45.0,),
                dt_factors=(1.0,), widths_m=(0.02, 0.04),
            )

        self.assertEqual(report["verdict"], "PASS", report)
        self.assertEqual(len(calls), 10)
        for first, second in zip(calls[::2], calls[1::2]):
            self.assertEqual(first, second)
        width_pairs = [
            (calls[i][0], calls[i][1]["bow_half_width_m"])
            for i in range(6, 10, 2)
        ]
        self.assertEqual(width_pairs, [(2, 0.02), (2, 0.04)])

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
