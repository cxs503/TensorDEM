"""Tests for the DEM3D engineering numerical screening report."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.report_dem3d_engineering_screen import build_acceptance_report


class DEM3DEngineeringScreenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.timestep = self.root / "timestep.json"
        self.resolution = self.root / "resolution.json"
        self.cross = self.root / "cross.json"
        self._write_reports()

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_reports(self, force_delta=0.02, force_nrmse=0.03, residual=0.001,
                       final_time=1.0):
        self._write(self.timestep, {
            "target_final_time_s": 1.0,
            "levels": [
                {"final_time_s": final_time, "peak_force_relative_delta_vs_finest": force_delta,
                 "max_abs_energy_balance_residual_J": residual, "final_mechanical_energy_J": 1.0},
                {"final_time_s": 1.0, "peak_force_relative_delta_vs_finest": 0.0,
                 "max_abs_energy_balance_residual_J": residual, "final_mechanical_energy_J": 1.0},
            ],
        })
        self._write(self.resolution, {
            "target_final_time_s": 1.0,
            "levels": [
                {"final_time_s": 1.0, "peak_force_relative_delta_vs_finest": force_delta,
                 "max_abs_energy_balance_residual_J": residual, "final_mechanical_energy_J": 1.0},
                {"final_time_s": 1.0, "peak_force_relative_delta_vs_finest": 0.0,
                 "max_abs_energy_balance_residual_J": residual, "final_mechanical_energy_J": 1.0},
            ],
        })
        self._write(self.cross, {
            "levels": [
                {"force_normalized_rmse_vs_finest": force_nrmse},
                {"force_normalized_rmse_vs_finest": 0.0},
            ],
        })

    def _build(self, **kwargs):
        return build_acceptance_report(self.timestep, self.resolution, self.cross, **kwargs)

    def test_passes_when_all_configured_gates_are_met(self):
        report = self._build()
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["fail_count"], 0)
        self.assertEqual(report["warn_count"], 0)
        self.assertEqual(report["protocol"], "tensordem-dem3d-engineering-screen-v1")

    def test_sensitivity_exceedance_warns_but_does_not_claim_failure(self):
        self._write_reports(force_delta=0.20, force_nrmse=0.2)
        report = self._build()
        self.assertEqual(report["verdict"], "WARN")
        self.assertGreater(report["warn_count"], 0)

    def test_energy_and_final_time_gate_failures_are_blocking(self):
        self._write_reports(residual=0.2, final_time=1.2)
        report = self._build()
        self.assertEqual(report["verdict"], "FAIL")
        self.assertGreaterEqual(report["fail_count"], 2)

    def test_missing_or_invalid_numeric_fields_are_rejected(self):
        self._write_reports()
        payload = json.loads(self.cross.read_text())
        payload["levels"][0]["force_normalized_rmse_vs_finest"] = float("nan")
        self._write(self.cross, payload)
        with self.assertRaises(ValueError):
            self._build()

    def test_negative_threshold_rejected(self):
        with self.assertRaises(ValueError):
            self._build(max_energy_residual_ratio=-1.0)


if __name__ == "__main__":
    unittest.main()
