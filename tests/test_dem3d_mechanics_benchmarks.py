"""Regression tests for the analytic DEM3D mechanics benchmark pack."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_dem3d_mechanics_benchmarks import run_mechanics_benchmarks


class DEM3DMechanicsBenchmarkTests(unittest.TestCase):
    def test_all_analytic_and_invariant_cases_pass_and_emit_audit_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mechanics"
            report = run_mechanics_benchmarks(output)
            self.assertEqual(report["protocol"], "tensordem-dem3d-mechanics-v1")
            self.assertEqual(report["verdict"], "PASS", report["cases"])
            self.assertEqual(report["case_count"], 6)
            self.assertEqual(report["pass_count"], 6)
            self.assertEqual(report["fail_count"], 0)
            self.assertTrue((output / "mechanics_benchmark_report.json").is_file())
            self.assertTrue((output / "mechanics_benchmark_report.csv").is_file())
            saved = json.loads(
                (output / "mechanics_benchmark_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["verdict"], "PASS")
            self.assertEqual(
                {row["case_id"] for row in saved["cases"]},
                {
                    "bond_axial_spring",
                    "linear_contact_penalty",
                    "force_balance",
                    "irreversible_tensile_failure",
                    "rigid_rotation_objectivity",
                    "timestep_guard",
                },
            )

    def test_report_distinguishes_supported_and_not_yet_supported_laws(self):
        manifest = json.loads(
            (Path(__file__).resolve().parents[1]
             / "benchmarks" / "dem3d" / "mechanics_cases.json").read_text(encoding="utf-8")
        )
        self.assertIn("linear central-force bonds", manifest["model_scope"])
        unsupported = {case["id"] for case in manifest["not_yet_supported"]}
        self.assertIn("hertz_mindlin_contact", unsupported)
        self.assertIn("macroscopic_ucs_brazilian_three_point_bending", unsupported)


if __name__ == "__main__":
    unittest.main()
