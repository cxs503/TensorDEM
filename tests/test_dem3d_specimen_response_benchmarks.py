from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_dem3d_specimen_response_benchmarks import run_specimen_response_benchmarks


class DEM3DSpecimenResponseBenchmarkTests(unittest.TestCase):
    def test_affine_compression_patch_is_repeatable_and_writes_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "specimen"
            report = run_specimen_response_benchmarks(output)
            self.assertEqual(report["verdict"], "PASS", report)
            self.assertEqual(report["case_id"], "affine_compression_patch")
            self.assertTrue(all(report["checks"].values()))
            self.assertGreater(report["mean_secant_modulus_Pa"], 0.0)
            self.assertTrue((output / "specimen_response_report.json").is_file())
            self.assertTrue((output / "specimen_stress_strain.csv").is_file())
            saved = json.loads((output / "specimen_response_report.json").read_text())
            self.assertEqual(len(saved["rows"]), 3)
            self.assertEqual(saved["signature_sha256"], saved["repeat_signature_sha256"])
            self.assertEqual(max(row["broken_bonds"] for row in saved["rows"]), 0)

    def test_manifest_explicitly_marks_standard_material_fixtures_as_future_work(self):
        path = Path(__file__).resolve().parents[1] / "benchmarks" / "dem3d" / "specimen_response_cases.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        future = {item["id"] for item in manifest["not_yet_supported"]}
        self.assertTrue({"standard_uniaxial_compression", "brazilian_splitting", "three_point_bending"} <= future)


if __name__ == "__main__":
    unittest.main()
