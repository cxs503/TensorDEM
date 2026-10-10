from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dem3d_ucs import run_ucs_benchmark


class DEM3DUCSBenchmarkTests(unittest.TestCase):
    def test_dynamic_ucs_history_is_finite_and_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "ucs"
            report = run_ucs_benchmark(output, steps=45)
            self.assertEqual(report["verdict"], "PASS", report)
            self.assertTrue(all(report["checks"].values()))
            self.assertGreater(report["peak_load_N"], 0.0)
            self.assertGreater(report["peak_engineering_stress_Pa"], 0.0)
            self.assertEqual(report["steps"], 45)
            self.assertEqual(len(report["rows"]), 46)
            self.assertTrue((output / "ucs_report.json").is_file())
            self.assertTrue((output / "ucs_history.csv").is_file())
            saved = json.loads((output / "ucs_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["signature_sha256"], report["signature_sha256"])
            self.assertEqual(saved["repeat_signature_sha256"], report["signature_sha256"])

    def test_nonpositive_step_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_ucs_benchmark(Path(tmp), steps=0)


if __name__ == "__main__":
    unittest.main()
