"""Tests for the named DEM3D benchmark case suite."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_dem3d_benchmark_suite import load_case_suite, run_benchmark_suite


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks" / "dem3d" / "cases.json"


class DEM3DBenchmarkCaseSuiteTests(unittest.TestCase):
    def test_suite_defines_unique_named_cases(self):
        suite = load_case_suite(SUITE)
        ids = [case["id"] for case in suite["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("indenter_smoke", ids)
        self.assertIn("indenter_reference", ids)
        self.assertIn("fracture_challenge", ids)

    def test_smoke_case_writes_summary_and_auditable_event_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            report = run_benchmark_suite(SUITE, output, ["indenter_smoke"])
            self.assertEqual(report["verdict"], "PASS")
            self.assertEqual(report["case_count"], 1)
            self.assertEqual(report["pass_case_count"], 1)
            self.assertTrue((output / "indenter_smoke" / "summary_3d.json").is_file())
            self.assertTrue((output / "indenter_smoke" / "history_3d.csv").is_file())
            self.assertTrue((output / "indenter_smoke" / "fracture_events_3d.csv").is_file())
            self.assertTrue((output / "benchmark_suite_summary.json").is_file())
            self.assertTrue((output / "benchmark_suite_summary.csv").is_file())
            saved = json.loads((output / "benchmark_suite_summary.json").read_text())
            self.assertEqual(saved["protocol"], "tensordem-dem3d-benchmark-suite-v1")
            self.assertEqual(saved["cases"][0]["case_id"], "indenter_smoke")
            self.assertTrue(all(check["status"] == "PASS" for check in saved["checks"]))

    def test_unknown_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unknown benchmark case"):
                run_benchmark_suite(SUITE, Path(tmp) / "suite", ["not-a-case"])

    def test_duplicate_case_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps({
                "cases": [
                    {"id": "same", "config": {}, "steps": 1, "sample_every": 1},
                    {"id": "same", "config": {}, "steps": 1, "sample_every": 1},
                ]
            }))
            with self.assertRaisesRegex(ValueError, "duplicate benchmark case"):
                load_case_suite(path)


if __name__ == "__main__":
    unittest.main()
