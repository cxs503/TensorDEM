"""Tests for DEM3D large-scale particle benchmark (10k/50k/100k scaling)."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results-dem3d-scale" / "scaling_report.json"


class DEM3DScaleBenchmarkTests(unittest.TestCase):
    """Acceptance tests for the DEM3D scale benchmark."""

    @classmethod
    def setUpClass(cls):
        if not REPORT.is_file():
            raise FileNotFoundError(f"scaling report not found: {REPORT}")
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_report_exists_and_valid(self):
        """Scaling report must exist and contain required keys."""
        self.assertIn("cases", self.report)
        self.assertIn("acceptance_criteria", self.report)
        self.assertIn("all_criteria_met", self.report)
        self.assertEqual(len(self.report["cases"]), 3)

    def test_wallclock_100k_ratio_vs_ideal(self):
        """wallclock_100k_ratio_vs_ideal must be < 2.0."""
        criteria = self.report["acceptance_criteria"]["wallclock_100k_ratio_vs_ideal"]
        self.assertEqual(criteria["status"], "PASS")
        self.assertIsNotNone(criteria["value"])
        self.assertLess(criteria["value"], 2.0)

    def test_energy_drift_per_step(self):
        """energy_drift_per_step must be < 1e-06."""
        criteria = self.report["acceptance_criteria"]["energy_drift_per_step"]
        self.assertEqual(criteria["status"], "PASS")
        self.assertIsNotNone(criteria["value"])
        self.assertLess(criteria["value"], 1e-06)

    def test_reproducible_hash(self):
        """reproducible_hash must be True."""
        criteria = self.report["acceptance_criteria"]["reproducible_hash"]
        self.assertEqual(criteria["status"], "PASS")
        self.assertTrue(criteria["value"])

    def test_all_criteria_met(self):
        """All acceptance criteria must be met."""
        self.assertTrue(self.report["all_criteria_met"])

    def test_case_particle_counts(self):
        """Each case must have approximately the target particle count."""
        expected = {
            "scale_10k": (10000, 0.15),   # ±15%
            "scale_50k": (50000, 0.15),
            "scale_100k": (100000, 0.15),
        }
        for case in self.report["cases"]:
            target, tol = expected[case["case_id"]]
            self.assertAlmostEqual(case["actual_n"], target, delta=target * tol)

    def test_scaling_near_linear(self):
        """Per-step time should scale approximately linearly with particle count."""
        cases = {c["case_id"]: c for c in self.report["cases"]}
        t10k = cases["scale_10k"]["wallclock_per_step_ms"]
        t50k = cases["scale_50k"]["wallclock_per_step_ms"]
        t100k = cases["scale_100k"]["wallclock_per_step_ms"]
        n10k = cases["scale_10k"]["actual_n"]
        n50k = cases["scale_50k"]["actual_n"]
        n100k = cases["scale_100k"]["actual_n"]

        # 50k should be roughly 5x 10k
        ratio_50k = (t50k / t10k) / (n50k / n10k)
        self.assertLess(ratio_50k, 2.0, "50k scaling deviates too much from linear")

        # 100k ratio already checked in acceptance criteria
        ratio_100k = self.report["wallclock_100k_ratio_vs_ideal"]
        self.assertLess(ratio_100k, 2.0)


if __name__ == "__main__":
    unittest.main()
