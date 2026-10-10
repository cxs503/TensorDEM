from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dem3d_bending_load_convergence import run_bending_load_convergence


class DEM3DBendingLoadConvergenceTests(unittest.TestCase):
    def test_refinement_campaign_emits_ordered_levels_and_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = run_bending_load_convergence(
                root / "campaign", base_steps=8, levels=2, peak_load_N=0.01
            )
            self.assertEqual(report["verdict"], "PASS", report)
            self.assertEqual(report["step_counts"], [8, 16])
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual(len(report["comparisons"]), 1)
            comparison = report["comparisons"][0]
            self.assertGreaterEqual(comparison["support_reaction_relative_change"], 0.0)
            self.assertGreaterEqual(comparison["midspan_deflection_relative_change"], 0.0)
            self.assertTrue((root / "campaign" / "bending_load_convergence.csv").is_file())
            saved = json.loads(
                (root / "campaign" / "bending_load_convergence.json").read_text()
            )
            self.assertEqual(saved["protocol"], "tensordem-dem3d-bending-load-sensitivity-v1")
            self.assertIn("effective loading rate", saved["interpretation"])
            self.assertGreater(report["runs"][1]["ramp_duration_s"], report["runs"][0]["ramp_duration_s"])

    def test_invalid_campaign_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kwargs in (
                {"base_steps": 1},
                {"levels": 1},
                {"levels": 7},
                {"peak_load_N": 0.0},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    run_bending_load_convergence(root / "bad", **kwargs)


if __name__ == "__main__":
    unittest.main()
