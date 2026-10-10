from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dem3d_bending_resolution import run_resolution_campaign


class DEM3DBendingResolutionTests(unittest.TestCase):
    def test_campaign_is_finite_repeatable_and_exports_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = run_resolution_campaign(root / "a", base_steps=6, factors=(1, 2))
            b = run_resolution_campaign(root / "b", base_steps=6, factors=(1, 2))
            self.assertEqual(a["verdict"], "PASS", a)
            self.assertEqual(
                [x["signature_sha256"] for x in a["cases"]],
                [x["signature_sha256"] for x in b["cases"]],
            )
            self.assertEqual([x["particle_count"] for x in a["cases"]], [81, 425])
            self.assertTrue((root / "a" / "bending_resolution_summary.csv").is_file())
            self.assertTrue((root / "a" / "bending_resolution_history.csv").is_file())
            saved = json.loads((root / "a" / "bending_resolution_report.json").read_text())
            self.assertEqual(saved["protocol"], "tensordem-dem3d-bending-resolution-sensitivity-v1")
            self.assertEqual(len(saved["cases"]), 2)

    def test_invalid_campaign_parameters_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_resolution_campaign(Path(tmp) / "bad", base_steps=0)
            with self.assertRaises(ValueError):
                run_resolution_campaign(Path(tmp) / "bad2", factors=(1, 1))
            with self.assertRaises(ValueError):
                run_resolution_campaign(Path(tmp) / "bad3", factors=(1, 0))


if __name__ == "__main__":
    unittest.main()
