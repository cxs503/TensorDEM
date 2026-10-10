from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dem3d_ucs_sensitivity import run_sensitivity_campaign


class DEM3DUCSSensitivityTests(unittest.TestCase):
    def test_sensitivity_campaign_is_finite_repeatable_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            report = run_sensitivity_campaign(out, target_strain=0.002)
            self.assertEqual(report["verdict"], "PASS", report["checks"])
            self.assertEqual(report["case_count"], 6)
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual({x["family"] for x in report["cases"]}, {"loading_rate", "resolution"})
            for name in (
                "ucs_sensitivity_report.json",
                "ucs_sensitivity_summary.csv",
                "ucs_sensitivity_histories.csv",
            ):
                self.assertTrue((out / name).is_file(), name)
            saved = json.loads((out / "ucs_sensitivity_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["case_count"], 6)

    def test_invalid_target_strain_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_sensitivity_campaign(Path(tmp), target_strain=0.0)


if __name__ == "__main__":
    unittest.main()
