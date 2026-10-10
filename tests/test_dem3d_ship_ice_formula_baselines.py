from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dem3d_ship_ice_formula_baselines import (
    DEFAULT_REFERENCE,
    score_formula_baselines,
)


class MTUikkuFormulaBaselineTests(unittest.TestCase):
    def test_formula_scores_are_complete_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            report = score_formula_baselines(DEFAULT_REFERENCE, output)
            self.assertEqual(report["protocol"], "tensordem-mt-uikku-formula-baselines-v1")
            self.assertEqual(report["case_count"], 10)
            self.assertEqual(
                [row["formula"] for row in report["formula_summaries"]],
                ["Lindqvist", "Riska", "Jeong", "Keinonen"],
            )
            for row in report["formula_summaries"]:
                self.assertEqual(row["case_count"], 10)
                self.assertGreaterEqual(row["mean_absolute_percentage_error_pct"], 0.0)
                self.assertGreaterEqual(row["root_mean_square_error_kN"], 0.0)
                self.assertEqual(
                    row["overprediction_cases"] + row["underprediction_cases"] + row["exact_match_cases"],
                    10,
                )
            with (output / "mt_uikku_formula_baselines_cases.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                case_rows = list(csv.DictReader(stream))
            self.assertEqual(len(case_rows), 10)
            self.assertEqual(case_rows[0]["test_id"], "103")
            saved = json.loads((output / "mt_uikku_formula_baselines.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
            self.assertTrue((output / "mt_uikku_formula_baselines_summary.csv").is_file())

    def test_invalid_reference_values_fail_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_csv = root / "bad.csv"
            bad_csv.write_text(
                "test_id,experimental_reference_table8_kN,lindqvist_prediction_kN,"
                "riska_prediction_kN,jeong_prediction_kN,keinonen_prediction_kN\n"
                "bad,0,1,1,1,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "experimental reference"):
                score_formula_baselines(bad_csv, root / "out")


if __name__ == "__main__":
    unittest.main()
