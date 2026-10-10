"""Regression tests for DEM3D cross-resolution time-history comparisons."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.compare_dem3d_resolutions import (
    compare_resolution_histories,
    read_resolution_history,
    write_report,
)


def row(factor, step, time_s, force, broken, energy=1.0, residual=0.01):
    return {
        "resolution_factor": factor,
        "step": step,
        "time_s": time_s,
        "reaction_z_N": force,
        "broken_bonds": broken,
        "mechanical_energy_J": energy,
        "energy_balance_residual_J": residual,
    }


class DEM3DCrossResolutionTests(unittest.TestCase):
    def setUp(self):
        self.histories = {
            1: [row(1, 0, 0.0, 0.0, 0), row(1, 1, 1.0, 2.0, 1),
                row(1, 2, 2.0, 0.0, 2)],
            2: [row(2, 0, 0.0, 0.0, 0), row(2, 2, 0.5, 1.5, 1),
                row(2, 4, 1.0, 2.0, 2), row(2, 6, 1.5, 1.0, 2),
                row(2, 8, 2.0, 0.0, 3)],
        }

    def test_common_time_grid_and_reference_metrics(self):
        report = compare_resolution_histories(self.histories)
        self.assertEqual(report["reference_resolution_factor"], 2)
        self.assertEqual(report["comparison_time_grid_points"], 5)
        levels = {r["resolution_factor"]: r for r in report["levels"]}
        self.assertEqual(levels[2]["force_rmse_vs_finest_N"], 0.0)
        self.assertEqual(levels[2]["final_broken_bonds_vs_finest_delta"], 0)
        self.assertTrue(math.isfinite(levels[1]["force_normalized_rmse_vs_finest"]))
        self.assertAlmostEqual(report["common_window_end_s"], 2.0)
        self.assertEqual(len(report["aligned_history"]), 10)

    def test_csv_reader_validates_time_and_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            fields = list(self.histories[1][0])
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.histories[1])
            parsed = read_resolution_history(path)
            self.assertEqual(len(parsed[1]), 3)
            bad = [dict(r) for r in self.histories[1]]
            bad[2]["broken_bonds"] = 0
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(bad)
            with self.assertRaises(ValueError):
                read_resolution_history(path)

    def test_disjoint_time_windows_rejected(self):
        disjoint = {
            1: [row(1, 0, 0.0, 0, 0), row(1, 1, 1.0, 1, 0)],
            2: [row(2, 0, 2.0, 0, 0), row(2, 1, 3.0, 1, 0)],
        }
        with self.assertRaises(ValueError):
            compare_resolution_histories(disjoint)

    def test_writes_json_and_csv_reports(self):
        report = compare_resolution_histories(self.histories)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_report(report, out)
            summary = json.loads((out / "cross_resolution_summary_3d.json").read_text())
            self.assertEqual(summary["protocol"], "tensordem-dem3d-cross-resolution-validation-v1")
            with (out / "cross_resolution_levels_3d.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            with (out / "cross_resolution_history_3d.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 10)


if __name__ == "__main__":
    unittest.main()
