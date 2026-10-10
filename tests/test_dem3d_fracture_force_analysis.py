"""End-to-end tests for DEM3D fracture/force post-processing."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_dem3d_fracture import analyze_fracture_force


class DEM3DFractureForceAnalysisTests(unittest.TestCase):
    def _write_csv(self, path, fields, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _inputs(self, root):
        history = root / "history_3d.csv"
        events = root / "fracture_events_3d.csv"
        self._write_csv(history, ["step", "time", "reaction_z", "broken_bonds"], [
            {"step": 0, "time": 0.0, "reaction_z": 0.0, "broken_bonds": 0},
            {"step": 10, "time": 1.0, "reaction_z": 2.0, "broken_bonds": 1},
            {"step": 20, "time": 2.0, "reaction_z": 4.0, "broken_bonds": 3},
            {"step": 30, "time": 3.0, "reaction_z": 0.0, "broken_bonds": 3},
        ])
        self._write_csv(events, [
            "bond_id", "failure_time_s", "failure_mode_label"
        ], [
            {"bond_id": 3, "failure_time_s": 0.5, "failure_mode_label": "tensile"},
            {"bond_id": 5, "failure_time_s": 1.5, "failure_mode_label": "shear"},
            {"bond_id": 8, "failure_time_s": 1.8, "failure_mode_label": "mixed"},
        ])
        return history, events

    def test_interval_reports_match_damage_and_event_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history, events = self._inputs(root)
            output = root / "analysis"
            summary = analyze_fracture_force(history, events, output)
            self.assertEqual(summary["total_broken_bonds"], 3)
            self.assertEqual(summary["interval_count"], 3)
            self.assertTrue(summary["interval_damage_matches_event_timestamps"])
            self.assertAlmostEqual(summary["total_signed_reaction_impulse_Ns"], 4.0)
            self.assertAlmostEqual(summary["total_absolute_reaction_impulse_Ns"], 4.0)
            self.assertIsNotNone(summary["pearson_fracture_rate_vs_abs_reaction"])

            with (output / "fracture_force_intervals_3d.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                intervals = list(csv.DictReader(stream))
            self.assertEqual([int(row["new_fractures"]) for row in intervals], [1, 2, 0])
            self.assertEqual([row["event_bond_ids"] for row in intervals], ["3", "5;8", ""])
            self.assertAlmostEqual(float(intervals[1]["fracture_rate_per_s"]), 2.0)
            saved = json.loads((output / "fracture_force_summary_3d.json").read_text())
            self.assertEqual(saved["protocol"], "tensordem-dem3d-fracture-force-analysis-v1")

    def test_rejects_event_count_or_timing_inconsistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history, events = self._inputs(root)
            # Three rows still exist, but one event is moved outside the sampled window.
            with events.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["failure_time_s"] = 9.0
            self._write_csv(events, list(rows[0]), rows)
            with self.assertRaisesRegex(ValueError, "outside the sampled history window"):
                analyze_fracture_force(history, events, root / "bad-output")

    def test_rejects_decreasing_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history, events = self._inputs(root)
            self._write_csv(history, ["step", "time", "reaction_z", "broken_bonds"], [
                {"step": 0, "time": 0.0, "reaction_z": 0.0, "broken_bonds": 0},
                {"step": 10, "time": 1.0, "reaction_z": 2.0, "broken_bonds": 2},
                {"step": 20, "time": 2.0, "reaction_z": 4.0, "broken_bonds": 1},
            ])
            with self.assertRaisesRegex(ValueError, "must never decrease"):
                analyze_fracture_force(history, events, root / "bad-output")


if __name__ == "__main__":
    unittest.main()
