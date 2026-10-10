from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.visualize_dem3d_benchmarks import visualize_case


class DEM3DVisualizationTests(unittest.TestCase):
    def test_clouds_animation_and_history_are_emitted_from_sampled_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "case_a"
            case.mkdir()
            initial = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
            pairs = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
            frames = [
                {"step": 0, "time_s": 0., "positions": initial.clone(),
                 "alive": torch.tensor([True, True]), "tool_position": torch.tensor([1e6, 1e6, 1e6])},
                {"step": 1, "time_s": 0.1, "positions": initial + torch.tensor([[0., 0., 0.], [0., 0., .1], [0., 0., .2]]),
                 "alive": torch.tensor([False, True]), "tool_position": torch.tensor([1e6, 1e6, 1e6])},
            ]
            torch.save({"format": "tensordem-dem3d-visualization-trajectory-v1",
                        "initial_positions": initial, "pairs": pairs,
                        "rest_lengths": torch.ones(2), "frames": frames}, case / "visualization_trajectory_3d.pt")
            with (case / "history_3d.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["step", "time", "reaction_z"])
                writer.writeheader()
                writer.writerows([{"step": 0, "time": 0, "reaction_z": 0},
                                  {"step": 1, "time": .1, "reaction_z": 1}])
            with (case / "fracture_events_3d.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "final_midpoint_x_m", "final_midpoint_y_m", "final_midpoint_z_m",
                    "failure_mode_label",
                ])
                writer.writeheader()
                writer.writerow({
                    "final_midpoint_x_m": 0.5, "final_midpoint_y_m": 0.0,
                    "final_midpoint_z_m": 0.05, "failure_mode_label": "tensile",
                })
            report = visualize_case(case, root / "out")
            self.assertTrue(report["trajectory_available"])
            self.assertEqual(report["frame_count"], 2)
            for filename in ("damage_initial.png", "damage_final.png",
                             "displacement_final.png", "damage_evolution.gif", "force_history.png",
                             "fracture_crack_map.png"):
                self.assertTrue((root / "out" / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
