from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.visualize_dem3d_benchmarks import visualize_case


class DEM3DFractureDiagnosticEdgeCases(unittest.TestCase):
    def _write_trajectory(self, case: Path, alive_frames: list[list[bool]], bond_count: int = 3) -> None:
        initial = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
        pairs = torch.tensor([[0, 1]] * bond_count, dtype=torch.long)
        frames = [
            {
                "step": index,
                "time_s": index * 0.01,
                "positions": initial.clone(),
                "alive": torch.tensor(alive, dtype=torch.bool),
                "tool_position": torch.tensor([1e6, 1e6, 1e6]),
            }
            for index, alive in enumerate(alive_frames)
        ]
        torch.save({
            "format": "tensordem-dem3d-visualization-trajectory-v1",
            "initial_positions": initial,
            "pairs": pairs,
            "rest_lengths": torch.ones(bond_count),
            "frames": frames,
        }, case / "visualization_trajectory_3d.pt")

    def test_zero_breakage_is_reported_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "intact_case"
            case.mkdir()
            self._write_trajectory(case, [[True, True, True], [True, True, True]])
            report = visualize_case(case, root / "out", make_gif=False)
            diagnostics = report["fracture_diagnostics"]
            self.assertEqual(diagnostics["final_broken_bonds"], 0)
            self.assertEqual(diagnostics["newly_broken_bonds"], 0)
            self.assertEqual(diagnostics["frames_with_new_breakage"], 0)
            self.assertEqual(diagnostics["warning"], "NO_VISIBLE_FRACTURE: no bonds broke in saved trajectory")
            manifest = json.loads((root / "out" / "visualization_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["fracture_diagnostics"]["warning"], diagnostics["warning"])

    def test_low_activity_warning_uses_broken_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "low_activity_case"
            case.mkdir()
            # One new break out of 200 bonds is below the 1% threshold.
            self._write_trajectory(case, [[True] * 200, [False] + [True] * 199], bond_count=200)
            report = visualize_case(case, root / "out", make_gif=False)
            self.assertEqual(report["fracture_diagnostics"]["warning"],
                             "LOW_FRACTURE_ACTIVITY: fewer than 1% of bonds broke")
            self.assertAlmostEqual(report["fracture_diagnostics"]["newly_broken_fraction"], 0.005)


if __name__ == "__main__":
    unittest.main()
