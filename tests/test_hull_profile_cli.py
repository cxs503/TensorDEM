import csv
import tempfile
import unittest
from pathlib import Path

import torch

from tensordem.cli import main, read_hull_profile


class HullProfileCliTests(unittest.TestCase):
    def test_read_hull_profile_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hull.csv"
            path.write_text("x,y\n-0.1,0.0\n0.0,0.1\n", encoding="utf-8")
            profile = read_hull_profile(path)
            self.assertEqual(tuple(profile.shape), (2, 2))
            self.assertEqual(profile.dtype, torch.float64)

    def test_reject_missing_columns_and_duplicate_vertices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hull.csv"
            path.write_text("x,z\n0,0\n1,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "x,y column"):
                read_hull_profile(path)
            path.write_text("x,y\n0,0\n0,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "zero-length"):
                read_hull_profile(path)

    def test_hull_cli_exports_position_and_reaction_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "hull.csv"
            profile.write_text("x,y\n0,-0.1\n0,0.2\n", encoding="utf-8")
            output = root / "out"
            main([
                "--nx", "5", "--ny", "3", "--steps", "2", "--save-every", "1",
                "--hull-profile", str(profile), "--hull-start-x", "0.01",
                "--hull-velocity-x", "0.2", "--speed", "0.2", "--output", str(output),
            ])
            with (output / "history.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertTrue({"tool_x", "tool_y", "reaction_x", "reaction_y"}.issubset(rows[0]))
            self.assertAlmostEqual(float(rows[-1]["tool_x"]), 0.01 + 0.2 * float(rows[-1]["time"]), places=10)
            self.assertTrue((output / "trajectory.pt").exists())


if __name__ == "__main__":
    unittest.main()
