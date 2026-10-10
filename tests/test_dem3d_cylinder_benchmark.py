"""End-to-end regression tests for the cylinder-compression benchmark driver."""
import csv
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts.run_dem3d_cylinder_benchmark import main as cylinder_benchmark_main


class CylinderBenchmarkIntegrationTests(unittest.TestCase):
    def test_smoke_run_writes_consistent_raw_and_derived_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cylinder"
            argv = [
                "run_dem3d_cylinder_benchmark.py",
                "--output", str(output),
                "--steps", "4",
                "--save-every", "2",
                "--nx", "5", "--ny", "2", "--nz", "5",
                "--platen-speed", "0.002",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                cylinder_benchmark_main()

            expected = (
                "history.csv", "load_displacement.csv", "checkpoint_3d.pt",
                "trajectory_3d.pt", "benchmark_summary.json",
            )
            for name in expected:
                self.assertTrue((output / name).is_file(), name)

            with (output / "history.csv").open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            with (output / "load_displacement.csv").open(newline="", encoding="utf-8") as stream:
                curve = list(csv.DictReader(stream))
            summary = json.loads((output / "benchmark_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(len(history), 3)  # step 0, 2, and 4
            self.assertEqual(len(curve), len(history))
            self.assertEqual(summary["samples"], len(history))
            self.assertAlmostEqual(float(history[-1]["time_s"]), summary["final_time_s"])
            self.assertAlmostEqual(
                float(curve[-1]["compression_displacement_m"]),
                2 * summary["configuration"]["platen_speed_m_s_each"] * summary["final_time_s"],
            )
            for row in history:
                for key in ("time_s", "top_platen_reaction_z_N",
                            "bottom_platen_reaction_z_N", "mechanical_energy_J"):
                    self.assertTrue(float(row[key]) == float(row[key]), f"non-finite {key}")
            self.assertIn("not experimental validation", summary["validation_note"])


if __name__ == "__main__":
    unittest.main()
