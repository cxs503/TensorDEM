from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dem3d_ucs import _run_once
from scripts.benchmark_dem3d_ucs_sensitivity import _base_config, _steps_for_strain
from scripts.calibrate_dem3d_ucs import load_experimental_curve, run_calibration_campaign


class DEM3DUCSCalibrationTests(unittest.TestCase):
    def _write_curve(self, path: Path, points):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["axial_strain", "compressive_stress_Pa"])
            writer.writeheader()
            writer.writerows(
                {"axial_strain": strain, "compressive_stress_Pa": stress}
                for strain, stress in points
            )

    def test_calibration_grid_is_ranked_repeatable_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _base_config()
            target = 0.001
            steps = _steps_for_strain(cfg, target)
            rows, _, _ = _run_once({"id": "test_curve", "config": cfg}, steps)
            sample_strains = (0.0, target / 3.0, 2.0 * target / 3.0, target)
            sample_points = []
            for strain in sample_strains:
                nearest = min(rows, key=lambda row: abs(float(row["axial_engineering_strain"]) - strain))
                sample_points.append((strain, float(nearest["engineering_stress_Pa"])))
            # Keep a positive curve even for a very short target strain.
            sample_points = [(s, max(p, 1e-12)) for s, p in sample_points]
            experimental = root / "measured.csv"
            self._write_curve(experimental, sample_points)

            output = root / "calibration"
            report = run_calibration_campaign(
                experimental, output,
                bond_stiffness_scales=(0.8, 1.0),
                breaking_strains=(0.012, 0.015),
                verify_repeat=True,
            )
            self.assertEqual(report["verdict"], "PASS", report["checks"])
            self.assertEqual(report["candidate_count"], 4)
            self.assertEqual([x["rank"] for x in report["candidates"]], [1, 2, 3, 4])
            self.assertTrue(all(report["checks"].values()))
            self.assertTrue(all(x["repeatable"] for x in report["candidates"]))
            for name in (
                "ucs_calibration_report.json",
                "ucs_calibration_summary.csv",
                "ucs_calibration_histories.csv",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_experimental_curve_schema_and_order_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            self._write_curve(path, [(0.0, 0.0), (0.01, 10.0), (0.01, 12.0)])
            with self.assertRaises(ValueError):
                load_experimental_curve(path)
            self._write_curve(path, [(0.0, 0.0), (0.01, 10.0)])
            with self.assertRaises(ValueError):
                load_experimental_curve(path)

    def test_invalid_parameter_grids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curve.csv"
            self._write_curve(path, [(0.0, 0.0), (0.001, 5.0), (0.002, 4.0)])
            with self.assertRaises(ValueError):
                run_calibration_campaign(path, Path(tmp) / "out", bond_stiffness_scales=(1.0, 1.0))
            with self.assertRaises(ValueError):
                run_calibration_campaign(path, Path(tmp) / "out", breaking_strains=(0.01, 0.01))


if __name__ == "__main__":
    unittest.main()
