from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.plot_dem3d_benchmark_paper import build_paper_figures


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class DEM3DPaperFiguresTests(unittest.TestCase):
    def test_generates_curves_from_existing_csv_without_fabricating_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / "results-dem3d-ucs" / "ucs_history.csv", [
                {"axial_engineering_strain": 0.0, "engineering_stress_Pa": 0.0},
                {"axial_engineering_strain": 0.1, "engineering_stress_Pa": 100.0},
            ])
            write_csv(root / "results-dem3d-three-point-bending" / "three_point_bending_history.csv", [
                {"midspan_deflection_m": 0.0, "support_reaction_z_N": 0.0},
                {"midspan_deflection_m": 0.01, "support_reaction_z_N": -0.02},
            ])
            write_csv(root / "results-dem3d-moving-wedge-bow" / "moving_wedge_bow_history.csv", [
                {"time_s": 0.0, "bow_tip_x_m": 0.0, "ice_resistance_N": 0.0},
                {"time_s": 0.1, "bow_tip_x_m": 0.01, "ice_resistance_N": 0.02},
            ])
            write_csv(root / "results-dem3d-bending-resolution" / "bending_resolution_summary.csv", [
                {"resolution_factor": 1, "peak_support_reaction_abs_N": 0.01,
                 "peak_midspan_deflection_abs_m": 0.002, "broken_bonds": 2},
                {"resolution_factor": 2, "peak_support_reaction_abs_N": 0.012,
                 "peak_midspan_deflection_abs_m": 0.0018, "broken_bonds": 3},
            ])
            write_csv(root / "results-dem3d-ucs-timestep" / "ucs_timestep_convergence_summary.csv", [
                {"dt_factor": 1.0, "peak_stress_Pa": 100.0},
                {"dt_factor": 0.5, "peak_stress_Pa": 102.0},
            ])
            write_csv(root / "results-dem3d-mt-uikku-formulas" / "mt_uikku_formula_baselines_cases.csv", [
                {"test_id": "case-a", "experimental_reference_kN": 10.0,
                 "lindqvist_prediction_kN": 9.0, "riska_prediction_kN": 11.0,
                 "jeong_prediction_kN": 10.5, "keinonen_prediction_kN": 8.5},
                {"test_id": "case-b", "experimental_reference_kN": 20.0,
                 "lindqvist_prediction_kN": 19.0, "riska_prediction_kN": 21.0,
                 "jeong_prediction_kN": 20.5, "keinonen_prediction_kN": 18.5},
            ])

            out = root / "figures"
            report = build_paper_figures(root, out)
            expected = {
                "ucs_stress_strain.png",
                "three_point_bending_load_deflection.png",
                "moving_wedge_resistance_time.png",
                "moving_wedge_resistance_vs_travel.png",
                "bending_resolution_sensitivity.png",
                "ucs_timestep_sensitivity.png",
                "formula_reference_comparison.png",
            }
            self.assertEqual(report["generated_count"], len(expected))
            for filename in expected:
                self.assertTrue((out / filename).is_file(), filename)
            self.assertTrue((out / "paper_figure_manifest.json").is_file())
            self.assertTrue((out / "paper_figures_index.md").is_file())

    def test_missing_input_is_reported_as_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_paper_figures(root, root / "figures")
            self.assertEqual(report["generated_count"], 0)
            self.assertGreater(report["skipped_count"], 0)
            manifest = (root / "figures" / "paper_figure_manifest.json").read_text(encoding="utf-8")
            self.assertIn('"SKIPPED"', manifest)


if __name__ == "__main__":
    unittest.main()
