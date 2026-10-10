"""Regression tests for physical-domain-preserving DEM3D resolution studies."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from tensordem.dem3d import DEM3DConfig
from scripts.validate_dem3d_resolution import (
    refined_config,
    run_resolution_study,
    write_report,
)


class DEM3DResolutionSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = DEM3DConfig(
            nx=3, ny=3, nz=2, drag=0.0, contact_damping=0.0,
            fix_x_edges=False, tool_radius=0.03, tool_gap=0.0,
        )

    def test_refinement_preserves_physical_extent_and_scales_parameters(self):
        base_dt = self.cfg.recommended_dt
        fine = refined_config(self.cfg, 2, base_dt)
        self.assertEqual((fine.nx, fine.ny, fine.nz), (5, 5, 3))
        self.assertAlmostEqual(
            (self.cfg.nx - 1) * 2 * self.cfg.radius,
            (fine.nx - 1) * 2 * fine.radius,
        )
        self.assertAlmostEqual(
            (self.cfg.ny - 1) * 2 * self.cfg.radius,
            (fine.ny - 1) * 2 * fine.radius,
        )
        self.assertAlmostEqual(fine.bond_stiffness, self.cfg.bond_stiffness / 2)
        self.assertAlmostEqual(fine.contact_stiffness, self.cfg.contact_stiffness / 2)
        self.assertAlmostEqual(fine.dt, base_dt / 2)

    def test_study_uses_same_physical_time_and_writes_auditable_outputs(self):
        report = run_resolution_study(
            self.cfg, base_steps=2, resolution_factors=(1, 2), sample_every=1
        )
        levels = report["levels"]
        self.assertEqual([level["resolution_factor"] for level in levels], [1, 2])
        self.assertTrue(report["same_final_time_within_tolerance"])
        for level in levels:
            self.assertTrue(math.isclose(
                level["final_time_s"], report["target_final_time_s"],
                rel_tol=1e-10, abs_tol=1e-14,
            ))
            self.assertGreater(level["particle_count"], 0)
            self.assertGreater(level["bond_count"], 0)
            for key in (
                "peak_abs_reaction_z_N", "final_mechanical_energy_J",
                "final_energy_balance_residual_J", "max_abs_energy_balance_residual_J",
                "peak_force_relative_delta_vs_finest",
            ):
                self.assertTrue(math.isfinite(float(level[key])), key)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_report(report, out)
            summary = json.loads((out / "resolution_sensitivity_3d.json").read_text())
            self.assertEqual(summary["protocol"], "tensordem-dem3d-resolution-sensitivity-v1")
            with (out / "resolution_levels_3d.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            with (out / "resolution_history_3d.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertGreater(len(list(csv.DictReader(stream))), 0)

    def test_invalid_resolution_schedule_rejected(self):
        with self.assertRaises(ValueError):
            run_resolution_study(self.cfg, base_steps=0)
        with self.assertRaises(ValueError):
            run_resolution_study(self.cfg, base_steps=1, resolution_factors=(1, 1))
        with self.assertRaises(ValueError):
            run_resolution_study(self.cfg, base_steps=1, resolution_factors=(0, 2))


if __name__ == "__main__":
    unittest.main()
