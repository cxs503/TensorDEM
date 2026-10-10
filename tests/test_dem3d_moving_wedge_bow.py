from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_wedge import MovingWedgeBowIceDEM3D
from scripts.benchmark_dem3d_moving_wedge_bow import run_campaign


class MovingWedgeBowTests(unittest.TestCase):
    def _model(self):
        return MovingWedgeBowIceDEM3D(
            DEM3DConfig(
                nx=3, ny=2, nz=3, radius=0.02, tool_speed=0.1,
                bond_stiffness=1000.0, contact_stiffness=3000.0,
                breaking_strain=0.5, shear_breaking_strain=0.5,
                drag=0.1, contact_damping=0.5,
                fix_x_edges=False, fix_bottom=False,
            ),
            bow_speed=0.1, wedge_angle_deg=45.0,
        )

    def test_wedge_force_reaction_is_finite_and_advances(self):
        sim = self._model()
        x0 = sim.bow_tip_x
        sim.step()
        d = sim.diagnostics()
        self.assertGreater(sim.bow_tip_x, x0)
        self.assertTrue(all(__import__("math").isfinite(float(d[k])) for k in (
            "bow_reaction_x_N", "bow_reaction_z_N", "bow_contact_energy_J",
            "kinetic_energy_J", "bond_elastic_energy_J",
        )))
        self.assertGreaterEqual(d["bow_contact_energy_J"], 0.0)

    def test_invalid_wedge_parameters_are_rejected(self):
        cfg = DEM3DConfig(nx=2, ny=2, nz=2)
        with self.assertRaises(ValueError):
            MovingWedgeBowIceDEM3D(cfg, bow_speed=0.0)
        with self.assertRaises(ValueError):
            MovingWedgeBowIceDEM3D(cfg, wedge_angle_deg=90.0)
        with self.assertRaises(ValueError):
            MovingWedgeBowIceDEM3D(cfg, initial_gap=-1.0)

    def test_campaign_persists_repeatable_histories(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_campaign(Path(tmp), steps=60, bow_speed=0.1, wedge_angle_deg=45.0)
            self.assertEqual(report["verdict"], "PASS", report)
            self.assertTrue(all(report["checks"].values()))
            self.assertGreater(report["metrics"]["bow_travel_m"], 0.0)
            self.assertTrue((Path(tmp) / "moving_wedge_bow_report.json").is_file())
            self.assertTrue((Path(tmp) / "moving_wedge_bow_history.csv").is_file())
            self.assertTrue((Path(tmp) / "moving_wedge_bow_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
