"""Independent mechanics and time-step sensitivity regressions for DEM3D."""
import csv
import math
import tempfile
import unittest
from pathlib import Path

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from scripts.validate_dem3d_convergence import run_convergence_study, write_report


class DEM3DMechanicsValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pair_internal_forces_conserve_linear_momentum(self):
        sim = IceDEM3D(DEM3DConfig(
            nx=3, ny=3, nz=2, drag=0.0, contact_damping=0.0,
            fix_x_edges=False, tool_radius=0.001, tool_gap=0.0,
        ))
        # No fracture update: this test isolates force assembly from damage history.
        force, reaction = sim.forces(update_fracture=False)
        self.assertTrue(torch.isfinite(force).all())
        torch.testing.assert_close(force.sum(dim=0), -reaction, atol=1e-10, rtol=0)
        # In the undeformed state, internal bond forces vanish and no contacts overlap.
        self.assertLess(float(force.abs().max()), 1e-8)

    def test_unbonded_particle_contact_matches_linear_spring_law(self):
        sim = IceDEM3D(DEM3DConfig(
            nx=3, ny=3, nz=2, fix_x_edges=False, drag=0.0,
            contact_damping=0.0, tool_radius=0.01, tool_gap=1.0,
        ))
        sim.alive[:] = False
        i, j = (int(v) for v in sim.pairs[0])
        delta = sim.positions[j] - sim.positions[i]
        normal = delta / torch.linalg.vector_norm(delta)
        overlap = 0.01
        sim.positions[j] = sim.positions[i] + normal * (
            2 * sim.config.radius - overlap
        )
        force, reaction = sim.forces(update_fracture=False)
        expected = sim.config.contact_stiffness * overlap
        torch.testing.assert_close(force[i], -expected * normal, atol=1e-9, rtol=1e-12)
        torch.testing.assert_close(force[j], expected * normal, atol=1e-9, rtol=1e-12)
        torch.testing.assert_close(force.sum(dim=0), -reaction, atol=1e-9, rtol=0)

    def test_fixed_particles_remain_fixed_under_external_load(self):
        sim = IceDEM3D(DEM3DConfig(nx=3, ny=3, nz=2, fix_x_edges=True, drag=0.0))
        before = sim.positions[sim.fixed].clone()
        load = torch.zeros_like(sim.positions)
        load[:, 2] = 0.1
        for _ in range(4):
            sim.step(load)
        torch.testing.assert_close(sim.positions[sim.fixed], before, atol=0, rtol=0)
        torch.testing.assert_close(sim.velocities[sim.fixed],
                                   torch.zeros_like(sim.velocities[sim.fixed]), atol=0, rtol=0)

    def test_timestep_study_uses_same_final_physical_time_and_finite_metrics(self):
        cfg = DEM3DConfig(
            nx=3, ny=3, nz=2, drag=0.0, fix_x_edges=False,
            tool_radius=0.03, tool_gap=0.0,
        )
        report = run_convergence_study(cfg, base_steps=3, refinement_factors=(1, 2))
        levels = report["levels"]
        self.assertEqual(len(levels), 2)
        self.assertAlmostEqual(levels[0]["final_time_s"], levels[1]["final_time_s"], places=12)
        self.assertEqual(levels[1]["steps"], 2 * levels[0]["steps"])
        for row in levels:
            for key in ("peak_abs_reaction_z_N", "final_mechanical_energy_J",
                        "final_energy_balance_residual_J",
                        "max_abs_energy_balance_residual_J"):
                self.assertTrue(math.isfinite(float(row[key])), key)
            self.assertGreater(row["dt_s"], 0)
        self.assertEqual(levels[-1]["broken_bond_delta_vs_finest"], 0)
        history = report["history"]
        self.assertEqual(len(history), sum(row["steps"] + 1 for row in levels))
        self.assertTrue(all(math.isfinite(row["reaction_z_N"]) for row in history))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_report(report, output)
            with (output / "reaction_history_3d.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(len(saved), len(history))
            self.assertTrue((output / "convergence_3d.csv").is_file())
            self.assertTrue((output / "convergence_3d.json").is_file())

    def test_invalid_refinement_schedule_rejected(self):
        cfg = DEM3DConfig(nx=3, ny=3, nz=2)
        with self.assertRaises(ValueError):
            run_convergence_study(cfg, base_steps=0)
        with self.assertRaises(ValueError):
            run_convergence_study(cfg, base_steps=1, refinement_factors=(1, 1))


if __name__ == "__main__":
    unittest.main()
