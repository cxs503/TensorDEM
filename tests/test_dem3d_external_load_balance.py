"""Regression guards for the DEM3D external-load and platen-damping path.

These tests pin two physics invariants that were silently broken between
``a68fa8a`` and ``a40e29a`` (PR #73) and that made the bending benchmarks fail:

1. ``IceDEM3D.forces(external_forces=...)`` must add the nodal loads to the
   internal force vector.  When the term is missing the solver returns a zero
   response for every externally loaded case (no exception, no drift).
2. The bottom-platen damper must oppose relative normal approach.  A flipped
   sign turns it into a negative-damping (energy-injecting) term, which is
   invisible while ``platen_damping = 0`` -- the reason it escaped review.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D


def _small(**overrides) -> IceDEM3D:
    params = dict(
        nx=5, ny=2, nz=2, radius=0.01, density=917.0, bond_stiffness=2_000.0,
        contact_stiffness=5_000.0, contact_damping=0.05, drag=0.2,
        fix_x_edges=False, tool_gap=1.0, tool_radius=0.01, tool_speed=0.01,
    )
    params.update(overrides)
    sim = IceDEM3D(DEM3DConfig(**params))
    # Park the indenter far away so only bonds/loads/platens act on the lattice.
    sim.tool_start[:] = torch.tensor([1e6, 1e6, 1e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()
    return sim


class DEM3DExternalLoadBalanceTests(unittest.TestCase):
    def test_external_loads_are_added_to_the_internal_force_vector(self):
        sim = _small()
        loads = torch.zeros_like(sim.positions)
        loads[3, 2] = -0.75
        loads[7, 0] = 1.25
        baseline, _ = sim.forces(update_fracture=False)
        loaded, _ = sim.forces(external_forces=loads, update_fracture=False)
        torch.testing.assert_close(loaded - baseline, loads, atol=0.0, rtol=0.0)

    def test_vertical_force_resultant_equals_applied_load(self):
        # With drag disabled the bond forces cancel pairwise, so the summed
        # vertical force equals the summed vertical external load exactly.
        sim = _small(drag=0.0)
        loads = torch.zeros_like(sim.positions)
        loads[:, 2] = -0.01
        force, _ = sim.forces(external_forces=loads, update_fracture=False)
        self.assertAlmostEqual(float(force[:, 2].sum()), float(loads[:, 2].sum()), places=15)

    def test_loaded_benchmark_transfers_load_to_supports(self):
        from scripts.benchmark_dem3d_three_point_bending import run_three_point_bending

        with tempfile.TemporaryDirectory() as tmp:
            report = run_three_point_bending(Path(tmp) / "bending", steps=12)
        self.assertEqual(report["verdict"], "PASS", report["failed_checks"])
        self.assertGreater(report["peak_support_reaction_abs_N"], 0.0)
        self.assertGreater(report["peak_midspan_deflection_abs_m"], 0.0)
        # The reaction must stay inside the applied-load envelope, and the
        # response must grow with the ramp rather than be a constant offset.
        self.assertLessEqual(
            report["peak_support_reaction_abs_N"], report["peak_applied_load_N"] * 1.05
        )
        first = abs(report["rows"][1]["midspan_deflection_m"])
        last = abs(report["rows"][-1]["midspan_deflection_m"])
        self.assertGreater(last, first)

    def test_fixed_supports_stay_pinned_under_load(self):
        sim = _small()
        sim.fixed[:] = False
        sim.fixed[[0, len(sim.positions) - 1]] = True
        loads = torch.zeros_like(sim.positions)
        loads[:, 2] = -0.02
        for _ in range(20):
            sim.step(external_forces=loads)
        torch.testing.assert_close(
            sim.positions[sim.fixed], sim.initial_positions[sim.fixed], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            sim.velocities[sim.fixed], torch.zeros_like(sim.velocities[sim.fixed]),
            atol=0.0, rtol=0.0,
        )

    def test_bottom_platen_damper_opposes_approach(self):
        sim = _small(bottom_platen_enabled=True, platen_stiffness=4_000.0,
                     platen_damping=50.0, drag=0.0)
        bottom = sim.positions[:, 2] == sim.positions[:, 2].min()
        # Press the bottom layer into the platen so the contact carries force.
        sim.positions[bottom, 2] -= 0.002
        count = int(bottom.sum())
        self.assertGreater(count, 0)

        def total_force_z(velocity_z: float) -> float:
            sim.velocities.zero_()
            sim.velocities[bottom, 2] = velocity_z
            force, _ = sim.forces(update_fracture=False)
            return float(force[:, 2].sum())

        static = total_force_z(0.0)
        approaching = total_force_z(-0.03)   # moving down, into the platen
        separating = total_force_z(+0.03)    # moving up, away from the platen
        damping = 50.0 * 0.03 * count
        self.assertAlmostEqual(approaching - static, damping, delta=1e-12)
        self.assertAlmostEqual(separating - static, -damping, delta=1e-12)

    def test_top_platen_damper_opposes_approach(self):
        sim = _small(top_platen_enabled=True, platen_stiffness=4_000.0,
                     platen_damping=50.0, drag=0.0)
        top = sim.positions[:, 2] == sim.positions[:, 2].max()
        sim.positions[top, 2] += 0.002
        count = int(top.sum())

        def total_force_z(velocity_z: float) -> float:
            sim.velocities.zero_()
            sim.velocities[top, 2] = velocity_z
            force, _ = sim.forces(update_fracture=False)
            return float(force[:, 2].sum())

        static = total_force_z(0.0)
        approaching = total_force_z(+0.03)   # moving up, into the top platen
        separating = total_force_z(-0.03)
        damping = 50.0 * 0.03 * count
        self.assertAlmostEqual(approaching - static, -damping, delta=1e-12)
        self.assertAlmostEqual(separating - static, +damping, delta=1e-12)


if __name__ == "__main__":
    unittest.main()