"""Regression tests: plane platen forces act only on intersecting particles.

The platen boundary used to add its viscous damping term to every particle,
whether or not the particle intersected the plane. With no particle in contact
that produced a fictitious reaction ``n * gamma_platen * |v_platen|`` and a
spurious drag on the whole specimen. These tests pin the corrected contract,
which is the same one the spherical tool contact already followed.
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D


def _model(**overrides) -> IceDEM3D:
    params = dict(
        nx=4, ny=4, nz=3, radius=0.025, density=917.0,
        bond_stiffness=2000.0, contact_stiffness=5000.0, breaking_strain=0.015,
        platen_stiffness=5000.0, platen_damping=0.5,
        top_platen_enabled=True, bottom_platen_enabled=True,
        top_platen_velocity=-0.05, bottom_platen_velocity=0.05,
        top_platen_gap=0.5, bottom_platen_gap=0.5,
    )
    params.update(overrides)
    sim = IceDEM3D(DEM3DConfig(**params))
    sim.tool_start[:] = torch.tensor([1.0e6, 1.0e6, 1.0e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()
    return sim


class PlatenContactMaskTests(unittest.TestCase):
    def test_no_contact_gives_zero_reaction(self):
        sim = _model()
        crossing = int((sim.positions[:, 2] + sim.config.radius > sim.top_platen_start_z).sum())
        self.assertEqual(crossing, 0)
        row = sim.diagnostics()
        self.assertEqual(row["top_platen_reaction_z_N"], 0.0)
        self.assertEqual(row["bottom_platen_reaction_z_N"], 0.0)
        self.assertEqual(row["platen_contact_energy_J"], 0.0)

    def test_no_contact_does_not_move_the_specimen(self):
        sim = _model()
        start_centre = sim.positions[:, 2].mean().item()
        start = sim.positions.clone()
        for _ in range(120):
            sim.step()
        self.assertAlmostEqual(sim.positions[:, 2].mean().item(), start_centre, places=12)
        torch.testing.assert_close(sim.positions, start, atol=1e-12, rtol=0)

    def test_reaction_equals_contacted_particle_sum(self):
        # Platens close enough to press the specimen, so the contact set is not empty.
        sim = _model(top_platen_gap=0.0, bottom_platen_gap=0.0)
        sim.time = 2.0e-3
        sim.forces(update_fracture=False)
        top_z = sim.top_platen_start_z + sim.time * sim.config.top_platen_velocity
        bottom_z = sim.bottom_platen_start_z + sim.time * sim.config.bottom_platen_velocity
        top_overlap = (sim.positions[:, 2] + sim.config.radius - top_z).clamp_min(0.0)
        bottom_overlap = (bottom_z + sim.config.radius - sim.positions[:, 2]).clamp_min(0.0)
        self.assertGreater(int((top_overlap > 0).sum()), 0)
        self.assertGreater(int((bottom_overlap > 0).sum()), 0)
        expected_top = (sim.config.platen_stiffness * top_overlap
                        + sim.config.platen_damping
                        * (sim.velocities[:, 2] - sim.config.top_platen_velocity)).clamp_min(0.0)
        expected_top = torch.where(top_overlap > 0, expected_top, torch.zeros_like(expected_top))
        expected_bottom = (sim.config.platen_stiffness * bottom_overlap
                           - sim.config.platen_damping
                           * (sim.velocities[:, 2] - sim.config.bottom_platen_velocity)).clamp_min(0.0)
        expected_bottom = torch.where(bottom_overlap > 0, expected_bottom,
                                      torch.zeros_like(expected_bottom))
        self.assertAlmostEqual(float(sim.last_top_platen_reaction_z), float(expected_top.sum()), places=9)
        self.assertAlmostEqual(float(sim.last_bottom_platen_reaction_z), -float(expected_bottom.sum()), places=9)

    def test_reaction_switches_on_only_when_the_plane_reaches_the_specimen(self):
        sim = _model(top_platen_gap=5.0e-5, bottom_platen_gap=5.0e-5)
        self.assertEqual(sim.diagnostics()["top_platen_reaction_z_N"], 0.0)
        activated = None
        for step in range(1, 400):
            sim.step()
            if sim.diagnostics()["top_platen_reaction_z_N"] > 0.0:
                activated = step
                break
        self.assertIsNotNone(activated)
        # Overlap appears only once the plane position crosses the outer particle
        # surface, so activation cannot happen on the first step.
        self.assertGreater(activated, 1)

    def test_tool_and_platen_agree_when_nothing_is_in_contact(self):
        sim = _model()
        _, reaction = sim.forces(update_fracture=False)
        self.assertTrue(math.isfinite(float(reaction.sum())))
        self.assertLess(abs(float(reaction[2])), 1e-9)


if __name__ == "__main__":
    unittest.main()
