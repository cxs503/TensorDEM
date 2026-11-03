"""Analytic verification and regression tests for the DEM3D Hertz contact law.

The reference values below come from the classical two-body Hertz relations,
written independently of ``tensordem.dem3d`` helpers:

    1/E* = sum_i (1 - nu_i^2)/E_i,   1/R* = sum_i 1/R_i,
    F    = (4/3) E* sqrt(R*) delta^(3/2),
    U    = (8/15) E* sqrt(R*) delta^(5/2).
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import torch

from tensordem.dem3d import (
    DEM3DConfig,
    IceDEM3D,
    hertz_contact_potential,
    hertz_effective_radius,
    hertz_equivalent_linear_stiffness,
    hertz_normal_force,
    hertz_reduced_modulus,
    hertz_secant_stiffness,
)

E_YOUNG = 1.0e9
NU = 0.3
RADIUS = 0.025
DENSITY = 917.0


def reference_sphere_sphere(overlap: float) -> float:
    e_eff = 1.0 / (2.0 * (1.0 - NU**2) / E_YOUNG)
    r_eff = 1.0 / (1.0 / RADIUS + 1.0 / RADIUS)
    return (4.0 / 3.0) * e_eff * math.sqrt(r_eff) * overlap**1.5


def reference_rigid_plane(overlap: float) -> float:
    e_eff = 1.0 / ((1.0 - NU**2) / E_YOUNG)
    return (4.0 / 3.0) * e_eff * math.sqrt(RADIUS) * overlap**1.5


class HertzLawTests(unittest.TestCase):
    def test_force_matches_independent_closed_form(self) -> None:
        for overlap in (1e-6, 1e-4, 1e-3, 1e-2):
            measured = float(hertz_normal_force(
                torch.tensor(overlap, dtype=torch.float64),
                hertz_reduced_modulus(E_YOUNG, NU),
                hertz_effective_radius(RADIUS)))
            reference = reference_sphere_sphere(overlap)
            self.assertLess(abs(measured / reference - 1.0), 1e-12)
            rigid = float(hertz_normal_force(
                torch.tensor(overlap, dtype=torch.float64),
                hertz_reduced_modulus(E_YOUNG, NU, rigid_counterpart=True),
                hertz_effective_radius(RADIUS, rigid_counterpart=True)))
            self.assertLess(abs(rigid / reference_rigid_plane(overlap) - 1.0), 1e-12)

    def test_reduced_modulus_mapping(self) -> None:
        self.assertAlmostEqual(hertz_reduced_modulus(E_YOUNG, NU),
                               E_YOUNG / (2.0 * (1.0 - NU**2)), places=6)
        self.assertAlmostEqual(hertz_reduced_modulus(E_YOUNG, NU, rigid_counterpart=True),
                               E_YOUNG / (1.0 - NU**2), places=6)
        self.assertAlmostEqual(hertz_effective_radius(RADIUS), 0.5 * RADIUS)
        self.assertAlmostEqual(hertz_effective_radius(RADIUS, rigid_counterpart=True), RADIUS)

    def test_force_exponent_is_three_halves(self) -> None:
        deltas = torch.logspace(-6, -2, 33, dtype=torch.float64)
        forces = hertz_normal_force(deltas, hertz_reduced_modulus(E_YOUNG, NU),
                                    hertz_effective_radius(RADIUS))
        slope = float((torch.log(forces).diff() / torch.log(deltas).diff()).mean())
        self.assertAlmostEqual(slope, 1.5, places=10)

    def test_potential_matches_force_integral(self) -> None:
        e_star = hertz_reduced_modulus(E_YOUNG, NU)
        r_eff = hertz_effective_radius(RADIUS)
        delta, intervals = 2e-3, 20000
        step = delta / intervals
        total = 0.0
        for i in range(intervals):
            f0 = float(hertz_normal_force(i * step, e_star, r_eff))
            f1 = float(hertz_normal_force((i + 1) * step, e_star, r_eff))
            total += 0.5 * (f0 + f1) * step
        measured = float(hertz_contact_potential(delta, e_star, r_eff))
        self.assertLess(abs(measured / total - 1.0), 1e-9)

    def test_secant_stiffness_identity(self) -> None:
        e_star, r_eff, delta = hertz_reduced_modulus(E_YOUNG, NU), hertz_effective_radius(RADIUS), 1e-3
        self.assertAlmostEqual(
            float(hertz_secant_stiffness(delta, e_star, r_eff)),
            float(hertz_normal_force(delta, e_star, r_eff)) / delta, places=6)

    def test_equivalent_linear_stiffness_seeds_matching_force(self) -> None:
        delta = 5e-4
        k = hertz_equivalent_linear_stiffness(E_YOUNG, NU, RADIUS, delta, rigid_counterpart=True)
        self.assertAlmostEqual(k * delta, reference_rigid_plane(delta), places=6)

    def test_negative_overlap_gives_zero_force(self) -> None:
        e_star, r_eff = hertz_reduced_modulus(E_YOUNG, NU), hertz_effective_radius(RADIUS)
        self.assertEqual(float(hertz_normal_force(-1e-4, e_star, r_eff)), 0.0)
        self.assertEqual(float(hertz_contact_potential(-1e-4, e_star, r_eff)), 0.0)


class HertzConfigTests(unittest.TestCase):
    def test_default_model_is_linear_and_positive(self) -> None:
        config = DEM3DConfig()
        self.assertEqual(config.contact_model, "linear")
        self.assertGreater(config.recommended_dt, 0.0)

    def test_invalid_contact_model_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DEM3DConfig(contact_model="mindlin")

    def test_invalid_elastic_parameters_rejected(self) -> None:
        for kwargs in ({"poisson_ratio": 0.5}, {"poisson_ratio": -0.1},
                       {"young_modulus": 0.0}, {"young_modulus": float("nan")},
                       {"hertz_stability_overlap_fraction": 0.0},
                       {"hertz_stability_overlap_fraction": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DEM3DConfig(contact_model="hertz", **kwargs)

    def test_hertz_bound_is_stiffer_than_linear_bound(self) -> None:
        linear = DEM3DConfig()
        hertz = DEM3DConfig(contact_model="hertz", young_modulus=E_YOUNG, poisson_ratio=NU)
        self.assertLess(hertz.recommended_dt, linear.recommended_dt)
        self.assertGreater(hertz._bound_stiffness(5_000.0, rigid_counterpart=False), 5_000.0)

    def test_step_guard_rejects_step_above_bounds(self) -> None:
        base = DEM3DConfig(nx=3, ny=3, nz=3, contact_model="hertz",
                           young_modulus=E_YOUNG, poisson_ratio=NU)
        with self.assertRaises(ValueError):
            IceDEM3D(replace(base, dt=2.0 * base.recommended_dt))


class HertzSolverPathTests(unittest.TestCase):
    def _static_indent(self, *, model: str, overlap: float) -> tuple[float, float]:
        base = DEM3DConfig(
            nx=3, ny=3, nz=3, radius=RADIUS, density=DENSITY,
            contact_model=model, young_modulus=E_YOUNG, poisson_ratio=NU,
            contact_damping=0.0, drag=0.0, tool_radius=0.1, tool_speed=0.01, tool_gap=0.0)
        sim = IceDEM3D(replace(base, dt=0.5 * base.recommended_dt))
        sim.fixed[:] = True
        sim.time = overlap / base.tool_speed
        _, reaction = sim.forces(update_fracture=False)
        return abs(float(reaction[2])), sim.time

    def test_static_indentation_matches_rigid_plane_closed_form(self) -> None:
        for overlap in (1e-5, 1e-3, 1e-2):
            measured, _ = self._static_indent(model="hertz", overlap=overlap)
            self.assertLess(abs(measured / reference_rigid_plane(overlap) - 1.0), 1e-9)

    def test_linear_model_keeps_penalty_identity(self) -> None:
        overlap, stiffness = 1e-3, 5_000.0
        measured, _ = self._static_indent(model="linear", overlap=overlap)
        # Single indenter contact with zero damping => exactly k * overlap.
        self.assertAlmostEqual(measured / (stiffness * overlap), 1.0, places=12)

    def test_contact_energy_uses_hertz_potential(self) -> None:
        overlap = 1e-3
        base = DEM3DConfig(
            nx=3, ny=3, nz=3, radius=RADIUS, density=DENSITY,
            contact_model="hertz", young_modulus=E_YOUNG, poisson_ratio=NU,
            contact_damping=0.0, drag=0.0, tool_radius=0.1, tool_speed=0.01, tool_gap=0.0)
        sim = IceDEM3D(replace(base, dt=0.5 * base.recommended_dt))
        sim.fixed[:] = True
        sim.time = overlap / base.tool_speed
        diagnostics = sim.diagnostics()
        e_star = hertz_reduced_modulus(E_YOUNG, NU, rigid_counterpart=True)
        expected = float(hertz_contact_potential(overlap, e_star, RADIUS))
        self.assertLess(abs(diagnostics["tool_contact_energy_J"] / expected - 1.0), 1e-9)
        self.assertAlmostEqual(diagnostics["particle_contact_energy_J"], 0.0, places=12)

    def test_platen_contact_uses_hertz_law(self) -> None:
        base = DEM3DConfig(
            nx=3, ny=3, nz=3, radius=RADIUS, density=DENSITY,
            contact_model="hertz", young_modulus=E_YOUNG, poisson_ratio=NU,
            contact_damping=0.0, drag=0.0, platen_damping=0.0,
            bottom_platen_enabled=True, bottom_platen_gap=0.0, fix_x_edges=False)
        sim = IceDEM3D(replace(base, dt=0.5 * base.recommended_dt))
        overlap = 1e-3
        # Push the lowest layer down by the target overlap and freeze it there.
        lowest = sim.positions[:, 2] == sim.positions[:, 2].min()
        sim.positions[lowest, 2] -= overlap
        sim.fixed[:] = True
        sim.forces(update_fracture=False)
        per_particle = abs(float(sim.last_bottom_platen_reaction_z)) / int(lowest.sum())
        self.assertLess(abs(per_particle / reference_rigid_plane(overlap) - 1.0), 1e-9)

    def test_elastic_impact_reproduces_hertz_peak_force(self) -> None:
        speed, dt_factor, duration = 0.5, 0.4, 1.5e-3
        base = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY, bond_stiffness=1e-4,
            contact_model="hertz", young_modulus=E_YOUNG, poisson_ratio=NU,
            contact_damping=0.0, drag=0.0, platen_damping=0.0,
            bottom_platen_enabled=True, bottom_platen_gap=0.0,
            fix_x_edges=False, tool_speed=1.0)
        sim = IceDEM3D(replace(base, dt=dt_factor * base.recommended_dt))
        xy = torch.linalg.vector_norm(sim.positions[:, :2], dim=1)
        depth = torch.isclose(sim.positions[:, 2], sim.positions[:, 2].min())
        index = int(torch.nonzero(depth, as_tuple=False).flatten()[torch.argmin(xy[depth])])
        sim.velocities[index, 2] = -speed
        peak = 0.0
        for _ in range(int(math.ceil(duration / sim.dt))):
            sim.step()
            peak = max(peak, abs(float(sim.last_bottom_platen_reaction_z)))
        e_plane = hertz_reduced_modulus(E_YOUNG, NU, rigid_counterpart=True)
        delta_max = (15.0 * base.mass * speed**2 / (16.0 * e_plane * math.sqrt(RADIUS))) ** 0.4
        reference = (4.0 / 3.0) * e_plane * math.sqrt(RADIUS) * delta_max**1.5
        self.assertLess(abs(peak / reference - 1.0), 0.03)


if __name__ == "__main__":
    unittest.main()