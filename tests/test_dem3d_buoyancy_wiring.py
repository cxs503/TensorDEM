"""Gates for the opt-in gravity + Archimedes buoyancy wiring in ``IceDEM3D``.

The analytic anchors live in :mod:`tensordem.buoyancy_wet3d`
(spherical-cap volume, Archimedes draft, waterplane stiffness); this file checks
that the *solver* reproduces them, that the new term is exactly inert when
disabled, and that the treated configuration validates its own inputs.

Acceptance thresholds (see ``docs/dem3d_buoyancy_wiring.md`` §6):
* static force identity — machine precision (round-off only);
* Archimedes draft — relative error < 3 % of the analytic draft;
* hydrostatic stiffness — relative error < 3 %;
* free fall with gravity only — error must shrink with ``dt``.
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.buoyancy_wet3d import equilibrium_centre_height, submerged_volume_sphere
from tensordem.buoyancy_wiring_verification import (
    free_fall_case,
    inert_when_disabled,
    isolated_particle,
    relaxation_case,
    static_force_cases,
    stiffness_case,
    wet_drag_case,
)
from tensordem.dem3d import DEM3DConfig, IceDEM3D


class StaticForceIdentityTest(unittest.TestCase):
    """Case 1: the body force equals ``-mg + rho_w g V_sub`` exactly."""

    def test_dry_half_and_fully_submerged_heights(self) -> None:
        cases = static_force_cases()
        self.assertEqual(len(cases), 14)
        worst = max(case["abs_error_N"] for case in cases)
        self.assertLess(worst, 1e-12)
        # Bound the *signal* too, so a degenerate all-zero force cannot pass.
        signal = max(abs(case["reference_Fz_N"]) for case in cases)
        self.assertGreater(signal, 0.5)

    def test_dry_particle_sees_weight_only(self) -> None:
        dem, cfg = isolated_particle(z0=0.4, water_level=None)
        self.assertFalse(dem.buoyancy_active)
        self.assertAlmostEqual(float(dem.body_forces()[0, 2]),
                               -cfg.mass * cfg.gravity, places=14)
        self.assertEqual(dem.body_force_ledger()["buoyancy_force_z_N"], 0.0)

    def test_net_force_sign_changes_with_draft(self) -> None:
        """Dry ice sinks, a fully submerged ice sphere is buoyant."""
        dem, cfg = isolated_particle(z0=0.06)
        dem.positions[0, 2] = 0.06
        self.assertLess(float(dem.body_forces()[0, 2]), 0.0)
        dem.positions[0, 2] = -1.0
        self.assertGreater(float(dem.body_forces()[0, 2]), 0.0)


class ArchimedesDraftTest(unittest.TestCase):
    """Case 2: a dropped particle settles on the analytic draft."""

    def test_settles_on_analytic_draft_ice(self) -> None:
        case = relaxation_case(density=917.0, n_steps=1200)
        self.assertLess(abs(case["rel_error_draft"]), 0.03)
        # Converging from above: the particle approaches h_eq monotonically late.
        tail = [z for _, z in case["trace"][-5:]]
        self.assertLess(tail[-1], 0.0)
        self.assertGreater(tail[-1], case["h_eq_analytic_m"] - 0.25 * 0.025)

    def test_draft_scales_with_density(self) -> None:
        light = relaxation_case(density=850.0, n_steps=1200)
        heavy = relaxation_case(density=917.0, n_steps=1200)
        self.assertLess(light["draft_analytic_m"], heavy["draft_analytic_m"])
        self.assertLess(abs(light["rel_error_draft"]), 0.03)
        self.assertLess(abs(heavy["rel_error_draft"]), 0.03)


class HydrostaticStiffnessTest(unittest.TestCase):
    """Case 3: central difference of the net force reproduces ``rho_w g A_wp``."""

    def test_central_difference_matches_waterplane_stiffness(self) -> None:
        case = stiffness_case()
        self.assertLess(case["rel_error"], 0.03)
        self.assertGreater(case["reference_K_N_m"], 1.0)


class GravityKinematicsTest(unittest.TestCase):
    """Case 4: dry box reproduces ``z = z0 - g t^2/2`` and the error shrinks with dt."""

    def test_free_fall_and_dt_refinement(self) -> None:
        coarse = free_fall_case(dt_scale=1.0)
        fine = free_fall_case(dt_scale=0.5)
        for case in (coarse, fine):
            self.assertLess(abs(case["rel_error"]), 0.01)
        self.assertLess(fine["abs_error_m"], coarse["abs_error_m"])


class InertWhenDisabledTest(unittest.TestCase):
    """Case 5: default configuration is bit-for-bit unchanged."""

    def test_default_config_has_zero_body_force(self) -> None:
        record = inert_when_disabled()
        self.assertTrue(record["forces_bitwise_equal"])
        self.assertEqual(record["default_body_force_abs_max_N"], 0.0)
        self.assertEqual(record["dt_default_s"], record["dt_with_body_fields_s"])
        self.assertFalse(record["buoyancy_active_default"])

    def test_water_far_above_lattice_contributes_nothing(self) -> None:
        dem = IceDEM3D(DEM3DConfig(water_level=-10.0, gravity=9.81))
        self.assertTrue(torch.all(dem.submerged_fraction() == 0.0))
        ledger = dem.body_force_ledger()
        self.assertEqual(ledger["buoyancy_force_z_N"], 0.0)
        self.assertLess(ledger["gravity_force_z_N"], 0.0)


class WetDragTest(unittest.TestCase):
    """Case 6: the linear wet drag is weighted by the submerged fraction."""

    def test_linear_wet_drag_matches_submerged_fraction(self) -> None:
        case = wet_drag_case()
        self.assertLess(case["abs_error_N"], 1e-14)
        self.assertTrue(0.0 < case["f_sub"] < 1.0)

    def test_wet_drag_absent_when_coefficient_is_zero(self) -> None:
        dem, cfg = isolated_particle(z0=-0.02, wet_drag_coefficient=0.0)
        dem.velocities[0] = torch.tensor([0.0, 0.0, -0.5], dtype=dem.dtype)
        body = dem.body_forces()
        expected = -cfg.mass * cfg.gravity + cfg.water_density * cfg.gravity * (
            4.0 / 3.0) * math.pi * cfg.radius ** 3 * float(dem.submerged_fraction()[0])
        self.assertAlmostEqual(float(body[0, 2]), expected, places=14)


class ConfigValidationTest(unittest.TestCase):
    """The new knobs reject non-physical input instead of silently degrading."""

    def test_invalid_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            DEM3DConfig(gravity=-1.0)
        with self.assertRaises(ValueError):
            DEM3DConfig(gravity=float("nan"))
        with self.assertRaises(ValueError):
            DEM3DConfig(water_level=float("inf"))
        with self.assertRaises(ValueError):
            DEM3DConfig(water_density=0.0)
        with self.assertRaises(ValueError):
            DEM3DConfig(wet_drag_coefficient=-0.1)

    def test_defaults_are_off(self) -> None:
        cfg = DEM3DConfig()
        self.assertEqual(cfg.gravity, 0.0)
        self.assertIsNone(cfg.water_level)
        self.assertEqual(cfg.wet_drag_coefficient, 0.0)

    def test_equilibrium_anchor_is_consistent_with_density(self) -> None:
        """The draft the solver is scored against depends only on rho_p/rho_w."""
        h = equilibrium_centre_height(0.025, 917.0, 1025.0)
        ratio = submerged_volume_sphere(h, 0.025) / ((4.0 / 3.0) * math.pi * 0.025 ** 3)
        self.assertAlmostEqual(ratio, 917.0 / 1025.0, places=12)


if __name__ == "__main__":
    unittest.main()