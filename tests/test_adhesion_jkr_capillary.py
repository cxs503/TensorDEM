"""Adhesion campaign P96-N4: JKR anchor, Laplace liquid bridge, cohesive pile.

Acceptance criteria exercised here (see docs/adhesion_jkr_capillary.md):

* F1 JKR pull-off force ``1.5 pi gamma R*`` reproduced numerically (<= 2 %)
* F2 Laplace-solved capillary force vs ``2 pi gamma R cos(theta)`` (<= 3 %)
* F3 ``gamma -> 0`` degeneration of the JKR law to Hertz (<= 1 %)
* F4 cohesive pile repose angle exceeds the cohesionless one by >= 5 deg
* F5 this module (>= 10 tests)
* Interface compatibility with the P96-N3 superquadric contact detector
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from tensordem.adhesion import (
    adhesive_contact_force,
    capillary_force_analytic,
    capillary_force_laplace,
    effective_radius,
    hertz_normal_force,
    jkr_contact_radius_from_force,
    jkr_force_from_contact_radius,
    jkr_hertz_relative_error,
    jkr_normal_force,
    jkr_overlap_from_contact_radius,
    jkr_pull_off_force,
    jkr_pull_off_numerical,
    reduced_modulus,
    solve_liquid_bridge,
)
from tensordem.adhesive_pile2d import PileCase, simulate_pile
from tensordem.superquadric import Superquadric, hertz_force_superquadric, superquadric_contact

E_MOD = 1.0e6
NU = 0.3
RADIUS = 1.0e-3


class TestJKRAnchor(unittest.TestCase):
    """F1: the JKR pull-off force is 1.5 pi gamma R* (sphere-plane convention)."""

    def setUp(self) -> None:
        self.e_star = reduced_modulus(E_MOD, NU, rigid_counterpart=True)

    def test_pull_off_force_sphere_plane_is_1p5_pi_gamma_R(self) -> None:
        gamma, r_eff = 0.05, RADIUS  # sphere of radius R against a rigid plane
        self.assertAlmostEqual(jkr_pull_off_force(gamma, r_eff),
                               1.5 * math.pi * gamma * r_eff, places=18)
        self.assertAlmostEqual(jkr_pull_off_force(gamma, r_eff),
                               1.5 * math.pi * gamma * RADIUS, places=18)

    def test_pull_off_force_equal_spheres_uses_effective_radius(self) -> None:
        gamma = 0.05
        r_eff = effective_radius(RADIUS, RADIUS)
        self.assertAlmostEqual(r_eff, 0.5 * RADIUS, places=18)
        # the same 1.5 pi gamma R* law, with R* = R/2 for two equal spheres
        self.assertAlmostEqual(jkr_pull_off_force(gamma, r_eff),
                               0.75 * math.pi * gamma * RADIUS, places=18)

    def test_pull_off_numerical_matches_analytic_within_2_percent(self) -> None:
        worst = 0.0
        for gamma in (0.005, 0.02, 0.05, 0.1):
            out = jkr_pull_off_numerical(self.e_star, RADIUS, gamma)
            worst = max(worst, out["rel_error"])
            self.assertLessEqual(out["rel_error"], 0.02,
                                 f"gamma={gamma}: {out['rel_error']:.3e}")
            self.assertAlmostEqual(out["f_pull"] / jkr_pull_off_force(gamma, RADIUS),
                                   1.0, places=6)
        self.assertLess(worst, 1e-6)  # measured error is at machine level

    def test_jkr_relations_are_mutually_consistent(self) -> None:
        """a(F(a)) round-trip and the pull-off point is the force minimum."""
        gamma = 0.05
        a_c = jkr_pull_off_numerical(self.e_star, RADIUS, gamma)["a_c"]
        # only the equilibrium branch a >= a_c has a force-load inversion
        a = torch.logspace(math.log10(a_c), -3.0, 13, dtype=torch.float64)
        f = jkr_force_from_contact_radius(a, self.e_star, RADIUS, gamma)
        a_back = jkr_contact_radius_from_force(f, self.e_star, RADIUS, gamma)
        rel = ((a_back - a).abs() / a).max().item()
        # worst case is at the pull-off point, where the JKR radicand vanishes and
        # the closed-form inversion loses digits to cancellation
        self.assertLess(rel, 1e-5)

        out = jkr_pull_off_numerical(self.e_star, RADIUS, gamma)
        f_at_ac = float(jkr_force_from_contact_radius(
            torch.tensor(out["a_c"], dtype=torch.float64), self.e_star, RADIUS, gamma))
        self.assertAlmostEqual(f_at_ac / -jkr_pull_off_force(gamma, RADIUS), 1.0, places=9)

    def test_displacement_controlled_pull_off_is_force_limit(self) -> None:
        """Unloading in displacement control cannot exceed the pull-off force."""
        gamma = 0.05
        pull = jkr_pull_off_numerical(self.e_star, RADIUS, gamma)
        delta_c = pull["delta_c"]
        delta_detach = pull["delta_detach"]
        deltas = torch.linspace(1.5 * delta_c, 5e-6, 120, dtype=torch.float64)
        f = jkr_normal_force(deltas, self.e_star, RADIUS, gamma)
        f_pull = jkr_pull_off_force(gamma, RADIUS)
        self.assertLessEqual(abs(float(f.min())), f_pull * (1.0 + 1e-9))
        # the sampled minimum sits one grid step inside the attached branch, so the
        # agreement is asserted at 1e-5 rather than at the machine level
        self.assertAlmostEqual(float(f.min()) / -f_pull, 1.0, places=5)
        # detached branch returns exactly the pull-off force
        detached = jkr_normal_force(1.2 * delta_detach, self.e_star, RADIUS, gamma,
                                    return_state=True)
        self.assertFalse(bool(detached.attached))
        self.assertAlmostEqual(float(detached.force) / -f_pull, 1.0, places=12)


class TestHertzDegeneration(unittest.TestCase):
    """F3: gamma -> 0 must degenerate to the Hertz law."""

    def setUp(self) -> None:
        self.e_star = reduced_modulus(E_MOD, NU, rigid_counterpart=True)

    def test_zero_surface_energy_is_exactly_hertz(self) -> None:
        deltas = torch.logspace(-8, -5, 13, dtype=torch.float64)
        f_jkr = jkr_normal_force(deltas, self.e_star, RADIUS, 0.0)
        f_hertz = hertz_normal_force(deltas, self.e_star, RADIUS)
        self.assertLess(float(((f_jkr - f_hertz).abs() / f_hertz).max()), 1e-14)
        out = jkr_hertz_relative_error(self.e_star, RADIUS, 0.0)
        self.assertEqual(out["max_rel_error"], 0.0)

    def test_small_surface_energy_within_1_percent(self) -> None:
        out = jkr_hertz_relative_error(self.e_star, RADIUS, 1e-10)
        self.assertLessEqual(out["max_rel_error"], 1e-2,
                             f"max rel error {out['max_rel_error']:.3e}")

    def test_deviation_scales_linearly_with_gamma(self) -> None:
        errs = [jkr_hertz_relative_error(self.e_star, RADIUS, g)["max_rel_error"]
                for g in (1e-12, 1e-10, 1e-8)]
        self.assertGreater(errs[1], errs[0])
        self.assertGreater(errs[2], errs[1])
        # leading order is linear in gamma (measured 103x for a 100x gamma step)
        self.assertAlmostEqual(errs[2] / errs[1], 100.0, delta=10.0)


class TestLiquidBridge(unittest.TestCase):
    """F2: the numerically solved Young-Laplace bridge vs 2 pi gamma R cos(theta)."""

    def test_analytic_anchor_limits(self) -> None:
        gamma, radius = 0.072, RADIUS
        self.assertAlmostEqual(capillary_force_analytic(gamma, radius, 0.0),
                               2.0 * math.pi * gamma * radius, places=18)
        self.assertAlmostEqual(capillary_force_analytic(gamma, radius, 60.0),
                               0.5 * 2.0 * math.pi * gamma * radius, places=15)
        with self.assertRaises(ValueError):
            capillary_force_analytic(gamma, radius, 90.0)

    def test_laplace_force_within_3_percent_for_small_bridges(self) -> None:
        rows = []
        for theta in (0.0, 30.0):
            for vol in (1e-7, 1e-6):
                res = solve_liquid_bridge(RADIUS, gap=0.0, contact_angle_deg=theta,
                                          volume_over_R3=vol, gamma=0.072)
                self.assertTrue(res.converged, f"theta={theta} V={vol}")
                rows.append(res)
                self.assertLessEqual(res.rel_error_vs_analytic, 0.03,
                                     f"theta={theta} V={vol}: "
                                     f"{res.rel_error_vs_analytic:.4f}")
        self.assertGreaterEqual(len(rows), 4)

    def test_two_force_expressions_agree(self) -> None:
        """Neck-cut vs sphere-cap force: an independent check of the solution."""
        for theta in (0.0, 30.0, 60.0):
            res = solve_liquid_bridge(RADIUS, gap=0.0, contact_angle_deg=theta,
                                      volume_over_R3=1e-6, gamma=0.072)
            self.assertLess(res.consistency, 1e-6)
            self.assertGreater(res.force, 0.0)
            self.assertGreater(res.suction, 0.0)  # wetting bridge holds by suction

    def test_deviation_shrinks_as_volume_goes_to_zero(self) -> None:
        errs = []
        for vol in (1e-7, 1e-6, 1e-5):
            res = solve_liquid_bridge(RADIUS, gap=0.0, contact_angle_deg=0.0,
                                      volume_over_R3=vol, gamma=0.072)
            errs.append(res.rel_error_vs_analytic)
        self.assertLess(errs[0], errs[1])
        self.assertLess(errs[1], errs[2])
        # Richardson-style extrapolation in the filling angle: the V -> 0 limit is
        # the analytic law.
        rows = [solve_liquid_bridge(RADIUS, gap=0.0, contact_angle_deg=0.0,
                                    volume_over_R3=v, gamma=0.072)
                for v in (1e-7, 1e-6, 1e-5)]
        beta = np.array([r.filling_angle_deg for r in rows])
        ratio = np.array([r.force / r.analytic_force for r in rows])
        slope, intercept = np.polyfit(beta, ratio, 1)
        self.assertLess(abs(intercept - 1.0), 2e-3)
        self.assertLess(slope, 0.0)

    def test_volume_round_trip(self) -> None:
        res = solve_liquid_bridge(RADIUS, gap=0.0, contact_angle_deg=0.0,
                                  volume_over_R3=1e-5, gamma=0.072)
        self.assertLess(res.volume_rel_error, 2e-2)
        self.assertLess(res.residual_angle, 5e-3)
        self.assertGreater(res.neck_radius, 0.0)
        self.assertLess(res.neck_radius, res.contact_radius)

    def test_convenience_wrapper_matches_solver(self) -> None:
        direct = solve_liquid_bridge(RADIUS, contact_angle_deg=0.0,
                                     volume_over_R3=1e-6, gamma=0.072)
        wrapped = capillary_force_laplace(RADIUS, contact_angle_deg=0.0,
                                          volume_over_R3=1e-6, gamma=0.072)
        self.assertAlmostEqual(wrapped / direct.force, 1.0, places=12)


class TestSuperquadricInterface(unittest.TestCase):
    """Adhesion stacks on the P96-N3 true-shape contact without touching it."""

    def test_sphere_limit_pull_off_from_contact_detector(self) -> None:
        R = RADIUS
        a = Superquadric(a=R, b=R, c=R, n1=2.0, n2=2.0, center=(0.0, 0.0, 0.0))
        b = Superquadric(a=R, b=R, c=R, n1=2.0, n2=2.0,
                         center=(0.0, 0.0, 2.0 * R - 1e-6))
        contact = superquadric_contact(a, b)
        self.assertTrue(contact.in_contact)
        self.assertAlmostEqual(contact.effective_radius / (0.5 * R), 1.0, delta=0.05)

        e_star = reduced_modulus(E_MOD, NU)
        out = adhesive_contact_force(contact, e_star, gamma_solid=0.05)
        expected_pull = jkr_pull_off_force(0.05, contact.effective_radius)
        self.assertAlmostEqual(out["f_pull"] / expected_pull, 1.0, places=12)
        # same 1.5 pi gamma R* law expressed with the sphere radius
        self.assertAlmostEqual(out["f_pull"] / (0.75 * math.pi * 0.05 * R), 1.0,
                              delta=0.05)
        # the adhesive contact is tensile at this tiny overlap
        self.assertLess(out["f_normal"], 0.0)

    def test_capillary_term_adds_to_normal_force(self) -> None:
        R = RADIUS
        a = Superquadric(a=R, b=R, c=R, n1=2.0, n2=2.0, center=(0.0, 0.0, 0.0))
        b = Superquadric(a=R, b=R, c=R, n1=2.0, n2=2.0,
                         center=(0.0, 0.0, 2.0 * R - 5e-6))
        contact = superquadric_contact(a, b)
        e_star = reduced_modulus(E_MOD, NU)
        dry = adhesive_contact_force(contact, e_star, gamma_solid=0.0)
        wet = adhesive_contact_force(contact, e_star, gamma_solid=0.0,
                                     gamma_liquid=0.072, contact_angle_deg=0.0)
        self.assertAlmostEqual(dry["f_normal"], dry["f_hertz"], places=12)
        self.assertAlmostEqual(dry["f_pull"], 0.0, places=15)
        self.assertGreater(wet["f_capillary"], 0.0)
        self.assertAlmostEqual(wet["f_normal"], dry["f_normal"] - wet["f_capillary"],
                               places=12)
        # Hertz reference from the superquadric module itself
        self.assertAlmostEqual(
            wet["f_hertz"] / hertz_force_superquadric(contact.overlap, e_star,
                                                     contact.effective_radius),
            1.0, places=12)


class TestCohesivePile(unittest.TestCase):
    """F4: adhesion must steepen the pile by at least 5 degrees."""

    @classmethod
    def setUpClass(cls) -> None:
        base = dict(wedge_half_base_in_d=8.0, settle_time=8.0)
        cls.dry = simulate_pile(PileCase(bond_number=0.0, **base))
        cls.wet = simulate_pile(PileCase(bond_number=20.0, **base))

    def test_cohesionless_heap_reference_state(self) -> None:
        """Negative control: without adhesion the disks roll out into a flat heap."""
        self.assertGreater(self.dry["theta_r_deg"], 0.0)
        self.assertLess(self.dry["theta_r_deg"], 35.0)
        self.assertEqual(self.dry["pull_off"], 0.0)
        # a cohesionless 2D disk heap keeps creeping, so only a loose bound applies
        self.assertLess(self.dry["kinetic_energy"], 40.0)

    def test_adhesion_increases_repose_angle_by_at_least_5_degrees(self) -> None:
        gain = self.wet["theta_r_deg"] - self.dry["theta_r_deg"]
        self.assertGreaterEqual(gain, 5.0,
                                f"dry {self.dry['theta_r_deg']:.2f} wet "
                                f"{self.wet['theta_r_deg']:.2f}")
        # the cohesive heap must be quasi-static and well fitted
        self.assertGreater(self.wet["r2"], 0.5)
        self.assertLess(self.wet["kinetic_energy"], 5.0)
        self.assertGreater(self.wet["theta_r_deg"], 20.0)

    def test_pull_off_force_scales_with_bond_number(self) -> None:
        self.assertEqual(self.dry["pull_off"], 0.0)
        self.assertAlmostEqual(self.wet["pull_off"], 20.0, places=12)


if __name__ == "__main__":
    unittest.main()