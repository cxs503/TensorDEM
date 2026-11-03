"""Regression guard for the finite-amplitude floe-heave verification (m2843).

Covers the item left open by ``docs/ice_floe_physics_validation.md`` §6
("the nonlinear (finite-amplitude) heave period is not verified").  The
reference is the closed-form potential-energy quadrature in
``benchmarks/floe_heave_finite_amplitude.py``; the code under test is the
repository force law ``tensordem.buoyancy_wet.compute_buoyancy_force`` driven by
the repository's symplectic-Euler update.  Each physics assertion comes with a
control (stubbed or linearised law) that must break it.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_benchmark():
    path = _REPO_ROOT / "benchmarks" / "floe_heave_finite_amplitude.py"
    spec = importlib.util.spec_from_file_location("floe_heave_finite_amplitude", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module      # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


H = _load_benchmark()
CFG = H.case_config()


class GeometryAndPotentialTests(unittest.TestCase):
    """The reference must be self-consistent before it judges anything else."""

    def test_submerged_area_matches_independent_closed_form(self):
        """Two independent derivations must agree: phi-substitution vs the direct primitive.

        ``int_{-R}^{u} 2 sqrt(R^2 - s^2) ds = u sqrt(R^2-u^2) + R^2 arcsin(u/R) + pi R^2/2``
        with ``u = -y`` (the free surface sits at relative height ``-y``) is derived
        without the ``s = R sin(phi)`` substitution used by the module and by this
        harness.  (A numerical Simpson reference is *not* used: the
        integrand has a square-root endpoint singularity and only converges at
        ~5e-5 with 20000 panels, which would look like a model discrepancy.)
        """
        R = H.RADIUS
        for y in (-0.9 * R, -0.5 * R, 0.0, 0.3 * R, 0.9 * R):
            # A_sub(centre height y) = area below the free surface = int_{-R}^{-y} 2 sqrt(R^2-s^2) ds
            u = -y
            independent = u * math.sqrt(R * R - u * u) + R * R * math.asin(u / R) + 0.5 * math.pi * R * R
            self.assertAlmostEqual(H.submerged_area(y) / independent, 1.0, places=12)

    def test_area_endpoints(self):
        R = H.RADIUS
        self.assertAlmostEqual(H.submerged_area(-R), math.pi * R * R, places=12)
        self.assertEqual(H.submerged_area(R), 0.0)

    def test_primitive_derivative_is_the_area(self):
        """d/dy [int A_sub] = A_sub -- catches the sign slip in the correction term."""
        for y in (-0.95, -0.4, 0.0, 0.35):
            step = 1e-7
            numeric = (H.area_primitive(y + step) - H.area_primitive(y - step)) / (2 * step)
            self.assertAlmostEqual(numeric / H.submerged_area(y), 1.0, places=7)

    def test_potential_gradient_is_minus_force(self):
        for y in (-0.45, -0.30, -0.10):
            step = 1e-7
            grad = (H.potential(y + step, -0.337767114588) - H.potential(y - step, -0.337767114588)) / (2 * step)
            force = H.net_force_reference(y)
            self.assertAlmostEqual(grad / -force, 1.0, places=5)


class ReferenceAgainstModuleTests(unittest.TestCase):
    """The closed-form reference and the module must agree where they overlap."""

    def setUp(self):
        self.y_eq, _, converged = H.solve_floating_equilibrium(
            H.RADIUS, H.THICKNESS, H.RHO_ICE, CFG)
        self.assertTrue(converged)
        self.y_eq_ref = H.equilibrium_reference()

    def test_equilibrium(self):
        self.assertAlmostEqual(self.y_eq, self.y_eq_ref, places=9)

    def test_heave_stiffness_matches_waterplane_closed_form(self):
        k_module, stable = H.heave_restoring_stiffness(H.RADIUS, H.THICKNESS, H.RHO_ICE, CFG)
        k_closed = H.RHO_WATER * H.GRAVITY * H.THICKNESS * H.waterplane_width(self.y_eq_ref)
        self.assertTrue(stable)
        self.assertAlmostEqual(k_module / k_closed, 1.0, places=8)

    def test_force_curve_matches_archimedes(self):
        y = torch.tensor([-0.45, -0.30, -0.10, 0.2], dtype=torch.float64)
        code = H.repo_net_force(y, CFG)
        for i in range(y.numel()):
            reference = H.net_force_reference(float(y[i]))
            self.assertAlmostEqual(float(code[i]) / reference, 1.0, places=9)

    def test_small_amplitude_limit_recovers_linear_period(self):
        k, _ = H.heave_restoring_stiffness(H.RADIUS, H.THICKNESS, H.RHO_ICE, CFG)
        t0 = H.linear_period(k)
        quad = H.period_and_mean_quadrature(self.y_eq + 2e-3 * H.RADIUS, self.y_eq)
        # shift scales as A^2: ~1e-6 at A/R = 2e-3, so 1e-4 is a real (not vacuous) bound
        self.assertLess(abs(quad["period"] / t0 - 1.0), 1e-4)


class FiniteAmplitudeTests(unittest.TestCase):
    """The item this milestone closes: nonlinear period and mean-draft shift."""

    @classmethod
    def setUpClass(cls):
        cls.y_eq, _, _ = H.solve_floating_equilibrium(H.RADIUS, H.THICKNESS, H.RHO_ICE, CFG)
        cls.k, _ = H.heave_restoring_stiffness(H.RADIUS, H.THICKNESS, H.RHO_ICE, CFG)
        cls.t0 = H.linear_period(cls.k)
        cls.ratios = [0.05, 0.30, -0.30]
        cls.quad = {r: H.period_and_mean_quadrature(cls.y_eq + r * H.RADIUS, cls.y_eq)
                    for r in cls.ratios}
        cls.sims = dict(zip(
            cls.ratios,
            H.simulate_heave_batch([cls.y_eq + r * H.RADIUS for r in cls.ratios],
                                   dt=2e-3, n_periods=3.0, cfg=CFG)))

    def test_quadrature_is_order_converged(self):
        for r in self.ratios:
            coarse = H.period_and_mean_quadrature(self.y_eq + r * H.RADIUS, self.y_eq, order=24)
            fine = H.period_and_mean_quadrature(self.y_eq + r * H.RADIUS, self.y_eq, order=64)
            self.assertLess(abs(coarse["period"] / fine["period"] - 1.0), 1e-8)

    def test_period_matches_simulation_at_all_amplitudes(self):
        for r in self.ratios:
            sim, quad = self.sims[r], self.quad[r]
            self.assertGreaterEqual(sim["n_turnings"], 4, f"A/R={r}: turning count")
            self.assertLess(abs(sim["period"] / quad["period"] - 1.0), 1e-5,
                            f"A/R={r}: sim {sim['period']} vs quadrature {quad['period']}")

    def test_nonlinear_period_shift_is_resolved_not_noise(self):
        """Period shift must exceed the discretisation error by a wide margin."""
        for r in (0.30, -0.30):
            shift = abs(self.quad[r]["period"] - self.t0) / self.t0
            self.assertGreater(shift, 3e-2 if r > 0 else 2e-2)
            self.assertLess(abs(self.sims[r]["period"] / self.quad[r]["period"] - 1.0), shift / 100.0)

    def test_amplitude_dependence_is_asymmetric(self):
        """Rising and sinking of equal magnitude give different periods (softening / stiffening)."""
        up = self.quad[0.30]["period"]
        down = self.quad[-0.30]["period"]
        self.assertGreater(up / self.t0 - 1.0, 1.05 * (down / self.t0 - 1.0))

    def test_mean_draft_rectification(self):
        """A symmetric release still settles at a mean draft below equilibrium."""
        for r in self.ratios:
            offset = self.quad[r]["mean_offset"]
            self.assertLess(offset, 0.0, f"A/R={r}: expected negative (rides lower)")
            self.assertLess(abs(self.sims[r]["mean_y"] - self.quad[r]["mean_y"]), 1e-4)

    def test_control_linearised_law_cannot_reproduce_the_quadrature(self):
        r = 0.30
        y0 = self.y_eq + r * H.RADIUS
        k_lin = H.RHO_WATER * H.GRAVITY * H.THICKNESS * H.waterplane_width(self.y_eq)
        sim = H.simulate_heave(y0, 2e-3, n_periods=3.0,
                               force_fn=lambda y: -k_lin * (y - self.y_eq))
        self.assertGreater(abs(sim["period"] / self.quad[r]["period"] - 1.0), 5e-2)

    def test_control_zero_buoyancy_has_no_oscillation(self):
        sim = H.simulate_heave(self.y_eq + 0.05 * H.RADIUS, 2e-3, n_periods=1.0,
                               force_fn=lambda y: torch.zeros_like(y))
        self.assertEqual(sim["n_turnings"], 0)
        self.assertTrue(math.isnan(sim["period"]))


if __name__ == "__main__":
    unittest.main()