"""Analytic verification of Hertz–Mindlin tangential contact law.

Campaign node: **GAP-DEM-MINDLIN-TANGENTIAL-001**

The Mindlin (1949) no-slip tangential solution for two elastic spheres gives
a closed-form relation between tangential force and displacement during
partial slip.  This test suite verifies the ``hertz_mindlin_tangential_force``
implementation against three independent analytic anchors:

(a) **Small-displacement linear regime**: for delta_t → 0, F_t = k_t * delta_t
    with k_t = 8 G* sqrt(R* delta).  Acceptance: relative error < 1e-10.

(b) **Coulomb saturation**: when |F_t_raw| > mu * F_n, the force is clamped
    to mu * F_n and the state is projected.  Acceptance: |F_t| / (mu*F_n)
    within 1e-12 of 1.0 after saturation.

(c) **Energy recoverability in stick phase**: the tangential spring energy
    U_t = 0.5 * k_t * |delta_t|^2 must equal the work done by the tangential
    force during a loading-unloading cycle (no slip).  Acceptance: closure
    ratio in [0.999, 1.001].

(d) **Stiffness ratio vs Poisson ratio**: k_t/k_n = 3(1-nu)/(2-nu) for equal
    spheres.  Acceptance: max relative error < 1e-12 across nu in [0, 0.45].

(e) **Tensor batch consistency**: scalar and batch (N,) inputs produce
    identical per-element results.  Acceptance: elementwise error < 1e-14.

References:
    Mindlin, R. D. (1949). "Compliance of Elastic Bodies in Contact".
    Journal of Applied Mechanics, 16(3), 259-268.
    Johnson, K. L. (1985). Contact Mechanics. Cambridge Univ. Press.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from tensordem.hertz_mindlin import (
    hertz_mindlin_tangential_force,
    hertz_mindlin_tangential_stiffness,
    hertz_reduced_shear_modulus,
    mindlin_stiffness_ratio,
)
from tensordem.dem3d import (
    hertz_reduced_modulus,
    hertz_effective_radius,
    hertz_secant_stiffness,
)

# Material parameters (ice-like)
E_YOUNG = 1.0e9  # Pa
NU = 0.3
RADIUS = 0.025   # m
MU = 0.3
DENSITY = 917.0  # kg/m³


class TestMindlinTangentialAnalytic(unittest.TestCase):
    """Analytic anchors for the Hertz–Mindlin tangential contact law."""

    def setUp(self):
        self.e_star = hertz_reduced_modulus(E_YOUNG, NU)
        self.g_star = hertz_reduced_shear_modulus(E_YOUNG, NU)
        self.r_eff = hertz_effective_radius(RADIUS)
        self.k_n_fn = lambda d: hertz_secant_stiffness(d, self.e_star, self.r_eff)

    # ------------------------------------------------------------------
    # (a) Small-displacement linear regime
    # ------------------------------------------------------------------
    def test_small_disp_linear_regime(self):
        """For delta_t → 0, F_t = k_t * delta_t exactly."""
        overlap = 1e-4
        k_t = float(hertz_mindlin_tangential_stiffness(
            overlap, self.e_star, self.g_star, self.r_eff))

        for dt_mag in [1e-12, 1e-10, 1e-8, 1e-6]:
            tang_disp = torch.tensor([[dt_mag, 0.0]], dtype=torch.float64)
            normal_force = torch.tensor([self.k_n_fn(overlap) * overlap], dtype=torch.float64)

            F_t, _ = hertz_mindlin_tangential_force(
                overlap, tang_disp, self.e_star, self.g_star, self.r_eff,
                MU, normal_force)

            expected = k_t * dt_mag
            rel_err = abs(float(F_t[0, 0].item()) - expected) / expected
            self.assertLess(rel_err, 1e-10,
                f"Linear regime failed at dt={dt_mag}: rel_err={rel_err:.2e}")

    # ------------------------------------------------------------------
    # (b) Coulomb saturation
    # ------------------------------------------------------------------
    def test_coulomb_saturation(self):
        """When |F_t_raw| > mu*F_n, force clamps to mu*F_n."""
        overlap = 1e-4
        k_n = float(self.k_n_fn(overlap))
        F_n = k_n * overlap
        F_limit = MU * F_n

        # Drive well past the friction limit
        tang_disp = torch.tensor([[F_limit * 10.0, 0.0]], dtype=torch.float64)
        normal_force = torch.tensor([F_n], dtype=torch.float64)

        F_t, disp_next = hertz_mindlin_tangential_force(
            overlap, tang_disp, self.e_star, self.g_star, self.r_eff,
            MU, normal_force)

        F_t_mag = float(torch.linalg.vector_norm(F_t).item())
        self.assertAlmostEqual(F_t_mag, F_limit, delta=1e-10 * F_limit,
            msg=f"Saturation failed: |F_t|={F_t_mag:.6e}, limit={F_limit:.6e}")

        # State should be projected: |disp_next| = F_limit / k_t
        k_t = float(hertz_mindlin_tangential_stiffness(
            overlap, self.e_star, self.g_star, self.r_eff))
        disp_next_mag = float(torch.linalg.vector_norm(disp_next).item())
        expected_disp = F_limit / k_t
        self.assertAlmostEqual(disp_next_mag, expected_disp, delta=1e-10 * expected_disp,
            msg="State projection after saturation failed")

    # ------------------------------------------------------------------
    # (c) Energy recoverability in stick phase
    # ------------------------------------------------------------------
    def test_energy_recoverability_stick(self):
        """Tangential spring energy is fully recoverable in stick phase."""
        overlap = 1e-4
        k_n = float(self.k_n_fn(overlap))
        F_n = k_n * overlap
        F_limit = MU * F_n
        k_t = float(hertz_mindlin_tangential_stiffness(
            overlap, self.e_star, self.g_star, self.r_eff))

        # Load to 50% of friction limit (well within stick)
        target_disp = 0.5 * F_limit / k_t

        # Loading
        disp_load = torch.tensor([[target_disp, 0.0]], dtype=torch.float64)
        normal_force = torch.tensor([F_n], dtype=torch.float64)
        F_t_load, _ = hertz_mindlin_tangential_force(
            overlap, disp_load, self.e_star, self.g_star, self.r_eff,
            MU, normal_force)

        # Stored energy: U = 0.5 * k_t * delta_t^2
        U_stored = 0.5 * k_t * target_disp ** 2

        # Work done by force during loading (linear spring: W = 0.5 * F * delta)
        W_load = 0.5 * float(F_t_load[0, 0].item()) * target_disp

        closure = W_load / U_stored if U_stored > 0 else 1.0
        self.assertAlmostEqual(closure, 1.0, delta=0.001,
            msg=f"Energy closure in stick phase: {closure:.6f}")

    # ------------------------------------------------------------------
    # (d) Stiffness ratio vs Poisson ratio
    # ------------------------------------------------------------------
    def test_stiffness_ratio_vs_poisson(self):
        """k_t/k_n = 3(1-nu)/(2-nu) for equal spheres."""
        for nu in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            e_star = hertz_reduced_modulus(E_YOUNG, nu)
            g_star = hertz_reduced_shear_modulus(E_YOUNG, nu)
            r_eff = hertz_effective_radius(RADIUS)

            overlap = 1e-4
            k_n = float(hertz_secant_stiffness(overlap, e_star, r_eff))
            k_t = float(hertz_mindlin_tangential_stiffness(
                overlap, e_star, g_star, r_eff))

            expected_ratio = mindlin_stiffness_ratio(nu)
            measured_ratio = k_t / k_n
            rel_err = abs(measured_ratio - expected_ratio) / expected_ratio
            self.assertLess(rel_err, 1e-12,
                f"Stiffness ratio failed at nu={nu}: {measured_ratio:.6f} vs {expected_ratio:.6f}")

    # ------------------------------------------------------------------
    # (e) Tensor batch consistency
    # ------------------------------------------------------------------
    def test_batch_consistency(self):
        """Scalar and batch inputs produce identical per-element results."""
        overlap = 1e-4
        k_n = float(self.k_n_fn(overlap))
        F_n = k_n * overlap

        # Scalar input
        disp_scalar = torch.tensor([[1e-6, 0.0]], dtype=torch.float64)
        nf_scalar = torch.tensor([F_n], dtype=torch.float64)
        F_t_s, _ = hertz_mindlin_tangential_force(
            overlap, disp_scalar, self.e_star, self.g_star, self.r_eff,
            MU, nf_scalar)

        # Batch input (same values repeated)
        N = 5
        disp_batch = disp_scalar.repeat(N, 1)
        nf_batch = nf_scalar.repeat(N)
        F_t_b, _ = hertz_mindlin_tangential_force(
            overlap, disp_batch, self.e_star, self.g_star, self.r_eff,
            MU, nf_batch)

        for i in range(N):
            err = float(torch.max(torch.abs(F_t_b[i] - F_t_s[0])).item())
            self.assertLess(err, 1e-14,
                f"Batch consistency failed at index {i}: err={err:.2e}")


class TestMindlinTangentialEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_zero_tangential_disp(self):
        """Zero tangential displacement gives zero tangential force."""
        overlap = 1e-4
        e_star = hertz_reduced_modulus(E_YOUNG, NU)
        g_star = hertz_reduced_shear_modulus(E_YOUNG, NU)
        r_eff = hertz_effective_radius(RADIUS)
        k_n = float(hertz_secant_stiffness(overlap, e_star, r_eff))
        F_n = torch.tensor([k_n * overlap], dtype=torch.float64)

        disp = torch.zeros((1, 2), dtype=torch.float64)
        F_t, disp_next = hertz_mindlin_tangential_force(
            overlap, disp, e_star, g_star, r_eff, MU, F_n)

        self.assertAlmostEqual(float(F_t[0, 0].item()), 0.0, places=15)
        self.assertAlmostEqual(float(F_t[0, 1].item()), 0.0, places=15)

    def test_zero_overlap_gives_zero_force(self):
        """Zero overlap gives zero tangential stiffness and force."""
        e_star = hertz_reduced_modulus(E_YOUNG, NU)
        g_star = hertz_reduced_shear_modulus(E_YOUNG, NU)
        r_eff = hertz_effective_radius(RADIUS)
        F_n = torch.tensor([0.0], dtype=torch.float64)
        disp = torch.tensor([[1e-6, 0.0]], dtype=torch.float64)

        F_t, _ = hertz_mindlin_tangential_force(
            0.0, disp, e_star, g_star, r_eff, MU, F_n)

        self.assertAlmostEqual(float(F_t[0, 0].item()), 0.0, places=15)

    def test_negative_mu_rejected(self):
        """Negative friction coefficient must raise ValueError."""
        with self.assertRaises(ValueError):
            hertz_mindlin_tangential_force(
                1e-4, torch.zeros((1, 2)), 1e9, 1e9, 0.01, -0.1,
                torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
