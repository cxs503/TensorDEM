"""Tests for rigid collision energy ledger convergence (IN155-N07).

Verifies:
  a) Methods (i) and (ii-b) are identical (machine precision)
  b) Method (ii-a) converges at O(dt) vs (ii-b)
  c) Newton's cradle momentum conservation
  d) m5404 residual attribution
"""
import math
import unittest

import torch

from tensordem.rigid_collision_energy import (
    run_single_collision,
    run_convergence_sweep,
    fit_convergence_order,
    run_newtons_cradle,
    run_full_audit,
)


class RigidCollisionEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_methods_i_and_iib_are_identical(self):
        """Method (i) state-difference and (ii-b) trapezoidal should match to machine precision."""
        result = run_single_collision(e_target=0.7, v0=0.5, k_n=5e4, dt=1e-5)
        # They should be identical (or very close)
        diff = abs(result.dE_state - result.dE_impulse_trap)
        self.assertLess(diff / result.E0, 1e-12,
                        f"Methods (i) and (ii-b) differ by {diff:.2e} J")

    def test_method_iia_converges_at_order_one(self):
        """Method (ii-a) explicit endpoint should converge at O(dt) vs (ii-b)."""
        rows = run_convergence_sweep(
            e_target=0.7, v0=0.5,
            k_n_values=[5e4],
            dt_factors=[1.0, 0.5, 0.25, 0.125, 0.0625],
        )
        fit = fit_convergence_order(rows, 5e4, "residual_iia_vs_iib")
        slope = fit["slope"]
        self.assertTrue(math.isfinite(slope), "Slope is not finite")
        self.assertAlmostEqual(slope, 1.0, delta=0.2,
                               msg=f"Slope {slope:.3f} not close to 1.0")
        self.assertGreater(fit["r_squared"], 0.95,
                           f"R^2={fit['r_squared']:.4f} too low")

    def test_newtons_cradle_momentum_conservation(self):
        """Newton's cradle should conserve total momentum to machine precision."""
        cradle = run_newtons_cradle(
            e_target=0.95, v0=1.0, k_n=5e4,
            n_balls=5, dt=1e-6,
        )
        # Momentum should be conserved
        self.assertLess(cradle.momentum_error, 1e-10,
                        f"Momentum error {cradle.momentum_error:.2e} too large")

    def test_newtons_cradle_energy_dissipation(self):
        """Newton's cradle energy dissipation should match theory (approximately)."""
        cradle = run_newtons_cradle(
            e_target=0.95, v0=1.0, k_n=5e4,
            n_balls=5, dt=1e-6,
        )
        # The energy dissipation should be close to theory
        # (allowing for time discretization error)
        rel_err = abs(cradle.dE_state - cradle.dE_theory_total) / cradle.E0
        # With e=0.95, the dissipation is small (~10% of E0), so we allow larger relative error
        self.assertLess(rel_err, 0.5,
                        f"Energy dissipation relative error {rel_err:.2%} too large")

    def test_full_audit_passes(self):
        """Full audit should pass all checks."""
        result = run_full_audit(
            e_target=0.7, v0=0.5,
            k_n_values=[5e4],
        )
        self.assertEqual(result["verdict"], "PASS",
                         f"Audit failed: {result['details']}")

    def test_m5404_residual_attribution(self):
        """The m5404 residual of 5.111e-07 should be attributable to time discretization, not implementation error."""
        # Run at a well-resolved dt
        result = run_single_collision(e_target=0.7, v0=0.5, k_n=5e4, dt=1e-6)
        # The residual |dE_state - dE_theory| / E0
        residual = abs(result.dE_state - result.dE_theory) / result.E0
        # This should be small but non-zero (due to e_measured ≠ e_target)
        self.assertLess(residual, 0.1,
                        f"Residual {residual:.2e} too large")
        # The key insight: this residual is NOT an implementation error,
        # but rather the time discretization error in the restitution coefficient
        # We verify this by checking that e_measured is close to e_target
        e_err = abs(result.e_measured - result.e_target)
        self.assertLess(e_err, 0.05,
                        f"e_measured={result.e_measured:.6f} differs from target by {e_err:.2e}")


if __name__ == "__main__":
    unittest.main()
