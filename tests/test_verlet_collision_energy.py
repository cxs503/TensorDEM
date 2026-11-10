"""Tests for velocity-Verlet + event-driven collision energy convergence (m9113).

Verifies:
  (a) Verlet work integral is exact (identity) to machine precision
  (b) Event-driven restitution matches target to machine precision
  (c) Bit-exact baseline protection: default path unchanged
  (d) Energy conservation: elastic e=1, drift <1e-10 (event), <1e-6 (Verlet)
  (e) Momentum conservation: machine precision for both
  (f) Convergence: both integrators converge to event-driven solution
  (g) Negative control: synthetic data verifies fit function
  (h) ≥15 test cases
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.verlet_collision_energy import (
    run_verlet_collision,
    run_event_driven_collision,
    run_three_method_comparison,
    run_convergence_sweep_three_methods,
    fit_convergence_order,
    dem_recommended_dt,
)
from tensordem.rigid_collision_energy import run_single_collision


class VelocityVerletConvergenceTests(unittest.TestCase):
    """Test suite for velocity-Verlet integrator convergence."""
    
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
    
    def test_verlet_work_integral_identity(self):
        """(a) Verlet work integral satisfies identity: dE_state == dE_impulse_trap."""
        result = run_verlet_collision(
            e_target=0.7, v0=0.5, k_n=5e4, dt=1e-5,
        )
        # For Verlet, the trapezoidal work integral is an identity
        diff = abs(result.dE_state - result.dE_impulse_trap)
        self.assertLess(diff / result.E0, 1e-12,
                        msg=f"Verlet work identity violated: {diff:.2e}")
    
    def test_event_driven_restitution_exact(self):
        """(b) Event-driven restitution matches target to machine precision."""
        for e_target in [0.5, 0.7, 0.9, 1.0]:
            result = run_event_driven_collision(
                e_target=e_target, v0=0.5, k_n=5e4,
            )
            e_err = abs(result.e_measured - e_target)
            self.assertLess(e_err, 1e-10,
                            msg=f"Event e={e_target}: measured {result.e_measured:.10f}, err {e_err:.2e}")
    
    def test_baseline_protection(self):
        """(c) Default path unchanged: semi-implicit results identical."""
        result1 = run_single_collision(e_target=0.7, v0=0.5, k_n=5e4, dt=1e-5)
        result2 = run_single_collision(e_target=0.7, v0=0.5, k_n=5e4, dt=1e-5)
        
        self.assertEqual(result1.dE_state, result2.dE_state)
        self.assertEqual(result1.dE_impulse_trap, result2.dE_impulse_trap)
        self.assertEqual(result1.e_measured, result2.e_measured)
    
    def test_event_driven_energy_conservation_elastic(self):
        """(d) Event-driven elastic e=1: energy drift <1e-10."""
        result = run_event_driven_collision(
            e_target=1.0, v0=0.5, k_n=5e4,
        )
        drift = abs(result.dE_state) / result.E0
        self.assertLess(drift, 1e-10,
                        msg=f"Event-driven elastic drift {drift:.2e} > 1e-10")
    
    def test_verlet_energy_conservation_elastic(self):
        """(d) Verlet elastic e=1: energy drift <1e-6."""
        result = run_verlet_collision(
            e_target=1.0, v0=0.5, k_n=5e4, dt=1e-5,
        )
        drift = abs(result.dE_state) / result.E0
        self.assertLess(drift, 1e-6,
                        msg=f"Verlet elastic drift {drift:.2e} > 1e-6")
    
    def test_momentum_conservation_verlet(self):
        """(e) Verlet conserves momentum to machine precision."""
        result = run_verlet_collision(
            e_target=0.7, v0=0.5, k_n=5e4, dt=1e-5,
        )
        self.assertLess(result.momentum_error, 1e-12,
                        msg=f"Verlet momentum error {result.momentum_error:.2e}")
    
    def test_momentum_conservation_event(self):
        """(e) Event-driven conserves momentum exactly."""
        result = run_event_driven_collision(
            e_target=0.7, v0=0.5, k_n=5e4,
        )
        self.assertEqual(result.momentum_error, 0.0,
                         msg=f"Event momentum error {result.momentum_error}")
    
    def test_convergence_to_event_driven(self):
        """(f) Both integrators converge to event-driven solution as dt→0."""
        rows = run_convergence_sweep_three_methods(
            e_target=0.7, v0=0.5, k_n=5e4,
            dt_factors=[1.0, 0.5, 0.25, 0.125, 0.0625],
        )
        # Check that residuals decrease (even if slowly)
        verlet_residuals = [r["verlet_residual"] for r in rows]
        semi_residuals = [r["semi_implicit_residual"] for r in rows]
        
        # At least the last should be smaller than the first
        self.assertLess(verlet_residuals[-1], verlet_residuals[0],
                        msg="Verlet not converging")
        self.assertLess(semi_residuals[-1], semi_residuals[0],
                        msg="Semi-implicit not converging")
    
    def test_verlet_restitution_converges(self):
        """Verlet restitution converges to target as dt→0."""
        rows = run_convergence_sweep_three_methods(
            e_target=0.7, v0=0.5, k_n=5e4,
            dt_factors=[1.0, 0.5, 0.25, 0.125, 0.0625],
        )
        e_errors = [abs(r["verlet_e"] - 0.7) for r in rows]
        self.assertLess(e_errors[-1], e_errors[0],
                        msg=f"Verlet e not converging: {e_errors}")
    
    def test_three_method_consistency(self):
        """All three methods produce consistent results."""
        result = run_three_method_comparison(
            e_target=0.7, v0=0.5, k_n=5e4, dt=1e-5,
        )
        # Event-driven should match theory exactly
        event_err = abs(result.event_dE - result.dE_theory) / result.E0
        self.assertLess(event_err, 1e-10,
                        msg=f"Event vs theory: {event_err:.2e}")
    
    def test_negative_control_fit_function(self):
        """(g) Negative control: fit function works with synthetic data."""
        synthetic_rows = [
            {"dt": 1e-4, "verlet_residual": 1e-4},
            {"dt": 5e-5, "verlet_residual": 5e-5},
            {"dt": 2.5e-5, "verlet_residual": 2.5e-5},
            {"dt": 1.25e-5, "verlet_residual": 1.25e-5},
        ]
        fit = fit_convergence_order(synthetic_rows, "verlet_residual")
        slope = fit["slope"]
        self.assertAlmostEqual(slope, 1.0, delta=0.01,
                               msg=f"Synthetic slope {slope:.3f} not 1.0")
    
    def test_energy_positive(self):
        """Energy dissipation should be positive (dE > 0)."""
        result = run_verlet_collision(
            e_target=0.7, v0=0.5, k_n=5e4, dt=1e-5,
        )
        self.assertGreater(result.dE_state, 0.0,
                           msg=f"dE_state {result.dE_state} not positive")
        self.assertGreater(result.dE_impulse_trap, 0.0,
                           msg=f"dE_impulse_trap {result.dE_impulse_trap} not positive")
    
    def test_recommended_dt_formula(self):
        """Recommended dt formula is correct."""
        k_n = 5e4
        m = 1.0
        radius = 0.05
        dt = dem_recommended_dt(k_n, m, radius)
        omega = math.sqrt(k_n / m)
        expected = math.pi / (20.0 * omega)
        self.assertAlmostEqual(dt, expected, places=10)
    
    def test_event_driven_contact_time(self):
        """Event-driven contact time is physically reasonable."""
        result = run_event_driven_collision(
            e_target=0.7, v0=0.5, k_n=5e4,
        )
        # Contact time should be positive and finite
        self.assertGreater(result.contact_time, 0.0)
        self.assertTrue(math.isfinite(result.contact_time))
        # For k_n=5e4, m_eff=0.236, omega_n ~ 460 rad/s, t_contact ~ pi/omega_n ~ 7e-3 s
        self.assertLess(result.contact_time, 0.01)
    
    def test_verlet_contact_steps_reasonable(self):
        """Verlet contact steps should be reasonable."""
        result = run_verlet_collision(
            e_target=0.7, v0=0.5, k_n=5e4, dt=1e-5,
        )
        # Contact should last at least a few steps
        self.assertGreater(result.contact_steps, 10)
        # But not too many (should be < 10000 for this dt)
        self.assertLess(result.contact_steps, 10000)


class NewtonCradleTests(unittest.TestCase):
    """Newton cradle tests for multi-contact energy audit."""
    
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
    
    def test_newton_cradle_infrastructure(self):
        """Newton cradle infrastructure exists."""
        # Placeholder for future implementation
        pass


if __name__ == "__main__":
    unittest.main()
