"""Rigid-collision energy-ledger convergence certification tests.

Verifies:
1. Three energy-accounting口径 (state-difference / work-integral / restitution-theory)
2. dt/2^k convergence sweep (k=0..3)
3. Negative control: explicit work O(dt), trapezoidal work O(dt) [limited by force eval]
4. 5-ball Newton cradle momentum conservation + energy residual order
5. m5404 re-evaluation: residual 5.111e-07 attributed to口径 not implementation error
"""
from __future__ import annotations

import math
import unittest

from tensordem.rigid_collision_ledger import (
    CollisionConfig,
    LedgerCriterion,
    NewtonCradleResult,
    dt_sweep,
    observed_order,
    run_newton_cradle,
    run_single_collision,
)


class SingleCollisionCriterionTests(unittest.TestCase):
    """Test that all three energy口径 produce consistent results."""

    def test_criterion_produces_finite_values(self):
        cfg = CollisionConfig(impact_speed=1.0, dt=1e-5)
        result = run_single_collision(cfg)
        c = result.criterion
        self.assertTrue(math.isfinite(c.delta_e_j))
        self.assertTrue(math.isfinite(c.work_explicit_j))
        self.assertTrue(math.isfinite(c.work_trapezoidal_j))
        self.assertTrue(math.isfinite(c.d_theory_j))
        self.assertTrue(math.isfinite(c.residual_state_diff))
        self.assertTrue(math.isfinite(c.residual_work_explicit))
        self.assertTrue(math.isfinite(c.residual_work_trap))

    def test_restitution_measured_close_to_theory(self):
        """Measured restitution should be within 5% of theory for small dt."""
        cfg = CollisionConfig(impact_speed=1.0, dt=1e-5)
        result = run_single_collision(cfg)
        rel_err = abs(result.restitution_measured - result.restitution_theory) / result.restitution_theory
        self.assertLess(rel_err, 0.05,
                        f"restitution error {rel_err:.4g} exceeds 5%")

    def test_energy_dissipation_positive(self):
        """Theoretical dissipation must be positive for e < 1."""
        cfg = CollisionConfig(impact_speed=1.0, dt=1e-5)
        result = run_single_collision(cfg)
        self.assertGreater(result.criterion.d_theory_j, 0.0)

    def test_state_diff_residual_bounded(self):
        """State-difference residual should be small relative to energy scale."""
        cfg = CollisionConfig(impact_speed=1.0, dt=1e-5)
        result = run_single_collision(cfg)
        c = result.criterion
        scale = abs(c.d_theory_j) + abs(c.delta_e_j) + 1e-30
        self.assertLess(c.residual_state_diff / scale, 0.1,
                        "state-diff residual exceeds 10% of energy scale")

    def test_work_residuals_bounded(self):
        """Work integral residuals should be small relative to energy change."""
        cfg = CollisionConfig(impact_speed=1.0, dt=1e-5)
        result = run_single_collision(cfg)
        c = result.criterion
        scale = abs(c.delta_e_j) + 1e-30
        self.assertLess(c.residual_work_explicit / scale, 0.1,
                        "explicit work residual exceeds 10%")
        self.assertLess(c.residual_work_trap / scale, 0.1,
                        "trapezoidal work residual exceeds 10%")


class DtSweepConvergenceTests(unittest.TestCase):
    """dt/2^k convergence sweep: prove order of convergence."""

    @classmethod
    def setUpClass(cls):
        """Run sweep with k=0..3 (takes ~70s total)."""
        from tensordem.dem3d import DEM3DConfig
        cls.results = []
        base_cfg = CollisionConfig(impact_speed=1.0)
        base = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=base_cfg.radius, density=base_cfg.density,
            contact_stiffness=base_cfg.stiffness, contact_damping=base_cfg.damping,
            drag=0.0,
        )
        for k in range(4):
            dt_k = 0.5 * base.recommended_dt / (2 ** k)
            cfg = CollisionConfig(
                radius=base_cfg.radius, density=base_cfg.density,
                stiffness=base_cfg.stiffness, damping=base_cfg.damping,
                impact_speed=base_cfg.impact_speed, dt=dt_k,
            )
            cls.results.append(run_single_collision(cfg))

    def test_residuals_decrease_with_dt(self):
        """Residuals should generally decrease as dt is refined."""
        results = self.results
        # Check first three refinements (k=0,1,2)
        for key in ("residual_state_diff", "residual_work_explicit",
                     "residual_work_trap"):
            vals = [getattr(r.criterion, key) for r in results[:3]]
            for i in range(len(vals) - 1):
                if vals[i] > 0:
                    self.assertLessEqual(
                        vals[i + 1], vals[i] * 1.5,
                        f"{key} not decreasing: {vals[i]:.6e} -> {vals[i+1]:.6e}")

    def test_explicit_work_converges_first_order(self):
        """Explicit (left-Riemann) work integral converges at O(dt)."""
        results = self.results
        vals = [r.criterion.residual_work_explicit for r in results]
        dts = [r.criterion.dt for r in results]
        order = observed_order(vals, dts)
        self.assertAlmostEqual(order, 1.0, delta=0.3,
                               msg=f"explicit work order {order:.3f} != 1.0")

    def test_trapezoidal_work_converges(self):
        """Trapezoidal work integral converges (order >= 0.8).

        Note: In semi-implicit Euler, the force evaluation is O(dt), so the
        trapezoidal work integral is limited to O(dt) overall, not O(dt^2).
        The quadrature rule itself is O(dt^2), but the force error dominates.
        """
        results = self.results
        vals = [r.criterion.residual_work_trap for r in results]
        dts = [r.criterion.dt for r in results]
        order = observed_order(vals, dts)
        self.assertGreater(order, 0.8,
                           msg=f"trapezoidal work order {order:.3f} < 0.8")

    def test_state_diff_converges(self):
        """State-difference residual converges (order >= 0.5)."""
        results = self.results
        vals = [r.criterion.residual_state_diff for r in results]
        dts = [r.criterion.dt for r in results]
        order = observed_order(vals, dts)
        self.assertGreater(order, 0.5,
                           msg=f"state-diff order {order:.3f} < 0.5")

    def test_finest_residual_small(self):
        """At finest dt, state-diff residual should be < 0.1% of energy scale."""
        results = self.results
        finest = results[-1]
        c = finest.criterion
        scale = abs(c.d_theory_j) + abs(c.delta_e_j) + 1e-30
        self.assertLess(c.residual_state_diff / scale, 0.001,
                        f"finest state-diff relative error exceeds 0.1%")


class NewtonCradleTests(unittest.TestCase):
    """5-ball Newton cradle: momentum conservation + energy residual order."""

    def test_momentum_conserved(self):
        """Total momentum should be conserved to high precision."""
        # Use larger dt and fewer steps for faster test
        result = run_newton_cradle(n=5, v0=1.0, dt=5e-5, max_steps=10000)
        self.assertLess(result.momentum_error_pct, 1.0,
                        f"momentum error {result.momentum_error_pct:.4g}% > 1%")

    def test_collisions_detected(self):
        """Should detect at least 2 collision events in the cradle."""
        result = run_newton_cradle(n=5, v0=1.0, dt=5e-5, max_steps=10000)
        self.assertGreaterEqual(len(result.collisions), 2,
                                f"only {len(result.collisions)} collisions detected")


class M5404ReevaluationTests(unittest.TestCase):
    """Re-evaluate m5404: the 5.111e-07 residual is due to口径, not implementation error."""

    def test_residual_at_reference_dt(self):
        """At the reference dt used in m5404, residual should be small."""
        cfg = CollisionConfig(
            radius=0.025, density=917.0,
            stiffness=5_000.0, damping=5.0,
            impact_speed=1.0,
        )
        from tensordem.dem3d import DEM3DConfig
        base = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=cfg.radius, density=cfg.density,
            contact_stiffness=cfg.stiffness, contact_damping=cfg.damping,
            drag=0.0,
        )
        ref_dt = 0.5 * base.recommended_dt
        cfg.dt = ref_dt

        result = run_single_collision(cfg)
        c = result.criterion
        self.assertLess(c.residual_state_diff, 1e-4,
                        f"state-diff residual {c.residual_state_diff:.6e} > 1e-4")

    def test_residual_scales_with_dt(self):
        """Prove the residual scales with dt (not a fixed implementation error)."""
        cfg = CollisionConfig(impact_speed=1.0)
        from tensordem.dem3d import DEM3DConfig
        base = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=cfg.radius, density=cfg.density,
            contact_stiffness=cfg.stiffness, contact_damping=cfg.damping,
            drag=0.0,
        )
        ref_dt = 0.5 * base.recommended_dt

        residuals = []
        dts = []
        for k in range(3):
            dt_k = ref_dt / (2 ** k)
            cfg_k = CollisionConfig(
                radius=cfg.radius, density=cfg.density,
                stiffness=cfg.stiffness, damping=cfg.damping,
                impact_speed=cfg.impact_speed, dt=dt_k,
            )
            r = run_single_collision(cfg_k)
            residuals.append(r.criterion.residual_state_diff)
            dts.append(dt_k)

        order = observed_order(residuals, dts)
        self.assertTrue(math.isfinite(order), "residual order is NaN")
        self.assertGreater(order, 0.3,
                           f"residual order {order:.3f} < 0.3: not dt-dependent")


if __name__ == "__main__":
    unittest.main()
