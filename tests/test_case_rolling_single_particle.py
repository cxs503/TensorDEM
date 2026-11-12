"""Case-pool acceptance test: CASE-DEM-ROLLING-SINGLE-PARTICLE.

Single solid sphere rolling without slip down a plane inclined at 15 deg.
The case demands the classical rigid-body reference

    a = g sin(theta) / (1 + I/(m R^2)),      alpha = a / R   (harness sign convention: alpha = -a / R for v > 0)

to be reproduced within 2 % by the existing TensorDEM capability
:func:`tensordem.rolling_resistance.simulate_sphere_on_incline` (Cundall-Strack
tangential contact, semi-implicit Euler).  Two negative controls prove the gate
can fail: a friction coefficient below the pure-rolling requirement, and the
inertia-free ("sliding") reference, which differs from the correct one by
1 - 1/(1 + I/(mR^2)) ~ 29 %.

The full case (convergence + stiffness sweeps + rolling-resistance branch +
figures) is driven by ``benchmarks/case_rolling_single_particle/run_case.py``;
the pre-registered design is ``prereg/CASE-DEM-ROLLING-SINGLE-PARTICLE.json``.
"""

from __future__ import annotations

import math
import unittest

import torch

from tensordem.rolling_resistance import (
    RollingResistanceConfig,
    SphereInclineConfig,
    critical_incline_angle_deg,
    simulate_sphere_on_incline,
)

TOL = 0.02          # declared tolerance of the case (2 %)
THETA = 15.0        # deg


def _ls_slope(t: torch.Tensor, y: torch.Tensor) -> float:
    t_mean, y_mean = t.mean(), y.mean()
    dt = t - t_mean
    return float((dt * (y - y_mean)).sum() / (dt ** 2).sum())


def _run(**kwargs) -> dict:
    duration = kwargs.pop("duration", 0.12)
    settle = kwargs.pop("settle", 0.04)
    mu_r = kwargs.pop("mu_r", 0.0)
    theta = kwargs.pop("theta", THETA)
    config = SphereInclineConfig(**kwargs)
    res = simulate_sphere_on_incline(
        [theta], RollingResistanceConfig(model="cdt", mu_r=mu_r),
        config=config, duration=duration, settle_time=settle,
    )
    mask = res["steady_mask"]
    t = res["time"][mask]
    a = _ls_slope(t, res["v"][mask, 0])
    alpha = _ls_slope(t, res["omega"][mask, 0])
    slip = res["slip_velocity"][mask, 0]
    cfg = SphereInclineConfig(**kwargs)
    theta_rad = math.radians(theta)
    a_ref = cfg.gravity * (math.sin(theta_rad) - mu_r * math.cos(theta_rad)) / (
        1.0 + cfg.inertia_factor)
    return {
        # harness sign convention: v > 0 down-slope, omega > 0 counter-clockwise,
        # so rolling without slip (v + R omega = 0) has alpha = -a / R
        "a": a, "alpha": alpha, "a_ref": a_ref, "alpha_ref": -a_ref / cfg.radius,
        "slip_rms": float(slip.pow(2).mean().sqrt()),
        "inertia_factor": cfg.inertia_factor, "radius": cfg.radius,
        "gravity": cfg.gravity,
    }


class TestSingleParticleRolling(unittest.TestCase):
    """Acceptance of the case-pool anchor (see prereg for the design)."""

    def test_translational_and_angular_acceleration_within_2pct(self):
        out = _run()
        err_a = abs(out["a"] - out["a_ref"]) / out["a_ref"]
        err_alpha = abs(out["alpha"] - out["alpha_ref"]) / abs(out["alpha_ref"])
        self.assertLess(err_a, TOL, f"a={out['a']:.6f} vs ref={out['a_ref']:.6f} ({err_a:.2%})")
        self.assertLess(err_alpha, TOL,
                        f"alpha={out['alpha']:.4f} vs ref={out['alpha_ref']:.4f} ({err_alpha:.2%})")
        # kinematic closure: v + R omega = 0 in the mean
        closure = abs(out["a"] + out["radius"] * out["alpha"]) / abs(out["a"])
        self.assertLess(closure, TOL, f"no-slip closure violated: {closure:.2%}")

    def test_inertia_factor_is_solid_sphere(self):
        out = _run()
        self.assertAlmostEqual(out["inertia_factor"], 0.4, places=12)
        # the reference therefore carries the factor 1/1.4 = 0.7143 of g sin(theta)
        self.assertAlmostEqual(out["a_ref"] / (out["gravity"] * math.sin(math.radians(THETA))),
                               1.0 / 1.4, places=12)

    def test_negative_control_sliding_when_friction_too_low(self):
        """mu = 0.01 is far below the 0.051 pure-rolling requirement -> must slip."""
        out = _run(mu=0.01)
        err_a = abs(out["a"] - out["a_ref"]) / out["a_ref"]
        self.assertGreater(err_a, TOL, "gate failed to reject a slipping contact")
        self.assertGreater(out["slip_rms"], 1e-2)

    def test_negative_control_inertia_free_reference_is_rejected(self):
        """The 'sliding' reference g sin(theta) is 28.6 % off the correct one."""
        out = _run()
        wrong = out["gravity"] * math.sin(math.radians(THETA))
        self.assertGreater(abs(out["a"] - wrong) / wrong, TOL)

    def test_rolling_resistance_branch_matches_ai2011_relation(self):
        """mu_r > 0: a = g (sin t - mu_r cos t)/(1 + I/(mR^2)) (Ai et al. 2011 model)."""
        mu_r = 0.1
        out = _run(mu_r=mu_r)
        err_a = abs(out["a"] - out["a_ref"]) / out["a_ref"]
        self.assertLess(err_a, TOL, f"a={out['a']:.6f} vs ref={out['a_ref']:.6f}")
        # consistency with the critical-angle law theta_c = arctan(mu_r)
        theta_c = critical_incline_angle_deg(mu_r)
        self.assertAlmostEqual(theta_c, math.degrees(math.atan(mu_r)), places=12)
        self.assertGreater(THETA, theta_c)

    def test_zero_rolling_resistance_reduces_to_classical_formula(self):
        out0 = _run(mu_r=0.0)
        theta_rad = math.radians(THETA)
        self.assertAlmostEqual(out0["a_ref"],
                               out0["gravity"] * math.sin(theta_rad) / (1.0 + 0.4), places=12)


if __name__ == "__main__":
    unittest.main()