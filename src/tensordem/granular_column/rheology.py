"""Steady-shear mu(I) calibration on the shared servo shear cell.

The measurement itself is *not* re-implemented here: it is the constant-stress
servo shear cell of :mod:`tensordem.mu_i_rheology`
(:class:`~tensordem.mu_i_rheology.ServoShearCase`,
:func:`~tensordem.mu_i_rheology.simulate_servo_shear`), which measures the wall
traction ``tau`` and the normal stress ``P`` on a plate sheared at an imposed
rate and reports the inertial number

.. math:: I = \\dot\\gamma d / \\sqrt{P / \\rho}.

This module only builds the sweep: for a target inertial number it sets
``gamma_dot = I sqrt(P/rho) / d`` and runs long enough for the shear stress to
reach a steady state.  Contact friction is set to the Jop et al. (2006) static
value ``mu_s = 0.36`` with rolling resistance off, so the low-``I`` limit of the
measured rheology is a *prediction* of the contact parameters rather than a
fitted quantity -- that is the content of criterion (2).
"""
from __future__ import annotations

import math

__all__ = ["ShearPoint", "run_shear_point", "INERTIAL_SWEEP", "JOP"]
from .scaling import JOP_I_0, JOP_MU_2, JOP_MU_S

JOP = {"mu_s": JOP_MU_S, "mu_2": JOP_MU_2, "i_0": JOP_I_0}

#: Campaign sweep of the inertial number.
INERTIAL_SWEEP = (1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 0.1, 0.2, 0.35, 0.5,
                  1.0, 2.0)


class ShearPoint:
    """Parameters of one stead-shear point at a target inertial number."""

    def __init__(self, inertial_number: float, *, radius: float = 0.005,
                 density: float = 2500.0, normal_stress: float = 2.0e3,
                 mu: float = JOP_MU_S, mu_rolling: float = 0.0,
                 restitution: float = 0.3, kn: float = 1.0e4,
                 shear_strain: float = 0.2, seed: int = 5279):
        self.inertial_number = float(inertial_number)
        self.radius = float(radius)
        self.density = float(density)
        self.normal_stress = float(normal_stress)
        self.mu = float(mu)
        self.mu_rolling = float(mu_rolling)
        self.restitution = float(restitution)
        self.kn = float(kn)
        self.shear_strain = float(shear_strain)
        self.seed = int(seed)

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def shear_rate(self) -> float:
        return self.inertial_number * math.sqrt(self.normal_stress / self.density) / self.diameter


def run_shear_point(point: ShearPoint) -> dict:
    """Run one shear point and return the measured ``(I, mu)`` plus diagnostics."""
    from ..mu_i_rheology import ServoShearCase, simulate_servo_shear

    case = ServoShearCase(
        radius=point.radius,
        density=point.density,
        mu=point.mu,
        mu_rolling=point.mu_rolling,
        restitution=point.restitution,
        kn=point.kn,
        gravity=0.0,
        normal_stress=point.normal_stress,
        shear_rate=point.shear_rate,
        inertial_number=None,
        shear_strain=point.shear_strain,
        settle_steps=2000,
        min_shear_steps=200,
        sample_stride=10,
        servo_stride=10,
        seed=point.seed,
    )
    result = simulate_servo_shear(case)
    keep = {k: v for k, v in result.items() if not isinstance(v, (list, dict))}
    keep["target_inertial_number"] = point.inertial_number
    keep["imposed_shear_rate"] = point.shear_rate
    keep["contact_mu"] = point.mu
    keep["mu_rolling"] = point.mu_rolling
    return keep
