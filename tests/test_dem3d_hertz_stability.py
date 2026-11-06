"""Verification of the Hertz contact stability margin at ``recommended_dt``.

What is verified
----------------
``DEM3DConfig.recommended_dt`` is the step the engine actually uses (the
constructor rejects anything larger).  For a Hertz contact the bound comes from
the secant stiffness at the reference overlap
``delta_ref = hertz_stability_overlap_fraction * radius``, so the step is a
*suffice* condition for the many-body lattice:

    w(delta) = sqrt(k_sec(delta) / m),   dt_crit = 2 / w,   d = dt_rec * w.

This module pins three statements about that bound, all in terms of the exact
closed form of the scheme (see ``tensordem.stability``):

1. at every documented design overlap ``delta <= 0.2 R`` the recommended step
   sits far inside the stability region (``d <= 0.5``, i.e. a factor >= 4 below
   the SDOF limit) and the amplification factor is *exactly* marginal
   (``rho == 1``), because symplectic Euler has ``det A = 1`` for zero damping:
   inside the region the spectrum lies on the unit circle, not inside it;
2. the step-limit crossing is at ``d = 2`` (the parabola boundary of
   ``tensordem.stability``), verified by straddling ``dt = 2 / w``;
3. the design point keeps a large margin against that limit.

Correction note (why this file no longer asserts ``rho < 1``)
-------------------------------------------------------------
An earlier revision of this file asserted ``rho(dt_rec) = 0.32 ... 0.82`` at
0.01-0.2 R and ``rho ~ 1`` at 0.3 R.  Both claims are refuted by the closed
form documented in :mod:`tensordem.stability`: for an *undamped* symplectic
Euler step the spectral radius is ``1`` everywhere inside the stability region
(distinct roots on the unit circle) and jumps to ``>1`` only past ``d = 2``; it
never takes the intermediate values 0.32-0.82.  Likewise the crossing is not
near ``0.3 R``: reaching ``d = 2`` at ``delta = 0.3 R`` would need a step 39x
larger than ``recommended_dt``, and because ``k_sec ~ sqrt(delta)`` the
crossing overlap is ``O(10^6 R)`` for this step.  The file therefore verifies
the criterion the repository actually implements, and keeps the design-overlap
set of the original claim.
"""
from __future__ import annotations

import math

import pytest

from tensordem import stability
from tensordem.dem3d import DEM3DConfig

RADIUS = 0.025
DENSITY = 917.0
YOUNG_MODULUS = 1.0e9
POISSON_RATIO = 0.3
DESIGN_OVERLAPS = (0.01, 0.05, 0.10, 0.20)


def _hertz_secant_stiffness(overlap, young_modulus=YOUNG_MODULUS,
                            poisson_ratio=POISSON_RATIO, radius=RADIUS):
    """Secant stiffness ``F/delta`` of an equal-sphere Hertz contact."""
    e_star = young_modulus / (2.0 * (1.0 - poisson_ratio ** 2))
    r_star = radius / 2.0
    force = (4.0 / 3.0) * e_star * math.sqrt(r_star) * overlap ** 1.5
    return force / overlap if overlap > 0 else 0.0


def _particle_mass(radius=RADIUS, density=DENSITY):
    return density * (4.0 / 3.0) * math.pi * radius ** 3


def _config(**overrides) -> DEM3DConfig:
    """Hertz contact, no supports, no drag, no tool contact.

    ``bond_stiffness``/``contact_stiffness`` are left at their defaults rather
    than set to zero: ``DEM3DConfig`` deliberately rejects zero stiffness (a
    zero bond stiffness would make every bonded pair force-free while its
    contact is masked, i.e. a silently vacuous kernel).  For a Hertz law these
    two values do not enter the step bound at all -- ``_bound_stiffness``
    returns the Hertz secant stiffness -- so their value cannot affect the
    numbers below; ``test_recommended_dt_is_governed_by_the_hertz_branch``
    pins that independence.
    """
    kwargs = dict(
        nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY,
        contact_model="hertz", young_modulus=YOUNG_MODULUS,
        poisson_ratio=POISSON_RATIO, drag=0.0, fix_x_edges=False,
        tool_speed=1e-3, tool_gap=1.0,
    )
    kwargs.update(overrides)
    return DEM3DConfig(**kwargs)


def _omega(overlap_frac: float) -> float:
    return math.sqrt(_hertz_secant_stiffness(overlap_frac * RADIUS) / _particle_mass())


class TestHertzStabilityMargin:
    """Stability margin of the Hertz contact at the recommended time step."""

    @pytest.mark.parametrize("overlap_frac", DESIGN_OVERLAPS)
    def test_recommended_step_is_well_inside_the_stability_region(self, overlap_frac):
        """``d = dt_rec * w <= 0.5`` and ``rho == 1`` (marginal, non-growing)."""
        cfg = _config()
        dt_rec = cfg.recommended_dt
        omega = _omega(overlap_frac)
        d = dt_rec * omega
        assert d <= 0.5, f"d={d:.4g} at overlap_frac={overlap_frac}: less than 4x margin"
        rho = stability.symplectic_euler_growth_factor(dt_rec, omega, 0.0)
        # Undamped symplectic Euler: det A = 1, so inside the region |rho| == 1.
        assert rho == pytest.approx(1.0, abs=1e-12)
        assert stability.symplectic_euler_stable(dt_rec, omega, 0.0)

    def test_design_point_margin(self):
        """At the design point (0.1 R) the step margin ``dt_crit/dt_rec - 1`` is large."""
        cfg = _config()
        dt_rec = cfg.recommended_dt
        omega = _omega(0.10)
        dt_crit = stability.elastic_timestep_limit(omega)
        margin = dt_crit / dt_rec - 1.0
        assert margin > 10.0, f"margin={margin:.3g} at design point, expected > 10"
        # ... and the amplification factor is marginal, not growing.
        assert stability.symplectic_euler_growth_factor(dt_rec, omega, 0.0) == pytest.approx(1.0, abs=1e-12)

    def test_instability_threshold_is_dt_equals_two_over_omega(self):
        """The crossing is at ``dt = 2 / w``: stable just below, unstable above."""
        omega = _omega(0.10)
        dt_crit = stability.elastic_timestep_limit(omega)
        assert dt_crit == pytest.approx(2.0 / omega, rel=1e-15)
        below = 0.999 * dt_crit
        above = 1.001 * dt_crit
        assert stability.symplectic_euler_growth_factor(below, omega, 0.0) == pytest.approx(1.0, abs=1e-9)
        assert stability.symplectic_euler_growth_factor(above, omega, 0.0) > 1.0
        # The recommended step is nowhere near the crossing for this contact.
        assert _config().recommended_dt < 0.1 * dt_crit

    def test_recommended_dt_is_governed_by_the_hertz_branch(self):
        """The bound uses the Hertz secant stiffness, not the linear stiffnesses."""
        reference = _config().recommended_dt
        for linear in (1.0, 1_000.0, 100_000.0):
            cfg = _config(bond_stiffness=linear, contact_stiffness=linear)
            if linear < _hertz_secant_stiffness(
                    cfg.hertz_stability_overlap_fraction * RADIUS):
                assert cfg.recommended_dt == pytest.approx(reference, rel=1e-15)
        # The Hertz secant value is what the bound is built from, for both the
        # particle-particle and the rigid-counterpart convention.
        cfg = _config()
        delta_ref = cfg.hertz_stability_overlap_fraction * RADIUS
        expected = 0.15 * math.sqrt(
            cfg.mass / ((2 * min(cfg.nx * cfg.ny * cfg.nz - 1, 26) + 1)
                        * _hertz_secant_stiffness(delta_ref))
        )
        assert cfg.recommended_dt <= expected * (1.0 + 1e-12)