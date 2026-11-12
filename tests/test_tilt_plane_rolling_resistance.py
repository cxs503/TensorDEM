"""Rolling-resistance branch of the tilt-plane surface (was NOT_SUPPORTED).

The tilt-plane benchmark exposed ``mu_s`` / ``mu_k`` only: ``build_case`` wired
:class:`FrictionPolyhedronConfig` with ``mu_r = k_r = 0``, so the production
wall law's capped EPSD rolling moment (Ai et al. 2011) was unreachable from the
tilt-plane path.  ``docs/CASE_DEM_ROLLING_SINGLE_PARTICLE.md`` recorded that
branch as **NOT_SUPPORTED** and drew the analytic family only as a dashed
reference.  ``TiltPlaneCase.mu_rolling`` / ``k_r`` now wire it through.

Closed-form anchor (moment balance about the contact, no-slip ``a = alpha R``,
capped moment ``M_r = mu_r R F_n``)::

    a = g (sin(theta) - mu_r cos(theta)) / (1 + I/(m R^2))
      = (5/7) g (sin(theta) - mu_r cos(theta))          (solid sphere)

so ``mu_r = 0`` reproduces the pure-rolling value ``(5/7) g sin`` (the
``tan(theta) = 3.5 mu`` limit of the module docstring) and ``tan(theta) <= mu_r``
must hold the sphere.

Predeclared acceptance thresholds (not relaxed after the fact):

* ``REL_ERR_LIMIT_PCT = 3.0`` -- measured vs analytic rolling acceleration;
* ``HOLD_TOLERANCE_RATIO = 1.0`` -- a held sphere may not move more than one
  slide tolerance (``0.01`` particle extent) over the run.
"""
from __future__ import annotations

import math

import pytest

from tensordem import tilt_plane_friction as tp

REL_ERR_LIMIT_PCT = 3.0
HOLD_TOLERANCE_RATIO = 1.0
T_END = 0.25
SPHERE = dict(shape="sphere", rotation_locked=False, mu_s=0.6,
              size=0.006, density=1500.0, t_end=T_END, samples=120)


def _case(**kwargs) -> tp.TiltPlaneCase:
    base = dict(SPHERE)
    base.update(kwargs)
    return tp.TiltPlaneCase(**base)


# ---------------------------------------------------------------------------
# 1. The closed form itself (independent of the solver)
# ---------------------------------------------------------------------------
def test_analytic_reduces_to_pure_rolling_without_resistance():
    anchor = tp.rolling_acceleration_analytic(_case(mu_rolling=0.0), 15.0)
    expected = 5.0 / 7.0 * 9.81 * math.sin(math.radians(15.0))
    assert anchor["a_m_s2"] == pytest.approx(expected, rel=1e-12)
    assert anchor["inertia_ratio_I_over_mR2"] == pytest.approx(2.0 / 5.0)


def test_analytic_hold_verdict_and_monotonicity():
    hold = tp.rolling_acceleration_analytic(_case(mu_rolling=0.25), 10.0)
    assert hold["rolling_hold"] is True
    assert hold["a_m_s2"] < 0.0
    # tan(theta) > mu_r -> rolls, and the acceleration falls with mu_r.
    values = [tp.rolling_acceleration_analytic(
        _case(mu_rolling=mu_r), 15.0)["a_m_s2"] for mu_r in (0.0, 0.02, 0.05, 0.10)]
    assert values == sorted(values, reverse=True)


def test_cube_case_has_no_rolling_anchor():
    """The closed form is a *rolling sphere* result; a cube must report None."""
    r = tp.simulate(tp.TiltPlaneCase(mu_s=0.6, rotation_locked=True,
                                    t_end=0.02, samples=20), 15.0)
    assert r["rolling_anchor"] is None
    assert r["rolling_moment_on"] is False


# ---------------------------------------------------------------------------
# 2. Default is unchanged: rolling moment stays identically zero
# ---------------------------------------------------------------------------
def test_default_case_keeps_rolling_moment_disabled():
    case = _case()
    assert case.mu_rolling == 0.0
    assert case.effective_k_r == 0.0
    assert case.rolling_active is False
    r = tp.simulate(case, 15.0)
    assert r["rolling_moment_on"] is False
    # Pure rolling: measured acceleration must be (5/7) g sin(theta).
    rel = abs(r["acceleration_m_s2"] / r["rolling_anchor"]["a_m_s2"] - 1.0)
    assert rel * 100.0 < REL_ERR_LIMIT_PCT


# ---------------------------------------------------------------------------
# 3. The new branch against the closed form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mu_r", [0.02, 0.05])
def test_measured_rolling_acceleration_matches_analytic(mu_r: float):
    case = _case(mu_rolling=mu_r)
    assert case.effective_k_r == pytest.approx(tp.ROLLING_SPRING_DEFAULT)
    r = tp.simulate(case, 15.0)
    assert r["rolling_moment_on"] is True
    got = r["acceleration_m_s2"]
    want = r["rolling_anchor"]["a_m_s2"]
    rel_err_pct = abs(got / want - 1.0) * 100.0
    assert rel_err_pct < REL_ERR_LIMIT_PCT, (got, want, rel_err_pct)
    # Roll without gross slip: the wall slip speed stays far below the COM speed.
    assert r["slip_speed_m_s"] < 0.05 * max(r["com_speed_m_s"], 1e-12)


def test_measured_rolling_acceleration_at_second_angle():
    case = _case(mu_rolling=0.03)
    r = tp.simulate(case, 25.0)
    rel_err_pct = abs(r["acceleration_m_s2"] / r["rolling_anchor"]["a_m_s2"] - 1.0) * 100.0
    assert rel_err_pct < REL_ERR_LIMIT_PCT


def test_rolling_resistance_holds_the_sphere():
    """tan(10 deg) = 0.176 < mu_r = 0.25: the sphere must not run away."""
    case = _case(mu_rolling=0.25)
    r = tp.simulate(case, 10.0)
    assert r["rolling_anchor"]["rolling_hold"] is True
    assert r["moved"] is False
    assert abs(r["slide_m"]) <= HOLD_TOLERANCE_RATIO * case.slide_tolerance


# ---------------------------------------------------------------------------
# 4. The moment actually enters the contact law (not just the report)
# ---------------------------------------------------------------------------
def test_rolling_moment_is_applied_by_the_wall_law():
    from tensordem.friction_polyhedron import (
        FrictionPolyhedronConfig, compute_wall_contact_force)

    normal = tp.plane_basis(15.0)[0]
    for mu_r, expect_nonzero in ((0.0, False), (0.02, True)):
        case = _case(mu_rolling=mu_r)
        system, particles, normal, _ = tp.build_case(case, 15.0)
        particle = particles[0]
        # Spin the sphere about the transverse axis at a rolling angular rate.
        rate = 1.0
        particle.omega = rate * (normal * 0.0 + tp.plane_basis(15.0)[1]) * 0.0
        particle.omega = 1e-3 * tp.plane_basis(15.0)[1]
        state = system.wall_states[(0, 0)]
        result, contacting = compute_wall_contact_force(
            particle.shape, particle.v, particle.omega, particle.mass,
            particle.inertia, normal, 0.0, state, system.config, 1.0e-5)
        assert contacting
        if expect_nonzero:
            assert result["m_r"] > 0.0
            limit = system.config.mu_r * case.size * result["f_n"]
            assert result["m_r"] <= limit * (1.0 + 1e-9)
        else:
            assert result["m_r"] == 0.0