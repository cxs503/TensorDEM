"""Case-pool test: DEM-ROLLING-SINGLE-PARTICLE (TensorDEM tilt plane).

Pool item ``DEM-ROLLING-SINGLE-PARTICLE`` (ref: classical rigid-body dynamics
``a = g sin(theta) / (1 + I/(m R^2)) = 5/7 g sin(theta)`` for a homogeneous
sphere; Ai et al. 2011 Powder Technol. 206:269 for the rolling-resistance
branch).  Acceptance: ``< 2%`` on the centre-of-mass acceleration and the
angular acceleration at ``theta = 15 deg`` without slip.

The measurements here are deliberately the *fast* variant of the benchmark
(``benchmarks/dem_rolling_single_particle.py``: ``size = 2 mm``,
``t_end = 0.06 s`` -- about 15 s per run on one core) so that the pool case stays
inside a normal test-suite budget; the benchmark script carries the full
``a(theta)`` curve and the slip-onset scan.

Scope note (honest): the tilt-plane engine (``TiltPlaneCase``) has no
rolling-resistance coefficient, so the ``mu_r`` branch of the pool reference is
NOT exercised here -- it is flagged NOT_SUPPORTED on this code path and only the
ideal no-slip rolling law + slip onset are certified.
"""
from __future__ import annotations

import math

import pytest

import tensordem.tilt_plane_friction as tp

GRAVITY = 9.81
RADIUS = 0.002
T_END = 0.06
MU_S = 0.4
#: No-slip limit for a rolling sphere: tan(theta) < 3.5 mu_s.
LIMIT_DEG = math.degrees(math.atan(3.5 * MU_S))


def _case(mu_s: float = MU_S) -> tp.TiltPlaneCase:
    return tp.TiltPlaneCase(shape="sphere", rotation_locked=False, mu_s=mu_s,
                            size=RADIUS, density=1500.0, t_end=T_END)


def _analytic(theta_deg: float, mu_r: float = 0.0) -> float:
    th = math.radians(theta_deg)
    return 5.0 / 7.0 * GRAVITY * (math.sin(th) - mu_r * math.cos(th))


# ---------------------------------------------------------------------------
# 1. Acceptance angle of the pool item
# ---------------------------------------------------------------------------

def test_acceptance_angle_15deg_matches_five_sevenths_law():
    """a(15 deg) vs 5/7 g sin(15 deg), acceptance < 2%."""
    theta = 15.0
    result = tp.dynamic_friction(_case(), theta)
    a_dem = result["acceleration_m_s2"]
    a_ref = _analytic(theta)
    assert a_dem == pytest.approx(a_ref, rel=0.02)
    # the fit is a straight line: free rolling, no stick-slip
    assert result["acceleration_r2"] == pytest.approx(1.0, abs=1e-6)
    # and it really is rolling, not sliding: slip speed stays negligible
    assert result["slip_speed_m_s"] < 1e-3


def test_angular_acceleration_is_a_over_radius():
    """alpha = a / R to < 2% (equivalently: no-slip rolling kinematics)."""
    theta = 15.0
    result = tp.dynamic_friction(_case(), theta)
    alpha_dem = result["acceleration_m_s2"] / RADIUS
    alpha_ref = _analytic(theta) / RADIUS
    assert alpha_dem == pytest.approx(alpha_ref, rel=0.02)


# ---------------------------------------------------------------------------
# 2. The measured mu is the naive-inversion artefact, not the contact mu
# ---------------------------------------------------------------------------

def test_naive_inversion_returns_two_sevenths_of_tan():
    """The pool case must not be mistaken for a friction calibration.

    A free-rolling sphere reports ``mu_measured = (2/7) tan(theta)``, i.e. a
    seventh-slope artefact far below the contact ``mu_s``; locking the rotation
    is what measures sliding friction (see ``test_tilt_plane_coulomb_anchor``).
    """
    theta = 20.0
    result = tp.dynamic_friction(_case(), theta)
    assert result["mu_measured"] == pytest.approx(
        math.tan(math.radians(theta)) * 2.0 / 7.0, rel=5e-2)
    assert result["mu_measured"] < 0.5 * MU_S


# ---------------------------------------------------------------------------
# 3. Slip onset at the no-slip limit atan(3.5 mu_s)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mu_s", [0.2, 0.4])
def test_slip_onset_brackets_the_roll_slip_limit(mu_s):
    """Below ``atan(3.5 mu_s)`` the sphere rolls; above it, gravity wins."""
    limit = math.degrees(math.atan(3.5 * mu_s))
    below = tp.dynamic_friction(_case(mu_s), max(1.0, limit - 3.0))
    above = tp.dynamic_friction(_case(mu_s), min(70.0, limit + 3.0))

    # rolling branch reproduces the analytic law
    assert below["acceleration_m_s2"] == pytest.approx(
        _analytic(max(1.0, limit - 3.0)), rel=0.02)
    # the threshold itself is where the two branches cross: the DEM must place
    # the transition inside the bracket, i.e. slip appears only just above it
    assert below["slip_speed_m_s"] < 1e-2
    assert above["slip_speed_m_s"] > below["slip_speed_m_s"] * 10.0