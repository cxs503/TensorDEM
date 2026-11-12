"""Falsifiable tests for the tilted-plane Coulomb friction anchor.

Companion to :mod:`tensordem.tilt_plane_friction` (method and thresholds are
documented in ``docs/TILT_PLANE_COULOMB_ANCHOR.md``).

Everything here is a *measurement* driven through the production contact kernel
(``PolyhedronDEMSystem`` + ``compute_wall_contact_force``) and compared with the
closed forms

    tan(theta_c) = mu_s ,      a = g (sin(theta) - mu_k cos(theta)) ,

plus the two controls that make the anchor falsifiable:

* **dual-friction semantics** -- a single coefficient must reproduce itself
  bit-for-bit through the static/dynamic path, and a genuine pair
  ``(mu_s, mu_k) = (0.6, 0.4)`` must separate the static limit from the
  residual dynamic one by ``atan 0.6 - atan 0.4 = 9.16 deg`` (>= 2 deg);
* **free-rotation regime probe** -- a sphere on an incline rolls without
  slipping below ``tan(theta) = 3.5 mu`` and then *fakes* the apparent
  coefficient ``(2/7) tan(theta)``, which is why the anchor is measured with the
  rotational degree of freedom frozen.

The test profile is a reduced-size/faster configuration (``size = 0.01 m``,
``t_end = 0.10 s``) so the whole file runs in about a minute; the reported
campaign uses ``size = 0.006 m``, ``t_end = 0.30 s`` and three time-step levels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem import tilt_plane_friction as tp
from tensordem.friction_polyhedron import (
    ContactState,
    FrictionPolyhedronConfig,
    friction_cone_limit,
)

MU_LIST = (0.2, 0.4, 0.6)
TEST_SIZE = 0.01
TEST_T_END = 0.10


def fast_case(**kwargs) -> tp.TiltPlaneCase:
    """Reduced-size case used by the tests (documented in the module docstring)."""
    base = dict(size=TEST_SIZE, t_end=TEST_T_END)
    base.update(kwargs)
    return tp.TiltPlaneCase(**base)


# ---------------------------------------------------------------------------
# 1. Coulomb cone helper: legacy scalar path and the static/dynamic latch
# ---------------------------------------------------------------------------
def test_legacy_cone_is_untouched_by_the_dual_path():
    """mu_dynamic=None must return mu*N and never write the latch."""
    config = FrictionPolyhedronConfig(mu=0.6)
    state = ContactState()
    for force in (0.1, 0.59, 10.0):
        limit = friction_cone_limit(config, state, force, 1.0, 1.0)
        assert limit == pytest.approx(0.6)
    assert state.sliding is False


def test_dual_cone_drops_to_the_residual_and_latches():
    """Static cone mu_s, then latched residual mu_d, then re-stick at rest."""
    config = FrictionPolyhedronConfig(mu=0.6, mu_dynamic=0.4, slip_stick_speed=1e-4)
    state = ContactState()
    # inside the static cone -> mu_s
    assert friction_cone_limit(config, state, 0.5, 1.0, 0.1) == pytest.approx(0.6)
    assert state.sliding is False
    # elastic force exceeds the static cone -> latch + residual mu_d
    assert friction_cone_limit(config, state, 0.7, 1.0, 0.1) == pytest.approx(0.4)
    assert state.sliding is True
    # while sliding the cone stays at mu_d even when the force is small
    assert friction_cone_limit(config, state, 0.05, 1.0, 0.1) == pytest.approx(0.4)
    # the body stops -> re-stick, static cone again
    assert friction_cone_limit(config, state, 0.05, 1.0, 1e-6) == pytest.approx(0.6)
    assert state.sliding is False


def test_single_coefficient_reproduces_itself_through_the_dual_path():
    """mu_k == mu_s must be bit-for-bit the single-coefficient trajectory."""
    legacy = tp.simulate(fast_case(mu_s=0.4), 30.0)
    dual = tp.simulate(fast_case(mu_s=0.4, mu_k=0.4), 30.0)
    assert dual["slide_m"] == legacy["slide_m"]
    assert dual["acceleration_m_s2"] == legacy["acceleration_m_s2"]


# ---------------------------------------------------------------------------
# 2. Geometry and static load transfer
# ---------------------------------------------------------------------------
def test_plane_basis_is_orthonormal_and_downhill():
    for theta in (0.0, 12.5, 30.0, 60.0):
        normal, downslope = tp.plane_basis(theta)
        assert np.linalg.norm(normal) == pytest.approx(1.0)
        assert np.linalg.norm(downslope) == pytest.approx(1.0)
        assert float(np.dot(normal, downslope)) == pytest.approx(0.0)
        # gravity projected on the normal is -g cos(theta): the downhill sense
        assert float(np.dot(np.array([0.0, 0.0, -9.81]), downslope)) == pytest.approx(
            9.81 * math.sin(math.radians(theta)))


def test_static_load_transfer_is_cos_theta():
    run = tp.simulate(fast_case(mu_s=0.4), 25.0)
    assert run["normal_load_over_weight"] == pytest.approx(run["cos_theta"], rel=1e-9)
    assert run["wall_contacts"] == 1.0


# ---------------------------------------------------------------------------
# 3. Static anchor: the closed form brackets the measured first motion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mu", MU_LIST)
def test_static_first_motion_brackets_atan_mu(mu):
    """Below atan(mu) the block rests, above it slides; inversion within 5 %."""
    theta_c = math.degrees(math.atan(mu))
    margin = 0.4
    below = tp.simulate(fast_case(mu_s=mu), theta_c - margin)
    above = tp.simulate(fast_case(mu_s=mu), theta_c + margin)
    assert below["moved"] is False, (
        f"mu={mu}: block moved at {theta_c - margin:.2f} deg, below atan(mu)")
    assert above["moved"] is True, (
        f"mu={mu}: block did not move at {theta_c + margin:.2f} deg, above atan(mu)")
    # the bracket must be tight enough to bound mu within the predeclared 5 %
    for theta in (theta_c - margin, theta_c + margin):
        mu_bracket = math.tan(math.radians(theta))
        assert 100.0 * abs(mu_bracket - mu) / mu <= tp.REL_ERR_LIMIT_PCT


def test_critical_angle_bisection_recovers_mu_0p4():
    """A real bisection (generic bracket 0/60 deg) returns atan(0.4) within 5 %."""
    case = fast_case(mu_s=0.4)
    result = tp.critical_angle(case, iterations=8, tolerance_deg=0.3)
    assert result["status"] == "ok"
    assert result["theta_c_bracket"][0] <= result["theta_c_deg"] <= result["theta_c_bracket"][1]
    assert result["theta_c_bracket"][1] - result["theta_c_bracket"][0] <= 0.3
    rel_err = 100.0 * abs(result["mu_measured"] - 0.4) / 0.4
    assert rel_err <= tp.REL_ERR_LIMIT_PCT, f"mu_measured={result['mu_measured']:.4f}"


# ---------------------------------------------------------------------------
# 4. Dynamic anchor: acceleration inverts to the same coefficient
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mu", MU_LIST)
def test_dynamic_acceleration_inverts_to_mu(mu):
    theta = math.degrees(math.atan(mu)) + 12.0
    result = tp.dynamic_friction(fast_case(mu_s=mu), theta)
    assert result["acceleration_r2"] > 0.999
    assert result["mu_measured"] == pytest.approx(mu, rel=tp.REL_ERR_LIMIT_PCT / 100.0)
    analytic = 9.81 * (math.sin(math.radians(theta)) - mu * math.cos(math.radians(theta)))
    assert result["acceleration_m_s2"] == pytest.approx(analytic, rel=2e-3)


# ---------------------------------------------------------------------------
# 5. Static/dynamic distinguishability (the >= 2 deg requirement)
# ---------------------------------------------------------------------------
def test_dual_friction_separates_static_and_dynamic_angles():
    mu_s, mu_k = tp.DUAL_CASE
    theta_s = math.degrees(math.atan(mu_s))
    # (a) holds below the static limit ...
    rest = tp.simulate(fast_case(mu_s=mu_s, mu_k=mu_k), theta_s - 2.5)
    assert rest["moved"] is False
    # (b) ... slides above it, at the *dynamic* acceleration
    dynamic = tp.dynamic_friction(fast_case(mu_s=mu_s, mu_k=mu_k), theta_s + 6.0)
    assert dynamic["mu_measured"] == pytest.approx(mu_k, rel=tp.REL_ERR_LIMIT_PCT / 100.0)
    theta_d = math.degrees(math.atan(dynamic["mu_measured"]))
    # (c) the two friction angles are separated by the analytic amount
    separation = theta_s - theta_d
    assert separation >= tp.DUAL_MIN_SEPARATION_DEG, (
        f"static/dynamic not distinguishable: {separation:.2f} deg")
    assert separation == pytest.approx(
        math.degrees(math.atan(mu_s) - math.atan(mu_k)), abs=1.5)


# ---------------------------------------------------------------------------
# 6. Free-rotation regime probe: the rolling trap
# ---------------------------------------------------------------------------
def test_sphere_rolling_trap_at_low_tilt():
    """A free sphere below atan(3.5 mu) rolls: a = 5/7 g sin(theta), no slip."""
    theta = 20.0
    case = fast_case(mu_s=0.4, shape="sphere", rotation_locked=False)
    result = tp.dynamic_friction(case, theta)
    th = math.radians(theta)
    assert result["acceleration_m_s2"] == pytest.approx(
        5.0 / 7.0 * 9.81 * math.sin(th), rel=2e-3)
    # the naive inversion returns (2/7) tan(theta), not mu
    assert result["mu_measured"] == pytest.approx(
        math.tan(th) / 7.0 * 2.0, rel=5e-2)
    assert result["mu_measured"] < 0.5 * 0.4
    assert result["slip_speed_m_s"] < 2e-3


def test_sphere_above_rolling_limit_measures_true_mu():
    """Above atan(3.5 mu) the sphere slips grossly and the inversion is unbiased."""
    theta = 60.0
    case = fast_case(mu_s=0.4, shape="sphere", rotation_locked=False)
    result = tp.dynamic_friction(case, theta)
    assert result["mu_measured"] == pytest.approx(0.4, rel=tp.REL_ERR_LIMIT_PCT / 100.0)


# ---------------------------------------------------------------------------
# 7. Artifact integrity (results.json + sha256), when the campaign has run
# ---------------------------------------------------------------------------
def test_results_sha256_is_canonical():
    payload = {"b": [1, 2, 3], "a": {"x": 0.5}}
    same = {"a": {"x": 0.5}, "b": [1, 2, 3]}
    assert tp.results_sha256(payload) == tp.results_sha256(same)
    assert len(tp.results_sha256(payload)) == 64


def test_published_results_artifact_if_present():
    path = Path("artifacts/tilted_plane/results.json")
    if not path.exists():
        pytest.skip("campaign artifact not built in this checkout")
    payload = json.loads(path.read_text())
    recorded = payload.pop("sha256")
    assert tp.results_sha256(payload) == recorded
    for key in ("static_inversion", "dynamic_inversion", "convergence",
                "static_dynamic", "raft_static", "verdicts", "thresholds"):
        assert key in payload
    # the acceptance table must be complete and inside the predeclared limit
    assert len(payload["static_inversion"]) == len(MU_LIST) * len(tp.DT_LEVELS)
    for row in payload["static_inversion"]:
        assert row["status"] == "ok"
        assert row["rel_err_pct"] <= tp.REL_ERR_LIMIT_PCT
    assert payload["static_dynamic"]["separation_deg"] >= tp.DUAL_MIN_SEPARATION_DEG