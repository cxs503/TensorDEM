"""Tests for the DEFECT-D3 repair: high-loading closure in the open-water BEM.

The closure is a *construction*, not a formula transplant: the high-loading
branch is fixed by the same three conditions Glauert's empirical actuator-disk
parabola uses (value and slope continuity with the classical momentum relation
at the breakdown induction, and ``CT(a = 1) = 2``).  Applied in the
wind-turbine convention the construction has to reproduce Buhl's published
coefficients exactly -- that identity is the anchor that says the method is
implemented correctly -- and it is then applied in *this* module's convention.

The tests also pin the defect that motivated the repair (a cap-limited state
claiming convergence while violating its own momentum identity) so the
pre-repair code cannot pass them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.propeller_ice.geometry import PropellerGeometry
from tensordem.propeller_ice.hydro import (
    OpenWaterConfig,
    OpenWaterState,
    _buhl,
    classical_induction,
    empirical_ct,
    empirical_ct_coefficients,
    empirical_ct_max,
    empirical_induction,
    high_loading_branch_end,
    momentum_breakdown_induction,
    solve_open_water,
)

ARTIFACT = (Path(__file__).resolve().parents[1]
            / "artifacts" / "open_water_high_induction" / "verification.json")

LIB_GEOM = dict(diameter=0.6, n_blades=4, hub_ratio=0.35, pitch_ratio=1.0,
                chord_ratio=0.40)
DEFECT_GEOM = dict(diameter=3.0, n_blades=4, hub_ratio=0.40, pitch_ratio=0.95,
                   chord_ratio=0.539, skew_tip_deg=0.0, thickness_ratio=0.05)


# --------------------------------------------------------------- algebra ---
def test_breakdown_induction_is_a_s_equals_0p2():
    """4a(1+a) = 0.96 has the exact root a_s = 0.2 in the propeller convention."""
    a_s = momentum_breakdown_induction(0.96)
    assert a_s == pytest.approx(0.2, abs=1e-14)
    assert 4.0 * a_s * (1.0 + a_s) == pytest.approx(0.96, rel=1e-15)


def test_classical_relation_inverts_exactly():
    for F in (0.25, 0.5, 0.75, 1.0):
        ct = np.linspace(-0.5 * F, 0.9 * F, 25)
        a = classical_induction(ct, np.full(ct.shape, F))
        assert np.max(np.abs(4.0 * F * a * (1.0 + a) - ct)) < 1e-14


@pytest.mark.parametrize("F", [0.3, 0.5, 0.7, 1.0])
def test_construction_reproduces_buhl_published_coefficients(F: float):
    """The 3-condition construction in the turbine convention == Buhl 2005."""
    c = empirical_ct_coefficients(F, 0.96, base="turbine")
    assert c[0] == pytest.approx(8.0 / 9.0, rel=1e-12)
    assert c[1] == pytest.approx(4.0 * F - 40.0 / 9.0, rel=1e-12)
    assert c[2] == pytest.approx(50.0 / 9.0 - 4.0 * F, rel=1e-12)


@pytest.mark.parametrize("F", [0.3, 0.5, 0.8, 1.0])
def test_closure_is_c1_continuous_at_the_breakdown_induction(F: float):
    a_s = momentum_breakdown_induction(0.96)
    c0, c1, c2 = empirical_ct_coefficients(F, 0.96, base="propeller")
    value = c0 + c1 * a_s + c2 * a_s ** 2
    slope = c1 + 2.0 * c2 * a_s
    assert value == pytest.approx(4.0 * F * a_s * (1.0 + a_s), rel=1e-12)
    assert slope == pytest.approx(4.0 * F * (1.0 + 2.0 * a_s), rel=1e-12)


@pytest.mark.parametrize("F", [0.4, 0.7, 1.0])
def test_closure_hits_the_glauert_high_loading_anchor_at_a_equal_one(F: float):
    """CT(a=1) = 2 is F-independent, as in the published relation."""
    assert float(empirical_ct(np.array([1.0]), F)[0]) == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize("F", [0.4, 0.7, 1.0])
def test_closure_inverse_is_exact_on_the_monotone_segment(F: float):
    a_hi = high_loading_branch_end(F, 0.96, 0.85)
    lo = 0.96 * F
    hi = float(empirical_ct(np.array([a_hi]), F)[0])
    ct = np.linspace(lo * 1.001, hi * 0.999, 17)
    a, exceeded = empirical_induction(ct, np.full(ct.shape, F), 0.96, 0.85)
    assert not exceeded.any()
    assert np.max(np.abs(empirical_ct(a, F) - ct)) < 1e-12
    assert np.all(np.diff(a) > 0.0)  # monotone in CT


def test_closure_reports_the_branch_end_when_loading_is_unrepresentable():
    F = 1.0
    a_hi = high_loading_branch_end(F, 0.96, 0.85)
    ct_max = empirical_ct_max(F, 0.96, 0.85)
    a, exceeded = empirical_induction(np.array([ct_max * 1.5]), np.array([F]), 0.96, 0.85)
    assert bool(exceeded[0])
    assert float(a[0]) == pytest.approx(a_hi, rel=1e-12)


def test_config_rejects_unknown_induction_model():
    with pytest.raises(ValueError):
        OpenWaterConfig(induction_model="buhl")
    with pytest.raises(ValueError):
        OpenWaterConfig(ct_limit=1.5)


# ----------------------------------------------------------------- solver ---
def test_legacy_mode_reproduces_the_registered_defect():
    """Repair guard: the pre-repair mode converges by a metric blind to momentum.

    On the geometry the defect was registered against, every strip sits on the
    induction cap while the blade-element thrust is unbalanced by the momentum
    side by per-cent amounts -- so this test fails on the pre-repair code.
    """
    geom = PropellerGeometry(**DEFECT_GEOM)
    state = solve_open_water(geom, 1.0, 0.2 * geom.diameter,
                             config=OpenWaterConfig(n_strips=24,
                                                    induction_model="clipped"))
    a = np.asarray(state.a)
    assert int(np.sum(a >= 0.849)) == a.size
    assert state.residual < 1.0e-6                    # "converged" by its own metric
    assert state.momentum_residual > 1.0e-2           # ~7 % of the thrust unbalance
    assert state.ct_max_loading > 10.0                # 19x above the closure bound
    assert not state.model_valid
    assert state.validity_flag == "thrust_closure_exceeded"


def test_default_mode_reports_the_invalid_operating_point_instead_of_hiding_it():
    geom = PropellerGeometry(**DEFECT_GEOM)
    state = solve_open_water(geom, 1.0, 0.2 * geom.diameter,
                             config=OpenWaterConfig(n_strips=24))
    assert state.induction_model == "classical"
    assert not state.model_valid
    assert state.n_strips_above_ct_limit > 0
    assert state.momentum_residual > 1.0e-2
    assert state.converged          # it does converge; validity is a separate question
    assert "closure" in state.validity_flag


def test_default_matches_legacy_coefficients_where_the_model_is_valid():
    geom = PropellerGeometry(**LIB_GEOM)
    for J in (0.6, 0.8, 1.0):
        old = solve_open_water(geom, 1.0, J * geom.diameter,
                               config=OpenWaterConfig(induction_model="clipped"))
        new = solve_open_water(geom, 1.0, J * geom.diameter)
        assert new.model_valid and old.model_valid
        assert new.thrust_coefficient == pytest.approx(old.thrust_coefficient, rel=1e-6)
        assert new.torque_coefficient == pytest.approx(old.torque_coefficient, rel=1e-6)


def test_high_loading_state_satisfies_its_own_momentum_identity():
    """Where the closure has a root, blade element and momentum agree to round-off."""
    geom = PropellerGeometry(**LIB_GEOM)
    checked = 0
    for J in (0.6, 0.8):
        state = solve_open_water(geom, 1.0, J * geom.diameter,
                                 config=OpenWaterConfig(induction_model="high_loading"))
        assert state.model_valid and state.converged
        assert state.momentum_residual < 1.0e-6
        checked += 1
    assert checked == 2


def test_validity_boundary_moves_with_geometry():
    """A low-solidity propeller stays inside the model far below J = 0.6."""
    lean = PropellerGeometry(diameter=0.6, n_blades=4, hub_ratio=0.35, pitch_ratio=1.0,
                             chord_ratio=0.12)
    fat = PropellerGeometry(**LIB_GEOM)
    for J in (0.5, 0.55):
        lean_state = solve_open_water(lean, 1.0, J * lean.diameter)
        fat_state = solve_open_water(fat, 1.0, J * fat.diameter)
        assert lean_state.ct_max_loading < fat_state.ct_max_loading
    lean_state = solve_open_water(lean, 1.0, 0.55 * lean.diameter)
    fat_state = solve_open_water(fat, 1.0, 0.55 * fat.diameter)
    assert lean_state.model_valid and not fat_state.model_valid


def test_bollard_is_flagged_not_silently_capped():
    geom = PropellerGeometry(**LIB_GEOM)
    state = solve_open_water(geom, 1.0, 0.0)
    assert state.validity_flag == "bollard_unsupported"
    assert not state.model_valid
    assert math.isfinite(state.thrust_coefficient)
    assert int(np.sum(np.asarray(state.a) >= 0.849)) == np.asarray(state.a).size


def test_efficiency_never_negative():
    geom = PropellerGeometry(**LIB_GEOM)
    for J in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
        state = solve_open_water(geom, 1.0, J * geom.diameter)
        assert 0.0 <= state.efficiency <= 1.0


def test_thrust_and_torque_are_monotone_inside_the_valid_envelope():
    geom = PropellerGeometry(diameter=0.6, n_blades=4, hub_ratio=0.35, pitch_ratio=1.0,
                             chord_ratio=0.12)
    states = [solve_open_water(geom, 1.0, J * geom.diameter) for J in (0.6, 0.8, 1.0)]
    assert all(s.model_valid for s in states)
    kt = [s.thrust_coefficient for s in states]
    assert all(a > b for a, b in zip(kt, kt[1:]))


def test_state_dict_exposes_the_validity_diagnostics():
    geom = PropellerGeometry(**LIB_GEOM)
    payload = solve_open_water(geom, 1.0, 0.6 * geom.diameter).as_dict()
    for key in ("momentum_residual", "model_valid", "validity_flag", "converged",
                "induction_model", "ct_max_loading", "thrust_momentum_N"):
        assert key in payload
    assert isinstance(payload["model_valid"], bool)


def test_open_water_state_is_still_a_frozen_dataclass():
    geom = PropellerGeometry(**LIB_GEOM)
    state = solve_open_water(geom, 1.0, 0.8 * geom.diameter)
    assert isinstance(state, OpenWaterState)
    with pytest.raises(Exception):
        state.thrust_coefficient = 1.0  # type: ignore[misc]


# ------------------------------------------------------- the announced branch ---
def test_coded_buhl_form_does_not_invert_its_own_parabola():
    """Finding: the constant form is not the inverse it is documented to be.

    At ``F = 1`` the coded form returns ``a = 0.9446``, whose ``CT`` on Buhl's
    own parabola is ``1.857``; the inverse of that parabola at ``CT = 0.96`` is
    ``a = 0.4``.  Anything that simply "calls the branch" inherits this.
    """
    # cap = 1.0 so the induction clip does not mask the value under test (with
    # the module's a_cap = 0.85 the branch is clipped to the cap, i.e. inert)
    a_coded = float(_buhl(np.array([0.9]), np.array([1.0]), 1.0)[0])
    assert a_coded == pytest.approx(0.944640868594, rel=1e-10)
    c0, c1, c2 = empirical_ct_coefficients(1.0, 0.96, base="turbine")
    assert c0 + c1 * a_coded + c2 * a_coded ** 2 == pytest.approx(1.857142857, rel=1e-8)
    disc = c1 * c1 - 4.0 * c2 * (c0 - 0.96)
    roots = [(-c1 - math.sqrt(disc)) / (2.0 * c2), (-c1 + math.sqrt(disc)) / (2.0 * c2)]
    assert min(roots, key=lambda v: abs(v - 0.4)) == pytest.approx(0.4, rel=1e-9)


def test_turbine_convention_is_not_a_change_of_variable_of_this_one():
    """a_t = -a_p maps the propeller relation onto *minus* the turbine CT."""
    a_p = 0.25
    ct_prop = 4.0 * a_p * (1.0 + a_p)
    ct_turb = 4.0 * (-a_p) * (1.0 - (-a_p))
    assert ct_prop > 0.0 > ct_turb
    assert abs(ct_prop + ct_turb) < 1e-15


# ------------------------------------------------------------- the artifact ---
@pytest.mark.skipif(not ARTIFACT.exists(), reason="verification artifact not built")
def test_verification_campaign_has_no_failing_gate():
    report = json.loads(ARTIFACT.read_text())
    assert report["protocol"] == "m16448::open_water_high_induction::v1"
    assert report["n_fail"] == 0, report["fails"]
    assert report["n_gates"] >= 12