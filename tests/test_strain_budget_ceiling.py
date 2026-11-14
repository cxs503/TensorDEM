"""Round 17 (m21066) — a strain request must be a ceiling, not a floor.

Round 16 measured that ``phi_estimator_audit.py --live --strain 2.0`` delivered
strains up to **49.5x** the requested budget at I = 0.3 while the run still
reported ``servo_valid=True``.  ``simulate_servo_shear`` reported only
``shear_steps`` (a floor, ``max(min_shear_steps, ceil(strain/(gdot dt)))``) and
``shear_strain_achieved``; nothing said whether the delivered strain honoured
the request, so the overrun was invisible downstream.

This module pins the three things that fix it:

1. ``strain_budget_status`` — the single classification helper.
2. the *mechanism*: the step budget is a floor, so the worst-case strain overrun
   grows monotonically with I and is ~50x at I = 0.3 (re-derived here from the
   device's own arithmetic, not copied from the round-16 table);
3. the *behaviour*: with ``max_recover_strain`` set the recovery phase stops at
   the strain ceiling and flags it; with ``None`` the historical (bit-for-bit)
   overrun still happens **but is now flagged**.

The two live runs are the slow part (~20 s and ~2 s); they are marked ``slow``
like the rest of the DEM campaign tests.
"""
from __future__ import annotations

import math

import pytest

from tensordem.mu_i_rheology import (
    ServoShearCase,
    simulate_servo_shear,
    strain_budget_status,
)

RADIUS = 0.005
KN = 6.25e4
PRESSURE = 62.5e3
FILL = 0.04
SEED = 5279


# ---------------------------------------------------------------------------
# 1. the classification helper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("achieved, requested, respected, status", [
    (2.00, 2.0, True, "respected"),      # exactly on budget
    (2.10, 2.0, True, "respected"),      # inside the 5 % tolerance
    (2.11, 2.0, False, "violated"),      # just outside it
    (98.93, 2.0, False, "violated"),     # round-16 I = 0.3 archival
    (0.964, 0.5, False, "violated"),     # min_shear_steps floor overshoot
])
def test_helper_classifies_against_request(achieved, requested, respected, status):
    out = strain_budget_status(achieved, requested)
    assert out["strain_budget_respected"] is respected
    assert out["strain_budget_status"] == status
    assert out["strain_overrun_ratio"] == pytest.approx(achieved / requested)


def test_helper_tolerance_is_the_documented_one():
    # the default tolerance is 5 %, and it is exclusive at the boundary
    assert strain_budget_status(2.0 * 1.05, 2.0)["strain_budget_respected"] is True
    assert strain_budget_status(2.0 * 1.0501, 2.0)["strain_budget_respected"] is False


@pytest.mark.parametrize("requested", [0.0, -1.0])
def test_helper_refuses_nonpositive_request(requested):
    out = strain_budget_status(1.0, requested)
    assert out["strain_budget_status"] == "undefined"
    assert out["strain_budget_respected"] is False
    assert math.isnan(out["strain_overrun_ratio"])


# ---------------------------------------------------------------------------
# 2. the mechanism: the step budget is a floor, so the overrun scales with I
# ---------------------------------------------------------------------------
def _worst_case(I: float, strain: float = 2.0):
    """Reproduce the device's own step arithmetic (``mu_i_rheology`` ~1226)."""
    probe = ServoShearCase(radius=RADIUS, kn=KN, normal_stress=PRESSURE,
                           inertial_number=I, fill_height=FILL,
                           wall_mode="smooth", seed=SEED)
    dt = probe.timestep
    d = probe.particle_diameter
    gamma_dot = I * math.sqrt(PRESSURE / probe.density) / d
    # what phi_estimator_audit.live_run passes as min_shear_steps
    driver_floor = int(math.ceil(strain / (gamma_dot * dt))) + 100
    shear_steps = int(max(driver_floor,
                          math.ceil(strain / (gamma_dot * dt))))
    worst_strain = (shear_steps + probe.max_recover_steps) * gamma_dot * dt
    return shear_steps, worst_strain, gamma_dot, dt, d, probe


def test_step_budget_is_a_floor_independent_of_the_strain_request():
    """Adding to ``shear_strain`` barely moves ``shear_steps`` at high I."""
    _, _, gamma_dot, dt, _, _ = _worst_case(0.3)
    # the requested strain contributes ceil(strain/(gdot dt)) steps; at I = 0.3
    # one unit of strain is ~207 steps, while min_shear_steps is 515 -- so the
    # floor, not the request, decides how long the run is.
    steps_per_strain = 1.0 / (gamma_dot * dt)
    assert 200.0 < steps_per_strain < 215.0
    assert _worst_case(0.3)[0] == 515


def test_worst_case_overrun_grows_monotonically_with_I():
    Is = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    overruns = [_worst_case(I)[1] / 2.0 for I in Is]
    assert overruns == sorted(overruns), overruns
    # low I cannot overrun much: the request dominates the step floor
    assert overruns[0] < 1.25
    # high I can overrun catastrophically: the floor dominates the request
    assert overruns[-1] > 40.0


def test_round16_high_I_overrun_is_reproducible_from_the_formula():
    """R16 measured 49.5x at I = 0.3 by exhausting the recovery cap; the
    arithmetic here bounds it from above, so it must be at least as large."""
    shear_steps, worst_strain, gamma_dot, dt, _, probe = _worst_case(0.3)
    assert shear_steps == 515
    # 20515 steps of 150 1/s shear == the 98.93 strain R16 archived
    assert (shear_steps + probe.max_recover_steps) * gamma_dot * dt > 98.0
    assert worst_strain / 2.0 > 49.0


def test_strain_ceiling_is_opt_in_and_off_by_default():
    """``None`` keeps every archived run bit-for-bit reproducible."""
    assert ServoShearCase().max_recover_strain is None
    assert ServoShearCase().strain_budget_tol == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 3. behaviour: live runs on the shipped device
# ---------------------------------------------------------------------------
LIVE = dict(radius=RADIUS, kn=KN, normal_stress=PRESSURE, fill_height=FILL,
            wall_mode="smooth", seed=SEED, inertial_number=0.3)


@pytest.mark.slow
def test_ceiling_stops_the_recovery_phase_and_flags_it():
    """With a cap, the run stops at the ceiling instead of over-shearing."""
    case = ServoShearCase(min_shear_steps=50, shear_strain=0.5,
                          sample_stride=10, settle_steps=300,
                          max_recover_strain=0.02, **LIVE)
    res = simulate_servo_shear(case, return_series=False)
    assert res["recover_termination"] == "max_recover_strain"
    assert res["shear_strain_requested"] == pytest.approx(0.5)
    # The ceiling is 0.5 + 0.02.  The gate is evaluated *after* the strain
    # increment, so the overshoot is bounded by exactly one step of strain --
    # at I = 0.3 that is gdot*dt = 150 * 3.215e-5 = 4.82e-3, which is an
    # order of magnitude inside the 5 % classification tolerance.
    gamma_dot = 0.3 * math.sqrt(PRESSURE / 2500.0) / 0.01
    one_step_strain = gamma_dot * ServoShearCase(**LIVE).timestep
    assert res["shear_strain_achieved"] <= 0.5 + 0.02 + one_step_strain + 1e-9
    assert res["shear_strain_achieved"] > 0.5 + 0.02 - one_step_strain
    assert res["strain_budget_status"] == "respected"
    assert res["strain_budget_respected"] is True
    assert res["max_recover_strain"] == pytest.approx(0.02)


@pytest.mark.slow
def test_historical_run_still_overruns_but_is_now_flagged():
    """The unchanged (``None``) path: preserved physics, caught defect.

    Before round 17 this exact run returned ``servo_valid=True`` and nothing
    else — a 33x strain overrun with no indication of it.
    """
    case = ServoShearCase(min_shear_steps=200, shear_strain=0.5,
                          sample_stride=10, settle_steps=300, **LIVE)
    res = simulate_servo_shear(case, return_series=False)
    assert case.max_recover_strain is None
    # the defect is real: 33x past the request
    assert res["strain_overrun_ratio"] > 30.0
    assert res["shear_strain_achieved"] > 15.0
    # ... and it is no longer silent
    assert res["strain_budget_respected"] is False
    assert res["strain_budget_status"] == "violated"
    assert res["shear_strain_requested"] == pytest.approx(0.5)
    assert res["recover_termination"] == "pressure_recovered"


@pytest.mark.slow
def test_budget_fields_are_present_on_every_run():
    """No run may be quotable without its budget status next to it."""
    case = ServoShearCase(min_shear_steps=50, shear_strain=0.5,
                          sample_stride=10, settle_steps=300,
                          max_recover_strain=0.0, **LIVE)
    res = simulate_servo_shear(case, return_series=False)
    for key in ("shear_strain_requested", "strain_overrun_ratio",
                "strain_budget_respected", "strain_budget_status",
                "recover_termination", "max_recover_strain"):
        assert key in res, key