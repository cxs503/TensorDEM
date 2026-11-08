"""m6457 — `beverloo_discharge.simulate_discharge` must *measure*, not echo.

Regression for a defect where the function returned the analytic Beverloo
correlation as ``flow_rate_kg_s`` with ``n_discharged = 0`` and ``r2 = 1.0``
(self-fulfilling "validation"), after running an unrelated repose simulation.
See ``docs/beverloo_discharge_measurement_audit.md``.

The DEM cases here use a small 80-particle silo (≈6 s each) so the file stays
fast; the campaign numbers live in ``runs/m6457_beverloo/``.
"""
from __future__ import annotations

import math

import pytest

from tensordem.beverloo_discharge import (
    BeverlooCase,
    BeverlooError,
    MAX_INITIAL_OVERLAP_OVER_RADIUS,
    beverloo_2d_prediction,
    beverloo_prediction,
    initial_interpenetration,
    recovered_C,
    simulate_discharge,
    to_silo_case,
)
from tensordem.beverloo_discharge import _steadiness_r2

QUICK = dict(radius=0.01, density=2500.0, silo_radius=0.06, fill_height=0.06,
             orifice_diameter=0.06, packing_gap=0.0, jitter=0.002,
             settle_time=0.05, window=0.04, n_windows=5, max_time=0.25)


@pytest.fixture(scope="module")
def quick_result():
    return simulate_discharge(BeverlooCase(**QUICK))


# ---------------------------------------------------------------- tautology --
def test_measurement_is_not_the_prediction(quick_result):
    """The headline defect: measured W must not be the correlation value."""
    r = quick_result
    w_lit = r["beverloo_prediction"]["W_kg_s"]
    assert r["flow_rate_source"] == "dem_mass_balance"
    assert r["n_discharged"] > 0, "no particle left the silo: nothing was measured"
    assert r["total_discharged_mass_kg"] > 0.0
    assert r["flow_rate_kg_s"] != w_lit, "measured rate is bit-identical to the prediction"
    assert abs(r["rel_err_vs_beverloo"]) > 1e-6


def test_mass_balance_closes(quick_result):
    """Particle count, discharged mass and the window average are consistent.

    The rate is a *window* average, so mass that leaves during settling is
    reported separately (``outside_window_discharged_mass_kg``) instead of being
    smeared into the steady rate.
    """
    r = quick_result
    mass = BeverlooCase(**QUICK).particle_mass
    assert abs(r["mass_balance_residual_kg"]) < 1e-9
    assert r["total_discharged_mass_kg"] == pytest.approx(r["n_discharged"] * mass, rel=1e-12)
    span = sum(r["flow_rate_windows_kg_s"]) * r["window"]
    assert r["window_span_discharged_mass_kg"] == pytest.approx(span, rel=1e-12)
    assert r["outside_window_discharged_mass_kg"] == pytest.approx(
        r["total_discharged_mass_kg"] - span, rel=1e-12)
    assert r["outside_window_discharged_mass_kg"] >= -1e-12
    assert r["n_discharged"] * mass == pytest.approx(
        span + max(r["outside_window_discharged_mass_kg"], 0.0), rel=1e-9)


def test_window_series_reproduces_measured_rate(quick_result):
    r = quick_result
    windows = r["flow_rate_windows_kg_s"]
    assert len(windows) == r["n_windows"]
    assert all(w >= 0.0 for w in windows)
    assert sum(windows) / len(windows) == pytest.approx(r["flow_rate_kg_s"], rel=1e-12)


def test_r2_is_steady_flow_not_agreement(quick_result):
    """r2 must be earned by the window series, not set to 1 by construction."""
    r = quick_result
    recomputed = _steadiness_r2(r["flow_rate_windows_kg_s"], r["window"], r["t_settle_s"])
    assert r["r2"] == pytest.approx(recomputed, abs=1e-12)
    assert r["r2_basis"] == "steady_flow_cumulative_mass_vs_time"
    # a stalled (start-up burst only) series must not score 1
    stall = _steadiness_r2([0.5, 0.0, 0.0, 0.0, 0.0], 0.2, 0.25)
    assert stall < 0.99
    assert _steadiness_r2([1.0] * 5, 0.2, 0.25) == pytest.approx(1.0, abs=1e-12)


def test_measurement_responds_to_physics():
    """A frictionless-ish bed must not return the same number as a frictional one."""
    base = dict(QUICK)
    fast = simulate_discharge(BeverlooCase(**{**base, "mu": 0.05, "mu_rolling": 0.0}))
    slow = simulate_discharge(BeverlooCase(**{**base, "mu": 0.6, "mu_rolling": 0.3}))
    assert fast["n_discharged"] != 0 or slow["n_discharged"] != 0
    assert fast["flow_rate_kg_s"] != slow["flow_rate_kg_s"]


def test_return_history_is_particle_state(quick_result):
    r = simulate_discharge(BeverlooCase(**QUICK), return_history=True)
    assert r["positions"].shape == (r["n_initial"], 3)
    assert r["alive"].shape == (r["n_initial"],)
    assert int((~r["alive"]).sum()) == r["n_discharged"]
    assert int((r["alive"]).sum()) == r["n_remaining"]


# ------------------------------------------------------------------- guards --
def test_initial_overlap_guard_rejects_divergent_start():
    bad = BeverlooCase(radius=0.0075, silo_radius=0.11, fill_height=0.12,
                       orifice_diameter=0.075, packing_gap=0.02, jitter=0.08)
    assert initial_interpenetration(bad) > MAX_INITIAL_OVERLAP_OVER_RADIUS
    with pytest.raises(BeverlooError, match="interpenetrate"):
        simulate_discharge(bad)


def test_overlap_metric_is_zero_for_clean_lattice():
    ok = BeverlooCase(radius=0.0075, silo_radius=0.11, fill_height=0.12,
                      orifice_diameter=0.075, packing_gap=0.02, jitter=0.002)
    assert initial_interpenetration(ok) == 0.0


def test_case_rejects_time_that_cannot_cover_the_windows():
    """Analytic use may build any case; simulating a too-short run must refuse."""
    short = BeverlooCase(**{**QUICK, "max_time": 0.1})
    beverloo_prediction(short)          # correlation still evaluable
    with pytest.raises(BeverlooError, match="max_time"):
        to_silo_case(short)
    with pytest.raises(BeverlooError, match="max_time"):
        simulate_discharge(short)


# ---------------------------------------------------------------- mapping ----
def test_to_silo_case_forwards_geometry():
    case = BeverlooCase(**QUICK)
    silo = to_silo_case(case)
    assert silo.radius == case.radius
    assert silo.silo_radius == case.silo_radius
    assert silo.orifice_diameter == case.orifice_diameter
    assert silo.plate_thickness == pytest.approx(2.0 * case.radius)
    assert silo.discharge_time == pytest.approx(case.max_time - case.settle_time)
    assert silo.n_windows == case.n_windows


def test_recovered_C_inverts_the_correlation():
    case = BeverlooCase(**QUICK)
    pred = beverloo_prediction(case, C=0.58, k=1.5)
    rho_ref = 0.64 * case.density
    assert recovered_C(case, pred["W_kg_s"], k=1.5, rho_b=rho_ref) == pytest.approx(0.58, rel=1e-12)
    # the 2-D form keeps its own exponent and stays positive
    two_d = beverloo_2d_prediction(0.1, 0.01, 1200.0, 9.81)
    assert two_d["valid"] and two_d["W_kg_ms"] > 0.0
    assert not beverloo_2d_prediction(0.01, 0.01, 1200.0, 9.81)["valid"]