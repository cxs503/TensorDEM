"""Tests for the ice-intake jamming / Ergun / cooling-envelope module."""

import json
import math

import numpy as np
import pytest

from tensordem.ice_intake import (
    BEVERLOO_C_LITERATURE,
    BEVERLOO_K_LITERATURE,
    IntakeGeometry,
    PumpModel,
    availability_envelope,
    beverloo_flow,
    beverloo_identifiability,
    deposition_rate,
    ergun_branch_audit,
    ergun_friction_factor,
    ergun_pressure_gradient,
    fit_beverloo,
    fit_power_law,
    funnel_trial_dem,
    kozeny_carman_gradient,
    leave_one_out,
    silo_discharge_dem,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# Beverloo
# ---------------------------------------------------------------------------


def test_beverloo_matches_manual_formula():
    D, d, C, k, rho_b, g = 0.09, 0.0125, 0.58, 1.4, 650.0, 9.81
    expected = C * rho_b * math.sqrt(g) * (D - k * d) ** 2.5
    assert beverloo_flow(D, d, C, k, rho_b, g) == pytest.approx(expected, rel=1e-15)


def test_beverloo_vanishes_below_critical_aperture():
    assert beverloo_flow(0.01, 0.0125, 0.58, 1.4, 650.0, 9.81) == 0.0
    assert beverloo_flow(0.017, 0.0125, 0.58, 1.4, 650.0, 9.81) == pytest.approx(0.0)


def test_beverloo_vectorised_input():
    flows = beverloo_flow(np.array([0.05, 0.06]), 0.0125, 0.58, 1.4, 650.0, 9.81)
    assert flows.shape == (2,)
    assert np.all(np.diff(flows) > 0)


def test_fit_recovers_exact_parameters_from_clean_data():
    D = np.array([0.050, 0.060, 0.075, 0.090])
    d = np.full(4, 0.0125)
    Q = beverloo_flow(D, d, 0.61, 1.45, 650.0, 9.81)
    fit = fit_beverloo(D, d, Q, 650.0, 9.81)
    assert fit["C"] == pytest.approx(0.61, rel=1e-6)
    assert fit["k"] == pytest.approx(1.45, rel=1e-6)
    assert fit["max_rel_error"] < 1e-10


def test_fit_rejects_non_finite_flow():
    D = np.array([0.05, 0.06, 0.075])
    d = np.full(3, 0.0125)
    with pytest.raises(ValueError):
        fit_beverloo(D, d, np.array([1.0, np.nan, 2.0]))


def test_leave_one_out_error_small_for_consistent_data():
    D = np.linspace(0.05, 0.09, 3)
    d = np.full(3, 0.0125)
    Q = beverloo_flow(D, d, 0.58, 1.4, 650.0, 9.81)
    loo = leave_one_out(D, d, Q, 650.0, 9.81)
    assert loo["max_rel_error"] < 1e-6


def test_identifiability_recovers_parameters_from_noisy_data():
    res = beverloo_identifiability(n_draws=40, seed=1)
    assert res["ensemble"]["C_rel_error_of_mean"] < 0.05
    assert res["ensemble"]["k_rel_error_of_mean"] < 0.05
    assert res["ensemble"]["C_std"] > 0.0
    assert len(res["orifices_m"]) == 4


def test_identifiability_reference_realisation_is_reported():
    res = beverloo_identifiability(n_draws=5, seed=3)
    assert res["reference"]["max_rel_error"] < 0.2
    assert 0.4 < res["reference"]["C_hat"] < 0.8


def test_literature_bands_are_ordered():
    assert BEVERLOO_C_LITERATURE[0] < BEVERLOO_C_LITERATURE[1]
    assert BEVERLOO_K_LITERATURE[0] < BEVERLOO_K_LITERATURE[1]


# ---------------------------------------------------------------------------
# jamming statistics
# ---------------------------------------------------------------------------


def test_wilson_interval_properties():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.2
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and 0.8 < lo < 1.0
    lo1, hi1 = wilson_interval(5, 10)
    lo2, hi2 = wilson_interval(50, 100)
    assert (hi2 - lo2) < (hi1 - lo1)
    assert lo1 < 0.5 < hi1


def test_power_law_fit_recovers_exponent():
    ratios = np.array([1.5, 2.0, 2.5, 3.0, 4.0])
    probs = 0.9 * ratios ** (-1.8)
    fit = fit_power_law(ratios, probs, n_boot=200)
    assert fit["alpha"] == pytest.approx(1.8, rel=1e-6)
    assert fit["ci95"][0] <= fit["alpha"] <= fit["ci95"][1]
    assert fit["r2"] > 0.999


def test_power_law_fit_handles_all_zero_probabilities():
    fit = fit_power_law([1.5, 2.0], [0.0, 0.0])
    assert math.isnan(fit["alpha"])
    assert fit["points"] == 0


def test_power_law_fit_ignores_zero_points():
    ratios = np.array([1.5, 2.0, 3.0, 4.0])
    probs = np.array([0.8, 0.4, 0.1, 0.0])
    fit = fit_power_law(ratios, probs, n_boot=100)
    assert fit["points"] == 3
    assert fit["alpha"] > 0


@pytest.fixture(scope="module")
def funnel_trials():
    out = {}
    for key, ratio in (("narrow", 1.6), ("wide", 4.0)):
        out[key] = [funnel_trial_dem(ratio, 60.0, t_max=0.35, seed=s,
                                     layers=6, half_width=9.0)
                    for s in range(3)]
    return out


def test_funnel_trial_is_reproducible(funnel_trials):
    a = funnel_trial_dem(2.0, 60.0, t_max=0.3, seed=11, layers=6,
                         half_width=9.0)
    b = funnel_trial_dem(2.0, 60.0, t_max=0.3, seed=11, layers=6,
                         half_width=9.0)
    assert a["jammed"] == b["jammed"]
    assert a["escaped"] == b["escaped"]


def test_funnel_trial_mass_ledger(funnel_trials):
    for group in funnel_trials.values():
        for res in group:
            assert abs(res["mass_residual"]) < 1e-12
            assert res["particles"] > 0


def test_narrow_slots_jam_more_often_than_wide_slots(funnel_trials):
    narrow = sum(r["jammed"] for r in funnel_trials["narrow"]) / 3
    wide = sum(r["jammed"] for r in funnel_trials["wide"]) / 3
    assert narrow >= wide


# ---------------------------------------------------------------------------
# Ergun
# ---------------------------------------------------------------------------


def test_ergun_viscous_branch_equals_blake_kozeny():
    audit = ergun_branch_audit()
    assert audit["residual_viscous_vs_blake"] < 1e-12
    assert audit["rel_viscous_vs_blake"] < 1e-12


def test_ergun_inertial_branch_equals_burke_plummer():
    audit = ergun_branch_audit()
    assert audit["rel_inertial_vs_burke_plummer"] < 0.05


def test_ergun_friction_factor_reconstruction():
    audit = ergun_branch_audit()
    assert audit["rel_friction_factor"] < 0.05


def test_ergun_numeric_value_matches_hand_calculation():
    u, d_p, eps, rho, mu = 0.15, 0.025, 0.45, 1025.0, 1.88e-3
    expected = (150.0 * mu * u * (1 - eps) ** 2 / (eps**3 * d_p**2)
                + 1.75 * rho * u**2 * (1 - eps) / (eps**3 * d_p))
    assert ergun_pressure_gradient(u, d_p, eps, rho, mu)["total"] == pytest.approx(
        expected, rel=1e-14
    )


def test_ergun_asymptotic_regimes():
    audit = ergun_branch_audit()
    assert audit["laminar_ratio_inertial_over_viscous"] < 1e-5
    assert audit["turbulent_ratio_viscous_over_inertial"] < 1e-4


def test_ergun_monotone_in_velocity_and_voidage():
    u = np.logspace(-3, 1, 20)
    totals = ergun_pressure_gradient(u, 0.025, 0.45)["total"]
    assert np.all(np.diff(totals) > 0)
    dense = ergun_pressure_gradient(0.2, 0.025, 0.40)["total"]
    open_bed = ergun_pressure_gradient(0.2, 0.025, 0.60)["total"]
    assert dense > open_bed


def test_ergun_rejects_invalid_voidage():
    with pytest.raises(ValueError):
        ergun_pressure_gradient(0.1, 0.02, 0.0)
    with pytest.raises(ValueError):
        ergun_pressure_gradient(0.1, 0.02, 1.5)


def test_kozeny_carman_carman_band_is_twenty_percent():
    blake = float(kozeny_carman_gradient(0.1, 0.025, 0.45))
    carman = float(kozeny_carman_gradient(0.1, 0.025, 0.45, blake_kozeny=5.0))
    assert carman / blake == pytest.approx(5.0 / (150.0 / 36.0), rel=1e-12)
    assert 0.15 < carman / blake - 1.0 < 0.25


def test_friction_factor_form():
    assert ergun_friction_factor(100.0) == pytest.approx(3.25)
    assert ergun_friction_factor(1e9) == pytest.approx(1.75, rel=1e-6)


# ---------------------------------------------------------------------------
# cooling envelope
# ---------------------------------------------------------------------------


def test_deposition_rate_is_linear_in_each_factor():
    base = deposition_rate(0.02, 5.0, 0.5, 0.03)
    assert base == pytest.approx(0.02 * 5.0 * 0.5 / 0.03)
    assert deposition_rate(0.04, 5.0, 0.5, 0.03) == pytest.approx(2 * base)
    assert deposition_rate(0.02, 5.0, 0.0, 0.03) == 0.0


@pytest.fixture(scope="module")
def envelope():
    geometry = IntakeGeometry()
    pump = PumpModel()
    rows, summary = availability_envelope(
        geometry, pump, ([1.5, 2.5, 4.0], [0.9, 0.4, 0.02]),
        [0.01, 0.05], [2.0, 8.0],
    )
    return rows, summary


def test_envelope_table_shape(envelope):
    rows, summary = envelope
    assert len(rows) == 4
    assert summary["rows"] == 4
    assert summary["D_over_d"] == pytest.approx(
        IntakeGeometry().slot_pitch / IntakeGeometry().floe_diameter
    )


def test_envelope_allowed_time_decreases_with_ice_concentration(envelope):
    rows, _ = envelope
    low = [r for r in rows if r.ice_concentration == 0.01]
    high = [r for r in rows if r.ice_concentration == 0.05]
    for a, b in zip(low, high):
        assert a.allowed_time_s >= b.allowed_time_s


def test_envelope_allowed_time_decreases_with_speed(envelope):
    rows, _ = envelope
    for conc in (0.01, 0.05):
        slow = [r for r in rows if r.ice_concentration == conc
                and r.ship_speed_kn == 2.0][0]
        fast = [r for r in rows if r.ice_concentration == conc
                and r.ship_speed_kn == 8.0][0]
        assert slow.allowed_time_s >= fast.allowed_time_s


def test_envelope_rows_are_json_serialisable(envelope):
    rows, summary = envelope
    payload = json.dumps({"rows": [r.as_dict() for r in rows], "summary": summary})
    assert "allowed_time_s" in payload
    assert all(r.pressure_drop_pa >= 0 for r in rows)


def test_envelope_reports_jam_probability_from_curve(envelope):
    rows, summary = envelope
    ratios = np.log(np.array([1.5, 2.5, 4.0]))
    probs = np.log(np.array([0.9, 0.4, 0.02]))
    expected = float(np.exp(np.interp(np.log(summary["D_over_d"]), ratios, probs)))
    assert summary["jam_probability"] == pytest.approx(expected, rel=1e-9)
    assert all(r.jam_probability == pytest.approx(expected) for r in rows)


def test_clean_grill_has_pressure_margin():
    geometry = IntakeGeometry()
    pump = PumpModel()
    clean = ergun_pressure_gradient(
        pump.volumetric_flow / geometry.initial_open_area,
        geometry.floe_diameter, geometry.voidage, pump.density, pump.viscosity,
    )["total"] * geometry.flow_length
    assert pump.max_pressure_drop / clean > 1.0


# ---------------------------------------------------------------------------
# DEM anchors (small, fast configurations)
# ---------------------------------------------------------------------------


def test_silo_discharge_smoke():
    res = silo_discharge_dem(5.0, half_width=8.0, layers=5, settle_steps=150,
                             window_start=0.3, window_end=0.8)
    assert res["particles"] > 100
    assert abs(res["mass_residual"]) < 1e-12
    assert res["max_wall_penetration"] < 0.25
    assert res["flow"] > 0.0
    assert res["r2"] > 0.5  # coarse config: stepwise cumulative curve


def test_silo_flow_increases_with_aperture():
    small = silo_discharge_dem(4.0, half_width=8.0, layers=5, settle_steps=150,
                               window_start=0.3, window_end=0.8)
    large = silo_discharge_dem(6.0, half_width=8.0, layers=5, settle_steps=150,
                               window_start=0.3, window_end=0.8)
    assert large["flow"] > small["flow"]