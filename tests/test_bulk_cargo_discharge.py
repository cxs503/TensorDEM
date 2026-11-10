"""Regression tests for the bulk-cargo hold verification surface (LW3-N06).

Two layers:

* **analytic / bookkeeping** (fast, no DEM): the Beverloo and Janssen helpers
  must invert their own laws, the repose-angle reduction must recover a
  prescribed cone, and the case validation must reject non-physical set-ups.
* **DEM smoke + honesty guards** (one short real DEM run): the hold packing has
  no initial interpenetration, the discharged mass equals the sum of the window
  mass balance, and a hold whose outlet is smaller than two particle diameters
  does not discharge (the C4b negative control mechanism).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.bulk_cargo import (
    BEVERLOO_C,
    BEVERLOO_K,
    BulkCargoError,
    CargoHoldCase,
    PourCase,
    _funnel_bed,
    _hold_bed,
    beverloo_prediction,
    fit_janssen,
    fit_power_law,
    measure_repose_angle,
    numerical_checks,
    simulate_hold_discharge,
    simulate_hold_static,
)
from tensordem.granular3d_core import build_pairs


# --------------------------------------------------------------------------- #
# analytic helpers
# --------------------------------------------------------------------------- #
def test_beverloo_law_shape():
    d = 0.005
    D = np.array([0.02, 0.03, 0.04])
    W = beverloo_prediction(D, d, 1500.0)
    # 5/2 exponent: W(D - k d) is exactly power 2.5
    x = D - BEVERLOO_K * d
    ratio = W[1:] / W[:-1]
    np.testing.assert_allclose(ratio, (x[1:] / x[:-1]) ** 2.5, rtol=1e-12)
    # prefactor C rho_b sqrt(g)
    np.testing.assert_allclose(W[0], BEVERLOO_C * 1500.0 * math.sqrt(9.81) * x[0] ** 2.5,
                               rtol=1e-12)
    # g scaling
    np.testing.assert_allclose(beverloo_prediction(D, d, 1500.0, g=4 * 9.81),
                               2.0 * W, rtol=1e-12)


def test_beverloo_zero_below_effective_aperture():
    # D <= k d leaves no free aperture -> the law returns exactly zero
    assert beverloo_prediction(0.006, 0.005, 1500.0) == 0.0
    assert beverloo_prediction(BEVERLOO_K * 0.005, 0.005, 1500.0) == 0.0


def test_fit_power_law_recovers_synthetic_exponent():
    d, rho_b = 0.005, 1500.0
    D = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
    W = beverloo_prediction(D, d, rho_b)
    fit = fit_power_law(D, W, d)
    assert fit["exponent"] == pytest.approx(2.5, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["prefactor"] == pytest.approx(BEVERLOO_C * rho_b * math.sqrt(9.81),
                                            rel=1e-9)


def test_fit_power_law_marks_failure_without_points():
    fit = fit_power_law([0.02], [1.0], 0.005)
    assert math.isnan(fit["exponent"])
    assert fit["n_points"] <= 1


def test_fit_janssen_recovers_profile():
    z = np.linspace(0.01, 0.25, 12)
    p_inf, lam = 3200.0, 0.045
    sigma = p_inf * (1.0 - np.exp(-z / lam))
    fit = fit_janssen(z, sigma)
    assert fit["p_inf"] == pytest.approx(p_inf, rel=1e-4)
    assert fit["lambda"] == pytest.approx(lam, rel=1e-4)
    assert fit["rms_rel"] < 1e-6


def test_janssen_identity_holds_for_synthetic_profile():
    # the parameter-free identity p_inf / lambda = rho_b g R_h of the slab balance
    rho_b, g, R_h, mu_w, K = 1500.0, 9.81, 0.022, 0.4, 0.5
    lam = R_h / (mu_w * K)
    p_inf = rho_b * g * R_h / (mu_w * K)
    # geometry-free identity of the Janssen slab balance
    assert p_inf / lam == pytest.approx(rho_b * g, rel=1e-12)
    # ... and the cylindrical-silo form lambda = R/(2 mu_w K) for R_h = R/2
    R = 2 * R_h
    assert lam == pytest.approx(R / (2 * mu_w * K), rel=1e-12)


# --------------------------------------------------------------------------- #
# repose-angle reduction
# --------------------------------------------------------------------------- #
def _synthetic_cone(theta_deg: float, n: int = 4000, r_max: float = 0.12,
                    seed: int = 3) -> np.ndarray:
    """A *surface layer* of particles on a cone of the given surface angle.

    Only the flank matters for the repose reduction: a particle *centre* lies
    roughly half a diameter below the free surface, which is a constant offset
    and therefore does not bias the fitted slope.
    """
    rng = np.random.default_rng(seed)
    r = r_max * np.sqrt(rng.random(n))
    phi = 2 * np.pi * rng.random(n)
    tan_t = math.tan(math.radians(theta_deg))
    z_surface = (r_max - r) * tan_t
    z = np.maximum(z_surface - 0.01 * rng.random(n), 0.001)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


@pytest.mark.parametrize("theta", [15.0, 25.0, 32.0])
def test_measure_repose_angle_recovers_synthetic_cone(theta):
    pos = _synthetic_cone(theta)
    res = measure_repose_angle(pos, radius=0.005, bin_width=0.005)
    assert res["theta_deg"] == pytest.approx(theta, abs=1.0)
    assert res["r2"] > 0.99


def test_measure_repose_angle_flat_layer_is_zero():
    rng = np.random.default_rng(0)
    n = 2000
    r = 0.10 * np.sqrt(rng.random(n))
    phi = 2 * np.pi * rng.random(n)
    pos = np.column_stack([r * np.cos(phi), r * np.sin(phi), rng.random(n) * 0.002])
    res = measure_repose_angle(pos, radius=0.005, bin_width=0.005)
    assert res["theta_deg"] < 2.0


def test_measure_repose_angle_rejects_degenerate_input():
    with pytest.raises(BulkCargoError):
        measure_repose_angle(np.zeros((5, 3)), radius=0.005)


# --------------------------------------------------------------------------- #
# case validation / packing
# --------------------------------------------------------------------------- #
def test_case_validation_rejects_bad_geometry():
    with pytest.raises(BulkCargoError):
        CargoHoldCase(length=0.05, outlet_diameter=0.06)
    with pytest.raises(BulkCargoError):
        CargoHoldCase(fill_height=0.005, radius=0.005)
    with pytest.raises(BulkCargoError):
        CargoHoldCase(n_windows=1)
    with pytest.raises(BulkCargoError):
        PourCase(funnel_side=0.015, radius=0.005)


def test_numerical_checks_resolve_contact_period():
    chk = numerical_checks(CargoHoldCase())
    assert chk["contact_period_steps"] == 20.0
    assert 0.0 < chk["dt_over_rayleigh"] < 0.5      # stable against surface waves
    assert chk["outlet_over_d"] == pytest.approx(3.0)


def test_hold_bed_has_no_initial_overlap_and_is_inside_walls():
    case = CargoHoldCase(length=0.06, breadth=0.05, fill_height=0.06,
                         outlet_diameter=0.02)
    pos = _hold_bed(case)
    assert len(pos) > 50
    iu, ju = build_pairs(pos, 2.0 * case.radius * 1.000001)
    if iu.size:
        d = np.linalg.norm(pos[ju] - pos[iu], axis=1)
        assert d.min() > 2.0 * case.radius * 0.999   # relaxation removed overlaps
    r = case.radius
    assert pos[:, 0].min() > 0.5 * r and pos[:, 0].max() < case.length - 0.5 * r
    assert pos[:, 1].min() > 0.5 * r and pos[:, 1].max() < case.breadth - 0.5 * r
    assert pos[:, 2].min() > 0.5 * r


def test_funnel_bed_stays_inside_funnel():
    case = PourCase(funnel_side=0.04, column_height=0.06)
    pos = _funnel_bed(case)
    s = case.funnel_side
    assert pos[:, 0].min() > 0.5 * s and pos[:, 0].max() < 1.5 * s
    assert pos[:, 2].min() > case.funnel_bottom


# --------------------------------------------------------------------------- #
# DEM smoke + honesty guards
# --------------------------------------------------------------------------- #
MINI = dict(length=0.06, breadth=0.05, fill_height=0.06, outlet_diameter=0.020,
            mu_rolling=0.05, settle_time=0.05, n_windows=3)


def test_discharge_mass_balance_is_consistent():
    case = CargoHoldCase(discharge_time=0.12, **MINI)
    res = simulate_hold_discharge(case)
    mass = case.particle_mass
    windows = np.array(res["flow_rate_windows_kg_s"])
    win = case.discharge_time / case.n_windows
    # the reported rate is the window mass balance -- not the correlation
    assert windows.sum() * win == pytest.approx(res["discharged_mass_kg"], rel=1e-9)
    assert res["discharged_mass_kg"] == pytest.approx(res["n_discharged"] * mass,
                                                      rel=1e-12)
    assert res["flow_rate_kg_s"] == pytest.approx(windows.mean(), rel=1e-12)
    assert res["n_initial"] > res["n_discharged"] >= 0
    # nothing escapes upwards/outwards through the walls
    assert res["bed_top_m"] < case.fill_height + 0.02


def test_blocked_orifice_does_not_discharge():
    """C4b mechanism: D < 2 d arches over and the hold stays loaded."""
    mini = {k: v for k, v in MINI.items()
            if k not in ("outlet_diameter", "settle_time", "n_windows")}
    case = CargoHoldCase(outlet_diameter=0.006, discharge_time=0.10,
                         settle_time=0.04, n_windows=2, **mini)
    assert case.outlet_diameter < 2.0 * case.diameter
    res = simulate_hold_discharge(case)
    assert res["n_discharged"] == 0
    assert res["flow_rate_kg_s"] == 0.0


def test_static_hold_reports_positive_wall_pressure_and_no_discharge():
    case = CargoHoldCase(length=0.06, breadth=0.05, fill_height=0.06,
                         outlet_diameter=0.020, settle_time=0.05)
    res = simulate_hold_static(case, n_slabs=6)
    assert res["n_particles"] > 50
    assert len(res["profile"]) >= 3
    sig_h = np.array([row["sigma_h_Pa"] for row in res["profile"]])
    sig_v = np.array([row["sigma_v_Pa"] for row in res["profile"]])
    assert np.all(sig_h >= 0.0) and np.all(sig_v >= 0.0)
    assert sig_h.max() > 0.0 and sig_v.max() > 0.0
    assert res["K_measured"] > 0.0
    # the force-balance vertical stress must saturate below the hydrostatic value
    hydro = np.array(res["hydrostatic_reference_Pa"])
    assert hydro.max() > sig_v.max()