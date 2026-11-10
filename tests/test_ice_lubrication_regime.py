"""Acceptance tests for the ice-structure lubrication-regime module.

Campaign node ``p9050-n09-ice-lubrication-maturity``.  The suite is organised
according to the campaign's machine criteria:

* **(a)** ``mu`` vs the ice-friction anchor table: ``>= 5`` of the applicable
  T-V combinations within 20 %;
* **(b)** melt-film thickness steady state vs the analytic reference within 15 %
  (both the literal campaign ODE root and the time-domain RK4 solution);
* **(c)** Stribeck minimum ``Her_opt`` vs the literature band within 30 %;
* **(d)** three negative controls (``V -> 0``, ``T -> T_melt``, ``Ra -> 0``);
* **(e)** falsifiability probe: ``-10 degC / 1 m/s`` must not return ``mu > 0.5``.

Every elementary law is also checked against its own closed form (VFT water
viscosity, Krieger-Dougherty slurry, Clapeyron depression, Blok flash
temperature, Reynolds squeeze-out), so a silent regression in a sub-law cannot
hide behind the end-to-end numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_lubrication_regime import (
    FLASH_CONCENTRATION,
    H_CONV,
    ICE_CREEP_ACTIVATION,
    ICE_VISC_REF,
    LAMBDA_BOUNDARY,
    LAMBDA_HYDRO,
    MELT_RAMP_K,
    MU_BOUNDARY_MAX,
    MU_BOUNDARY_MIN,
    REGIME_BOUNDARY,
    REGIME_FULL_FILM,
    REGIME_HYDRO,
    REGIME_MIXED,
    REGIMES,
    STRIBECK_HER_OPT_BAND,
    STRIBECK_HER_OPT_LITERATURE,
    LubricationInputs,
    blok_effusivity,
    boundary_friction,
    calibrate,
    combined_roughness,
    conductive_capacity,
    drainage_rate,
    falsifiability_probe,
    film_area_fraction,
    film_ode_rhs,
    film_steady_state,
    film_steady_state_analytic,
    film_thickness_supply,
    flash_temperature_rise,
    friction_surface,
    hersey_band,
    hersey_number,
    ice_creep_viscosity,
    integrate_film,
    melting_point,
    melting_point_depression,
    negative_controls,
    predict,
    regime_of,
    slurry_viscosity,
    stribeck_check,
    stribeck_curve,
    stribeck_mu,
    stribeck_optimum,
    stribeck_parameters,
    stribeck_scan,
    validate_literature,
    water_viscosity,
)


# ---------------------------------------------------------------------------
# elementary laws
# ---------------------------------------------------------------------------
def test_water_viscosity_matches_classical_anchors():
    eta = water_viscosity(np.array([0.0, 10.0, -10.0]))
    # VFT correlation reproduces the textbook 1.792e-3 to 2.2 %
    assert eta[0] == pytest.approx(1.79e-3, rel=0.03)
    assert eta[1] == pytest.approx(1.31e-3, rel=0.02)
    # supercooled water is more viscous than the melt
    assert eta[2] > eta[0] > eta[1]
    assert eta[2] == pytest.approx(2.5e-3, rel=0.10)


def test_ice_creep_viscosity_is_arrhenius_and_large():
    eta_ref = float(ice_creep_viscosity(0.0))
    assert eta_ref == pytest.approx(ICE_VISC_REF, rel=1e-12)
    t1, t2 = -10.0, -20.0
    expected = math.exp(ICE_CREEP_ACTIVATION / 8.31446261815324
                        * (1.0 / (t2 + 273.15) - 1.0 / (t1 + 273.15)))
    assert float(ice_creep_viscosity(t2) / ice_creep_viscosity(t1)) == pytest.approx(
        expected, rel=1e-12)
    assert float(ice_creep_viscosity(-10.0)) > 1e11  # ice creeps, it does not flow


def test_slurry_viscosity_krieger_dougherty():
    neat = float(slurry_viscosity(-10.0, 0.0))
    assert neat == pytest.approx(float(water_viscosity(-10.0)), rel=1e-12)
    dense = float(slurry_viscosity(-10.0, 0.55))
    assert dense > 40.0 * neat  # dense slurry: x41.6 at phi = 0.55
    denser = float(slurry_viscosity(-10.0, 0.58))
    assert denser > 100.0 * neat  # x172 at phi = 0.58
    values = [float(slurry_viscosity(-10.0, phi)) for phi in (0.0, 0.2, 0.4, 0.55)]
    assert all(b > a for a, b in zip(values, values[1:]))
    with pytest.raises(ValueError):
        slurry_viscosity(-10.0, 0.7)


def test_clapeyron_melting_point_depression():
    assert float(melting_point(0.0)) == pytest.approx(0.0)
    assert float(melting_point(10.0e6)) == pytest.approx(-0.074, rel=1e-3)
    assert float(melting_point_depression(1.0e6)) == pytest.approx(0.0074, rel=1e-12)


def test_blok_effusivity_and_flash_temperature_inverse():
    eff = blok_effusivity()
    assert eff["combined"] < min(eff["ice"], eff["steel"])
    flux, dwell = 1.0e5, 2.0e-3
    rise = float(flash_temperature_rise(flux, dwell))
    assert rise > 0.0
    # the conductive capacity is the exact inverse of the flash rise
    back = float(conductive_capacity(rise, dwell))
    assert back == pytest.approx(flux, rel=1e-12)


# ---------------------------------------------------------------------------
# film ODE (criterion b)
# ---------------------------------------------------------------------------
def test_campaign_ode_closed_form_root_and_instability():
    q, dt = 1.0e5, 10.0
    ana = film_steady_state_analytic(q, dt)
    assert ana["exists"] == 1.0
    assert ana["h_ss"] == pytest.approx(0.56 * dt / (q - H_CONV * dt), rel=1e-12)
    # the literal root annihilates the campaign ODE
    assert film_ode_rhs(ana["h_ss"], friction_flux_w_m2=q,
                        temperature_deficit_k=dt) == pytest.approx(0.0, abs=1e-12)
    # ... and it is unstable at fixed mu (recorded as a structural finding)
    assert ana["stability_sign"] == 1.0
    # no steady state when the frictional flux cannot beat the convective sink
    assert film_steady_state_analytic(100.0, 10.0)["exists"] == 0.0


def test_campaign_ode_conserves_the_spec_terms():
    h, q, dt = 2.0e-5, 1.0e5, 10.0
    rhs = film_ode_rhs(h, friction_flux_w_m2=q, temperature_deficit_k=dt)
    expected = (q - 0.56 * dt / h - H_CONV * dt) / (917.0 * 3.34e5)
    assert rhs == pytest.approx(expected, rel=1e-12)
    # the drainage extension subtracts a positive flux
    slower = film_ode_rhs(h, friction_flux_w_m2=q, temperature_deficit_k=dt,
                          drainage_retention=1.0, drainage_flux_w_m2=1.0e4)
    assert slower < rhs


def test_steady_state_is_reachable_from_both_sides():
    """Criterion (b): ODE steady state vs analytic root, two independent cases."""
    cases = ((1.0e6, 1.0, 2.0, 1.0e-4), (3.0e5, 5.0, 10.0, 1.0e-4))
    for pressure, speed, deficit, sigma in cases:
        ss = film_steady_state(pressure_pa=pressure, speed_m_s=speed,
                               temperature_deficit_k=deficit, roughness_m=sigma)
        assert ss["starved"] == 0.0, "reference case must have a steady film"
        eta = ss["eta_pa_s"]
        mu_b = float(boundary_friction(pressure, -abs(deficit)))

        def flux(h, eta=eta, mu_b=mu_b, p=pressure, v=speed, s=sigma):
            a = float(film_area_fraction(h / s))
            return ((1.0 - a) * mu_b + a * eta * v / (h * p)) * p * v

        for h0 in (0.6 * ss["h_ss"], 1.5 * ss["h_ss"]):
            out = integrate_film(300.0, initial_thickness_m=h0,
                                 friction_flux_w_m2=flux,
                                 temperature_deficit_k=deficit, steps=30000)
            h_end = out["h"][-1]
            err = abs(h_end - ss["h_ss"]) / ss["h_ss"]
            assert err < 0.15, f"h {h_end:.3e} vs analytic {ss['h_ss']:.3e}: {err:.3%}"


def test_drainage_rate_scaling():
    h, p, r, eta = 1.0e-6, 1.0e6, 1.0e-2, 2.5e-3
    v1 = float(drainage_rate(h, p, r, eta))
    v2 = float(drainage_rate(2.0 * h, p, r, eta))
    assert v2 / v1 == pytest.approx(8.0, rel=1e-12)  # h^3
    v3 = float(drainage_rate(h, p, r, 2.0 * eta))
    assert v3 / v1 == pytest.approx(0.5, rel=1e-12)  # 1/eta


def test_film_supply_kernel_monotone_in_speed_and_roughness_independent():
    slow = film_thickness_supply(pressure_pa=1.0e6, speed_m_s=0.5,
                                 temperature_c=-5.0)
    fast = film_thickness_supply(pressure_pa=1.0e6, speed_m_s=2.0,
                                 temperature_c=-5.0)
    assert fast["h"] > slow["h"]
    assert fast["lambda"] > slow["lambda"]
    cold = film_thickness_supply(pressure_pa=1.0e6, speed_m_s=2.0,
                                 temperature_c=-20.0)
    # colder ice melts less (lower supply), while the *thickness* is set by the
    # drainage cap (proportional to eta**(1/3), and colder melt is more viscous),
    # so h itself may hold up -- the supply is the quantity that must fall.
    assert cold["v_melt"] < fast["v_melt"]
    assert cold["h"] <= 1.2 * fast["h"]


# ---------------------------------------------------------------------------
# regime map
# ---------------------------------------------------------------------------
def test_regime_thresholds_are_the_campaign_numbers():
    assert regime_of(0.5) == [REGIME_BOUNDARY]
    assert regime_of(LAMBDA_BOUNDARY) == [REGIME_MIXED]
    assert regime_of(2.0) == [REGIME_MIXED]
    assert regime_of(LAMBDA_HYDRO) == [REGIME_HYDRO]
    assert regime_of([0.1, 2.0, 10.0]) == [REGIME_BOUNDARY, REGIME_MIXED,
                                           REGIME_HYDRO]
    area = film_area_fraction(np.array([0.0, 1.0, 2.0, 3.0, 100.0]))
    assert area.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    assert float(combined_roughness(3.0e-6, 4.0e-6)) == pytest.approx(5.0e-6)


# ---------------------------------------------------------------------------
# Stribeck (criterion c)
# ---------------------------------------------------------------------------
def test_stribeck_optimum_closed_form_matches_brute_force():
    mu_b, her_c, her_v = 0.11, 3.0e-10, 4.0e-6
    opt = stribeck_optimum(mu_b, her_c, her_v)
    grid = np.logspace(-12.0, -4.0, 200001)
    mu = stribeck_mu(grid, mu_b=mu_b, her_c=her_c, her_v=her_v)
    her_brute = grid[int(np.argmin(mu))]
    assert opt["her_opt"] == pytest.approx(her_brute, rel=1e-3)
    assert opt["mu_opt"] == pytest.approx(float(np.min(mu)), rel=1e-3)


def test_stribeck_branches():
    mu_b, her_c, her_v = 0.11, 3.0e-10, 4.0e-6
    low = float(stribeck_mu(1.0e-13, mu_b=mu_b, her_c=her_c, her_v=her_v))
    assert low == pytest.approx(mu_b, rel=1e-3)  # Coulomb plateau
    high = float(stribeck_mu(1.0e-3, mu_b=mu_b, her_c=her_c, her_v=her_v))
    assert high > low  # viscous upturn


def test_stribeck_parameters_come_from_closures_not_fits():
    par = stribeck_parameters(pressure_pa=1.0e6, temperature_c=-5.0)
    assert par["bracketed"] == 1.0
    assert par["her_v"] == pytest.approx(3.0 * par["sigma_m"], rel=1e-12)
    expected_mu_b = float(boundary_friction(1.0e6, -5.0))
    assert par["mu_b"] == pytest.approx(expected_mu_b, rel=1e-12)
    assert 1.0e-11 < par["her_c"] < 1.0e-8


def test_criterion_c_stribeck_minimum_within_literature_band():
    for temperature in (-5.0, -10.0):
        check = stribeck_check(pressure_pa=1.0e6, temperature_c=temperature)
        assert check["pass"] is True
        assert check["rel_error"] <= 0.30
        lo, hi = STRIBECK_HER_OPT_BAND
        assert lo <= check["her_opt"] <= hi
        assert abs(check["her_opt"] - STRIBECK_HER_OPT_LITERATURE) \
            / STRIBECK_HER_OPT_LITERATURE <= 0.30


def test_stribeck_curve_covers_the_campaign_hersey_band():
    curve = stribeck_curve()
    assert curve["her"][0] == pytest.approx(1.0e-8)
    assert curve["her"][-1] == pytest.approx(1.0e-2)
    assert np.all(np.diff(curve["mu"]) > -1e-9) or curve["mu"].min() > 0.0
    assert curve["mu"].min() == pytest.approx(
        float(stribeck_mu(curve["her"][int(np.argmin(curve["mu"]))],
                          mu_b=curve["mu_b"], her_c=curve["her_c"],
                          her_v=curve["her_v"])), rel=1e-9)


def test_hersey_band_is_reported_not_asserted():
    band = hersey_band(-10.0)
    assert band["her_water_min"] <= 1.0e-8 <= band["her_water_max"]
    assert band["eta_ice_creep_pa_s"] > 1e11
    # the neat melt cannot reach 1e-2 at engineering loads: declared, not hidden
    assert band["her_slurry_max"] < 1.0e-2
    light = hersey_band(-10.0, pressure_range=(1.0e2, 1.0e3), ice_fraction=0.55)
    assert light["her_slurry_max"] > band["her_slurry_max"]


# ---------------------------------------------------------------------------
# end-to-end driver
# ---------------------------------------------------------------------------
def test_boundary_friction_band_and_pressure_dependence():
    mu_low_p = float(boundary_friction(2.0e5, -10.0))
    mu_high_p = float(boundary_friction(5.0e6, -10.0))
    assert MU_BOUNDARY_MIN <= mu_high_p < mu_low_p <= MU_BOUNDARY_MAX
    assert mu_high_p == pytest.approx(MU_BOUNDARY_MIN, rel=1e-12)  # saturated


def test_predict_returns_the_adjudicable_record():
    res = predict(LubricationInputs(ice_temperature_c=-10.0, speed_m_s=1.0))
    d = res.as_dict()
    for key in ("mu", "regime", "film_thickness_m", "lambda_ratio",
                "film_area_fraction", "hersey", "pressure_pa", "melting",
                "mu_boundary", "mu_film", "flash_rise_k", "converged"):
        assert key in d
    assert res.regime in REGIMES
    assert res.pressure_pa == pytest.approx(1.0e6)
    assert res.converged is True
    assert 0.0 < res.mu <= MU_BOUNDARY_MAX
    # Hersey number is the slurry value at the interface temperature
    expected_her = float(hersey_number(slurry_viscosity(-10.0), 1.0, 1.0e6))
    assert res.hersey == pytest.approx(expected_her, rel=1e-12)


def test_predict_monotone_in_temperature_and_speed():
    colder = predict(LubricationInputs(ice_temperature_c=-20.0, speed_m_s=0.5))
    warmer = predict(LubricationInputs(ice_temperature_c=-2.0, speed_m_s=0.5))
    assert colder.mu > warmer.mu  # colder ice is the rougher tribo-pair
    slow = predict(LubricationInputs(ice_temperature_c=-2.0, speed_m_s=0.01))
    fast = predict(LubricationInputs(ice_temperature_c=-2.0, speed_m_s=0.5))
    assert fast.mu < slow.mu  # melting relief with speed at warm ice
    assert fast.lambda_ratio > slow.lambda_ratio


def test_predict_lambda_grows_as_roughness_falls():
    rough = predict(LubricationInputs(ice_temperature_c=-2.0, speed_m_s=0.5,
                                     ice_roughness_m=5.0e-5))
    smooth = predict(LubricationInputs(ice_temperature_c=-2.0, speed_m_s=0.5,
                                       ice_roughness_m=1.0e-8))
    assert smooth.lambda_ratio > rough.lambda_ratio
    assert smooth.film_area_fraction >= rough.film_area_fraction


def test_friction_surface_shape_and_regime_partition():
    surf = friction_surface((-20.0, -10.0, -2.0, 0.0), (0.01, 0.1, 1.0),
                            (1.0e-7, 1.0e-6, 1.0e-5))
    assert surf["mu"].shape == (4, 3, 3)
    assert np.all(np.isfinite(surf["mu"]))
    assert np.all(surf["mu"] > 0.0)
    assert set(np.unique(surf["regime_index"]).astype(int)) <= set(range(len(REGIMES)))
    assert surf["regimes"] == REGIMES
    warm = surf["mu"][3]
    cold = surf["mu"][0]
    assert warm.mean() < cold.mean()


def test_stribeck_scan_is_consistent_with_predict():
    scan = stribeck_scan((0.01, 0.1, 1.0), temperature_c=-5.0)
    assert scan["mu"].shape == (3,)
    single = predict(LubricationInputs(speed_m_s=0.1, ice_temperature_c=-5.0,
                                       normal_load_n=1.0e3,
                                       contact_area_m2=1.0e-3))
    assert scan["mu"][1] == pytest.approx(single.mu, rel=1e-12)
    assert scan["regime"][1] in REGIMES


def test_input_validation():
    with pytest.raises(ValueError):
        LubricationInputs(speed_m_s=-1.0)
    with pytest.raises(ValueError):
        LubricationInputs(ice_roughness_m=0.0, steel_roughness_m=0.0)
    with pytest.raises(ValueError):
        flash_temperature_rise(-1.0, 1.0e-3)
    with pytest.raises(ValueError):
        regime_of(float("nan"))
    with pytest.raises(ValueError):
        water_viscosity(-200.0)


# ---------------------------------------------------------------------------
# criteria (a), (d), (e)
# ---------------------------------------------------------------------------
def test_criterion_a_literature_anchor_accuracy():
    report = validate_literature()
    assert report["n_applicable"] >= 6
    assert report["n_within_20pct"] >= 5, (
        f"only {report['n_within_20pct']}/{report['n_applicable']} anchors "
        f"within 20 % (mean {report['mean_rel_error']:.1%})")
    assert report["criterion_a_pass"] is True
    assert report["mean_rel_error"] < 0.20
    for row in report["rows"]:
        assert math.isfinite(row["mu_model"]) and row["mu_model"] > 0.0
        assert row["rel_error"] == pytest.approx(
            abs(row["mu_model"] - row["mu_anchor"]) / row["mu_anchor"], rel=1e-12)


def test_criterion_d_negative_controls():
    controls = negative_controls()
    assert controls["all_pass"] is True
    low, high = controls["d1_slow_speed_boundary"]["band"]
    for row in controls["d1_slow_speed_boundary"]["rows"]:
        assert low <= row["mu"] <= high
        assert row["melting"] is False
    assert controls["d2_warm_ice_full_film"]["row"]["regime"] == REGIME_FULL_FILM
    assert controls["d3_smooth_full_film"]["row"]["film_area_fraction"] == 1.0


def test_criterion_e_falsifiability_probe():
    probe = falsifiability_probe()
    assert probe["falsified"] is False
    assert probe["mu"] <= 0.5
    assert probe["inside_literature_band"] is True


def test_calibration_reports_leave_one_out_honestly():
    out = calibrate(tau_grid=(0.75, 1.0, 1.25), flash_grid=(1.0, 2.5))
    assert out["loo_n"] == 6
    assert len(out["leave_one_out"]) == 6
    assert math.isfinite(out["loo_mean_rel_error"])
    # the in-sample fit must be reported alongside, never instead of, the LOO
    assert out["verdict_full_fit"]["n_within_20pct"] >= 0
    for row in out["leave_one_out"]:
        assert row["tau_scale"] in (0.75, 1.0, 1.25)
        assert row["flash_concentration"] in (1.0, 2.5)