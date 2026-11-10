"""Physics of the phase-change closure: Clausius-Clapeyron and the friction law.

Every assertion here is against an independent reference, not against the code:

* the Clapeyron slope against its three material properties and against the
  tabulated 0.0072 K/atm;
* ``T_m(p)`` and the pressure-melting threshold against hand-checked arithmetic;
* the exponential law against a finite-difference derivative and against a
  synthetic set of anchors (a fit that is off by any amount fails here);
* monotonicity of the melt fraction in the two variables that matter
  (temperature and pressure) -- this is the mechanism, so it is tested directly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_friction_thermo import (
    BETA_RANGE_PER_K,
    CC_SLOPE_K_PER_MPA,
    CC_SLOPE_K_PER_PA,
    ICE,
    ICE_DENSITY_KG_M3,
    LATENT_HEAT_J_KG,
    MELT_TEMPERATURE_K,
    STEEL,
    WATER_DENSITY_KG_M3,
    ContactFrictionModel,
    blok_partition,
    cc_slope_k_per_pa,
    clapeyron_slope_check,
    film_latent_energy_j_m2,
    fit_exponential_temperature_law,
    melt_fraction,
    melting_temperature_c,
    mu_exponential,
    pressure_melting_threshold_pa,
    real_contact_pressure,
    stefan_melt_rate_m_per_s,
)
from tensordem.ice_steel_friction import CALIBRATION_ANCHORS


# ---------------------------------------------------------------------------
# Clausius-Clapeyron
# ---------------------------------------------------------------------------
def test_clapeyron_slope_matches_its_properties():
    """``dT_m/dp = T dv / L`` reproduced by hand from the material properties."""
    dv = 1.0 / WATER_DENSITY_KG_M3 - 1.0 / ICE_DENSITY_KG_M3
    expected = MELT_TEMPERATURE_K * dv / LATENT_HEAT_J_KG
    assert cc_slope_k_per_pa() == pytest.approx(expected, rel=1e-15)
    assert CC_SLOPE_K_PER_PA == pytest.approx(-7.40e-8, rel=1e-2)
    assert CC_SLOPE_K_PER_MPA == pytest.approx(0.0740, rel=1e-2)


def test_clapeyron_slope_matches_the_tabulated_atmosphere_value():
    """The derived slope is 0.0075 K/atm, 4 % above the tabulated 0.0072."""
    assert clapeyron_slope_check() < 0.05
    assert -CC_SLOPE_K_PER_PA * 101325.0 == pytest.approx(0.0074, rel=5e-2)


def test_melting_point_is_depressed_by_pressure():
    assert melting_temperature_c(0.0) == pytest.approx(0.0, abs=1e-12)
    assert melting_temperature_c(2.0e7) == pytest.approx(-1.480, rel=1e-2)
    assert melting_temperature_c(1.0e8) == pytest.approx(-7.402, rel=1e-2)
    # strictly decreasing in pressure
    pressures = np.array([0.0, 1.0e6, 2.0e7, 1.0e8, 2.0e8])
    temperatures = [melting_temperature_c(p) for p in pressures]
    assert all(b < a for a, b in zip(temperatures, temperatures[1:]))


def test_pressure_melting_threshold_inverts_the_melting_curve():
    for temperature in (-1.0, -2.0, -5.0, -10.0, -30.0):
        pressure = pressure_melting_threshold_pa(temperature)
        assert melting_temperature_c(pressure) == pytest.approx(temperature, rel=1e-12)
    assert pressure_melting_threshold_pa(-5.0) / 1.0e6 == pytest.approx(67.5, rel=1e-2)
    with pytest.raises(ValueError):
        pressure_melting_threshold_pa(0.5)


def test_melt_fraction_rises_with_temperature_and_with_pressure():
    # temperature direction, fixed pressure
    series = [melt_fraction(t, 2.0e7) for t in (-10.0, -5.0, -3.0, -1.5, -1.0)]
    assert all(b >= a for a, b in zip(series, series[1:]))
    assert series[0] == 0.0 and series[-1] == 1.0
    # pressure direction, fixed temperature: the Clausius-Clapeyron signature
    pressures = np.array([0.0, 1.0e6, 5.0e6, 2.0e7, 5.0e7, 1.0e8])
    pressureseries = [melt_fraction(-3.0, p) for p in pressures]
    assert all(b >= a for a, b in zip(pressureseries, pressureseries[1:]))
    assert pressureseries[0] == 0.0
    assert pressureseries[-1] == 1.0


def test_melt_fraction_is_bounded_and_continuous_in_the_onset_width():
    width = 1.0
    for temperature in np.linspace(-5.0, 0.0, 51):
        for pressure in (0.0, 2.0e7, 1.0e8):
            s = melt_fraction(float(temperature), pressure, width)
            assert 0.0 <= s <= 1.0
    # continuity: neighbouring temperatures differ by at most dt/width
    ts = np.linspace(-3.0, -0.5, 251)
    values = [melt_fraction(float(t), 2.0e7, width) for t in ts]
    steps = np.abs(np.diff(values))
    assert steps.max() <= (ts[1] - ts[0]) / width + 1e-12


def test_real_contact_pressure_saturates_at_hardness():
    p_real, fraction = real_contact_pressure(1.0e6, 2.0e7)
    assert p_real == pytest.approx(2.0e7)          # plastic: saturates at H
    assert fraction == pytest.approx(0.05)
    # above the hardness the area fraction is one and the pressure is nominal
    p_real2, fraction2 = real_contact_pressure(4.0e7, 2.0e7)
    assert p_real2 == pytest.approx(4.0e7)
    assert fraction2 == pytest.approx(1.0)
    # no load, no contact
    assert real_contact_pressure(0.0, 2.0e7) == (0.0, 0.0)


def test_stefan_melt_rate_and_film_energy():
    flux = 1.0e6
    rate = stefan_melt_rate_m_per_s(flux)
    assert rate == pytest.approx(flux / (ICE_DENSITY_KG_M3 * LATENT_HEAT_J_KG), rel=1e-15)
    assert stefan_melt_rate_m_per_s(-5.0) == 0.0
    assert film_latent_energy_j_m2(1.0e-6) == pytest.approx(
        1.0e-6 * ICE_DENSITY_KG_M3 * LATENT_HEAT_J_KG, rel=1e-15)


# ---------------------------------------------------------------------------
# material properties / partition
# ---------------------------------------------------------------------------
def test_blok_partition_is_the_effusivity_ratio():
    gamma = blok_partition(ICE, STEEL)
    e_i = ICE.effusivity_j_m2_k_sqrt_s
    e_s = STEEL.effusivity_j_m2_k_sqrt_s
    assert gamma == pytest.approx(e_i / (e_i + e_s), rel=1e-15)
    assert 0.10 < gamma < 0.20      # steel takes ~86 % of the frictional heat
    assert ICE.alpha_m2_s == pytest.approx(ICE.k_w_m_k / (ICE.rho_kg_m3 * ICE.cp_j_kg_k))


# ---------------------------------------------------------------------------
# the friction law
# ---------------------------------------------------------------------------
def test_exponential_law_matches_its_definition_and_derivative():
    mu_ref, beta, t_ref = 0.07, 0.027, -2.0
    for temperature in (-30.0, -10.0, -2.5, -0.1):
        value = mu_exponential(temperature, mu_ref, beta, t_ref)
        assert value == pytest.approx(mu_ref * math.exp(-beta * (temperature - t_ref)),
                                      rel=1e-15)
    # analytic derivative vs central difference
    model = ContactFrictionModel(mu_ref=mu_ref, beta_per_k=beta, t_ref_c=t_ref,
                                 allow_melting=False)
    for temperature in (-20.0, -10.0, -3.0):
        h = 1e-4
        numerical = (model.mu(temperature + h, 0.0) - model.mu(temperature - h, 0.0)) / (2 * h)
        assert model.dmu_d_temperature(temperature, 0.0) == pytest.approx(numerical, rel=1e-6)


def test_law_is_strictly_decreasing_in_temperature_and_pressure():
    model = ContactFrictionModel(allow_melting=False)
    temperatures = np.linspace(-30.0, -0.5, 40)
    values = [model.mu(float(t), 2.0e7) for t in temperatures]
    assert all(b < a for a, b in zip(values, values[1:]))
    model_on = ContactFrictionModel()
    pressures = [0.0, 1.0e6, 2.0e7, 5.0e7, 1.0e8]
    pressed = [model_on.mu(-3.0, p) for p in pressures]
    assert all(b <= a for a, b in zip(pressed, pressed[1:]))
    assert pressed[-1] < pressed[0]


def test_melting_branch_never_increases_friction():
    """Enabling the phase change can only lower mu -- a structural invariant."""
    model_off = ContactFrictionModel(allow_melting=False)
    model_on = ContactFrictionModel()
    for temperature in np.linspace(-20.0, -0.5, 25):
        for pressure in (0.0, 2.0e7, 6.0e7):
            assert model_on.mu(float(temperature), pressure) <= \
                model_off.mu(float(temperature), pressure) + 1e-15


def test_wet_ratio_controls_the_melted_coefficient():
    dry = ContactFrictionModel(wet_ratio=0.0)
    control = ContactFrictionModel(wet_ratio=1.0)
    melted = -0.1   # well inside the melting branch at 100 MPa
    assert dry.mu(melted, 1.0e8) == pytest.approx(dry.mu_floor, rel=1e-12)
    assert control.mu(melted, 1.0e8) == pytest.approx(control.mu_dry(melted), rel=1e-12)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        ContactFrictionModel(beta_per_k=-0.01)
    with pytest.raises(ValueError):
        ContactFrictionModel(beta_per_k=BETA_RANGE_PER_K[1] * 2)
    with pytest.raises(ValueError):
        ContactFrictionModel(t_ref_c=1.0)
    with pytest.raises(ValueError):
        ContactFrictionModel(wet_ratio=1.5)
    with pytest.raises(ValueError):
        ContactFrictionModel(hardness_pa=0.0)
    with pytest.raises(ValueError):
        mu_exponential(-5.0, 0.07, 0.0)


def test_fit_recovers_a_synthetic_law_exactly():
    truth = ContactFrictionModel(mu_ref=0.08, beta_per_k=0.031, t_ref_c=-5.0,
                                 allow_melting=False)
    rows = [(t, truth.mu_dry(t), 0.005) for t in (-30.0, -20.0, -12.0, -6.0, -1.0)]
    fit = fit_exponential_temperature_law(rows, t_ref_c=-5.0)
    assert fit["mu_ref"] == pytest.approx(0.08, rel=1e-10)
    assert fit["beta_per_k"] == pytest.approx(0.031, rel=1e-10)
    assert fit["max_rel_residual"] < 1e-10


def test_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        fit_exponential_temperature_law([(-5.0, 0.08), (-5.0, 0.09), (-5.0, 0.1)])
    with pytest.raises(ValueError):
        fit_exponential_temperature_law([(-5.0, 0.08), (-6.0, 0.09)])


def test_calibration_on_the_barnes_sweep_is_documented_and_physical():
    rows = [row for row in CALIBRATION_ANCHORS if "Barnes" in row.source]
    fit = fit_exponential_temperature_law(rows, t_ref_c=-10.0)
    assert BETA_RANGE_PER_K[0] <= fit["beta_per_k"] <= BETA_RANGE_PER_K[1]
    # the exponential form describes a 28 K temperature sweep to better than 10 %
    assert fit["max_rel_residual"] < 0.10
    assert fit["n_anchors"] == len(rows)