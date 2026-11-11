"""Properties and plastic-heat budget of the ice thermal-stress chain.

These tests pin down what the rest of the package assumes:

* the engineering property fits stay inside the tabulated bands (``ANCHORS``);
* the strength used by the fracture criterion *is* the library's Timco &
  Frederking law (no second strength model sneaks in);
* the Taylor-Quinney / Blok split conserves energy and the deposition volume is
  what it says it is;
* the adiabatic estimate agrees with the definition ``beta Q /(rho cp V)``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_material import pure_ice_strength
from tensordem.ice_thermal_stress.plastic_heat import (
    TAYLOR_QUINNEY_BETA,
    TAYLOR_QUINNEY_RANGE,
    adiabatic_temperature_rise,
    deposition_case,
    deposition_volume,
    dissipated_energy,
    hertz_patch_radius,
    layer_depth_sensitivity,
    partition_to_ice,
)
from tensordem.ice_thermal_stress.properties import (
    ANCHORS,
    ANCHOR_TOLERANCE,
    NU_ICE,
    TEMPERATURE_BAND_C,
    compressive_strength,
    ice_properties,
    thermal_stress_coefficient,
)


def test_property_fits_reproduce_the_tabulated_anchors():
    for temperature, row in ANCHORS.items():
        props = ice_properties(temperature)
        for key, expected in row.items():
            got = getattr(props, key)
            assert got == pytest.approx(expected, rel=ANCHOR_TOLERANCE), (temperature, key)


def test_property_trends_are_physical():
    warm = ice_properties(-1.0)
    cold = ice_properties(-30.0)
    assert cold.rho_kg_m3 > warm.rho_kg_m3          # ice densifies as it cools
    assert cold.cp_j_kg_k < warm.cp_j_kg_k          # Debye: cp falls with T
    assert cold.k_w_m_k > warm.k_w_m_k              # conductivity rises
    assert cold.youngs_modulus_pa > warm.youngs_modulus_pa
    assert cold.alpha_linear_per_k > warm.alpha_linear_per_k


def test_derived_quantities_are_consistent():
    props = ice_properties(-12.0)
    assert props.diffusivity_m2_s == pytest.approx(
        props.k_w_m_k / (props.rho_kg_m3 * props.cp_j_kg_k))
    assert props.effusivity_j_m2_k_sqrt_s ** 2 == pytest.approx(
        props.k_w_m_k * props.rho_kg_m3 * props.cp_j_kg_k)
    assert props.thermal_stress_coefficient_pa_k == pytest.approx(
        props.youngs_modulus_pa * props.alpha_linear_per_k / (1.0 - NU_ICE))
    assert thermal_stress_coefficient(-12.0) == pytest.approx(
        props.thermal_stress_coefficient_pa_k)


def test_properties_reject_temperatures_outside_the_fit_band():
    low, high = TEMPERATURE_BAND_C
    with pytest.raises(ValueError):
        ice_properties(low - 1.0)
    with pytest.raises(ValueError):
        ice_properties(high + 1.0)
    with pytest.raises(ValueError):
        ice_properties(float("nan"))


def test_strength_is_delegated_not_reinvented():
    for temperature in (-2.0, -5.0, -10.0, -20.0, -35.0):
        assert compressive_strength(temperature) == pytest.approx(
            pure_ice_strength(temperature), rel=1e-12)
    # and the brine branch matches the ice_material FYI law
    from tensordem.ice_material import compressive_strength_fyi

    assert compressive_strength(-10.0, salinity_ppt=5.0) == pytest.approx(
        compressive_strength_fyi(-10.0, 5.0), rel=1e-12)


def test_dissipated_energy_limits():
    assert dissipated_energy(100.0, 1.0, 0.0) == pytest.approx(50.0)
    assert dissipated_energy(100.0, 1.0, 1.0) == 0.0
    with pytest.raises(ValueError):
        dissipated_energy(100.0, 1.0, 1.2)
    with pytest.raises(ValueError):
        dissipated_energy(0.0, 1.0, 0.2)


def test_hertz_patch_is_the_classical_cube_root():
    force, radius, modulus = 5.0e4, 0.30, 5.06e9
    a = hertz_patch_radius(force, radius, modulus)
    assert a == pytest.approx((3.0 * force * radius / (4.0 * modulus)) ** (1.0 / 3.0))
    # doubling the load grows the radius by 2^(1/3)
    assert hertz_patch_radius(2.0 * force, radius, modulus) / a == pytest.approx(2.0 ** (1 / 3))


def test_blok_split_conserves_the_dissipated_energy():
    props = ice_properties(-10.0)
    assert partition_to_ice(props, None) == 1.0                 # adiabatic face
    assert partition_to_ice(props, props) == pytest.approx(0.5)  # ice on ice
    # a steel face takes almost nothing: the ice keeps the heat
    steel = ice_properties(-10.0).__class__(
        temperature_c=-10.0, rho_kg_m3=7850.0, cp_j_kg_k=460.0, k_w_m_k=50.0,
        youngs_modulus_pa=210e9, poisson=0.30, alpha_linear_per_k=1.2e-5)
    assert 0.05 < partition_to_ice(props, steel) < 0.3


def test_adiabatic_rise_matches_its_definition():
    props = ice_properties(-10.0)
    energy, volume = 40.0, 1.0e-6
    expect = TAYLOR_QUINNEY_BETA * energy / (props.volumetric_heat_capacity_j_m3_k * volume)
    assert adiabatic_temperature_rise(energy, volume, props) == pytest.approx(expect)


def test_deposition_case_is_self_consistent():
    budget = deposition_case(
        mass_kg=103.8, impact_speed_m_s=1.0, restitution=0.3, peak_force_n=1.5e5,
        effective_radius_m=0.3, effective_modulus_pa=1.07e10,
        temperature_c=-10.0, contact_duration_s=3e-3, depth_model="fraction",
        layer_fraction=0.15)
    assert budget.deposition_volume_m3 == pytest.approx(
        deposition_volume(budget.patch_radius_m, budget.layer_depth_m))
    assert budget.layer_depth_m == pytest.approx(0.15 * budget.patch_radius_m)
    assert budget.delta_t_adiabatic_k > 0.0
    # the deposition Fourier number must be small for the patch to stay adiabatic
    assert budget.fourier_number_deposition < 1e-2
    assert budget.plastic_heat_ice_j == pytest.approx(
        budget.partition_to_ice * budget.dissipated_energy_j)
    assert TAYLOR_QUINNEY_RANGE[0] <= budget.beta <= TAYLOR_QUINNEY_RANGE[1]


def test_layer_depth_sensitivity_scales_like_one_over_depth():
    rows = layer_depth_sensitivity(
        mass_kg=103.8, impact_speed_m_s=1.0, restitution=0.3, peak_force_n=1.5e5,
        effective_radius_m=0.3, effective_modulus_pa=1.07e10,
        temperature_c=-10.0, contact_duration_s=3e-3,
        fractions=(0.1, 0.2, 0.4))
    depths = np.array([r.layer_depth_m for r in rows])
    rises = np.array([r.delta_t_adiabatic_k for r in rows])
    # Delta T ~ 1/depth for a fixed deposition power over a fixed patch area
    ratios = rises * depths
    assert np.allclose(ratios, ratios[0], rtol=1e-9)
    assert np.all(np.diff(depths) > 0.0)


def test_beta_default_is_inside_its_declared_band():
    assert TAYLOR_QUINNEY_RANGE[0] <= TAYLOR_QUINNEY_BETA <= TAYLOR_QUINNEY_RANGE[1]
    assert math.isfinite(TAYLOR_QUINNEY_BETA)