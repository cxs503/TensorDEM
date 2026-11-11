"""Thermal stress, superposition/fracture criterion and the coupled impact.

The invariants pinned here are the ones that would silently destroy the
physics if they broke:

* the restrained and free-slab bounds differ by exactly a factor of two, and a
  linear gradient reproduces the free-slab bending peak;
* a self-equilibrated (restrained) thermal field carries no membrane force;
* heating softens the ice *and* adds compressive stress -- both act through the
  same strength surface, never double counted;
* the coupled impact conserves energy (dissipated -> enthalpy) and reproduces
  the coefficient of restitution it was asked for;
* the thermal term is *bounded* by the strength: the breakage increment may not
  exceed ``|sigma_th| / sigma_f``.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.ice_material import pure_ice_strength
from tensordem.ice_thermal_stress.collision import (
    BlockImpactCase,
    layer_depth_sweep,
    simulate_block_impact,
)
from tensordem.ice_thermal_stress.fracture import (
    RATIO_COMPRESSION_TO_TENSION,
    strength_budget_bound,
    breakage_increment_curve,
    critical_heating_for_crushing,
    failure_margin,
    superpose,
    tensile_strength,
    thermal_breakage_increment,
    thermal_margin_fraction,
)
from tensordem.ice_thermal_stress.properties import ice_properties
from tensordem.ice_thermal_stress.thermal_stress import (
    field_stress_from_temperature,
    free_slab_gradient_peak,
    restrained_gradient_stress,
    restrained_uniform_stress,
    restrained_vs_free_ratio,
    stress_equilibrium_residual,
    thermal_shock_critical_delta_t,
)


# ---------------------------------------------------------------------------
# thermal stress
# ---------------------------------------------------------------------------
def test_restrained_and_free_bounds_differ_by_two():
    props = ice_properties(-10.0)
    for delta_t in (1.0, 5.0, 20.0):
        restrained = float(restrained_uniform_stress(delta_t, props))
        free = float(free_slab_gradient_peak(delta_t, props))
        assert restrained < 0.0 < free                 # heating compresses, cooling tensions
        assert abs(restrained) / free == pytest.approx(2.0)
    assert restrained_vs_free_ratio() == pytest.approx(2.0)


def test_linear_gradient_reproduces_the_free_slab_solution():
    props = ice_properties(-10.0)
    linear = np.linspace(-10.0, 10.0, 401)             # dT = 20 K across the section
    sigma = restrained_gradient_stress(linear, props)
    assert np.max(np.abs(sigma)) == pytest.approx(float(free_slab_gradient_peak(20.0, props)))
    assert stress_equilibrium_residual(sigma) < 1e-12


def test_restrained_field_is_self_equilibrated():
    props = ice_properties(-12.0)
    x = np.linspace(0.0, 0.02, 401)
    profile = 8.0 * np.exp(-x / 1.5e-3)
    sigma = restrained_gradient_stress(profile, props)
    assert stress_equilibrium_residual(sigma) < 1e-12
    assert sigma[0] < 0.0 and sigma[-1] > 0.0          # hot face compressive
    # free branch is the same shape, halved
    free = field_stress_from_temperature(profile, props, restraint="free")
    assert np.allclose(free, 0.5 * sigma)


def test_critical_temperature_jump_is_the_strength_over_the_coefficient():
    for temperature in (-2.0, -5.0, -10.0, -20.0):
        sigma_f = float(pure_ice_strength(temperature)) * 1.0e6
        props = ice_properties(temperature)
        expect = sigma_f * (1.0 - props.poisson) / (
            props.youngs_modulus_pa * props.alpha_linear_per_k)
        assert thermal_shock_critical_delta_t(sigma_f, temperature) == pytest.approx(expect)
        assert thermal_shock_critical_delta_t(
            sigma_f, temperature, restraint="free") == pytest.approx(2.0 * expect)


# ---------------------------------------------------------------------------
# superposition and fracture
# ---------------------------------------------------------------------------
def test_superposition_and_branch_selection():
    assert superpose(-40.0e6, -5.0e6) == pytest.approx(-45.0e6)
    comp = failure_margin(-20.0e6, -5.0e6, -10.0)
    assert comp.branch == "compression" and comp.sigma_total_pa == pytest.approx(-25.0e6)
    tens = failure_margin(+2.0e6, 0.0, -10.0)
    assert tens.branch == "tension"
    # ice is three times weaker in tension than in compression
    assert tens.strength_pa == pytest.approx(comp.strength_pa / RATIO_COMPRESSION_TO_TENSION)


def test_tensile_strength_tracks_the_compressive_law():
    for temperature in (-2.0, -10.0, -25.0):
        assert tensile_strength(temperature) == pytest.approx(
            float(pure_ice_strength(temperature)) * 1.0e6 / RATIO_COMPRESSION_TO_TENSION)


def test_heating_softens_and_loads_at_the_same_time():
    cold = failure_margin(-10.0e6, 0.0, -10.0, delta_t_k=0.0)
    hot = failure_margin(-10.0e6, -3.0e6, -10.0, delta_t_k=6.0)
    assert hot.strength_pa < cold.strength_pa            # sigma_f(T) softening
    assert abs(hot.sigma_total_pa) > abs(cold.sigma_total_pa)
    assert hot.margin_pa > cold.margin_pa


def test_melting_is_flagged_not_silently_modelled():
    state = failure_margin(-5.0e6, -20.0e6, -2.0, delta_t_k=20.0)
    assert state.melted is True
    # the strength is clamped just under the melting point, never above it
    assert state.strength_pa > 0.0


def test_thermal_margin_fraction_is_zero_without_thermal_load():
    assert thermal_margin_fraction(-10.0e6, 0.0, -10.0) == 0.0
    assert thermal_margin_fraction(-10.0e6, -5.0e6, -10.0) == pytest.approx(1.0 / 3.0)
    assert np.isinf(thermal_margin_fraction(-1.0e9, -1.0e6, -10.0))


def test_breakage_increment_is_monotone_and_bounded():
    sigma_f = float(pure_ice_strength(-10.0)) * 1.0e6
    # a stress population spread uniformly over the whole strength budget
    population = -sigma_f * np.linspace(0.0, 1.0, 201)
    curve = breakage_increment_curve(population, np.linspace(0.0, 30.0, 31), -10.0)
    increments = curve["increment"]
    assert increments[0] == 0.0
    assert np.all(np.diff(increments) >= -1e-12)
    bound = np.array([strength_budget_bound(float(th), -10.0, delta_t_k=float(dt))
                      for th, dt in zip(curve["sigma_thermal_pa"], curve["delta_t_k"])])
    # the bound is tight for a uniform population: the increment cannot exceed it
    assert np.all(increments <= bound + 1e-9)
    assert increments[-1] > 0.99                        # 30 K closes the whole budget
    assert increments[3] > 0.0                          # 3 K already breaks something


def test_critical_heating_matches_the_inverse_of_the_increment():
    sigma_f = float(pure_ice_strength(-10.0)) * 1.0e6
    mech = -0.5 * sigma_f
    dt_c = critical_heating_for_crushing(mech, -10.0)
    props = ice_properties(-10.0)
    assert abs(float(restrained_uniform_stress(dt_c, props))) == pytest.approx(0.5 * sigma_f)
    assert critical_heating_for_crushing(-2.0 * sigma_f, -10.0) == 0.0


# ---------------------------------------------------------------------------
# coupled impact
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def impact():
    return simulate_block_impact(BlockImpactCase(cells=120))


def test_impact_reproduces_the_prescribed_restitution(impact):
    assert impact.restitution_measured == pytest.approx(0.30, abs=2e-3)


def test_impact_energy_ledger_closes(impact):
    ledger = impact.energy_ledger
    assert ledger["dissipated_energy_relative_error"] < 0.01
    assert ledger["heat_to_enthalpy_relative_error"] < 1e-6
    # dissipated cannot exceed the available kinetic energy
    assert impact.dissipated_energy_j <= ledger["kinetic_energy_j"]
    assert impact.plastic_heat_ice_j <= impact.dissipated_energy_j


def test_thermal_field_and_stress_are_consistent_with_the_heat(impact):
    props = ice_properties(impact.case.bulk_temperature_c)
    assert impact.peak_delta_t_k > 0.0
    # lumped-capacity estimate is the upper bound of the conducted peak
    assert impact.peak_delta_t_k <= impact.peak_delta_t_adiabatic_k * 1.05
    assert impact.peak_delta_t_adiabatic_k == pytest.approx(
        impact.plastic_heat_ice_j / (
            props.volumetric_heat_capacity_j_m3_k * impact.deposition_volume_m3))
    assert impact.sigma_thermal_patch_pa == pytest.approx(
        float(restrained_uniform_stress(impact.peak_delta_t_k, props)))
    assert impact.sigma_thermal_free_pa == pytest.approx(0.5 * impact.sigma_thermal_patch_pa)
    # the stress field is self-equilibrated and peaks at the heated face
    assert stress_equilibrium_residual(
        impact.sigma_thermal_profile_pa, impact.cell_widths_m) < 1e-10
    assert impact.sigma_thermal_profile_pa[0] < 0.0
    # heating can only soften the ice
    assert impact.sigma_f_heated_pa < impact.sigma_f_bulk_pa


def test_breakage_increment_is_bounded_by_the_strength_budget(impact):
    got = impact.breakage_coupled["increment"]
    bound = strength_budget_bound(impact.sigma_thermal_patch_pa,
                                  impact.case.bulk_temperature_c,
                                  delta_t_k=impact.peak_delta_t_k)
    assert impact.breakage_coupled["increment"] >= impact.breakage_plain["increment"]
    assert 0.0 <= got <= bound + 1e-12
    assert impact.breakage_bound["increment"] == pytest.approx(bound)


def test_zero_beta_turns_the_coupling_off():
    off = simulate_block_impact(BlockImpactCase(beta=1e-12, cells=80))
    assert off.peak_delta_t_k == pytest.approx(0.0, abs=1e-9)
    assert off.breakage_coupled["increment"] == pytest.approx(
        off.breakage_plain["increment"])


def test_thermal_numbers_grow_with_impact_speed():
    slow = simulate_block_impact(BlockImpactCase(impact_speed_m_s=0.5, cells=80))
    fast = simulate_block_impact(BlockImpactCase(impact_speed_m_s=2.0, cells=80))
    assert fast.peak_delta_t_k > slow.peak_delta_t_k
    assert abs(fast.sigma_thermal_patch_pa) > abs(slow.sigma_thermal_patch_pa)
    assert fast.dissipated_energy_j > slow.dissipated_energy_j


def test_layer_depth_sweep_obeys_the_inverse_scaling():
    rows = layer_depth_sweep(fractions=(0.1, 0.2, 0.4), cells=80)
    rises = np.array([r["peak_delta_t_k"] for r in rows])
    depths = np.array([r["layer_depth_m"] for r in rows])
    assert np.all(np.diff(rises) < 0.0)                 # thicker layer -> cooler
    # Delta T ~ 1/depth is the adiabatic scaling; conduction weakens it slightly
    assert 0.8 < (rises * depths)[0] / (rises * depths)[-1] < 3.0


def test_case_validation():
    with pytest.raises(ValueError):
        BlockImpactCase(impact_speed_m_s=0.0)
    with pytest.raises(ValueError):
        BlockImpactCase(restitution=1.5)
    with pytest.raises(ValueError):
        BlockImpactCase(beta=0.0)
    with pytest.raises(ValueError):
        BlockImpactCase(layer_fraction=2.0)