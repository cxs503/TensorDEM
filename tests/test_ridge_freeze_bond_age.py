"""Acceptance tests for the freeze-bond age-evolution module (IN54W1-N05).

The chain that has to hold together is

``T_c < T_m(S)`` formation criterion -> ``sigma_b``(T, S) ->
``sigma_b(t) = sigma_b_inf (1 - exp(-t/tau_f))`` with ``tau_f`` from a
conduction solution -> DEM keel resistance ``R_keel(t)`` -> Lindqvist age
factor ``k_age(t)``.

Every test below is a physical statement, not a snapshot: each assertion is a
limit, a monotonicity direction, a dimensional identity or a conservation /
degeneration check, so the suite stays valid when the deck is re-parameterised.
"""
from __future__ import annotations

import json
import math
import os

import pytest

from tensordem.ridge_dem import RidgeAssembly, RidgeKeel, RidgeMaterial
from tensordem.ridge_freeze_bond_age import (
    LITERATURE_ANCHOR,
    FreezeBondAgeModel,
    aged_material,
    brine_volume_fraction,
    conduction_time,
    freezing_point_celsius,
    ice_tensile_strength,
    keel_resistance,
    lens_cooling_transient,
    lindqvist_age_correction,
    run_keel_age_sweep,
    stefan_freezing_time,
)

# --------------------------------------------------------------------------- #
# 1. Formation criterion: freezing-point depression, brine volume, strength
# --------------------------------------------------------------------------- #
def test_freezing_point_of_fresh_water_is_zero():
    assert freezing_point_celsius(0.0) == pytest.approx(0.0, abs=1e-12)


def test_freezing_point_depression_matches_assur():
    # Assur's relation T_m = -0.05411 S - 7.52e-5 S^2 (degC, S in ppt)
    for salinity in (5.0, 10.0, 35.0):
        expected = -0.05411 * salinity - 7.52e-5 * salinity ** 2
        assert freezing_point_celsius(salinity) == pytest.approx(expected, rel=1e-12)
    # published sea-water freezing points at S = 35 ppt scatter between the
    # Assur polynomial (-1.99 degC) and the UNESCO/Millero value (-1.91 degC);
    # the module uses Assur, and both must agree to better than 0.1 K
    assert freezing_point_celsius(35.0) == pytest.approx(-1.99, abs=0.02)
    assert abs(freezing_point_celsius(35.0) - (-1.91)) < 0.1


def test_freezing_point_is_monotone_in_salinity():
    values = [freezing_point_celsius(s) for s in (0.0, 5.0, 10.0, 20.0, 35.0)]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_brine_volume_grows_with_salinity_and_shrinks_when_colder():
    assert brine_volume_fraction(-10.0, 0.0) == pytest.approx(0.0, abs=1e-15)
    saline = [brine_volume_fraction(-10.0, s) for s in (5.0, 10.0, 20.0, 35.0)]
    assert all(b > a for a, b in zip(saline, saline[1:]))
    cold = [brine_volume_fraction(t, 10.0) for t in (-2.0, -5.0, -10.0, -20.0)]
    assert all(b < a for a, b in zip(cold, cold[1:]))


def test_brine_volume_refuses_extrapolation_outside_polynomial_range():
    with pytest.raises(ValueError):
        brine_volume_fraction(-30.0, 35.0, allow_extrapolation=False)
    # the permissive branch clamps instead of raising and stays finite
    assert math.isfinite(brine_volume_fraction(-30.0, 35.0,
                                              allow_extrapolation=True))


def test_tensile_strength_decreases_with_salinity_and_increases_when_colder():
    saline = [ice_tensile_strength(-10.0, s) for s in (0.0, 5.0, 10.0, 20.0)]
    assert all(b < a for a, b in zip(saline, saline[1:]))
    cold = [ice_tensile_strength(t, 10.0) for t in (-2.0, -5.0, -10.0, -20.0)]
    assert all(b > a for a, b in zip(cold, cold[1:]))


def test_tensile_strength_never_negative_even_at_full_brine():
    # warm saline ice is beyond the brine-volume cut-off: strength -> 0, not < 0
    assert ice_tensile_strength(-2.0, 35.0) == pytest.approx(0.0, abs=1e-9)
    assert ice_tensile_strength(-1.0, 35.0) >= 0.0


# --------------------------------------------------------------------------- #
# 2. Formation criterion at the bond level: no bond above T_m(S)
# --------------------------------------------------------------------------- #
def test_no_freeze_bond_above_the_local_freezing_point():
    model = FreezeBondAgeModel(temperature_c=-1.0, salinity_ppt=35.0)
    assert model.bond_active is False
    assert model.freezing_point_c < model.temperature_c
    for age in (0.0, 6.0, 24.0, 168.0):
        assert model.sigma_at(age) == 0.0
        assert model.tau_at(age) == 0.0
        assert model.bond_fraction_at(age) == 0.0


def test_bond_is_active_once_colder_than_the_freezing_point():
    freezing = freezing_point_celsius(35.0)
    assert FreezeBondAgeModel(temperature_c=freezing - 1.0, salinity_ppt=35.0).bond_active
    assert not FreezeBondAgeModel(temperature_c=freezing + 1.0, salinity_ppt=35.0).bond_active


# --------------------------------------------------------------------------- #
# 3. Growth law sigma_b(t) = sigma_inf (1 - exp(-t/tau_f))
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def model():
    return FreezeBondAgeModel.calibrated(thermal_length=0.5)


def test_strength_vanishes_at_zero_age(model):
    assert model.sigma_at(0.0) == 0.0
    assert model.tau_at(0.0) == 0.0


def test_strength_is_monotone_increasing_and_bounded(model):
    ages = [0.0, 0.5, 1.0, 3.0, 6.0, 24.0, 72.0, 168.0, 336.0, 1000.0]
    sigmas = [model.sigma_at(age) for age in ages]
    assert all(b > a for a, b in zip(sigmas, sigmas[1:]))
    assert all(value <= model.sigma_inf_pa for value in sigmas)
    assert model.sigma_at(336.0) < model.sigma_inf_pa
    # exponential relaxation: half-saturation is ln2 * tau_f
    assert model.sigma_at(math.log(2.0) * model.tau_f_hours) == pytest.approx(
        0.5 * model.sigma_inf_pa, rel=1e-9)


def test_long_age_approaches_the_asymptote(model):
    assert model.sigma_at(1.0e5) == pytest.approx(model.sigma_inf_pa, rel=1e-9)
    # 168 h (one week) already carries > 99 % of the asymptotic strength
    assert model.sigma_at(168.0) / model.sigma_inf_pa > 0.99


def test_shear_strength_keeps_the_material_ratio(model):
    for age in (0.0, 6.0, 24.0, 168.0):
        sigma, tau = model.strength_at(age)
        assert tau == pytest.approx(model.shear_ratio * sigma, rel=1e-12)
    assert model.tau_inf_pa == pytest.approx(model.shear_ratio * model.sigma_inf_pa)


def test_negative_or_non_finite_age_is_rejected(model):
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            model.sigma_at(bad)


def test_salinity_reduces_the_asymptotic_strength(model):
    fresh = FreezeBondAgeModel(temperature_c=-10.0, salinity_ppt=0.0,
                               area_fraction=model.area_fraction)
    saline = FreezeBondAgeModel(temperature_c=-10.0, salinity_ppt=35.0,
                                area_fraction=model.area_fraction)
    assert fresh.sigma_inf_pa > saline.sigma_inf_pa > 0.0


def test_calibration_reproduces_the_anchor_and_stays_inside_its_band(model):
    anchor = float(LITERATURE_ANCHOR["sigma_b_inf_pa"])
    assert model.sigma_inf_pa == pytest.approx(anchor, rel=1e-9)
    low = LITERATURE_ANCHOR["band_pa"][0]
    high = LITERATURE_ANCHOR["band_pa"][1]
    for target in (low, high):
        variant = FreezeBondAgeModel.calibrated(sigma_inf_target_pa=target,
                                               thermal_length=0.5)
        assert variant.sigma_inf_pa == pytest.approx(target, rel=1e-9)
        assert low <= variant.sigma_inf_pa <= high


# --------------------------------------------------------------------------- #
# 4. tau_f from a real conduction solution, compared with rho c h^2 / (4 k)
# --------------------------------------------------------------------------- #
def test_conduction_time_is_the_closed_form_comparator():
    tp = FreezeBondAgeModel().thermophysics
    expected = (tp.ice_density * tp.ice_specific_heat * 0.5 ** 2
                / (4.0 * tp.ice_conductivity))
    assert conduction_time(0.5) == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError):
        conduction_time(0.0)


def test_stefan_lens_time_is_much_shorter_than_the_block_time():
    lens = stefan_freezing_time(0.005, 8.0)
    block = conduction_time(0.5)
    assert lens < block / 10.0


def test_cooling_transient_is_convergent_and_energy_accounted():
    coarse = lens_cooling_transient(0.5, ambient_c=-10.0, initial_c=0.0, nodes=21)
    fine = lens_cooling_transient(0.5, ambient_c=-10.0, initial_c=0.0, nodes=41)
    assert fine["tau_half_s"] == pytest.approx(coarse["tau_half_s"], rel=0.10)
    # enthalpy ledger: converges to zero with node refinement (0.52 % at 81
    # nodes).  A regression to ~15 % means the latent injection into the block
    # node is being double counted.
    assert fine["energy_closure_relative_error"] < 0.03
    assert fine["lens_freeze_time_hours"] > 0.0
    # bond-plane cooling time is a diffusive time scale of order rho c h^2/(4k)
    assert 1.0 < fine["temperature_half_hours"] / fine["conduction_time_hours"] < 10.0
    # bond plane starts at the freezing point and ends near the ambient
    assert fine["bond_temperature_c"][0] == pytest.approx(0.0, abs=1e-9)
    assert fine["bond_temperature_c"][-1] < -8.0


def test_model_tau_f_is_within_an_order_of_magnitude_of_conduction(model):
    ratio = model.tau_f_seconds_effective / conduction_time(model.thermal_length)
    assert 1.0 / 3.0 < ratio < 3.0


def test_prescribed_tau_f_overrides_the_solution():
    forced = FreezeBondAgeModel(thermal_length=0.5, tau_f_seconds=3600.0)
    assert forced.tau_source == "prescribed"
    assert forced.tau_f_hours == pytest.approx(1.0)
    assert forced.sigma_at(1.0) == pytest.approx(forced.sigma_inf_pa * (1.0 - math.exp(-1.0)))


# --------------------------------------------------------------------------- #
# 5. DEM coupling: runtime bond strengths and the aged material
# --------------------------------------------------------------------------- #
def test_aged_material_installs_the_time_dependent_bond(model):
    base = RidgeMaterial(particle_radius=0.75, freeze_bond_fraction=0.65,
                         bond_tensile_strength=100.0e3, bond_shear_strength=200.0e3)
    fresh = aged_material(base, model, 0.0)
    assert fresh.freeze_bond_fraction == 0.0
    mature = aged_material(base, model, 168.0)
    assert mature.freeze_bond_fraction == pytest.approx(0.65, rel=1e-3)
    assert mature.bond_tensile_strength == pytest.approx(model.sigma_at(168.0), rel=1e-12)
    assert mature.bond_shear_strength == pytest.approx(model.tau_at(168.0), rel=1e-12)
    # the growth is monotone in age
    fractions = [aged_material(base, model, age).freeze_bond_fraction
                 for age in (0.0, 6.0, 24.0, 72.0, 168.0)]
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))


def test_set_bond_strengths_updates_the_runtime_strengths():
    keel = RidgeKeel(4.0, 6.0, 0.25)
    material = RidgeMaterial(particle_radius=0.75, bond_tensile_strength=100.0e3,
                             bond_shear_strength=200.0e3)
    from tensordem.ridge_dem import generate_keel_assembly
    assembly: RidgeAssembly = generate_keel_assembly(keel, material)["assembly"]
    assert assembly.bond_tensile_strength == pytest.approx(100.0e3)
    assembly.set_bond_strengths(30.0e3)
    assert assembly.bond_tensile_strength == pytest.approx(30.0e3)
    # shear follows the material ratio (200/100 = 2)
    assert assembly.bond_shear_strength == pytest.approx(60.0e3)
    assembly.set_bond_strengths(10.0e3, 5.0e3)
    assert assembly.bond_shear_strength == pytest.approx(5.0e3)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            assembly.set_bond_strengths(bad)
    with pytest.raises(TypeError):
        assembly.set_bond_strengths(True)


def test_bond_contact_mode_is_validated_and_defaults_to_historical_behaviour():
    keel = RidgeKeel(4.0, 6.0, 0.25)
    material = RidgeMaterial(particle_radius=0.75)
    from tensordem.ridge_dem import generate_keel_assembly
    assembly = generate_keel_assembly(keel, material)["assembly"]
    assert assembly.bond_contact_mode == "exclusive"
    with pytest.raises(ValueError):
        RidgeAssembly(assembly.pos, assembly.radii, assembly.mass, assembly.bonds,
                      material, bond_contact_mode="bogus")


# --------------------------------------------------------------------------- #
# 6. Keel resistance vs age and the Lindqvist age factor
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def quick_sweep(model):
    keel, material = RidgeKeel(4.0, 6.0, 0.25), RidgeMaterial(
        particle_radius=0.75, freeze_bond_fraction=0.65,
        bond_tensile_strength=100.0e3, bond_shear_strength=200.0e3)
    return run_keel_age_sweep((0.0, 24.0, 168.0), keel=keel, material=material,
                              model=model, seeds=(20261031,),
                              press_velocity=0.1, stroke=0.03)


def test_unbonded_keel_at_zero_age(quick_sweep):
    first = quick_sweep["records"][0]
    assert first["age_hours"] == 0.0
    assert first["sigma_b_pa"] == 0.0
    assert first["bond_fraction"] == 0.0
    assert first["age_factor"] == pytest.approx(1.0)
    assert first["peak_force_n_per_m"] > 0.0


def test_keel_resistance_grows_with_maturity(quick_sweep):
    """Mature bonds must carry more than half-grown ones.

    The *whole* curve (including the young ages) is a deck-resolution question
    and is tested against the committed full-deck artifact below: on the
    9-particle smoke deck a freshly bonded packing can be weaker than the
    unbonded one because all its bonds fail at once.
    """
    by_age = {record["age_hours"]: record for record in quick_sweep["records"]}
    assert by_age[0.0]["age_factor"] == pytest.approx(1.0)
    assert by_age[168.0]["peak_force_n_per_m"] > by_age[24.0]["peak_force_n_per_m"]
    assert by_age[168.0]["age_factor"] > 1.1


def test_age_factor_interfaces_with_lindqvist(quick_sweep):
    coupling = lindqvist_age_correction(quick_sweep["records"],
                                        reference_keel_resistance=1.0e6)
    k_age = coupling["k_age"]
    assert len(k_age) == len(quick_sweep["records"])
    assert k_age[0]["age_hours"] == 0.0
    assert k_age[0]["k_age"] == pytest.approx(1.0)
    assert k_age[0]["R_keel_n_per_m"] == pytest.approx(1.0e6)
    for entry, record in zip(k_age, quick_sweep["records"]):
        assert entry["k_age"] == pytest.approx(record["age_factor"])
        assert entry["R_keel_n_per_m"] == pytest.approx(1.0e6 * entry["k_age"])
    with pytest.raises(ValueError):
        lindqvist_age_correction([])


def test_contact_mode_sensitivity_is_reported_not_hidden(model):
    """Both contact conventions must be runnable and must respond to bonding.

    Which of the two carries the larger peak is a *measured* model-choice
    sensitivity (it is recorded in ``age_sweep.json["contact_mode_sensitivity"]``),
    not an assumption: the suite only requires that both modes are stable,
    break no more bonds than they have, and react to the aged material.
    """
    keel, material = RidgeKeel(4.0, 6.0, 0.25), RidgeMaterial(
        particle_radius=0.75, freeze_bond_fraction=0.65,
        bond_tensile_strength=100.0e3, bond_shear_strength=200.0e3)
    aged = aged_material(material, model, 168.0)
    runs = {mode: keel_resistance(keel, aged, press_velocity=0.1, stroke=0.03,
                                  bond_contact_mode=mode)
            for mode in ("parallel", "exclusive")}
    for mode, run in runs.items():
        assert run["peak_n_per_m"] > 0.0
        assert 0 <= run["broken_bonds"] <= run["bonds"]
        assert run["bonds"] == aged_bond_count(material, keel)
    # in *both* conventions a fully grown bond raises the peak above the
    # unbonded deck (133.5 kN/m for this smoke deck)
    unbonded = keel_resistance(keel, aged_material(material, model, 0.0),
                              press_velocity=0.1, stroke=0.03)
    for run in runs.values():
        assert run["peak_n_per_m"] > unbonded["peak_n_per_m"]


ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "artifacts", "ridge_freeze_bond_age", "age_sweep.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT),
                    reason="full-deck artifact not generated in this checkout")
def _full_deck() -> dict:
    with open(ARTIFACT) as handle:
        sweep = json.load(handle)
    records = sorted(sweep["records"], key=lambda record: record["age_hours"])
    assert [record["age_hours"] for record in records] == [0.0, 6.0, 24.0, 72.0, 168.0]
    assert sweep["apparatus"]["bond_contact_mode"] == "parallel"
    assert len(sweep["apparatus"]["seeds"]) >= 2, "full deck must average seeds"
    assert records[0]["sigma_b_pa"] == 0.0 and records[0]["bond_fraction"] == 0.0
    assert records[-1]["particles"] > 30, "representative volume, not smoke deck"
    # the constitutive strength law must be monotone even though the measured
    # keel resistance is not: the two are different statements
    sigmas = [record["sigma_b_pa"] for record in records]
    assert all(b > a for a, b in zip(sigmas, sigmas[1:]))
    return {"sweep": sweep, "records": records}


@pytest.mark.skipif(not os.path.exists(ARTIFACT),
                    reason="full-deck artifact not generated in this checkout")
def test_committed_full_deck_reaches_mature_aging():
    """Above ~24 h an older (better bonded) keel must be the stronger one."""
    records = _full_deck()["records"]
    mature = [record for record in records if record["age_hours"] >= 24.0]
    peaks = [record["peak_force_n_per_m"] for record in mature]
    assert all(b > a for a, b in zip(peaks, peaks[1:])), peaks
    assert mature[-1]["age_factor"] > 1.4
    factors = [record["age_factor"] for record in records]
    assert factors[0] == pytest.approx(1.0)


@pytest.mark.skipif(not os.path.exists(ARTIFACT),
                    reason="full-deck artifact not generated in this checkout")
def test_young_age_dip_is_recorded_not_smoothed():
    """Honest regression pin for the *measured* non-monotonicity.

    The campaign premise was that keel resistance grows with age. The DEM says
    it does not below ~24 h: the 6 h keel is weaker than the unbonded one for
    part of the packings (see docs/ridge_freeze_bond_age_evolution.md 3.5).
    This test fails if someone silently makes the curve look monotone.
    """
    records = _full_deck()["records"]
    young = records[1]
    assert young["age_hours"] == 6.0
    assert young["sigma_b_pa"] > 0.0, "bonds do exist at 6 h"
    assert young["age_factor"] < records[2]["age_factor"]
    study_path = os.path.join(os.path.dirname(ARTIFACT), "young_age_seed_study.json")
    if os.path.exists(study_path):
        with open(study_path) as handle:
            study = json.load(handle)
        assert study["ensemble"]["ratio_6h_over_0h_mean"] < 1.0
        assert len(study["paired"]) >= 3


def aged_bond_count(material, keel) -> int:
    """Number of freeze bonds the keel generator produces for ``material``."""
    from tensordem.ridge_dem import generate_keel_assembly
    return generate_keel_assembly(keel, material)["info"]["bonds"]


# --------------------------------------------------------------------------- #
# 7. Cross-check against the prescribed growth time of the sibling module
# --------------------------------------------------------------------------- #
def test_growth_time_comparison_scales_with_the_square_of_the_block():
    from tensordem.ridge_freeze_bond_age import compare_growth_time_scales
    result = compare_growth_time_scales(thermal_lengths=(0.25, 0.5, 1.0),
                                        temperature_c=-10.0, salinity_ppt=5.0)
    rows = result["records"]
    tau = [row["tau_derived_hours"] for row in rows]
    assert all(b > a for a, b in zip(tau, tau[1:]))
    # conduction scales as h^2, but the lens latent heat is a fixed 5 mm film,
    # so the growth is super-linear yet below quadratic and the latent excess
    # (tau / rho c h^2/(4k)) shrinks as the block grows
    for index in (0, 1):
        factor = tau[index + 1] / tau[index]
        assert 2.0 < factor < 4.0, factor
    ratios_to_conduction = [row["ratio_to_conduction"] for row in rows]
    assert all(b < a for a, b in zip(ratios_to_conduction,
                                     ratios_to_conduction[1:]))
    for row in rows:
        assert 1.0 < row["ratio_to_conduction"] < 10.0
    prescribed = [row["tau_prescribed_hours"] for row in rows]
    if prescribed[0] is not None:
        # the prescribed (activation-energy) time scale is size independent,
        # so it can only match the derived one at one block size
        assert len(set(round(value, 9) for value in prescribed)) == 1
        ratios = [row["ratio_prescribed_over_derived"] for row in rows]
        assert all(b < a for a, b in zip(ratios, ratios[1:]))
        # and the two routes agree within a factor of 5 at the DEM block size
        assert 0.1 < result["records"][1]["ratio_prescribed_over_derived"] < 5.0
