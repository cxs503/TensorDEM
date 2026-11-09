"""Acceptance tests for the ice-ridge keel freeze-bond aging model (P54-W2-N03).

Covers the four machine-checkable criteria of the node:

(a) ``sigma_b(t) -> 0`` as ``t -> 0`` and ``sigma_b(t) -> sigma_b,inf`` as
    ``t -> inf``, monotonically increasing;
(b) ``sigma_b,inf`` against the Ettema & Schaefer (1986) experiments, every
    datum inside 20 % (holdouts included);
(c) ``R_keel(t)`` monotonically increasing and degenerate at ``t = 0`` to the
    unbonded keel resistance;
(d) the DEM integration: ``tensordem.ridge_dem`` bond elements fail exactly at
    the law's strength, the keel generator realises the aged bonded fraction,
    plus the salinity (S) dependence tests.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest
import torch

from tensordem.ridge_dem import RidgeKeel, RidgeMaterial, generate_keel_assembly
from tensordem.ridge_freeze_bond_aging import (
    DEFAULT_APPARENT_ACTIVATION_ENERGY,
    FREEZING_POINT_K,
    REFERENCE_DATA,
    KeelAgingResistance,
    SaturationFreezeBondModel,
    aged_material,
    aging_state,
    cohesion_transfer_coefficient,
    validate_against_ettema_schaefer,
)
from tensordem.ridge_freeze_bond_aging_dem import (dem_bond_element_capacity,
                                                   dem_keel_penetration)

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs" / "p54w2_n03_ridge_freeze_bond_aging" / "study.json"
MODEL = SaturationFreezeBondModel()
SIGMA_N = 5297.4      # hydrostatic contact pressure at mid-depth of a 10 m keel
T_COLD = 263.15       # -10 degC


# --------------------------------------------------------------------- criterion (a)
class TestLawLimits:
    def test_zero_age_gives_zero_strength(self):
        assert MODEL.shear_strength(0.0, SIGMA_N, temperature_K=T_COLD) == 0.0
        assert MODEL.tensile_strength(0.0, SIGMA_N, temperature_K=T_COLD) == 0.0

    def test_large_age_approaches_asymptote(self):
        sigma_inf = MODEL.asymptotic_tensile_strength(SIGMA_N, temperature_K=T_COLD)
        sigma = float(MODEL.tensile_strength(1.0e7, SIGMA_N, temperature_K=T_COLD))
        assert sigma == pytest.approx(sigma_inf, rel=1e-3)

    def test_monotone_increasing(self):
        ages = np.linspace(0.0, 240.0 * 3600.0, 500)
        sigma = MODEL.tensile_strength(ages, SIGMA_N, temperature_K=T_COLD)
        assert np.all(np.diff(sigma) >= 0.0)
        assert sigma[0] == 0.0

    def test_initial_slope_is_sigma_inf_over_tau(self):
        tau = MODEL.growth_time_s(temperature_K=T_COLD)
        slope = float(MODEL.tensile_rate(0.0, SIGMA_N, temperature_K=T_COLD))
        sigma_inf = MODEL.asymptotic_tensile_strength(SIGMA_N, temperature_K=T_COLD)
        assert slope == pytest.approx(sigma_inf / tau, rel=1e-12)

    def test_tensile_is_kappa_times_shear(self):
        ages = np.array([0.0, 1.0, 1.0e4])
        shear = MODEL.shear_strength(ages, SIGMA_N, temperature_K=T_COLD)
        tensile = MODEL.tensile_strength(ages, SIGMA_N, temperature_K=T_COLD)
        assert np.allclose(tensile, MODEL.shear_to_tension_ratio * shear)

    def test_negative_age_and_pressure_rejected(self):
        with pytest.raises(ValueError):
            MODEL.shear_strength(-1.0, SIGMA_N)
        with pytest.raises(ValueError):
            MODEL.shear_strength(1.0, -5.0)

    def test_bonded_fraction_runs_from_zero_to_one(self):
        assert MODEL.bonded_fraction(0.0) == 0.0
        assert float(MODEL.bonded_fraction(1.0e9)) == pytest.approx(1.0, rel=1e-9)
        frac = MODEL.bonded_fraction(np.linspace(0.0, 1.0e6, 100))
        assert np.all(np.diff(frac) >= 0.0)


# --------------------------------------------------------------------- criterion (b)
class TestEttemaSchaeferValidation:
    def test_all_data_inside_twenty_percent(self):
        report = validate_against_ettema_schaefer(MODEL)
        assert report["all_pass"], report["rows"]
        assert report["max_rel_error"] < 0.20
        assert report["max_holdout_rel_error"] < 0.20

    def test_reference_data_is_traceable(self):
        for datum in REFERENCE_DATA:
            assert datum.source, datum.key
            assert datum.kind in {"coefficient", "plateau", "ratio", "time_form"}

    def test_calibration_anchors_reproduce_exactly(self):
        report = validate_against_ettema_schaefer(MODEL)
        anchors = {r["key"]: r for r in report["rows"] if r["role"] == "calibration"}
        assert set(anchors) == {"distilled_k", "air_k"}
        for row in anchors.values():
            assert row["rel_error"] < 1e-12

    def test_holdout_count_and_identity(self):
        report = validate_against_ettema_schaefer(MODEL)
        assert report["n_holdout"] == report["n_data"] - 2
        keys = {r["key"] for r in report["rows"] if r["role"] == "holdout"}
        assert "tap_k" in keys and "saline_12p5_plateau" in keys

    def test_caption_pressure_variant_is_documented_failure(self):
        """The Fig. 11 caption (3.8 kPa) is inconsistent with Eq. (6): it fails."""
        variant = validate_against_ettema_schaefer(MODEL, normal_pressure_override=3800.0)
        assert not variant["all_pass"]
        assert variant["max_rel_error"] > 1.0

    def test_salinity_law_matches_equation_six(self):
        for salinity in (0.0, 3.0, 6.0, 12.5):
            expected = 2.01 * (1.0 - 0.06 * salinity)
            assert MODEL.salinity_coefficient(salinity) == pytest.approx(expected, rel=1e-12)

    def test_salinity_coefficient_plateaus_above_cap(self):
        assert MODEL.salinity_coefficient(12.5) == pytest.approx(
            MODEL.salinity_coefficient(25.0), rel=1e-12)

    def test_water_bond_about_four_times_air(self):
        ratio = MODEL.salinity_coefficient(0.0) / MODEL.salinity_coefficient(0.0, dry=True)
        assert 3.5 < ratio < 4.5


# ---------------------------------------------------------------- salinity (criterion d)
class TestSalinityDependence:
    def test_asymptote_decreases_with_salinity(self):
        values = [MODEL.asymptotic_shear_strength(SIGMA_N, salinity_percent=s,
                                                  temperature_K=T_COLD)
                  for s in (0.0, 3.0, 6.0, 12.5, 25.0)]
        assert all(a >= b for a, b in zip(values[:-1], values[1:]))
        assert values[-1] < values[0]

    def test_growth_time_decreases_with_salinity(self):
        taus = [MODEL.growth_time_s(salinity_percent=s, temperature_K=T_COLD)
                for s in (0.0, 3.0, 12.5)]
        assert all(a > b for a, b in zip(taus[:-1], taus[1:]))

    def test_brine_bond_saturates_within_minutes(self):
        """Ettema & Schaefer's time-independence in brine, at their 0 degC test state."""
        tau = MODEL.growth_time_s(salinity_percent=12.5, temperature_K=FREEZING_POINT_K)
        assert tau < 60.0
        fraction = float(MODEL.shear_strength(180.0, SIGMA_N, salinity_percent=12.5,
                                              temperature_K=FREEZING_POINT_K)
                         / MODEL.asymptotic_shear_strength(
                             SIGMA_N, salinity_percent=12.5,
                             temperature_K=FREEZING_POINT_K))
        assert fraction > 0.9
        # even at -10 degC the brine bond saturates an order of magnitude faster
        assert (MODEL.growth_time_s(salinity_percent=12.5, temperature_K=T_COLD)
                < MODEL.growth_time_s(temperature_K=T_COLD) / 10.0)

    def test_fresh_water_bond_still_grows_at_three_minutes(self):
        fraction = float(MODEL.shear_strength(180.0, SIGMA_N, temperature_K=T_COLD)
                         / MODEL.asymptotic_shear_strength(SIGMA_N, temperature_K=T_COLD))
        assert fraction < 0.5

    def test_freezing_point_depression(self):
        assert MODEL.freezing_temperature_K(0.0) == FREEZING_POINT_K
        assert MODEL.freezing_temperature_K(35.0) == pytest.approx(FREEZING_POINT_K - 1.89, rel=1e-3)
        # a 35 ppt brine at -5 degC is 5 K below 0 degC, hence 3.11 K below its
        # own freezing point (the depression adds to the driving temperature)
        assert MODEL.temperature_drop_K(FREEZING_POINT_K - 5.0, 35.0) == pytest.approx(
            3.11, rel=1e-3)
        # above the brine freezing point there is no driving temperature at all
        assert MODEL.temperature_drop_K(FREEZING_POINT_K, 35.0) == 0.0

    def test_colder_pore_water_gives_stronger_and_slower_bond(self):
        warm = MODEL.asymptotic_shear_strength(SIGMA_N, temperature_K=FREEZING_POINT_K - 1.0)
        cold = MODEL.asymptotic_shear_strength(SIGMA_N, temperature_K=253.15)
        assert cold > warm
        assert MODEL.growth_time_s(temperature_K=253.15) > MODEL.growth_time_s(
            temperature_K=FREEZING_POINT_K - 1.0)

    def test_keel_resistance_falls_with_salinity(self):
        totals = []
        for salinity in (0.0, 3.0, 12.5):
            keel = KeelAgingResistance(keel_depth_m=10.0, salinity_percent=salinity,
                                       temperature_K=T_COLD)
            totals.append(float(keel.series(MODEL, [240.0 * 3600.0])["total_N_per_m"][-1]))
        assert all(a > b for a, b in zip(totals[:-1], totals[1:]))


# --------------------------------------------------------------------- criterion (c)
class TestKeelResistance:
    def test_hydrostatic_pressure_and_buoyant_weight(self):
        keel = KeelAgingResistance(keel_depth_m=10.0, porosity=0.30)
        expected_sigma = (1025.0 - 917.0) * 9.81 * 5.0
        assert keel.normal_pressure_Pa == pytest.approx(expected_sigma, rel=1e-12)
        expected_weight = 108.0 * 9.81 * 0.70 * 0.5 * 100.0
        assert keel.buoyant_weight_per_m == pytest.approx(expected_weight, rel=1e-12)

    def test_t0_degenerates_to_unbonded_keel(self):
        keel = KeelAgingResistance(keel_depth_m=10.0, temperature_K=T_COLD)
        series = keel.series(MODEL, [0.0, 1.0, 1.0e3, 240.0 * 3600.0])
        assert series["sigma_b_shear_Pa"][0] == 0.0
        assert series["cohesive_N_per_m"][0] == 0.0
        assert series["total_N_per_m"][0] == pytest.approx(
            keel.degenerate_resistance_N_per_m(), rel=1e-12)
        assert series["total_N_per_m"][0] == pytest.approx(
            keel.friction_coefficient * keel.buoyant_weight_per_m, rel=1e-12)

    def test_monotone_increasing_and_growing(self):
        keel = KeelAgingResistance(keel_depth_m=10.0, temperature_K=T_COLD)
        ages = np.concatenate(([0.0], np.logspace(0.0, math.log10(240.0 * 3600.0), 200)))
        summary = keel.summary(MODEL, ages)
        assert summary["monotone_increasing"]
        assert summary["t0_degeneration_rel_error"] < 1e-9
        assert summary["R_growth_factor"] > 1.0
        assert summary["cohesive_dominates_at_infinity"]

    def test_unbonded_resistance_equals_friction_only(self):
        """One hour is still effectively unbonded: bonds are ~zero at 1 h."""
        keel = KeelAgingResistance(keel_depth_m=10.0, temperature_K=T_COLD,
                                   cohesion_transfer=1.0)
        early = float(keel.series(MODEL, [1.0])["total_N_per_m"][0])
        assert early == pytest.approx(keel.degenerate_resistance_N_per_m(), rel=0.05)

    def test_transfer_coefficient(self):
        assert cohesion_transfer_coefficient(7.9e3, 2.1e4) == pytest.approx(0.376, rel=0.01)
        with pytest.raises(ValueError):
            cohesion_transfer_coefficient(0.0, 1.0)
        with pytest.raises(ValueError):
            cohesion_transfer_coefficient(1.0, -1.0)

    def test_invalid_geometry_rejected(self):
        with pytest.raises(ValueError):
            KeelAgingResistance(ice_density=1100.0)
        with pytest.raises(ValueError):
            KeelAgingResistance(porosity=1.2)


# --------------------------------------------------------------------- criterion (d)
class TestDemIntegration:
    def test_aged_material_carries_the_law(self):
        base = RidgeMaterial()
        for age_h in (0.0, 1.0, 6.0, 24.0, 240.0):
            material, state = aged_material(base, MODEL, age_h * 3600.0,
                                           normal_pressure_Pa=SIGMA_N,
                                           temperature_K=T_COLD)
            assert material.bond_tensile_strength == pytest.approx(
                max(state.sigma_b_tension_Pa, 1.0), rel=1e-12)
            assert material.bond_shear_strength == pytest.approx(
                max(state.sigma_b_shear_Pa, 1.0), rel=1e-12)
            assert material.freeze_bond_fraction == pytest.approx(
                state.bonded_fraction, abs=1e-12)

    def test_aged_material_at_zero_age_has_no_bonds(self):
        material, state = aged_material(RidgeMaterial(), MODEL, 0.0,
                                       normal_pressure_Pa=SIGMA_N, temperature_K=T_COLD)
        assert state.sigma_b_shear_Pa == 0.0
        assert material.freeze_bond_fraction == 0.0

    def test_dem_bond_element_fails_at_the_law_strength(self):
        base = RidgeMaterial()
        for age_h in (0.0, 1.0, 6.0, 24.0, 240.0):
            material, state = aged_material(base, MODEL, age_h * 3600.0,
                                           normal_pressure_Pa=SIGMA_N,
                                           temperature_K=T_COLD)
            probe = dem_bond_element_capacity(material)
            assert probe["bond_alive_below"]
            assert probe["bond_broken_above"]
            # the probe is evaluated one step below the threshold
            assert probe["force_below_threshold_N"] == pytest.approx(
                material.bond_tensile_strength * material.bond_area, rel=1e-5)

    def test_keel_generator_realises_aged_bond_network(self):
        keel = RidgeKeel(6.0, 12.0, 0.25)
        counts = []
        for age_h in (0.0, 1.0, 6.0, 24.0, 1000.0):
            material, state = aged_material(RidgeMaterial(), MODEL, age_h * 3600.0,
                                           normal_pressure_Pa=SIGMA_N,
                                           temperature_K=T_COLD)
            info = generate_keel_assembly(keel, material, tip_height=0.5)["info"]
            assert 0.0 <= material.freeze_bond_fraction <= 1.0
            counts.append(info["bonds"])
        assert counts[0] == 0
        assert counts == sorted(counts)
        assert counts[-1] > 0

    def test_keel_penetration_probe_runs(self):
        keel = RidgeKeel(4.0, 8.0, 0.25)
        material, _ = aged_material(RidgeMaterial(particle_radius=0.5), MODEL,
                                   240.0 * 3600.0, normal_pressure_Pa=SIGMA_N,
                                   temperature_K=T_COLD)
        assembly = generate_keel_assembly(keel, material, tip_height=0.5)["assembly"]
        probe = dem_keel_penetration(assembly, platen_velocity=0.5, stroke=0.4,
                                     settle_steps=100, max_steps=500)
        assert math.isfinite(probe["peak_resistance_N_per_m"])
        assert probe["max_penetration_m"] > 0.0


# ------------------------------------------------------------------ study artefacts
@pytest.mark.skipif(not STUDY.exists(), reason="study artefacts not generated")
class TestStudyArtefacts:
    def test_all_criteria_pass(self):
        payload = json.loads(STUDY.read_text())
        failing = {k: v for k, v in payload["criteria"].items() if not v["pass"]}
        assert not failing, failing

    def test_numbers_are_reported(self):
        payload = json.loads(STUDY.read_text())
        assert payload["criteria"]["b_ettema_schaefer"]["max_rel_error"] < 0.20
        assert payload["criteria"]["c_keel_resistance"]["R_at_0_N_per_m"] > 0.0
        assert payload["criteria"]["d_dem_integration"]["max_bond_element_rel_error"] < 1e-6
        assert payload["validation"]["n_data"] == len(REFERENCE_DATA)
        assert payload["figures"], "study must record its figures"

    def test_activation_energy_is_declared(self):
        payload = json.loads(STUDY.read_text())
        assert payload["model"]["apparent_activation_energy_J_mol"] == pytest.approx(
            DEFAULT_APPARENT_ACTIVATION_ENERGY)