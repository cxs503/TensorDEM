"""Tests for the level-ice / channel resistance layer."""
from __future__ import annotations

import csv
import math

import pytest

from tensordem.formation_resistance import (GRAVITY, IceResistanceModel, IceSheet,
                                            IcebreakerShip, calibrate_ice_resistance,
                                            ice_properties_from_reference_row,
                                            level_ice_resistance, strip_resistance)

SHIP = IcebreakerShip(name="T1", length=100.0, beam=20.0, draft=8.0)


def test_plate_foundation_bending_load_matches_the_closed_form():
    sheet = IceSheet()
    thickness = 1.0
    breaking, _submergence, _friction = sheet.line_load(thickness)
    rigidity = sheet.elastic_modulus * thickness ** 3 / (12.0 * (1.0 - sheet.poisson_ratio ** 2))
    beta = (sheet.water_density * GRAVITY / (4.0 * rigidity)) ** 0.25
    expected = (2.0 / 3.0) * sheet.flexural_strength * thickness ** 2 * beta
    assert breaking == pytest.approx(expected, rel=1e-12)
    assert abs(beta - sheet.characteristic_length(thickness)) < 1e-15


def test_thin_ice_is_limited_by_the_crushing_cap():
    sheet = IceSheet(flexural_strength=5.0e5)
    thickness = 0.02  # bending load becomes smaller than the crushing cap
    breaking, _s, _f = sheet.line_load(thickness)
    crushing = sheet.compressive_strength * thickness
    beta = sheet.characteristic_length(thickness)
    bending = (2.0 / 3.0) * sheet.flexural_strength * thickness ** 2 * beta
    assert breaking == pytest.approx(min(bending, crushing), rel=1e-12)


def test_strip_sum_reproduces_the_uniform_baseline():
    model = IceResistanceModel(scale_factor=1.3)
    uniform = level_ice_resistance(SHIP, model, 0.8)
    width = SHIP.contact_width
    breaking, submergence, friction = model.sheet.line_load(0.8)
    assert uniform["breaking_resistance_N"] == pytest.approx(breaking * width * 1.3, rel=1e-12)
    assert uniform["total_resistance_N"] == pytest.approx(
        (breaking + submergence + friction) * width * 1.3, rel=1e-12)
    assert len(uniform["strips"]) == SHIP.strips


def test_resistance_is_monotone_in_thickness_strength_and_speed_of_ice():
    model = IceResistanceModel()
    thin = level_ice_resistance(SHIP, model, 0.5)["total_resistance_N"]
    thick = level_ice_resistance(SHIP, model, 1.0)["total_resistance_N"]
    assert thick > thin > 0.0
    weak = level_ice_resistance(SHIP, IceResistanceModel(sheet=IceSheet(flexural_strength=3.0e5)),
                                1.0)["total_resistance_N"]
    assert thick > weak


def test_brash_strips_are_cheaper_than_intact_sheet():
    model = IceResistanceModel()
    intact = strip_resistance(SHIP, model, [1.0] * SHIP.strips)
    brash = strip_resistance(SHIP, model, [1.0] * SHIP.strips,
                             brash_flags=[True] * SHIP.strips)
    assert brash["total_resistance_N"] < intact["total_resistance_N"]
    mixed = strip_resistance(SHIP, model, [1.0] * SHIP.strips,
                            brash_flags=[True] + [False] * (SHIP.strips - 1))
    assert brash["total_resistance_N"] < mixed["total_resistance_N"] < intact["total_resistance_N"]


def test_scale_factor_scales_linearly():
    base = level_ice_resistance(SHIP, IceResistanceModel(scale_factor=1.0), 0.9)
    scaled = level_ice_resistance(SHIP, IceResistanceModel(scale_factor=2.0), 0.9)
    assert scaled["total_resistance_N"] == pytest.approx(2.0 * base["total_resistance_N"], rel=1e-12)


def test_strip_argument_validation():
    with pytest.raises(ValueError):
        strip_resistance(SHIP, IceResistanceModel(), [1.0, 1.0])
    with pytest.raises(ValueError):
        strip_resistance(SHIP, IceResistanceModel(), [1.0] * SHIP.strips,
                         brash_flags=[True])
    with pytest.raises(ValueError):
        IceSheet(water_density=900.0)
    with pytest.raises(ValueError):
        IceResistanceModel(scale_factor=0.0)


def test_ship_geometry_validation_and_report():
    report = SHIP.geometry_report()
    assert report["length_beam_ratio"] == pytest.approx(5.0)
    assert report["beam_draft_ratio"] == pytest.approx(2.5)
    assert report["displacement_volume_m3"] == pytest.approx(
        SHIP.block_coefficient * 100.0 * 20.0 * 8.0)
    with pytest.raises(ValueError):
        IcebreakerShip(length=100.0, beam=20.0, draft=8.0, block_coefficient=1.2)
    with pytest.raises(ValueError):
        IcebreakerShip(length=100.0, beam=4.0, draft=1.0)  # L/B out of window
    with pytest.raises(ValueError):
        IcebreakerShip(length=20.0, beam=20.0, draft=1.0)  # L/B out of window


def test_reference_row_conversion():
    row = {"flexural_strength_kPa": "724", "ice_elastic_modulus_MPa": "929"}
    sheet = ice_properties_from_reference_row(row)
    assert sheet.flexural_strength == pytest.approx(724.0e3)
    assert sheet.elastic_modulus == pytest.approx(929.0e6)
    with pytest.raises(ValueError):
        ice_properties_from_reference_row({"flexural_strength_kPa": ""})


def test_calibration_is_a_single_parameter_fit_with_holdout(tmp_path):
    rows = []
    ship = IcebreakerShip(name="MT Uikku equivalent", length=105.0, beam=19.2, draft=7.0,
                          block_coefficient=0.75)
    base = IceResistanceModel()
    # Synthetic reference set: measured = 1.4 * unscaled prediction of thick ice.
    for index, thickness in enumerate([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]):
        prediction = level_ice_resistance(ship, base, thickness)["total_resistance_N"]
        rows.append({
            "test_id": f"T{index:03d}",
            "ice_thickness_m": str(thickness),
            "flexural_strength_kPa": "500",
            "ice_elastic_modulus_MPa": "5000",
            "experimental_reference_table8_kN": str(1.4 * prediction / 1000.0),
        })
    result = calibrate_ice_resistance(rows, ship, holdout=3)
    assert result["scale_factor"] == pytest.approx(1.4, rel=1e-6)
    assert result["holdout_mape_percent"] < 1e-6
    assert result["training_mape_percent"] < 1e-6
    assert result["holdout_count"] == 3
    assert result["ship"]["length_m"] == pytest.approx(105.0)


def test_calibration_rejects_too_few_cases():
    ship = IcebreakerShip(length=100.0, beam=20.0, draft=8.0)
    with pytest.raises(ValueError):
        calibrate_ice_resistance([{"test_id": "x", "ice_thickness_m": "1.0",
                                   "flexural_strength_kPa": "500",
                                   "ice_elastic_modulus_MPa": "5000",
                                   "experimental_reference_table8_kN": "500"}], ship)