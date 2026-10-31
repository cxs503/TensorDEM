"""Tests for the friction-coupled level-ice resistance model and its API."""
import math

import pytest

from tensordem.api_hull_ice_friction import (
    DEFAULT_REQUEST,
    _equivalent_power_law,
    handle_hull_ice_friction,
    hull_ice_friction,
)
from tensordem.hull_ice_friction import MU_LITERATURE_LOW, micro_contact_friction
from tensordem.ice_resistance import (
    VesselIceGeometry,
    ice_resistance_components,
    resistance_temperature_comparison,
    resistance_temperature_sweep,
)

FIT = {"mu0": 0.02, "alpha": 0.062, "beta": 0.45}


def test_components_sum_to_total():
    record = ice_resistance_components(-10.0, mu_parameters=FIT)
    total = (
        record["crushing_resistance_N"]
        + record["breaking_resistance_N"]
        + record["submersion_resistance_N"]
        + record["friction_resistance_N"]
    )
    assert record["total_resistance_N"] == pytest.approx(total)
    assert 0.0 < record["friction_share"] < 1.0
    assert record["friction_coefficient"] == pytest.approx(
        0.02 + 0.062 * 10.0**0.45
    )


def test_resistance_grows_as_the_ice_cools():
    warm = ice_resistance_components(-5.0, mu_parameters=FIT)
    cold = ice_resistance_components(-20.0, mu_parameters=FIT)
    assert cold["total_resistance_N"] > warm["total_resistance_N"]
    assert cold["friction_coefficient"] > warm["friction_coefficient"]


def test_temperature_comparison_threshold():
    comparison = resistance_temperature_comparison(-5.0, -20.0, mu_parameters=FIT)
    assert comparison["warm"]["temperature_C"] == -5.0
    assert comparison["cold"]["temperature_C"] == -20.0
    assert comparison["delta_resistance_N"] > 0.0
    assert comparison["delta_relative_to_warm"] > 0.10
    assert comparison["friction_coefficient_ratio_cold_over_warm"] > 1.0
    assert 0.0 < comparison["friction_share_warm"] < comparison["friction_share_cold"] < 1.0


def test_temperature_comparison_ordering_validation():
    with pytest.raises(ValueError):
        resistance_temperature_comparison(-20.0, -5.0)
    with pytest.raises(ValueError):
        resistance_temperature_comparison(-5.0, -5.0)


def test_geometry_validation():
    with pytest.raises(ValueError):
        VesselIceGeometry(beam_m=0.0)
    with pytest.raises(ValueError):
        VesselIceGeometry(bow_inclination_deg=0.0)
    with pytest.raises(ValueError):
        VesselIceGeometry(bow_inclination_deg=90.0)
    with pytest.raises(ValueError):
        VesselIceGeometry(ice_thickness_m=float("nan"))


def test_resistance_sweep_is_monotone():
    records = resistance_temperature_sweep([-5.0, -15.0, -25.0], mu_parameters=FIT)
    totals = [record["total_resistance_N"] for record in records]
    assert totals == sorted(totals)


def test_friction_falls_back_to_the_interface_law():
    assert hull_ice_friction(-10.0) == pytest.approx(micro_contact_friction(-10.0))
    assert hull_ice_friction(-10.0, mu_parameters=FIT) == pytest.approx(
        0.02 + 0.062 * 10.0**0.45
    )


def test_equivalent_power_law_matches_the_interface_law():
    law = _equivalent_power_law({"mu_ref": MU_LITERATURE_LOW, "exponent": 0.45})
    for temperature in (-5.0, -10.0, -25.0):
        assert (
            0.0 + law["alpha"] * abs(temperature) ** law["beta"]
        ) == pytest.approx(micro_contact_friction(temperature), rel=1e-12)


def test_api_default_request_runs():
    response = handle_hull_ice_friction({})
    assert response["status"] == "ok"
    assert response["protocol"].startswith("tensordem-hull-ice-friction")
    assert len(response["friction"]) == len(DEFAULT_REQUEST["temperatures_C"])
    assert response["criteria"]["friction_monotone_decreasing_in_temperature"] is True
    assert response["criteria"]["resistance_increase_cold_vs_warm"]["greater_than_10_percent"] is True
    assert "WITHHELD" in response["evidence"]["absolute_resistance_validation"]
    for entry in response["friction"]:
        assert entry["resistance"]["total_resistance_N"] > 0.0


def test_api_accepts_calibrated_coefficients_and_vessel_override():
    response = handle_hull_ice_friction(
        {
            "temperatures_C": [-5.0, -20.0],
            "mu_parameters": FIT,
            "vessel": {"beam_m": 22.0, "ice_thickness_m": 1.0},
            "comparison": {"warm_temperature_C": -5.0, "cold_temperature_C": -20.0},
        }
    )
    assert response["request"]["vessel"]["beam_m"] == 22.0
    assert response["friction"][0]["friction_coefficient"] == pytest.approx(
        0.02 + 0.062 * 5.0**0.45
    )
    assert response["evidence"]["friction_calibration"].startswith("REAL")
    reference = response["criteria"]["reference_friction_within_literature_band"]
    assert reference["value"] == pytest.approx(0.02 + 0.062 * 10.0**0.45)
    assert reference["within_band"] is True


def test_api_rejects_invalid_requests():
    with pytest.raises(ValueError):
        handle_hull_ice_friction({"temperatures_C": [5.0]})
    with pytest.raises(ValueError):
        handle_hull_ice_friction({"temperatures_C": []})
    with pytest.raises(ValueError):
        handle_hull_ice_friction({"unknown_field": 1})
    with pytest.raises(ValueError):
        handle_hull_ice_friction({"mu_parameters": {"mu0": 0.1, "alpha": 0.1}})
    with pytest.raises(ValueError):
        handle_hull_ice_friction({"vessel": {"beam_width": 20.0}})
    with pytest.raises(ValueError):
        handle_hull_ice_friction({"temperatures_C": ["cold"]})


def test_api_router_builds_when_fastapi_is_available():
    fastapi = pytest.importorskip("fastapi")
    from tensordem.api_hull_ice_friction import create_router

    router = create_router(prefix="/api/v1")
    paths = {route.path for route in router.routes}
    assert "/api/v1/calculations/hull-ice-friction" in paths
    assert fastapi.__version__