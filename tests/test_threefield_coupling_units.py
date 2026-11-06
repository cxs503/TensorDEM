"""Unit coverage for :mod:`tensordem.threefield.coupling`.

Six of the ten public names of the three-field coupling module were never
referenced by any test: the mass/diameter conversion pair, the wake-profile
sampler, the irrigation-free ledger helper, the DEM inventory adapter, the
config validator and the follower-ship impact split.  They are the glue that
turns DEM/LBM/thermal results into engineering numbers, so an inverted
conversion factor or a silently clamped profile would show up in an artifact
only as a small percentage error.

Positive cases check the algebraic identities (round trip, interpolation,
ledger closure, mass conservation through the fragmentation and thermal
stages).  Negative cases check that the ``_finite``-based guards still raise
``TypeError`` / ``ValueError`` instead of propagating NaN into a report.
"""
from __future__ import annotations

import math

import pytest

from tensordem.threefield import coupling as cp
from tensordem.threefield.fragmentation import BreakupProperties
from tensordem.threefield.thermal import ThermalProperties

RHO_ICE = 917.0


def _inventory() -> list[dict[str, float]]:
    return [
        {"index": 0.0, "mass_kg": 12.0, "diameter_m": 1.0},
        {"index": 1.0, "mass_kg": 3.5, "diameter_m": 0.6},
    ]


# ---------------------------------------------------------------------------
# Mass / diameter conversions
# ---------------------------------------------------------------------------

def test_sphere_mass_diameter_round_trip() -> None:
    for diameter in (0.3, 0.6, 1.0, 2.5):
        mass = cp.ice_mass_from_sphere(diameter, RHO_ICE)
        assert mass == pytest.approx(RHO_ICE * math.pi / 6.0 * diameter ** 3, rel=1e-15)
        assert cp.diameter_from_mass(mass, RHO_ICE) == pytest.approx(diameter, rel=1e-12)

    # a zero mass has a zero equivalent diameter (no division by zero)
    assert cp.diameter_from_mass(0.0, RHO_ICE) == 0.0
    # diameter scales with the cube root of the mass
    assert cp.diameter_from_mass(8.0 * 10.0, RHO_ICE) == pytest.approx(
        2.0 * cp.diameter_from_mass(10.0, RHO_ICE), rel=1e-12)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: cp.ice_mass_from_sphere(-0.5, RHO_ICE), ValueError),
        (lambda: cp.ice_mass_from_sphere(1.0, 0.0), ValueError),
        (lambda: cp.ice_mass_from_sphere(1.0, -917.0), ValueError),
        (lambda: cp.ice_mass_from_sphere(float("nan"), RHO_ICE), ValueError),
        (lambda: cp.ice_mass_from_sphere(True, RHO_ICE), TypeError),
        (lambda: cp.diameter_from_mass(-1.0, RHO_ICE), ValueError),
        (lambda: cp.diameter_from_mass(1.0, 0.0), ValueError),
        (lambda: cp.diameter_from_mass(float("inf"), RHO_ICE), ValueError),
    ],
)
def test_mass_conversions_reject_invalid_inputs(call, error) -> None:
    with pytest.raises(error):
        call()


# ---------------------------------------------------------------------------
# Wake-profile sampler
# ---------------------------------------------------------------------------

_PROFILE = {
    "radius": [0.0, 1.0, 2.0],
    "mean_axial_velocity": [5.0, 4.0, 3.0],
    "axial_rms_velocity": [0.5, 0.4, 0.3],
}


def test_race_velocity_interpolates_and_clamps() -> None:
    interior = cp.race_velocity_at(0.5, _PROFILE)
    assert interior["mean_velocity"] == pytest.approx(4.5, rel=1e-15)
    assert interior["rms_velocity"] == pytest.approx(0.45, rel=1e-15)

    assert cp.race_velocity_at(1.0, _PROFILE)["mean_velocity"] == pytest.approx(4.0)
    # outside the tabulated range the end values are held (no extrapolation)
    assert cp.race_velocity_at(0.0, _PROFILE)["mean_velocity"] == pytest.approx(5.0)
    assert cp.race_velocity_at(9.0, _PROFILE)["mean_velocity"] == pytest.approx(3.0)
    assert cp.race_velocity_at(9.0, _PROFILE)["rms_velocity"] == pytest.approx(0.3)


def test_race_velocity_defaults_rms_to_zero() -> None:
    profile = {"radius": [0.0, 1.0], "mean_axial_velocity": [5.0, 3.0]}
    samples = cp.race_velocity_at(0.5, profile)
    assert samples["mean_velocity"] == pytest.approx(4.0)
    assert samples["rms_velocity"] == 0.0


def test_race_velocity_rejects_empty_profile_and_negative_radius() -> None:
    with pytest.raises(ValueError, match="wake profile is empty"):
        cp.race_velocity_at(0.5, {"radius": [], "mean_axial_velocity": []})
    with pytest.raises(ValueError, match="nonnegative"):
        cp.race_velocity_at(-0.1, _PROFILE)


# ---------------------------------------------------------------------------
# Ledger and inventory adapters
# ---------------------------------------------------------------------------

def test_mass_ledger_closes_exactly() -> None:
    ledger = cp.mass_ledger(
        initial_mass_kg=10.0, surviving_mass_kg=8.5, melted_mass_kg=1.5)
    assert ledger["balance_kg"] == pytest.approx(10.0, rel=1e-15)
    assert ledger["residual_kg"] == pytest.approx(0.0, abs=1e-15)
    assert ledger["relative_error"] < 1e-15


def test_mass_ledger_reports_the_residual_and_extra_terms() -> None:
    ledger = cp.mass_ledger(
        initial_mass_kg=10.0, surviving_mass_kg=8.0, melted_mass_kg=1.0,
        extra_terms={"spray_kg": 0.5},
    )
    assert ledger["balance_kg"] == pytest.approx(9.5)
    assert ledger["residual_kg"] == pytest.approx(-0.5)
    assert ledger["relative_error"] == pytest.approx(0.05, rel=1e-12)
    assert ledger["extra_terms"] == {"spray_kg": 0.5}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_mass_kg": 0.0, "surviving_mass_kg": 0.0, "melted_mass_kg": 0.0},
        {"initial_mass_kg": -1.0, "surviving_mass_kg": 0.0, "melted_mass_kg": 0.0},
        {"initial_mass_kg": 10.0, "surviving_mass_kg": -1.0, "melted_mass_kg": 1.0},
        {"initial_mass_kg": 10.0, "surviving_mass_kg": 8.0, "melted_mass_kg": -1.0},
        {"initial_mass_kg": float("nan"), "surviving_mass_kg": 0.0, "melted_mass_kg": 0.0},
    ],
)
def test_mass_ledger_rejects_invalid_terms(kwargs) -> None:
    with pytest.raises(ValueError):
        cp.mass_ledger(**kwargs)


def test_dem_inventory_preserves_the_census() -> None:
    case = {
        "fragments": {
            "masses_kg": [12.0, 3.5],
            "diameters_m": [1.0, 0.6],
        }
    }
    inventory = cp.dem_inventory(case)
    assert [item["index"] for item in inventory] == [0.0, 1.0]
    assert [item["mass_kg"] for item in inventory] == [12.0, 3.5]
    assert [item["diameter_m"] for item in inventory] == [1.0, 0.6]


def test_dem_inventory_rejects_inconsistent_census() -> None:
    case = {"fragments": {"masses_kg": [12.0, 3.5], "diameters_m": [1.0]}}
    with pytest.raises(ValueError, match="inconsistent"):
        cp.dem_inventory(case)
    with pytest.raises(KeyError):
        cp.dem_inventory({"fragments": {"masses_kg": [1.0]}})


# ---------------------------------------------------------------------------
# Stage-level conservation
# ---------------------------------------------------------------------------

def test_fragmentation_stage_conserves_mass() -> None:
    inventory = _inventory()
    stage = cp.fragmentation_stage(inventory, 0.6, BreakupProperties())

    assert stage["fragment_count_in"] == len(inventory)
    assert stage["fragment_count_out"] == len(stage["masses_kg"]) == len(stage["lengths_m"])
    assert stage["initial_mass_kg"] == pytest.approx(15.5, rel=1e-15)
    assert abs(stage["mass_error_kg"]) <= 1e-12 * stage["initial_mass_kg"]
    assert all(mass > 0.0 for mass in stage["masses_kg"])
    assert stage["critical_length_m"] > 0.0


def test_thermal_stage_closes_the_mass_ledger() -> None:
    config = cp.CouplingConfig(residence_time_s=2.0, melt_dt_s=0.25)
    stage = cp.thermal_stage(_inventory(), config)

    assert stage["initial_mass_kg"] == pytest.approx(15.5, rel=1e-15)
    assert len(stage["survivors"]) == len(_inventory())
    assert abs(stage["mass_error_kg"]) <= 1e-12 * stage["initial_mass_kg"]
    assert stage["remaining_mass_kg"] + stage["melted_mass_kg"] == pytest.approx(
        stage["initial_mass_kg"], rel=1e-12)
    assert 0.0 < stage["melted_mass_kg"] < stage["initial_mass_kg"]
    assert 0.0 < stage["melt_fraction"] < 1.0
    # melting can only remove mass
    for survivor in stage["survivors"]:
        assert survivor["mass_kg"] <= survivor["initial_mass_kg"]
        assert survivor["melted_kg"] >= 0.0


def test_thermal_stage_skips_zero_mass_fragments() -> None:
    config = cp.CouplingConfig(residence_time_s=1.0, melt_dt_s=0.25)
    stage = cp.thermal_stage(
        [{"index": 0.0, "mass_kg": 0.0, "diameter_m": 0.0},
         {"index": 1.0, "mass_kg": 5.0, "diameter_m": 0.6}],
        config,
    )
    assert len(stage["survivors"]) == 1
    assert stage["initial_mass_kg"] == pytest.approx(5.0)
    assert abs(stage["mass_error_kg"]) <= 1e-12 * stage["initial_mass_kg"]


def test_follower_ship_impact_splits_the_two_relief_mechanisms() -> None:
    impact = cp.follower_ship_impact(
        initial_mass_kg=1000.0,
        surviving_mass_kg=800.0,
        melted_mass_kg=200.0,
        mean_floe_diameter_m=1.2,
        reference_floe_diameter_m=3.0,
        channel_width_m=20.0,
        ice_thickness_m=1.0,
        track_length_m=100.0,
    )
    assert impact["melted_fraction"] == pytest.approx(0.2, rel=1e-12)
    assert impact["floe_size_ratio"] == pytest.approx(0.4, rel=1e-12)
    assert impact["resistance_relief_from_floe_refinement"] == pytest.approx(0.6, rel=1e-12)
    assert impact["ice_area_concentration"] == pytest.approx(
        800.0 / (RHO_ICE * 1.0) / (20.0 * 100.0), rel=1e-12)

    # floes larger than the reference may not report a negative relief
    clamped = cp.follower_ship_impact(
        initial_mass_kg=10.0,
        surviving_mass_kg=10.0,
        melted_mass_kg=0.0,
        mean_floe_diameter_m=4.0,
        reference_floe_diameter_m=3.0,
        channel_width_m=20.0,
        ice_thickness_m=1.0,
        track_length_m=100.0,
    )
    assert clamped["resistance_relief_from_floe_refinement"] == 0.0


def test_follower_ship_impact_rejects_zero_channel() -> None:
    with pytest.raises(ValueError, match="channel_width_m"):
        cp.follower_ship_impact(
            initial_mass_kg=10.0,
            surviving_mass_kg=10.0,
            melted_mass_kg=0.0,
            mean_floe_diameter_m=1.0,
            reference_floe_diameter_m=3.0,
            channel_width_m=0.0,
            ice_thickness_m=1.0,
            track_length_m=100.0,
        )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def test_coupling_config_accepts_and_exposes_valid_fields() -> None:
    config = cp.CouplingConfig()
    assert config.residence_time_s > 0.0
    assert config.race_velocity_m_s > 0.0
    assert isinstance(config.thermal, ThermalProperties)
    assert isinstance(config.breakup, BreakupProperties)

    custom = cp.CouplingConfig(
        residence_time_s=12.0, melt_dt_s=0.5, relative_velocity_m_s=0.0,
        race_velocity_m_s=4.2, turbulence_intensity=0.12, seed=7,
    )
    assert custom.residence_time_s == 12.0
    assert custom.seed == 7


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"residence_time_s": 0.0}, ValueError),
        ({"residence_time_s": float("inf")}, ValueError),
        ({"melt_dt_s": -0.25}, ValueError),
        ({"relative_velocity_m_s": -0.1}, ValueError),
        ({"race_velocity_m_s": 0.0}, ValueError),
        ({"turbulence_intensity": 0.0}, ValueError),
        ({"seed": True}, TypeError),
        ({"seed": 1.5}, TypeError),
    ],
)
def test_coupling_config_rejects_invalid_fields(kwargs, error) -> None:
    with pytest.raises(error):
        cp.CouplingConfig(**kwargs)