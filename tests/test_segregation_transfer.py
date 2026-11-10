"""Ship-side mapping: zone moments -> TCG/VCG, dGM, free-surface bounds, pressures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.segregation.transfer import (
    CargoDistribution,
    HoldGeometry,
    cargo_state,
    distribution_from_record,
    floor_pressure,
    free_surface_bounds,
    gm_mapping,
    slice_to_hold,
    zone_moment_check,
)


def _geometry(n_long: int = 4) -> HoldGeometry:
    return HoldGeometry(length_m=20.0, breadth_m=12.0, depth_m=6.0,
                        n_longitudinal=n_long, n_transverse=1, repose_angle_deg=35.0)


def _uniform_distribution(geo: HoldGeometry, mass_per_zone: float, zc: float = 3.0,
                          xc: np.ndarray | None = None) -> CargoDistribution:
    n = geo.n_zones
    if xc is None:
        xc = (np.arange(n) + 0.5) * geo.zone_length_m
    mass = np.full(n, mass_per_zone)
    return CargoDistribution(
        geometry=geo, mass_kg=mass, z_moment_kg_m=mass * zc,
        x_moment_kg_m=mass * xc, y_moment_kg_m=mass * geo.breadth_m / 2.0,
        z2_moment_kg_m2=mass * zc**2,
    )


def test_hold_geometry_rejects_bad_input():
    with pytest.raises(ValueError):
        HoldGeometry(length_m=0.0, breadth_m=1.0, depth_m=1.0)
    with pytest.raises(ValueError):
        HoldGeometry(length_m=1.0, breadth_m=1.0, depth_m=1.0, repose_angle_deg=95.0)


def test_cargo_state_moments():
    geo = _geometry()
    dist = _uniform_distribution(geo, 1000.0, zc=2.5)
    state = cargo_state(dist)
    assert state["total_mass_kg"] == pytest.approx(4000.0)
    assert state["vcg_m"] == pytest.approx(2.5)
    assert state["lcg_m"] == pytest.approx(10.0)
    assert state["tcg_m"] == pytest.approx(6.0)
    assert state["z_spread_m"] == pytest.approx(0.0, abs=1e-9)


def test_zone_moment_shard_check_is_exact():
    geo = _geometry()
    rng = np.random.default_rng(3)
    n = geo.n_zones
    mass = rng.uniform(100.0, 200.0, n)
    dist = CargoDistribution(
        geometry=geo, mass_kg=mass, z_moment_kg_m=mass * rng.uniform(1.0, 4.0, n),
        x_moment_kg_m=mass * rng.uniform(0.0, 20.0, n),
        y_moment_kg_m=mass * 6.0, z2_moment_kg_m2=mass * 4.0,
    )
    out = zone_moment_check(dist)
    assert out["max_rel"] < 1e-12


def test_free_surface_bounds_are_ordered_and_void_corrected():
    geo = _geometry()
    dist = _uniform_distribution(geo, 1000.0)
    solid = free_surface_bounds(dist, void_fraction=0.0)
    void = free_surface_bounds(dist, void_fraction=0.4)
    assert solid["bulk_density_kg_m3"] == pytest.approx(1000.0)
    assert void["bulk_density_kg_m3"] == pytest.approx(600.0)
    assert void["free_surface_moment_kg_m"] == pytest.approx(
        0.6 * solid["free_surface_moment_kg_m"], rel=1e-12
    )
    assert void["free_surface_moment_grain_kg_m"] <= void["free_surface_moment_kg_m"] + 1e-12
    expected_i = geo.length_m * geo.breadth_m**3 / 12.0
    assert solid["free_surface_inertia_m4"] == pytest.approx(expected_i)


def test_gm_mapping_is_linear_and_mass_conserving():
    geo = _geometry()
    heavy = _uniform_distribution(geo, 1000.0, zc=3.0)
    # same mass, cargo piled higher (segregated large particles at the top)
    zc = np.array([2.0, 2.5, 3.4, 4.1])
    mass = np.full(geo.n_zones, 1000.0)
    piled = CargoDistribution(
        geometry=geo, mass_kg=mass, z_moment_kg_m=mass * zc,
        x_moment_kg_m=heavy.x_moment_kg_m, y_moment_kg_m=heavy.y_moment_kg_m,
        z2_moment_kg_m2=mass * zc**2,
    )
    out = gm_mapping(heavy, piled, displacement_t=20000.0, km_m=9.0,
                     void_before=0.4, void_after=0.35)
    d_vcg = float(np.mean(zc)) - 3.0
    assert out["d_vcg_m"] == pytest.approx(d_vcg)
    assert out["d_kg_vessel_m"] == pytest.approx(4000.0 / 20_000_000.0 * d_vcg)
    assert out["d_gm_liquid_m"] == pytest.approx(-out["d_kg_vessel_m"] - out["d_fsc_liquid_m"])
    assert out["d_gm_grain_m"] == pytest.approx(-out["d_kg_vessel_m"] - out["d_fsc_grain_m"])
    # lowering the void ratio raises the bulk density and therefore the FSC
    assert out["d_fsc_liquid_m"] > 0.0
    assert out["d_fsc_grain_m"] < out["d_fsc_liquid_m"]
    with pytest.raises(ValueError):
        gm_mapping(heavy, _uniform_distribution(geo, 900.0), displacement_t=20000.0,
                   km_m=9.0, void_before=0.4, void_after=0.4)


def test_floor_pressure_matches_the_static_weight():
    geo = _geometry()
    dist = _uniform_distribution(geo, 1000.0)
    load = np.full(geo.n_zones, 1000.0 * 9.81 / geo.zone_length_m)  # N/m of floor length
    dist.base_load_N_m = load
    dist.base_load_peak_N_m = 2.0 * load
    out = floor_pressure(dist)
    assert out["total_load_rel_error"] == pytest.approx(0.0, abs=1e-9)
    assert out["peak_pressure_Pa"] == pytest.approx(2.0 * 1000.0 * 9.81 / geo.zone_length_m**2)
    assert out["zone_peak_over_mean"][0] == pytest.approx(2.0)


def test_slice_to_hold_extrudes_the_slice_across_the_breadth():
    geo = HoldGeometry(length_m=8.0, breadth_m=4.0, depth_m=3.0, n_longitudinal=4)
    slice_mass = np.array([10.0, 12.0, 11.0, 9.0])          # kg per unit length
    z_com = np.array([0.4, 0.5, 0.6, 0.45])
    x_com = np.array([1.0, 3.0, 5.0, 7.0])
    dist = slice_to_hold(
        zone_mass_slice_kg=slice_mass,
        zone_z_moment_slice_kg_m=slice_mass * z_com,
        zone_x_moment_slice_kg_m=slice_mass * x_com,
        zone_x_com_m=x_com, zone_z_com_m=z_com,
        geometry=geo, bed_width_m=8.0,
    )
    assert dist.total_mass_kg == pytest.approx(float(slice_mass.sum()) * geo.breadth_m)
    state = cargo_state(dist)
    assert state["vcg_m"] == pytest.approx(float(np.sum(slice_mass * z_com) / slice_mass.sum()))
    assert state["tcg_m"] == pytest.approx(geo.breadth_m / 2.0)


def test_distribution_from_record_round_trip():
    geo = HoldGeometry(length_m=0.4, breadth_m=1.0, depth_m=0.2, n_longitudinal=4)
    n = 4
    area = np.array([0.01, 0.012, 0.011, 0.009])
    z_com = np.array([0.03, 0.04, 0.05, 0.035])
    record = {
        "spec": {"name": "unit-test"},
        "stats": {
            "x_area": (area * 10).tolist(),
            "x_moment": (area * z_com * 10).tolist(),
            "x_z_moment": (area * 0.05 * 10).tolist(),
            "x_moment2": (area * z_com**2 * 10).tolist(),
            "x_edges": np.linspace(0.0, 0.4, 5).tolist(),
            "n_samples": 10,
        },
        "base_load": {"mean_pressure_Pa": [100.0] * n, "peak_pressure_Pa": [200.0] * n},
    }
    dist = distribution_from_record(record, geo)
    assert dist.total_mass_kg == pytest.approx(float(area.sum()) * 1000.0 * geo.breadth_m)
    assert cargo_state(dist)["vcg_m"] == pytest.approx(float(np.sum(area * z_com) / area.sum()))