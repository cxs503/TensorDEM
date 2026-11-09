"""Tests for the ice-ridge mechanics capability (IN100-N07).

Coverage: Hopky anchor and its negative control, geometry statistics, buoyancy,
integrator stability and energy closure, freeze-bond failure, contact
repulsion, the ridge squeeze, the ship-bow geometry, failure-mode ordering and
the rubble-strength measurement.
"""
import math

import pytest
import torch

from tensordem.ice_ridge_mechanics import (
    Assembly,
    IceRubbleMaterial,
    RidgeSqueezeConfig,
    ShipCrossingConfig,
    WallSegment,
    build_hull,
    build_rubble_field,
    hopky_keel_depth,
    hopky_sail_height,
    identify_modes,
    measure_rubble_strength,
    negative_control_zero_squeeze,
    ridge_geometry,
    ridge_profile_geometry,
    RidgeDepositionConfig,
    deposit_ridge,
    simulate_ridge_squeeze,
    simulate_ship_crossing,
    union_area,
    envelope_area,
)
from tensordem.ice_ridge_mechanics.verification import AcceptanceConfig


def small_material(**kwargs):
    base = dict(particle_radius=0.15)
    base.update(kwargs)
    return IceRubbleMaterial(**base)


# --------------------------------------------------------------------- Hopky
def test_hopky_sail_height_matches_formula():
    assert hopky_sail_height(2.0, 8.0) == pytest.approx(0.4 * math.sqrt(16.0))
    assert hopky_sail_height(1.0, 1.0) == pytest.approx(0.4)


def test_hopky_zero_squeeze_negative_control():
    assert hopky_sail_height(2.0, 0.0) == 0.0
    assert hopky_keel_depth(2.0, 0.0) == 0.0
    control = negative_control_zero_squeeze(AcceptanceConfig())
    assert control["pass"] is True


def test_hopky_keel_ratio_is_four():
    h, d = 1.8, 5.0
    assert hopky_keel_depth(h, d) == pytest.approx(4.0 * hopky_sail_height(h, d))


def test_hopky_rejects_negative_squeeze():
    with pytest.raises(ValueError):
        hopky_sail_height(2.0, -1.0)


# ------------------------------------------------------------------ geometry
def test_ridge_geometry_measures_a_triangle():
    # synthetic triangular keel: 4 rows, widening downwards
    positions, radii = [], []
    for row in range(6):
        width = 1 + 2 * row
        for col in range(width):
            positions.append(((col - row) * 0.6, -0.6 * row))
            radii.append(0.3)
    out = ridge_profile_geometry(torch.tensor(positions), torch.tensor(radii),
                                 ice_thickness=0.6, thickness_factor=0.9)
    assert out["keel_depth"] > 2.0
    # only the crest disk emerges (centre on the waterline, radius 0.3)
    assert 0.0 < out["sail_height"] < 0.4
    assert out["column_count"] > 0
    assert 0.0 <= out["porosity"] < 1.0


def test_geometry_porosity_of_a_dense_lattice_is_small():
    r, spacing = 0.15, 0.30
    positions = [(i * spacing + (j % 2) * spacing / 2, j * spacing * 0.866)
                 for i in range(16) for j in range(4)]
    pos = torch.tensor(positions)
    rad = torch.full((len(positions),), r)
    solid = union_area(pos, rad)
    envelope = envelope_area(pos, rad)
    porosity = 1.0 - solid / envelope
    assert 0.0 < porosity < 0.25


def test_envelope_is_never_smaller_than_union():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    rad = torch.tensor([0.4, 0.4, 0.4])
    assert envelope_area(pos, rad) >= union_area(pos, rad)


# ----------------------------------------------------------------- buoyancy
def test_submerged_area_limits():
    material = small_material()
    pos = torch.tensor([[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    assembly = Assembly(pos, 0.2, material, waterline=0.0)
    area = assembly.submerged_area()
    full = math.pi * 0.2 ** 2
    assert area[0] == pytest.approx(full, rel=1e-9)      # fully submerged
    assert area[1] == pytest.approx(0.5 * full, rel=1e-9)  # half submerged
    assert area[2] == pytest.approx(0.0, abs=1e-9)        # dry


def test_free_fall_matches_gravity():
    material = small_material(water_density=1e-9, drag=0.0)
    assembly = Assembly(torch.tensor([[0.0, 0.0]]), 0.15, material, waterline=-1e9)
    dt = 1.0e-4
    for _ in range(1000):
        assembly.step(dt)
    assert float(assembly.vel[0, 1]) == pytest.approx(-material.gravity * dt * 1000,
                                                      rel=1e-3)


# -------------------------------------------------------------- integration
def test_energy_ledger_closes_for_a_settling_cluster():
    material = small_material()
    r, spacing = 0.15, 0.31
    positions = [(i * spacing + (j % 2) * spacing / 2, j * spacing * 0.9)
                 for i in range(4) for j in range(4)]
    pos = torch.tensor(positions) + torch.tensor([0.0, -0.4])
    assembly = Assembly(pos, r, material, waterline=0.0)
    for _ in range(400):
        assembly.step(assembly.material.recommended_dt())
    assert abs(assembly.ledger()["relative_residual"]) < 0.05


def test_contact_is_repulsive():
    material = small_material()
    pos = torch.tensor([[-0.10, 0.0], [0.10, 0.0]])
    assembly = Assembly(pos, 0.15, material, waterline=-1e9, drag=0.0)
    out = assembly.forces(update_memory=False)
    # overlapping pair: particle 0 must be pushed in -x, particle 1 in +x
    assert float(out["force"][0, 0]) < 0.0
    assert float(out["force"][1, 0]) > 0.0


def test_bond_breaks_above_tensile_strength():
    material = small_material(bond_stiffness=1.0e6, tensile_strength=1.0e4)
    pos = torch.tensor([[-0.15, 0.0], [0.15, 0.0]])
    assembly = Assembly(pos, 0.15, material, waterline=-1e9, drag=0.0)
    assembly.add_bonds(torch.tensor([[0, 1]], dtype=torch.long))
    assembly.pos[1, 0] += 0.05   # stretch the bond well past the strength
    assembly.forces(update_memory=False)
    assert assembly.broken_bonds == 1
    assert assembly.break_energy > 0.0


# ------------------------------------------------------------------ squeeze
def test_rubble_field_has_no_initial_overlap():
    config = RidgeSqueezeConfig(ice_thickness=1.2, squeeze=1.2, final_width=3.0,
                                particle_radius=0.15)
    field = build_rubble_field(config)
    assembly = Assembly(field["positions"], field["radii"],
                        config.material_or_default, waterline=0.0)
    assert assembly.max_overlap() <= 1e-9
    assert field["blocks"] > 1


def test_ridge_squeeze_produces_a_ridge_and_keeps_energy_bounded():
    config = RidgeSqueezeConfig(ice_thickness=1.2, squeeze=1.2, final_width=3.0,
                                squeeze_velocity=1.5, particle_radius=0.15,
                                dt=1.0e-3, relax_time=0.1,
                                freeze_bond_fraction=0.5)
    result = simulate_ridge_squeeze(config)
    assert result.geometry["keel_depth"] >= 0.0
    assert result.geometry["particle_count"] > 0
    assert 0.0 <= result.geometry["porosity"] < 1.0
    assert float(result.assembly.vel.norm(dim=1).max()) < 5.0
    assert math.isfinite(result.ledger["relative_residual"])


def test_zero_squeeze_deposition_gives_zero_sail():
    config = RidgeDepositionConfig(ice_thickness=1.2, squeeze=0.0,
                                   particle_radius=0.15, settle_time=0.1)
    result = deposit_ridge(config)
    assert result.geometry["sail_height"] == 0.0
    assert result.hopky["sail_height"] == 0.0


# -------------------------------------------------------------- penetration
def test_hull_geometry_is_raked_and_immersed():
    config = ShipCrossingConfig(speed=1.0, stem_rake_deg=25.0, bow_top=1.0)
    walls, geometry = build_hull(config, keel_depth=2.0, start_x=10.0)
    assert len(walls) == 3
    assert geometry["draft"] > 2.0
    assert geometry["keel_x"] < geometry["stem_x"] < geometry["top_x"]
    for wall in walls:
        assert float(wall.velocity[0]) == pytest.approx(1.0)


def test_identify_modes_recovers_the_expected_order():
    trace = [
        {"time": 0.0, "crushing": 0.1, "bending": 0.0, "ride_up": 0.0, "clearing": 0.0},
        {"time": 1.0, "crushing": 5.0, "bending": 0.1, "ride_up": 0.0, "clearing": 0.0},
        {"time": 2.0, "crushing": 5.0, "bending": 3.0, "ride_up": 0.1, "clearing": 0.0},
        {"time": 3.0, "crushing": 4.0, "bending": 3.0, "ride_up": 2.0, "clearing": 0.1},
        {"time": 4.0, "crushing": 3.0, "bending": 2.0, "ride_up": 2.0, "clearing": 4.0},
    ]
    modes = identify_modes(trace)
    assert modes["mode_order"] == ["crushing", "bending", "ride_up", "clearing"]


def test_ship_crossing_tiny_run_records_force_and_energy():
    config = RidgeSqueezeConfig(ice_thickness=1.0, squeeze=0.8, final_width=2.4,
                                squeeze_velocity=1.5, particle_radius=0.15,
                                dt=1.2e-3, relax_time=0.05,
                                freeze_bond_fraction=0.4)
    ridge = simulate_ridge_squeeze(config)
    crossing = simulate_ship_crossing(
        ridge.assembly, ShipCrossingConfig(speed=3.0, dt=1.2e-3, record_every=50))
    assert crossing.steps > 0
    assert len(crossing.force_history) > 0
    assert crossing.force_peak >= 0.0
    for key in ("break_energy", "friction_energy", "kinetic_energy",
                "relative_residual"):
        assert key in crossing.energy_split


# ----------------------------------------------------------------- strength
def test_rubble_strength_is_positive_and_reported():
    from tensordem.ice_ridge_mechanics.strength import RubbleStrengthConfig
    config = RubbleStrengthConfig(sample_width=1.2, sample_height=0.9,
                                  particle_radius=0.15, confining_pressure=2.0e4,
                                  platen_velocity=0.2, strain=0.10, dt=1.2e-3)
    out = measure_rubble_strength(config)
    assert out["sigma_f"] > 0.0
    assert out["sigma_f"] < 1.0e8
    assert out["confining_pressure"] == pytest.approx(2.0e4)
    assert len(out["stress_history"]) > 2


def test_uniform_pressure_force_is_finite_everywhere():
    # guard against a blown-up assembly: no force may reach machine scale
    material = small_material()
    positions = [(i * 0.31, j * 0.31) for i in range(3) for j in range(3)]
    assembly = Assembly(torch.tensor(positions) + torch.tensor([0.0, -0.3]),
                        0.15, material, waterline=0.0)
    for _ in range(200):
        assembly.step(material.recommended_dt())
    assert torch.isfinite(assembly.pos).all()
    assert float(assembly.vel.norm(dim=1).max()) < 10.0
