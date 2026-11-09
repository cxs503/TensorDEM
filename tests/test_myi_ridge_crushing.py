"""Plane-strain ridge crushing DEM: contact laws, ledger and reproducibility."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.myi_ridge import crushing


def _two_disc_dem(*, gap: float = -0.02, bond: bool = True, **kwargs) -> crushing.RidgeCrushDem:
    """Two discs side by side (``gap`` is the surface distance, negative = overlap)."""
    radius = 0.5
    positions = np.array([[1.0, 1.0], [2.0 + gap, 1.0]])
    bonds = np.array([[0, 1]], dtype=np.int64) if bond else np.zeros((0, 2), dtype=np.int64)
    particles = {
        "positions": positions,
        "radius": radius,
        "bonds": bonds,
        "porosity": 0.5,
        "count": 2,
        "bond_count": int(bonds.shape[0]),
    }
    specimen = crushing.RidgeSpecimen(height=4.0, length=8.0, porosity=0.5,
                                      particle_radius=radius, seed=1)
    params = crushing.CrushParameters(**kwargs)
    return crushing.RidgeCrushDem(specimen, params, particles=particles)


def test_specimen_is_stress_free_and_matches_porosity():
    specimen = crushing.RidgeSpecimen(height=6.0, length=8.0, porosity=0.30,
                                      particle_radius=0.4, seed=5)
    built = crushing.build_specimen(specimen)
    positions = np.asarray(built["positions"])
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    assert distance.min() > 2.0 * built["radius"] - 1e-12  # no initial overlap
    assert built["porosity"] == pytest.approx(0.30, abs=0.03)
    assert built["bond_count"] > built["count"]


def test_buoyant_free_acceleration_matches_analytic():
    specimen = crushing.RidgeSpecimen(height=3.0, length=3.0, porosity=0.45,
                                      particle_radius=0.3, seed=2)
    dem = crushing.RidgeCrushDem(
        specimen, crushing.CrushParameters(buoyancy=True, hull_velocity=0.0,
                                           advance=0.1, drag=0.0,
                                           velocity_damping=0.0))
    steps = 50
    for _ in range(steps):
        dem.step()
    free = dem.positions[:, 0] < 1e-9
    assert not bool(free.any())
    speed = float(dem.velocities[:, 1].mean())
    expected = (1025.0 - 917.0) / 917.0 * 9.81 * steps * dem.dt
    assert speed == pytest.approx(expected, rel=0.05)


def test_contact_force_is_stiffness_times_overlap():
    dem = _two_disc_dem(gap=-0.02, bond=False, contact_damping=0.0, drag=0.0)
    out = dem.forces()
    normal = float(out["force"][0, 0])
    assert normal == pytest.approx(-1.0e9 * 0.02, rel=1e-6)
    assert float(out["force"][1, 0]) == pytest.approx(1.0e9 * 0.02, rel=1e-6)


def test_tensile_cut_off_bounds_bond_force():
    dem = _two_disc_dem(gap=0.5, bond=True, drag=0.0)
    out = dem.forces()
    limit = dem.bond_tensile_force
    assert abs(float(out["force"][0, 0])) <= limit * (1.0 + 1e-9)


def test_cohesion_projection_bounds_shear_force():
    dem = _two_disc_dem(gap=0.0, bond=True, drag=0.0, velocity_damping=0.0)
    dem.velocities[1, 1] = 50.0  # large tangential velocity -> plastic slip
    for _ in range(3):
        dem.step()
    dem.forces()  # re-applies the plastic projection at the current state
    delta = dem.positions[dem.bond_j] - dem.positions[dem.bond_i]
    normal = delta / torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    shear = dem.bond_shear - (dem.bond_shear * normal).sum(dim=1, keepdim=True) * normal
    magnitude = float(torch.linalg.vector_norm(
        dem.params.bond_shear_stiffness * shear, dim=1)[0])
    cohesion = float(dem.bond_cohesion[0])
    assert magnitude <= cohesion * (1.0 + 1e-6)
    assert cohesion < dem.bond_shear_force
    assert dem.energy_fracture > 0.0


def test_rest_state_does_not_drift():
    specimen = crushing.RidgeSpecimen(height=6.0, length=6.0, porosity=0.35,
                                      particle_radius=0.35, seed=3)
    dem = crushing.RidgeCrushDem(specimen, crushing.CrushParameters(
        hull_velocity=0.0, advance=0.1, velocity_damping=0.05))
    for _ in range(400):
        dem.step()
    kinetic = float(0.5 * (dem.mass[:, None] * dem.velocities ** 2).sum())
    assert kinetic < 1e-2
    assert float(dem.positions.norm(dim=1).max()) < 1.0e3


def test_energy_ledger_closes_on_a_crushing_run():
    specimen = crushing.RidgeSpecimen(height=4.0, length=5.0, porosity=0.30,
                                      particle_radius=0.4, seed=4)
    params = crushing.CrushParameters(ice_strength=7.0e6, bond_strength_fraction=0.10,
                                      hull_velocity=2.0, advance=0.15)
    dem = crushing.RidgeCrushDem(specimen, params)
    out = dem.run(max_steps=20000)
    ledger = out["ledger"]
    assert ledger["closure_of_exchanged"] < 0.05
    assert ledger["fracture_energy"] > 0.0
    assert out["pressure_steady"] > 0.0


def test_crushing_run_is_reproducible():
    specimen = crushing.RidgeSpecimen(height=4.0, length=5.0, porosity=0.30,
                                      particle_radius=0.4, seed=4)
    params = crushing.CrushParameters(ice_strength=7.0e6, bond_strength_fraction=0.10,
                                      hull_velocity=2.0, advance=0.15)
    first = crushing.RidgeCrushDem(specimen, params).run(max_steps=20000)
    second = crushing.RidgeCrushDem(specimen, params).run(max_steps=20000)
    assert first["force_peak"] == second["force_peak"]
    assert first["bonds_broken"] == second["bonds_broken"]


def test_stiffer_bond_makes_the_specimen_stronger():
    specimen = crushing.RidgeSpecimen(height=4.0, length=5.0, porosity=0.30,
                                      particle_radius=0.4, seed=6)
    weak = crushing.RidgeCrushDem(specimen, crushing.CrushParameters(
        ice_strength=5.0e6, bond_strength_fraction=0.10, hull_velocity=2.0,
        advance=0.12)).run(max_steps=20000)
    strong = crushing.RidgeCrushDem(specimen, crushing.CrushParameters(
        ice_strength=10.0e6, bond_strength_fraction=0.10, hull_velocity=2.0,
        advance=0.12)).run(max_steps=20000)
    assert strong["pressure_steady"] > weak["pressure_steady"]


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        crushing.RidgeSpecimen(height=-1.0, length=5.0)
    with pytest.raises(ValueError):
        crushing.RidgeSpecimen(height=4.0, length=5.0, porosity=0.9)
    with pytest.raises(ValueError):
        crushing.CrushParameters(hull_velocity=-1.0)
    with pytest.raises(ValueError):
        crushing.CrushParameters(velocity_damping=1.5)
    with pytest.raises(ValueError):
        crushing.CrushParameters(pair_interval=0)