"""Unit tests for the vectorised soft-sphere granular core."""

import math

import numpy as np
import pytest

from tensordem.granular import (
    BoxWalls,
    FunnelWalls2D,
    GranularConfig,
    GranularSystem,
    HoledFloor,
    PlaneWall,
    _cell_pairs,
    hex_lattice,
)


def test_config_derived_quantities():
    sphere = GranularConfig(radius=0.5, density=917.0)
    disk = GranularConfig(radius=0.5, density=917.0, dimensionality=2)
    assert sphere.diameter == pytest.approx(1.0)
    assert sphere.volume == pytest.approx(4.0 / 3.0 * math.pi * 0.5**3)
    assert disk.volume == pytest.approx(math.pi * 0.25 * 1.0)
    assert sphere.mass == pytest.approx(917.0 * sphere.volume)
    assert sphere.inertia_factor == pytest.approx(0.4)
    assert disk.inertia_factor == pytest.approx(0.5)
    assert sphere.recommended_dt() > 0.0


@pytest.mark.parametrize("bad", [
    {"radius": 0.0},
    {"radius": -1.0},
    {"density": 0.0},
    {"normal_stiffness": 0.0},
    {"friction": -0.1},
    {"dimensionality": 4},
])
def test_config_rejects_invalid(bad):
    kwargs = {"radius": 1.0}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        GranularConfig(**kwargs)


def test_lattice_is_overlap_free():
    x = hex_lattice(250, 1.0, rng=np.random.default_rng(0), jitter=0.0)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 2.0 - 1e-12


def test_lattice_respects_requested_count():
    for count in (10, 100, 1000):
        assert hex_lattice(count, 1.0, rng=np.random.default_rng(1)).shape == (count, 3)


def test_cell_pairs_match_brute_force():
    x = hex_lattice(200, 1.0, rng=np.random.default_rng(4), jitter=0.05)
    lo, hi = x.min(axis=0) - 4.0, x.max(axis=0) + 4.0
    i, j = _cell_pairs(x, 1.0, lo, hi, 2.1, 0.1)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    bi, bj = np.nonzero(np.triu(d < 2.1, 1))
    assert set(zip(i.tolist(), j.tolist())) == set(zip(bi.tolist(), bj.tolist()))


def test_cell_pairs_returns_empty_for_isolated_particles():
    x = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    i, j = _cell_pairs(x, 1.0, np.array([-1.0, -1.0, -1.0]),
                       np.array([51.0, 51.0, 51.0]), 2.1, 0.1)
    assert i.size == 0 and j.size == 0


def test_free_fall_matches_analytic_trajectory():
    cfg = GranularConfig(radius=0.5, gravity=9.81, normal_stiffness=1e6)
    system = GranularSystem(np.array([[0.0, 0.0, 10.0]]), cfg)
    dt = cfg.recommended_dt()
    steps = 400
    for _ in range(steps):
        system.step(dt)
    t = steps * dt
    z_analytic = 10.0 - 0.5 * 9.81 * t**2
    v_analytic = -9.81 * t
    assert system.x[0, 2] == pytest.approx(z_analytic, rel=2e-3)
    assert system.v[0, 2] == pytest.approx(v_analytic, rel=2e-3)


def test_closed_box_conserves_mass_and_keeps_particles_inside():
    cfg = GranularConfig(radius=1.0, gravity=1.0, normal_stiffness=2e4)
    x = hex_lattice(64, 1.0, rng=np.random.default_rng(2))
    lo, hi = x.min(axis=0) - 2.0, x.max(axis=0) + 2.0
    system = GranularSystem(x, cfg, [BoxWalls(lo, hi)])
    dt = cfg.recommended_dt()
    for _ in range(300):
        system.step(dt)
    assert system.mass_residual() == pytest.approx(0.0, abs=1e-12)
    assert system.max_wall_penetration() < 0.05 * cfg.radius
    assert np.all(system.x >= lo - cfg.radius) and np.all(system.x <= hi + cfg.radius)


def test_bed_energy_decays_and_stays_bounded():
    cfg = GranularConfig(radius=1.0, gravity=1.0, normal_stiffness=2e4)
    x = hex_lattice(64, 1.0, rng=np.random.default_rng(3))
    lo, hi = x.min(axis=0) - 2.0, x.max(axis=0) + 2.0
    system = GranularSystem(x, cfg, [BoxWalls(lo, hi)])
    dt = cfg.recommended_dt()
    energies = []
    for _ in range(3000):
        system.step(dt)
        energies.append(system.kinetic_energy())
    # never explosively larger than the available potential energy, and the
    # settled bed must be far quieter than the collapsing column
    potential = abs(system.potential_energy())
    assert max(energies) < max(potential, 1.0)
    assert energies[-1] < 0.2 * max(energies)


def test_single_particle_stays_at_rest_on_floor():
    cfg = GranularConfig(radius=0.5, gravity=9.81, normal_stiffness=2e5)
    system = GranularSystem(np.array([[0.0, 0.0, 0.5]]), cfg,
                            [PlaneWall((0, 0, 1), 0.0)])
    dt = cfg.recommended_dt()
    for _ in range(1500):
        system.step(dt)
    assert system.x[0, 2] == pytest.approx(0.5, abs=0.02)
    assert system.kinetic_energy() < 1e-6


def test_holed_floor_lets_particles_through_only_inside_the_hole():
    cfg = GranularConfig(radius=0.5, gravity=9.81, normal_stiffness=2e5)
    x = np.array([[0.0, 0.0, 0.6], [2.0, 0.0, 0.6]])
    system = GranularSystem(x, cfg, [HoledFloor(0.0, 0.6, 0.6)])
    dt = cfg.recommended_dt()
    for _ in range(4000):
        system.step(dt)
    assert system.x[0, 2] < 0.0  # fell through the hole
    assert system.x[1, 2] == pytest.approx(0.5, abs=0.02)  # held by the floor


def test_funnel_walls_are_confining():
    funnel = FunnelWalls2D(aperture=4.0, inclination_deg=60.0)
    cfg = GranularConfig(radius=0.5, gravity=0.0, normal_stiffness=2e5)
    x = np.array([[-6.0, 0.0, 3.0]])
    system = GranularSystem(x, cfg, [funnel])
    dt = cfg.recommended_dt()
    for _ in range(2000):
        system.step(dt)
    # pushed back inside the funnel: x*sin + z*cos >= -aperture/2*sin
    s = math.sin(math.radians(60.0))
    c = math.cos(math.radians(60.0))
    assert system.x[0, 0] * s + system.x[0, 2] * c > -0.5 * 4.0 * s


def test_remove_updates_mass_ledger_exactly():
    cfg = GranularConfig(radius=1.0, gravity=1.0)
    x = hex_lattice(10, 1.0, rng=np.random.default_rng(0), jitter=0.0)
    system = GranularSystem(x, cfg)
    initial = cfg.mass * 10
    system.remove(np.array([1, 3, 5]))
    assert system.removed_mass == pytest.approx(3.0 * cfg.mass, rel=1e-15)
    assert system.mass_residual() == pytest.approx(0.0, abs=1e-12)
    assert system.initial_mass == pytest.approx(initial, rel=1e-15)