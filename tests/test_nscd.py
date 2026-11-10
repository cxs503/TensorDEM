"""Tests for the NSCD (Moreau) time-stepping contact solver."""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.nonlocal_rheology.nscd import NSCDError, NSCDWorld, Wall
from tensordem.nonlocal_rheology.validation import (SoftSphereWorld, _lattice_box,
                                                    tier1_single_impact, tier2_chain)


def _ball(radius=0.01, restitution=0.5, dt=2e-5, v0=1.0, sweep="gauss_seidel"):
    world = NSCDWorld(radius, mu=0.0, restitution=restitution,
                      gravity=(0.0, 0.0, 0.0), walls=[Wall(2, 0.0, 1)],
                      dt=dt, tol=1e-13, max_iter=200, sweep=sweep,
                      position_correction=False)
    world.set_state([[0.0, 0.0, 0.05]], vel=[[0.0, 0.0, -v0]])
    return world


def test_single_impact_restitution_exact_and_no_penetration():
    world = _ball()
    penetration = 0.0
    for _ in range(4000):
        world.step()
        penetration = max(penetration, world._max_overlap())
    assert penetration == 0.0
    assert abs(world.vel[0, 2] - 0.5) < 1e-12


def test_impact_series_backends_agree():
    results = {}
    for sweep in ("gauss_seidel", "jacobi", "colored"):
        world = _ball(sweep=sweep)
        for _ in range(4000):
            world.step()
        results[sweep] = float(world.vel[0, 2])
    assert abs(results["jacobi"] - results["gauss_seidel"]) < 1e-9
    assert abs(results["colored"] - results["gauss_seidel"]) < 1e-9


def test_anticipated_contact_offset_scales_with_dt():
    gaps = []
    for dt in (8e-5, 4e-5, 2e-5, 1e-5):
        row = tier1_single_impact(dt=dt, n_steps=int(0.12 / dt) + 50)
        gaps.append(row["bounce_gap"])
        assert row["restitution_error"] < 1e-9
        assert row["max_penetration"] == 0.0
    # the offset is bounded by one step of travel (it is O(dt) by construction)
    for gap, dt in zip(gaps, (8e-5, 4e-5, 2e-5, 1e-5)):
        assert gap <= 1.6 * dt
    assert gaps[-1] < gaps[0]


def test_resting_contact_carries_the_weight():
    world = NSCDWorld(0.01, mu=0.5, restitution=0.0, gravity=(0.0, 0.0, -9.81),
                      walls=[Wall(2, 0.0, 1)], dt=2e-5, tol=1e-12, sweep="gauss_seidel")
    world.set_state([[0.0, 0.0, 0.0100]])
    for _ in range(2000):
        world.step()
    assert abs(world.vel[0, 2]) < 1e-9
    assert world._max_overlap() == 0.0
    assert abs(world.wall_force(0)[2] + world.mass[0] * 9.81) < 1e-6


def test_head_on_collision_momentum_and_restitution():
    world = NSCDWorld(np.array([0.01, 0.01]), mu=0.3, restitution=0.6,
                      gravity=(0.0, 0.0, 0.0), dt=1e-5, tol=1e-12,
                      sweep="gauss_seidel")
    world.set_state([[0.0, 0.0, 0.0], [0.025, 0.0, 0.0]],
                    vel=[[0.5, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    p0 = world.total_momentum().copy()
    ke0 = world.total_kinetic_energy()
    for _ in range(3000):
        world.step()
        assert world._max_overlap() == 0.0
    assert np.linalg.norm(world.total_momentum() - p0) == 0.0
    rel = (world.vel[1, 0] - world.vel[0, 0]) / 0.7
    assert abs(abs(rel) - 0.6) < 1e-9
    assert world.total_kinetic_energy() < ke0


def test_chain_conserves_momentum_without_penetration():
    out = tier2_chain(n=8, steps=3000)
    assert out["momentum_error"] < 1e-14
    assert out["max_penetration"] == 0.0
    assert out["energy_monotone"]
    assert out["ke_ratio"] < 1.0
    assert out["final_velocities"][-1] > out["final_velocities"][0]


def test_colored_sweep_matches_sequential_on_a_cluster():
    """Colour partitioning must reproduce the sequential NLGS impulse."""
    n = 6
    r = 0.01
    pos = np.array([[r * 2.0 * (i % 3) * 1.02, r * 2.0 * (i // 3) * 1.02, 0.0]
                    for i in range(n)])
    outcomes = {}
    for sweep in ("gauss_seidel", "colored"):
        world = NSCDWorld(np.full(n, r), mu=0.4, restitution=0.3,
                          gravity=(0.0, 0.0, -9.81), walls=[Wall(2, 0.0, 1)],
                          dt=1e-5, tol=1e-12, max_iter=60, sweep=sweep)
        world.set_state(pos + np.array([0.0, 0.0, r]))
        world.vel[:] = np.array([0.05, 0.0, -0.02])
        for _ in range(200):
            world.step()
        outcomes[sweep] = (world.pos.copy(), world.vel.copy())
    assert np.allclose(outcomes["colored"][1], outcomes["gauss_seidel"][1], atol=2e-4)


def test_penalty_model_penetrates_where_nscd_does_not():
    """Adjudication: the penalty contact overlaps, the rigid contact does not."""
    penalty = SoftSphereWorld(0.01, kn=1.0e5, mu=0.0, restitution=0.5,
                              gravity=(0.0, 0.0, 0.0), walls=[Wall(2, 0.0, 1)],
                              dt=2e-5, tol=1e-13, max_iter=1)
    penalty.set_state([[0.0, 0.0, 0.05]], vel=[[0.0, 0.0, -1.0]])
    pen_max = 0.0
    for _ in range(4000):
        penalty.step()
        pen_max = max(pen_max, penalty._max_overlap())
    rigid = _ball()
    for _ in range(4000):
        rigid.step()
    assert pen_max > 0.0
    assert rigid._max_overlap() == 0.0
    assert pen_max > 10.0 * rigid._max_overlap()


def test_lattice_packing_has_no_initial_overlap_at_phi_085():
    from scipy.spatial import cKDTree
    pos, spacing = _lattice_box(4, 5, 3, 0.005, 0.85)
    assert abs(pos[:, 2].min() - 0.005) < 1e-12
    d, _ = cKDTree(pos).query(pos, 2)
    assert d[:, 1].min() >= 2.0 * 0.005 - 1e-12
    assert spacing > 2.0 * 0.005


def test_nscd_rejects_bad_configuration():
    with pytest.raises(NSCDError):
        NSCDWorld(0.01, restitution=1.5)
    with pytest.raises(NSCDError):
        NSCDWorld(0.01, mu=-0.1)
    with pytest.raises(NSCDError):
        NSCDWorld(0.01, sweep="mystery")
    with pytest.raises(NSCDError):
        Wall(3, 0.0)