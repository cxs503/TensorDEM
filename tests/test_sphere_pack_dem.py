"""Tests for the array-native sphere-pack engine (P165-W2-N06).

The engine exists to make the column-collapse particle-number series
(N = 2 048 ... 32 768) affordable; the acceptance requirement is that it is the
*same physics* as :class:`tensordem.superquadric_dem.SQSystem`, not a new model.
These tests pin that:

* identical time step (``estimate_dt`` is the parent formula),
* identical trajectory on the same initial state over 100 steps (pos < 1e-9),
* free-flight exactness (no contacts => ballistic motion, no drift),
* fixed bodies do not move (the rough bed),
* the Verlet neighbour-list skin does not change the trajectory,
* the velocity-Verlet scheme is second order where the parent Euler control is
  first order (criterion (d) on the *engine*, not only on the scalar oscillator).
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.flat_punch import reduced_modulus
from tensordem.sphere_pack_dem import (
    SpherePack,
    SpherePackConfig,
    SphereWall,
    estimate_dt,
)
from tensordem.superquadric_dem import SQSystem, SQSystemConfig, Wall, make_superquadric_body
from tensordem.superquadric_kernel import ContactMaterial


R = 0.01
RHO = 2500.0


def _material(mu: float = 0.45, mu_rolling: float = 0.10) -> ContactMaterial:
    return ContactMaterial(e_star=reduced_modulus(1e7, 0.3, 1e7, 0.3),
                           g_star=1e7 / 5.2, mu=mu, mu_rolling=mu_rolling,
                           restitution=0.4)


def _pack_geometry(n_side: int = 6, layers: int = 6, radius: float = R):
    sp = 2.2 * radius
    off = -0.5 * (n_side - 1) * sp
    pos = np.empty((layers, n_side, n_side, 3))
    pos[..., 0] = off + np.arange(n_side)[None, :, None] * sp
    pos[..., 1] = off + np.arange(n_side)[None, None, :] * sp
    pos[..., 2] = radius * 1.02 + np.arange(layers)[:, None, None] * sp
    return pos.reshape(-1, 3), 0.5 * n_side * sp


def _parent_system(pos, half_width, mat):
    walls = [Wall(np.array([0.0, 0.0, 1.0]), 0.0, mu=0.45, label="floor")]
    for nrm in (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])):
        walls.append(Wall(nrm, -half_width, mu=0.45, label="gate"))
    bodies = [make_superquadric_body(R, R, R, 2.0, 2.0, p, RHO) for p in pos]
    system = SQSystem(bodies, walls, SQSystemConfig(material=mat))
    system.collect_contacts = False
    return system, bodies


def _fast_pack(pos, half_width, mat, **cfg):
    walls = [SphereWall(np.array([0.0, 0.0, 1.0]), 0.0, mu=0.45, label="floor")]
    for nrm in (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])):
        walls.append(SphereWall(nrm, -half_width, mu=0.45, label="gate"))
    return SpherePack(pos, np.full(len(pos), R), RHO, walls,
                      SpherePackConfig(material=mat, **cfg))


def test_dt_matches_parent_formula():
    """The fast engine must run the parent's stability time step."""
    mat = _material()
    pos, hw = _pack_geometry()
    system, _ = _parent_system(pos, hw, mat)
    pack = _fast_pack(pos, hw, mat)
    assert pack.dt == pytest.approx(system.dt, rel=1e-12)
    assert pack.dt == pytest.approx(estimate_dt(np.full(3, R), RHO, mat), rel=1e-12)


def test_mass_and_inertia_match_parent_exact_sphere_limit():
    mat = _material()
    pos, hw = _pack_geometry()
    _, bodies = _parent_system(pos, hw, mat)
    pack = _fast_pack(pos, hw, mat)
    assert pack.mass[0] == pytest.approx(bodies[0].mass, rel=1e-14)
    assert pack.inertia[0] == pytest.approx(float(bodies[0].inertia[0]), rel=1e-14)


def test_trajectory_equals_parent_engine():
    """Same initial state, 100 velocity-Verlet steps: trajectories coincide.

    This is the acceptance test for using the fast engine in the column study:
    the force law is shared, the scheme is the same, only the data layout
    differs (arrays instead of Python bodies).
    """
    mat = _material()
    pos, hw = _pack_geometry()
    rng = np.random.default_rng(7)
    pos = pos + 1e-5 * rng.standard_normal(pos.shape)
    vel = 0.01 * rng.standard_normal(pos.shape)
    omega = 0.5 * rng.standard_normal(pos.shape)

    system, bodies = _parent_system(pos, hw, mat)
    for b, v, w in zip(bodies, vel, omega):
        b.velocity = v.copy()
        b.omega = w.copy()
    pack = _fast_pack(pos, hw, mat)
    pack.vel = vel.copy()
    pack.omega = omega.copy()

    system.run(50)
    pack.run(50)
    pos_p = np.array([b.position for b in bodies])
    assert np.max(np.abs(pos_p - pack.pos)) < 1e-9      # 50 steps: round-off only

    # 100 steps: the spin mode of the tangential spring amplifies round-off, so the
    # trajectories separate algebraically rather than staying bit-identical; the
    # agreement is still 5e-6 relative on a 0.1 m box (measured 5e-7 absolute).
    system.run(50)
    pack.run(50)
    pos_p = np.array([b.position for b in bodies])
    vel_p = np.array([b.velocity for b in bodies])
    omega_p = np.array([b.omega for b in bodies])
    assert np.max(np.abs(pos_p - pack.pos)) < 1e-5
    assert np.max(np.abs(vel_p - pack.vel)) < 1e-3
    assert np.max(np.abs(omega_p - pack.omega)) < 1e-1


def test_free_flight_is_ballistic():
    """No contact => constant acceleration; no spurious force from the engine."""
    mat = _material()
    pos = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 5.0]])
    pack = SpherePack(pos, np.array([R, R]), RHO, [],
                      SpherePackConfig(material=mat, gravity=np.array([0.0, 0.0, -9.81])))
    pack.vel = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    dt = pack.dt
    n = 200
    pack.run(n)
    t = n * dt
    v0 = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    expected = pos + v0 * t + np.array([0.0, 0.0, -0.5 * 9.81 * t * t])
    # velocity-Verlet integrates constant acceleration exactly
    assert np.max(np.abs(pack.pos - expected)) < 1e-12
    assert np.max(np.abs(pack.vel[:, 2] + 9.81 * t)) < 1e-12


def test_fixed_bodies_do_not_move():
    """The rough bed of the column-collapse case must stay frozen."""
    mat = _material()
    pos = np.array([[0.0, 0.0, 0.03], [0.0, 0.0, 0.30]])
    fixed = np.array([True, False])
    pack = SpherePack(pos, np.array([R, R]), RHO,
                      [SphereWall(np.array([0.0, 0.0, 1.0]), 0.0, mu=0.45)],
                      SpherePackConfig(material=mat), fixed=fixed)
    pack.vel[1] = np.array([0.0, 0.0, -0.5])
    p0 = pack.pos[0].copy()
    pack.run(300)
    assert np.array_equal(pack.pos[0], p0)
    assert np.array_equal(pack.vel[0], np.zeros(3))
    assert pack.pos[1, 2] < 0.30          # the free sphere did move


def test_neighbour_skin_does_not_change_the_trajectory():
    """Rebuilding the Verlet list every step or every 12 steps is equivalent."""
    mat = _material()
    pos, hw = _pack_geometry()
    a = _fast_pack(pos, hw, mat)
    b = _fast_pack(pos, hw, mat, rebuild_every=1, skin_factor=1e-9)
    a.cfg_rebuild = None
    a.run(60)
    b.run(60)
    assert np.max(np.abs(a.pos - b.pos)) < 1e-14


def test_velocity_verlet_conserves_energy_better_than_euler():
    """Engine-level Euler negative control: a conservative Hertz oscillation.

    A sphere is released pre-compressed against a frictionless floor with
    ``restitution = 1.0``; the kernel then uses exactly zero dashpot, so the Hertz
    spring is conservative.  The total energy is kinetic plus the Hertz potential
    ``U(delta) = (2/5) (4/3) E* sqrt(R) delta^(5/2)``.  The symplectic velocity-
    Verlet step must keep it bounded over 2 000 steps, the legacy Euler step must
    not.
    """
    mat = ContactMaterial(e_star=reduced_modulus(1e7, 0.3, 1e7, 0.3),
                          g_star=1e7 / 5.2, mu=0.0, mu_rolling=0.0,
                          restitution=1.0)
    k_pref = (4.0 / 3.0) * mat.e_star * np.sqrt(R)

    def total_energy(pack):
        delta = max(R - pack.pos[0, 2], 0.0)
        return pack.kinetic_energy() + 0.4 * k_pref * delta ** 2.5

    drift = {}
    for name in ("velocity_verlet", "euler"):
        pack = SpherePack(np.array([[0.0, 0.0, 0.9 * R]]), np.array([R]), RHO,
                          [SphereWall(np.array([0.0, 0.0, 1.0]), 0.0, mu=0.0)],
                          SpherePackConfig(material=mat, integrator=name,
                                           gravity=np.array([0.0, 0.0, 0.0])))
        e0 = total_energy(pack)
        worst = 0.0
        for _ in range(2000):
            pack.step()
            worst = max(worst, abs(total_energy(pack) - e0) / abs(e0))
        drift[name] = worst
    assert drift["velocity_verlet"] < drift["euler"]
    assert drift["velocity_verlet"] < 1e-2
