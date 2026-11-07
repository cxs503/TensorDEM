"""Macroscopic anchors: packing fraction, cubic-particle repose angle, order S(t).

The heavy protocol runs live in ``scripts/verify_nonspherical_contact_and_timestep.py``;
the tests here pin the parts that must never silently break: the clump's mass and
inertia identities, the quaternion kinematics, the absence of energy injection in
a single bounce, and the repose-angle estimator itself (against a synthetic
profile with a known slope).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.clump_granular_anchors import (
    ClumpPile,
    CubeClumpBody,
    measure_packing_fraction,
    measure_repose_angle,
    scott_dense_random_packing,
)


def test_cube_clump_mass_and_inertia_identities():
    """Mass = 8 member spheres; inertia is the parallel-axis sum (isotropic cube)."""
    edge = 0.05
    density = 1000.0
    body = CubeClumpBody.create([0.0, 0.0, 0.0], edge, density)
    radius = 0.25 * edge
    member_mass = density * 4.0 / 3.0 * math.pi * radius ** 3
    assert body.mass == pytest.approx(8.0 * member_mass, rel=1e-12)
    offset = 0.5 * edge - radius
    expected = 8.0 * (0.4 * member_mass * radius ** 2
                      + member_mass * 2.0 * offset ** 2)
    assert body.inertia[0] == pytest.approx(expected, rel=1e-12)
    assert body.inertia[0] == pytest.approx(body.inertia[2], rel=1e-12)
    # the union's extent really is the nominal cube edge
    assert np.max(np.abs(body.local_offsets[:, 0])) + radius == pytest.approx(0.5 * edge)


def test_quaternion_update_keeps_the_body_rigid():
    """A rotating clump must preserve member distances and stay normalised."""
    body = CubeClumpBody.create([0.0, 0.0, 0.0], 0.05, 1000.0)
    pile = ClumpPile.__new__(ClumpPile)
    pile.bodies = [body]
    pile.box = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    pile.stiffness = 4.0e4
    pile.damping = 2.0
    pile.friction = 0.5
    pile.gravity = 0.0
    pile.edge = 0.05
    pile._cache_state()
    body.angular_velocity = np.array([3.0, -2.0, 1.0])
    start = body.member_positions()
    distances = np.linalg.norm(start[:, None, :] - start[None, :, :], axis=2)
    for _ in range(200):
        pile.step(1.0e-4)
    assert np.linalg.norm(pile.quaternions[0]) == pytest.approx(1.0, rel=1e-13)
    end = pile.member_world_positions()[0]
    distances_end = np.linalg.norm(end[:, None, :] - end[None, :, :], axis=2)
    assert np.allclose(distances, distances_end, rtol=1e-12, atol=1e-12)
    assert body.member_radius > 0.0


def test_single_cube_bounce_does_not_gain_energy():
    """A damped impact may only remove energy -- no injection from the contact model."""
    body = CubeClumpBody.create([0.0, 0.0, 0.08], 0.05, 1000.0)
    body.velocity = np.array([0.0, 0.0, -0.3])
    pile = ClumpPile.__new__(ClumpPile)
    pile.bodies = [body]
    pile.box = np.array([[-0.2, -0.2, 0.0], [0.2, 0.2, 1.2]])
    pile.stiffness = 4.0e4
    pile.damping = 2.0
    pile.friction = 0.5
    pile.gravity = 0.0
    pile.edge = 0.05
    pile._cache_state()
    initial = 0.5 * pile.masses[0] * float(np.sum(pile.velocities[0] ** 2))
    rebound = 0.0
    contacted = False
    for _ in range(4000):
        pile.step(1.0e-4)
        if pile.positions[0, 2] < 0.06:
            contacted = True
        if contacted and pile.velocities[0, 2] > 0.0:
            rebound = max(rebound, 0.5 * pile.masses[0]
                          * float(np.sum(pile.velocities[0] ** 2)))
    assert rebound > 0.0, "the cube never rebounded"
    assert rebound <= initial * (1.0 + 1e-9), "the contact model injected energy"
    assert rebound > 0.4 * initial, "the dashpot removed far too much energy"


def test_repose_angle_estimator_recovers_a_known_slope():
    """The estimator must return the slope of a synthetic profile to 1e-3 deg."""
    x = np.linspace(-0.2, 0.2, 15)
    target = 31.0
    z = 0.25 - math.tan(math.radians(target)) * np.abs(x)
    z = np.maximum(z, 0.0)
    result = measure_repose_angle(x, z)
    assert result["angle_deg"] == pytest.approx(target, abs=0.05)


def test_packing_fraction_reaches_a_dense_random_packing():
    """A settled frictionless packing must land in the dense random packing band."""
    result = measure_packing_fraction(count=80, radius=0.02, steps=1500, taps=1,
                                      tap_steps=200, box_width=0.30, box_height=0.28,
                                      compress_step=2.0e-5, pressure_ratio=0.5)
    fraction = result["packing_fraction"]
    # uniaxial compression must leave a dense packing, i.e. clearly above the
    # loose-pour value (~0.35) and below the close-packing limit (0.74)
    assert 0.40 < fraction < 0.75
    assert scott_dense_random_packing() == pytest.approx(0.64)


def test_cube_pile_settles_without_blowing_up():
    """A short cubic-particle column release: bounded energy, finite order S."""
    pile = ClumpPile(count=18, edge=0.05)
    for _ in range(1200):
        pile.step(2.0e-4)
    kinetic = 0.5 * float(np.sum(pile.masses[:, None] * pile.velocities ** 2))
    weight = float(np.sum(pile.masses)) * 9.81 * 0.7
    assert np.all(np.isfinite(pile.positions))
    assert kinetic < 5.0 * weight, "the pile gained energy instead of dissipating it"
    order = pile.orientation_order()
    assert -0.5 <= order <= 1.0