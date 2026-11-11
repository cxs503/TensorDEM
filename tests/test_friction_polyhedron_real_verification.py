"""Falsifiable verification tests for the friction + polyhedron DEM kernel.

Companion to :mod:`tensordem.friction_polyhedron_real_verification`.

Every anchor here is a *measurement* driven through ``PolyhedronDEMSystem`` /
``compute_wall_contact_force``; each carries a control that must move the
result in the physically required direction, so a tautological implementation
cannot pass.  The three historical defects fixed on 2026-11-11 are locked down
by dedicated regression tests:

* ``test_wall_penetration_is_exact_for_flat_face`` / ``... bounding_radius ...``
  -- the wall contact used the bounding radius as the surface distance, so a
  resting cube carried k_n*(sqrt(3)-1)*a = 1.7e4 x its own weight.
* ``test_zero_rolling_resistance_injects_no_angular_energy`` -- the EPSD spring
  was left un-clamped when mu_r = 0 (negative damping).
* ``test_rolling_resistance_opposes_rolling`` -- the EPSD moment was subtracted
  in the wall branch, i.e. applied as a *driving* torque.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.friction_polyhedron import (
    ContactState,
    PolyhedronDEMSystem,
    PolyhedronParticle,
    compute_wall_contact_force,
    regular_polyhedron,
    wall_penetration,
)
from tensordem import friction_polyhedron_real_verification as rv


# ---------------------------------------------------------------------------
# Wall penetration / static load transfer
# ---------------------------------------------------------------------------
def test_wall_penetration_is_exact_for_flat_face():
    """A face-flat cube's surface sits a (not sqrt(3) a) below its centre."""
    size = 0.006
    verts = regular_polyhedron("cube", np.zeros(3), size)
    p = PolyhedronParticle.from_polytope(verts, density=1500.0)
    depth, point = wall_penetration(p.shape, np.array([0.0, 0.0, 1.0]))
    assert depth == pytest.approx(-size, rel=1e-12)
    assert point[2] == pytest.approx(-size, rel=1e-12)
    # face centroid, not a corner: |x|,|y| must be ~0
    assert abs(point[0]) < 1e-12 and abs(point[1]) < 1e-12


def test_wall_penetration_sphere_matches_radius():
    p = PolyhedronParticle.from_sphere(0.005, density=1500.0,
                                       position=np.array([0.0, 0.0, 0.005]))
    depth, point = wall_penetration(p.shape, np.array([0.0, 0.0, 1.0]))
    assert depth == pytest.approx(0.0, abs=1e-15)
    assert point[2] == pytest.approx(0.0, abs=1e-15)


def test_bounding_radius_penetration_defect_is_quantified():
    """The pre-fix formula is 1.7e4 x the weight for a resting cube."""
    rep = rv.bounding_radius_penetration_error()
    assert rep["force_ratio_vs_weight"] > 1.0e4
    assert rep["spurious_overlap_m"] > 0.004
    # the fixed routine reports the physical load, m g cos(theta)
    load = rep["static_load"]
    assert load["contacting"] is True
    assert load["ratio"] == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("theta", [0.0, 10.0, 25.0])
def test_static_normal_load_equals_weight_cosine(theta):
    """F_n / (m g) = cos(theta) to better than 2 % on a real contact."""
    system, particle = rv.build_incline_case(theta, mu=0.3, shape="cube")
    n_wall, d_wall = system.walls[0]
    # put the particle into the physical equilibrium penetration
    penetration = particle.mass * 9.81 * math.cos(math.radians(theta)) / system.config.k_n
    rv.seat_particle(system, particle, penetration)
    res, contacting = compute_wall_contact_force(
        particle.shape, particle.v, particle.omega, particle.mass,
        particle.inertia, n_wall, d_wall, ContactState(), system.config, 1e-5)
    assert contacting
    ratio = res["f_n"] / (particle.mass * 9.81)
    assert ratio == pytest.approx(math.cos(math.radians(theta)), rel=0.02)


# ---------------------------------------------------------------------------
# Rolling resistance: sign and zero limit
# ---------------------------------------------------------------------------
def _sphere_rolling_moment(mu_r, n_steps=200, dt=1e-5):
    """Return the accumulated EPSD moment for a sphere rolling on a plane."""
    system, particle = rv.build_incline_case(5.0, mu=0.6, mu_r=mu_r, shape="sphere")
    state = system._get_wall_state((0, 0))
    n_wall, d_wall = system.walls[0]
    for _ in range(n_steps):
        compute_wall_contact_force(particle.shape, particle.v, particle.omega,
                                   particle.mass, particle.inertia, n_wall,
                                   d_wall, state, system.config, dt)
        system.step(dt)
    return state.rolling_moment.copy(), particle


def test_zero_rolling_resistance_injects_no_angular_energy():
    """mu_r = 0 must leave the EPSD spring identically zero (no anti-damping)."""
    moment, _ = _sphere_rolling_moment(0.0)
    assert np.linalg.norm(moment) == 0.0


def test_rolling_resistance_opposes_rolling():
    """With mu_r > 0 the resisting moment must oppose the rolling direction."""
    moment, particle = _sphere_rolling_moment(0.2)
    assert np.linalg.norm(moment) > 0.0
    omega_roll = np.array([particle.omega[0], 0.0, particle.omega[2]])
    if np.linalg.norm(omega_roll) < 1e-12:
        pytest.skip("sphere did not roll on this configuration")
    assert float(np.dot(moment, omega_roll)) < 0.0, (
        "EPSD moment points along omega -> it drives instead of resisting")


def test_rolling_resistance_holds_a_sphere_on_a_shallow_incline():
    """Below atan(mu_r) a sphere must stay put; above it must roll."""
    static = rv.simulate_incline(5.0, mu=0.6, mu_r=0.3, shape="sphere")
    rolling = rv.simulate_incline(20.0, mu=0.6, mu_r=0.3, shape="sphere")
    assert static["slide_m"] < 1e-5
    assert rolling["slide_m"] > 1e-4


# ---------------------------------------------------------------------------
# A1: Coulomb sliding limit (rotation-locked control volume)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mu,expected", [(0.30, 16.70), (0.60, 30.96)])
def test_coulomb_sliding_limit_is_reproduced(mu, expected):
    """A translationally free block slides at tan(theta) = mu."""
    res = rv.find_critical_angle_sim(mu, 0.0, shape="cube", rotation_locked=True,
                                     n_bisect=6, t_end=0.06)
    assert res["method"] == "dem_bisection"
    assert res["critical_angle_deg"] == pytest.approx(expected, rel=0.05)


def test_sliding_limit_is_monotone_in_mu():
    """Control: a wrong/constant "measurement" cannot be monotone in mu."""
    angles = [rv.find_critical_angle_sim(mu, 0.0, shape="cube",
                                         rotation_locked=True, n_bisect=4,
                                         t_end=0.03)["critical_angle_deg"]
              for mu in (0.2, 0.45)]
    assert angles[0] < angles[1]


def test_negative_control_without_tangential_stiffness():
    """k_t = 0 removes the Coulomb cap -> the block must slide at 4 deg."""
    res = rv.negative_control_no_tangential_stiffness()
    assert res["slides_without_friction"] is True


# ---------------------------------------------------------------------------
# A2: EPSD rolling limit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mu_r,expected", [(0.15, 8.53), (0.30, 16.70)])
def test_rolling_limit_is_reproduced(mu_r, expected):
    res = rv.find_critical_angle_sim(0.6, mu_r, shape="sphere",
                                     n_bisect=6, t_end=0.06)
    # 6 bisections over (2, 60) deg resolve the bracket to +-0.45 deg (~5 % at
    # 8.5 deg); 10 % leaves room for the discrete slide-onset threshold.
    assert res["critical_angle_deg"] == pytest.approx(expected, rel=0.10)


# ---------------------------------------------------------------------------
# Known limitation: single-point face contact cannot shift its centre of pressure
# ---------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason=(
    "KNOWN LIMITATION: flat-face contact is collapsed into one contact point at "
    "the face centroid, so the normal reaction cannot shift to balance the "
    "friction moment; a free block creeps at angles below atan(mu). "
    "Fixing this (multi-point vertex-face contact) must flip this test -- then "
    "update docs/p273w3_n02_friction_polyhedron_verification.md"))
def test_face_flat_block_is_static_below_coulomb_limit():
    res = rv.simulate_incline(4.0, mu=0.3, shape="cube", rotation_locked=False)
    assert res["slide_m"] < 1e-5


def test_face_flat_block_creep_is_measured_and_nonzero():
    """Characterisation of the limitation above (must not silently vanish)."""
    free = rv.simulate_incline(4.0, mu=0.3, shape="cube", rotation_locked=False)
    locked = rv.simulate_incline(4.0, mu=0.3, shape="cube", rotation_locked=True)
    assert free["slide_m"] > 10.0 * max(locked["slide_m"], 1e-9)
    assert locked["slide_m"] < 1e-5


# ---------------------------------------------------------------------------
# Metrics helpers used by the report
# ---------------------------------------------------------------------------
def test_energy_ledger_stays_finite_for_settling_pair():
    """Two spheres settling under gravity must not gain energy (no blow-up)."""
    a = PolyhedronParticle.from_sphere(0.005, density=1500.0,
                                       position=np.array([0.0, 0.0, 0.012]))
    b = PolyhedronParticle.from_sphere(0.005, density=1500.0,
                                       position=np.array([0.0, 0.0, 0.0021]))
    cfg = __import__("tensordem.friction_polyhedron", fromlist=["x"]).FrictionPolyhedronConfig(
        k_n=1e5, gamma_n=2.0, k_t=1e5, mu=0.4, mu_r=0.1, k_r=1e3, density=1500.0)
    system = PolyhedronDEMSystem([a, b], cfg, gravity=np.array([0.0, 0.0, -9.81]))
    system.add_wall(np.array([0.0, 0.0, 1.0]), 0.0)
    system.record_initial_energy()
    for _ in range(3000):
        system.step(1e-5)
    assert np.isfinite(system.total_energy())
    assert abs(system.energy_balance_error()) < 0.05