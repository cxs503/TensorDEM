"""Multi-point (vertex-distributed) wall contact -- verification anchors (m16086).

The historical single-point wall law collapses a face-flat contact to the face
centroid, so the normal reaction passes through the particle centroid and a
freely rotating block creeps below ``atan(mu)`` (A5 of
``docs/p273w3_n02_friction_polyhedron_verification.md``).  The opt-in law
``FrictionPolyhedronConfig.wall_multi_point`` distributes the contact over the
tied vertices; these tests pin the properties that make it a *fix* rather than a
different model:

* default (``False``) path is untouched (no new result keys, same normal force);
* a face-flat contact keeps the historic total normal stiffness ``k_n``;
* the centre of pressure moves under pitch (the quantity the old law lost);
* a single-vertex (corner) contact falls back to the single-point law exactly;
* creep at ``theta < atan(mu)`` is suppressed, the critical angle recovers
  ``atan(mu)``, and sliding above the limit is unchanged;
* a block whose centre of mass leaves the support still tips (no spurious
  over-stabilisation).

Numbers behind the report: ``scripts/m16086_wall_multipoint_contact.py``,
``artifacts/m16086/wall_multipoint_contact.{json,csv}``.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tensordem.friction_polyhedron import (
    ContactState,
    FrictionPolyhedronConfig,
    compute_wall_contact_force,
    compute_wall_contact_force_multi_point,
)
from tensordem.friction_polyhedron_real_verification import (
    _rotation_y,
    build_incline_case,
    incline_basis,
)

MU = 0.30
FLAT_NORMAL = np.array([0.0, 0.0, 1.0])


HALF = 3e-3
DELTA = 1e-8  # imposed penetration of the deepest vertex, m


def _place(verts: np.ndarray, normal: np.ndarray, delta: float = DELTA) -> np.ndarray:
    """Translate ``verts`` so that the deepest vertex penetrates the plane by ``delta``."""
    depth = float((verts @ normal).min())
    return verts + normal * (-delta - depth)


def _make(verts: np.ndarray, cfg=None, label="cube"):
    from tensordem.friction_polyhedron import PolyhedronParticle
    cfg = cfg or FrictionPolyhedronConfig(k_n=1e5, gamma_n=20.0, k_t=1e5,
                                          mu=MU, mu_r=0.0, k_r=1e3, density=1500.0)
    center = verts.mean(axis=0)
    particle = PolyhedronParticle.from_polytope(verts, density=1500.0,
                                                position=center, label=label)
    return cfg, particle


def _flat_cube():
    """Cube with its bottom face parallel to the wall, deepest face 10 nm inside."""
    from tensordem.friction_polyhedron_real_verification import regular_polyhedron
    local = regular_polyhedron("cube", np.zeros(3), HALF)
    verts = _place(local, FLAT_NORMAL)
    return _make(verts)


def _state():
    return ContactState()


def test_default_path_unchanged():
    """``wall_multi_point=False`` must return the legacy result dictionary."""
    cfg, particle = _flat_cube()
    assert cfg.wall_multi_point is False
    state = _state()
    result, contacting = compute_wall_contact_force(
        particle.shape, particle.v, particle.omega, particle.mass, particle.inertia,
        FLAT_NORMAL, 0.0, state, cfg, 1e-5)
    assert contacting
    assert "n_contact_points" not in result
    # Face-flat single-point contact: the contact point is the face centroid,
    # i.e. directly below the CoM -> no lever arm for the normal force.
    r = result["contact_point"] - particle.shape.centroid()
    # face centroid sits straight below the CoM: no tangential lever arm
    assert abs(r[0]) < 1e-12 and abs(r[1]) < 1e-12
    assert r[2] == pytest.approx(-HALF, rel=1e-6)
    assert abs(result["force"][0]) < 1e-15 and abs(result["force"][1]) < 1e-15


def test_flat_face_keeps_total_normal_stiffness():
    """Shared per-vertex stiffness must reproduce the single-point ``k_n``."""
    cfg, particle = _flat_cube()
    single, _ = compute_wall_contact_force(
        particle.shape, particle.v, particle.omega, particle.mass, particle.inertia,
        FLAT_NORMAL, 0.0, _state(), cfg, 1e-5)
    multi, contacting = compute_wall_contact_force_multi_point(
        particle.shape, particle.v, particle.omega, particle.mass, particle.inertia,
        FLAT_NORMAL, 0.0, _state(), cfg, 1e-5)
    assert contacting
    assert multi["n_contact_points"] == 4
    assert multi["f_n"] == pytest.approx(single["f_n"], rel=1e-12)
    assert multi["energy_elastic"] == pytest.approx(single["energy_elastic"], rel=1e-12)
    # Face-flat -> the pressure centroid is the face centroid.
    assert multi["cop_offset_m"] == pytest.approx(0.0, abs=1e-15)


def test_centre_of_pressure_shifts_under_pitch():
    """A pitched block must carry its normal resultant off the face centroid."""
    from tensordem.friction_polyhedron_real_verification import regular_polyhedron
    phi = 2e-7  # rad; the induced depth split stays far below the 1 um baseline
    local = regular_polyhedron("cube", np.zeros(3), HALF)
    tilt = np.array([[math.cos(phi), 0.0, -math.sin(phi)],
                     [0.0, 1.0, 0.0],
                     [math.sin(phi), 0.0, math.cos(phi)]])
    verts = _place(local @ tilt.T, FLAT_NORMAL, delta=1e-6)
    cfg, pitched = _make(verts)
    multi, contacting = compute_wall_contact_force_multi_point(
        pitched.shape, pitched.v, pitched.omega, pitched.mass, pitched.inertia,
        FLAT_NORMAL, 0.0, _state(), cfg, 1e-5)
    single, _ = compute_wall_contact_force(
        pitched.shape, pitched.v, pitched.omega, pitched.mass, pitched.inertia,
        FLAT_NORMAL, 0.0, _state(), cfg, 1e-5)
    assert contacting
    assert multi["n_contact_points"] == 4
    assert multi["cop_offset_m"] > 0.0
    # pressure centroid moves towards the downhill (-x) edge; the single-point
    # law cannot represent this: it reports the tied-vertex mean
    assert abs(multi["contact_point"][0]) > 1e-7
    # the single-point law reports the *tied-vertex geometric* mean (here the
    # two deepest vertices, x = -HALF), not the force-weighted pressure centroid
    assert single["contact_point"][0] == pytest.approx(-HALF, rel=1e-3)
    assert abs(single["contact_point"][0] - multi["contact_point"][0]) > 1e-7


def test_corner_contact_falls_back_to_single_point():
    """One active vertex -> identical to the single-point law (tipping preserved)."""
    from tensordem.friction_polyhedron_real_verification import regular_polyhedron
    cfg = FrictionPolyhedronConfig(k_n=1e5, gamma_n=20.0, k_t=1e5, mu=MU, mu_r=0.0)
    local = regular_polyhedron("cube", np.zeros(3), HALF)
    # rotate the cube so that a body diagonal is vertical: one corner touches
    phi = math.atan(math.sqrt(2.0))
    rot = _rotation_y(math.degrees(phi)) @ np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(0.6155), -math.sin(0.6155)],
         [0.0, math.sin(0.6155), math.cos(0.6155)]])
    verts = _place(local @ rot.T, FLAT_NORMAL)
    cfg, particle = _make(verts, cfg)
    multi, contacting = compute_wall_contact_force_multi_point(
        particle.shape, particle.v, particle.omega, particle.mass, particle.inertia,
        FLAT_NORMAL, 0.0, _state(), cfg, 1e-5)
    single, _ = compute_wall_contact_force(
        particle.shape, particle.v, particle.omega, particle.mass, particle.inertia,
        FLAT_NORMAL, 0.0, _state(), cfg, 1e-5)
    assert contacting
    assert multi["n_contact_points"] == 1, multi["n_contact_points"]
    assert multi["f_n"] == pytest.approx(single["f_n"], rel=1e-12)
    assert np.allclose(multi["force"], single["force"], rtol=1e-12, atol=1e-15)


def _creep_slide(law: str, t_end: float = 0.02, dt: float = 1e-5) -> float:
    system, particle = build_incline_case(4.0, MU, 0.0)
    system.config = dataclasses.replace(
        system.config, wall_multi_point=(law == "multi"))
    _, downslope = incline_basis(4.0)
    x0 = particle.x.copy()
    for _ in range(int(round(t_end / dt))):
        system.step(dt)
    return float(np.dot(particle.x - x0, downslope))


def test_creep_below_friction_limit_is_suppressed():
    """A resting block must not creep at 4 deg when mu = 0.30 (atan = 16.7 deg)."""
    single = _creep_slide("single")
    multi = _creep_slide("multi")
    assert single > 5e-6, f"baseline defect disappeared ({single:.3e} m)"
    assert multi < 0.2 * single, (single, multi)


def test_sliding_above_limit_unchanged():
    """Above atan(mu) both laws must give the same sliding acceleration."""
    theta, t_end = 30.0, 5e-3
    a_analytic = 9.81 * (math.sin(math.radians(theta)) - MU * math.cos(math.radians(theta)))

    def slide(law: str) -> float:
        system, particle = build_incline_case(theta, MU, 0.0)
        system.config = dataclasses.replace(
            system.config, wall_multi_point=(law == "multi"))
        _, downslope = incline_basis(theta)
        x0 = particle.x.copy()
        for _ in range(int(round(t_end / 1e-5))):
            system.step(1e-5)
        return float(np.dot(particle.x - x0, downslope))

    s_single, s_multi = slide("single"), slide("multi")
    assert s_multi == pytest.approx(s_single, rel=2e-3)
    a_meas = 2.0 * s_multi / t_end**2
    assert abs(a_meas - a_analytic) / a_analytic < 0.05


def test_tall_block_still_tips():
    """A block whose CoM leaves the support polygon must tip in both laws."""
    theta, mu, t_end = 30.0, 0.90, 0.04
    tilts = {}
    for law in ("single", "multi"):
        normal, _ = incline_basis(theta)
        rot = _rotation_y(theta)
        a, h = 2e-3, 6e-3
        local = np.array([[sx * a, sy * a, sz * h]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                         dtype=np.float64)
        center = normal * (h + 1e-6)
        verts = local @ rot.T + center
        from tensordem.friction_polyhedron import PolyhedronDEMSystem, PolyhedronParticle
        cfg = FrictionPolyhedronConfig(k_n=1e5, gamma_n=20.0, k_t=1e5,
                                       mu=mu, mu_r=0.0, k_r=1e3, density=1500.0,
                                       wall_multi_point=(law == "multi"))
        particle = PolyhedronParticle.from_polytope(verts, density=1500.0,
                                                    position=center)
        system = PolyhedronDEMSystem([particle], cfg,
                                     gravity=np.array([0.0, 0.0, -9.81]))
        system.add_wall(normal, 0.0)
        for _ in range(int(round(t_end / 1e-5))):
            system.step(1e-5)
        v = np.asarray(particle.shape.vertices, dtype=np.float64)
        axis = v[4:].mean(axis=0) - v[:4].mean(axis=0)
        axis = axis / np.linalg.norm(axis)
        tilts[law] = math.degrees(math.acos(min(1.0, abs(float(np.dot(axis, normal))))))
    assert tilts["single"] > 20.0, tilts
    assert tilts["multi"] > 20.0, tilts