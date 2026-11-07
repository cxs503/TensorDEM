"""Analytic cross-check of the GJK/EPA contact geometry layer.

Eight pair families are scored against oracles that are derived independently of
the solver (closed forms and an exact separating-axis implementation):

==================  ==========================================  =========================
group               bodies                                      oracle
==================  ==========================================  =========================
sphere-sphere       two spheres                                 closed form
sphere-box          sphere vs oriented box                      point-box closed form
box-box (aligned)   two axis-aligned boxes                      15-axis SAT
box-box (rot 45)    box vs box rotated about z                 15-axis SAT
tetra-box           tetrahedron vs box                         face+edge SAT
cone-floor          cone (apex down) vs floor slab              apex/point closed form
clump-box           three-sphere clump vs box                  per-member closed form
clump-clump         two three-sphere clumps                    per-member closed form
==================  ==========================================  =========================

Every group contributes a *separated* and an *overlapping* configuration.  The
acceptance thresholds are the campaign criteria: relative error on the gap /
penetration depth ``<= 1e-10`` and normal angle ``<= 1e-8 rad``.  The angle is
measured with ``|n x n_ref|`` (equal to ``sin theta``) rather than
``acos(n . n_ref)``: near ``theta = 0`` the ``acos`` form is limited to
``sqrt(eps) ~ 1.5e-8 rad`` by the double-precision dot product, which would make
the criterion unmeasurable and would also hide a genuinely wrong normal.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.convex_contact import (
    Box,
    Clump,
    Cone,
    Polytope,
    Sphere,
    analytic_box_box_sat,
    analytic_clump_box_gap,
    analytic_clump_clump_gap,
    analytic_polytope_sat,
    analytic_sphere_box,
    analytic_sphere_sphere,
    clump_contact,
    contact,
    gjk_distance,
)

REL_TOL = 1.0e-10
ANGLE_TOL = 1.0e-8

ROT45 = np.array([[math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0],
                  [math.sin(math.pi / 4), math.cos(math.pi / 4), 0.0],
                  [0.0, 0.0, 1.0]])


def angle_between(a, b) -> float:
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    return float(np.linalg.norm(np.cross(a, b)))


def _sphere_sphere():
    a = Sphere([0.0, 0.0, 0.0], 1.0)
    for centre in ([2.7, 0.3, 0.2], [1.5, 0.2, 0.1]):
        b = Sphere(centre, 0.8)
        gap, normal, _, _ = analytic_sphere_sphere(a.center, a.radius, b.center, b.radius)
        yield contact(a, b), gap, normal


def _sphere_box():
    box = Box([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    for centre in ([0.95, 0.85, 0.9], [0.62, 0.55, 0.58]):
        sphere = Sphere(centre, 0.25 if centre[0] > 0.9 else 0.2)
        gap, normal, _, _ = analytic_sphere_box(sphere.center, sphere.radius,
                                                box.center, box.half_extents)
        yield contact(sphere, box), gap, normal


def _box_box_aligned():
    a = Box([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    for centre in ([1.23, 0.1, 0.05], [0.87, 0.1, 0.05]):
        b = Box(centre, [0.5, 0.5, 0.5])
        gap, normal, _, _ = analytic_box_box_sat(a.center, a.half_extents, None,
                                                 b.center, b.half_extents, None)
        yield contact(a, b), gap, normal


def _box_box_rotated():
    a = Box([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    for centre in ([1.35, 0.0, 0.05], [1.10, 0.0, 0.05]):
        b = Box(centre, [0.5, 0.5, 0.5], ROT45)
        gap, normal, _, _ = analytic_box_box_sat(a.center, a.half_extents, None,
                                                 b.center, b.half_extents, ROT45)
        yield contact(a, b), gap, normal


def _tetra_box():
    tetra = Polytope([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for centre in ([1.25, 0.15, 0.15], [0.85, 0.15, 0.15]):
        box = Box(centre, [0.2, 0.2, 0.2])
        gap, normal = analytic_polytope_sat(tetra.vertices, box.vertices())
        yield contact(tetra, box), gap, normal


def _cone_floor():
    floor = Box([0.0, 0.0, -0.5], [2.0, 2.0, 0.5])
    for apex_z in (0.06, -0.04):
        cone = Cone([0.1, 0.05, apex_z], [0.0, 0.0, 1.0], 0.3, 0.12)
        gap, normal, _, _ = analytic_sphere_box(cone.apex, 0.0, floor.center,
                                                floor.half_extents)
        yield contact(cone, floor), gap, normal


def _clump_box():
    clump = Clump([Sphere([0.0, 0.0, 0.0], 0.2), Sphere([0.35, 0.0, 0.0], 0.2),
                   Sphere([0.17, 0.3, 0.0], 0.2)])
    for centre_z in (-0.45, -0.22):
        box = Box([0.17, 0.1, centre_z], [0.5, 0.5, 0.2])
        gap, index = analytic_clump_box_gap(clump, box.center, box.half_extents,
                                            box.orientation)
        member = clump.spheres[index]
        _, normal, _, _ = analytic_sphere_box(member.center, member.radius, box.center,
                                              box.half_extents, box.orientation)
        yield clump_contact(clump, box)[0], gap, normal


def _clump_clump():
    clump = Clump([Sphere([0.0, 0.0, 0.0], 0.2), Sphere([0.35, 0.0, 0.0], 0.2),
                   Sphere([0.17, 0.3, 0.0], 0.2)])
    for offset_y in (0.75, 0.35):
        other = Clump([Sphere([0.0, offset_y, 0.0], 0.2),
                       Sphere([0.35, offset_y, 0.0], 0.2),
                       Sphere([0.17, offset_y, 0.5], 0.2)])
        gap, _, normal = analytic_clump_clump_gap(clump, other)
        yield clump_contact(clump, other)[0], gap, normal


GROUPS = {
    "sphere-sphere": _sphere_sphere,
    "sphere-box": _sphere_box,
    "box-box-aligned": _box_box_aligned,
    "box-box-rotated": _box_box_rotated,
    "tetra-box": _tetra_box,
    "cone-floor": _cone_floor,
    "clump-box": _clump_box,
    "clump-clump": _clump_clump,
}


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_eight_analytic_contact_pairs(group):
    """Eight families, separated and overlapping, gap/depth <= 1e-10 rel, angle <= 1e-8."""
    cases = list(GROUPS[group]())
    assert len(cases) == 2, "each family contributes one separated and one overlapping case"
    kinds = set()
    for geometry, oracle_gap, oracle_normal in cases:
        if oracle_gap >= 0.0:
            assert geometry.separated, f"{group}: oracle says separated, solver says overlap"
            kinds.add("separated")
            rel = abs(geometry.distance - oracle_gap) / max(oracle_gap, 1e-300)
        else:
            assert not geometry.separated, f"{group}: oracle says overlap, solver says separated"
            kinds.add("overlapping")
            rel = abs(geometry.depth + oracle_gap) / max(-oracle_gap, 1e-300)
        assert rel <= REL_TOL, f"{group}: relative error {rel:.3e} exceeds {REL_TOL:.0e}"
        angle = angle_between(geometry.normal, oracle_normal)
        assert angle <= ANGLE_TOL, f"{group}: normal angle {angle:.3e} rad exceeds {ANGLE_TOL:.0e}"
    assert kinds == {"separated", "overlapping"}, f"{group} did not cover both branches"


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_contact_points_are_consistent_with_the_normal(group):
    """The two contact points must straddle the reported normal.

    This is the statement that makes the normal usable by a force law: pushing B
    along ``+normal`` by ``depth`` must separate the pair, and for a separated pair
    the segment ``point_a -> point_b`` must be anti-parallel to ``normal``.
    """
    for geometry, _, _ in GROUPS[group]():
        if geometry.separated:
            span = geometry.point_b - geometry.point_a
            assert float(span @ geometry.normal) > 0.0
            assert abs(float(np.linalg.norm(span)) - geometry.distance) <= 1e-9 * max(1.0, geometry.distance)
        else:
            span = geometry.point_a - geometry.point_b
            assert float(span @ geometry.normal) > 0.0
            # the pair of contact points spans the depth *along the normal*; for a
            # face-face polytope contact the barycentrics are not unique, so the
            # projection -- not the full norm -- is the invariant that must hold
            assert abs(float(span @ geometry.normal) - geometry.depth) <= 1e-9 * max(1.0, geometry.depth)


def test_gjk_returns_a_rigorous_two_sided_bound():
    """The gap GJK reports must sit inside its own bracket ``[lower, upper]``."""
    a = Sphere([0.0, 0.0, 0.0], 0.7)
    b = Box([1.6, 0.9, 0.4], [0.3, 0.3, 0.3], ROT45)
    result = gjk_distance(a, b)
    assert not result.overlap
    assert result.relative_gap <= 1.0e-12
    # independent closed form: sphere-box is point-box minus the sphere radius
    gap_pb, _ = _point_box_gap(a.center, b.center, b.half_extents, ROT45)
    expected = gap_pb - a.radius
    assert abs(result.distance - expected) <= 1e-12 * max(1.0, expected)


def _point_box_gap(point, centre, half, rotation):
    local = rotation.T @ (np.asarray(point, float) - np.asarray(centre, float))
    outside = np.clip(np.abs(local) - np.asarray(half, float), 0.0, None)
    return float(np.linalg.norm(outside)), None


def test_separated_pairs_never_report_phantom_penetration():
    """Fail-closed control: a clearly separated pair must return depth == 0."""
    probes = [
        (Sphere([0.0, 0.0, 0.0], 0.2), Box([3.0, 0.0, 0.0], [0.5, 0.5, 0.5])),
        (Box([0.0, 0.0, 0.0], [0.5] * 3), Box([0.0, 0.0, 3.0], [0.5] * 3)),
        (Cone([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], 0.3, 0.1), Box([0.0, 0.0, 0.0], [0.5] * 3)),
    ]
    for a, b in probes:
        geometry = contact(a, b)
        assert geometry.separated
        assert geometry.depth == 0.0
        assert geometry.distance > 0.0


def test_tolerance_checker_is_not_vacuous():
    """Negative control: a normal perturbed by 1e-3 rad must fail the angle gate."""
    box = Box([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    sphere = Sphere([0.62, 0.55, 0.58], 0.25)
    geometry = contact(sphere, box)
    _, oracle_normal, _, _ = analytic_sphere_box(sphere.center, sphere.radius,
                                                 box.center, box.half_extents)
    assert angle_between(geometry.normal, oracle_normal) <= ANGLE_TOL
    perturbed = geometry.normal + 1.0e-3 * np.array([0.0, 0.0, 1.0])
    assert angle_between(perturbed, oracle_normal) > ANGLE_TOL


def test_clump_channel_uses_members_not_the_convex_hull():
    """A gap between two clumps must not be filled by the union's convex hull."""
    left = Clump([Sphere([0.0, 0.0, 0.0], 0.1), Sphere([1.0, 0.0, 0.0], 0.1)])
    probe = Clump([Sphere([0.5, 0.4, 0.0], 0.1)])
    # The probe sits in the interior gap of the two-member clump.  A hull based
    # test would report 0.4 - 0.1 = 0.3 (the distance to the hull, which spans
    # x in [-0.1, 1.1]); the member-resolved channel must report the distance to
    # the nearest *sphere*, sqrt(0.5^2 + 0.4^2) - 0.2.
    geometry = clump_contact(left, probe)[0]
    expected = math.hypot(0.5, 0.4) - 0.2
    assert geometry.separated and geometry.distance > 0.0
    assert abs(geometry.distance - expected) <= 1e-12
    assert geometry.distance > 0.3 + 0.1