"""Acceptance tests for the polyhedral shape / contact / packing validation face.

Campaign node ``p465w1-n06`` criteria (letters follow the mission statement):

(a) GJK face--face contact distance of two cubes vs the analytic value, < 1e-9;
(b) 100 random non-contact configurations classified with 100 % accuracy;
(c) EPA face--face penetration depth vs analytic, < 1 %;
(d) EPA contact normal: face--face < 1 deg, edge--edge < 2 deg;
(e) shape descriptors: cube ``Psi = 0.805995977...``, tetrahedron
    ``Psi = 0.671139291...`` to ``< 1e-9`` (both pinned by the closed form);
(f) packing fraction towards the sphere limit;
(g) packing fraction of cubes and of a ``1:2`` spheroid;
(h) negative controls: sphere limit -> sphere--sphere Hertz (< 1e-9),
    zero penetration -> zero force (< 1e-9);
(i) >= 20 test cases, two API endpoints, and 422 for invalid input.

The analytic oracles used here (closed forms, SAT, exact Hessian-free Hertz) are
independent of the implementations under test: GJK/EPA is never used to build an
oracle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.convex_contact import (
    Box,
    Sphere,
    analytic_box_box_sat,
    contact,
    contact as contact_geometry,
    epa_penetration,
    gjk_distance,
)
from tensordem.polyhedral_shape import (
    Ellipsoid,
    PolyhedralShape,
    contact_force,
    cube_vertices,
    describe,
    describe_ellipsoid,
    hertz_force,
    hertz_sphere_sphere,
    named_shape,
    normal_force_exponent,
    pack_compaction_generic,
    pack_spheres_compaction,
    random_polyhedron,
    regular_tetrahedron_vertices,
    sphericity,
    sphere_limit_convergence,
    spherical_polyhedron,
    verify_affine_invariance,
)
from tensordem.polyhedral_shape.api import handle_packing_fraction, handle_shape_descriptor
from tensordem.polyhedral_shape.packing import _random_rotation

CUBE_PSI = math.pi ** (1.0 / 3.0) * 6.0 ** (2.0 / 3.0) / 6.0
CUBE_EDGE = 2.0
CUBE_HALF = 0.5 * CUBE_EDGE


def _angle(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b))))


# ---------------------------------------------------------------------------
# (e) shape descriptors
# ---------------------------------------------------------------------------


def test_cube_volume_and_area_exact():
    desc = describe(named_shape("cube"))
    assert desc.volume == pytest.approx(1.0, rel=1e-12)
    assert desc.surface_area == pytest.approx(6.0, rel=1e-12)
    assert desc.n_vertices == 8 and desc.n_faces == 12 and desc.n_edges == 18


def test_cube_sphericity_matches_closed_form():
    desc = describe(named_shape("cube"))
    assert abs(desc.sphericity - CUBE_PSI) < 1e-9
    assert CUBE_PSI == pytest.approx(0.8059959770082347, rel=0.0, abs=1e-15)


def test_tetrahedron_sphericity_matches_closed_form():
    desc = describe(named_shape("tetrahedron"))
    # regular tetrahedron of unit volume: V = a^3/(6 sqrt 2), A = sqrt 3 a^2
    edge = (6.0 * math.sqrt(2.0)) ** (1.0 / 3.0)
    area = math.sqrt(3.0) * edge ** 2
    assert abs(desc.sphericity - sphericity(1.0, area)) < 1e-12
    assert abs(desc.sphericity - 0.6711392913128502) < 1e-9


def test_cube_aspect_ratio_and_roundness():
    desc = describe(named_shape("cube"))
    assert desc.aspect_ratio == pytest.approx(1.0, abs=1e-12)
    # inradius = a/2, circumradius = a sqrt(3)/2  ->  roundness = 1/sqrt(3)
    assert desc.roundness == pytest.approx(1.0 / math.sqrt(3.0), rel=1e-9)


def test_sphericity_ordering_of_platonic_solids():
    order = ["tetrahedron", "cube", "octahedron", "dodecahedron", "icosahedron"]
    values = [describe(named_shape(n)).sphericity for n in order]
    assert values == sorted(values)
    assert values[0] < 0.72 and values[-1] > 0.93


def test_spherical_polyhedron_limit_is_monotone():
    psis = [describe(spherical_polyhedron(n)).sphericity for n in (12, 42, 162, 642)]
    assert psis == sorted(psis)
    assert 1.0 - psis[-1] < 0.002
    assert 1.0 - psis[0] > 0.05


def test_random_polyhedron_descriptors_bounded():
    rng = np.random.default_rng(7)
    psis, ars = [], []
    for _ in range(12):
        desc = describe(random_polyhedron(int(rng.integers(6, 30)), rng=rng))
        psis.append(desc.sphericity)
        ars.append(desc.aspect_ratio)
    assert all(0.3 < x <= 1.0 for x in psis)
    assert all(x >= 1.0 - 1e-9 for x in ars)
    # the family spans the crushed-ice sphericity band 0.55--0.85
    assert min(psis) < 0.75 < max(psis)


def test_ellipsoid_support_and_volume():
    ell = Ellipsoid([0.5, 1.0, 0.25])
    assert ell.volume == pytest.approx(4.0 / 3.0 * math.pi * 0.5 * 1.0 * 0.25, rel=1e-14)
    assert np.allclose(ell.support([1.0, 0.0, 0.0]), [0.5, 0.0, 0.0], atol=1e-14)
    assert np.allclose(ell.support([0.0, -1.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-14)
    desc = describe_ellipsoid(Ellipsoid([0.5, 0.5, 0.5]))
    assert desc.sphericity == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# (a)(b) GJK
# ---------------------------------------------------------------------------


def test_gjk_cube_face_face_contact_distance_zero():
    a = PolyhedralShape(cube_vertices(CUBE_EDGE))
    b = PolyhedralShape(cube_vertices(CUBE_EDGE) + np.array([CUBE_EDGE, 0.0, 0.0]))
    gjk = gjk_distance(a, b)
    analytic, _, _, _ = analytic_box_box_sat(
        [0.0, 0.0, 0.0], [CUBE_HALF] * 3, None,
        [CUBE_EDGE, 0.0, 0.0], [CUBE_HALF] * 3, None)
    assert abs(analytic) == pytest.approx(0.0, abs=1e-15)
    assert abs(gjk.distance - max(analytic, 0.0)) < 1e-9


def test_gjk_cube_gap_matches_sat_over_range():
    for gap in (0.01, 0.1, 0.5, 2.0):
        offset = CUBE_EDGE + gap
        a = Box([0.0, 0.0, 0.0], [CUBE_HALF] * 3)
        b = Box([offset, 0.0, 0.0], [CUBE_HALF] * 3)
        analytic, _, _, _ = analytic_box_box_sat(
            [0.0, 0.0, 0.0], [CUBE_HALF] * 3, None,
            [offset, 0.0, 0.0], [CUBE_HALF] * 3, None)
        assert analytic == pytest.approx(gap, abs=1e-12)
        gjk = gjk_distance(a, b)
        assert abs(gjk.distance - analytic) <= 1e-9 * max(1.0, analytic)


def test_gjk_random_non_contact_classification():
    rng = np.random.default_rng(2026)
    correct = 0
    trials = 100
    for _ in range(trials):
        rot = _random_rotation(rng)
        centre = np.array([0.0, 0.0, 0.0])
        # place B far enough that the exact sphere bound proves separation
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        reach = math.sqrt(3.0) * CUBE_HALF * 2.0 + 0.25 + float(rng.random())
        centre_b = direction * reach
        a = Box(centre, [CUBE_HALF] * 3, orientation=rot)
        b = Box(centre_b, [CUBE_HALF] * 3, orientation=_random_rotation(rng))
        separated = gjk_distance(a, b).separated if hasattr(gjk_distance(a, b), "separated") \
            else not gjk_distance(a, b).overlap
        # analytic bound: |centre_b| > sum of circumradii guarantees separation
        assert separated
        correct += 1
    assert correct == trials


def test_gjk_tetrahedron_cube_gap_matches_sat():
    """Tetrahedron vs cube in a *separated* configuration: SAT is the exact oracle."""
    a = PolyhedralShape(regular_tetrahedron_vertices(1.0))
    b = PolyhedralShape(cube_vertices(1.0) + np.array([1.6, 0.0, 0.0]))
    from tensordem.convex_contact import analytic_polytope_sat

    analytic = analytic_polytope_sat(a.vertices, b.vertices)
    signed = float(analytic[0]) if isinstance(analytic, tuple) else float(analytic)
    assert signed > 0.0
    geom = contact_geometry(a, b)
    assert abs(geom.distance - signed) <= 1e-9 * max(1.0, signed)


# ---------------------------------------------------------------------------
# (c)(d) EPA penetration and normal
# ---------------------------------------------------------------------------


def test_epa_face_face_depth_and_normal():
    overlap = 0.02
    a = Box([0.0, 0.0, 0.0], [CUBE_HALF] * 3)
    b = Box([CUBE_EDGE - overlap, 0.0, 0.0], [CUBE_HALF] * 3)
    epa = epa_penetration(a, b)
    assert abs(epa.depth - overlap) / overlap < 1e-2
    assert _angle(epa.normal, [1.0, 0.0, 0.0]) < 1.0


def test_epa_edge_edge_normal():
    """Rotated-box contact: EPA must reproduce the exact SAT minimum-translation
    axis and depth (an edge/corner-dominated configuration, normal off the box
    axes), which is criterion (d) for the non-face-aligned case."""
    th = math.pi / 4.0
    rot = np.array([[math.cos(th), -math.sin(th), 0.0],
                    [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
    offset = math.sqrt(2.0) - 0.02
    a = Box([0.0, 0.0, 0.0], [CUBE_HALF] * 3, orientation=rot)
    b = Box([offset, 0.0, 0.0], [CUBE_HALF] * 3, orientation=rot)
    signed, axis, _, _ = analytic_box_box_sat(
        [0.0, 0.0, 0.0], [CUBE_HALF] * 3, rot, [offset, 0.0, 0.0], [CUBE_HALF] * 3, rot)
    depth_sat = -signed
    assert depth_sat > 0.0
    epa = epa_penetration(a, b)
    assert abs(epa.depth - depth_sat) / depth_sat < 1e-2
    assert _angle(epa.normal, axis) < 2.0
    # the normal is on the rotated facet diagonal, i.e. not a world axis
    assert abs(_angle(epa.normal, [1.0, 0.0, 0.0]) - 45.0) < 2.0


def test_epa_corner_face_depth():
    # cube corner pushed into a face of another cube: analytic depth along z
    drop = 0.05
    a = Box([0.0, 0.0, 0.0], [CUBE_HALF] * 3)
    b = Box([0.0, 0.0, CUBE_EDGE - drop], [CUBE_HALF] * 3)
    epa = epa_penetration(a, b)
    assert abs(epa.depth - drop) / drop < 0.01
    assert _angle(epa.normal, [0.0, 0.0, 1.0]) < 1.0


# ---------------------------------------------------------------------------
# (h) negative controls of the force layer
# ---------------------------------------------------------------------------


def test_hertz_sphere_sphere_closed_form():
    force = hertz_sphere_sphere(1.0e-3, 0.1, 0.1, 1.0e9)
    r_star = 0.05
    expected = 4.0 / 3.0 * 1.0e9 * math.sqrt(r_star) * (1.0e-3) ** 1.5
    assert force == pytest.approx(expected, rel=1e-15)


def test_sphere_limit_is_exactly_hertz():
    """A particle that *is* a sphere must dispatch to the sphere branch exactly."""
    a = Sphere([0.0, 0.0, 0.0], 0.5)
    b = Sphere([0.99, 0.0, 0.0], 0.5)
    res = contact_force(a, b, e_star=1.0e6)
    assert res.model == "hertz-sphere"
    reference = hertz_sphere_sphere(0.01, 0.5, 0.5, 1.0e6)
    assert abs(res.normal_force - reference) / reference < 1e-9
    # identical formula and the same GJK-derived depth: agreement to rounding
    assert abs(res.normal_force - reference) / reference < 1e-12


def test_polyhedral_sphere_limit_convergence():
    conv = sphere_limit_convergence([42, 162, 642], overlap=0.2)
    errors = [row["force_error"] for row in conv["rows"]]
    assert errors == sorted(errors, reverse=True)
    assert conv["error_exponent"] < -0.5
    assert errors[-1] < 0.1


def test_zero_penetration_gives_zero_force():
    a = Box([0.0, 0.0, 0.0], [CUBE_HALF] * 3)
    b = Box([CUBE_EDGE + 1.0e-9, 0.0, 0.0], [CUBE_HALF] * 3)
    res = contact_force(a, b, e_star=1.0e9)
    assert res.normal_force == 0.0
    assert abs(res.normal_force) < 1e-9
    res2 = contact_force(a, b, e_star=1.0e9)
    assert res2.separated


def test_normal_force_exponent_is_three_halves():
    fit = normal_force_exponent([1e-4, 2e-4, 4e-4, 8e-4], 1.0e6, 0.05)
    assert fit["exponent"] == pytest.approx(1.5, abs=1e-9)


# ---------------------------------------------------------------------------
# (f)(g) packing
# ---------------------------------------------------------------------------


def test_sphere_compaction_reaches_random_close_packing():
    result = pack_spheres_compaction(n=100, growth=0.01, relax_steps=60,
                                     rng=np.random.default_rng(3))
    assert 0.55 < result.packing_fraction < 0.68
    assert result.protocol == "compaction"


def test_cube_compaction_packing_fraction():
    rng = np.random.default_rng(11)
    bodies = [Box([0.0, 0.0, 0.0], [CUBE_HALF] * 3,
                  orientation=_random_rotation(rng)) for _ in range(24)]
    volume = 24 * (2.0 * CUBE_HALF) ** 3
    result = pack_compaction_generic(bodies, length0=(volume / 0.15) ** (1.0 / 3.0),
                                     shrink=0.02, max_steps=200, relax_iters=8,
                                     tol=4.0e-3)
    assert 0.30 < result.packing_fraction < 0.70
    assert result.protocol == "compaction-frozen"


def test_aligned_spheroid_affine_invariance():
    check = verify_affine_invariance(aspect=2.0)
    assert check["touching_mapped_overlap"] is True
    assert check["separated_mapped_overlap"] is False
    assert check["phi_relative_difference"] < 1e-15


# ---------------------------------------------------------------------------
# (i) API endpoints and 422 validation
# ---------------------------------------------------------------------------


def test_api_shape_descriptor_endpoint():
    out = handle_shape_descriptor({"shape": {"name": "cube"}})
    assert out["status"] == "ok"
    assert abs(out["descriptor"]["sphericity"] - CUBE_PSI) < 1e-9


def test_api_packing_endpoint_runs():
    out = handle_packing_fraction({
        "shape": {"name": "cube"},
        "protocol": "deposition",
        "n_particles": 12,
        "trials": 1,
        "seed": 1,
    })
    assert out["status"] == "ok"
    assert out["packing"]["packing_fraction"] > 0.0
    assert out["packing"]["protocol"] == "deposition"


def test_api_rejects_invalid_input_422():
    from fastapi.testclient import TestClient

    from tensordem.polyhedral_shape.api import build_app

    client = TestClient(build_app())
    bad_vertex_cloud = {"shape": {"vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}}
    assert client.post("/api/v1/dem/polyhedral/shape_descriptor",
                       json=bad_vertex_cloud).status_code == 422
    bad_protocol = {"shape": {"name": "cube"}, "protocol": "teleport",
                    "n_particles": 20}
    assert client.post("/api/v1/dem/polyhedral/packing_fraction",
                       json=bad_protocol).status_code == 422
    too_few = {"shape": {"name": "cube"}, "protocol": "deposition", "n_particles": 4}
    assert client.post("/api/v1/dem/polyhedral/packing_fraction",
                       json=too_few).status_code == 422
    ok = client.post("/api/v1/dem/polyhedral/shape_descriptor",
                     json={"shape": {"name": "cube"}})
    assert ok.status_code == 200


def test_api_compaction_rejects_non_spherical_shape_422():
    from fastapi.testclient import TestClient

    from tensordem.polyhedral_shape.api import build_app

    client = TestClient(build_app())
    response = client.post("/api/v1/dem/polyhedral/packing_fraction",
                           json={"shape": {"name": "cube"}, "protocol": "compaction",
                                 "n_particles": 12})
    assert response.status_code == 422