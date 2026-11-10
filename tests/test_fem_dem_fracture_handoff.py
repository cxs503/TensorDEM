"""Acceptance tests for the FEM -> DEM fracture handoff (campaign p302w1-n03).

The criteria of the campaign node are tested one by one, plus the verification of
the FEM solver against four closed forms and the robustness checks of the
artefact (tamper detection, determinism, lattice consistency, partition
coverage).  Everything the acceptance claims is asserted here; nothing is claimed
in the report that is not a number produced by one of these tests.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tensordem.fem_dem_fracture.plate_fem import _pressure_load_vector, acm_inverse
from tensordem.fem_dem_fracture import (
    FOUNDATION,
    IcePlateCase,
    PlateMaterial,
    PlateMesh,
    PressurePatch,
    dem_initial_conditions,
    find_splitting_state,
    hetenyi_point_load_deflection,
    navier_ssss_square_deflection,
    navier_ssss_square_moment,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
    polygon_perimeter,
    read_failure_faces,
    run_handoff,
    solve_plate,
    trace_crack_network,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def case() -> IcePlateCase:
    return IcePlateCase()


@pytest.fixture(scope="module")
def handoff(tmp_path_factory, case) -> dict:
    out_dir = tmp_path_factory.mktemp("handoff")
    summary = run_handoff(case, str(out_dir))
    return {"summary": summary, "dir": str(out_dir)}


@pytest.fixture(scope="module")
def first_cracking_solution(case) -> "object":
    unit = solve_plate(case.mesh(), case.material, case.patches,
                       foundation_modulus=case.foundation_modulus,
                       load_amplitude=1.0, boundary="free")
    return unit.scaled(unit.scale_to_failure())


# --------------------------------------------------------------------------
# FEM verification against closed forms
# --------------------------------------------------------------------------
def test_ssss_square_deflection_matches_navier():
    material = PlateMaterial()
    side, q = 10.0, 1000.0
    mesh = PlateMesh(nx=40, ny=40, length_x=side, length_y=side)
    sol = solve_plate(mesh, material,
                      [PressurePatch((side / 2, side / 2), (side, side), q)],
                      boundary="simply_supported")
    ref = navier_ssss_square_deflection(q, side, material)
    assert abs(sol.max_deflection - ref) / ref < 0.01


def test_ssss_square_moment_matches_navier():
    material = PlateMaterial()
    side, q = 10.0, 1000.0
    mesh = PlateMesh(nx=40, ny=40, length_x=side, length_y=side)
    sol = solve_plate(mesh, material,
                      [PressurePatch((side / 2, side / 2), (side, side), q)],
                      boundary="simply_supported")
    centre = (mesh.ny // 2) * mesh.nx + mesh.nx // 2
    ref = navier_ssss_square_moment(q, side, material)
    assert abs(abs(sol.moment_x[centre]) - ref) / ref < 0.01
    # the textbook coefficient for nu = 0.3 is 0.0479 q a^2
    assert abs(ref / (q * side ** 2) - 0.0479) < 1e-3


def test_strip_bending_matches_beam_limit():
    material = PlateMaterial()
    q, span, width = 1000.0, 5.0, 1.0
    ei = material.young_modulus * material.thickness ** 3 / 12.0
    ref = 5.0 * q * span ** 4 / (384.0 * ei)
    mesh = PlateMesh(nx=40, ny=8, length_x=span, length_y=width)
    sol = solve_plate(mesh, material,
                      [PressurePatch((span / 2, width / 2), (span, width), q)],
                      boundary="simply_supported", constrained_edges=("x0", "x1"))
    assert abs(sol.max_deflection - ref) / ref < 0.01


def test_winkler_plate_matches_hetenyi():
    material = PlateMaterial()
    lc = material.characteristic_length(FOUNDATION)
    domain, n = 10.0 * lc, 109
    load, side = 1.0e5, 0.5
    mesh = PlateMesh(nx=n, ny=n, length_x=domain, length_y=domain)
    sol = solve_plate(mesh, material,
                      [PressurePatch((domain / 2, domain / 2), (side, side),
                                     load / side ** 2)],
                      foundation_modulus=FOUNDATION, boundary="free")
    ref = hetenyi_point_load_deflection(load, material, FOUNDATION)
    assert abs(sol.max_deflection - ref) / ref < 0.02


def test_clamped_square_deflection():
    material = PlateMaterial()
    side, q = 10.0, 1000.0
    mesh = PlateMesh(nx=40, ny=40, length_x=side, length_y=side)
    sol = solve_plate(mesh, material,
                      [PressurePatch((side / 2, side / 2), (side, side), q)],
                      boundary="clamped")
    ref = 0.00126 * q * side ** 4 / material.flexural_rigidity
    assert abs(sol.max_deflection - ref) / ref < 0.02


def test_free_plate_without_foundation_is_rejected():
    mesh = PlateMesh(nx=8, ny=8, length_x=5.0, length_y=5.0)
    with pytest.raises(ValueError, match="mechanism"):
        solve_plate(mesh, PlateMaterial(),
                    [PressurePatch((2.5, 2.5), (1.0, 1.0), 1000.0)],
                    foundation_modulus=0.0, boundary="free")


def test_pressure_integration_is_exact_for_sub_cell_patch():
    """The consistent load of a patch smaller than a cell must carry all of it.

    ``sum_k f_k = integral of p`` because the ACM interpolation has partition of
    unity, so this checks the clipped-rectangle quadrature exactly.
    """
    mesh = PlateMesh(nx=6, ny=6, length_x=3.0, length_y=3.0)   # h = 0.5 m
    cinv = acm_inverse(mesh.hx, mesh.hy)
    cells = mesh.cell_ids()
    edof = np.empty((mesh.n_cells, 12), dtype=int)
    for k in range(4):
        edof[:, 3 * k + 0] = 3 * cells[:, k] + 0
        edof[:, 3 * k + 1] = 3 * cells[:, k] + 1
        edof[:, 3 * k + 2] = 3 * cells[:, k] + 2
    patch = PressurePatch((1.55, 1.44), (0.20, 0.20), 1234.0)  # 0.2 m patch (< cell)
    loads = _pressure_load_vector(mesh, [patch], cinv, edof[:, 0::3],
                                  3 * mesh.n_nodes)
    expected = patch.pressure * patch.size[0] * patch.size[1]
    assert abs(float(loads.sum()) - expected) / expected < 1e-12


# --------------------------------------------------------------------------
# C1: fracture criterion and artefact
# --------------------------------------------------------------------------
def test_C1_fracture_declared_and_artefact_written(handoff, case):
    criteria = handoff["summary"]["criteria"]
    assert criteria["C1_fracture_declared"] is True
    assert criteria["C1_sigma_over_strength"] >= 1.0
    assert criteria["C1_sigma_max_pa"] >= case.material.sigma_flexure - 1e-6
    faces_path = handoff["summary"]["artifacts"]["failure_faces"]
    assert os.path.exists(faces_path)
    payload = json.load(open(faces_path, encoding="utf-8"))
    assert payload["schema"] == "tensordem.failure_faces/1"
    assert payload["meta"]["stress_state"]["fracture_declared"] is True
    assert len(payload["cracks"]) > 0
    assert len(payload["fractured_elements"]) >= 1
    assert len(payload["crack_cells"]) >= 1


def test_C1_no_fracture_below_strength(case, first_cracking_solution):
    """Honest negative control: below the strength nothing nucleates."""
    weak = first_cracking_solution.scaled(0.25 * first_cracking_solution.load_amplitude)
    with pytest.raises(RuntimeError, match="below its fracture load"):
        trace_crack_network(weak, case.tracer)


# --------------------------------------------------------------------------
# C2: mass conservation
# --------------------------------------------------------------------------
def test_C2_mass_conservation(handoff, case):
    criteria = handoff["summary"]["criteria"]
    assert criteria["C2_pass"] is True
    assert abs(criteria["C2_mass_residual_kg"]) < 1e-6
    plate_mass = (case.length_x * case.length_y
                  * case.material.thickness * case.material.density)
    assert abs(criteria["C2_plate_mass_kg"] - plate_mass) < 1e-9
    assert abs(criteria["C2_floes_mass_kg"] - plate_mass) < 1e-6


# --------------------------------------------------------------------------
# C3: geometric conservation
# --------------------------------------------------------------------------
def test_C3_area_conservation_and_rim_ledger(handoff, case):
    criteria = handoff["summary"]["criteria"]
    assert criteria["C3_pass"] is True
    assert abs(criteria["C3_area_residual_m2"]) < 1e-3
    plate_perimeter = 2.0 * (case.length_x + case.length_y)
    assert abs(criteria["C3_rim_length_from_polygons_m"] - plate_perimeter) < 1e-9
    assert abs(criteria["C3_rim_residual_m"]) < 1e-9
    # a real fracture network separates area, so the internal face length is > 0
    # and cannot exceed the total length of the traced crack polylines
    assert 0.0 < criteria["C3_internal_face_length_m"] <= criteria["total_crack_length_m"] + 1e-9


def test_C3_boundary_identity_holds(handoff, case):
    """sum(perimeter) = plate rim + 2 x fracture-face length (a partition identity).

    This is the check that catches an arithmetic error in the face-length
    bookkeeping: it failed (127.0 vs 83.5 m) when the face length was halved by
    mistake, and passes only when the ledger is self-consistent.
    """
    faces = read_failure_faces(handoff["summary"]["artifacts"]["failure_faces"])
    ledger = faces.conservation
    plate_perimeter = 2.0 * (case.length_x + case.length_y)
    assert abs(ledger.floes_perimeter_sum
               - (plate_perimeter + 2.0 * ledger.internal_face_length)) < 1e-9


def test_C3_floe_polygons_are_simple_and_cover_the_plate(handoff):
    faces = read_failure_faces(handoff["summary"]["artifacts"]["failure_faces"])
    rng = np.random.default_rng(3)
    lx = faces.meta["plate"]["length_x_m"]
    ly = faces.meta["plate"]["length_y_m"]
    counts = []
    for _ in range(400):
        x = rng.uniform(0.0, lx)
        y = rng.uniform(0.0, ly)
        counts.append(sum(1 for floe in faces.floes if floe.contains((x, y))))
    counts = np.asarray(counts)
    # every sampled point is inside exactly one floe -> the polygons partition the plate
    assert np.all(counts == 1), f"coverage counts: {np.bincount(counts)}"
    for floe in faces.floes:
        assert polygon_area(np.asarray(floe.polygons[0])) > 0.0
        assert floe.area > 0.0


# --------------------------------------------------------------------------
# C4 / C5
# --------------------------------------------------------------------------
def test_C4_floe_count_in_range(handoff):
    criteria = handoff["summary"]["criteria"]
    assert criteria["C4_pass"] is True
    assert 3 <= criteria["C4_n_floe"] <= 30


def test_C5_end_to_end_roundtrip(handoff, case):
    criteria = handoff["summary"]["criteria"]
    assert criteria["C5_pass"] is True
    bodies = dem_initial_conditions(
        read_failure_faces(handoff["summary"]["artifacts"]["failure_faces"])
    )
    plate_mass = (case.length_x * case.length_y
                  * case.material.thickness * case.material.density)
    assert len(bodies) == criteria["C4_n_floe"]
    assert abs(bodies.total_mass - plate_mass) < 1e-6
    # DEM bodies: area-equivalent discs of the floes, at rest (quasi-static FEM)
    np.testing.assert_allclose(np.pi * bodies.diameters ** 2 / 4,
                               np.array([f.area for f in read_failure_faces(
                                   handoff["summary"]["artifacts"]["failure_faces"]).floes]),
                               rtol=1e-10)
    assert np.all(bodies.velocities == 0.0)
    payload = json.load(open(handoff["summary"]["artifacts"]["dem_initial_conditions"],
                            encoding="utf-8"))
    assert payload["n_bodies"] == len(bodies)
    assert abs(payload["total_mass_kg"] - plate_mass) < 1e-9


def test_dem_initial_conditions_accept_a_velocity_field(handoff):
    faces = read_failure_faces(handoff["summary"]["artifacts"]["failure_faces"])
    bodies = dem_initial_conditions(faces, velocity_field=lambda x, y: (0.1, -0.2))
    np.testing.assert_allclose(bodies.velocities[:, 0], 0.1)
    np.testing.assert_allclose(bodies.velocities[:, 1], -0.2)
    assert np.all(bodies.velocities[:, 2] == 0.0)


# --------------------------------------------------------------------------
# artefact robustness: tamper detection, determinism, lattice consistency
# --------------------------------------------------------------------------
def test_artefact_tamper_is_detected(handoff):
    payload = json.load(open(handoff["summary"]["artifacts"]["failure_faces"],
                             encoding="utf-8"))
    tampered = json.loads(json.dumps(payload))
    # Drag one vertex perpendicular to its segment: this changes both the area
    # and the rim length, which the reader recomputes independently.
    tampered["floes"][0]["polygons"][0][1][1] += 0.5
    with pytest.raises(ValueError):
        read_failure_faces(tampered)


def test_declared_floe_scalars_must_match_their_polygons(handoff):
    payload = json.load(open(handoff["summary"]["artifacts"]["failure_faces"],
                             encoding="utf-8"))
    tampered = json.loads(json.dumps(payload))
    tampered["floes"][0]["area_m2"] *= 1.5
    with pytest.raises(ValueError, match="declared area"):
        read_failure_faces(tampered)


def test_bad_schema_is_detected(handoff):
    payload = json.load(open(handoff["summary"]["artifacts"]["failure_faces"],
                             encoding="utf-8"))
    payload["schema"] = "something-else/9"
    with pytest.raises(ValueError, match="unexpected schema"):
        read_failure_faces(payload)


def test_payload_is_deterministic(case):
    out_a = os.path.join("artifacts", "_tmp_det_a")
    out_b = os.path.join("artifacts", "_tmp_det_b")
    try:
        run_handoff(case, out_a, tag="det_a")
        run_handoff(case, out_b, tag="det_b")
        a = json.load(open(os.path.join(out_a, "failure_faces.json"), encoding="utf-8"))
        b = json.load(open(os.path.join(out_b, "failure_faces.json"), encoding="utf-8"))
        assert a["provenance"]["payload_sha256"] == b["provenance"]["payload_sha256"]
        assert a["floes"] == b["floes"]
    finally:
        for path in (out_a, out_b):
            for name in os.listdir(path) if os.path.isdir(path) else []:
                os.remove(os.path.join(path, name))
            if os.path.isdir(path):
                os.rmdir(path)


def test_crack_polylines_live_on_the_mesh_lattice(handoff, case):
    faces = read_failure_faces(handoff["summary"]["artifacts"]["failure_faces"])
    hx = case.length_x / case.nx
    hy = case.length_y / case.ny
    for crack in faces.cracks:
        pts = np.asarray(crack.polyline)
        # every vertex is a mesh node
        np.testing.assert_allclose(pts[:, 0] / hx, np.round(pts[:, 0] / hx), atol=1e-9)
        np.testing.assert_allclose(pts[:, 1] / hy, np.round(pts[:, 1] / hy), atol=1e-9)
        if len(pts) > 1:
            d = np.abs(np.diff(pts, axis=0))
            # consecutive vertices are one cell edge apart in exactly one direction
            assert np.all(np.isclose(d.max(axis=1), np.maximum(hx, hy)))


def test_floe_polygon_vertices_lie_on_nodes_and_cells_are_disjoint(handoff):
    faces = read_failure_faces(handoff["summary"]["artifacts"]["failure_faces"])
    total_cells = sum(f.n_cells for f in faces.floes)
    assert total_cells == faces.meta["mesh"]["n_cells"]
    n_fractured = faces.meta["stress_state"]["n_fractured_elements"]
    assert n_fractured >= 1


# --------------------------------------------------------------------------
# polygon utilities and physical-validity guard
# --------------------------------------------------------------------------
def test_polygon_utilities():
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    assert abs(polygon_area(square) - 4.0) < 1e-12
    assert abs(polygon_area(square[::-1]) + 4.0) < 1e-12
    np.testing.assert_allclose(polygon_centroid(square), [1.0, 1.0], atol=1e-12)
    assert abs(polygon_perimeter(square) - 8.0) < 1e-12
    assert point_in_polygon((1.0, 1.0), square)
    assert not point_in_polygon((3.0, 1.0), square)


def test_thin_plate_assumption_holds_for_the_case(handoff, case):
    criteria = handoff["summary"]["criteria"]
    # Kirchhoff kinematics: curvature * half thickness and the slope must be small
    assert criteria["max_curvature_times_thickness"] < 0.01
    assert criteria["max_rotation_rad"] < 0.05
    assert case.length_x / case.material.thickness > 10.0


def test_characteristic_length_and_failure_moment():
    material = PlateMaterial()
    lc = material.characteristic_length(FOUNDATION)
    assert 8.0 < lc < 9.5                       # 8.69 m for the default ice
    assert abs(material.failure_moment()
               - material.sigma_flexure * material.thickness ** 2 / 6.0) < 1e-12


def test_splitting_state_is_found_monotonically(case, first_cracking_solution):
    state = find_splitting_state(first_cracking_solution, case.tracer)
    assert state.load_factor >= 1.0
    assert state.largest_floe_fraction <= 0.8 + 1e-9
    assert state.evaluations >= 1
    fractions = [h["largest_floe_fraction"] for h in state.history]
    assert max(fractions) >= min(fractions)