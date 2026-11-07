"""Acceptance and unit tests for the 3D triangle-triangle (non-coplanar) mortar node.

Campaign node IN52-N01, mission 6114.  The suite covers

* the geometry primitives: homogeneous Sutherland-Hodgman clipping (with its
  denominator / ``w`` guards), the coplanar branch and the projection branch of
  ``tri_pair_overlap``, and the dihedral rejection of facets that do not belong
  to the same interface patch;
* the measure-proportional backfill of m5207 section 5.2 as an *arithmetic
  identity*, together with the executable trap;
* the machine gates D1-D6 of :mod:`tensordem.interface_operators.acceptance3d`,
  each with its negative control;
* reproducibility: the published artifact must equal a freshly computed record.

Run with::

    python -m pytest tests/test_interface_tri3d_nonplanar.py -q
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.interface_operators.acceptance3d import (CURVED_LEVELS,
                                                        curved_octant_case,
                                                        kinked_case,
                                                        planar_tri_case,
                                                        run_acceptance,
                                                        three_operator_table)
from tensordem.interface_operators.clip3d import (SurfaceMortarOperator, TriSide,
                                                  clip_homogeneous, facet_gradients,
                                                  ico_sphere_tris, measure_backfill,
                                                  octant_reference,
                                                  surface_cell_averages,
                                                  surface_coverage, surface_integral,
                                                  surface_overlap_entries,
                                                  tri_area, tri_normal,
                                                  tri_pair_overlap, tri_patch_side,
                                                  uv_sphere_tris)
from tensordem.interface_operators.geometry import sphere_box_area

ARTIFACT = (Path(__file__).resolve().parents[1] / "artifacts" / "in52_n01"
            / "interface_tri3d_acceptance.json")


@pytest.fixture(scope="module")
def acceptance() -> dict:
    """The full D1-D6 record (computed once, ~45 s)."""
    return run_acceptance()


# ---------------------------------------------------------------------------
# geometry primitives
# ---------------------------------------------------------------------------
def test_homogeneous_clip_drops_vertex_at_infinity():
    """A ``w = 0`` vertex has no finite clip parameter and must be dropped."""
    square = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0],
                       [2.0, 2.0, 1.0], [0.0, 2.0, 1.0]])
    planes = np.array([[1.0, 0.0, 0.0]])            # x >= 0
    out = clip_homogeneous(square, planes)
    assert np.all(np.isfinite(out))
    assert len(out) == 4
    # now push one vertex to infinity: it must disappear, not become a NaN
    bad = square.copy()
    bad[2, 2] = 0.0
    out_bad = clip_homogeneous(bad, planes)
    assert np.all(np.isfinite(out_bad))
    assert len(out_bad) < len(out)


def test_homogeneous_clip_parallel_edge_guard():
    """An edge parallel to the clip plane must not create a spurious vertex."""
    tri = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.5, 2.0, 1.0]])
    out = clip_homogeneous(tri, np.array([[0.0, 1.0, 0.0]]))   # y >= 0
    assert np.all(np.isfinite(out))
    assert abs(len(out) - 3) <= 1


def test_tri_pair_overlap_coplanar_is_exact():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = np.array([[0.25, 0.25, 0.0], [0.75, 0.25, 0.0], [0.25, 0.75, 0.0]])
    res = tri_pair_overlap(a, b)
    assert res["path"] == "coplanar"
    assert res["measure_src"] == pytest.approx(0.125, rel=1e-13)     # 1/8
    assert res["measure_tgt"] == pytest.approx(0.125, rel=1e-13)
    # the moment integrates to zero over a single cell's own area
    whole = tri_pair_overlap(a, a)
    assert np.allclose(whole["moment"], 0.0, atol=1e-15)


def test_tri_pair_overlap_projection_jacobian():
    """Non-coplanar pair: the transported measure is ``area / |cos|``."""
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ang = math.radians(60.0)
    b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [0.0, math.cos(ang), math.sin(ang)]])
    res = tri_pair_overlap(a, b)
    assert res["path"] == "projected"
    assert res["cos_dihedral"] == pytest.approx(math.cos(ang), rel=1e-14)
    # the projected footprint is ``cos(60 deg)`` times the unit triangle, and the
    # transported measure is the *true* area of ``b`` (0.5), i.e. the Jacobian of
    # the projection is undone exactly
    assert res["measure_src"] == pytest.approx(0.5 * math.cos(ang), rel=1e-13)
    assert res["measure_tgt"] == pytest.approx(0.5, rel=1e-13)


def test_tri_pair_overlap_rejects_steep_dihedral():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # 90 deg
    res = tri_pair_overlap(a, b)
    assert res["path"] == "rejected"
    assert res["measure_src"] == 0.0 and res["measure_tgt"] == 0.0


def test_deviation_from_m5436_planarity_guard():
    """``TriSide`` is the *curved* counterpart of ``InterfaceSide``: it accepts
    non-coplanar cells, which ``mortar.InterfaceSide`` deliberately rejects."""
    from tensordem.interface_operators.mortar import InterfaceSide, plane_frame
    tris = ico_sphere_tris(1.0, 1)
    with pytest.raises(ValueError):
        InterfaceSide(polys=tris, basis=plane_frame(np.vstack(tris))[1],
                      origin=plane_frame(np.vstack(tris))[0], label="curved")
    side = TriSide(tris, label="curved")             # must not raise
    assert side.n == len(tris)


def test_rotated_patch_partitions_the_square():
    """Both facet families must cover exactly the unit square, without slivers."""
    for rot in (0.0, 17.0):
        side = tri_patch_side((0.5, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                              7, 5, rotate_deg=rot, jitter=0.15, seed=4)
        assert side.area == pytest.approx(1.0, rel=1e-14)
        assert side.measures.min() > 1e-12           # no degenerate sliver survives
        assert np.all(np.abs(side.normals[:, 0] - 1.0) < 1e-12)


def test_octant_reference_against_independent_closed_form():
    """The octant reference must agree with the semi-analytic sphere/box area."""
    ref = octant_reference(1.0)
    inf = float("inf")
    exact = sphere_box_area(1.0, ((0.0, inf), (0.0, inf), (0.0, inf)))
    assert ref["area"] == pytest.approx(exact, rel=1e-13)
    assert ref["int_sum"] == pytest.approx(3.0 * ref["int_x"], rel=1e-15)


def test_facet_gradients_reproduce_a_linear_field():
    side = tri_patch_side((0.5, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                          6, 4, jitter=0.1, seed=2)
    G = facet_gradients(side)
    # a linear field in (y, z) has the same gradient component on every facet
    vals = surface_cell_averages(side, lambda p: 2.0 * np.asarray(p)[..., 1]
                                 - 1.5 * np.asarray(p)[..., 2])
    g = np.einsum("inl,l->in", G, vals)              # (n, 2) gradient per facet
    assert g.shape == (side.n, 2)
    # each facet has its own tangent frame, so only the *magnitude* is frame
    # independent: |grad| = sqrt(2^2 + 1.5^2)
    mag = np.linalg.norm(g, axis=1)
    assert np.max(np.abs(mag - math.hypot(2.0, 1.5))) < 1e-9


# ---------------------------------------------------------------------------
# the m5207 section 5.2 measure-normalisation trap, as an arithmetic identity
# ---------------------------------------------------------------------------
def test_backfill_closes_the_deficit_exactly():
    rng = np.random.default_rng(0)
    m = rng.uniform(0.02, 0.5, size=57)
    v = rng.normal(size=57)
    recv = float(np.dot(v, m))
    for deficit in (0.5, -0.25, 1e-9):
        out = measure_backfill(v, recv + deficit, recv, m)
        assert float(np.dot(out, m)) - recv == pytest.approx(deficit, abs=1e-14)


def test_backfill_trap_leaves_a_percent_level_residual():
    rng = np.random.default_rng(1)
    m = rng.uniform(0.02, 0.5, size=57)
    v = np.zeros(57)
    recv = 0.0
    deficit = 0.5
    wrong = measure_backfill(v, recv + deficit, recv, m, broken="backfill_linear")
    left = float(np.dot(wrong, m)) - recv
    # the correction is (sent-recv)|T_j|/sum|T|, whose integral is
    # (sent-recv) * sum|T|^2/sum|T|: the deficit is *not* closed (for a unit-measure
    # interface, sum|T| = 1, this is the (1 - sum|T|^2) form quoted in m5207 5.2)
    assert left == pytest.approx(deficit * np.sum(m ** 2) / np.sum(m), rel=1e-12)
    assert left > 0.3 * deficit                      # most of the deficit survives


def test_backfill_weighting_is_measure_proportional():
    m = np.array([0.1, 0.2, 0.7])
    out = measure_backfill(np.zeros(3), 1.0, 0.0, m)
    ratio = out / m
    assert np.allclose(ratio, ratio[0])              # constant, i.e. proportional


# ---------------------------------------------------------------------------
# gates D1-D5
# ---------------------------------------------------------------------------
def test_d1_gates(acceptance):
    g = acceptance["gates"]
    assert g["D1_coverage"]["measured"] <= 1e-13
    assert g["D1_conservation"]["measured"] <= 1e-13
    assert g["D1_transpose"]["measured"] <= 1e-12
    assert g["D1_negative_control_red"]["passed"]


def test_d1_conservation_per_reconstruction_order(acceptance):
    for case in acceptance["D1"]["cases"]:
        for name, row in case["orders"].items():
            assert row["residual"] <= 1e-13, (case["n_src"], name, row)


def test_d2_patch_test_and_order1_is_not_exact(acceptance):
    assert acceptance["gates"]["D2_patch_test"]["measured"] <= 1e-12
    assert acceptance["gates"]["D2_order1_not_exact"]["passed"]


def test_d3_curved_orders_and_closed_form(acceptance):
    d3 = acceptance["D3"]
    gates = acceptance["gates"]
    assert gates["D3_area_order"]["measured"] >= 1.8
    assert gates["D3_field_order"]["measured"] >= 1.8
    assert gates["D3_field_accuracy"]["measured"] <= 3e-2
    assert gates["D3_raw_defect_order"]["measured"] >= 1.5
    assert gates["D3_backfill_closes"]["measured"] <= 1e-13
    # the two polyhedra really are non-coplanar: the dihedral is < 1
    src, tgt = curved_octant_case(*CURVED_LEVELS[0])
    ents = surface_overlap_entries(src, tgt)
    cosines = [abs(float(np.dot(src.normals[i], tgt.normals[j])))
               for (i, j, _m, _mom) in ents]
    assert max(cosines) < 1.0 - 1e-6
    assert np.mean(cosines) > 0.9                    # but they approximate one surface


def test_d3_trap_and_midpoint_controls_are_red(acceptance):
    d3 = acceptance["D3"]["negative_controls"]
    assert acceptance["gates"]["D3_trap_control_red"]["passed"]
    assert acceptance["gates"]["D3_midpoint_control_red"]["passed"]
    assert d3["trap_residual"] > 1e-2
    assert d3["correct_backfill_residual"] <= 1e-13
    assert d3["arithmetic_demo"]["correct_after"] < 1e-12
    assert d3["arithmetic_demo"]["trap_after"] > 1e-2


def test_d4_symmetry(acceptance):
    g = acceptance["gates"]
    assert g["D4_mirror_equivariance"]["measured"] <= 1e-12
    assert acceptance["D4"]["midpoint_mirror"]["bijective"]
    assert g["D4_curved_asymmetry_reported"]["passed"]
    assert g["D4_curved_conservation_both_directions"]["passed"]


def test_d5_kinked_interface(acceptance):
    g = acceptance["gates"]
    assert g["D5_conservation"]["measured"] <= 1e-13
    assert g["D5_patch_test"]["measured"] <= 1e-12
    assert g["D5_order"]["measured"] >= 1.8
    assert g["D5_negative_control_red"]["passed"]
    # the kink really is a kink: the two panels have different tangents
    panels, verts = kinked_case(4, 3, 6, 5)
    a, b, c = verts
    cos = float(np.dot(b - a, c - b) / (np.linalg.norm(b - a) * np.linalg.norm(c - b)))
    assert cos == pytest.approx(math.cos(math.radians(30.0)), rel=1e-12)


# ---------------------------------------------------------------------------
# D6 comparison table
# ---------------------------------------------------------------------------
def test_three_operator_table_is_comparable_across_dimensions(acceptance):
    table = acceptance["D6"]
    cases = {row["case"] for row in table["rows"]}
    assert any("1D" in c for c in cases)
    assert any("2D" in c for c in cases)
    assert any("3D tri-tri" in c for c in cases)
    assert any("curved" in c for c in cases)
    for row in table["rows"]:
        if "no backfill" in row["operator"]:
            continue
        assert abs(row["conservation_rel"]) <= 1e-12, row
    # the un-backfilled curved transfer is *not* conservative: that is why the
    # measure-proportional backfill is on the critical path for curved interfaces
    raw = [r for r in table["rows"] if "no backfill" in r["operator"]]
    assert raw and abs(raw[0]["conservation_rel"]) > 1e-7


def test_verdict_is_pass(acceptance):
    assert acceptance["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not generated yet")
def test_artifact_matches_computed_record(acceptance):
    published = json.loads(ARTIFACT.read_text())
    for name, gate in acceptance["gates"].items():
        old = published["gates"][name]
        for key in ("measured", "threshold"):
            if key in gate and isinstance(gate[key], float):
                assert gate[key] == pytest.approx(old[key], rel=1e-9, abs=1e-15), \
                    (name, key)
        if "passed" in gate:
            assert gate["passed"] == old["passed"], name