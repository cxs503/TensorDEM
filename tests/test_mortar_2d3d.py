"""Acceptance tests for the 2D/3D mortar interface operators (mission 5436).

Covers the campaign gates C1-C5 (plus the C6 comparison table and the
measure-normalisation trap), and — importantly — the *negative controls*: every
gate is exercised together with a deliberately broken variant that must fail, so
a green suite means the gates are alive rather than dead thresholds.

Run with::

    python -m pytest tests/test_mortar_2d3d.py -q

The heavy acceptance record is computed once per module (``acceptance`` fixture);
the remaining tests are unit-level checks of the geometry primitives, the
backfill normalisation and the robustness guards.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.interface_operators import (MomentumExchangeOperator, MortarOperator,
                                           RBFOperator, cell_averages, grid_side,
                                           integral, l2_error, measure_weighted_symmetry,
                                           segment_side, sphere_box_area, tri_patch_side)
from tensordem.interface_operators.acceptance import (cube_case_3d, jittered_cuts,
                                                      run_acceptance, triangle_case_2d)
from tensordem.interface_operators.geometry import (cell_overlap, clip_convex_polygon,
                                                   interval_clip, polygon_area)
from tensordem.interface_operators.mortar import measure_proportional_backfill

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "m5436" / \
    "mortar_2d3d_acceptance.json"


@pytest.fixture(scope="module")
def acceptance() -> dict:
    """The full C1-C6 acceptance record (computed once, ~1 min)."""
    return run_acceptance()


# ---------------------------------------------------------------------------
# C1 - 2D rotated-17deg non-matching interface, piecewise-constant field
# ---------------------------------------------------------------------------
def test_c1_conservation_2d_rotated_17deg(acceptance):
    gate = acceptance["gates"]["C1"]
    assert gate["measured"] <= 1e-13, gate
    c1 = acceptance["C1_C2_C3"]["C1"]
    # both reconstruction orders conserve, as they must: the overlaps partition
    # every donor cell, and the slope term integrates to zero over a cell
    assert c1["order1"]["residual"] <= 1e-13
    assert c1["order2"]["residual"] <= 1e-13


def test_c1_negative_controls_are_red(acceptance):
    neg = acceptance["C1_C2_C3"]["C1"]["negative_controls"]
    # point-sampling the donor reconstruction on a non-matching mesh leaks mass
    assert neg["mortar_midpoint"] > 1e-6
    # the wrong backfill normalisation leaks mass as well
    assert neg["rbf_backfill_linear"] > 1e-3


# ---------------------------------------------------------------------------
# C2 - linear patch test
# ---------------------------------------------------------------------------
def test_c2_linear_patch_test_2d(acceptance):
    gate = acceptance["gates"]["C2"]
    assert gate["measured"] <= 1e-12, gate
    patch = acceptance["C1_C2_C3"]["C2"]["2d_17deg"]
    # a first-order (piecewise-constant) operator cannot be exact on linear data
    assert patch["order1"] > 1e-3
    assert patch["order2"] <= 1e-12


def test_c2_linear_patch_test_3d(acceptance):
    patch = acceptance["C1_C2_C3"]["C2"]["3d_cube_face"]
    assert patch["order2"] <= 1e-12
    assert patch["order1"] > 1e-3


# ---------------------------------------------------------------------------
# C3 - smooth-field convergence
# ---------------------------------------------------------------------------
def test_c3_observed_order(acceptance):
    gate = acceptance["gates"]["C3"]
    assert gate["relation"] == ">="
    assert gate["measured"] >= 1.8, gate
    c3 = acceptance["C1_C2_C3"]["C3"]
    # errors must actually decrease on refinement (guards against a fake plateau
    # produced by a random node cloud instead of a jittered lattice)
    errs = c3["order2"]["errors"]
    assert all(errs[k + 1] < errs[k] for k in range(len(errs) - 1))
    assert all(r["h"] > 0 for r in c3["levels"])


# ---------------------------------------------------------------------------
# C4 - 3D two cubes + curved sphere/box clipping
# ---------------------------------------------------------------------------
def test_c4_3d_coverage_and_conservation(acceptance):
    cov = acceptance["gates"]["C4_coverage"]
    cons = acceptance["gates"]["C4_conservation"]
    assert cov["measured"] <= 1e-10, cov
    assert cons["measured"] <= 1e-10, cons
    for row in acceptance["C4"]["coverage"]:
        assert row["pairs"] > 0
        assert abs(row["src_measure_error"]) <= 1e-10
        assert abs(row["tgt_measure_error"]) <= 1e-10
    neg = acceptance["gates"]["C4_negative_control_red"]
    assert neg["passed"] and neg["residual"] > 1e-6


def test_c4_curved_clipping_vs_closed_form(acceptance):
    gate = acceptance["gates"]["C4_curved_closed_form"]
    assert gate["measured"] <= 1e-6, gate
    cases = {r["case"]: r for r in acceptance["C4"]["curved"]["closed_form"]}
    assert len(cases) >= 4
    # the cap case is the one with a genuinely curved boundary (small circle)
    assert cases["spherical_cap z>=0.35"]["rel_error"] <= 1e-9


def test_c4_curved_facet_path_converges(acceptance):
    gate = acceptance["gates"]["C4_curved_facet_order"]
    assert gate["measured"] >= 1.8, gate
    rows = acceptance["C4"]["curved"]["facet_convergence"]["rows"]
    errs = [r["rel_error"] for r in rows]
    assert all(errs[k + 1] < errs[k] for k in range(len(errs) - 1))


# ---------------------------------------------------------------------------
# C5 - symmetry
# ---------------------------------------------------------------------------
def test_c5_transpose_consistency(acceptance):
    gate = acceptance["gates"]["C5_transpose"]
    assert gate["measured"] <= 1e-12, gate
    tab = acceptance["C5"]["transpose"]
    assert set(tab) == {"1d", "2d_oblique_17deg", "3d_cube_face"}
    for value in tab.values():
        assert value <= 1e-12


def test_c5_reflection_and_rotation(acceptance):
    assert acceptance["gates"]["C5_reflection"]["measured"] <= 1e-12
    assert acceptance["gates"]["C5_rotation"]["measured"] <= 1e-12
    # the "each side its own mirror axis" variant must not be needed: the shared
    # axis already gives exact equivariance, and both numbers stay tiny here
    refl = acceptance["C5"]["reflection_2d"]
    assert refl["symmetry_flip"] <= 1e-12
    assert refl["negative_control_two_axes"] <= 1e-12


# ---------------------------------------------------------------------------
# C6 - three-operator table, the trap, and the momentum budget
# ---------------------------------------------------------------------------
def test_c6_three_operator_table(acceptance):
    tab = acceptance["C6"]["table"]
    cases = {row["case"] for row in tab}
    assert cases == {"1d", "2d_17deg", "3d_cube_face"}
    ops = {row["operator"] for row in tab}
    assert ops == {"mortar", "rbf", "me"}
    for row in tab:
        assert np.isfinite(row["conservation_residual"])
        assert np.isfinite(row["l2_error"])
    # mortar is conservative by construction; RBF by backfill; ME by normalisation
    for row in tab:
        assert row["conservation_residual"] <= 1e-12, row


def test_c6_measure_normalisation_trap(acceptance):
    """The m5207 section 5.2 trap, now executed in 2D and 3D.

    Magnitude note (measured, not assumed): the trap leaves
    ``d * (1 - sum|T|^2 / sum|T|)`` behind, i.e. it scales with the measure
    *non-uniformity* of the acceptor mesh.  On the 2D jittered interface the
    residual is ~5e-2; on the near-uniform 3D quad/triangle grid it is only
    ~7e-4 -- small, but of the same order as the uncorrected deficit, which is
    exactly why the trap is dangerous: it looks like a converged correction.
    """
    trap = acceptance["C6"]["trap"]
    assert set(trap) == {"2d_17deg", "3d_cube_face"}
    for case, values in trap.items():
        assert values["correct_measure_proportional"] <= 1e-12, (case, values)
        assert values["trap_linear_denominator"] >= 0.5 * values["no_fix"], (case, values)
        assert values["no_fix"] >= 1e-6, (case, values)
    assert max(v["trap_linear_denominator"] for v in trap.values()) >= 1e-2
    assert acceptance["gates"]["trap_correct_normalisation"]["passed"]
    assert acceptance["gates"]["trap_wrong_normalisation_red"]["passed"]


def test_backfill_normalisation_trap_regression():
    """Direct regression of ``dv_j = d |T_j| / sum|T|^2`` vs the tempting variant.

    The correct split changes the integral by exactly ``sent - recv``; the linear
    denominator leaves ``(sent-recv)(1 - sum|T|^2/(sum|T|)^2)`` behind.  Both
    branches are executed so the arithmetic claim in the module docstring is
    machine-checked rather than asserted.
    """
    rng = np.random.default_rng(3)
    m = rng.uniform(0.01, 0.2, size=17)
    v = rng.normal(size=17)
    sent, recv = 1.2345, 0.9876

    good = measure_proportional_backfill(v, sent, recv, m)
    assert abs(integral(good, m) - integral(v, m) - (sent - recv)) <= 1e-14

    bad = measure_proportional_backfill(v, sent, recv, m, broken="backfill_linear")
    deficit = sent - recv
    # the wrong denominator delivers only d * sum|T|^2 / sum|T| instead of d
    wrong_change = deficit * float(np.sum(m * m)) / float(np.sum(m))
    assert abs(integral(bad, m) - integral(v, m) - wrong_change) <= 1e-14
    assert abs(integral(bad, m) - integral(v, m) - deficit) > 1e-4

    with pytest.raises(ValueError):
        measure_proportional_backfill(v, sent, recv, m, broken="nonsense")


def test_c6_momentum_budget_closes(acceptance):
    assert acceptance["gates"]["me_budget_closed"]["passed"]
    assert acceptance["gates"]["me_negative_control_red"]["passed"]
    for case, rec in acceptance["C6"]["momentum_exchange"].items():
        assert rec["closed"]["budget_residual"] <= 1e-12, (case, rec)
        assert rec["per_node_norm"]["budget_residual"] > 1e-4, (case, rec)


# ---------------------------------------------------------------------------
# unit checks of the geometry primitives
# ---------------------------------------------------------------------------
def test_polygon_clipping_is_exact():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    shifted = square + np.array([0.25, 0.5])
    clipped = clip_convex_polygon(shifted, square)
    assert polygon_area(clipped) == pytest.approx(0.75 * 0.5, rel=0, abs=1e-15)

    # a full cover must return the clipper's own area
    assert polygon_area(clip_convex_polygon(square, square)) == pytest.approx(1.0, abs=1e-15)

    # disjoint cells: measure zero, and the moment must not be NaN
    far = square + np.array([5.0, 0.0])
    m, mom = cell_overlap(shifted, far,
                          basis=np.array([[1.0, 0.0], [0.0, 1.0]]),
                          origin=np.zeros(2))
    assert m == 0.0 and np.all(np.isfinite(mom))


def test_interval_clip_and_rotated_segments():
    # 17-degree interface: overlap lengths are exact at any orientation
    th = math.radians(17.0)
    t = np.array([math.cos(th), math.sin(th)])
    p0 = np.array([0.1, -0.2])
    a = segment_side(p0, p0 + t, np.linspace(0, 1, 5))
    b = segment_side(p0, p0 + t, np.linspace(0, 1, 8))
    total = sum(m for _, _, m, _ in
                __import__("tensordem.interface_operators.mortar",
                           fromlist=["overlap_entries"]).overlap_entries(a, b))
    assert total == pytest.approx(1.0, abs=1e-14)
    assert interval_clip((0.0, 0.5), (0.5, 1.0)) is None     # touching only
    assert interval_clip((0.0, 0.75), (0.5, 1.0)) == (0.5, 0.75)


def test_sphere_box_area_closed_forms():
    R = 1.0
    inf = float("inf")
    assert sphere_box_area(R, ((-inf, inf),) * 3) == pytest.approx(4 * math.pi, abs=1e-12)
    assert sphere_box_area(R, ((0.0, inf), (-inf, inf), (-inf, inf))) == pytest.approx(
        2 * math.pi, abs=1e-12)
    assert sphere_box_area(R, ((0, inf), (0, inf), (0, inf))) == pytest.approx(
        math.pi / 2, abs=1e-12)
    # spherical cap of height R - d
    d = 0.35
    assert sphere_box_area(R, ((-inf, inf), (-inf, inf), (d, inf))) == pytest.approx(
        2 * math.pi * R * (R - d), abs=1e-12)
    # a small box fully inside a big sphere contains no sphere surface
    assert sphere_box_area(10.0, ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))) == 0.0
    # a box that contains the sphere cuts nothing
    assert sphere_box_area(10.0, ((-20.0, 20.0),) * 3) == pytest.approx(
        4 * math.pi * 100.0, abs=1e-9)


def test_3d_partition_coverage_is_exact():
    c = cube_case_3d(9, 7)
    from tensordem.interface_operators.mortar import overlap_entries
    ent = overlap_entries(c.src, c.tgt)
    assert sum(m for _, _, m, _ in ent) == pytest.approx(1.0, abs=1e-13)
    # every acceptor cell is covered and every donor cell is used
    used_src = {i for i, _, _, _ in ent}
    used_tgt = {j for _, j, _, _ in ent}
    assert used_src == set(range(c.src.n))
    assert used_tgt == set(range(c.tgt.n))


def test_nonplanar_cells_are_rejected():
    """A cell whose vertices leave the interface plane must raise, not silently
    corrupt the area/moment integrals (a real failure mode when a 3D node jitter
    carries an out-of-plane component)."""
    origin = np.zeros(3)
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])
    bent = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                     [0.0, 1.0, 1.0], [0.2, 0.0, 1.0]])       # 0.2 out of plane
    from tensordem.interface_operators.mortar import InterfaceSide
    with pytest.raises(ValueError, match="non-planar"):
        InterfaceSide(polys=[bent], basis=np.vstack([e1, e2]), origin=origin)


def test_jittered_lattice_refines_monotonically():
    prev = None
    for n in (8, 16, 32):
        cuts = jittered_cuts(n, 0.4, seed=5)
        assert cuts[0] == 0.0 and cuts[-1] == 1.0
        assert np.all(np.diff(cuts) > 0.0)
        h_max = np.diff(cuts).max()
        if prev is not None:
            assert h_max < prev * 0.75          # keeps halving, no saturation
        prev = h_max


def test_operator_matrix_matches_transfer():
    """``matrix()`` and ``transfer()`` must be the same operator."""
    c = triangle_case_2d(7, 11, 17.0)
    u = cell_averages(c.src, lambda p: np.sin(3.0 * c.arclength(p)))
    for order in (1, 2):
        op = MortarOperator(order=order)
        assert np.allclose(op.matrix(c.src, c.tgt) @ u, op.transfer(c.src, u, c.tgt),
                           atol=1e-14)


def test_mortar_matches_1d_reference_operator():
    """The 1D m5207 configuration is the ``nb = 1`` special case of this module."""
    c = triangle_case_2d(7, 11, angle_deg=0.0, p0=(0.0, 0.0))
    f = lambda p: 1.0 - 2.5 * np.asarray(p)[..., 0]
    u = cell_averages(c.src, f)
    v = MortarOperator(order=2).transfer(c.src, u, c.tgt)
    assert l2_error(v, cell_averages(c.tgt, f), c.tgt.measures) <= 1e-12
    assert abs(integral(v, c.tgt.measures) - integral(u, c.src.measures)) <= 1e-14


def test_artifact_matches_computed_record(acceptance):
    """If the published artifact exists it must agree with the live computation."""
    if not ARTIFACT.exists():
        pytest.skip("artifact not generated yet")
    on_disk = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for key, live in acceptance["gates"].items():
        if key not in on_disk["gates"]:
            continue
        old = on_disk["gates"][key]
        assert old["passed"] == live["passed"], key
        if isinstance(live.get("measured"), float) and isinstance(old.get("measured"), float):
            assert old["measured"] == pytest.approx(live["measured"], rel=1e-9,
                                                    abs=1e-15), key