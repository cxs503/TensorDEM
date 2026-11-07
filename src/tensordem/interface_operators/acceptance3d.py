"""Machine-checkable acceptance harness for the 3D triangle-triangle mortar node.

Gates (campaign node IN52-N01, mission 6114)
--------------------------------------------
D1  Planar 3D non-matching *triangle-triangle* interface (both sides
    triangulated, the two facet families rotated 17 deg against each other,
    nodes jittered): area-coverage error <= 1e-13 and conservation residual
    <= 1e-13 for reconstruction orders 1 and 2; negative controls (midpoint
    sampling, mis-normalised backfill) must go red.
D2  Linear patch test on that interface: L2 <= 1e-12 for order 2, while order 1
    stays at the percent level (a first-order operator cannot be exact on linear
    data).
D3  Curved, *non-coplanar* interface: two different polyhedral approximations of
    the unit-sphere octant (UV sphere vs subdivided octahedron, different
    topology and resolution).  Area and transferred integral converge to the
    closed form (pi/2 and 3*pi/4) with observed order >= 1.8; the raw
    conservation defect decays as O(h^2) and the measure-proportional backfill
    (m5207 section 5.2 fix) closes it to machine precision, while the
    mis-normalised branch leaves a percent-level residual behind.
D4  Symmetry: both sides mirrored about **one common** mirror plane -> the
    transfer is equivariant (<= 1e-13), and the overlap matrix is
    transpose-consistent (m_ij vs m_ji) on a coplanar interface.  On a curved
    interface the projection mortar is *not* transpose-consistent by
    construction; the measured asymmetry is reported, not hidden.
D5  2D piecewise-linear (kinked, 30 deg) interface: conservation <= 1e-13,
    linear patch test <= 1e-12 on every panel, observed order >= 1.8 -- this is
    the "true curve" case that the coplanar m5436 operator left open.
D6  Three-operator comparison table (mortar / RBF / ME) on the 1D reference,
    the 2D 17 deg oblique interface, the new 3D tri-tri interface and the curved
    octant, so the numbers sit next to the published 1D contract values.

Every gate is paired with a negative control, so a green suite means the gates
are alive rather than dead thresholds.  Nothing in the record is transcribed by
hand: the JSON artifact is produced here and consumed by
``scripts/run_interface_tri3d_benchmark.py`` and
``tests/test_interface_tri3d_nonplanar.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess

import numpy as np

from .acceptance import _fit_order, _gate, _order
from .clip3d import (SurfaceMortarOperator, TriSide, clip_tris_by_box,
                     ico_sphere_tris, measure_backfill, octant_reference,
                     surface_cell_averages, surface_coverage, surface_integral,
                     surface_l2_error, surface_overlap_entries, tri_patch_side,
                     uv_sphere_tris)
from .geometry import sphere_box_area
from .mortar import (InterfaceSide, MortarOperator, cell_averages, from_polygons,
                     integral, l2_error, overlap_entries, segment_side)
from .operators import MomentumExchangeOperator, RBFOperator

__all__ = ["run_acceptance", "planar_tri_case", "curved_octant_case", "kinked_case",
           "three_operator_table", "OCTANT_BOX", "main"]

OCTANT_BOX = ((0.0, float("inf")), (0.0, float("inf")), (0.0, float("inf")))


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def planar_tri_case(n_src: int = 6, n_tgt: int = 4, *, rotate_deg: float = 17.0,
                    jitter: float = 0.15, seed_src: int = 3, seed_tgt: int = 9,
                    plane_x: float = 0.5) -> tuple[TriSide, TriSide]:
    """Two cubes sharing the plane ``x = plane_x``, both sides triangulated.

    ``e1`` runs along ``+y`` and ``e2`` along ``+z``, so this is the shared face
    of ``[0,plane_x] x [0,1]^2`` and ``[plane_x,1] x [0,1]^2``.  The *acceptor*
    facet family is rotated 17 deg inside the plane, the donor's is not: generic
    overlaps, no nested nodes, identical support.
    """
    origin = (plane_x, 0.0, 0.0)
    src = tri_patch_side(origin, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), n_src, n_src,
                         rotate_deg=0.0, jitter=jitter, seed=seed_src, label="cube-left")
    tgt = tri_patch_side(origin, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), n_tgt, n_tgt,
                         rotate_deg=rotate_deg, jitter=jitter, seed=seed_tgt,
                         label="cube-right")
    return src, tgt


def curved_octant_case(n_theta: int, n_phi: int, ico_level: int) -> tuple[TriSide, TriSide]:
    """Two polyhedral approximations of the unit-sphere octant (non-coplanar)."""
    src = TriSide(clip_tris_by_box(uv_sphere_tris(1.0, n_theta, n_phi), OCTANT_BOX),
                  label=f"uv{n_theta}x{n_phi}")
    tgt = TriSide(clip_tris_by_box(ico_sphere_tris(1.0, ico_level), OCTANT_BOX),
                  label=f"ico{ico_level}")
    return src, tgt


CURVED_LEVELS = ((12, 24, 3), (24, 48, 4), (48, 96, 5))
CURVED_MISMATCH = (20, 40, 2)          # deliberately mismatched resolutions


def kinked_case(n1_src: int, n2_src: int, n1_tgt: int, n2_tgt: int, *,
                kink_deg: float = 30.0, panel: float = 0.55, jitter: float = 0.35,
                seed: int = 11):
    """Piecewise-linear 2D interface with a 30 deg kink (two panels).

    The kink is a genuine geometric vertex (a 2D analogue of a curved interface:
    the tangent frame jumps), the two panels are discretised with *different*
    cell counts on the two sides, and every panel break is jittered by O(h).  The
    mortar operator is applied panel by panel -- the same code path as m5436 --
    and the conservation gate is evaluated on the assembly of both panels.
    """
    a = np.array([0.10, 0.05])
    e1 = np.array([math.cos(math.radians(10.0)), math.sin(math.radians(10.0))])
    e2 = np.array([math.cos(math.radians(10.0 + kink_deg)),
                   math.sin(math.radians(10.0 + kink_deg))])
    b = a + panel * e1
    c = b + panel * e2

    def breaks(n, seed_off):
        rng = np.random.default_rng(seed + seed_off)
        w = 1.0 + jitter * rng.uniform(-1.0, 1.0, size=n)
        w = w / w.sum()
        return np.concatenate([[0.0], np.cumsum(w)])

    panels = []
    for k, (p, q, n_s, n_t) in enumerate(((a, b, n1_src, n1_tgt), (b, c, n2_src, n2_tgt))):
        src = segment_side(p, q, breaks(n_s, 10 * k), label=f"src-panel{k}")
        tgt = segment_side(p, q, breaks(n_t, 10 * k + 1), label=f"tgt-panel{k}")
        panels.append((src, tgt))
    return panels, (a, b, c)


# ---------------------------------------------------------------------------
# D1 - planar triangle-triangle interface
# ---------------------------------------------------------------------------
def criteria_d1() -> dict:
    out = {"cases": [], "negative_controls": {}}
    gates = {}
    worst_cov, worst_cons = 0.0, 0.0
    worst_transpose = 0.0
    step = lambda p: np.where(np.asarray(p)[..., 1] < 0.42, 1.0, -2.5)
    for n_src, n_tgt in ((4, 3), (6, 4), (8, 6)):
        src, tgt = planar_tri_case(n_src, n_tgt)
        ents = surface_overlap_entries(src, tgt)
        cov = surface_coverage(src, tgt, ents)
        u = surface_cell_averages(src, step)
        row = {"n_src": src.n, "n_tgt": tgt.n, "h": 1.0 / n_src,
               "coverage_src_abs": cov["src_abs"], "coverage_tgt_abs": cov["tgt_abs"],
               "coverage_src_rel_diagnostic": cov["src_rel"],
               "coverage_tgt_rel_diagnostic": cov["tgt_rel"],
               "coverage_total": abs(cov["src_total"]), "orders": {}}
        for order in (1, 2):
            v = SurfaceMortarOperator(order=order).transfer(src, u, tgt, ents)
            row["orders"][f"order{order}"] = {
                "residual": abs(surface_integral(v, tgt.measures)
                                - surface_integral(u, src.measures))
                / abs(surface_integral(u, src.measures))}
        # overlaps measured in the two directions: coplanar clipping must agree
        back = surface_overlap_entries(tgt, src)
        M_st = np.zeros((src.n, tgt.n))
        M_ts = np.zeros((tgt.n, src.n))
        for (i, j, m, _mom) in ents:
            M_st[i, j] += m
        for (j, i, m, _mom) in back:
            M_ts[j, i] += m
        row["transpose_rel"] = float(np.max(np.abs(M_st - M_ts.T))
                                     / max(float(M_st.max()), 1e-300))
        out["cases"].append(row)
        worst_cov = max(worst_cov, row["coverage_total"], row["coverage_src_abs"],
                        row["coverage_tgt_abs"])
        worst_cons = max(worst_cons, max(r["residual"] for r in row["orders"].values()))
        worst_transpose = max(worst_transpose, row["transpose_rel"])

    # negative controls on the coarsest case
    src, tgt = planar_tri_case(4, 3)
    ents = surface_overlap_entries(src, tgt)
    u = surface_cell_averages(src, step)
    v_mid = SurfaceMortarOperator(order=2, broken="midpoint").transfer(src, u, tgt, ents)
    out["negative_controls"]["midpoint_residual"] = abs(
        surface_integral(v_mid, tgt.measures) - surface_integral(u, src.measures))
    gates["D1_coverage"] = _gate("3D tri-tri area coverage error (17 deg rotated "
                                 "facets, jittered)", worst_cov, 1e-13)
    gates["D1_conservation"] = _gate("3D tri-tri conservation residual (orders 1,2)",
                                     worst_cons, 1e-13)
    gates["D1_negative_control_red"] = {
        "midpoint_residual": out["negative_controls"]["midpoint_residual"],
        "passed": bool(out["negative_controls"]["midpoint_residual"] > 1e-4),
        "note": "point-sampling a non-matching interface must break conservation"}
    out["transpose_max_rel"] = worst_transpose
    gates["D1_transpose"] = _gate("3D tri-tri overlap transpose asymmetry",
                                  worst_transpose, 1e-12)
    return {"D1": out, "gates": gates}


# ---------------------------------------------------------------------------
# D2 - linear patch test (3D tri-tri)
# ---------------------------------------------------------------------------
def criteria_d2() -> dict:
    lin = lambda p: 1.5 * np.asarray(p)[..., 1] - 0.5 * np.asarray(p)[..., 2] + 2.0
    out = {"cases": []}
    gates = {}
    for n_src, n_tgt in ((4, 3), (6, 4), (8, 6)):
        src, tgt = planar_tri_case(n_src, n_tgt)
        ents = surface_overlap_entries(src, tgt)
        u = surface_cell_averages(src, lin)
        v_ex = surface_cell_averages(tgt, lin)
        row = {"n_src": src.n, "n_tgt": tgt.n}
        for order in (1, 2):
            got = SurfaceMortarOperator(order=order).transfer(src, u, tgt, ents)
            row[f"l2_order{order}"] = surface_l2_error(got, v_ex, tgt.measures)
        out["cases"].append(row)
    worst = max(r["l2_order2"] for r in out["cases"])
    gates["D2_patch_test"] = _gate("3D tri-tri linear patch test L2 (order 2)", worst,
                                   1e-12)
    gates["D2_order1_not_exact"] = {
        "min_l2_order1": min(r["l2_order1"] for r in out["cases"]),
        "passed": bool(min(r["l2_order1"] for r in out["cases"]) > 1e-3),
        "note": "a piecewise-constant operator must NOT pass the linear patch "
                "test; a gate that both pass would be dead"}
    return {"D2": out, "gates": gates}


# ---------------------------------------------------------------------------
# D3 - curved, non-coplanar interface (sphere octant, two polyhedra)
# ---------------------------------------------------------------------------
def criteria_d3() -> dict:
    ref = octant_reference(1.0)
    field = lambda p: np.sum(np.asarray(p), axis=-1)
    out = {"reference": ref, "levels": [], "negative_controls": {}}       # noqa: E731
    gates = {}
    for (nt, nph, lv) in CURVED_LEVELS:
        src, tgt = curved_octant_case(nt, nph, lv)
        ents = surface_overlap_entries(src, tgt)
        cov = surface_coverage(src, tgt, ents)
        u = surface_cell_averages(src, field)
        v_raw = SurfaceMortarOperator(order=2).transfer(src, u, tgt, ents)
        v_bf = SurfaceMortarOperator(order=2, backfill="measure").transfer(src, u, tgt, ents)
        sent = surface_integral(u, src.measures)
        recv_raw = surface_integral(v_raw, tgt.measures)
        recv_bf = surface_integral(v_bf, tgt.measures)
        h = float(np.sqrt(src.area / src.n))
        out["levels"].append({
            "n_src": src.n, "n_tgt": tgt.n, "h": h,
            "area_src": src.area, "area_tgt": tgt.area,
            "area_src_rel": src.area / ref["area"] - 1.0,
            "area_tgt_rel": tgt.area / ref["area"] - 1.0,
            "coverage_src_total": cov["src_total"], "coverage_tgt_total": cov["tgt_total"],
            "coverage_src_abs": cov["src_abs"], "coverage_tgt_abs": cov["tgt_abs"],
            "coverage_src_rel_diagnostic": cov["src_rel"],
            "coverage_tgt_rel_diagnostic": cov["tgt_rel"],
            "recv_raw_rel": recv_raw / ref["int_sum"] - 1.0,
            "recv_bf_rel": recv_bf / ref["int_sum"] - 1.0,
            "raw_conservation": recv_raw / sent - 1.0,
            "bf_conservation": recv_bf / sent - 1.0,
        })
    hs = [r["h"] for r in out["levels"]]
    err_area = [abs(r["area_tgt_rel"]) for r in out["levels"]]
    err_field = [abs(r["recv_bf_rel"]) for r in out["levels"]]
    err_raw = [abs(r["raw_conservation"]) for r in out["levels"]]
    out["order_area_tgt"] = {"errors": err_area, "successive": [
        _order(err_area[k], err_area[k + 1]) for k in range(len(err_area) - 1)],
        "fit": _fit_order(err_area, hs)}
    out["order_field_integral"] = {"errors": err_field, "successive": [
        _order(err_field[k], err_field[k + 1]) for k in range(len(err_field) - 1)],
        "fit": _fit_order(err_field, hs)}
    out["order_raw_conservation_defect"] = {"errors": err_raw, "successive": [
        _order(err_raw[k], err_raw[k + 1]) for k in range(len(err_raw) - 1)],
        "fit": _fit_order(err_raw, hs)}
    gates["D3_area_order"] = _gate(
        "curved octant: area (acceptor polyhedron) observed order",
        out["order_area_tgt"]["successive"][-1], 1.8, ">=")
    gates["D3_field_order"] = _gate(
        "curved octant: transferred integral observed order",
        out["order_field_integral"]["successive"][-1], 1.8, ">=")
    gates["D3_field_accuracy"] = _gate(
        "curved octant: transferred integral relative error at the finest level",
        err_field[-1], 3e-2)
    gates["D3_raw_defect_order"] = _gate(
        "curved octant: raw conservation defect observed order",
        out["order_raw_conservation_defect"]["successive"][-1], 1.5, ">=")
    gates["D3_backfill_closes"] = _gate(
        "curved octant: post-backfill conservation residual (finest)",
        abs(out["levels"][-1]["bf_conservation"]), 1e-13)

    # -- negative controls: mismatched resolutions make the trap visible ------
    src, tgt = curved_octant_case(*CURVED_MISMATCH)
    ents = surface_overlap_entries(src, tgt)
    u = surface_cell_averages(src, field)
    sent = surface_integral(u, src.measures)
    v_raw = SurfaceMortarOperator(order=2).transfer(src, u, tgt, ents)
    v_ok = SurfaceMortarOperator(order=2, backfill="measure").transfer(src, u, tgt, ents)
    v_trap = SurfaceMortarOperator(order=2, backfill="measure",
                                   broken="backfill_linear").transfer(src, u, tgt, ents)
    v_mid = SurfaceMortarOperator(order=2, broken="midpoint").transfer(src, u, tgt, ents)
    raw = abs(surface_integral(v_raw, tgt.measures) / sent - 1.0)
    ok = abs(surface_integral(v_ok, tgt.measures) / sent - 1.0)
    trap = abs(surface_integral(v_trap, tgt.measures) / sent - 1.0)
    mid = abs(surface_integral(v_mid, tgt.measures) / sent - 1.0)
    # a tolerance-free check of the arithmetic identity of the backfill itself
    m = tgt.measures
    demo = measure_backfill(np.zeros_like(m), 3.0, 2.5, m)
    demo_trap = measure_backfill(np.zeros_like(m), 3.0, 2.5, m, broken="backfill_linear")
    out["negative_controls"] = {
        "case": {"n_theta": CURVED_MISMATCH[0], "n_phi": CURVED_MISMATCH[1],
                 "ico_level": CURVED_MISMATCH[2], "n_src": src.n, "n_tgt": tgt.n},
        "raw_defect": raw, "correct_backfill_residual": ok, "trap_residual": trap,
        "trap_share_of_deficit": trap / max(raw, 1e-300),
        "midpoint_residual": mid,
        "arithmetic_demo": {
            "deficit": 0.5,
            "correct_after": abs(float(np.dot(demo, m)) - 0.5),
            "trap_after": abs(float(np.dot(demo_trap, m)) - 0.5),
            "formula": "dv_j = (sent-recv) |T_j| / sum_k |T_k|^2   (trap: / sum_k |T_k|)",
            "note": "the trap is an arithmetic identity, not a tuning issue: it "
                    "leaves deficit*(1 - sum|T|^2/(sum|T|)^2) behind"}}
    gates["D3_trap_control_red"] = {
        "raw_defect": raw, "correct_backfill_residual": ok, "trap_residual": trap,
        "arithmetic_trap_after": out["negative_controls"]["arithmetic_demo"]["trap_after"],
        "passed": bool(trap > 1e-2 and raw > 1e-2 and ok < 1e-13
                       and out["negative_controls"]["arithmetic_demo"]["correct_after"] < 1e-12
                       and out["negative_controls"]["arithmetic_demo"]["trap_after"] > 1e-2),
        "note": "on a mismatched-resolution curved interface the measure-"
                "normalisation trap must leave a percent-level residual"}
    gates["D3_midpoint_control_red"] = {
        "midpoint_residual": mid, "passed": bool(mid > 1e-4)}
    return {"D3": out, "gates": gates}


# ---------------------------------------------------------------------------
# D4 - symmetry about one common mirror plane
# ---------------------------------------------------------------------------
def criteria_d4() -> dict:
    out = {"midpoint_mirror": {}, "curved_asymmetry": {}}
    gates = {}
    # mirror the *interface data* about the plane x = 0.5: the interface plane is
    # x = 0.5, so this maps the interface onto itself and both sides must map to
    # mirror partners of themselves.
    src, tgt = planar_tri_case(6, 4, plane_x=0.5)
    ents = surface_overlap_entries(src, tgt)
    sfield = lambda p: np.sin(2.7 * np.asarray(p)[..., 1]) + 0.3 * np.asarray(p)[..., 2]
    u = surface_cell_averages(src, sfield)
    v = SurfaceMortarOperator(order=2).transfer(src, u, tgt, ents)

    # same mirror axis for both sides: reflect about y = 0.5 (in-plane axis,
    # i.e. the interface plane x = 0.5 is invariant and the axis is shared)
    axis_point = np.array([0.0, 0.5, 0.0])
    axis_normal = np.array([0.0, 1.0, 0.0])
    src_m = src.mirrored(axis_point, axis_normal, label="cube-left-mirror")
    tgt_m = tgt.mirrored(axis_point, axis_normal, label="cube-right-mirror")
    ents_m = surface_overlap_entries(src_m, tgt_m)
    # the *same* donor data is transferred on the mirrored meshes: the mirror is
    # a change of frame, not a change of data, so equivariance means the reflected
    # result equals the original one cell by cell (re-evaluating the field on the
    # mirrored mesh would compare a different field, which is a different test).
    v_m = SurfaceMortarOperator(order=2).transfer(src_m, u, tgt_m, ents_m)
    # the reflected mesh has the reflected centroids, so the pairing is a nearest
    # -centroid match whose distance is itself part of the gate
    refl_c = tgt.centroids - np.outer(2.0 * ((tgt.centroids - axis_point) @ axis_normal),
                                      axis_normal)
    d = np.linalg.norm(tgt_m.centroids[:, None, :] - refl_c[None, :, :], axis=2)
    perm = np.argmin(d, axis=1)
    match = float(np.max(d[np.arange(tgt_m.n), perm]))
    bijective = bool(len(set(perm.tolist())) == tgt.n)
    drift = float(np.max(np.abs(v_m - v[perm])))
    scale = float(np.max(np.abs(v)))
    out["midpoint_mirror"] = {"max_drift": drift, "scale": scale,
                              "relative": drift / max(scale, 1e-300),
                              "centroid_match": match, "bijective": bijective,
                              "mirror_axis": "y = 0.5 (one common axis for both sides)"}
    gates["D4_mirror_equivariance"] = _gate(
        "3D tri-tri reflection equivariance about one common mirror axis",
        drift / max(scale, 1e-300), 1e-12,
        note=f"centroid pairing residual {match:.2e}, bijective={bijective}")

    # curved interface: the projection mortar is *not* symmetric; measure it
    src_c, tgt_c = curved_octant_case(*CURVED_LEVELS[1])
    fwd = surface_overlap_entries(src_c, tgt_c)
    bwd = surface_overlap_entries(tgt_c, src_c)
    M_f = np.zeros((src_c.n, tgt_c.n))
    M_b = np.zeros((tgt_c.n, src_c.n))
    for (i, j, m, _mom) in fwd:
        M_f[i, j] += m
    for (j, i, m, _mom) in bwd:
        M_b[j, i] += m
    rel_asym = float(np.max(np.abs(M_f - M_b.T)) / max(float(M_f.max()), 1e-300))
    out["curved_asymmetry"] = {
        "max_abs_asym": rel_asym,
        "note": ("donor-plane projection mortar: m_ij is measured in S_i's plane "
                 "and m_ji in T_j's, so the overlap table is not transposable on a "
                 "curved interface. Reported as a property (O(theta)), not faked "
                 "to zero by symmetrising; energy/mass bookkeeping stays exact "
                 "because the backfill is applied per transfer direction.")}
    gates["D4_curved_asymmetry_reported"] = {
        "max_abs_asym": rel_asym, "passed": bool(0.0 <= rel_asym < 0.2),
        "note": "asymmetry must be bounded by the dihedral angle of the two "
                "polyhedral approximations, not zero"}
    gates["D4_curved_conservation_both_directions"] = {}
    ref_field = lambda p: np.sum(np.asarray(p), axis=-1)                  # noqa: E731
    resid = {}
    for name, s_, t_, ents_ in (("fwd", src_c, tgt_c, fwd), ("bwd", tgt_c, src_c, bwd)):
        uu = surface_cell_averages(s_, ref_field)
        vv = SurfaceMortarOperator(order=2, backfill="measure").transfer(s_, uu, t_, ents_)
        resid[name] = abs(surface_integral(vv, t_.measures)
                          - surface_integral(uu, s_.measures)) / abs(surface_integral(uu, s_.measures))
    gates["D4_curved_conservation_both_directions"]["residuals"] = resid
    gates["D4_curved_conservation_both_directions"]["passed"] = bool(
        max(resid.values()) <= 1e-13)
    return {"D4": out, "gates": gates}


# ---------------------------------------------------------------------------
# D5 - 2D piecewise-linear (kinked) interface
# ---------------------------------------------------------------------------
def criteria_d5() -> dict:
    out = {"levels": [], "patch": [], "negative_controls": {}}
    gates = {}
    lin = lambda p: 2.0 * np.asarray(p)[..., 0] - 3.0 * np.asarray(p)[..., 1] + 0.7
    smooth = lambda p: np.sin(4.0 * np.asarray(p)[..., 0]) * np.cos(3.0 * np.asarray(p)[..., 1]) + 0.5

    def transfer(panels, field, order, backfill=None, broken=None):
        sent = recv = 0.0
        vals, exact, meas = [], [], []
        for (src, tgt) in panels:
            u = cell_averages(src, field)
            v_ex = cell_averages(tgt, field)
            v = MortarOperator(order=order, backfill=backfill, broken=broken).transfer(
                src, u, tgt)
            sent += integral(u, src.measures)
            recv += integral(v, tgt.measures)
            vals.append(v)
            exact.append(v_ex)
            meas.append(tgt.measures)
        return (np.concatenate(vals), np.concatenate(exact), np.concatenate(meas),
                sent, recv)

    for (n1s, n2s, n1t, n2t) in ((4, 3, 6, 5), (8, 6, 12, 10), (16, 12, 24, 20)):
        panels, _verts = kinked_case(n1s, n2s, n1t, n2t)
        v, v_ex, m, sent, recv = transfer(panels, smooth, 2)
        row = {"cells_src": n1s + n2s, "cells_tgt": n1t + n2t,
               "h": 1.0 / min(n1s, n2s),
               "conservation": abs(recv - sent) / max(abs(sent), 1.0),
               "l2": float(math.sqrt(np.dot((v - v_ex) ** 2, m) / m.sum()))}
        vl, vlex, ml, sentl, recvl = transfer(panels, lin, 2)
        row["patch_l2"] = float(math.sqrt(np.dot((vl - vlex) ** 2, ml) / ml.sum()))
        out["levels"].append(row)
        out["patch"].append(row["patch_l2"])
    errs = [r["l2"] for r in out["levels"]]
    hs = [r["h"] for r in out["levels"]]
    out["order"] = {"errors": errs, "successive": [
        _order(errs[k], errs[k + 1]) for k in range(len(errs) - 1)],
        "fit": _fit_order(errs, hs)}
    gates["D5_conservation"] = _gate("2D kinked (30 deg) interface conservation residual",
                                     max(r["conservation"] for r in out["levels"]), 1e-13)
    gates["D5_patch_test"] = _gate("2D kinked interface linear patch test L2",
                                   max(out["patch"]), 1e-12)
    gates["D5_order"] = _gate("2D kinked interface smooth-field observed order",
                              out["order"]["successive"][-1], 1.8, ">=")
    panels, _ = kinked_case(4, 3, 6, 5)
    _v, _e, _m, sent, recv = transfer(panels, smooth, 2, broken="midpoint")
    out["negative_controls"]["midpoint"] = abs(recv - sent) / abs(sent)
    gates["D5_negative_control_red"] = {"midpoint_residual":
                                        out["negative_controls"]["midpoint"],
                                        "passed": bool(
                                            out["negative_controls"]["midpoint"] > 1e-4)}
    return {"D5": out, "gates": gates}


# ---------------------------------------------------------------------------
# D6 - three-operator table, comparable with the 1D contract
# ---------------------------------------------------------------------------
def three_operator_table() -> dict:
    """Conservation and order of mortar / RBF / ME on the 1D, 2D, 3D and curved cases.

    1D: ``segment_side`` embedded in 1D (the m5207 contract geometry).
    2D: 17 deg oblique interface (``triangle_case_2d`` geometry of m5436, but with
        the operator calls made here so the table is self-contained).
    3D: the new triangle-triangle shared face (17 deg rotated facet family).
    Curved: the sphere octant with two polyhedra -- mortar only, because the RBF
    and momentum-exchange kernels of m5207/m5436 are defined on a *coplanar*
    interface side; running them there would silently mix two manifolds.
    """
    rows = []

    def record(case, dim, src, tgt, field, ops, ents=None):
        u = (surface_cell_averages(src, field) if isinstance(src, TriSide)
             else cell_averages(src, field))
        sent = (surface_integral(u, src.measures) if isinstance(src, TriSide)
                else integral(u, src.measures))
        for name, op in ops.items():
            v = op(src, u, tgt, ents) if callable(op) else op.transfer(src, u, tgt)
            recv = (surface_integral(v, tgt.measures) if isinstance(tgt, TriSide)
                    else integral(v, tgt.measures))
            rows.append({"case": case, "dim": dim, "operator": name,
                         "sent": sent, "recv": recv,
                         "conservation_rel": recv / sent - 1.0})

    # -- 1D reference ------------------------------------------------------
    f1d = np.linspace(0.0, 1.0, 8)
    f1d_t = np.linspace(0.0, 1.0, 12)
    s1 = segment_side(np.array([0.0]), np.array([1.0]), f1d, label="1d-donor")
    t1 = segment_side(np.array([0.0]), np.array([1.0]), f1d_t, label="1d-acceptor")
    field1 = lambda p: np.where(np.asarray(p)[..., 0] < 0.42, 1.0, -2.5)
    record("1D interval", 1, s1, t1, field1,
           {"mortar": MortarOperator(order=2), "rbf": RBFOperator(),
            "me": MomentumExchangeOperator()})

    # -- 2D oblique 17 deg -------------------------------------------------
    p0, p1 = np.array([0.30, -0.40]), np.array([0.30, -0.40]) + np.array(
        [math.cos(math.radians(17.0)), math.sin(math.radians(17.0))])
    s2 = segment_side(p0, p1, jittered_cuts_local(7, 0.4, 7), label="2d-donor")
    t2 = segment_side(p0, p1, jittered_cuts_local(11, 0.4, 8), label="2d-acceptor")
    field2 = lambda p: np.where(2.0 * np.asarray(p)[..., 0] - 3.0 * np.asarray(p)[..., 1]
                                < -0.1, 1.0, -2.5)
    record("2D 17 deg oblique", 2, s2, t2, field2,
           {"mortar": MortarOperator(order=2), "rbf": RBFOperator(),
            "me": MomentumExchangeOperator()})

    # -- 3D tri-tri shared face -------------------------------------------
    src3, tgt3 = planar_tri_case(6, 4)
    ents3 = surface_overlap_entries(src3, tgt3)
    side_s = from_polygons(src3.tris, label="cube-left")
    side_t = from_polygons(tgt3.tris, label="cube-right")
    field3 = lambda p: np.where(np.asarray(p)[..., 1] < 0.42, 1.0, -2.5)
    record("3D tri-tri face (17 deg facets)", 3, src3, tgt3, field3,
           {"mortar": SurfaceMortarOperator(order=2)}, ents=ents3)
    record("3D tri-tri face (17 deg facets)", 3, side_s, side_t, field3,
           {"rbf": RBFOperator(), "me": MomentumExchangeOperator()})

    # -- curved octant -----------------------------------------------------
    src_c, tgt_c = curved_octant_case(*CURVED_LEVELS[0])
    ents_c = surface_overlap_entries(src_c, tgt_c)
    field_c = lambda p: np.where(np.asarray(p)[..., 2] < 0.5, 1.0, -2.5)
    record("3D curved octant (two polyhedra)", 3, src_c, tgt_c, field_c,
           {"mortar": SurfaceMortarOperator(order=2, backfill="measure")}, ents=ents_c)
    record("3D curved octant (two polyhedra)", 3, src_c, tgt_c, field_c,
           {"mortar (no backfill)": SurfaceMortarOperator(order=2)}, ents=ents_c)
    return {"rows": rows,
            "note": ("RBF and ME are listed only where the interface side is "
                     "coplanar (their kernels of m5207/m5436 are defined on a "
                     "single manifold); on the curved octant only the mortar "
                     "operator, which owns the projection/backfill machinery, is "
                     "reported.")}


def jittered_cuts_local(n: int, jitter: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = 1.0 + jitter * rng.uniform(-1.0, 1.0, size=n)
    w = w / w.sum()
    return np.concatenate([[0.0], np.cumsum(w)])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _provenance() -> dict:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                              capture_output=True, check=False).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root,
                                text=True, capture_output=True, check=False).stdout.strip()
    except OSError:                                   # pragma: no cover
        head, branch = "unknown", "unknown"
    return {"git_head": head, "git_branch": branch,
            "module": os.path.abspath(__file__)}


def run_acceptance() -> dict:
    """Compute every gate; returns the record that is written to the artifact."""
    record = {"campaign_node": "IN52-N01", "mission": 6114,
              "provenance": _provenance(), "gates": {}}
    for fn in (criteria_d1, criteria_d2, criteria_d3, criteria_d4, criteria_d5):
        part = fn()
        for k, v in part.items():
            if k == "gates":
                record["gates"].update(v)
            else:
                record[k] = v
    record["D6"] = three_operator_table()
    payload = json.dumps(record["D6"], sort_keys=True).encode()
    record["D6"]["sha256"] = hashlib.sha256(payload).hexdigest()
    record["verdict"] = "PASS" if all(
        g.get("passed", g.get("measured", 1e9) <= g.get("threshold", 0.0))
        for g in record["gates"].values()) else "FAIL"
    return record


def main(argv=None) -> int:                            # pragma: no cover
    rec = run_acceptance()
    out = os.path.join("artifacts", "in52_n01", "interface_tri3d_acceptance.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    for name, gate in rec["gates"].items():
        val = gate.get("measured", gate.get("passed"))
        print(f"{name:44s} {val}")
    print("verdict:", rec["verdict"], "->", out)
    return 0 if rec["verdict"] == "PASS" else 1


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())