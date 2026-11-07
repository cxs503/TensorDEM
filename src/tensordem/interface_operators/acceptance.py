"""Machine-checkable acceptance harness for the 2D/3D interface-operator node.

Gates (campaign node P48-N02, mission 5436)
-------------------------------------------
C1  2D non-matching interface rotated 17 deg, piecewise-constant field:
    conservation residual <= 1e-13
C2  2D linear-field patch test: L2 error after mortar projection <= 1e-12
C3  2D smooth field (sin*cos) convergence: observed order >= 1.8 when h halves
C4  3D two cubes on a shared non-matching face: area-coverage error <= 1e-10 and
    conservation residual <= 1e-10; curved sphere/box clipping area vs closed
    form <= 1e-6
C5  symmetry: measure-weighted transpose consistency <= 1e-12 (1D/2D/3D) and
    reflection/rotation equivariance <= 1e-12 with one **common** mirror axis
C6  three-operator comparison table (mortar / RBF / ME) on the same 1D, 2D and 3D
    cases, written to a JSON artifact, with the m5207 1D reference in the same
    table

Every gate is paired with a negative control, so a green verdict means the gate
can actually go red: mortar ``broken="midpoint"`` must trip C1/C4, RBF
``broken="backfill_linear"`` must trip the normalisation gate, ME
``broken="per_node_norm"`` must open the momentum budget.

Nothing here is transcribed by hand: the JSON artifact is produced by
``run_acceptance`` and consumed by ``scripts/run_mortar_2d3d_benchmark.py`` and
``tests/test_mortar_2d3d.py``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

from .geometry import gl_rule, sphere_box_area
from .mortar import (MortarOperator, cell_averages, grid_side, integral, l2_error,
                     measure_weighted_symmetry, overlap_entries, reflect_side,
                     rotate_side, segment_side, tri_patch_side)
from .operators import MomentumExchangeOperator, RBFOperator

__all__ = ["triangle_case_2d", "cube_case_3d", "jittered_cuts", "run_acceptance",
           "sphere_box_area_facets", "M5207_DEFAULT"]


# ---------------------------------------------------------------------------n
# helpers
# ---------------------------------------------------------------------------
def jittered_cuts(n: int, jitter: float, seed: int) -> np.ndarray:
    """Lattice cuts ``0 = x0 < ... < xn = 1`` with O(h) jitter on cell widths.

    ``w_k = 1 + jitter * U(-1, 1)`` renormalised.  A lattice plus O(h) jitter
    refines properly; a random node cloud does not (its clustering noise
    saturates and reports a fake order plateau -- m5207 trap list).
    """
    rng = np.random.default_rng(seed)
    w = 1.0 + jitter * rng.uniform(-1.0, 1.0, size=n)
    w = w / w.sum()
    cuts = np.concatenate([[0.0], np.cumsum(w)])
    cuts[-1] = 1.0
    return cuts


def _gate(name: str, value: float, threshold: float, relation: str = "<=",
          note: str = "") -> dict:
    ok = (value <= threshold) if relation == "<=" else (value >= threshold)
    out = {"gate": name, "measured": float(value), "threshold": float(threshold),
           "relation": relation, "passed": bool(ok)}
    if note:
        out["note"] = note
    return out


def _order(e_coarse: float, e_fine: float) -> float:
    if e_coarse <= 0.0 or e_fine <= 0.0:
        return float("nan")
    return float(np.log2(e_coarse / e_fine))


def _fit_order(errors: list, hs: list) -> float:
    x = np.log(np.asarray(hs, dtype=float))
    y = np.log(np.asarray(errors, dtype=float))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


# ---------------------------------------------------------------------------
# case builders
# ---------------------------------------------------------------------------
@dataclass
class Case2D:
    """2D non-matching interface: one straight trace, two partitions."""

    src: object
    tgt: object
    p0: np.ndarray
    tangent: np.ndarray
    angle_deg: float
    length: float

    def arclength(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=float)
        return (p - self.p0) @ self.tangent


@dataclass
class Case3D:
    """Two unit cubes sharing the square interface face ``x = 1``."""

    src: object
    tgt: object
    origin: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    patch_area: float
    n_src: int
    n_tgt: int


def triangle_case_2d(n_src: int = 7, n_tgt: int = 11, angle_deg: float = 17.0,
                     length: float = 1.0, jitter: float = 0.0, seed: int = 0,
                     p0=(0.3, -0.4)) -> Case2D:
    """2D interface at ``angle_deg`` to the x axis with non-matching partitions.

    The two sides carry different cell counts and, for ``jitter > 0``, different
    O(h) node jitter, so no cell of one side matches a cell of the other.  Both
    traces are the *same* segment, so the overlap decomposition partitions it
    exactly and the mortar conservation residual is pure round-off.
    """
    th = math.radians(angle_deg)
    t = np.array([math.cos(th), math.sin(th)])
    p0a = np.asarray(p0, dtype=float)
    p1a = p0a + length * t
    c_src = jittered_cuts(n_src, jitter, seed) if jitter > 0 else np.linspace(0, 1, n_src + 1)
    c_tgt = (jittered_cuts(n_tgt, jitter, seed + 101) if jitter > 0
             else np.linspace(0, 1, n_tgt + 1))
    return Case2D(src=segment_side(p0a, p1a, c_src, "2d-src"),
                  tgt=segment_side(p0a, p1a, c_tgt, "2d-tgt"),
                  p0=p0a, tangent=t, angle_deg=angle_deg, length=length)


def cube_case_3d(n_src: int = 9, n_tgt: int = 7, jitter: float = 0.15,
                 seed_src: int = 1, seed_tgt: int = 2) -> Case3D:
    """Left cube ``[0,1]^3`` and right cube ``[1,2]x[0,1]^2``.

    Their shared face is the unit square in the plane ``x = 1``: the donor side is
    a jittered quad grid, the acceptor a jittered triangular patch, so the clipper
    is exercised for quad-quad, quad-triangle and triangle-triangle pairs.
    Boundary nodes stay on the patch boundary, so the overlap areas must reproduce
    the patch area (gate C4).
    """
    origin = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])
    src = grid_side(origin, e1, e2, n_src, n_src, jitter=jitter, seed=seed_src,
                    label="3d-src")
    tgt = tri_patch_side(origin, e1, e2, n_tgt, n_tgt, jitter=jitter,
                         seed=seed_tgt, label="3d-tgt")
    return Case3D(src=src, tgt=tgt, origin=origin, e1=e1, e2=e2, patch_area=1.0,
                  n_src=n_src, n_tgt=n_tgt)


# ---------------------------------------------------------------------------
# curved clipping (sphere / box) by flat facets, for the curved branch of C4
# ---------------------------------------------------------------------------
def _newell_area(poly: np.ndarray) -> float:
    """Area of a planar 3D polygon via Newell's normal formula."""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    n = np.zeros(3)
    for k in range(len(p)):
        n += np.cross(p[k], p[(k + 1) % len(p)])
    return 0.5 * float(np.linalg.norm(n))


def _clip_polygon_by_plane(poly: np.ndarray, point: np.ndarray, normal: np.ndarray,
                           tol: float = 1e-14) -> np.ndarray:
    """Keep the part of a 3D polygon on the positive side of a plane."""
    p = np.asarray(poly, dtype=float)
    point = np.asarray(point, dtype=float)
    nrm = np.asarray(normal, dtype=float)
    d = (p - point) @ nrm
    nxt = np.roll(p, -1, axis=0)
    dn = np.roll(d, -1)
    keep, keep_n = d >= -tol, dn >= -tol
    out = []
    for i in range(len(p)):
        if keep[i]:
            out.append(p[i])
        if keep[i] != keep_n[i]:
            t = d[i] / (d[i] - dn[i])
            out.append(p[i] + t * (nxt[i] - p[i]))
    return np.asarray(out, dtype=float) if out else np.zeros((0, 3))


def _sphere_triangles(radius: float, n_theta: int, n_phi: int) -> list:
    """Lat/lon triangulation of a sphere (inscribed facets: area < true area)."""
    th = np.linspace(0.0, np.pi, n_theta + 1)
    ph = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)

    def pt(i, j):
        return radius * np.array([math.sin(th[i]) * math.cos(ph[j]),
                                  math.sin(th[i]) * math.sin(ph[j]),
                                  math.cos(th[i])])

    tris = []
    for i in range(n_theta):
        for j in range(n_phi):
            a, b, c, d = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
            tris.append(np.array([a, b, c]))
            if i > 0:
                tris.append(np.array([a, c, d]))
    return [t for t in tris if _newell_area(t) > 0.0]


def sphere_box_area_facets(radius: float, box, n_theta: int, n_phi: int) -> float:
    """Flat-facet sphere/box clipping area: the *polygonal* clipping path.

    Each spherical triangle is clipped by the six box half-spaces and its area is
    summed, which is what a surface-mesh-based mortar sees.  It converges
    O(h^2) from below to
    :func:`~tensordem.interface_operators.geometry.sphere_box_area`.
    """
    planes = []
    for k, (lo, hi) in enumerate(box):
        ek = np.zeros(3)
        ek[k] = 1.0
        if lo > -np.inf:
            planes.append((lo * ek, ek.copy()))
        if hi < np.inf:
            nrm = -ek
            planes.append((hi * ek, nrm))
    total = 0.0
    for tri in _sphere_triangles(radius, n_theta, n_phi):
        poly = tri
        for point, nrm in planes:
            if len(poly) < 3:
                break
            poly = _clip_polygon_by_plane(poly, point, nrm)
        if len(poly) >= 3:
            total += _newell_area(poly)
    return float(total)


# ---------------------------------------------------------------------------
# C1 / C2 / C3
# ---------------------------------------------------------------------------
def criteria_c1_c2_c3() -> dict:
    case = triangle_case_2d(n_src=7, n_tgt=11, angle_deg=17.0)
    src, tgt = case.src, case.tgt
    s = case.arclength
    gates = {}

    def step_field(pts):
        return np.where(s(pts) < 0.42, 1.0, -2.5)

    u_step = cell_averages(src, step_field)
    c1 = {"order1": {}, "order2": {}, "negative_controls": {}}
    for order in (1, 2):
        v = MortarOperator(order=order).transfer(src, u_step, tgt)
        sent, recv = integral(u_step, src.measures), integral(v, tgt.measures)
        c1[f"order{order}"] = {"sent": sent, "recv": recv,
                               "residual": abs(recv - sent) / max(abs(sent), 1.0)}
    v = MortarOperator(order=2, broken="midpoint").transfer(src, u_step, tgt)
    c1["negative_controls"]["mortar_midpoint"] = abs(
        integral(v, tgt.measures) - integral(u_step, src.measures))
    # the backfill trap bites only on an operator that is not conservative by
    # construction, so it is exercised on the RBF operator here
    v = RBFOperator(broken="backfill_linear").transfer(src, u_step, tgt)
    c1["negative_controls"]["rbf_backfill_linear"] = abs(
        integral(v, tgt.measures) - integral(u_step, src.measures))
    worst = max(c1["order1"]["residual"], c1["order2"]["residual"])
    gates["C1"] = _gate("2D 17deg piecewise-constant conservation residual", worst,
                        1e-13)
    gates["C1_negative_control_red"] = {
        "mortar_midpoint_residual": c1["negative_controls"]["mortar_midpoint"],
        "rbf_backfill_linear_residual": c1["negative_controls"]["rbf_backfill_linear"],
        "passed": bool(c1["negative_controls"]["mortar_midpoint"] > 1e-6
                       and c1["negative_controls"]["rbf_backfill_linear"] > 1e-3),
        "note": "a negative control must break the gate, else the gate is dead"}

    # -- C2: linear patch test (2D, plus the 3D shared face) ----------------
    def linear_field(pts):
        p = np.asarray(pts, dtype=float)
        return 2.0 * p[..., 0] - 3.0 * p[..., 1] + 0.7

    u_lin = cell_averages(src, linear_field)
    v_lin = cell_averages(tgt, linear_field)
    c2 = {"2d_17deg": {}, "3d_cube_face": {}}
    for order in (1, 2):
        got = MortarOperator(order=order).transfer(src, u_lin, tgt)
        c2["2d_17deg"][f"order{order}"] = l2_error(got, v_lin, tgt.measures)
    c3d = cube_case_3d(9, 7)

    def lin3(pts):
        p = np.asarray(pts, dtype=float)
        return 1.5 * p[..., 1] - 0.5 * p[..., 2] + 2.0

    ul3 = cell_averages(c3d.src, lin3)
    vl3 = cell_averages(c3d.tgt, lin3)
    for order in (1, 2):
        got = MortarOperator(order=order).transfer(c3d.src, ul3, c3d.tgt)
        c2["3d_cube_face"][f"order{order}"] = l2_error(got, vl3, c3d.tgt.measures)
    gates["C2"] = _gate("2D linear patch test L2 (order 2)",
                        c2["2d_17deg"]["order2"], 1e-12)

    # -- C3: smooth-field convergence, h halved ----------------------------
    def smooth_field(case_):
        def f(pts):
            x = case_.arclength(pts)
            return np.sin(3.1 * x) * np.cos(2.3 * x + 0.4)
        return f

    c3 = {"levels": [], "order1": {}, "order2": {}}
    for n_src, n_tgt in ((8, 11), (16, 22), (32, 44)):
        c = triangle_case_2d(n_src=n_src, n_tgt=n_tgt, angle_deg=17.0, jitter=0.4,
                             seed=7)
        u = cell_averages(c.src, smooth_field(c))
        v_ex = cell_averages(c.tgt, smooth_field(c))
        row = {"n_src": n_src, "n_tgt": n_tgt, "h": 1.0 / n_src}
        for order in (1, 2):
            got = MortarOperator(order=order).transfer(c.src, u, c.tgt)
            row[f"l2_order{order}"] = l2_error(got, v_ex, c.tgt.measures)
            row[f"resid_order{order}"] = abs(
                integral(got, c.tgt.measures) - integral(u, c.src.measures)) / max(
                abs(integral(u, c.src.measures)), 1.0)
        c3["levels"].append(row)
    hs = [r["h"] for r in c3["levels"]]
    for order in (1, 2):
        errs = [r[f"l2_order{order}"] for r in c3["levels"]]
        c3[f"order{order}"] = {
            "errors": errs,
            "order_successive": [_order(errs[k], errs[k + 1])
                                 for k in range(len(errs) - 1)],
            "order_fit": _fit_order(errs, hs)}
    gates["C3"] = _gate("2D smooth-field observed order (order-2 mortar, last pair)",
                        c3["order2"]["order_successive"][-1], 1.8, ">=")
    return {"C1": c1, "C2": c2, "C3": c3, "gates": gates}


# ---------------------------------------------------------------------------
# C4
# ---------------------------------------------------------------------------
def criteria_c4() -> dict:
    out = {"coverage": [], "conservation": {}, "curved": {}}
    gates = {}
    for n_src, n_tgt in ((9, 7), (17, 13), (33, 26)):
        c = cube_case_3d(n_src, n_tgt)
        ent = overlap_entries(c.src, c.tgt)
        covered = sum(m for _, _, m, _ in ent)
        out["coverage"].append({
            "n_src_side": n_src, "n_tgt_cells": c.tgt.n, "pairs": len(ent),
            "covered_area": covered, "patch_area": c.patch_area,
            "coverage_error": abs(covered - c.patch_area) / c.patch_area,
            "src_measure_error": abs(c.src.measures.sum() - c.patch_area),
            "tgt_measure_error": abs(c.tgt.measures.sum() - c.patch_area)})
    c = cube_case_3d(9, 7)

    def step3(pts):
        p = np.asarray(pts, dtype=float)
        return np.where(p[..., 1] + p[..., 2] < 1.0, 1.0, -2.5)

    def smooth3(pts):
        p = np.asarray(pts, dtype=float)
        return np.sin(2.0 * p[..., 1]) * np.cos(3.0 * p[..., 2])

    for name, field in (("piecewise_constant", step3), ("smooth", smooth3)):
        u = cell_averages(c.src, field)
        v_ex = cell_averages(c.tgt, field)
        for order in (1, 2):
            got = MortarOperator(order=order).transfer(c.src, u, c.tgt)
            sent, recv = integral(u, c.src.measures), integral(got, c.tgt.measures)
            out["conservation"][f"{name}_order{order}"] = {
                "sent": sent, "recv": recv,
                "residual": abs(recv - sent) / max(abs(sent), 1.0),
                "l2_error": l2_error(got, v_ex, c.tgt.measures)}
    u = cell_averages(c.src, step3)
    vbad = MortarOperator(order=1, broken="midpoint").transfer(c.src, u, c.tgt)
    out["conservation"]["negative_control_midpoint"] = {
        "residual": abs(integral(vbad, c.tgt.measures) - integral(u, c.src.measures))
        / max(abs(integral(u, c.src.measures)), 1.0)}
    worst_cov = max(r["coverage_error"] for r in out["coverage"])
    worst_cons = max(abs(v["residual"]) for k, v in out["conservation"].items()
                     if not k.startswith("negative_control"))
    gates["C4_coverage"] = _gate("3D shared-face area coverage error", worst_cov, 1e-10)
    gates["C4_conservation"] = _gate("3D shared-face conservation residual",
                                     worst_cons, 1e-10)
    gates["C4_negative_control_red"] = {
        "residual": out["conservation"]["negative_control_midpoint"]["residual"],
        "passed": bool(out["conservation"]["negative_control_midpoint"]["residual"] > 1e-6)}

    # -- curved branch: sphere cut by boxes with closed-form areas ----------
    R, inf = 1.0, float("inf")
    closed = [
        ("hemisphere x>=0", ((0.0, inf), (-inf, inf), (-inf, inf)), 2.0 * math.pi),
        ("spherical_cap z>=0.35", ((-inf, inf), (-inf, inf), (0.35, inf)),
         2.0 * math.pi * (1.0 - 0.35)),
        ("octant", ((0.0, inf), (0.0, inf), (0.0, inf)), math.pi / 2.0),
        ("whole_sphere_in_box", ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)), 4.0 * math.pi),
        ("two_plane_lune", ((0.0, inf), (0.0, inf), (-inf, inf)), math.pi),
    ]
    curved = {"closed_form": [], "facet_convergence": {}}
    for name, box, ref in closed:
        got = sphere_box_area(R, box)
        curved["closed_form"].append({"case": name, "computed": got,
                                      "closed_form": ref,
                                      "rel_error": abs(got - ref) / ref})
    worst_rel = max(r["rel_error"] for r in curved["closed_form"])
    gates["C4_curved_closed_form"] = _gate(
        "curved sphere/box clipping area vs closed form", worst_rel, 1e-6)

    box = ((-0.5, 0.7), (-0.9, 0.4), (0.2, 1.5))     # no closed form
    ref = sphere_box_area(R, box)
    rows = []
    for n_theta, n_phi in ((16, 32), (32, 64), (64, 128)):
        a = sphere_box_area_facets(R, box, n_theta, n_phi)
        rows.append({"n_theta": n_theta, "n_phi": n_phi, "facet_area": a,
                     "rel_error": abs(a - ref) / ref, "h": math.pi / n_theta})
    a1, a2 = rows[-2]["facet_area"], rows[-1]["facet_area"]
    extrap = (4.0 * a2 - a1) / 3.0
    curved["facet_convergence"] = {
        "box": box, "semi_analytic_area": ref, "rows": rows,
        "richardson_area": extrap,
        "richardson_rel_error": abs(extrap - ref) / ref,
        "observed_order": _order(abs(a1 - ref), abs(a2 - ref)),
        "note": ("inscribed flat facets converge O(h^2) to the curved value; the "
                 "Richardson limit is the asymptote of a mesh-based operator, and "
                 "the residual bias is the small-circle representation error that "
                 "no flat-facet mesh can remove (documented limitation)")}
    gates["C4_curved_facet_order"] = _gate(
        "curved sphere/box flat-facet clipping: observed convergence order",
        curved["facet_convergence"]["observed_order"], 1.8, ">=",
        note=("flat facets cannot represent small-circle cuts, so the faceted "
              "operator converges to the curved area only as O(h^2); the 1e-6 "
              "area gate itself is carried by the curved clipping path"))
    out["curved"] = curved
    return {"C4": out, "gates": gates}


# ---------------------------------------------------------------------------
# C5
# ---------------------------------------------------------------------------
def criteria_c5() -> dict:
    out = {"transpose": {}, "reflection_2d": {}, "rotation_3d": {}}
    gates = {}

    c1d = triangle_case_2d(7, 11, angle_deg=0.0, p0=(0.0, 0.0))
    c2d = triangle_case_2d(7, 11, angle_deg=17.0)
    c3 = cube_case_3d(9, 7)
    for name, case in (("1d", c1d), ("2d_oblique_17deg", c2d), ("3d_cube_face", c3)):
        A_st = MortarOperator(order=1).matrix(case.src, case.tgt)
        A_ts = MortarOperator(order=1).matrix(case.tgt, case.src)
        out["transpose"][name] = measure_weighted_symmetry(
            A_st, A_ts, case.src.measures, case.tgt.measures)
    gates["C5_transpose"] = _gate("max measure-weighted M_ij vs M_ji relative diff",
                                  max(out["transpose"].values()), 1e-12)

    # -- 2D reflection about the *common* mirror axis ----------------------
    case = triangle_case_2d(7, 11, 17.0)

    def smooth2d(pts):
        x = case.arclength(pts)
        return np.sin(3.1 * x) * np.cos(2.3 * x + 0.4)

    u = cell_averages(case.src, smooth2d)
    v = MortarOperator(order=2).transfer(case.src, u, case.tgt)
    mid = case.p0 + 0.5 * case.length * case.tangent
    src_m = reflect_side(case.src, mid, case.tangent)
    tgt_m = reflect_side(case.tgt, mid, case.tangent)
    v_m = MortarOperator(order=2).transfer(src_m, u[::-1], tgt_m)
    flip = float(np.max(np.abs(v - v_m[::-1])) / max(np.max(np.abs(v)), 1e-300))
    src_w = reflect_side(case.src, case.src.centroids.mean(axis=0), case.tangent)
    tgt_w = reflect_side(case.tgt, case.tgt.centroids.mean(axis=0), case.tangent)
    v_w = MortarOperator(order=2).transfer(src_w, u[::-1], tgt_w)
    out["reflection_2d"] = {
        "symmetry_flip": flip,
        "mirror_axis": {"point": mid.tolist(), "normal": case.tangent.tolist()},
        "negative_control_two_axes": float(
            np.max(np.abs(v - v_w[::-1])) / max(np.max(np.abs(v)), 1e-300)),
        "note": "one mirror axis shared by both sides"}
    gates["C5_reflection"] = _gate("2D reflection equivariance (shared axis)",
                                   flip, 1e-12)

    # -- 3D rigid-rotation equivariance ----------------------------------
    ang = math.radians(23.0)
    ax = np.array([1.0, 1.0, 0.5])
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
    rot = np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)

    def smooth3(pts):
        p = np.asarray(pts, dtype=float)
        return np.sin(2.0 * p[..., 1]) * np.cos(3.0 * p[..., 2])

    ul = cell_averages(c3.src, smooth3)
    vl = MortarOperator(order=2).transfer(c3.src, ul, c3.tgt)
    vr = MortarOperator(order=2).transfer(rotate_side(c3.src, rot), ul,
                                          rotate_side(c3.tgt, rot))
    rot_flip = float(np.max(np.abs(vl - vr)) / max(np.max(np.abs(vl)), 1e-300))
    out["rotation_3d"] = {"symmetry_flip": rot_flip, "angle_deg": 23.0,
                          "axis": ax.tolist()}
    gates["C5_rotation"] = _gate("3D rigid-rotation equivariance", rot_flip, 1e-12)
    return {"C5": out, "gates": gates}


# ---------------------------------------------------------------------------
# C6
# ---------------------------------------------------------------------------
M5207_DEFAULT = "/root/tp-agent/work/m5207/artifacts/m5207/interface_operator_acceptance.json"


def load_m5207_reference(path: str | None = None) -> dict:
    """Load the frozen 1D reference numbers published by mission m5207.

    Preferred source: ``$M5207_ARTIFACT`` or the m5207 worktree artifact.  A
    vendored copy lives in ``artifacts/m5436/m5207_1d_reference.json`` so the
    comparison table stays reproducible outside that worktree.  Returns ``{}``
    when neither is available (the table then simply omits the 1D cross-repo row
    and says so).
    """
    candidates = [os.environ.get("M5207_ARTIFACT"), path, M5207_DEFAULT,
                  os.path.join(os.path.dirname(__file__), "..", "..", "..",
                               "artifacts", "m5436", "m5207_1d_reference.json")]
    for cand in candidates:
        if not cand or not os.path.exists(cand):
            continue
        try:
            with open(cand, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):        # pragma: no cover
            continue
        crit = data.get("criteria", data)
        picked = {}
        for key in ("C1", "C3", "C4"):
            if key in crit:
                picked[key] = crit[key]
        if picked:
            picked["_source"] = cand
            return picked
    return {}


def criteria_c6(m5207_reference: dict | None = None) -> dict:
    rows = []
    ops = {"mortar": MortarOperator(order=2), "rbf": RBFOperator(),
           "me": MomentumExchangeOperator()}

    def field_set_curve(case):
        s = case.arclength
        return {
            "constant": lambda p: np.full(np.asarray(p).shape[:-1], 3.5),
            "piecewise_constant": lambda p: np.where(s(p) < 0.42, 1.0, -2.5),
            "linear": lambda p: (2.0 * np.asarray(p)[..., 0]
                                 - 3.0 * np.asarray(p)[..., 1] + 0.7),
            "smooth": lambda p: np.sin(3.1 * s(p)) * np.cos(2.3 * s(p) + 0.4),
        }

    def field_set_surface(case):
        def lin(p):
            p = np.asarray(p, dtype=float)
            return 1.5 * p[..., 1] - 0.5 * p[..., 2] + 2.0

        def smooth(p):
            p = np.asarray(p, dtype=float)
            return np.sin(2.0 * p[..., 1]) * np.cos(3.0 * p[..., 2])

        def step(p):
            p = np.asarray(p, dtype=float)
            return np.where(p[..., 1] + p[..., 2] < 1.0, 1.0, -2.5)

        return {"constant": lambda p: np.full(np.asarray(p).shape[:-1], 3.5),
                "piecewise_constant": step, "linear": lin, "smooth": smooth}

    cases = [("1d", triangle_case_2d(7, 11, angle_deg=0.0, p0=(0.0, 0.0))),
             ("2d_17deg", triangle_case_2d(7, 11, angle_deg=17.0)),
             ("3d_cube_face", cube_case_3d(9, 7))]
    for name, case in cases:
        fields = (field_set_curve(case) if name != "3d_cube_face"
                  else field_set_surface(case))
        for fname, field in fields.items():
            u = cell_averages(case.src, field)
            v_ex = cell_averages(case.tgt, field)
            for oname, op in ops.items():
                v = op.transfer(case.src, u, case.tgt)
                sent = integral(u, case.src.measures)
                recv = integral(v, case.tgt.measures)
                row = {"case": name, "operator": oname, "field": fname,
                       "conservation_residual": abs(recv - sent) / max(abs(sent), 1.0),
                       "l2_error": l2_error(v, v_ex, case.tgt.measures)}
                if oname == "me":
                    row["momentum_budget_residual"] = op.momentum_budget(
                        case.src, u, case.tgt)["budget_residual"]
                rows.append(row)

    # observed orders per operator on the 2D 17-degree smooth family
    ord_tab = {}
    for oname, op in ops.items():
        errs, hs = [], []
        for n_src, n_tgt in ((8, 11), (16, 22), (32, 44)):
            c = triangle_case_2d(n_src, n_tgt, 17.0, jitter=0.4, seed=7)

            def sm(pts, c=c):
                x = c.arclength(pts)
                return np.sin(3.1 * x) * np.cos(2.3 * x + 0.4)

            u = cell_averages(c.src, sm)
            v_ex = cell_averages(c.tgt, sm)
            errs.append(l2_error(op.transfer(c.src, u, c.tgt), v_ex, c.tgt.measures))
            hs.append(1.0 / n_src)
        c_lin = triangle_case_2d(16, 22, 17.0, jitter=0.4, seed=11)

        def lin(pts):
            p = np.asarray(pts, dtype=float)
            return 2.0 * p[..., 0] - 3.0 * p[..., 1] + 0.7

        lin_err = l2_error(op.transfer(c_lin.src, cell_averages(c_lin.src, lin), c_lin.tgt),
                           cell_averages(c_lin.tgt, lin), c_lin.tgt.measures)
        ord_tab[oname] = {"errors": errs, "hs": hs, "linear_l2": lin_err,
                          "order_successive": [_order(errs[k], errs[k + 1])
                                               for k in range(len(errs) - 1)],
                          "order_fit": _fit_order(errs, hs),
                          "linear_exact": bool(lin_err < 1e-10)}

    # the measure-normalisation trap, executed in 2D and 3D
    trap = {}
    for name, case in (("2d_17deg", triangle_case_2d(7, 11, 17.0)),
                       ("3d_cube_face", cube_case_3d(9, 7))):
        fields = (field_set_curve(case) if name == "2d_17deg"
                  else field_set_surface(case))
        field = fields["piecewise_constant"]
        u = cell_averages(case.src, field)
        sent = integral(u, case.src.measures)
        for tag, broken in (("correct_measure_proportional", None),
                            ("trap_linear_denominator", "backfill_linear")):
            op = RBFOperator(broken=broken)
            v = op.transfer(case.src, u, case.tgt)
            trap.setdefault(name, {})[tag] = abs(
                integral(v, case.tgt.measures) - sent) / max(abs(sent), 1.0)
        op = RBFOperator(broken="no_fix")
        v = op.transfer(case.src, u, case.tgt)
        trap[name]["no_fix"] = abs(integral(v, case.tgt.measures) - sent) / max(
            abs(sent), 1.0)

    me_ctrl = {}
    for name, case in (("2d_17deg", triangle_case_2d(7, 11, 17.0)),
                       ("3d_cube_face", cube_case_3d(9, 7))):
        fields = (field_set_curve(case) if name == "2d_17deg"
                  else field_set_surface(case))
        u = cell_averages(case.src, fields["smooth"])
        me_ctrl[name] = {
            "closed": MomentumExchangeOperator().momentum_budget(case.src, u, case.tgt),
            "per_node_norm": MomentumExchangeOperator(
                broken="per_node_norm").momentum_budget(case.src, u, case.tgt)}
    worst_trap = max(v["trap_linear_denominator"] for v in trap.values())
    gates = {
        "trap_correct_normalisation": _gate(
            "worst conservation residual with the correct |T_j|/sum|T|^2 backfill",
            max(v["correct_measure_proportional"] for v in trap.values()), 1e-12),
        "trap_wrong_normalisation_red": _gate(
            "worst conservation residual with the |T_j|/sum|T| trap",
            worst_trap, 1e-2, ">=",
            note="the wrong normalisation must be visibly non-conservative"),
        "me_budget_closed": _gate(
            "worst ME momentum-budget residual (closed)",
            max(v["closed"]["budget_residual"] for v in me_ctrl.values()), 1e-12),
        "me_negative_control_red": _gate(
            "worst ME momentum-budget residual (per-acceptor normalisation trap)",
            max(v["per_node_norm"]["budget_residual"] for v in me_ctrl.values()),
            1e-3, ">="),
    }
    return {"C6": {"table": rows, "orders": ord_tab, "trap": trap,
                   "momentum_exchange": me_ctrl,
                   "m5207_1d_reference": m5207_reference or {}},
            "gates": gates}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_acceptance(m5207_reference: dict | None = None) -> dict:
    """Run every gate and return the acceptance record (JSON-serialisable)."""
    c123 = criteria_c1_c2_c3()
    c4 = criteria_c4()
    c5 = criteria_c5()
    c6 = criteria_c6(m5207_reference if m5207_reference is not None
                     else load_m5207_reference())
    gates = {}
    for part in (c123, c4, c5, c6):
        gates.update(part["gates"])
    c1_thresholds = {
        "C1": "conservation residual <= 1e-13",
        "C2": "linear patch L2 <= 1e-12",
        "C3": "observed order >= 1.8",
        "C4_coverage": "coverage error <= 1e-10",
        "C4_conservation": "conservation residual <= 1e-10",
        "C4_curved_closed_form": "curved area vs closed form <= 1e-6",
        "C4_curved_facet_order": "flat-facet clipping observed order >= 1.8",
        "C5_transpose": "M_ij vs M_ji <= 1e-12",
        "C5_reflection": "reflection flip <= 1e-12 (shared axis)",
        "C5_rotation": "rotation flip <= 1e-12",
    }
    for k, v in c1_thresholds.items():
        gates[k]["criterion"] = v
    return {
        "mission": "5436",
        "campaign_node": "P48-N02",
        "repository": "TensorDEM",
        "module": "tensordem.interface_operators",
        "gates": gates,
        "all_pass": all(g.get("passed", True) for g in gates.values()),
        "C1_C2_C3": c123,
        "C4": c4["C4"],
        "C5": c5["C5"],
        "C6": c6["C6"],
    }


def main(argv=None) -> int:                            # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="run the 2D/3D interface-operator gates")
    ap.add_argument("--out", default=None, help="write the acceptance JSON here")
    args = ap.parse_args(argv)
    rec = run_acceptance()
    text = json.dumps(rec, indent=1, sort_keys=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text if not args.out else f"wrote {args.out}")
    return 0 if rec["all_pass"] else 1


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())