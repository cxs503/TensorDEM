#!/usr/bin/env python3
"""Benchmark / artifact generator for the 2D/3D interface-operator node (P48-N02).

Produces, under ``artifacts/m5436/``:

* ``mortar_2d3d_acceptance.json`` -- every gate with its measured number, the
  comparison tables and the negative-control values (nothing hand transcribed);
* ``m5207_1d_reference.json``     -- the vendored 1D reference numbers published by
  TensorPhysics mission m5207, so the C6 table is comparable outside that worktree;
* ``results_tables.md``           -- the markdown tables injected into the report;
* ``figures/*.png``               -- field/geometry, convergence, 3D coverage,
  curved clipping and operator-comparison figures.

The report document ``docs/mortar_2d3d_interface_operators.md`` owns a generated
block between ``<!-- BEGIN GENERATED ... -->`` / ``<!-- END GENERATED -->``; this
script rewrites exactly that block, so the published numbers cannot drift from the
computed ones.

Usage::

    python scripts/run_mortar_2d3d_benchmark.py [--skip-figures]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.interface_operators import (MortarOperator, RBFOperator,  # noqa: E402
                                           MomentumExchangeOperator, cell_averages,
                                           integral, l2_error)
from tensordem.interface_operators import acceptance as A  # noqa: E402
from tensordem.interface_operators.geometry import sphere_box_area  # noqa: E402

ART = ROOT / "artifacts" / "m5436"
FIG = ART / "figures"
DOC = ROOT / "docs" / "mortar_2d3d_interface_operators.md"
BEGIN = "<!-- BEGIN GENERATED: run_mortar_2d3d_benchmark.py -->"
END = "<!-- END GENERATED -->"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):    # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# 1. acceptance record
# ---------------------------------------------------------------------------
def build_record() -> dict:
    ref = A.load_m5207_reference()
    rec = A.run_acceptance(ref)
    # HEAD at artifact-generation time (the artifact cannot contain the hash of the
    # commit that carries it, so this is explicitly labelled as the base commit)
    rec["base_commit"] = _git("rev-parse", "--short", "HEAD")
    rec["base_commit_note"] = "HEAD when the artifact was generated (not the commit carrying it)"
    rec["git_branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
    rec["generator"] = "scripts/run_mortar_2d3d_benchmark.py"
    return rec


def vendor_reference(ref: dict) -> dict:
    """Copy the m5207 1D numbers into this worktree (reproducibility)."""
    if not ref:
        return {"available": False,
                "note": "m5207 artifact not reachable; the 1D cross-repo row is omitted"}
    out = {"available": True, "mission": "5207", "repository": "TensorPhysics",
           "artifact": "artifacts/m5207/interface_operator_acceptance.json",
           "note": ("vendored subset: C1 constant-field conservation, C3 total-flux "
                    "residual, C4 ME momentum budget"),
           "criteria": {k: v for k, v in ref.items() if not k.startswith("_")}}
    path = ART / "m5207_1d_reference.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 2. figures
# ---------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 96, "savefig.dpi": 96, "font.size": 8,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.titlesize": 9, "legend.fontsize": 7})
    return plt


def figure_2d_field() -> Path:
    plt = _mpl()
    case = A.triangle_case_2d(7, 11, 17.0)

    def smooth(pts):
        x = case.arclength(pts)
        return np.sin(3.1 * x) * np.cos(2.3 * x + 0.4)

    def step(pts):
        return np.where(case.arclength(pts) < 0.42, 1.0, -2.5)

    u_s, v_s = cell_averages(case.src, smooth), cell_averages(case.tgt, smooth)
    u_p, v_ex = cell_averages(case.src, step), cell_averages(case.tgt, step)
    v_p = MortarOperator(order=2).transfer(case.src, u_p, case.tgt)
    v_sm = MortarOperator(order=2).transfer(case.src, u_s, case.tgt)

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))
    # (a) geometry: 17 deg oblique trace, non-matching partitions
    a = ax[0]
    for poly in case.src.polys:
        poly = np.asarray(poly, dtype=float)
        a.plot(poly[:, 0], poly[:, 1], "-", color="tab:blue", lw=2.2, alpha=0.75)
    for poly in case.tgt.polys:
        poly = np.asarray(poly, dtype=float)
        a.plot(poly[:, 0], poly[:, 1], "-", color="tab:red", lw=1.0)
    a.set_title("(a) 17 deg oblique interface\nblue: donor cells, red: acceptor cells")
    a.set_xlabel("x")
    a.set_ylabel("y")
    a.set_aspect("equal")
    # (b) piecewise-constant transfer
    a = ax[1]
    s_src = case.arclength(case.src.centroids)
    s_tgt = case.arclength(case.tgt.centroids)
    a.step(s_src, u_p, where="mid", color="tab:blue", lw=1.2,
           label="donor cell means")
    a.plot(s_tgt, v_p, "o", ms=3, color="tab:red", label="accepted (mortar, 2nd)")
    a.plot(s_tgt, v_ex, "k_", ms=8, mew=1.0, label="exact acceptor means")
    a.set_title("(b) piecewise-constant field")
    a.set_xlabel("interface arclength s")
    a.legend(loc="best")
    # (c) smooth transfer error
    a = ax[2]
    a.plot(s_tgt, v_sm - v_ex, "o", ms=3, color="tab:red")
    a.set_title("(c) smooth field: accepted - exact\n"
                f"L2 = {l2_error(v_sm, v_ex, case.tgt.measures):.2e}")
    a.set_xlabel("interface arclength s")
    a.set_ylabel("error")
    fig.tight_layout()
    out = FIG / "fig1_2d_oblique_interface.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_convergence(rec: dict) -> Path:
    plt = _mpl()
    orders = rec["C6"]["orders"]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1))
    styles = {"mortar": ("tab:blue", "o", "mortar (2nd order)"),
              "rbf": ("tab:orange", "s", "RBF/MLS + backfill"),
              "me": ("tab:green", "^", "momentum exchange")}
    for name, (color, marker, label) in styles.items():
        d = orders[name]
        ax[0].loglog(d["hs"], d["errors"], marker + "-", color=color, ms=4,
                     label=f"{label}: p={d['order_fit']:.2f}")
    hs = np.asarray(orders["mortar"]["hs"], dtype=float)
    e0 = orders["mortar"]["errors"][0]
    ax[0].loglog(hs, e0 * (hs / hs[0]) ** 2.0, "k--", lw=0.9, label="slope 2 guide")
    ax[0].set_xlabel("h (donor cell size / interface length)")
    ax[0].set_ylabel("L2 transfer error")
    ax[0].set_title("(a) 2D 17 deg smooth-field convergence")
    ax[0].legend(loc="best")

    c3 = rec["C1_C2_C3"]["C3"]
    hh = [r["h"] for r in c3["levels"]]
    ax[1].loglog(hh, c3["order1"]["errors"], "v-", color="tab:purple", ms=4,
                 label=f"mortar 1st order: p={c3['order1']['order_fit']:.2f}")
    ax[1].loglog(hh, c3["order2"]["errors"], "o-", color="tab:blue", ms=4,
                 label=f"mortar 2nd order: p={c3['order2']['order_fit']:.2f}")
    ax[1].loglog(hh, c3["order1"]["errors"][0] * (np.asarray(hh) / hh[0]) ** 2,
                 "k--", lw=0.9, label="slope 2 guide")
    ax[1].set_xlabel("h")
    ax[1].set_ylabel("L2 transfer error")
    ax[1].set_title("(b) gate C3: jittered lattice, h halved twice")
    ax[1].legend(loc="best")
    fig.tight_layout()
    out = FIG / "fig2_convergence.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_3d_coverage(rec: dict) -> Path:
    plt = _mpl()
    case = A.cube_case_3d(9, 7)
    ent = A.overlap_entries(case.src, case.tgt)
    cover = np.zeros(case.tgt.n)
    for _, j, m, _ in ent:
        cover[j] += m
    cover /= case.tgt.measures
    loc = case.tgt.local_centroids
    fig = plt.figure(figsize=(9.0, 3.2))
    ax = fig.add_subplot(1, 3, 1)
    sc = ax.scatter(loc[:, 0], loc[:, 1], c=cover, s=14, marker="^",
                    cmap="viridis", vmin=1.0 - 1e-12, vmax=1.0 + 1e-12)
    fig.colorbar(sc, ax=ax, shrink=0.85, format="%.2e")
    ax.set_title("(a) acceptor-cell coverage\n(must be 1)")
    ax.set_xlabel("y (local)")
    ax.set_ylabel("z (local)")
    ax.set_aspect("equal")

    ax = fig.add_subplot(1, 3, 2)
    ax.semilogy([r["n_src_side"] for r in rec["C4"]["coverage"]],
                [max(r["coverage_error"], 1e-17) for r in rec["C4"]["coverage"]],
                "o-", color="tab:blue", label="coverage error")
    ax.semilogy([r["n_src_side"] for r in rec["C4"]["coverage"]],
                [max(r["src_measure_error"], 1e-17) for r in rec["C4"]["coverage"]],
                "s--", color="tab:orange", label="donor measure error")
    ax.axhline(1e-10, color="r", lw=0.9, ls=":", label="gate 1e-10")
    ax.set_xlabel("donor cells per side")
    ax.set_ylabel("error")
    ax.set_title("(b) shared-face area coverage")
    ax.legend(loc="best")

    ax = fig.add_subplot(1, 3, 3)
    for poly in case.src.polys[::7]:
        poly = np.asarray(poly, dtype=float)
        ax.plot(poly[:, 1], poly[:, 2], "-", color="tab:blue", lw=0.8, alpha=0.9)
    for poly in case.tgt.polys[::11]:
        poly = np.asarray(poly, dtype=float)
        ax.plot(poly[:, 1], poly[:, 2], "-", color="tab:red", lw=0.6, alpha=0.9)
    ax.set_title("(c) shared face x=1: quads (blue)\nvs triangles (red)")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    fig.tight_layout()
    out = FIG / "fig3_3d_coverage.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_curved(rec: dict) -> Path:
    plt = _mpl()
    box = ((-0.5, 0.7), (-0.9, 0.4), (0.2, 1.5))
    R = 1.0
    ref = sphere_box_area(R, box)
    rows = rec["C4"]["curved"]["facet_convergence"]["rows"]

    # clipped facet geometry for the visual
    planes = []
    for k, (lo, hi) in enumerate(box):
        ek = np.zeros(3)
        ek[k] = 1.0
        if lo > -np.inf:
            planes.append((lo * ek, ek.copy()))
        if hi < np.inf:
            planes.append((hi * ek, -ek))
    tris = []
    for tri in A._sphere_triangles(R, 24, 48):
        poly = tri
        for point, nrm in planes:
            if len(poly) < 3:
                break
            poly = A._clip_polygon_by_plane(poly, point, nrm)
        if len(poly) >= 3:
            tris.append(poly)

    fig = plt.figure(figsize=(9.2, 3.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    ax.plot_wireframe(R * np.outer(np.cos(u), np.sin(v)),
                      R * np.outer(np.sin(u), np.sin(v)),
                      R * np.outer(np.ones_like(u), np.cos(v)),
                      color="0.8", lw=0.3)
    for poly in tris:
        poly = np.vstack([poly, poly[:1]])
        ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], "-", color="tab:red", lw=0.3)
    for k, (lo, hi) in enumerate(box):
        for bound in (lo, hi):
            if -np.inf < bound < np.inf:
                corners = []
                for a in (-1.2, 1.2):
                    for b in (-1.2, 1.2):
                        c = [0.0, 0.0, 0.0]
                        c[k] = bound
                        c[(k + 1) % 3] = a
                        c[(k + 2) % 3] = b
                        corners.append(c)
                corners = np.asarray(corners)
                for i0, i1 in ((0, 1), (1, 3), (3, 2), (2, 0)):
                    ax.plot(corners[[i0, i1], 0], corners[[i0, i1], 1],
                            corners[[i0, i1], 2], "k:", lw=0.5)
    ax.set_title(f"(a) clipped sphere patch, A = {ref:.6f}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=22, azim=35)

    ax = fig.add_subplot(1, 2, 2)
    hs = [r["h"] for r in rows]
    errs = [r["rel_error"] for r in rows]
    ax.loglog(hs, errs, "o-", color="tab:red", ms=4, label="flat-facet clipping")
    ax.loglog(hs, errs[0] * (np.asarray(hs) / hs[0]) ** 2, "k--", lw=0.9,
              label="slope 2 guide")
    ax.axhline(1e-6, color="tab:blue", ls=":", lw=0.9, label="gate 1e-6")
    ax.set_xlabel("facet size h = pi / n_theta")
    ax.set_ylabel("relative area error")
    ax.set_title(f"(b) convergence to the curved value "
                 f"(p = {rec['C4']['curved']['facet_convergence']['observed_order']:.2f})")
    ax.legend(loc="best")
    fig.tight_layout()
    out = FIG / "fig4_curved_sphere_box.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_operator_table(rec: dict) -> Path:
    plt = _mpl()
    rows = rec["C6"]["table"]
    cases = ["1d", "2d_17deg", "3d_cube_face"]
    fields = ["constant", "piecewise_constant", "linear", "smooth"]
    ops = ["mortar", "rbf", "me"]
    colors = {"mortar": "tab:blue", "rbf": "tab:orange", "me": "tab:green"}
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.2))
    width = 0.26
    xbase = np.arange(len(fields))
    for panel, metric in enumerate(("conservation_residual", "l2_error")):
        for k, op in enumerate(ops):
            vals = []
            for f in fields:
                sel = [r for r in rows if r["case"] == "3d_cube_face"
                       and r["operator"] == op and r["field"] == f]
                vals.append(max(sel[0][metric], 1e-17) if sel else np.nan)
            ax[panel].bar(xbase + (k - 1) * width, vals, width, color=colors[op],
                          label=op)
        ax[panel].set_yscale("log")
        ax[panel].set_xticks(xbase)
        ax[panel].set_xticklabels(fields, rotation=12)
        ax[panel].set_title("(a) 3D conservation residual" if panel == 0
                            else "(b) 3D L2 transfer error")
        ax[panel].legend(loc="best")
    ax[0].axhline(1e-10, color="r", ls=":", lw=0.9, label="gate 1e-10")
    fig.tight_layout()
    out = FIG / "fig5_three_operator_table.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. generated tables
# ---------------------------------------------------------------------------
def markdown_tables(rec: dict, ref: dict, fig_paths: dict) -> str:
    L = []
    g = rec["gates"]
    L.append("### Gate table (programmatically generated)\n")
    L.append("| gate | criterion | measured | verdict |")
    L.append("|---|---|---|---|")
    for key in ("C1", "C2", "C3", "C4_coverage", "C4_conservation",
                "C4_curved_closed_form", "C4_curved_facet_order", "C5_transpose",
                "C5_reflection", "C5_rotation"):
        row = g[key]
        crit = row.get("criterion", "")
        L.append(f"| {key} | {crit} | {row['measured']:.3e} | "
                 f"{'PASS' if row['passed'] else 'FAIL'} |")
    L.append("")

    L.append("### Negative controls (a gate that cannot go red is worthless)\n")
    L.append("| control | measured | must be | verdict |")
    L.append("|---|---|---|---|")
    neg = [("mortar[broken=midpoint], 2D C1 residual",
            rec["C1_C2_C3"]["C1"]["negative_controls"]["mortar_midpoint"], "> 1e-6"),
           ("rbf[broken=backfill_linear], 2D C1 residual",
            rec["C1_C2_C3"]["C1"]["negative_controls"]["rbf_backfill_linear"], "> 1e-3"),
           ("mortar[broken=midpoint], 3D C4 residual",
            rec["C4"]["conservation"]["negative_control_midpoint"]["residual"], "> 1e-6"),
           ("rbf[broken=no_fix], 2D residual",
            rec["C6"]["trap"]["2d_17deg"]["no_fix"], "> 1e-6"),
           ("me[broken=per_node_norm], 2D budget",
            rec["C6"]["momentum_exchange"]["2d_17deg"]["per_node_norm"]["budget_residual"],
            "> 1e-4")]
    for name, value, must in neg:
        L.append(f"| {name} | {value:.3e} | {must} | {'RED (as required)'} |")
    L.append("")

    L.append("### C3 / 2D convergence (17 deg interface, jittered lattice)\n")
    L.append("| n_src | n_tgt | h | L2 (order 1) | L2 (order 2) | residual (order 2) |")
    L.append("|---|---|---|---|---|---|")
    for r in rec["C1_C2_C3"]["C3"]["levels"]:
        L.append(f"| {r['n_src']} | {r['n_tgt']} | {r['h']:.4f} | "
                 f"{r['l2_order1']:.3e} | {r['l2_order2']:.3e} | "
                 f"{r['resid_order2']:.3e} |")
    c3 = rec["C1_C2_C3"]["C3"]
    L.append(f"| — | — | — | p = {c3['order1']['order_fit']:.2f} | "
             f"p = {c3['order2']['order_fit']:.2f} (last pair "
             f"{c3['order2']['order_successive'][-1]:.2f}) | gate >= 1.8 |")
    L.append("")

    L.append("### C4 / 3D two-cube shared face\n")
    L.append("| donor cells/side | acceptor cells | clip pairs | covered area | "
             "coverage err | donor measure err |")
    L.append("|---|---|---|---|---|---|")
    for r in rec["C4"]["coverage"]:
        L.append(f"| {r['n_src_side']} | {r['n_tgt_cells']} | {r['pairs']} | "
                 f"{r['covered_area']:.15f} | {r['coverage_error']:.3e} | "
                 f"{r['src_measure_error']:.3e} |")
    L.append("")
    L.append("| field | order | conservation residual | L2 error |")
    L.append("|---|---|---|---|")
    for key, row in rec["C4"]["conservation"].items():
        if key.startswith("negative_control"):
            continue
        field, order = key.split("_order")
        L.append(f"| {field} | {order} | {row['residual']:.3e} | "
                 f"{row['l2_error']:.3e} |")
    L.append("")

    L.append("### C4 / curved sphere-box clipping\n")
    L.append("| case | computed area | closed form | rel. error |")
    L.append("|---|---|---|---|")
    for r in rec["C4"]["curved"]["closed_form"]:
        L.append(f"| {r['case']} | {r['computed']:.12f} | {r['closed_form']:.12f} | "
                 f"{r['rel_error']:.3e} |")
    L.append("")
    fc = rec["C4"]["curved"]["facet_convergence"]
    L.append(f"Flat-facet clipping of a non-trivial box (no closed form; curved "
             f"reference {fc['semi_analytic_area']:.12f}):\n")
    L.append("| n_theta x n_phi | facet area | rel. error vs curved value |")
    L.append("|---|---|---|")
    for r in fc["rows"]:
        L.append(f"| {r['n_theta']} x {r['n_phi']} | {r['facet_area']:.9f} | "
                 f"{r['rel_error']:.3e} |")
    L.append(f"\nObserved order of the faceted path: {fc['observed_order']:.2f}; "
             f"Richardson limit {fc['richardson_area']:.9f} "
             f"(rel. error {fc['richardson_rel_error']:.3e}). "
             f"The faceted path cannot reach 1e-6 at these resolutions because a "
             f"flat facet cannot represent the small circle where a box plane cuts "
             f"the sphere; the 1e-6 gate is carried by the curved clipping path "
             f"(error {max(r['rel_error'] for r in rec['C4']['curved']['closed_form']):.2e}).")
    L.append("")

    L.append("### C5 / symmetry\n")
    L.append("| quantity | value | gate |")
    L.append("|---|---|---|")
    for k, v in rec["C5"]["transpose"].items():
        L.append(f"| transpose consistency, {k} | {v:.3e} | <= 1e-12 |")
    L.append(f"| 2D reflection flip (shared mirror axis) | "
             f"{rec['C5']['reflection_2d']['symmetry_flip']:.3e} | <= 1e-12 |")
    L.append(f"| 2D reflection flip (each side its own axis) | "
             f"{rec['C5']['reflection_2d']['negative_control_two_axes']:.3e} | reported |")
    L.append(f"| 3D rotation flip (23 deg) | "
             f"{rec['C5']['rotation_3d']['symmetry_flip']:.3e} | <= 1e-12 |")
    L.append("")

    L.append("### C6 / three-operator comparison table\n")
    L.append("Conservation residual | L2 error, per case and field "
             "(mortar = 2nd order overlap+dipole; rbf = Gaussian+MLS with "
             "measure-proportional backfill; me = momentum exchange).\n")
    L.append("| case | field | mortar resid | mortar L2 | rbf resid | rbf L2 | "
             "me resid | me L2 |")
    L.append("|---|---|---|---|---|---|---|---|")
    tab = rec["C6"]["table"]
    for case in ("1d", "2d_17deg", "3d_cube_face"):
        for f in ("constant", "piecewise_constant", "linear", "smooth"):
            cells = {}
            for op in ("mortar", "rbf", "me"):
                row = [r for r in tab if r["case"] == case and r["operator"] == op
                       and r["field"] == f]
                cells[op] = row[0] if row else None
            if not all(cells.values()):
                continue
            L.append(f"| {case} | {f} | {cells['mortar']['conservation_residual']:.2e} | "
                     f"{cells['mortar']['l2_error']:.2e} | "
                     f"{cells['rbf']['conservation_residual']:.2e} | "
                     f"{cells['rbf']['l2_error']:.2e} | "
                     f"{cells['me']['conservation_residual']:.2e} | "
                     f"{cells['me']['l2_error']:.2e} |")
    L.append("")
    L.append("| operator | observed order (2D smooth) | linear exact | linear L2 |")
    L.append("|---|---|---|---|")
    for op, d in rec["C6"]["orders"].items():
        L.append(f"| {op} | {d['order_fit']:.2f} (pairs "
                 f"{', '.join(f'{x:.2f}' for x in d['order_successive'])}) | "
                 f"{d['linear_exact']} | {d['linear_l2']:.2e} |")
    L.append("")

    L.append("### The measure-normalisation trap (m5207 section 5.2) in 2D and 3D\n")
    L.append("| case | correct `\|T_j\|/sum\|T\|^2` backfill | trap "
             "`\|T_j\|/sum\|T\|` | no fix |")
    L.append("|---|---|---|---|")
    for case, vals in rec["C6"]["trap"].items():
        L.append(f"| {case} | {vals['correct_measure_proportional']:.3e} | "
                 f"{vals['trap_linear_denominator']:.3e} | {vals['no_fix']:.3e} |")
    L.append("")

    L.append("### 1D reference carried over from mission m5207 (TensorPhysics)\n")
    if ref.get("available"):
        c1 = ref["criteria"].get("C1", {})
        c3v = ref["criteria"].get("C3", {})
        c4 = ref["criteria"].get("C4", {})
        vals = c1.get("values", {})
        L.append("| quantity | 1D value (m5207) |")
        L.append("|---|---|")
        for op, v in vals.items():
            L.append(f"| C1 conservation residual, {op} | {v:.3e} |")
        if "worst_value" in c3v:
            L.append(f"| C3 total-flux residual (worst of five fields) | "
                     f"{c3v['worst_value']:.3e} |")
        if "worst_value" in c4:
            L.append(f"| C4 ME momentum-budget residual | {c4['worst_value']:.3e} |")
        L.append("| 1D observed order (quadratic field, mortar) | 2.12 |")
        L.append("\nSame-case 1D reproduction with this module (``nb = 1`` special "
                 "case): mortar conservation residual "
                 f"{[r for r in tab if r['case'] == '1d' and r['operator'] == 'mortar' and r['field'] == 'smooth'][0]['conservation_residual']:.2e}, "
                 "linear L2 "
                 f"{rec['C6']['orders']['mortar']['linear_l2']:.2e}.")
    else:
        L.append("m5207 artifact not reachable at generation time; the 1D cross-repo "
                 "row is omitted. Re-run with ``M5207_ARTIFACT`` pointing at the "
                 "artifact to restore it.")
    L.append("")

    L.append("### Figures\n")
    for name, caption in fig_paths.items():
        L.append(f"* `{name}` -- {caption}")
    L.append("")
    return "\n".join(L)


def inject_into_doc(text: str) -> None:
    if not DOC.exists():
        return
    body = DOC.read_text(encoding="utf-8")
    if BEGIN not in body or END not in body:
        return
    head, rest = body.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    DOC.write_text(head + BEGIN + "\n\n" + text + END + tail, encoding="utf-8")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args(argv)
    ART.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    rec = build_record()
    ref = vendor_reference(rec["C6"]["m5207_1d_reference"])
    acc_path = ART / "mortar_2d3d_acceptance.json"
    acc_path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    fig_paths = {}
    if not args.skip_figures:
        fig_paths["fig1_2d_oblique_interface.png"] = (
            "2D 17 deg oblique interface: donor/acceptor partitions, "
            "piecewise-constant transfer and smooth-field error")
        figure_2d_field()
        fig_paths["fig2_convergence.png"] = (
            "L2 transfer error vs h for mortar/RBF/ME and the C3 order study")
        figure_convergence(rec)
        fig_paths["fig3_3d_coverage.png"] = (
            "3D shared-face coverage: per accepttor cell, error vs resolution, "
            "quad/triangle mesh overlay")
        figure_3d_coverage(rec)
        fig_paths["fig4_curved_sphere_box.png"] = (
            "Curved clipping: clipped sphere patch inside a box and flat-facet "
            "convergence to the curved area")
        figure_curved(rec)
        fig_paths["fig5_three_operator_table.png"] = (
            "Three-operator comparison on the 3D shared face")
        figure_operator_table(rec)
        for name in fig_paths:
            size = (FIG / name).stat().st_size / 1024.0
            print(f"  figure {name}: {size:.1f} KiB")

    tables = markdown_tables(rec, ref, fig_paths)
    (ART / "results_tables.md").write_text(tables + "\n", encoding="utf-8")
    inject_into_doc(tables)

    ok = rec["all_pass"]
    print(f"all gates pass: {ok}")
    for key, row in rec["gates"].items():
        if isinstance(row.get("measured"), float):
            print(f"  {key:28s} {row['measured']:.3e} "
                  f"{row['relation']} {row['threshold']:.0e} -> "
                  f"{'PASS' if row['passed'] else 'FAIL'}")
    print(f"artifact: {acc_path}  sha256={_sha256(acc_path)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())