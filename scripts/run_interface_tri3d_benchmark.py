#!/usr/bin/env python3
"""Benchmark + figures for the 3D triangle-triangle (non-coplanar) mortar node.

Campaign node IN52-N01 (mission 6114).  Produces

    artifacts/in52_n01/interface_tri3d_acceptance.json    machine gates D1-D6
    artifacts/in52_n01/results_tables.md                  same numbers as tables
    artifacts/in52_n01/figures/fig1..fig5.png             report figures

Run::

    PYTHONPATH=src python3 scripts/run_interface_tri3d_benchmark.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.interface_operators.acceptance3d import (CURVED_LEVELS, OCTANT_BOX,
                                                        curved_octant_case,
                                                        kinked_case, planar_tri_case,
                                                        run_acceptance)
from tensordem.interface_operators.clip3d import (SurfaceMortarOperator, TriSide,
                                                  clip_tris_by_box, octant_reference,
                                                  surface_cell_averages,
                                                  surface_integral,
                                                  surface_overlap_entries,
                                                  tri_patch_side, uv_sphere_tris,
                                                  ico_sphere_tris)

OUT = os.path.join("artifacts", "in52_n01")
FIGS = os.path.join(OUT, "figures")


def _order(e1: float, e2: float) -> float:
    return math.log(e1 / e2) / math.log(2.0) if e2 > 0.0 else float("nan")


def figures(record: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGS, exist_ok=True)
    made = []

    # -- fig1: planar tri-tri convergence of coverage / conservation ----------
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8))
    cases = record["D1"]["cases"]
    h = [c["h"] for c in cases]
    ax[0].semilogy(h, [c["coverage_src_abs"] for c in cases], "o-", label="coverage (donor)")
    ax[0].semilogy(h, [c["coverage_tgt_abs"] for c in cases], "s-", label="coverage (acceptor)")
    ax[0].semilogy(h, [max(o["residual"] for o in c["orders"].values())
                       for c in cases], "^-", label="conservation")
    ax[0].set_xlabel("h (patch cells / side)")
    ax[0].set_ylabel("relative defect")
    ax[0].set_title("3D tri-tri, 17$^\\circ$ rotated facets: exactness")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    pt = record["D2"]["cases"]
    ax[1].semilogy(h, [c["l2_order1"] for c in pt], "o-", label="order 1 (piecewise const.)")
    ax[1].semilogy(h, [c["l2_order2"] for c in pt], "s-", label="order 2 (with slope term)")
    ax[1].set_xlabel("h")
    ax[1].set_ylabel("linear patch-test L2")
    ax[1].set_title("linear patch test: only order 2 is exact")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGS, "fig1_planar_tri_tri.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    # -- fig2: curved octant convergence to closed form -----------------------
    lv = record["D3"]["levels"]
    hs = np.array([r["h"] for r in lv])
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8))
    ax[0].loglog(hs, [abs(r["area_tgt_rel"]) for r in lv], "o-", label="acceptor polyhedron area")
    ax[0].loglog(hs, [abs(r["area_src_rel"]) for r in lv], "s-", label="donor polyhedron area")
    ax[0].loglog(hs, [abs(r["recv_bf_rel"]) for r in lv], "^-", label="transferred $\\int(x+y+z)\\,dA$")
    ref = np.abs(lv[0]["area_tgt_rel"]) * (hs / hs[0]) ** 2
    ax[0].loglog(hs, ref, "k--", lw=1, label="slope 2 reference")
    ax[0].set_xlabel("h = sqrt(area/cells)")
    ax[0].set_ylabel("relative error vs closed form")
    ax[0].set_title("curved octant: two polyhedra, closed-form reference")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    ax[1].loglog(hs, [abs(r["raw_conservation"]) for r in lv], "o-",
                 label="raw (no backfill)")
    ax[1].loglog(hs, [max(abs(r["bf_conservation"]), 1e-17) for r in lv], "s-",
                 label="measure-proportional backfill")
    ax[1].axhline(1e-13, color="k", ls=":", lw=1, label="1e-13 gate")
    ax[1].set_xlabel("h")
    ax[1].set_ylabel("|received/sent - 1|")
    ax[1].set_title("conservation: the m5207 $\\S$5.2 backfill is on the critical path")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGS, "fig2_curved_octant.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    # -- fig3: the trap, bar chart -------------------------------------------
    nc = record["D3"]["negative_controls"]
    labels = ["raw\n(no backfill)", "correct backfill\n$|T_j|/\\sum|T|^2$",
              "trap\n$|T_j|/\\sum|T|$", "midpoint\nsampling"]
    vals = [nc["raw_defect"], max(nc["correct_backfill_residual"], 1e-17),
            nc["trap_residual"], nc["midpoint_residual"]]
    colors = ["tab:gray", "tab:green", "tab:red", "tab:orange"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(labels, vals, color=colors)
    ax.set_yscale("log")
    ax.axhline(1e-13, color="k", ls=":", lw=1)
    ax.set_ylabel("|received/sent - 1|")
    ax.set_title("measure-normalisation trap (m5207 $\\S$5.2), mismatched curved mesh")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.4, f"{v:.1e}", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    path = os.path.join(FIGS, "fig3_measure_trap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    # -- fig4: kinked 2D interface + three-operator table ---------------------
    k = record["D5"]["levels"]
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8))
    hk = np.array([r["h"] for r in k])
    ax[0].loglog(hk, [r["l2"] for r in k], "o-", label="order-2 mortar L2")
    ax[0].loglog(hk, [r["patch_l2"] for r in k], "s-", label="linear patch test")
    ax[0].loglog(hk, k[0]["l2"] * (hk / hk[0]) ** 2, "k--", lw=1, label="slope 2")
    ax[0].set_xlabel("h")
    ax[0].set_ylabel("L2 error")
    ax[0].set_title("2D interface with a 30$^\\circ$ kink (two panels)")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    rows = [r for r in record["D6"]["rows"] if "no backfill" not in r["operator"]]
    cases = []
    for r in rows:
        if r["case"] not in cases:
            cases.append(r["case"])
    ops = sorted({r["operator"] for r in rows})
    width = 0.8 / max(len(ops), 1)
    for i, op in enumerate(ops):
        ys = []
        for c in cases:
            vals_ = [abs(r["conservation_rel"]) for r in rows
                     if r["case"] == c and r["operator"] == op]
            ys.append(vals_[0] if vals_ else np.nan)
        ax[1].bar(np.arange(len(cases)) + i * width, np.maximum(ys, 1e-17), width,
                  label=op)
    ax[1].set_yscale("log")
    ax[1].set_xticks(np.arange(len(cases)) + 0.4 - width / 2)
    ax[1].set_xticklabels([c.replace(" (", "\n(") for c in cases], fontsize=7)
    ax[1].set_ylabel("|conservation residual|")
    ax[1].set_title("three-operator comparison (with the 1D contract row)")
    ax[1].axhline(1e-13, color="k", ls=":", lw=1)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    path = os.path.join(FIGS, "fig4_kinked_and_operators.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    # -- fig5: mesh picture of the curved interface --------------------------
    src, tgt = curved_octant_case(*CURVED_LEVELS[1])
    fig = plt.figure(figsize=(10.4, 3.6))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for side, colour, lab in ((src, "tab:blue", f"UV sphere ({src.n} facets)"),
                              (tgt, "tab:red", f"icosphere ({tgt.n} facets)")):
        for t in side.tris:
            p = np.vstack([t, t[0]])
            ax.plot(p[:, 0], p[:, 1], p[:, 2], colour, lw=0.25, alpha=0.6)
        ax.plot([], [], [], colour, label=lab, lw=1.2)
    ax.set_title("sphere octant: two non-matching polyhedra")
    ax.legend(fontsize=7)
    ax.view_init(24, 34)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_trisurf(np.array([c[0] for c in src.centroids]),
                     np.array([c[1] for c in src.centroids]),
                     np.array([c[2] for c in src.centroids]),
                     triangles=None, alpha=0.35, color="tab:blue")
    ax2.scatter(src.centroids[:, 0], src.centroids[:, 1], src.centroids[:, 2],
                s=2, color="tab:blue", label="donor centroids")
    ax2.scatter(tgt.centroids[:, 0], tgt.centroids[:, 1], tgt.centroids[:, 2],
                s=14, color="tab:red", label="acceptor centroids")
    ax2.set_title("cell centres of the two partitions")
    ax2.legend(fontsize=7)
    ax2.view_init(24, 34)

    # dihedral-angle distribution of the overlap pairs: the projection defect
    ents = surface_overlap_entries(src, tgt)
    cosines = np.array([abs(float(np.dot(src.normals[i], tgt.normals[j])))
                        for (i, j, _m, _mom) in ents])
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(np.degrees(np.arccos(np.clip(cosines, -1, 1))), bins=30, color="tab:purple")
    ax3.set_xlabel("dihedral angle between the two facets (deg)")
    ax3.set_ylabel("overlap pairs")
    ax3.set_title(f"overlap-pair dihedral (mean {np.degrees(np.arccos(cosines.mean())):.2f}$^\\circ$)")
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGS, "fig5_mesh_and_dihedral.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)
    return made


def tables(record: dict) -> str:
    lines = ["# IN52-N01 3D tri-tri non-coplanar mortar: result tables",
             "",
             "All numbers are produced by `scripts/run_interface_tri3d_benchmark.py`",
             "(no hand transcription).  Gate values are in",
             "`artifacts/in52_n01/interface_tri3d_acceptance.json`.", ""]
    lines += ["## D1-D5 gates", "", "| gate | measured | threshold | relation | pass |",
              "|---|---|---|---|---|"]
    for name, g in record["gates"].items():
        val = g.get("measured")
        lines.append(f"| {name} | {val:.6g} | {g.get('threshold', float('nan')):.3g} | "
                     f"{g.get('relation', '-')} | {g.get('passed', '-')} |"
                     if isinstance(val, float) else
                     f"| {name} | - | - | - | {g.get('passed')} |")
    lines += ["", "## D1 planar tri-tri (17 deg rotated facets, jittered)", "",
              "| n_src | n_tgt | h | coverage (donor, abs) | conservation o1 | "
              "conservation o2 | transpose |",
              "|---|---|---|---|---|---|---|"]
    for c in record["D1"]["cases"]:
        lines.append(f"| {c['n_src']} | {c['n_tgt']} | {c['h']:.4f} | "
                     f"{c['coverage_src_abs']:.3e} | {c['orders']['order1']['residual']:.3e} | "
                     f"{c['orders']['order2']['residual']:.3e} | {c['transpose_rel']:.3e} |")
    lines += ["", "## D2 linear patch test", "",
              "| n_src | n_tgt | L2 order 1 | L2 order 2 |", "|---|---|---|---|"]
    for c in record["D2"]["cases"]:
        lines.append(f"| {c['n_src']} | {c['n_tgt']} | {c['l2_order1']:.3e} | "
                     f"{c['l2_order2']:.3e} |")
    lines += ["", "## D3 curved octant (UV sphere donor / icosphere acceptor)", "",
              "| n_src | n_tgt | h | area err (acceptor) | field-integral err | "
              "raw conservation | post-backfill |",
              "|---|---|---|---|---|---|---|"]
    for r in record["D3"]["levels"]:
        lines.append(f"| {r['n_src']} | {r['n_tgt']} | {r['h']:.4f} | "
                     f"{r['area_tgt_rel']:+.3e} | {r['recv_bf_rel']:+.3e} | "
                     f"{r['raw_conservation']:+.3e} | {r['bf_conservation']:+.1e} |")
    o = record["D3"]
    lines += ["", f"Observed orders: area {o['order_area_tgt']['fit']:.2f}, "
              f"field integral {o['order_field_integral']['fit']:.2f}, "
              f"raw conservation defect {o['order_raw_conservation_defect']['fit']:.2f}", ""]
    lines += ["## D3 negative controls (mismatched curved mesh)", "",
              f"* raw defect: {o['negative_controls']['raw_defect']:.3e}",
              f"* correct backfill: {o['negative_controls']['correct_backfill_residual']:.3e}",
              f"* trap (linear normalisation): {o['negative_controls']['trap_residual']:.3e} "
              f"({100 * o['negative_controls']['trap_share_of_deficit']:.1f}% of the deficit)",
              f"* midpoint sampling: {o['negative_controls']['midpoint_residual']:.3e}", ""]
    lines += ["## D5 2D kinked (30 deg) interface", "",
              "| cells_src | cells_tgt | h | conservation | L2 | patch L2 |",
              "|---|---|---|---|---|---|"]
    for r in record["D5"]["levels"]:
        lines.append(f"| {r['cells_src']} | {r['cells_tgt']} | {r['h']:.4f} | "
                     f"{r['conservation']:.1e} | {r['l2']:.3e} | {r['patch_l2']:.1e} |")
    lines += ["", f"Observed order: {record['D5']['order']['fit']:.2f}", ""]
    lines += ["## D6 three-operator comparison", "",
              "| case | dim | operator | sent | recv | conservation (rel) |",
              "|---|---|---|---|---|---|"]
    for r in record["D6"]["rows"]:
        lines.append(f"| {r['case']} | {r['dim']} | {r['operator']} | {r['sent']:.6f} | "
                     f"{r['recv']:.6f} | {r['conservation_rel']:+.2e} |")
    lines += ["", record["D6"]["note"], ""]
    return "\n".join(lines)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    record = run_acceptance()
    with open(os.path.join(OUT, "interface_tri3d_acceptance.json"), "w",
              encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    with open(os.path.join(OUT, "results_tables.md"), "w", encoding="utf-8") as fh:
        fh.write(tables(record))
    figs = figures(record)
    print("verdict:", record["verdict"])
    for p in figs:
        print("figure:", p, os.path.getsize(p))
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())