"""Figures for the LEFM ice-crack campaign node.

Reads ``artifacts/lefm_ice_crack/results.json`` (written by
``benchmarks/lefm_ice_crack_validation.py``) plus one fresh SEN B solution for the
field plot, and writes ``artifacts/lefm_ice_crack/fig*.png``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.lefm import (SENBGeometry, build_senb, contour_j, domain_j,  # noqa: E402
                            stress_intensity_from_j, trace_crack_path)

OUT = ROOT / "artifacts" / "lefm_ice_crack"
E, B = 5.0e9, 0.05
KIC_LO, KIC_HI = 150e3, 250e3


def fig_field():
    geom = SENBGeometry(width=0.10, spacing=0.005, notch_ratio=0.3, thickness=B,
                        young_modulus=E, load_offset_cells=3)
    lat = build_senb(geom, load=100.0)
    lat.solve()
    tip = lat.metadata["tip"]
    sig = lat.node_stress()[:, 1, 1]
    fig, ax = plt.subplots(figsize=(4.2, 8.4))
    sc = ax.tripcolor(lat.x, lat.y, sig, shading="gouraud", cmap="RdBu_r")
    ax.set_aspect("equal")
    ax.plot([0, tip[0]], [tip[1], tip[1]], "k-", lw=1.4)
    res = trace_crack_path(lat, tip, 6)
    ax.plot(res.path[:, 0], res.path[:, 1], "k--o", ms=2.5, lw=1.0,
            label="MTS crack path")
    ax.set_xlabel("x [m]  (depth direction, crack grows in +x)")
    ax.set_ylabel("y [m]  (span direction)")
    ax.set_title("SEN B lattice, sigma_yy [Pa]\noffset load (mixed mode)")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="sigma_yy [Pa]")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_senb_field_path.png", dpi=140)
    plt.close(fig)


def fig_routes(data):
    senb = data["senb"]
    labels = ["ASTM", "domain J (EDI)", "contour J", "M-integral M", "energy dU/da"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.35
    for i, s in enumerate(senb):
        vals = [s["k_astm"], s["k_domain"][-1], s["k_contour"][-1], s["k_m_I"], s["k_energy"]]
        norm = np.array(vals) / s["k_astm"]
        ax.bar(np.arange(len(vals)) + (i - 0.5) * width, norm, width,
               label=f"{s['cells']} cells (a = {1000*s['spacing']:.1f} mm)")
        for j, v in enumerate(norm):
            ax.text(j + (i - 0.5) * width, v * 1.05, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(1.0, color="k", ls=":" , lw=1)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("K / K_ASTM")
    ax.set_title("SEN B crack driving force by route (a/W = 0.3, S/W = 4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_k_routes.png", dpi=140)
    plt.close(fig)


def fig_path_independence(data):
    senb = data["senb"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for s in senb:
        ax.plot(s["contour_radii"], np.array(s["k_contour"]) / s["k_astm"], "o-",
                label=f"contour J, {s['cells']} cells")
        rings = [0.5 * (r1 + r2) for r1, r2 in s["rings"]]
        ax.plot(rings, np.array(s["k_domain"]) / s["k_astm"], "s--",
                label=f"domain J, {s['cells']} cells")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("contour radius / mean annulus radius  [lattice cells]")
    ax.set_ylabel("K / K_ASTM")
    ax.set_title("Path dependence of the J routes")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_path_independence.png", dpi=140)
    plt.close(fig)


def fig_calibration(data):
    cal = data["kic_calibration"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.axhspan(KIC_LO, KIC_HI, color="0.85", label="Lu et al. (2015) band")
    for sr, marker in ((2.0, "o"), (4.0, "s"), (6.0, "^")):
        rows = [r for r in cal["rows"] if r["span_ratio"] == sr]
        alphas = [r["alpha"] for r in rows]
        after = [r["k_apparent_over_kappa"] for r in rows]
        ax.plot(alphas, after, marker + "-", label=f"S/W = {sr:.0f}")
    ax.axhline(cal["k_ic_input"], color="k", ls=":", lw=1, label="input K_IC = 200 kPa m^0.5")
    ax.set_xlabel("a/W")
    ax.set_ylabel("calibrated K_IC [Pa m^0.5]")
    ax.set_title(f"K_IC after bond calibration (kappa = {cal['kappa']:.2f})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_kic_calibration.png", dpi=140)
    plt.close(fig)


def fig_crack_path(data):
    cp = data["crack_path"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for tag, style in (("symmetric", "o-"), ("mixed_mode", "s--")):
        p = cp[tag]
        path = np.array(p["path"])
        ref = np.array(p["reference_path"])
        ax.plot(path[:, 0] * 1e3, path[:, 1] * 1e3, "k" + style, ms=3,
                label=f"{tag}: MTS lattice path")
        ax.plot(ref[:, 0] * 1e3, ref[:, 1] * 1e3, ":", lw=1.0,
                label=f"{tag}: first-order LEFM reference")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Crack path prediction (MTS, mixed mode via offset load)")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_crack_path.png", dpi=140)
    plt.close(fig)


def fig_ccp(data):
    c = data["ccp"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    labels = ["domain J\n(rings)", "contour J\n(radii)", "energy\ndU/da"]
    vals = [np.mean(c["rel_error_domain"]), np.mean(c["rel_error_contour"]),
            c["rel_error_energy"]]
    ax.bar(labels, np.array(vals) * 100.0, color=["C0", "C1", "C2"])
    for i, v in enumerate(vals):
        ax.text(i, v * 100.0, f"{v*100:+.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0.0, color="k", lw=1)
    ax.set_ylabel("K_error vs sigma sqrt(pi a) F  [%]")
    ax.set_title("Centre-cracked plate ground-truth check")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_ccp_validation.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    data = json.loads((OUT / "results.json").read_text())
    fig_field()
    fig_routes(data)
    fig_path_independence(data)
    fig_calibration(data)
    fig_crack_path(data)
    fig_ccp(data)
    print("figures written to", OUT)