#!/usr/bin/env python3
"""Figures for the W27 ice-ridge keel grounding artifact (matplotlib only).

Reads ``papers/artifacts/w27/ice-ridge-keel-grounding/ridge_keel_anchor.json``
and writes three PNGs next to it:

  fig1_kerr_beam.png      Kerr analytic vs independent FEM: deflection + reaction
  fig2_keel_footing.png   bonded keel vs Kerr: resistance vs footing width, and
                          the pressure-settlement curve of the confined test
  fig3_refinement.png     particle-count / mesh refinement convergence with GCI
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensordem import ridge_keel_grounding as rg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=os.path.join(
        "papers", "artifacts", "w27", "ice-ridge-keel-grounding", "ridge_keel_anchor.json"))
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    art = json.load(open(args.artifact))
    outdir = args.outdir or os.path.dirname(args.artifact)
    os.makedirs(outdir, exist_ok=True)
    made = []

    # --- fig 1: Kerr analytic vs FEM -------------------------------------
    cfg = art["configuration"]["kerr_sets"]["balanced"]
    fnd = rg.KerrFoundation(k_upper=cfg["k_upper"], k_lower=cfg["k_lower"], shear=cfg["shear"])
    a, q0 = 2.0, 1.0e5
    x = np.linspace(-20.0, 20.0, 1201)
    an = rg.kerr_beam_analytic(a, q0, ei=cfg["ei"], foundation=fnd, x=x)
    fem = rg.kerr_beam_fem(np.linspace(-20.0, 20.0, 601), ei=cfg["ei"], foundation=fnd,
                           load=lambda xx: np.where(np.abs(xx) <= a, q0, 0.0))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(an.x, an.w * 1e3, "k-", lw=2, label="Kerr closed form")
    axes[0].plot(fem.x, fem.w * 1e3, "r--", lw=1.5, label="mixed Galerkin FEM")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("deflection w [mm]")
    axes[0].set_title("Kerr beam, patch load: analytic vs FEM")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(an.x, an.p / 1e3, "k-", lw=2, label="Kerr closed form")
    axes[1].plot(fem.x, fem.p / 1e3, "r--", lw=1.5, label="mixed Galerkin FEM")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("foundation reaction p [kPa]")
    axes[1].set_title("reaction (edge jumps are the c q'' term)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(outdir, "fig1_kerr_beam.png")
    fig.savefig(p1, dpi=150); plt.close(fig); made.append(p1)

    # --- fig 2: bonded keel vs Kerr --------------------------------------
    red = art["C_keel_to_kerr_reduction"]
    preds = red["footing_predictions"]
    keys = list(preds)
    b = np.array([preds[k]["footing_width_m"] for k in keys])
    dem = np.array([preds[k]["dem_N_per_m"] for k in keys])
    ker = np.array([preds[k]["kerr_N_per_m"] for k in keys])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = np.argsort(b)
    axes[0].plot(b[order], dem[order] / 1e6, "o", ms=8, label="bonded keel (DEM)")
    axes[0].plot(b[order], ker[order] / 1e6, "s--", ms=6, label="Kerr closed form (identified)")
    axes[0].set_xlabel("footing width b [m]"); axes[0].set_ylabel("resistance R [MN/m]")
    axes[0].set_title("footing resistance, same geometry and settlement")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    ws = np.array(red["confined_pressure_settlement"]["settlements_m"]) * 1e3
    ps = np.array(red["confined_pressure_settlement"]["pressures_Pa"]) / 1e6
    axes[1].plot(ws, ps, "o-", label="DEM confined compression")
    axes[1].axhline(red["k_subgrade_N_per_m3"] * art["configuration"]["w0_m"] / 1e6,
                    color="k", ls=":", label="K from secant at w0")
    axes[1].set_xlabel("settlement w [mm]"); axes[1].set_ylabel("pressure [MPa]")
    axes[1].set_title("keel subgrade response (wide lid)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(outdir, "fig2_keel_footing.png")
    fig.savefig(p2, dpi=150); plt.close(fig); made.append(p2)

    # --- fig 3: refinement ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    summ = art["B_bonded_keel_dem"]["summary"]
    npart = np.array([s["n_particles"] for s in summ], dtype=float)
    for key, label in (("1", "b = H_k (validation)"), ("0.5", "b = 0.5 H_k (identification)"),
                       ("2", "b = 2 H_k")):
        r = np.array([s["resistance_N_per_m"][key] for s in summ])
        axes[0].plot(npart / 1e3, r / 1e6, "o-", label=label)
    axes[0].set_xlabel("particles [thousand]"); axes[0].set_ylabel("R [MN/m]")
    axes[0].set_title("particle-count convergence of the footing resistance")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    mesh = art["A_kerr_beam_verification"]["mesh_refinement"]
    ne = np.array([m["n_elements"] for m in mesh], dtype=float)
    wc = np.array([m["w_centre_m"] for m in mesh])
    axes[1].semilogx(ne, wc * 1e3, "o-", label="FEM centre deflection")
    axes[1].axhline(mesh[-1]["w_centre_m"] * 1e3, color="k", ls=":",
                    label="finest mesh / analytic")
    axes[1].set_xlabel("beam elements"); axes[1].set_ylabel("w(0) [mm]")
    axes[1].set_title("beam mesh convergence")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(outdir, "fig3_refinement.png")
    fig.savefig(p3, dpi=150); plt.close(fig); made.append(p3)
    print(json.dumps({"figures": made}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())