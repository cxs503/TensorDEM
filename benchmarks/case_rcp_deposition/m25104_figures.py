#!/usr/bin/env python3
"""m25104 — figures for the RCP fresh-verification report.

Reads ``runs/m25104_rcp_verify/verify_summary.json`` (from m25104_verify.py)
and the settled positions (``.npz``) to produce the report figures:

* ``fig_m25104_phi_vs_seed.png``   — per-seed phi_corr, archived vs fresh
* ``fig_m25104_phi_vs_mu.png``     — pooled phi vs mu, band + reference
* ``fig_m25104_snapshot.png``      — settled packing cross-section (mu=0)

All numbers are read from the evidence JSON (zero hand-copy).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "runs" / "m25104_rcp_verify" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PHI_REF = 0.6366
BAND = (0.62, 0.66)


def load_summary() -> dict:
    with open(ROOT / "runs" / "m25104_rcp_verify" / "verify_summary.json") as fh:
        return json.load(fh)


def fig_phi_vs_seed(summary: dict) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhspan(BAND[0], BAND[1], color="green", alpha=0.12,
              label="acceptance band 0.62–0.66")
    ax.axhline(PHI_REF, color="k", ls="--", lw=1,
               label=f"Bernal & Mason 1960  φ_ref={PHI_REF}")
    colors = {"mu=0": "tab:blue", "mu=0.1": "tab:red"}
    markers = {"archived": "o", "fresh_m25104": "^"}
    for mukey, lv in summary["levels"].items():
        mu = mukey.split("=")[1]
        c = colors[mukey]
        for r in lv["rows"]:
            mk = markers[r["origin"]]
            ax.plot(r["seed"], r["phi_corr"], marker=mk, ms=9,
                    color=c, mec="k", mew=0.6,
                    label=(f"μ={mu} {r['origin']}"
                           if r["seed"] in (1, 4) else None))
        m = lv["phi_pooled"]["mean"]
        s = lv["phi_pooled"]["std"]
        ax.axhline(m, color=c, ls=":", lw=1.2)
        ax.annotate(f"μ={mu} pooled {m:.4f}±{s:.4f}",
                    xy=(0.02, 0.04 + 0.12 * (0 if mu == "0" else 1)),
                    xycoords="axes fraction", color=c, fontsize=9)
    ax.set_xlabel("seed")
    ax.set_ylabel("φ (overlap-corrected bulk volume fraction)")
    ax.set_title("RCP volume fraction per seed — archived (1–3) vs fresh m25104 (4–6)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    p = OUT / "fig_m25104_phi_vs_seed.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def fig_phi_vs_mu(summary: dict) -> str:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhspan(BAND[0], BAND[1], color="green", alpha=0.12,
              label="acceptance band 0.62–0.66")
    ax.axhline(PHI_REF, color="k", ls="--", lw=1,
               label=f"RCP ref {PHI_REF} (Bernal & Mason 1960)")
    mus, phis, errs, labels = [], [], [], []
    for mukey, lv in summary_levels(summary):
        mu = float(mukey.split("=")[1])
        mus.append(mu)
        phis.append(lv["phi_pooled"]["mean"])
        errs.append(lv["phi_pooled"]["std"])
    ax.errorbar(mus, phis, yerr=errs, fmt="o", capsize=5, ms=9,
                color="tab:purple", ecolor="gray",
                label="TensorDEM deposition (N=10000, 6 seeds pooled)")
    # literature anchors
    ax.plot(0.0, PHI_REF, "s", ms=9, mfc="none", mec="k",
            label="frictionless RCP 0.6366 (Bernal & Mason 1960)")
    ax.plot(0.5, 0.60, "s", ms=9, mfc="none", mec="gray",
              label="frictional RCP ≈0.60 (Silbert et al. 2002, μ→∞ trend)")
    ax.set_xlabel("inter-particle friction μ")
    ax.set_ylabel("φ (overlap-corrected bulk volume fraction)")
    ax.set_title("RCP volume fraction vs friction — fresh m25104 verification")
    ax.set_xticks([0.0, 0.1])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = OUT / "fig_m25104_phi_vs_mu.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def summary_levels(summary: dict):
    return list(summary["levels"].items())


def fig_snapshot(summary: dict) -> str | None:
    """Cross-section of the settled mu=0 packing (fresh seed 4)."""
    import glob
    cands = sorted(glob.glob(str(
        ROOT / "runs" / "m25104_rcp_verify" / "runs"
        / "deposition_n10000_mu0_kn4000_e0.1_seed4.npz")))
    if not cands:
        return None
    d = np.load(cands[0])
    pos = d["positions"] if "positions" in d else d["pos"]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    # mid-plane cross-section: take particles within a slab of the y-midplane
    ymid = float(np.median(y))
    slab = 0.6
    sel = np.abs(y - ymid) < slab
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x[sel], z[sel], s=6, c=z[sel], cmap="viridis",
                     vmin=0, vmax=float(z.max()))
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Settled N=10000 μ=0 packing — mid-y slab (fresh seed 4)")
    fig.colorbar(sc, ax=ax, label="z")
    ax.set_aspect("equal")
    p = OUT / "fig_m25104_snapshot.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def main() -> int:
    summary = load_summary()
    paths = [fig_phi_vs_seed(summary), fig_phi_vs_mu(summary)]
    snap = fig_snapshot(summary)
    if snap:
        paths.append(snap)
    out = {"figures": paths}
    with open(OUT / "figures.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
