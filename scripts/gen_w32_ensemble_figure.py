#!/usr/bin/env python3
"""W32-5: Generate the DEM-vs-Kerr ensemble comparison figure for the report.

Reads the verified artifact (papers/data/w32-2026-11-16/ice/granular_vs_kerr.json)
and renders:
  (a) DEM footing resistance R_dem vs footing width b/Hk -- per-seed points,
      ensemble mean +/- 1 std band, and the Kerr closed-form prediction;
  (b) per-seed DEM-vs-Kerr relative error vs b/Hk with the 15% engineering
      criterion and the 10% ensemble-CV criterion marked.

All numbers are injected from the artifact (no hand-copy).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path(__file__).resolve().parent.parent / "papers" / "data" / "w32-2026-11-16" / "ice" / "granular_vs_kerr.json"
OUT = Path(__file__).resolve().parent.parent / "docs" / "m28264" / "fig_w32_ensemble.png"


def main() -> None:
    d = json.loads(ART.read_text())
    widths = ["0.5", "1.0", "2.0"]
    b_over_h = [float(w) for w in widths]  # b/Hk
    ens = d["ensemble"]["widths"]

    # Per-seed DEM and Kerr resistance
    R_dem_seeds = {w: ens[w]["R_dem_per_seed_N_per_m"] for w in widths}
    R_kerr_seeds = {w: ens[w]["R_kerr_per_seed_N_per_m"] for w in widths}
    mean_dem = {w: ens[w]["mean_R_dem_N_per_m"] for w in widths}
    std_dem = {w: ens[w]["std_R_dem_N_per_m"] for w in widths}
    mean_kerr = {w: float(np.mean(R_kerr_seeds[w])) for w in widths}
    rel_err = {w: [r * 100 for r in ens[w]["rel_error_per_seed"]] for w in widths}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel (a): resistance vs b/Hk ----
    for w in widths:
        b = float(w)
        for r in R_dem_seeds[w]:
            ax1.plot(b, r / 1e6, "o", color="tab:blue", ms=5, alpha=0.55,
                     label="DEM per-seed" if w == "0.5" else None)
        ax1.errorbar(b, mean_dem[w] / 1e6, yerr=std_dem[w] / 1e6,
                     fmt="s", color="tab:blue", ms=7, capsize=4, lw=1.5,
                     label="DEM mean ± 1σ" if w == "0.5" else None)
        ax1.plot(b, mean_kerr[w] / 1e6, "D", color="tab:red", ms=7,
                 label="Kerr closed form" if w == "0.5" else None)
    ax1.set_xlabel("Footing width  b / H$_k$  (-)")
    ax1.set_ylabel("Footing resistance  R  (MN/m)")
    ax1.set_title("(a) DEM vs Kerr: resistance at w$_0$=0.008 m")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)

    # ---- Panel (b): relative error vs b/Hk ----
    for w in widths:
        b = float(w)
        for e in rel_err[w]:
            ax2.plot(b, e, "o", color="tab:blue", ms=5, alpha=0.55,
                     label="per-seed" if w == "0.5" else None)
    ax2.axhline(15.0, color="tab:red", ls="--", lw=1.5, label="15% engineering criterion")
    ax2.axhline(10.0, color="tab:green", ls=":", lw=1.5, label="10% ensemble-CV criterion")
    ax2.axhline(3.0, color="tab:orange", ls="-.", lw=1.2, label="3% strict threshold")
    ax2.set_xlabel("Footing width  b / H$_k$  (-)")
    ax2.set_ylabel("|R$_{kerr}$ − R$_{dem}$| / R$_{dem}$  (%)")
    ax2.set_title("(b) Per-seed DEM-vs-Kerr relative error")
    ax2.set_ylim(-2, 16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle("W32-5 ice-ridge keel grounding: 5-seed granular ensemble vs Kerr model",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Figure written to {OUT}")


if __name__ == "__main__":
    main()
