#!/usr/bin/env python3
"""Generate figures for the CASE-DEM-SILO-BEVERLOO validation report."""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.silo_flow.beverloo import BEVERLOO_C, BEVERLOO_K, beverloo_prediction, fit_power_law

RUN_DIR = ROOT / "runs" / "m20707_beverloo"
FIG_DIR = ROOT / "docs" / "case_beverloo" / "figures"
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"


def fig_to_base64(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def main(run_dir: str | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir or RUN_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ref = json.loads(REF_PATH.read_text())
    C_ref = float(ref["coefficients"]["C"])
    k_ref = float(ref["coefficients"]["k"])

    cases = {}
    for p in sorted(run_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        d = json.loads(p.read_text())
        cases[d["case_id"]] = d

    if not cases:
        print("No results found")
        return {}

    figures = {}
    primary_ids = ["D4", "D5", "D6"]
    prim = {cid: cases[cid] for cid in primary_ids if cid in cases}

    # ---- Figure 1: W vs D/d ----
    if len(prim) >= 2:
        D = np.array([prim[cid]["case"]["orifice_diameter"] for cid in prim])
        W = np.array([prim[cid]["result"]["flow_rate_kg_s"] for cid in prim])
        Wsd = np.array([prim[cid]["result"]["flow_rate_std_kg_s"] for cid in prim])
        rho = float(np.mean([prim[cid]["result"]["rho_bulk"] for cid in prim]))
        dpart = prim[list(prim.keys())[0]]["case"]["particle_diameter"]
        ratios = np.array([prim[cid]["case"]["D_over_d"] for cid in prim])

        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.errorbar(ratios, W * 1000, yerr=Wsd * 1000, fmt="o", capsize=4, ms=7,
                    label="DEM (measured)", color="C0", zorder=5)
        D_cont = np.linspace(D.min() * 0.9, D.max() * 1.1, 100)
        W_bev = np.array([beverloo_prediction(Di, dpart, rho) for Di in D_cont])
        ratios_cont = D_cont / dpart
        ax.plot(ratios_cont, W_bev * 1000, "--", color="C1", lw=1.5,
                label=f"Beverloo C={C_ref}, k={k_ref}")
        ax.set_xlabel("D / d", fontsize=11)
        ax.set_ylabel("W  [g/s]", fontsize=11)
        ax.set_title("Silo discharge rate vs orifice diameter", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path1 = FIG_DIR / "beverloo_rate_vs_orifice.png"
        fig.savefig(path1, dpi=150)
        figures["rate_vs_orifice"] = {
            "name": "rate_vs_orifice",
            "caption": "DEM-measured discharge rate vs D/d, compared with Beverloo correlation",
            "png_base64": fig_to_base64(fig),
        }
        plt.close(fig)
        print(f"Wrote {path1}")

        # ---- Figure 2: log-log ----
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        eff = D - k_ref * dpart
        ax.errorbar(np.log(eff), np.log(W), yerr=Wsd / np.maximum(W, 1e-30),
                    fmt="o", capsize=4, ms=7, label="DEM", color="C0", zorder=5)
        eff_cont = D_cont - k_ref * dpart
        ax.plot(np.log(eff_cont), np.log(W_bev), "--", color="C1", lw=1.5,
                label="Beverloo (5/2 exponent)")
        fit = fit_power_law(eff, W)
        xx = np.linspace(eff.min() * 0.9, eff.max() * 1.1, 100)
        ax.plot(np.log(xx), np.log(fit["amplitude"]) + fit["exponent"] * np.log(xx),
                "-", color="C2", lw=1.2,
                label=f"fit p={fit['exponent']:.2f}, R²={fit['r2']:.3f}")
        ax.set_xlabel("ln(D − k·d)  [m]", fontsize=11)
        ax.set_ylabel("ln W  [kg/s]", fontsize=11)
        ax.set_title("Beverloo scaling (log–log)", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path2 = FIG_DIR / "beverloo_loglog.png"
        fig.savefig(path2, dpi=150)
        figures["loglog"] = {
            "name": "loglog",
            "caption": "Log-log Beverloo scaling with fitted exponent",
            "png_base64": fig_to_base64(fig),
        }
        plt.close(fig)
        print(f"Wrote {path2}")

    # ---- Figure 3: Window series ----
    n_cases = len(cases)
    ncols = min(3, n_cases)
    nrows = (n_cases + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for idx, cid in enumerate(sorted(cases)):
        if idx >= nrows * ncols:
            break
        ax = axes[idx]
        c = cases[cid]
        wins = c["result"]["flow_rate_windows_kg_s"]
        n_win = len(wins)
        ax.bar(range(1, n_win + 1), [w * 1000 for w in wins], color="C0", alpha=0.7)
        W_ref = c["beverloo"]["W_ref_kg_s"]
        ax.axhline(W_ref * 1000, color="C1", ls="--", lw=1.2, label="Beverloo")
        ax.set_title(f"{cid} (D/d={c['case']['D_over_d']:.0f})", fontsize=9)
        ax.set_xlabel("Window", fontsize=8)
        ax.set_ylabel("W [g/s]", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(n_cases, nrows * ncols):
        axes[idx].set_visible(False)
    fig.suptitle("Per-window discharge rates", fontsize=11)
    fig.tight_layout()
    path3 = FIG_DIR / "beverloo_windows.png"
    fig.savefig(path3, dpi=150)
    figures["windows"] = {
        "name": "windows",
        "caption": "Per-window discharge rates with Beverloo reference",
        "png_base64": fig_to_base64(fig),
    }
    plt.close(fig)
    print(f"Wrote {path3}")

    print(f"\nGenerated {len(figures)} figures")
    return figures


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
