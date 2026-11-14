#!/usr/bin/env python3
"""Generate figures for the m20707 Beverloo campaign."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.silo_flow.beverloo import BEVERLOO_C, BEVERLOO_K, beverloo_prediction

FIG_DIR = ROOT / "docs" / "figures"
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"


def main(run_dir: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir or ROOT / "runs" / "m20707_beverloo")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ref = json.loads(REF_PATH.read_text())
    C_ref = float(ref["coefficients"]["C"])
    k_ref = float(ref["coefficients"]["k"])

    cases = {}
    for p in sorted(run_dir.glob("*.json")):
        d = json.loads(p.read_text())
        cases[d["case_id"]] = d

    if not cases:
        print("No results found")
        return

    primary_ids = ["D4", "D5", "D6"]
    prim = {cid: cases[cid] for cid in primary_ids if cid in cases}

    if len(prim) >= 2:
        D = np.array([prim[cid]["case"]["orifice_diameter"] for cid in prim])
        W = np.array([prim[cid]["result"]["flow_rate_kg_s"] for cid in prim])
        Wsd = np.array([prim[cid]["result"]["flow_rate_std_kg_s"] for cid in prim])
        rho = float(np.mean([prim[cid]["result"]["rho_bulk"] for cid in prim]))
        dpart = prim[primary_ids[0]]["case"]["particle_diameter"]
        ratios = np.array([prim[cid]["case"]["D_over_d"] for cid in prim])

        # ---- Figure 1: W vs D/d ----
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.errorbar(ratios, W, yerr=Wsd, fmt="o", capsize=4, ms=7,
                    label="DEM (measured)", color="C0", zorder=5)
        D_cont = np.linspace(D.min() * 0.9, D.max() * 1.1, 100)
        W_bev = np.array([beverloo_prediction(Di, dpart, rho) for Di in D_cont])
        ratios_cont = D_cont / dpart
        ax.plot(ratios_cont, W_bev, "--", color="C1", lw=1.5,
                label=f"Beverloo C={C_ref}, k={k_ref}")
        ax.set_xlabel("D / d", fontsize=11)
        ax.set_ylabel("W  [kg/s]", fontsize=11)
        ax.set_title("Silo discharge rate vs orifice diameter", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "m20707_beverloo_rate_vs_orifice.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {FIG_DIR / 'm20707_beverloo_rate_vs_orifice.png'}")

        # ---- Figure 2: log-log Beverloo scaling ----
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        eff = D - k_ref * dpart
        ax.errorbar(np.log(eff), np.log(W), yerr=Wsd / np.maximum(W, 1e-30),
                    fmt="o", capsize=4, ms=7, label="DEM", color="C0", zorder=5)
        eff_cont = D_cont - k_ref * dpart
        ax.plot(np.log(eff_cont), np.log(W_bev), "--", color="C1", lw=1.5,
                label="Beverloo (5/2 exponent)")
        # Fit line
        from tensordem.silo_flow.beverloo import fit_power_law
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
        fig.savefig(FIG_DIR / "m20707_beverloo_loglog.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {FIG_DIR / 'm20707_beverloo_loglog.png'}")

    # ---- Figure 3: Window series ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=False)
    axes = axes.ravel()
    for idx, cid in enumerate(sorted(cases)):
        if idx >= 6:
            break
        ax = axes[idx]
        c = cases[cid]
        wins = c["result"]["flow_rate_windows_kg_s"]
        n_win = len(wins)
        ax.bar(range(1, n_win + 1), wins, color="C0", alpha=0.7)
        W_ref = c["beverloo"]["W_ref_kg_s"]
        ax.axhline(W_ref, color="C1", ls="--", lw=1.2, label="Beverloo")
        ax.set_title(f"{cid} (D/d={c['case']['D_over_d']:.0f}, "
                     f"μ={c['case']['mu']:.2f})", fontsize=9)
        ax.set_xlabel("Window", fontsize=8)
        ax.set_ylabel("W [kg/s]", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(len(cases), 6):
        axes[idx].set_visible(False)
    fig.suptitle("Per-window discharge rates", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m20707_beverloo_windows.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'm20707_beverloo_windows.png'}")

    # ---- Figure 4: Friction sensitivity ----
    mu_cases = [cid for cid in cases if cid.startswith("D5_mu") or cid == "D5"]
    if len(mu_cases) >= 2:
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        mus = [cases[cid]["case"]["mu"] for cid in mu_cases]
        C_recs = [cases[cid]["beverloo"]["C_recovered"] for cid in mu_cases]
        errs = [cases[cid]["beverloo"]["rel_err_pct"] for cid in mu_cases]
        ax.plot(mus, C_recs, "o-", ms=6, label="C recovered")
        ax.axhline(C_ref, color="C1", ls="--", lw=1.2, label=f"C literature = {C_ref}")
        ax.fill_between([min(mus) - 0.02, max(mus) + 0.02], 0.55, 0.60,
                        alpha=0.15, color="C1", label="C range [0.55, 0.60]")
        ax.set_xlabel("Inter-particle friction μ", fontsize=10)
        ax.set_ylabel("C recovered", fontsize=10)
        ax.set_title("Beverloo C vs friction coefficient", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "m20707_beverloo_friction_sensitivity.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {FIG_DIR / 'm20707_beverloo_friction_sensitivity.png'}")

    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
