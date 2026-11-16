#!/usr/bin/env python3
"""Generate the figures for the m20707 Beverloo campaign.

All curves are derived from the per-case JSON artefacts in
``benchmarks/m20707_beverloo/results/<preset>/`` — nothing is hand-entered.

Figures
-------
m20707_beverloo_rate_vs_orifice.png          W(D/d) vs the Beverloo correlation
m20707_beverloo_loglog.png                   log-log law check, fixed-k fit
m20707_beverloo_windows.png                  per-window discharge rates
m20707_beverloo_gravity_scaling.png          W at g vs g/4 (sqrt(g) check)
m20707_beverloo_friction_sensitivity.png     recovered C vs inter-particle mu

Usage
-----
    PYTHONPATH=src python benchmarks/m20707_beverloo/render_figures.py
    PYTHONPATH=src python benchmarks/m20707_beverloo/render_figures.py --preset pilot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.silo_flow.beverloo import beverloo_prediction, fit_power_law

FIG_DIR = ROOT / "docs" / "figures"
RESULTS_ROOT = ROOT / "benchmarks" / "m20707_beverloo" / "results"
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"

LAW_CASES = ["D4", "D5", "D6"]
PRESET_STYLE = {"full": dict(color="C0", marker="o", label="DEM (full, d=3 mm)"),
                "pilot": dict(color="C3", marker="s", label="DEM (pilot, d=2 mm)")}


def load_cases(run_dir: Path) -> dict:
    cases = {}
    for p in sorted(Path(run_dir).glob("*.json")):
        d = json.loads(p.read_text())
        if "case_id" not in d:
            continue
        cases[d["case_id"]] = d
    return cases


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="both", choices=["full", "pilot", "both"])
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ref = json.loads(REF_PATH.read_text())
    C_ref = float(ref["coefficients"]["C"])
    k_ref = float(ref["coefficients"]["k"])

    if args.run_dir:
        sets = {"run": load_cases(Path(args.run_dir))}
    else:
        wanted = ["full", "pilot"] if args.preset == "both" else [args.preset]
        sets = {name: load_cases(RESULTS_ROOT / name) for name in wanted}
    sets = {k: v for k, v in sets.items() if v}
    if not sets:
        print("No results found")
        return
    all_cases = {cid: c for s in sets.values() for cid, c in s.items()}

    # ---------------- Figure 1: W vs D/d ----------------
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for name, cases in sets.items():
        law = [cid for cid in LAW_CASES if cid in cases]
        if not law:
            continue
        style = PRESET_STYLE.get(name, dict(color="C0", marker="o", label=name))
        ratios = np.array([cases[cid]["case"]["D_over_d"] for cid in law])
        W = np.array([cases[cid]["result"]["flow_rate_kg_s"] for cid in law])
        Wsd = np.array([cases[cid]["result"]["flow_rate_std_kg_s"] for cid in law])
        ax.errorbar(ratios, W, yerr=np.maximum(Wsd, 1e-12), fmt=style["marker"] + "-",
                    capsize=4, ms=7, color=style["color"], label=style["label"], zorder=5)

        dpart = cases[law[0]]["case"]["particle_diameter"]
        rho = float(np.mean([cases[cid]["result"]["rho_bulk"] for cid in law]))
        D_cont = np.linspace(dpart * 3.5, dpart * 6.5, 100)
        W_bev = np.array([beverloo_prediction(Di, dpart, rho, c=C_ref, k=k_ref)
                          for Di in D_cont])
        ax.plot(D_cont / dpart, W_bev, "--", color=style["color"], lw=1.2, alpha=0.75,
                label=f"Beverloo {C_ref}/{k_ref}, ρ_b={rho:.0f} ({name})")

    ax.set_xlabel("D / d  (orifice / particle diameter)", fontsize=11)
    ax.set_ylabel("W  [kg/s]", fontsize=11)
    ax.set_title("Silo discharge rate vs orifice size", fontsize=12)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m20707_beverloo_rate_vs_orifice.png", dpi=150)
    plt.close(fig)
    print("Wrote", FIG_DIR / "m20707_beverloo_rate_vs_orifice.png")

    # ---------------- Figure 2: log-log law check ----------------
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, cases in sets.items():
        law = [cid for cid in LAW_CASES if cid in cases]
        if len(law) < 2:
            continue
        style = PRESET_STYLE.get(name, dict(color="C0", marker="o", label=name))
        dpart = cases[law[0]]["case"]["particle_diameter"]
        D = np.array([cases[cid]["case"]["orifice_diameter"] for cid in law])
        W = np.array([cases[cid]["result"]["flow_rate_kg_s"] for cid in law])
        eff = D - k_ref * dpart
        ax.plot(np.log(eff), np.log(W), style["marker"], ms=8, color=style["color"],
                label=f"{style['label']}: points", zorder=5)
        fit = fit_power_law(eff, W)
        xx = np.linspace(eff.min() * 0.92, eff.max() * 1.08, 50)
        ax.plot(np.log(xx), np.log(fit["amplitude"]) + fit["exponent"] * np.log(xx), "-",
                color=style["color"], lw=1.2, alpha=0.8,
                label=f"fit p={fit['exponent']:.2f} (target 2.50)")

    ax.set_xlabel("ln(D − k·d)   [m]", fontsize=11)
    ax.set_ylabel("ln W   [kg/s]", fontsize=11)
    ax.set_title("Beverloo scaling check (log–log, slope should be 5/2)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m20707_beverloo_loglog.png", dpi=150)
    plt.close(fig)
    print("Wrote", FIG_DIR / "m20707_beverloo_loglog.png")

    # ---------------- Figure 3: per-window rates ----------------
    ids = sorted(all_cases)
    ncol = 3
    nrow = max(1, (len(ids) + ncol - 1) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.0 * nrow), squeeze=False)
    axes = axes.ravel()
    for idx, cid in enumerate(ids):
        ax = axes[idx]
        c = all_cases[cid]
        wins = c["result"]["flow_rate_windows_kg_s"]
        cv = float(np.std(wins, ddof=1) / max(np.mean(wins), 1e-30)) if len(wins) > 1 else 0.0
        ax.bar(range(1, len(wins) + 1), wins, color="C0", alpha=0.75)
        ax.axhline(c["beverloo"]["W_ref_kg_s"], color="C1", ls="--", lw=1.2,
                   label="Beverloo")
        ax.set_title(f"{c.get('preset', '?')}/{cid} (D/d={c['case']['D_over_d']:.0f}, "
                     f"μ={c['case']['mu']:.2f})\nCV={cv:.3f}, err={c['beverloo']['rel_err_pct']:+.1f}%",
                     fontsize=8.5)
        ax.set_xlabel("window", fontsize=8)
        ax.set_ylabel("W [kg/s]", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(len(ids), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Per-window discharge rates (dashed = Beverloo at measured ρ_b)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m20707_beverloo_windows.png", dpi=150)
    plt.close(fig)
    print("Wrote", FIG_DIR / "m20707_beverloo_windows.png")

    # ---------------- Figure 4: gravity scaling ----------------
    pairs = [(name, cases["D5"], cases["D5_g4"]) for name, cases in sets.items()
             if "D5" in cases and "D5_g4" in cases]
    if pairs:
        fig, ax = plt.subplots(figsize=(5.0, 3.8))
        labels, vals, colors = [], [], []
        for name, cg, cg4 in pairs:
            labels += [f"g\n({name})", f"g/4\n({name})"]
            vals += [cg["result"]["flow_rate_kg_s"], cg4["result"]["flow_rate_kg_s"]]
            colors += ["C0", "C2"]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, alpha=0.8, width=0.6)
        # sqrt(g) prediction anchored on the g case
        W_g = pairs[0][1]["result"]["flow_rate_kg_s"]
        for k in range(0, len(vals), 2):
            ax.plot([x[k], x[k] + 1], [vals[k], vals[k] * 0.5], "k--", lw=1.0,
                    label="√g prediction (0.5×)" if k == 0 else None)
        ratio = vals[1] / vals[0] if len(vals) >= 2 else float("nan")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("W  [kg/s]", fontsize=10)
        ax.set_title(f"Gravity scaling: W(g/4)/W(g) = {ratio:.3f} (expected 0.500)",
                     fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "m20707_beverloo_gravity_scaling.png", dpi=150)
        plt.close(fig)
        print("Wrote", FIG_DIR / "m20707_beverloo_gravity_scaling.png")

    # ---------------- Figure 5: friction sensitivity ----------------
    has_mu = False
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for name, cases in sets.items():
        mu_ids = [cid for cid in cases if cid.startswith("D5_mu")]
        if len(mu_ids) < 2:
            continue
        has_mu = True
        style = PRESET_STYLE.get(name, dict(color="C0", marker="o", label=name))
        mu_ids = sorted(mu_ids, key=lambda c: cases[c]["case"]["mu"])
        mus = [cases[c]["case"]["mu"] for c in mu_ids]
        Cs = [cases[c]["beverloo"]["C_recovered"] for c in mu_ids]
        ax.plot(mus, Cs, style["marker"] + "-", ms=7, color=style["color"],
                label=f"C recovered ({name})")
    if has_mu:
        ax.axhline(C_ref, color="C1", ls="--", lw=1.2, label=f"C literature = {C_ref}")
        ax.fill_between([0.35, 0.60], 0.55, 0.60, alpha=0.15, color="C1",
                        label="literature band [0.55, 0.60]")
        ax.set_xlim(0.35, 0.60)
        ax.set_xlabel("inter-particle friction μ", fontsize=10)
        ax.set_ylabel("C recovered", fontsize=10)
        ax.set_title("Beverloo coefficient vs friction", fontsize=11)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "m20707_beverloo_friction_sensitivity.png", dpi=150)
        print("Wrote", FIG_DIR / "m20707_beverloo_friction_sensitivity.png")
    plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()