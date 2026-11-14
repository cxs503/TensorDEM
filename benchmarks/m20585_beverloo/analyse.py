#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO: comprehensive analysis and report generation (m20585).

Reads results from:
  - runs/m20055_beverloo/*.json  (m20055 baseline, kn=1e5)
  - runs/m20585_beverloo_kn_sensitivity/*.json  (kn sensitivity study)

Generates:
  - docs/figures/m20585_beverloo_*.png
  - runs/m20585_beverloo/summary.json
  - runs/m20585_beverloo/report_tables.md
  - docs/CASE-DEM-SILO-BEVERLOO_m20585.md

Usage
-----
    PYTHONPATH=src python benchmarks/m20585_beverloo/analyse.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REF = json.loads((ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json").read_text())
C_LIT = float(REF["coefficients"]["C"])
K_LIT = float(REF["coefficients"]["k"])

RUNS_20055 = ROOT / "runs" / "m20055_beverloo"
RUNS_KN = ROOT / "runs" / "m20585_beverloo_kn_sensitivity"
RUNS_OUT = ROOT / "runs" / "m20585_beverloo"
FIGS = ROOT / "docs" / "figures"


def load_m20055() -> dict:
    out = {}
    for p in sorted(RUNS_20055.glob("*.json")):
        if p.name == "summary.json":
            continue
        d = json.loads(p.read_text())
        if "telemetry" in d and "verdict" in d:
            out[d["case_id"]] = d
    return out


def load_kn_sensitivity() -> list:
    summary_path = RUNS_KN / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())["results"]
    results = []
    for p in sorted(RUNS_KN.glob("kn_*.json")):
        d = json.loads(p.read_text())
        results.append(d)
    return results


def fig_kn_sensitivity(kn_results: list, out: Path) -> None:
    """Plot error vs kn."""
    if not kn_results:
        return
    kn_vals = [r["kn"] for r in kn_results]
    errs = [r["rel_err_pct"] for r in kn_results if math.isfinite(r.get("rel_err_pct", float("nan")))]
    kn_vals = [r["kn"] for r in kn_results if math.isfinite(r.get("rel_err_pct", float("nan")))]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogx(kn_vals, errs, "o-", color="C0", markersize=8)
    ax.axhspan(-10, 10, color="C2", alpha=0.12, label="±10% acceptance")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Normal stiffness kn  [N/m]")
    ax.set_ylabel("Relative error vs Beverloo  [%]")
    ax.set_title("kn sensitivity: DEM silo discharge vs Beverloo (1961)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    for x, y in zip(kn_vals, errs):
        ax.text(x, y + (2 if y >= 0 else -4), f"{y:+.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_combined_comparison(m20055_runs: dict, kn_results: list, out: Path) -> None:
    """Combined bar chart: m20055 cases + kn sensitivity."""
    keys = ["D4", "D5", "D6"]
    m20055_errs = []
    m20055_labels = []
    for k in keys:
        if k in m20055_runs:
            m20055_errs.append(m20055_runs[k]["verdict"]["rel_err_W"] * 100)
            m20055_labels.append(k)

    kn_labels = [f"kn={r['kn']:.0e}" for r in kn_results]
    kn_errs = [r["rel_err_pct"] for r in kn_results if math.isfinite(r.get("rel_err_pct", float("nan")))]
    kn_labels = [f"kn={r['kn']:.0e}" for r in kn_results if math.isfinite(r.get("rel_err_pct", float("nan")))]

    all_labels = m20055_labels + kn_labels
    all_errs = m20055_errs + kn_errs

    if not all_labels:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    colors = ["C0" if abs(e) < 10 else "C3" for e in all_errs]
    ax.bar(range(len(all_labels)), all_errs, color=colors)
    ax.axhspan(-10, 10, color="C2", alpha=0.12, label="±10% acceptance")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Relative error vs Beverloo  [%]")
    ax.set_title("CASE-DEM-SILO-BEVERLOO: error summary")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, e in enumerate(all_errs):
        ax.text(i, e + (1.5 if e >= 0 else -3.5), f"{e:+.1f}%", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    RUNS_OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    m20055_runs = load_m20055()
    kn_results = load_kn_sensitivity()

    if not m20055_runs and not kn_results:
        print("No results found. Run the cases first.")
        return 1

    # Generate figures
    if kn_results:
        fig_kn_sensitivity(kn_results, FIGS / "m20585_beverloo_kn_sensitivity.png")
    if m20055_runs or kn_results:
        fig_combined_comparison(m20055_runs, kn_results, FIGS / "m20585_beverloo_combined_errors.png")

    # Build summary
    summary = {
        "case": "CASE-DEM-SILO-BEVERLOO",
        "mission_id": 20585,
        "reference": REF["source"],
        "reference_coefficients": REF["coefficients"],
        "m20055_baseline": {
            k: {
                "D_over_d": v["dimensionless"]["D_over_d"],
                "W_got": v["verdict"]["W_got_kg_s"],
                "W_ref": v["verdict"]["W_ref_kg_s"],
                "rel_err_pct": v["verdict"]["rel_err_W"] * 100,
                "pass_10pct": v["verdict"]["pass_10pct"],
            }
            for k, v in m20055_runs.items()
        },
        "kn_sensitivity": kn_results,
        "hypothesis": "Soft particles (kn=1e5) cause systematic over-prediction of flow rate; "
                      "stiffer kn (1e6) converges to Beverloo prediction",
        "figures": {
            "kn_sensitivity": str((FIGS / "m20585_beverloo_kn_sensitivity.png").relative_to(ROOT))
                if (FIGS / "m20585_beverloo_kn_sensitivity.png").exists() else None,
            "combined_errors": str((FIGS / "m20585_beverloo_combined_errors.png").relative_to(ROOT))
                if (FIGS / "m20585_beverloo_combined_errors.png").exists() else None,
        },
    }
    (RUNS_OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # Markdown tables
    lines = []
    lines.append("# CASE-DEM-SILO-BEVERLOO — m20585 分析报告")
    lines.append("")
    lines.append("## m20055 基线结果（kn=1e5）")
    lines.append("")
    lines.append("| case | D/d | W [kg/s] | W_ref [kg/s] | err [%] | <10% |")
    lines.append("|---|---|---|---|---|---|")
    for k in ["D4", "D5", "D6"]:
        if k in m20055_runs:
            v = m20055_runs[k]
            lines.append(
                f"| {k} | {v['dimensionless']['D_over_d']:.2f} | "
                f"{v['verdict']['W_got_kg_s']:.4f} | {v['verdict']['W_ref_kg_s']:.4f} | "
                f"{v['verdict']['rel_err_W']*100:+.1f} | "
                f"{'PASS' if v['verdict']['pass_10pct'] else 'FAIL'} |")

    lines.append("")
    lines.append("## kn 敏感性（D5 算例）")
    lines.append("")
    lines.append("| kn [N/m] | phi | rho_b [kg/m3] | W [kg/s] | W_ref [kg/s] | err [%] | <10% |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in kn_results:
        passed = abs(r.get("rel_err_pct", 999)) < 10 if math.isfinite(r.get("rel_err_pct", float("nan"))) else False
        lines.append(
            f"| {r['kn']:.0e} | {r['packing_fraction_bed']:.3f} | "
            f"{r['bulk_density_bed_kg_m3']:.0f} | {r['W_got_kg_s']:.4f} | "
            f"{r['W_ref_kg_s']:.4f} | {r.get('rel_err_pct', 'nan'):+.1f} | "
            f"{'PASS' if passed else 'FAIL'} |")

    tbl = RUNS_OUT / "report_tables.md"
    tbl.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsummary -> {RUNS_OUT/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
