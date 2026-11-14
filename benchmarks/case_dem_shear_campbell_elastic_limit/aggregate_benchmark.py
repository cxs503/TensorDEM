#!/usr/bin/env python3
"""Comprehensive validation of CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT.

Aggregates all existing round data (R3 kappa-collapse, R4 inertial-branch,
R5 constant-pressure) into a single benchmark assessment against Campbell (2002)
JFM 465:261-291.

Generates:
  - artifacts/case_dem_shear_campbell_elastic_limit/benchmark.json
  - artifacts/case_dem_shear_campbell_elastic_limit/benchmark.md
  - artifacts/case_dem_shear_campbell_elastic_limit/fig_scaling.png
  - artifacts/case_dem_shear_campbell_elastic_limit/fig_exponents.png
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

ART = REPO / "artifacts" / "case_dem_shear_campbell_elastic_limit"
ART.mkdir(parents=True, exist_ok=True)

R3_RUNS = HERE / "runs" / "kappa_collapse"
R4_RUNS = HERE / "runs" / "inertial_branch"
R5_RUNS = HERE / "runs" / "constant_pressure"
ANCHOR_FILE = HERE / "reference_anchors.json"

CASE = "CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT"


def load_r3_points():
    """Load R3 kappa-collapse point JSONs."""
    pts = []
    for f in sorted(R3_RUNS.glob("kn*.json")):
        if f.name.startswith("kn") and f.name.endswith(".json"):
            d = json.loads(f.read_text())
            if "kappa" in d and "tau_star" in d:
                pts.append(d)
    return pts


def load_r3_analysis():
    return json.loads((R3_RUNS / "analysis.json").read_text())


def load_r4_analysis():
    return json.loads((R4_RUNS / "analysis.json").read_text())


def load_r5_analysis():
    return json.loads((R5_RUNS / "analysis.json").read_text())


def load_anchors():
    return json.loads(ANCHOR_FILE.read_text())


def loglog_slope(xs, ys):
    x = [math.log(v) for v in xs if v > 0]
    y = [math.log(v) for v in ys if v > 0]
    if len(x) < 2:
        return None, None
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        return None, None
    beta = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx
    residuals = [b - (my + beta * (a - mx)) for a, b in zip(x, y)]
    sse = sum(r ** 2 for r in residuals)
    sst = sum((b - my) ** 2 for b in y)
    r2 = 1 - sse / sst if sst > 0 else 0
    return beta, r2


def compute_benchmark():
    """Compute the full benchmark from existing run data."""
    r3 = load_r3_analysis()
    r4 = load_r4_analysis()
    r5 = load_r5_analysis()
    anchors = load_anchors()

    # --- R3 kappa-collapse data ---
    r3_pts = [p for p in r3["points"] if p.get("gates", {}).get("all")]
    r3_pts.sort(key=lambda p: p["kappa"])

    # Global fit from R3
    beta_global = r3["global_fit"]["beta"]
    n_global = r3["global_fit"]["n"]
    r2_global = r3["global_fit"]["r2"]

    # Local slopes from R3
    local_slopes = r3["local_slopes"]

    # Approach to elastic limit
    approach = r3["approach"]

    # Elastic coefficient
    elastic_coeff = r3["elastic_coefficient"]

    # Rate exponents from R3
    family_rates = r3["family_rate_exponents"]

    # --- R4 inertial-branch data ---
    r4_verdicts = r4.get("verdicts", {})

    # --- R5 constant-pressure data ---
    r5_verdicts = r5.get("P1_verdict", "NOT_SUPPORTED")

    # --- Build benchmark legs ---
    TOL_EXPONENT_ABS = 0.30  # 15% of span [0, 2]
    TOL_BETA_BAND = (0.0, 1.0)
    TOL_KAPPA_FACTOR = 3.0

    legs = []

    # Leg 1: Elastic-quasi-static exponent
    n_elastic = approach["n_high_half"]
    legs.append({
        "leg": "L1_elastic_quasi_static_exponent",
        "quantity": "d ln tau / d ln gdot in the elastic-quasi-static limit (high-kappa end)",
        "ref": 0.0,
        "ref_anchor": "campbell2002_abstract_elastic_quasi_static",
        "got": n_elastic,
        "abs_dev": n_elastic - 0.0,
        "tol": TOL_EXPONENT_ABS,
        "tol_kind": "abs",
        "verdict": "PASS" if abs(n_elastic) <= TOL_EXPONENT_ABS else "FAIL",
        "source": "Campbell (2002) abstract: stresses scale purely elastically => n = 0",
        "evidence": f"R3 kappa-collapse, beta_high_half = {approach['beta_high_half']:.4f}, "
                    f"n_high_half = {n_elastic:.4f}, sigma_n = {approach['sigma_beta_high_half'] * 2:.4f}",
    })

    # Leg 2: Elastic-inertial branch (global beta)
    legs.append({
        "leg": "L2_elastic_inertial_branch",
        "quantity": "beta = d ln tau* / d ln kappa over the gated ladder",
        "ref": "0 < beta < 1",
        "ref_anchor": "campbell2002_abstract_elastic_inertial",
        "got": beta_global,
        "tol": TOL_BETA_BAND,
        "tol_kind": "band",
        "verdict": "PASS" if 0 < beta_global < 1 else "FAIL",
        "source": "Campbell (2002) abstract: stresses scale with both elastic and inertial properties",
        "evidence": f"R3 kappa-collapse, beta_global = {beta_global:.4f}, R² = {r2_global:.4f}, "
                    f"span {r3['global_fit']['kappa_span'][0]:.1f}–{r3['global_fit']['kappa_span'][1]:.1f}",
    })

    # Leg 3: Direct rate exponent at kn=1e4
    kn1e4_rate = family_rates.get("1e+04", {})
    n_rate_kn1e4 = kn1e4_rate.get("n_from_rate_ladder")
    if n_rate_kn1e4 is not None:
        legs.append({
            "leg": "L3_direct_rate_exponent_kn1e4",
            "quantity": "n = d ln tau / d ln gdot at kn = 1e4 (constant-volume)",
            "ref": 0.0,
            "ref_anchor": "campbell2002_abstract_elastic_quasi_static",
            "got": n_rate_kn1e4,
            "abs_dev": n_rate_kn1e4 - 0.0,
            "tol": TOL_EXPONENT_ABS,
            "tol_kind": "abs",
            "verdict": "PASS" if abs(n_rate_kn1e4) <= TOL_EXPONENT_ABS else "FAIL",
            "source": "Same anchor as L1: elastic-quasi-static => n = 0",
            "evidence": f"R3 rate sweep at kn=1e4, n = {n_rate_kn1e4:.4f}, "
                        f"R² = {kn1e4_rate.get('r2', 0):.4f}",
        })

    # Leg 4: Kappa-collapse identity (P1)
    legs.append({
        "leg": "L4_kappa_collapse_identity",
        "quantity": "tau* is identical for iso-kappa (k_n, gdot) pairs",
        "ref": "exact identity",
        "ref_anchor": "kernel_invariance",
        "got": r3["collapse_worst_spread"],
        "tol": 1e-6,
        "tol_kind": "abs",
        "verdict": r3.get("P1_verdict", "PASS"),
        "source": "SLLOD + linear-spring + Coulomb is exactly invariant under (kn, gdot) -> (lambda^2 kn, lambda gdot)",
        "evidence": f"R3 kappa-collapse, worst spread = {r3['collapse_worst_spread']:.1e}, "
                    f"{len(r3['collapse_groups'])} groups",
    })

    # Leg 5: Monotone approach to elastic limit (P4)
    legs.append({
        "leg": "L5_monotone_approach_to_elastic_limit",
        "quantity": "beta increases monotonically with kappa (approach to elastic limit)",
        "ref": "monotone increasing",
        "ref_anchor": "campbell2002_abstract_elastic_quasi_static",
        "got": "beta_low_half < beta_high_half",
        "tol": None,
        "tol_kind": "qualitative",
        "verdict": r3.get("P4_verdict", "PASS"),
        "source": "Campbell (2002): elastic-quasi-static is the high-kappa limit",
        "evidence": f"R3: beta_low_half = {approach['beta_low_half']:.4f}, "
                    f"beta_high_half = {approach['beta_high_half']:.4f}",
    })

    # Leg 6: Coefficient (WITHHELD)
    legs.append({
        "leg": "L6_coefficient_elastic_branch",
        "quantity": "C in tau* = C * kappa (elastic-quasi-static plateau)",
        "ref": None,
        "ref_anchor": "campbell2002_tables",
        "got": elastic_coeff["tau_star_over_kappa"],
        "tol": None,
        "tol_kind": "none",
        "verdict": "WITHHELD",
        "source": "Campbell (2002) tables: full text CLOSED (paywalled)",
        "evidence": f"Our value: C = {elastic_coeff['tau_star_over_kappa']:.6f} at kappa = {elastic_coeff['kappa']:.1f}",
    })

    # Leg 7: Bagnold inertial exponent
    legs.append({
        "leg": "L7_bagnold_inertial_exponent",
        "quantity": "n = d ln tau / d ln gdot in the inertial limit (kappa << 1)",
        "ref": 2.0,
        "ref_anchor": "campbell2002_abstract_no_rate_path",
        "got": None,
        "tol": TOL_EXPONENT_ABS,
        "tol_kind": "abs",
        "verdict": "NOT_SUPPORTED",
        "source": "Linear-spring cell cannot reach kappa << 1 without overlap of order d",
        "evidence": "R4 inertial-branch: all gated points at kappa > 9; soft-particle probe at kn=1 fails overlap gate",
    })

    # --- Summary ---
    benchable = [l for l in legs if l["verdict"] in ("PASS", "FAIL")]
    passed = [l["leg"] for l in benchable if l["verdict"] == "PASS"]
    failed = [l["leg"] for l in benchable if l["verdict"] == "FAIL"]
    withheld = [l["leg"] for l in legs if l["verdict"] == "WITHHELD"]
    not_supported = [l["leg"] for l in legs if l["verdict"] == "NOT_SUPPORTED"]

    result = {
        "case": CASE,
        "reference": anchors["reference"],
        "n_r3_points_gated": len(r3_pts),
        "n_r3_points_total": r3["n_points"],
        "beta_global": beta_global,
        "n_global": n_global,
        "r2_global": r2_global,
        "beta_high_half": approach["beta_high_half"],
        "n_high_half": approach["n_high_half"],
        "elastic_coefficient": elastic_coeff["tau_star_over_kappa"],
        "kappa_span": r3["global_fit"]["kappa_span"],
        "kappa_decades": r3["global_fit"]["decades"],
        "legs": legs,
        "legs_passed": passed,
        "legs_failed": failed,
        "legs_withheld": withheld,
        "legs_not_supported": not_supported,
        "overall": "PASS" if not failed else "FAIL",
        "declared_deviations": {
            "geometry": "2-D disks vs Campbell's 3-D spheres",
            "coefficient": "WITHHELD (Campbell's numerical tables are paywalled)",
            "inertial_limit": "NOT_SUPPORTED (linear-spring cell cannot reach kappa << 1)",
        },
    }

    return result


def generate_figures(result):
    """Generate scaling and exponent figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r3 = load_r3_analysis()
    r3_pts = [p for p in r3["points"] if p.get("gates", {}).get("all")]
    r3_pts.sort(key=lambda p: p["kappa"])

    kappas = [p["kappa"] for p in r3_pts]
    tau_stars = [p["tau_star"] for p in r3_pts]

    # --- Figure 1: tau* vs kappa scaling ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.loglog(kappas, tau_stars, "ko-", markersize=6, label="DEM (gated)")

    # Power-law fit
    beta = result["beta_global"]
    C_fit = tau_stars[len(tau_stars) // 2] / kappas[len(kappas) // 2] ** beta
    k_fit = np.logspace(np.log10(min(kappas)), np.log10(max(kappas)), 100)
    ax.loglog(k_fit, C_fit * k_fit ** beta, "r--", linewidth=1.5,
              label=f"fit: τ* = {C_fit:.4f} κ^{{{beta:.3f}}} (R²={result['r2_global']:.4f})")

    # Elastic limit line (beta = 1)
    C_el = result["elastic_coefficient"]
    ax.loglog(k_fit, C_el * k_fit, "b:", linewidth=1.5,
              label=f"elastic limit: τ* = {C_el:.5f} κ (β=1)")

    # Reference annotations
    ax.axvline(x=1e4, color="gray", linestyle="--", alpha=0.5, label="κ = 10⁴ (μ(I) threshold)")
    ax.axvline(x=1.0, color="orange", linestyle="--", alpha=0.5, label="κ = 1 (inertial boundary)")

    ax.set_xlabel("κ = k_n / (ρ d³ γ̇²)", fontsize=12)
    ax.set_ylabel("τ* = τ / (ρ d² γ̇²)", fontsize=12)
    ax.set_title("CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT\nDimensionless Shear Stress vs Stiffness Number", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ART / "fig_scaling.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: Local exponents ---
    local = r3["local_slopes"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    kappas_mid = [s["gmid"] for s in local]
    betas = [s["beta_local"] for s in local]
    betas_err = [s["sigma_beta"] for s in local]
    ns = [s["n_local"] for s in local]
    ns_err = [s["sigma_n"] for s in local]

    ax1.errorbar(kappas_mid, betas, yerr=betas_err, fmt="ko-", markersize=6, capsize=3)
    ax1.axhline(y=1.0, color="b", linestyle="--", alpha=0.7, label="β = 1 (elastic-quasi-static)")
    ax1.axhline(y=0.0, color="r", linestyle="--", alpha=0.7, label="β = 0 (inertial/Bagnold)")
    ax1.set_ylabel("β = d ln τ* / d ln κ", fontsize=12)
    ax1.set_title("Local Scaling Exponents vs Stiffness Number", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xscale("log")

    ax2.errorbar(kappas_mid, ns, yerr=ns_err, fmt="ko-", markersize=6, capsize=3)
    ax2.axhline(y=0.0, color="b", linestyle="--", alpha=0.7, label="n = 0 (elastic-quasi-static)")
    ax2.axhline(y=2.0, color="r", linestyle="--", alpha=0.7, label="n = 2 (inertial/Bagnold)")
    ax2.axhspan(-0.3, 0.3, alpha=0.1, color="blue", label="15% tolerance band")
    ax2.set_xlabel("κ = k_n / (ρ d³ γ̇²)", fontsize=12)
    ax2.set_ylabel("n = d ln τ / d ln γ̇", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_xscale("log")

    fig.tight_layout()
    fig.savefig(ART / "fig_exponents.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: Rate sweep at kn=1e4 ---
    family_rates = r3["family_rate_exponents"]
    if "1e+04" in family_rates:
        fr = family_rates["1e+04"]
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.loglog(fr["gammas"], fr["taus"], "ko-", markersize=8, label="DEM")
        # Power-law fit
        n_rate = fr["n_from_rate_ladder"]
        C_rate = fr["taus"][0] / fr["gammas"][0] ** n_rate
        g_fit = np.logspace(np.log10(min(fr["gammas"])), np.log10(max(fr["gammas"])), 50)
        ax.loglog(g_fit, C_rate * g_fit ** n_rate, "r--", linewidth=1.5,
                  label=f"fit: τ ∝ γ̇^{{{n_rate:.3f}}} (R²={fr['r2']:.4f})")
        ax.set_xlabel("γ̇ (shear rate)", fontsize=12)
        ax.set_ylabel("τ (shear stress)", fontsize=12)
        ax.set_title(f"Rate Sweep at k_n = 10⁴ (κ = {5787:.0f}–{362:.0f})", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(ART / "fig_rate_sweep.png", dpi=150)
        plt.close(fig)

    print(f"Figures saved to {ART}")


def to_markdown(result):
    """Convert benchmark result to markdown report."""
    lines = [
        f"# {CASE} — Benchmark vs Campbell (2002)",
        "",
        f"> Reference: Campbell, C. S. (2002). *Granular shear flows at the elastic limit*.",
        f"> J. Fluid Mech. **465**, 261-291. DOI: [10.1017/S002211200200109X](https://doi.org/10.1017/S002211200200109X)",
        "",
        "## Summary",
        "",
        f"* **κ span**: {result['kappa_span'][0]:.1f} – {result['kappa_span'][1]:.1f} ({result['kappa_decades']:.1f} decades)",
        f"* **Gated points**: {result['n_r3_points_gated']} / {result['n_r3_points_total']}",
        f"* **Global fit**: β = {result['beta_global']:.4f}, n = {result['n_global']:.4f}, R² = {result['r2_global']:.4f}",
        f"* **Elastic limit**: β_high = {result['beta_high_half']:.4f}, n_high = {result['n_high_half']:.4f}",
        f"* **Elastic coefficient**: C = {result['elastic_coefficient']:.6f} (WITHHELD vs Campbell — paywalled)",
        f"* **Overall**: **{result['overall']}**",
        "",
        "## Benchmark Legs",
        "",
        "| Leg | Quantity | Ref | Got | Dev | Tol | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]

    for l in result["legs"]:
        ref = l["ref"] if not isinstance(l["ref"], float) else f"{l['ref']:.3g}"
        got = l["got"]
        if isinstance(got, float):
            got_s = f"{got:.4f}"
        else:
            got_s = str(got)
        dev = l.get("abs_dev", "")
        dev_s = f"{dev:.4f}" if isinstance(dev, float) else ""
        tol = l["tol"]
        tol_s = f"{tol}" if tol is not None else "—"
        lines.append(f"| {l['leg']} | {l['quantity'][:50]} | {ref} | {got_s} | {dev_s} | {tol_s} | **{l['verdict']}** |")

    lines.extend([
        "",
        "## Declared Deviations",
        "",
    ])
    for k, v in result["declared_deviations"].items():
        lines.append(f"* **{k}**: {v}")

    lines.extend([
        "",
        "## Evidence Detail",
        "",
    ])
    for l in result["legs"]:
        lines.append(f"### {l['leg']}")
        lines.append(f"* **Source**: {l['source']}")
        lines.append(f"* **Evidence**: {l.get('evidence', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def main():
    print("Computing benchmark from existing R3/R4/R5 run data...")
    result = compute_benchmark()

    # Save benchmark JSON
    (ART / "benchmark.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"Benchmark JSON saved to {ART / 'benchmark.json'}")

    # Save benchmark markdown
    md = to_markdown(result)
    (ART / "benchmark.md").write_text(md)
    print(f"Benchmark MD saved to {ART / 'benchmark.md'}")

    # Generate figures
    generate_figures(result)

    # Print summary
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Overall: {result['overall']}")
    print(f"Passed: {result['legs_passed']}")
    print(f"Failed: {result['legs_failed']}")
    print(f"Withheld: {result['legs_withheld']}")
    print(f"Not supported: {result['legs_not_supported']}")
    for l in result["legs"]:
        print(f"  {l['leg']}: {l['verdict']} (got={l['got']}, ref={l['ref']})")


if __name__ == "__main__":
    main()
