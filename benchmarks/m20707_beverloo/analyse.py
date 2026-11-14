#!/usr/bin/env python3
"""Analyze the m20707 Beverloo campaign results.

Reads the JSON artefacts from ``runs/m20707_beverloo/``, fits the Beverloo
law, produces comparison tables and figures, and writes a summary JSON.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.silo_flow.beverloo import (
    BEVERLOO_C, BEVERLOO_K, beverloo_prediction, fit_law,
)

REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"


def main(run_dir: str | None = None) -> dict:
    run_dir = Path(run_dir or ROOT / "runs" / "m20707_beverloo")
    ref = json.loads(REF_PATH.read_text())
    C_ref = float(ref["coefficients"]["C"])
    k_ref = float(ref["coefficients"]["k"])

    # Load all case results
    cases = {}
    for p in sorted(run_dir.glob("*.json")):
        d = json.loads(p.read_text())
        cases[d["case_id"]] = d

    if not cases:
        print("No results found in", run_dir)
        return {}

    # ---- Table 1: primary results ----
    print("\n=== Table 1: Flow rate vs Beverloo ===")
    print(f"{'case':<10} {'D/d':>5} {'mu':>5} {'g':>6} {'n_bed':>6} "
          f"{'rho_b':>7} {'W_dem':>10} {'W_ref':>10} {'rel%':>7} "
          f"{'C_rec':>6} {'pass':>5}")
    print("-" * 90)

    primary_cases = ["D4", "D5", "D6"]
    law_D = []
    law_W = []
    law_rho = []

    for cid in sorted(cases):
        c = cases[cid]
        bev = c["beverloo"]
        cs = c["case"]
        res = c["result"]
        print(
            f"{cid:<10} {cs['D_over_d']:>5.1f} {cs['mu']:>5.2f} "
            f"{c['gravity']:>6.2f} {res['n_bed']:>6} "
            f"{res['rho_bulk']:>7.1f} {res['flow_rate_kg_s']:>10.6f} "
            f"{bev['W_ref_kg_s']:>10.6f} {bev['rel_err_pct']:>+7.1f} "
            f"{bev['C_recovered']:>6.3f} "
            f"{'PASS' if bev['pass_10pct'] else 'FAIL':>5}"
        )
        if cid in primary_cases:
            law_D.append(cs["orifice_diameter"])
            law_W.append(res["flow_rate_kg_s"])
            law_rho.append(res["rho_bulk"])

    # ---- Power-law fit ----
    summary = {"cases": {cid: cases[cid] for cid in cases}}

    if len(law_D) >= 3:
        D = np.array(law_D)
        W = np.array(law_W)
        rho = float(np.mean(law_rho))
        dpart = cases[primary_cases[0]]["case"]["particle_diameter"]

        fits = fit_law(D, dpart, W, rho)
        summary["fit"] = fits

        print("\n=== Power-law fit ===")
        fk = fits["fixed_k"]
        frk = fits["free_k"]
        print(f"Fixed k={BEVERLOO_K}: exponent={fk['exponent']:.3f} "
              f"(target 2.5), C={fk['c']:.3f} (target {C_ref}), "
              f"R²={fk['r2']:.4f}")
        print(f"Free k:           exponent={frk['exponent']:.3f}, "
              f"k={frk['k']:.3f}, C={frk['c']:.3f}, R²={frk['r2']:.4f}")

        # Exponent check
        exp_err = abs(fk["exponent"] - 2.5)
        exp_pass = exp_err <= 0.2
        print(f"\nExponent deviation from 5/2: {exp_err:.3f} "
              f"({'PASS' if exp_pass else 'FAIL'}, tolerance 0.2)")

        summary["exponent_check"] = {
            "exponent_fixed_k": fk["exponent"],
            "exponent_free_k": frk["exponent"],
            "target": 2.5,
            "deviation": exp_err,
            "pass": exp_pass,
            "r2_fixed_k": fk["r2"],
            "c_fixed_k": fk["c"],
            "c_free_k": frk["c"],
            "k_free": frk["k"],
        }

    # ---- Gravity scaling check ----
    if "D5" in cases and "D5_g4" in cases:
        W5 = cases["D5"]["result"]["flow_rate_kg_s"]
        W5g4 = cases["D5_g4"]["result"]["flow_rate_kg_s"]
        ratio = W5g4 / W5 if W5 > 0 else float("nan")
        expected = 0.5  # sqrt(g/4) / sqrt(g) = 0.5
        print(f"\n=== Gravity scaling ===")
        print(f"W(g/4)/W(g) = {ratio:.3f} (expected 0.500)")
        grav_pass = abs(ratio - 0.5) < 0.10
        print(f"{'PASS' if grav_pass else 'FAIL'} (tolerance ±0.10)")
        summary["gravity_scaling"] = {
            "W_g4": W5g4,
            "W_g": W5,
            "ratio": ratio,
            "expected": 0.5,
            "pass": grav_pass,
        }

    # ---- Friction sensitivity ----
    mu_cases = [cid for cid in cases if cid.startswith("D5_mu") or cid == "D5"]
    if len(mu_cases) >= 2:
        print(f"\n=== Friction sensitivity ===")
        for cid in sorted(mu_cases):
            c = cases[cid]
            print(f"  {cid}: mu={c['case']['mu']:.2f}, "
                  f"C={c['beverloo']['C_recovered']:.3f}, "
                  f"err={c['beverloo']['rel_err_pct']:+.1f}%")

    # ---- Window steadiness ----
    print(f"\n=== Window steadiness ===")
    for cid in sorted(cases):
        c = cases[cid]
        wins = c["result"]["flow_rate_windows_kg_s"]
        if len(wins) > 1:
            cv = np.std(wins, ddof=1) / max(np.mean(wins), 1e-30)
            print(f"  {cid}: CV={cv:.3f}, windows={[f'{w:.6f}' for w in wins]}")

    # ---- Overall verdict ----
    primary = cases.get("D5")
    if primary:
        bev = primary["beverloo"]
        verdict = "PASS" if bev["pass_10pct"] else "FAIL"
        print(f"\n=== VERDICT (primary case D5) ===")
        print(f"W = {primary['result']['flow_rate_kg_s']:.6f} kg/s")
        print(f"W_ref = {bev['W_ref_kg_s']:.6f} kg/s")
        print(f"Relative error = {bev['rel_err_pct']:+.1f}%")
        print(f"C recovered = {bev['C_recovered']:.3f} (literature {C_ref})")
        print(f"Verdict: {verdict} (tolerance <10%)")
        summary["verdict"] = {
            "case": "D5",
            "W_dem": primary["result"]["flow_rate_kg_s"],
            "W_ref": bev["W_ref_kg_s"],
            "rel_err_pct": bev["rel_err_pct"],
            "C_recovered": bev["C_recovered"],
            "C_literature": C_ref,
            "pass_10pct": bev["pass_10pct"],
            "verdict": verdict,
        }

    # Write summary
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSummary written to {summary_path}")

    return summary


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
