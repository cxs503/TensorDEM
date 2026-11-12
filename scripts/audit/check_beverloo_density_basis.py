#!/usr/bin/env python3
"""Reusable linter: is a Beverloo benchmark artifact's rel_err on a consistent
density basis?

The Beverloo (1961) correlation ``W = C · ρ_b · √g · (D − k·d)^{5/2}`` is linear
in the *in-situ* bulk density of the bed that is draining.  A DEM artifact that
pairs a measured flow rate with a prediction evaluated at a different ρ_b
(e.g. the ``0.64 ρ_p`` literature packing reference while the simulated bed sits
at φ≈0.40) reports a relative error that is biased low by exactly
``ρ_b(pred) / ρ_b(measured)``.

This tool recomputes both bases straight from the artifact's own case/results
blocks, so it can be run against any mission's JSON record (m15259, m15381, …)
and against future re-runs.

Usage
-----
    python scripts/audit/check_beverloo_density_basis.py ARTIFACT.json [ARTIFACT2.json ...]
    python scripts/audit/check_beverloo_density_basis.py --json out.json ARTIFACT.json

Exit code 0 = every artifact passes |rel_err| < tol on BOTH bases.
Exit code 1 = at least one artifact's verdict flips when the density basis is
              made consistent (the "PASS" is a normalization artifact).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

C_LIT = 0.58
K_LIT = 1.4
PHI_LIT = 0.64          # hard-coded in tensordem.beverloo_discharge.beverloo_prediction
TOL = 0.15


def audit_record(rec: dict, tol: float = TOL) -> dict:
    case, res = rec["case"], rec["results"]
    d, D = case["particle_diameter_m"], case["orifice_diameter_m"]
    rho_p, g = case["particle_density_kg_m3"], case["gravity_m_s2"]
    k = rec.get("reference", {}).get("k", K_LIT)
    c = rec.get("reference", {}).get("C", C_LIT)
    der = D - k * d
    w = res["flow_rate_kg_s"]

    def basis(rho_b: float) -> dict:
        w_ref = c * rho_b * math.sqrt(g) * der ** 2.5
        err = (w - w_ref) / w_ref
        return {"rho_b": rho_b, "W_ref_kg_s": w_ref, "rel_err": err,
                "abs_pass": abs(err) < tol,
                "recovered_C": w / (rho_b * math.sqrt(g) * der ** 2.5)}

    lit = basis(PHI_LIT * rho_p)
    cons = basis(res["bulk_density_kg_m3"])
    win = res["flow_rate_windows_kg_s"]
    mean_s = sum(win[1:]) / max(1, len(win) - 1)
    return {
        "artifact": rec.get("case_label"),
        "D_over_d": D / d,
        "geometry_in_beverloo_regime": D / d >= 4.0,
        "phi_measured": res["packing_fraction"],
        "density_inflation": lit["rho_b"] / cons["rho_b"],
        "as_filed": lit, "consistent": cons,
        "verdict_flips": lit["abs_pass"] != cons["abs_pass"],
        "window_swing": {"mean_all": sum(win) / len(win), "mean_excl_startup": mean_s,
                         "rel_err_excl_startup_consistent": (mean_s - cons["W_ref_kg_s"]) / cons["W_ref_kg_s"]},
        "mass_residual_kg": res["mass_balance_residual_kg"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="+")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    out, bad = [], 0
    for a in args.artifacts:
        rec = json.loads(Path(a).read_text())
        r = audit_record(rec, args.tol)
        r["source"] = a
        out.append(r)
        bad += int(r["verdict_flips"] or not r["consistent"]["abs_pass"])
        print(f"{a} [{r['artifact']}] D/d={r['D_over_d']:.1f} "
              f"(regime ok={r['geometry_in_beverloo_regime']}) phi={r['phi_measured']:.3f} "
              f"rho_inflation=x{r['density_inflation']:.2f}")
        print(f"   as filed   : rho_b={r['as_filed']['rho_b']:.1f} "
              f"W_ref={r['as_filed']['W_ref_kg_s']:.5f} rel_err={100*r['as_filed']['rel_err']:+.2f}% "
              f"C={r['as_filed']['recovered_C']:.4f} -> {'PASS' if r['as_filed']['abs_pass'] else 'FAIL'}")
        print(f"   consistent : rho_b={r['consistent']['rho_b']:.1f} "
              f"W_ref={r['consistent']['W_ref_kg_s']:.5f} rel_err={100*r['consistent']['rel_err']:+.2f}% "
              f"C={r['consistent']['recovered_C']:.4f} -> {'PASS' if r['consistent']['abs_pass'] else 'FAIL'}")
        if r["verdict_flips"]:
            print("   !! VERDICT FLIPS with a consistent density basis "
                  "(the filed PASS is a normalization artifact)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())