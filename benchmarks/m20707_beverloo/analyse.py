#!/usr/bin/env python3
"""Analyze the m20707 Beverloo campaign results.

Reads the per-case JSON artefacts from ``benchmarks/m20707_beverloo/results/``
(one directory per preset), checks the predeclared acceptance criteria, fits the
Beverloo law and emits comparison tables (console + optional markdown) plus one
combined machine-readable summary.

Every number printed or written by this script is recomputed from the case
artefacts, so the verification note can be regenerated instead of hand-edited.

Usage
-----
    PYTHONPATH=src python benchmarks/m20707_beverloo/analyse.py            # both presets
    PYTHONPATH=src python benchmarks/m20707_beverloo/analyse.py --preset full
    PYTHONPATH=src python benchmarks/m20707_beverloo/analyse.py \\
        --md-out /tmp/tables.md \\
        --summary-out benchmarks/m20707_beverloo/results/analysis_summary.json
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


REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"
RESULTS_ROOT = ROOT / "benchmarks" / "m20707_beverloo" / "results"

# ---- predeclared tolerances (fixed before the campaign was run) -------------
TOL_REL = 0.10            # |W - W_ref| / W_ref  (primary decisive criterion)
TOL_EXPONENT = 0.20       # |p - 5/2|
TOL_GRAVITY = 0.10        # |W(g/4)/W(g) - 0.5|
TOL_CV = 0.05             # window-rate coefficient of variation

# A case off by more than this is outside the correlation's applicability band
# (arching / no steady discharge) and is excluded from the exponent fit.  Points
# that are excluded are always still reported in Table 1.
LAW_VALIDITY_BAND = 50.0

PRIMARY = "D5"
LAW_CASES = ["D4", "D5", "D6"]


def load_cases(run_dir: Path) -> dict:
    cases = {}
    for p in sorted(Path(run_dir).glob("*.json")):
        d = json.loads(p.read_text())
        if "case_id" not in d:      # summary.json & friends
            continue
        cases[d["case_id"]] = d
    return cases


def _fit(D: np.ndarray, W: np.ndarray, dpart: float, rho: float, k: float) -> dict:
    """Least-squares ``ln W = ln A + p ln(D - k d)`` with C = A / (rho sqrt g)."""
    eff = np.asarray(D, dtype=np.float64) - k * dpart
    x = np.log(eff)
    y = np.log(np.asarray(W, dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    amplitude = float(np.exp(intercept))
    return {
        "k": float(k),
        "exponent": float(slope),
        "amplitude": amplitude,
        "c": amplitude / (rho * np.sqrt(9.81)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan"),
        "n_points": int(x.size),
        "residual_rms_log": float(np.sqrt(ss_res / max(x.size - 2, 1))),
    }


def _best_free_k(D: np.ndarray, W: np.ndarray, dpart: float, rho: float,
                 k_max: float = 3.0, n: int = 301) -> dict:
    """Free-k fit. A solution on the ``k_max`` boundary is not identified."""
    best = None
    for k in np.linspace(0.0, k_max, n):
        if np.any(np.asarray(D) - k * dpart <= 0.0):
            continue
        f = _fit(D, W, dpart, rho, k)
        if best is None or f["r2"] > best["r2"]:
            best = f
    if best is not None:
        best["at_boundary"] = bool(best["k"] >= k_max - (k_max / (n - 1)) * 1.5)
    return best


def analyse_preset(run_dir: Path, preset: str) -> dict:
    ref = json.loads(REF_PATH.read_text())
    C_ref = float(ref["coefficients"]["C"])
    k_ref = float(ref["coefficients"]["k"])
    cases = load_cases(run_dir)
    if not cases:
        print(f"[{preset}] no results in {run_dir}")
        return {}

    try:
        shown = Path(run_dir).resolve().relative_to(ROOT)
    except ValueError:
        shown = Path(run_dir)
    md = [f"### Preset `{preset}` — run directory `{shown}`\n"]
    md.append("| case | D/d | μ | g (m/s²) | n_bed | ρ_b (kg/m³) | W_DEM (kg/s) "
              "| σ_W (kg/s) | n_steady | W_Beverloo (kg/s) | rel. err | C_rec | <10 %? |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    print(f"\n=== [{preset}] Table 1: flow rate vs Beverloo ===")
    for cid in sorted(cases, key=lambda c: (cases[c]["case"]["D_over_d"], c)):
        c = cases[cid]
        bev, cs, res = c["beverloo"], c["case"], c["result"]
        md.append(
            f"| {cid} | {cs['D_over_d']:.1f} | {cs['mu']:.2f} | {c['gravity']:.3f} "
            f"| {res['n_bed']} | {res['rho_bulk']:.0f} | {res['flow_rate_kg_s']:.6f} "
            f"| {res['flow_rate_std_kg_s']:.6f} "
            f"| {res.get('flow_rate_steady_windows', max(1, len(res['flow_rate_windows_kg_s']) // 2))} "
            f"| {bev['W_ref_kg_s']:.6f} "
            f"| {bev['rel_err_pct']:+.1f} % | {bev['C_recovered']:.3f} "
            f"| {'yes' if bev['pass_10pct'] else 'no'} |"
        )
        print(f"  {cid:<9} D/d={cs['D_over_d']:.1f} mu={cs['mu']:.2f} "
              f"W={res['flow_rate_kg_s']:.6f} W_ref={bev['W_ref_kg_s']:.6f} "
              f"rel={bev['rel_err_pct']:+.1f}% C_rec={bev['C_recovered']:.3f} "
              f"{'PASS' if bev['pass_10pct'] else 'FAIL'}")

    summary = {"preset": preset, "run_dir": str(shown), "cases": cases}

    # ---------------- exponent / power-law check ----------------
    law_all = [cid for cid in LAW_CASES if cid in cases]
    law = [cid for cid in law_all
           if abs(cases[cid]["beverloo"]["rel_err_pct"]) <= LAW_VALIDITY_BAND]
    excluded = [cid for cid in law_all if cid not in law]

    if len(law_all) >= 2:
        D_all = np.array([cases[cid]["case"]["orifice_diameter"] for cid in law_all])
        W_all = np.array([cases[cid]["result"]["flow_rate_kg_s"] for cid in law_all])
        rho = float(np.mean([cases[cid]["result"]["rho_bulk"] for cid in law_all]))
        dpart = cases[law_all[0]]["case"]["particle_diameter"]

        fit_all = _fit(D_all, W_all, dpart, rho, k_ref)      # all points, incl. arching
        D = np.array([cases[cid]["case"]["orifice_diameter"] for cid in law])
        W = np.array([cases[cid]["result"]["flow_rate_kg_s"] for cid in law])
        fit_use = _fit(D, W, dpart, rho, k_ref) if len(law) >= 2 else None
        free_k = _best_free_k(D, W, dpart, rho) if len(law) >= 3 else None

        exp_err = abs(fit_use["exponent"] - 2.5) if fit_use else float("nan")
        determined = len(law) >= 3
        if not determined:
            status = "NOT_DETERMINED"
        else:
            status = "PASS" if exp_err <= TOL_EXPONENT else "FAIL"

        print(f"\n=== [{preset}] Power-law fit (exponent should be 5/2) ===")
        if excluded:
            print(f"  outside validity band (|err| > {LAW_VALIDITY_BAND:.0f} %), "
                  f"excluded from fit: {', '.join(excluded)}")
        print(f"  all {len(law_all)} points, k={k_ref}: p={fit_all['exponent']:.3f} "
              f"R2={fit_all['r2']:.4f}")
        if fit_use:
            print(f"  {len(law)} usable points, k={k_ref}: p={fit_use['exponent']:.3f} "
                  f"|p-2.5|={exp_err:.3f} -> {status} (tol {TOL_EXPONENT}), "
                  f"C={fit_use['c']:.3f} R2={fit_use['r2']:.4f}")
        if free_k:
            print(f"  free k: k={free_k['k']:.3f} p={free_k['exponent']:.3f} "
                  f"C={free_k['c']:.3f} R2={free_k['r2']:.4f}"
                  + ("  [at k_max boundary -> not identified]" if free_k["at_boundary"] else ""))

        md.append("")
        md.append("| fit | points used | exponent p | C | k | R² | verdict |")
        md.append("|---|---|---|---|---|---|---|")
        md.append(f"| fixed k = {k_ref}, all sweep points | "
                  f"{', '.join(law_all)} | {fit_all['exponent']:.3f} | n/a "
                  f"| {k_ref} | {fit_all['r2']:.4f} | diagnostic (arching point included) |")
        if fit_use:
            md.append(f"| fixed k = {k_ref}, usable points | {', '.join(law)} "
                      f"| {fit_use['exponent']:.3f} | {fit_use['c']:.3f} | {k_ref} "
                      f"| {fit_use['r2']:.4f} | **{status}** (|p−2.5| = {exp_err:.3f}, "
                      f"tol {TOL_EXPONENT}) |")
        if free_k:
            md.append(f"| free k | {', '.join(law)} | {free_k['exponent']:.3f} "
                      f"| {free_k['c']:.3f} | {free_k['k']:.3f} | {free_k['r2']:.4f} "
                      f"| diagnostic{' (k at grid boundary)' if free_k['at_boundary'] else ''} |")

        summary["exponent_check"] = {
            "cases_all": law_all, "cases_used": law, "cases_excluded": excluded,
            "validity_band_pct": LAW_VALIDITY_BAND,
            "exponent_all_points": fit_all["exponent"],
            "exponent_used_points": fit_use["exponent"] if fit_use else None,
            "target": 2.5, "deviation": exp_err, "tolerance": TOL_EXPONENT,
            "status": status, "n_points_used": len(law),
            "c_used_points": fit_use["c"] if fit_use else None,
            "r2_used_points": fit_use["r2"] if fit_use else None,
            "free_k": free_k,
        }

    # ---------------- gravity scaling ----------------
    if "D5" in cases and "D5_g4" in cases:
        W5 = cases["D5"]["result"]["flow_rate_kg_s"]
        W5g4 = cases["D5_g4"]["result"]["flow_rate_kg_s"]
        ratio = W5g4 / W5 if W5 > 0 else float("nan")
        grav_pass = abs(ratio - 0.5) < TOL_GRAVITY
        print(f"\n=== [{preset}] Gravity scaling ===")
        print(f"  W(g/4)/W(g) = {ratio:.3f} (expected 0.500) -> "
              f"{'PASS' if grav_pass else 'FAIL'} (tol ±{TOL_GRAVITY})")
        md.append("")
        md.append("| check | measured | expected | deviation | verdict |")
        md.append("|---|---|---|---|---|")
        md.append(f"| W(g/4)/W(g) | {ratio:.3f} | 0.500 | {abs(ratio - 0.5) * 100:.1f} % "
                  f"| {'PASS' if grav_pass else 'FAIL'} |")
        summary["gravity_scaling"] = {
            "W_g4": W5g4, "W_g": W5, "ratio": ratio, "expected": 0.5,
            "tolerance": TOL_GRAVITY, "pass": grav_pass,
        }

    # ---------------- window steadiness ----------------
    print(f"\n=== [{preset}] Window steadiness (CV of window rates, tol {TOL_CV}) ===")
    md.append("")
    md.append("| case | windows (kg/s) | CV | steady? |")
    md.append("|---|---|---|---|")
    cvs = {}
    for cid in sorted(cases):
        wins = cases[cid]["result"]["flow_rate_windows_kg_s"]
        if len(wins) < 2:
            continue
        cv = float(np.std(wins, ddof=1) / max(np.mean(wins), 1e-30))
        cvs[cid] = cv
        ok = cv < TOL_CV
        print(f"  {cid:<9} CV={cv:.3f} -> {'PASS' if ok else 'FAIL'}")
        md.append(f"| {cid} | {', '.join(f'{w:.6f}' for w in wins)} | {cv:.3f} "
                  f"| {'yes' if ok else 'no'} |")
    summary["window_cv"] = cvs
    summary["window_cv_pass"] = all(v < TOL_CV for v in cvs.values()) if cvs else None

    # ---------------- friction sensitivity ----------------
    mu_cases = [cid for cid in cases if cid.startswith("D5_mu")] + \
               (["D5"] if "D5" in cases else [])
    if len(mu_cases) >= 2:
        print(f"\n=== [{preset}] Friction sensitivity ===")
        md.append("")
        md.append("| μ | C recovered | rel. err vs Beverloo |")
        md.append("|---|---|---|")
        for cid in sorted(mu_cases, key=lambda c: cases[c]["case"]["mu"]):
            c = cases[cid]
            print(f"  mu={c['case']['mu']:.2f}: C={c['beverloo']['C_recovered']:.3f} "
                  f"err={c['beverloo']['rel_err_pct']:+.1f}%")
            md.append(f"| {c['case']['mu']:.2f} | {c['beverloo']['C_recovered']:.3f} "
                      f"| {c['beverloo']['rel_err_pct']:+.1f} % |")
        summary["friction_sensitivity"] = {
            cid: {"mu": cases[cid]["case"]["mu"],
                  "C_recovered": cases[cid]["beverloo"]["C_recovered"],
                  "rel_err_pct": cases[cid]["beverloo"]["rel_err_pct"]}
            for cid in mu_cases
        }

    # ---------------- verdict ----------------
    if PRIMARY in cases:
        bev = cases[PRIMARY]["beverloo"]
        verdict = "PASS" if bev["pass_10pct"] else "FAIL"
        print(f"\n=== [{preset}] VERDICT (primary case {PRIMARY}) ===")
        print(f"  W_DEM   = {cases[PRIMARY]['result']['flow_rate_kg_s']:.6f} kg/s")
        print(f"  W_ref   = {bev['W_ref_kg_s']:.6f} kg/s")
        print(f"  rel err = {bev['rel_err_pct']:+.1f} %")
        print(f"  C_rec   = {bev['C_recovered']:.3f} (literature {C_ref})")
        print(f"  Verdict = {verdict} (tolerance < {TOL_REL * 100:.0f} %)")
        summary["verdict"] = {
            "case": PRIMARY,
            "W_dem": cases[PRIMARY]["result"]["flow_rate_kg_s"],
            "sigma_W": cases[PRIMARY]["result"]["flow_rate_std_kg_s"],
            "W_ref": bev["W_ref_kg_s"],
            "rel_err_pct": bev["rel_err_pct"],
            "C_recovered": bev["C_recovered"],
            "C_literature": C_ref,
            "pass_10pct": bev["pass_10pct"],
            "verdict": verdict,
        }

    summary["markdown"] = "\n".join(md)
    return summary


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="both", choices=["full", "pilot", "both"])
    ap.add_argument("--run-dir", default=None, help="override run directory (single preset)")
    ap.add_argument("--md-out", default=None, help="write the markdown tables here")
    ap.add_argument("--summary-out", default=None,
                    help="write one combined, machine-readable summary here")
    args = ap.parse_args(argv)

    presets = ["full", "pilot"] if args.preset == "both" else [args.preset]
    results, md_chunks = {}, []
    for preset in presets:
        run_dir = Path(args.run_dir) if args.run_dir else RESULTS_ROOT / preset
        summary = analyse_preset(run_dir, preset)
        if summary:
            results[preset] = summary
            md_chunks.append(summary["markdown"])

    if results and args.summary_out:
        payload = {p: {k: v for k, v in s.items() if k != "markdown"}
                   for p, s in results.items()}
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nCombined summary written to {out}")

    if args.md_out and md_chunks:
        Path(args.md_out).write_text("\n\n".join(md_chunks) + "\n")
        print(f"Markdown tables written to {args.md_out}")
    return results


if __name__ == "__main__":
    main()