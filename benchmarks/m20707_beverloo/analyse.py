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
import hashlib
import json
import sys
from pathlib import Path
from typing import List

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


# ---------------------------------------------------------------------------
# Document injection: the verification note is *rendered* from the committed
# artefacts instead of being hand-edited, so a number in the note cannot drift
# away from the run that produced it.  ``--check-doc`` is the guard the pytest
# suite calls; any hand edit inside a generated block makes it fail.
# ---------------------------------------------------------------------------
DOC_PATH = ROOT / "docs" / "m20707_beverloo_verification.md"
BEGIN_FMT = "<!-- BEGIN:GENERATED {name} (rendered by analyse.py --inject-doc) -->"
END_FMT = "<!-- END:GENERATED {name} -->"
SKIP_ARTEFACTS = {"summary.json", "analysis_summary.json"}

# Steady mass-flow rates printed by the *pre-binding* revision of the note
# (git b2a67b44, i.e. before the artefacts were tracked and the note rendered).
# They are pinned here so the reproduction check below is a measurement of the
# run against the published claim, not a comparison of the note with itself.
PUBLISHED_BASELINE = {
    "D4": 0.000458, "D5": 0.009817, "D5_g4": 0.004974,
    "D5_mu40": 0.010799, "D5_mu55": 0.010930, "D6": 0.020616,
}
TOL_REPRODUCTION = 0.1        # predeclared |ΔW| / W in percent


def artefact_digest(run_dir: Path) -> str:
    """sha256 over the sorted per-case artefacts, truncated to 12 hex chars."""
    h = hashlib.sha256()
    for p in sorted(Path(run_dir).glob("*.json")):
        if p.name in SKIP_ARTEFACTS:
            continue
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _rel_to_repo(path) -> str:
    """Path relative to the repo root when possible (stable in the note)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:          # artefacts staged outside the checkout
        return p.as_posix()


def _steady_windows(case: dict) -> int:
    res = case["result"]
    return int(res.get("flow_rate_steady_windows",
                       max(1, len(res["flow_rate_windows_kg_s"]) // 2)))


def _arrow(ok: bool) -> str:
    return "yes" if ok else "no"


def render_results_block(results: dict) -> str:
    """Render §3-§5 of the note (results, error judgement, full-preset status)."""
    pilot = results.get("pilot")
    full = results.get("full")
    if not pilot:
        raise SystemExit("no pilot artefacts to render the note from")
    cases = pilot["cases"]
    lines: List[str] = []

    lines.append("## 3. Results — preset `pilot` (d = 2 mm)")
    lines.append("")
    lines.append(
        "Everything in §3–§5 is rendered by `benchmarks/m20707_beverloo/analyse.py "
        "--inject-doc` from the per-case artefacts committed next to the script; hand "
        "editing these blocks makes `--check-doc` (and "
        "`tests/test_m20707_beverloo_campaign.py`) fail."
    )
    lines.append("")
    lines.append("| provenance | value |")
    lines.append("|---|---|")
    lines.append(f"| run directory | `{_rel_to_repo(pilot['run_dir'])}` |")
    lines.append(f"| artefacts | {', '.join(sorted(c + '.json' for c in cases))} |")
    lines.append(f"| artefact fingerprint (sha256[:12]) | `{artefact_digest(Path(pilot['run_dir']))}` |")
    revs = sorted({str(c.get('git_sha', 'unknown')) for c in cases.values()})
    hosts = sorted({str(c.get('host', 'unknown')) for c in cases.values()})
    lines.append(f"| producing git revision | {', '.join('`' + r + '`' for r in revs)} |")
    lines.append(f"| host | {', '.join(hosts)} |")
    wall = ", ".join(f"{cid} {cases[cid].get('wall_clock_s', float('nan')):.0f}"
                     for cid in sorted(cases))
    lines.append(f"| wall clock (s) | {wall} |")
    lines.append("")
    lines.append("### Table 1a — reproduction of the pre-binding revision")
    lines.append("")
    lines.append("Pinned values are the rates printed by the note before the artefacts were")
    lines.append(f"tracked; the predeclared reproduction criterion is |ΔW| < {TOL_REPRODUCTION} %.")
    lines.append("")
    lines.append("| case | W published (pre-binding note) | W this run | ΔW | reproduced? |")
    lines.append("|---|---|---|---|---|")
    repro_ok = True
    for cid in sorted(PUBLISHED_BASELINE):
        if cid not in cases:
            continue
        w_now = float(cases[cid]["result"]["flow_rate_kg_s"])
        w_pub = float(PUBLISHED_BASELINE[cid])
        delta = (w_now - w_pub) / w_pub * 100.0 if w_pub else float("nan")
        ok = abs(delta) < TOL_REPRODUCTION
        repro_ok = repro_ok and ok
        lines.append(f"| {cid} | {w_pub:.6f} kg/s | {w_now:.6f} kg/s | {delta:+.4f} % "
                     f"| {_arrow(ok)} |")
    lines.append("")
    lines.append(f"Reproduction verdict: **{'PASS' if repro_ok else 'FAIL'}** "
                 f"(all cases within {TOL_REPRODUCTION} %).")
    lines.append("")
    lines.append("### Table 1 — flow rate vs Beverloo")
    lines.append("")
    lines.append("| case | D/d | μ | g (m/s²) | n_bed | ρ_b (kg/m³) | W_DEM (kg/s) "
                 "| σ_W (kg/s) | n_steady | W_Beverloo (kg/s) | rel. err | C_rec | <10 %? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid in sorted(cases, key=lambda c: (cases[c]["case"]["D_over_d"], c)):
        c = cases[cid]
        cs, res, bev = c["case"], c["result"], c["beverloo"]
        lines.append(
            f"| {cid} | {cs['D_over_d']:.1f} | {cs['mu']:.2f} | {c['gravity']:.3f} "
            f"| {res['n_bed']} | {res['rho_bulk']:.0f} | {res['flow_rate_kg_s']:.6f} "
            f"| {res['flow_rate_std_kg_s']:.6f} | {_steady_windows(c)} "
            f"| {bev['W_ref_kg_s']:.6f} | {bev['rel_err_pct']:+.1f} % "
            f"| {bev['C_recovered']:.3f} | {_arrow(bev['pass_10pct'])} |"
        )
    lines.append("")
    lines.append("`W_Beverloo` uses the *measured* ρ_b of the same case, with C and k from "
                 "`benchmarks/m20055_beverloo/ref_beverloo_1961.json`.")
    lines.append("")

    # ---- power-law / exponent check -------------------------------------
    exp = pilot.get("exponent_check")
    lines.append("### Table 2 — power-law check")
    lines.append("")
    if exp:
        k_fix = float(cases[exp["cases_used"][0]]["beverloo"]["k_ref"]) \
            if exp["cases_used"] else 1.4
        lines.append("| fit | points used | exponent p | C | k | R² | verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        all_pts = ", ".join(exp["cases_used"]
                            + [f"{c} (excluded)" for c in exp["cases_excluded"]])
        lines.append(
            f"| fixed k = {k_fix:.1f}, all sweep points | {all_pts} "
            f"| {exp['exponent_all_points']:.3f} | n/a | {k_fix:.1f} | n/a "
            f"| diagnostic (points outside the validity band included) |"
        )
        if exp.get("exponent_used_points") is not None:
            lines.append(
                f"| fixed k = {k_fix:.1f}, usable points | {', '.join(exp['cases_used'])} "
                f"| {exp['exponent_used_points']:.3f} | {exp['c_used_points']:.3f} "
                f"| {k_fix:.1f} | {exp['r2_used_points']:.4f} "
                f"| **{exp['status']}** (|p−2.5| = {exp['deviation']:.3f}, "
                f"tol {exp['tolerance']}) |"
            )
        fk = exp.get("free_k")
        if fk:
            note = " (k at the search bound → not identified)" if fk.get("at_boundary") else ""
            lines.append(
                f"| free k | {', '.join(exp['cases_used'])} | {fk['exponent']:.3f} "
                f"| {fk['c']:.3f} | {fk['k']:.3f} | {fk['r2']:.4f} | diagnostic{note} |"
            )
        lines.append("")
        if exp["cases_excluded"]:
            lines.append(
                f"Points with |rel. err| > {exp['validity_band_pct']:.0f} % "
                f"({', '.join(exp['cases_excluded'])}) are outside the correlation's "
                "applicability band (arching at this system size) and are excluded from "
                "the fit, but always tabulated above."
            )
            lines.append("")
    else:
        lines.append("Not enough cases to fit the law (needs at least two sweep points).")
        lines.append("")

    # ---- gravity + steadiness + friction --------------------------------
    grav = pilot.get("gravity_scaling")
    lines.append("### Table 3 — √g scaling, discharge steadiness, friction sensitivity")
    lines.append("")
    if grav:
        lines.append("| check | measured | expected | deviation | verdict |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| W(g/4)/W(g) | {grav['ratio']:.3f} | 0.500 "
                     f"| {abs(grav['ratio'] - 0.5) * 100:.1f} % "
                     f"| {'PASS' if grav['pass'] else 'FAIL'} |")
        lines.append("")
    cvs = pilot.get("window_cv") or {}
    if cvs:
        lines.append(f"| case | windows (kg/s) | CV (tol {TOL_CV}) | steady? |")
        lines.append("|---|---|---|---|")
        for cid in sorted(cvs):
            wins = cases[cid]["result"]["flow_rate_windows_kg_s"]
            lines.append(f"| {cid} | {', '.join(f'{w:.6f}' for w in wins)} "
                         f"| {cvs[cid]:.3f} | {_arrow(cvs[cid] < TOL_CV)} |")
        lines.append("")
    fr = pilot.get("friction_sensitivity") or {}
    if fr:
        lines.append("| μ | C recovered | rel. err vs Beverloo |")
        lines.append("|---|---|---|")
        for cid in sorted(fr, key=lambda c: fr[c]["mu"]):
            lines.append(f"| {fr[cid]['mu']:.2f} | {fr[cid]['C_recovered']:.3f} "
                         f"| {fr[cid]['rel_err_pct']:+.1f} % |")
        lines.append("")

    # ---- §4 error judgement ---------------------------------------------
    lines.append("## 4. Error judgement (explicit, per quantity)")
    lines.append("")

    def row(name: str, measured: float, reference: float, unit: str,
            rel_pct: float, note: str = "") -> str:
        if unit:                       # dimensional quantity, 6 decimals + unit
            m, r = f"{measured:.6f} {unit}", f"{reference:.6f} {unit}"
        else:                          # dimensionless (C, ratio)
            m, r = f"{measured:.3f}", f"{reference:.3f}"
        return (f"| {name} | {m} | {r} "
                f"| {rel_pct:+.1f} % | **{_arrow(abs(rel_pct) < 3.0)}** "
                f"| {_arrow(abs(rel_pct) < 10.0)}{note} |")

    lines.append("| quantity | DEM | reference | relative error | < 3 % ? | < 10 % ? |")
    lines.append("|---|---|---|---|---|---|")
    prim = cases.get("D5")
    if prim:
        bev = prim["beverloo"]
        lines.append(row("W at D/d = 5 (primary)", prim["result"]["flow_rate_kg_s"],
                         bev["W_ref_kg_s"], "kg/s", bev["rel_err_pct"]))
        lines.append(row(f"C recovered (D/d = 5, μ = {prim['case']['mu']:.2f})",
                         bev["C_recovered"], bev["C_ref"], "", bev["rel_err_pct"]))
        for cid in ("D6", "D4"):
            if cid in cases:
                b = cases[cid]["beverloo"]
                extra = " (arching)" if abs(b["rel_err_pct"]) > LAW_VALIDITY_BAND else ""
                lines.append(row(f"C recovered (D/d = {cases[cid]['case']['D_over_d']:.0f})",
                                 b["C_recovered"], b["C_ref"], "", b["rel_err_pct"], extra))
    if grav:
        lines.append(row("W(g/4)/W(g)", grav["ratio"], 0.5, "",
                         (grav["ratio"] - 0.5) * 100.0))
    if exp and exp.get("exponent_used_points") is not None:
        p = exp["exponent_used_points"]
        lines.append(row("exponent p (usable sweep points)", p, 2.5, "",
                         (p - 2.5) / 2.5 * 100.0))
    lines.append("")
    answers = []
    if prim:
        rel = prim["beverloo"]["rel_err_pct"]
        inside3, inside10 = abs(rel) < 3.0, abs(rel) < 10.0
        where = ("inside the 3 % band" if inside3 else
                 ("inside the predeclared 10 % band but outside a 3 % band" if inside10
                  else "outside both the 3 % and the 10 % band"))
        answers.append(f"(i) Is the primary rate within 3 % of Beverloo? "
                       f"**{'Yes' if inside3 else 'No'}** — it is {rel:+.1f} %, {where}.")
    if exp:
        if exp["status"] == "NOT_DETERMINED":
            answers.append(
                f"(ii) Is the 5/2 exponent verified? **No** — it is not even determined at "
                f"pilot scale: with {exp['n_points_used']} usable diameters the fitted "
                f"slope is {exp['exponent_used_points']:.3f} (deviation "
                f"{exp['deviation']:.3f} against a {exp['tolerance']} tolerance), so the "
                f"check is reported as **NOT_DETERMINED** rather than as a pass.")
        else:
            answers.append(
                f"(ii) Is the 5/2 exponent verified? "
                f"**{'Yes' if exp['status'] == 'PASS' else 'No'}** — fitted slope "
                f"{exp['exponent_used_points']:.3f} on {exp['n_points_used']} points, "
                f"deviation {exp['deviation']:.3f} against a {exp['tolerance']} tolerance "
                f"(**{exp['status']}**).")
    if cvs:
        failing = [cid for cid in sorted(cvs) if cvs[cid] >= TOL_CV]
        passing = [cid for cid in sorted(cvs) if cvs[cid] < TOL_CV]
        detail = ("all cases are inside the band." if not failing else
                  f"{len(failing)} of {len(cvs)} cases are outside: "
                  + ", ".join(f"{cid}={cvs[cid]:.3f}" for cid in failing)
                  + ("; the cases inside are " + ", ".join(passing) + "." if passing
                     else "; no case is inside the band."))
        answers.append(f"(iii) Is the steady-discharge criterion (CV < {TOL_CV}) met? "
                       f"**{'Yes' if not failing else 'No'}** — {detail}")
    if grav:
        answers.append(f"(iv) Does the rate scale as √g? "
                       f"**{'Yes' if grav['pass'] else 'No'}** — W(g/4)/W(g) = "
                       f"{grav['ratio']:.3f} vs 0.500 "
                       f"({abs(grav['ratio'] - 0.5) * 100:.1f} % deviation, "
                       f"tol {TOL_GRAVITY}).")
    lines.append("**Direct answers.** " + " ".join(answers))
    lines.append("")

    # ---- §5 full preset --------------------------------------------------
    lines.append("## 5. Status of the `full` preset (d = 3 mm)")
    lines.append("")
    if full:
        lines.append(
            f"The `full` preset is measured: {len(full['cases'])} cases in "
            f"`{_rel_to_repo(full['run_dir'])}/` "
            f"(artefact fingerprint `{artefact_digest(Path(full['run_dir']))}`), "
            f"primary-case rel. err "
            f"{full['cases']['D5']['beverloo']['rel_err_pct']:+.1f} %.")
    else:
        run_dir = Path(pilot["run_dir"]).parent / "full"
        rel_dir = _rel_to_repo(run_dir)
        lines.append(
            "**Withheld — not claimed.** No artefact exists in "
            f"`{rel_dir}/`, so this note reports no `d = 3 mm` number at all. A `full` "
            "case costs roughly 4.4× the particles, 4.5× the simulated time and a "
            "stiffer contact (≈ 50 min of wall clock per case), which is outside the "
            "control-node short-test budget. Evidence grade for the `d = 3 mm` "
            "campaign: **WITHHELD**. One case per command:")
        lines.append("")
        lines.append("```bash")
        lines.append("python benchmarks/m20707_beverloo/campaign.py --preset full --case D5")
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_comparison_block(results: dict) -> str:
    """Render the m20707 row of the campaign-comparison table (§6)."""
    pilot = results.get("pilot")
    if not pilot or "D5" not in pilot["cases"]:
        raise SystemExit("no pilot D5 artefact to render the comparison row from")
    c = pilot["cases"]["D5"]
    rel = c["beverloo"]["rel_err_pct"]
    verdict = "PASS (<10 %)" if c["beverloo"]["pass_10pct"] else "FAIL (≥10 %)"
    return (f"| **m20707 (pilot)** | **silo_flow** | **disordered** "
            f"| **{c['case']['mu']:.2f}** | **{c['case']['D_over_d']:.0f}** "
            f"| **{rel:+.1f} %** | **{verdict}** | this note (§3, fingerprint "
            f"`{artefact_digest(Path(pilot['run_dir']))}`) |")


def inject_block(text: str, name: str, body: str, path: Path) -> str:
    begin, end = BEGIN_FMT.format(name=name), END_FMT.format(name=name)
    if begin not in text or end not in text:
        raise SystemExit(f"marker {name!r} not found in {path}")
    i = text.index(begin) + len(begin)
    j = text.index(end, i)
    return text[:i] + "\n" + body.strip("\n") + "\n" + text[j:]


def render_doc(results: dict, path: Path = DOC_PATH) -> str:
    """Return the verification note with its generated blocks re-rendered."""
    text = Path(path).read_text()
    text = inject_block(text, "results", render_results_block(results), Path(path))
    text = inject_block(text, "comparison-row", render_comparison_block(results), Path(path))
    return text


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="both", choices=["full", "pilot", "both"])
    ap.add_argument("--run-dir", default=None, help="override run directory (single preset)")
    ap.add_argument("--md-out", default=None, help="write the markdown tables here")
    ap.add_argument("--summary-out", default=None,
                    help="write one combined, machine-readable summary here")
    ap.add_argument("--inject-doc", nargs="?", const=str(DOC_PATH), default=None,
                    metavar="DOC",
                    help="render the generated blocks of the verification note from the "
                         "committed artefacts")
    ap.add_argument("--check-doc", nargs="?", const=str(DOC_PATH), default=None,
                    metavar="DOC",
                    help="exit non-zero if the generated blocks of the note disagree "
                         "with the artefacts")
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

    if args.inject_doc or args.check_doc:
        doc = Path(args.inject_doc or args.check_doc)
        rendered = render_doc(results, doc)
        current = doc.read_text()
        if args.check_doc:
            if rendered != current:
                import difflib
                diff = list(difflib.unified_diff(
                    current.splitlines(), rendered.splitlines(),
                    fromfile=f"{doc} (as committed)", tofile="rendered from artefacts",
                    lineterm="", n=1))
                print("\n".join(diff[:60]))
                raise SystemExit(
                    f"{doc}: generated blocks disagree with the artefacts "
                    f"({len(diff)} diff lines) - re-run "
                    f"`python benchmarks/m20707_beverloo/analyse.py --inject-doc`")
            print(f"{doc}: generated blocks match the artefacts")
        else:
            doc.write_text(rendered)
            print(f"Rendered the generated blocks of {doc} from the artefacts")
    return results


if __name__ == "__main__":
    main()
