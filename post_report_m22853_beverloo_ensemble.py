#!/usr/bin/env python3
"""Generate and publish the m22657 Beverloo ensemble report.

Usage:
    python post_report_m22853_beverloo_ensemble.py [--dry-run]

This script:
1. Runs the m22657 ensemble analysis (reads runs/m22657_beverloo_ensemble/seed*.json)
2. Generates comparison figures (matplotlib)
3. Builds the paper-grade report JSON with six elements
4. Posts to /api/v1/tensor/reports

All numbers are read from the run artifacts — nothing is hand-typed.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs" / "m22657_beverloo_ensemble"
FIG_DIR = ROOT / "docs" / "figures" / "m22657"
ANALYSIS_SCRIPT = ROOT / "benchmarks" / "m22657_beverloo_ensemble" / "analyse.py"

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
if not base.endswith("/api/v1"):
    base += "/api/v1"


def login() -> str:
    data = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    tok = body.get("access_token") or body.get("token")
    if not tok:
        raise SystemExit(f"login did not return a token: {body}")
    return tok


def run_analysis() -> dict:
    """Run the frozen estimator and return the summary."""
    import subprocess
    summary_path = RUN_DIR / "summary.json"
    cmd = [sys.executable, str(ANALYSIS_SCRIPT),
           "--dir", str(RUN_DIR),
           "--out", str(summary_path),
           "--figures", str(FIG_DIR)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Analysis failed:\n{proc.stderr}", file=sys.stderr)
        raise SystemExit(1)
    print(proc.stdout)
    return json.loads(summary_path.read_text())


def make_figures_base64(summary: dict) -> list:
    """Read generated PNGs and return [{name, caption, png_base64}]."""
    figures = []
    for png in sorted(FIG_DIR.glob("*.png")):
        with open(png, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        name = png.stem
        caption = {
            "m22657_ensemble_errors": "Single-seed scatter vs ensemble mean with 95% CI. Grey band = ledger ±10% tolerance.",
            "m22657_window_history": "Window-rate history for each seed. Shaded region = counted windows (last 50%). Black dashed = Beverloo prediction.",
        }.get(name, "")
        figures.append({"name": name, "caption": caption, "png_base64": b64})
    return figures


def build_report(summary: dict, figures: list) -> dict:
    """Build the paper-grade report JSON."""
    e = summary.get("ensemble", {})
    verdicts = summary.get("verdicts", {})
    
    # Build markdown
    md = f"""# CASE-DEM-SILO-BEVERLOO: Seed-Ensemble Validation (m22657)

## 1. Case Introduction

**Physical background**: The Beverloo law describes the steady-state gravity-driven discharge of cohesionless granular material through a circular orifice in a flat-bottom silo. The flow rate is set by the free-fall arch above the orifice and is independent of the filling height (for sufficiently deep beds).

**Geometry & parameters** (frozen in preregistration):
- Particle diameter: d = 3.0 mm
- Orifice diameter: D = 15.0 mm (D/d = 5.0)
- Silo radius: R = 24.0 mm (silo_D/D = 3.2)
- Fill height: H = 36.0 mm (H/D = 2.4)
- Particle friction: μ = 0.50, rolling μ_r = 0.05, restitution e = 0.30
- Normal stiffness: k_n = 5.0×10³ N/m
- Wall friction: μ_w = 0 (smooth walls)
- Seeds: 1, 2, 3, 4, 5, 6 (disordered packing)

**Acceptance criterion**: |W − W_ref| / W_ref < 10% (feature_ledger CASE-DEM-SILO-BEVERLOO)

## 2. Method

**DEM kernel**: `tensordem.silo_flow.kernel` implements Cundall–Strack contact with:
- Linear spring-dashpot normal force: f_n = max(k_n δ − γ v_n, 0)
- Incremental tangential spring saturated at μ f_n
- Coulomb rolling torque
- Analytic wall contacts (cylinder + flat plate with orifice)
- Disordered packing generator (no HCP arching)

**Measurement protocol**:
- Settle time: 0.06 s (plug closed)
- Discard time: 0.12 s (plug open, transient discarded)
- Measurement: 8 windows × 0.05 s = 0.40 s (drains ≈ 14% of bed)
- Flow rate: mean of last 50% of windows (4 windows)

**Estimator** (frozen in `benchmarks/m22657_beverloo_ensemble/analyse.py`):
- Ensemble mean: mean_s(W_s) with Student-t 95% CI
- Reference: Beverloo with C = 0.58, k = 1.4 (literature, no refit)
- Verdicts: V1_ledger, V2_reproducibility, V3_steady_state, V4_rho_control

## 3. Results

### 3.1 Per-seed results

| seed | n_bed | ρ_b [kg/m³] | W_dem [kg/s] | W_ref [kg/s] | rel err [%] | C_rec | drift [%] |
|------|-------|-------------|--------------|--------------|-------------|-------|-----------|
"""
    for p in summary.get("per_seed", []):
        md += f"| {p['seed']} | {p['n_bed']} | {p['rho_bulk_kg_m3']:.1f} | {p['W_dem_kg_s']:.6f} | {p['W_ref_kg_s']:.6f} | {100*p['rel_err']:+.2f} | {p['C_recovered']:.3f} | {100*p['window_trend']['rel_drift']:+.2f} |\n"
    
    md += f"""
### 3.2 Ensemble summary

- **Ensemble mean**: W_ens = {e.get('W_ens_kg_s', 0):.6f} kg/s
- **Reference**: W_ref = {e.get('W_ref_kg_s', 0):.6f} kg/s
- **Relative error**: {100*e.get('rel_err_ensemble', 0):+.2f}%
- **95% CI**: [{100*e.get('rel_err_ci95', [0,0])[0]:+.2f}%, {100*e.get('rel_err_ci95', [0,0])[1]:+.2f}%]
- **CV**: {100*e.get('cv', 0):.2f}%
- **Recovered C**: {e.get('C_recovered_ensemble', 0):.3f}

### 3.3 Verdicts

- **V1_ledger**: {verdicts.get('V1_ledger', 'NOT_RUN')}
- **V2_reproducibility**: {verdicts.get('V2_reproducibility', 'NOT_RUN')}
- **V3_steady_state**: {verdicts.get('V3_steady_state', 'NOT_RUN')}
- **V4_rho_control**: {verdicts.get('V4_rho_control', 'NOT_RUN')}

## 4. Figures

"""
    for i, fig in enumerate(figures, 1):
        md += f"**Figure {i}**: {fig['caption']}\n\n"
    
    md += f"""
## 5. Caveats & Limitations

1. **Linear-spring contact**: The DEM kernel uses linear spring-dashpot, not Hertzian. Large overlaps (δ/d ~ 1%) violate the rigid-particle assumption.
2. **Low stiffness**: k_n = 5×10³ N/m is 1–2 orders of magnitude below typical DEM practice. Higher k_n reduces overlaps but increases cost.
3. **Axisymmetric geometry**: 2D axisymmetric, not full 3D. Sufficient for silo discharge but limits extension.
4. **Monodisperse spheres**: Beverloo's k factor accounts for size. Polydisperse would be more rigorous.
5. **Smooth walls**: μ_w = 0 matches Beverloo's glass-bead experiments. Real silos have wall friction.

## 6. Error Judgment

**Key quantity**: Ensemble mean relative error = {100*e.get('rel_err_ensemble', 0):+.2f}%

**Is |rel err| < 10%?**: **{'YES' if abs(e.get('rel_err_ensemble', 0)) < 0.10 else 'NO'}**

**95% CI within ±10%?**: **{'YES' if all(abs(x) < 0.10 for x in e.get('rel_err_ci95', [0,0])) else 'NO'}**

**Verdict**: {verdicts.get('V1_ledger', 'NOT_RUN')}

**Interpretation**: {'The DEM kernel passes the Beverloo validation with statistical rigor.' if verdicts.get('V1_ledger') == 'PASS' else 'The DEM kernel fails the Beverloo validation. The error is systematic (not packing scatter) and requires model improvements (higher stiffness, Hertzian contact, or 3D full simulation).' if verdicts.get('V1_ledger') == 'FAIL' else 'The ensemble scatter is too large to decide. More seeds or parameter tuning needed.'}
"""
    
    report = {
        "title": "CASE-DEM-SILO-BEVERLOO: Seed-Ensemble Validation (m22657)",
        "summary": f"Beverloo law validation with 6-seed ensemble at D/d=5. Ensemble mean rel err = {100*e.get('rel_err_ensemble', 0):+.2f}%, 95% CI = [{100*e.get('rel_err_ci95', [0,0])[0]:+.2f}%, {100*e.get('rel_err_ci95', [0,0])[1]:+.2f}%]. Verdict: {verdicts.get('V1_ledger', 'NOT_RUN')}.",
        "scenario_key": "CASE-DEM-SILO-BEVERLOO",
        "mission_id": 22853,
        "tags": ["DEM", "Beverloo", "silo", "discharge", "ensemble", "validation"],
        "markdown": md,
        "figures": figures,
    }
    return report


def main(dry_run: bool = False) -> int:
    print("=" * 70)
    print("CASE-DEM-SILO-BEVERLOO Ensemble Report (m22853)")
    print("=" * 70)
    
    # Check if runs exist
    seed_files = sorted(RUN_DIR.glob("seed*.json"))
    if len(seed_files) < 6:
        print(f"ERROR: Expected 6 seed files, found {len(seed_files)} in {RUN_DIR}")
        print("The cluster job (cj4437) may still be running. Check with:")
        print("  run_cluster ps --jid cj4437")
        return 1
    
    print(f"Found {len(seed_files)} seed files in {RUN_DIR}")
    
    # Run analysis
    print("\n[1/3] Running ensemble analysis...")
    summary = run_analysis()
    print(f"Analysis complete: {summary.get('n_cases')} cases")
    
    # Generate figures
    print("\n[2/3] Generating figures...")
    figures = make_figures_base64(summary)
    print(f"Generated {len(figures)} figures")
    
    # Build report
    print("\n[3/3] Building report...")
    report = build_report(summary, figures)
    
    if dry_run:
        print("\n[DRY RUN] Report built but not posted:")
        print(json.dumps(report, indent=2)[:1000] + "...")
        return 0
    
    # Post report
    print("\nPosting report to platform API...")
    token = login()
    payload = json.dumps(report).encode()
    req = urllib.request.Request(
        base + "/tensor/reports",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read())
    
    report_id = result.get("id") or result.get("report_id")
    print(f"\n✓ Report posted successfully: id={report_id}")
    print(f"  Title: {report['title']}")
    print(f"  Summary: {report['summary'][:100]}...")
    
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Build report but don't post")
    args = ap.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
