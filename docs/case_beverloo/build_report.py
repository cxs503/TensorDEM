#!/usr/bin/env python3
"""Build and publish the CASE-DEM-SILO-BEVERLOO validation report.

Reads campaign results, generates figures, and posts the report to the
TensorPlatform reports API.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.silo_flow.beverloo import BEVERLOO_C, BEVERLOO_K, beverloo_prediction

RUN_DIR = ROOT / "runs" / "m20707_beverloo"
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"


def build_report() -> dict:
    """Build the full validation report as a dict ready for the API."""
    ref = json.loads(REF_PATH.read_text())
    C_ref = float(ref["coefficients"]["C"])
    k_ref = float(ref["coefficients"]["k"])

    # Load campaign results
    cases = {}
    for p in sorted(RUN_DIR.glob("*.json")):
        if p.name == "summary.json":
            continue
        d = json.loads(p.read_text())
        cases[d["case_id"]] = d

    summary_path = RUN_DIR / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    # Build results table
    table_rows = []
    for cid in sorted(cases):
        c = cases[cid]
        bev = c["beverloo"]
        cs = c["case"]
        res = c["result"]
        table_rows.append(
            f"| {cid:<8} | {cs['D_over_d']:>4.1f} | {cs['mu']:>4.2f} | "
            f"{c['gravity']:>5.2f} | {res['n_bed']:>5} | "
            f"{res['rho_bulk']:>7.1f} | {res['flow_rate_kg_s']*1000:>8.3f} | "
            f"{bev['W_ref_kg_s']*1000:>8.3f} | {bev['rel_err_pct']:>+6.1f} | "
            f"{bev['C_recovered']:>5.3f} | "
            f"{'✓' if bev['pass_10pct'] else '✗':>3} |"
        )

    # Primary case verdict
    d5 = cases.get("D5", {})
    d5_verdict = "N/A"
    d5_err = "N/A"
    d5_C = "N/A"
    if d5:
        bev = d5["beverloo"]
        d5_verdict = "PASS" if bev["pass_10pct"] else "FAIL"
        d5_err = f"{bev['rel_err_pct']:+.1f}%"
        d5_C = f"{bev['C_recovered']:.3f}"

    # Exponent check
    exp_info = "N/A"
    if "exponent_check" in summary:
        ec = summary["exponent_check"]
        exp_info = f"p = {ec['exponent_fixed_k']:.3f} (target 2.5), R² = {ec['r2_fixed_k']:.3f}"

    # Gravity scaling
    grav_info = "N/A"
    if "gravity_scaling" in summary:
        gs = summary["gravity_scaling"]
        grav_info = f"W(g/4)/W(g) = {gs['ratio']:.3f} (expected 0.500), {'PASS' if gs['pass'] else 'FAIL'}"

    # Generate figures
    sys.path.insert(0, str(ROOT / "docs" / "case_beverloo"))
    from render_figures import main as render_main
    figures = render_main()

    # Build markdown report
    md = f"""# CASE-DEM-SILO-BEVERLOO: Beverloo Law Validation Report

## 1. Case Introduction

This report validates the **Beverloo orifice-discharge law** (Beverloo, Leniger & van de Velde, 1961)
against 3-D DEM simulations of gravity-driven granular discharge from a flat-bottom silo.

**Beverloo correlation:**

$$W = C \\cdot \\rho_b \\cdot \\sqrt{{g}} \\cdot (D - k \\cdot d)^{{5/2}}$$

| Parameter | Symbol | Value | Range |
|-----------|--------|-------|-------|
| Discharge coefficient | C | 0.58 | 0.55–0.60 |
| Shape factor | k | 1.4 | 1.2–1.6 |
| Particle diameter | d | 3 mm | — |
| Particle density | ρ_p | 2500 kg/m³ | — |
| Silo radius | R | 25 mm | — |
| Fill height | H | 60 mm | — |
| Wall friction | μ_w | 0 (smooth) | — |
| Inter-particle friction | μ_p | 0.50 | — |
| Rolling friction | μ_r | 0.05 | — |
| Normal stiffness | k_n | 5×10³ N/m | — |
| Restitution | e | 0.30 | — |

**Acceptance criterion:** |W − W_ref| / W_ref < 10% for the primary case D5 (D/d = 5).

## 2. Method

### DEM Kernel
- **Contact model:** Linear spring–dashpot (normal) + Cundall–Strack tangential spring + Coulomb rolling resistance
- **Wall friction:** Per-surface Cundall–Strack history (cylinder, plate, rim)
- **Packing:** Disordered deposition (jittered FCC lattice + overlap relaxation + gravity settling)
- **Energy ledger:** Closed per-timestep balance (kinetic + gravitational + elastic + dissipated + exit)
- **Integration:** Velocity-Verlet with dt = 0.05·T_contact

### Measurement Protocol
1. Build disordered bed with plugged outlet
2. Settle under gravity (0.08 s)
3. Open orifice, discard start-up transient (0.15 s)
4. Measure mass flow rate over 6 windows of 0.20 s each
5. Steady-state rate = mean of last 50% of windows

### Case Matrix

| Case ID | D/d | D [mm] | μ_p | g [m/s²] | Purpose |
|---------|-----|--------|-----|-----------|---------|
| D4 | 4 | 12 | 0.50 | 9.81 | Law check point |
| D5 | 5 | 15 | 0.50 | 9.81 | **Primary case** |
| D6 | 6 | 18 | 0.50 | 9.81 | Law check point |
| D5_g4 | 5 | 15 | 0.50 | 2.45 | Gravity scaling |

## 3. Results

### Flow Rate Comparison

| Case     | D/d  |  μ_p |    g | n_bed | ρ_b [kg/m³] | W_dem [g/s] | W_ref [g/s] | err%  | C_rec | OK  |
|----------|------|------|------|-------|-------------|-------------|-------------|-------|-------|-----|
{chr(10).join(table_rows)}

### Primary Case Verdict (D5)

- **W_dem** = {d5['result']['flow_rate_kg_s']*1000:.3f} g/s
- **W_ref** = {d5['beverloo']['W_ref_kg_s']*1000:.3f} g/s
- **Relative error** = {d5_err}
- **C recovered** = {d5_C} (literature: {C_ref})
- **Verdict: {d5_verdict}** (tolerance <10%)

### Power-Law Exponent

{exp_info}

### Gravity Scaling

{grav_info}

## 4. Figures

See attached figures below.

## 5. Notes and Limitations

1. **Smooth walls (μ_w = 0):** The Beverloo experiments used glass beads in smooth-walled
   containers. Wall friction would reduce the flow rate and increase C_recovered.
2. **Monodisperse spheres:** The correlation applies to spherical particles; polydisperse
   or non-spherical particles require different k values.
3. **D/d ≥ 4:** Below D/d ≈ 4, intermittent arching can occur and the Beverloo law
   breaks down. The D4 case (D/d = 4) is near this limit.
4. **Finite fill height:** The Beverloo law assumes the flow rate is independent of fill
   height. For very shallow beds (H/D < 1.5), this assumption fails.
5. **Stiffness sensitivity:** The spring stiffness k_n affects the contact duration and
   thus the effective restitution. Very soft contacts (low k_n) can lead to sustained
   overlaps that modify the effective orifice geometry.

## 6. Error Assessment

| Quantity | DEM Value | Reference | Relative Error | <10%? |
|----------|-----------|-----------|----------------|-------|
| W (D5) | {d5['result']['flow_rate_kg_s']*1000:.3f} g/s | {d5['beverloo']['W_ref_kg_s']*1000:.3f} g/s | {d5_err} | {'Yes' if d5.get('beverloo', {}).get('pass_10pct', False) else 'No'} |
"""

    report = {
        "title": "CASE-DEM-SILO-BEVERLOO: Beverloo Orifice-Discharge Law Validation",
        "summary": f"Beverloo law validation: D5 primary case {d5_verdict} with {d5_err} error, C_recovered={d5_C}",
        "scenario_key": "CASE-DEM-SILO-BEVERLOO",
        "mission_id": 22262,
        "tags": ["DEM", "Beverloo", "silo-discharge", "validation", "granular-flow"],
        "markdown": md,
        "figures": list(figures.values()) if figures else [],
    }

    return report


def publish_report(report: dict) -> None:
    """Publish the report to the TensorPlatform API."""
    import urllib.request
    import urllib.parse

    api_base = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1")
    user = os.environ.get("TENSOR_AGENT_USER", "")
    passwd = os.environ.get("TENSOR_AGENT_PASS", "")

    # Login
    login_url = f"{api_base}/auth/login"
    login_data = json.dumps({"username": user, "password": passwd}).encode()
    req = urllib.request.Request(login_url, data=login_data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            token = json.loads(resp.read())["access_token"]
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # Post report
    report_url = f"{api_base}/tensor/reports"
    report_data = json.dumps(report, default=float).encode()
    req = urllib.request.Request(report_url, data=report_data,
                                  headers={
                                      "Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}",
                                  })
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            print(f"Report published: id={result.get('id', 'unknown')}")
    except Exception as e:
        print(f"Report publish failed: {e}")
        # Save locally as fallback
        fallback = ROOT / "docs" / "case_beverloo" / "report.json"
        fallback.write_text(json.dumps(report, indent=2, default=float))
        print(f"Report saved locally to {fallback}")


def main():
    report = build_report()

    # Save locally
    local_path = ROOT / "docs" / "case_beverloo" / "report.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"Report saved to {local_path}")

    # Publish
    publish_report(report)


if __name__ == "__main__":
    main()
