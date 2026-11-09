#!/usr/bin/env python3
"""Run the full Hu & Zhou (2016) MT Uikku 10-condition campaign with 3D sheet.

This script runs all 10 level-ice cases across W/B ∈ {1.0, 1.5, 2.0}
with mass-conserving lateral escape (reservoir enabled).

Usage:
    PYTHONPATH=src python benchmarks/uikku_3d_campaign.py --output results-uikku-3d
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensordem.level_ice.sheet3d import (
    load_uikku_cases,
    run_uikku_case,
    Sheet3DConfig,
    LevelIceSheet3D,
)
from tensordem.level_ice.sheet import SheetConfig


def run_campaign(wb_ratios=(1.0, 1.5, 2.0), layers=4, advance_m=3.0,
                 beam_m=26.0, output_dir="results-uikku-3d"):
    """Run the full campaign and save results."""
    cases = load_uikku_cases()
    all_results = []
    t0 = time.time()

    for i, case in enumerate(cases):
        for wb in wb_ratios:
            print(f"  [{i+1}/{len(cases)}] Test {case['test_id']}, W/B={wb}...", end=" ", flush=True)
            tc = time.time()
            result = run_uikku_case(
                case, wb_ratio=wb, layers=layers,
                enable_reservoir=True, advance_m=advance_m,
                beam_m=beam_m,
            )
            dt = time.time() - tc
            err = f"{result['error_pct']:.1f}%" if result['error_pct'] is not None else "N/A"
            print(f"predicted={result['predicted_kN']:.1f} kN, error={err}, {dt:.1f}s")
            all_results.append(result)

    total_time = time.time() - t0

    # Compute summary statistics
    summary = {}
    for wb in wb_ratios:
        wb_results = [r for r in all_results if r["wb_ratio"] == wb]
        errors = [abs(r["error_pct"]) for r in wb_results if r["error_pct"] is not None]
        mape = float(np.mean(errors)) if errors else float("nan")

        max_mass_errors = [r["mass_ledger"]["max_rel_error"] for r in wb_results]
        max_mass_error = max(max_mass_errors)

        summary[f"wb_{wb}"] = {
            "mape_pct": mape,
            "n_cases": len(wb_results),
            "max_mass_conservation_error": max_mass_error,
            "per_case": [
                {
                    "test_id": r["test_id"],
                    "h_m": r["ice_thickness_m"],
                    "v_m_s": r["drift_speed_m_s"],
                    "measured_kN": r["measured_kN"],
                    "predicted_kN": round(r["predicted_kN"], 2),
                    "error_pct": round(r["error_pct"], 2) if r["error_pct"] is not None else None,
                    "lindqvist_kN": r["lindqvist_kN"],
                }
                for r in wb_results
            ],
        }

    # Also run negative control (no reservoir) for 2 cases
    print("\n  Running negative control (no reservoir)...")
    neg_control = []
    for case in cases[:2]:
        r = run_uikku_case(case, wb_ratio=1.5, layers=4,
                          enable_reservoir=False, advance_m=advance_m,
                          beam_m=beam_m)
        neg_control.append({
            "test_id": r["test_id"],
            "measured_kN": r["measured_kN"],
            "predicted_kN": round(r["predicted_kN"], 2),
            "error_pct": round(r["error_pct"], 2) if r["error_pct"] is not None else None,
        })
        print(f"    Test {r['test_id']}: error={r['error_pct']:.1f}%" if r['error_pct'] else f"    Test {r['test_id']}: N/A")

    payload = {
        "protocol": "uikku-3d-mass-conserving-v1",
        "wb_ratios": list(wb_ratios),
        "layers": layers,
        "advance_m": advance_m,
        "beam_m": beam_m,
        "total_runtime_s": round(total_time, 1),
        "summary": summary,
        "negative_control": neg_control,
        "all_results": [
            {
                "test_id": r["test_id"],
                "wb_ratio": r["wb_ratio"],
                "layers": r["layers"],
                "ice_thickness_m": r["ice_thickness_m"],
                "drift_speed_m_s": r["drift_speed_m_s"],
                "measured_kN": r["measured_kN"],
                "predicted_kN": round(r["predicted_kN"], 2),
                "error_pct": round(r["error_pct"], 2) if r["error_pct"] is not None else None,
                "lindqvist_kN": r["lindqvist_kN"],
                "mass_ledger": r["mass_ledger"],
                "lateral_flux": r["lateral_flux"],
            }
            for r in all_results
        ],
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "uikku_3d_campaign.json").write_text(
        json.dumps(payload, indent=2, default=float)
    )
    print(f"\n  Results saved to {out / 'uikku_3d_campaign.json'}")
    print(f"  Total runtime: {total_time:.1f}s")

    # Print summary table
    print("\n  === SUMMARY ===")
    for wb in wb_ratios:
        s = summary[f"wb_{wb}"]
        print(f"  W/B={wb}: MAPE={s['mape_pct']:.1f}%, max_mass_err={s['max_mass_conservation_error']:.2e}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results-uikku-3d")
    parser.add_argument("--wb-ratios", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--advance", type=float, default=3.0)
    args = parser.parse_args()

    run_campaign(
        wb_ratios=tuple(args.wb_ratios),
        layers=args.layers,
        advance_m=args.advance,
        output_dir=args.output,
    )
