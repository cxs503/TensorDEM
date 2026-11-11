#!/usr/bin/env python3
"""Validate CASE-PAIR-COLLISION benchmark results against tolerance criteria."""
from __future__ import annotations

import json
import pathlib
import sys

RESULTS_PATH = pathlib.Path("benchmarks/case_pair_collision_hertz/results.json")
TOLERANCE = 0.03  # 3%


def main():
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run run.py first.")
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    print("=" * 60)
    print("CASE-PAIR-COLLISION: Validation Report")
    print("=" * 60)

    elastic = results["primary_elastic"]
    damped = results["primary_damped"]
    sweep = results["velocity_sweep"]
    conv = results["convergence_study"]

    all_pass = True

    # 1. Peak overlap (elastic)
    err = elastic["peak_overlap_rel_error"]
    status = "PASS" if err < TOLERANCE else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"\n1. Peak overlap (elastic, v=0.5 m/s)")
    print(f"   DEM:     {elastic['peak_overlap_m']:.6e} m")
    print(f"   Analytic: {elastic['peak_overlap_ref_m']:.6e} m")
    print(f"   Error:   {err:.4%} [{status}] (threshold: {TOLERANCE:.0%})")

    # 2. Contact time (elastic)
    err = elastic["contact_time_rel_error"]
    status = "PASS" if err < TOLERANCE else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"\n2. Contact time (elastic, v=0.5 m/s)")
    print(f"   DEM:     {elastic['contact_time_s']:.6e} s")
    print(f"   Analytic: {elastic['contact_time_ref_s']:.6e} s")
    print(f"   Error:   {err:.4%} [{status}] (threshold: {TOLERANCE:.0%})")

    # 3. Restitution coefficient (damped)
    err = damped["restitution_rel_error"]
    status = "PASS" if err < TOLERANCE else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"\n3. Restitution coefficient (damped, e_target=0.8, v=0.5 m/s)")
    print(f"   DEM engine: {damped['restitution_engine']:.6f}")
    print(f"   RK4 ref:    {damped['restitution_rk4']:.6f}")
    print(f"   Error:      {err:.4%} [{status}] (threshold: {TOLERANCE:.0%})")
    print("   (code-vs-code check: the RK4 reference integrates the SAME Tsuji law,")
    print("    so this row only proves the engine implements that law faithfully.)")

    # 3b. Restitution closure against the REQUESTED coefficient.
    #     The anchor is "restitution coefficient"; the linear zeta(e) mapping is
    #     not exact for the Hertzian delta^(1/4) dashpot, so the requested value
    #     is not delivered. Reported honestly instead of being folded into 3.
    closure = damped.get("closure_error_vs_target")
    if closure is not None:
        status = "PASS" if closure < TOLERANCE else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"\n3b. Restitution closure vs e_target={damped.get('restitution_target')}")
        print(f"   DEM engine: {damped['restitution_engine']:.6f}")
        print(f"   Requested:  {damped.get('restitution_target')}")
        print(f"   Error:      {closure:.4%} [{status}] (threshold: {TOLERANCE:.0%})")
        print("   Known model bias: zeta(e) from linear spring-dashpot theory applied")
        print("   to a Hertzian dashpot; see docs/board-audit-m15296/AUDIT.md.")

    # 4. Velocity independence
    max_err = max(r["restitution_rel_error"] for r in sweep)
    status = "PASS" if max_err < TOLERANCE else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"\n4. Restitution velocity-independence (e_target=0.8)")
    for r in sweep:
        print(f"   v={r['speed_m_per_s']} m/s: e={r['restitution_engine']:.4f}, err={r['restitution_rel_error']:.4%}")
    print(f"   Max error: {max_err:.4%} [{status}]")

    # 5. Convergence
    print(f"\n5. Timestep convergence (peak overlap)")
    for r in conv:
        err = r["peak_overlap_rel_error"]
        status = "PASS" if err < TOLERANCE else "FAIL"
        print(f"   dt×{r['dt_factor']}: error={err:.4%} [{status}]")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
