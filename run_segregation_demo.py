#!/usr/bin/env python3
"""Quick demonstration of segregation physics for the validation report.

Runs a minimal subset of cases to show:
- Negative control (d_l/d_s = 1) has |S| < 0.02
- Segregation increases with diameter ratio
- Mass conservation is exact
- Transfer mapping works

This is a proof-of-concept for the report, not the full campaign.
"""

import sys
import json
import os
import numpy as np

sys.path.insert(0, 'src')

from tensordem.segregation.campaign import CaseSpec, run_case
from tensordem.segregation.analysis import evaluate_criteria, load_records
from tensordem.segregation.transfer import HoldGeometry

def make_demo_specs():
    """Create a minimal set of cases for demonstration."""
    specs = []
    
    # Negative control: d_l/d_s = 1, 3 seeds
    for seed in range(3):
        specs.append(CaseSpec(
            name=f"demo_ctl_R1.0_s{seed}",
            drive="vib",
            ratio=1.0,
            large_fraction=0.3,
            gamma=3.0,
            frequency=60.0,
            seed=seed,
            n_particles=5000,  # smaller for speed
            width=0.4,
            height=0.18,
            settle_time=0.1,
            run_time=0.2,
            sample_every=100,
        ))
    
    # Diameter ratio sweep: 1.5, 2.0, 3.0, 4.0, 1 seed each
    for ratio in [1.5, 2.0, 3.0, 4.0]:
        specs.append(CaseSpec(
            name=f"demo_ratio_R{ratio}_s0",
            drive="vib",
            ratio=ratio,
            large_fraction=0.3,
            gamma=3.0,
            frequency=60.0,
            seed=0,
            n_particles=5000,
            width=0.4,
            height=0.18,
            settle_time=0.1,
            run_time=0.2,
            sample_every=100,
        ))
    
    # Gamma threshold sweep: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
    for gamma in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        specs.append(CaseSpec(
            name=f"demo_gamma_G{gamma}_s0",
            drive="vib",
            ratio=2.0,
            large_fraction=0.3,
            gamma=gamma,
            frequency=60.0,
            seed=0,
            n_particles=5000,
            width=0.4,
            height=0.18,
            settle_time=0.1,
            run_time=0.2,
            sample_every=100,
        ))
    
    return specs

def main():
    out_dir = "runs/segregation/demo_out"
    os.makedirs(out_dir, exist_ok=True)
    
    specs = make_demo_specs()
    print(f"Running {len(specs)} demo cases...")
    
    for i, spec in enumerate(specs):
        path = os.path.join(out_dir, f"{spec.name}.json")
        if os.path.exists(path):
            print(f"[{i+1}/{len(specs)}] skip {spec.name}")
            continue
        
        print(f"[{i+1}/{len(specs)}] running {spec.name}...", end=" ", flush=True)
        record = run_case(spec)
        
        with open(path, "w") as f:
            json.dump(record, f)
        
        s_bn = record["metrics"]["segregation"]["S_bn"]
        v_up = record["metrics"]["rise_large"]["v_up_m_s"]
        mass_err = record["metrics"]["mass_conservation_rel"]
        print(f"S_bn={s_bn:.3f} v_up={v_up:.2e} m/s mass_err={mass_err:.2e}")
    
    print("\nEvaluating acceptance criteria...")
    records = load_records(out_dir)
    geo = HoldGeometry(length_m=20.0, breadth_m=12.0, depth_m=6.0, n_longitudinal=8)
    criteria = evaluate_criteria(records, geometry=geo, displacement_t=20000.0, km_m=9.0)
    
    # Save criteria
    with open(os.path.join(out_dir, "criteria.json"), "w") as f:
        json.dump(criteria, f, indent=2, default=str)
    
    print("\n=== Acceptance Criteria ===")
    for key, val in criteria.items():
        if isinstance(val, dict) and "pass" in val:
            status = "✓ PASS" if val["pass"] else "✗ FAIL"
            print(f"{key}: {status}")
        else:
            print(f"{key}: {val}")
    
    return criteria

if __name__ == "__main__":
    main()
