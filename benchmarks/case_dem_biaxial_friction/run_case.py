#!/usr/bin/env python
"""Entry point for CASE-DEM-BIAXIAL-FRICTION benchmark.

Usage:
    TENSORDEM_THREADS=4 PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/run_case.py
    TENSORDEM_THREADS=4 PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/run_case.py --quick
"""
import sys
import os

# Ensure the repo root is on the path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tensordem.biaxial_friction_case import run_case

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CASE-DEM-BIAXIAL-FRICTION")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    result = run_case(quick=args.quick, out_dir=args.out_dir)
    sys.exit(0 if result["judgement"]["passed"] else 1)
