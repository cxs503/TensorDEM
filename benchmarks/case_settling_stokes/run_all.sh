#!/usr/bin/env bash
# CASE-SETTLING-STOKES: one-command reproduce (run + report).
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=src python benchmarks/case_settling_stokes/run_case.py --out artifacts/case_settling_stokes
PYTHONPATH=src python benchmarks/case_settling_stokes/report.py --out artifacts/case_settling_stokes --doc docs/CASE_SETTLING_STOKES.md
echo "OK: artifacts/case_settling_stokes/ + docs/CASE_SETTLING_STOKES.md"
