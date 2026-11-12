#!/usr/bin/env bash
# CASE-SETTLING-STOKES / transient half: regenerate artifacts + docs.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=src python benchmarks/case_settling_transient/run_case.py \
    --out artifacts/case_settling_transient
