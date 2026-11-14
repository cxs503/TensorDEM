#!/usr/bin/env bash
# Sequential remaining-case batch for CASE-DEM-SILO-BEVERLOO (m20055).
# One core; used because shard dispatch of cj2460 launched only shard 1.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
for c in D5 D6 D5_mu15 D5_mu45 D5_dthalf; do
  echo "=== $c $(date -Is)"
  python benchmarks/m20055_beverloo/run_case.py --case "$c" || echo "CASE $c FAILED rc=$?"
done
python benchmarks/m20055_beverloo/analyse.py || echo "ANALYSE FAILED"
python benchmarks/m20055_beverloo/render_doc.py || echo "RENDER FAILED"
echo "=== done $(date -Is)"