#!/usr/bin/env bash
# One-command reproduction of CASE-DEM-SILO-BEVERLOO (m20055).
# Each case is a single-core DEM run; 6 cases; heavy enough to submit:
#   run_compute submit repo=TensorDEM stage_dir=<this worktree> \
#       cmd_tpl='PYTHONPATH=src python benchmarks/m20055_beverloo/run_case.py --shard {i} --nshards {n}' shards=6
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
for c in D4 D5 D6 D5_mu15 D5_mu45 D5_dthalf; do
  python benchmarks/m20055_beverloo/run_case.py --case "$c"
done
python benchmarks/m20055_beverloo/analyse.py
python benchmarks/m20055_beverloo/render_doc.py
python -m pytest tests/test_silo_discharge_steady.py -q
