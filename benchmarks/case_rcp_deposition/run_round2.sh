#!/usr/bin/env bash
# Round-2 runner for CASE-DEM-RCP-PACKING-FRACTION (mission 18143).
#
# The cluster queue was congested when this round started (submitted job cj1807
# sat queued while the dispatcher showed no running jobs), so the wave is
# executed on the worktree host in three single-threaded lanes.  Every run is
# the same entry point the cluster template uses (run_case.py), so provenance
# (impl_sha256 in each JSON) is identical; only the host differs.
#
# Usage: bash benchmarks/case_rcp_deposition/run_round2.sh [lane]
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=benchmarks/case_rcp_deposition/runs
LOGS=$OUT/logs
mkdir -p "$OUT" "$LOGS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src

run() {  # name n mu seed kn dtfrac
  local name=$1 n=$2 mu=$3 seed=$4 kn=$5 dtf=$6
  echo "[$(date +%H:%M:%S)] START $name" >> "$LOGS/round2.log"
  nice -n 5 python3 benchmarks/case_rcp_deposition/run_case.py \
      --n "$n" --mu "$mu" --seed "$seed" --kn "$kn" --dt-frac "$dtf" --out "$OUT" \
      > "$LOGS/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name rc=$?" >> "$LOGS/round2.log"
}

case "${1:-all}" in
  a)
    run prod_mu0_s2      10000 0.0   2 64000  0.05
    run prod_mu0.1_s2    10000 0.1   2 64000  0.05
    run fric_mu0.05      10000 0.05  1 64000  0.05
    run dtprobe_mu0      10000 0.0   1 64000  0.025
    ;;
  b)
    run prod_mu0_s3      10000 0.0   3 64000  0.05
    run prod_mu0.1_s3    10000 0.1   3 64000  0.05
    run fric_mu0.2       10000 0.2   1 64000  0.05
    run dtprobe_mu0.1    10000 0.1   1 64000  0.025
    ;;
  c)
    run stiff_mu0.1_kn256000 10000 0.1 1 256000 0.05
    ;;
  *)
    echo "lanes: a b c" >&2
    ;;
esac