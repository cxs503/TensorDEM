#!/usr/bin/env bash
# CASE-TRIAXIAL-CD: full matrix (ladder + sensitivity + convergence) + aggregation.
#
#   bash benchmarks/case_triaxial_cd/run_all.sh [OUT_DIR] [JOBS]
#
# Each point is one process; the points are independent (same seed, deterministic
# apparatus), so they can be sharded across cores or submitted to a cluster.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$ROOT/artifacts/case_triaxial_cd}"
JOBS="${2:-8}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
cd "$ROOT"

run() { python3 benchmarks/case_triaxial_cd/run_case.py --out "$OUT" "$@"; }

pids=()
launch() {
  run "$@" &
  pids+=($!)
  if (( ${#pids[@]} >= JOBS )); then
    wait "${pids[0]}" || true
    pids=("${pids[@]:1}")
  fi
}

# --- confining-pressure ladder (base material: mu = 0.5, mu_r = 0.25) ---
launch --tag ladder_s50000  --sigma3 5.0e4  --mu 0.5 --mu-r 0.25
launch --tag ladder_s100000 --sigma3 1.0e5  --mu 0.5 --mu-r 0.25
launch --tag ladder_s200000 --sigma3 2.0e5  --mu 0.5 --mu-r 0.25
# --- interparticle-friction sensitivity ---
launch --tag sens_mu_0p30 --sigma3 1.0e5 --mu 0.30 --mu-r 0.25
launch --tag sens_mu_0p70 --sigma3 1.0e5 --mu 0.70 --mu-r 0.25
# --- rolling-resistance (angularity proxy) sensitivity ---
launch --tag sens_mur_0p10 --sigma3 1.0e5 --mu 0.5 --mu-r 0.10
launch --tag sens_mur_0p40 --sigma3 1.0e5 --mu 0.5 --mu-r 0.40
# --- convergence: strain rate (quasi-static limit) ---
launch --tag conv_rate_0p25 --sigma3 1.0e5 --mu 0.5 --mu-r 0.25 --rate 0.25 --strain 0.20
# --- convergence: specimen size (96 vs 360 particles) ---
launch --tag conv_size_n96 --sigma3 1.0e5 --mu 0.5 --mu-r 0.25 --nx 4 --ny 4 --nz 3
for pid in "${pids[@]}"; do wait "$pid" || true; done

python3 benchmarks/case_triaxial_cd/run_case.py --aggregate --out "$OUT"
echo "done: $OUT"