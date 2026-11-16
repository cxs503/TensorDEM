#!/usr/bin/env bash
# CASE-DEM-PLANE-SHEAR-MUIDI, round 27 (m22662): strain-resolved saturation runs.
#
# Why this matrix exists
# ----------------------
# Round 26 left the mu(I)/phi(I) sweep (`cj4227`) unmeasured and, in the archived
# probe series, mu_wall *falls* through the pre-registered measurement window
# (I = 0.05: first half 0.1215, second half 0.0792 -- a 35 % drop inside one
# strain; audits/round 27 quantify this on the stored time series).  A quantity
# that is still moving when the window closes is not a steady-state mu, so the
# ledger verdict computed from a fixed 1-strain window is a verdict on a
# transient, not on the rheology.
#
# This matrix holds *the device fixed* (same probe cell, same protocol) and
# changes exactly one thing: the measurement window is long enough (1 + 8
# strains instead of 0.5 + 1) to see whether mu saturates and where.  The
# no-polydispersity control (w = 0) is included so that any saturation level can
# be attributed to the crystal fix (round 26) rather than to the longer window.
#
#   w = 0.5  I = 0.02 / 0.05 / 0.2   polydisperse, crystal-free (round 26 seed work)
#   w = 0.0  I = 0.05                monodisperse control (round 25 known crystal)
#
# Cost: ~1e5-3e5 steps per shard, 4 shards in parallel, one core each.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=benchmarks/plane_shear_mu_i/runs/plane_shear_2d
mkdir -p "$OUT"
TAG=r27sat

run() {  # run <w> <I>
  local W="$1" I="$2"
  python3 benchmarks/plane_shear_mu_i/plane_shear_2d.py \
    --I "$I" --init random --seed 0 --polydispersity "$W" \
    --width 0.30 --gap 0.15 --np 458 --settle-units 40 \
    --shear-setup-strain 1.0 --strain-measure 8.0 \
    --out "$OUT/${TAG}_w${W}_I${I}.json" \
    > "$OUT/${TAG}_w${W}_I${I}.log" 2>&1
}

run 0.5 0.02 & run 0.5 0.05 & run 0.5 0.2 & run 0.0 0.05 &
wait
echo "strain-saturation sweep done"
ls -l "$OUT"/${TAG}_*.json