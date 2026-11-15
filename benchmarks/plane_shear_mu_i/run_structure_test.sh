#!/usr/bin/env bash
# Round 25 (m22440): initial-condition structure test.
#
# Same probe cell as the returned sweep (458 disks, W = 0.30 m, gap = 0.15 m,
# P = 10 kPa, mu_p = 0.4, settle 40 d/sqrt(P/rho), 1.0 strain of measurement),
# two initial packings:
#   lattice -> the round-24 triangular-lattice seed (a crystal seed)
#   random  -> the same density, disordered topology (_random_packing)
# The question: does the -61...-72 % mu deficit of the returned sweep come from
# the crystalline initial condition rather than from the dimension?
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=benchmarks/plane_shear_mu_i/runs/plane_shear_2d
mkdir -p "$OUT"
for I in 0.02 0.05 0.2; do
  for INIT in lattice random; do
    python3 benchmarks/plane_shear_mu_i/plane_shear_2d.py \
      --I "$I" --init "$INIT" --seed 0 \
      --width 0.30 --gap 0.15 --np 458 --settle-units 40 \
      --shear-setup-strain 0.5 --strain-measure 1.0 \
      --out "$OUT/structure_${INIT}_I${I}.json" \
      > "$OUT/structure_${INIT}_I${I}.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  done
done
wait
echo "structure test done"