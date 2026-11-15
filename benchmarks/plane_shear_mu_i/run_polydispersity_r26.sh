#!/usr/bin/env bash
# CASE-DEM-PLANE-SHEAR-MUIDI, round 26 (m22550): does polydispersity break the
# compaction crystal that round 25 identified as the blocking defect?
#
# Round 25 result: a disordered seed does not survive the constant-P compaction of
# a *monodisperse* disk bed - the packing re-crystallises (psi6 = 0.914...0.957,
# hexatic fraction up to 0.93, Z ~ 4) and mu_wall stays on the frictionless branch
# (0.059...0.129 against the law's 0.246...0.359).  Round 25 recommendation (2):
# break the symmetry *physically* with polydispersity, which is exactly what the
# archived law's provenance note allows (the friction law is preserved up to 50 %
# polydispersity).
#
# Matrix: probe cell (458 disks, W = 0.30 m, gap = 0.15 m, P = 10 kPa, mu_p = 0.4,
# disordered seed, seed = 0) x I in {0.02, 0.05, 0.2} x w in {0.0, 0.25, 0.5}.
# w = 0 is the round-25 control, so any change is attributable to the size spread.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=benchmarks/plane_shear_mu_i/runs/plane_shear_2d
mkdir -p "$OUT"
for I in 0.02 0.05 0.2; do
  for W in 0.0 0.25 0.5; do
    python3 benchmarks/plane_shear_mu_i/plane_shear_2d.py \
      --I "$I" --init random --seed 0 --polydispersity "$W" \
      --width 0.30 --gap 0.15 --np 458 --settle-units 40 \
      --shear-setup-strain 0.5 --strain-measure 1.0 \
      --out "$OUT/poly_w${W}_I${I}.json" \
      > "$OUT/poly_w${W}_I${I}.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  done
done
wait
echo "polydispersity sweep done"
