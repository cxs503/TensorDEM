#!/bin/sh
# m8235 silo-flow verification campaign: every case is an independent
# single-core process, so the matrix is sharded with xargs -P.
#
#   OUT=runs/m8235_silo_flow sh scripts/m8235_silo_flow_campaign.sh
#
# Stages: 1 Beverloo orifice sweep (D/d = 4,5,6,7), 2 Janssen deep silo,
#         3 hopper regime sweep (6 angles x 2 wall frictions), 4 energy ledger.
set -u
OUT=${OUT:-runs/m8235_silo_flow}
PY=${PYTHON:-python3}
mkdir -p "$OUT"
export PYTHONPATH=src

cat > /tmp/m8235_jobs.txt <<EOF
$PY -u -m tensordem.silo_flow.study beverloo --ratio 4 --silo-radius 0.045 --fill-height 0.10 --wall-mu 0.30 --out $OUT/beverloo_d4.json
$PY -u -m tensordem.silo_flow.study beverloo --ratio 6 --silo-radius 0.045 --fill-height 0.10 --wall-mu 0.30 --out $OUT/beverloo_d6.json
$PY -u -m tensordem.silo_flow.study beverloo --ratio 7 --silo-radius 0.045 --fill-height 0.10 --wall-mu 0.30 --out $OUT/beverloo_d7.json
$PY -u -m tensordem.silo_flow.study janssen --silo-radius 0.025 --fill-height 0.30 --outlet-radius 0.008 --wall-mu 0.40 --n-layers 20 --out $OUT/janssen_muw40.json
$PY -u -m tensordem.silo_flow.study hopper --angle 20 --silo-radius 0.035 --outlet-radius 0.010 --cylinder-height 0.060 --wall-mu 0.40 --out $OUT/hopper_a20_muw40.json
$PY -u -m tensordem.silo_flow.study hopper --angle 30 --silo-radius 0.035 --outlet-radius 0.010 --cylinder-height 0.060 --wall-mu 0.40 --out $OUT/hopper_a30_muw40.json
$PY -u -m tensordem.silo_flow.study hopper --angle 40 --silo-radius 0.035 --outlet-radius 0.010 --cylinder-height 0.060 --wall-mu 0.40 --out $OUT/hopper_a40_muw40.json
$PY -u -m tensordem.silo_flow.study hopper --angle 50 --silo-radius 0.035 --outlet-radius 0.010 --cylinder-height 0.060 --wall-mu 0.40 --out $OUT/hopper_a50_muw40.json
$PY -u -m tensordem.silo_flow.study hopper --angle 60 --silo-radius 0.035 --outlet-radius 0.010 --cylinder-height 0.060 --wall-mu 0.40 --out $OUT/hopper_a60_muw40.json
$PY -u -m tensordem.silo_flow.study hopper --angle 70 --silo-radius 0.035 --outlet-radius 0.010 --cylinder-height 0.060 --wall-mu 0.40 --out $OUT/hopper_a70_muw40.json
$PY -u -m tensordem.silo_flow.study hopper --angle 30 --silo-radius 0.035 --outlet-radius 0.010 --cylinder-height 0.060 --wall-mu 0.70 --out $OUT/hopper_a30_muw70.json
$PY -u -m tensordem.silo_flow.study hopper --angle 45 --silo-radius 0.035 --outlet-radius 0.010 --cylinder-height 0.060 --wall-mu 0.70 --out $OUT/hopper_a45_muw70.json
$PY -u -m tensordem.silo_flow.study hopper --angle 60 --silo-radius 0.035 --outlet-radius 0.010 --cylinder-height 0.060 --wall-mu 0.70 --out $OUT/hopper_a60_muw70.json
$PY -u -m tensordem.silo_flow.study energy --silo-radius 0.035 --fill-height 0.080 --orifice-ratio 5 --wall-mu 0.30 --out $OUT/energy_dt.json
EOF

echo "[$(date +%H:%M:%S)] launching $(wc -l < /tmp/m8235_jobs.txt) cases"
xargs -P "${JOBS:-12}" -I CMD sh -c 'echo "[$(date +%H:%M:%S)] start CMD"; CMD; echo "[$(date +%H:%M:%S)] rc=$? CMD"' < /tmp/m8235_jobs.txt
echo "[$(date +%H:%M:%S)] campaign done"