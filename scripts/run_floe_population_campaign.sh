#!/usr/bin/env bash
# Produce the floe-population campaign artefacts (node p80w2-n05).
#
#   artifacts/floe_population/study_summary.json  criteria ledger C1..C7
#   artifacts/floe_population/*.png               campaign figures
#   runs/floe_population/*.json                   raw stage records
#
# The three DEM experiments are independent per seed, so they are run in
# parallel processes; the field stage needs the breakage kernel and therefore
# starts after the breakage stages have finished.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
SEEDS="${SEEDS:-0 1 2}"
PY="${PYTHON:-python}"

mkdir -p runs/floe_population artifacts/floe_population

for seed in $SEEDS; do
  "$PY" -m tensordem.floe_population.study --stage collision --seed "$seed" &
  "$PY" -m tensordem.floe_population.study --stage breakage  --seed "$seed" &
  "$PY" -m tensordem.floe_population.study --stage control   --seed "$seed" &
done
wait

# the breakage kernel is fitted from every seed's impact campaign
"$PY" - <<'PY'
import json
import tensordem.floe_population.study as S
kernel = S._kernel_from_runs(S.RUN_DIR, tuple(int(s) for s in __import__("os").environ.get("SEEDS", "0 1 2").split()))
S._write(S.RUN_DIR / "breakage_kernel.json", kernel)
print("breakage kernel:", {k: round(v, 4) for k, v in kernel.items() if isinstance(v, float)})
PY

for seed in $SEEDS; do
  "$PY" -m tensordem.floe_population.study --stage field --seed "$seed" \
        --breakage-kernel runs/floe_population/breakage_kernel.json &
done
wait

"$PY" -m tensordem.floe_population.study --stage assemble --seeds $SEEDS \
      --breakage-kernel runs/floe_population/breakage_kernel.json