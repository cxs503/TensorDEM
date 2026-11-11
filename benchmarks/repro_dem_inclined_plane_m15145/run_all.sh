#!/bin/bash
# run the whole reproduction campaign, 6 parallel workers
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src
run_one () {
  out=$(echo "$1" | sed -n 's/.*--out \(.*\)\.json/\1/p')
  mkdir -p "$(dirname "$out")"
  python benchmarks/repro_dem_inclined_plane_m15145/run_point.py $1 > "$out.log" 2>&1
}
export -f run_one
cat benchmarks/repro_dem_inclined_plane_m15145/jobs.txt | xargs -P 6 -I{} bash -c 'run_one "{}"'
echo DONE > runs/chute/ALLDONE
