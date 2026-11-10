#!/bin/sh
# Wait for the local campaign workers, then build every deliverable.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for i in $(seq 1 60); do
  n=$(ps ax | grep -c "[r]un_bidisperse_campaign.py")
  if [ "$n" -eq 0 ]; then break; fi
  sleep 30
done
mkdir -p runs/all
cp runs/probe/case_*.json runs/all/ 2>/dev/null || true
cp runs/ratios/case_*.json runs/all/ 2>/dev/null || true
python scripts/run_bidisperse_campaign.py --mode probe --outdir runs/all \
  --merge --report artifacts/bidisperse_campaign.json > /tmp/p155_merge.log 2>&1
python scripts/audit_bidisperse_criteria.py --inputs runs/all \
  --out artifacts/bidisperse_segregation_criteria.json > /tmp/p155_audit.log 2>&1
python scripts/make_bidisperse_figures.py \
  --report artifacts/bidisperse_campaign.json \
  --out artifacts/bidisperse_segregation.png > /tmp/p155_fig.log 2>&1
python scripts/make_bidisperse_report.py \
  --criteria artifacts/bidisperse_segregation_criteria.json \
  --campaign artifacts/bidisperse_campaign.json \
  --figure artifacts/bidisperse_segregation.png \
  --doc docs/bidisperse_segregation_verification.md \
  --payload /tmp/report.json > /tmp/p155_report.log 2>&1
echo FINALIZE_OK > /tmp/p155_done
