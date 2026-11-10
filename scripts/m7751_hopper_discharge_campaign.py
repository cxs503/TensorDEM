#!/usr/bin/env python3
"""Driver for the m7751 hopper-discharge verification campaign (TensorDEM).

Usage
-----
    python scripts/m7751_hopper_discharge_campaign.py --list
    python scripts/m7751_hopper_discharge_campaign.py --case discharge_D100_s11 \
        --out-dir runs/hopper_discharge
    python scripts/m7751_hopper_discharge_campaign.py --shard 3/8 \
        --out-dir runs/hopper_discharge
    python scripts/m7751_hopper_discharge_campaign.py --merge \
        --out-dir runs/hopper_discharge --artifacts artifacts/hopper_discharge

``--shard i/n`` runs the cases with ``index % n == i - 1`` so the campaign can
be spread over the compute cluster; ``--merge`` reduces the collected records
to the acceptance table and writes ``case.json`` / ``results.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from verification.granular.hopper_discharge.campaign import (  # noqa: E402
    PROTOTYPE,
    campaign_plan,
    evaluate_criteria,
    load_records,
    run_face,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print the case plan")
    ap.add_argument("--case", help="run a single case id")
    ap.add_argument("--shard", help="i/n: run the i-th of n shards (1-based)")
    ap.add_argument("--faces", help="comma separated face filter, e.g. repose,janssen")
    ap.add_argument("--out-dir", default="runs/hopper_discharge")
    ap.add_argument("--artifacts", default="artifacts/hopper_discharge")
    ap.add_argument("--merge", action="store_true",
                    help="reduce collected records to case.json/results.json")
    args = ap.parse_args(argv)

    plan = campaign_plan()

    if args.list:
        print(json.dumps(plan, indent=2))
        return 0

    if args.merge:
        out_dir = Path(args.out_dir)
        records = load_records(out_dir)
        print(f"loaded {len(records)} case records from {out_dir}")
        criteria = evaluate_criteria(records)
        art = Path(args.artifacts)
        art.mkdir(parents=True, exist_ok=True)
        payload = {
            "mission": "m7751",
            "node": "P84W2-N05",
            "repository": "TensorDEM",
            "surface": "granular/hopper_discharge",
            "prototype": PROTOTYPE,
            "plan": plan,
            "n_cases": len(records),
            "criteria": criteria["criteria"],
            "n_passed": criteria["n_passed"],
            "n_criteria": criteria["n_criteria"],
            "all_passed": criteria["all_passed"],
            "summary": criteria["summary"],
            "runs": {cid: {k: v for k, v in rec.items() if k != "case"}
                     for cid, rec in sorted(records.items())},
        }
        (art / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        case_payload = {
            "mission": "m7751",
            "node": "P84W2-N05",
            "surface": "granular/hopper_discharge",
            "plan": plan,
            "cases": {cid: rec["case"] for cid, rec in sorted(records.items())},
        }
        (art / "case.json").write_text(json.dumps(case_payload, indent=2, sort_keys=True))
        print(f"wrote {art/'results.json'} and {art/'case.json'}")
        for c in criteria["criteria"]:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['criterion']}: "
                  f"{c['value']} ({c['threshold']})")
        return 0 if criteria["all_passed"] else 0

    if args.case:
        entries = [e for e in plan if e["case_id"] == args.case]
        if not entries:
            print(f"unknown case {args.case}", file=sys.stderr)
            return 2
    elif args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        if not 1 <= i <= n:
            print("shard index out of range", file=sys.stderr)
            return 2
        entries = [e for idx, e in enumerate(plan) if idx % n == i - 1]
    else:
        entries = plan
    if args.faces:
        wanted = {f.strip() for f in args.faces.split(",") if f.strip()}
        entries = [e for e in entries if e["face"] in wanted]

    out_dir = Path(args.out_dir)
    t0 = time.time()
    for entry in entries:
        t_case = time.time()
        path = run_face(entry, out_dir)
        print(f"[{entry['case_id']}] {time.time() - t_case:.1f} s -> {path}",
              flush=True)
    print(f"finished {len(entries)} cases in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())