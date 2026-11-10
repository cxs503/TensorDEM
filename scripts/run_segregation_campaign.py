"""Run the whole segregation campaign in parallel and write one JSON per case.

One process per case (the solver is single-threaded NumPy), so a 128-core node
finishes the entire matrix in the wall time of its slowest case.  A failing case
(``BoxTooSmall``, a stalled packing, ...) is recorded as an error entry instead of
aborting the campaign: an honest FAIL of one point must not hide the others.

Usage::

    PYTHONPATH=src python scripts/run_segregation_campaign.py \
        --spec runs/segregation/specs.json --out runs/segregation/out --jobs 49
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from multiprocessing import Pool

from tensordem.segregation.campaign import CaseSpec, run_case



def _run_one(payload: tuple[dict, str]) -> tuple[str, str]:
    spec_dict, out_dir = payload
    spec = CaseSpec(**spec_dict)
    path = os.path.join(out_dir, f"{spec.name}.json")
    try:
        record = run_case(spec)
        with open(path + ".tmp", "w") as fh:
            json.dump(record, fh)
        os.replace(path + ".tmp", path)
        return spec.name, f"ok wall={record['audit']['wall_clock_s']:.0f}s"
    except Exception as exc:  # noqa: BLE001 - one failing case must not stop the run
        err = {
            "spec": spec.to_json(),
            "error": repr(exc),
            "traceback": traceback.format_exc()[-2000:],
        }
        with open(path + ".tmp", "w") as fh:
            json.dump(err, fh)
        os.replace(path + ".tmp", path)
        return spec.name, f"ERROR {exc!r}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default=None, help="comma separated case names")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    with open(args.spec) as fh:
        specs = [CaseSpec(**s) for s in json.load(fh)]
    if args.only:
        keep = set(args.only.split(","))
        specs = [s for s in specs if s.name in keep]
    todo = [s.to_json() for s in specs
            if not os.path.exists(os.path.join(args.out, f"{s.name}.json"))]
    print(f"{len(specs)} cases, {len(todo)} to run, jobs={args.jobs}", flush=True)
    with Pool(processes=min(args.jobs, max(len(todo), 1))) as pool:
        for name, status in pool.imap_unordered(
            _run_one, [(s, args.out) for s in todo]
        ):
            print(f"[{name}] {status}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())