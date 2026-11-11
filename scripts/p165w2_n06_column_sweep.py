"""Sharded driver for the P165-W2-N06 column-collapse sweep.

Each shard runs exactly one simulation from the run table below, so the sweep can
be spread over the cluster (``run_cluster`` with ``shards=N``); every run writes
its own JSON to ``runs/p165w2_n06/`` and
``scripts/p165w2_n06_column_summary.py`` merges them into
``artifacts/dem/p165w2_n06/column_collapse.json``.

Axes
----
* particle-number convergence at fixed particle size (N = 2 k ... 16 k)
* friction scan (mu = 0.30 / 0.45 / 0.60) at the smallest N
* time-step convergence (dt and dt/2 of the stability estimate) at the smallest N

Usage::

    PYTHONPATH=src python scripts/p165w2_n06_column_sweep.py --shard 3 --shards 7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from p165w2_n06_column_collapse import run_case                   # noqa: E402

REPO = os.path.dirname(HERE)
OUTDIR = os.path.join(REPO, "runs", "p165w2_n06")

#: (label, kwargs for run_case)
#: ``gravity`` is scaled up 1.5x relative to 9.81 m/s^2 so that the free-fall
#: time of the squat column fits the engine throughput (measured 0.5 s/step at
#: N ~ 2 k).  The scaling is *identical for every run*, so the particle-number
#: and friction comparisons are unaffected; the gravity sensitivity is measured
#: separately by the ``n2k_g9.81`` run, which repeats the reference case at
#: physical gravity.
RUNS = [
    ("n2k_mu0.45", dict(n_target=2000, mu=0.45, gravity=15.0, max_steps=12000)),
    ("n4k_mu0.45", dict(n_target=4000, mu=0.45, gravity=15.0, max_steps=9000)),
    ("n8k_mu0.45", dict(n_target=8000, mu=0.45, gravity=15.0, max_steps=6000)),
    ("n16k_mu0.45", dict(n_target=16000, mu=0.45, gravity=15.0, max_steps=3500)),
    ("n32k_mu0.45", dict(n_target=32000, mu=0.45, gravity=15.0, max_steps=1800)),
    ("n2k_mu0.30", dict(n_target=2000, mu=0.30, gravity=15.0, max_steps=12000)),
    ("n2k_mu0.60", dict(n_target=2000, mu=0.60, gravity=15.0, max_steps=12000)),
    ("n2k_dt2", dict(n_target=2000, mu=0.45, gravity=15.0, dt_scale=2.0, max_steps=6000)),
    ("n2k_seed7", dict(n_target=2000, mu=0.45, gravity=15.0, seed=7, max_steps=12000)),
    ("n2k_g9.81", dict(n_target=2000, mu=0.45, gravity=9.81, max_steps=16000)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=len(RUNS))
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--budget-s", type=float, default=6900.0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    for k in range((args.shard - 1) % max(args.shards, 1), len(RUNS), args.shards):
        if k >= len(RUNS):
            continue
        label, kwargs = RUNS[k]
        path = os.path.join(OUTDIR, f"column_{label}.json")
        if os.path.exists(path):
            print(f"[shard {args.shard}] {label} already done -> {path}", flush=True)
            continue
        kw = dict(kwargs)
        kw.setdefault("max_steps", args.max_steps)
        # keep each shard inside the job budget: the plateau loop stops on its own
        t0 = time.perf_counter()
        print(f"[shard {args.shard}] start {label} {kw}", flush=True)
        res = run_case(label=label, **kw)
        res["shard"] = args.shard
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"[shard {args.shard}] done {label}: theta={res['repose_angle_deg']:.2f} "
              f"n={res['n_actual']} steps={res['settle_steps']} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()