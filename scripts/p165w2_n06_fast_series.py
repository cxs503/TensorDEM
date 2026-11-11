"""Sharded driver for the P165-W2-N06 column-collapse series (array engine).

Run list (``--shard i --shards n`` picks the i-th case, so the series can be
spread over the cluster nodes):

======  ==============  ======================  =====================================
case    N (target)      dt scale / mu           what it measures
======  ==============  ======================  =====================================
n2k     2048            1.0                     particle-number series (criterion (f))
n4k     4096            1.0                     "
n8k     8192            1.0                     "
n16k    16384           1.0                     "
n32k    32768           1.0                     "
dt05    8192            0.5                     time-step convergence (f)
dt2     8192            2.0                     time-step convergence (f)
mu30    8192            1.0 (mu = 0.30)         material sensitivity
mu60    8192            1.0 (mu = 0.60)         material sensitivity
smooth  8192            1.0 (no rough bed)      negative control (no wall rolling
                                                resistance => no repose angle)
======  ==============  ======================  =====================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p165w2_n06_column_fast import run_case                     # noqa: E402

import numpy as np                                              # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "runs", "p165w2_n06", "fast")

CASES = [
    dict(name="n2k", n=2048),
    dict(name="n4k", n=4096),
    dict(name="n8k", n=8192),
    dict(name="n16k", n=16384),
    dict(name="n32k", n=32768),
    dict(name="dt05", n=8192, dt_scale=0.5),
    dict(name="dt2", n=8192, dt_scale=2.0),
    dict(name="mu30", n=8192, mu=0.30),
    dict(name="mu60", n=8192, mu=0.60),
    dict(name="smooth", n=8192, bed_layers=0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--aspect", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=16000)
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--plateau-deg", type=float, default=0.5)
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--outdir", type=str, default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    # the cluster {i} placeholder may be 0- or 1-based; take it modulo the count
    shard = int(args.shard) % int(args.shards)
    todo = [c for k, c in enumerate(CASES) if k % args.shards == shard]
    for case in todo:
        out = os.path.join(args.outdir, f"{case['name']}.json")
        if os.path.exists(out):
            print(f"skip {case['name']} (exists)", flush=True)
            continue
        t0 = time.perf_counter()
        res, cloud = run_case(case["n"], radius=args.radius,
                              mu=case.get("mu", 0.45),
                              aspect=args.aspect,
                              dt_scale=case.get("dt_scale", 1.0),
                              max_steps=args.max_steps, block=args.block,
                              plateau_deg=args.plateau_deg, jitter=args.jitter,
                              bed_layers=case.get("bed_layers", 2),
                              label=case["name"])
        np.savez_compressed(os.path.join(args.outdir, f"cloud_{case['name']}.npz"),
                            pos=cloud, radius=args.radius)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"{case['name']}: angle={res['repose_angle_deg']:.3f} deg "
              f"n={res['n_actual']} steps={res['settle_steps']} "
              f"plateau={res['plateau_reached']} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()