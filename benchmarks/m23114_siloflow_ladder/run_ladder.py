#!/usr/bin/env python3
"""Shard driver for the m23114 numerical ladder (compute-gate friendly).

One shard runs exactly one (level, seed) point, so the ladder can be spread over
the cluster without any shared state:

    PYTHONPATH=src python benchmarks/m23114_siloflow_ladder/run_ladder.py --shard 1 --nshards 12
    PYTHONPATH=src python benchmarks/m23114_siloflow_ladder/run_ladder.py --list

The task list is frozen here (see `prereg.md` §2) so a shard index is
reproducible: the point that shard *i* computes never depends on what has
already finished.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "runs" / "m23114_siloflow_ladder"

#: frozen task list — order matters (shard i computes TASKS[i-1])
TASKS = [(level, seed)
         for level in ("nom20", "nom40", "nom80", "kn2e4", "kn8e4")
         for seed in (1, 2)]


def out_path(level: str, seed: int) -> Path:
    return OUT / f"{level}_s{seed}.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args(argv)

    if args.list:
        for i, (level, seed) in enumerate(TASKS, start=1):
            print(f"{i:2d} {level:>6s} seed={seed} -> {out_path(level, seed).relative_to(ROOT)}")
        return 0

    if not (1 <= args.shard <= args.nshards <= len(TASKS)):
        raise SystemExit(f"shard {args.shard}/{args.nshards} outside 1..{len(TASKS)}")
    todo = [TASKS[i] for i in range(args.shard - 1, len(TASKS), args.nshards)]
    for level, seed in todo:
        out = out_path(level, seed)
        if out.exists() and json.loads(out.read_text()).get("m23114"):
            print(f"skip (done) {level} seed {seed}")
            continue
        if out.exists():
            out.unlink()
        cmd = [sys.executable, str(HERE / "run_level.py"), "--level", level,
               "--seed", str(seed), "--out", str(out)]
        print(" ".join(cmd), flush=True)
        if subprocess.run(cmd, cwd=str(ROOT)).returncode != 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())