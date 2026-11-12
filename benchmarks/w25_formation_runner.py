"""W25 ``formation-ice-channel`` DEM runner.

Runs one shard of the pre-registered case matrix and writes one JSON per case
into ``--out``.  The script contains no physics: it builds
``tensordem.ice_channel_formation.FormationCase`` objects from
``tensordem.ice_channel_formation_cases.build_matrix()``, calls
``run_case`` and dumps the measured ledger.

Usage::

    python benchmarks/w25_formation_runner.py --list
    python benchmarks/w25_formation_runner.py --shard 3/49 --out benchmarks/artifacts/w25/raw
    python benchmarks/w25_formation_runner.py --only single_V1p00_reference ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src",):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from tensordem.ice_channel_formation import run_case  # noqa: E402
from tensordem.ice_channel_formation_cases import build_matrix, cost_proxy  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "benchmarks" / "artifacts" / "w25" / "raw"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list the case matrix and exit")
    parser.add_argument("--shard", default=None, help="i/n (1-based) shard of the matrix")
    parser.add_argument("--only", nargs="*", default=None, help="run these labels only")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true", help="overwrite existing results")
    args = parser.parse_args(argv)

    matrix = build_matrix()
    if args.list:
        for item in matrix:
            print(item.label)
        return 0

    cases = matrix
    if args.only:
        wanted = set(args.only)
        cases = [item for item in matrix if item.label in wanted]
        missing = wanted - {item.label for item in matrix}
        if missing:
            parser.error(f"unknown labels: {sorted(missing)}")
    elif args.shard:
        index, total = (int(part) for part in args.shard.split("/"))
        if not 1 <= index <= total:
            parser.error("shard must be i/n with 1 <= i <= n")
        # cost-balanced round robin: the heaviest cases land on distinct shards
        ordered = sorted(matrix, key=cost_proxy, reverse=True)
        cases = [item for i, item in enumerate(ordered) if i % total == index - 1]

    out_dir = Path(args.out)
    started = time.time()
    for item in cases:
        target = out_dir / f"{item.label}.json"
        if target.exists() and not args.force:
            print(f"[skip] {item.label} (exists)", flush=True)
            continue
        t0 = time.time()
        print(f"[run ] {item.label} n_ships={item.n_ships} V={item.speed_m_s} "
              f"S={item.spacing_m:.0f}m L_field={item.field_length_m:.0f}m "
              f"r={item.mean_radius_refined_m:.2f}m dt={item.dt:.4f}", flush=True)
        result = run_case(item, progress=True)
        wall = time.time() - t0
        result["wall_time_s"] = wall
        _write(target, result)
        print(f"[done] {item.label} steps={result['steps_done']} "
              f"t={result['sim_time_s']:.1f}s n_floes={result['n_floes']} "
              f"load={result['formation_total_load_N'] / 1e3:.2f}kN "
              f"({wall:.1f}s)", flush=True)
    print(f"[shard] {len(cases)} case(s) in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())