"""IN32-N02 stage 2+3+4: jamming density from energy minima.

Runs the preparation-protocol matrix (``growth`` / ``decompression`` /
``rsa_direct`` / ``fire_from_rsa``) x ``phi`` step 0.01 x seeds x
``N in {256, 512, 1024, 2048, 4096}`` on a **doubly periodic** cell, in reduced
units (``kn = 1``, mean radius 1, bidisperse ratio 1.4), and writes one JSON per
``(protocol, mu, N, seed)`` into ``artifacts/jamming_protocol/raw/``.

Every task is independent, so the driver is embarrassingly parallel
(``--workers``) and resumable (existing files are skipped).

Example
-------
    python benchmarks/jamming_protocol_ablation.py --workers 96
    python benchmarks/jamming_protocol_ablation.py --mu 0.5 --protocols growth --workers 32
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.granular2d.jamming_protocol import run_protocol  # noqa: E402

DEFAULT_NS = (256, 512, 1024, 2048, 4096)
DEFAULT_PROTOCOLS = ("growth", "decompression", "rsa_direct", "fire_from_rsa")
PHI_LO, PHI_HI, PHI_STEP = 0.78, 0.88, 0.01


def _phi_grid() -> list[float]:
    return [round(PHI_LO + k * PHI_STEP, 10)
            for k in range(int(round((PHI_HI - PHI_LO) / PHI_STEP)) + 1)]


def _run_one(args: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    out = run_protocol(
        args["protocol"], n=args["n"], seed=args["seed"], phi_grid=args["phi_grid"],
        dphi=PHI_STEP, mu=args["mu"], kt_ratio=args["kt_ratio"],
        method=args["method"], tol=args["tol"], max_iter=args["max_iter"],
        top_phi=args["top_phi"], placement_phi=args["placement_phi"])
    out["elapsed_s"] = time.time() - t0
    out["kt_ratio"] = args["kt_ratio"]
    out["phi_grid"] = args["phi_grid"]
    out["settings"] = {k: args[k] for k in ("tol", "max_iter", "top_phi", "placement_phi")}
    out["settings"]["dphi"] = PHI_STEP
    return out


def _path(root: Path, protocol: str, mu: float, n: int, seed: int) -> Path:
    return root / f"{protocol}_mu{mu:.2f}_N{n}_s{seed:03d}.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-list", default=",".join(str(v) for v in DEFAULT_NS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    ap.add_argument("--mu", type=float, nargs="+", default=[0.0, 0.5],
                    help="friction coefficients; a non-zero mu also needs --kt-ratio")
    ap.add_argument("--kt-ratio", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-iter", type=int, default=30000)
    ap.add_argument("--top-phi", type=float, default=0.90)
    ap.add_argument("--placement-phi", type=float, default=0.45)
    ap.add_argument("--method", default="fire", choices=["fire", "cg"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--out", default="artifacts/jamming_protocol/raw")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N tasks")
    args = ap.parse_args(argv)

    ns = [int(v) for v in args.n_list.split(",") if v.strip()]
    protocols = [v for v in args.protocols.split(",") if v.strip()]
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    grid = _phi_grid()

    tasks = []
    for mu in args.mu:
        for protocol in protocols:
            for n in ns:
                for k in range(args.seeds):
                    seed = args.seed_start + k
                    path = _path(root, protocol, mu, n, seed)
                    if path.exists():
                        continue
                    tasks.append({
                        "protocol": protocol, "n": n, "seed": seed, "mu": mu,
                        "kt_ratio": args.kt_ratio if mu > 0.0 else 0.0,
                        "method": args.method, "tol": args.tol,
                        "max_iter": args.max_iter, "top_phi": args.top_phi,
                        "placement_phi": args.placement_phi, "phi_grid": grid,
                        "path": str(path),
                    })
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} tasks, {args.workers} workers, out={root}", flush=True)
    if not tasks:
        return 0
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                out = fut.result()
            except Exception as exc:  # pragma: no cover - reported, not hidden
                print(f"FAIL {task['protocol']} mu={task['mu']} N={task['n']} "
                      f"seed={task['seed']}: {exc}", flush=True)
                continue
            Path(task["path"]).write_text(json.dumps(out))
            done += 1
            if done % 10 == 0 or done == len(tasks):
                rate = (time.time() - t0) / done
                print(f"{done}/{len(tasks)} tasks, {rate:.1f} s/task, "
                      f"eta {(len(tasks) - done) * rate / 60:.1f} min", flush=True)
    print(f"finished {done} tasks in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())