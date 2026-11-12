"""Cell-list (candidate-pair search) timing + digest bench for ``IceDEM3D``.

Mission m16007 / GAP-DEM-G6.  The function measured here is
``IceDEM3D._contact_candidate_pairs`` -- the uniform-cell-list search that is
rebuilt on every force evaluation.  The bench does three things at once so the
same run can serve as *before* and *after* evidence:

1. **Timing**: ``_contact_candidate_pairs`` and a full ``step()`` at a size
   ladder (1.3 k -> 103 k particles).
2. **Digests**: SHA-256 of the pair list in *ordered* form (the exact tensor the
   solver consumes) and in *canonical* form (sorted rows).  Comparing these two
   files before/after a change answers "did the search return the same pairs, in
   the same order?" without storing O(1e6) pairs.
3. **Trajectory digest**: K steps of the solver followed by a hash of the
   particle state.  An identical digest before/after means the change is
   bit-for-bit transparent to the physics.

Usage
-----
    PYTHONPATH=src python scripts/m16007_cell_list_bench.py --out artifacts/cell_list/baseline.json
    PYTHONPATH=src python scripts/m16007_cell_list_bench.py --out artifacts/cell_list/vectorized.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from tensordem.dem3d import IceDEM3D, DEM3DConfig

LADDER = [(11, 11, 11), (22, 22, 22), (37, 37, 37), (47, 47, 47)]


def _digest(t: torch.Tensor) -> str:
    arr = t.detach().to("cpu", torch.int64).numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _canonical_digest(t: torch.Tensor) -> str:
    arr = t.detach().to("cpu", torch.int64).numpy()
    if arr.size:
        arr = arr[arr[:, 0].argsort(kind="stable")]
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _state_digest(sim: IceDEM3D) -> str:
    h = hashlib.sha256()
    for name in ("positions", "velocities"):
        h.update(_digest(getattr(sim, name)).encode())
    return h.hexdigest()


def bench_case(nx: int, ny: int, nz: int, *, repeat: int, steps: int) -> dict:
    t_build = time.perf_counter()
    sim = IceDEM3D(DEM3DConfig(nx=nx, ny=ny, nz=nz))
    t_build = time.perf_counter() - t_build
    n = len(sim.positions)
    sim.forces()  # warm-up (torch kernels, allocator)

    t0 = time.perf_counter()
    for _ in range(repeat):
        pairs = sim._contact_candidate_pairs()
    t_pairs = (time.perf_counter() - t0) / repeat

    ordered = _digest(pairs)
    canonical = _canonical_digest(pairs)

    t0 = time.perf_counter()
    for _ in range(repeat):
        sim.step()
    t_step = (time.perf_counter() - t0) / repeat

    # Trajectory: fresh deterministic state, K steps, hash.
    ref = IceDEM3D(DEM3DConfig(nx=nx, ny=ny, nz=nz))
    for _ in range(steps):
        ref.step()

    return {
        "grid": [nx, ny, nz],
        "n_particles": n,
        "n_candidate_pairs": int(pairs.shape[0]),
        "pairs_ordered_sha256": ordered,
        "pairs_canonical_sha256": canonical,
        "build_s": round(t_build, 4),
        "t_pairs_s": round(t_pairs, 6),
        "t_step_s": round(t_step, 6),
        "pairs_fraction_of_step": round(t_pairs / t_step, 4),
        "trajectory_sha256": _state_digest(ref),
        "trajectory_steps": steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="", help="comma list of g (cubic g^3 grids); default = built-in ladder")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.sizes:
        ladder = [(int(g),) * 3 for g in args.sizes.split(",")]
    else:
        ladder = LADDER

    cases = [bench_case(nx, ny, nz, repeat=args.repeat, steps=args.steps) for nx, ny, nz in ladder]
    payload = {
        "mission": "m16007",
        "gap": "GAP-DEM-G6",
        "function": "IceDEM3D._contact_candidate_pairs",
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "repeat": args.repeat,
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    for c in cases:
        print(
            f"N={c['n_particles']:>7d} pairs={c['n_candidate_pairs']:>8d} "
            f"t_pairs={c['t_pairs_s']*1e3:9.2f} ms  t_step={c['t_step_s']*1e3:9.1f} ms  "
            f"pairs/step={c['pairs_fraction_of_step']*100:5.1f}%"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()