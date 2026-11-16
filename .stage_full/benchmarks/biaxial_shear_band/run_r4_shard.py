"""Shard wrapper for the R4 cumulative ensemble (m20774).

The cluster queue on 2026-11-14 is deep for 8-core jobs, so the 11-specimen
R4 DEM half is submitted as 11 single-specimen shards (one per specimen,
disjoint cache entries).  Each shard reuses the *identical* production code
path (``_run_one`` / ``_run_cached`` of ``run_r4_cumulative_ensemble.py``)
with the predeclared specimen list, so the numbers are bit-identical to a
single 8-worker run.  After all shards finish, the local driver re-assembles
the ensemble JSON from the per-specimen cache with ``--resume``.

Usage (one shard): ``python benchmarks/biaxial_shear_band/run_r4_shard.py <i>``
with ``i in 0..10``.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

# Predeclared R4 specimen list (run_r4_cumulative_ensemble.py):
# 8 anchor specimens at p = 100 kPa (out-of-sample seeds, disjoint from R3),
# then 3 confinement-sensitivity specimens at p = 200 kPa.
SPECIMENS = [
    (53, 100.0e3), (67, 100.0e3), (79, 100.0e3), (83, 100.0e3),
    (89, 100.0e3), (97, 100.0e3), (101, 100.0e3), (103, 100.0e3),
    (53, 200.0e3), (67, 200.0e3), (79, 200.0e3),
]


def main() -> int:
    i = int(sys.argv[1])
    seed, p = SPECIMENS[i]
    from benchmarks.biaxial_shear_band.run_r4_cumulative_ensemble import _run_cached

    cfg = {
        "name": ("anchor" if p == 100.0e3 else "p200") + f"_s{seed}",
        "axis": "anchor" if p == 100.0e3 else "confining_pressure",
        "seed": seed,
        "confining_stress": p,
        "quick": False,
    }
    _run_cached(cfg, os.path.join(ROOT, "artifacts", "biaxial_shear_band", "r4_cache"), True)
    print(f"[shard {i}] {cfg['name']} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
