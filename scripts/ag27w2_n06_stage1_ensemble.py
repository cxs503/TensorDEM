"""Stage 1 of node AG27W2-N06: build the independent-repeat DEM wear ensemble.

Runs ``K`` independent ``run_wear_season`` DEM wear seasons through
``dem_activity_rate_field`` (different contact-ensemble seed = different initial
packing of the ice fragments, identical physics) and caches the DEM-grid rate
fields to ``runs/AG27W2_N06/dem_ensemble.npz``.

Usage::

    PYTHONPATH=src python scripts/ag27w2_n06_stage1_ensemble.py [--n 41]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensordem import ice_abrasion as ia
from tensordem import ice_abrasion_spatiotemporal as st

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "AG27W2_N06"
CACHE = RUN_DIR / "dem_ensemble.npz"

BASE_SEED = 20261106
SEED_STRIDE = 37


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=41)
    ap.add_argument("--events", type=int, default=20000)
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if args.reuse_cache and CACHE.exists():
        with np.load(CACHE) as d:
            return {"reused": True, "path": str(CACHE), "shape": list(d["rates"].shape)}

    t0 = time.perf_counter()
    grid = ia.build_waterline_grid()
    voyage = ia.reference_baltic_voyage()
    ens = st.dem_activity_rate_field(
        grid=grid, voyage=voyage, n_realizations=args.n,
        n_events_per_leg=args.events, base_seed=BASE_SEED, seed_stride=SEED_STRIDE)
    elapsed = time.perf_counter() - t0
    np.savez_compressed(CACHE, rates=ens.rate_realizations, mean=ens.mean_rate,
                        seeds=np.array(ens.seeds, dtype=np.int64),
                        depths_um=ens.depth_per_voyage_um,
                        duration_s=ens.voyage_duration_s,
                        events=int(args.events))
    meta = {
        "n_realizations": int(args.n),
        "events_per_leg": int(args.events),
        "base_seed": BASE_SEED,
        "seed_stride": SEED_STRIDE,
        "dem_grid_shape": list(grid.shape),
        "voyage_duration_s": float(ens.voyage_duration_s),
        "elapsed_s": float(elapsed),
        "p95_rate_um_per_h": [float(v) for v in
                              np.percentile(ens.rate_realizations.reshape(args.n, -1),
                                            95.0, axis=1) * 1e6 * 3600.0],
        "path": str(CACHE),
    }
    (RUN_DIR / "stage1_ensemble.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "p95_rate_um_per_h"}, indent=2))
    return meta


if __name__ == "__main__":
    main()