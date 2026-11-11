"""GAP-DEM-FRICTION retry (m15159): shard driver for the direct-shear cell.

The cluster dispatcher passes one shard index per job; the table below maps it to
one (micro friction, shear strain, normal stress) point.  Every point shares the
same device (:mod:`tensordem.shear_cell`) and the same post-processing
(:mod:`scripts.m15159.study`), so a distributed sweep and a single-process sweep
produce byte-identical level records for the same seed.

    PYTHONPATH=src python scripts/m15159/sweep_shard.py --tag head --i 0

Writes ``artifacts/gap_dem_friction_m15159/<tag>.json`` (compact: level summary
plus a down-sampled shear history) and echoes a one-line JSON record.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from scripts.m15159.study import run_point  # noqa: E402

# (tag, mu_p, gamma, sigma_n [Pa], geom)
SHARDS = [
    ("headline_mu050_g100_s4000", 0.5, 1.0, 4000.0, "medium"),
    ("stationarity_mu050_g150_s4000", 0.5, 1.5, 4000.0, "medium"),
    ("level_mu050_g100_s2000", 0.5, 1.0, 2000.0, "medium"),
    ("level_mu050_g100_s8000", 0.5, 1.0, 8000.0, "medium"),
    ("material_mu030_g100_s4000", 0.3, 1.0, 4000.0, "medium"),
    ("material_mu070_g100_s4000", 0.7, 1.0, 4000.0, "medium"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i", type=int, default=0)
    ap.add_argument("--tag", default=None, help="override the shard tag")
    args = ap.parse_args()
    tag, mu, gamma, sigma, geom = SHARDS[args.i % len(SHARDS)]
    tag = args.tag or tag
    rec = run_point(tag=tag, mu=mu, gamma=gamma, sigma=sigma, geom=geom, tol=0.15)
    print("SHARD_RECORD " + json.dumps(rec, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())