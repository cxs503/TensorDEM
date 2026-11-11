"""Shard runner for the GAP-DEM-CLUMP round-2 scan (cluster-friendly).

Enumerates the (aspect ratio, seed, configuration) grid deterministically and
runs the slice ``[i::n]`` so that ``run_cluster --cmd-tpl`` can spread it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "gap_dem_clump"))

from run_gap_dem_clump import AR_LIST  # noqa: E402
from run_round2_scan import CONFIGS, SEEDS, fill_heights, run_case  # noqa: E402


def cases():
    return [(ar, seed, cfg) for cfg, ar, seed in
            itertools.product(CONFIGS, AR_LIST, SEEDS)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--outdir", default=str(ROOT / "artifacts" / "GAP-DEM-CLUMP-R2"))
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_cases = cases()
    mine = all_cases[args.i :: args.n]
    print(f"shard {args.i}/{args.n}: {len(mine)} of {len(all_cases)} cases", flush=True)
    if not mine:
        return 0
    fills = fill_heights()
    (outdir / "fills.json").write_text(json.dumps(fills, indent=2))
    for ar, seed, cfg in mine:
        tag = f"case_ar{ar:.1f}_seed{seed}_{cfg.replace('/', '-')}.json"
        dst = outdir / tag
        if dst.exists():
            print(f"skip {tag}", flush=True)
            continue
        row = run_case(ar, seed, cfg, fills)
        dst.write_text(json.dumps(row, indent=2))
        print(f"{tag}: {row['angle_profile_deg']:.3f} deg, n={row['n_particles']}, "
              f"quiesced={row['quiesced']}, {row['wall_seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())