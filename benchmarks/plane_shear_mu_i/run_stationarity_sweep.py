"""Driver for the round-5 strain-stationarity ladder (CASE-DEM-PLANE-SHEAR-MUIDI).

Fans the ``(I, strain)`` ladder of :mod:`run_stationarity` out over shards so a
cluster job can run it in parallel::

    python run_stationarity_sweep.py --shard 0/6     # one I target per shard

The ladder is deliberately uneven: the cheap high-``I`` points get the longest
strain rungs (a stiff, fast-shearing bed reaches a large strain for a few
hundred steps), while the expensive low-``I`` points are capped so that no
single shard needs more than ~25 min of CPU.  The rung set per ``I`` is stored
in the payload so the analysis never re-derives it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_stationarity import ARMS, OUT, run_ladder_point  # noqa: E402

# I target -> strain rungs.  Cost per rung ~ steps / (gamma_dot*dt), i.e. it
# scales as 1/I for a fixed strain, hence the shorter ladders at low I.
LADDER: dict[float, tuple[float, ...]] = {
    1e-3: (0.2, 2.0),
    3e-3: (0.2, 4.0),
    1e-2: (0.2, 1.0, 4.0, 8.0),
    3e-2: (0.2, 1.0, 4.0, 8.0),
    1e-1: (0.2, 1.0, 4.0, 16.0),
    3e-1: (0.2, 1.0, 4.0, 16.0),
}
I_ORDER = sorted(LADDER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n, i in [0,n)")
    ap.add_argument("--arm", choices=tuple(ARMS), default="thick")
    ap.add_argument("--index", default=None,
                    help="optional json summary of the assigned rungs")
    args = ap.parse_args()
    i, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit("shard index out of range")
    assigned = [I for k, I in enumerate(I_ORDER) if k % n == i]
    print(f"shard {i}/{n}: I targets {assigned}", flush=True)
    summary = {}
    for I in assigned:
        for strain in LADDER[I]:
            rec = run_ladder_point(I, strain, arm=args.arm)
            tag = f"{args.arm}__I{I:g}__s{strain:g}"
            (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=1))
            summary[tag] = {k: rec[k] for k in (
                "I_target", "strain_target", "I_measured", "mu_tail",
                "mu_ratio_of_means", "mu_block_drift_pct", "phi_bed_steady",
                "mu_err_pct", "phi_err_pct", "wall_clock_s",
                "plate_travel_in_d")}
            print(f"[{tag}] I={rec['I_measured']:.4g} mu={rec['mu_tail']:.4f} "
                  f"({rec['mu_err_pct']:+.1f}%) phi={rec['phi_bed_steady']:.4f} "
                  f"({rec['phi_err_pct']:+.1f}%) drift={rec['mu_block_drift_pct']:+.2f}% "
                  f"wc={rec['wall_clock_s']:.0f}s", flush=True)
    out = Path(args.index) if args.index else OUT / f"shard_{i}_of_{n}.json"
    out.write_text(json.dumps({"shard": f"{i}/{n}", "arm": args.arm,
                               "ladder": {str(k): list(v) for k, v in LADDER.items()},
                               "points": summary}, indent=1))
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()