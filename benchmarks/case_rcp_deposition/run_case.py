"""One gravity-deposition run of CASE-DEM-RCP-PACKING-FRACTION.

Writes ``<out>/deposition_n<N>_mu<MU>_kn<KN>_e<E>_seed<S>.json`` (metrics,
profile, history, provenance) and the settled positions as ``.npz``.  Runs that
deviate from the default timestep add ``_dtf<dt_frac>`` to the tag so a
temporal-convergence probe never overwrites a production run.

Repro::

    PYTHONPATH=src python benchmarks/case_rcp_deposition/run_case.py \
        --n 10000 --mu 0.0 --seed 1 --out benchmarks/case_rcp_deposition/runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np

from tensordem.deposition_packing import DepositionSpec, run_deposition


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kn", type=float, default=4000.0)
    ap.add_argument("--restitution", type=float, default=0.1)
    ap.add_argument("--dt-frac", type=float, default=1.0 / 20.0,
                    help="dt = dt_frac * t_c (default 1/20; halve for a "
                         "temporal-convergence probe)")
    ap.add_argument("--t-max", type=float, default=200.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    spec = DepositionSpec(n_particles=args.n, mu=args.mu, seed=args.seed,
                          kn=args.kn, restitution=args.restitution,
                          dt_frac=args.dt_frac, t_max=args.t_max)
    t0 = time.perf_counter()

    def progress(rec):
        print(f"[t={rec['t']:7.2f}] ke_norm={rec['ke_norm']:.3e} "
              f"pairs={rec['n_pair']:6d} wall={rec['n_wall']:5d}", flush=True)

    res = run_deposition(spec, progress=progress)
    payload = res.summary()
    payload["wall_clock_total"] = time.perf_counter() - t0
    tag = (f"deposition_n{args.n}_mu{args.mu:g}_kn{args.kn:g}_e{args.restitution:g}"
           f"_seed{args.seed}")
    if args.dt_frac != 1.0 / 20.0:
        tag += f"_dtf{args.dt_frac:g}"
    with open(os.path.join(args.out, tag + ".json"), "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                        positions=res.positions, velocities=res.velocities,
                        omegas=res.omegas)
    m = res.metrics
    print(f"RESULT {tag} phi_bulk={m['phi_bulk']:.5f} "
          f"phi_corr={m['phi_bulk_overlap_corrected']:.5f} "
          f"Z={m['coordination_bulk']:.3f} settled={res.settled} "
          f"steps={res.steps} wall={res.wall_clock:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())