"""CASE-COLUMN-COLLAPSE — ensemble (initial-realization) uncertainty wave.

Anchor: ``CASE-COLUMN-COLLAPSE`` (算例池·颗粒柱坍塌; ref Lube et al. 2004 *JFM*
508:175-199, doi:10.1017/S0022112004009036; tolerance < 15 %).

Why this wave exists
--------------------
Rounds 1-5 closed the runout law 5/5 inside the 15 % band with the arm
``(mu, mu_r, mu_wall) = (0.5, 0.5, 0.8)``, and three later audits
(m17534, m17917, m18261, m18385) all list the *same* remaining debt:

    "单种子、单一初始构型（HCP+gap，jitter=0），无系综散布"

i.e. every reported error is one deterministic realisation of an infinitely
sharp HCP lattice with ``jitter = 0``.  A 15 % acceptance verdict whose margin
is 6 % is only meaningful if the realisation-to-realisation scatter is smaller
than that margin, and nothing in the archive measures it.  This wave varies
**only the initial packing realisation** at the closing arm and leaves every
other lever (material, contacts, dt, geometry, stopping rule) untouched.

Pre-registered hypotheses (fixed before the first run, adjudicated
mechanically by ``scripts/analyze_column_collapse_ensemble_m19922.py``)
--------------------------------------------------------------------------
``H_REPRO``  the ``jitter = 0`` member reproduces the round-5b archived
             ``runout_ratio_p995`` at the same aspect ratio to < 0.5 %
             (provenance / library-drift gate: the deterministic control must
             come back unchanged).
``H_SPREAD`` ensemble 1-sigma of ``r_inf / r_i`` over the jittered members is
             <= 5 % of the ensemble mean at every aspect ratio (the 15 % band is
             not consumed by realisation scatter).
``H_BAND``   *every* ensemble member stays within 15 % of the Lube (2004) law
             at every aspect ratio (worst-case verdict, not ensemble mean).
``H_SIGN``   at ``a = 0.5`` the emplacement-time error stays ``> +15 %`` for
             *every* member (the shallow-column time overrun of m17534/m17917 is
             sign-stable and therefore not a realisation artifact).
``H_JITTER`` moving from the perfect lattice (``jitter = 0``) to 10 %
             positional disorder (``jitter = 0.10``, i.e. sigma = 0.1 r_p
             per coordinate) changes ``r_inf / r_i`` by < 5 % at every aspect
             ratio (positional disorder is not a first-order lever for runout).

A falsified hypothesis is recorded as falsified; nothing is retuned after the
fact.

Design
------
* Closing arm, unchanged: ``mu = 0.5``, ``mu_r = 0.5``, ``mu_wall = 0.8``,
  ``e = 0.5``, ``kn = 6e4 N/m``, ``r_p = 0.03 m``, 8 grains per column radius,
  ``packing_gap = 0.02``, ``rho = 2500 kg/m^3``.
* Aspects ``a in {0.5, 1.0, 2.0}``: shallow (where the time law fails),
  transitional, deep (largest throughput, the runout law is inertial).
* Five members per aspect: the deterministic lattice control plus four
  disordered packings (two amplitudes x two seeds).

CLI
---
``python benchmarks/case_column_collapse_ensemble_m19922.py [--shard k]
[--nshards n] [--aspects 0.5,1,2] [--progress]``

Per-case JSON goes to
``runs/case_column_collapse_ensemble_m19922/cases/<case_id>.json``
(``return_positions=False``: the archive keeps scalars + time histories only,
which is all this study reduces; deposit geometry was audited separately by
m18385).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

from tensordem.granular_column.axisym import (AxisymCase, lube_2004_emplacement_time,
                                             lube_2004_runout_ratio,
                                             simulate_axisymmetric_column, verdict)

TOLERANCE = 0.15
ASPECTS = (0.5, 1.0, 2.0)

#: Closing-arm parameters (identical to round 5b, docs/case_column_collapse_round6_m18691.md).
CLOSING_ARM = dict(radius=0.03, density=2500.0, mu=0.5, mu_rolling=0.5,
                   mu_wall=0.8, restitution=0.5, kn=6.0e4, kt_ratio=0.5,
                   grains_per_radius=8.0, packing_gap=0.02, gravity=9.81,
                   dt_scale=1.0, duration_factor=8.0, min_time=0.5,
                   rest_ke_tol=1.0e-4, rest_hold=200, t_reach_fraction=0.99)

#: (tag, jitter, seed).  ``m0`` is the deterministic reproduction control.
MEMBERS = (
    ("m0_lattice", 0.00, 12345),
    ("m1_j05_s101", 0.05, 101),
    ("m2_j05_s202", 0.05, 202),
    ("m3_j10_s303", 0.10, 303),
    ("m4_j10_s404", 0.10, 404),
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(os.path.dirname(HERE), "runs",
                       "case_column_collapse_ensemble_m19922")
CASE_DIR = os.path.join(RUN_DIR, "cases")


def ensemble_cases(aspects=ASPECTS) -> list[dict]:
    """Every pre-declared ensemble member (order is the sharding order)."""
    cases = []
    for a in aspects:
        for tag, jitter, seed in MEMBERS:
            cases.append({
                "id": f"ens_a{a:g}_{tag}",
                "phase": "ensemble",
                "aspect": float(a),
                "member": tag,
                "jitter": float(jitter),
                "seed": int(seed),
                "params": dict(CLOSING_ARM, aspect_ratio=float(a),
                               jitter=float(jitter), seed=int(seed)),
            })
    return cases


def case_path(case_id: str) -> str:
    return os.path.join(CASE_DIR, f"{case_id}.json")


def evaluate(res: dict, case: AxisymCase) -> dict:
    """Pre-declared criteria; every target is evaluated from the Lube formulas."""
    a = case.aspect_ratio
    return {
        "runout_ratio": verdict(lube_2004_runout_ratio(a), res["runout_ratio_p995"],
                                TOLERANCE),
        "runout_ratio_max_metric": verdict(lube_2004_runout_ratio(a),
                                           res["runout_ratio_max"], TOLERANCE),
        "emplacement_time": verdict(
            lube_2004_emplacement_time(a, case.column_radius, case.gravity),
            res["t_reach"], TOLERANCE),
    }


def run_case(spec: dict, progress: bool = False) -> dict:
    t0 = time.perf_counter()
    case = AxisymCase(**spec["params"], label=spec["id"])
    res = simulate_axisymmetric_column(case, return_positions=False, progress=progress)
    res["wall_clock_s"] = time.perf_counter() - t0
    res["phase"] = spec["phase"]
    res["ensemble"] = {"member": spec["member"], "jitter": spec["jitter"],
                       "seed": spec["seed"]}
    res["criteria"] = evaluate(res, case)
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(case_path(spec["id"]), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def load_case(case_id: str) -> dict | None:
    path = case_path(case_id)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--aspects", default=",".join(f"{a:g}" for a in ASPECTS))
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip cases whose per-case JSON already exists")
    args = ap.parse_args(argv)

    aspects = tuple(float(x) for x in args.aspects.split(",") if x.strip())
    cases = ensemble_cases(aspects)
    nshards = max(1, args.nshards)
    mine = [c for i, c in enumerate(cases) if i % nshards == args.shard % nshards]
    if args.resume:
        mine = [c for c in mine if not os.path.exists(case_path(c["id"]))]
    print(f"[ensemble] shard {args.shard}/{nshards}: {len(mine)}/{len(cases)} cases "
          f"-> {CASE_DIR}", flush=True)
    for i, spec in enumerate(mine, 1):
        t0 = time.perf_counter()
        res = run_case(spec, progress=args.progress)
        crit = res["criteria"]["runout_ratio"]
        print(f"  [{i}/{len(mine)}] {spec['id']}: a={res['aspect_measured']:.3f} "
              f"N={res['n_particles']} r_inf/r_i={res['runout_ratio_p995']:.3f} "
              f"ref={crit['reference']:.3f} err={100 * crit['relative_error']:+.1f}% "
              f"t95={res['t_reach_by_fraction']['0.95']:.3f} "
              f"stop={res['stop_reason']} {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())