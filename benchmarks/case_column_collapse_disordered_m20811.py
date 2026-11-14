"""CASE-COLUMN-COLLAPSE — equilibrated disordered columns (m20811).

Anchor: ``CASE-COLUMN-COLLAPSE`` (算例池·颗粒柱坍塌; ref Lube et al. 2004 *JFM*
508:175-199, doi:10.1017/S0022112004009036; tolerance < 15 %).

Why this wave exists
--------------------
Every archived member of the ensemble waves (m19922, m20259, m20533) starts from
one and the same **HCP lattice** with a positional jitter of at most
``0.005 r_p`` -- above that the lattice overlaps and the m20259 admissibility
gate rejects it.  The m20533 audit records this as its first open debt (§5-1):
the ensemble "只括住「晶格附近」的实现散布，**不是**真正的无序平衡堆积
（需沉积/沉降生成目标 φ）".  m20533 further shows that the shallow aspect
(``a = 0.5``) is where the verdict is decided (worst member −16.1 %, one member
outside the 15 % band) *and* that lattice-to-jitter drift (``C_SCALE`` = 10.3 %)
is larger than the resolution drift -- i.e. the verdict is sensitive to the
initial state in a way that a lattice-adjacent ensemble cannot resolve.

This wave replaces the initial state: a dense lattice is jittered by up to
``0.3 d`` (far beyond the admissible-jitter limit) and **relaxed to static
equilibrium inside the container** (cylindrical wall + floor) before release
(:func:`tensordem.granular_column.pluviation.prepare_disordered_column`).  Only
the equilibrated state is released, so the m20259 objection (an overlapping
lattice stores impulse-releasing elastic energy) does not apply, and the
disorder is *reported* (``hex_order``) instead of asserted.  Nothing else is
touched: ``CLOSING_ARM`` is imported, not retuned, and the floor/contact law is
the one used by every earlier wave.

Pre-registered in this file before the first case of this wave was run
---------------------------------------------------------------------
``C_PHI``      the equilibrated packing fraction stays in ``[0.52, 0.68]``
               (random-loose to dense); outside that the preparation is not a
               physical column and the case is reported, not silently used.
``C_EQ``       every released state is at rest (``KE <= 1e-3 J``) with
               equilibrium overlaps only (``max_overlap <= 0.02 r_p``).
``C_DISORDER`` the equilibrated state is measurably disordered: ``hex_order``
               at least 20 % below the lattice control of the same geometry.
``C_BAND``     every prepared member at every aspect stays within 15 % of the
               Lube (2004) runout law evaluated at the *measured* aspect ratio
               (the paper-consistent caliber of m20533).
``C_SPREAD``   1-sigma of ``r_inf/r_i`` across the members is <= 5 % of their
               mean, at each aspect ratio.
``B_LATTICE``  the wave also runs the deterministic lattice control (same arm,
               ``jitter = 0``) so the disordered members are compared against the
               archived baseline *in the same run*, not against a number from a
               previous wave.

CLI
---
``python benchmarks/case_column_collapse_disordered_m20811.py
[--aspects 0.5,1] [--shard k --nshards n] [--resume] [--progress]``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from tensordem.granular_column.axisym import (AxisymCase, build_axisymmetric_column,
                                             simulate_axisymmetric_column)
from tensordem.granular_column.pluviation import (RelaxCase, disorder_metrics,
                                                  prepare_disordered_column)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_column_collapse_ensemble_m19922 import CLOSING_ARM  # noqa: E402

#: Aspects that decide the anchor verdict (m20533: shallow fails, deep passes).
ASPECTS = (0.5, 1.0)

#: (tag, jitter in grain diameters, seed).  ``LATTICE`` is the in-run control;
#: the jitter amplitudes are 20-60x the m20259 admissible limit.
LATTICE = ("m0_lattice", 0.0, 12345)
MEMBERS = (
    ("d1_j005_s505", 0.05, 505),
    ("d2_j010_s505", 0.10, 505),
    ("d3_j020_s505", 0.20, 505),
    ("d4_j030_s707", 0.30, 707),
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(os.path.dirname(HERE), "runs",
                       "case_column_collapse_disordered_m20811")
CASE_DIR = os.path.join(RUN_DIR, "cases")


def disordered_cases(aspects=ASPECTS) -> list[dict]:
    """Lattice control + the four disordered members, at every aspect."""
    cases = []
    for a in aspects:
        for tag, jitter, seed in (LATTICE,) + tuple(MEMBERS):
            cases.append({
                "id": f"dis_a{a:g}_{tag}",
                "aspect": float(a),
                "member": tag,
                "jitter_diameters": float(jitter),
                "seed": int(seed),
                "phase": "lattice_control" if tag == LATTICE[0] else "disordered",
            })
    return cases


def case_path(case_id: str) -> str:
    return os.path.join(CASE_DIR, f"{case_id}.json")


def relax_case(spec: dict) -> RelaxCase:
    return RelaxCase(
        column_radius=float(CLOSING_ARM["grains_per_radius"]) * 2.0
        * float(CLOSING_ARM["radius"]),
        radius=float(CLOSING_ARM["radius"]),
        aspect_ratio=float(spec["aspect"]),
        density=float(CLOSING_ARM["density"]),
        mu=float(CLOSING_ARM["mu"]), mu_rolling=float(CLOSING_ARM["mu_rolling"]),
        mu_wall=float(CLOSING_ARM["mu_wall"]),
        restitution=float(CLOSING_ARM["restitution"]),
        kn=float(CLOSING_ARM["kn"]), kt_ratio=float(CLOSING_ARM["kt_ratio"]),
        gravity=float(CLOSING_ARM["gravity"]),
        packing_gap=float(CLOSING_ARM["packing_gap"]),
        jitter_diameters=float(spec["jitter_diameters"]),
        seed=int(spec["seed"]), dt_scale=float(CLOSING_ARM["dt_scale"]),
        label=spec["id"])


def collapse_case(spec: dict) -> AxisymCase:
    params = dict(CLOSING_ARM, aspect_ratio=float(spec["aspect"]),
                  jitter=float(spec["jitter_diameters"]),
                  seed=int(spec["seed"]))
    return AxisymCase(**params, label=spec["id"])


def run_case(spec: dict, progress: bool = False) -> dict:
    prep = prepare_disordered_column(relax_case(spec), progress=progress)
    # The equilibrium gate is *recorded*, not fatal: a case that fails it is
    # still run and reported with ``gate_pass = false`` so the reader sees how
    # many members are physical instead of a shard that silently died.
    gate_pass = bool(prep["admissible"])
    pos = np.asarray(prep["positions"], dtype=float)
    case = collapse_case(spec)
    t0 = time.perf_counter()
    res = simulate_axisymmetric_column(case, return_positions=False,
                                       initial_positions=pos)
    wall_clock = time.perf_counter() - t0
    # the lattice control is, by construction, the historical HCP state; record
    # its own disorder metric so C_DISORDER is an in-run comparison
    if spec["phase"] == "lattice_control":
        lat_pos, _ = build_axisymmetric_column(
            AxisymCase(**dict(CLOSING_ARM, aspect_ratio=float(spec["aspect"]),
                              jitter=0.0, seed=int(spec["seed"]))))
        res["lattice_disorder"] = disorder_metrics(lat_pos, case.radius)
    res["preparation"] = {k: v for k, v in prep.items() if k != "positions"}
    res["member"] = {"tag": spec["member"],
                     "jitter_diameters": spec["jitter_diameters"],
                     "seed": spec["seed"], "phase": spec["phase"],
                     "aspect_nominal": float(spec["aspect"])}
    res["wall_clock_collapse_s"] = wall_clock
    res["gate_pass"] = gate_pass
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(case_path(spec["id"]), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--aspects", default=",".join(f"{a:g}" for a in ASPECTS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--print-json", action="store_true",
                    help="print one JSON line per case (parseable fallback if "
                         "the job fetch does not return worktree outputs)")
    args = ap.parse_args(argv)
    aspects = tuple(float(x) for x in args.aspects.split(",") if x.strip())
    cases = disordered_cases(aspects)
    nshards = max(1, args.nshards)
    mine = [c for i, c in enumerate(cases) if i % nshards == args.shard % nshards]
    if args.resume:
        mine = [c for c in mine if not os.path.exists(case_path(c["id"]))]
    print(f"[disordered] shard {args.shard}/{nshards}: {len(mine)}/{len(cases)} "
          f"cases -> {CASE_DIR}", flush=True)
    for i, spec in enumerate(mine, 1):
        t0 = time.perf_counter()
        res = run_case(spec, progress=args.progress)
        prep = res["preparation"]
        a = res["aspect_measured"]
        from tensordem.granular_column.axisym import lube_2004_runout_ratio
        ref = lube_2004_runout_ratio(a)
        print(f"  [{i}/{len(mine)}] {spec['id']}: a_nom={spec['aspect']:g} "
              f"a_meas={a:.3f} phi={prep['packing_fraction']:.3f} "
              f"hex={prep['hex_order']:.3f} r_inf/r_i={res['runout_ratio_p995']:.3f} "
              f"ref={ref:.3f} err={100 * (res['runout_ratio_p995'] / ref - 1):+.1f}% "
              f"stop={res['stop_reason']} {time.perf_counter() - t0:.1f}s", flush=True)
        if args.print_json:
            payload = {k: v for k, v in res.items()
                       if k not in ("positions", "history", "energy", "inertial")}
            payload["case_id"] = spec["id"]
            print("CASEJSON " + json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
