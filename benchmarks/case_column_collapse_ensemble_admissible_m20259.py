"""CASE-COLUMN-COLLAPSE — *admissible* ensemble wave (m20259).

Anchor: ``CASE-COLUMN-COLLAPSE`` (算例池·颗粒柱坍塌; ref Lube et al. 2004 *JFM*
508:175-199, doi:10.1017/S0022112004009036; tolerance < 15 %).

Why a second ensemble wave exists
---------------------------------
The m19922 wave declared five members per aspect: the deterministic lattice plus
four disordered packings at ``jitter = 0.05`` and ``0.10`` (in units of ``r_p``
per coordinate).  Pure-geometry re-examination (``tensordem.granular_column.
admissibility``) shows those amplitudes are **not admissible initial states** of
the same column: the prepared HCP lattice has a 2 % clearance (1.18 mm at
``r_p = 30 mm``), while a 5 % jitter already pushes 1400+ neighbour pairs into
penetration (up to 6.6 mm deep) and a 10 % jitter gives 1850+ pairs up to
14.2 mm deep.  Such a state stores elastic energy that the contact law releases
at ``t = 0``; it measures an initial-overlap artefact, not realisation-to-
realisation uncertainty.  The measured consequence at ``a = 0.5`` is a doubling
of the peak front speed and a runout of ``r_inf/r_i = 2.30 … 2.35`` against the
Lube law reference 1.620 (+42 %, +45 %) — i.e. the pre-registered H_BAND and
H_JITTER of m19922 are falsified by an artefact.

This wave therefore re-runs the realisation axis strictly **inside the
admissible band**:

* ``jitter = 0.0025 r_p`` and ``jitter = 0.005 r_p``, two seeds each, verified
  to have ``overlap_pairs == 0`` *before* the run (the run aborts otherwise);
* aspects ``a in {0.5, 1.0, 2.0}`` (same as m19922);
* everything else identical to the closing arm — the same ``CLOSING_ARM`` dict
  imported from the m19922 benchmark, no retuning.

Pre-registered here, before the first admissible member was run
----------------------------------------------------------------
``A_ADMISS``  every member's initial packing has ``overlap_pairs == 0``
              (gate; a rejecting member is never integrated).
``A_BAND``    every admissible member stays within 15 % of the Lube (2004)
              runout law at every aspect ratio.
``A_SPREAD``  1-sigma of ``r_inf/r_i`` over the admissible members is <= 5 % of
              the ensemble mean at every aspect ratio.
``A_SCALE``   the runout shift from the lattice to the largest admissible
              jitter is <= 5 % at every aspect ratio (i.e. within the
              admissible band positional disorder is not a first-order lever).
``A_GAP``     the m19922 ``jitter = 0.05`` and ``0.10`` members lie far outside
              the admissible band, so their out-of-band runout cannot be used as
              evidence about physical realisation uncertainty (the falsification
              of H_BAND/H_JITTER is an initial-overlap artefact).

Honest scope: because admissible jitter is only ~0.5 % of ``r_p``, this wave
brackets realisation sensitivity *near the lattice*; a genuinely disordered
*equilibrium* packing (lower packing fraction, prepared by deposition and
settling) is a different generator and is left as the explicit next step.

CLI
---
``python benchmarks/case_column_collapse_ensemble_admissible_m20259.py
[--shard k --nshards n] [--aspects 0.5,1,2] [--resume]``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from tensordem.granular_column.admissibility import overlap_stats
from tensordem.granular_column.axisym import (AxisymCase, build_axisymmetric_column,
                                             simulate_axisymmetric_column)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_column_collapse_ensemble_m19922 import (  # noqa: E402
    ASPECTS, CLOSING_ARM, evaluate)

#: (tag, jitter, seed) — all verified admissible by the geometry scan below.
MEMBERS = (
    ("n1_j0025_s505", 0.0025, 505),
    ("n2_j0025_s606", 0.0025, 606),
    ("n3_j005_s707", 0.005, 707),
    ("n4_j005_s808", 0.005, 808),
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(os.path.dirname(HERE), "runs",
                       "case_column_collapse_ensemble_admissible_m20259")
CASE_DIR = os.path.join(RUN_DIR, "cases")


def admissible_cases(aspects=ASPECTS) -> list[dict]:
    cases = []
    for a in aspects:
        for tag, jitter, seed in MEMBERS:
            cases.append({"id": f"adm_a{a:g}_{tag}", "phase": "admissible",
                          "aspect": float(a), "member": tag,
                          "jitter": float(jitter), "seed": int(seed),
                          "params": dict(CLOSING_ARM, aspect_ratio=float(a),
                                         jitter=float(jitter), seed=int(seed))})
    return cases


def case_path(case_id: str) -> str:
    return os.path.join(CASE_DIR, f"{case_id}.json")


def gate(spec: dict) -> dict:
    """Admissibility gate on the *initial* state (no integration involved)."""
    case = AxisymCase(**spec["params"])
    pos, _ = build_axisymmetric_column(case)
    return overlap_stats(pos, case.radius)


def run_case(spec: dict) -> dict:
    st = gate(spec)
    if not st["admissible"]:
        raise RuntimeError(f"{spec['id']} rejected by the admissibility gate: "
                           f"overlap_pairs={st['overlap_pairs']} "
                           f"max_overlap={st['max_overlap']:.5f} m")
    t0 = time.perf_counter()
    case = AxisymCase(**spec["params"], label=spec["id"])
    res = simulate_axisymmetric_column(case, return_positions=False)
    res["wall_clock_s"] = time.perf_counter() - t0
    res["phase"] = spec["phase"]
    res["ensemble"] = {"member": spec["member"], "jitter": spec["jitter"],
                       "seed": spec["seed"]}
    res["admissibility"] = st
    res["criteria"] = evaluate(res, case)
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
    args = ap.parse_args(argv)
    aspects = tuple(float(x) for x in args.aspects.split(",") if x.strip())
    cases = admissible_cases(aspects)
    nshards = max(1, args.nshards)
    mine = [c for i, c in enumerate(cases) if i % nshards == args.shard % nshards]
    if args.resume:
        mine = [c for c in mine if not os.path.exists(case_path(c["id"]))]
    print(f"[admissible] shard {args.shard}/{nshards}: {len(mine)}/{len(cases)} "
          f"cases -> {CASE_DIR}", flush=True)
    for i, spec in enumerate(mine, 1):
        t0 = time.perf_counter()
        res = run_case(spec)
        crit = res["criteria"]["runout_ratio"]
        print(f"  [{i}/{len(mine)}] {spec['id']}: a={res['aspect_measured']:.3f} "
              f"jitter={spec['jitter']:g} N={res['n_particles']} "
              f"r_inf/r_i={res['runout_ratio_p995']:.3f} ref={crit['reference']:.3f} "
              f"err={100 * crit['relative_error']:+.1f}% "
              f"overlap_pairs={res['admissibility']['overlap_pairs']} "
              f"stop={res['stop_reason']} {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())