"""CASE-COLUMN-COLLAPSE — completion of the admissible ensemble wave (m20533).

Anchor: ``CASE-COLUMN-COLLAPSE`` (算例池·颗粒柱坍塌; ref Lube et al. 2004 *JFM*
508:175-199, doi:10.1017/S0022112004009036; tolerance < 15 %).

Why this wave exists
--------------------
m20259 (``docs/AUDIT_COLUMN_COLLAPSE_REALISATION_AXIS_m20259.md``) re-ran the
realisation axis strictly inside the geometrically admissible band and closed
``a = 0.5``, but declared the other two pre-registered aspects **not run**
(``summary.json: incomplete_aspects = [1.0, 2.0]``; the cluster shards died on a
missing path).  Its ``A_BAND`` (every admissible member within 15 % of the Lube
law) and ``A_SPREAD`` verdicts therefore rest on a single aspect ratio.

This wave fills exactly that hole and nothing else: the *same* four admissible
members (identical ``(jitter, seed)`` tuples) plus the deterministic lattice
control, at ``a = 1.0`` and ``a = 2.0``, with the *same* ``CLOSING_ARM``
parameter dict imported from the m19922 benchmark — no retuning, no new axes.

Second, independent question: aspect bookkeeping
------------------------------------------------
The module normalises runout by the **measured** initial extent
(``r_i = p995`` of the prepared packing) and reports the **measured** aspect
ratio ``a_meas = h_meas / r_i_meas``, while the Lube law is evaluated at the
*nominal* ``a``.  Lube et al. define ``a = h_i / r_i`` of the column that was
actually released, so the paper-consistent reference is the one evaluated at
``a_meas``.  The prepared HCP packing is measurably taller than nominal
(``a_meas / a = 1.099`` at ``a = 0.5``), so the two variants cannot be assumed
equal a priori.  Both errors are recorded per case here; the pre-registered
criterion is evaluated on the paper-consistent one (``B_ASPECT_BOOK`` below).

Pre-registered in this file before the first case of this wave was run
---------------------------------------------------------------------
``C_ADMISS``  every member starts with ``overlap_pairs == 0`` (gate; a rejecting
              member is never integrated).
``C_BAND``    every member at every aspect (this wave: ``a in {1.0, 2.0}``) stays
              within 15 % of the Lube (2004) runout law evaluated at ``a_meas``.
``C_SPREAD``  1-sigma of ``r_inf/r_i`` across the four admissible members is
              <= 5 % of their mean, at each aspect ratio.
``C_SCALE``   |shift| from the lattice control to the largest admissible jitter
              is <= 5 %, at each aspect ratio.
``B_ASPECT_BOOK``  the difference between the nominal-``a`` and measured-``a``
              reference variants is reported per case; a verdict that flips
              between the two variants is flagged, not silently resolved.

CLI
---
``python benchmarks/case_column_collapse_admissible_completion_m20533.py
[--aspects 1,2] [--shard k --nshards n] [--resume]``
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
from case_column_collapse_ensemble_m19922 import CLOSING_ARM, evaluate  # noqa: E402
from case_column_collapse_ensemble_admissible_m20259 import MEMBERS  # noqa: E402

#: Aspects still open after m20259 (``summary.json: incomplete_aspects``).
COMPLETION_ASPECTS = (1.0, 2.0)

#: The deterministic lattice control (same tag/seed as the m19922/m20259 waves).
LATTICE = ("m0_lattice", 0.0, 12345)

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(os.path.dirname(HERE), "runs",
                       "case_column_collapse_admissible_completion_m20533")
CASE_DIR = os.path.join(RUN_DIR, "cases")


def completion_cases(aspects=COMPLETION_ASPECTS) -> list[dict]:
    """Lattice control + the four admissible members, at every open aspect."""
    cases = []
    for a in aspects:
        for tag, jitter, seed in (LATTICE,) + tuple(MEMBERS):
            phase = "lattice" if tag == LATTICE[0] else "admissible"
            cases.append({"id": f"cmp_a{a:g}_{tag}", "phase": phase,
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
                       "seed": spec["seed"], "aspect": float(spec["aspect"])}
    res["admissibility"] = st
    # criteria at nominal a (the m19922/m20259 convention) -- kept for
    # comparability with the archived wave
    res["criteria"] = evaluate(res, case)
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(case_path(spec["id"]), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--aspects", default=",".join(f"{a:g}" for a in COMPLETION_ASPECTS))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    aspects = tuple(float(x) for x in args.aspects.split(",") if x.strip())
    cases = completion_cases(aspects)
    nshards = max(1, args.nshards)
    mine = [c for i, c in enumerate(cases) if i % nshards == args.shard % nshards]
    if args.resume:
        mine = [c for c in mine if not os.path.exists(case_path(c["id"]))]
    print(f"[completion] shard {args.shard}/{nshards}: {len(mine)}/{len(cases)} "
          f"cases -> {CASE_DIR}", flush=True)
    for i, spec in enumerate(mine, 1):
        t0 = time.perf_counter()
        res = run_case(spec)
        crit = res["criteria"]["runout_ratio"]
        print(f"  [{i}/{len(mine)}] {spec['id']}: a_nom={spec['aspect']:g} "
              f"a_meas={res['aspect_measured']:.3f} jitter={spec['jitter']:g} "
              f"N={res['n_particles']} r_inf/r_i={res['runout_ratio_p995']:.3f} "
              f"ref_nom={crit['reference']:.3f} err={100 * crit['relative_error']:+.1f}% "
              f"overlap_pairs={res['admissibility']['overlap_pairs']} "
              f"stop={res['stop_reason']} {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())