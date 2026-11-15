"""Resolution ladder for CASE-COLUMN-COLLAPSE (m21656).

Why this exists
---------------
The anchor's verdict is a *reference comparison* at whatever resolution
happens to be available: ``a = 3`` at 12 grains/``r_i`` reported +15.4 %
(out of the 15 % band) while the coarser 8 and the finer 16 reported +8.1 %
and +11.1 %.  A single miss that sits *between* two passers cannot be
adjudicated by a tolerance test -- it needs a grid-convergence statement, and
that needs a ladder with

* a **uniform refinement ratio** on at least three levels (``(4, 8, 16)`` gives
  ``r = 2``) so the observed order and the GCI are defined at all, and
* **intermediate rungs** (5, 10, 14) so a non-monotone sequence can be
  resolved into a decaying oscillation instead of unexplained scatter.

This script fills those rungs on the frozen closed arm (same
``BASE`` protocol as ``case_column_collapse_time_m18261``, same column radius
``r_i = 0.48 m``, same stiffness law ``k_n = 6e4 (d/0.03)^2`` which keeps the
static sag ratio -- i.e. the dimensionless contact stiffness -- constant across
the ladder).  Nothing about the physics is touched; only ``grains_per_radius``
changes.

It also records the **metric** alternatives (``p995 + r``, ``p999 + r``,
``max + r``, ``p95 + r``) for every rung, because the primary runout metric is
a 99.5 % quantile: an extreme order statistic whose gap to the outermost grain
depends on the particle count, i.e. on the resolution being varied.  Without
that, a resolution trend and a metric artefact are indistinguishable.

Provenance
----------
Rungs that already exist in committed artifacts (``R_a*_g*`` from the m20873 /
m18261 rounds, ``SB_a3_g16`` from m20931) are **reused rather than re-run**,
but only after :func:`assert_reusable` proves that the committed protocol is
field-for-field identical to what this round would have generated (the
``protocol_sha256`` differs only through ``case_id``/``round`` provenance).
The reused JSON's sha256 and path are recorded in the summary.

Retrieval
---------
Cluster ``fetch`` returns the committed repo state, not the node worktree
(``docs/AUDIT_COLUMN_COLLAPSE_REFINEMENT_FETCH_m20559.md``), so each case JSON
is also emitted into the job log as a base64/gzip payload with a sha256 header
-- the same envelope as ``scripts/emit_column_collapse_payload_m20873.py``, and
decoded by the same function.

Usage (node, repo checkout)
---------------------------
::

    python3 benchmarks/case_column_collapse_grid_study_m21656.py --list
    python3 benchmarks/case_column_collapse_grid_study_m21656.py --case GS_a1_g4
    python3 benchmarks/case_column_collapse_grid_study_m21656.py --shard 0/4 --emit
    python3 benchmarks/case_column_collapse_grid_study_m21656.py --collect
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), _HERE, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import case_column_collapse_time_m18261 as base  # noqa: E402  (frozen arm)
from emit_column_collapse_payload_m20873 import encode  # noqa: E402
from tensordem.granular_column.arrival import arrival_report  # noqa: E402
from tensordem.granular_column.axisym import (  # noqa: E402
    simulate_axisymmetric_column)

ROUND = "m21656"
OUT_DIR = os.path.join("runs", "case_column_collapse_grid_study_m21656")
CASE_DIR = os.path.join(OUT_DIR, "cases")
TOLERANCE = base.TOLERANCE

#: Committed artifacts that hold rungs of this ladder already.
REUSE = {
    "R_a3_g8": "docs/artifacts/column_collapse_deep_refinement_m20873/cases/R_a3_g8.json",
    "R_a3_g12": "docs/artifacts/column_collapse_deep_refinement_m20873/cases/R_a3_g12.json",
    "SB_a3_g16": "docs/artifacts/column_collapse_sqrt_branch_m20931/cases/SB_a3_g16.json",
    "R_a1_g8": "docs/artifacts/column_collapse_deep_refinement_m20873/cases/R_a1_g8.json",
    "R_a1_g12": "docs/artifacts/column_collapse_deep_refinement_m20873/cases/R_a1_g12.json",
    "R_a1_g16": "docs/artifacts/column_collapse_deep_refinement_m20873/cases/R_a1_g16.json",
}

#: Ladder rungs: (ladder id, aspect, grains per column radius, reused case id).
RUNGS = [
    ("a3", 3.0, 4.0, None),
    ("a3", 3.0, 5.0, None),
    ("a3", 3.0, 8.0, "R_a3_g8"),
    ("a3", 3.0, 10.0, None),
    ("a3", 3.0, 12.0, "R_a3_g12"),
    ("a3", 3.0, 14.0, None),
    ("a3", 3.0, 16.0, "SB_a3_g16"),
    ("a1", 1.0, 4.0, None),
    ("a1", 1.0, 5.0, None),
    ("a1", 1.0, 8.0, "R_a1_g8"),
    ("a1", 1.0, 10.0, None),
    ("a1", 1.0, 12.0, "R_a1_g12"),
    ("a1", 1.0, 16.0, "R_a1_g16"),
]


def case_id(ladder: str, gpr: float) -> str:
    return f"GS_{ladder}_g{gpr:g}"


def _case_spec(spec: dict) -> dict:
    """The subset of a ladder spec that ``base.make_case`` accepts."""
    return {k: v for k, v in spec.items() if k != "ladder"}


def build_spec(ladder: str, aspect: float, gpr: float) -> dict:
    return {"case_id": case_id(ladder, gpr), "axis": "R",
            "aspect_ratio": aspect, "grains_per_radius": gpr,
            "ladder": ladder}


def assert_reusable(path: str, spec: dict) -> dict:
    """Prove a committed case JSON is this ladder's rung, then return it.

    Checks (all must hold, otherwise ``ValueError``):

    1. the physics fields of the protocol equal the frozen ``BASE`` arm,
    2. ``radius`` equals ``r_i / (2 gpr)``,
    3. ``kn`` equals the sag-preserving stiffness law ``6e4 (d/0.03)**2``,
    4. the **measured column radius** equals the ladder constant
       ``r_i = 0.48 m`` to 10 %.  This -- not the aspect ratio -- is the
       invariant of a refinement ladder, and it is what catches a scale
       mistake of the kind documented for round 5 ``w5c_gpr12_*`` (which kept
       ``radius = 0.03 m`` at 12 grains per radius and so moved ``r_i`` by
       +52 %).  The measured aspect ratio legitimately drifts with resolution
       (the lattice truncates differently against the cylinder wall: at
       ``a = 1``, 8 grains per ``r_i`` measures ``a = 1.078``), which is why
       every reference is evaluated at ``a_meas`` and never at the nominal.
    """
    with open(path) as fh:
        raw = fh.read()
    case = json.loads(raw)
    proto = case["protocol"]
    want = base.make_case(_case_spec(spec))
    checked = ("mu", "mu_rolling", "mu_wall", "restitution", "kt_ratio",
               "gravity", "packing_gap", "jitter", "seed", "dt_scale",
               "duration_factor", "min_time", "contact_mode")
    mismatch = [(k, proto.get(k), base.BASE[k]) for k in checked
                if proto.get(k) != base.BASE[k]]
    if mismatch:
        raise ValueError(f"{path}: physics fields differ from the frozen arm: "
                         f"{mismatch}")
    if abs(float(proto["radius"]) - float(want.radius)) > 1e-12:
        raise ValueError(f"{path}: radius {proto['radius']} != ladder law "
                         f"{want.radius}")
    if abs(float(proto["kn"]) - float(want.kn)) > 1e-9 * float(want.kn):
        raise ValueError(f"{path}: kn {proto['kn']} != sag-preserving law {want.kn}")
    r_i = float(case["column_radius_measured"])
    if abs(r_i - base.COLUMN_RADIUS) / base.COLUMN_RADIUS > 0.10:
        raise ValueError(f"{path}: measured column radius {r_i} is not the ladder "
                         f"constant {base.COLUMN_RADIUS} (scale mistake, not a "
                         "refinement rung)")
    return {"case": case, "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "path": path}


def run_case(spec: dict, *, progress: bool = False) -> dict:
    """Integrate one rung (cache-first).  Records every metric alternative."""
    cid = spec["case_id"]
    path = os.path.join(CASE_DIR, f"{cid}.json")
    if os.path.exists(path):
        with open(path) as fh:
            out = json.load(fh)
        out["cached"] = True
        return out
    case = base.make_case(_case_spec(spec))
    sig = base.protocol_signature(_case_spec(spec))
    t0 = time.perf_counter()
    res = simulate_axisymmetric_column(case, return_positions=False,
                                       progress=progress)
    wall = time.perf_counter() - t0
    hist = res["history"]
    time_hist = np.asarray(hist["time"], dtype=float)
    front_hist = np.asarray(hist["front"], dtype=float)
    arrivals = arrival_report(time_hist, front_hist,
                              radius=res["column_radius_measured"],
                              grain_diameter=2.0 * case.radius)
    refs = base.reference_values(spec, res["aspect_measured"],
                                 res["column_radius_measured"], case.gravity)
    radius = float(case.radius)
    r_i = float(res["column_radius_measured"])
    # Both extent metrics are produced by the kernel on the *full* particle set
    # (``return_positions`` only ships a strided copy, which would bias a
    # recomputed quantile), so they are read straight off the result.
    extent = {"p995": float(res["runout_radius_p995"]),
              "max": float(res["runout_radius_max"])}
    out = {
        "case_id": cid,
        "ladder": spec["ladder"],
        "axis": "R",
        "protocol": sig,
        "protocol_sha256": base.signature_sha256(sig),
        "n_particles": res["n_particles"],
        "dt": res["dt"],
        "n_steps": res["n_steps"],
        "time_end": res["time_end"],
        "stop_reason": res["stop_reason"],
        "aspect_nominal": float(spec["aspect_ratio"]),
        "aspect_measured": res["aspect_measured"],
        "column_radius_measured": r_i,
        "column_height_measured": res["column_height_measured"],
        "solid_fraction_initial": res["solid_fraction_initial"],
        "grain_diameter": 2.0 * radius,
        "grains_per_radius": float(spec["grains_per_radius"]),
        "extent": extent,
        "runout_ratio_p995": extent["p995"] / r_i,
        "runout_ratio_by_metric": {k: v / r_i for k, v in extent.items()},
        "reference": refs,
        "arrival": arrivals,
        "energy": res["energy"],
        "inertial": res["inertial"],
        "coordination_max": res["coordination_max"],
        "history": {"time": hist["time"], "front": hist["front"],
                    "kinetic": hist["kinetic"], "contacts": hist["contacts"]},
        "wall_clock_s": wall,
        "source": "run",
        "round": ROUND,
    }
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def collect(emit: bool = False) -> dict:
    """Assemble the ladder summary from case JSONs and reused artifacts."""
    rows, missing = [], []
    for ladder, aspect, gpr, reuse_id in RUNGS:
        spec = build_spec(ladder, aspect, gpr)
        cid = spec["case_id"]
        path = os.path.join(CASE_DIR, f"{cid}.json")
        if os.path.exists(path):
            with open(path) as fh:
                case = json.load(fh)
            case["source"] = case.get("source", "run")
        elif reuse_id is not None:
            chk = assert_reusable(REUSE[reuse_id], spec)
            case = chk["case"]
            case["source"] = "reused"
            case["reused_from"] = chk["path"]
            case["reused_sha256"] = chk["sha256"]
            os.makedirs(CASE_DIR, exist_ok=True)
            with open(path, "w") as fh:
                json.dump(case, fh, indent=1)
        else:
            missing.append(cid)
            continue
        proto = case.get("protocol") or {}

        def field(key, default=None):
            """Top level of a this-round JSON, else the protocol block."""
            if key in case:
                return case[key]
            if key in proto:
                return proto[key]
            return default

        r_i = float(case["column_radius_measured"])
        extent = case.get("extent") or {}
        by_metric = case.get("runout_ratio_by_metric") or {}
        if not by_metric:
            by_metric = {"p995": float(case["runout_ratio_p995"])}
        rows.append({
            "case_id": cid,
            "ladder": ladder,
            "aspect_nominal": float(case["aspect_nominal"]),
            "aspect_measured": float(case["aspect_measured"]),
            "grains_per_radius": float(field("grains_per_radius")),
            # d = 2 r_p and r_p = r_i / (2 gpr) -> d = r_i / gpr.  The 2x error
            # here silently broke the refinement-ratio uniformity check.
            "grain_diameter": float(field("grain_diameter") or
                                    (r_i / float(field("grains_per_radius")))),
            "n_particles": int(case["n_particles"]),
            "solid_fraction_initial": float(case["solid_fraction_initial"]),
            "stop_reason": field("stop_reason"),
            "time_end": float(case["time_end"]),
            "runout_ratio": {k: float(v) for k, v in by_metric.items()},
            "reference": case["reference"],
            "arrival": case.get("arrival"),
            "source": case["source"],
            "wall_clock_s": case.get("wall_clock_s"),
        })
    rows.sort(key=lambda r: (r["ladder"], -r["grains_per_radius"]))
    summary = {
        "round": ROUND,
        "anchor": "CASE-COLUMN-COLLAPSE",
        "reference": "Lube et al. 2004 JFM 508:175-199",
        "tolerance": TOLERANCE,
        "n_declared": len(RUNGS),
        "n_present": len(rows),
        "missing": missing,
        "cases": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    if emit:
        for row in rows:
            p = os.path.join(CASE_DIR, f"{row['case_id']}.json")
            print(encode(p), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--shard", help="i/n")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--emit", action="store_true",
                    help="collect and print base64 payloads of every case JSON")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for ladder, aspect, gpr, reuse in RUNGS:
            case = base.make_case(_case_spec(build_spec(ladder, aspect, gpr)))
            print(f"{case_id(ladder, gpr):<14} a={aspect:<4g} gpr={gpr:<4g} "
                  f"d={2 * case.radius:.5f}m kn={case.kn:.6g} "
                  f"dt={case.material().timestep:.3e}s reuse={reuse}")
        return 0
    if args.case:
        spec = None
        for ladder, aspect, gpr, _reuse in RUNGS:
            if case_id(ladder, gpr) == args.case:
                spec = build_spec(ladder, aspect, gpr)
        if spec is None:
            raise SystemExit(f"unknown case {args.case}")
        out = run_case(spec, progress=args.progress)
        print(f"{args.case}: n={out['n_particles']} a_meas={out['aspect_measured']:.4f} "
              f"runout={out['runout_ratio_p995']:.4f} wall={out['wall_clock_s']:.0f}s")
        if args.emit:
            print(encode(os.path.join(CASE_DIR, f"{args.case}.json")), flush=True)
        return 0
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        todo = [r for k, r in enumerate(RUNGS) if k % n == i]
        for _ladder, aspect, gpr, _reuse in todo:
            spec = build_spec(_ladder, aspect, gpr)
            if os.path.exists(os.path.join(CASE_DIR, f"{spec['case_id']}.json")):
                continue
            out = run_case(spec, progress=args.progress)
            print(f"{spec['case_id']}: n={out['n_particles']} "
                  f"runout={out['runout_ratio_p995']:.4f} "
                  f"wall={out['wall_clock_s']:.0f}s", flush=True)
            print(encode(os.path.join(CASE_DIR, f"{spec['case_id']}.json")),
                  flush=True)
        return 0
    summary = collect(emit=args.emit)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=1))
    if summary["missing"]:
        print(f"MISSING {summary['missing']}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())