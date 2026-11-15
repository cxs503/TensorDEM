"""Initial-packing-fraction axis for CASE-COLUMN-COLLAPSE (m21723).

Question
--------
The resolution ladder of m21656 held ``r_i`` fixed and varied ``gpr`` -- but the
HCP lattice fills the cylinder better at finer resolution, so ``phi_0`` slid
from 0.546 (4 grains/``r_i``) to 0.662 (16 grains/``r_i``) on the ``a = 3``
ladder.  The ladder therefore sweeps **two** axes at once.  This runner measures
the other axis on its own:

    at fixed ``(a, gpr)``, sweep the lattice gap ``packing_gap`` ->
    ``phi_0`` falls, everything else (radius, stiffness law, contact law,
    floor, integrator, metrics) is the frozen closed arm.

Arms (pre-registered)
---------------------
================  =====  ================  ==================================
arm               ``a``  grains/``r_i``    ``packing_gap`` levels
================  =====  ================  ==================================
``A3g8``          3.0    8                 {0.02 (reused), 0.05, 0.08, 0.12}
``A3g12``         3.0    12                {0.02 (reused), 0.05, 0.08}
``A1g8``          1.0    8                 {0.02 (reused), 0.05, 0.08, 0.12}
================  =====  ================  ==================================

``gap = 0`` is deliberately **not** an arm: the HCP lattice at zero gap already
starts in penetration (``min_gap = -2e-5 m``, thousands of overlapping pairs),
which the case's own admissibility rule rejects (``admissibility.py``:
``overlap_pairs == 0``).  Every member of this sweep is gated by
:func:`~tensordem.granular_column.admissibility.overlap_stats` *before*
integration; a member that fails is recorded as rejected, never run.

Reading of the axis (stated up front, not after the fact)
---------------------------------------------------------
``packing_gap`` sets the initial free volume of a lattice: grains start apart
and free-fall a distance ``gap * d`` before the first contact.  It is therefore
an *initial-porosity* axis, **not** a prepared loose deposit (that is the
pluviation route of m20811).  The free-fall drop is recorded per member in
grain diameters so its kinetic-energy scale can be compared with the collapse
velocity scale ``sqrt(g h_i)``; at ``gap = 0.12`` and ``d = 0.059 m`` the drop
is 7.1 mm, i.e. an impact velocity of ~0.37 m/s against a collapse scale of
~3.8 m/s.

Usage (node, repo checkout)
---------------------------
::

    python3 benchmarks/case_column_collapse_packing_axis_m21723.py --list
    python3 benchmarks/case_column_collapse_packing_axis_m21723.py --case PH_A3g8_gap0.05
    python3 benchmarks/case_column_collapse_packing_axis_m21723.py --shard 0/4
    python3 benchmarks/case_column_collapse_packing_axis_m21723.py --collect
    python3 benchmarks/case_column_collapse_packing_axis_m21723.py --phi-map
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
from tensordem.granular_column.admissibility import overlap_stats  # noqa: E402
from tensordem.granular_column.arrival import arrival_report  # noqa: E402
from tensordem.granular_column.axisym import (  # noqa: E402
    build_axisymmetric_column, radial_extent, simulate_axisymmetric_column)
from emit_column_collapse_payload_m20873 import encode  # noqa: E402
from tensordem.granular_column.packing_axis import solid_fraction  # noqa: E402

ROUND = "m21723"
OUT_DIR = os.path.join("runs", "case_column_collapse_packing_axis_m21723")
CASE_DIR = os.path.join(OUT_DIR, "cases")
TOLERANCE = base.TOLERANCE

#: Committed artifacts that already hold the ``packing_gap = 0.02`` member of
#: each arm (the frozen arm itself), keyed by arm.
REUSE = {
    "A3g8": "docs/artifacts/column_collapse_deep_refinement_m20873/cases/R_a3_g8.json",
    "A3g12": "docs/artifacts/column_collapse_deep_refinement_m20873/cases/R_a3_g12.json",
    "A1g8": "docs/artifacts/column_collapse_deep_refinement_m20873/cases/R_a1_g8.json",
}

#: Ladder rungs of m21656, used only to map ``phi_0`` onto the resolution axis
#: (pure geometry -- no integration, see ``--phi-map``).
LADDER = [(3.0, g) for g in (4.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0)] + \
         [(1.0, g) for g in (4.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0)]

#: (arm, aspect, grains per column radius, gap levels)
ARMS = [
    ("A3g8", 3.0, 8.0, (0.02, 0.05, 0.08, 0.12)),
    ("A3g12", 3.0, 12.0, (0.02, 0.05, 0.08)),
    ("A1g8", 1.0, 8.0, (0.02, 0.05, 0.08, 0.12)),
]


def case_id(arm: str, gap: float) -> str:
    return f"PH_{arm}_gap{gap:.2f}"


def build_cases() -> list[dict]:
    cases = []
    for arm, aspect, gpr, gaps in ARMS:
        for gap in gaps:
            cases.append({"case_id": case_id(arm, gap), "arm": arm,
                          "axis": "PHI", "aspect_ratio": aspect,
                          "grains_per_radius": gpr, "packing_gap": gap})
    return cases


CASES = build_cases()
CASE_BY_ID = {c["case_id"]: c for c in CASES}


def case_path(cid: str) -> str:
    return os.path.join(CASE_DIR, f"{cid}.json")


def _spec(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if k not in ("arm",)}


def make_case(spec: dict):
    """Frozen arm + this member's lattice gap; everything else inherited."""
    full = {**base.BASE}
    full["packing_gap"] = float(spec["packing_gap"])
    full["case_id"] = spec["case_id"]
    full["aspect_ratio"] = float(spec["aspect_ratio"])
    full["grains_per_radius"] = float(spec["grains_per_radius"])
    return base.make_case(full)


def protocol_signature(spec: dict) -> dict:
    sig = base.protocol_signature(_spec(spec))
    sig.pop("case_id", None)
    sig["case_id"] = spec["case_id"]
    sig["arm"] = spec["arm"]
    sig["axis"] = "PHI"
    sig["round"] = ROUND
    return sig


def signature_sha256(sig: dict) -> str:
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()


def assert_reusable(path: str, spec: dict) -> dict:
    """Prove a committed case JSON is the ``packing_gap = 0.02`` member of this arm.

    Four gates -- all must pass, else ``ValueError``:

    1. every physics field equals the frozen ``BASE`` arm **except**
       ``packing_gap``, which must equal this member's gap (0.02 for the reused
       artifacts) -- a reused artifact may never smuggle in a different gap,
    2. ``radius = r_i / (2 gpr)``,
    3. ``kn`` follows the sag-preserving law ``6e4 (d/0.03)^2``,
    4. the measured column radius is the ladder constant ``0.48 m`` to 10 %.

    The measured aspect ratio is *allowed* to differ from nominal (the lattice
    truncates differently against the wall), and is recorded.
    """
    with open(path) as fh:
        raw = fh.read()
    case = json.loads(raw)
    proto = case["protocol"]
    checked = ("mu", "mu_rolling", "mu_wall", "restitution", "kt_ratio",
               "gravity", "jitter", "seed", "dt_scale", "duration_factor",
               "min_time", "contact_mode")
    mismatch = [(k, proto.get(k), base.BASE[k]) for k in checked
                if proto.get(k) != base.BASE[k]]
    if mismatch:
        raise ValueError(f"{path}: physics differs from the frozen arm: {mismatch}")
    if proto.get("packing_gap") != float(spec["packing_gap"]):
        raise ValueError(f"{path}: packing_gap {proto.get('packing_gap')!r} is not "
                         f"this member's {spec['packing_gap']!r}")
    gpr = float(spec["grains_per_radius"])
    radius = float(proto["radius"])
    if abs(radius - base.case_radius(gpr)) > 1e-12:
        raise ValueError(f"{path}: radius {radius} is not r_i/(2 gpr) "
                         f"= {base.case_radius(gpr)}")
    if abs(float(proto["kn"]) - base.case_stiffness(radius)) > 1e-6 * float(proto["kn"]):
        raise ValueError(f"{path}: kn {proto['kn']} is off the sag-preserving law")
    r_meas = float(case["column_radius_measured"])
    if abs(r_meas - base.COLUMN_RADIUS) > 0.10 * base.COLUMN_RADIUS:
        raise ValueError(f"{path}: measured r_i {r_meas} is off the ladder "
                         f"constant {base.COLUMN_RADIUS} by more than 10 %")
    return {"source_path": path,
            "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "case": case}


def run_case(spec: dict, progress: bool = False) -> dict:
    """Integrate one member of the ``phi_0`` axis (cache-first)."""
    cid = spec["case_id"]
    path = case_path(cid)
    if os.path.exists(path):
        with open(path) as fh:
            res = json.load(fh)
        res["cached"] = True
        return res
    if float(spec["packing_gap"]) == 0.02 and spec["arm"] in REUSE:
        reuse = assert_reusable(REUSE[spec["arm"]], spec)
        case = reuse["case"]
        out = dict(case)
        out["case_id"] = cid
        out["arm"] = spec["arm"]
        out["axis"] = "PHI"
        out["reused_from"] = {"path": reuse["source_path"],
                              "sha256": reuse["source_sha256"],
                              "original_case_id": case["case_id"]}
        out["protocol"] = protocol_signature(spec)
        out["protocol_sha256"] = signature_sha256(out["protocol"])
        out["phi0"] = float(case["solid_fraction_initial"])
        out["cached"] = True
        os.makedirs(CASE_DIR, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(out, fh)
        return out

    case = make_case(spec)
    pos, mat = build_axisymmetric_column(case)
    adm = overlap_stats(pos, mat.radius)
    if not adm["admissible"]:
        out = {"case_id": cid, "arm": spec["arm"], "axis": "PHI",
               "rejected": True, "admissibility": adm,
               "reason": "initial packing has overlapping pairs"}
        os.makedirs(CASE_DIR, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(out, fh)
        raise SystemExit(f"{cid}: rejected by admissibility gate {adm}")
    ext = radial_extent(pos, mat.radius)
    h_i = float(pos[:, 2].max() + mat.radius)
    phi0_geom = solid_fraction(pos.shape[0], mat.radius, ext["p995"], h_i)

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
    errors = {conv: base.error(res["runout_ratio_p995"],
                               refs[conv]["runout_ratio"])
              for conv in ("nominal", "measured")}
    out = {
        "case_id": cid,
        "arm": spec["arm"],
        "axis": "PHI",
        "protocol": protocol_signature(spec),
        "protocol_sha256": signature_sha256(protocol_signature(spec)),
        "packing_gap": float(case.packing_gap),
        "gap_in_grain_diameters": float(case.packing_gap),
        "free_fall_drop_m": float(case.packing_gap * 2.0 * case.radius),
        "free_fall_velocity_scale": float(math.sqrt(2.0 * case.gravity *
                                                    case.packing_gap * 2.0 * case.radius)),
        "collapse_velocity_scale": float(math.sqrt(case.gravity * h_i)),
        "admissibility": adm,
        "n_particles": res["n_particles"],
        "dt": res["dt"],
        "n_steps": res["n_steps"],
        "time_end": res["time_end"],
        "stop_reason": res["stop_reason"],
        "aspect_nominal": float(spec["aspect_ratio"]),
        "aspect_measured": res["aspect_measured"],
        "column_radius_measured": res["column_radius_measured"],
        "column_height_measured": res["column_height_measured"],
        "phi0": float(res["solid_fraction_initial"]),
        "phi0_geometry_check": phi0_geom,
        "phi0_consistency": abs(phi0_geom - float(res["solid_fraction_initial"])),
        "runout_ratio_p995": float(res["runout_ratio_p995"]),
        "runout_ratio_max": float(res["runout_ratio_max"]),
        "reference": refs,
        "runout_error_nominal": errors["nominal"],
        "runout_error_measured": errors["measured"],
        "arrival": arrivals,
        "energy": res["energy"],
        "inertial": res["inertial"],
        "history": {"time": hist["time"], "front": hist["front"],
                    "kinetic": hist["kinetic"]},
        "wall_clock_s": wall,
    }
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


def phi_map() -> dict:
    """``phi_0`` along the m21656 resolution ladder -- pure geometry, no DEM."""
    rows = []
    for aspect, gpr in LADDER:
        case = base.make_case({**base.BASE, "case_id": "MAP", "aspect_ratio": aspect,
                               "grains_per_radius": gpr})
        pos, mat = build_axisymmetric_column(case)
        ext = radial_extent(pos, mat.radius)
        h_i = float(pos[:, 2].max() + mat.radius)
        adm = overlap_stats(pos, mat.radius)
        rows.append({"aspect": aspect, "grains_per_radius": gpr,
                     "n_particles": int(pos.shape[0]),
                     "aspect_measured": h_i / ext["p995"],
                     "phi0": solid_fraction(pos.shape[0], mat.radius,
                                            ext["p995"], h_i),
                     "admissible": bool(adm["admissible"])})
    out = {"ladder": rows,
           "phi0_span": {f"{a:g}": [min(r["phi0"] for r in rows if r["aspect"] == a),
                                    max(r["phi0"] for r in rows if r["aspect"] == a)]
                         for a in sorted({r["aspect"] for r in rows})}}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "phi_map.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def emit_case(cid: str) -> None:
    """Push one result JSON into the job log (``fetch`` returns committed state only)."""
    path = case_path(cid)
    if os.path.exists(path):
        print(encode(path), flush=True)


def run_shard(index: int, total: int, progress: bool = False,
              emit: bool = False) -> list[str]:
    done = []
    for k, spec in enumerate(CASES):
        if k % total != index:
            continue
        print(f"[shard {index}/{total}] {spec['case_id']} ...", flush=True)
        res = run_case(spec, progress=progress)
        if res.get("rejected"):
            print("  rejected by admissibility", flush=True)
            done.append(spec["case_id"])
            continue
        print(f"  N={res['n_particles']} phi0={res['phi0']:.4f} "
              f"rho={res['runout_ratio_p995']:.4f} "
              f"err={res['runout_error_measured']:+.2%} "
              f"wall={res['wall_clock_s']:.0f}s stop={res['stop_reason']}", flush=True)
        if emit:
            emit_case(spec["case_id"])
        done.append(spec["case_id"])
    return done


def collect(out_dir: str = OUT_DIR) -> dict:
    rows, missing = [], []
    for spec in CASES:
        path = os.path.join(out_dir, "cases", f"{spec['case_id']}.json")
        if not os.path.exists(path):
            missing.append(spec["case_id"])
            continue
        with open(path) as fh:
            case = json.load(fh)
        if case.get("rejected"):
            rows.append({"case_id": spec["case_id"], "arm": spec["arm"],
                         "rejected": True})
            continue
        rows.append({
            "case_id": spec["case_id"], "arm": spec["arm"],
            "aspect_measured": case["aspect_measured"],
            "grains_per_radius": spec["grains_per_radius"],
            "packing_gap": case["packing_gap"],
            "phi0": case["phi0"],
            "runout_ratio_p995": case["runout_ratio_p995"],
            "runout_error_measured": case["runout_error_measured"],
            "t90": case["arrival"]["fraction"].get("0.9"),
            "stop_reason": case["stop_reason"],
            "wall_clock_s": case["wall_clock_s"],
            "source": ("reused" if case.get("reused_from") else "run"),
        })
    summary = {"round": ROUND, "case": "CASE-COLUMN-COLLAPSE", "axis": "PHI",
               "reference": "Lube et al. 2004 JFM 508:175-199",
               "tolerance": TOLERANCE,
               "n_cases_declared": len(CASES), "n_cases_present": len(rows),
               "missing": missing, "rows": rows}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--shard", nargs=1)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--phi-map", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--emit", action="store_true",
                    help="print base64/gzip payloads of every case touched")
    args = ap.parse_args(argv)
    if args.list:
        for spec in CASES:
            print(json.dumps(spec, sort_keys=True))
        return 0
    if args.phi_map:
        print(json.dumps(phi_map(), indent=1))
        return 0
    if args.case:
        spec = CASE_BY_ID[args.case]
        res = run_case(spec, progress=args.progress)
        print(json.dumps({k: v for k, v in res.items() if k != "history"},
                         indent=1, default=float))
        return 0
    if args.shard:
        i, n = (int(v) for v in args.shard[0].split("/"))
        run_shard(i, n, progress=args.progress, emit=args.emit)
        return 0
    if args.collect:
        print(json.dumps(collect(), indent=1, default=float))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())