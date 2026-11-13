"""CASE-COLUMN-COLLAPSE — is the shallow column's late emplacement *physical*?

The anchor's acceptance is ``|error| < 15 %`` against Lube et al. (2004).  The
runout law passes 5/5 on the closure arm, but the *second* law
(``t_inf = 3 sqrt(h_i/g)``) fails on the shallow branch: at ``a = 0.5`` the
99 % arrival is +27 ... +56 % late (``docs/AUDIT_COLUMN_COLLAPSE_TIME_LAW_m17534.md``),
and the coupled front-speed prediction is off by -20 ... -40 % there
(``docs/AUDIT_COLUMN_COLLAPSE_FRONT_KINEMATICS_m17770.md``).  Both audits name
the *same* remaining debt: every existing convergence control sits at
``a in {1, 3}`` -- i.e. exactly where the law already holds -- so nobody has
ever shown that the shallow-branch excess survives a change of discretisation.
That question cannot be answered from stored provenance; it needs new runs.

This module is the reducer/adjudicator for those runs (benchmark wave 5d,
``benchmarks/case_column_collapse_lube2004.py --wave5d``): for each aspect ratio
in ``{0.5, 1.5}`` it compares the baseline discretisation (8 grains per column
radius, ``dt_ref``) against

* resolution-refined   (12 grains per column radius),
* time-refined         (``dt/2``),
* fully refined        (12 grains per radius **and** ``dt/2``, at ``a = 0.5``),

and decides, from **pre-registered** rules stated before the runs (they are
constants below: ``H_RES_PP``, ``H_DT_REL``, ``H_RUNOUT_PP``):

* ``H_RES``    -- the ``a = 0.5`` emplacement-time error moves by less than
  15 percentage points between 8 and 12 grains per column radius;
* ``H_DT``     -- ``dt -> dt/2`` changes ``t95`` by less than 5 % relative;
* ``H_SIGN``   -- every refined ``a = 0.5`` variant stays *outside* the 15 %
  band with the same sign (no refinement flips the verdict);
* ``H_RUNOUT`` -- the runout law is equally stable (< 15 pp drift), so the two
  laws cannot be traded against each other by adjusting the discretisation.

``H_RES`` and ``H_DT`` both holding means the shallow-branch excess is **not**
a discretisation artifact: the honest verdict stays "the second law fails at
``a = 0.5``", now with a discretisation-independent magnitude.  If either is
refuted, the anchor's shallow verdict is downgraded to "undetermined at this
discretisation" instead -- both outcomes are informative and neither is
allowed to be overwritten.

A cross-architecture consistency check is included: the baseline cases are
re-run on the cluster (loong64) although they already exist from the control
plane (x86_64), so the reducer reports the relative difference of every scalar
between the two architectures.  It is a *byproduct* check, not the main claim.

Usage::

    PYTHONPATH=src python3 scripts/analyze_column_collapse_shallow_convergence_m17917.py \
        --src-dir /path/to/fetched/cases \
        --baseline-dir /root/tp-agent/work/m17111/runs/case_column_collapse_lube2004/cases \
        --out-dir docs/artifacts/column_collapse_shallow_convergence_m17917
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass

# --- pre-registered decision constants (fixed before the wave-5d runs) ------
TOLERANCE = 0.15            # anchor acceptance band, |relative error| < 15 %
H_RES_PP = 0.15             # H_RES: error drift across resolution < 15 pp
H_DT_REL = 0.05             # H_DT: t95 drift across dt < 5 % relative
H_RUNOUT_PP = 0.15          # H_RUNOUT: runout error drift < 15 pp
ASPECTS = (0.5, 1.5)
SHALLOW = 0.5

# baseline (8 grains per column radius, dt_ref) case ids of wave 5b
BASELINE_ID = "w5b_confirm_wall{wall:g}_a{a:g}"
WALL = 0.8
# wave-5d case ids, produced by benchmark --wave5d
VARIANT_ID = {
    ("resolution", 0.5): "w5d_gpr12_a0.5",
    ("resolution", 1.5): "w5d_gpr12_a1.5",
    ("timestep", 0.5): "w5d_dt0.5_a0.5",
    ("timestep", 1.5): "w5d_dt0.5_a1.5",
    ("combined", 0.5): "w5d_fine_a0.5",
}
METRICS = ("t90", "t95", "t99")

REDUCED_FIELDS = (
    "case", "axis", "aspect_nominal", "aspect_measured", "grains_per_radius",
    "dt_scale", "n_particles", "dt", "n_steps", "stop_reason",
    "runout_ratio", "ref_runout_ratio", "err_runout",
    "t90", "t95", "t99", "ref_emplacement_time", "err_t90", "err_t95", "err_t99",
    "vbar", "vbar_ref", "vbar_gap", "energy_residual", "peak_front_speed",
)


# --------------------------------------------------------------------------- #
# derivation (all reference numbers are evaluated, never typed)
# --------------------------------------------------------------------------- #
def lube_runout_ratio(a: float) -> float:
    """Lube et al. (2004) runout law (same branch logic as the solver)."""
    if a < 1.7:
        return 1.0 + 1.24 * a
    return 1.0 + 1.6 * math.sqrt(a)


def lube_emplacement_time(a: float, r_i: float, g: float = 9.81) -> float:
    """``t_inf = 3 sqrt(h_i/g) = 3 sqrt(a r_i / g)``."""
    return 3.0 * math.sqrt(a * r_i / g)


def mean_front_speed_gap(rows_like: dict) -> dict:
    """Dimensionless mean front speed and its gap to the coupled-law prediction.

    ``v_mean = runout / t_arrival``, ``V = v_mean / sqrt(g r_i)``; the reference
    value combines the two laws and is evaluated, not typed::

        V_ref = (R r_i / t_ref) / sqrt(g r_i),  R = ratio_law - 1
    """
    a = rows_like["aspect_nominal"]
    r_i = rows_like["column_radius_measured"]
    g = rows_like.get("gravity", 9.81)
    t_ref = lube_emplacement_time(a, r_i, g)
    runout = rows_like["runout_ratio"] - 1.0
    vbar = runout * r_i / rows_like["t99"] / math.sqrt(g * r_i)
    vbar_ref = (lube_runout_ratio(a) - 1.0) * r_i / t_ref / math.sqrt(g * r_i)
    return {"vbar": vbar, "vbar_ref": vbar_ref,
            "vbar_gap": vbar / vbar_ref - 1.0}


def reduce_case(payload: dict, axis: str) -> dict:
    """One per-case provenance dict -> the scalars this study adjudicates."""
    case = payload["case"]
    a = float(case["aspect_ratio"])
    r_i = float(payload["column_radius_measured"])
    g = float(case.get("gravity", 9.81))
    frac = payload.get("t_reach_by_fraction", {})
    row = {
        "case": case["label"],
        "axis": axis,
        "aspect_nominal": a,
        "aspect_measured": float(payload["aspect_measured"]),
        "grains_per_radius": float(case["grains_per_radius"]),
        "dt_scale": float(case["dt_scale"]),
        "n_particles": int(payload["n_particles"]),
        "dt": float(payload["dt"]),
        "n_steps": int(payload["n_steps"]),
        "stop_reason": payload["stop_reason"],
        "runout_ratio": float(payload["runout_ratio_p995"]),
        "ref_runout_ratio": lube_runout_ratio(a),
        "t90": float(frac.get("0.9", payload["t_reach"])),
        "t95": float(frac.get("0.95", payload["t_reach"])),
        "t99": float(payload["t_reach"]),
        "ref_emplacement_time": lube_emplacement_time(a, r_i, g),
        "column_radius_measured": r_i,
        "gravity": g,
        "peak_front_speed": float(payload.get("peak_front_speed", float("nan"))),
        "energy_residual": float(
            payload["energy"]["residual_fraction_of_potential"]),
        "wall_clock_s": float(payload.get("wall_clock_s", float("nan"))),
    }
    row["err_runout"] = row["runout_ratio"] / row["ref_runout_ratio"] - 1.0
    for m in METRICS:
        row[f"err_{m}"] = row[m] / row["ref_emplacement_time"] - 1.0
    row.update(mean_front_speed_gap(row))
    return row


def drift(base: float, variant: float, relative: bool) -> float:
    """Drift between two variants of a metric.

    ``relative=True``  -> relative change ``variant/base - 1``;
    ``relative=False`` -> difference in the *error* (percentage points / 100).
    """
    if relative:
        if base == 0.0:
            raise ValueError("relative drift of a zero baseline is undefined")
        return variant / base - 1.0
    return variant - base


# --------------------------------------------------------------------------- #
# adjudication
# --------------------------------------------------------------------------- #
def adjudicate(rows: list[dict]) -> dict:
    """Apply the pre-registered rules of the module docstring."""
    by_id = {r["case"]: r for r in rows}
    out = {"tolerance": TOLERANCE,
           "rules": {"H_RES_pp": H_RES_PP, "H_DT_rel": H_DT_REL,
                     "H_RUNOUT_pp": H_RUNOUT_PP},
           "aspects": {}, "hypotheses": {}}

    base_err95, res_err95, dt_err95 = {}, {}, {}
    for a in ASPECTS:
        base = by_id.get(BASELINE_ID.format(wall=WALL, a=a))
        if base is None:
            raise KeyError(f"missing baseline case {BASELINE_ID.format(wall=WALL, a=a)}")
        entry = {"baseline": {k: base[k] for k in
                              ("case", "t95", "t99", "err_t95", "err_t99",
                               "runout_ratio", "err_runout", "vbar_gap")}}
        base_err95[a] = base["err_t95"]
        for axis in ("resolution", "timestep", "combined"):
            cid = VARIANT_ID.get((axis, a))
            if cid is None:
                continue
            var = by_id.get(cid)
            if var is None:
                entry[axis] = None
                continue
            entry[axis] = {
                "case": cid,
                "t95": var["t95"], "t99": var["t99"],
                "err_t95": var["err_t95"], "err_t99": var["err_t99"],
                "runout_ratio": var["runout_ratio"],
                "err_runout": var["err_runout"], "vbar_gap": var["vbar_gap"],
                "n_particles": var["n_particles"], "dt": var["dt"],
                "stop_reason": var["stop_reason"],
                "drift_t95_rel": drift(base["t95"], var["t95"], True),
                "drift_err95_pp": drift(base["err_t95"], var["err_t95"], False),
                "drift_err99_pp": drift(base["err_t99"], var["err_t99"], False),
                "drift_err_runout_pp": drift(base["err_runout"], var["err_runout"],
                                             False),
            }
            if axis == "resolution":
                res_err95[a] = var["err_t95"]
            if axis == "timestep":
                dt_err95[a] = var["err_t95"]
        out["aspects"][f"{a:g}"] = entry

    shallow = out["aspects"].get(f"{SHALLOW:g}", {})
    res = shallow.get("resolution")
    dts = shallow.get("timestep")
    comb = shallow.get("combined")

    h_res = bool(res and abs(res["drift_err95_pp"]) < H_RES_PP)
    h_dt = bool(dts and abs(dts["drift_t95_rel"]) < H_DT_REL)
    refined = [v for v in (res, dts, comb) if v]
    h_sign = bool(refined) and all(
        abs(v["err_t95"]) >= TOLERANCE and
        math.copysign(1.0, v["err_t95"]) == math.copysign(
            1.0, base_err95.get(SHALLOW, 1.0))
        for v in refined)
    h_runout = bool(refined) and all(
        abs(v["drift_err_runout_pp"]) < H_RUNOUT_PP for v in refined)

    out["hypotheses"] = {
        "H_RES": {"holds": h_res,
                  "detail": None if res is None else res["drift_err95_pp"],
                  "rule": f"|d(err_t95)| < {H_RES_PP} across 8 -> 12 grains/radius"},
        "H_DT": {"holds": h_dt,
                 "detail": None if dts is None else dts["drift_t95_rel"],
                 "rule": f"|d t95 / t95| < {H_DT_REL} across dt -> dt/2"},
        "H_SIGN": {"holds": h_sign,
                   "detail": [v["err_t95"] for v in refined],
                   "rule": f"every refined a={SHALLOW:g} variant keeps the sign "
                           f"of the baseline error and stays outside {TOLERANCE}"},
        "H_RUNOUT": {"holds": h_runout,
                     "detail": None if not refined else
                     [v["drift_err_runout_pp"] for v in refined],
                     "rule": f"|d(err_runout)| < {H_RUNOUT_PP} in every variant"},
    }
    physical = h_res and h_dt and h_sign
    out["verdict"] = {
        "shallow_excess_is_physical": physical,
        "statement": (
            "a=0.5 emplacement-time excess is discretisation-independent: the "
            "second Lube law genuinely fails on the shallow branch in this "
            "model" if physical else
            "the shallow-branch excess is NOT shown to be discretisation-independent; "
            "the anchor's shallow verdict stays undetermined at this discretisation"),
    }
    return out


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_cases(src_dir: str, baseline_dir: str | None = None) -> list[dict]:
    """Load the wave-5d variants plus the re-run wave-5b baselines."""
    rows, seen = [], set()
    ids = [BASELINE_ID.format(wall=WALL, a=a) for a in ASPECTS]
    ids += list(VARIANT_ID.values())
    for cid in ids:
        for directory, tag in ((src_dir, "wave5d"), (baseline_dir, "wave5b")):
            if not directory:
                continue
            path = os.path.join(directory, f"{cid}.json")
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                payload = json.load(fh)
            axis = "baseline" if cid.startswith("w5b_") else _axis_of(cid)
            rows.append(reduce_case(payload, axis))
            seen.add(cid)
            break
    missing = [cid for cid in ids if cid not in seen and not cid.startswith("w5d_")]
    if missing:
        raise FileNotFoundError(f"missing baseline provenance: {missing}")
    return rows


def _axis_of(cid: str) -> str:
    if cid == "w5d_fine_a0.5":
        return "combined"
    if cid.startswith("w5d_gpr"):
        return "resolution"
    return "timestep"


def cross_architecture(cases: str, reference: str) -> dict:
    """Relative difference of every scalar between two runs of the same case."""
    with open(cases) as fh:
        a = json.load(fh)
    with open(reference) as fh:
        b = json.load(fh)
    keys = ("runout_ratio_p995", "t_reach", "column_radius_measured",
            "aspect_measured", "n_particles", "dt")
    out = {"case": a["case"]["label"], "n_particles": a["n_particles"],
           "cluster_n_particles": b["n_particles"], "fields": {}}
    for k in keys:
        va, vb = float(a[k]), float(b[k])
        out["fields"][k] = {"cluster": va, "control_plane": vb,
                            "rel_diff": (va - vb) / vb if vb else float("nan")}
    frac = ("0.9", "0.95")
    for f in frac:
        va = a["t_reach_by_fraction"][f]
        vb = b["t_reach_by_fraction"][f]
        out["fields"][f"t{f}"] = {"cluster": va, "control_plane": vb,
                                  "rel_diff": (va - vb) / vb}
    return out


def write_reduced(rows: list[dict], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "cases_reduced.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(REDUCED_FIELDS), extrasaction="ignore")
        wr.writeheader()
        for row in rows:
            wr.writerow(row)
    return path


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(paths: list[str], out_dir: str) -> str:
    manifest = {os.path.relpath(p, out_dir): sha256_of(p) for p in paths}
    path = os.path.join(out_dir, "manifest_sha256.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", required=True,
                    help="directory holding the fetched wave-5d case JSONs")
    ap.add_argument("--baseline-dir", default=None,
                    help="directory holding the wave-5b baseline case JSONs")
    ap.add_argument("--out-dir", default="docs/artifacts/"
                                        "column_collapse_shallow_convergence_m17917")
    ap.add_argument("--cross-arch-a", default=None, help="cluster re-run case JSON")
    ap.add_argument("--cross-arch-b", default=None, help="control-plane case JSON")
    args = ap.parse_args(argv)

    rows = load_cases(args.src_dir, args.baseline_dir)
    bundle = adjudicate(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = write_reduced(rows, args.out_dir)
    if args.cross_arch_a and args.cross_arch_b:
        bundle["cross_architecture"] = cross_architecture(args.cross_arch_a,
                                                          args.cross_arch_b)
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump({"cases": rows, "adjudication": bundle}, fh, indent=1)
    write_manifest([summary_path, csv_path], args.out_dir)
    print(json.dumps(bundle["hypotheses"], indent=1))
    print(json.dumps(bundle["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())