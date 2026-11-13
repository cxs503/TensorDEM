"""XVAL: rolling friction as a particle-shape surrogate (Wensrich & Katterfeld 2012).

Two subcommands, no hand-typed numbers anywhere.

``point``
    Runs one calibration point of the sandpile protocol at a given ``mu_r`` and
    writes the full protocol record to
    ``artifacts/xval_rolling_shape_surrogate/points/<tag>.json``.

``aggregate``
    Reads every point file, builds the ``mu_r -> repose angle`` calibration
    surface for the reference condition, fits the saturating surrogate, inverts
    it for the literature angles (parsed from the in-repository literature
    table), scores out-of-sample transfer against the points whose condition
    differs, and writes ``result.json``.

Usage::

    PYTHONPATH=src python benchmarks/xval_rolling_shape_surrogate/run_sweep.py \
        point --mu-r 0.3 --tag ref_mu0p30
    PYTHONPATH=src python benchmarks/xval_rolling_shape_surrogate/run_sweep.py aggregate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from tensordem.protocols.sandpile import PRIORS
from tensordem.protocols.sandpile_calibration import (
    CalibrationCondition,
    CalibrationPoint,
    UnattainableTarget,
    fit_saturation,
    invert_target,
    parse_wk2012_reference,
    run_point,
    transfer_error,
)

ARTIFACT_DIR = Path("artifacts/xval_rolling_shape_surrogate")
POINT_DIR = ARTIFACT_DIR / "points"

#: Reference condition of the calibration surface (the one the curve is fit on).
REFERENCE = CalibrationCondition(
    n_particles=1600, seed=11, pour_rate=890.0, drop_height=0.005,
    settle_time=0.15)

#: Repose-angle acceptance gate (owner standard: >15 % is not acceptable).
GATE_REL_ERROR = 0.15

#: A surface point enters the fit only if the flank is a measurable heap:
#: ``r2`` of the linear height-field fit and the number of grain layers (apex
#: height in radii).  Failing points are reported, never silently dropped.
MIN_VALID_R2 = 0.9
MIN_VALID_LAYERS = 3.0


def _git_rev() -> str:
    head = Path(".git/HEAD")
    try:
        ref = head.read_text().strip().split()[-1]
        return (Path(".git") / ref).read_text().strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition_of(point: dict) -> dict:
    """Protocol condition of a point file (stored under ``row``)."""
    return dict(point["row"]["condition"])


def cmd_point(args: argparse.Namespace) -> int:
    condition = CalibrationCondition(
        n_particles=args.n_particles, seed=args.seed,
        pour_rate=args.pour_rate, drop_height=args.drop_height,
        settle_time=args.settle, dt=(REFERENCE.dt if not args.dt_half else None))
    overrides: dict = {}
    if args.dt_half:
        base = condition.config(args.mu_r)
        overrides["dt"] = 0.5 * base.dt_effective
    point = run_point(args.mu_r, condition, mu_s=args.mu_s, **overrides)
    POINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": args.tag,
        "mu_r": args.mu_r,
        "mu_s": args.mu_s,
        "row": point.as_row(),
        "record": point.record,
        "wall_seconds": point.wall_seconds,
        "provenance": {
            "script": "benchmarks/xval_rolling_shape_surrogate/run_sweep.py",
            "git_rev": _git_rev(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    out = POINT_DIR / f"{args.tag}.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"{args.tag}: mu_r={args.mu_r} theta={point.theta_deg:.4f} deg "
          f"r2={point.r2:.4f} h/R={point.height_over_radius:.2f} "
          f"steps={point.steps} wall={point.wall_seconds:.1f}s -> {out}")
    return 0


def _load_points() -> list[dict]:
    if not POINT_DIR.is_dir():
        raise SystemExit(f"no points directory: {POINT_DIR}")
    out = []
    for path in sorted(POINT_DIR.glob("*.json")):
        out.append(json.loads(path.read_text()))
    if not out:
        raise SystemExit(f"no point files in {POINT_DIR}")
    return out


def _as_point(entry: dict) -> CalibrationPoint:
    return CalibrationPoint.from_record(
        entry["mu_r"], entry["mu_s"], entry["record"], entry["wall_seconds"])


def _same_condition(a: dict, b: dict, *, keys=("n_particles", "seed",
                                               "pour_rate_grains_s", "drop_height_m",
                                               "settle_time_s", "dt_s")) -> bool:
    return all(a.get(k) == b.get(k) for k in keys)


def cmd_aggregate(args: argparse.Namespace) -> int:
    entries = _load_points()
    ref_condition = REFERENCE.as_record()
    # ``dt_s=None`` in the condition means "the protocol default step", which the
    # config resolves deterministically -- compare against the resolved value.
    ref_condition["dt_s"] = REFERENCE.config(0.0).dt_effective
    reference_entries = [e for e in entries
                         if _same_condition(_condition_of(e), ref_condition)]
    off_reference = [e for e in entries
                     if not _same_condition(_condition_of(e), ref_condition)]
    if len(reference_entries) < 3:
        raise SystemExit("need >=3 points on the reference condition")

    all_points = [_as_point(e) for e in reference_entries]
    all_points.sort(key=lambda p: p.mu_r)
    points = [p for p in all_points if p.r2 >= MIN_VALID_R2
              and p.height_over_radius >= MIN_VALID_LAYERS]
    rejected = [p.as_row() for p in all_points if p not in points]
    if len(points) < 3:
        raise SystemExit("need >=3 measurement-valid points on the reference condition")
    fit = fit_saturation(points)

    ref = parse_wk2012_reference(".")
    ref_rec = ref.as_record()

    # -- literature comparison: measured angle at the two reference mu_r -----
    def measured_at(mu_r: float) -> dict:
        exact = [p for p in points if abs(p.mu_r - mu_r) < 1e-12]
        if exact:
            p = exact[0]
            return {"mu_r": mu_r, "theta_deg": p.theta_deg,
                    "theta_stderr_deg": p.theta_stderr_deg,
                    "r2": p.r2, "source": "measured", "height_over_radius": p.height_over_radius}
        pred = float(fit(mu_r))
        return {"mu_r": mu_r, "theta_deg": pred, "source": "saturation_fit",
                "height_over_radius": float("nan")}

    lit_rows = []
    for mu_r, theta_ref in zip(ref.mu_r, ref.theta_deg):
        got = measured_at(mu_r)
        rel = abs(got["theta_deg"] - theta_ref) / theta_ref if theta_ref else float("inf")
        lit_rows.append({
            "mu_r": mu_r,
            "theta_ref_deg": theta_ref,
            "theta_got_deg": got["theta_deg"],
            "got_source": got["source"],
            "theta_got_stderr_deg": got.get("theta_stderr_deg"),
            "height_over_radius": got["height_over_radius"],
            "rel_error": rel,
            "within_gate": bool(rel <= GATE_REL_ERROR),
        })

    # -- sensitivity over the literature interval ---------------------------
    lit_sens = ref.sensitivity_deg_per_unit_mu_r
    got_lo, got_hi = measured_at(ref.mu_r[0])["theta_deg"], measured_at(ref.mu_r[1])["theta_deg"]
    got_sens = (got_hi - got_lo) / (ref.mu_r[1] - ref.mu_r[0])
    sens_rel = (abs(got_sens - lit_sens) / lit_sens) if lit_sens else float("nan")

    # -- monotonicity of the measured surface -------------------------------
    thetas = np.array([p.theta_deg for p in points])
    mu_rs = np.array([p.mu_r for p in points])
    diffs = np.diff(thetas)
    monotone = bool(np.all(diffs >= -1e-9))
    # rank correlation (Spearman) without scipy.stats dependency on ordering
    spearman = float(np.corrcoef(
        np.argsort(np.argsort(mu_rs)), np.argsort(np.argsort(thetas)))[0, 1])

    # -- out-of-sample transfer --------------------------------------------
    holdout = [_as_point(e) for e in off_reference]
    transfer = transfer_error(points, holdout) if holdout else None

    # -- protocol spread: same mu_r, different condition --------------------
    spread_rows = []
    for target in points:
        same = [p for p in holdout if abs(p.mu_r - target.mu_r) < 1e-12]
        if not same:
            continue
        vals = [target.theta_deg] + [p.theta_deg for p in same]
        spread_rows.append({
            "mu_r": target.mu_r,
            "theta_deg": vals,
            "spread_deg": float(max(vals) - min(vals)),
            "mean_deg": float(sum(vals) / len(vals)),
            "spread_rel": float((max(vals) - min(vals)) / (sum(vals) / len(vals))),
            "conditions": [target.condition] + [p.condition for p in same],
        })

    # -- inversion for the literature band ---------------------------------
    inversions = []
    for mu_r, theta_ref in zip(ref.mu_r, ref.theta_deg):
        try:
            inv = invert_target(points, theta_ref)
            inv["reference_theta_deg"] = theta_ref
            inv["mu_r_error"] = inv["mu_r_star"] - mu_r
            inversions.append(inv)
        except UnattainableTarget as exc:
            inversions.append({
                "reference_theta_deg": theta_ref,
                "reference_mu_r": mu_r,
                "attainable_deg": list(exc.attainable),
                "status": "unattainable",
            })

    checks = {
        "C1_monotone_increase": monotone,
        "C2_literature_points_within_gate": all(r["within_gate"] for r in lit_rows),
        "C3_sensitivity_within_gate": bool(abs(sens_rel) <= GATE_REL_ERROR),
        "C4_invertible_at_literature_angles": all(
            inv.get("status") != "unattainable" for inv in inversions),
        "C5_transfer_within_gate": bool(
            transfer is not None and transfer["worst_rel_error"] <= GATE_REL_ERROR),
    }

    result = {
        "anchor": "XVAL-DEM-ROLLING-FRICTION-AS-SHAPE-SURROGATE",
        "question": ("does a sphere carrying a calibrated rolling-friction "
                     "coefficient reproduce the literature repose-vs-mu_r "
                     "correspondence (shape surrogate)?"),
        "gate_rel_error": GATE_REL_ERROR,
        "reference_condition": ref_condition,
        "beta_mu_s": PRIORS["mu_s"],
        "literature": ref_rec,
        "validity_rule": {"min_r2": MIN_VALID_R2,
                          "min_height_over_radius": MIN_VALID_LAYERS,
                          "note": "points outside the rule are reported in "
                                  "rejected_surface_points and excluded from "
                                  "the fit and the inversions"},
        "surface": [p.as_row() for p in all_points],
        "valid_surface": [p.as_row() for p in points],
        "rejected_surface_points": rejected,
        "saturation_fit": fit.as_record(),
        "literature_comparison": lit_rows,
        "sensitivity": {
            "literature_deg_per_unit_mu_r": lit_sens,
            "measured_deg_per_unit_mu_r": got_sens,
            "rel_error": sens_rel,
        },
        "monotonicity": {"monotone_nondecreasing": monotone,
                         "spearman_rank": spearman},
        "inversions": inversions,
        "off_reference_points": [p.as_row() for p in holdout],
        "transfer": transfer,
        "protocol_spread": spread_rows,
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "provenance": {
            "script": "benchmarks/xval_rolling_shape_surrogate/run_sweep.py",
            "git_rev": _git_rev(),
            "literature_doc_sha256": ref.source_sha256,
            "point_files": {p.name: _sha256(p) for p in sorted(POINT_DIR.glob("*.json"))},
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "result.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=False))

    print(json.dumps({k: result[k] for k in
                      ("checks", "all_checks_pass", "literature_comparison",
                       "sensitivity", "monotonicity")}, indent=1))
    print(f"-> {out}")
    return 0 if result["all_checks_pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("point", help="run one calibration point")
    p.add_argument("--mu-r", type=float, required=True)
    p.add_argument("--mu-s", type=float, default=PRIORS["mu_s"])
    p.add_argument("--n-particles", type=int, default=REFERENCE.n_particles)
    p.add_argument("--seed", type=int, default=REFERENCE.seed)
    p.add_argument("--pour-rate", type=float, default=REFERENCE.pour_rate)
    p.add_argument("--drop-height", type=float, default=REFERENCE.drop_height)
    p.add_argument("--settle", type=float, default=REFERENCE.settle_time)
    p.add_argument("--dt-half", action="store_true")
    p.add_argument("--tag", required=True)
    p.set_defaults(func=cmd_point)

    a = sub.add_parser("aggregate", help="build surface, invert, score transfer")
    a.set_defaults(func=cmd_aggregate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())