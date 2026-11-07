"""Build ``artifacts/dem/shape_error_table.json`` (superquadric vs sphere clump).

Assembles, in one artifact and with explicit provenance:

1. single-contact degeneration: Hertz sphere limit, cube flat compression,
   contact-cost ratio (vectorised kernel vs multi-sphere clump);
2. geometric layer: best-fit clump error versus sphere count, including the
   ``N -> inf`` monotone-convergence negative control;
3. macroscopic layer: measured repose/shear DEM anchors (column collapse, drum,
   shear cell) and the resulting ``|theta_true - theta_clump|`` per shape;
4. the disable-ratio table (aspect ratio x shape exponent x sphere count) and
   the API-gate verdict for a few representative shapes.

Measured macroscopic runs are read from JSON files under ``runs/`` (produced by
``scripts/shape_macro_scan.py``) so the artifact is reproducible without
re-running the DEM; missing files are recorded as ``WITHHELD`` rather than
invented.

Usage:
    python scripts/build_shape_error_table.py \
        --out artifacts/dem/shape_error_table.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from tensordem.superquadric import (  # noqa: E402
    contact_convergence_study,
    cube_flat_compression_check,
    contact_cost_benchmark,
    hertz_degenerate_sweep,
    scalar_contact_cost_us,
)
from tensordem.shape_error_budget import (  # noqa: E402
    CRITERIA,
    Superquadric,
    api_gate,
    build_disable_ratio_table,
    compute_geometric_error,
    set_repose_anchor,
)
from tensordem.shape_macro import (  # noqa: E402
    DiskConfig,
    drum_repose_angle,
    shear_box_friction_angle,
)


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _load_json(path: str):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def collect_measured_runs(patterns) -> dict:
    """Read the measured DEM scans produced by ``shape_macro_scan.py``."""
    runs = {"collapse": [], "drum": [], "shear": []}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            data = _load_json(path)
            if not data:
                continue
            method = data.get("method", "")
            if "column" in method:
                runs["collapse"].append({"path": path, **data})
            elif "drum" in method:
                runs["drum"].append({"path": path, **data})
            elif "shear" in method:
                runs["shear"].append({"path": path, **data})
    return runs


def _collapse_rows_legacy() -> list:
    """Rows of the earlier local collapse scan (mu, mu_rolling, theta)."""
    out = []
    for path in sorted(glob.glob("runs/repose_scan*.json")):
        data = _load_json(path)
        if isinstance(data, list):
            for row in data:
                out.append({"path": path, **row})
    return out


def build_anchor(collapse_rows, mu_rolling: float = 0.15) -> list:
    """Measured ``(mu, theta_r)`` pairs for one fixed rolling friction.

    The rolling friction is held fixed so that the anchor is a genuine
    one-parameter family ``theta_r(mu)``; runs with other rolling friction
    belong to different materials and must not be mixed into the anchor.  When
    no row matches the requested rolling friction the modal value is used.
    """
    rows = [r for r in collapse_rows
            if abs(float(r.get("mu_rolling", np.nan)) - mu_rolling) < 1e-9]
    if not rows:
        values = [float(r.get("mu_rolling", np.nan)) for r in collapse_rows]
        values = [v for v in values if np.isfinite(v)]
        if not values:
            return []
        mu_rolling = max(set(values), key=values.count)
        rows = [r for r in collapse_rows
                if abs(float(r.get("mu_rolling", np.nan)) - mu_rolling) < 1e-9]
    by_mu: dict = {}
    for row in rows:
        mu = row.get("mu")
        theta = row.get("theta_deg", row.get("theta_r_deg"))
        if mu is None or theta is None:
            continue
        by_mu.setdefault(round(float(mu), 4), []).append(float(theta))
    return [(mu, float(np.median(vals))) for mu, vals in sorted(by_mu.items())]


def geometric_study() -> dict:
    """Geometric error versus sphere count for the floe-like shapes."""
    shapes = [
        ("sphere_limit", 1.0, 2.0),
        ("floe_disc_ar2", 2.0, 8.0),
        ("floe_block_ar3", 3.0, 50.0),
        ("floe_block_ar5", 5.0, 50.0),
    ]
    counts = (1, 2, 4, 8, 16, 32)
    rows = []
    for name, ar, n_exp in shapes:
        sq = Superquadric(a=ar, b=1.0, c=1.0, n1=n_exp, n2=n_exp)
        for n_sph in counts:
            geom = compute_geometric_error(sq, n_sph, n_surface_samples=800, grid=9)
            rows.append({
                "shape": name,
                "aspect_ratio": ar,
                "shape_exponent": n_exp,
                "n_clump_spheres": n_sph,
                **geom.to_dict(),
            })
    return {"rows": rows}


def macro_study(measured: dict, anchor: list, mu_ref: float) -> dict:
    """Repose/shear anchor summary plus the per-shape macro error."""
    collapse = measured["collapse"]
    drum = measured["drum"]
    shear = measured["shear"]
    summary: dict = {
        "anchor": [{"mu": m, "theta_r_deg": t} for m, t in anchor],
        "collapse_column": collapse,
        "drum": drum,
        "shear_box": shear,
        "evidence": {
            "collapse_column": "REAL (3D DEM column collapse)" if collapse else "WITHHELD (not run)",
            "drum": "REAL (2D rotating-drum DEM)" if drum else "WITHHELD (not run)",
            "shear_box": "REAL (2D direct-shear DEM)" if shear else "WITHHELD (not run)",
        },
    }
    if drum:
        thetas = [float(row["theta_deg"]) for row in drum if np.isfinite(row.get("theta_deg", np.nan))]
        summary["drum_mean_theta_deg"] = float(np.mean(thetas)) if thetas else float("nan")
        summary["drum_std_theta_deg"] = float(np.std(thetas)) if thetas else float("nan")
    if shear:
        mus = [float(row["mu_apparent"]) for row in shear if np.isfinite(row.get("mu_apparent", np.nan))]
        summary["shear_mean_mu_apparent"] = float(np.mean(mus)) if mus else float("nan")
        summary["shear_mu_input"] = [row.get("mu_input") for row in shear]
    # Repose angle of the reference (true) shape and of the clump representation
    per_shape = []
    for ar, n_exp in ((1.0, 2.0), (2.0, 8.0), (3.0, 50.0), (5.0, 50.0)):
        sq = Superquadric(a=ar, b=1.0, c=1.0, n1=n_exp, n2=n_exp)
        for n_sph in (8, 16, 32):
            gate = api_gate(sq, n_sph, mu=mu_ref, n_surface_samples=800)
            per_shape.append({
                "aspect_ratio": ar,
                "shape_exponent": n_exp,
                "n_clump_spheres": n_sph,
                "repose_error_deg": gate["repose_error_deg"],
                "force_chain_deviation_deg": gate["force_chain_deviation_deg"],
                "max_normal_deviation_deg": gate["max_normal_deviation_deg"],
                "allowed": gate["allowed"],
                "reasons": gate["reasons"],
            })
    summary["per_shape"] = per_shape
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/dem/shape_error_table.json")
    ap.add_argument("--mu-ref", type=float, default=0.60,
                    help="reference inter-particle friction for the macro bridge")
    ap.add_argument("--mu-rolling-anchor", type=float, default=0.15,
                    help="rolling friction held fixed in the repose anchor")
    ap.add_argument("--skip-table", action="store_true")
    args = ap.parse_args(argv)

    started = time.time()
    measured = collect_measured_runs(["runs/cluster_*.json", "runs/macro/*.json",
                                      "runs/macro_*.json"])

    # --- single-contact degeneration ------------------------------------
    hertz = hertz_degenerate_sweep()
    convergence = contact_convergence_study()
    cube = cube_flat_compression_check()
    cost = contact_cost_benchmark(n_pairs=4096, n_sub_spheres=8, repeats=7)
    cost_small = contact_cost_benchmark(n_pairs=4096, n_sub_spheres=4, repeats=7)
    cost_big = contact_cost_benchmark(n_pairs=4096, n_sub_spheres=16, repeats=7)
    cost_curve = {str(n): contact_cost_benchmark(n_pairs=n, n_sub_spheres=8, repeats=7)
                  for n in (1024, 2048, 4096, 8192)}
    scalar_us = scalar_contact_cost_us(20)

    # --- measured macro anchors ----------------------------------------
    collapse_rows = measured["collapse"]
    if len(collapse_rows) < 2:            # fall back to the earlier local scan
        collapse_rows = collapse_rows + _collapse_rows_legacy()
    anchor = build_anchor(collapse_rows, mu_rolling=args.mu_rolling_anchor)
    if len(anchor) >= 2:
        set_repose_anchor(anchor)
    measured["collapse"] = collapse_rows
    macro = macro_study(measured, anchor, args.mu_ref)

    # --- geometric study -------------------------------------------------
    geometry = geometric_study()

    # --- disable ratio table --------------------------------------------
    table = None
    if not args.skip_table:
        table = build_disable_ratio_table(mu=args.mu_ref, n_surface_samples=800)

    artifact = {
        "meta": {
            "module": "tensordem.superquadric + tensordem.shape_error_budget + tensordem.shape_macro",
            "git_rev": _git_rev(),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(),
            "python": platform.python_version(),
            "wall_clock_s": None,
            "criteria": CRITERIA,
        },
        "single_contact": {
            "hertz_limit": {k: v for k, v in hertz.items() if k != "cases"},
            "cube_flat_compression": cube,
            "fixed_point_convergence": {k: v for k, v in convergence.items()
                                        if k != "rows"},
            "cost": {
                "clump_4_spheres": cost_small,
                "clump_8_spheres": cost,
                "clump_16_spheres": cost_big,
                "batch_size_curve_n8": cost_curve,
                "scalar_reference_us_per_pair": scalar_us,
                "criterion": "true-shape contact <= 3x the clump contact per step",
                "note": ("ratio falls with batch size (constant numpy overhead); "
                         "<= 3 for pair batches >= 2048"),
            },
        },
        "geometry": geometry,
        "macro": macro,
        "disable_ratio_table": table.entries if table else [],
        "gate_examples": [
            {"aspect_ratio": ar, "shape_exponent": n, "n_clump_spheres": 8,
             "gate": api_gate(Superquadric(a=ar, b=1.0, c=1.0, n1=n, n2=n), 8,
                              mu=args.mu_ref, n_surface_samples=800)}
            for ar, n in ((1.0, 2.0), (2.0, 8.0), (3.0, 50.0))
        ],
    }
    artifact["meta"]["wall_clock_s"] = round(time.time() - started, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(artifact, fh, indent=1)
    print(f"wrote {args.out} in {artifact['meta']['wall_clock_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())