"""XVAL: TensorDEM's scatter-point Radon band-angle measurement vs the OSS image-domain Radon.

Runs are driven by ``prereg/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.json`` (thresholds,
angles, seeds, pixel sizes frozen before the first production run).

Usage (cluster, sharded)::

    python benchmarks/radon_projection_xval/run_xval.py --shard 0 4
    python benchmarks/radon_projection_xval/run_xval.py --merge

Every row carries the two estimators' angles, the four projection criteria of the
reference and the errors against the *known injected* inclination, so the report
tables can be injected programmatically (no hand transcription).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:                                                             # torch only for kNN strain
    import torch

    torch.set_num_threads(1)
except Exception:                                                # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.biaxial_band_diagnostics import synthetic_band_configuration  # noqa: E402
from tensordem.biaxial_shear_band import local_shear_strain, radon_band_angle  # noqa: E402
from tensordem.radon_reference import (  # noqa: E402
    OSS_SOURCE,
    radon_projection_angle,
)

PREREG = ROOT / "prereg" / "XVAL_DEM_RADON_PROJECTION_MEASUREMENT.json"
OUTDIR = ROOT / "artifacts" / "XVAL-DEM-RADON-PROJECTION-MEASUREMENT"


# ---------------------------------------------------------------------------
# synthetic fields
# ---------------------------------------------------------------------------
def lattice(columns: int, rows: int, *, spacing: float = 1.0, jitter: float = 0.03,
            seed: int = 11) -> np.ndarray:
    """Jittered triangular lattice, identical construction to the biaxial specimen."""
    rng = np.random.default_rng(seed)
    pts = []
    for r in range(rows):
        for c in range(columns):
            pts.append((c * spacing + 0.5 * (r % 2) * spacing,
                        r * spacing * math.sqrt(3.0) / 2.0))
    pts = np.asarray(pts, dtype=np.float64)
    pts += jitter * spacing * rng.standard_normal(pts.shape)
    return pts


def gaussian_ridge(points: np.ndarray, angle_deg: float, sigma: float) -> np.ndarray:
    """Weight field collapsing onto a line of known inclination ``angle_deg``."""
    rad = math.radians(float(angle_deg))
    s = -points[:, 0] * math.sin(rad) + points[:, 1] * math.cos(rad)
    return np.exp(-(s / float(sigma)) ** 2)


def production_field(points: np.ndarray, angle_deg: float, *, spacing: float,
                     band_width: float) -> np.ndarray:
    """Strain field the production pipeline actually feeds to the estimator.

    A finite-width tangential slip band of known inclination is imposed on the
    lattice (:func:`tensordem.biaxial_band_diagnostics.synthetic_band_configuration`),
    and the weights are the local maximum shear strain of that displacement
    (:func:`tensordem.biaxial_shear_band.local_shear_strain`, k = 8), i.e. exactly
    what ``measure_band`` projects.
    """
    height = float(np.ptp(points[:, 1]))
    moved = synthetic_band_configuration(
        points, angle_deg=float(angle_deg), slip=0.04 * height,
        band_width=float(band_width))
    gamma = local_shear_strain(points, moved, k=8)
    return np.clip(gamma, 0.0, None)


# ---------------------------------------------------------------------------
# case matrix
# ---------------------------------------------------------------------------
def build_cases(cfg: dict) -> list[dict]:
    cm = cfg["case_matrix"]
    rs = cfg["reference_settings"]
    res = float(rs["resolution_deg"])
    cases: list[dict] = []

    # (1) controlled ridge, case lattice, all angles x seeds x pixel sizes
    for seed in cm["seeds"]:
        for angle in cm["angles_deg"]:
            for h in cm["primary_pixel_sizes"]:
                cases.append({"group": "primary", "lattice": cm["lattice"],
                              "seed": seed, "angle_true_deg": angle,
                              "field": "gaussian_ridge", "pixel_size": h,
                              "deposit_sigma_spacings": cm["primary_deposit_sigma_spacings"],
                              "resolution_deg": res})
    # (2) square lattice control (isolates lattice aspect as a confounder)
    square = {"columns": cm["lattice"]["columns"], "rows": cm["lattice"]["columns"]}
    for angle in cm["control_angles_deg"]:
        for h in cm["primary_pixel_sizes"]:
            cases.append({"group": "control_square", "lattice": square,
                          "seed": cm["seeds"][0], "angle_true_deg": angle,
                          "field": "gaussian_ridge", "pixel_size": h,
                          "deposit_sigma_spacings": 1.0, "resolution_deg": res})
    # (3) resolution / deposition convergence + aliasing failure mode
    for angle in cm["resolution_angles_deg"]:
        for h in cm["resolution_pixel_sizes"]:
            for dep in cm["deposit_sigma_sweep"]:
                cases.append({"group": "resolution", "lattice": cm["lattice"],
                              "seed": cm["seeds"][0], "angle_true_deg": angle,
                              "field": "gaussian_ridge", "pixel_size": h,
                              "deposit_sigma_spacings": dep,
                              "resolution_deg": float(rs["resolution_deg_coarse"])})
    # (4) angular-resolution sensitivity (ours and the reference)
    for angle in cm["sensitivity_angles_deg"]:
        for ares in cm["angle_resolution_sweep"]:
            cases.append({"group": "angle_resolution", "lattice": cm["lattice"],
                          "seed": cm["seeds"][0], "angle_true_deg": angle,
                          "field": "gaussian_ridge",
                          "pixel_size": cm["primary_pixel_sizes"][0],
                          "deposit_sigma_spacings": 1.0, "resolution_deg": float(ares)})
    # (5) end-to-end: the production strain field of an injected slip band
    for angle in cm["production_angles_deg"]:
        for h in cm["primary_pixel_sizes"]:
            cases.append({"group": "production_field", "lattice": cm["lattice"],
                          "seed": cm["seeds"][0], "angle_true_deg": angle,
                          "field": "production_strain", "pixel_size": h,
                          "deposit_sigma_spacings": 1.0, "resolution_deg": res})
    for i, case in enumerate(cases):
        case["case_id"] = i
    return cases


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------
def run_case(case: dict) -> dict:
    t0 = time.time()
    lat = case["lattice"]
    points = lattice(int(lat["columns"]), int(lat["rows"]), seed=int(case["seed"]))
    # particle spacing of the jittered lattice (median nearest-neighbour distance)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    spacing = float(np.median(d.min(axis=1)))
    angle = float(case["angle_true_deg"])
    h = float(case["pixel_size"])
    dep = float(case["deposit_sigma_spacings"]) * spacing

    if case["field"] == "gaussian_ridge":
        weight = gaussian_ridge(points, angle, sigma=2.0 * spacing)
    elif case["field"] == "production_strain":
        weight = production_field(points, angle, spacing=spacing,
                                  band_width=3.0 * spacing)
    else:                                                    # pragma: no cover
        raise ValueError(case["field"])

    ours = radon_band_angle(points[:, 0], points[:, 1], weight,
                            resolution_deg=float(case["resolution_deg"]))
    ref = radon_projection_angle(points[:, 0], points[:, 1], weight,
                                 pixel_size=h, deposit_sigma=dep,
                                 resolution_deg=float(case["resolution_deg"]))
    ours_angle = float(ours["folded_angle_deg"])
    ref_angle = float(ref["angle_deg"])
    return {
        **{k: case[k] for k in ("group", "seed", "angle_true_deg", "field",
                                "pixel_size", "deposit_sigma_spacings",
                                "resolution_deg", "case_id")},
        "n_particles": int(points.size // 2),
        "particle_spacing": spacing,
        "deposit_sigma": dep,
        "ours_angle_deg": ours_angle,
        "ours_eta2": float(ours["eta2"]),
        "ours_error_deg": ours_angle - angle,
        "ref_angle_deg": ref_angle,
        "ref_error_deg": ref_angle - angle,
        "ref_criteria_angles_deg": ref["criteria_angles_deg"],
        "agreement_deg": abs(ours_angle - ref_angle),
        "rel_ours": abs(ours_angle - angle) / angle,
        "rel_ref": abs(ref_angle - angle) / angle,
        "rel_agreement": abs(ours_angle - ref_angle) / angle,
        "raster_shape": ref["raster_shape"],
        "wall_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args(argv)

    cfg = json.loads(PREREG.read_text())
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        rows: list[dict] = []
        for path in sorted(OUTDIR.glob("shard_*.json")):
            rows.extend(json.loads(path.read_text())["rows"])
        rows.sort(key=lambda r: r["case_id"])
        summary = summarise(rows, cfg)
        (OUTDIR / "results.json").write_text(json.dumps(
            {"prereg": str(PREREG.relative_to(ROOT)),
             "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
             "oss_source": OSS_SOURCE, "rows": rows, "summary": summary},
            indent=1))
        print(json.dumps(summary, indent=1))
        return 0

    cases = build_cases(cfg)
    i, n = args.shard if args.shard else (0, 1)
    mine = [c for c in cases if c["case_id"] % n == i]
    rows = []
    for case in mine:
        rows.append(run_case(case))
        print(json.dumps({k: rows[-1][k] for k in
                          ("case_id", "group", "angle_true_deg", "ours_angle_deg",
                           "ref_angle_deg", "wall_s")}), flush=True)
    (OUTDIR / f"shard_{i}.json").write_text(json.dumps(
        {"shard": [i, n], "rows": rows}, indent=1))
    return 0


def summarise(rows: list[dict], cfg: dict) -> dict:
    tol = float(cfg["acceptance"]["relative_error_max"])
    target = float(cfg["acceptance"]["relative_error_target"])

    def stats(sel: list[dict]) -> dict:
        if not sel:
            return {"n": 0}
        e_ours = [abs(r["ours_error_deg"]) for r in sel]
        e_ref = [abs(r["ref_error_deg"]) for r in sel]
        rel_o = [r["rel_ours"] for r in sel]
        rel_r = [r["rel_ref"] for r in sel]
        rel_a = [r["rel_agreement"] for r in sel]
        return {
            "n": len(sel),
            "ours_max_abs_deg": max(e_ours), "ours_mean_abs_deg": float(np.mean(e_ours)),
            "ref_max_abs_deg": max(e_ref), "ref_mean_abs_deg": float(np.mean(e_ref)),
            "agreement_max_deg": max(r["agreement_deg"] for r in sel),
            "agreement_mean_deg": float(np.mean([r["agreement_deg"] for r in sel])),
            "ours_max_rel": max(rel_o), "ref_max_rel": max(rel_r),
            "agreement_max_rel": max(rel_a),
            "ours_within_tol": all(v < tol for v in rel_o),
            "ref_within_tol": all(v < tol for v in rel_r),
            "agreement_within_tol": all(v < tol for v in rel_a),
            "ours_within_target": all(v < target for v in rel_o),
            "ref_within_target": all(v < target for v in rel_r),
            "agreement_within_target": all(v < target for v in rel_a),
        }

    groups = sorted({r["group"] for r in rows})
    per_group = {g: stats([r for r in rows if r["group"] == g]) for g in groups}

    primary = [r for r in rows if r["group"] == "primary"]
    by_pixel: dict[str, dict] = {}
    for h in sorted({r["pixel_size"] for r in primary}):
        by_pixel[str(h)] = stats([r for r in primary if r["pixel_size"] == h])
    by_angle: dict[str, dict] = {}
    for a in sorted({r["angle_true_deg"] for r in primary}):
        by_angle[str(a)] = stats([r for r in primary if r["angle_true_deg"] == a])

    resolution = [r for r in rows if r["group"] == "resolution"]
    grid: dict[str, dict[str, dict]] = {}
    for r in resolution:
        key = f"h={r['pixel_size']}|deposit={r['deposit_sigma_spacings']}"
        grid.setdefault(str(r["angle_true_deg"]), {})[key] = {
            "ref_angle_deg": r["ref_angle_deg"], "ref_error_deg": r["ref_error_deg"],
            "ours_angle_deg": r["ours_angle_deg"]}

    criteria: dict[str, dict] = {}
    for name in ("energy", "entropy", "peak", "variance"):
        errs = [abs(r["ref_criteria_angles_deg"][name] - r["angle_true_deg"])
                for r in primary]
        criteria[name] = {"max_abs_deg": max(errs), "mean_abs_deg": float(np.mean(errs)),
                          "max_rel": max(errs) / float(np.min([r["angle_true_deg"] for r in primary]))}

    overall = stats(rows)
    return {
        "acceptance": {"relative_error_max": tol, "relative_error_target": target},
        "overall": overall,
        "per_group": per_group,
        "primary_by_pixel_size": by_pixel,
        "primary_by_angle": by_angle,
        "criteria_primary": criteria,
        "resolution_grid": grid,
        "verdict": {
            "xval_agreement_within_tol": overall["agreement_within_tol"],
            "ours_vs_truth_within_tol": overall["ours_within_tol"],
            "ref_vs_truth_within_tol": overall["ref_within_tol"],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())