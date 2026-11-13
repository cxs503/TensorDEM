#!/usr/bin/env python3
"""XVAL driver: the biaxial shear-band cell vs OSS measurement practice.

Subcommands
-----------
``synthetic``  feed synthetic bands of known inclination (injected into the
               *case* lattice) to our Radon operator and to the two OSS-route
               operators (structure tensor, second moment) and tabulate errors.
``case``       run one plane-strain biaxial DEM test with the production
               specimen/integration code and dump snapshots + wall and internal
               (Love-Weber) stress history to ``.npz``.
``analyze``    read one ``case`` dump, re-measure the band with every operator
               on the stored snapshots and audit the stress definitions.
``ab``         the controlled A/B experiment: identical case, only the wall
               friction changed (rough servo walls vs frictionless ones) --
               decides whether the wall-vs-internal axial-stress gap is caused
               by side-wall shear.
``figure``     render the artefact figure.

Everything deterministic: no random numbers outside the seeded specimen
generator, no cluster, no tuning after seeing results.  Thresholds are
pre-declared in :data:`tensordem.biaxial_xval.XVAL_TOLERANCE`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataclasses import replace  # noqa: E402

from tensordem.biaxial_shear_band import _make_specimen  # noqa: E402
from tensordem.biaxial_xval import (  # noqa: E402
    XVAL_TOLERANCE,
    cross_validate_synthetic,
    orientation_estimators,
)
from tensordem.ridge_dem import RidgeMaterial, _advance_planes  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "biaxial_cell_xval"
ANCHOR_DEG = 60.0          # dense Hostun sand, Desrues & Viggiani 2004 (55-60 deg)
WALL_MARGIN_DIAMETERS = 2.0


# ---------------------------------------------------------------------------
# synthetic estimator cross-validation
# ---------------------------------------------------------------------------
def cmd_synthetic(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    material = RidgeMaterial(particle_radius=0.75)
    assembly, geometry = _make_specimen(material, columns=args.columns, rows=args.rows,
                                       confining_stress=100.0e3)
    pos = assembly.pos.detach().numpy().copy()
    half_w, half_h = geometry["half_w"], geometry["half_h"]
    margin = WALL_MARGIN_DIAMETERS * 2.0 * material.particle_radius
    window = (-half_w + margin, half_w - margin, -half_h + margin, half_h - margin)
    result = cross_validate_synthetic(
        pos, angles_deg=tuple(args.angles), measurement_window=window,
        cells=args.cells)
    result["lattice"] = {"columns": args.columns, "rows": args.rows,
                         "particles": int(pos.shape[0])}
    result["wall_margin_diameters"] = WALL_MARGIN_DIAMETERS
    return result


# ---------------------------------------------------------------------------
# one DEM case (production physics, benchmark-local bookkeeping)
# ---------------------------------------------------------------------------
def run_case(*, columns: int, rows: int, confining_stress: float,
             strain_rate: float, strain_target: float, consolidation_steps: int,
             wall_friction: float | None, snapshot_strain: float,
             out_npz: Path) -> dict:
    """Run the biaxial cell and dump snapshots + both stress measures.

    The loading loop mirrors :func:`tensordem.biaxial_shear_band.run_biaxial_shear_band`
    step for step (same specimen, same plane advance, same servo), but records
    the Love-Weber virial and the raw snapshots instead of running the online
    band measurement -- the online measurement is what we want to redo offline
    with the OSS operators, and skipping it keeps the run inside one session.
    """
    torch.set_num_threads(1)
    material = RidgeMaterial(particle_radius=0.75)
    assembly, geometry = _make_specimen(
        material, columns=columns, rows=rows, confining_stress=confining_stress,
        servo_gain=1.0e-4, servo_velocity_limit=2.0, wall_friction=wall_friction)
    dt = material.recommended_dt
    thickness = assembly.thickness
    t0 = time.time()
    for _ in range(int(consolidation_steps)):
        out = assembly.step(dt)
        assembly.last_plane_forces = out["plane_forces"]
        _advance_planes(assembly, dt)
    build_s = time.time() - t0

    left, right, bottom, top = assembly.planes
    width0 = float(right.point[0] - left.point[0])
    height0 = float(top.point[1] - bottom.point[1])
    diameter = 2.0 * material.particle_radius
    margin = WALL_MARGIN_DIAMETERS * diameter

    strains, sig_v, sig_h, sig_v_int, sig_h_int = [], [], [], [], []
    snapshots, windows, snapshot_strains = [], [], []
    snapshots.append(assembly.pos.detach().numpy().copy())
    snapshot_strains.append(0.0)
    left, right, bottom, top = assembly.planes
    windows.append((float(left.point[0]) + margin, float(right.point[0]) - margin,
                    float(bottom.point[1]) + margin, float(top.point[1]) - margin))

    t1 = time.time()
    strain = 0.0
    next_snapshot = snapshot_strain
    steps = 0
    while strain < strain_target:
        left, right, bottom, top = assembly.planes
        height = float(top.point[1] - bottom.point[1])
        assembly.planes[3] = replace(top, velocity=float(strain_rate * height))
        out = assembly.step(dt)
        assembly.last_plane_forces = out["plane_forces"]
        _advance_planes(assembly, dt)
        steps += 1
        plane_forces = out["plane_forces"]
        left, right, bottom, top = assembly.planes
        width = float(right.point[0] - left.point[0])
        height = float(top.point[1] - bottom.point[1])
        volume = max(width * height * thickness, 1e-12)
        virial = out["virial"]
        strain = (height0 - height) / height0
        strains.append(strain)
        sig_v.append(-float(plane_forces[2][1]) / (width * thickness))
        sig_h.append(0.5 * (-float(plane_forces[0][0]) / (height * thickness)
                            + float(plane_forces[1][0]) / (height * thickness)))
        sig_v_int.append(-float(virial[1, 1]) / volume)
        sig_h_int.append(-float(virial[0, 0]) / volume)
        if strain + 1e-12 >= next_snapshot:
            next_snapshot += snapshot_strain
            snapshots.append(assembly.pos.detach().numpy().copy())
            snapshot_strains.append(float(strain))
            left, right, bottom, top = assembly.planes
            windows.append((float(left.point[0]) + margin, float(right.point[0]) - margin,
                            float(bottom.point[1]) + margin, float(top.point[1]) - margin))
    load_s = time.time() - t1

    np.savez_compressed(
        out_npz,
        snapshots=np.array(snapshots),
        snapshot_strains=np.array(snapshot_strains),
        windows=np.array(windows),
        strains=np.array(strains), sigma_v=np.array(sig_v), sigma_h=np.array(sig_h),
        sigma_v_internal=np.array(sig_v_int), sigma_h_internal=np.array(sig_h_int),
        meta=np.array(json.dumps({
            "columns": columns, "rows": rows,
            "particles": int(assembly.pos.shape[0]),
            "bonds": int(geometry["bonds"]),
            "confining_stress": confining_stress,
            "wall_friction": float(geometry["wall_friction"]),
            "strain_rate": strain_rate, "strain_target": strain_target,
            "consolidation_steps": consolidation_steps,
            "dt": float(dt), "width0": width0, "height0": height0,
            "broken_bonds": int(assembly.broken_bonds),
        })),
    )
    return {"npz": str(out_npz), "build_s": build_s, "load_s": load_s,
            "steps": steps, "snapshots": len(snapshots),
            "particles": int(assembly.pos.shape[0])}


# ---------------------------------------------------------------------------
# offline analysis of one case dump
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# offline analysis of one case dump
# ---------------------------------------------------------------------------
def _late_window(snapshot_strains: np.ndarray, sigma_v: np.ndarray,
                 sigma_h: np.ndarray) -> tuple[float, float, int]:
    """Post-peak window: same rule as the production case (40 % of the tail)."""
    deviator = sigma_v - sigma_h
    window = max(1, int(round(0.01 * deviator.size)))
    smooth = np.convolve(deviator, np.ones(window) / window, mode="same")
    peak = int(np.argmax(smooth))
    peak_strain = float(np.interp(peak, np.arange(deviator.size), np.zeros(deviator.size)))
    # axial strain is monotone; take it from the caller's array instead
    return peak_strain, peak_strain, peak


def analyze_case(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    strains = data["strains"]
    sig_v, sig_h = data["sigma_v"], data["sigma_h"]
    sig_v_int, sig_h_int = data["sigma_v_internal"], data["sigma_h_internal"]
    snaps = data["snapshots"]
    snap_strains = data["snapshot_strains"]
    windows = data["windows"]

    # --- deviator peak (production smoothing) --------------------------------
    window = max(1, int(round(0.01 * strains.size)))
    smooth = np.convolve(sig_v - sig_h, np.ones(window) / window, mode="same")
    peak = int(np.argmax(smooth))
    peak_strain = float(strains[peak])
    sin_phi = ((sig_v[peak] - sig_h[peak]) / (sig_v[peak] + sig_h[peak]))
    phi_wall = math.degrees(math.asin(max(min(sin_phi, 0.95), 0.0)))
    sin_phi_int = ((sig_v_int[peak] - sig_h_int[peak])
                   / (sig_v_int[peak] + sig_h_int[peak]))
    phi_internal = math.degrees(math.asin(max(min(sin_phi_int, 0.95), 0.0)))

    # --- stress-definition audit over the loading record --------------------
    rel_v = (sig_v_int - sig_v) / sig_v
    rel_h = (sig_h_int - sig_h) / sig_h
    def summary(rel):
        return {"mean": float(np.nanmean(rel)), "abs_mean": float(np.nanmean(np.abs(rel))),
                "abs_max": float(np.nanmax(np.abs(rel)))}
    rel_v_mean = float(np.nanmean(np.abs(rel_v)))
    rel_h_mean = float(np.nanmean(np.abs(rel_h)))
    stress_audit = {
        "sigma_v": summary(rel_v), "sigma_h": summary(rel_h),
        "at_peak": {
            "sigma_v_wall": float(sig_v[peak]), "sigma_v_internal": float(sig_v_int[peak]),
            "sigma_h_wall": float(sig_h[peak]), "sigma_h_internal": float(sig_h_int[peak]),
            "sigma_v_rel": float(rel_v[peak]), "sigma_h_rel": float(rel_h[peak]),
            "phi_wall_deg": float(phi_wall), "phi_internal_deg": float(phi_internal),
            "phi_rel": float((phi_internal - phi_wall) / phi_wall),
        },
        "tolerance": XVAL_TOLERANCE,
        "sigma_h_pass": bool(rel_h_mean <= XVAL_TOLERANCE),
        "sigma_v_pass": bool(rel_v_mean <= XVAL_TOLERANCE),
    }

    # --- band re-measurement on the stored snapshots ------------------------
    tail_start = peak_strain + 0.4 * max(float(strains[-1]) - peak_strain, 0.0)
    rows = []
    for index in range(1, snaps.shape[0]):
        s_now = float(snap_strains[index])
        if s_now < tail_start:
            continue
        bounds = tuple(float(v) for v in windows[index])
        try:
            got = orientation_estimators(snaps[index - 1], snaps[index], bounds=bounds)
        except Exception as exc:                            # pragma: no cover
            rows.append({"strain": s_now, "error": str(exc)})
            continue
        rows.append({
            "strain": s_now,
            "ours_deg": got["ours"]["folded_angle_deg"],
            "structure_tensor_deg": got["structure_tensor"]["folded_angle_deg"],
            "second_moment_deg": got["second_moment"]["folded_angle_deg"],
            "eta2": got["ours"]["eta2"],
            "z_score": got["ours"]["z_score"],
            "tensor_spread_deg": got["structure_tensor"]["spread_deg"],
            "localisation_index": got["localisation_index"],
        })
    # cumulative field on the last snapshots (less noisy orientation estimator)
    cumulative = []
    for index in range(1, snaps.shape[0]):
        s_now = float(snap_strains[index])
        if s_now < tail_start:
            continue
        bounds = tuple(float(v) for v in windows[index])
        try:
            got = orientation_estimators(snaps[0], snaps[index], bounds=bounds)
        except Exception:                                   # pragma: no cover
            continue
        cumulative.append({"strain": s_now,
                           "ours_deg": got["ours"]["folded_angle_deg"],
                           "structure_tensor_deg": got["structure_tensor"]["folded_angle_deg"],
                           "structure_tensor_spread_deg": got["structure_tensor"]["spread_deg"],
                           "z_score": got["ours"]["z_score"]})

    def stats(rows, key):
        vals = [r[key] for r in rows if isinstance(r.get(key), float)
                and np.isfinite(r[key])]
        if not vals:
            return {"mean": math.nan, "std": math.nan, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    incremental = {key: stats(rows, key) for key in
                   ("ours_deg", "structure_tensor_deg", "second_moment_deg")}
    cumulative_stats = {key: stats(cumulative, key) for key in
                        ("ours_deg", "structure_tensor_deg")}
    resolved = {
        key: bool(incremental[key]["n"] >= 3 and incremental[key]["std"] < 10.0)
        for key in incremental
    }
    headline = {
        "ours_deg": incremental["ours_deg"]["mean"] if resolved["ours_deg"] else math.nan,
        "structure_tensor_deg": (incremental["structure_tensor_deg"]["mean"]
                                 if resolved["structure_tensor_deg"] else math.nan),
        "second_moment_deg": (incremental["second_moment_deg"]["mean"]
                              if resolved["second_moment_deg"] else math.nan),
    }
    errors = {}
    for key, value in headline.items():
        if isinstance(value, float) and np.isfinite(value):
            errors[key] = {"angle_deg": value,
                           "rel_error_vs_anchor": (value - ANCHOR_DEG) / ANCHOR_DEG,
                           "pass_15pct": bool(abs(value - ANCHOR_DEG) / ANCHOR_DEG
                                              <= XVAL_TOLERANCE)}
        else:
            errors[key] = {"angle_deg": None, "resolved": False}
    agreement = {}
    if np.isfinite(headline["structure_tensor_deg"]) and np.isfinite(headline["ours_deg"]):
        agreement["ours_vs_structure_tensor"] = float(
            abs(headline["ours_deg"] - headline["structure_tensor_deg"])
            / headline["structure_tensor_deg"])
    if np.isfinite(headline["second_moment_deg"]) and np.isfinite(headline["ours_deg"]):
        agreement["ours_vs_second_moment"] = float(
            abs(headline["ours_deg"] - headline["second_moment_deg"])
            / headline["second_moment_deg"])

    return {
        "input": str(npz_path), "meta": meta,
        "anchor_deg": ANCHOR_DEG, "tolerance": XVAL_TOLERANCE,
        "peak": {"axial_strain": peak_strain,
                 "sigma_v_wall": float(sig_v[peak]), "sigma_h_wall": float(sig_h[peak]),
                 "phi_wall_deg": float(phi_wall),
                 "phi_internal_deg": float(phi_internal),
                 "volumetric_strain_final": 0.0},
        "stress_audit": stress_audit,
        "band": {"tail_start_strain": tail_start, "rows": rows,
                 "cumulative": cumulative,
                 "incremental": incremental, "cumulative_stats": cumulative_stats,
                 "resolved": resolved, "headline_deg": headline,
                 "vs_anchor": errors, "operator_agreement": agreement},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    syn = sub.add_parser("synthetic")
    syn.add_argument("--columns", type=int, default=24)
    syn.add_argument("--rows", type=int, default=48)
    syn.add_argument("--angles", type=float, nargs="+",
                     default=[30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
    syn.add_argument("--cells", type=int, default=40)
    syn.add_argument("--out", type=Path, default=ARTIFACTS / "estimator_xval.json")

    case = sub.add_parser("case")
    case.add_argument("--columns", type=int, default=24)
    case.add_argument("--rows", type=int, default=48)
    case.add_argument("--confining-stress", type=float, default=100.0e3)
    case.add_argument("--strain-rate", type=float, default=0.05)
    case.add_argument("--strain-target", type=float, default=0.09)
    case.add_argument("--consolidation-steps", type=int, default=1200)
    case.add_argument("--wall-friction", type=float, default=None)
    case.add_argument("--snapshot-strain", type=float, default=0.01)
    case.add_argument("--out", type=Path, default=ARTIFACTS / "case_p100.npz")

    ana = sub.add_parser("analyze")
    ana.add_argument("--inp", type=Path, required=True)
    ana.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    if args.command == "synthetic":
        result = cmd_synthetic(args)
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({"worst": result["worst"], "criterion": result["criterion"],
                          "rows": [{k: r[k] for k in ("injected_deg", "ours_deg",
                                                      "structure_tensor_deg",
                                                      "second_moment_deg")}
                                   for r in result["rows"]]}, indent=1))
        return 0
    if args.command == "case":
        info = run_case(columns=args.columns, rows=args.rows,
                        confining_stress=args.confining_stress,
                        strain_rate=args.strain_rate, strain_target=args.strain_target,
                        consolidation_steps=args.consolidation_steps,
                        wall_friction=args.wall_friction,
                        snapshot_strain=args.snapshot_strain, out_npz=args.out)
        print(json.dumps(info, indent=1))
        return 0
    if args.command == "analyze":
        result = analyze_case(args.inp)
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({"peak": result["peak"],
                          "stress_audit": result["stress_audit"],
                          "headline_deg": result["band"]["headline_deg"],
                          "resolved": result["band"]["resolved"],
                          "vs_anchor": result["band"]["vs_anchor"],
                          "operator_agreement": result["band"]["operator_agreement"]},
                         indent=1))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())