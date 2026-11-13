"""CASE-PENETRATION sweep runner: one case per shard, JSON records out.

The m15404 case (``docs/CASE_PENETRATION_QUASI_STATIC.md``) recorded an honest
FAIL: ``alpha_fit = 349`` against the Brzinski et al. (2013) value ``35 +- 5``,
i.e. the measured resistance was ~10x the continuum drag law.  That case ran
with ``bed_depth = 7 d = 3.5 R`` while penetrating ``1.6 R``: the rigid floor sat
only ``1.9 R`` below the deepest point of the intruder, so the material under the
intruder was ~4 grains thick.  A thin layer squeezed between a big sphere and a
rigid floor is *confined compression*, not the free-surface granular drag that
Eq. (3c)/(3d) describes.

This runner varies one axis at a time (all other knobs from ``--base``) so the
contamination can be *measured* rather than assumed:

``floor``  floor clearance below the deepest intruder point, in R
``rate``   penetration speed (checks the inertial/Froude term by rate doubling)
``rd``     intruder radius in grain diameters, at fixed floor clearance in R

Each case writes ``result.json`` (the standard case record) and ``study.json``
(the swept parameter, the measured alpha, and the window-resolved alpha ladder
of :func:`tensordem.penetration_quasistatic.alpha_vs_window`).

Usage (shard ``i`` of ``n`` runs case ``i``, ``i+n``, ...)::

    PYTHONPATH=src python scripts/penetration_sweep.py --axis floor --shard 0/3 \
        --out-root artifacts/case_penetration/floor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from tensordem.penetration_quasistatic import (  # noqa: E402
    PenetrationConfig, QuasiStaticPenetration, alpha_vs_window, preset)

AXES = {
    # floor clearance below the deepest point, in units of R
    "floor": [1.9, 4.0, 8.0],
    # penetration speed, m/s (base = 0.05)
    "rate": [0.0125, 0.05, 0.2],
    # intruder radius in grain diameters (floor clearance kept at 4 R)
    "rd": [2.0, 3.0],
}


def case_config(axis: str, value: float, base: str = "rd2") -> PenetrationConfig:
    """One swept case; every knob except the swept one comes from ``base``."""
    cfg = preset(base)
    if axis == "floor":
        rd = cfg.intruder_radius_over_d
        cfg.bed_depth_over_d = cfg.penetrate_over_r * rd + value * rd
        tag = f"rd{rd:g}_floor{value:g}R"
    elif axis == "rate":
        cfg.penetration_speed = float(value)
        tag = f"rd{cfg.intruder_radius_over_d:g}_v{value:g}"
    elif axis == "rd":
        rd = float(value)
        cfg.intruder_radius_over_d = rd
        cfg.bed_depth_over_d = cfg.penetrate_over_r * rd + 4.0 * rd
        tag = f"rd{rd:g}_floor4R"
    else:  # pragma: no cover - guarded by argparse choices
        raise SystemExit(f"unknown axis {axis!r}")
    cfg.label = tag
    cfg.validate()
    return cfg


def run_case(cfg: PenetrationConfig, out_dir: str, verbose: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    case = QuasiStaticPenetration(cfg)
    result = case.run(verbose=verbose)
    result["wall_clock_s"] = time.time() - t0

    profile = result["profile"]
    z = np.asarray(profile["z"], dtype=float)
    F = np.asarray(profile["F"], dtype=float)
    geometry = result["geometry"]
    metrics = result["metrics"]
    study = dict(
        label=cfg.label,
        axis=None,
        rd=float(cfg.intruder_radius_over_d),
        bed_depth_over_d=float(cfg.bed_depth_over_d),
        n_grains=int(result["bed"]["n_grains"]),
        penetration_speed=float(cfg.penetration_speed),
        speed_ratio=float(cfg.penetration_speed / np.sqrt(9.81 * cfg.R)),
        penetration_depth_over_r=float(geometry["penetration_depth"]) / cfg.R,
        floor_clearance_over_r=float(geometry["floor_clearance_over_r"]),
        floor_clearance_over_d=float(geometry["floor_clearance_over_d"]),
        packing_fraction=float(result["bed"]["packing_fraction"]),
        bulk_density=float(result["bed"]["bulk_density"]),
        # single global fit (the m15404 estimator)
        alpha_shape=metrics.get("alpha_fit"),
        alpha_free=metrics.get("alpha_fit_free"),
        mean_force_N=metrics.get("mean_force_N"),
        reference_mean_force_N=metrics.get("reference_force_mean_N"),
        force_variation_coefficient=metrics.get("force_variation_coefficient"),
        kinetic_energy_max_J=metrics.get("kinetic_energy_max_J"),
        max_overlap_ratio=metrics.get("max_overlap_ratio"),
        status=result["status"],
        failed=result["failed_criteria"],
        protocol_warnings=result.get("protocol_warnings", []),
        # additive-contamination diagnostic
        alpha_window=alpha_vs_window(z, F, radius=cfg.R, mu=cfg.mu_grain,
                                     rho_bulk=result["bed"]["bulk_density"],
                                     smoothing=cfg.smoothing_over_r * cfg.R,
                                     fit_from=cfg.fit_from_over_r * cfg.R),
        timing=result["timing"],
        wall_clock_s=result["wall_clock_s"],
    )
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(out_dir, "study.json"), "w") as fh:
        json.dump(study, fh, indent=2)
    return study


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", choices=sorted(AXES), required=True)
    ap.add_argument("--values", default=None,
                    help="override the axis values, comma separated")
    ap.add_argument("--base", default="rd2")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: run cases i, i+n, ...")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    values = ([float(v) for v in args.values.split(",")] if args.values
              else AXES[args.axis])
    i, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"bad shard {args.shard!r}")

    summary = []
    for k, value in enumerate(values):
        if k % n != i:
            continue
        cfg = case_config(args.axis, value, args.base)
        if args.seed is not None:
            cfg.seed = int(args.seed)
        out = os.path.join(args.out_root, cfg.label)
        print(f"[shard {i}/{n}] {args.axis} = {value:g} -> {out}", flush=True)
        study = run_case(cfg, out, verbose=not args.quiet)
        summary.append({key: study[key] for key in (
            "label", "rd", "penetration_speed", "speed_ratio",
            "floor_clearance_over_r", "n_grains", "alpha_shape", "alpha_free",
            "mean_force_N", "status", "wall_clock_s")})
        print(json.dumps(summary[-1], indent=2), flush=True)
    print(json.dumps({"shard": args.shard, "cases": summary}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())