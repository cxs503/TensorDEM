#!/usr/bin/env python
"""Cluster driver: one keel-rubble DEM case per shard (campaign p302w2-n04).

Each case deposits a polydisperse block set in the periodic cell that
represents the keel parcel and then shakes it at the wave period, recording the
void ratio.  Output is a self-contained JSON so the analysis (composite model,
fits, figures) can run locally without the DEM.

Usage
-----
    python benchmarks/keel_porosity/p302_n04_dem_run.py --list
    python benchmarks/keel_porosity/p302_n04_dem_run.py --shard 3 --nshards 15
    python benchmarks/keel_porosity/p302_n04_dem_run.py --case main
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from tensordem.ice_ridge_keel_porosity import (
    ConsolidationConfig,
    KeelGeometry,
    build_cell,
    dilute_cell_positions,
    net_effective_gravity,
    polydisperse_radii,
    RubbleDEM,
)

OUTDIR_DEFAULT = "artifacts/keel_porosity/p302_n04"

BASE = dict(n_blocks=400, d_min=0.2, d_max=0.5, distribution="uniform",
            wave_amplitude=3.0, wave_period=10.0, deposit_time=20.0,
            total_time=400.0, record_interval=10.0, dt=1.0e-3, kn=2.0e4,
            damping_ratio=0.10, friction=0.70, rolling_friction=0.05,
            mode="cell", seed=42, porosity_points=30000,
            gravity=9.81, submerged=False, rotation=False)


def _mk(name: str, **kw) -> dict:
    case = dict(BASE)
    case.update(kw)
    case["name"] = name
    return case


def case_list() -> list[dict]:
    """The campaign's DEM cases (one shard each)."""
    cases = [
        _mk("main", n_blocks=1000, wave_amplitude=3.0, total_time=600.0,
            deposit_time=25.0, porosity_points=40000, keep_state=True),
        _mk("amp_1.0", wave_amplitude=1.0),
        _mk("amp_3.0", wave_amplitude=3.0),
        _mk("amp_6.0", wave_amplitude=6.0),
        _mk("amp_12.0", wave_amplitude=12.0),
        _mk("grad_mono", d_min=0.30, d_max=0.40),
        _mk("grad_ref", d_min=0.20, d_max=0.50),
        _mk("grad_wide", d_min=0.10, d_max=0.50),
        _mk("buoyant", submerged=True),
        _mk("dt_fine", n_blocks=300, total_time=200.0, dt=5.0e-4,
            record_interval=5.0),
        _mk("dt_ref", n_blocks=300, total_time=200.0, dt=1.0e-3,
            record_interval=5.0),
        _mk("damp_low", n_blocks=300, total_time=200.0, damping_ratio=0.05,
            record_interval=5.0),
        _mk("damp_high", n_blocks=300, total_time=200.0, damping_ratio=0.20,
            record_interval=5.0),
        _mk("seed_1", seed=1),
        _mk("seed_2", seed=2),
        _mk("keel_walls", n_blocks=300, mode="keel", total_time=200.0,
            record_interval=5.0),
    ]
    return cases


def _jsonable(ledger: dict) -> dict:
    """Flatten the (possibly nested) energy ledger into JSON-safe floats."""
    out = {}
    for key, value in ledger.items():
        if isinstance(value, dict):
            out[key] = {k: float(v) for k, v in value.items()}
        else:
            out[key] = float(value)
    return out


def run_case(case: dict) -> dict:
    geom = KeelGeometry()
    gravity = net_effective_gravity(BASE["gravity"], submerged=case["submerged"])
    radii = polydisperse_radii(case["n_blocks"], case["d_min"], case["d_max"],
                               case["distribution"], -2.0,
                               np.random.default_rng(case["seed"]))
    cfg = ConsolidationConfig(
        dt=case["dt"], deposit_time=case["deposit_time"],
        total_time=case["total_time"], record_interval=case["record_interval"],
        gravity=gravity, kn=case["kn"], damping_ratio=case["damping_ratio"],
        friction=case["friction"], rolling_friction=case["rolling_friction"],
        wave_amplitude=case["wave_amplitude"], wave_period=case["wave_period"],
        placement_porosity=0.75, porosity_points=case["porosity_points"],
        mode=case["mode"], seed=case["seed"],
        rotation=case.get("rotation", False))

    parcel = build_cell(radii, geom, cfg)
    rng = np.random.default_rng(case["seed"])
    dilute = dilute_cell_positions(parcel, radii, rng,
                                   placement_porosity=cfg.placement_porosity)
    t0 = time.time()
    sim = RubbleDEM(dilute.positions, dilute.radii, geom, cfg.to_dem(parcel))
    dem = sim.run()
    wall = time.time() - t0
    keep = dem.time >= case["deposit_time"] - 1e-9
    out = dict(
        case=case, parcel=dict(lx=parcel.lx, ly=parcel.ly, floor=parcel.floor,
                               target_height=parcel.target_height,
                               z_lo=parcel.z_lo, z_mid=parcel.z_mid,
                               volume=parcel.volume),
        gravity=gravity, wall_seconds=wall, steps=int(sim.steps),
        ms_per_step=1e3 * wall / max(sim.steps, 1),
        dt_max_stable=sim.dt_max_stable,
        radii_summary=dict(n=int(len(radii)), d_mean=float(2 * radii.mean()),
                           d_min=float(2 * radii.min()), d_max=float(2 * radii.max()),
                           solid_volume=float(np.sum(4 / 3 * np.pi * radii ** 3))),
        time_dem=np.round(dem.time, 4).tolist(),
        porosity_full=np.round(dem.porosity_slab, 5).tolist(),
        porosity_point_full=np.round(dem.porosity_point, 5).tolist(),
        surface_full=np.round(dem.surface_depth, 4).tolist(),
        coordination_full=np.round(dem.coordination, 4).tolist(),
        kinetic_full=np.round(dem.kinetic_energy, 3).tolist(),
        time=np.round(dem.time[keep] - case["deposit_time"], 4).tolist(),
        porosity=np.round(dem.porosity_slab[keep], 5).tolist(),
        porosity_point=np.round(dem.porosity_point[keep], 5).tolist(),
        surface=np.round(dem.surface_depth[keep], 4).tolist(),
        coordination=np.round(dem.coordination[keep], 4).tolist(),
        kinetic=np.round(dem.kinetic_energy[keep], 3).tolist(),
        energy_ledger=_jsonable(dem.energy_ledger),
        wall_penetration_max=float(getattr(sim, "wall_penetration_max", 0.0)),
        placement_porosity=dilute.placement_porosity,
    )
    if case.get("keep_state"):
        out["final_positions"] = np.round(dem.positions, 4).tolist()
        out["final_radii"] = np.round(dem.radii, 5).tolist()
        out["final_spin_mean"] = float(np.mean(np.linalg.norm(sim.spin, axis=1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=None)
    ap.add_argument("--case", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    args = ap.parse_args()

    cases = case_list()
    if args.list:
        for i, c in enumerate(cases):
            print(i, c["name"])
        return
    if args.case is not None:
        selected = [c for c in cases if c["name"] == args.case]
        if not selected:
            raise SystemExit(f"unknown case {args.case}")
    else:
        shard = 0 if args.shard is None else args.shard
        nshards = len(cases) if args.nshards is None else args.nshards
        selected = [c for i, c in enumerate(cases) if i % nshards == shard]
    os.makedirs(args.outdir, exist_ok=True)
    for case in selected:
        out = run_case(case)
        path = os.path.join(args.outdir, f"{case['name']}.json")
        with open(path, "w") as fh:
            json.dump(out, fh)
        print(f"[done] {case['name']} -> {path} "
              f"({out['wall_seconds']:.0f} s, {out['ms_per_step']:.2f} ms/step, "
              f"n0={out['porosity'][0]:.3f}, n_end={out['porosity'][-1]:.3f})",
              flush=True)


if __name__ == "__main__":
    main()
