"""Dynamics consequence of the righting-moment model in ``tensordem.ice_rideup``.

Runs the same ride-up parameter points twice -- once with the historical
near-level righting moment, once with the exact large-angle hydrostatics -- and
records the resistance split, the contact statistics and the floe attitude
distribution.  The point is *model-form sensitivity*: how much of the reported
``R_up`` / ``R_total`` is an artefact of a righting law that the envelope audit
(`scripts/audit_ice_rideup_righting.py`) shows to be outside its validity range
in exactly the heel band the bow imposes.

Every number is written to ``artifacts/ice_rideup_righting/dynamics.json``; the
markdown table in the report is generated from that file, not transcribed.

Run (single core, minutes):
  PYTHONPATH=src python scripts/audit_ice_rideup_righting_dynamics.py
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

from tensordem.floe_6dof import capsize_angle_from_moment
from tensordem.ice_channel import WaterEnvironment
from tensordem.ice_rideup import (GRAVITY, RHO_ICE, RHO_WATER,
                                  RIGHTING_MODEL_EXACT, RIGHTING_MODEL_NEAR_LEVEL, quat_rotate,
                                  run_case)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ice_rideup_righting"
ANGLES = (20.0, 30.0)
SEEDS = (0, 1)
STEPS = 1800
WARMUP = 400
N_FLOES = 40
FR = 0.3
THICKNESS = 0.5
RADIUS_MEAN = 4.0


def attitude_stats(sim) -> dict:
    """Heel distribution of the floe field at the end of a run (deg)."""
    f = sim.field
    ez = torch.zeros_like(f.quats[:, :3])
    ez[:, 2] = 1.0
    axis = quat_rotate(f.quats, ez)                      # world direction of the floe axis
    tilt = torch.rad2deg(torch.acos(axis[:, 2].abs().clamp(max=1.0)))
    fold = math.degrees(float(capsize_angle_from_moment(
        RADIUS_MEAN, THICKNESS, WaterEnvironment(
            water_density=RHO_WATER, ice_density=RHO_ICE, gravity=GRAVITY))["capsize_angle"]))
    return {
        "heel_mean_deg": float(tilt.mean()),
        "heel_p50_deg": float(tilt.median()),
        "heel_p90_deg": float(torch.quantile(tilt, 0.9)),
        "heel_max_deg": float(tilt.max()),
        "share_past_capsize_fold": float((tilt > fold).to(torch.float64).mean()),
        "capsize_fold_deg": fold,
    }


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.time()
    total = len(ANGLES) * len(SEEDS) * 2
    done = 0
    for angle in ANGLES:
        for seed in SEEDS:
            for model in (RIGHTING_MODEL_NEAR_LEVEL, RIGHTING_MODEL_EXACT):
                t0 = time.time()
                result, sim = run_case(fr=FR, thickness=THICKNESS, angle_deg=angle,
                                       seed=seed, steps=STEPS, warmup_steps=WARMUP,
                                       n_floes=N_FLOES, radius_mean=RADIUS_MEAN,
                                       return_sim=True, righting_model=model)
                d = {k: getattr(result, k) for k in vars(result)} if not hasattr(result, "to_dict") else result.to_dict()
                row = {"angle_deg": angle, "seed": seed, "righting_model": model,
                       "wall_clock_s": time.time() - t0}
                for key in ("r_break", "r_clear", "r_up", "r_total", "r_up_channel",
                            "lift_force_z", "n_climbing", "n_contact", "n_stacked",
                            "fracture_energy_rate", "r_break_energy"):
                    if key in d:
                        row[key] = float(d[key])
                row.update(attitude_stats(sim))
                rows.append(row)
                done += 1
                print(f"[{done}/{total}] alpha={angle:g} seed={seed} {model}: "
                      f"R_up={row.get('r_up', float('nan')):.4g} "
                      f"R_total={row.get('r_total', float('nan')):.4g} "
                      f"heel_p50={row['heel_p50_deg']:.1f} deg "
                      f"({row['wall_clock_s']:.1f}s)", flush=True)

    # pairwise ratios per (angle, seed)
    pairs = []
    for angle in ANGLES:
        for seed in SEEDS:
            near = next(r for r in rows if r["angle_deg"] == angle and r["seed"] == seed
                        and r["righting_model"] == RIGHTING_MODEL_NEAR_LEVEL)
            exact = next(r for r in rows if r["angle_deg"] == angle and r["seed"] == seed
                         and r["righting_model"] == RIGHTING_MODEL_EXACT)
            entry = {"angle_deg": angle, "seed": seed}
            for key in ("r_up", "r_clear", "r_total", "r_up_channel", "n_climbing",
                        "n_contact", "heel_p50_deg", "share_past_capsize_fold"):
                n, e = near.get(key), exact.get(key)
                entry[f"{key}_near"] = n
                entry[f"{key}_exact"] = e
                entry[f"{key}_ratio_exact_over_near"] = (e / n) if n else None
            pairs.append(entry)

    summary = {}
    for key in ("r_up", "r_clear", "r_total", "n_contact", "heel_p50_deg",
                "share_past_capsize_fold"):
        vals = [p[f"{key}_ratio_exact_over_near"] for p in pairs
                if p[f"{key}_ratio_exact_over_near"] is not None]
        if vals:
            summary[key] = {"ratio_min": min(vals), "ratio_max": max(vals),
                            "ratio_mean": sum(vals) / len(vals)}
    out = {
        "setup": {"fr": FR, "thickness_m": THICKNESS, "radius_mean_m": RADIUS_MEAN,
                  "n_floes": N_FLOES, "steps": STEPS, "warmup_steps": WARMUP,
                  "dt": 0.01, "seeds": list(SEEDS), "angles_deg": list(ANGLES),
                  "water": {"rho_water": RHO_WATER, "rho_ice": RHO_ICE,
                            "gravity": GRAVITY}},
        "runs": rows,
        "pairs": pairs,
        "ratio_summary_exact_over_near": summary,
        "wall_clock_total_s": time.time() - t_start,
    }
    (OUT / "dynamics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return out


if __name__ == "__main__":
    main()