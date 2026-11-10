#!/usr/bin/env python
"""Shard dispatcher for the granular elastic-wave pressure campaign.

One case = one confining pressure (or the frictionless negative control).  Each
case builds its own specimen, trims it to the target pressure, and measures the
P and S wave speeds with two independent routes (free ring-down frequency and
steady-state resonance sweep) plus the two attenuation estimators (free-decay
log decrement and half-power bandwidth).

Usage
-----
``python scripts/run_granular_wave_campaign.py --shard 0 --shards 7 --out-dir runs/wave_campaign``

The case table is deliberately data, not code: ``CASES`` below.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import torch

from tensordem.granular_wave import dynamic as dyn
from tensordem.granular_wave.emt import digby_hertz_mindlin
from tensordem.granular_wave.engine import GranularPacking, GranularWaveConfig

# label, pressure [Pa], friction, nx, nz
CASES: list[dict[str, Any]] = [
    {"label": "p010kPa", "pressure": 1.0e4, "friction": 0.30, "nx": 6, "nz": 36},
    {"label": "p025kPa", "pressure": 2.5e4, "friction": 0.30, "nx": 6, "nz": 36},
    {"label": "p050kPa", "pressure": 5.0e4, "friction": 0.30, "nx": 6, "nz": 36},
    {"label": "p100kPa", "pressure": 1.0e5, "friction": 0.30, "nx": 6, "nz": 36},
    {"label": "p200kPa", "pressure": 2.0e5, "friction": 0.30, "nx": 6, "nz": 36},
    {"label": "p400kPa", "pressure": 4.0e5, "friction": 0.30, "nx": 6, "nz": 36},
    {"label": "neg100kPa_mu0", "pressure": 1.0e5, "friction": 1.0e-6, "nx": 6, "nz": 36},
]


def run_case(case: dict[str, Any], out_dir: str, *, threads: int,
             modes: tuple[str, ...] = ("P", "S"),
             sweep_points: int = 5, warmup_cycles: int = 3,
             measure_cycles: int = 3, keep_signals: bool = True) -> dict[str, Any]:
    torch.set_num_threads(max(1, threads))
    cfg = GranularWaveConfig(nx=case["nx"], ny=case["nx"], nz=case["nz"],
                             friction=case["friction"], seed=case.get("seed", 0))
    dcfg = dyn.DynamicConfig(pressure_Pa=case["pressure"],
                             sweep_points=sweep_points,
                             sweep_warmup_cycles=warmup_cycles,
                             sweep_measure_cycles=measure_cycles,
                             trim_passes=2, hold_tol=0.10)
    t_start = time.time()
    init = dyn.build_loose_lattice(cfg, gap=dcfg.gap, jitter=dcfg.jitter)
    sim = GranularPacking(cfg, init=init)
    dt_info = dyn.safe_timestep(sim, max_overlap_fraction=0.01, factor=0.10)
    sim.dt = dt_info["dt_s"]
    prep = dyn.prepare_state(sim, case["pressure"], dcfg)
    state = prep["state"]
    pred = digby_hertz_mindlin(
        coordination_number=state["coordination_number"],
        porosity=state["porosity"],
        grain_shear_modulus=cfg.shear_modulus,
        grain_poisson_ratio=cfg.poisson_ratio,
        pressure=state["pressure_mean_Pa"],
        bulk_density=state["bulk_density_kg_m3"])
    out: dict[str, Any] = {
        "label": case["label"],
        "case": case,
        "config": {"nx": case["nx"], "ny": case["nx"], "nz": case["nz"],
                   "radius_m": cfg.radius, "polydispersity": cfg.polydispersity,
                   "density": cfg.density, "young_modulus": cfg.young_modulus,
                   "poisson_ratio": cfg.poisson_ratio,
                   "friction": cfg.friction, "wall_friction": cfg.wall_friction,
                   "normal_damping": cfg.normal_damping,
                   "gap": dcfg.gap, "jitter": dcfg.jitter},
        "timestep": dt_info,
        "preparation": prep,
        "digby": pred,
        "waves": {},
    }
    spectrum: dict[str, np.ndarray] = {}
    for mode in modes:
        v_ref = pred["v_p_m_s"] if mode == "P" else pred["v_s_m_s"]
        t0 = time.time()
        ring = dyn.ringdown_velocity(sim, mode=mode, v_ref=v_ref, cfg=dcfg,
                                     keep_signals=keep_signals)
        ring["elapsed_s"] = time.time() - t0
        if keep_signals:
            sig = ring.pop("signals")
            spectrum[f"ringdown_{mode}_t"] = np.asarray(sig["times_s"])
            spectrum[f"ringdown_{mode}_trace"] = np.asarray(sig["trace"])
            ring["n_kick"] = int(sig["n_kick"])
        t0 = time.time()
        sweep = dyn.steady_resonance(sim, mode=mode, v_ref=v_ref, cfg=dcfg,
                                     f0_guess=ring["frequency_hz"])
        sweep["elapsed_s"] = time.time() - t0
        ring["sweep"] = sweep
        ring["speed_ratio_ring_to_sweep"] = (
            ring["speed_m_s"] / sweep["speed_m_s"]
            if sweep.get("speed_m_s") and np.isfinite(sweep["speed_m_s"]) else float("nan"))
        out["waves"][mode] = ring
    out["wall_time_s"] = time.time() - t_start
    os.makedirs(out_dir, exist_ok=True)
    payload = {"report": np.array(json.dumps(out, default=float))}
    payload.update(spectrum)
    path = os.path.join(out_dir, f"{case['label']}.npz")
    np.savez_compressed(path, **payload)
    print(json.dumps({
        "label": out["label"], "path": path,
        "pressure_Pa": state["pressure_mean_Pa"],
        "solid_fraction": 1.0 - state["porosity"],
        "coordination_number": state["coordination_number"],
        "max_overlap_fraction": state["max_overlap_fraction"],
        "digby_v_p": pred["v_p_m_s"], "digby_v_s": pred["v_s_m_s"],
        "waves": {m: {"v_ring": out["waves"][m]["speed_m_s"],
                      "v_sweep": out["waves"][m]["sweep"]["speed_m_s"],
                      "f_ring": out["waves"][m]["frequency_hz"],
                      "q_decay": out["waves"][m]["q_decay"],
                      "q_bandwidth": out["waves"][m]["sweep"]["q_bandwidth"]}
                  for m in out["waves"]},
        "wall_time_s": out["wall_time_s"]}, default=float), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out-dir", type=str, default="runs/wave_campaign")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--modes", type=str, default="P,S")
    ap.add_argument("--sweep-points", type=int, default=5)
    ap.add_argument("--warmup-cycles", type=int, default=3)
    ap.add_argument("--measure-cycles", type=int, default=3)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    cases = CASES
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        cases = [c for c in cases if c["label"] in wanted]
    mine = [c for i, c in enumerate(cases) if i % max(1, args.shards) == args.shard]
    for case in mine:
        run_case(case, args.out_dir, threads=args.threads,
                 modes=tuple(m for m in args.modes.split(",") if m),
                 sweep_points=args.sweep_points,
                 warmup_cycles=args.warmup_cycles,
                 measure_cycles=args.measure_cycles)


if __name__ == "__main__":
    main()