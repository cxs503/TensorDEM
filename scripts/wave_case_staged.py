#!/usr/bin/env python
"""Staged, resumable runner for one granular wave-speed case (pressure state).

The heavy object is a DEM specimen; a case is therefore split into stages that
checkpoint the simulation state to disk, so a job queue with a short wall-time
limit can carry a case forward across several invocations:

``prep``     build + jam + trim to the target pressure        -> state + prep log
``ringP``    free ring-down, P mode                            -> f_P, Q_P
``ringS``    free ring-down, S mode                            -> f_S, Q_S
``sweepP``   steady resonance sweep, P mode                    -> f0_P, Q_P(bw)
``sweepS``   steady resonance sweep, S mode                    -> f0_S, Q_S(bw)
``report``   assemble the case JSON

``--budget`` runs stages until the wall-time budget is exhausted and exits with a
zero status, so the next call simply continues where this one stopped.

Example
-------
``python scripts/wave_case_staged.py --label p100kPa --pressure 1e5 --budget 90``
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any

import numpy as np
import torch

from tensordem.granular_wave import dynamic as dyn
from tensordem.granular_wave.emt import digby_hertz_mindlin
from tensordem.granular_wave.engine import GranularPacking, GranularWaveConfig

STAGES = ("prep", "ringP", "ringS", "sweepP", "sweepS", "report")

_STATE_FIELDS = ("pos", "radius", "mass", "inertia", "wall_pos", "vel", "omega",
                 "wall_vnormal", "wall_tangential_drive", "wall_tan", "pair_tan",
                 "pair_force_normal", "pairs")


def save_state(sim: GranularPacking, path: str, meta: dict[str, Any]) -> None:
    torch.save({
        "meta": meta,
        "dt": sim.dt,
        "time": sim.time,
        "step_count": sim.step_count,
        "tensors": {name: getattr(sim, name) for name in _STATE_FIELDS},
    }, path)


def load_state(cfg: GranularWaveConfig, path: str) -> tuple[GranularPacking, dict[str, Any]]:
    blob = torch.load(path, weights_only=False)
    sim = GranularPacking(cfg)
    for name, value in blob["tensors"].items():
        setattr(sim, name, value)
    sim.n = len(sim.pos)
    sim.dt = blob["dt"]
    sim.time = blob["time"]
    sim.step_count = blob["step_count"]
    sim.rebuild_pairs()
    return sim, blob["meta"]


def build_sim(cfg: GranularWaveConfig, dcfg: dyn.DynamicConfig) -> GranularPacking:
    init = dyn.build_loose_lattice(cfg, gap=dcfg.gap, jitter=dcfg.jitter)
    sim = GranularPacking(cfg, init=init)
    sim.dt = dyn.safe_timestep(sim, max_overlap_fraction=0.01, factor=0.10)["dt_s"]
    return sim


def case_paths(out_dir: str, label: str) -> dict[str, str]:
    base = os.path.join(out_dir, label)
    return {"state": base + "_state.pt", "json": base + ".json",
            "trace": base + "_trace.npz"}


def load_json(path: str) -> dict[str, Any]:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def write_json(path: str, data: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1, default=float)
    os.replace(tmp, path)


def done_stages(record: dict[str, Any]) -> set[str]:
    done = set(record.get("stages_done", []))
    if os.path.exists(record.get("paths", {}).get("state", "")):
        done.add("prep")
    return done


def _with_dt_retry(sim: GranularPacking, cfg: GranularWaveConfig, paths: dict[str, str], fn,
                   attempts: int = 3):
    """Run a wave measurement, halving the step if the packing diverges.

    A marginal specimen (4 grains across) can reach overlaps that make the
    preparation step marginal; rather than losing the whole case, the measurement
    is retried with a smaller step and the reduction is recorded.
    """
    last = None
    for k in range(attempts):
        try:
            return fn(sim), sim
        except RuntimeError as exc:  # divergence
            last = exc
            sim, _meta = load_state(cfg, paths["state"])
            sim.dt = sim.dt * 0.5
    raise RuntimeError(f"wave measurement diverged after {attempts} attempts: {last}")


def run_stage(stage: str, args: argparse.Namespace, paths: dict[str, str]) -> dict[str, Any]:
    cfg = GranularWaveConfig(nx=args.nx, ny=args.nx, nz=args.nz,
                             friction=args.friction, seed=args.seed)
    dcfg = dyn.DynamicConfig(pressure_Pa=args.pressure,
                             slab_thickness_factor=args.slab_thickness,
                             kick_cycles=args.kick_cycles,
                             ringdown_cycles=args.ringdown_cycles,
                             sweep_points=args.sweep_points,
                             sweep_amplitude=args.sweep_amplitude,
                             pulse_amplitude=args.kick_amplitude,
                             sweep_span=args.sweep_span,
                             sweep_warmup_cycles=args.warmup_cycles,
                             sweep_measure_cycles=args.measure_cycles,
                             trim_passes=args.trim_passes, hold_tol=args.hold_tol,
                             hold_steps=args.hold_steps, hold_window=args.hold_window)
    record = load_json(paths["json"])
    record.setdefault("label", args.label)
    record.setdefault("case", {"pressure": args.pressure, "friction": args.friction,
                               "nx": args.nx, "nz": args.nz, "seed": args.seed})
    record.setdefault("config", {"radius_m": cfg.radius,
                                 "polydispersity": cfg.polydispersity,
                                 "density": cfg.density,
                                 "young_modulus": cfg.young_modulus,
                                 "poisson_ratio": cfg.poisson_ratio,
                                 "friction": cfg.friction,
                                 "wall_friction": cfg.wall_friction,
                                 "normal_damping": cfg.normal_damping})
    record.setdefault("paths", paths)
    record.setdefault("waves", {})
    record.setdefault("stages_done", [])

    if stage == "prep" or not os.path.exists(paths["state"]):
        t0 = time.time()
        sim = build_sim(cfg, dcfg)
        record["timestep"] = dyn.safe_timestep(sim, max_overlap_fraction=0.01, factor=0.10)
        prep = dyn.prepare_state(sim, args.pressure, dcfg)
        state = prep["state"]
        record["preparation"] = prep
        record["digby"] = digby_hertz_mindlin(
            coordination_number=state["coordination_number"],
            porosity=state["porosity"],
            grain_shear_modulus=cfg.shear_modulus,
            grain_poisson_ratio=cfg.poisson_ratio,
            pressure=state["pressure_mean_Pa"],
            bulk_density=state["bulk_density_kg_m3"])
        record["state"] = {k: state[k] for k in (
            "n_particles", "pressure_mean_Pa", "pressure_per_axis_Pa", "porosity",
            "bulk_density_kg_m3", "coordination_number", "box_lengths_m",
            "max_overlap_fraction", "kinetic_energy_J", "dt_s")}
        record["state"]["solid_fraction"] = 1.0 - state["porosity"]
        record["state"]["n_contacts"] = state["n_load_contacts"]
        record["prep_elapsed_s"] = time.time() - t0
        save_state(sim, paths["state"], {"label": args.label})
        record["stages_done"] = sorted(set(record["stages_done"]) | {"prep"})
        write_json(paths["json"], record)
        return record

    sim, _meta = load_state(cfg, paths["state"])
    state = record["state"]
    pred = record["digby"]
    # the step is re-derived from the *realised* overlap: at 4 grains across, wall
    # compaction produces overlaps well above the 1% the preparation assumed, and
    # a step sized for 1% diverges on the stiffer realised contacts
    realised = max(float(state.get("max_overlap_fraction", 0.01)), 0.01)
    dt_safe = dyn.safe_timestep(sim, max_overlap_fraction=min(realised * 1.2, 0.05),
                                factor=0.08)["dt_s"]
    sim.dt = min(sim.dt, dt_safe)
    record["timestep"]["dt_wave_s"] = sim.dt
    record["timestep"]["realised_overlap_fraction"] = realised

    if stage in ("ringP", "ringS"):
        mode = "P" if stage == "ringP" else "S"
        v_ref = pred["v_p_m_s"] if mode == "P" else pred["v_s_m_s"]
        t0 = time.time()
        ring, sim = _with_dt_retry(
            sim, cfg, paths, lambda s: dyn.ringdown_velocity(
                s, mode=mode, v_ref=v_ref, cfg=dcfg, keep_signals=True))
        ring["dt_s"] = sim.dt
        ring["elapsed_s"] = time.time() - t0
        sig = ring.pop("signals")
        np.savez_compressed(paths["trace"].replace(".npz", f"_{mode}.npz"),
                            t=np.asarray(sig["times_s"]),
                            trace=np.asarray(sig["trace"]),
                            n_kick=np.asarray(sig["n_kick"]))
        record["waves"][mode] = {**record["waves"].get(mode, {}), "ring": ring}
        save_state(sim, paths["state"], {"label": args.label})
        record["stages_done"] = sorted(set(record["stages_done"]) | {stage})
        write_json(paths["json"], record)
        return record

    if stage in ("sweepP", "sweepS"):
        mode = "P" if stage == "sweepP" else "S"
        v_ref = pred["v_p_m_s"] if mode == "P" else pred["v_s_m_s"]
        ring = record["waves"].get(mode, {}).get("ring", {})
        f0 = ring.get("frequency_hz") if ring.get("frequency_hz") else v_ref / (
            2.0 * state["box_lengths_m"][2])
        t0 = time.time()
        def _fresh() -> GranularPacking:
            fresh, _ = load_state(cfg, paths["state"])
            fresh.dt = sim.dt
            return fresh

        sweep, sim = _with_dt_retry(
            sim, cfg, paths,
            lambda s: dyn.steady_resonance(s, mode=mode, v_ref=v_ref, cfg=dcfg,
                                           f0_guess=f0, reset=_fresh))
        sweep["dt_s"] = sim.dt
        sweep["elapsed_s"] = time.time() - t0
        record["waves"][mode] = {**record["waves"].get(mode, {}), "sweep": sweep}
        save_state(sim, paths["state"], {"label": args.label})
        record["stages_done"] = sorted(set(record["stages_done"]) | {stage})
        write_json(paths["json"], record)
        return record

    # report: summarise the case
    summary = {"label": args.label, "pressure_Pa": state["pressure_mean_Pa"],
               "solid_fraction": state["solid_fraction"],
               "coordination_number": state["coordination_number"],
               "max_overlap_fraction": state["max_overlap_fraction"],
               "digby_v_p": pred["v_p_m_s"], "digby_v_s": pred["v_s_m_s"], "waves": {}}
    for mode in ("P", "S"):
        w = record["waves"].get(mode, {})
        ring, sweep = w.get("ring", {}), w.get("sweep", {})
        v_digby = pred["v_p_m_s"] if mode == "P" else pred["v_s_m_s"]
        summary["waves"][mode] = {
            "v_ring": ring.get("speed_m_s"), "v_sweep": sweep.get("speed_m_s"),
            "f_ring": ring.get("frequency_hz"), "f_sweep": sweep.get("frequency_hz"),
            "two_method_rel_diff": ((ring.get("speed_m_s") / sweep.get("speed_m_s") - 1.0)
                                    if ring.get("speed_m_s") and sweep.get("speed_m_s") else None),
            "q_decay": ring.get("q_decay"), "q_bandwidth": sweep.get("q_bandwidth"),
            "dev_ring_vs_digby": (ring.get("speed_m_s") / v_digby - 1.0
                                  if ring.get("speed_m_s") else None),
        }
    record["summary"] = summary
    record["stages_done"] = sorted(set(record["stages_done"]) | {"report"})
    write_json(paths["json"], record)
    print(json.dumps(summary, default=float), flush=True)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--pressure", type=float, required=True)
    ap.add_argument("--friction", type=float, default=0.30)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default="runs/wave_campaign")
    ap.add_argument("--budget", type=float, default=90.0,
                    help="wall-clock budget in seconds; stages run until it is spent")
    ap.add_argument("--slab-thickness", type=float, default=3.0)
    ap.add_argument("--kick-cycles", type=float, default=1.5)
    ap.add_argument("--ringdown-cycles", type=int, default=8)
    ap.add_argument("--sweep-points", type=int, default=5)
    ap.add_argument("--sweep-amplitude", type=float, default=1.0e-3)
    ap.add_argument("--kick-amplitude", type=float, default=2.0e-3)
    ap.add_argument("--sweep-span", type=float, default=0.40)
    ap.add_argument("--warmup-cycles", type=int, default=3)
    ap.add_argument("--measure-cycles", type=int, default=3)
    ap.add_argument("--trim-passes", type=int, default=1)
    ap.add_argument("--hold-steps", type=int, default=120)
    ap.add_argument("--hold-window", type=int, default=25)
    ap.add_argument("--hold-tol", type=float, default=0.12)
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    os.makedirs(args.out_dir, exist_ok=True)
    paths = case_paths(args.out_dir, args.label)
    record = load_json(paths["json"])
    done = done_stages(record)
    t_start = time.time()
    for stage in STAGES:
        if stage in done or stage == "report":
            continue
        if time.time() - t_start > args.budget:
            print(f"budget spent before {stage}", flush=True)
            break
        t0 = time.time()
        run_stage(stage, args, paths)
        print(f"stage {stage} done in {time.time() - t0:.1f}s", flush=True)
    record = load_json(paths["json"])
    if all(s in done_stages(record) for s in STAGES[:-1]):
        run_stage("report", args, paths)


if __name__ == "__main__":
    main()