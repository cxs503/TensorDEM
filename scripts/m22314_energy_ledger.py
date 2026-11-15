#!/usr/bin/env python
"""m22314: energy ledger of the isochoric biaxial cell (verification, no physics change).

R8 (m22106) registered an "energy budget non-closure": in the 10x20 isochoric
smoke run the contact-layer friction dissipation read from the friction meter
was 24.53 MW while ``q * eps_dot_a * V`` of the same window was 1.59 MW, a
residual of -1464 %.  That comparison silently assumes

    (i)  the boundary work rate equals ``q * eps_dot * V``, and
    (ii) ``sum_friction_power`` is the only dissipation channel.

Neither is checked anywhere.  This script audits the budget *without* either
assumption, using only quantities the integrator already returns per step
(``RidgeAssembly.forces()`` -> ``exchange`` / ``dissipation``):

    P_driver(t)   = sum_planes  F_particles<-wall . v_wall     (driver work rate)
    P_particle(t) = sum_planes  F_particles<-wall . v_particle (``exchange.plane``)
    D(t)          = sum(dissipation.values())                  (all sinks, W >= 0)
    K(t)          = 1/2 sum m v^2                              (kinetic)
    U(t)          = 1/2 sum kn delta^2  (disk-disk) + wall springs

Ledger identities (must hold to integration error):

    R_particle = W[P_particle] - W[D] - dK - dU_contact   -> 0
    R_driver   = W[P_driver]   - W[D] - dK - dU_contact - dU_wall -> 0

and the *macro* estimate used by the analysis pipeline is compared against
``W[P_driver]`` so that the size of the "non-closure" can be attributed.

Usage:  PYTHONPATH=src python scripts/m22314_energy_ledger.py --columns 8 --rows 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.biaxial_shear_band import _make_specimen  # noqa: E402
from tensordem.ridge_dem import RidgeMaterial, _advance_planes  # noqa: E402

OUT_SUBDIR = os.path.join("artifacts", "CASE-DEM-BIAXIAL-FRICTION", "energy_ledger")


def _elastic_energy(assembly) -> tuple[float, float]:
    """(disk-disk, wall) spring energy of the current configuration, J."""
    pos, radii = assembly.pos, assembly.radii
    m = assembly.material
    n = int(pos.shape[0])
    u_contact = 0.0
    if n > 1:
        idx_i, idx_j = torch.triu_indices(n, n, offset=1)
        dist = torch.linalg.vector_norm(pos[idx_j] - pos[idx_i], dim=1)
        overlap = (radii[idx_i] + radii[idx_j] - dist).clamp_min(0.0)
        u_contact = float(0.5 * m.contact_stiffness * (overlap ** 2).sum())
    u_wall = 0.0
    for plane in assembly.planes:
        normal = torch.tensor(plane.normal, dtype=torch.float64)
        signed = (pos - torch.tensor(plane.point, dtype=torch.float64)) @ normal
        gap = (assembly.radii - signed).clamp_min(0.0)
        u_wall += float(0.5 * plane.stiffness * (gap ** 2).sum())
    return u_contact, u_wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", type=int, default=8)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--radius", type=float, default=0.75)
    ap.add_argument("--mu", type=float, default=0.55)
    ap.add_argument("--strain-rate", type=float, default=0.05)
    ap.add_argument("--strain-target", type=float, default=0.05)
    ap.add_argument("--confining-stress", type=float, default=100.0e3)
    ap.add_argument("--consolidation-steps", type=int, default=600)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--dt-scale", type=float, default=1.0,
                    help="multiply the material's recommended dt (time-step refinement)")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("TENSORDEM_THREADS", "4")))
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    material = RidgeMaterial(particle_radius=args.radius,
                             friction_coefficient=args.mu,
                             freeze_bond_fraction=0.0)
    assembly, geometry = _make_specimen(
        material, columns=args.columns, rows=args.rows,
        confining_stress=args.confining_stress, jitter=0.03,
        servo_gain=1.0e-4, servo_velocity_limit=2.0,
        wall_friction=0.0, axial_servo=False)
    dt = material.recommended_dt * float(args.dt_scale)
    thickness = assembly.thickness

    for _ in range(int(args.consolidation_steps)):
        out = assembly.step(dt)
        assembly.last_plane_forces = out["plane_forces"]
        _advance_planes(assembly, dt)

    left, right, bottom, top = assembly.planes
    width0 = float(right.point[0] - left.point[0])
    height0 = float(top.point[1] - bottom.point[1])
    volume0 = width0 * height0 * thickness
    mass = assembly.mass

    started = time.time()
    max_steps = int(math.ceil(args.strain_target
                              / (args.strain_rate * dt))) + 5
    assembly.enable_friction_meter()
    acc = {"p_driver": 0.0, "p_particle": 0.0, "diss": 0.0, "steps": 0,
           "meter_friction": 0.0, "meter_calls": 0.0, "absdiff": 0.0}
    diss_parts = {}
    samples = []
    u_c0, u_w0 = _elastic_energy(assembly)
    k0 = float(0.5 * (mass * assembly.vel.square().sum(dim=1)).sum())

    for _step in range(max_steps):
        left, right, bottom, top = assembly.planes
        height = float(top.point[1] - bottom.point[1])
        width_now = float(right.point[0] - left.point[0])
        v_lateral = -0.5 * args.strain_rate * width_now
        for _lat in (0, 1):
            plane = assembly.planes[_lat]
            if plane.servo_stress is not None:
                plane = plane.__class__(**{**plane.__dict__, "servo_stress": None})
            assembly.planes[_lat] = plane.__class__(
                **{**plane.__dict__, "velocity": v_lateral})
        assembly.planes[3] = assembly.planes[3].__class__(
            **{**assembly.planes[3].__dict__,
               "velocity": float(args.strain_rate * height)})

        out = assembly.step(dt)
        assembly.last_plane_forces = out["plane_forces"]
        plane_forces = out["plane_forces"]

        # driver work rate: force on the particles (=-plane force) dotted with the
        # wall velocity the driver actually imposes
        p_driver = 0.0
        for plane, force in zip(assembly.planes, plane_forces):
            normal = torch.tensor(plane.normal, dtype=torch.float64)
            v_wall = plane.velocity * normal
            p_driver += float(-(force @ v_wall))
        p_particle = float(out["exchange"]["plane"])
        meter = assembly.read_friction_meter(reset=True)
        d_parts = {k: float(v) for k, v in out["dissipation"].items()}
        d_total = sum(d_parts.values())
        for k, v in d_parts.items():
            diss_parts[k] = diss_parts.get(k, 0.0) + v * dt
        acc["p_driver"] += p_driver * dt
        acc["p_particle"] += p_particle * dt
        acc["diss"] += d_total * dt
        meter_fric = float(meter["sum_friction_power"])   # one forces() call per step
        acc["meter_friction"] += meter_fric * dt
        acc["meter_calls"] += float(meter["calls"])
        acc["absdiff"] += abs(meter_fric - d_parts.get("contact_friction", 0.0)) * dt
        acc["steps"] += 1

        _advance_planes(assembly, dt)

        if acc["steps"] % 100 == 0:
            width = float(right.point[0] - left.point[0])
            height = float(top.point[1] - bottom.point[1])
            sigma_v = -float(plane_forces[2][1]) / (width * thickness)
            sigma_v_b = float(plane_forces[3][1]) / (width * thickness)
            sigma_h = 0.5 * (-float(plane_forces[0][0]) / (height * thickness)
                             + float(plane_forces[1][0]) / (height * thickness))
            vol = width * height * thickness
            u_c, u_w = _elastic_energy(assembly)
            k_now = float(0.5 * (mass * assembly.vel.square().sum(dim=1)).sum())
            samples.append({
                "U_contact_J": u_c, "U_wall_J": u_w, "KE_J": k_now,
                "step": acc["steps"], "_dt": dt,
                "axial_strain_log": math.log(height0 / max(height, 1e-12)),
                "p_driver_W": p_driver, "p_particle_W": p_particle,
                "diss_total_W": d_total,
                "diss_parts_W": d_parts,
                "meter_friction_W": meter_fric,
                "meter_calls": float(meter["calls"]),
                "sigma_v_bottom_Pa": sigma_v, "sigma_v_top_Pa": sigma_v_b,
                "sigma_h_Pa": sigma_h,
                "q_macro_Pa": sigma_v - sigma_h,
                "volume_m3": vol,
                "macro_power_W": (sigma_v - sigma_h) * args.strain_rate * vol,
            })
        if math.log(height0 / max(height, 1e-12)) >= args.strain_target:
            break

    elapsed = time.time() - started
    u_c1, u_w1 = _elastic_energy(assembly)
    k1 = float(0.5 * (mass * assembly.vel.square().sum(dim=1)).sum())

    w_driver = acc["p_driver"]
    w_particle = acc["p_particle"]
    w_diss = acc["diss"]
    w_meter_friction = acc["meter_friction"]
    w_meter_absdiff = acc["absdiff"]
    dk = k1 - k0
    du_c = u_c1 - u_c0
    du_w = u_w1 - u_w0
    r_particle = w_particle - w_diss - dk - du_c
    r_driver = w_driver - w_diss - dk - du_c - du_w
    macro_w = float(np.trapezoid([s["macro_power_W"] for s in samples], dx=100 * dt)) \
        if len(samples) > 1 else float("nan")

    record = {
        "tag": args.tag or f"ledger_{args.columns}x{args.rows}",
        "module": "CASE-DEM-BIAXIAL-FRICTION",
        "purpose": "audit the R8 energy-budget 'non-closure' against the integrator's own ledger",
        "config": {"columns": args.columns, "rows": args.rows, "mu": args.mu,
                   "strain_rate": args.strain_rate,
                   "strain_target": args.strain_target,
                   "confining_stress": args.confining_stress,
                   "consolidation_steps": args.consolidation_steps,
                   "dt": dt, "dt_scale": float(args.dt_scale),
                   "thickness": thickness,
                   "freeze_bond_fraction": 0.0, "wall_friction": 0.0,
                   "lateral_control": "isochoric"},
        "specimen": {"width0": width0, "height0": height0, "volume0": volume0,
                     "particles": int(assembly.pos.shape[0])},
        "steps_run": acc["steps"], "sim_time_s": acc["steps"] * dt,
        "elapsed_s": elapsed,
        "work_J": {"w_driver": w_driver, "w_particle": w_particle,
                   "w_dissipated": w_diss, "macro_qV": macro_w,
                   "w_meter_friction": w_meter_friction,
                   "delta_KE": dk, "delta_U_contact": du_c,
                   "delta_U_wall": du_w},
        "dissipation_by_channel_J": diss_parts,
        "residual_J": {"particle_ledger": r_particle, "driver_ledger": r_driver},
        "meter_check": {
            "meter_calls_total": acc["meter_calls"],
            "meter_friction_J": w_meter_friction,
            "ledger_contact_friction_J": diss_parts.get("contact_friction", float("nan")),
            "relative_disagreement": ((w_meter_friction
                                       - diss_parts.get("contact_friction", float("nan")))
                                      / w_meter_friction if w_meter_friction else float("nan")),
            "integrated_abs_difference_J": w_meter_absdiff,
        },
        "residual_fraction": {
            "particle_ledger": r_particle / w_particle if w_particle else float("nan"),
            "driver_ledger": r_driver / w_driver if w_driver else float("nan"),
            "macro_vs_driver": ((macro_w - w_driver) / w_driver
                                if w_driver else float("nan")),
        },
        "samples": samples,
        "window_analysis": _window_analysis(samples),
    }
    path = os.path.join(out_dir, f"{record['tag']}.json")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=1)
    print(json.dumps({k: record[k] for k in
                      ("tag", "steps_run", "sim_time_s", "elapsed_s",
                       "work_J", "residual_J", "residual_fraction")}, indent=1))
    print(f"wrote {path}")
    return 0


def _window_analysis(samples: list[dict]) -> dict:
    """Late-window (last 20 %) power balance: is the cell energetically steady?"""
    if len(samples) < 6:
        return {"error": "too few samples"}
    strains = [s["axial_strain_log"] for s in samples]
    lo = strains[-1] - 0.2 * (strains[-1] - strains[0])
    idx = [i for i, e in enumerate(strains) if e >= lo]
    if len(idx) < 3:
        return {"error": "late window too short"}
    first, last = samples[idx[0]], samples[idx[-1]]
    t_win = (idx[-1] - idx[0]) * 100 * first["_dt"] if "_dt" in first else float("nan")
    def _mean(key):
        return float(np.mean([samples[i][key] for i in idx]))
    def _rate(key):
        return ((last[key] - first[key]) / t_win) if t_win else float("nan")
    out = {
        "strain_window": [first["axial_strain_log"], last["axial_strain_log"]],
        "window_s": t_win,
        "mean_p_driver_W": _mean("p_driver_W"),
        "mean_p_particle_W": _mean("p_particle_W"),
        "mean_diss_total_W": _mean("diss_total_W"),
        "mean_contact_friction_W": _mean("meter_friction_W"),
        "mean_macro_power_W": _mean("macro_power_W"),
        "dUdt_contact_W": _rate("U_contact_J"),
        "dUdt_wall_W": _rate("U_wall_J"),
        "dKdt_W": _rate("KE_J"),
        "mean_U_contact_J": _mean("U_contact_J"),
        "mean_KE_J": _mean("KE_J"),
    }
    p_in = out["mean_p_particle_W"]
    sinks = (out["mean_diss_total_W"] + out["dUdt_contact_W"] + out["dKdt_W"])
    out["closure_residual_W"] = p_in - sinks
    out["closure_residual_fraction"] = ((p_in - sinks) / p_in) if p_in else float("nan")
    out["dissipation_over_input"] = (out["mean_diss_total_W"] / p_in
                                     if p_in else float("nan"))
    return out


if __name__ == "__main__":
    raise SystemExit(main())