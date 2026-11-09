#!/usr/bin/env python3
"""Campaign p41w3-n04 witness run: thermal contact-conduction chain + granular collapse.

Modes
-----
``thermal``     all thermal verification cases (contact law, 1D chain steady and
                transient, 2D lattice, ice-bond gate) -> ``thermal.json``
``collapse``    one task of the aspect-ratio / inertial-number table
                (``--index i --num-tasks n``) -> ``case_<name>.json``
``assemble``    read every ``case_*.json``, fit the runout scaling law, grade the
                gates and write ``summary.json`` plus the figures

Everything is written under ``--outdir`` (default ``runs/dem_thermal_granular``)
so that the same script runs locally or sharded over the compute cluster.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import replace

import numpy as np

from tensordem.granular_collapse import (
    AXISYMMETRIC_EXPONENT,
    JOP_I_0,
    CollapseCase,
    deposit_front,
    fit_power_law,
    column_collapse,
    jop_mu_I,
    scaling_comparison,
    LAJEUNESSE_RECTANGULAR_EXPONENT_HIGH,
    LAJEUNESSE_RECTANGULAR_EXPONENT_LOW,
)
from tensordem.thermal.contact_chain import (
    ChainCase,
    ThermalChain,
    TransientChain,
    bond_failure_gate,
    chain_resistance,
    contact_constriction_resistance,
    hertz_contact_radius,
    ice_bond_tensile_strength,
    lattice_network_temperatures,
    sphere_bulk_resistance,
    square_lattice_resistance,
)

# --------------------------------------------------------------- task table
#: aspect-ratio scan at g = 9.81 m/s^2 (rectangular channel)
ASPECTS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
#: gravity factors at fixed aspect ratio 2 -> inertial-number sweep
GRAVITY_FACTORS = (0.25, 0.5, 2.0, 4.0)
FEATURED_ASPECT = 2.0


def collapse_tasks() -> list[dict[str, object]]:
    tasks = [{"kind": "aspect", "aspect": a, "gravity_factor": 1.0} for a in ASPECTS]
    tasks += [{"kind": "gravity", "aspect": FEATURED_ASPECT, "gravity_factor": f}
              for f in GRAVITY_FACTORS]
    # one full contact-level ledger run (gate c) on a mid aspect ratio
    tasks += [{"kind": "ledger", "aspect": FEATURED_ASPECT, "gravity_factor": 1.0}]
    return tasks


TAG = ""


def task_name(task: dict[str, object]) -> str:
    return (f"{task['kind']}_a{float(task['aspect']):g}_g{float(task['gravity_factor']):g}")


def build_case(task: dict[str, object], *, width: int, radius: float,
               max_steps: int, mu_rolling: float = 0.0, mu_s: float = 0.36,
               mu_2: float = 0.84, dt_safety: float = 1.0) -> CollapseCase:
    return CollapseCase(
        aspect=float(task["aspect"]),
        width=width,
        radius=radius,
        mu_rolling=mu_rolling,
        mu_s=mu_s,
        mu_2=mu_2,
        gravity=9.81 * float(task["gravity_factor"]),
        max_steps=max_steps,
        ke_tol=1e-2,
        ke_hold=500,
        dt_safety=dt_safety,
    )


# ------------------------------------------------------------------- thermal
def _node_x(chain: ThermalChain, case: ChainCase) -> np.ndarray:
    """Normalised node abscissa of the chain (interface nodes are duplicated)."""
    length = 2.0 * case.n_spheres * case.radius
    idx = np.arange(chain.n_nodes)
    seg = idx // (chain.m + 1)
    local = idx % (chain.m + 1)
    return (2.0 * case.radius * (seg + local / chain.m)) / length


def run_thermal(outdir: str) -> dict[str, object]:
    out: dict[str, object] = {}

    case = ChainCase()
    chain = ThermalChain(case)
    numeric = chain.solve_steady()
    analytic = chain.analytic_steady()
    span = float(np.ptp(analytic))
    jump = chain.contact_jump(numeric, 0)
    out["contact_law"] = {
        "contact_force_n": case.contact_force,
        "hertz_contact_radius_m": case.contact_radius,
        "contact_constriction_resistance_k_per_w": contact_constriction_resistance(
            case.contact_radius, case.conductivity),
        "sphere_bulk_resistance_k_per_w": sphere_bulk_resistance(case.radius,
                                                                 case.conductivity),
        "chain_resistance_k_per_w": chain_resistance(case.n_spheres,
                                                     case.contact_radius, case.radius,
                                                     case.conductivity),
        "ratios": {
            "contact_over_bulk": contact_constriction_resistance(
                case.contact_radius, case.conductivity)
            / sphere_bulk_resistance(case.radius, case.conductivity),
            "k_eff_over_k": case.conductivity_ratio,
        },
    }
    out["chain_steady"] = {
        "n_spheres": case.n_spheres,
        "cells_per_sphere": case.cells_per_sphere,
        "nodes": int(chain.n_nodes),
        "t_hot_k": case.t_hot,
        "t_cold_k": case.t_cold,
        "heat_flow_w": case.heat_flow,
        "measured_heat_flow_w": chain.measured_heat_flow(numeric),
        "contact_jump_numeric_k": jump,
        "contact_jump_analytic_k": case.heat_flow * contact_constriction_resistance(
            case.contact_radius, case.conductivity),
        "max_abs_error_k": float(np.max(np.abs(numeric - analytic))),
        "max_rel_error": float(np.max(np.abs(numeric - analytic)) / span),
        "k_eff_numeric_w_m_k": chain.measured_effective_conductivity(numeric),
        "k_eff_analytic_w_m_k": case.effective_conductivity,
        "gate": {"criterion": "steady chain temperature vs analytic < 5 %",
                 "value": float(np.max(np.abs(numeric - analytic)) / span),
                 "threshold": 0.05,
                 "passes": bool(np.max(np.abs(numeric - analytic)) / span < 0.05)},
        "profile_x_over_l": _node_x(chain, case).tolist(),
        "profile_numeric_k": numeric.tolist(),
        "profile_analytic_k": analytic.tolist(),
    }

    transient_case = ChainCase(n_spheres=8, cells_per_sphere=4, t_hot=272.15,
                               t_cold=253.15)
    tr = TransientChain(transient_case)
    tau = tr.relaxation_time()
    conv = []
    for steps_per_tau in (500, 1000, 2000):
        res = tr.march(dt=tau / steps_per_tau, n_steps=3 * steps_per_tau,
                       record_every=steps_per_tau)
        exact = tr.analytic_transient(res["time"])
        conv.append({"steps_per_tau": steps_per_tau,
                     "max_rel_error": float(np.max(np.abs(res["nodes"] - exact))
                                            / np.ptp(exact))})
    orders = [math.log(conv[i]["max_rel_error"] / conv[i + 1]["max_rel_error"]) / math.log(2.0)
              for i in range(len(conv) - 1)]
    res = tr.march(dt=tau / 2000, n_steps=6000, record_every=300)
    exact = tr.analytic_transient(res["time"])
    out["chain_transient"] = {
        "n_spheres": transient_case.n_spheres,
        "relaxation_time_s": tau,
        "continuum_relaxation_time_s": tr.continuum_relaxation_time(),
        "convergence": conv,
        "convolution_orders": orders,
        "max_rel_error_at_finest": conv[-1]["max_rel_error"],
        "time_s": res["time"].tolist(),
        "hot_flux_w": res["hot_flux"].tolist(),
        "exact_hot_flux_w": [float(tr.link_conductance[0] * (f[0] - f[1]))
                             for f in exact],
        "sphere_temperature_history": tr.march(dt=tau / 2000, n_steps=6000,
                                              record_every=300)["nodes"][:, -2].tolist(),
    }

    nx, ny, r_link = 9, 6, 1.0
    lattice = lattice_network_temperatures(nx, ny, r_link, 320.0, 280.0)
    exact_r = square_lattice_resistance(nx, ny, r_link)
    q_total = ny * float((lattice[1] - lattice[2]).mean()) / r_link
    out["lattice_2d"] = {
        "nx": nx, "ny": ny, "r_link_k_per_w": r_link,
        "exact_resistance_k_per_w": exact_r,
        "measured_heat_flow_w": q_total,
        "exact_heat_flow_w": 40.0 / exact_r,
        "rel_error": abs(q_total - 40.0 / exact_r) / (40.0 / exact_r),
        "field_k": lattice.tolist(),
    }

    gate_case = ChainCase(n_spheres=6, radius=0.005, contact_force=20.0,
                          t_hot=273.15 - 0.5, t_cold=273.15 - 20.0)
    gate = bond_failure_gate(gate_case, bond_index=1, bond_radius=1e-4,
                             applied_force=0.042, n_steps=20000)
    out["ice_bond_gate"] = {
        **gate.as_dict(),
        "strength_at_0c_pa": ice_bond_tensile_strength(273.15),
        "strength_at_minus10c_pa": ice_bond_tensile_strength(263.15),
        "gate": {"criterion": "simulated vs analytic bond failure time < 5 %",
                 "value": gate.failure_time_error, "threshold": 0.05,
                 "passes": bool(gate.failure_time_error < 0.05)},
        "strength_curve": {
            "t_k": np.linspace(253.15, 273.15, 41).tolist(),
            "sigma_pa": [ice_bond_tensile_strength(t) for t in np.linspace(253.15, 273.15, 41)],
        },
    }
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "thermal.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    return out


# ------------------------------------------------------------------ collapse
def run_collapse_task(index: int, num_tasks: int, outdir: str, *, width: int,
                      radius: float, max_steps: int, mu_rolling: float = 0.0,
                      mu_s: float = 0.36, mu_2: float = 0.84,
                      dt_safety: float = 1.0) -> dict[str, object]:
    tasks = collapse_tasks()
    if not 0 <= index < len(tasks):
        raise SystemExit(f"index {index} outside the {len(tasks)} task table")
    task = tasks[index]
    # the contact-level ledger needs every dissipating channel inside the
    # contact records; the engine keeps its rolling-spring state private, so the
    # ledger case runs without rolling resistance (the rolling configuration is
    # reported separately as an open item)
    case_rolling = 0.0 if task["kind"] == "ledger" else mu_rolling
    case = build_case(task, width=width, radius=radius, max_steps=max_steps,
                      mu_rolling=case_rolling, mu_s=mu_s, mu_2=mu_2,
                      dt_safety=dt_safety)
    with_ledger = task["kind"] == "ledger"
    started = time.time()
    result = column_collapse(case, record_every=200, with_ledger=with_ledger)
    payload = {
        "task": task,
        "name": task_name(task),
        "case": {k: getattr(case, k) for k in
                 ("aspect", "width", "radius", "density", "kn", "restitution",
                  "mu_s", "mu_2", "i_0", "mu_rolling", "gravity", "max_steps",
                  "ke_tol", "ke_hold", "dt_safety",
                  "pile_threshold_diameters")},
        "result": result.as_dict(),
        "front_history": result.front_history,
        "wall_clock_s": time.time() - started,
        "inertial_number_series": None,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"case_{payload['name']}{TAG}.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def assemble(outdir: str) -> dict[str, object]:
    files = sorted(f for f in os.listdir(outdir)
                   if f.startswith("case_") and f.endswith(".json"))
    if not files:
        raise SystemExit("no case_*.json found; run the collapse tasks first")
    cases = []
    for name in files:
        with open(os.path.join(outdir, name)) as handle:
            cases.append(json.load(handle))
    aspect_cases = [c for c in cases if c["task"]["kind"] == "aspect"]
    gravity_cases = [c for c in cases if c["task"]["kind"] == "gravity"]
    ledger_cases = [c for c in cases if c["task"]["kind"] == "ledger"]
    aspect_cases.sort(key=lambda c: c["task"]["aspect"])

    a = [c["result"]["aspect"] for c in aspect_cases]
    rx = [max(c["result"]["relative_runout"], 1e-6) for c in aspect_cases]
    fit_all = fit_power_law(a, rx) if len(a) >= 2 else None
    low = [c for c in aspect_cases if c["task"]["aspect"] <= 3.0]
    high = [c for c in aspect_cases if c["task"]["aspect"] >= 3.0]
    fit_low = (fit_power_law([c["result"]["aspect"] for c in low],
                             [max(c["result"]["relative_runout"], 1e-6) for c in low])
               if len(low) >= 2 else None)
    fit_high = (fit_power_law([c["result"]["aspect"] for c in high],
                              [max(c["result"]["relative_runout"], 1e-6) for c in high])
                if len(high) >= 2 else None)

    verdict_low = (scaling_comparison(fit_low["exponent"],
                                      reference=LAJEUNESSE_RECTANGULAR_EXPONENT_LOW)
                   if fit_low else None)
    verdict_high = (scaling_comparison(fit_high["exponent"],
                                       reference=LAJEUNESSE_RECTANGULAR_EXPONENT_HIGH)
                    if fit_high else None)
    verdict_axis = (scaling_comparison(fit_all["exponent"],
                                       reference=AXISYMMETRIC_EXPONENT)
                    if fit_all else None)

    ledger_gate = None
    if ledger_cases:
        closure = ledger_cases[0]["result"]["energy_ledger"]["closure_error"]
        ledger_gate = {"criterion": "potential-energy / dissipation ledger closure < 2 %",
                       "value": closure, "threshold": 0.02,
                       "passes": bool(closure < 0.02),
                       "case": ledger_cases[0]["name"]}

    threshold_fit = {}
    for key in ("0.5d", "1.0d", "2.0d", "3.0d", "4.0d"):
        vals = [max((c["result"]["runout_thresholds_m"][key] - c["result"]["initial_length_m"])
                    / c["result"]["initial_length_m"], 1e-6) for c in aspect_cases]
        fit = fit_power_law(a, vals) if len(a) >= 2 else None
        threshold_fit[key] = {
            "relative_runout": vals,
            "fit_all": fit,
            "verdict_vs_rectangular_low": (scaling_comparison(
                fit["exponent"], reference=LAJEUNESSE_RECTANGULAR_EXPONENT_LOW)
                if fit else None),
        }

    gravity = sorted(gravity_cases, key=lambda c: c["task"]["gravity_factor"])
    inertial_scan = [{
        "gravity_factor": c["task"]["gravity_factor"],
        "gravity": c["case"]["gravity"],
        "inertial_number_peak": c["result"]["inertial_number_peak"],
        "inertial_number_mean": c["result"]["inertial_number_mean"],
        "mu_flow_mean": c["result"]["mu_flow_mean"],
        "relative_runout": c["result"]["relative_runout"],
        "final_height_over_length": c["result"]["relative_height"],
    } for c in gravity]

    summary = {
        "campaign_node": "p41w3-n04-dem-thermal-granular",
        "repo": "TensorDEM",
        "cases": [{"name": c["name"], "task": c["task"],
                   "relative_runout": c["result"]["relative_runout"],
                   "runout_p99_m": c["result"]["runout_p99_m"],
                   "runout_max_m": c["result"]["runout_max_m"],
                   "runout_thresholds_m": c["result"]["runout_thresholds_m"],
                   "relative_height": c["result"]["relative_height"],
                   "initial_length_m": c["result"]["initial_length_m"],
                   "initial_height_m": c["result"]["initial_height_m"],
                   "steps": c["result"]["steps"],
                   "time_s": c["result"]["time_s"],
                   "free_fall_time_s": c["result"]["free_fall_time_s"],
                   "inertial_number_peak": c["result"]["inertial_number_peak"],
                   "inertial_number_mean": c["result"]["inertial_number_mean"],
                   "mu_flow_mean": c["result"]["mu_flow_mean"],
                   "max_overlap_ratio": c["result"]["max_overlap_ratio"],
                   "energy_ledger": c["result"]["energy_ledger"],
                   "state_digest": c["result"]["state_digest"],
                   "wall_clock_s": c["wall_clock_s"]}
                  for c in cases],
        "fit_all": fit_all,
        "fit_low_aspect": fit_low,
        "fit_high_aspect": fit_high,
        "reference_laws": {
            "lube_2004_axisymmetric": {"exponent": AXISYMMETRIC_EXPONENT,
                                       "form": "r_inf = r_i (1 + 1.6 a^0.5), a > 1.7",
                                       "doi": "10.1017/S0022112004009036"},
            "lajeunesse_2005_rectangular_low": {
                "exponent": LAJEUNESSE_RECTANGULAR_EXPONENT_LOW,
                "form": "dL/L0 ~ a, a < 3", "doi": "10.1063/1.2087687"},
            "lajeunesse_2005_rectangular_high": {
                "exponent": LAJEUNESSE_RECTANGULAR_EXPONENT_HIGH,
                "form": "dL/L0 ~ a^(2/3), a > 3", "doi": "10.1063/1.2087687"},
        },
        "threshold_fit": threshold_fit,
        "verdict_low_aspect": verdict_low,
        "verdict_high_aspect": verdict_high,
        "verdict_vs_axisymmetric": verdict_axis,
        "ledger_gate": ledger_gate,
        "inertial_number_scan": inertial_scan,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _run_one(args_tuple):
    index, width, radius, max_steps, mu_rolling, mu_s, mu_2, outdir, dt_safety = args_tuple
    started = time.time()
    payload = run_collapse_task(index, len(collapse_tasks()), outdir, width=width,
                                radius=radius, max_steps=max_steps,
                                mu_rolling=mu_rolling, mu_s=mu_s, mu_2=mu_2,
                                dt_safety=dt_safety)
    return payload["name"], time.time() - started


def run_scan(outdir: str, *, jobs: int, width: int, radius: float,
             max_steps: int, mu_rolling: float, mu_s: float, mu_2: float,
             dt_safety: float = 1.0) -> None:
    """Run every task of the table with a process pool (one cluster node)."""
    from concurrent.futures import ProcessPoolExecutor

    tasks = [(i, width, radius, max_steps, mu_rolling, mu_s, mu_2, outdir,
              dt_safety)
             for i in range(len(collapse_tasks()))]
    with ProcessPoolExecutor(max_workers=max(1, int(jobs))) as pool:
        for name, seconds in pool.map(_run_one, tasks):
            print(f"{name:24s} {seconds:7.1f} s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["thermal", "collapse", "scan", "assemble"])
    parser.add_argument("--outdir", default="runs/dem_thermal_granular")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=len(collapse_tasks()))
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--mu-rolling", type=float, default=0.0)
    parser.add_argument("--mu-s", type=float, default=0.36)
    parser.add_argument("--mu-2", type=float, default=0.84)
    parser.add_argument("--tag", default="")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--dt-safety", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.mode == "thermal":
        out = run_thermal(args.outdir)
        print(json.dumps({k: (v if not isinstance(v, dict) else "...") for k, v in out.items()},
                         indent=2)[:2000])
    elif args.mode == "scan":
        run_scan(args.outdir, jobs=args.jobs, width=args.width, radius=args.radius,
                 max_steps=args.max_steps, mu_rolling=args.mu_rolling,
                 mu_s=args.mu_s, mu_2=args.mu_2, dt_safety=args.dt_safety)
        summary = assemble(args.outdir)
        print(json.dumps({k: summary[k] for k in
                          ("fit_all", "fit_low_aspect", "fit_high_aspect",
                           "verdict_low_aspect", "ledger_gate")}, indent=2))
    elif args.mode == "collapse":
        global TAG
        TAG = f"_{args.tag}" if args.tag else ""
        payload = run_collapse_task(args.index, args.num_tasks, args.outdir,
                                    width=args.width, radius=args.radius,
                                    max_steps=args.max_steps,
                                    mu_rolling=args.mu_rolling, mu_s=args.mu_s,
                                    mu_2=args.mu_2)
        print(json.dumps({"name": payload["name"],
                          "relative_runout": payload["result"]["relative_runout"],
                          "runout_max_over_l": payload["result"]["runout_max_m"]
                          / payload["result"]["initial_length_m"],
                          "relative_height": payload["result"]["relative_height"],
                          "steps": payload["result"]["steps"],
                          "time_s": payload["result"]["time_s"],
                          "stopped": payload["result"]["stopped"],
                          "peak_ke_fraction": payload["result"]["peak_kinetic_fraction"],
                          "overlap": payload["result"]["max_overlap_ratio"],
                          "Ipeak": payload["result"]["inertial_number_peak"],
                          "wall_clock_s": payload["wall_clock_s"]}, indent=2))
    else:
        summary = assemble(args.outdir)
        print(json.dumps({k: summary[k] for k in
                          ("fit_all", "fit_low_aspect", "fit_high_aspect",
                           "verdict_low_aspect", "ledger_gate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())