"""Channel-clearing width with 6-DOF floe capsize dynamics (IN22-N02).

Runs the campaign scan: coverage ``0.6 / 0.7 / 0.8`` x floe thickness
``0.3 / 0.5 m`` x 3 seeds (18 cases), each in three groups

* ``tilt_locked`` -- the H1 engine (flat floes, heeling locked);
* ``six_dof``     -- full attitude, in-plane (vertical-wall) hull contact;
* ``six_dof_shoulder`` -- full attitude plus the hull-shoulder submersion term
  (``tan 15 deg``) that drives floe subduction under the hull,

at two time steps (8 ms and 4 ms).  It writes a machine-readable artifact with
every acceptance number (injected programmatically, nothing hand-copied), the
three-way cleared-width table and the tables needed by the campaign criteria.

Usage
-----
``python benchmarks/channel_6dof_capsize.py --shard 0 --nshards 8``
    run one slice of the case list, write ``artifacts/channel_6dof_shard_<i>.json``
``python benchmarks/channel_6dof_capsize.py --aggregate``
    merge the shards, evaluate the criteria, write
    ``artifacts/channel_6dof.json`` and the figures
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.floe_6dof import (  # noqa: E402
    CapsizeParameters,
    SixDofFloeSimulation,
    analytic_metacentric_height,
    analytic_roll_stiffness,
    capsize_angle_from_hessian,
    capsize_angle_from_moment,
    capsize_curve,
    disk_quadrature,
    heave_equilibrium_height,
    quat_from_euler,
    quat_identity,
    submerged_hydrostatics,
)
from tensordem.ice_channel import (  # noqa: E402
    HullGeometry,
    IceChannelSimulation,
    WaterEnvironment,
)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
FIGURE_DIR = os.path.join(ARTIFACT_DIR, "channel_6dof")

COVERAGES = (0.6, 0.7, 0.8)
THICKNESSES = (0.3, 0.5)
SEEDS = (11, 12, 13)
GROUPS = ("tilt_locked", "six_dof", "six_dof_shoulder")
TIMESTEPS = (0.008, 0.004)

DOMAIN_LENGTH = 120.0
DOMAIN_WIDTH = 60.0
TRANSIT_SECONDS = 80.0
SHOULDER_TAN = math.tan(math.radians(15.0))
ANALYSIS_QUAD = (24, 32)          # high-order rule for the capsize analysis
SIM_QUAD = (10, 16)               # production rule inside the time stepping


# ---------------------------------------------------------------------------
# Floe field
# ---------------------------------------------------------------------------

def build_field(concentration: float, thickness: float, seed: int) -> Dict[str, object]:
    """Reuse the H1 field generator (hex lattice + coverage inversion)."""
    demo = IceChannelSimulation.from_field(concentration,
                                           domain_length=DOMAIN_LENGTH,
                                           domain_width=DOMAIN_WIDTH,
                                           seed=seed)
    radius = [float(v) for v in demo.radius]
    pos = demo.pos.clone()
    stats = demo.field_statistics()
    return {
        "radius": radius,
        "pos": pos,
        "thickness": thickness,
        "floe_count": len(radius),
        "realised_concentration": float(stats["realised_concentration"]),
        "radius_mean": float(np.mean(radius)),
        "radius_min": float(np.min(radius)),
        "radius_max": float(np.max(radius)),
        "max_initial_overlap": float(stats["max_initial_overlap"]),
        "seed": seed,
    }


def make_simulation(field: Dict[str, object], group: str, dt: float,
                    capsize_angles: Optional[Sequence[float]] = None,
                    quad: Tuple[int, int] = SIM_QUAD,
                    hull_start: float = -40.0) -> SixDofFloeSimulation:
    """Build one simulation for a group (all groups share the initial state)."""
    thickness = float(field["thickness"])
    radius = list(field["radius"])
    n = len(radius)
    mode = "tilt_locked" if group == "tilt_locked" else "six_dof"
    shoulder = SHOULDER_TAN if group == "six_dof_shoulder" else 0.0
    caps = CapsizeParameters(shoulder_submergence=shoulder,
                             quad_radial_order=quad[0],
                             quad_azimuth_order=quad[1])
    sim = SixDofFloeSimulation(radius, [thickness] * n, mode=mode,
                               caps_params=caps,
                               capsize_angles=capsize_angles,
                               hull=HullGeometry(x_start=hull_start),
                               dt=dt)
    sim.pos = field["pos"].clone()
    sim.domain = (DOMAIN_LENGTH, DOMAIN_WIDTH)
    sim.target_concentration = float(field["realised_concentration"])
    sim.set_flat_equilibrium()
    return sim


# ---------------------------------------------------------------------------
# Static capsize analysis (criterion a and the small-angle anchor)
# ---------------------------------------------------------------------------

def capsize_analysis(thickness: float, radius: float = 5.0) -> Dict[str, object]:
    water = WaterEnvironment()
    coarse = disk_quadrature(*ANALYSIS_QUAD)
    moment = capsize_angle_from_moment(radius, thickness, water, coarse,
                                       scan_points=361)
    hessian = capsize_angle_from_hessian(radius, thickness, water, coarse,
                                         scan_points=721)
    curve = capsize_curve(radius, thickness, water, tilt_count=181,
                          tilt_max=0.5 * math.pi, quad=coarse)
    # resolution sensitivity of the fold
    fold_lo = capsize_angle_from_moment(radius, thickness, water,
                                        disk_quadrature(12, 16), scan_points=181)
    fold_hi = capsize_angle_from_moment(radius, thickness, water,
                                        disk_quadrature(36, 48), scan_points=181)
    # small-angle anchor: numeric slope at phi -> 0 against rho_w g V GM
    phi = 1e-4
    q = quat_from_euler([phi], [0.0], [0.0])
    z = heave_equilibrium_height(radius, thickness, q, water, coarse)
    state = submerged_hydrostatics([radius], [thickness], q, [z], water, coarse)
    slope = float(-state.moment[0, 0]) / phi
    analytic = analytic_roll_stiffness(radius, thickness, water.ice_density,
                                       water.water_density, water.gravity)
    # quadrature accuracy of the flat submerged volume against Archimedes
    flat = submerged_hydrostatics([radius], [thickness], quat_identity(1),
                                 [0.0], water, coarse)
    analytic_flat_volume = 0.5 * math.pi * radius ** 2 * thickness
    flat_error = abs(float(flat.volume[0]) - analytic_flat_volume) / analytic_flat_volume
    # force balance at the relaxed equilibrium
    balance = abs(float(state.buoyancy[0, 2] + state.weight[0, 2])) / abs(
        float(state.weight[0, 2]))
    return {
        "thickness": thickness,
        "radius": radius,
        "capsize_angle_moment_deg": math.degrees(moment["capsize_angle"]),
        "capsize_angle_hessian_deg": math.degrees(hessian["capsize_angle"]),
        "capsize_consistency_rel": abs(moment["capsize_angle"]
                                       - hessian["capsize_angle"])
        / hessian["capsize_angle"],
        "peak_righting_moment_nm": moment["peak_righting_moment"],
        "fold_quadrature_12x16_deg": math.degrees(fold_lo["capsize_angle"]),
        "fold_quadrature_36x48_deg": math.degrees(fold_hi["capsize_angle"]),
        "fold_quadrature_spread_rel": abs(fold_lo["capsize_angle"]
                                          - fold_hi["capsize_angle"])
        / fold_hi["capsize_angle"],
        "metacentric_height_m": analytic_metacentric_height(
            radius, thickness, water.ice_density, water.water_density),
        "roll_stiffness_analytic_nm_per_rad": analytic,
        "roll_stiffness_numeric_nm_per_rad": slope,
        "roll_stiffness_rel_error": abs(slope - analytic) / analytic,
        "flat_volume_quadrature_rel_error": flat_error,
        "equilibrium_balance_rel_error": balance,
        "curve": curve.to_dict(),
    }


# ---------------------------------------------------------------------------
# Regression lock (criterion b)
# ---------------------------------------------------------------------------

def digest_lock(concentration: float, thickness: float, seed: int,
                steps: int = 400, dt: float = 0.008) -> Dict[str, object]:
    """Tilt-locked mode must reproduce the H1 digest bit-for-bit."""
    field = build_field(concentration, thickness, seed)
    radius = list(field["radius"])
    n = len(radius)
    reference = IceChannelSimulation(radius, [thickness] * n,
                                     hull=HullGeometry(x_start=-40.0), dt=dt)
    reference.pos = field["pos"].clone()
    locked = SixDofFloeSimulation(radius, [thickness] * n, mode="tilt_locked",
                                 hull=HullGeometry(x_start=-40.0), dt=dt)
    locked.pos = field["pos"].clone()
    for _ in range(int(steps)):
        reference.step()
        locked.step()
    reference_rerun = IceChannelSimulation(radius, [thickness] * n,
                                           hull=HullGeometry(x_start=-40.0), dt=dt)
    reference_rerun.pos = field["pos"].clone()
    for _ in range(int(steps)):
        reference_rerun.step()
    return {
        "concentration": concentration,
        "thickness": thickness,
        "seed": seed,
        "steps": int(steps),
        "dt": dt,
        "h1_digest": reference.state_digest(),
        "h1_rerun_digest": reference_rerun.state_digest(),
        "tilt_locked_digest": locked.state_digest(),
        "digest_identical": bool(reference.state_digest() == locked.state_digest()),
        "rerun_identical": bool(reference.state_digest() == reference_rerun.state_digest()),
    }


# ---------------------------------------------------------------------------
# One channel transit
# ---------------------------------------------------------------------------

def quick_capsize_angle(thickness: float, radius: float = 5.0,
                        quad: Tuple[int, int] = SIM_QUAD) -> float:
    """Fold angle [rad] with the production quadrature (used inside a shard)."""
    return float(capsize_angle_from_moment(radius, thickness, WaterEnvironment(),
                                           disk_quadrature(*quad),
                                           scan_points=181)["capsize_angle"])


def run_case(case: Dict[str, object], group: str, dt: float,
             capsize_angle: float, transit: float,
             quad: Tuple[int, int] = SIM_QUAD) -> Dict[str, object]:
    field = build_field(float(case["concentration"]), float(case["thickness"]),
                        int(case["seed"]))
    n = len(field["radius"])
    sim = make_simulation(field, group, dt, capsize_angles=[float(capsize_angle)] * n,
                          quad=quad)
    record_every = max(1, int(round(0.2 / dt)))
    started = time.time()
    out = sim.run_transit(transit, record_every=record_every)
    elapsed = time.time() - started
    summary = {key: value for key, value in out.items()
               if key not in ("cleared_width_series", "time_series",
                              "max_tilt_series_deg")}
    summary.update({
        "case": case,
        "group": group,
        "dt": dt,
        "floe_count": int(sim.n),
        "realised_concentration": float(field["realised_concentration"]),
        "radius_mean": float(field["radius_mean"]),
        "wall_clock_s": elapsed,
        "cleared_width_final": out["cleared_width_final"],
        "cleared_width_tail_mean": out["cleared_width_tail_mean"],
        "resistance_tail_mean": out["resistance_tail_mean"],
        "capsized_fraction": out["capsized_fraction"],
        "max_tilt_deg": out["max_tilt_deg"],
        "tilt_p95_deg": out["tilt_p95_deg"],
        "quaternion_norm_error": out["quaternion_norm_error"],
        "worst_quaternion_norm_error": out["worst_quaternion_norm_error"],
        "worst_hydrostatic_balance_error": out["worst_hydrostatic_balance_error"],
        "initial_hydrostatic_balance_error": out["initial_hydrostatic_balance_error"],
        "cleared_width_series": out["cleared_width_series"],
        "max_tilt_series_deg": out["max_tilt_series_deg"],
        "time_series": out["time_series"],
    })
    return summary


def case_list() -> List[Dict[str, object]]:
    cases = []
    for coverage in COVERAGES:
        for thickness in THICKNESSES:
            for seed in SEEDS:
                cases.append({"concentration": coverage, "thickness": thickness,
                              "seed": seed})
    return cases


# ---------------------------------------------------------------------------
# Shard / aggregate drivers
# ---------------------------------------------------------------------------

def shard_jobs(cases: Sequence[Dict[str, object]], nshards: int) -> List[List[Dict[str, object]]]:
    """Assign (case, group, dt) jobs to shards in a balanced round-robin."""
    jobs = []
    for case in cases:
        for group in GROUPS:
            for dt in TIMESTEPS:
                jobs.append({"case": case, "group": group, "dt": dt})
    shards: List[List[Dict[str, object]]] = [[] for _ in range(int(nshards))]
    for index, job in enumerate(jobs):
        shards[index % int(nshards)].append(job)
    return shards


def run_shard(index: int, nshards: int, transit: float) -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    cases = case_list()
    folds = {thickness: quick_capsize_angle(thickness) for thickness in THICKNESSES}
    jobs = shard_jobs(cases, nshards)[int(index)]
    results = []
    for job in jobs:
        case = job["case"]
        thickness = float(case["thickness"])
        results.append(run_case(case, job["group"], job["dt"], folds[thickness], transit))
        print(f"[shard {index}] {case} {job['group']} dt={job['dt']} "
              f"W={results[-1]['cleared_width_tail_mean']:.3f} "
              f"caps={results[-1]['capsized_fraction']:.3f} "
              f"({results[-1]['wall_clock_s']:.0f}s)", flush=True)
    payload = {"shard": int(index), "nshards": int(nshards),
               "folds_rad": folds, "runs": results}
    path = os.path.join(ARTIFACT_DIR, f"channel_6dof_shard_{index:02d}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    return path


def _rel_diff(a: float, b: float, floor: float = 1e-12) -> float:
    denominator = max(abs(b), floor)
    return abs(a - b) / denominator


def aggregate(nshards: int) -> str:
    runs: List[Dict[str, object]] = []
    analysis: Dict[str, object] = {}
    for index in range(int(nshards)):
        path = os.path.join(ARTIFACT_DIR, f"channel_6dof_shard_{index:02d}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing shard {path}; run all shards first")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        runs.extend(payload["runs"])
    analysis = {str(thickness): capsize_analysis(thickness)
                for thickness in THICKNESSES}

    # criterion (a) + small-angle anchor
    criterion_a = {
        "per_thickness": {key: {name: value[name] for name in (
            "capsize_angle_moment_deg", "capsize_angle_hessian_deg",
            "capsize_consistency_rel", "peak_righting_moment_nm",
            "fold_quadrature_12x16_deg", "fold_quadrature_36x48_deg",
            "fold_quadrature_spread_rel", "metacentric_height_m",
            "roll_stiffness_analytic_nm_per_rad",
            "roll_stiffness_numeric_nm_per_rad", "roll_stiffness_rel_error",
            "flat_volume_quadrature_rel_error", "equilibrium_balance_rel_error")}
            for key, value in analysis.items()},
        "threshold": 0.02,
    }
    criterion_a["worst_consistency_rel"] = max(
        value["capsize_consistency_rel"] for value in criterion_a["per_thickness"].values())
    criterion_a["passed"] = bool(criterion_a["worst_consistency_rel"] < 0.02)

    # criterion (b): digest lock on three representative fields
    locks = [digest_lock(0.6, 0.5, 11), digest_lock(0.7, 0.5, 12),
             digest_lock(0.8, 0.5, 13), digest_lock(0.8, 0.3, 11)]
    criterion_b = {"cases": locks,
                   "all_identical": all(item["digest_identical"] for item in locks),
                   "all_reruns_identical": all(item["rerun_identical"] for item in locks)}
    criterion_b["passed"] = bool(criterion_b["all_identical"]
                                 and criterion_b["all_reruns_identical"])

    # index the runs
    index: Dict[Tuple[float, float, int, str, float], Dict[str, object]] = {}
    for run in runs:
        case = run["case"]
        index[(float(case["concentration"]), float(case["thickness"]),
               int(case["seed"]), run["group"], float(run["dt"]))] = run

    # criterion (c): time-step sensitivity
    dt_rows = []
    for (coverage, thickness, seed, group, dt) in list(index.keys()):
        if abs(dt - TIMESTEPS[0]) > 1e-12:
            continue
        fine = index.get((coverage, thickness, seed, group, TIMESTEPS[1]))
        if fine is None:
            continue
        coarse = index[(coverage, thickness, seed, group, dt)]
        dt_rows.append({
            "concentration": coverage, "thickness": thickness, "seed": seed,
            "group": group,
            "cleared_width_dt8": coarse["cleared_width_tail_mean"],
            "cleared_width_dt4": fine["cleared_width_tail_mean"],
            "cleared_width_rel_diff": _rel_diff(coarse["cleared_width_tail_mean"],
                                                fine["cleared_width_tail_mean"]),
            "capsize_fraction_dt8": coarse["capsized_fraction"],
            "capsize_fraction_dt4": fine["capsized_fraction"],
            "capsize_fraction_abs_diff": abs(coarse["capsized_fraction"]
                                             - fine["capsized_fraction"]),
            "resistance_dt8": coarse["resistance_tail_mean"],
            "resistance_dt4": fine["resistance_tail_mean"],
            "resistance_rel_diff": _rel_diff(coarse["resistance_tail_mean"],
                                             fine["resistance_tail_mean"]),
            "max_tilt_deg_dt8": coarse["max_tilt_deg"],
            "max_tilt_deg_dt4": fine["max_tilt_deg"],
        })
    # why a single-floe extreme statistic does not converge: the seed-to-seed
    # spread of the same quantity is compared with the time-step spread
    seed_spread = {}
    for group in GROUPS:
        values = [row["cleared_width_tilt_locked"] for row in dt_rows
                  if row["group"] == group and False]
        values = [run["cleared_width_tail_mean"] for run in runs
                  if run["group"] == group and abs(run["dt"] - TIMESTEPS[0]) < 1e-12]
        seed_spread[group] = {"mean": float(np.mean(values)) if values else float("nan"),
                              "std": float(np.std(values, ddof=1)) if len(values) > 1
                              else float("nan"),
                              "rel_std": float(np.std(values, ddof=1) / np.mean(values))
                              if len(values) > 1 else float("nan")}
    criterion_c = {
        "rows": dt_rows,
        "resistance_worst_rel_diff": max((row["resistance_rel_diff"]
                                          for row in dt_rows), default=float("nan")),
        "resistance_median_rel_diff": float(np.median([row["resistance_rel_diff"]
                                                       for row in dt_rows]))
        if dt_rows else float("nan"),
        "cleared_width_median_rel_diff": float(np.median(
            [row["cleared_width_rel_diff"] for row in dt_rows])) if dt_rows else float("nan"),
        "seed_to_seed_spread": seed_spread,
        "worst_cleared_width_rel_diff": max((row["cleared_width_rel_diff"]
                                             for row in dt_rows), default=float("nan")),
        "worst_capsize_fraction_abs_diff": max((row["capsize_fraction_abs_diff"]
                                                for row in dt_rows), default=float("nan")),
        "threshold_cleared_width": 0.05,
        "threshold_capsize": 0.05,
    }
    criterion_c["passed"] = bool(
        criterion_c["worst_cleared_width_rel_diff"] < 0.05
        and criterion_c["worst_capsize_fraction_abs_diff"] < 0.05)

    # criterion (d): three-way comparison table at the production time step
    table = []
    for coverage in COVERAGES:
        for thickness in THICKNESSES:
            for seed in SEEDS:
                row = {"concentration": coverage, "thickness": thickness, "seed": seed}
                for group in GROUPS:
                    run = index.get((coverage, thickness, seed, group, TIMESTEPS[0]))
                    row[f"cleared_width_{group}"] = (run["cleared_width_tail_mean"]
                                                     if run else float("nan"))
                    row[f"capsize_fraction_{group}"] = (run["capsized_fraction"]
                                                        if run else float("nan"))
                    row[f"max_tilt_deg_{group}"] = (run["max_tilt_deg"]
                                                    if run else float("nan"))
                    row[f"resistance_{group}"] = (run["resistance_tail_mean"]
                                                  if run else float("nan"))
                base = row["cleared_width_tilt_locked"]
                for group in ("six_dof", "six_dof_shoulder"):
                    value = row[f"cleared_width_{group}"]
                    row[f"delta_width_{group}"] = value - base
                    row[f"delta_width_rel_{group}"] = (value - base) / base if base else float("nan")
                table.append(row)

    def summarise(rows: List[Dict[str, object]], key: str) -> Dict[str, float]:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))
                  and not math.isnan(row[key])]
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))
                if len(values) > 1 else 0.0, "n": len(values)}

    grouped_table = []
    for coverage in COVERAGES:
        for thickness in THICKNESSES:
            rows = [row for row in table
                    if row["concentration"] == coverage and row["thickness"] == thickness]
            grouped_table.append({
                "concentration": coverage, "thickness": thickness,
                "cleared_width_tilt_locked": summarise(rows, "cleared_width_tilt_locked"),
                "cleared_width_six_dof": summarise(rows, "cleared_width_six_dof"),
                "cleared_width_six_dof_shoulder": summarise(rows, "cleared_width_six_dof_shoulder"),
                "delta_width_rel_six_dof": summarise(rows, "delta_width_rel_six_dof"),
                "delta_width_rel_six_dof_shoulder": summarise(rows, "delta_width_rel_six_dof_shoulder"),
                "capsize_fraction_six_dof": summarise(rows, "capsize_fraction_six_dof"),
                "capsize_fraction_six_dof_shoulder": summarise(rows, "capsize_fraction_six_dof_shoulder"),
                "max_tilt_deg_six_dof_shoulder": summarise(rows, "max_tilt_deg_six_dof_shoulder"),
            })

    # criteria (e) and (f) over every run
    quat_error = max((run["worst_quaternion_norm_error"] for run in runs),
                     default=float("nan"))
    balance_error = max((max(run["initial_hydrostatic_balance_error"],
                              run["worst_hydrostatic_balance_error"]) for run in runs),
                        default=float("nan"))
    analysis_balance_error = max(
        (value["equilibrium_balance_rel_error"] for value in analysis.values()),
        default=float("nan"))
    balance_error = max(balance_error, analysis_balance_error)
    criterion_e = {"worst_quaternion_norm_error": quat_error, "threshold": 1e-12,
                   "passed": bool(quat_error < 1e-12)}
    criterion_f = {"worst_hydrostatic_balance_rel_error": balance_error,
                   "threshold": 1e-10, "passed": bool(balance_error < 1e-10)}

    artifact = {
        "task": "IN22-N02 6-DOF floe capsize dynamics and channel clearing width",
        "module": "src/tensordem/floe_6dof.py",
        "benchmark": "benchmarks/channel_6dof_capsize.py",
        "configuration": {
            "domain_length": DOMAIN_LENGTH, "domain_width": DOMAIN_WIDTH,
            "transit_seconds": TRANSIT_SECONDS,
            "coverage": list(COVERAGES), "thickness": list(THICKNESSES),
            "seeds": list(SEEDS), "groups": list(GROUPS),
            "timesteps": list(TIMESTEPS),
            "shoulder_submergence": SHOULDER_TAN,
            "analysis_quadrature": list(ANALYSIS_QUAD),
            "simulation_quadrature": list(SIM_QUAD),
            "water": {
                "water_density": WaterEnvironment().water_density,
                "ice_density": WaterEnvironment().ice_density,
                "gravity": WaterEnvironment().gravity,
                "wind_velocity": list(WaterEnvironment().wind_velocity),
            },
            "hull": {"length": 80.0, "beam": 20.0, "speed": 1.6, "x_start": -40.0},
        },
        "capsize_analysis": analysis,
        "criteria": {
            "a_capsize_consistency": criterion_a,
            "b_digest_regression_lock": criterion_b,
            "c_timestep_convergence": criterion_c,
            "d_width_table": {"rows": table, "grouped": grouped_table},
            "e_quaternion_norm": criterion_e,
            "f_buoyancy_balance": criterion_f,
        },
        "run_count": len(runs),
        "total_wall_clock_s": float(sum(run["wall_clock_s"] for run in runs)),
    }
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, "channel_6dof.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=1)
    print(json.dumps({name: artifact["criteria"][name].get("passed")
                      for name in ("a_capsize_consistency", "b_digest_regression_lock",
                                   "c_timestep_convergence", "e_quaternion_norm",
                                   "f_buoyancy_balance")}, indent=1))
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--transit", type=float, default=TRANSIT_SECONDS)
    args = parser.parse_args(argv)
    if args.aggregate:
        print(aggregate(args.nshards))
        return 0
    if args.shard is None:
        parser.error("either --shard or --aggregate is required")
    print(run_shard(args.shard, args.nshards, args.transit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())