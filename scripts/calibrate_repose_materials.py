#!/usr/bin/env python
"""Drive the material-class repose-angle calibration closed loop.

Usage
-----
    python scripts/calibrate_repose_materials.py --stage scan      --out artifacts/repose_material_calibration
    python scripts/calibrate_repose_materials.py --stage calibrate --out artifacts/repose_material_calibration
    python scripts/calibrate_repose_materials.py --stage all       --out artifacts/repose_material_calibration

Stages
------
``scan``       mu in [0.2, 0.8] x seeds x materials  -> one JSON per run.
``calibrate``  aggregate scan runs -> monotone table -> invert theta_target ->
               verification runs at mu* + negative control + mu_r sensitivity +
               stiffness check + analytic anchors -> ``calibration.json``.
``all``        both, in order.

Every run file carries the audit chain (git commit, torch version, host, spec).
The heavy lifting stays in :mod:`tensordem.repose_material_calibration`; this
script only orchestrates, parallelises (``--workers``) and serialises.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import platform
import socket
import subprocess
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _audit() -> dict:
    import torch
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         text=True).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    return {
        "commit": commit,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "host": socket.gethostname(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "module": "tensordem.repose_material_calibration",
    }


def _run_one(job: dict) -> dict:
    """Worker entry point (picklable, module level)."""
    import torch

    torch.set_num_threads(1)
    from tensordem import repose_material_calibration as rmc

    spec = rmc.material(job["material"])
    overrides = dict(job.get("spec_overrides") or {})
    if overrides:
        spec = rmc.MaterialSpec(**{**spec.__dict__, **overrides})
    if job["kind"] == "negative_control":
        run = rmc.negative_control(spec, seed=job["seed"])
    else:
        run = rmc.simulate(spec, job["mu"], seed=job["seed"],
                           mu_rolling=job.get("mu_rolling"),
                           keep_particles=bool(job.get("keep_particles")))
    positions = run.pop("positions", None)
    record = {
        "job": job,
        "audit": _audit(),
        "result": run,
    }
    if positions is not None:
        record["positions_xyz"] = positions.tolist()
    return record


def _path(out: str, job: dict) -> str:
    tag = f"{job['material']}__{job['kind']}__mu{job['mu']:.3f}"
    if job.get("mu_rolling") is not None:
        tag += f"__mur{job['mu_rolling']:.3f}"
    tag += f"__seed{job['seed']}"
    if job.get("spec_overrides"):
        tag += "__" + "_".join(f"{k}{v}" for k, v in sorted(job["spec_overrides"].items()))
    return os.path.join(out, "runs", tag + ".json")


def _execute(jobs: list[dict], out: str, workers: int, skip_existing: bool = True) -> int:
    pending = [job for job in jobs
               if not (skip_existing and os.path.exists(_path(out, job)))]
    print(f"[runner] {len(jobs)} jobs, {len(jobs) - len(pending)} cached, "
          f"{len(pending)} to run, workers={workers}", flush=True)
    if not pending:
        return 0
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    done = 0
    with Pool(workers) as pool:
        for record in pool.imap_unordered(_run_one, pending):
            path = _path(out, record["job"])
            with open(path, "w") as fh:
                json.dump(record, fh, indent=1, sort_keys=True)
            done += 1
            res = record["result"]
            print(f"[runner] {done}/{len(pending)} {os.path.basename(path)} -> "
                  f"theta={res.get('theta_r_deg', float('nan')):.2f} deg "
                  f"(r2={res.get('r2', float('nan')):.3f})", flush=True)
    return done


#: Diagnostic configurations that bound the *scope* of the calibration result:
#: pile size, rolling-resistance prior, deposition rate and contact stiffness.
#: Every entry is a documented, reproducible run (same runner, same audit chain)
#: whose purpose is to show which knob moves the repose angle and by how much.
EVIDENCE_JOBS: list[dict] = [
    {"kind": "evidence", "material": "glass_bead", "mu": 0.5, "seed": 39907,
     "label": "baseline"},
    {"kind": "evidence", "material": "glass_bead", "mu": 0.5, "seed": 39907,
     "label": "mu_r_x2", "mu_rolling": 1.30},
    {"kind": "evidence", "material": "gravel", "mu": 0.6, "seed": 39907,
     "label": "lift_x0.4",
     "spec_overrides": {"lift_time": 0.48}},
    {"kind": "evidence", "material": "gravel", "mu": 0.6, "seed": 39907,
     "label": "kn_x4",
     "spec_overrides": {"kn": 2000.0}},
]


def scan_jobs(materials: list[str], mu_nodes: list[float], seeds: list[int]) -> list[dict]:
    jobs: list[dict] = []
    for key in materials:
        for mu in mu_nodes:
            for seed in seeds:
                jobs.append({"kind": "scan", "material": key, "mu": float(mu),
                             "seed": int(seed)})
    return jobs


def load_scan(out: str, materials: list[str]) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {key: [] for key in materials}
    for path in sorted(glob.glob(os.path.join(out, "runs", "*__scan__*.json"))):
        with open(path) as fh:
            record = json.load(fh)
        key = record["job"]["material"]
        if key in runs and record["job"].get("spec_overrides") is None:
            runs[key].append(record["result"])
    for key, group in runs.items():
        if not group:
            raise SystemExit(f"no scan runs for material {key!r} in {out}/runs")
    return runs


def calibration_jobs(plans: dict[str, dict], verify_seeds: list[int]) -> list[dict]:
    jobs: list[dict] = []
    for key, plan in plans.items():
        jobs.append({"kind": "negative_control", "material": key, "mu": 0.0,
                     "mu_rolling": 0.0, "seed": verify_seeds[0]})
        for seed in verify_seeds:
            jobs.append({"kind": "verification", "material": key,
                         "mu": plan["mu_star"], "seed": int(seed),
                         "keep_particles": seed == verify_seeds[0]})
        for factor in (0.5, 1.0, 1.5):
            jobs.append({"kind": "sensitivity", "material": key, "mu": plan["mu_star"],
                         "mu_rolling": round(plan["mu_rolling"] * factor, 10),
                         "seed": verify_seeds[0]})
        for factor in (1.0, 4.0):
            jobs.append({"kind": "stiffness", "material": key, "mu": plan["mu_star"],
                         "seed": verify_seeds[0],
                         "spec_overrides": {"kn": plan["kn"] * factor}})
    return jobs


def _load(out: str, job: dict) -> dict:
    with open(_path(out, job)) as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",
                        choices=("scan", "calibrate", "evidence", "all"), default="all")
    parser.add_argument("--out", default="artifacts/repose_material_calibration")
    parser.add_argument("--materials", default="glass_bead,sand,gravel")
    parser.add_argument("--mu-nodes", type=int, default=7)
    parser.add_argument("--mu-nodes-list", default="",
                        help="explicit comma separated mu nodes, overrides --mu-nodes")
    parser.add_argument("--seeds", default="39907,39908")
    parser.add_argument("--verify-seeds", default="39909,39910")
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    args = parser.parse_args(argv)

    from tensordem import repose_material_calibration as rmc

    materials = [m.strip() for m in args.materials.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    verify_seeds = [int(s) for s in args.verify_seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)

    if args.stage in ("scan", "all"):
        nodes = ([float(v) for v in args.mu_nodes_list.split(",")]
                 if args.mu_nodes_list else rmc.default_mu_nodes(count=args.mu_nodes))
        jobs = scan_jobs(materials, nodes, seeds)
        _execute(jobs, args.out, args.workers)
        with open(os.path.join(args.out, "scan_plan.json"), "w") as fh:
            json.dump({"mu_nodes": nodes, "seeds": seeds, "materials": materials,
                       "n_jobs": len(jobs), "audit": _audit()}, fh, indent=1)

    if args.stage in ("evidence", "all"):
        _execute(EVIDENCE_JOBS, args.out, args.workers)

    if args.stage in ("calibrate", "all"):
        scan = load_scan(args.out, materials)
        plans: dict[str, dict] = {}
        tables: dict[str, dict] = {}
        for key in materials:
            spec = rmc.material(key)
            node_rows = rmc.aggregate_nodes(scan[key])
            table, table_report = rmc.build_table(spec, node_rows)
            inside = table.theta_deg[0] <= spec.target_theta_deg <= table.theta_deg[-1]
            plans[key] = {
                # verification/diagnostic runs are always executed; when the
                # target is outside the bracket they probe the closest reachable
                # operating point, and the calibration step reports the FAIL
                "mu_star": (float(table.mu_from_theta(spec.target_theta_deg))
                            if inside else float(table.mu[-1])),
                "target_inside_bracket": bool(inside),
                "mu_rolling": spec.mu_rolling, "kn": spec.kn,
            }
            tables[key] = {"table": table, "report": table_report,
                           "nodes": node_rows, "spec": spec}
        jobs = calibration_jobs(plans, verify_seeds)
        _execute(jobs, args.out, args.workers)

        payload = {"kind": "tensordem-repose-material-calibration-v1",
                   "audit": _audit(), "materials": {}}
        verdicts: list[bool] = []
        cones = [rmc.analytic_cone_check(20.0), rmc.analytic_cone_check(35.0)]
        onset = [rmc.sliding_onset_angle(0.6, 0.2), rmc.sliding_onset_angle(0.25, 0.5)]
        for key in materials:
            spec = rmc.material(key)
            bundle = tables[key]
            table, table_report = bundle["table"], bundle["report"]
            entry: dict = {
                "material": spec.to_json(),
                "scan_nodes": bundle["nodes"],
                "table": table_report,
                "analytic_anchors": {"cone": cones, "slider_onset": onset},
            }
            entry["negative_control"] = _load(args.out, {
                "kind": "negative_control", "material": key, "mu": 0.0,
                "mu_rolling": 0.0, "seed": verify_seeds[0]})["result"]
            try:
                entry["calibration"] = rmc.calibrate(spec, table, seeds=verify_seeds)
            except rmc.TargetOutOfBracketError as exc:
                entry["calibration"] = {"material": key, "passed": False,
                                        "error": str(exc),
                                        "target_theta_deg": spec.target_theta_deg,
                                        "bracket": table.bracket()}
            entry["sensitivity"] = [
                _load(args.out, {"kind": "sensitivity", "material": key,
                                 "mu": plans[key]["mu_star"],
                                 "mu_rolling": round(spec.mu_rolling * f, 10),
                                 "seed": verify_seeds[0]})["result"]
                for f in (0.5, 1.0, 1.5)]
            entry["stiffness"] = [
                _load(args.out, {"kind": "stiffness", "material": key,
                                 "mu": plans[key]["mu_star"], "seed": verify_seeds[0],
                                 "spec_overrides": {"kn": spec.kn * f}})["result"]
                for f in (1.0, 4.0)]
            cal = entry["calibration"]
            checks = {
                "angle_within_10pct": bool(cal.get("passed", False)),
                "negative_control_below_5deg": bool(entry["negative_control"]["passed"]),
                "mu_star_inside_scan_bracket": bool(
                    cal.get("mu_star") is not None
                    and rmc.MU_LOWER <= float(cal.get("mu_star", float("nan"))) <= rmc.MU_UPPER),
                "stiffness_independent_under_2deg": bool(
                    abs(entry["stiffness"][1]["theta_r_deg"]
                        - entry["stiffness"][0]["theta_r_deg"]) < 2.0) if len(entry["stiffness"]) == 2 else False,
                "cone_estimator_within_1p5deg": all(c["passed"] for c in cones),
                "slider_onset_within_0p3deg": all(o["passed"] for o in onset),
            }
            bracket_hi = table.theta_deg[-1]
            closest_mu = table.mu[-1]
            entry["target_inside_bracket"] = plans[key]["target_inside_bracket"]
            entry["closest_achievable"] = {
                "theta_deg": bracket_hi,
                "mu": closest_mu,
                "gap_deg": float(bracket_hi - spec.target_theta_deg),
                "gap_rel": float((bracket_hi - spec.target_theta_deg)
                                 / spec.target_theta_deg),
                "note": ("target above the scanned bracket: the DEM set-up cannot "
                         "reach it inside mu <= 0.8 (honest FAIL, no extrapolation)"),
            }
            entry["checks"] = checks
            verdicts.extend(checks.values())
            payload["materials"][key] = entry
        payload["evidence"] = []
        for job in EVIDENCE_JOBS:
            if not os.path.exists(_path(args.out, job)):
                payload["evidence"].append({
                    "label": job["label"], "material": job["material"],
                    "mu": job["mu"], "mu_rolling": job.get("mu_rolling"),
                    "spec_overrides": job.get("spec_overrides", {}),
                    "theta_r_deg": None, "r2": None, "n_particles": None,
                    "max_speed_m_s": None, "status": "not_run"})
                continue
            record = _load(args.out, job)
            payload["evidence"].append({
                "label": job["label"], "material": job["material"],
                "mu": job["mu"], "mu_rolling": job.get("mu_rolling"),
                "spec_overrides": job.get("spec_overrides", {}),
                "theta_r_deg": record["result"]["theta_r_deg"],
                "r2": record["result"]["r2"],
                "n_particles": record["result"]["n_particles"],
                "max_speed_m_s": record["result"].get("max_speed_m_s"),
            })
        payload["verdict"] = {"all_checks_passed": bool(all(verdicts)),
                              "n_checks": len(verdicts),
                              "n_passed": int(sum(verdicts))}
        with open(os.path.join(args.out, "calibration.json"), "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print("[runner] verdict:", payload["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())