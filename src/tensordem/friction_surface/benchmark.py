"""Repose-angle benchmark harness for the friction-contact surface (P152W1-N04).

The harness runs the *pre-registered* repose protocol (fixed-height nozzle pour,
bursts paced by the charge fall time -- see
``prereg/P152W1_N04_repose_protocol_v2_prereg.json``) on the DEM kernel in
:mod:`tensordem.friction_surface.repose` and emits a machine-adjudicable
reference-vs-simulation table:

* anchor A -- negative control ``mu = mu_rolling = 0`` must give ``theta < 5 deg``;
* anchor B -- glass beads (calibration of the single shape parameter
  ``mu_rolling``) and quartz sand (out-of-sample prediction) against the
  literature repose bands;
* the ``theta_r(mu)`` surface at the calibrated ``mu_rolling`` (for figures and
  for the friction-angle reading ``mu_eff = tan(theta_r)``).

Design rules
------------
* **Nothing is typed in by hand.**  Thresholds, material micro-parameters and
  experimental bands are read from the pre-registration file; their SHA-256 is
  recorded in the payload so a reader can prove which protocol produced the
  numbers.
* **The kernel is not modified.**  The harness only drives
  ``ReposeCase``/``simulate_repose`` and aggregates the returned scalars.
* **No silent pass.**  A run that fails to reach rest, or that produces a NaN
  angle, is reported as ``WITHHELD`` -- never as a pass by default.

Entry points
------------
    python -m tensordem.friction_surface.benchmark --particles 300 --jobs 12
    python -m tensordem.friction_surface            # same, module entry point
    python -m tensordem.friction_surface.study --stage benchmark

The written artifact defaults to ``artifacts/P152W1-N04/repose_benchmark.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool

import numpy as np

from .repose import ReposeCase, simulate_repose

DEFAULT_PREREG = os.path.join("prereg", "P152W1_N04_repose_protocol_v2_prereg.json")
DEFAULT_OUT = os.path.join("artifacts", "P152W1-N04", "repose_benchmark.json")
SCHEMA = "tensordem.friction_surface.repose_benchmark/v1"

#: bracket of the shape-parameter bisection (pre-registered, not tuned here)
MU_ROLLING_BRACKET = (0.02, 0.40)


# ---------------------------------------------------------------------------
# pre-registration (single source of truth for every threshold)
# ---------------------------------------------------------------------------

def _candidate_prereg_paths(path: str | None):
    if path:
        yield path
        return
    yield DEFAULT_PREREG
    here = os.path.dirname(os.path.abspath(__file__))
    # src/tensordem/friction_surface -> repository root
    yield os.path.join(here, "..", "..", "..", DEFAULT_PREREG)


def load_prereg(path: str | None = None) -> dict:
    """Load the pre-registration document; raise a useful error if absent."""
    tried = []
    for cand in _candidate_prereg_paths(path):
        cand = os.path.normpath(cand)
        tried.append(cand)
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as fh:
                raw = fh.read()
            doc = json.loads(raw)
            doc["_path"] = cand
            doc["_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            return doc
    raise FileNotFoundError(
        "pre-registration not found; tried: " + ", ".join(tried)
        + " (run from the repository root or pass --prereg)"
    )


def _criteria(prereg: dict) -> tuple[dict, dict, float]:
    """Return ``(negative_control_threshold, materials, rel_err_max)``."""
    by_id = {c["id"]: c for c in prereg["criteria_inherited"]}
    control = by_id["A-repose-negative-control"]["threshold"]
    experiment = by_id["B-repose-experiment"]
    return control, experiment["materials"], float(
        experiment["threshold"]["relative_error_max"])


# ---------------------------------------------------------------------------
# one DEM repose run (worker; must stay picklable for multiprocessing)
# ---------------------------------------------------------------------------

def run_case(mu: float, mu_rolling: float, seed: int, n_particles: int,
             source_mode: str = "fixed", keep_positions: bool = False,
             max_time: float | None = None) -> dict:
    """Run one repose case and return the scalars the anchor needs."""
    kwargs = {}
    if max_time is not None:
        kwargs["max_time"] = float(max_time)
    case = ReposeCase(mu=mu, mu_rolling=mu_rolling, seed=seed,
                      n_particles=n_particles, source_mode=source_mode, **kwargs)
    res = simulate_repose(case)
    positions = res.pop("final_positions", None)
    out = {
        "mu": mu,
        "mu_rolling": mu_rolling,
        "seed": seed,
        "theta_r_deg": res["theta_r_deg"],
        "reached_rest": res["reached_rest"],
        "pour_closed": res["pour_closed"],
        "n_particles": res["n_particles"],
        "n_particles_requested": case.n_particles,
        "t_end": res["t_end"],
        "dt": res.get("dt"),
        "apex_height_m": res["apex_height_m"],
        "nozzle_z_max": res["nozzle_z_max"],
        "bursts_released": res["bursts_released"],
        "bursts_skipped": res["bursts_skipped"],
        "measurement": res["measurement"],
        "source_mode": res["source_mode"],
    }
    if keep_positions and positions is not None:
        out["final_positions"] = np.asarray(positions, dtype=float)
    return out


def _run_pool(pool, jobs) -> list[dict]:
    return pool.map(_run_pool_job, jobs)


def _run_pool_job(args) -> dict:
    return run_case(*args)


# ---------------------------------------------------------------------------
# plan + benchmark driver
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkPlan:
    """Compute budget of one benchmark run."""

    particles: int = 300
    bisection_steps: int = 5
    seeds: tuple = (15204, 15205, 15206)
    surface_mus: tuple = (0.1, 0.2, 0.4, 0.6)
    source_mode: str = "fixed"
    jobs: int = 12
    budget: str = "full"
    keep_positions: bool = False
    max_time: float | None = None
    notes: list = field(default_factory=list)


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and float(v) == float(v)]


def _mean_theta(runs) -> tuple[float, float, int]:
    vals = _finite(r["theta_r_deg"] for r in runs)
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def _jsonable(obj):
    """Convert numpy scalars/arrays and non-finite floats to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5, check=False).stdout.strip() or "unknown"
    except Exception:                                    # pragma: no cover - env dependent
        return "unknown"


def run_benchmark(plan: BenchmarkPlan | None = None,
                  prereg: dict | None = None) -> dict:
    """Execute the benchmark and return the adjudicable payload."""
    plan = plan or BenchmarkPlan()
    prereg = prereg or load_prereg()
    control_thr, materials, rel_err_max = _criteria(prereg)
    theta_max_control = float(control_thr["theta_r_deg_max"])
    fitted_on = prereg.get("protocol_v2", {}).get("calibration", "")
    fitted_material = "glass_beads" if "glass" in fitted_on else sorted(materials)[0]

    n = int(plan.particles)
    seeds = tuple(int(s) for s in plan.seeds)
    if not seeds:
        raise ValueError("plan.seeds must not be empty")
    t0 = time.time()
    with Pool(max(1, min(plan.jobs, plan.particles, 32))) as pool:
        # --- anchor A: negative control ------------------------------------
        control_jobs = [(0.0, 0.0, s, n, plan.source_mode, False, plan.max_time)
                        for s in seeds]
        control = _run_pool(pool, control_jobs)

        # --- anchor B: bisection of mu_rolling on the fitted material -------
        mu_fit = float(materials[fitted_material]["mu"])
        target = float(materials[fitted_material]["theta_exp_deg"])
        lo, hi = MU_ROLLING_BRACKET
        history = []
        for _ in range(int(plan.bisection_steps)):
            mid = 0.5 * (lo + hi)
            runs = _run_pool(pool, [(mu_fit, mid, s, n, plan.source_mode, False,
                                     plan.max_time) for s in seeds])
            theta, std, count = _mean_theta(runs)
            history.append({"mu_rolling": mid, "theta_r_deg": None if theta != theta else theta,
                            "theta_std_deg": None if std != std else std,
                            "n_runs": count, "target_deg": target})
            if theta == theta:                     # NaN never moves the bracket
                if theta < target:
                    lo = mid
                else:
                    hi = mid
        mu_rolling_star = 0.5 * (lo + hi)

        # --- final runs: every material, every seed ------------------------
        final_jobs = [(float(mat["mu"]), mu_rolling_star, s, n, plan.source_mode,
                       False, plan.max_time)
                      for mat in materials.values() for s in seeds]
        finals = _run_pool(pool, final_jobs)

        # --- theta(mu) surface at the calibrated shape parameter -----------
        surface_mus = [float(m) for m in plan.surface_mus
                       if all(abs(float(m) - float(mat["mu"])) > 1e-9
                              for mat in materials.values())]
        surface = _run_pool(pool, [(m, mu_rolling_star, seeds[0], n, plan.source_mode,
                                    False, plan.max_time) for m in surface_mus])

        # --- one run per material with positions kept (figure data) --------
        keep = []
        if plan.keep_positions:
            keep = _run_pool(pool, [(float(mat["mu"]), mu_rolling_star, seeds[0], n,
                                     plan.source_mode, True, plan.max_time)
                                    for mat in materials.values()])
    wall = time.time() - t0

    # ------------------------- adjudication -------------------------------
    control_vals = _finite(r["theta_r_deg"] for r in control)
    control_mean = float(np.mean(control_vals)) if control_vals else float("nan")
    control_ok = bool(control_vals) and control_mean < theta_max_control
    verdicts = [{
        "id": "A-repose-negative-control",
        "reference": {"theta_r_deg_max": theta_max_control,
                      "source": prereg["_path"]},
        "got": {"theta_r_deg": None if control_mean != control_mean else control_mean,
                "runs": control_vals, "n_runs": len(control_vals)},
        "status": "PASS" if control_ok else ("WITHHELD" if not control_vals else "FAIL"),
        "note": "mu = mu_rolling = 0 must give a flat heap",
    }]

    table = {}
    for name, mat in materials.items():
        sel = [r for r in finals if abs(float(r["mu"]) - float(mat["mu"])) < 1e-9]
        mean, std, count = _mean_theta(sel)
        ref = float(mat["theta_exp_deg"])
        band = [float(mat["band"][0]), float(mat["band"][1])]
        rel = abs(mean - ref) / ref if mean == mean else float("nan")
        if not count:
            status = "WITHHELD"
        elif rel <= rel_err_max:
            status = "PASS"
        else:
            status = "FAIL"
        table[name] = {
            "mu": float(mat["mu"]),
            "mu_rolling": mu_rolling_star,
            "reference_deg": ref,
            "reference_band_deg": band,
            "reference_source": mat.get("note", "literature value from the pre-registration"),
            "got_deg": None if mean != mean else mean,
            "got_std_deg": None if std != std else std,
            "got_runs_deg": [None if v != v else v
                             for v in (r["theta_r_deg"] for r in sel)],
            "n_runs": count,
            "abs_error_deg": None if mean != mean else abs(mean - ref),
            "relative_error": None if rel != rel else rel,
            "relative_error_max": rel_err_max,
            "within_band": None if mean != mean else bool(band[0] <= mean <= band[1]),
            "mu_eff_sim": None if mean != mean else math.tan(math.radians(mean)),
            "mu_eff_reference": math.tan(math.radians(ref)),
            "out_of_sample": name != fitted_material,
            "status": status,
        }
        verdicts.append({
            "id": f"B-repose-experiment/{name}",
            "reference": {"theta_exp_deg": ref, "band_deg": band,
                          "relative_error_max": rel_err_max},
            "got": {"theta_r_deg": table[name]["got_deg"],
                    "theta_std_deg": table[name]["got_std_deg"],
                    "n_runs": count},
            "relative_error": table[name]["relative_error"],
            "status": status,
        })

    statuses = [v["status"] for v in verdicts]
    overall = ("FAIL" if "FAIL" in statuses
               else ("PASS" if all(s == "PASS" for s in statuses) else "PARTIAL"))

    payload = {
        "schema": SCHEMA,
        "node": prereg.get("node", "P152W1-N04"),
        "prereg": {"path": prereg["_path"], "sha256": prereg["_sha256"]},
        "plan": asdict(plan),
        "provenance": {
            "command": " ".join([sys.executable] + list(sys.argv)),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.node(),
            "wall_seconds": round(wall, 3),
            "started_unix": round(t0, 3),
            "cwd": os.getcwd(),
        },
        "negative_control": {
            "mu": 0.0, "mu_rolling": 0.0,
            "theta_r_deg": None if control_mean != control_mean else control_mean,
            "runs_deg": control_vals,
            "criterion_theta_r_deg_max": theta_max_control,
        },
        "calibration": {
            "parameter": "mu_rolling (single shape surrogate)",
            "fitted_on": fitted_material,
            "target_deg": target,
            "bracket": list(MU_ROLLING_BRACKET),
            "bisection_steps": int(plan.bisection_steps),
            "bisection_history": history,
            "mu_rolling_star": mu_rolling_star,
        },
        "materials": table,
        "verdicts": verdicts,
        "overall": overall,
        "runs": {"control": control, "finite": finals, "surface": surface},
        "theta_mu_surface": [{"mu": float(r["mu"]),
                              "theta_r_deg": None if r["theta_r_deg"] != r["theta_r_deg"]
                              else float(r["theta_r_deg"])} for r in surface],
        "figure_positions": {
            name: (None if not keep else
                   np.asarray(keep[i].get("final_positions")).tolist())
            for i, name in enumerate(materials)
        },
        "honest_note": ("WITHHELD means the anchor produced no adjudicable number "
                        "(NaN angle or no run reached rest); it is neither a pass "
                        "nor a pass by default."),
    }
    return _jsonable(payload)


def write_payload(payload: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    return path


def summarise(payload: dict) -> str:
    lines = [f"overall: {payload['overall']}"]
    for v in payload["verdicts"]:
        got = v["got"]
        if "theta_r_deg" in got:
            val = "n/a" if got["theta_r_deg"] is None else f"{got['theta_r_deg']:.2f} deg"
        else:
            val = "n/a"
        lines.append(f"  {v['id']:<34} {v['status']:<9} got={val}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Repose-angle benchmark (anchor A/B of node P152W1-N04): "
                    "pre-registered thresholds vs DEM output.")
    ap.add_argument("--particles", type=int, default=300,
                    help="spheres in the pour (pre-registered production value: 300)")
    ap.add_argument("--bisection-steps", type=int, default=5,
                    help="shape-parameter bisection steps (pre-registered: 5)")
    ap.add_argument("--seeds", default="15204,15205,15206",
                    help="comma-separated seeds (pre-registered: 15204,15205,15206)")
    ap.add_argument("--surface-mus", default="0.1,0.2,0.4,0.6",
                    help="extra friction coefficients for the theta(mu) surface")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--source-mode", default="fixed", choices=["fixed", "rising"])
    ap.add_argument("--max-time", type=float, default=None,
                    help="override the per-run simulated-time cap (tests only)")
    ap.add_argument("--budget", default=None,
                    help="label written into the artifact (e.g. reduced)")
    ap.add_argument("--keep-positions", action="store_true",
                    help="store the final deposits (figure data; file gets large)")
    ap.add_argument("--prereg", default=None, help="pre-registration JSON path")
    ap.add_argument("--out", default=DEFAULT_OUT)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    surface = tuple(float(m) for m in args.surface_mus.split(",") if m.strip())
    prereg = load_prereg(args.prereg)
    plan = BenchmarkPlan(particles=args.particles,
                         bisection_steps=args.bisection_steps,
                         seeds=seeds, surface_mus=surface,
                         source_mode=args.source_mode, jobs=args.jobs,
                         budget=args.budget or ("full" if args.particles >= 300 else "reduced"),
                         keep_positions=args.keep_positions,
                         max_time=args.max_time)
    payload = run_benchmark(plan, prereg)
    write_payload(payload, args.out)
    print(summarise(payload))
    print("artifact:", args.out)
    return 0


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())