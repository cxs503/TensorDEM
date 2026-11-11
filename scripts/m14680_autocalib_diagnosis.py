"""GAP-DEM-AUTOCALIB round 1: is the repose forward model calibratable?

The mission asks for the closed loop

    micro parameters (stiffness / friction / damping)
        -> macro responses (angle of repose / elastic modulus)
        -> optimiser (BayesOpt or Nelder-Mead) against literature experiments
        -> prediction error of the angle < 10 %

Before an optimiser can be pointed at literature data, the *forward map* has to
be (i) at rest when the response is measured, and (ii) monotone in the
parameter that is supposed to explain the response.  This script measures both
properties of the two forward protocols shipped with :mod:`tensordem.dem_calibration`
and writes the numbers to ``artifacts/gap_dem_autocalib/diagnosis.json``.

Three experiments, all deterministic and seeded:

``P1 friction response``  pluviation heap, mu = 0.20 ... 0.85, everything else
                          fixed -> theta_r(mu) must increase.
``P2 replica noise``      same parameters, three seeds -> the run-to-run spread
                          is the noise floor any calibration must beat.
``P3 rest criterion``     lifting-cylinder collapse (a) on the fixed time
                          window used by the shipped pipeline and (b) with the
                          adaptive ``ke_stop`` termination of
                          :func:`tensordem.dem_calibration.heap.run_heap`.

Usage (fast, ~1 min)::

    PYTHONPATH=src python scripts/m14680_autocalib_diagnosis.py

Usage (writes the artefact only)::

    PYTHONPATH=src python scripts/m14680_autocalib_diagnosis.py --json-only
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import time

import numpy as np

from tensordem.dem_calibration.heap import build_pile, run_heap
from tensordem.dem_calibration.materials import MATERIALS

#: Pluviation parameters of the shipped pipeline (materials._PROTOCOL).
MU_SWEEP = (0.20, 0.35, 0.52, 0.70, 0.85)
SEED_REPLICATES = (0, 1, 2)
SETTLE_CHECK_EVERY = 25
KE_STOP = 0.02
ARTIFACT = os.path.join("artifacts", "gap_dem_autocalib", "diagnosis.json")


def _rel(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / abs(float(b)) if b else float("nan")


def friction_response(spec) -> list[dict]:
    """P1: repose angle versus sliding friction at fixed e and k_n/k_s."""
    rows = []
    for mu in MU_SWEEP:
        case = spec.case("calibration", {"mu": mu, "restitution": 0.38,
                                         "stiffness_ratio": 2.2})
        t0 = time.time()
        out = run_heap(case)
        rows.append({
            "mu": mu,
            "theta_r_deg": float(out["theta_r_deg"]),
            "phi_solid": float(out["phi_solid"]),
            "r2": float(out["r2"]),
            "ke_ratio": float(out["ke_ratio"]),
            "n_particles": int(out["n_particles"]),
            "wall_seconds": round(time.time() - t0, 2),
        })
    return rows


def replica_noise(spec) -> list[dict]:
    """P2: identical parameters, different seeds -> the noise floor."""
    rows = []
    for offset in SEED_REPLICATES:
        case = spec.case("calibration", {"mu": 0.52, "restitution": 0.38,
                                         "stiffness_ratio": 2.2})
        case = dataclasses.replace(case, seed=case.seed + 1000 * offset)
        out = run_heap(case)
        rows.append({"seed_offset": 1000 * offset,
                     "theta_r_deg": float(out["theta_r_deg"]),
                     "digest": out["digest"]})
    return rows


def rest_criterion(spec, *, lift_ratio: float = 4.0, settle_ratio: float = 6.0,
                   max_steps: int = 40000) -> dict:
    """P3: fixed window versus adaptive ``ke_stop`` termination."""
    base = spec.case("calibration")
    tau = math.sqrt(base.column_height / base.gravity)
    rows = []
    for mu in (0.40, 0.65):
        case = dataclasses.replace(base, shape="column", mu=mu, restitution=0.40,
                                   stiffness_ratio=2.0,
                                   lift_time=lift_ratio * tau,
                                   settle_time=settle_ratio * tau)
        fixed = run_heap(case)
        adaptive = run_heap(case, ke_stop=KE_STOP,
                            settle_check_every=SETTLE_CHECK_EVERY,
                            max_steps=max_steps)
        rows.append({
            "mu": mu,
            "lift_time_s": case.lift_time,
            "settle_time_s": case.settle_time,
            "fixed_window": {
                "theta_r_deg": float(fixed["theta_r_deg"]),
                "r2": float(fixed["r2"]),
                "phi_solid": float(fixed["phi_solid"]),
                "ke_ratio": float(fixed["ke_ratio"]),
                "steps": int(fixed["steps_used"]),
                "settled": bool(fixed["settled"]),
            },
            "ke_stop_window": {
                "ke_stop": KE_STOP,
                "theta_r_deg": float(adaptive["theta_r_deg"]),
                "r2": float(adaptive["r2"]),
                "phi_solid": float(adaptive["phi_solid"]),
                "ke_ratio": float(adaptive["ke_ratio"]),
                "steps": int(adaptive["steps_used"]),
                "settled": bool(adaptive["settled"]),
            },
        })
    return {"lift_ratio": lift_ratio, "settle_ratio": settle_ratio,
            "tau_s": tau, "rows": rows}


def summarise(spec, p1, p2, p3) -> dict:
    """Machine-checkable verdicts: monotone? noise floor? at rest?"""
    thetas = [r["theta_r_deg"] for r in p1]
    diffs = np.diff(thetas)
    # Spearman rank correlation of theta on mu, sign only (no scipy needed for
    # a 5-point monotonicity check).
    order_mu = np.argsort([r["mu"] for r in p1])
    ranks = np.empty(len(p1))
    ranks[order_mu] = np.arange(1, len(p1) + 1)
    order_th = np.argsort(thetas)
    rank_th = np.empty(len(p1))
    rank_th[order_th] = np.arange(1, len(p1) + 1)
    spearman = float(np.corrcoef(ranks, rank_th)[0, 1])
    noise = [r["theta_r_deg"] for r in p2]
    noise_span = float(max(noise) - min(noise))
    signal_span = float(max(thetas) - min(thetas))
    fixed_ke = [r["fixed_window"]["ke_ratio"] for r in p3["rows"]]
    return {
        "material": spec.name,
        "literature_theta_band_deg": list(spec.literature_theta_deg),
        "theta_simulated_band_deg": [min(thetas), max(thetas)],
        "literature_reachable": bool(
            min(thetas) <= spec.literature_theta_deg[1]
            and max(thetas) >= spec.literature_theta_deg[0]),
        "monotone_in_mu": bool(np.all(diffs > 0.0)),
        "spearman_mu_theta": spearman,
        "n_inversions": int(np.sum(diffs <= 0.0)),
        "signal_span_deg": signal_span,
        "replica_noise_span_deg": noise_span,
        "signal_to_noise": float(signal_span / noise_span) if noise_span else float("inf"),
        "fixed_window_ke_ratio": fixed_ke,
        "fixed_window_at_rest": bool(all(k < KE_STOP for k in fixed_ke)),
        "n_particles": int(p1[0]["n_particles"]),
        "heap_height_over_grain_diameter": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="sand")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--out", default=ARTIFACT)
    args = ap.parse_args()

    spec = {s.name: s for s in MATERIALS}[args.material]
    t0 = time.time()
    p1 = friction_response(spec)
    p2 = replica_noise(spec)
    p3 = rest_criterion(spec)
    summary = summarise(spec, p1, p2, p3)
    summary["heap_height_over_grain_diameter"] = round(
        float(spec.calibration["fill_height"] / (2.0 * spec.calibration["radius"])), 2)

    record = {
        "mission": 14680,
        "gap": "GAP-DEM-AUTOCALIB",
        "round": 1,
        "material": spec.name,
        "material_spec": {
            "radius_m": spec.radius, "density_kg_m3": spec.density,
            "mu_rolling": spec.mu_rolling, "literature_theta_deg": list(spec.literature_theta_deg),
        },
        "fixed_parameters": {"restitution": 0.38, "stiffness_ratio": 2.2},
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform(),
                        "machine": platform.machine()},
        "p1_friction_response": p1,
        "p2_replica_noise": p2,
        "p3_rest_criterion": p3,
        "summary": summary,
        "wall_seconds": round(time.time() - t0, 1),
        "repro_cmd": "PYTHONPATH=src python scripts/m14680_autocalib_diagnosis.py",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)

    if not args.json_only:
        print(f"[{spec.name}] theta(mu): " + "  ".join(
            f"mu={r['mu']:.2f}->{r['theta_r_deg']:.2f}deg" for r in p1))
        print(f"[{spec.name}] replica thetas: " +
              "  ".join(f"{r['theta_r_deg']:.2f}" for r in p2))
        for row in p3["rows"]:
            print(f"[{spec.name}] mu={row['mu']}: fixed ke_ratio="
                  f"{row['fixed_window']['ke_ratio']:.3f} theta="
                  f"{row['fixed_window']['theta_r_deg']:.2f} | ke_stop theta="
                  f"{row['ke_stop_window']['theta_r_deg']:.2f} "
                  f"ke_ratio={row['ke_stop_window']['ke_ratio']:.4f} "
                  f"steps={row['ke_stop_window']['steps']}")
        print(json.dumps({k: summary[k] for k in
                          ("monotone_in_mu", "spearman_mu_theta", "signal_span_deg",
                           "replica_noise_span_deg", "signal_to_noise",
                           "literature_reachable", "fixed_window_at_rest")}, indent=2))
        print(f"artifact -> {args.out}  ({record['wall_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())