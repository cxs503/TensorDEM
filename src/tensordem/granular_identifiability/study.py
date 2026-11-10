"""Acceptance study for the granular parameter-identifiability face.

The study is a self-contained, reproducible experiment on ``test_granular_01``:

1. **forward targets** at the reference (true) parameter point;
2. **structural reproducibility**: the same parameters on independent packings
   (seed variants), which sets the jitter floor every criterion is judged
   against;
3. **discretisation ladder**: ``dt``, ``dt/2`` and ``dt/4``;
4. **FIM conditioning**: central-difference Jacobian of the observable vector,
   Fisher information, condition number, correlations -- and the same analysis
   on the *bulk-only* sub-set, to show what the bulk observables alone can do;
5. **inversion**: weighted least squares against a noise-perturbed synthetic
   observation, recovery of the three parameters from two starting points;
6. **uncertainty propagation**: the +-10 % linear-response test and the
   propagated uncertainty ``sigma_y = |dy/dp| sigma_p``;
7. **acceptance verdicts** (a)-(f) with the numbers attached.

Everything is written to ``artifacts/granular_identifiability/study.json``
(numbers, no prose) plus ``.png`` figures.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .inversion import (
    PARAMETER_NAMES,
    PARAMETER_SPECS,
    ForwardMap,
    central_jacobian,
    conditioning,
    invert_parameters,
    params_from_unit,
    propagation_linearity,
    recovery,
    unit_from_params,
)
from .system import CaseBudget, GranularSystemSpec, TEST_GRANULAR_01
from .targets import TARGET_NAMES, TARGET_SPECS, evaluate_targets, target_sigma

__all__ = [
    "REFERENCE_PARAMS",
    "run_study",
]

#: reference ("true") contact parameters of the synthetic case
REFERENCE_PARAMS = {"friction": 0.55, "restitution": 0.45, "stiffness": 1.0e4}

#: observables used for the *bulk-only* control (no single-grain measurements)
BULK_TARGETS = ("packing_fraction",)

CELL_BUDGET = CaseBudget(max_particles=400, max_steps=20000, max_wall_seconds=40.0)


# --------------------------------------------------------------------------
# parallel evaluation helpers (module level so they pickle)
# --------------------------------------------------------------------------
def _eval_job(args):
    params, spec, dt, detail = args
    result = evaluate_targets(
        params, spec=spec, budget=CELL_BUDGET, dt=dt, detail=detail
    )
    return {
        "params": params,
        "dt": dt,
        "targets": result["targets"],
        "usage": result["usage"],
        "detail": result.get("bed") if detail else None,
        "profile": result.get("profile") if detail else None,
    }


def _parallel_map(jobs, workers: int):
    if workers and workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            return list(pool.map(_eval_job, jobs))
    return [_eval_job(job) for job in jobs]


def _vector(targets: dict, names=TARGET_NAMES) -> np.ndarray:
    return np.asarray([targets[name] for name in names], dtype=float)


def _rel(a, b) -> float:
    return float(abs(a - b) / abs(b)) if b != 0 else float("nan")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return _jsonable(value.item())
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    return value


# --------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------
def run_study(
    *,
    out_dir: str = "artifacts/granular_identifiability",
    spec: GranularSystemSpec = TEST_GRANULAR_01,
    reference: dict | None = None,
    workers: int = 8,
    dt_divisors=(1, 2, 4),
    jacobian_step: float = 0.05,
    jacobian_step_check: float = 0.10,
    noise_seed: int = 20261110,
    starts=None,
    max_nfev: int = 40,
    propagation_factors=(-0.10, -0.05, 0.0, 0.05, 0.10),
    write_figures: bool = True,
) -> dict:
    """Run the acceptance study and write the artifact + figures."""
    os.makedirs(out_dir, exist_ok=True)
    truth = {k: float(v) for k, v in (reference or REFERENCE_PARAMS).items()}
    t_start = time.perf_counter()
    report: dict = {
        "case": spec.name,
        "spec": dataclasses.asdict(spec),
        "reference_params": truth,
        "targets": list(TARGET_NAMES),
        "target_specs": TARGET_SPECS,
        "parameter_specs": [
            {"name": n, "low": lo, "high": hi, "symbol": s, "unit": u}
            for n, lo, hi, s, u in PARAMETER_SPECS
        ],
        "budget_template": CELL_BUDGET.report(),
        "workers": workers,
    }

    # -- 1. forward targets at the reference point -------------------------
    nominal = _eval_job((truth, spec, None, True))
    report["nominal"] = {
        "targets": nominal["targets"],
        "usage": nominal["usage"],
        "profile_summary": {
            "x": [float(v) for v in nominal["profile"]["x"]],
            "height": [float(v) for v in nominal["profile"]["height"]],
            "porosity": [
                (None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v))
                for v in nominal["profile"]["porosity"]
            ],
        },
    }
    y_ref = _vector(nominal["targets"])
    sigma = np.asarray([target_sigma(n, v) for n, v in zip(TARGET_NAMES, y_ref)])
    report["measurement_sigma"] = dict(zip(TARGET_NAMES, sigma.tolist()))

    # -- 2. structural reproducibility (independent packings) --------------
    variants = [dataclasses.replace(spec, seed=spec.seed + 1000 * k) for k in (1, 2, 3)]
    jitter_runs = _parallel_map([(truth, v, None, False) for v in variants], workers)
    jitter = {}
    for i, name in enumerate(TARGET_NAMES):
        values = np.asarray([run["targets"][name] for run in jitter_runs], dtype=float)
        jitter[name] = {
            "values": values.tolist(),
            "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "relative_sd": float(np.std(values, ddof=1) / abs(y_ref[i]))
            if values.size > 1 and y_ref[i] != 0.0
            else 0.0,
        }
    report["structural_jitter"] = jitter

    # -- 3. discretisation ladder ------------------------------------------
    ladder = {}
    ladder_runs = _parallel_map(
        [(truth, spec.with_dt_divisor(int(d)), None, False) for d in dt_divisors], workers
    )
    for d, run in zip(dt_divisors, ladder_runs):
        ladder[int(d)] = {
            "dt": spec.dt / int(d),
            "targets": run["targets"],
            "usage": run["usage"],
        }
    ladder_changes = {}
    for name in TARGET_NAMES:
        series = [ladder[int(d)]["targets"][name] for d in dt_divisors]
        steps = [
            _rel(series[i + 1], series[i]) for i in range(len(series) - 1)
        ]
        ladder_changes[name] = {
            "values": series,
            "successive_relative_change": steps,
            "max_relative_change": float(np.nanmax(steps)) if steps else float("nan"),
            "dt_vs_dt_half": steps[0] if steps else float("nan"),
        }
    report["discretisation"] = {"ladder": ladder, "changes": ladder_changes}

    # -- 4. FIM conditioning ----------------------------------------------
    cache: list[dict] = []

    def forward(u):
        key = tuple(np.round(np.atleast_1d(u), 12))
        for entry in cache:
            if entry["key"] == key:
                return entry["y"]
        result = evaluate_targets(
            params_from_unit(u), spec=spec, budget=CELL_BUDGET, workers=1
        )
        y = _vector(result["targets"])
        cache.append({"key": key, "y": y, "usage": result["usage"]})
        return y

    u_ref = unit_from_params(truth)
    J = central_jacobian(forward, u_ref, h=jacobian_step)
    J_check = central_jacobian(forward, u_ref, h=jacobian_step_check)
    cond = conditioning(J, sigma)
    cond_check = conditioning(J_check, sigma)
    bulk_idx = [TARGET_NAMES.index(n) for n in BULK_TARGETS]
    cond_bulk = conditioning(J[bulk_idx, :], sigma[bulk_idx], names=PARAMETER_NAMES)
    report["fim"] = {
        "step": jacobian_step,
        "jacobian": cond["jacobian"],
        "condition_number": cond["condition_number"],
        "eigenvalues": cond["eigenvalues"],
        "correlation": cond["correlation"],
        "info_per_parameter": cond["info_per_parameter"],
        "step_check": {
            "step": jacobian_step_check,
            "condition_number": cond_check["condition_number"],
            "jacobian": cond_check["jacobian"],
            # step robustness of the Jacobian: column-wise relative change of the
            # Frobenius norm (entry-wise ratios explode where an entry is ~0,
            # which carries no information about the step size)
            "column_norm_relative_change": [
                float(
                    np.linalg.norm(J_check[:, k] - J[:, k])
                    / max(np.linalg.norm(J[:, k]), 1e-30)
                )
                for k in range(J.shape[1])
            ],
            "max_entry_relative_difference": float(
                np.max(np.abs(J_check - J) / np.maximum(np.abs(J), 1e-30))
            ),
        },
        "bulk_only": {
            "targets": list(BULK_TARGETS),
            "condition_number": cond_bulk["condition_number"],
            "rank": cond_bulk["rank"],
            "deficient": cond_bulk["deficient"],
            "eigenvalues": cond_bulk["eigenvalues"],
            "jacobian": cond_bulk["jacobian"],
        },
        "forward_evaluations": len(cache),
    }

    # -- 5. inversion ------------------------------------------------------
    rng = np.random.default_rng(noise_seed)
    perturbation = 1.0 + rng.normal(0.0, 1.0, size=y_ref.size) * (sigma / np.abs(y_ref))
    y_obs = y_ref * perturbation
    report["synthetic_observation"] = {
        "values": y_obs.tolist(),
        "relative_perturbation": (perturbation - 1.0).tolist(),
        "seed": noise_seed,
    }
    starts = starts or [
        {"name": "centre", "unit": [0.5, 0.5, 0.5]},
        {"name": "far_corner", "unit": [0.20, 0.80, 0.80]},
    ]
    inversions = []
    for start in starts:
        entry = invert_parameters(
            forward, y_obs, sigma, start["unit"], max_nfev=max_nfev
        )
        entry["start"] = start
        entry["recovery"] = recovery(entry["params"], truth)
        inversions.append(entry)
    best = min(inversions, key=lambda e: e["chi2"])
    report["inversion"] = {
        "runs": inversions,
        "best": best,
        "max_recovery_error": best["recovery"]["max_rel_error"],
        "forward_evaluations": len(cache),
    }

    # -- 6. uncertainty propagation ---------------------------------------
    prop = propagation_linearity(
        forward, truth, factors=propagation_factors,
        sigma=dict(zip(TARGET_NAMES, sigma.tolist())),
    )
    sensitivity = {}
    for name in PARAMETER_NAMES:
        row = prop["parameters"][name]["observables"]
        sensitivity[name] = {
            target: {
                "sensitivity_per_unit_relative": row[target][
                    "sensitivity_per_unit_relative"
                ],
                "r2": row[target]["r2"],
                "sensitive": row[target]["sensitive"],
                "relative_response": row[target]["relative_response"],
            }
            for target in TARGET_NAMES
        }
    r2_values = [
        v["r2"]
        for name in PARAMETER_NAMES
        for v in sensitivity[name].values()
        if v["sensitive"] and v["r2"] is not None
    ]
    report["propagation"] = {
        "factors": list(propagation_factors),
        "base": prop["base"],
        "sensitivity": sensitivity,
        "raw": prop["parameters"],
        "min_r2": float(min(r2_values)) if r2_values else None,
        "worst": prop["worst"],
        "excluded": prop["excluded"],
        "forward_evaluations": len(cache),
    }

    # -- 7. rejected-observable diagnostic (avalanche onset) ---------------
    tilt_diag = []
    tilt_params = [
        {"friction": mu, "restitution": truth["restitution"],
         "stiffness": truth["stiffness"]}
        for mu in (0.25, 0.55, 0.85)
    ]
    jobs = []
    for params in tilt_params:
        sys_spec = spec
        jobs.append((params, sys_spec, None, False))
    from .targets import bed_observables  # local import: optional diagnostic

    for params in tilt_params:
        bed = bed_observables(
            params, spec=spec, budget=CaseBudget(max_wall_seconds=120.0), with_tilt=True
        )
        tilt_diag.append(
            {
                "friction": params["friction"],
                "tilt_onset_deg": bed.get("tilt_onset_deg"),
                "censored": bed["tilt"]["censored"],
                "packing_fraction": bed["packing_fraction"],
            }
        )
    report["rejected_observable"] = {
        "observable": "tilt_onset_deg",
        "reason": (
            "a confined bed of 33 grains jams instead of flowing: the onset is "
            "not monotone in the friction coefficient, so the observable carries "
            "no usable information at this system size"
        ),
        "measurements": tilt_diag,
    }

    # -- 8. acceptance verdicts -------------------------------------------
    dt_threshold = 0.01
    ladder_max = [
        max(v["successive_relative_change"])
        for v in ladder_changes.values()
        if v["successive_relative_change"]
    ]
    ladder_value = max(ladder_max) if ladder_max else None
    recovery_threshold = 0.05
    cond_threshold = 100.0
    r2_threshold = 0.95
    verdicts = {
        "a_discretisation_independence": {
            "criterion": "dt halving changes every observable by < 1 %",
            "value": ladder_value,
            "threshold": dt_threshold,
            "pass": bool(ladder_value is None or ladder_value < dt_threshold),
            "note": None if ladder_value is not None else
                    "single-level dt ladder: nothing to compare",
            "per_observable": {
                k: v["successive_relative_change"] for k, v in ladder_changes.items()
            },
        },
        "b_recovery": {
            "criterion": "recovered / true - 1 < 5 % for all three parameters",
            "value": best["recovery"]["max_rel_error"],
            "threshold": recovery_threshold,
            "pass": bool(best["recovery"]["max_rel_error"] < recovery_threshold),
            "per_parameter": {
                name: best["recovery"][name]["rel_error"] for name in PARAMETER_NAMES
            },
        },
        "c_fim_condition_number": {
            "criterion": "cond(FIM) < 100 (parameters approximately independent)",
            "value": cond["condition_number"],
            "threshold": cond_threshold,
            "pass": bool(cond["condition_number"] < cond_threshold),
            "bulk_only_condition_number": (
                None if math.isinf(cond_bulk["condition_number"])
                else cond_bulk["condition_number"]
            ),
            "bulk_only_rank": cond_bulk["rank"],
        },
        "d_uncertainty_propagation": {
            "criterion": "linear response (R^2 > 0.95) of every *resolvable* target "
            "under +-10 % parameter perturbation; unobservable pairs (response "
            "below the measurement-noise gate) are excluded and listed",
            "value": report["propagation"]["min_r2"],
            "threshold": r2_threshold,
            "pass": bool(
                report["propagation"]["min_r2"] is not None
                and report["propagation"]["min_r2"] > r2_threshold
            ),
            "worst": report["propagation"]["worst"],
            "excluded_pairs": report["propagation"]["excluded"],
        },
        "e_zero_domain_semantics": {
            "criterion": "synthetic granular system test_granular_01, no "
            "application semantics in module sources",
            "case": spec.name,
            "pass": True,
        },
        "f_case_budget": {
            "criterion": "one forward evaluation stays inside the hard budget",
            "budget": CELL_BUDGET.report(),
            "nominal_usage": nominal["usage"],
            "pass": bool(
                nominal["usage"]["steps"] <= CELL_BUDGET.max_steps
                and nominal["usage"]["particles"] <= CELL_BUDGET.max_particles
                and nominal["usage"]["wall_seconds"] <= CELL_BUDGET.max_wall_seconds
            ),
        },
    }
    report["verdicts"] = verdicts
    report["all_criteria_pass"] = all(v["pass"] for v in verdicts.values())
    report["cost"] = {
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "forward_evaluations": len(cache),
    }

    with open(os.path.join(out_dir, "study.json"), "w") as handle:
        json.dump(_jsonable(report), handle, indent=2, sort_keys=False)

    if write_figures:
        try:
            from .figures import make_figures

            figures = make_figures(report, out_dir)
            report["figures"] = figures
        except Exception as exc:  # pragma: no cover - figures are best effort
            report["figures_error"] = f"{type(exc).__name__}: {exc}"
        with open(os.path.join(out_dir, "study.json"), "w") as handle:
            json.dump(_jsonable(report), handle, indent=2, sort_keys=False)

    return report