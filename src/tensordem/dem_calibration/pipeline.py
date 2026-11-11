"""Calibration pipeline: objective, BayesOpt search, UQ, verification verdicts.

The pipeline for one material class is

1. **twin-experiment targets** -- forward responses at the declared truth, plus
   observation noise (``OBSERVATION_SIGMA``);
2. **objective** ``J(theta) = sum_i w_i |sim_i - exp_i| / |exp_i|`` over the
   responses ``theta_r`` (repose angle) and ``phi_solid`` (bulk solid fraction);
3. **BayesOpt search** -- Latin-hypercube design, Matern-5/2 GP surrogate,
   expected-improvement acquisition (:mod:`inverse`);
4. **posterior** -- random-walk Metropolis on the response surface for the
   credible intervals (:func:`inverse.posterior_sample`);
5. **verification** -- the calibrated parameters are re-run with a different
   packing seed, so the acceptance number is not the number the optimiser
   minimised;
6. **generalisation** -- the calibrated parameters are transferred to the
   independent validation geometry (different grain radius, container and fill
   height) and compared with that geometry's twin-experiment target;
7. **reproducibility** -- the forward model is deterministic, so a repeated run
   must give a bit-identical position digest.

All verdict thresholds are the campaign-node acceptance criteria; they are
constants of this module so that the tests and the report cannot drift apart.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .heap import HeapCase, HeapError, run_heap, run_pluviation
from .inverse import ParameterSpace, bayesian_optimize, posterior_sample
from .materials import MATERIALS, OBSERVATION_SIGMA, RESPONSE_WEIGHTS, MaterialSpec, case_for

__all__ = [
    "THRESHOLDS",
    "PARAMETER_SPACE",
    "ForwardModel",
    "objective_function",
    "calibrate_material",
    "calibration_verdicts",
    "run_all",
]

#: Acceptance thresholds of the campaign node.
THRESHOLDS = {
    "angle_error_deg": 5.0,          # (a) |theta_calibrated - theta_target| < 5 deg
    "objective": 0.05,               # (b) objective < 0.05
    "max_iterations": 50,            # (b) inside 50 forward evaluations
    "generalisation_rel": 0.10,      # (c) independent-condition relative error < 10 %
    "credible_level": 0.95,          # (d) 95 % credible interval
}

PARAMETER_SPACE = ParameterSpace()

RESPONSES = ("theta_r_deg", "phi_solid")


class ForwardModel:
    """Cached wrapper around :func:`run_heap` for one experiment condition.

    Repeated evaluations of the same parameter triple are served from the cache
    (expected-improvement candidates can coincide with earlier probes), and a
    run that fails to produce a measurable flank is reported as the declared
    penalty ``failure_penalty`` instead of crashing the search.
    """

    def __init__(self, case: HeapCase, *, failure_penalty: float = 10.0,
                 forward: Callable[[HeapCase], dict] | None = None,
                 seed_offset: int = 0):
        self.case = case
        self.failure_penalty = float(failure_penalty)
        self._forward = forward or _default_forward
        self.seed_offset = int(seed_offset)
        self.cache: dict[tuple, dict | None] = {}
        self.n_calls = 0

    def _key(self, params: dict) -> tuple:
        return (round(float(params["mu"]), 10),
                round(float(params["restitution"]), 10),
                round(float(params["stiffness_ratio"]), 10))

    def case_for(self, params: dict) -> HeapCase:
        case = self.case.with_parameters(params["mu"], params["restitution"],
                                         params["stiffness_ratio"])
        if self.seed_offset:
            case = replace(case, seed=case.seed + self.seed_offset)
        return case

    def responses(self, params: dict) -> dict | None:
        key = self._key(params)
        if key in self.cache:
            return self.cache[key]
        self.n_calls += 1
        try:
            out = self._forward(self.case_for(params))
            record = {name: float(out[name]) for name in RESPONSES}
            record["digest"] = out.get("digest")
            record["r2"] = float(out.get("r2", float("nan")))
            record["ke_ratio"] = float(out.get("ke_ratio", float("nan")))
            if not all(math.isfinite(record[name]) for name in RESPONSES):
                record = None
        except HeapError:
            record = None
        self.cache[key] = record
        return record


def _default_forward(case: HeapCase) -> dict:
    """Run the forward model selected by the case (growth or release protocol)."""
    if case.shape == "pluviation":
        return run_pluviation(case)
    return run_heap(case)


def objective_function(sim: dict | None, targets: dict,
                       weights: dict | None = None,
                       *, failure_penalty: float = 10.0) -> float:
    """``J = sum_i w_i |sim_i - exp_i| / |exp_i|`` (smaller is better)."""
    weights = weights or RESPONSE_WEIGHTS
    if sim is None:
        return float(failure_penalty)
    total = 0.0
    for name, weight in weights.items():
        target = float(targets[name])
        if target == 0.0:
            raise ValueError(f"target {name} is zero; relative error undefined")
        total += float(weight) * abs(float(sim[name]) - target) / abs(target)
    return float(total)


def _make_objective(forward: ForwardModel, targets: dict,
                    weights: dict | None = None) -> Callable[[dict], float]:
    def objective(params: dict) -> float:
        return objective_function(forward.responses(params), targets, weights,
                                  failure_penalty=forward.failure_penalty)
    return objective


def calibrate_material(spec: MaterialSpec, *, n_init: int = 8, n_iter: int = 32,
                       seed: int = 0, forward: Callable[[HeapCase], dict] | None = None,
                       space: ParameterSpace = PARAMETER_SPACE,
                       n_posterior: int = 2000) -> dict:
    """Run the full calibration for one material class and return the record."""
    t_start = time.time()
    cal_case = spec.case("calibration")
    val_case = spec.case("validation")

    cal_forward = ForwardModel(cal_case, forward=forward)
    val_forward = ForwardModel(val_case, forward=forward)

    # 1. twin-experiment targets (deterministic truth response)
    truth = {k: float(v) for k, v in spec.truth.items()}
    truth_responses = cal_forward.responses(truth)
    if truth_responses is None:
        raise RuntimeError(f"{spec.name}: truth parameters produce no measurable heap")
    targets = {name: truth_responses[name] for name in RESPONSES}

    # 2. + 3. search
    objective = _make_objective(cal_forward, targets)
    search = bayesian_optimize(objective, space, n_init=n_init, n_iter=n_iter,
                               seed=seed, y_stop=THRESHOLDS["objective"])

    # 4. posterior on the visited response surface
    table = search["table"]
    x_unit = np.asarray([row["unit"] for row in table])
    response_matrix = {name: np.asarray([forward_value(row, name, cal_forward)
                                         for row in table]) for name in RESPONSES}
    post = posterior_sample(x_unit, response_matrix, targets, OBSERVATION_SIGMA,
                            space, n_samples=n_posterior, seed=seed + 7)

    # 5. verification on a different packing seed
    best = search["best_params"]
    verify_forward = ForwardModel(cal_case, forward=forward, seed_offset=1)
    verify_responses = verify_forward.responses(best)
    theta_verify = (verify_responses["theta_r_deg"] if verify_responses
                    else float("nan"))
    angle_error_deg = abs(theta_verify - targets["theta_r_deg"])

    # 6. generalisation to the independent validation geometry
    val_truth = {k: float(v) for k, v in spec.truth.items()}
    val_targets = val_forward.responses(val_truth)
    val_calibrated = ForwardModel(val_case, forward=forward, seed_offset=1).responses(best)
    if val_targets is None or val_calibrated is None:
        generalisation_rel = float("inf")
        val_theta_pred = float("nan")
        val_theta_target = float("nan")
    else:
        val_theta_target = val_targets["theta_r_deg"]
        val_theta_pred = val_calibrated["theta_r_deg"]
        generalisation_rel = abs(val_theta_pred - val_theta_target) / abs(val_theta_target)

    # 7. reproducibility witness: repeat the verification case
    repeat = ForwardModel(cal_case, forward=forward, seed_offset=1).responses(best)
    reproducible = bool(repeat is not None and verify_responses is not None
                        and repeat["digest"] == verify_responses["digest"])

    record = {
        "material": spec.name,
        "experiment": {
            "calibration": cal_case.signature(),
            "validation": val_case.signature(),
        },
        "truth": truth,
        "targets": targets,
        "search": {
            "n_evaluations": search["n_evaluations"],
            "n_init": n_init,
            "n_iter": n_iter,
            "best_params": best,
            "best_objective": search["best_objective"],
            "converged_iteration": search["converged_iteration"],
            "table": table,
            "seed": seed,
        },
        "posterior": {
            "summary": post["summary"],
            "acceptance_rate": post["acceptance_rate"],
            "n_samples": post["n_samples"],
        },
        "verification": {
            "theta_target_deg": targets["theta_r_deg"],
            "theta_verified_deg": theta_verify,
            "angle_error_deg": angle_error_deg,
            "seed_offset": 1,
            "responses": verify_responses,
        },
        "generalisation": {
            "theta_target_deg": val_theta_target,
            "theta_predicted_deg": val_theta_pred,
            "relative_error": generalisation_rel,
            "geometry": val_case.signature(),
        },
        "reproducible": reproducible,
        "forward_evaluations": {
            "calibration": cal_forward.n_calls,
            "validation": val_forward.n_calls,
            "verify": verify_forward.n_calls,
        },
        "wall_seconds": round(time.time() - t_start, 2),
    }
    record["verdicts"] = calibration_verdicts(record)
    return record


def forward_value(row: dict, name: str, forward: ForwardModel) -> float:
    """Response of ``name`` at the design point of ``row`` (cache hit after search)."""
    if "responses" in row and row["responses"] is not None:
        return float(row["responses"][name])
    record = forward.responses(row["params"])
    if record is None:
        return float("nan")
    row["responses"] = record
    return float(record[name])


def calibration_verdicts(record: dict) -> dict:
    """Machine-checkable verdicts of criteria (a)-(e) for one material."""
    post = record["posterior"]["summary"]
    truth = record["truth"]
    ci_covers = {}
    for name, value in truth.items():
        lo = post[name]["lo95"]
        hi = post[name]["hi95"]
        ci_covers[name] = bool(lo <= value <= hi)
    verdicts = {
        "angle_error_deg": record["verification"]["angle_error_deg"],
        "angle_within_5deg": bool(record["verification"]["angle_error_deg"]
                                  < THRESHOLDS["angle_error_deg"]),
        "best_objective": record["search"]["best_objective"],
        "objective_below_0p05": bool(record["search"]["best_objective"]
                                     < THRESHOLDS["objective"]),
        "iterations_within_50": bool(record["search"]["n_evaluations"]
                                     <= THRESHOLDS["max_iterations"]),
        "generalisation_rel": record["generalisation"]["relative_error"],
        "generalisation_within_10pct": bool(
            record["generalisation"]["relative_error"] < THRESHOLDS["generalisation_rel"]),
        "ci_covers_truth": ci_covers,
        "ci_covers_all": bool(all(ci_covers.values())),
        "reproducible": bool(record["reproducible"]),
    }
    verdicts["all_pass"] = bool(
        verdicts["angle_within_5deg"] and verdicts["objective_below_0p05"]
        and verdicts["iterations_within_50"] and verdicts["generalisation_within_10pct"]
        and verdicts["ci_covers_all"] and verdicts["reproducible"])
    return verdicts


def run_all(materials: tuple[MaterialSpec, ...] = MATERIALS, *, n_init: int = 8,
            n_iter: int = 32, seed: int = 0, progress=None) -> dict:
    """Calibrate every material class and assemble the campaign record."""
    records = []
    for spec in materials:
        if progress is not None:
            progress(f"calibrating {spec.name}")
        records.append(calibrate_material(spec, n_init=n_init, n_iter=n_iter, seed=seed))
    overall = {
        "criterion_accuracy": all(r["verdicts"]["angle_within_5deg"] for r in records),
        "criterion_convergence": all(r["verdicts"]["objective_below_0p05"]
                                     and r["verdicts"]["iterations_within_50"]
                                     for r in records),
        "criterion_generalisation": all(r["verdicts"]["generalisation_within_10pct"]
                                        for r in records),
        "criterion_uncertainty": all(r["verdicts"]["ci_covers_all"] for r in records),
        "criterion_reproducibility": all(r["verdicts"]["reproducible"] for r in records),
    }
    overall["all_pass"] = all(overall.values())
    return {"records": records, "criteria": overall}


def dump(result: dict, path: str) -> str:
    """Write the campaign record as JSON (zero hand-copied numbers)."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return path