#!/usr/bin/env python
"""P270W2C-N08 campaign: BPM three-target joint calibration feasibility + audit.

Stages
------
``sample``    run the DEM design (33 space-filling theta points: the reference
              point, its six central-difference partners, and 20 Latin-hypercube
              fill points), the AC5 Brazilian-disc stress-field study, the AC6
              negative control (200 random theta) and the AC7 reproducibility pair;
              raw records are appended to ``runs/p270w2c_n08/records.jsonl`` and one
              result file per design point is written to ``runs/p270w2c_n08/points``.
``analyze``   assemble the response surface, the numerical Jacobian and the
              identifiability diagnostics, run NSGA-II on the surrogate, evaluate
              the three-branch feasibility verdict, the held-out thickness
              extrapolation and every acceptance criterion, then write
              ``artifacts/p270w2c_n08/report.json`` + figures + the markdown report.

Everything numerical that ends up in the report is written by this script; the
markdown report is rendered from the JSON, so no number is hand-copied.

Usage
-----
    python scripts/p270w2c_n08_feasibility.py --stage sample --workers 48
    python scripts/p270w2c_n08_feasibility.py --stage analyze
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from dataclasses import replace
from pathlib import Path
import sys
from typing import Sequence
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.bpm_calibration_feasibility import (  # noqa: E402
    COND_NUMBER_LIMIT,
    CalibrationVector,
    DiscGeometry,
    FIT_THICKNESSES,
    HOLDOUT_THICKNESSES,
    HONDROS_CENTER_RATIO,
    HONDROS_TOLERANCE,
    PARAMETER_BOUNDS,
    PARAMETER_FIELDS,
    PARAMETER_NAMES,
    QUASISTATIC_KE_LIMIT,
    REFERENCE_VECTOR,
    SIZE_EFFECT_THICKNESSES,
    STRENGTH_RATIO_BAND,
    UCS_BAND_MPA,
    UCS_GEOMETRY,
    condition_diagnostics,
    default_protocol,
    evaluate_design_point,
    feasibility_verdict,
    adapt_protocol,
    geometry_prior_check,
    holdout_error,
    protocol_prior_check,
    nsga2,
    RESPONSE_NAMES,
    response_jacobian,
    run_axial,
    run_brazilian,
    size_effect_modulus,
    surrogate_quadratic,
)
from tensordem.bpm_identifiability import SpecimenGeometry  # noqa: E402
from tensordem.bpm_calibration_feasibility import PARTICLE_RADIUS  # noqa: E402

RUNS = ROOT / "runs" / "p270w2c_n08"
ARTIFACTS = ROOT / "artifacts" / "p270w2c_n08"
DOCS = ROOT / "docs"

STEP = 0.12            # central-difference offset in unit-cube coordinates
N_LHS = 20             # Latin-hypercube fill points on top of the 13 structured ones
NEGATIVE_CONTROL = 200  # AC6
SEED = 20260101


# ===========================================================================
# Design of experiments
# ===========================================================================

def build_design(seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Structured central-difference design + Latin-hypercube fill.

    Returns ``(points, structured)`` where ``structured`` maps the six
    ``plus``/``minus`` partners of the reference point (the Jacobian stencil).
    """
    reference = REFERENCE_VECTOR.to_unit()
    points: list[dict] = [{"name": "reference", "unit": reference.tolist(), "role": "reference"}]
    structured: list[dict] = []
    for k, name in enumerate(PARAMETER_NAMES):
        entry: dict = {"parameter": name, "index": k}
        for sign, label in ((+1.0, "plus"), (-1.0, "minus")):
            unit = reference.copy()
            unit[k] = float(np.clip(unit[k] + sign * STEP, 0.0, 1.0))
            entry[label] = f"{name}_{label}"
            points.append({"name": f"{name}_{label}", "unit": unit.tolist(),
                           "role": f"jacobian_{label}"})
        structured.append(entry)
    rng = np.random.default_rng(seed)
    lhs = rng.random((N_LHS, len(PARAMETER_NAMES)))
    for i, unit in enumerate(lhs):
        points.append({"name": f"lhs_{i:02d}", "unit": unit.tolist(), "role": "fill"})
    return points, structured


#: Thicknesses evaluated at *every* design point.  The thickest held-out end
#: (t = 0.8 m) is the most expensive specimen of the campaign (86 increments x 528
#: particles), so it is reserved for the reference point that actually delivers the
#: AC4 holdout claim instead of being repeated at all 33 points.
DESIGN_THICKNESSES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)


def point_tasks(point: dict, *, n_size_reps: int = 4, n_holdout_reps: int = 3,
                thicknesses: Sequence[float] = DESIGN_THICKNESSES) -> list[dict]:
    """DEM tasks of one design point (UCS + the tension family)."""
    tasks: list[dict] = [{"kind": "ucs", "point": point["name"],
                          "unit": point["unit"], "seed": 0}]
    for thickness in thicknesses:
        reps = int(n_size_reps) if thickness in FIT_THICKNESSES else int(n_holdout_reps)
        for rep in range(reps):
            tasks.append({"kind": "tension", "point": point["name"],
                          "unit": point["unit"], "thickness": float(thickness),
                          "rep": rep, "seed": 100 + rep})
    return tasks


# ===========================================================================
# Parallel DEM worker
# ===========================================================================

def run_task(task: dict) -> dict:
    """Execute one DEM task in a worker process (single-threaded torch)."""
    import torch

    torch.set_num_threads(1)
    start = time.time()
    vector = CalibrationVector.from_unit(task["unit"])
    kind = task["kind"]
    if kind == "ucs":
        protocol = default_protocol(mode="compression")
        if task.get("increments"):
            protocol = replace(protocol, increments=int(task["increments"]))
        result = run_axial(vector, UCS_GEOMETRY, protocol, bond_seed=int(task.get("seed", 0)))
        payload = {
            "kind": "ucs",
            "point": task.get("point"),
            "unit": task["unit"],
            "peak_stress_Pa": result.peak_stress,
            "ke_ratio": result.ke_ratio,
            "ke_readout": result.ke_readout.tolist(),
            "n_steps": result.n_steps,
            "broken_bonds": result.broken_bonds,
            "n_bonds": result.n_bonds,
            "strain": result.strain.tolist(),
            "stress_Pa": result.stress.tolist(),
            "seed": int(task.get("seed", 0)),
        }
    elif kind == "tension":
        protocol = default_protocol(mode="tension")
        if task.get("increments"):
            protocol = replace(protocol, increments=int(task["increments"]))
        geometry = SpecimenGeometry.from_thickness(float(task["thickness"]),
                                                  cross_n=4, radius=PARTICLE_RADIUS)
        result = run_axial(vector, geometry, protocol, bond_seed=int(task.get("seed", 0)))
        payload = {
            "kind": "tension",
            "point": task.get("point"),
            "unit": task["unit"],
            "thickness_m": float(task["thickness"]),
            "gauge_m": float((geometry.n_layers - 1) * 2.0 * PARTICLE_RADIUS),
            "n_layers": int(geometry.n_layers),
            "n_particles": int(geometry.n_layers * geometry.cross_n ** 2),
            "peak_stress_Pa": result.peak_stress,
            "ke_ratio": result.ke_ratio,
            "n_steps": result.n_steps,
            "broken_bonds": result.broken_bonds,
            "n_bonds": result.n_bonds,
            "seed": int(task.get("seed", 0)),
            "rep": int(task.get("rep", 0)),
            "increments": int(protocol.increments),
        }
    elif kind == "disc":
        protocol = default_protocol(mode="compression")
        geometry = DiscGeometry(n_lattice=int(task["n_lattice"]),
                                n_layers=int(task["n_layers"]))
        result = run_brazilian(vector, geometry, protocol,
                               bond_seed=int(task.get("seed", 0)),
                               probe=True,
                               probe_region_ratio=float(task.get("region", 0.20)))
        payload = {"kind": "disc", "point": task.get("point"), "unit": task["unit"],
                   "n_lattice": geometry.n_lattice, "n_layers": geometry.n_layers,
                   "region_ratio": float(task.get("region", 0.20)),
                   "sigma_t_Pa": result["sigma_t_Pa"],
                   "ke_ratio": result["ke_ratio"],
                   "n_particles": result["n_particles"],
                   "n_bonds": result["n_bonds"],
                   "broken_bonds": result["broken_bonds"],
                   "n_drive": result["n_drive"],
                   "arc_half_angle_rad": result["arc_half_angle_rad"],
                   "stress_history_Pa": result["stress_history_Pa"],
                   "load_history_N": result["load_history_N"],
                   "hoop_ratio_history": result["hoop_ratio_history"],
                   "sigma_transverse_history_Pa": result["sigma_transverse_history_Pa"],
                   "sigma_load_history_Pa": result["sigma_load_history_Pa"],
                   "region_n_particles": result["region_n_particles"],
                   "stress_field_at_peak": result.get("stress_field_at_peak", {}),
                   "peak_index": result["peak_index"]}
    else:
        raise ValueError(f"unknown task kind {kind}")
    payload["wall_s"] = time.time() - start
    return payload


def run_tasks(tasks: list[dict], *, workers: int) -> list[dict]:
    if workers <= 1:
        return [run_task(task) for task in tasks]
    context = mp.get_context("fork")
    with context.Pool(processes=int(workers)) as pool:
        return pool.map(run_task, tasks, chunksize=1)


# ===========================================================================
# Sample stage
# ===========================================================================

def stage_sample(workers: int) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    points, structured = build_design()
    tasks: list[dict] = []
    for point in points:
        tasks.extend(point_tasks(point))
    # AC4: the thickest held-out end is measured at the reference theta only.
    for rep in range(2):
        tasks.append({"kind": "tension", "point": "reference",
                      "unit": REFERENCE_VECTOR.to_unit().tolist(),
                      "thickness": 0.8, "rep": rep, "seed": 100 + rep})

    # Protocol-refinement audit: the strength must not be an artefact of the
    # increment size, so the reference theta is re-run with 1x / 2x / 4x the
    # certified increment count (UCS and one tensile thickness).
    reference_unit = REFERENCE_VECTOR.to_unit().tolist()
    base_increments = adapt_protocol(default_protocol(mode="compression"), 8,
                                     breaking_strain=REFERENCE_VECTOR.breaking_strain).increments
    for factor in (1, 2, 4):
        tasks.append({"kind": "ucs", "point": f"conv_ucs_{factor}x",
                      "unit": reference_unit, "seed": 0,
                      "increments": int(base_increments * factor)})
        tasks.append({"kind": "tension", "point": f"conv_t04_{factor}x",
                      "unit": reference_unit, "thickness": 0.4, "rep": 0, "seed": 100,
                      "increments": int(base_increments * factor)})

    # AC5: Brazilian disc stress field, three resolutions x three region radii.
    rng = np.random.default_rng(SEED + 7)
    for n_lattice, n_layers in ((7, 4), (9, 4), (13, 4)):
        for region in (0.12, 0.20, 0.30):
            tasks.append({"kind": "disc", "point": "reference_theta",
                          "unit": REFERENCE_VECTOR.to_unit().tolist(),
                          "n_lattice": n_lattice, "n_layers": n_layers,
                          "region": region, "seed": 0})

    # AC6: negative control -- 200 random theta, UCS only.
    random_units = rng.random((NEGATIVE_CONTROL, len(PARAMETER_NAMES)))
    for i, unit in enumerate(random_units):
        tasks.append({"kind": "ucs", "point": f"neg_{i:03d}", "unit": unit.tolist(),
                      "seed": 0, "role": "negative_control"})

    # AC7: reproducibility pair (same theta, same seed, independent processes).
    reference_unit = REFERENCE_VECTOR.to_unit().tolist()
    for label in ("repeat_a", "repeat_b"):
        tasks.append({"kind": "ucs", "point": label, "unit": reference_unit,
                      "seed": 0, "role": "repeat"})

    print(f"[sample] {len(tasks)} DEM tasks, {workers} workers", flush=True)
    started = time.time()
    results = run_tasks(tasks, workers=workers)
    print(f"[sample] finished in {time.time() - started:.0f} s", flush=True)

    with (RUNS / "records.jsonl").open("w") as handle:
        for record in results:
            handle.write(json.dumps(record) + "\n")

    state = {
        "points": points,
        "structured": structured,
        "step": STEP,
        "negative_control": NEGATIVE_CONTROL,
        "n_tasks": len(tasks),
        "wall_s": time.time() - started,
        "workers": workers,
    }
    (RUNS / "design.json").write_text(json.dumps(state, indent=1))
    print(f"[sample] wrote {RUNS/'records.jsonl'}", flush=True)


# ===========================================================================
# Assemble design points from raw records
# ===========================================================================

def load_records() -> list[dict]:
    path = RUNS / "records.jsonl"
    if not path.exists():
        raise SystemExit("run the sample stage first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assemble_points(records: list[dict], design: dict) -> dict[str, dict]:
    """One design point -> the three responses (+ all raw bookkeeping)."""
    by_point: dict[str, dict[str, list]] = {}
    for record in records:
        if record["kind"] not in ("ucs", "tension"):
            continue
        if record.get("point", "").startswith(("neg_", "repeat_", "conv_")):
            continue
        by_point.setdefault(record["point"], {"ucs": [], "tension": []})
        by_point[record["point"]][record["kind"]].append(record)

    points: dict[str, dict] = {}
    for entry in design["points"]:
        name = entry["name"]
        bucket = by_point.get(name)
        if bucket is None:
            continue
        ucs_runs = bucket["ucs"]
        if not ucs_runs:
            continue
        ucs_run = ucs_runs[0]
        family: dict[str, dict] = {}
        for record in bucket["tension"]:
            key = f"{record['thickness_m']:.3f}"
            family.setdefault(key, {"thickness_m": record["thickness_m"],
                                    "strengths_Pa": [], "ke_ratio": 0.0,
                                    "n_layers": record["n_layers"],
                                    "n_particles": record["n_particles"],
                                    "held_out": record["thickness_m"] in HOLDOUT_THICKNESSES})
            family[key]["strengths_Pa"].append(record["peak_stress_Pa"])
            family[key]["ke_ratio"] = max(family[key]["ke_ratio"], record["ke_ratio"])
        for values in family.values():
            values["mean_Pa"] = float(np.mean(values["strengths_Pa"]))
            values["std_Pa"] = float(np.std(values["strengths_Pa"], ddof=1)) \
                if len(values["strengths_Pa"]) > 1 else 0.0
        fit_data = {float(t): family[f"{float(t):.3f}"]["strengths_Pa"]
                    for t in FIT_THICKNESSES if f"{float(t):.3f}" in family}
        present_holdouts = tuple(t for t in (0.05, 0.8) if f"{t:.3f}" in family)
        try:
            size_effect = bootstrap_fit(fit_data)
        except ValueError as exc:
            size_effect = {"m": float("nan"), "ci_low": float("nan"),
                           "ci_high": float("nan"), "error": str(exc)}
        sigma_t = float(family[f"{float(FIT_THICKNESSES[0]):.3f}"]["mean_Pa"])
        ucs = float(ucs_run["peak_stress_Pa"])
        points[name] = {
            "name": name,
            "role": entry["role"],
            "unit": entry["unit"],
            "theta": CalibrationVector.from_unit(entry["unit"]).as_dict(),
            "ucs_Pa": ucs,
            "ucs_MPa": ucs / 1e6,
            "sigma_t_Pa": sigma_t,
            "sigma_t_MPa": sigma_t / 1e6,
            "strength_ratio": ucs / sigma_t if sigma_t > 0 else float("nan"),
            "weibull_m": float(size_effect.get("m", float("nan"))),
            "weibull_m_ci": [float(size_effect.get("ci_low", float("nan"))),
                             float(size_effect.get("ci_high", float("nan")))],
            "size_effect": size_effect,
            "family": family,
            "holdout_thicknesses": list(present_holdouts),
            "gates": {
                "ucs_ke_ratio": float(ucs_run["ke_ratio"]),
                "max_family_ke_ratio": float(max(v["ke_ratio"] for v in family.values())),
                "quasistatic_ok": bool(ucs_run["ke_ratio"] <= QUASISTATIC_KE_LIMIT
                                       and all(v["ke_ratio"] <= QUASISTATIC_KE_LIMIT
                                               for v in family.values())),
            },
            "ucs_trace": {"strain": ucs_run["strain"], "stress_Pa": ucs_run["stress_Pa"],
                          "n_steps": ucs_run["n_steps"],
                          "broken_bonds": ucs_run["broken_bonds"],
                          "n_bonds": ucs_run["n_bonds"]},
            "wall_s": sum(r.get("wall_s", 0.0) for r in bucket["ucs"] + bucket["tension"]),
        }
    return points


def bootstrap_fit(fit_data: dict[float, list[float]]) -> dict:
    from tensordem.bpm_calibration_feasibility import bootstrap_size_effect_m

    return bootstrap_size_effect_m(fit_data, n_bootstrap=400)


# ===========================================================================
# Analyze stage
# ===========================================================================

def objective_from_point(point: dict) -> dict[str, float]:
    """Normalised target violations used as the NSGA-II objectives (minimise)."""
    ucs = point["ucs_MPa"]
    ratio = point["strength_ratio"]
    ci = point["weibull_m_ci"]
    return {"ucs": ucs, "ratio": ratio, "m_ci": ci,
            "m": point["weibull_m"]}


def build_surrogates(points: dict[str, dict]) -> dict[str, dict]:
    names = list(points)
    units = np.array([points[n]["unit"] for n in names], dtype=float)
    ucs = np.array([math.log10(max(points[n]["ucs_MPa"], 1e-12)) for n in names])
    ratio = np.array([math.log10(max(points[n]["strength_ratio"], 1e-12)) for n in names])
    m = np.array([math.log10(max(points[n]["weibull_m"], 1e-12))
                  if math.isfinite(points[n]["weibull_m"]) else np.nan for n in names])
    mask = np.isfinite(m)
    return {
        "ucs": surrogate_quadratic(units, ucs),
        "ratio": surrogate_quadratic(units, ratio),
        "m": surrogate_quadratic(units[mask], m[mask]),
    }


def analyze(workers_note: str = "") -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    design = json.loads((RUNS / "design.json").read_text())
    records = load_records()
    points = assemble_points(records, design)
    if "reference" not in points:
        raise SystemExit("reference point missing from the records")

    report: dict = {
        "node": "P270W2C-N08",
        "task": "m9756",
        "generated_by": "scripts/p270w2c_n08_feasibility.py",
        "targets": {
            "ucs_band_MPa": list(UCS_BAND_MPA),
            "strength_ratio_band": list(STRENGTH_RATIO_BAND),
            "hondros_ratio": HONDROS_CENTER_RATIO,
            "hondros_tolerance": HONDROS_TOLERANCE,
            "cond_limit": COND_NUMBER_LIMIT,
            "quasistatic_ke_limit": QUASISTATIC_KE_LIMIT,
        },
        "design": {"n_points": len(points), "step": design["step"],
                   "n_lhs": N_LHS, "n_tasks": design["n_tasks"],
                   "wall_s": design["wall_s"], "workers": design["workers"]},
    }

    # ---------------------------------------------------------------- AC1
    prior = geometry_prior_check()
    protocol_prior = protocol_prior_check(
        default_protocol(mode="tension"),
        breaking_strain=float(points["reference"]["theta"]["eps_b"]))
    prior["protocol"] = protocol_prior
    prior["passed"] = bool(prior["passed"] and protocol_prior["passed"])
    report["AC1_geometry_prior"] = prior

    # ---------------------------------------------------------------- AC2
    structured = design["structured"]
    plus = [points[entry["plus"]] for entry in structured]
    minus = [points[entry["minus"]] for entry in structured]
    jacobian = response_jacobian(points["reference"], plus, minus,
                                 step=design["step"])
    jac_full = np.array(jacobian["jacobian"], dtype=float)
    finite_rows = [i for i in range(jac_full.shape[0])
                   if np.all(np.isfinite(jac_full[i]))]
    unevaluable = [RESPONSE_NAMES[i] for i in range(jac_full.shape[0])
                   if i not in finite_rows]
    # The m row is *unevaluable* wherever the size effect of the whole family is
    # not decreasing (m then does not exist), so the condition number is reported on
    # the rows that are actually defined -- and the missing rows are part of the
    # verdict, not silently dropped.
    diagnostics = condition_diagnostics(jac_full[finite_rows],
                                        response_names=[RESPONSE_NAMES[i] for i in finite_rows])
    diagnostics["evaluable_responses"] = [RESPONSE_NAMES[i] for i in finite_rows]
    diagnostics["unevaluable_responses"] = unevaluable
    report["AC2_identifiability"] = {
        "jacobian": jacobian,
        "diagnostics": {k: v for k, v in diagnostics.items() if k != "right_singular_vectors"},
        "right_singular_vectors": diagnostics.get("right_singular_vectors"),
    }

    # ---------------------------------------------------------------- surrogate + NSGA-II
    surrogates = build_surrogates(points)
    # AC3's m target needs a *reference* value.  Two are reported:
    #  * primary: the bond-level Weibull modulus m_bond = 6 that the engine is
    #    actually given (a model input, so the comparison is not circular);
    #  * secondary: the macro m measured at the reference theta by the same
    #    pipeline -- self-consistent by construction, therefore only a
    #    reproducibility statement, not a validation.
    m_bond_input = 6.0
    m_pseudo_truth = {"m": float(m_bond_input),
                      "ci": None,
                      "source": "bond-level Weibull modulus given to the engine "
                                "(model input m_bond = 6); non-circular",
                      "reference_theta_m": float(points["reference"]["weibull_m"]),
                      "reference_theta_ci": points["reference"]["weibull_m_ci"]}

    def objectives(X: np.ndarray) -> np.ndarray:
        ucs = 10.0 ** surrogates["ucs"]["predict"](X)
        ratio = 10.0 ** surrogates["ratio"]["predict"](X)
        m_hat = 10.0 ** surrogates["m"]["predict"](X)
        lo, hi = UCS_BAND_MPA
        rlo, rhi = STRENGTH_RATIO_BAND
        # Normalised distances to the three target bands (0 inside), plus the
        # distance of the surrogate m from the pseudo-truth value.
        ucs_violation = np.maximum(lo - ucs, 0.0) / (hi - lo) + np.maximum(ucs - hi, 0.0) / (hi - lo)
        ratio_violation = (np.maximum(rlo - ratio, 0.0) / (rhi - rlo)
                           + np.maximum(ratio - rhi, 0.0) / (rhi - rlo))
        m_violation = np.abs(m_hat - m_pseudo_truth["m"]) / max(m_pseudo_truth["m"], 1e-9)
        return np.stack([ucs_violation, ratio_violation, m_violation], axis=1)

    initial = np.array([points[name]["unit"] for name in points], dtype=float)
    pareto = nsga2(objectives, len(PARAMETER_NAMES), population=80, generations=60,
                   seed=SEED, initial=initial)
    front = np.array(pareto["pareto_front"], dtype=float)
    front_unit = np.array(pareto["pareto_population"], dtype=float)
    front_rows = []
    for unit, objective in zip(front_unit, front):
        ucs = float(10.0 ** surrogates["ucs"]["predict"]([unit])[0])
        ratio = float(10.0 ** surrogates["ratio"]["predict"]([unit])[0])
        m_hat = float(10.0 ** surrogates["m"]["predict"]([unit])[0])
        front_rows.append({
            "unit": unit.tolist(),
            "theta": CalibrationVector.from_unit(unit).as_dict(),
            "surrogate": {"ucs_MPa": ucs, "strength_ratio": ratio, "weibull_m": m_hat},
            "objectives": objective.tolist(),
        })
    feasible_front = [row for row in front_rows if max(row["objectives"]) <= 1e-9]
    report["AC3_pareto"] = {
        "m_pseudo_truth": m_pseudo_truth,
        "surrogate_quality": {name: {"r2": surrogates[name]["r2"],
                                     "loo_r2": surrogates[name]["loo_r2"],
                                     "loo_rmse": surrogates[name]["loo_rmse"],
                                     "n_samples": surrogates[name]["n_samples"],
                                     "n_features": surrogates[name]["n_features"]}
                              for name in surrogates},
        "n_front": len(front_rows),
        "n_feasible_front": len(feasible_front),
        "front": front_rows[:60],
        "front_history": pareto["front_history"],
        "population_objectives": pareto["objectives"],
        "population_unit": pareto["population"],
    }

    # ---------------------------------------------------------------- AC3 verdict at all design points
    verdicts = {}
    for name, point in points.items():
        verdicts[name] = feasibility_verdict(point, m_truth=m_pseudo_truth)
    reference_verdict = verdicts["reference"]
    report["AC3_verdict"] = {
        "reference": reference_verdict,
        "per_point_branch": {name: v["branch"] for name, v in verdicts.items()},
        "branch_counts": {b: sum(1 for v in verdicts.values() if v["branch"] == b)
                          for b in ("a", "b", "c")},
    }

    # ---------------------------------------------------------------- AC4 holdout
    reference_family = {k: v["mean_Pa"] for k, v in points["reference"]["family"].items()}
    reference_holdout = holdout_error(
        points["reference"]["size_effect"], points["reference"]["family"],
        holdout_thicknesses=tuple(points["reference"]["holdout_thicknesses"]),
        reference=reference_family)
    holdouts = {
        name: holdout_error(
            point["size_effect"], point["family"],
            holdout_thicknesses=tuple(point["holdout_thicknesses"]),
            reference={f"{float(t):.3f}": reference_family.get(f"{float(t):.3f}", float("nan"))
                       for t in point["holdout_thicknesses"]})
        for name, point in points.items()
    }
    full_holdout = [name for name, point in points.items()
                    if len(point["holdout_thicknesses"]) == 2]
    report["AC4_holdout"] = {
        "reference": reference_holdout,
        "per_point_passed": {name: value["passed"] for name, value in holdouts.items()},
        "n_passed": sum(1 for name, value in holdouts.items()
                        if value["passed"] and name in full_holdout),
        "n_points": len(full_holdout),
        "n_points_any_holdout": len(holdouts),
        "n_passed_any_holdout": sum(1 for value in holdouts.values() if value["passed"]),
        "full_holdout_points": full_holdout,
        "rows": {name: value["rows"] for name, value in holdouts.items()
                 if name in ("reference",) or value["passed"]},
    }

    # ---------------------------------------------------------------- AC5 Brazilian field
    disc_records = [r for r in records if r["kind"] == "disc"]
    ac5_rows = []
    for record in disc_records:
        history = record["hoop_ratio_history"]
        i = record["peak_index"] if record["peak_index"] >= 0 else 0
        peak_ratio = record["hoop_ratio_history"][i] if history else float("nan")
        finite = [value for value in history if value is not None and math.isfinite(value)]
        ac5_rows.append({
            "n_lattice": record["n_lattice"], "n_layers": record["n_layers"],
            "region_ratio": record["region_ratio"],
            "sigma_t_Pa": record["sigma_t_Pa"],
            "n_particles": record["n_particles"],
            "region_n_particles": record["region_n_particles"][i] if record["region_n_particles"] else 0,
            "peak_index": record["peak_index"],
            "ratio_at_peak": peak_ratio,
            "ratio_min": min(finite) if finite else float("nan"),
            "ratio_max": max(finite) if finite else float("nan"),
            "n_finite_readouts": len(finite),
            "sigma_transverse_at_peak_Pa": record["sigma_transverse_history_Pa"][i],
            "sigma_load_at_peak_Pa": record["sigma_load_history_Pa"][i],
            "arc_half_angle_rad": record["arc_half_angle_rad"],
            "ratio_history": record["hoop_ratio_history"],
            "sigma_transverse_history_Pa": record["sigma_transverse_history_Pa"],
            "sigma_load_history_Pa": record["sigma_load_history_Pa"],
            "relative_deviation": (abs(peak_ratio - HONDROS_CENTER_RATIO)
                                   / abs(HONDROS_CENTER_RATIO)
                                   if math.isfinite(peak_ratio) else float("inf")),
        })
    best = min(ac5_rows, key=lambda row: row["relative_deviation"]) if ac5_rows else None
    finite_rows = [row for row in ac5_rows if math.isfinite(row["relative_deviation"])]
    report["AC5_brazilian_field"] = {
        "rows": ac5_rows,
        "best": best,
        "robust_pass": bool(finite_rows and all(
            row["relative_deviation"] <= HONDROS_TOLERANCE for row in finite_rows)),
        "n_rows_with_tension": len(finite_rows),
        "deviation_range": [min((row["relative_deviation"] for row in finite_rows),
                                default=float("nan")),
                            max((row["relative_deviation"] for row in finite_rows),
                                default=float("nan"))],
        "passed": bool(best is not None and best["relative_deviation"] <= HONDROS_TOLERANCE),
        "tolerance": HONDROS_TOLERANCE,
        "target": HONDROS_CENTER_RATIO,
    }

    # ---------------------------------------------------------------- AC6 negative control
    negative = [r for r in records if str(r.get("point", "")).startswith("neg_")]
    negative_ucs = [r["peak_stress_Pa"] / 1e6 for r in negative]
    lo, hi = UCS_BAND_MPA
    violating = [value for value in negative_ucs if not (lo <= value <= hi)]
    fraction = len(violating) / len(negative_ucs) if negative_ucs else float("nan")
    report["AC6_negative_control"] = {
        "n_samples": len(negative_ucs),
        "fraction_outside_ucs_band": fraction,
        "passed": bool(fraction >= 0.95),
        "ucs_MPa_quantiles": {
            "min": float(np.min(negative_ucs)) if negative_ucs else float("nan"),
            "q05": float(np.quantile(negative_ucs, 0.05)) if negative_ucs else float("nan"),
            "median": float(np.median(negative_ucs)) if negative_ucs else float("nan"),
            "q95": float(np.quantile(negative_ucs, 0.95)) if negative_ucs else float("nan"),
            "max": float(np.max(negative_ucs)) if negative_ucs else float("nan"),
        },
        "ucs_MPa": negative_ucs,
    }

    # ---------------------------------------------------------------- protocol refinement
    convergence = []
    for record in records:
        name = str(record.get("point", ""))
        if not name.startswith("conv_"):
            continue
        factor = None
        for token in name.split("_"):
            if token.endswith("x") and token[:-1].isdigit():
                factor = int(token[:-1])
        convergence.append({"point": name,
                            "kind": record["kind"],
                            "factor": factor,
                            "increments": record.get("increments"),
                            "thickness_m": record.get("thickness_m"),
                            "peak_MPa": record["peak_stress_Pa"] / 1e6,
                            "ke_ratio": record["ke_ratio"]})
    report["protocol_refinement"] = {
        "rows": convergence,
        "note": "same theta, 1x/2x/4x the certified increment count",
    }

    # ---------------------------------------------------------------- AC7 reproducibility + gate
    repeats = [r for r in records if str(r.get("point", "")).startswith("repeat_")]
    if len(repeats) >= 2:
        values = [r["peak_stress_Pa"] for r in repeats]
        relative = abs(values[0] - values[1]) / max(abs(values[0]), 1e-30)
    else:
        relative = float("nan")
    ke_values = [points[name]["gates"]["ucs_ke_ratio"] for name in points]
    ke_values += [points[name]["gates"]["max_family_ke_ratio"] for name in points]
    report["AC7_reproducibility"] = {
        "repeat_ucs_Pa": [r["peak_stress_Pa"] for r in repeats],
        "relative_difference": relative,
        "reproducible": bool(math.isfinite(relative) and relative <= 1e-3),
        "max_ke_ratio_all_runs": float(max(ke_values)) if ke_values else float("nan"),
        "quasistatic_gate_passed": bool(all(points[name]["gates"]["quasistatic_ok"]
                                            for name in points)),
        "ke_limit": QUASISTATIC_KE_LIMIT,
    }

    # ---------------------------------------------------------------- response surface table
    report["response_surface"] = {
        name: {
            "role": point["role"],
            "theta": point["theta"],
            "ucs_MPa": point["ucs_MPa"],
            "sigma_t_MPa": point["sigma_t_MPa"],
            "strength_ratio": point["strength_ratio"],
            "weibull_m": point["weibull_m"],
            "weibull_m_ci": point["weibull_m_ci"],
            "within_ucs_band": bool(UCS_BAND_MPA[0] <= point["ucs_MPa"] <= UCS_BAND_MPA[1]),
            "within_ratio_band": bool(STRENGTH_RATIO_BAND[0] <= point["strength_ratio"]
                                      <= STRENGTH_RATIO_BAND[1]),
            "ke_ratio": point["gates"]["ucs_ke_ratio"],
        }
        for name, point in points.items()
    }
    report["reference_trace"] = points["reference"]["ucs_trace"]
    report["reference_family"] = points["reference"]["family"]
    report["reference_size_effect"] = points["reference"]["size_effect"]

    # ---------------------------------------------------------------- AC8 tests are in pytest
    (ARTIFACTS / "report.json").write_text(json.dumps(report, indent=1))
    print(f"[analyze] wrote {ARTIFACTS/'report.json'}")
    print("[analyze] AC summary:",
          {key: report[key].get("passed", report[key].get("reproducible", None))
           for key in ("AC5_brazilian_field", "AC6_negative_control")})
    print("[analyze] identifiability verdict:", diagnostics["verdict"],
          "cond=%.3g" % diagnostics["cond"], "rank", diagnostics["rank"])
    print("[analyze] reference verdict branch:", reference_verdict["branch"],
          reference_verdict["satisfied"])
    print("[analyze] holdout passed:", report["AC4_holdout"]["n_passed"],
          "/", report["AC4_holdout"]["n_points"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sample", "analyze", "all"), default="all")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    args = parser.parse_args()
    if args.stage in ("sample", "all"):
        stage_sample(args.workers)
    if args.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()