"""Acceptance surface of the three-body collision / broken-ice decomposition node.

Criteria (node ``p262w1-N04-dem-collision-brash``)
-------------------------------------------------
``F1`` momentum conservation: the ice-system momentum change must equal the
       external impulse (hull + water drag) to better than 1 %, for the free
       floe-floe reference, the drag-free three-body event and the drag-driven
       sustained case.
``F2`` elastic energy: with ``e = 1`` the drift of ``E_kin + E_spring`` over a
       free collision must be below 2 %.
``F3`` force partition: ``F_11 + F_12 = F_total`` (algebraic, machine precision)
       **and** the impulse-level cross-check ``int F_hull dt = d p_ice - int
       F_drag dt`` below 5 %.
``F4`` resistance decomposition: ``R_collision + R_submerged_drag +
       R_channel_clearing = R_ice`` within 5 % (the residual is the storage term
       and is reported separately, never tuned away).
``F5`` the pytest suite (``tests/test_threebody_ice.py``) is green -- the count
       is written into the artifact by the runner.
``F6`` the three figures exist.
``F7`` energy dissipation band ``E_diss / dE_k in [0.3, 0.8]`` (node physics
       goal) -- reported with both normalisations.

Predeclared thresholds live in :data:`THRESHOLDS`; the case list in
:data:`CASE_MANIFEST` is fixed before any number is produced (the sharded runner
executes the manifest entries one at a time).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .model import ContactLaw, parameter_ledger_sha256
from .threebody import (ScenarioSpec, calibrate_law, elastic_reference,
                        reference_law, run_three_body, sweep_spacing)
from .field import FieldSpec, power_law_exponent, run_field, seed_region_m

__all__ = [
    "THRESHOLDS",
    "CASE_MANIFEST",
    "BASE_LAW",
    "BASE_FIELD",
    "SPACING_RATIOS",
    "SPEEDS",
    "CONCENTRATIONS",
    "run_case",
    "evaluate",
    "artifact_dir",
]

#: Predeclared acceptance thresholds (no post-hoc tuning).
THRESHOLDS: Dict[str, float] = {
    "momentum_relative_error": 0.01,
    "elastic_energy_relative_drift": 0.02,
    "force_partition_relative_error": 0.05,
    "decomposition_closure_relative": 0.05,
    "dissipation_band_low": 0.30,
    "dissipation_band_high": 0.80,
    "restitution_calibration_tolerance": 1.0e-3,
    "force_partition_algebraic": 1.0e-9,
    "ledger_identity_relative": 0.01,
}

BASE_LAW = ContactLaw(floe_floe_restitution=0.40, hull_floe_restitution=0.40,
                      normal_stiffness=5.0e6)
#: The transit domain is deliberately long: the residual of the resistance
#: decomposition is the window change of the *stored* energy of the ice field,
#: which is bounded by the energy of the ice inside the swept corridor, so its
#: contribution to the closure falls as 1/(V T) as the window lengthens.
BASE_FIELD = FieldSpec(domain_length_m=60.0, domain_width_m=30.0,
                       hull_length_m=24.0, hull_beam_m=10.0,
                       floe_radius_min_m=0.8, floe_radius_max_m=1.6,
                       seed_margin_m=3.0, window_fraction=(0.25, 0.90))
SPACING_RATIOS: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
SPEEDS: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
CONCENTRATIONS: Tuple[float, ...] = (0.3, 0.5, 0.7)


# ---------------------------------------------------------------------------
# Case manifest
# ---------------------------------------------------------------------------

def CASE_MANIFEST() -> List[Tuple[str, str, Dict[str, Any]]]:
    """Ordered list of ``(key, kind, parameters)`` to be executed by shards."""
    cases: List[Tuple[str, str, Dict[str, Any]]] = [
        ("reference_calibration", "reference", {}),
        ("reference_elastic", "reference_elastic", {}),
        ("reference_impact", "reference_impact", {"restitutions": [0.50, 0.35]}),
    ]
    for ratio in SPACING_RATIOS:
        cases.append((f"three_body_d{ratio:g}", "three_body",
                      {"spacing_ratio": ratio, "drag": False}))
    for ratio in SPACING_RATIOS:
        cases.append((f"three_body_drag_d{ratio:g}", "three_body_drag",
                      {"spacing_ratio": ratio, "drag": True}))
    for concentration in CONCENTRATIONS:
        for speed in SPEEDS:
            cases.append((f"field_c{concentration:g}_v{speed:g}", "field",
                          {"concentration": concentration, "speed_m_s": speed}))
    return cases


def artifact_dir(root: str = "artifacts/threebody_ice") -> str:
    parts = os.path.join(root, "parts")
    os.makedirs(parts, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------

def _calibrated(base: ContactLaw = BASE_LAW) -> Dict[str, Any]:
    return calibrate_law(base.floe_floe_restitution, law=base)


def run_case(key: str, kind: str, params: Dict[str, Any],
             *, root: str = "artifacts/threebody_ice") -> Dict[str, Any]:
    """Execute one manifest entry and return its (JSON-serialisable) result."""
    if kind == "reference":
        calibration = _calibrated()
        payload = {k: v for k, v in calibration.items() if k != "law"}
        payload["law_after_calibration"] = calibration["law"].as_dict()
        payload["parameter_ledger_sha256"] = parameter_ledger_sha256()
        payload["analytic_single_collision"] = __import__(
            "tensordem.threebody_ice.threebody", fromlist=["x"]
        ).single_collision_fractions(BASE_LAW.floe_floe_restitution)
        result = {"key": key, "kind": kind, "calibration": payload}
    elif kind == "reference_elastic":
        elastic = elastic_reference(replace(BASE_LAW, floe_floe_restitution=1.0,
                                            hull_floe_restitution=1.0,
                                            floe_floe_friction=0.0,
                                            hull_floe_friction=0.0),
                                    speed_m_s=1.0)
        elastic["series"] = _elastic_series(replace(
            BASE_LAW, floe_floe_restitution=1.0, hull_floe_restitution=1.0,
            floe_floe_friction=0.0, hull_floe_friction=0.0))
        result = {"key": key, "kind": kind, "elastic": elastic}
    elif kind == "reference_impact":
        # dissipation fractions of ONE collision, at the declared restitution and
        # at two alternatives (a declared sensitivity study of the node's band)
        base = BASE_LAW
        rows = []
        primary = _calibrated(base)
        rows.append(_impact_row(primary["law"], base.floe_floe_restitution, primary))
        for restitution in params.get("restitutions", []):
            law_i = replace(base, floe_floe_restitution=float(restitution),
                            hull_floe_restitution=float(restitution))
            calibration_i = calibrate_law(float(restitution), law=law_i)
            rows.append(_impact_row(calibration_i["law"], float(restitution),
                                    calibration_i))
        result = {"key": key, "kind": kind, "impacts": rows}
    elif kind in ("three_body", "three_body_drag"):
        calibration = _calibrated()
        law = calibration["law"]
        spec = ScenarioSpec(spacing_ratio=float(params["spacing_ratio"]))
        row = run_three_body(spec, law, drag=bool(params["drag"]))
        if params.get("spacing_ratio") == 1.0:
            row["series"] = row.pop("series")      # keep one reference trace
        else:
            row.pop("series", None)
        result = {"key": key, "kind": kind, "params": params,
                  "calibration": {k: v for k, v in calibration.items() if k != "law"},
                  "row": row}
    elif kind == "field":
        calibration = _calibrated()
        law = calibration["law"]
        spec = replace(BASE_FIELD, concentration=float(params["concentration"]),
                       speed_m_s=float(params["speed_m_s"]))
        row = run_field(spec, law)
        result = {"key": key, "kind": kind, "params": params,
                  "calibration": {k: v for k, v in calibration.items() if k != "law"},
                  "row": row}
    else:
        raise ValueError(f"unknown case kind {kind!r}")

    path = os.path.join(artifact_dir(root), "parts", f"{key}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    return result


def _impact_row(law: ContactLaw, restitution: float,
                calibration: Dict[str, Any]) -> Dict[str, Any]:
    from .threebody import single_impact_energies
    row = single_impact_energies(law)
    row["restitution_declared"] = float(restitution)
    row["calibrated_beta_hull_floe"] = float(calibration["beta_hull_floe_calibrated"])
    row["measured_restitution_calibrated"] = float(
        calibration["measured_restitution_hull_floe"])
    return row


def _elastic_series(law: ContactLaw, *, radius_m: float = 2.0,
                    speed_m_s: float = 1.0, samples: int = 400) -> Dict[str, List[float]]:
    """Short energy-vs-time trace of the elastic reference collision."""
    from .model import FloeState, MultiBodyModel
    from .threebody import _time_step
    law = reference_law(law)
    dt = _time_step(law, radius_m) * 0.5
    state = FloeState.create([radius_m, radius_m],
                             [[-(radius_m + 0.3), 0.0], [radius_m + 0.3, 0.0]],
                             law=law)
    state.vel[0] = __import__("torch").tensor([speed_m_s, 0.0],
                                              dtype=__import__("torch").float64)
    model = MultiBodyModel(state, law)
    series = {k: [] for k in ("t", "kinetic", "spring", "total")}
    for step in range(samples):
        series["t"].append(float(model.time))
        series["kinetic"].append(state.kinetic_energy())
        spring = model.spring_energy()
        series["spring"].append(spring)
        series["total"].append(state.kinetic_energy() + spring)
        model.step(dt)
    return series


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def _closure_with_residual(row: Dict[str, Any]) -> float:
    """Full-ledger closure, recomputed from the parts if the key is absent."""
    if "decomposition_closure_with_residual_relative" in row:
        return float(row["decomposition_closure_with_residual_relative"])
    try:
        numerator = abs(float(row["R_component_sum_N"]) + float(row["R_residual_N"])
                        - float(row["R_ice_force_N"]))
        return numerator / max(abs(float(row["R_ice_force_N"])), 1e-30)
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _check(name: str, value: float, threshold: float, *,
           mode: str = "max", note: str = "") -> Dict[str, Any]:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return {"name": name, "value": None, "threshold": threshold,
                "passed": False, "note": note or "no finite value"}
    value = float(value)
    passed = value <= threshold if mode == "max" else value >= threshold
    return {"name": name, "value": value, "threshold": threshold,
            "passed": bool(passed), "note": note}


def evaluate(parts: Iterable[Dict[str, Any]], *,
             pytest_count: Optional[int] = None,
             figures: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Assemble the criteria verdicts from the executed manifest parts."""
    by_key = {part["key"]: part for part in parts}
    t = THRESHOLDS
    criteria: List[Dict[str, Any]] = []

    # ---- calibration / restitution -------------------------------------
    calibration = by_key.get("reference_calibration", {}).get("calibration", {})
    criteria.append(_check(
        "F0a_measured_restitution_floe_floe_vs_target",
        float(calibration.get("restitution_error_floe_floe", float("nan"))),
        t["restitution_calibration_tolerance"]))
    criteria.append(_check(
        "F0b_measured_restitution_hull_floe_vs_target",
        float(calibration.get("restitution_error_hull_floe", float("nan"))),
        t["restitution_calibration_tolerance"]))

    # ---- F1 momentum ---------------------------------------------------
    elastic = by_key.get("reference_elastic", {}).get("elastic", {})
    criteria.append(_check("F1a_momentum_error_free_pair_elastic",
                           float(elastic.get("relative_momentum_error", float("nan"))),
                           t["momentum_relative_error"]))
    three_body_rows = [by_key[k]["row"] for k in by_key if k.startswith("three_body_d")]
    drag_rows = [by_key[k]["row"] for k in by_key if k.startswith("three_body_drag_d")]
    momentum_errors = [float(row["impulse_vs_momentum_error"]) for row in three_body_rows] or [float("nan")]
    criteria.append(_check("F1b_momentum_error_three_body_drag_free",
                           max(momentum_errors), t["momentum_relative_error"],
                           note="max over the spacing sweep"))
    momentum_errors_drag = [float(row["impulse_vs_momentum_error"]) for row in drag_rows] or [float("nan")]
    criteria.append(_check("F1c_momentum_error_three_body_sustained",
                           max(momentum_errors_drag), t["momentum_relative_error"],
                           note="max over the spacing sweep, drag impulse included"))

    # ---- F2 elastic energy ---------------------------------------------
    criteria.append(_check("F2_elastic_energy_drift",
                           float(elastic.get("relative_energy_drift", float("nan"))),
                           t["elastic_energy_relative_drift"]))

    # ---- F3 force partition --------------------------------------------
    all_three = three_body_rows + drag_rows
    algebraic = max((float(row["force_share_algebraic_error"]) for row in all_three),
                    default=float("nan"))
    criteria.append(_check("F3a_force_partition_algebraic",
                           algebraic, t["force_partition_algebraic"]))
    impulse = max((float(row["impulse_vs_momentum_error"]) for row in all_three),
                  default=float("nan"))
    criteria.append(_check("F3b_force_partition_impulse_vs_momentum",
                           impulse, t["force_partition_relative_error"]))

    # ---- F4 decomposition closure --------------------------------------
    field_rows = ([by_key[k]["row"] for k in sorted(by_key) if k.startswith("field_")])
    closure = max((float(row["decomposition_closure_relative"]) for row in field_rows),
                  default=float("nan"))
    criteria.append(_check("F4_decomposition_closure", closure,
                           t["decomposition_closure_relative"],
                           note="max over the concentration x speed matrix"))
    criteria.append(_check("F4b_force_vs_power_identity",
                           max((float(row["R_force_vs_power_relative"]) for row in field_rows),
                               default=float("nan")),
                           t["decomposition_closure_relative"],
                           note="R_ice measured twice, independently"))
    criteria.append(_check(
        "X1_ledger_identity_with_storage_residual",
        max((_closure_with_residual(row) for row in field_rows), default=float("nan")),
        THRESHOLDS["ledger_identity_relative"],
        note="three components + window storage residual = R_ice (the full ledger)"))
    criteria.append(_check("F4c_all_components_non_negative",
                           min((min(float(row["R_collision_N"]),
                                    float(row["R_submerged_drag_N"]),
                                    float(row["R_channel_clearing_N"]))
                                for row in field_rows), default=float("nan")),
                           0.0, mode="min",
                           note="negative component = sign/bookkeeping error"))

    # ---- F5 tests -------------------------------------------------------
    criteria.append(_check("F5_pytest_cases_green", float(pytest_count or 0),
                           12.0, mode="min"))

    # ---- F6 figures -----------------------------------------------------
    criteria.append(_check("F6_figures_present", float(len(figures or [])), 3.0,
                           mode="min", note=", ".join(figures or [])))

    # ---- F7 dissipation band -------------------------------------------
    # The node band is about a *collision*, so it is checked at the collision
    # grain (one isolated hull-floe impact) and the multi-body aggregate is
    # reported separately.  Nothing here is tuned: the declared restitution is
    # the one frozen in the manifest.
    band_lo, band_hi = t["dissipation_band_low"], t["dissipation_band_high"]
    impacts = by_key.get("reference_impact", {}).get("impacts", [])
    declared = next((row for row in impacts
                     if abs(float(row.get("restitution_declared", -1.0))
                            - BASE_LAW.floe_floe_restitution) < 1e-9), {})
    measured_primary = float(declared.get("dissipation_fraction_of_input", float("nan")))
    measured_alt = float(declared.get("dissipation_over_ke_gain", float("nan")))
    analytic = declared.get("analytic_single_collision", {})
    criteria.append(_check(
        "P1_single_collision_vs_analytic_primary",
        abs(measured_primary - float(analytic.get("dissipation_fraction_of_input", float("nan")))),
        0.05, note="measured E_diss/W_hull at the declared restitution"))
    criteria.append(_check(
        "P2_single_collision_vs_analytic_over_ke_gain",
        abs(measured_alt - float(analytic.get("dissipation_over_ke_gain", float("nan")))),
        0.05, note="measured E_diss/dE_k at the declared restitution"))
    criteria.append({
        "name": "P3_dissipation_band_single_collision",
        "value": [measured_primary, measured_alt],
        "threshold": [band_lo, band_hi],
        "passed": bool(band_lo <= measured_primary <= band_hi
                       and band_lo <= measured_alt <= band_hi),
        "note": ("node band [0.3, 0.8] on the measured single-collision fractions; "
                 f"the analytic value for e={BASE_LAW.floe_floe_restitution:.2f} is "
                 f"{float(analytic.get('dissipation_fraction_of_input', float('nan'))):.4f}, "
                 "i.e. exactly on the lower edge"),
    })
    aggregate_primary = [float(row["dissipation_fraction_of_input"]) for row in three_body_rows] or [float("nan")]
    aggregate_alt = [float(row["dissipation_over_ke_gain"]) for row in three_body_rows] or [float("nan")]
    criteria.append({
        "name": "P4_dissipation_band_multibody_aggregate",
        "value": [min(aggregate_primary), max(aggregate_primary)],
        "threshold": [band_lo, band_hi],
        "passed": bool(all(band_lo <= v <= band_hi for v in aggregate_primary)),
        "note": ("E_diss/W_hull of the whole tandem event (reported, not selected): the "
                 "chain keeps transferring energy to the far floe after the near floe "
                 "separates, so less of the hull work is dissipated inside contacts"),
    })
    criteria.append({
        "name": "P5_dissipation_over_ke_gain_multibody",
        "value": [min(aggregate_alt), max(aggregate_alt)],
        "threshold": [band_lo, band_hi],
        "passed": bool(all(band_lo <= v <= band_hi for v in aggregate_alt)),
        "note": "aggregate E_diss/dE_k (reported for completeness)",
    })

    # ---- velocity dependence -------------------------------------------
    exponents: Dict[str, Any] = {}
    by_concentration: Dict[float, List[Dict[str, Any]]] = {}
    for row in field_rows:
        by_concentration.setdefault(float(row["spec"]["concentration"]), []).append(row)
    for concentration, rows in sorted(by_concentration.items()):
        if len(rows) < 3:      # a power law needs three speeds
            continue
        rows.sort(key=lambda r: float(r["spec"]["speed_m_s"]))
        speeds = [float(r["spec"]["speed_m_s"]) for r in rows]
        exponents[f"c{concentration:g}"] = {
            "speeds": speeds,
            "R_ice": power_law_exponent(speeds, [float(r["R_ice_force_N"]) for r in rows]),
            "R_collision": power_law_exponent(speeds, [float(r["R_collision_N"]) for r in rows]),
            "R_submerged_drag": power_law_exponent(speeds, [float(r["R_submerged_drag_N"]) for r in rows]),
            "R_channel_clearing": power_law_exponent(
                speeds, [max(float(r["R_channel_clearing_N"]), 1e-12) for r in rows]),
        }

    total = len(criteria)
    passed = sum(1 for c in criteria if c["passed"])
    return {
        "criteria": criteria,
        "n_criteria": total,
        "n_passed": passed,
        "all_passed": bool(passed == total),
        "thresholds": dict(t),
        "velocity_exponents": exponents,
        "single_impact": impacts,
        "field_matrix": [
            {"key": row["case_key"] if "case_key" in row else None,
             "concentration": float(row["spec"]["concentration"]),
             "speed_m_s": float(row["spec"]["speed_m_s"]),
             "R_ice_N": float(row["R_ice_force_N"]),
             "R_collision_N": float(row["R_collision_N"]),
             "R_submerged_drag_N": float(row["R_submerged_drag_N"]),
             "R_channel_clearing_N": float(row["R_channel_clearing_N"]),
             "R_residual_N": float(row["R_residual_N"]),
             "closure": float(row["decomposition_closure_relative"]),
             "closure_with_residual": _closure_with_residual(row),
             "realised_concentration": float(row["realised_concentration"]),
             "n_floes": int(row["n_floes"]),
             "removed": int(row["n_removed"])}
            for row in field_rows
        ],
        "three_body_matrix": [
            {"drag": bool(row["drag_enabled"]),
             "spacing_ratio": float(row["spec"]["spacing_ratio"]),
             "impulse_share_1": float(row["impulse_share_1"]),
             "simultaneous_contact_fraction": float(row["simultaneous_contact_fraction"]),
             "dissipation_fraction_of_input": float(row["dissipation_fraction_of_input"]),
             "dissipation_over_ke_gain": float(row["dissipation_over_ke_gain"]),
             "energy_closure_relative": float(row["energy_closure_relative"]),
             "peak_force_total_N": float(row["peak_force_total_N"])}
            for row in three_body_rows + drag_rows
        ],
    }