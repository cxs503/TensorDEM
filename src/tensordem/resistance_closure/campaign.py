"""Campaign driver for the 3-D contact-to-resistance mapping closure (P316-W1-N09).

The campaign produces, for the pre-registered acceptance criteria of the node:

(a) **closure** -- the measured relative residual of
    ``R = R_crush + R_bend + R_sub`` over every sampled step of every case;
(b) **Lindqvist total** -- the relative deviation of the DEM total from the
    Lindqvist total at the same geometry, per-unit-width reduced;
(c) **crushing trend** -- the crushing share as a function of ice thickness;
(d) **grid refinement** -- the change of the resistance when the particle radius
    is halved at constant realised thickness;
(e) **negative control** -- zero ice load must give exactly zero components;
(f) **engineering deliverable** -- the pytest count and ``results.json``.

Nothing in the driver decides a verdict by judgement: every criterion is a
numeric comparison written into ``results.json`` next to the numbers it uses, and
``honest FAIL`` is a legal outcome (it is recorded, not overwritten).  Two extra
numerical axes are reported because they can masquerade as physics: the mass
scaling and the contact-penalty stiffness.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .dem import ClosureCase, run_case
from .mechanisms import plate_characteristic_length
from .reference import build_reference, compare_components

CLOSURE_TOLERANCE = 1e-3          # criterion (a): 0.1 %
LINDQVIST_TOLERANCE_PCT = 15.0    # criterion (b)
REFINEMENT_TOLERANCE_PCT = 5.0    # criterion (d)
MASS_SCALE_TOLERANCE_PCT = 10.0   # extra axis (documented, not a node criterion)


@dataclass(frozen=True)
class CampaignSpec:
    """Which cases the campaign runs; all parameters in SI."""

    base: ClosureCase = ClosureCase.physical(
        mass_scale=1600.0, particle_radius_m=0.04, thickness_m=0.40,
        n_length=16, n_width=4, bow_speed_m_s=1.0, wedge_angle_deg=30.0,
        contact_mask_every=20, embed_m=0.02, steps=1500,
    )
    thicknesses_m: tuple[float, ...] = (0.24, 0.40, 0.56)
    refinement_radius_m: float = 0.02
    mass_scale_axis: tuple[float, ...] = (400.0, 1600.0)

    def describe(self) -> dict[str, Any]:
        return {
            "base_case": _case_dict(self.base),
            "thicknesses_m": list(self.thicknesses_m),
            "refinement_radius_m": self.refinement_radius_m,
            "mass_scale_axis": list(self.mass_scale_axis),
        }


def _case_dict(case: ClosureCase) -> dict[str, Any]:
    return {
        key: (value if isinstance(value, (int, float, str, bool)) or value is None else str(value))
        for key, value in case.__dict__.items()
    }


def _reference_for(case: ClosureCase) -> dict[str, Any]:
    """Lindqvist reference at the model geometry plus the length-convention axis."""
    common = dict(
        thickness_m=case.realised_thickness_m,
        beam_m=case.model_width_m,
        entry_angle_deg=case.wedge_angle_deg,
        speed_m_s=case.bow_speed_m_s,
        friction_coefficient=case.friction_coefficient,
        compressive_strength_pa=case.compressive_strength_pa,
        flexural_strength_pa=case.flexural_strength_pa,
        density_kg_m3=case.density_kg_m3,
        water_density_kg_m3=case.water_density_kg_m3,
        gravity_m_s2=case.gravity_m_s2,
    )
    characteristic = plate_characteristic_length(
        case.elastic_modulus_pa, case.realised_thickness_m,
        case.water_density_kg_m3, case.gravity_m_s2,
    )
    model = build_reference(length_m=case.model_length_m, **common)
    physical = build_reference(length_m=characteristic, **common)
    return {
        "model_length": model.as_dict(),
        "characteristic_length": physical.as_dict(),
        "characteristic_length_m": characteristic,
    }


def evaluate_case(case: ClosureCase, *, label: str) -> dict[str, Any]:
    """Run one case and attach the Lindqvist comparison."""
    result = run_case(case)
    reference = _reference_for(case)
    comparison = compare_components(
        result["steady_per_width"],
        result["steady_per_width"]["r_total_N_m"],
        build_reference(
            thickness_m=case.realised_thickness_m,
            beam_m=case.model_width_m,
            length_m=case.model_length_m,
            entry_angle_deg=case.wedge_angle_deg,
            speed_m_s=case.bow_speed_m_s,
            friction_coefficient=case.friction_coefficient,
            compressive_strength_pa=case.compressive_strength_pa,
            flexural_strength_pa=case.flexural_strength_pa,
            density_kg_m3=case.density_kg_m3,
            water_density_kg_m3=case.water_density_kg_m3,
            gravity_m_s2=case.gravity_m_s2,
        ),
        tolerance_pct=LINDQVIST_TOLERANCE_PCT,
    )
    comparison_physical_length = compare_components(
        result["steady_per_width"],
        result["steady_per_width"]["r_total_N_m"],
        build_reference(
            thickness_m=case.realised_thickness_m,
            beam_m=case.model_width_m,
            length_m=reference["characteristic_length_m"],
            entry_angle_deg=case.wedge_angle_deg,
            speed_m_s=case.bow_speed_m_s,
            friction_coefficient=case.friction_coefficient,
            compressive_strength_pa=case.compressive_strength_pa,
            flexural_strength_pa=case.flexural_strength_pa,
            density_kg_m3=case.density_kg_m3,
            water_density_kg_m3=case.water_density_kg_m3,
            gravity_m_s2=case.gravity_m_s2,
        ),
        tolerance_pct=LINDQVIST_TOLERANCE_PCT,
    )
    # Line-load convention: Lindqvist's crushing term carries no beam, so the
    # per-metre reduction of that term may or may not include /B.  Both are
    # reported so the reader can see how much of the gap is a convention.
    line_load_total = (
        comparison["reference"]["components_per_width_N_m"]["crushing"] * case.model_width_m
        + comparison["reference"]["components_per_width_N_m"]["bending"]
        + comparison["reference"]["components_per_width_N_m"]["submersion"]
    )
    dem_total = result["steady_per_width"]["r_total_N_m"]
    line_load_error = (
        (dem_total - line_load_total) / line_load_total * 100.0 if line_load_total else None
    )
    return {
        "label": label,
        "case": result["case"],
        "case_config": _case_dict(case),
        "steady": result["steady"],
        "steady_per_width": result["steady_per_width"],
        "peak_total_N": result["peak_total_N"],
        "max_relative_residual": result["max_relative_residual"],
        "characteristic_length_m": result["characteristic_length_m"],
        "sheet_length_over_characteristic_length": result["sheet_length_over_characteristic_length"],
        "flexural_demand_ratio": result["flexural_demand_ratio"],
        "average_contact_pressure_pa": result["average_contact_pressure_pa"],
        "contact_patch_area_m2": result["contact_patch_area_m2"],
        "contact_patch_area_ratio": result["contact_patch_area_ratio"],
        "lindqvist": comparison,
        "lindqvist_characteristic_length": comparison_physical_length,
        "lindqvist_line_load": {
            "lindqvist_total_N_m": line_load_total,
            "error_pct": line_load_error,
        },
        "reference": reference,
        "series": result["series"],
    }


def negative_control(case: ClosureCase | None = None) -> dict[str, Any]:
    """No-load control: an empty contact set must decompose to exactly zero.

    Two independent controls are combined:

    * an empty :class:`~tensordem.resistance_closure.mechanisms.HullContactSet`
      (the pure-layer control, exercised in the test suite), and
    * a DEM case whose bow never reaches the ice (``embed_m = 0`` and a travel
      far below the initial clearance), so the *solver* must report zero on
      every component.
    """
    base = case or ClosureCase.physical(
        mass_scale=1.0, particle_radius_m=0.04, thickness_m=0.40, n_length=8, n_width=2,
        bow_speed_m_s=0.05, steps=5, embed_m=0.0, initial_gap_m=0.20, contact_mask_every=1,
    )
    result = run_case(base)
    components = result["steady"]["components"]
    return {
        "case": result["case"],
        "steady": result["steady"],
        "series_max_abs_total": max((abs(row["r_total_N"]) for row in result["series"]), default=0.0),
        "series_max_abs_component": max(
            (abs(row[f"r_{name}_N"]) for row in result["series"] for name in components),
            default=0.0,
        ),
        "zero": all(abs(value) == 0.0 for value in components.values()),
        "max_relative_residual": result["max_relative_residual"],
    }


def run_campaign(outdir: str | os.PathLike[str], spec: CampaignSpec | None = None,
                 *, log: Any = print) -> dict[str, Any]:
    """Run every case, evaluate the criteria, write ``results.json`` (resumable)."""
    spec = spec or CampaignSpec()
    out = Path(outdir)
    cases_dir = out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    def cached(label: str, case: ClosureCase) -> dict[str, Any]:
        path = cases_dir / f"{label}.json"
        if path.is_file():
            log(f"[cache] {label}")
            return json.loads(path.read_text())
        log(f"[run]   {label} ...")
        result = evaluate_case(case, label=label)
        path.write_text(json.dumps(result, indent=1))
        log(f"[done]  {label}: R={result['steady']['r_total']:.1f} N "
            f"shares={ {k: round(v, 3) for k, v in result['steady']['shares'].items()} }")
        return result

    base = spec.base
    results: dict[str, Any] = {}

    # --- thickness sweep (criterion c, and the base case of everything else) ---
    thickness_cases: dict[str, Any] = {}
    for thickness in spec.thicknesses_m:
        case = base.with_changes(thickness_m=thickness)
        label = f"thickness_{thickness:.2f}".replace(".", "p")
        thickness_cases[f"{thickness:.2f}"] = cached(label, case)
        results[label] = thickness_cases[f"{thickness:.2f}"]
    base_label = f"thickness_{base.thickness_m:.2f}".replace(".", "p")
    base_result = thickness_cases[f"{base.thickness_m:.2f}"]

    # --- grid refinement (criterion d), compared at equal bow advance ---
    base_travel = base.steps * base.bow_speed_m_s * base.to_config().recommended_dt
    refined_case = base.with_changes(particle_radius_m=spec.refinement_radius_m)
    refined = refined_case.with_changes(steps=refined_case.steps_for_travel(base_travel))
    refined_label = f"refined_r_{spec.refinement_radius_m:.3f}".replace(".", "p")
    log(f"[plan]  base travel {base_travel * 1000:.1f} mm, refined steps {refined.steps}")
    refined_result = cached(refined_label, refined)
    results[refined_label] = refined_result

    # --- mass-scaling axis (numerical, not a node criterion) ---
    mass_results: dict[str, Any] = {}
    for scale in spec.mass_scale_axis:
        if scale == base.mass_scale:
            mass_results[f"{scale:g}"] = base_result
            continue
        case = base.with_changes(mass_scale=scale)
        case = case.with_changes(steps=case.steps_for_travel(base_travel))
        label = f"mass_scale_{scale:g}"
        mass_results[f"{scale:g}"] = cached(label, case)
        results[label] = mass_results[f"{scale:g}"]

    # --- negative control (criterion e) ---
    control = negative_control()
    results["negative_control"] = control

    criteria = _evaluate_criteria(
        base_result=base_result,
        thickness_cases=thickness_cases,
        refined_result=refined_result,
        control=control,
        mass_results=mass_results,
    )
    report = {
        "protocol": "p316-n09-dem-resistance-3d-mapping-v1",
        "spec": spec.describe(),
        "base_label": base_label,
        "base": {
            key: base_result[key]
            for key in (
                "case", "steady", "steady_per_width", "peak_total_N", "max_relative_residual",
                "characteristic_length_m", "sheet_length_over_characteristic_length",
                "flexural_demand_ratio", "average_contact_pressure_pa", "contact_patch_area_m2",
                "contact_patch_area_ratio", "lindqvist", "lindqvist_characteristic_length",
                "lindqvist_line_load", "reference",
            )
        },
        "thickness_sweep": {
            key: _summary(value) for key, value in thickness_cases.items()
        },
        "refinement": {
            "coarse": _summary(base_result),
            "fine": _summary(refined_result),
        },
        "mass_scale_axis": {key: _summary(value) for key, value in mass_results.items()},
        "negative_control": {
            key: value for key, value in control.items() if key != "series"
        },
        "criteria": criteria,
        "verdict": "PASS" if all(c["passed"] for c in criteria.values()) else "FAIL",
        "case_files": sorted(path.name for path in cases_dir.glob("*.json")),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=1))
    (out / "series_base.json").write_text(json.dumps(base_result["series"], indent=1))
    log(f"[verdict] {report['verdict']}")
    for name, entry in criteria.items():
        log(f"   {name}: {'PASS' if entry['passed'] else 'FAIL'} -- {entry['detail']}")
    return report


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": result["case"],
        "steady": result["steady"],
        "steady_per_width": result["steady_per_width"],
        "max_relative_residual": result["max_relative_residual"],
        "lindqvist": result["lindqvist"],
        "lindqvist_characteristic_length": result["lindqvist_characteristic_length"],
        "lindqvist_line_load": result["lindqvist_line_load"],
        "lindqvist_total_error_pct": result["lindqvist"]["total_error_pct"],
        "lindqvist_total_error_pct_characteristic_length":
            result["lindqvist_characteristic_length"]["total_error_pct"],
        "lindqvist_line_load_error_pct": result["lindqvist_line_load"]["error_pct"],
        "contact_patch_area_ratio": result["contact_patch_area_ratio"],
        "average_contact_pressure_pa": result["average_contact_pressure_pa"],
        "sheet_length_over_characteristic_length": result["sheet_length_over_characteristic_length"],
        "flexural_demand_ratio": result["flexural_demand_ratio"],
    }


def _evaluate_criteria(*, base_result: dict[str, Any], thickness_cases: dict[str, Any],
                       refined_result: dict[str, Any], control: dict[str, Any],
                       mass_results: dict[str, Any]) -> dict[str, Any]:
    residuals = [entry["max_relative_residual"] for entry in thickness_cases.values()]
    residuals.append(refined_result["max_relative_residual"])
    worst_residual = max([abs(value) for value in residuals] + [abs(control["max_relative_residual"])])
    closure_passed = worst_residual <= CLOSURE_TOLERANCE

    total_error = base_result["lindqvist"]["total_error_pct"]
    lindqvist_passed = abs(total_error) <= LINDQVIST_TOLERANCE_PCT

    shares = [
        (float(key), thickness_cases[key]["steady"]["shares"]["crushing"])
        for key in sorted(thickness_cases, key=float)
    ]
    crush_trend_passed = all(
        later[1] > earlier[1] for earlier, later in zip(shares, shares[1:])
    )
    coarse_total = base_result["steady_per_width"]["r_total_N_m"]
    fine_total = refined_result["steady_per_width"]["r_total_N_m"]
    refinement_change = (
        abs(fine_total - coarse_total) / abs(coarse_total) * 100.0 if coarse_total else math.inf
    )
    refinement_passed = refinement_change <= REFINEMENT_TOLERANCE_PCT

    control_passed = bool(control["zero"]) and control["series_max_abs_total"] == 0.0

    totals = list(mass_results.values())
    spread = 0.0
    if len(totals) > 1:
        values = [entry["steady_per_width"]["r_total_N_m"] for entry in totals]
        mean = sum(values) / len(values)
        if mean:
            spread = (max(values) - min(values)) / abs(mean) * 100.0
    return {
        "a_closure_residual": {
            "passed": bool(closure_passed),
            "worst_relative_residual": worst_residual,
            "tolerance": CLOSURE_TOLERANCE,
            "detail": f"worst |residual|/|R| = {worst_residual:.3e} (tol {CLOSURE_TOLERANCE:g})",
        },
        "b_lindqvist_total": {
            "passed": bool(lindqvist_passed),
            "error_pct": total_error,
            "tolerance_pct": LINDQVIST_TOLERANCE_PCT,
            "dem_total_N_m": base_result["steady_per_width"]["r_total_N_m"],
            "lindqvist_total_N_m": base_result["lindqvist"]["lindqvist_total_N_m"],
            "signed_error_pct": base_result["lindqvist"]["total_signed_error_pct"],
            "detail": "DEM vs Lindqvist total error "
                      f"{base_result['lindqvist']['total_signed_error_pct']:+.1f}% "
                      f"(tol +/-{LINDQVIST_TOLERANCE_PCT:g}%)",
        },
        "c_crushing_trend": {
            "passed": bool(crush_trend_passed),
            "crushing_share_vs_thickness": shares,
            "detail": "crushing share increases with ice thickness: "
                      + ", ".join(f"{h:.2f} m -> {s:.3f}" for h, s in shares),
        },
        "d_grid_refinement": {
            "passed": bool(refinement_passed),
            "relative_change_pct": refinement_change,
            "tolerance_pct": REFINEMENT_TOLERANCE_PCT,
            "detail": f"halving the particle radius changes R by {refinement_change:.2f}% "
                      f"(tol {REFINEMENT_TOLERANCE_PCT:g}%)",
        },
        "e_negative_control": {
            "passed": bool(control_passed),
            "zero_components": bool(control["zero"]),
            "series_max_abs_total_N": control["series_max_abs_total"],
            "detail": "no-ice control gives exactly zero on every component: "
                      + ("yes" if control_passed else "NO"),
        },
        "mass_scale_axis": {
            "spread_pct": spread,
            "tolerance_pct": MASS_SCALE_TOLERANCE_PCT,
            "passed": bool(spread <= MASS_SCALE_TOLERANCE_PCT),
            "detail": f"mass-scaling spread {spread:.2f}% over {sorted(mass_results)} "
                      f"(documented numerical axis, tol {MASS_SCALE_TOLERANCE_PCT:g}%)",
        },
    }