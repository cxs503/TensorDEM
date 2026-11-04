"""Verification helpers for the ice-ridge grounding campaign (verification only).

The campaign of ``docs/ice_ridge_pipeline_grounding.md`` publishes a set of
numbers that come from two very different sources:

* **solver output** -- the coupled DEM / subgrade / beam record
  (``artifacts/ridge_grounding/*.json``);
* **closed form** -- Skempton undrained bearing capacity, Hetenyi beam-on-Winkler
  solutions, and the linear-in-``s_u`` scaling of the subgrade capacity.

This module exists because those two sources drifted apart once already.  The
undrained capacity branch carried a double-counted depth factor until m2983
(see ``docs/ridge_capacity_skempton_audit.md``); after the fix the committed
artifacts of the campaign were **not** regenerated, so the published soil-strength
sweep (``73.8 / 221.5 / 442.9 kN/m``) was 20 % too high while the code that
produced it no longer existed.  The functions here re-derive every closed-form
quantity from the artifacts and compare the *documented* numbers with the
committed artifacts, so a stale number is a red test rather than a footnote.

Nothing in this module touches the force law, the subgrade model or the beam
solver: it only reads dictionaries and returns a verdict table.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping

from .ridge_geotech import MohrCoulombSoil, skempton_undrained_nc

#: Skempton (1951) surface value for a weightless strip on phi = 0 clay.
SKEMPTON_STRIP_SURFACE = 5.14
#: Skempton deep-mechanism caps: strip 7.5, square/circular 9.0 (= 7.5 * 1.2).
SKEMPTON_STRIP_CAP = 7.5
SKEMPTON_SQUARE_CAP = 9.0
#: Square/circular shape factor ``1 + 0.2 B/L`` at ``B/L = 1``.
SQUARE_SHAPE_FACTOR = 1.2


def legacy_undrained_nc_pre_m2983(depth_below_mudline: float, width: float) -> float:
    """Undrained ``N_c`` **as implemented before m2983**, kept for the audit ratio.

    The pre-fix branch multiplied the Skempton strip surface value by the
    *square-footing* shape factor ``1.2`` and capped the result at the square
    cap ``9.0`` even for the plane-strain strip model the subgrade represents.
    That is the double count documented in ``docs/ridge_capacity_skempton_audit.md``:

        N_c_old = min(5.14 * 1.2 * (1 + 0.2 z/B), 9.0)
        N_c_new = min(5.14       * (1 + 0.2 z/B), 7.5)

    so ``N_c_old / N_c_new = 1.2`` whenever neither cap binds.  This function is
    *only* an audit reference; it must never be used for a design capacity.
    """
    if not (math.isfinite(depth_below_mudline) and math.isfinite(width)):
        raise ValueError("inputs must be finite")
    if width <= 0.0:
        raise ValueError("width must be positive")
    ratio = depth_below_mudline / width
    return min(SKEMPTON_STRIP_SURFACE * SQUARE_SHAPE_FACTOR * (1.0 + 0.2 * ratio),
               SKEMPTON_SQUARE_CAP)


def _rel_error(value: float, reference: float) -> float:
    if reference == 0.0:
        return math.inf if value else 0.0
    return abs(value - reference) / abs(reference)


def _row(quantity: str, value: float, reference: float, *, tolerance: float,
         unit: str = "", note: str = "") -> dict[str, Any]:
    error = _rel_error(value, reference)
    return {
        "quantity": quantity,
        "value": value,
        "reference": reference,
        "unit": unit,
        "relative_error": error,
        "tolerance": tolerance,
        "passed": bool(error <= tolerance),
        "note": note,
    }


# --------------------------------------------------------------------------- soil
def soil_strength_sweep_closed_form(
    rows: Iterable[Mapping[str, Any]], *, diameter: float, embedment: float
) -> list[dict[str, Any]]:
    """Cross-check the analytic subgrade sweep against Skempton in closed form.

    ``SofteningVerticalSpring.ultimate`` is
    ``s_u * N_c(z/B) * D`` with ``z = embedment + D`` (the pipe invert).  The
    reference here is :func:`tensordem.ridge_geotech.skempton_undrained_nc`
    evaluated *outside* the spring class, so the check is not self-referential.
    """
    depth = embedment + diameter
    reference_nc = skempton_undrained_nc(depth, diameter)
    out = []
    for row in rows:
        strength = float(row["undrained_shear_strength"])
        value = float(row["soil_ultimate_capacity"])
        reference = strength * reference_nc * diameter
        item = _row(
            f"subgrade p_ult at s_u = {strength / 1e3:g} kPa", value, reference,
            tolerance=1e-12, unit="N/m",
            note="s_u * N_c(z/B) * D, z = embedment + D",
        )
        item["implied_nc"] = value / (strength * diameter)
        item["skempton_nc"] = reference_nc
        item["depth_over_width"] = depth / diameter
        out.append(item)
    return out


def soil_strength_linearity(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """``p_ult / s_u`` must be one constant: the undrained branch is linear."""
    ratios = [float(r["soil_ultimate_capacity"]) / float(r["undrained_shear_strength"])
              for r in rows]
    spread = (max(ratios) - min(ratios)) / (0.5 * (max(ratios) + min(ratios)))
    return {
        "quantity": "p_ult / s_u spread over the sweep",
        "value": spread,
        "reference": 0.0,
        "unit": "-",
        "relative_error": spread,
        "tolerance": 1e-12,
        "passed": bool(spread <= 1e-12),
        "note": "constant ratio proves the linear undrained relation",
    }


def skempton_fix_factor(rows: Iterable[Mapping[str, Any]], *,
                        diameter: float, embedment: float) -> list[dict[str, Any]]:
    """Prove the published pre-fix sweep is exactly ``1.2 x`` the fixed sweep.

    A ratio that comes out to ``1.2`` to machine precision identifies *which*
    factor was double counted.  It also shows that the re-derivation is a genuine
    re-execution of the fixed model and not a coincidence of the DEM.
    """
    depth = embedment + diameter
    legacy_nc = legacy_undrained_nc_pre_m2983(depth, diameter)
    fixed_nc = skempton_undrained_nc(depth, diameter)
    out = []
    for row in rows:
        strength = float(row["undrained_shear_strength"])
        fixed = float(row["soil_ultimate_capacity"])
        legacy = strength * legacy_nc * diameter
        out.append(_row(
            f"pre-fix / fixed subgrade capacity at s_u = {strength / 1e3:g} kPa",
            legacy / fixed, SQUARE_SHAPE_FACTOR, tolerance=1e-12, unit="-",
            note="legacy N_c carries the square shape factor on a strip model",
        ))
    return out


def check_published_soil_sweep(
    rows: Iterable[Mapping[str, Any]], published: Iterable[float], *,
    tolerance: float = 3e-3,
) -> list[dict[str, Any]]:
    """Compare the numbers *published in the doc* with the committed artifacts."""
    rows = list(rows)
    published = list(published)
    if len(rows) != len(published):
        raise ValueError("published sweep and artifact sweep differ in length")
    out = []
    for row, quoted in zip(rows, published):
        strength = float(row["undrained_shear_strength"]) / 1e3
        out.append(_row(
            f"published subgrade p_ult at s_u = {strength:g} kPa (kN/m)",
            float(quoted), float(row["soil_ultimate_capacity"]) / 1e3,
            tolerance=tolerance, unit="kN/m",
            note="doc number vs artifacts/ridge_grounding/sensitivity.json",
        ))
    return out


# ---------------------------------------------------------------------- winkler
def check_winkler(main: Mapping[str, Any], beam: Mapping[str, Any], *,
                  moment_tolerance: float = 0.03,
                  deflection_tolerance: float = 0.03) -> list[dict[str, Any]]:
    """Coupled moment/deflection and the standalone beam against Hetenyi."""
    profile = main["pipe_profile"]
    out = [
        _row("coupled peak moment vs Hetenyi", float(profile["peak_moment"]),
             float(profile["peak_moment_reference"]), tolerance=moment_tolerance,
             unit="N m", note="infinite-beam closed form on the same load set"),
        _row("coupled peak deflection vs Hetenyi", float(profile["peak_deflection"]),
             float(profile["peak_deflection_reference"]), tolerance=deflection_tolerance,
             unit="m", note="infinite-beam closed form on the same load set"),
    ]
    for case in ("point_load", "patch_load"):
        block = beam[case]
        out.append(_row(f"FE beam moment vs Hetenyi ({case})",
                        float(block["moment_fe"]),
                        float(block["moment_reference"]),
                        tolerance=moment_tolerance, unit="N m",
                        note="finite-element beam against the closed form"))
    out.append({
        "quantity": "beam finite-element convergence flag",
        "value": float(bool(profile.get("converged", True))),
        "reference": 1.0, "unit": "bool", "relative_error": 0.0, "tolerance": 0.0,
        "passed": bool(profile.get("converged", True)),
        "note": "mesh refinement flag from the beam solve",
    })
    return out


# ------------------------------------------------------------------------- rates
def check_rate_study(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Zero-rate extrapolation of the peak capacity.

    The DEM cannot reach the quasi-static limit at this resolution, so the
    campaign reports the trend and the linear extrapolation to ``v -> 0``
    instead of a single rate-dependent number.
    """
    rows = sorted(rows, key=lambda r: float(r["platen_velocity"]))
    v = [float(r["platen_velocity"]) for r in rows]
    p = [float(r["capacity_per_length"]) for r in rows]
    if len(v) < 2:
        raise ValueError("rate study needs at least two velocities")
    mean_v = sum(v) / len(v)
    mean_p = sum(p) / len(p)
    sxx = sum((x - mean_v) ** 2 for x in v)
    sxy = sum((x - mean_v) * (y - mean_p) for x, y in zip(v, p))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_p - slope * mean_v
    residual = max(abs(y - (intercept + slope * x)) for x, y in zip(v, p))
    scale = max(abs(y) for y in p)
    return {
        "quantity": "peak capacity extrapolated to zero platen velocity",
        "value": intercept,
        "reference": p[0],
        "unit": "N/m",
        "relative_error": _rel_error(intercept, p[0]),
        "tolerance": 0.15,
        "passed": bool(_rel_error(intercept, p[0]) <= 0.15),
        "note": "linear fit in v; reference is the slowest affordable run",
        "slope_per_metre_per_second": slope,
        "linearity_residual_relative": residual / scale if scale else math.inf,
    }


# ---------------------------------------------------------------------- document
_DOC_PATTERNS = {
    "softening_percent": r"([0-9.]+) % post-peak\s+softening",
    "winkler_percent": r"\(([0-9.]+) % moment, ([0-9.]+) %\s*deflection",
    "flat_percent": r"flat within ±([0-9.]+) %",
    "soil_sweep": r"\(([0-9.]+) / ([0-9.]+) / ([0-9.]+) kN/m for\s+s_u = 10 / 30 / 60 kPa\)",
}


def parse_documented_numbers(text: str) -> dict[str, Any]:
    """Extract the campaign numbers that the narrative page quotes."""
    found: dict[str, Any] = {}
    for key, pattern in _DOC_PATTERNS.items():
        match = re.search(pattern, text)
        if match is None:
            continue
        groups = match.groups()
        found[key] = [float(g) for g in groups] if len(groups) > 1 else float(groups[0])
    return found


def check_documented_numbers(
    documented: Mapping[str, Any], main: Mapping[str, Any],
    sensitivity: Mapping[str, Any], *, tolerance: float = 0.05,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Every parsed doc number must match the artifact that produced it."""
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    metrics = main["metrics"]

    def add(name: str, quoted: float, actual: float, unit: str, note: str) -> None:
        checks.append(_row(name, quoted, actual, tolerance=tolerance, unit=unit,
                           note=note))

    soil_rows = list(sensitivity["soil_strength"])
    if "soil_sweep" in documented:
        quoted = documented["soil_sweep"]
        checks.extend(check_published_soil_sweep(soil_rows, quoted,
                                                 tolerance=5e-3))
    else:
        missing.append("soil_sweep")

    for key in ("softening_percent", "winkler_percent", "flat_percent"):
        if key not in documented:
            missing.append(key)

    if "softening_percent" in documented:
        add("(a) post-peak softening %", documented["softening_percent"],
            float(metrics["softening_ratio"]) * 1e2, "%",
            "main_case.json metrics.softening_ratio")
    if "winkler_percent" in documented:
        moment, deflection = documented["winkler_percent"]
        add("(c) coupled moment error %", moment,
            float(main["pipe_profile"]["moment_relative_error"]) * 1e2, "%",
            "main_case.json pipe_profile.moment_relative_error")
        add("(c) coupled deflection error %", deflection,
            float(main["pipe_profile"]["deflection_relative_error"]) * 1e2, "%",
            "main_case.json pipe_profile.deflection_relative_error")
    if "flat_percent" in documented:
        spreads = []
        for key in ("keel_depth", "keel_width"):
            values = [float(r["capacity_per_length"]) for r in sensitivity[key]]
            spreads.append((max(values) - min(values)) / (max(values) + min(values)))
        add("(d) keel sweep ± half-spread", documented["flat_percent"],
            max(spreads) * 1e2, "%", "±(max-min)/(max+min) of each keel sweep; doc quotes the worse one")
    return checks, missing


# ------------------------------------------------------------------------ report
def build_report(
    artifacts: Mapping[str, Any], documented: Mapping[str, Any], *,
    diameter: float, embedment: float,
) -> dict[str, Any]:
    """Assemble every check into one machine-readable table plus a verdict."""
    main = artifacts["main_case"]
    sensitivity = artifacts["sensitivity"]
    document_checks, missing = check_documented_numbers(documented, main, sensitivity)

    soil_rows = list(sensitivity["soil_strength"])
    closed_form = soil_strength_sweep_closed_form(
        soil_rows, diameter=diameter, embedment=embedment)
    linearity = soil_strength_linearity(soil_rows)
    legacy = skempton_fix_factor(soil_rows, diameter=diameter, embedment=embedment)
    winkler = check_winkler(main, artifacts["beam"])
    rate = check_rate_study(artifacts["rate"]["rows"])

    checks = closed_form + [linearity] + legacy + winkler + [rate] + document_checks
    failures = [c for c in checks if not c["passed"]]
    return {
        "schema": "tensordem-ridge-grounding-rederivation-v1",
        "scenario_key": "ice_ridge_pipeline_grounding",
        "main_case_capacity_per_length": float(main["capacity_per_length"]),
        "subgrade_peak_capacity_per_length": float(
            main["soil_spring"]["peak_reaction"]),
        "subgrade_ultimate_capacity_per_length": float(
            main["soil_spring"]["ultimate_bearing_capacity"]),
        "checks": checks,
        "documented_numbers": dict(documented),
        "missing_documented_numbers": missing,
        "failure_count": len(failures),
        "failed_quantities": [c["quantity"] for c in failures],
        "verdict": "PASS" if not failures else "FAIL",
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Markdown table of the verdicts (used by the docs and the report)."""
    lines = [
        "| quantity | published / value | independent reference | rel. error | tol. | pass |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for check in report["checks"]:
        value = check["value"]
        reference = check["reference"]
        lines.append(
            f"| {check['quantity']} | {value:.6g} {check['unit']} | "
            f"{reference:.6g} {check['unit']} | "
            f"{check['relative_error']:.2e} | {check['tolerance']:.0e} | "
            f"{'yes' if check['passed'] else '**no**'} |")
    lines.append("")
    lines.append(f"verdict: **{report['verdict']}** "
                 f"({report['failure_count']} failed checks)")
    return "\n".join(lines)


def beaufort_soil(undrained_shear_strength: float = 30.0e3,
                  *, saturated_unit_weight: float = 18.0e3,
                  water_unit_weight: float = 10.06e3) -> MohrCoulombSoil:
    """The campaign soil deck (rebuilt here so the checks need no artifact)."""
    return MohrCoulombSoil(
        friction_angle_deg=0.0, cohesion=undrained_shear_strength,
        saturated_unit_weight=saturated_unit_weight, young_modulus=10.0e6,
        poisson_ratio=0.3, water_unit_weight=water_unit_weight, undrained=True)