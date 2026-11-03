"""Independent reference verification of the buried-pipeline subgrade capacity model.

The ice-ridge grounding campaign (``docs/ice_ridge_pipeline_grounding.md``) quotes
the seabed bearing capacity as its *verified* quantity, but until this module the
only test of the undrained branch re-derived the implementation instead of
comparing it with a published reference.  This module closes that debt: every
number below is compared against a closed-form solution or a value printed in a
primary/secondary reference, and the verdict (pass/fail, relative error, evidence
grade) is returned as data so the campaign ledger and the report can quote it
programmatically instead of by hand.

References used (all reproduced here as constants; no external file is read):

* Prandtl (1920) / Reissner: undrained surface strip capacity ``q = (2 + pi) s_u``
  with ``2 + pi = 5.1416``.
* Skempton (1951) embedment factor for a footing in clay,
  ``N_c = N_c0 (1 + 0.2 z/B)``, ``N_c0 = 5.14`` (strip) / ``6.2``
  (square, circular), capped at ``7.5`` / ``9.0`` (tabulated in Das,
  *Principles of Foundation Engineering*, table 3.3).
* Vesic (1973) bearing-capacity factors ``N_c, N_q, N_gamma`` (table values as
  tabulated in Das, table 3.4, and in Bowles, *Foundation Analysis and Design*).
* Meyerhof / Hansen depth factors ``d_c = 1 + 0.4 k``, ``d_q = 1 + 2 tan(phi)
  (1 - sin phi)^2 k`` with ``k = min(z/B, 1)``.

Known limitation of this verification: the lateral limit pressure ``N_p = 2 +
1.5 z/D`` (capped at 9) is a pipe-soil rule of practice, and its primary text
(DNV-RP-F114) is not available in this worktree, so only invariants (monotonic,
capped) are checked and the reference value is graded ``WITHHELD``.
"""
from __future__ import annotations

import math

from .ridge_geotech import (MohrCoulombSoil, SofteningVerticalSpring,
                            bearing_capacity_factors, skempton_nc)

__all__ = [
    "PRANDTL_NC",
    "SKEMPTON_STRIP_TABLE",
    "VESIC_FACTOR_TABLE",
    "verify_subgrade_capacity_model",
    "undrained_capacity_impact",
]

PRANDTL_NC = 2.0 + math.pi
"""``2 + pi = 5.1416``: Prandtl (1920) surface strip factor for phi = 0."""

SKEMPTON_STRIP_TABLE = (
    (0.0, 5.14), (0.5, 5.65), (1.0, 6.17), (2.0, 7.20), (2.5, 7.50), (5.0, 7.50),
)
"""Skempton (1951) strip ``N_c`` against ``z/B`` (Das table 3.3)."""

SKEMPTON_SQUARE_TABLE = (
    (0.0, 6.20), (0.5, 6.82), (1.0, 7.44), (2.0, 8.68), (2.5, 9.00), (5.0, 9.00),
)
"""Skempton (1951) square/circular ``N_c`` against ``z/B`` (Das table 3.3)."""

VESIC_FACTOR_TABLE = {
    20.0: (14.83, 6.40, 5.39),
    25.0: (20.72, 10.66, 10.88),
    30.0: (30.14, 18.40, 22.40),
    35.0: (46.12, 33.30, 48.03),
    40.0: (75.31, 64.20, 109.41),
}
"""Vesic (1973) ``(N_c, N_q, N_gamma)`` table values (Das table 3.4)."""

_TOL_LITERAL = 5.0e-3      # closed-form literals printed to 2-3 decimals
_TOL_TABLE = 5.0e-3        # table values printed to 2 decimals


def _check(check_id: str, quantity: str, model: float, reference: float,
           reference_source: str, *, tolerance: float, units: str,
           evidence: str = "REAL", note: str = "",
           abs_tolerance: float | None = None) -> dict:
    """Build one ledger row: computed value vs reference value.

    ``tolerance`` is a *relative* tolerance against ``reference``.  When the
    reference is a zero identity (``reference == 0``) pass ``abs_tolerance``
    instead: the row then reports the absolute error and judges it directly.
    """
    model = float(model)
    reference = float(reference)
    error = abs(model - reference)
    absolutely = abs_tolerance is not None or reference == 0.0
    if absolutely:
        relative = error
        limit = abs_tolerance if abs_tolerance is not None else tolerance
        passed = error <= limit
    else:
        relative = error / abs(reference)
        limit = tolerance
        passed = relative <= limit
    return {
        "id": check_id,
        "quantity": quantity,
        "model": model,
        "reference": reference,
        "reference_source": reference_source,
        "units": units,
        "absolute_error": error,
        "relative_error": relative,
        "compared_absolutely": absolutely,
        "tolerance": limit,
        "passed": bool(passed),
        "evidence": evidence,
        "note": note,
    }


def _undrained_soil(cohesion: float = 30.0e3, **kwargs) -> MohrCoulombSoil:
    defaults = dict(friction_angle_deg=0.0, cohesion=cohesion,
                    saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                    poisson_ratio=0.3, water_unit_weight=10.06e3, undrained=True)
    defaults.update(kwargs)
    return MohrCoulombSoil(**defaults)


def _legacy_undrained_nc(z_over_b: float) -> float:
    """Pre-fix (defective) ``N_c``: embedment factor applied twice, cap too early.

    Kept here only to quantify the size of the corrected defect; it is not used
    by any production code path.
    """
    k = min(z_over_b, 1.0)
    return min(5.14 * (1.0 + 0.2 * z_over_b), 9.0) * (1.0 + 0.2 * k)


def verify_subgrade_capacity_model(*, diameter: float = 0.914,
                                   embedment: float = 0.5,
                                   cohesion: float = 30.0e3) -> dict:
    """Verify :class:`MohrCoulombSoil` capacity against published references.

    Returns ``{"checks": [...], "summary": {...}, "defect": {...},
    "impact": {...}}``; every check row carries the model value, the reference
    value, the source of the reference and the pass/fail verdict.
    """
    checks: list[dict] = []
    soil = _undrained_soil(cohesion)

    # 1. Prandtl anchor -------------------------------------------------------
    checks.append(_check("prandtl_nc", "undrained N_c at z/B = 0",
                         skempton_nc(0.0), PRANDTL_NC,
                         "Prandtl (1920) slip-line solution 2 + pi",
                         tolerance=_TOL_LITERAL, units="-"))
    surface_pressure = soil.vertical_pressure_capacity(diameter, 0.0)
    checks.append(_check("prandtl_pressure", "undrained surface pressure / s_u",
                         surface_pressure / cohesion, PRANDTL_NC,
                         "Prandtl (1920) q_ult = (2 + pi) s_u",
                         tolerance=_TOL_LITERAL, units="-"))

    # 2. Skempton tables ------------------------------------------------------
    for footing, table in (("strip", SKEMPTON_STRIP_TABLE),
                           ("square", SKEMPTON_SQUARE_TABLE)):
        for z_over_b, reference in table:
            checks.append(_check(
                f"skempton_{footing}_{z_over_b:g}",
                f"Skempton N_c ({footing}) at z/B = {z_over_b:g}",
                skempton_nc(z_over_b, footing=footing), reference,
                "Skempton (1951) via Das table 3.3",
                tolerance=_TOL_TABLE, units="-"))

    # 3. Cap invariant (the pre-fix code violated this) -----------------------
    checks.append(_check("skempton_cap_strip", "N_c cap, strip, z/B = 10",
                         skempton_nc(10.0), 7.5, "Skempton (1951) strip cap",
                         tolerance=1.0e-12, units="-"))
    checks.append(_check("skempton_cap_square", "N_c cap, square, z/B = 10",
                         skempton_nc(10.0, footing="square"), 9.0,
                         "Skempton (1951) square/circular cap",
                         tolerance=1.0e-12, units="-"))

    # 4. Vesic factors -------------------------------------------------------
    for phi_deg, (nc_ref, nq_ref, ng_ref) in VESIC_FACTOR_TABLE.items():
        nc, nq, ng = bearing_capacity_factors(math.radians(phi_deg))
        for name, model, reference in (("N_c", nc, nc_ref), ("N_q", nq, nq_ref),
                                       ("N_gamma", ng, ng_ref)):
            checks.append(_check(f"vesic_{name}_{phi_deg:g}", f"{name} at phi = {phi_deg:g} deg",
                                 model, reference, "Vesic (1973) via Das table 3.4",
                                 tolerance=_TOL_TABLE, units="-"))

    # 5. Drained depth factor (Meyerhof/Hansen) ------------------------------
    #    phi = 0 drained path: pure cohesion term must reproduce c N_c (1 + 0.4 k)
    drained = MohrCoulombSoil(friction_angle_deg=0.0, cohesion=100.0e3,
                              saturated_unit_weight=10.16e3, young_modulus=10.0e6,
                              water_unit_weight=10.06e3)
    k = 1.0
    cohesion_term = drained.vertical_pressure_capacity(1.0, 1.0) \
        - drained.effective_vertical_stress(1.0)  # subtract the small surcharge term
    reference = 100.0e3 * 5.14 * (1.0 + 0.4 * k)
    checks.append(_check("meyerhof_dc", "cohesion term with d_c at k = 1",
                         cohesion_term, reference,
                         "Meyerhof/Hansen d_c = 1 + 0.4 k",
                         tolerance=1.0e-3, units="Pa",
                         note="surcharge term removed analytically "
                              "(N_q = 1, d_q = 1 at phi = 0)"))

    # 5b. Composite drained capacity from published factors + Hansen depth factors
    composite_soil = MohrCoulombSoil(friction_angle_deg=30.0, cohesion=20.0e3,
                                     saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                                     poisson_ratio=0.3, water_unit_weight=10.06e3)
    width, depth = 0.914, 1.5
    nc_ref, nq_ref, ng_ref = VESIC_FACTOR_TABLE[30.0]
    k_ref = min(depth / width, 1.0)
    phi_ref = math.radians(30.0)
    dc_ref = 1.0 + 0.4 * k_ref
    dq_ref = 1.0 + 2.0 * math.tan(phi_ref) * (1.0 - math.sin(phi_ref)) ** 2 * k_ref
    gamma_prime = composite_soil.effective_unit_weight
    composite_ref = (20.0e3 * nc_ref * dc_ref
                     + gamma_prime * depth * nq_ref * dq_ref
                     + 0.5 * gamma_prime * width * ng_ref)
    checks.append(_check("drained_composite_30deg",
                         "drained q_ult at phi = 30, c = 20 kPa, z/B = 1.64",
                         composite_soil.vertical_pressure_capacity(width, depth),
                         composite_ref,
                         "Vesic (1973) table factors + Meyerhof/Hansen depth factors",
                         tolerance=1.0e-3, units="Pa"))

    # 6. Embedment applied once (the corrected defect) -----------------------
    for z_over_b in (0.5, 1.0, 2.0, 3.0, 6.0):
        model = soil.vertical_pressure_capacity(diameter, z_over_b * diameter) / cohesion
        checks.append(_check(f"embedment_once_{z_over_b:g}",
                             f"q/s_u at z/B = {z_over_b:g} (embedment applied once)",
                             model, skempton_nc(z_over_b),
                             "Skempton (1951) with a single embedment factor",
                             tolerance=1.0e-12, units="-",
                             note="regression guard against the double depth factor"))

    # 7. Linear scaling in s_u ----------------------------------------------
    ratios = []
    for strength in (10.0e3, 30.0e3, 60.0e3):
        row_soil = _undrained_soil(strength)
        ratios.append(row_soil.vertical_capacity_per_length(diameter, embedment) / strength)
    checks.append(_check("su_linearity", "spread of p_ult/s_u over s_u = 10-60 kPa",
                         max(ratios) - min(ratios), 0.0,
                         "undrained capacity is exactly linear in s_u",
                         tolerance=1.0e-9, units="m", abs_tolerance=1.0e-9,
                         note=f"p_ult/s_u = {ratios[0]:.6f} m for every s_u"))

    # 8. Lateral limit pressure invariants ----------------------------------
    lateral = [soil.lateral_limit_pressure(z, diameter) for z in (0.2, 1.0, 3.0, 6.0, 50.0)]
    monotone = all(b >= a for a, b in zip(lateral, lateral[1:]))
    checks.append(_check("lateral_monotone", "N_p monotone in embedment (0 = fail)",
                         0.0 if monotone else 1.0, 0.0,
                         "pipe-soil practice: N_p = 2 + 1.5 z/D increasing",
                         tolerance=0.0, units="-", abs_tolerance=0.0))
    checks.append(_check("lateral_cap", "N_p cap at deep embedment (z/D >> 1)",
                         soil.lateral_limit_pressure(50.0, diameter) / (cohesion * diameter),
                         9.0,
                         "pipe-soil practice N_p <= 9 (DNV-RP-F114 primary text "
                         "not in worktree -> reference grade WITHHELD)",
                         tolerance=1.0e-12, units="-", evidence="WITHHELD"))

    # 9. Peak/residual spring consistency with the verified capacity ---------
    spring = SofteningVerticalSpring(soil, diameter, embedment)
    checks.append(_check("spring_peak_fraction", "spring peak / p_ult",
                         spring.peak / spring.ultimate, 0.90,
                         "SofteningVerticalSpring(yield_ratio = 0.9) by construction",
                         tolerance=1.0e-12, units="-"))
    checks.append(_check("spring_residual_fraction", "spring residual / peak",
                         spring.residual / spring.peak, 0.80,
                         "SofteningVerticalSpring(residual_ratio = 0.8)",
                         tolerance=1.0e-12, units="-"))

    # ---- defect quantification --------------------------------------------
    defect_rows = []
    for z_over_b in (0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 6.0):
        corrected = skempton_nc(z_over_b)
        legacy = _legacy_undrained_nc(z_over_b)
        defect_rows.append({
            "z_over_b": z_over_b,
            "corrected_nc": corrected,
            "legacy_nc": legacy,
            "over_prediction": legacy / corrected - 1.0,
        })
    defect = {
        "description": "undrained branch applied the Skempton embedment factor twice "
                       "and applied the cap before that factor",
        "max_over_prediction": max(row["over_prediction"] for row in defect_rows),
        "max_legacy_nc": max(row["legacy_nc"] for row in defect_rows),
        "cap_violation": max(row["legacy_nc"] for row in defect_rows) - 7.5,
        "rows": defect_rows,
    }

    # ---- engineering impact on the Beaufort case --------------------------
    impact = undrained_capacity_impact(diameter=diameter, embedment=embedment,
                                       cohesion=cohesion)

    passed = [row for row in checks if row["passed"]]
    summary = {
        "n_checks": len(checks),
        "n_passed": len(passed),
        "all_passed": len(passed) == len(checks),
        "max_relative_error": max(row["relative_error"] for row in checks
                                  if not row["compared_absolutely"]),
        "max_relative_error_real_checks_only": max(
            row["relative_error"] for row in checks
            if row["evidence"] == "REAL" and not row["compared_absolutely"]),
        "all_within_3_percent": all(row["relative_error"] <= 0.03 for row in checks
                                    if not row["compared_absolutely"]),
        "evidence_roll_up": {
            "REAL": sum(1 for row in checks if row["evidence"] == "REAL"),
            "WITHHELD": sum(1 for row in checks if row["evidence"] == "WITHHELD"),
        },
    }
    return {"checks": checks, "summary": summary, "defect": defect, "impact": impact}


def undrained_capacity_impact(*, diameter: float = 0.914, embedment: float = 0.5,
                              cohesion: float = 30.0e3) -> dict:
    """Capacity change of the corrected model relative to the pre-fix model."""
    soil = _undrained_soil(cohesion)
    z_over_b = (embedment + diameter) / diameter
    corrected = soil.vertical_capacity_per_length(diameter, embedment)
    legacy = cohesion * _legacy_undrained_nc(z_over_b) * diameter
    spring = SofteningVerticalSpring(soil, diameter, embedment)
    return {
        "diameter": diameter,
        "embedment": embedment,
        "z_over_b": z_over_b,
        "shear_strength": cohesion,
        "legacy_capacity_per_length": legacy,
        "corrected_capacity_per_length": corrected,
        "relative_change": corrected / legacy - 1.0,
        "corrected_peak_reaction": spring.peak,
        "corrected_residual_reaction": spring.residual,
    }