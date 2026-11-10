"""Independent references the DEM transit is compared against.

Three references are used, and each one's role is stated so that a comparison
cannot be mistaken for a validation of the wrong regime:

``lindqvist_reference``
    The in-repository Lindqvist three-component anchor
    (:func:`tensordem.level_ice.formula.lindqvist_components`) evaluated for the
    *same* ship (MT Uikku equivalent transport hull) and the *same* ice material
    as the DEM case.  Lindqvist is a **level-ice (intact sheet)** formula: it is
    an upper bound for a 60 %-concentration broken field, so it is used as the
    analytic cross-check of magnitude and of the speed/thickness trend, not as
    ground truth for broken ice.
``broken_ice_closed_form``
    The in-repository broken-ice closed-form anchor
    (:func:`tensordem.ice_resistance_scaling.broken_ice_anchor`): the same
    Lindqvist structure with the bending term suppressed when the floe diameter
    is below the characteristic bending length, plus a momentum-flux term.  It
    is the like-for-like *broken-ice* analytic reference.
``mt_uikku_level_ice_cases``
    The MT Uikku full-scale model-test table that ships with the repository.
    **Only the ten level-ice cases are transcribed**; the paper's broken-ice
    channel cases (304-306, 404-406) are not part of the shipped CSV, so the
    broken-ice field cannot be validated against experiment here.  The table is
    used (i) as the source of the DEM's measured ice material, and (ii) to score
    the Lindqvist anchor implementation against the paper's own prediction
    column -- an implementation check, not an independent DEM validation.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

from ..ice_resistance_scaling import ShipCase, broken_ice_anchor
from ..level_ice.formula import (TRANSPORT, HullForm, IceState, level_ice_cases,
                                 lindqvist_components)
from .ledger import CaseSpec, MaterialState


def _hull_form(spec: CaseSpec) -> HullForm:
    """The transport hull of the DEM case (same L, B, T, entry and stem angle)."""
    return HullForm(length_m=TRANSPORT.length_m, beam_m=TRANSPORT.beam_m,
                    draught_m=TRANSPORT.draught_m,
                    entry_angle_deg=TRANSPORT.entry_angle_deg,
                    stem_angle_deg=TRANSPORT.stem_angle_deg,
                    friction_coefficient=spec.friction_coefficient,
                    name="transport")


def _ice_state(spec: CaseSpec) -> IceState:
    material = MaterialState(thickness_m=spec.thickness_m,
                             compressive_strength_pa=spec.compressive_strength_pa)
    return IceState(thickness_m=material.thickness_m,
                    compressive_strength_pa=material.compressive_strength_pa,
                    flexural_strength_pa=material.flexural_strength_pa,
                    elastic_modulus_pa=material.elastic_modulus_pa,
                    density_kg_m3=material.ice_density_kg_m3)


def lindqvist_reference(spec: CaseSpec) -> dict[str, Any]:
    """Lindqvist three-component anchor for the case's ship, ice and speed."""
    payload = lindqvist_components(_ice_state(spec), _hull_form(spec),
                                   spec.speed_m_s,
                                   water_density_kg_m3=MaterialState().water_density_kg_m3,
                                   gravity_m_s2=MaterialState().gravity_m_s2)
    payload["role"] = ("level-ice (intact sheet) analytic upper bound for a "
                       "broken field; cross-check of magnitude and trend")
    payload["total_kN"] = payload["total_N"] / 1e3
    return payload


def broken_ice_closed_form(spec: CaseSpec) -> dict[str, Any]:
    """In-repository broken-ice closed-form anchor (like-for-like reference)."""
    material = MaterialState(thickness_m=spec.thickness_m,
                             compressive_strength_pa=spec.compressive_strength_pa)
    ship = ShipCase(
        length_m=TRANSPORT.length_m, beam_m=TRANSPORT.beam_m,
        draught_m=TRANSPORT.draught_m, speed_m_s=spec.speed_m_s,
        thickness_m=spec.thickness_m, concentration=spec.concentration,
        entry_angle_deg=TRANSPORT.entry_angle_deg,
        stem_angle_deg=TRANSPORT.stem_angle_deg,
        friction_coefficient=spec.friction_coefficient,
        strength_pa=material.compressive_strength_pa,
        flexural_strength_pa=material.flexural_strength_pa,
        elastic_modulus_pa=material.elastic_modulus_pa,
        floe_diameter_m=2.0 * spec.mean_floe_radius_m,
    )
    payload = broken_ice_anchor(ship)
    payload["total_kN"] = payload["total_N"] / 1e3
    payload["role"] = ("broken-ice closed-form reference (bending suppressed for "
                       "sub-bending-length floes, momentum flux added)")
    return payload


def mt_uikku_level_ice_cases(path: str | None = None) -> list[dict[str, Any]]:
    """The ten MT Uikku level-ice model-test cases (shipped reference table)."""
    return level_ice_cases(path)


def lindqvist_anchor_scores(cases: Iterable[dict[str, Any]] | None = None
                            ) -> dict[str, Any]:
    """Score the Lindqvist implementation against the paper's own column.

    Per case: the published ``lindqvist_prediction_kN`` is compared with this
    repository's Lindqvist anchor evaluated at the case's *own* measured ice
    properties, the transport hull and the case speed.  The residual measures
    how far the implementation's coefficient set sits from the published
    prediction column; it is an implementation check, not an independent
    validation (both sides are the same formula family).
    """
    rows = list(cases) if cases is not None else mt_uikku_level_ice_cases()
    scored = []
    for row in rows:
        published = row.get("lindqvist_prediction_kN")
        measured = row.get("experimental_reference_table8_kN")
        if not published:
            continue
        spec = CaseSpec(name=f"uikku_{row['test_id']}",
                        thickness_m=row["ice_thickness_m"],
                        speed_m_s=row["ice_drift_speed_m_s"],
                        compressive_strength_pa=1e3 * row["compressive_strength_kPa"],
                        friction_coefficient=0.05)
        anchor = lindqvist_reference(spec)
        # flexural strength / modulus of the *case* enter the bending term
        from ..level_ice.formula import lindqvist_components
        ice = IceState(
            thickness_m=row["ice_thickness_m"],
            compressive_strength_pa=1e3 * row["compressive_strength_kPa"],
            flexural_strength_pa=1e3 * row["flexural_strength_kPa"],
            elastic_modulus_pa=1e6 * row["ice_elastic_modulus_MPa"],
        )
        hull = HullForm(length_m=TRANSPORT.length_m, beam_m=TRANSPORT.beam_m,
                        draught_m=TRANSPORT.draught_m,
                        entry_angle_deg=TRANSPORT.entry_angle_deg,
                        stem_angle_deg=TRANSPORT.stem_angle_deg,
                        friction_coefficient=0.05, name="transport")
        ours = lindqvist_components(ice, hull, row["ice_drift_speed_m_s"])["total_N"] / 1e3
        scored.append({
            "test_id": row["test_id"],
            "ice_thickness_m": row["ice_thickness_m"],
            "ice_drift_speed_m_s": row["ice_drift_speed_m_s"],
            "published_lindqvist_kN": published,
            "implemented_lindqvist_kN": ours,
            "published_error_vs_experiment": (
                (published - measured) / measured if measured else None),
            "implemented_error_vs_experiment": (
                (ours - measured) / measured if measured else None),
            "implementation_vs_published_rel_error": (ours - published) / published,
            "anchor_total_kN": anchor["total_kN"],
        })
    errors = [abs(s["implementation_vs_published_rel_error"]) for s in scored]
    return {
        "cases": scored,
        "case_count": len(scored),
        "mean_abs_rel_error_vs_published": sum(errors) / len(errors) if errors else float("nan"),
        "note": ("both sides are Lindqvist-family; a small residual says the "
                 "coefficient set spans the published column, it is NOT an "
                 "independent validation of Lindqvist"),
    }