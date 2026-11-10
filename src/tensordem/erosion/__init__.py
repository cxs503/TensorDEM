"""Solid-particle erosion (sand / slurry / ice-mélange abrasion) models.

Public surface
--------------
* :mod:`tensordem.erosion.materials` -- target-material library with provenance
  and the published constants of each model.
* :mod:`tensordem.erosion.models`    -- Finnie (1960), Oka et al. (2005),
  DNV-RP-O501 ductile model, brittle lateral-crack scaling, plus the
  velocity-exponent extractor used to check the models' scaling laws.
* :mod:`tensordem.erosion.ledger`    -- event-by-event mass/energy ledger that
  closes ``mass removed`` against ``particle kinetic-energy loss``.

Typical use for a ship/structure application (sand-laden seawater in a sea-chest
or a ballast line, ice-mélange abrasion of a hull coating)::

    from tensordem.erosion import MATERIALS, dnv_ductile_wall_loss_rate

    res = dnv_ductile_wall_loss_rate(
        v_m_s=12.0, alpha_deg=45.0, material="carbon_steel",
        sand_rate_kg_s=1e-3, area_m2=0.05, wall_thickness_mm=8.0,
    )
    res["wall_loss_mm_per_year"], res["life_years_to_thickness"]

The module is a constitutive sink for a DEM/CFD particle stream: it contains no
DEM machinery of its own, and it makes no claim that a published constant
transfers to a different erodent or wall alloy -- see the provenance strings.
"""
from __future__ import annotations

from .ledger import (
    FINNIE_GEOMETRIC_EFFICIENCY_MAX,
    ErosionLedger,
    ImpactEvent,
    closed_form_finnie_stream,
    implied_efficiency,
)
from .materials import (
    BRITTLE_CONSTANTS,
    DNV_DUCTILE_COEFFICIENTS,
    DNVDuctileCoefficients,
    LITERATURE_PEAK_ANGLES_DEG,
    LITERATURE_VELOCITY_EXPONENTS,
    MATERIALS,
    OKA_CONSTANTS,
    Material,
    dnv_coefficients,
    get_material,
)
from .models import (
    FINNIE_BRANCH_ANGLE_DEG,
    FINNIE_PEAK_ANGLE_DEG,
    SECONDS_PER_YEAR,
    brittle_angle_function,
    brittle_peak_angle_deg_numeric,
    brittle_relative_erosion,
    dnv_angle_function,
    dnv_ductile_erosion_ratio,
    dnv_ductile_wall_loss_rate,
    emergent_velocity_exponent,
    finnie_angle_function,
    finnie_erosion_ratio,
    finnie_peak_angle_deg,
    finnie_peak_angle_deg_numeric,
    finnie_peak_value,
    fit_velocity_exponent,
    oka_angle_function,
    oka_exponents,
    oka_peak_angle_deg,
    oka_peak_angle_deg_numeric,
    oka_relative_erosion,
)

__all__ = [
    "BRITTLE_CONSTANTS",
    "DNV_DUCTILE_COEFFICIENTS",
    "DNVDuctileCoefficients",
    "ErosionLedger",
    "FINNIE_BRANCH_ANGLE_DEG",
    "FINNIE_GEOMETRIC_EFFICIENCY_MAX",
    "FINNIE_PEAK_ANGLE_DEG",
    "ImpactEvent",
    "LITERATURE_PEAK_ANGLES_DEG",
    "LITERATURE_VELOCITY_EXPONENTS",
    "MATERIALS",
    "Material",
    "OKA_CONSTANTS",
    "SECONDS_PER_YEAR",
    "brittle_angle_function",
    "brittle_peak_angle_deg_numeric",
    "brittle_relative_erosion",
    "closed_form_finnie_stream",
    "dnv_angle_function",
    "dnv_coefficients",
    "dnv_ductile_erosion_ratio",
    "dnv_ductile_wall_loss_rate",
    "emergent_velocity_exponent",
    "finnie_angle_function",
    "finnie_erosion_ratio",
    "finnie_peak_angle_deg",
    "finnie_peak_angle_deg_numeric",
    "finnie_peak_value",
    "fit_velocity_exponent",
    "get_material",
    "implied_efficiency",
    "oka_angle_function",
    "oka_exponents",
    "oka_peak_angle_deg",
    "oka_peak_angle_deg_numeric",
    "oka_relative_erosion",
]