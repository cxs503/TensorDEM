"""Salinity-strength differentiation of multi-year (MYI) and first-year (FYI) ice.

Public surface: :func:`brine_salinity` (Cox & Weeks liquidus),
:func:`brine_volume_fraction` (Cox & Weeks mass balance),
:func:`compressive_strength` (Timco & Frederking porosity law, FY/MY classes),
:func:`monte_carlo_strength` (confidence bands) and the grid helpers used by
``scripts/run_salinity_strength.py`` for the report figures and ``results.json``.

This module is deliberately a *constitutive* library: it produces the strength
envelope that a DEM/BEM contact model consumes as a material parameter.  It
contains no DEM machinery of its own.
"""
from __future__ import annotations

from .core import (
    BRINE_DENSITY_SLOPE,
    DEFAULT_PARAMETERS,
    ICECLASS_DEFAULT_SALINITY,
    LIQUIDUS_A1,
    LIQUIDUS_A2,
    PARAMETER_PROVENANCE,
    RHO_ICE_G_CM3,
    StrengthParameters,
    brine_density,
    brine_salinity,
    brine_volume_fraction,
    compressive_strength,
    monte_carlo_strength,
    porosity_strength_factor,
    salinity_scan,
    spearman_rho,
    strength_surface,
    temperature_scan,
)

__all__ = [
    "BRINE_DENSITY_SLOPE",
    "DEFAULT_PARAMETERS",
    "ICECLASS_DEFAULT_SALINITY",
    "LIQUIDUS_A1",
    "LIQUIDUS_A2",
    "PARAMETER_PROVENANCE",
    "RHO_ICE_G_CM3",
    "StrengthParameters",
    "brine_density",
    "brine_salinity",
    "brine_volume_fraction",
    "compressive_strength",
    "monte_carlo_strength",
    "porosity_strength_factor",
    "salinity_scan",
    "spearman_rho",
    "strength_surface",
    "temperature_scan",
]