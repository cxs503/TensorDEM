"""Submersion and momentum-flux correction for plan-view DEM ice resistance.

The plan-view DEM only captures horizontal contact forces (crushing-like).
It misses two key mechanisms:

1. Submersion resistance: ice floes must be pushed down against buoyancy
   as the hull clears a channel. This is the largest missing term (~240 kN
   for the standard case).

2. Momentum flux: ice deflected sideways carries momentum, requiring a
   reaction force on the hull (~21 kN for the standard case).

This module provides analytical corrections based on the broken-ice closed-form
anchor (ice_resistance_scaling.broken_ice_anchor) that can be added to the
DEM contact forces to produce a hybrid prediction.

The correction is physics-based, not curve-fitted: the submersion term comes
from buoyant depression of the cleared channel, and the momentum term from
conservation of momentum for the deflected ice.

All quantities SI; resistances in N unless noted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .ledger import CaseSpec, MaterialState
from ..level_ice.formula import TRANSPORT

WATER_DENSITY = 1025.0  # kg/m^3
ICE_DENSITY = 917.0  # kg/m^3
GRAVITY = 9.80665  # m/s^2


@dataclass(frozen=True)
class SubmersionCorrection:
    """Analytical correction for missing submersion and momentum physics.
    
    Attributes
    ----------
    submersion_N : float
        Buoyant depression + rubble friction resistance [N].
    momentum_N : float
        Momentum flux of deflected ice [N].
    total_correction_N : float
        Sum of submersion and momentum [N].
    dem_contact_N : float
        DEM contact force (input) [N].
    hybrid_total_N : float
        DEM + correction [N].
    correction_fraction : float
        Correction as fraction of hybrid total [-].
    """
    submersion_N: float
    momentum_N: float
    total_correction_N: float
    dem_contact_N: float
    hybrid_total_N: float
    correction_fraction: float
    
    def as_dict(self) -> dict[str, Any]:
        return {
            "submersion_N": self.submersion_N,
            "momentum_N": self.momentum_N,
            "total_correction_N": self.total_correction_N,
            "dem_contact_N": self.dem_contact_N,
            "hybrid_total_N": self.hybrid_total_N,
            "correction_fraction": self.correction_fraction,
        }


def compute_submersion_resistance(spec: CaseSpec) -> float:
    """Compute submersion resistance for the case.
    
    R_s = rho_w * g * h^2 * B * (k_sub + k_fric * mu * L / B)
    
    where:
    - k_sub = 0.9 (channel-opening coefficient, displaced ice volume fraction)
    - k_fric = 0.05 (rubble friction over hull)
    - B = ship beam
    - L = ship length
    - h = ice thickness
    - mu = friction coefficient
    
    This is the same formula as in ice_resistance_scaling.broken_ice_anchor,
    extracted here for clarity and independent use.
    """
    h = spec.thickness_m
    B = TRANSPORT.beam_m
    L = TRANSPORT.length_m
    mu = spec.friction_coefficient
    
    k_sub = 0.9
    k_fric = 0.05
    
    R_s = WATER_DENSITY * GRAVITY * h**2 * B * (k_sub + k_fric * mu * L / B)
    return R_s


def compute_momentum_flux(spec: CaseSpec) -> float:
    """Compute momentum flux resistance for the case.
    
    R_m = k_m * rho_i * h * B * V^2
    
    where:
    - k_m = 0.1 (momentum flux coefficient, from BROKEN_ICE_LEDGER)
    - rho_i = ice density
    - h = ice thickness
    - B = ship beam
    - V = ship speed
    
    This is the same formula as in ice_resistance_scaling.broken_ice_anchor.
    """
    h = spec.thickness_m
    B = TRANSPORT.beam_m
    V = spec.speed_m_s
    
    k_m = 0.1  # momentum flux coefficient (from BROKEN_ICE_LEDGER)
    
    R_m = k_m * ICE_DENSITY * h * B * V**2
    return R_m


def apply_correction(spec: CaseSpec, dem_contact_N: float) -> SubmersionCorrection:
    """Apply submersion and momentum correction to DEM contact force.
    
    Parameters
    ----------
    spec : CaseSpec
        Case specification (ship geometry, ice properties, speed).
    dem_contact_N : float
        DEM contact force (horizontal reaction) [N].
    
    Returns
    -------
    SubmersionCorrection
        Correction details and hybrid total.
    """
    R_sub = compute_submersion_resistance(spec)
    R_mom = compute_momentum_flux(spec)
    R_corr = R_sub + R_mom
    R_hybrid = dem_contact_N + R_corr
    
    frac = R_corr / R_hybrid if R_hybrid > 0 else 0.0
    
    return SubmersionCorrection(
        submersion_N=R_sub,
        momentum_N=R_mom,
        total_correction_N=R_corr,
        dem_contact_N=dem_contact_N,
        hybrid_total_N=R_hybrid,
        correction_fraction=frac,
    )


def concentration_correction_factor(concentration: float) -> float:
    """Correction factor for ice concentration.
    
    The analytical formulas assume a fully covered channel (C=1).
    For partial concentration, the resistance scales approximately as C^delta
    where delta ~ 1 (linear in concentration).
    
    Parameters
    ----------
    concentration : float
        Ice concentration (0 to 1).
    
    Returns
    -------
    float
        Multiplicative correction factor.
    """
    if not 0.0 <= concentration <= 1.0:
        raise ValueError(f"concentration must be in [0, 1], got {concentration}")
    
    # Linear scaling: R ~ C^1
    # For C=0.6, factor = 0.6
    return concentration


def apply_full_correction(
    spec: CaseSpec,
    dem_contact_N: float,
    apply_concentration: bool = True
) -> SubmersionCorrection:
    """Apply full correction including concentration scaling.
    
    Parameters
    ----------
    spec : CaseSpec
        Case specification.
    dem_contact_N : float
        DEM contact force [N].
    apply_concentration : bool
        Whether to apply concentration correction (default True).
    
    Returns
    -------
    SubmersionCorrection
        Correction with concentration scaling applied.
    """
    # Base correction (submersion + momentum)
    base = apply_correction(spec, dem_contact_N)
    
    # Apply concentration scaling if requested
    if apply_concentration and spec.concentration < 1.0:
        c_factor = concentration_correction_factor(spec.concentration)
        # Scale the correction by concentration
        R_sub_scaled = base.submersion_N * c_factor
        R_mom_scaled = base.momentum_N * c_factor
        R_corr_scaled = R_sub_scaled + R_mom_scaled
        R_hybrid_scaled = dem_contact_N + R_corr_scaled
        frac_scaled = R_corr_scaled / R_hybrid_scaled if R_hybrid_scaled > 0 else 0.0
        
        return SubmersionCorrection(
            submersion_N=R_sub_scaled,
            momentum_N=R_mom_scaled,
            total_correction_N=R_corr_scaled,
            dem_contact_N=dem_contact_N,
            hybrid_total_N=R_hybrid_scaled,
            correction_fraction=frac_scaled,
        )
    
    return base
