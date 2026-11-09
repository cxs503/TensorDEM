"""Ice ridge resistance in mixed ice fields.

This module extends the mixed ice field resistance model to include ice ridges,
which are common in Arctic waters and significantly increase resistance.

Ice ridges are formed when ice floes collide and raft/ridge. They consist of:
- A sail above water (typically 1-4 m)
- A keel below water (typically 5-20 m)
- A consolidated core with higher strength than level ice

The resistance model follows ISO 19906 and IACS UR I guidelines for ridge
interaction with ship hulls.

References
----------
- ISO 19906 (2019) "Petroleum and natural gas industries — Arctic offshore structures"
- IACS Unified Requirements for Polar Ships (UR I)
- Timco & Burden (2005) "Analysis of ice ridge loads on a wide structure"
- Keerthikan et al. (2015) "Ice ridge keel morphology"

All quantities are SI: m, Pa, kg/m^3, m/s, N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import aging
from .field import MixedFieldSpec
from .resistance import mixed_resistance
from ..level_ice.formula import ICEBREAKER, IceState, HullForm

__all__ = [
    "IceRidge",
    "RidgeFieldSpec",
    "ridge_resistance",
    "mixed_field_with_ridges",
]


@dataclass(frozen=True)
class IceRidge:
    """Ice ridge geometry and material properties.

    Parameters
    ----------
    sail_height_m : float
        Height of the ridge sail above water [m].
    keel_depth_m : float
        Depth of the ridge keel below water [m].
    width_m : float
        Width of the ridge at waterline [m].
    sail_angle_deg : float
        Angle of the sail face from horizontal [deg].
    keel_angle_deg : float
        Angle of the keel face from horizontal [deg].
    consolidated_fraction : float
        Fraction of the keel that is consolidated (0-1).
    porosity : float
        Porosity of the unconsolidated keel (0-1).
    compressive_strength_pa : float
        Compressive strength of consolidated core [Pa].
    """

    sail_height_m: float = 2.0
    keel_depth_m: float = 10.0
    width_m: float = 30.0
    sail_angle_deg: float = 30.0
    keel_angle_deg: float = 45.0
    consolidated_fraction: float = 0.3
    porosity: float = 0.3
    compressive_strength_pa: float = 2.0e6

    def __post_init__(self) -> None:
        for name in ("sail_height_m", "keel_depth_m", "width_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("sail_angle_deg", "keel_angle_deg"):
            value = float(getattr(self, name))
            if not (0.0 < value < 90.0):
                raise ValueError(f"{name} must be in (0, 90) degrees")
        for name in ("consolidated_fraction", "porosity"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        value = float(self.compressive_strength_pa)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("compressive_strength_pa must be positive and finite")

    def as_dict(self) -> dict[str, Any]:
        return {
            "sail_height_m": self.sail_height_m,
            "keel_depth_m": self.keel_depth_m,
            "width_m": self.width_m,
            "sail_angle_deg": self.sail_angle_deg,
            "keel_angle_deg": self.keel_angle_deg,
            "consolidated_fraction": self.consolidated_fraction,
            "porosity": self.porosity,
            "compressive_strength_pa": self.compressive_strength_pa,
        }


@dataclass(frozen=True)
class RidgeFieldSpec:
    """A mixed ice field with ice ridges.

    Parameters
    ----------
    my_fraction : float
        MY ice area fraction (0-1).
    ridge_density_per_km : float
        Number of ridges per kilometer of ship track [1/km].
    ridge : IceRidge
        Ridge geometry and properties.
    level_ice_thickness_m : float
        Thickness of the surrounding level ice [m].
    """

    my_fraction: float = 0.0
    ridge_density_per_km: float = 5.0
    ridge: IceRidge = None
    level_ice_thickness_m: float = 1.2

    def __post_init__(self) -> None:
        p = float(self.my_fraction)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("my_fraction must be in [0, 1]")
        density = float(self.ridge_density_per_km)
        if not math.isfinite(density) or density < 0.0:
            raise ValueError("ridge_density_per_km must be nonnegative")
        h = float(self.level_ice_thickness_m)
        if not math.isfinite(h) or h <= 0.0:
            raise ValueError("level_ice_thickness_m must be positive")
        if self.ridge is None:
            object.__setattr__(self, "ridge", IceRidge())

    def as_dict(self) -> dict[str, Any]:
        return {
            "my_fraction": self.my_fraction,
            "ridge_density_per_km": self.ridge_density_per_km,
            "ridge": self.ridge.as_dict(),
            "level_ice_thickness_m": self.level_ice_thickness_m,
        }


def ridge_resistance(
    ridge: IceRidge,
    hull: HullForm,
    speed_m_s: float,
    *,
    water_density_kg_m3: float = 1025.0,
    ice_density_kg_m3: float = 917.0,
    gravity_m_s2: float = 9.80665,
) -> dict[str, Any]:
    """Compute resistance from a single ice ridge crossing.

    The ridge resistance has three components:
    1. Sail breaking: breaking the sail above water
    2. Keel clearing: pushing the keel rubble aside
    3. Consolidated core crushing: crushing the consolidated portion

    Parameters
    ----------
    ridge : IceRidge
        Ridge geometry and properties.
    hull : HullForm
        Hull form.
    speed_m_s : float
        Ship speed [m/s].
    water_density_kg_m3 : float
        Water density [kg/m^3].
    ice_density_kg_m3 : float
        Ice density [kg/m^3].
    gravity_m_s2 : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    dict
        Ridge resistance components and total [N].
    """
    # Sail breaking resistance
    # R_sail = 0.5 * sigma_c * h_sail^2 * (tan(alpha) + mu/cos(alpha))
    alpha_sail = math.radians(ridge.sail_angle_deg)
    mu = hull.friction_coefficient
    r_sail = (
        0.5
        * ridge.compressive_strength_pa
        * ridge.sail_height_m**2
        * (math.tan(alpha_sail) + mu / math.cos(alpha_sail))
    )

    # Keel clearing resistance
    # R_keel = rho_w * g * A_keel * d_keel
    # where A_keel is the cross-sectional area of the keel
    alpha_keel = math.radians(ridge.keel_angle_deg)
    # Keel cross-section area (triangular approximation)
    a_keel = 0.5 * ridge.keel_depth_m**2 / math.tan(alpha_keel)
    r_keel_buoyancy = water_density_kg_m3 * gravity_m_s2 * a_keel * ridge.width_m

    # Consolidated core crushing
    # R_core = sigma_c * A_core * consolidated_fraction
    # A_core is the cross-sectional area of the consolidated core
    a_core = a_keel * ridge.consolidated_fraction
    r_core = ridge.compressive_strength_pa * a_core

    # Total ridge resistance
    r_total = r_sail + r_keel_buoyancy + r_core

    return {
        "sail_breaking_N": r_sail,
        "keel_clearing_N": r_keel_buoyancy,
        "core_crushing_N": r_core,
        "total_N": r_total,
        "ridge_width_m": ridge.width_m,
        "ridge_keel_depth_m": ridge.keel_depth_m,
    }


def mixed_field_with_ridges(
    spec: RidgeFieldSpec,
    *,
    hull: HullForm = None,
    speed_m_s: float = 3.0,
    model: str = "lindqvist",
    track_length_km: float = 1.0,
) -> dict[str, Any]:
    """Compute resistance of a mixed ice field with ridges.

    The total resistance is the sum of:
    1. Level ice resistance (from mixed_resistance)
    2. Ridge resistance (averaged over the track length)

    Parameters
    ----------
    spec : RidgeFieldSpec
        Mixed field with ridges specification.
    hull : HullForm, optional
        Hull form. If None, uses ICEBREAKER.
    speed_m_s : float
        Ship speed [m/s].
    model : str
        Resistance model ("lindqvist" or "riska").
    track_length_km : float
        Track length for averaging [km].

    Returns
    -------
    dict
        Total resistance with level and ridge components.
    """
    if hull is None:
        hull = ICEBREAKER

    # Level ice resistance
    level_spec = MixedFieldSpec(my_fraction=spec.my_fraction)
    level_result = mixed_resistance(level_spec, hull=hull, speed_m_s=speed_m_s, model=model)
    r_level = level_result.resistance_mean_n

    # Ridge resistance
    ridge_result = ridge_resistance(spec.ridge, hull, speed_m_s)
    r_ridge_single = ridge_result["total_N"]

    # Number of ridges in the track
    n_ridges = spec.ridge_density_per_km * track_length_km

    # Average ridge resistance per unit length
    # R_ridge_avg = n_ridges * R_ridge_single / (track_length_km * 1000)
    r_ridge_avg = n_ridges * r_ridge_single / (track_length_km * 1000.0)

    # Total resistance
    r_total = r_level + r_ridge_avg

    return {
        "level_resistance_N": r_level,
        "ridge_resistance_avg_N": r_ridge_avg,
        "total_resistance_N": r_total,
        "ridge_contribution_fraction": r_ridge_avg / r_total if r_total > 0 else 0.0,
        "n_ridges": n_ridges,
        "track_length_km": track_length_km,
        "level_details": level_result.as_dict(),
        "ridge_details": ridge_result,
        "ridge_spec": spec.as_dict(),
    }
