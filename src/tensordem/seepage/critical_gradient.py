"""Critical hydraulic gradient for piping initiation (Terzaghi).

Physical background
-------------------
When upward seepage flow through a granular soil becomes strong enough, the
seepage force counteracts the submerged weight of the soil skeleton.  At the
**critical hydraulic gradient** ``i_c``, the effective stress drops to zero
and the soil loses all shear strength -- this is the onset of **piping**
(boiling / quick condition).

Terzaghi's formula for the critical gradient is derived from force equilibrium:

    i_c = gamma' / gamma_w = (G_s - 1) / (1 + e)

where:
- ``gamma'`` = buoyant unit weight = ``gamma_sat - gamma_w``
- ``gamma_w`` = unit weight of water
- ``G_s`` = specific gravity of solids = ``rho_s / rho_w``
- ``e`` = void ratio = ``n / (1 - n)``

For typical quartz sand (G_s ≈ 2.65, e ≈ 0.65), i_c ≈ 1.0.

This module provides the analytic reference for acceptance criterion C2:
DEM-simulated critical gradient vs Terzaghi formula, error < 15 %.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "terzaghi_critical_gradient",
    "buoyant_unit_weight",
    "piping_factor_of_safety",
    "seepage_force_per_volume",
    "PipingAnalysis",
]

#: Unit weight of water at 20 degC (N/m^3).
WATER_UNIT_WEIGHT = 9810.0
#: Density of water (kg/m^3).
WATER_DENSITY = 1000.0
#: Gravity (m/s^2).
GRAVITY = 9.81


def buoyant_unit_weight(
    rho_s: float,
    porosity: float,
    rho_w: float = WATER_DENSITY,
    g: float = GRAVITY,
) -> float:
    """Buoyant (submerged) unit weight ``gamma' = gamma_sat - gamma_w``.

    Parameters
    ----------
    rho_s : float
        Solid particle density (kg/m^3).
    porosity : float
        Porosity n (void volume / total volume), in (0, 1).
    rho_w : float
        Water density (kg/m^3).
    g : float
        Gravitational acceleration (m/s^2).

    Returns
    -------
    float
        Buoyant unit weight (N/m^3).
    """
    if not (0.0 < porosity < 1.0):
        raise ValueError(f"porosity must be in (0, 1), got {porosity}")
    gamma_sat = (1.0 - porosity) * rho_s * g + porosity * rho_w * g
    return gamma_sat - rho_w * g


def terzaghi_critical_gradient(
    rho_s: float,
    porosity: float,
    rho_w: float = WATER_DENSITY,
    g: float = GRAVITY,
) -> float:
    """Terzaghi critical hydraulic gradient ``i_c = gamma' / gamma_w``.

    Equivalently: ``i_c = (G_s - 1) / (1 + e)`` where ``G_s = rho_s/rho_w``
    and ``e = n/(1-n)``.

    Parameters
    ----------
    rho_s : float
        Solid particle density (kg/m^3).
    porosity : float
        Porosity n.
    rho_w : float
        Water density (kg/m^3).
    g : float
        Gravitational acceleration (m/s^2).

    Returns
    -------
    float
        Critical hydraulic gradient (dimensionless).
    """
    gamma_prime = buoyant_unit_weight(rho_s, porosity, rho_w, g)
    gamma_w = rho_w * g
    return gamma_prime / gamma_w


def seepage_force_per_volume(
    hydraulic_gradient: float,
    gamma_w: float = WATER_UNIT_WEIGHT,
) -> float:
    """Seepage force per unit volume ``j = i * gamma_w`` (N/m^3).

    The seepage force acts in the direction of flow.  For upward flow it
    opposes the submerged weight of the soil skeleton.

    Parameters
    ----------
    hydraulic_gradient : float
        Hydraulic gradient i = dh/dL (dimensionless).
    gamma_w : float
        Unit weight of water (N/m^3).

    Returns
    -------
    float
        Seepage force per unit volume (N/m^3).
    """
    return hydraulic_gradient * gamma_w


def piping_factor_of_safety(
    hydraulic_gradient: float,
    rho_s: float,
    porosity: float,
    rho_w: float = WATER_DENSITY,
    g: float = GRAVITY,
) -> float:
    """Factor of safety against piping ``FS = i_c / i``.

    FS > 1: stable; FS = 1: critical; FS < 1: piping occurs.

    Parameters
    ----------
    hydraulic_gradient : float
        Applied hydraulic gradient.
    rho_s : float
        Solid particle density (kg/m^3).
    porosity : float
        Porosity n.
    rho_w : float
        Water density (kg/m^3).
    g : float
        Gravitational acceleration (m/s^2).

    Returns
    -------
    float
        Factor of safety (dimensionless).
    """
    i_c = terzaghi_critical_gradient(rho_s, porosity, rho_w, g)
    if hydraulic_gradient <= 0.0:
        return float("inf")
    return i_c / hydraulic_gradient


@dataclass(frozen=True)
class PipingAnalysis:
    """Complete piping analysis for a given soil and hydraulic condition.

    Computes the critical gradient, the applied gradient, the factor of safety,
    and the seepage force.
    """

    rho_s: float = 2650.0
    porosity: float = 0.40
    rho_w: float = WATER_DENSITY
    g: float = GRAVITY
    applied_gradient: float = 0.0

    @property
    def gamma_w(self) -> float:
        return self.rho_w * self.g

    @property
    def gamma_prime(self) -> float:
        return buoyant_unit_weight(self.rho_s, self.porosity, self.rho_w, self.g)

    @property
    def specific_gravity(self) -> float:
        return self.rho_s / self.rho_w

    @property
    def void_ratio(self) -> float:
        return self.porosity / (1.0 - self.porosity)

    @property
    def i_c(self) -> float:
        """Critical hydraulic gradient (Terzaghi)."""
        return terzaghi_critical_gradient(
            self.rho_s, self.porosity, self.rho_w, self.g,
        )

    @property
    def i_c_alt(self) -> float:
        """Critical gradient via (G_s - 1)/(1 + e) -- should equal i_c."""
        G_s = self.specific_gravity
        e = self.void_ratio
        return (G_s - 1.0) / (1.0 + e)

    @property
    def seepage_force(self) -> float:
        """Seepage force per unit volume at the applied gradient (N/m^3)."""
        return seepage_force_per_volume(self.applied_gradient, self.gamma_w)

    @property
    def factor_of_safety(self) -> float:
        """FS = i_c / i_applied."""
        return piping_factor_of_safety(
            self.applied_gradient, self.rho_s, self.porosity, self.rho_w, self.g,
        )

    @property
    def is_piping(self) -> bool:
        """True if the applied gradient exceeds the critical value."""
        return self.applied_gradient > self.i_c

    def summary(self) -> dict:
        """Full analysis summary."""
        return {
            "rho_s": self.rho_s,
            "porosity": self.porosity,
            "void_ratio": self.void_ratio,
            "specific_gravity": self.specific_gravity,
            "gamma_prime": self.gamma_prime,
            "gamma_w": self.gamma_w,
            "i_c_terzaghi": self.i_c,
            "i_c_alt_formula": self.i_c_alt,
            "applied_gradient": self.applied_gradient,
            "seepage_force_N_per_m3": self.seepage_force,
            "factor_of_safety": self.factor_of_safety,
            "piping_onset": self.is_piping,
        }
