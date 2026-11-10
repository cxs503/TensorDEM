"""Terzaghi effective stress principle verification.

Physical background
-------------------
Terzaghi's effective stress principle states that the mechanical behaviour of
a saturated granular assembly is governed by the **effective stress**

    sigma' = sigma - u

where ``sigma`` is the total stress and ``u`` is the pore water pressure.
For a saturated soil column of height ``H`` under self-weight, the total
vertical stress at depth ``z`` (measured from the top) is

    sigma_v(z) = gamma_sat * z

and the hydrostatic pore pressure (with the water table at the top surface) is

    u(z) = gamma_w * z

so the effective stress is

    sigma'(z) = (gamma_sat - gamma_w) * z = gamma' * z

where ``gamma'`` is the buoyant (submerged) unit weight.

This module provides the analytic reference solutions against which DEM
simulations of saturated columns are judged (acceptance criterion C1:
pore-pressure profile error < 10 %).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "hydrostatic_pore_pressure",
    "self_weight_pore_pressure",
    "effective_stress_profile",
    "terzaghi_effective_stress",
    "SaturatedColumn",
]

#: Unit weight of water at 20 degC (N/m^3).
WATER_UNIT_WEIGHT = 9810.0
#: Density of water (kg/m^3).
WATER_DENSITY = 1000.0
#: Gravity (m/s^2).
GRAVITY = 9.81


def hydrostatic_pore_pressure(
    z: np.ndarray | float,
    gamma_w: float = WATER_UNIT_WEIGHT,
    water_table_depth: float = 0.0,
) -> np.ndarray:
    """Hydrostatic pore pressure ``u = gamma_w * (z - z_wt)`` for ``z >= z_wt``.

    Parameters
    ----------
    z : array or scalar
        Depth below the ground surface (m), positive downward.
    gamma_w : float
        Unit weight of water (N/m^3).
    water_table_depth : float
        Depth of the water table below the surface (m).

    Returns
    -------
    np.ndarray
        Pore water pressure (Pa); zero above the water table.
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    u = gamma_w * np.maximum(z_arr - water_table_depth, 0.0)
    return u if np.ndim(z) != 0 else float(u[0])


def self_weight_pore_pressure(
    z: np.ndarray | float,
    height: float,
    gamma_sat: float,
    gamma_w: float = WATER_UNIT_WEIGHT,
    water_table_depth: float = 0.0,
) -> np.ndarray:
    """Pore pressure in a saturated column under self-weight with top drainage.

    For a column of height ``H`` fully saturated with the water table at the
    top, the steady-state pore pressure at depth ``z`` is simply hydrostatic:
    ``u(z) = gamma_w * z``.  This function returns that profile, clipped to
    the column height.

    Parameters
    ----------
    z : array or scalar
        Depth below the top surface (m).
    height : float
        Column height (m).
    gamma_sat : float
        Saturated unit weight of the soil (N/m^3).
    gamma_w : float
        Unit weight of water (N/m^3).
    water_table_depth : float
        Depth of the water table (m).

    Returns
    -------
    np.ndarray
        Pore water pressure (Pa).
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    z_clipped = np.clip(z_arr, 0.0, height)
    return hydrostatic_pore_pressure(z_clipped, gamma_w, water_table_depth)


def effective_stress_profile(
    z: np.ndarray | float,
    height: float,
    gamma_sat: float,
    gamma_w: float = WATER_UNIT_WEIGHT,
    water_table_depth: float = 0.0,
    surcharge: float = 0.0,
) -> np.ndarray:
    """Terzaghi effective stress ``sigma' = sigma - u`` under self-weight.

    Parameters
    ----------
    z : array or scalar
        Depth below the top surface (m).
    height : float
        Column height (m).
    gamma_sat : float
        Saturated unit weight (N/m^3).
    gamma_w : float
        Unit weight of water (N/m^3).
    water_table_depth : float
        Water table depth (m).
    surcharge : float
        Additional surface surcharge (Pa).

    Returns
    -------
    np.ndarray
        Effective vertical stress (Pa).
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    z_clipped = np.clip(z_arr, 0.0, height)
    sigma_total = gamma_sat * z_clipped + surcharge
    u = hydrostatic_pore_pressure(z_clipped, gamma_w, water_table_depth)
    sigma_prime = sigma_total - u
    return sigma_prime if np.ndim(z) != 0 else float(sigma_prime[0])


def terzaghi_effective_stress(
    z: np.ndarray | float,
    gamma_prime: float,
    surcharge: float = 0.0,
) -> np.ndarray:
    """Effective stress using the buoyant unit weight directly.

    ``sigma'(z) = gamma' * z + surcharge``

    Parameters
    ----------
    z : array or scalar
        Depth (m).
    gamma_prime : float
        Buoyant (submerged) unit weight (N/m^3).
    surcharge : float
        Surface surcharge (Pa).

    Returns
    -------
    np.ndarray
        Effective stress (Pa).
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    sigma_prime = gamma_prime * np.maximum(z_arr, 0.0) + surcharge
    return sigma_prime if np.ndim(z) != 0 else float(sigma_prime[0])


@dataclass(frozen=True)
class SaturatedColumn:
    """Analytic reference for a saturated soil column under self-weight.

    All derived properties are computed from the primitive inputs; no
    empirical closures are used.
    """

    height: float                    # column height (m)
    porosity: float                  # void ratio n
    rho_s: float = 2650.0            # solid density (kg/m^3)
    rho_w: float = WATER_DENSITY     # water density (kg/m^3)
    g: float = GRAVITY               # gravity (m/s^2)

    @property
    def gamma_s(self) -> float:
        """Solid unit weight (N/m^3)."""
        return self.rho_s * self.g

    @property
    def gamma_w(self) -> float:
        """Water unit weight (N/m^3)."""
        return self.rho_w * self.g

    @property
    def gamma_sat(self) -> float:
        """Saturated unit weight (N/m^3)."""
        return (1.0 - self.porosity) * self.gamma_s + self.porosity * self.gamma_w

    @property
    def gamma_prime(self) -> float:
        """Buoyant (submerged) unit weight (N/m^3)."""
        return self.gamma_sat - self.gamma_w

    @property
    def void_ratio(self) -> float:
        """Void ratio e = n / (1 - n)."""
        return self.porosity / (1.0 - self.porosity)

    @property
    def specific_gravity(self) -> float:
        """G_s = rho_s / rho_w."""
        return self.rho_s / self.rho_w

    def pore_pressure(self, z: np.ndarray | float) -> np.ndarray:
        """Hydrostatic pore pressure profile."""
        return hydrostatic_pore_pressure(z, self.gamma_w)

    def total_stress(self, z: np.ndarray | float) -> np.ndarray:
        """Total vertical stress under self-weight."""
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        return self.gamma_sat * np.clip(z_arr, 0.0, self.height)

    def effective_stress(self, z: np.ndarray | float) -> np.ndarray:
        """Terzaghi effective stress profile."""
        return effective_stress_profile(
            z, self.height, self.gamma_sat, self.gamma_w,
        )

    def profile_at(self, n_points: int = 101) -> dict:
        """Full stress/pore-pressure profile as a dictionary."""
        z = np.linspace(0.0, self.height, n_points)
        return {
            "z": z.tolist(),
            "u": self.pore_pressure(z).tolist(),
            "sigma_total": self.total_stress(z).tolist(),
            "sigma_prime": self.effective_stress(z).tolist(),
            "gamma_sat": self.gamma_sat,
            "gamma_prime": self.gamma_prime,
            "gamma_w": self.gamma_w,
        }
