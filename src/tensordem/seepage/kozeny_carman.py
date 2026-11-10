"""Kozeny–Carman permeability inversion.

Physical background
-------------------
The Kozeny–Carman equation relates the intrinsic permeability ``k_int`` of a
porous medium to its microstructural properties:

    k_int = phi^3 / (C * S^2 * (1 - phi)^2)

where:
- ``phi`` = porosity
- ``S`` = specific surface area per unit volume of solids (1/m)
- ``C`` = Kozeny constant (typically 180 for packed spheres, or 5 * tau^2
  where tau is tortuosity)

For monodisperse spheres of diameter ``d``:
    S = 6 / d
    k_int = phi^3 * d^2 / (180 * (1 - phi)^2)

The hydraulic conductivity is then:
    K = k_int * gamma_w / mu_w

where ``gamma_w`` is the unit weight of water and ``mu_w`` is the dynamic
viscosity.

This module provides the analytic reference for acceptance criterion C3:
DEM-inferred permeability vs K-C formula at multiple porosities, error < 20 %.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "kozeny_carman_intrinsic",
    "kozeny_carman_conductivity",
    "kozeny_carman_porosity_sweep",
    "specific_surface_sphere",
    "KozenyCarmanModel",
]

#: Unit weight of water at 20 degC (N/m^3).
WATER_UNIT_WEIGHT = 9810.0
#: Dynamic viscosity of water at 20 degC (Pa s).
WATER_VISCOSITY = 1.0e-3
#: Default Kozeny constant for packed spheres.
KOZENY_CONSTANT = 180.0


def specific_surface_sphere(diameter: float) -> float:
    """Specific surface area per unit volume of solid for spheres: ``S = 6/d``.

    Parameters
    ----------
    diameter : float
        Particle diameter (m).

    Returns
    -------
    float
        Specific surface area (1/m).
    """
    if diameter <= 0.0:
        raise ValueError("diameter must be positive")
    return 6.0 / diameter


def kozeny_carman_intrinsic(
    porosity: float,
    diameter: float,
    kozeny_c: float = KOZENY_CONSTANT,
) -> float:
    """Intrinsic permeability from Kozeny–Carman for monodisperse spheres.

    ``k_int = phi^3 * d^2 / (C * (1 - phi)^2)``

    Parameters
    ----------
    porosity : float
        Porosity phi in (0, 1).
    diameter : float
        Particle diameter (m).
    kozeny_c : float
        Kozeny constant (default 180 for packed spheres).

    Returns
    -------
    float
        Intrinsic permeability (m^2).
    """
    if not (0.0 < porosity < 1.0):
        raise ValueError(f"porosity must be in (0, 1), got {porosity}")
    if diameter <= 0.0:
        raise ValueError("diameter must be positive")
    return (porosity ** 3 * diameter ** 2) / (kozeny_c * (1.0 - porosity) ** 2)


def kozeny_carman_conductivity(
    porosity: float,
    diameter: float,
    gamma_w: float = WATER_UNIT_WEIGHT,
    mu_w: float = WATER_VISCOSITY,
    kozeny_c: float = KOZENY_CONSTANT,
) -> float:
    """Hydraulic conductivity from Kozeny–Carman.

    ``K = k_int * gamma_w / mu_w``

    Parameters
    ----------
    porosity : float
        Porosity phi.
    diameter : float
        Particle diameter (m).
    gamma_w : float
        Unit weight of water (N/m^3).
    mu_w : float
        Dynamic viscosity of water (Pa s).
    kozeny_c : float
        Kozeny constant.

    Returns
    -------
    float
        Hydraulic conductivity (m/s).
    """
    k_int = kozeny_carman_intrinsic(porosity, diameter, kozeny_c)
    return k_int * gamma_w / mu_w


def kozeny_carman_porosity_sweep(
    porosities: Sequence[float],
    diameter: float,
    gamma_w: float = WATER_UNIT_WEIGHT,
    mu_w: float = WATER_VISCOSITY,
    kozeny_c: float = KOZENY_CONSTANT,
) -> list[dict]:
    """Kozeny–Carman permeability over a range of porosities.

    Parameters
    ----------
    porosities : sequence of float
        Porosity values to evaluate.
    diameter : float
        Particle diameter (m).
    gamma_w : float
        Unit weight of water (N/m^3).
    mu_w : float
        Dynamic viscosity of water (Pa s).
    kozeny_c : float
        Kozeny constant.

    Returns
    -------
    list[dict]
        List of {porosity, k_intrinsic, k_hydraulic} dicts.
    """
    results = []
    for phi in porosities:
        k_int = kozeny_carman_intrinsic(phi, diameter, kozeny_c)
        k_hyd = k_int * gamma_w / mu_w
        results.append({
            "porosity": phi,
            "k_intrinsic_m2": k_int,
            "k_hydraulic_ms": k_hyd,
        })
    return results


@dataclass(frozen=True)
class KozenyCarmanModel:
    """Kozeny–Carman permeability model with configurable parameters.

    Supports both the sphere-specific form and the general form with
    user-specified specific surface area.
    """

    diameter: float = 2.0e-4          # particle diameter (m)
    kozeny_c: float = KOZENY_CONSTANT  # Kozeny constant
    gamma_w: float = WATER_UNIT_WEIGHT
    mu_w: float = WATER_VISCOSITY
    specific_surface: float | None = None  # override S (1/m); None = 6/d

    @property
    def S(self) -> float:
        """Specific surface area (1/m)."""
        if self.specific_surface is not None:
            return self.specific_surface
        return specific_surface_sphere(self.diameter)

    def intrinsic(self, porosity: float) -> float:
        """Intrinsic permeability (m^2)."""
        if not (0.0 < porosity < 1.0):
            raise ValueError(f"porosity must be in (0, 1), got {porosity}")
        return porosity ** 3 / (self.kozeny_c * self.S ** 2 * (1.0 - porosity) ** 2)

    def conductivity(self, porosity: float) -> float:
        """Hydraulic conductivity (m/s)."""
        return self.intrinsic(porosity) * self.gamma_w / self.mu_w

    def sweep(self, porosities: Sequence[float]) -> list[dict]:
        """Permeability over a porosity range."""
        return kozeny_carman_porosity_sweep(
            porosities, self.diameter, self.gamma_w, self.mu_w, self.kozeny_c,
        )

    def compare_with_measured(
        self,
        measured_porosities: Sequence[float],
        measured_conductivities: Sequence[float],
    ) -> dict:
        """Compare K-C predictions with measured permeability values.

        Returns a dict with per-point deviations and summary statistics.
        """
        devs = []
        for phi_meas, k_meas in zip(measured_porosities, measured_conductivities):
            k_pred = self.conductivity(phi_meas)
            if k_meas > 0.0:
                dev = abs(k_pred - k_meas) / k_meas
            else:
                dev = float("inf")
            devs.append({
                "porosity": phi_meas,
                "k_measured": k_meas,
                "k_predicted": k_pred,
                "relative_deviation": dev,
            })
        max_dev = max(d["relative_deviation"] for d in devs)
        mean_dev = np.mean([d["relative_deviation"] for d in devs])
        return {
            "points": devs,
            "max_relative_deviation": max_dev,
            "mean_relative_deviation": mean_dev,
            "n_points": len(devs),
        }
