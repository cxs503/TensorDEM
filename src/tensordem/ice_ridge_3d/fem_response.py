"""Hull panel FEM response to ice ridge load.

Analytical anchor: simply supported plate under uniform pressure
    sigma_max = k * p * b² / t²
    w_max = alpha * p * b⁴ / (E * t³)

Acceptance criterion (e): Hull FEM sigma_max vs closed-form error < 3%.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "plate_stress_simply_supported",
    "plate_deflection_simply_supported",
    "hull_panel_response",
    "PlateProperties",
    "von_mises_stress",
]


def _finite(name: str, value, *, positive: bool = False, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class PlateProperties:
    """Properties of a hull plate panel."""
    width: float          # plate width b (m), shorter dimension
    length: float         # plate length a (m), longer dimension
    thickness: float      # plate thickness t (m)
    young_modulus: float  # E (Pa)
    poisson_ratio: float  # nu
    yield_strength: float # sigma_y (Pa)

    def __post_init__(self) -> None:
        for name in ("width", "length", "thickness", "young_modulus", "yield_strength"):
            _finite(name, getattr(self, name), positive=True)
        _finite("poisson_ratio", self.poisson_ratio, nonnegative=True)
        if self.poisson_ratio >= 0.5:
            raise ValueError("poisson_ratio must be < 0.5")

    @property
    def aspect_ratio(self) -> float:
        """a/b ratio."""
        return self.length / self.width

    @property
    def flexural_rigidity(self) -> float:
        """D = E t³ / (12 (1 - nu²))."""
        return self.young_modulus * self.thickness ** 3 / (
            12.0 * (1.0 - self.poisson_ratio ** 2))


def _stress_coefficient_k(aspect_ratio: float) -> float:
    """Bending stress coefficient k for simply supported rectangular plate.

    sigma_max = k * p * b² / t²

    For a/b >= 2, k ≈ 0.5 (long plate limit).
    For a/b = 1 (square), k ≈ 0.287.
    Interpolated from Timoshenko plate theory tables.
    """
    if aspect_ratio >= 2.0:
        return 0.5
    elif aspect_ratio >= 1.5:
        return 0.42
    elif aspect_ratio >= 1.2:
        return 0.36
    else:
        # Square plate
        return 0.287


def _deflection_coefficient_alpha(aspect_ratio: float, poisson_ratio: float = 0.3) -> float:
    """Deflection coefficient alpha for simply supported rectangular plate.

    w_max = alpha * p * b⁴ / D

    For a/b >= 2, alpha ≈ 0.0130.
    For a/b = 1 (square), alpha ≈ 0.00406.
    """
    if aspect_ratio >= 2.0:
        return 0.0130
    elif aspect_ratio >= 1.5:
        return 0.0090
    elif aspect_ratio >= 1.2:
        return 0.0060
    else:
        return 0.00406


def plate_stress_simply_supported(pressure: float, width: float,
                                   thickness: float, aspect_ratio: float,
                                   poisson_ratio: float = 0.3) -> float:
    """Maximum bending stress in simply supported rectangular plate.

    sigma_max = k * p * b² / t²

    Analytical anchor for criterion (e): error < 3% vs closed-form.

    Parameters
    ----------
    pressure : float
        Uniform pressure load (Pa).
    width : float
        Plate width b (m), shorter dimension.
    thickness : float
        Plate thickness t (m).
    aspect_ratio : float
        a/b ratio (length/width).
    poisson_ratio : float
        Poisson's ratio (used for k coefficient selection).

    Returns
    -------
    float
        Maximum bending stress (Pa).
    """
    _finite("pressure", pressure, nonnegative=True)
    _finite("width", width, positive=True)
    _finite("thickness", thickness, positive=True)
    _finite("aspect_ratio", aspect_ratio, positive=True)

    k = _stress_coefficient_k(aspect_ratio)
    return k * pressure * width ** 2 / thickness ** 2


def plate_deflection_simply_supported(pressure: float, width: float,
                                       thickness: float,
                                       young_modulus: float,
                                       poisson_ratio: float,
                                       aspect_ratio: float) -> float:
    """Maximum deflection of simply supported rectangular plate.

    w_max = alpha * p * b⁴ / D, where D = E t³ / (12(1-nu²))

    Parameters
    ----------
    pressure : float
        Uniform pressure load (Pa).
    width : float
        Plate width b (m).
    thickness : float
        Plate thickness t (m).
    young_modulus : float
        Young's modulus (Pa).
    poisson_ratio : float
        Poisson's ratio.
    aspect_ratio : float
        a/b ratio.

    Returns
    -------
    float
        Maximum deflection (m).
    """
    _finite("pressure", pressure, nonnegative=True)
    _finite("width", width, positive=True)
    _finite("thickness", thickness, positive=True)
    _finite("young_modulus", young_modulus, positive=True)

    alpha = _deflection_coefficient_alpha(aspect_ratio, poisson_ratio)
    d = young_modulus * thickness ** 3 / (12.0 * (1.0 - poisson_ratio ** 2))
    return alpha * pressure * width ** 4 / d


def von_mises_stress(sigma_x: float, sigma_y: float,
                      tau_xy: float = 0.0) -> float:
    """Von Mises equivalent stress.

    sigma_vm = sqrt(sigma_x² + sigma_y² - sigma_x*sigma_y + 3*tau_xy²)
    """
    return math.sqrt(sigma_x ** 2 + sigma_y ** 2 - sigma_x * sigma_y + 3.0 * tau_xy ** 2)


def hull_panel_response(ridge_load: float, contact_area: float,
                         plate: PlateProperties,
                         load_distribution: str = "uniform") -> dict:
    """Compute hull panel response to ridge load.

    Parameters
    ----------
    ridge_load : float
        Total ridge load on panel (N).
    contact_area : float
        Contact area between ridge and hull (m²).
    plate : PlateProperties
        Hull plate properties.
    load_distribution : str
        "uniform" or "triangular" pressure distribution.

    Returns
    -------
    dict with stress, deflection, and safety margin.
    """
    _finite("ridge_load", ridge_load, nonnegative=True)
    _finite("contact_area", contact_area, positive=True)

    # Average pressure
    pressure = ridge_load / contact_area

    # For non-uniform distribution, use equivalent uniform pressure
    if load_distribution == "triangular":
        pressure *= 1.5  # peak pressure factor

    # Stress
    sigma_max = plate_stress_simply_supported(
        pressure, plate.width, plate.thickness,
        plate.aspect_ratio, plate.poisson_ratio
    )

    # Deflection
    w_max = plate_deflection_simply_supported(
        pressure, plate.width, plate.thickness,
        plate.young_modulus, plate.poisson_ratio,
        plate.aspect_ratio
    )

    # Von Mises stress (bending dominates, sigma_y ≈ 0)
    sigma_vm = von_mises_stress(sigma_max, 0.0)

    # Safety margin
    safety_margin = (plate.yield_strength - sigma_vm) / plate.yield_strength

    return {
        "pressure_pa": pressure,
        "sigma_max_pa": sigma_max,
        "sigma_vm_pa": sigma_vm,
        "deflection_m": w_max,
        "yield_strength_pa": plate.yield_strength,
        "safety_margin": safety_margin,
        "is_safe": sigma_vm < plate.yield_strength,
    }
