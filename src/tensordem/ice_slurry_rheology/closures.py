"""Analytical closure models for ice-water slurry rheology.

Implements:
- Krieger-Dougherty relative viscosity law
- Eilers relative viscosity law
- Shape-factor correction for angularity
- Temperature-dependent carrier viscosity (Arrhenius-type for water near 0 °C)
- Cross-adjudication between KD and Eilers
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

__all__ = [
    "IceSlurrySpec",
    "krieger_dougherty",
    "eilers",
    "shape_factor_angularity",
    "viscosity_temperature",
    "cross_adjudicate",
]

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO_ICE = 917.0       # kg/m³, ice density
RHO_WATER = 1000.0    # kg/m³, fresh water density
MU_WATER_0C = 1.787e-3  # Pa·s, water viscosity at 0 °C
MU_WATER_20C = 1.002e-3  # Pa·s, water viscosity at 20 °C

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IceSlurrySpec:
    """Specification of an ice-water slurry state.

    Parameters
    ----------
    phi : float
        Volumetric ice concentration (solid fraction), 0 < phi < phi_max.
    phi_max : float
        Maximum packing fraction (0.58-0.64 for random loose/close packing
        of spheres; lower for angular particles).
    intrinsic_viscosity : float
        Einstein intrinsic viscosity [η]; 2.5 for spheres, higher for
        non-spherical particles.
    temperature_c : float
        Temperature in °C (affects carrier viscosity).
    shape_factor : float
        Angularity correction multiplier on [η]; 1.0 = spherical,
        >1.0 = angular.
    d50 : float
        Mean ice particle diameter (m).
    """
    phi: float = 0.10
    phi_max: float = 0.60
    intrinsic_viscosity: float = 2.5
    temperature_c: float = 0.0
    shape_factor: float = 1.0
    d50: float = 0.010  # 10 mm

    def __post_init__(self) -> None:
        if not 0.0 <= self.phi < self.phi_max:
            raise ValueError(f"phi={self.phi} must be in [0, phi_max={self.phi_max})")
        if not 0.0 < self.phi_max <= 1.0:
            raise ValueError(f"phi_max={self.phi_max} must be in (0, 1]")
        if self.intrinsic_viscosity < 2.0:
            raise ValueError(f"[η]={self.intrinsic_viscosity} must be >= 2.0")
        if self.shape_factor < 1.0:
            raise ValueError(f"shape_factor={self.shape_factor} must be >= 1.0")
        if self.d50 <= 0.0:
            raise ValueError(f"d50={self.d50} must be positive")

    @property
    def effective_intrinsic_viscosity(self) -> float:
        """[η] corrected for particle shape."""
        return self.intrinsic_viscosity * self.shape_factor

    @property
    def carrier_viscosity(self) -> float:
        """Water viscosity at the specified temperature (Pa·s)."""
        return viscosity_temperature(self.temperature_c)

    def as_dict(self) -> Dict[str, float]:
        return {
            "phi": self.phi,
            "phi_max": self.phi_max,
            "intrinsic_viscosity": self.intrinsic_viscosity,
            "effective_intrinsic_viscosity": self.effective_intrinsic_viscosity,
            "temperature_c": self.temperature_c,
            "carrier_viscosity_Pa_s": self.carrier_viscosity,
            "shape_factor": self.shape_factor,
            "d50_m": self.d50,
        }


# ---------------------------------------------------------------------------
# Krieger-Dougherty law
# ---------------------------------------------------------------------------

def krieger_dougherty(phi: float, phi_max: float = 0.60,
                      intrinsic_viscosity: float = 2.5) -> float:
    r"""Krieger-Dougherty relative viscosity.

    .. math:: \eta_r = \left(1 - \frac{\phi}{\phi_m}\right)^{-[\eta]\phi_m}

    Parameters
    ----------
    phi : float
        Volumetric solid fraction.
    phi_max : float
        Maximum packing fraction.
    intrinsic_viscosity : float
        Einstein intrinsic viscosity [η].

    Returns
    -------
    float
        Relative viscosity η_r = η_slurry / η_carrier.

    Notes
    -----
    Diverges as phi → phi_max. For phi = 0, returns exactly 1.0.
    """
    if phi < 0.0:
        raise ValueError("phi must be non-negative")
    if phi >= phi_max:
        raise ValueError(f"phi={phi} must be < phi_max={phi_max}")
    if phi == 0.0:
        return 1.0
    exponent = -intrinsic_viscosity * phi_max
    return (1.0 - phi / phi_max) ** exponent


# ---------------------------------------------------------------------------
# Eilers law
# ---------------------------------------------------------------------------

def eilers(phi: float, phi_max: float = 0.60,
           intrinsic_viscosity: float = 2.5) -> float:
    r"""Eilers relative viscosity.

    .. math:: \eta_r = \left(1 + \frac{[\eta]\phi}{2(1 - \phi/\phi_m)}\right)^2

    Parameters
    ----------
    phi : float
        Volumetric solid fraction.
    phi_max : float
        Maximum packing fraction.
    intrinsic_viscosity : float
        Einstein intrinsic viscosity [η].

    Returns
    -------
    float
        Relative viscosity η_r = η_slurry / η_carrier.

    Notes
    -----
    Also diverges as phi → phi_max. For phi = 0, returns exactly 1.0.
    Generally gives slightly lower values than KD at moderate concentrations.
    """
    if phi < 0.0:
        raise ValueError("phi must be non-negative")
    if phi >= phi_max:
        raise ValueError(f"phi={phi} must be < phi_max={phi_max}")
    if phi == 0.0:
        return 1.0
    denom = 2.0 * (1.0 - phi / phi_max)
    return (1.0 + intrinsic_viscosity * phi / denom) ** 2


# ---------------------------------------------------------------------------
# Shape factor correction
# ---------------------------------------------------------------------------

def shape_factor_angularity(shape: str = "spherical") -> Tuple[float, str]:
    """Return the intrinsic-viscosity multiplier for a given particle shape.

    Based on literature values for non-spherical particle suspensions
    (e.g., Mueller et al. 2010, J. Volcanol. Geotherm. Res.;
    Maron & Pierce 1956, J. Colloid Sci.).

    Parameters
    ----------
    shape : str
        One of: "spherical", "sub-angular", "angular", "very_angular".

    Returns
    -------
    Tuple[float, str]
        (shape_factor, description).
    """
    table = {
        "spherical": (1.0, "perfect spheres, [η]=2.5"),
        "sub-angular": (1.15, "slightly irregular, [η]≈2.875"),
        "angular": (1.35, "angular fragments, [η]≈3.375"),
        "very_angular": (1.60, "highly angular, [η]≈4.0"),
    }
    if shape not in table:
        raise ValueError(f"Unknown shape '{shape}'; choose from {list(table.keys())}")
    return table[shape]


# ---------------------------------------------------------------------------
# Temperature-dependent carrier viscosity
# ---------------------------------------------------------------------------

def viscosity_temperature(temperature_c: float) -> float:
    """Water dynamic viscosity as a function of temperature (Pa·s).

    Uses a quadratic fit to log(μ) vs T based on IAPWS data for the
    0-40 °C range:

    - μ(0 °C) = 1.787e-3 Pa·s
    - μ(20 °C) = 1.002e-3 Pa·s
    - μ(40 °C) = 0.653e-3 Pa·s
    """
    if temperature_c < -5.0 or temperature_c > 50.0:
        raise ValueError("Temperature must be in [-5, 50] °C for this correlation")
    # Quadratic fit to log(mu) vs Tc: log(mu) = a*Tc^2 + b*Tc + c
    a = 1.87597439e-04
    b = -3.25720364e-02
    c = -6.32918385e+00
    return math.exp(a * temperature_c ** 2 + b * temperature_c + c)


# ---------------------------------------------------------------------------
# Cross-adjudication
# ---------------------------------------------------------------------------

def cross_adjudicate(phi_values: Sequence[float], phi_max: float = 0.60,
                     intrinsic_viscosity: float = 2.5) -> Dict[str, object]:
    """Cross-adjudicate Krieger-Dougherty vs Eilers over a concentration range.

    Parameters
    ----------
    phi_values : Sequence[float]
        Volumetric fractions to evaluate.
    phi_max : float
        Maximum packing fraction.
    intrinsic_viscosity : float
        Einstein intrinsic viscosity [η].

    Returns
    -------
    dict
        Contains per-point values, max relative deviation, and pass verdicts.
    """
    rows = []
    for phi in phi_values:
        eta_kd = krieger_dougherty(phi, phi_max, intrinsic_viscosity)
        eta_ei = eilers(phi, phi_max, intrinsic_viscosity)
        # Relative deviation: |KD - Eilers| / mean(KD, Eilers)
        mean_val = 0.5 * (eta_kd + eta_ei)
        rel_dev = abs(eta_kd - eta_ei) / mean_val if mean_val > 0 else 0.0
        rows.append({
            "phi": phi,
            "eta_kd": eta_kd,
            "eta_eilers": eta_ei,
            "rel_deviation": rel_dev,
        })

    max_dev = max(r["rel_deviation"] for r in rows) if rows else 0.0
    return {
        "rows": rows,
        "max_relative_deviation": max_dev,
        "kd_threshold": 0.30,
        "eilers_threshold": 0.25,
        "pass_kd": bool(max_dev <= 0.30),
        "pass_eilers": bool(max_dev <= 0.25),
        "n_points": len(rows),
    }
