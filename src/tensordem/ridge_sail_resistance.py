"""Sail resistance component of ice ridge — Kujala 1994 / Riska 1997 model.

Physical background
-------------------
An ice ridge sail is the above-water portion of a pressure ridge, consisting of
broken ice blocks with porosity *n*.  When a ship hull pushes against the sail,
two resistance mechanisms act in parallel:

1. **Breaking resistance** (F_sb): the hull fractures the sail ice blocks.
   Governed by the flexural strength σ_f of the ice and the sail geometry.
   Based on Kujala (1994) full-scale measurements on the Baltic icebreaker
   *Sisu* and *Urho*.

2. **Pushing resistance** (F_sp): the hull pushes the already-broken rubble
   pile.  Governed by passive earth pressure theory with the effective unit
   weight of the porous ice rubble.  Based on Riska (1997) model-scale tests
   and full-scale validation.

Total sail resistance:  F_sail = F_sb + F_sp

Key references
--------------
- Kujala, K. (1994). "Ice ridge loads on ship hulls — full-scale measurements
  on icebreakers Sisu and Urho." *Proceedings of IAHR Ice Symposium*,
  Trondheim, Norway.
- Riska, K. (1997). "A model for ice ridge keel resistance." *Proceedings of
  POAC 97*, Vol. 2, pp. 454–463.
- Timco, G.W. & Burden, R.P. (1994). "Mechanical properties of sea ice ridges."
  *Cold Regions Science and Technology*, 22, 285–298.

Formulas
--------
Breaking resistance (Kujala 1994):

    F_sb = C_b · σ_f · h_s² · B / sin(α)

where:
    C_b  = empirical breaking coefficient (≈ 0.6–0.8, default 0.7)
    σ_f  = ice flexural strength [Pa]
    h_s  = sail height [m]
    B    = ship beam / contact width [m]
    α    = hull waterline angle [deg]

Pushing resistance (Riska 1997):

    F_sp = C_p · γ' · h_s² · B · K_p / 2

where:
    C_p  = empirical pushing coefficient (≈ 0.8–1.2, default 1.0)
    γ'   = effective unit weight of porous ice rubble [N/m³]
         = (1 - n) · (ρ_ice - ρ_water) · g   (submerged, buoyant)
    K_p  = passive earth pressure coefficient
         = tan²(45° + φ/2)
    φ    = internal friction angle of ice rubble [deg] (≈ 35–45°)

Total sail resistance:

    F_sail = F_sb + F_sp

Notes
-----
- The sail is *above* the waterline, but the rubble is porous and partially
  saturated.  The effective unit weight accounts for buoyancy of the ice blocks
  in the water-filled pores.
- For ridges with very high porosity (n > 0.5), the pushing resistance becomes
  small because the effective unit weight approaches zero.
- The breaking coefficient C_b was calibrated by Kujala against full-scale
  measurements on two Finnish icebreakers; the recommended range is 0.6–0.8.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Optional

import numpy as np

__all__ = [
    "SailResistanceConfig",
    "SailResistanceResult",
    "compute_sail_resistance",
    "kujala_breaking_resistance",
    "riska_pushing_resistance",
    "passive_earth_pressure_coefficient",
    "effective_unit_weight",
]

# Default physical constants
ICE_DENSITY = 917.0        # kg/m³
WATER_DENSITY = 1025.0     # kg/m³
GRAVITY = 9.81             # m/s²


@dataclass(frozen=True)
class SailResistanceConfig:
    """Configuration for sail resistance calculation.

    Attributes
    ----------
    h_s : float
        Sail height above waterline [m].
    B : float
        Ship beam / contact width [m].
    alpha : float
        Hull waterline angle [deg].
    sigma_f : float
        Ice flexural strength [Pa].  Typical range: 0.3–1.0 MPa for first-year
        ridge ice; 0.1–0.5 MPa for multi-year ridge ice.
    n : float
        Sail porosity (void fraction), 0 < n < 1.  Typical range: 0.2–0.5.
    phi : float
        Internal friction angle of ice rubble [deg].  Typical: 35–45°.
    C_b : float
        Breaking resistance coefficient.  Kujala (1994) recommends 0.6–0.8.
    C_p : float
        Pushing resistance coefficient.  Riska (1997) recommends 0.8–1.2.
    rho_ice : float
        Ice density [kg/m³].
    rho_water : float
        Water density [kg/m³].
    g : float
        Gravitational acceleration [m/s²].
    """

    h_s: float
    B: float
    alpha: float = 30.0
    sigma_f: float = 500e3       # 0.5 MPa, typical first-year ridge ice
    n: float = 0.35              # typical sail porosity
    phi: float = 40.0            # internal friction angle
    C_b: float = 0.7             # Kujala breaking coefficient
    C_p: float = 1.0             # Riska pushing coefficient
    rho_ice: float = ICE_DENSITY
    rho_water: float = WATER_DENSITY
    g: float = GRAVITY

    def __post_init__(self) -> None:
        for name in ("h_s", "B", "sigma_f", "C_b", "C_p", "rho_ice", "rho_water", "g"):
            val = getattr(self, name)
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not (0 < self.n < 1):
            raise ValueError("porosity n must be in (0, 1)")
        if not (0 < self.alpha < 90):
            raise ValueError("alpha must be in (0, 90) degrees")
        if not (20 < self.phi < 60):
            raise ValueError("phi must be in (20, 60) degrees")


@dataclass
class SailResistanceResult:
    """Result of sail resistance calculation.

    Attributes
    ----------
    F_sb : float
        Breaking resistance [N].
    F_sp : float
        Pushing resistance [N].
    F_sail : float
        Total sail resistance [N].
    gamma_eff : float
        Effective unit weight of porous ice rubble [N/m³].
    K_p : float
        Passive earth pressure coefficient.
    config : SailResistanceConfig
        Input configuration.
    """

    F_sb: float
    F_sp: float
    F_sail: float
    gamma_eff: float
    K_p: float
    config: SailResistanceConfig

    def to_dict(self) -> Dict[str, float]:
        return {
            "F_sb_N": self.F_sb,
            "F_sp_N": self.F_sp,
            "F_sail_N": self.F_sail,
            "gamma_eff_N_m3": self.gamma_eff,
            "K_p": self.K_p,
            "h_s_m": self.config.h_s,
            "B_m": self.config.B,
            "alpha_deg": self.config.alpha,
            "sigma_f_Pa": self.config.sigma_f,
            "n": self.config.n,
            "phi_deg": self.config.phi,
            "C_b": self.config.C_b,
            "C_p": self.config.C_p,
        }


def passive_earth_pressure_coefficient(phi_deg: float) -> float:
    """Passive earth pressure coefficient K_p = tan²(45° + φ/2).

    Parameters
    ----------
    phi_deg : float
        Internal friction angle [deg].

    Returns
    -------
    K_p : float
        Passive earth pressure coefficient.
    """
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4 + phi_rad / 2) ** 2


def effective_unit_weight(n: float, rho_ice: float = ICE_DENSITY,
                          rho_water: float = WATER_DENSITY,
                          g: float = GRAVITY) -> float:
    """Effective (submerged) unit weight of porous ice rubble.

    γ' = (1 - n) · (ρ_ice - ρ_water) · g

    Note: for ice in seawater, (ρ_ice - ρ_water) < 0, so γ' is negative
    (buoyant).  The magnitude is used in the passive earth pressure formula.

    Parameters
    ----------
    n : float
        Porosity (void fraction).
    rho_ice : float
        Ice density [kg/m³].
    rho_water : float
        Water density [kg/m³].
    g : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    gamma_eff : float
        Effective unit weight [N/m³].  Negative for buoyant ice.
    """
    return (1 - n) * (rho_ice - rho_water) * g


def kujala_breaking_resistance(h_s: float, B: float, alpha_deg: float,
                                sigma_f: float, C_b: float = 0.7) -> float:
    """Sail breaking resistance after Kujala (1994).

    F_sb = C_b · σ_f · h_s² · B / sin(α)

    Parameters
    ----------
    h_s : float
        Sail height [m].
    B : float
        Ship beam / contact width [m].
    alpha_deg : float
        Hull waterline angle [deg].
    sigma_f : float
        Ice flexural strength [Pa].
    C_b : float
        Breaking coefficient (default 0.7, range 0.6–0.8).

    Returns
    -------
    F_sb : float
        Breaking resistance [N].
    """
    alpha_rad = math.radians(alpha_deg)
    return C_b * sigma_f * h_s ** 2 * B / math.sin(alpha_rad)


def riska_pushing_resistance(h_s: float, B: float, n: float, phi_deg: float,
                              C_p: float = 1.0,
                              rho_ice: float = ICE_DENSITY,
                              rho_water: float = WATER_DENSITY,
                              g: float = GRAVITY) -> float:
    """Sail pushing resistance after Riska (1997).

    F_sp = C_p · |γ'| · h_s² · B · K_p / 2

    where γ' = (1-n)·(ρ_ice - ρ_water)·g is the effective unit weight
    (negative for buoyant ice; we use the magnitude).

    Parameters
    ----------
    h_s : float
        Sail height [m].
    B : float
        Ship beam / contact width [m].
    n : float
        Sail porosity.
    phi_deg : float
        Internal friction angle [deg].
    C_p : float
        Pushing coefficient (default 1.0, range 0.8–1.2).
    rho_ice : float
        Ice density [kg/m³].
    rho_water : float
        Water density [kg/m³].
    g : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    F_sp : float
        Pushing resistance [N].
    """
    gamma_eff = effective_unit_weight(n, rho_ice, rho_water, g)
    K_p = passive_earth_pressure_coefficient(phi_deg)
    # Use magnitude of effective unit weight (buoyancy reduces the load)
    return C_p * abs(gamma_eff) * h_s ** 2 * B * K_p / 2


def compute_sail_resistance(config: SailResistanceConfig) -> SailResistanceResult:
    """Compute total sail resistance (breaking + pushing).

    Parameters
    ----------
    config : SailResistanceConfig
        Input configuration.

    Returns
    -------
    SailResistanceResult
    """
    F_sb = kujala_breaking_resistance(
        config.h_s, config.B, config.alpha, config.sigma_f, config.C_b
    )
    F_sp = riska_pushing_resistance(
        config.h_s, config.B, config.n, config.phi, config.C_p,
        config.rho_ice, config.rho_water, config.g
    )
    gamma_eff = effective_unit_weight(
        config.n, config.rho_ice, config.rho_water, config.g
    )
    K_p = passive_earth_pressure_coefficient(config.phi)

    return SailResistanceResult(
        F_sb=F_sb,
        F_sp=F_sp,
        F_sail=F_sb + F_sp,
        gamma_eff=gamma_eff,
        K_p=K_p,
        config=config,
    )
