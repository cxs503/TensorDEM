"""Literature correlations for Nusselt number in rotary dryers.

This module provides empirical correlations from the literature for
validating the heat transfer predictions.

Correlations implemented:
1. Tscheng & Watkinson (1979) - convective heat transfer
2. Descoin (1990) - comprehensive heat transfer model

References
----------
- Tscheng, S.H., Watkinson, A.P., 1979. Convective heat transfer in a rotary dryer.
  Can. J. Chem. Eng. 57, 433-437.
- Descoin, C., 1990. Etude du transfert thermique dans un sechoir rotatif.
  PhD thesis, ENSMA Poitiers.
- Baker, C.G.J., 1988. The design and performance of rotary dryers.
  Drying Technology 6(3), 395-434.
"""
from __future__ import annotations

import math
from typing import Optional

from .kinematics import RotaryDryerConfig

__all__ = [
    "tscheng_correlation",
    "descoin_correlation",
    "nusselt_literature",
    "reynolds_number",
    "prandtl_number",
]


def reynolds_number(
    omega: float,
    R: float,
    d_p: float,
    rho_g: float = 1.2,
    mu_g: float = 1.8e-5,
) -> float:
    """Calculate particle Reynolds number.

    Re_p = ρ_g · v_slip · d_p / μ_g

    For a rotary dryer, the characteristic velocity is the slip velocity
    between particles and gas, approximated by the peripheral velocity.

    Parameters
    ----------
    omega : float
        Angular velocity (rad/s).
    R : float
        Drum radius (m).
    d_p : float
        Particle diameter (m).
    rho_g : float
        Gas density (kg/m³). Default 1.2 for air.
    mu_g : float
        Gas dynamic viscosity (Pa·s). Default 1.8e-5 for air.

    Returns
    -------
    float
        Particle Reynolds number (-).
    """
    v_char = omega * R  # Peripheral velocity as characteristic
    if mu_g <= 0:
        raise ValueError("viscosity must be positive")
    return rho_g * v_char * d_p / mu_g


def prandtl_number(
    cp_g: float = 1005.0,
    mu_g: float = 1.8e-5,
    k_g: float = 0.026,
) -> float:
    """Calculate Prandtl number for gas.

    Pr = μ·cp / k

    Parameters
    ----------
    cp_g : float
        Gas specific heat (J/(kg·K)). Default 1005 for air.
    mu_g : float
        Gas dynamic viscosity (Pa·s). Default 1.8e-5 for air.
    k_g : float
        Gas thermal conductivity (W/(m·K)). Default 0.026 for air.

    Returns
    -------
    float
        Prandtl number (-).
    """
    if k_g <= 0:
        raise ValueError("thermal conductivity must be positive")
    return mu_g * cp_g / k_g


def tscheng_correlation(
    Re_p: float,
    Pr: float = 0.71,
    fill: float = 0.15,
) -> float:
    """Tscheng & Watkinson (1979) correlation for Nusselt number.

    Nu = 0.35 · Re_p^0.6 · Pr^(1/3) · (1 + 3.5·F)^1.2

    where F is the fill fraction.

    This correlation is valid for:
    - Re_p: 10 - 1000
    - Fill fraction: 0.05 - 0.30

    Parameters
    ----------
    Re_p : float
        Particle Reynolds number (-).
    Pr : float
        Prandtl number (-). Default 0.71 for air.
    fill : float
        Fill fraction (-). Default 0.15.

    Returns
    -------
    float
        Nusselt number (-).

    Reference
    ---------
    Tscheng, S.H., Watkinson, A.P., 1979. Convective heat transfer in a
    rotary dryer. Can. J. Chem. Eng. 57, 433-437.
    """
    if Re_p < 0:
        raise ValueError("Reynolds number must be non-negative")
    if not 0 < fill < 1:
        raise ValueError("fill fraction must be in (0, 1)")

    # Base correlation
    Nu = 0.35 * (Re_p ** 0.6) * (Pr ** (1.0 / 3.0))

    # Fill fraction correction
    fill_factor = (1.0 + 3.5 * fill) ** 1.2
    Nu *= fill_factor

    # Minimum Nusselt number (conduction limit)
    Nu = max(Nu, 2.0)

    return Nu


def descoin_correlation(
    Re_p: float,
    Pr: float = 0.71,
    fill: float = 0.15,
    omega: float = 0.5,
    R: float = 0.75,
) -> float:
    """Descoin (1990) correlation for overall Nusselt number.

    Nu = 2.0 + 0.6 · Re_p^0.5 · Pr^(1/3) + 0.036 · Fr^0.2 · Re_p^0.8

    This correlation includes:
    - Conduction limit (Nu=2 for stagnant)
    - Convective contribution
    - Froude number effect (bed dynamics)

    Parameters
    ----------
    Re_p : float
        Particle Reynolds number (-).
    Pr : float
        Prandtl number (-). Default 0.71 for air.
    fill : float
        Fill fraction (-). Default 0.15.
    omega : float
        Angular velocity (rad/s).
    R : float
        Drum radius (m).

    Returns
    -------
    float
        Nusselt number (-).

    Reference
    ---------
    Descoin, C., 1990. Etude du transfert thermique dans un sechoir rotatif.
    PhD thesis, ENSMA Poitiers.
    """
    if Re_p < 0:
        raise ValueError("Reynolds number must be non-negative")

    g = 9.81
    Fr = (omega ** 2) * R / g if R > 0 else 0.0

    # Conduction limit
    Nu_cond = 2.0

    # Convective contribution
    Nu_conv = 0.6 * (Re_p ** 0.5) * (Pr ** (1.0 / 3.0))

    # Froude effect (bed dynamics)
    Nu_fr = 0.036 * (Fr ** 0.2) * (Re_p ** 0.8) if Fr > 0 else 0.0

    Nu = Nu_cond + Nu_conv + Nu_fr

    return Nu


def nusselt_literature(
    cfg: RotaryDryerConfig,
    rho_g: float = 1.2,
    mu_g: float = 1.8e-5,
    cp_g: float = 1005.0,
) -> dict:
    """Calculate Nusselt numbers from literature correlations.

    Parameters
    ----------
    cfg : RotaryDryerConfig
        Dryer configuration.
    rho_g : float
        Gas density (kg/m³). Default 1.2 for air.
    mu_g : float
        Gas dynamic viscosity (Pa·s). Default 1.8e-5 for air.
    cp_g : float
        Gas specific heat (J/(kg·K)). Default 1005 for air.

    Returns
    -------
    dict
        Dictionary with Nusselt numbers from different correlations,
        Reynolds number, Prandtl number, and metadata.
    """
    # Calculate dimensionless numbers
    Re_p = reynolds_number(
        cfg.omega, cfg.radius, cfg.particle_diameter, rho_g, mu_g
    )
    Pr = prandtl_number(cp_g, mu_g, cfg.gas_conductivity)

    # Literature correlations
    Nu_tscheng = tscheng_correlation(Re_p, Pr, cfg.fill_fraction)
    Nu_descoin = descoin_correlation(
        Re_p, Pr, cfg.fill_fraction, cfg.omega, cfg.radius
    )

    return {
        "nusselt_tscheng": Nu_tscheng,
        "nusselt_descoin": Nu_descoin,
        "reynolds_number": Re_p,
        "prandtl_number": Pr,
        "froude_number": cfg.froude,
        "fill_fraction": cfg.fill_fraction,
        "correlations": {
            "tscheng_1979": {
                "Nu": Nu_tscheng,
                "reference": "Tscheng & Watkinson (1979), Can. J. Chem. Eng. 57, 433-437",
                "validity": "Re_p: 10-1000, fill: 0.05-0.30",
            },
            "descoin_1990": {
                "Nu": Nu_descoin,
                "reference": "Descoin (1990), PhD thesis ENSMA Poitiers",
                "validity": "General rotary dryer conditions",
            },
        },
    }
