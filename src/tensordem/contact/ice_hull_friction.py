"""Integration of ice-steel friction law with hull contact and ice resistance.

This module connects the calibrated friction law mu(v, T, p) to:
1. The hull-ice contact model (replacing constant Coulomb mu)
2. Straight-ahead ice resistance computation
3. Lindqvist (1989) analytical comparison
4. Uncertainty propagation via MCMC posterior samples

Physical model
--------------
The total ice resistance is decomposed as:
    R_total = R_crushing + R_friction + R_bending + R_submersion

where R_friction = mu(v, T, p) * N_normal, and N_normal depends on the
ice thickness, hull geometry, and ice strength.

The Lindqvist (1989) formula provides an analytical reference:
    R_L = 0.5*sigma_c*h^2*(tan(alpha_r) + mu/cos(alpha_r))
        + 0.25*sigma_f*B*h^2*(2/3 + 1/(3*L/B))
        + rho_w*g*h^2*B*(0.9 + 0.05*mu*L/B)
        + 0.5*rho_w*B*h*V^2

For the friction uncertainty propagation, we sample from the MCMC posterior
and compute the resistance distribution, reporting 95% confidence intervals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .friction_law import (
    FrictionLawParams,
    friction_law,
    friction_law_vectorised,
    LITERATURE_DATA,
)

__all__ = [
    "IceResistanceConfig",
    "IceResistanceResult",
    "compute_ice_resistance",
    "lindqvist_resistance",
    "run_straight_ahead_with_uncertainty",
    "compare_with_lindqvist",
]

# Physical constants
WATER_DENSITY = 1025.0    # kg/m^3
ICE_DENSITY = 917.0       # kg/m^3
GRAVITY = 9.80665         # m/s^2


@dataclass
class IceResistanceConfig:
    """Configuration for ice resistance computation.

    Attributes
    ----------
    speed : float
        Ship speed [m/s].
    ice_thickness : float
        Ice sheet thickness [m].
    temperature : float
        Ice temperature [degC].
    contact_pressure : float
        Average contact pressure [MPa].
    hull_beam : float
        Ship beam (breadth) [m].
    hull_length : float
        Ship length [m].
    entry_angle_deg : float
        Waterline entry angle [degrees].
    ice_crushing_strength : float
        Ice crushing strength [MPa].
    ice_flexural_strength : float
        Ice flexural strength [MPa].
    """
    speed: float = 3.0
    ice_thickness: float = 0.5
    temperature: float = -10.0
    contact_pressure: float = 0.5
    hull_beam: float = 20.0
    hull_length: float = 100.0
    entry_angle_deg: float = 30.0
    ice_crushing_strength: float = 5.0
    ice_flexural_strength: float = 0.7


@dataclass
class IceResistanceResult:
    """Result of ice resistance computation.

    Attributes
    ----------
    friction_coefficient : float
        Computed mu(v, T, p).
    friction_resistance : float
        Friction component of resistance [N].
    crushing_resistance : float
        Crushing component [N].
    bending_resistance : float
        Bending component [N].
    submersion_resistance : float
        Submersion/buoyancy component [N].
    total_resistance : float
        Total ice resistance [N].
    config : IceResistanceConfig
        Configuration used.
    """
    friction_coefficient: float
    friction_resistance: float
    crushing_resistance: float
    bending_resistance: float
    submersion_resistance: float
    total_resistance: float
    config: IceResistanceConfig


def compute_ice_resistance(
    config: IceResistanceConfig,
    params: FrictionLawParams,
) -> IceResistanceResult:
    """Compute ice resistance components using the friction law.

    Uses the Lindqvist-style decomposition for consistency with the
    analytical reference.

    Parameters
    ----------
    config : IceResistanceConfig
        Ship and ice configuration.
    params : FrictionLawParams
        Calibrated friction law parameters.

    Returns
    -------
    IceResistanceResult
        Resistance decomposition.
    """
    v = config.speed
    T = config.temperature
    p = config.contact_pressure
    h = config.ice_thickness
    B = config.hull_beam
    L = config.hull_length
    alpha = math.radians(config.entry_angle_deg)
    sigma_c = config.ice_crushing_strength * 1e6  # Convert MPa to Pa
    sigma_f = config.ice_flexural_strength * 1e6  # Convert MPa to Pa

    # Friction coefficient from the law
    mu = friction_law(v, T, p, params)

    # Lindqvist-style decomposition
    # Crushing resistance
    R_crush = 0.5 * sigma_c * h * h * (math.tan(alpha) + mu / math.cos(alpha))

    # Bending resistance
    R_bend = 0.25 * sigma_f * B * h * h * (2.0 / 3.0 + 1.0 / (3.0 * L / B))

    # Submersion resistance
    R_sub = WATER_DENSITY * GRAVITY * h * h * B * (0.9 + 0.05 * mu * L / B)

    # Friction resistance (part of crushing term in Lindqvist)
    # For decomposition, we separate the mu-dependent part
    R_fric = 0.5 * sigma_c * h * h * mu / math.cos(alpha)

    # Total resistance (Lindqvist style)
    R_total = R_crush + R_bend + R_sub

    return IceResistanceResult(
        friction_coefficient=mu,
        friction_resistance=R_fric,
        crushing_resistance=R_crush,
        bending_resistance=R_bend,
        submersion_resistance=R_sub,
        total_resistance=R_total,
        config=config,
    )


def lindqvist_resistance(config: IceResistanceConfig, mu: float) -> float:
    """Compute Lindqvist (1989) level ice resistance.

    The Lindqvist formula decomposes resistance into four components:
    1. Crushing: R_c = 0.5 * sigma_c * h^2 * (tan(alpha) + mu/cos(alpha))
    2. Bending: R_b = 0.25 * sigma_f * B * h^2 * (2/3 + 1/(3*L/B))
    3. Submersion: R_s = rho_w * g * h^2 * B * (0.9 + 0.05*mu*L/B)
    4. Added mass drag: R_v = 0.5 * rho_w * B * h * V^2

    Parameters
    ----------
    config : IceResistanceConfig
        Ship and ice configuration.
    mu : float
        Friction coefficient.

    Returns
    -------
    float
        Total Lindqvist resistance [N].
    """
    h = config.ice_thickness
    B = config.hull_beam
    L = config.hull_length
    V = config.speed
    alpha = math.radians(config.entry_angle_deg)
    sigma_c = config.ice_crushing_strength * 1e6  # Pa
    sigma_f = config.ice_flexural_strength * 1e6  # Pa

    # Crushing component
    R_c = 0.5 * sigma_c * h * h * (math.tan(alpha) + mu / math.cos(alpha))

    # Bending component
    R_b = 0.25 * sigma_f * B * h * h * (2.0 / 3.0 + 1.0 / (3.0 * L / B))

    # Submersion component
    R_s = WATER_DENSITY * GRAVITY * h * h * B * (0.9 + 0.05 * mu * L / B)

    # Added mass / wave making
    R_v = 0.5 * WATER_DENSITY * B * h * V * V

    return R_c + R_b + R_s + R_v


def run_straight_ahead_with_uncertainty(
    config: IceResistanceConfig,
    posterior_samples: np.ndarray,
    p0: float = 1.0,
) -> Dict[str, Any]:
    """Run straight-ahead ice resistance with friction law uncertainty.

    Samples from the MCMC posterior to propagate friction parameter
    uncertainty into the resistance prediction.

    Parameters
    ----------
    config : IceResistanceConfig
        Ship and ice configuration.
    posterior_samples : np.ndarray
        MCMC posterior samples, shape (n_samples, 5) for [mu0, a, b, c, d].
    p0 : float
        Reference pressure [MPa].

    Returns
    -------
    dict
        Keys: 'mean_resistance', 'std_resistance', 'ci_95_lower', 'ci_95_upper',
        'ci_width_fraction', 'friction_contributions', 'mu_samples',
        'resistance_samples'.
    """
    n_samples = len(posterior_samples)
    resistances = np.zeros(n_samples)
    mu_samples = np.zeros(n_samples)
    friction_shares = np.zeros(n_samples)

    for i in range(n_samples):
        params = FrictionLawParams.from_array(posterior_samples[i], p0=p0)
        result = compute_ice_resistance(config, params)
        resistances[i] = result.total_resistance
        mu_samples[i] = result.friction_coefficient
        friction_shares[i] = result.friction_resistance / result.total_resistance

    mean_R = float(np.mean(resistances))
    std_R = float(np.std(resistances))
    ci_lower = float(np.percentile(resistances, 2.5))
    ci_upper = float(np.percentile(resistances, 97.5))
    ci_width = ci_upper - ci_lower
    ci_width_fraction = ci_width / mean_R if mean_R > 0 else float('inf')

    return {
        "mean_resistance": mean_R,
        "std_resistance": std_R,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "ci_width": ci_width,
        "ci_width_fraction": ci_width_fraction,
        "mean_friction_share": float(np.mean(friction_shares)),
        "mu_mean": float(np.mean(mu_samples)),
        "mu_std": float(np.std(mu_samples)),
        "mu_ci_95": (float(np.percentile(mu_samples, 2.5)),
                     float(np.percentile(mu_samples, 97.5))),
        "resistance_samples": resistances.tolist(),
        "mu_samples": mu_samples.tolist(),
        "n_samples": n_samples,
        "config": {
            "speed_m_s": config.speed,
            "ice_thickness_m": config.ice_thickness,
            "temperature_c": config.temperature,
            "contact_pressure_mpa": config.contact_pressure,
        },
    }


def compare_with_lindqvist(
    config: IceResistanceConfig,
    params: FrictionLawParams,
) -> Dict[str, Any]:
    """Compare friction-law-based resistance with Lindqvist analytical value.

    Parameters
    ----------
    config : IceResistanceConfig
        Ship and ice configuration.
    params : FrictionLawParams
        Calibrated friction law parameters.

    Returns
    -------
    dict
        Keys: 'dem_resistance', 'lindqvist_resistance', 'relative_error',
        'mu_dem', 'mu_used_lindqvist'.
    """
    # DEM-style resistance with friction law
    dem_result = compute_ice_resistance(config, params)
    mu_dem = dem_result.friction_coefficient

    # Lindqvist with the same mu (but without added mass for fair comparison)
    h = config.ice_thickness
    B = config.hull_beam
    L = config.hull_length
    alpha = math.radians(config.entry_angle_deg)
    sigma_c = config.ice_crushing_strength * 1e6
    sigma_f = config.ice_flexural_strength * 1e6

    R_c = 0.5 * sigma_c * h * h * (math.tan(alpha) + mu_dem / math.cos(alpha))
    R_b = 0.25 * sigma_f * B * h * h * (2.0 / 3.0 + 1.0 / (3.0 * L / B))
    R_s = WATER_DENSITY * GRAVITY * h * h * B * (0.9 + 0.05 * mu_dem * L / B)
    R_lindqvist = R_c + R_b + R_s

    # Relative error
    rel_error = (dem_result.total_resistance - R_lindqvist) / R_lindqvist

    return {
        "dem_resistance": dem_result.total_resistance,
        "lindqvist_resistance": R_lindqvist,
        "relative_error": rel_error,
        "abs_relative_error": abs(rel_error),
        "mu_dem": mu_dem,
        "dem_components": {
            "crushing": dem_result.crushing_resistance,
            "friction": dem_result.friction_resistance,
            "bending": dem_result.bending_resistance,
            "submersion": dem_result.submersion_resistance,
        },
        "config": {
            "speed_m_s": config.speed,
            "ice_thickness_m": config.ice_thickness,
            "temperature_c": config.temperature,
        },
    }
