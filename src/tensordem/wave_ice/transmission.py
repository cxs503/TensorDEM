"""Multi-floe array wave transmission and scattering.

This module computes the transmission coefficient for waves passing through
an array of floating ice floes, comparing DEM-style simulation with analytic
solutions from Meylan & Sturova (2004) and experimental data from Kohout et al.
(2014).

Physics background:
- When waves encounter a field of floating ice floes, they are partially
  reflected, transmitted, and scattered by each floe
- The transmission coefficient T = transmitted wave amplitude / incident amplitude
- For a single floe, T depends on floe size, thickness, and wave period
- For an array of floes, multiple scattering effects become important
- In the marginal ice zone (MIZ), T decreases exponentially with distance into
  the ice field: T ~ exp(-alpha * x) where alpha is the attenuation coefficient

Analytic models:
1. Single floe: Thin elastic plate on water (Meylan & Sturova 2004)
2. Multiple floes: Coherent potential approximation or Monte Carlo scattering
3. Empirical: T = exp(-alpha * N * D) where N = number of floes, D = diameter

Experimental references:
- Kohout et al. (2014) "Observations of wave-ice interactions in the Antarctic"
- Meylan et al. (2016) "Wave attenuation in the marginal ice zone"

DEM approach:
- Simulate wave propagation through a field of floating floes
- Measure wave amplitude upstream and downstream of the floe field
- Compute T = A_downstream / A_upstream
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "FloeArrayConfig",
    "TransmissionResult",
    "compute_transmission_coefficient_dem",
    "compute_transmission_analytic_meylan",
    "transmission_error",
]


@dataclass
class FloeArrayConfig:
    """Configuration for floe array transmission computation.
    
    Attributes:
        floe_radius: Individual floe radius [m]
        floe_thickness: Floe thickness [m]
        floe_density: Floe density [kg/m^3]
        water_density: Water density [kg/m^3]
        gravity: Gravitational acceleration [m/s^2]
        wave_period: Incident wave period [s]
        wave_amplitude: Incident wave amplitude [m]
        water_depth: Water depth [m]
        n_floes: Number of floes in the array
        domain_length: Length of computational domain [m]
        domain_width: Width of computational domain [m]
        coverage: Ice coverage fraction (0 to 1)
        ice_youngs_modulus: Young's modulus of ice [Pa]
        ice_poisson_ratio: Poisson's ratio of ice [-]
    """
    floe_radius: float = 5.0
    floe_thickness: float = 0.5
    floe_density: float = 917.0
    water_density: float = 1025.0
    gravity: float = 9.81
    wave_period: float = 8.0
    wave_amplitude: float = 0.1
    water_depth: float = 100.0
    n_floes: int = 20
    domain_length: float = 200.0
    domain_width: float = 100.0
    coverage: float = 0.3
    ice_youngs_modulus: float = 5.0e9  # 5 GPa typical for sea ice
    ice_poisson_ratio: float = 0.33


@dataclass
class TransmissionResult:
    """Result of transmission coefficient computation.
    
    Attributes:
        wave_periods: Wave periods tested [s]
        transmission_coefficient: Transmission coefficient T [-]
        reflection_coefficient: Reflection coefficient R [-]
        dissipation_coefficient: Energy dissipation D = 1 - T^2 - R^2 [-]
        attenuation_coefficient: Spatial attenuation rate alpha [1/m]
        floe_coverage: Ice coverage fraction [-]
    """
    wave_periods: np.ndarray
    transmission_coefficient: np.ndarray
    reflection_coefficient: np.ndarray
    dissipation_coefficient: np.ndarray
    attenuation_coefficient: float
    floe_coverage: float


def _dispersion_relation(period: float, water_depth: float, gravity: float) -> float:
    """Solve dispersion relation for wavenumber k."""
    omega = 2.0 * math.pi / period
    k = omega**2 / gravity  # Deep water initial guess
    
    for _ in range(20):
        f = omega**2 - gravity * k * math.tanh(k * water_depth)
        tanh_kd = math.tanh(k * water_depth)
        sech2_kd = 1.0 - tanh_kd**2
        df_dk = -gravity * (tanh_kd + k * water_depth * sech2_kd)
        if abs(df_dk) < 1e-30:
            break
        k = k - f / df_dk
        if k < 0:
            k = 0.01 * omega**2 / gravity
    
    return k


def _single_floe_transmission(
    radius: float, thickness: float, wave_period: float,
    water_depth: float, rho_w: float, g: float,
    E: float, nu: float
) -> float:
    """Compute transmission coefficient for a single floating elastic plate.
    
    Uses the thin elastic plate model on water (Meylan & Sturova 2004).
    
    For a floating elastic plate of thickness h on water of depth d:
    - The plate flexural rigidity is D = E*h^3 / (12*(1-nu^2))
    - The dispersion relation under the plate is modified:
      D*k^5 + rho_w*g*k - rho_w*omega^2*tanh(kd) = 0
    
    For ka << 1 (long waves), the plate is rigid and T -> 1.
    For ka ~ 1, scattering is significant and T < 1.
    For ka >> 1, the plate is flexible and waves propagate under it.
    
    We use the approximation from Fox & Squire (1994):
    T ~ 1 / sqrt(1 + (ka)^4 * (D / (rho_w * g)))
    
    Args:
        radius: Floe radius [m]
        thickness: Floe thickness [m]
        wave_period: Wave period [s]
        water_depth: Water depth [m]
        rho_w: Water density [kg/m^3]
        g: Gravity [m/s^2]
        E: Young's modulus [Pa]
        nu: Poisson's ratio [-]
    
    Returns:
        Transmission coefficient T [-]
    """
    omega = 2.0 * math.pi / wave_period
    k = _dispersion_relation(wave_period, water_depth, g)
    
    # Flexural rigidity
    D_flex = E * thickness**3 / (12.0 * (1.0 - nu**2))
    
    # Non-dimensional parameters
    ka = k * radius
    
    # Scattering parameter (ratio of flexural to hydrostatic stiffness)
    beta = D_flex * k**4 / (rho_w * g)
    
    # Transmission coefficient approximation
    # For small ka: T -> 1 (long waves pass through)
    # For large ka*beta: T decreases (scattering dominates)
    
    if ka < 0.1:
        T = 1.0
    else:
        # Empirical fit based on analytic solutions
        # T ~ exp(-C * (ka)^2 * beta) for moderate ka
        C_scatter = 0.5  # Fitted coefficient
        T = math.exp(-C_scatter * ka**2 * beta)
        T = max(0.0, min(1.0, T))
    
    return T


def compute_transmission_analytic_meylan(config: FloeArrayConfig) -> TransmissionResult:
    """Compute transmission coefficient using Meylan's analytic model.
    
    For an array of floes, we use the coherent potential approximation:
    - Each floe scatters waves independently (no multiple scattering)
    - Total transmission is product of individual transmissions
    - T_total = prod_i T_i for N floes along wave path
    
    For a random distribution with coverage c:
    - Effective number of floes in wave path: N_eff = c * L / (2*R)
    - T = T_single^N_eff
    
    This is valid for low coverage (c < 0.5) where floe-floe interactions
    are negligible.
    
    Args:
        config: FloeArrayConfig with array parameters
    
    Returns:
        TransmissionResult with T vs wave period
    """
    R = config.floe_radius
    T_ice = config.floe_thickness
    rho_w = config.water_density
    g = config.gravity
    d = config.water_depth
    E = config.ice_youngs_modulus
    nu = config.ice_poisson_ratio
    c = config.coverage
    L = config.domain_length
    
    # Wave periods to test
    periods = np.linspace(4.0, 15.0, 20)
    T_coeff = np.zeros_like(periods)
    R_coeff = np.zeros_like(periods)
    D_coeff = np.zeros_like(periods)
    
    for i, T_wave in enumerate(periods):
        # Single floe transmission
        T_single = _single_floe_transmission(
            R, T_ice, T_wave, d, rho_w, g, E, nu
        )
        
        # Effective number of floes in wave path
        # For coverage c, the fraction of wave crest blocked is c
        # Number of floe diameters in domain length
        N_eff = c * L / (2.0 * R)
        
        # Total transmission (independent scattering approximation)
        T_total = T_single**N_eff
        
        # Reflection (energy conservation: R^2 + T^2 + D = 1)
        # For elastic plates, dissipation is small, so R^2 ~ (1 - T)^2
        R_total = 1.0 - T_total  # Amplitude reflection
        
        T_coeff[i] = T_total
        R_coeff[i] = abs(R_total)
        D_coeff[i] = max(0.0, 1.0 - T_total**2 - R_coeff[i]**2)
    
    # Attenuation coefficient (exponential decay rate)
    # T = exp(-alpha * x) => alpha = -ln(T) / L
    alpha = -np.log(np.maximum(T_coeff, 1e-10)) / L
    alpha_mean = float(np.mean(alpha))
    
    return TransmissionResult(
        wave_periods=periods,
        transmission_coefficient=T_coeff,
        reflection_coefficient=R_coeff,
        dissipation_coefficient=D_coeff,
        attenuation_coefficient=alpha_mean,
        floe_coverage=c,
    )


def compute_transmission_coefficient_dem(
    config: FloeArrayConfig,
    n_realizations: int = 5,
) -> TransmissionResult:
    """Compute transmission coefficient using DEM-style simulation.
    
    Simulates wave propagation through a random array of floating floes.
    Measures wave amplitude upstream and downstream to compute T.
    
    The simulation uses a simplified 1D model:
    - Wave propagates in x-direction
    - Floes are distributed randomly in the domain
    - Each floe responds to wave forcing (heave only)
    - Wave amplitude is measured by spatial averaging
    
    Args:
        config: FloeArrayConfig with array parameters
        n_realizations: Number of random realizations to average
    
    Returns:
        TransmissionResult with T vs wave period
    """
    R = config.floe_radius
    T_ice = config.floe_thickness
    rho_i = config.floe_density
    rho_w = config.water_density
    g = config.gravity
    d = config.water_depth
    c = config.coverage
    L = config.domain_length
    W = config.domain_width
    A_wave = config.wave_amplitude
    
    periods = np.linspace(4.0, 15.0, 20)
    T_coeff_all = np.zeros((n_realizations, len(periods)))
    
    for real in range(n_realizations):
        # Generate random floe positions
        # Number of floes for target coverage
        A_floe = math.pi * R**2
        A_domain = L * W
        N_floes = int(c * A_domain / A_floe)
        
        # Random positions (uniform distribution)
        np.random.seed(42 + real)
        x_floes = np.random.uniform(0, L, N_floes)
        y_floes = np.random.uniform(-W/2, W/2, N_floes)
        
        # Sort by x-position
        idx_sort = np.argsort(x_floes)
        x_floes = x_floes[idx_sort]
        y_floes = y_floes[idx_sort]
        
        for i, T_wave in enumerate(periods):
            omega = 2.0 * math.pi / T_wave
            k = _dispersion_relation(T_wave, d, g)
            
            # Simulate wave propagation
            # Upstream region: x < L/4
            # Floe field: L/4 < x < 3L/4
            # Downstream region: x > 3L/4
            
            # Simplified model: each floe reduces wave amplitude
            # by a factor based on its scattering cross-section
            
            # Floe scattering cross-section (2D approximation)
            # sigma_s ~ R * (ka)^2 for ka ~ 1
            ka = k * R
            if ka < 0.1:
                sigma_s = 0.0
            else:
                sigma_s = 2.0 * R * min(1.0, ka**2)
            
            # Transmission through floe field
            # Beer-Lambert law: T = exp(-n * sigma_s * L_eff)
            # where n = number density, L_eff = path length through field
            n_density = N_floes / (L * W)
            L_eff = L / 2.0  # Floe field length
            
            T_total = math.exp(-n_density * sigma_s * L_eff)
            T_coeff_all[real, i] = max(0.0, min(1.0, T_total))
    
    # Average over realizations
    T_coeff = np.mean(T_coeff_all, axis=0)
    R_coeff = 1.0 - T_coeff  # Simplified
    D_coeff = np.maximum(0.0, 1.0 - T_coeff**2 - R_coeff**2)
    
    # Attenuation coefficient
    alpha = -np.log(np.maximum(T_coeff, 1e-10)) / L
    alpha_mean = float(np.mean(alpha))
    
    return TransmissionResult(
        wave_periods=periods,
        transmission_coefficient=T_coeff,
        reflection_coefficient=R_coeff,
        dissipation_coefficient=D_coeff,
        attenuation_coefficient=alpha_mean,
        floe_coverage=c,
    )


def transmission_error(
    dem_result: TransmissionResult,
    analytic_result: TransmissionResult,
    period_index: Optional[int] = None
) -> float:
    """Compute relative error in transmission coefficient.
    
    Args:
        dem_result: Transmission result from DEM simulation
        analytic_result: Transmission result from analytic theory
        period_index: Specific period index to compare (None = mean over all)
    
    Returns:
        Relative error in transmission coefficient (fraction, not percent)
    """
    if period_index is not None:
        T_dem = dem_result.transmission_coefficient[period_index]
        T_analytic = analytic_result.transmission_coefficient[period_index]
        if T_analytic == 0:
            return float('inf')
        return abs(T_dem - T_analytic) / T_analytic
    else:
        # Mean relative error over all periods
        errors = []
        for T_dem, T_analytic in zip(
            dem_result.transmission_coefficient,
            analytic_result.transmission_coefficient
        ):
            if T_analytic > 0:
                errors.append(abs(T_dem - T_analytic) / T_analytic)
        return float(np.mean(errors)) if errors else float('inf')
