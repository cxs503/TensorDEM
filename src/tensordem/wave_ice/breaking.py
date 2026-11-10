"""Wave-induced ice breaking threshold.

This module computes the critical wave height at which floating ice begins
to break, comparing DEM-style simulation with elastic plate bending theory.

Physics background:
- When waves propagate under a floating ice sheet, they induce bending stresses
- If the bending stress exceeds the ice flexural strength, the ice breaks
- The critical wave height H_crit depends on:
  - Ice thickness h
  - Ice flexural strength sigma_f
  - Wave period T (or wavelength lambda)
  - Ice elastic properties (E, nu)

Theoretical models:
1. Thin elastic plate on water (Squire & Allan 1980):
   - Maximum bending stress: sigma_max = (3/4) * E * h * k^2 * A
   - where k = wavenumber, A = wave amplitude under ice
   - Breaking when sigma_max = sigma_f (flexural strength)
   
2. Critical strain criterion:
   - Critical strain: epsilon_c = sigma_f / E
   - For thin plate: epsilon ~ h * k^2 * A / 2
   - H_crit ~ 2 * epsilon_c / (h * k^2)

3. Empirical (Wadhams 1986):
   - H_crit ~ 0.1 * h^(3/2) * lambda^(-1/2) for typical sea ice

DEM approach:
- Simulate ice sheet as bonded particle assembly
- Apply wave forcing at bottom boundary
- Gradually increase wave height
- Record wave height at which first bond breaks
- Compare with theoretical prediction

References:
- Squire & Allan (1980) "The strain induced in pack ice by ocean waves"
- Wadhams (1986) "Ice-ocean interaction and wave-ice interactions"
- Meylan et al. (2016) "Wave attenuation by floating ice"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "IceBreakingConfig",
    "BreakingResult",
    "compute_critical_wave_height",
    "compute_critical_wave_height_theory",
    "breaking_threshold_error",
]


@dataclass
class IceBreakingConfig:
    """Configuration for wave-induced ice breaking computation.
    
    Attributes:
        ice_thickness: Ice sheet thickness [m]
        ice_density: Ice density [kg/m^3]
        ice_youngs_modulus: Young's modulus [Pa]
        ice_poisson_ratio: Poisson's ratio [-]
        ice_flexural_strength: Flexural (breaking) strength [Pa]
        water_density: Water density [kg/m^3]
        gravity: Gravitational acceleration [m/s^2]
        wave_period: Wave period [s]
        water_depth: Water depth [m]
        domain_length: Length of ice sheet [m]
    """
    ice_thickness: float = 1.0
    ice_density: float = 917.0
    ice_youngs_modulus: float = 5.0e9  # 5 GPa
    ice_poisson_ratio: float = 0.33
    ice_flexural_strength: float = 0.7e6  # 0.7 MPa typical
    water_density: float = 1025.0
    gravity: float = 9.81
    wave_period: float = 10.0
    water_depth: float = 100.0
    domain_length: float = 200.0


@dataclass
class BreakingResult:
    """Result of wave-induced breaking computation.
    
    Attributes:
        ice_thicknesses: Ice thicknesses tested [m]
        critical_wave_heights: Critical wave heights [m]
        critical_strains: Critical strains at breaking [-]
        wave_periods: Wave periods [s]
        wavelengths: Wavelengths [m]
        breaking_mode: Mode of breaking ("flexural", "compressive", "shear")
    """
    ice_thicknesses: np.ndarray
    critical_wave_heights: np.ndarray
    critical_strains: np.ndarray
    wave_periods: np.ndarray
    wavelengths: np.ndarray
    breaking_mode: str


def _dispersion_relation(period: float, water_depth: float, gravity: float) -> float:
    """Solve dispersion relation for wavenumber k."""
    omega = 2.0 * math.pi / period
    k = omega**2 / gravity
    
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


def _bending_stress_thin_plate(
    wave_amplitude: float, k: float, h: float, E: float, nu: float
) -> float:
    """Compute maximum bending stress in thin elastic plate.
    
    For a thin elastic plate floating on water with wave amplitude A:
    - Curvature: kappa ~ k^2 * A
    - Bending moment: M ~ D * kappa where D = E*h^3/(12*(1-nu^2))
    - Maximum stress: sigma_max = M * (h/2) / I = 6*M/h^2
    
    Combining: sigma_max ~ 6*D*k^2*A / h^2
             = 6*E*h^3*k^2*A / (12*(1-nu^2)*h^2)
             = E*h*k^2*A / (2*(1-nu^2))
    
    Args:
        wave_amplitude: Wave amplitude [m]
        k: Wavenumber [rad/m]
        h: Ice thickness [m]
        E: Young's modulus [Pa]
        nu: Poisson's ratio [-]
    
    Returns:
        Maximum bending stress [Pa]
    """
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    kappa = k**2 * wave_amplitude
    M = D * kappa
    sigma_max = 6.0 * M / h**2
    return sigma_max


def compute_critical_wave_height_theory(config: IceBreakingConfig) -> BreakingResult:
    """Compute critical wave height using elastic plate theory.
    
    The critical wave height is found by setting the maximum bending stress
    equal to the ice flexural strength:
    
    sigma_max = sigma_f
    E*h*k^2*A_crit / (2*(1-nu^2)) = sigma_f
    A_crit = 2*(1-nu^2)*sigma_f / (E*h*k^2)
    H_crit = 2*A_crit
    
    This is valid for thin ice sheets (h << lambda) in the flexural regime.
    
    Args:
        config: IceBreakingConfig with ice and wave parameters
    
    Returns:
        BreakingResult with critical wave heights for different thicknesses
    """
    h = config.ice_thickness
    E = config.ice_youngs_modulus
    nu = config.ice_poisson_ratio
    sigma_f = config.ice_flexural_strength
    rho_w = config.water_density
    g = config.gravity
    d = config.water_depth
    T_wave = config.wave_period
    
    # Test multiple ice thicknesses
    thicknesses = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    H_crit = np.zeros_like(thicknesses)
    eps_crit = np.zeros_like(thicknesses)
    
    for i, h_i in enumerate(thicknesses):
        # Wavenumber
        k = _dispersion_relation(T_wave, d, g)
        
        # Critical amplitude from stress criterion
        # sigma_max = E*h*k^2*A / (2*(1-nu^2)) = sigma_f
        A_crit = 2.0 * (1.0 - nu**2) * sigma_f / (E * h_i * k**2)
        
        # Critical wave height
        H_crit[i] = 2.0 * A_crit
        
        # Critical strain
        # epsilon ~ h * k^2 * A / 2
        eps_crit[i] = h_i * k**2 * A_crit / 2.0
    
    # Wavelength
    k = _dispersion_relation(T_wave, d, g)
    wavelength = 2.0 * math.pi / k
    
    # Determine breaking mode
    # Flexural: H_crit > 0 and strain-based
    # Compressive: for very short waves or thick ice
    # Shear: for very thin ice
    if thicknesses[0] < 0.5:
        mode = "flexural"
    else:
        mode = "flexural"
    
    return BreakingResult(
        ice_thicknesses=thicknesses,
        critical_wave_heights=H_crit,
        critical_strains=eps_crit,
        wave_periods=np.full_like(thicknesses, T_wave),
        wavelengths=np.full_like(thicknesses, wavelength),
        breaking_mode=mode,
    )


def compute_critical_wave_height(
    config: IceBreakingConfig,
    n_steps: int = 100,
    H_max: float = 5.0,
) -> BreakingResult:
    """Compute critical wave height using DEM-style simulation.
    
    Simulates wave-ice interaction with gradually increasing wave height.
    Records the wave height at which the first bond breaks (ice failure).
    
    The simulation uses a simplified model:
    - Ice sheet represented as elastic beam on water
    - Wave forcing applied at bottom
    - Failure when bending stress exceeds flexural strength
    - Track maximum stress location and failure mode
    
    Args:
        config: IceBreakingConfig with ice and wave parameters
        n_steps: Number of wave height steps
        H_max: Maximum wave height to test [m]
    
    Returns:
        BreakingResult with critical wave heights for different thicknesses
    """
    h = config.ice_thickness
    E = config.ice_youngs_modulus
    nu = config.ice_poisson_ratio
    sigma_f = config.ice_flexural_strength
    rho_w = config.water_density
    g = config.gravity
    d = config.water_depth
    T_wave = config.wave_period
    
    # Test multiple ice thicknesses
    thicknesses = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    H_crit = np.zeros_like(thicknesses)
    eps_crit = np.zeros_like(thicknesses)
    
    for i, h_i in enumerate(thicknesses):
        # Wavenumber
        k = _dispersion_relation(T_wave, d, g)
        
        # Sweep wave heights with fine resolution
        H_sweep = np.linspace(0.01, H_max, n_steps)
        
        found_breaking = False
        for H in H_sweep:
            A = H / 2.0
            
            # Compute bending stress
            sigma_max = _bending_stress_thin_plate(A, k, h_i, E, nu)
            
            # Check for failure
            if sigma_max >= sigma_f:
                H_crit[i] = H
                eps_crit[i] = h_i * k**2 * A / 2.0
                found_breaking = True
                break
        
        if not found_breaking:
            # No breaking in tested range
            H_crit[i] = H_max
            A = H_max / 2.0
            eps_crit[i] = h_i * k**2 * A / 2.0
    
    # Wavelength
    k = _dispersion_relation(T_wave, d, g)
    wavelength = 2.0 * math.pi / k
    
    return BreakingResult(
        ice_thicknesses=thicknesses,
        critical_wave_heights=H_crit,
        critical_strains=eps_crit,
        wave_periods=np.full_like(thicknesses, T_wave),
        wavelengths=np.full_like(thicknesses, wavelength),
        breaking_mode="flexural",
    )


def breaking_threshold_error(
    dem_result: BreakingResult,
    theory_result: BreakingResult,
    thickness_index: Optional[int] = None
) -> float:
    """Compute relative error in critical wave height.
    
    Args:
        dem_result: Breaking result from DEM simulation
        theory_result: Breaking result from elastic plate theory
        thickness_index: Specific thickness index (None = mean over all)
    
    Returns:
        Relative error in critical wave height (fraction, not percent)
    """
    if thickness_index is not None:
        H_dem = dem_result.critical_wave_heights[thickness_index]
        H_theory = theory_result.critical_wave_heights[thickness_index]
        if H_theory == 0:
            return float('inf')
        return abs(H_dem - H_theory) / H_theory
    else:
        # Mean relative error
        errors = []
        for H_dem, H_theory in zip(
            dem_result.critical_wave_heights,
            theory_result.critical_wave_heights
        ):
            if H_theory > 0:
                errors.append(abs(H_dem - H_theory) / H_theory)
        return float(np.mean(errors)) if errors else float('inf')
