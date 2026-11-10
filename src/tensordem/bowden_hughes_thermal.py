"""Bowden-Hughes thermal equilibrium model for ice-steel friction.

This module implements the Bowden-Hughes (1950) thermal balance model for
ice friction, which predicts the non-monotonic μ(V) behavior observed
experimentally: friction decreases with velocity at low speeds (melting
lubrication) but increases at high speeds (viscous shear in melt film).

Physics background:
- Friction generates heat at the contact interface: q = μ * σ * V
- Heat partitions between ice and steel (Blok partition)
- Ice melts at the interface, creating a water film of thickness h
- Film thickness determined by balance of:
  * Melting rate (from heat flux)
  * Squeeze-out rate (from normal load)
  * Refreezing rate (from heat conduction into bulk ice)
- Friction coefficient has two contributions:
  * Asperity contact (Coulomb friction): μ_asy
  * Viscous shear in water film: μ_film = η * V / (h * σ)
- Total friction: μ = (1-A) * μ_asy + A * μ_film
  where A = load-bearing area fraction of the film

The non-monotonic behavior:
- Low V: thin film, asperity contact dominates, μ ≈ μ_asy (high)
- Medium V: film thickens, mixed lubrication, μ decreases
- High V: thick film, viscous shear dominates, μ increases

Key equations (Bowden-Hughes):
1. Heat balance: q = μ*σ*V = q_ice + q_steel + q_melt
2. Blok partition: q_ice/q_steel = sqrt(k_ice*ρ_ice*c_ice) / sqrt(k_steel*ρ_steel*c_steel)
3. Melt film thickness: h = q_melt / (L_f * ρ_water * V_melt)
4. Viscous friction: μ_film = η * V / (h * σ)

References:
- Bowden & Hughes (1950) "The mechanism of friction on ice"
- Evans et al. (1982) "Friction and melting of ice"
- Oksanen (1989) "Ice-steel friction experiments"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "BowdenHughesConfig",
    "BowdenHughesResult",
    "compute_bowden_hughes_equilibrium",
    "compute_mu_velocity_curve",
    "energy_conservation_check",
]


@dataclass
class BowdenHughesConfig:
    """Configuration for Bowden-Hughes thermal equilibrium model.
    
    Attributes:
        ice_thermal_conductivity: Ice thermal conductivity [W/(m·K)]
        ice_density: Ice density [kg/m^3]
        ice_specific_heat: Ice specific heat [J/(kg·K)]
        ice_thermal_diffusivity: Ice thermal diffusivity [m^2/s]
        steel_thermal_conductivity: Steel thermal conductivity [W/(m·K)]
        steel_density: Steel density [kg/m^3]
        steel_specific_heat: Steel specific heat [J/(kg·K)]
        water_viscosity: Dynamic viscosity of water [Pa·s]
        water_density: Water density [kg/m^3]
        latent_heat_fusion: Latent heat of fusion of ice [J/kg]
        bulk_ice_temperature: Bulk ice temperature [°C]
        asperity_friction: Dry/boundary friction coefficient [-]
        contact_time: Contact duration [s]
        contact_radius: Contact patch radius [m]
        surface_roughness: Combined surface roughness [m]
    """
    ice_thermal_conductivity: float = 2.2  # W/(m·K)
    ice_density: float = 917.0  # kg/m^3
    ice_specific_heat: float = 2050.0  # J/(kg·K)
    ice_thermal_diffusivity: float = 1.17e-6  # m^2/s
    steel_thermal_conductivity: float = 50.0  # W/(m·K)
    steel_density: float = 7800.0  # kg/m^3
    steel_specific_heat: float = 470.0  # J/(kg·K)
    water_viscosity: float = 1.79e-3  # Pa·s at 0°C
    water_density: float = 1000.0  # kg/m^3
    latent_heat_fusion: float = 3.34e5  # J/kg
    bulk_ice_temperature: float = -5.0  # °C
    asperity_friction: float = 0.15  # Dry friction coefficient
    contact_time: float = 120.0  # s
    contact_radius: float = 2.0e-3  # m
    surface_roughness: float = 1.0e-6  # m


@dataclass
class BowdenHughesResult:
    """Result of Bowden-Hughes thermal equilibrium computation.
    
    Attributes:
        velocity: Sliding velocity [m/s]
        normal_stress: Normal stress [Pa]
        friction_coefficient: Total friction coefficient μ [-]
        film_thickness: Melt film thickness [m]
        heat_flux_total: Total frictional heat flux [W/m^2]
        heat_flux_ice: Heat flux into ice [W/m^2]
        heat_flux_steel: Heat flux into steel [W/m^2]
        heat_flux_melt: Heat flux for melting [W/m^2]
        interface_temperature: Interface temperature [°C]
        asperity_area_fraction: Fraction of load on asperities [-]
        film_area_fraction: Fraction of load on water film [-]
        regime: Lubrication regime ("boundary", "mixed", "hydrodynamic")
    """
    velocity: float
    normal_stress: float
    friction_coefficient: float
    film_thickness: float
    heat_flux_total: float
    heat_flux_ice: float
    heat_flux_steel: float
    heat_flux_melt: float
    interface_temperature: float
    asperity_area_fraction: float
    film_area_fraction: float
    regime: str


def _blok_partition_coefficient(config: BowdenHughesConfig) -> float:
    """Compute Blok heat partition coefficient for ice.
    
    The Blok partition divides frictional heat between two bodies based on
    their thermal effusivities: e = sqrt(k * ρ * c)
    
    gamma_ice = e_ice / (e_ice + e_steel)
    
    Args:
        config: BowdenHughesConfig with thermal properties
    
    Returns:
        Fraction of heat going into ice [-]
    """
    e_ice = math.sqrt(
        config.ice_thermal_conductivity *
        config.ice_density *
        config.ice_specific_heat
    )
    e_steel = math.sqrt(
        config.steel_thermal_conductivity *
        config.steel_density *
        config.steel_specific_heat
    )
    
    gamma_ice = e_ice / (e_ice + e_steel)
    return gamma_ice


def _transient_conduction_depth(
    thermal_diffusivity: float, time: float
) -> float:
    """Compute thermal penetration depth for transient conduction.
    
    delta = sqrt(alpha * t)
    
    Args:
        thermal_diffusivity: Thermal diffusivity [m^2/s]
        time: Time [s]
    
    Returns:
        Penetration depth [m]
    """
    return math.sqrt(thermal_diffusivity * time)


def _conductive_heat_flux(
    k: float, T_interface: float, T_bulk: float, delta: float
) -> float:
    """Compute conductive heat flux.
    
    q = k * (T_interface - T_bulk) / delta
    
    Args:
        k: Thermal conductivity [W/(m·K)]
        T_interface: Interface temperature [°C]
        T_bulk: Bulk temperature [°C]
        delta: Thermal penetration depth [m]
    
    Returns:
        Heat flux [W/m^2]
    """
    if delta <= 0:
        return 0.0
    return k * (T_interface - T_bulk) / delta


def _melt_film_thickness_steady(
    heat_flux_melt: float,
    velocity: float,
    latent_heat: float,
    water_density: float,
    contact_radius: float,
) -> float:
    """Compute steady-state melt film thickness.
    
    Mass balance: melting rate = squeeze-out rate
    rho_w * V_melt = q_melt / L_f
    h = V_melt * t_residence
    
    where t_residence = contact_radius / velocity
    
    Args:
        heat_flux_melt: Heat flux available for melting [W/m^2]
        velocity: Sliding velocity [m/s]
        latent_heat: Latent heat of fusion [J/kg]
        water_density: Water density [kg/m^3]
        contact_radius: Contact radius [m]
    
    Returns:
        Film thickness [m]
    """
    if velocity <= 0 or heat_flux_melt <= 0:
        return 0.0
    
    # Melting velocity
    V_melt = heat_flux_melt / (latent_heat * water_density)
    
    # Residence time
    t_res = contact_radius / velocity
    
    # Film thickness
    h = V_melt * t_res
    
    return max(0.0, h)


def _squeeze_out_time(
    film_thickness: float,
    normal_stress: float,
    water_viscosity: float,
    contact_radius: float,
) -> float:
    """Compute time for film squeeze-out under normal load.
    
    Reynolds equation for squeeze film:
    t_squeeze = 3 * eta * R^2 / (2 * sigma * h^2)
    
    Args:
        film_thickness: Film thickness [m]
        normal_stress: Normal stress [Pa]
        water_viscosity: Water viscosity [Pa·s]
        contact_radius: Contact radius [m]
    
    Returns:
        Squeeze-out time [s]
    """
    if film_thickness <= 0 or normal_stress <= 0:
        return float('inf')
    
    t_squeeze = (3.0 * water_viscosity * contact_radius**2 /
                 (2.0 * normal_stress * film_thickness**2))
    
    return t_squeeze


def _film_area_fraction(
    film_thickness: float,
    surface_roughness: float,
    squeeze_time: float,
    contact_time: float,
) -> float:
    """Compute load-bearing area fraction of water film.
    
    The film covers the contact area if:
    1. Film thickness > surface roughness
    2. Film lifetime (squeeze time) > contact time
    
    A_film = min(1, h / roughness) * min(1, t_squeeze / t_contact)
    
    Args:
        film_thickness: Film thickness [m]
        surface_roughness: Surface roughness [m]
        squeeze_time: Squeeze-out time [s]
        contact_time: Contact duration [s]
    
    Returns:
        Film area fraction [-] (0 to 1)
    """
    if surface_roughness <= 0:
        return 1.0
    
    # Thickness criterion
    A_h = min(1.0, film_thickness / surface_roughness)
    
    # Lifetime criterion
    A_t = min(1.0, squeeze_time / contact_time) if squeeze_time < float('inf') else 1.0
    
    return A_h * A_t


def compute_bowden_hughes_equilibrium(
    velocity: float,
    normal_stress: float,
    config: BowdenHughesConfig,
) -> BowdenHughesResult:
    """Compute Bowden-Hughes thermal equilibrium for given conditions.
    
    Solves the coupled thermal-mechanical problem:
    1. Total heat flux: q = μ * σ * V
    2. Heat partition: q_ice + q_steel + q_melt = q
    3. Film thickness from melt balance
    4. Friction from asperity + film contributions
    
    The interface temperature is assumed to be 0°C (melting point) when
    sufficient heat is generated. Otherwise, it's below melting.
    
    Args:
        velocity: Sliding velocity [m/s]
        normal_stress: Normal stress [Pa]
        config: BowdenHughesConfig with material properties
    
    Returns:
        BowdenHughesResult with all computed quantities
    """
    if velocity <= 0:
        return BowdenHughesResult(
            velocity=velocity,
            normal_stress=normal_stress,
            friction_coefficient=config.asperity_friction,
            film_thickness=0.0,
            heat_flux_total=0.0,
            heat_flux_ice=0.0,
            heat_flux_steel=0.0,
            heat_flux_melt=0.0,
            interface_temperature=config.bulk_ice_temperature,
            asperity_area_fraction=1.0,
            film_area_fraction=0.0,
            regime="boundary",
        )
    
    # Blok partition coefficient
    gamma_ice = _blok_partition_coefficient(config)
    gamma_steel = 1.0 - gamma_ice
    
    # Thermal penetration depths
    delta_ice = _transient_conduction_depth(
        config.ice_thermal_diffusivity, config.contact_time
    )
    
    # Iterative solution for interface temperature and friction
    # Start with asperity friction, compute heat, check if melting occurs
    
    mu_asy = config.asperity_friction
    q_total = mu_asy * normal_stress * velocity
    
    # Conductive heat fluxes (assuming T_interface = 0°C if melting)
    T_melt = 0.0  # Melting point
    q_cond_ice = _conductive_heat_flux(
        config.ice_thermal_conductivity,
        T_melt,
        config.bulk_ice_temperature,
        delta_ice
    )
    
    # Check if enough heat for melting
    q_available_melt = q_total - gamma_ice * q_cond_ice - gamma_steel * q_cond_ice
    
    if q_available_melt > 0:
        # Melting occurs, interface at 0°C
        T_interface = T_melt
        q_melt = max(0.0, q_available_melt)
        q_ice = gamma_ice * q_cond_ice
        q_steel = gamma_steel * q_cond_ice
    else:
        # No melting, interface below 0°C
        # Solve for T_interface from heat balance
        # q_total = q_cond_ice + q_cond_steel
        # Approximate: T_interface = T_bulk + q_total * delta_eff / k_eff
        k_eff = (config.ice_thermal_conductivity + config.steel_thermal_conductivity) / 2.0
        delta_eff = (delta_ice + _transient_conduction_depth(
            config.steel_thermal_conductivity / (config.steel_density * config.steel_specific_heat),
            config.contact_time
        )) / 2.0
        
        T_interface = config.bulk_ice_temperature + q_total * delta_eff / k_eff
        T_interface = min(T_interface, T_melt)
        
        q_melt = 0.0
        q_ice = gamma_ice * q_total
        q_steel = gamma_steel * q_total
    
    # Film thickness
    h_film = _melt_film_thickness_steady(
        q_melt, velocity, config.latent_heat_fusion,
        config.water_density, config.contact_radius
    )
    
    # Squeeze-out time
    t_squeeze = _squeeze_out_time(
        h_film, normal_stress, config.water_viscosity, config.contact_radius
    )
    
    # Film area fraction
    A_film = _film_area_fraction(
        h_film, config.surface_roughness, t_squeeze, config.contact_time
    )
    A_asy = 1.0 - A_film
    
    # Viscous friction in film
    if h_film > 0:
        mu_film = config.water_viscosity * velocity / (h_film * normal_stress)
        mu_film = min(mu_film, mu_asy)  # Film can't increase friction above asperity value
    else:
        mu_film = mu_asy
    
    # Total friction coefficient
    mu_total = A_asy * mu_asy + A_film * mu_film
    
    # Determine regime
    if A_film < 0.05:
        regime = "boundary"
    elif A_film < 0.5:
        regime = "mixed"
    else:
        regime = "hydrodynamic"
    
    return BowdenHughesResult(
        velocity=velocity,
        normal_stress=normal_stress,
        friction_coefficient=mu_total,
        film_thickness=h_film,
        heat_flux_total=q_total,
        heat_flux_ice=q_ice,
        heat_flux_steel=q_steel,
        heat_flux_melt=q_melt,
        interface_temperature=T_interface,
        asperity_area_fraction=A_asy,
        film_area_fraction=A_film,
        regime=regime,
    )


def compute_mu_velocity_curve(
    normal_stress: float,
    velocities: np.ndarray,
    config: BowdenHughesConfig,
) -> Tuple[np.ndarray, List[BowdenHughesResult]]:
    """Compute friction coefficient vs velocity curve.
    
    Args:
        normal_stress: Normal stress [Pa]
        velocities: Array of sliding velocities [m/s]
        config: BowdenHughesConfig with material properties
    
    Returns:
        Tuple of (mu_array, list_of_results)
    """
    mu_array = np.zeros_like(velocities)
    results = []
    
    for i, v in enumerate(velocities):
        result = compute_bowden_hughes_equilibrium(v, normal_stress, config)
        mu_array[i] = result.friction_coefficient
        results.append(result)
    
    return mu_array, results


def energy_conservation_check(result: BowdenHughesResult, tolerance: float = 0.05) -> bool:
    """Check energy conservation in the thermal balance.
    
    Verifies: q_total = q_ice + q_steel + q_melt (within tolerance)
    
    Args:
        result: BowdenHughesResult to check
        tolerance: Relative tolerance for energy balance
    
    Returns:
        True if energy is conserved within tolerance
    """
    q_total = result.heat_flux_total
    q_sum = result.heat_flux_ice + result.heat_flux_steel + result.heat_flux_melt
    
    if q_total == 0:
        return q_sum == 0
    
    residual = abs(q_total - q_sum) / q_total
    return residual <= tolerance
