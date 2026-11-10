"""M2: Ice-ice collision fracture energy ledger.

Implements and verifies:
- Fracture energy G_c from fracture toughness K_IC
- Energy balance: E_k = E_elastic + E_friction + E_fracture + E_kinetic_residual
- Verification: E_fracture/E_k vs Timco 2008 experimental values (≤20%)
- Negative control: v=0 → E_fracture=0

References:
- Timco, G.W. (2008): "Mechanical properties of sea ice"
- Schulson, E.M. (2001): "Brittle failure of ice"
- Dempsey, J.P. et al. (1999): "Strength of ice"
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class IceMaterialProperties:
    """Ice material properties for fracture mechanics."""
    youngs_modulus: float      # Pa
    poisson_ratio: float       # dimensionless
    density: float             # kg/m^3
    fracture_toughness: float  # K_IC, Pa·m^{1/2}
    flexural_strength: float   # sigma_f, Pa
    compressive_strength: float  # sigma_c, Pa
    
    @property
    def fracture_energy(self) -> float:
        """Fracture energy G_c = K_IC^2 / E' (plane strain).
        
        E' = E / (1 - nu^2) for plane strain
        
        Returns:
            G_c: Fracture energy, J/m^2
        """
        E_prime = self.youngs_modulus / (1.0 - self.poisson_ratio**2)
        return self.fracture_toughness**2 / E_prime


def fracture_energy_from_kic(
    K_IC: float,
    E: float,
    nu: float,
) -> float:
    """Compute fracture energy G_c from fracture toughness K_IC.
    
    G_c = K_IC^2 / E'
    E' = E / (1 - nu^2)  (plane strain)
    
    Args:
        K_IC: Fracture toughness, Pa·m^{1/2}
        E: Young's modulus, Pa
        nu: Poisson's ratio, dimensionless
    
    Returns:
        G_c: Fracture energy, J/m^2
    """
    E_prime = E / (1.0 - nu**2)
    return K_IC**2 / E_prime


def crack_area_from_collision(
    R: float,
    delta_max: float,
    sigma_c: float,
    sigma_t: float,
) -> float:
    """Estimate crack area from collision geometry.
    
    For ice-ice collision, the crack area is approximated as:
    A_c = pi * a_c^2
    where a_c is the crack radius at which tensile stress exceeds strength.
    
    Hertzian contact: sigma_max = (3/2) * F_n / (pi * a^2)
    Contact radius: a = sqrt(R * delta)
    
    Crack initiates when sigma_max > sigma_t (tensile strength).
    
    Args:
        R: Effective radius, m
        delta_max: Maximum overlap, m
        sigma_c: Compressive strength, Pa
        sigma_t: Tensile strength (flexural), Pa
    
    Returns:
        A_c: Crack area, m^2
    """
    # Contact radius at maximum overlap
    a_contact = math.sqrt(R * delta_max)
    
    # Crack area is a fraction of contact area (empirical)
    # For brittle materials, crack area ~ 0.1-0.5 of contact area
    # Use 0.3 as typical value for ice
    A_c = 0.3 * math.pi * a_contact**2
    
    return A_c


@dataclass
class IceIceCollisionResult:
    """Result of ice-ice collision with fracture energy accounting."""
    time: np.ndarray              # s
    delta: np.ndarray             # overlap, m
    F_n: np.ndarray               # normal force, N
    v_rel: np.ndarray             # relative velocity, m/s
    E_kinetic: np.ndarray         # kinetic energy, J
    E_elastic: np.ndarray         # elastic potential energy, J
    E_friction: np.ndarray        # friction dissipation, J
    E_fracture: float             # fracture energy, J
    E_total: np.ndarray           # total energy, J
    fracture_occurred: bool       # whether fracture occurred
    crack_area: float             # crack area, m^2


def ice_ice_collision_energy_balance(
    R: float,
    m1: float,
    m2: float,
    v_impact: float,
    ice_props: IceMaterialProperties,
    mu: float = 0.1,
    dt: float = 1e-5,
    t_total: float = 0.01,
) -> IceIceCollisionResult:
    """Simulate ice-ice collision with energy balance ledger.
    
    Energy conservation:
    E_k(t) + E_elastic(t) + E_friction(t) + E_fracture = E_k(0)
    
    Args:
        R: Effective radius, m (1/R = 1/R1 + 1/R2)
        m1: Mass of ice floe 1, kg
        m2: Mass of ice floe 2, kg
        v_impact: Impact velocity, m/s
        ice_props: Ice material properties
        mu: Friction coefficient
        dt: Time step, s
        t_total: Total simulation time, s
    
    Returns:
        IceIceCollisionResult with energy ledger
    """
    # Effective properties
    E_star = ice_props.youngs_modulus / (2.0 * (1.0 - ice_props.poisson_ratio**2))
    m_eff = 1.0 / (1.0 / m1 + 1.0 / m2)
    
    # Time integration
    n_steps = int(t_total / dt)
    time = np.linspace(0, t_total, n_steps)
    
    delta = np.zeros(n_steps)
    F_n = np.zeros(n_steps)
    v_rel = np.zeros(n_steps)
    
    E_kinetic = np.zeros(n_steps)
    E_elastic = np.zeros(n_steps)
    E_friction = np.zeros(n_steps)
    
    # Initial conditions
    delta[0] = 0.0
    v_rel[0] = v_impact
    E_kinetic[0] = 0.5 * m_eff * v_impact**2
    E_elastic[0] = 0.0
    E_friction[0] = 0.0
    
    # Track maximum overlap for fracture calculation
    delta_max = 0.0
    
    for i in range(n_steps - 1):
        # Contact force (Hertz)
        if delta[i] > 0:
            F_n[i] = (4.0 / 3.0) * E_star * math.sqrt(R) * delta[i]**1.5
        else:
            F_n[i] = 0.0
        
        # Elastic energy
        if delta[i] > 0:
            E_elastic[i] = (8.0 / 15.0) * E_star * math.sqrt(R) * delta[i]**2.5
        
        # Acceleration
        a_rel = -F_n[i] / m_eff
        
        # Update (Verlet)
        delta[i+1] = delta[i] + v_rel[i] * dt + 0.5 * a_rel * dt**2
        v_rel[i+1] = v_rel[i] + a_rel * dt
        
        # Track maximum overlap
        if delta[i+1] > delta_max:
            delta_max = delta[i+1]
        
        # Friction dissipation (simplified: proportional to normal force)
        if delta[i] > 0 and abs(v_rel[i]) > 0:
            F_friction = mu * F_n[i]
            E_friction[i+1] = E_friction[i] + F_friction * abs(v_rel[i]) * dt
        else:
            E_friction[i+1] = E_friction[i]
        
        # Kinetic energy
        E_kinetic[i+1] = 0.5 * m_eff * v_rel[i+1]**2
    
    # Fracture energy calculation
    fracture_occurred = False
    E_fracture = 0.0
    crack_area = 0.0
    
    if delta_max > 0:
        # Check if stress exceeds tensile strength
        a_contact = math.sqrt(R * delta_max)
        F_peak = (4.0 / 3.0) * E_star * math.sqrt(R) * delta_max**1.5
        sigma_max = (3.0 / 2.0) * F_peak / (math.pi * a_contact**2)
        
        if sigma_max > ice_props.flexural_strength:
            fracture_occurred = True
            crack_area = crack_area_from_collision(
                R, delta_max, ice_props.compressive_strength, ice_props.flexural_strength
            )
            E_fracture = ice_props.fracture_energy * crack_area
    
    # Total energy
    E_total = E_kinetic + E_elastic + E_friction + E_fracture
    
    return IceIceCollisionResult(
        time=time,
        delta=delta,
        F_n=F_n,
        v_rel=v_rel,
        E_kinetic=E_kinetic,
        E_elastic=E_elastic,
        E_friction=E_friction,
        E_fracture=E_fracture,
        E_total=E_total,
        fracture_occurred=fracture_occurred,
        crack_area=crack_area,
    )


def verify_fracture_energy(
    R: float = 2.0,
    m1: float = 500.0,
    m2: float = 500.0,
    v_impact: float = 2.0,
) -> dict:
    """Verify fracture energy against Timco 2008 experimental values.
    
    Criteria:
    - E_fracture/E_k deviation from Timco 2008 ≤ 20%
    - Negative control: v=0 → E_fracture=0
    - Energy conservation: E_k(0) = E_k(t) + E_elastic(t) + E_friction(t) + E_fracture
    
    Args:
        R: Effective radius, m
        m1: Mass of floe 1, kg
        m2: Mass of floe 2, kg
        v_impact: Impact velocity, m/s
    
    Returns:
        dict with verification results
    """
    # Ice material properties (typical values)
    ice_props = IceMaterialProperties(
        youngs_modulus=5.0e9,        # 5 GPa
        poisson_ratio=0.3,
        density=917.0,
        fracture_toughness=0.1e6,    # 0.1 MPa·m^{1/2} (Timco 2008)
        flexural_strength=1.0e6,     # 1 MPa
        compressive_strength=5.0e6,  # 5 MPa
    )
    
    # Simulation
    result = ice_ice_collision_energy_balance(
        R=R,
        m1=m1,
        m2=m2,
        v_impact=v_impact,
        ice_props=ice_props,
        dt=1e-5,
        t_total=0.01,
    )
    
    # Initial kinetic energy
    m_eff = 1.0 / (1.0 / m1 + 1.0 / m2)
    E_k_initial = 0.5 * m_eff * v_impact**2
    
    # Final energies
    E_k_final = result.E_kinetic[-1]
    E_elastic_final = result.E_elastic[-1]
    E_friction_final = result.E_friction[-1]
    E_fracture = result.E_fracture
    
    # Energy conservation check
    E_total_final = E_k_final + E_elastic_final + E_friction_final + E_fracture
    energy_conservation_error = abs(E_total_final - E_k_initial) / E_k_initial
    
    # Fracture energy ratio
    E_fracture_ratio = E_fracture / E_k_initial if E_k_initial > 0 else 0.0
    
    # Timco 2008 reference: E_fracture/E_k ~ 0.1-0.3 for typical ice collisions
    # Use 0.2 as reference value
    timco_reference = 0.2
    fracture_ratio_error = abs(E_fracture_ratio - timco_reference) / timco_reference
    
    # Negative control: v=0 → E_fracture=0
    result_zero_v = ice_ice_collision_energy_balance(
        R=R,
        m1=m1,
        m2=m2,
        v_impact=0.0,
        ice_props=ice_props,
        dt=1e-5,
        t_total=0.01,
    )
    negative_control_pass = (result_zero_v.E_fracture == 0.0)
    
    return {
        "E_k_initial": E_k_initial,
        "E_k_final": E_k_final,
        "E_elastic_final": E_elastic_final,
        "E_friction_final": E_friction_final,
        "E_fracture": E_fracture,
        "E_total_final": E_total_final,
        "energy_conservation_error": energy_conservation_error,
        "energy_conservation_pass": energy_conservation_error <= 0.05,
        "E_fracture_ratio": E_fracture_ratio,
        "timco_reference": timco_reference,
        "fracture_ratio_error": fracture_ratio_error,
        "fracture_ratio_criterion_pass": fracture_ratio_error <= 0.20,
        "fracture_occurred": result.fracture_occurred,
        "crack_area": result.crack_area,
        "negative_control_pass": negative_control_pass,
        "G_c": ice_props.fracture_energy,
        "K_IC": ice_props.fracture_toughness,
    }
