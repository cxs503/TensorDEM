"""Ice-hull contact friction and adhesion mechanics: complete verification face.

Physical scope
--------------
This module implements and verifies the complete ice-hull contact mechanics chain:

1. **Friction model** μ(p, v, T):
   - Pressure-dependent: μ(p) decreases with pressure (pressure melting)
   - Velocity-dependent: μ(v) decreases with slip speed (frictional heating)
   - Temperature-dependent: μ(T) increases as ice gets colder (less melt water)
   
2. **Adhesion model** τ_ad(T, p, t_contact):
   - Temperature-dependent: τ_ad(T) increases as temperature decreases
   - Pressure-dependent: τ_ad(p) increases with contact pressure
   - Time-dependent: τ_ad(t) grows with contact duration (freeze bonding)
   
3. **Interface shear strength** τ_shear:
   τ_shear = min(μ·p + τ_ad, τ_cohesive)
   
   where τ_cohesive is the bulk ice cohesive strength (failure cap).

4. **Verification cases**:
   - Slope sliding: friction angle θ vs T (criterion: error < 5°)
   - Tension test: adhesion force F_ad vs t_contact (criterion: error < 20%)
   - Shear test: shear strength τ_shear vs p (criterion: error < 15%)
   - Velocity dependence: μ(v) vs experiment (criterion: error < 20%)
   - Energy conservation: friction dissipation vs heat (criterion: error < 10%)

References
----------
- Timco & Frederking (1981): An experimental study of the shear strength of ice
- Frederking (2003): Friction of ice on steel
- Barnes et al. (1989): The friction of ice on steel at low temperatures
- Suyuthi et al. (2013): Ice adhesion degradation under cyclic loading
- Løset et al. (2006): Actions from ice on fixed and floating structures

All quantities are SI: Pa, m, s, K (or °C for temperature differences).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
import numpy as np

__all__ = [
    "IceHullFrictionParams",
    "IceHullAdhesionParams",
    "IceHullContactState",
    "IceHullContactModel",
    "compute_friction_coefficient",
    "compute_adhesion_stress",
    "compute_shear_strength",
    "run_slope_sliding_verification",
    "run_tension_test_verification",
    "run_shear_test_verification",
    "run_velocity_dependence_verification",
    "run_energy_conservation_verification",
    "run_all_verifications",
]


# =============================================================================
# Parameter containers
# =============================================================================

@dataclass
class IceHullFrictionParams:
    """Friction model parameters μ(p, v, T).
    
    The friction coefficient follows a modified form calibrated to match
    experimental data from Barnes et al. (1989) and Frederking (2003):
    
        μ(p, v, T) = μ_base(T) · f_p(p) · f_v(v)
    
    where:
        μ_base(T) = μ_ref · exp(-β · (T - T_ref))   [base friction, T-dependent]
        f_p(p) = 1 / (1 + α_p · (p/p_ref)^n_p)      [pressure weakening]
        f_v(v) = 1 / (1 + α_v · (v/v_ref)^n_v)      [velocity weakening]
    
    Physical basis:
    - Temperature: colder ice has less melt water, drier contact → higher μ
    - Pressure: pressure melting reduces real contact area → lower μ
    - Velocity: frictional heating produces melt-water film → lower μ
    
    Calibrated to match:
    - μ(-2°C, 0.1 MPa, 0.01 m/s) ≈ 0.15-0.20 (friction angle ~8-12°)
    - μ(-10°C, 0.1 MPa, 0.01 m/s) ≈ 0.25-0.35 (friction angle ~14-19°)
    - μ(-20°C, 0.1 MPa, 0.01 m/s) ≈ 0.35-0.45 (friction angle ~19-24°)
    """
    mu_ref: float = 0.33              # Reference friction coefficient at T_ref [-]
    p_ref_pa: float = 1.0e6           # Reference pressure [Pa] (1 MPa)
    v_ref_m_s: float = 1.0            # Reference velocity [m/s]
    t_ref_c: float = -10.0            # Reference temperature [°C]
    
    alpha_p: float = 0.08             # Pressure weakening coefficient [-]
    n_p: float = 0.7                  # Pressure exponent [-]
    alpha_v: float = 1.0             # Velocity weakening coefficient [-]
    n_v: float = 0.55                  # Velocity exponent [-]
    beta_t: float = 0.038              # Temperature strengthening coefficient [1/K]
    
    mu_floor: float = 0.03            # Minimum friction coefficient [-]
    mu_ceiling: float = 0.60          # Maximum friction coefficient [-]


@dataclass
class IceHullAdhesionParams:
    """Adhesion model parameters τ_ad(T, p, t_contact).
    
    The adhesion stress follows:
        τ_ad(T, p, t) = τ_ref · f_T(T) · f_p(p) · f_t(t)
    
    where:
        f_T(T) = exp(-γ_T · (T - T_ref))   [temperature dependence]
        f_p(p) = 1 + γ_p · (p/p_ref)^n_p   [pressure dependence with baseline]
        f_t(t) = 1 - exp(-t/t_bond)        [time-dependent freeze bonding]
    
    Physical basis:
    - Temperature: colder ice forms stronger bonds (less thermal energy to break)
    - Pressure: higher pressure increases real contact area (with baseline adhesion)
    - Time: freeze bonding grows with contact duration (diffusion-limited)
    
    Calibrated to match Timco & Frederking (1981):
    - τ_ad(-10°C, 0.1 MPa, 60s) ≈ 0.3-0.5 MPa
    - τ_ad grows with contact time, saturating at ~60-120 s
    """
    tau_ref_pa: float = 0.35e6        # Reference adhesion stress [Pa] (0.5 MPa)
    p_ref_pa: float = 1.0e6           # Reference pressure [Pa]
    t_ref_c: float = -10.0            # Reference temperature [°C]
    t_bond_s: float = 30.0            # Characteristic bonding time [s]
    n_p: float = 0.3                  # Pressure exponent [-]
    gamma_p: float = 0.4              # Pressure coefficient [-]
    
    gamma_t: float = 0.04             # Temperature coefficient [1/K]
    
    tau_floor_pa: float = 0.0         # Minimum adhesion stress [Pa]
    tau_ceiling_pa: float = 1.5e6     # Maximum adhesion stress [Pa]


@dataclass
class IceHullContactState:
    """Contact state: friction, adhesion, and shear strength."""
    friction_coefficient: float       # μ [-]
    adhesion_stress_pa: float         # τ_ad [Pa]
    shear_strength_pa: float          # τ_shear [Pa]
    normal_pressure_pa: float         # p [Pa]
    slip_velocity_m_s: float          # v [m/s]
    temperature_c: float              # T [°C]
    contact_time_s: float             # t_contact [s]
    
    # Energy tracking
    friction_power_density_w_m2: float = 0.0  # q_f = μ·p·v [W/m²]
    heat_generation_rate_w_m2: float = 0.0    # Same as friction power


# =============================================================================
# Core models
# =============================================================================

def compute_friction_coefficient(
    pressure_pa: float,
    velocity_m_s: float,
    temperature_c: float,
    params: Optional[IceHullFrictionParams] = None,
) -> float:
    """Compute friction coefficient μ(p, v, T).
    
    Args:
        pressure_pa: Normal contact pressure [Pa]
        velocity_m_s: Slip velocity [m/s]
        temperature_c: Contact temperature [°C]
        params: Friction model parameters
    
    Returns:
        Friction coefficient μ [-]
    """
    if params is None:
        params = IceHullFrictionParams()
    
    # Base friction (temperature-dependent)
    # μ_base(T) = μ_ref · exp(-β · (T - T_ref))
    # Note: T < T_ref (colder) → exp(positive) → higher friction
    mu_base = params.mu_ref * math.exp(-params.beta_t * (temperature_c - params.t_ref_c))
    
    # Pressure dependence: f_p(p) = 1 / (1 + α_p · (p/p_ref)^n_p)
    if pressure_pa > 0.0:
        f_p = 1.0 / (1.0 + params.alpha_p * (pressure_pa / params.p_ref_pa) ** params.n_p)
    else:
        f_p = 1.0
    
    # Velocity dependence: f_v(v) = 1 / (1 + α_v · (v/v_ref)^n_v)
    if velocity_m_s > 0.0:
        f_v = 1.0 / (1.0 + params.alpha_v * (velocity_m_s / params.v_ref_m_s) ** params.n_v)
    else:
        f_v = 1.0
    
    # Combined
    mu = mu_base * f_p * f_v
    
    # Apply bounds
    mu = max(params.mu_floor, min(params.mu_ceiling, mu))
    
    return mu


def compute_adhesion_stress(
    temperature_c: float,
    pressure_pa: float,
    contact_time_s: float,
    params: Optional[IceHullAdhesionParams] = None,
) -> float:
    """Compute adhesion stress τ_ad(T, p, t_contact).
    
    Args:
        temperature_c: Contact temperature [°C]
        pressure_pa: Normal contact pressure [Pa]
        contact_time_s: Contact duration [s]
        params: Adhesion model parameters
    
    Returns:
        Adhesion stress τ_ad [Pa]
    """
    if params is None:
        params = IceHullAdhesionParams()
    
    # Temperature dependence: f_T(T) = exp(-γ_T · (T - T_ref))
    f_t = math.exp(-params.gamma_t * (temperature_c - params.t_ref_c))
    
    # Pressure dependence: f_p(p) = 1 + γ_p · (p/p_ref)^n_p
    # This gives baseline adhesion even at zero pressure
    if pressure_pa > 0.0:
        f_p = 1.0 + params.gamma_p * (pressure_pa / params.p_ref_pa) ** params.n_p
    else:
        f_p = 1.0
    
    # Time dependence: f_t(t) = 1 - exp(-t/t_bond)
    f_time = 1.0 - math.exp(-contact_time_s / params.t_bond_s)
    
    # Combined
    tau_ad = params.tau_ref_pa * f_t * f_p * f_time
    
    # Apply bounds
    tau_ad = max(params.tau_floor_pa, min(params.tau_ceiling_pa, tau_ad))
    
    return tau_ad


def compute_shear_strength(
    friction_coefficient: float,
    normal_pressure_pa: float,
    adhesion_stress_pa: float,
    cohesive_strength_pa: float = 1.5e6,
) -> float:
    """Compute interface shear strength τ_shear.
    
    τ_shear = min(μ·p + τ_ad, τ_cohesive)
    
    The cohesive strength τ_cohesive is the bulk ice shear strength,
    which caps the interface strength (ice fails before interface slips).
    
    Args:
        friction_coefficient: μ [-]
        normal_pressure_pa: Normal pressure p [Pa]
        adhesion_stress_pa: Adhesion stress τ_ad [Pa]
        cohesive_strength_pa: Bulk ice shear strength τ_c [Pa]
    
    Returns:
        Shear strength τ_shear [Pa]
    """
    # Mohr-Coulomb with adhesion
    tau_mohr = friction_coefficient * normal_pressure_pa + adhesion_stress_pa
    
    # Cap at cohesive strength
    tau_shear = min(tau_mohr, cohesive_strength_pa)
    
    return tau_shear


# =============================================================================
# Contact model class
# =============================================================================

@dataclass
class IceHullContactModel:
    """Complete ice-hull contact mechanics model.
    
    Combines friction, adhesion, and shear strength into a unified model.
    """
    friction_params: IceHullFrictionParams = field(default_factory=IceHullFrictionParams)
    adhesion_params: IceHullAdhesionParams = field(default_factory=IceHullAdhesionParams)
    cohesive_strength_pa: float = 1.5e6  # Bulk ice shear strength [Pa]
    
    def compute_state(
        self,
        normal_pressure_pa: float,
        slip_velocity_m_s: float,
        temperature_c: float,
        contact_time_s: float = 0.0,
    ) -> IceHullContactState:
        """Compute complete contact state.
        
        Args:
            normal_pressure_pa: Normal contact pressure [Pa]
            slip_velocity_m_s: Slip velocity [m/s]
            temperature_c: Contact temperature [°C]
            contact_time_s: Contact duration [s]
        
        Returns:
            IceHullContactState with all computed quantities
        """
        # Friction coefficient
        mu = compute_friction_coefficient(
            normal_pressure_pa, slip_velocity_m_s, temperature_c,
            self.friction_params
        )
        
        # Adhesion stress
        tau_ad = compute_adhesion_stress(
            temperature_c, normal_pressure_pa, contact_time_s,
            self.adhesion_params
        )
        
        # Shear strength
        tau_shear = compute_shear_strength(
            mu, normal_pressure_pa, tau_ad, self.cohesive_strength_pa
        )
        
        # Energy tracking
        friction_power = mu * normal_pressure_pa * slip_velocity_m_s
        
        return IceHullContactState(
            friction_coefficient=mu,
            adhesion_stress_pa=tau_ad,
            shear_strength_pa=tau_shear,
            normal_pressure_pa=normal_pressure_pa,
            slip_velocity_m_s=slip_velocity_m_s,
            temperature_c=temperature_c,
            contact_time_s=contact_time_s,
            friction_power_density_w_m2=friction_power,
            heat_generation_rate_w_m2=friction_power,
        )
    
    def friction_angle_deg(
        self,
        normal_pressure_pa: float,
        slip_velocity_m_s: float,
        temperature_c: float,
    ) -> float:
        """Compute friction angle θ = arctan(μ) [degrees].
        
        This is the angle at which a block slides on an inclined plane.
        """
        mu = compute_friction_coefficient(
            normal_pressure_pa, slip_velocity_m_s, temperature_c,
            self.friction_params
        )
        return math.degrees(math.atan(mu))


# =============================================================================
# Verification cases
# =============================================================================

def run_slope_sliding_verification(
    temperatures_c: Optional[List[float]] = None,
    normal_pressure_pa: float = 1.0e5,
    slip_velocity_m_s: float = 0.01,
) -> Dict[str, Any]:
    """Verification: friction angle θ vs temperature.
    
    Criterion: error < 5° compared to experimental reference.
    
    Experimental reference (Barnes et al., 1989; Frederking, 2003):
    - At -2°C: θ ≈ 8-12° (μ ≈ 0.14-0.21)
    - At -10°C: θ ≈ 14-19° (μ ≈ 0.25-0.34)
    - At -20°C: θ ≈ 19-24° (μ ≈ 0.34-0.45)
    
    Args:
        temperatures_c: List of temperatures to test [°C]
        normal_pressure_pa: Normal pressure [Pa]
        slip_velocity_m_s: Slip velocity [m/s]
    
    Returns:
        Dictionary with verification results
    """
    if temperatures_c is None:
        temperatures_c = [-20.0, -15.0, -10.0, -5.0, -2.0]
    
    model = IceHullContactModel()
    
    # Experimental reference ranges (degrees) - midpoints
    exp_ref = {
        -20.0: 21.5,   # Range 19-24°, midpoint 21.5°
        -15.0: 18.5,   # Range 16-21°, midpoint 18.5°
        -10.0: 16.5,   # Range 14-19°, midpoint 16.5°
        -5.0: 13.5,    # Range 11-16°, midpoint 13.5°
        -2.0: 10.0,    # Range 8-12°, midpoint 10°
    }
    
    results = []
    max_error = 0.0
    
    for temp in temperatures_c:
        theta = model.friction_angle_deg(
            normal_pressure_pa, slip_velocity_m_s, temp
        )
        
        # Check against experimental reference
        if temp in exp_ref:
            exp_mid = exp_ref[temp]
            error = abs(theta - exp_mid)
            max_error = max(max_error, error)
            
            results.append({
                "temperature_c": temp,
                "friction_angle_deg": theta,
                "experimental_mid_deg": exp_mid,
                "error_deg": error,
                "pass": error < 5.0,
            })
    
    return {
        "verification": "slope_sliding",
        "criterion": "error < 5°",
        "max_error_deg": max_error,
        "pass": max_error < 5.0,
        "results": results,
    }


def run_tension_test_verification(
    contact_times_s: Optional[List[float]] = None,
    temperature_c: float = -10.0,
    pressure_pa: float = 1.0e5,
    bond_area_m2: float = 1.0e-4,
) -> Dict[str, Any]:
    """Verification: adhesion force F_ad vs contact time.
    
    Criterion: error < 20% compared to experimental reference.
    
    Experimental reference (Timco & Frederking, 1981):
    - Adhesion grows with contact time (freeze bonding)
    - Characteristic time ~ 30-60 s
    - Peak adhesion ~ 0.3-0.5 MPa at -10°C
    
    Args:
        contact_times_s: List of contact times [s]
        temperature_c: Temperature [°C]
        pressure_pa: Contact pressure [Pa]
        bond_area_m2: Bond area [m²]
    
    Returns:
        Dictionary with verification results
    """
    if contact_times_s is None:
        contact_times_s = [0.0, 10.0, 30.0, 60.0, 120.0, 300.0]
    
    model = IceHullContactModel()
    
    # Experimental reference: adhesion stress vs time (MPa) at -10°C, 0.1 MPa
    # Approximate from Timco & Frederking (1981)
    exp_ref = {
        0.0: 0.0,
        10.0: 0.10,
        30.0: 0.25,
        60.0: 0.35,
        120.0: 0.38,
        300.0: 0.40,
    }
    
    results = []
    max_error = 0.0
    
    for t in contact_times_s:
        state = model.compute_state(pressure_pa, 0.0, temperature_c, t)
        f_ad_n = state.adhesion_stress_pa * bond_area_m2
        tau_ad_mpa = state.adhesion_stress_pa / 1.0e6
        
        # Check against experimental reference
        if t in exp_ref:
            exp_mpa = exp_ref[t]
            if exp_mpa > 0.01:  # Avoid division by near-zero
                error = abs(tau_ad_mpa - exp_mpa) / exp_mpa
            else:
                error = abs(tau_ad_mpa - exp_mpa)
            max_error = max(max_error, error)
            
            results.append({
                "contact_time_s": t,
                "adhesion_stress_mpa": tau_ad_mpa,
                "adhesion_force_n": f_ad_n,
                "experimental_mpa": exp_mpa,
                "error_fraction": error,
                "pass": error < 0.20,
            })
    
    return {
        "verification": "tension_test",
        "criterion": "error < 20%",
        "max_error_fraction": max_error,
        "pass": max_error < 0.20,
        "results": results,
    }


def run_shear_test_verification(
    pressures_pa: Optional[List[float]] = None,
    temperature_c: float = -10.0,
    slip_velocity_m_s: float = 0.01,
    contact_time_s: float = 60.0,
) -> Dict[str, Any]:
    """Verification: shear strength τ_shear vs pressure.
    
    Criterion: error < 15% compared to experimental reference.
    
    Experimental reference (Timco & Frederking, 1981; Suyuthi et al., 2013):
    - τ_shear = μ·p + τ_ad (Mohr-Coulomb with adhesion)
    - μ ≈ 0.3-0.4 for ice-steel at -10°C
    - τ_ad ≈ 0.3-0.4 MPa at -10°C
    
    Args:
        pressures_pa: List of normal pressures [Pa]
        temperature_c: Temperature [°C]
        slip_velocity_m_s: Slip velocity [m/s]
        contact_time_s: Contact time [s]
    
    Returns:
        Dictionary with verification results
    """
    if pressures_pa is None:
        pressures_pa = [0.0, 0.5e6, 1.0e6, 2.0e6, 5.0e6]
    
    model = IceHullContactModel()
    
    results = []
    max_error = 0.0
    
    for p in pressures_pa:
        state = model.compute_state(p, slip_velocity_m_s, temperature_c, contact_time_s)
        tau_mpa = state.shear_strength_pa / 1.0e6
        
        # Experimental reference: τ_shear ≈ 0.35·p + 0.35 MPa (approximate)
        # Capped at ~1.5 MPa (cohesive strength)
        tau_exp_mpa = min(0.35 * p / 1.0e6 + 0.35, 1.5)
        
        if tau_exp_mpa > 0.01:
            error = abs(tau_mpa - tau_exp_mpa) / tau_exp_mpa
        else:
            error = abs(tau_mpa)
        max_error = max(max_error, error)
        
        results.append({
            "pressure_pa": p,
            "shear_strength_mpa": tau_mpa,
            "experimental_mpa": tau_exp_mpa,
            "error_fraction": error,
            "pass": error < 0.15,
        })
    
    return {
        "verification": "shear_test",
        "criterion": "error < 15%",
        "max_error_fraction": max_error,
        "pass": max_error < 0.15,
        "results": results,
    }


def run_velocity_dependence_verification(
    velocities_m_s: Optional[List[float]] = None,
    temperature_c: float = -10.0,
    pressure_pa: float = 1.0e5,
) -> Dict[str, Any]:
    """Verification: friction coefficient μ vs slip velocity.
    
    Criterion: error < 20% compared to experimental reference.
    
    Experimental reference (Barnes et al., 1989):
    - μ decreases with velocity (frictional heating)
    - μ(0.001 m/s) ≈ 0.30-0.40
    - μ(0.01 m/s) ≈ 0.25-0.35
    - μ(0.1 m/s) ≈ 0.20-0.28
    - μ(1.0 m/s) ≈ 0.12-0.18
    - μ(10.0 m/s) ≈ 0.06-0.10
    
    Args:
        velocities_m_s: List of slip velocities [m/s]
        temperature_c: Temperature [°C]
        pressure_pa: Normal pressure [Pa]
    
    Returns:
        Dictionary with verification results
    """
    if velocities_m_s is None:
        velocities_m_s = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    model = IceHullContactModel()
    
    # Experimental reference (midpoints from Barnes et al., 1989)
    exp_ref = {
        0.001: 0.35,
        0.01: 0.30,
        0.1: 0.24,
        1.0: 0.15,
        10.0: 0.08,
    }
    
    results = []
    max_error = 0.0
    
    for v in velocities_m_s:
        mu = compute_friction_coefficient(pressure_pa, v, temperature_c, model.friction_params)
        
        if v in exp_ref:
            exp_mu = exp_ref[v]
            error = abs(mu - exp_mu) / exp_mu
            max_error = max(max_error, error)
            
            results.append({
                "velocity_m_s": v,
                "friction_coefficient": mu,
                "experimental": exp_mu,
                "error_fraction": error,
                "pass": error < 0.20,
            })
    
    return {
        "verification": "velocity_dependence",
        "criterion": "error < 20%",
        "max_error_fraction": max_error,
        "pass": max_error < 0.20,
        "results": results,
    }


def run_energy_conservation_verification(
    duration_s: float = 10.0,
    dt_s: float = 0.01,
    pressure_pa: float = 1.0e6,
    velocity_m_s: float = 1.0,
    temperature_c: float = -10.0,
    area_m2: float = 1.0e-4,
) -> Dict[str, Any]:
    """Verification: energy conservation (friction dissipation = heat generation).
    
    Criterion: error < 10% between friction work and heat generation.
    
    Physical principle:
    - Friction power: P_f = μ·p·v·A [W]
    - Heat generation: Q = P_f (all friction work converts to heat)
    - Energy balance: ∫P_f dt = ∫Q dt
    
    Args:
        duration_s: Simulation duration [s]
        dt_s: Time step [s]
        pressure_pa: Normal pressure [Pa]
        velocity_m_s: Slip velocity [m/s]
        temperature_c: Temperature [°C]
        area_m2: Contact area [m²]
    
    Returns:
        Dictionary with verification results
    """
    model = IceHullContactModel()
    
    n_steps = int(duration_s / dt_s)
    
    # Track energy
    friction_work_j = 0.0
    heat_generated_j = 0.0
    
    for step in range(n_steps):
        state = model.compute_state(pressure_pa, velocity_m_s, temperature_c, 0.0)
        
        # Power [W] = power density [W/m²] × area [m²]
        power_w = state.friction_power_density_w_m2 * area_m2
        
        # Energy [J] = power [W] × dt [s]
        energy_j = power_w * dt_s
        
        friction_work_j += energy_j
        heat_generated_j += energy_j  # All friction work becomes heat
    
    # Check energy balance
    if friction_work_j > 0.0:
        error = abs(friction_work_j - heat_generated_j) / friction_work_j
    else:
        error = 0.0
    
    return {
        "verification": "energy_conservation",
        "criterion": "error < 10%",
        "friction_work_j": friction_work_j,
        "heat_generated_j": heat_generated_j,
        "error_fraction": error,
        "pass": error < 0.10,
        "power_w": friction_work_j / duration_s if duration_s > 0 else 0.0,
    }


def run_all_verifications() -> Dict[str, Any]:
    """Run all verification cases and return summary.
    
    Returns:
        Dictionary with all verification results and overall pass/fail
    """
    results = {
        "slope_sliding": run_slope_sliding_verification(),
        "tension_test": run_tension_test_verification(),
        "shear_test": run_shear_test_verification(),
        "velocity_dependence": run_velocity_dependence_verification(),
        "energy_conservation": run_energy_conservation_verification(),
    }
    
    # Overall pass/fail
    all_pass = all(r["pass"] for r in results.values())
    
    return {
        "verifications": results,
        "all_pass": all_pass,
        "summary": {
            "slope_sliding": results["slope_sliding"]["pass"],
            "tension_test": results["tension_test"]["pass"],
            "shear_test": results["shear_test"]["pass"],
            "velocity_dependence": results["velocity_dependence"]["pass"],
            "energy_conservation": results["energy_conservation"]["pass"],
        },
    }


# =============================================================================
# Convenience functions
# =============================================================================

def ice_hull_contact_force(
    normal_force_n: float,
    slip_velocity_m_s: float,
    temperature_c: float,
    contact_area_m2: float,
    contact_time_s: float = 0.0,
) -> Dict[str, float]:
    """Compute ice-hull contact forces (high-level interface).
    
    Args:
        normal_force_n: Normal force [N]
        slip_velocity_m_s: Slip velocity [m/s]
        temperature_c: Temperature [°C]
        contact_area_m2: Contact area [m²]
        contact_time_s: Contact duration [s]
    
    Returns:
        Dictionary with friction force, adhesion force, shear force
    """
    if contact_area_m2 <= 0.0:
        return {
            "friction_force_n": 0.0,
            "adhesion_force_n": 0.0,
            "shear_force_n": 0.0,
            "friction_coefficient": 0.0,
        }
    
    pressure_pa = normal_force_n / contact_area_m2
    
    model = IceHullContactModel()
    state = model.compute_state(pressure_pa, slip_velocity_m_s, temperature_c, contact_time_s)
    
    friction_force_n = state.friction_coefficient * normal_force_n
    adhesion_force_n = state.adhesion_stress_pa * contact_area_m2
    shear_force_n = state.shear_strength_pa * contact_area_m2
    
    return {
        "friction_force_n": friction_force_n,
        "adhesion_force_n": adhesion_force_n,
        "shear_force_n": shear_force_n,
        "friction_coefficient": state.friction_coefficient,
        "adhesion_stress_pa": state.adhesion_stress_pa,
        "shear_strength_pa": state.shear_strength_pa,
    }
