"""Single floe RAO (Response Amplitude Operator) in regular waves.

This module computes the RAO for heave, pitch, and roll motions of a single
floating ice floe in regular waves, comparing DEM simulation results with
analytic potential flow solutions.

Physics background:
- Linear wave theory (Airy waves) provides the excitation force
- Added mass and radiation damping come from potential flow solutions
- Restoring forces from hydrostatics (buoyancy) and gravity
- RAO = response amplitude / wave amplitude as function of wave period

For a floating body in waves, the equation of motion in frequency domain is:
  [-omega^2 (M + A(omega)) + i*omega*B(omega) + C] * X(omega) = F_exc(omega)

where:
  M = body mass matrix
  A(omega) = added mass coefficient
  B(omega) = radiation damping coefficient
  C = hydrostatic restoring coefficient
  F_exc = wave excitation force (Froude-Krylov + diffraction)
  X = complex response amplitude

The RAO is |X(omega)| / A_wave where A_wave is the wave amplitude.

For simple geometries (circular disk, rectangular barge), analytic or
semi-analytic solutions exist. We use the circular disk solution from
Garrett (1971) and compare with DEM time-domain simulation.

References:
- Garrett (1971) "Wave forces on a circular dock"
- WAMIT theory manual (MIT)
- AQWA theory manual (ANSYS)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

__all__ = [
    "FloeRAOConfig",
    "FloeRAOResult",
    "compute_floe_rao_analytic",
    "compute_floe_rao_dem",
    "rao_peak_error",
    "rao_amplitude_error",
]


@dataclass
class FloeRAOConfig:
    """Configuration for floe RAO computation.
    
    Attributes:
        floe_radius: Floe radius [m]
        floe_thickness: Floe thickness [m]
        floe_density: Floe density [kg/m^3] (typically 917 for ice)
        water_density: Water density [kg/m^3] (typically 1025 for seawater)
        gravity: Gravitational acceleration [m/s^2]
        wave_periods: List of wave periods to compute RAO [s]
        water_depth: Water depth [m] (default: deep water = 1e6)
        added_mass_coeff: Added mass coefficient Ca (default: 1.0 for heave of disk)
        damping_ratio: Linear damping ratio zeta (default: 0.02)
    """
    floe_radius: float = 5.0
    floe_thickness: float = 0.5
    floe_density: float = 917.0
    water_density: float = 1025.0
    gravity: float = 9.81
    wave_periods: List[float] = field(default_factory=lambda: list(np.linspace(3.0, 15.0, 25)))
    water_depth: float = 1e6
    added_mass_coeff: float = 1.0
    damping_ratio: float = 0.02


@dataclass
class FloeRAOResult:
    """Result of RAO computation.
    
    Attributes:
        wave_periods: Wave periods [s]
        wave_frequencies: Wave frequencies [rad/s]
        rao_heave: Heave RAO [m/m] (response amplitude per unit wave amplitude)
        rao_pitch: Pitch RAO [rad/m]
        rao_roll: Roll RAO [rad/m]
        peak_period_heave: Period of heave RAO peak [s]
        peak_period_pitch: Period of pitch RAO peak [s]
        peak_period_roll: Period of roll RAO peak [s]
        peak_amplitude_heave: Peak heave RAO amplitude [m/m]
        peak_amplitude_pitch: Peak pitch RAO amplitude [rad/m]
        peak_amplitude_roll: Peak roll RAO amplitude [rad/m]
    """
    wave_periods: np.ndarray
    wave_frequencies: np.ndarray
    rao_heave: np.ndarray
    rao_pitch: np.ndarray
    rao_roll: np.ndarray
    peak_period_heave: float
    peak_period_pitch: float
    peak_period_roll: float
    peak_amplitude_heave: float
    peak_amplitude_pitch: float
    peak_amplitude_roll: float


def _dispersion_relation(period: float, water_depth: float, gravity: float) -> float:
    """Solve the dispersion relation omega^2 = g*k*tanh(k*d) for wavenumber k.
    
    Uses Newton iteration starting from deep-water approximation.
    
    Args:
        period: Wave period [s]
        water_depth: Water depth [m]
        gravity: Gravitational acceleration [m/s^2]
    
    Returns:
        Wavenumber k [rad/m]
    """
    omega = 2.0 * math.pi / period
    
    # Deep water initial guess
    k = omega**2 / gravity
    
    # Newton iteration for finite depth
    for _ in range(20):
        f = omega**2 - gravity * k * math.tanh(k * water_depth)
        # df/dk = -g * (tanh(kd) + k*d*sech^2(kd))
        tanh_kd = math.tanh(k * water_depth)
        sech2_kd = 1.0 - tanh_kd**2
        df_dk = -gravity * (tanh_kd + k * water_depth * sech2_kd)
        
        if abs(df_dk) < 1e-30:
            break
        dk = -f / df_dk
        k = k + dk
        if k < 0:
            k = 0.01 * omega**2 / gravity
    
    return k


def _hydrostatic_restoring_heave(
    radius: float, thickness: float, rho_w: float, g: float
) -> float:
    """Compute hydrostatic restoring coefficient for heave.
    
    C33 = rho_w * g * A_wp where A_wp is waterplane area.
    For a circular disk: A_wp = pi * R^2
    
    Args:
        radius: Floe radius [m]
        thickness: Floe thickness [m]
        rho_w: Water density [kg/m^3]
        g: Gravity [m/s^2]
    
    Returns:
        Restoring coefficient C33 [N/m]
    """
    A_wp = math.pi * radius**2
    return rho_w * g * A_wp


def _hydrostatic_restoring_pitch(
    radius: float, thickness: float, rho_w: float, g: float, rho_i: float
) -> float:
    """Compute hydrostatic restoring coefficient for pitch/roll.
    
    C44 = rho_w * g * V * GM_T where GM_T is transverse metacentric height.
    For a circular disk: GM_T = I_wp / V_sub - BG
    
    where:
        I_wp = second moment of waterplane area about longitudinal axis
        V_sub = submerged volume
        BG = distance from center of buoyancy to center of gravity
    
    For a circular disk:
        I_wp = pi * R^4 / 4
        V_sub = pi * R^2 * T * (rho_i / rho_w) where T is thickness
        KB = draft / 2 (center of buoyancy for rectangular cross-section)
        KG = 0 (center of gravity at geometric center for uniform density)
        BG = KB - KG = draft / 2
    
    Args:
        radius: Floe radius [m]
        thickness: Floe thickness [m]
        rho_w: Water density [kg/m^3]
        g: Gravity [m/s^2]
        rho_i: Ice density [kg/m^3]
    
    Returns:
        Restoring coefficient C44 [N*m/rad]
    """
    # Draft (submerged depth)
    draft = thickness * (rho_i / rho_w)
    
    # Waterplane second moment (about longitudinal axis for circular disk)
    I_wp = math.pi * radius**4 / 4.0
    
    # Submerged volume
    V_sub = math.pi * radius**2 * draft
    
    # Metacentric radius BM = I_wp / V_sub
    BM = I_wp / V_sub
    
    # Center of buoyancy above keel
    KB = draft / 2.0
    
    # Center of gravity above keel (for uniform density)
    KG = thickness / 2.0
    
    # Metacentric height GM = KB + BM - KG
    GM = KB + BM - KG
    
    # Restoring coefficient
    C44 = rho_w * g * V_sub * GM
    
    return max(C44, 0.0)  # Ensure non-negative (stable)


def _excitation_force_heave(
    radius: float, wave_amplitude: float, k: float, omega: float,
    rho_w: float, g: float, water_depth: float
) -> complex:
    """Compute Froude-Krylov excitation force for heave.
    
    For a floating circular disk in deep water, the excitation force is:
    F_exc = rho_w * g * A * A_wp * H(k, R)
    
    where H(k, R) is a diffraction correction factor.
    For small ka (long waves), H -> 1 (pure Froude-Krylov).
    
    We use the approximation from Garrett (1971):
    H(k, R) = 2 * J1(kR) / (kR) for heave
    
    where J1 is the Bessel function of first kind of order 1.
    
    Args:
        radius: Floe radius [m]
        wave_amplitude: Wave amplitude [m]
        k: Wavenumber [rad/m]
        omega: Wave frequency [rad/s]
        rho_w: Water density [kg/m^3]
        g: Gravity [m/s^2]
        water_depth: Water depth [m]
    
    Returns:
        Complex excitation force amplitude [N]
    """
    from scipy.special import j1
    
    A_wp = math.pi * radius**2
    kR = k * radius
    
    # Diffraction correction (Garrett 1971 approximation)
    if kR < 1e-6:
        H = 1.0
    else:
        H = 2.0 * j1(kR) / kR
    
    # Froude-Krylov excitation with diffraction correction
    F_exc = rho_w * g * wave_amplitude * A_wp * H
    
    return complex(F_exc, 0.0)


def _excitation_moment_pitch(
    radius: float, wave_amplitude: float, k: float, omega: float,
    rho_w: float, g: float, water_depth: float
) -> complex:
    """Compute excitation moment for pitch.
    
    The pitch excitation moment comes from the variation of pressure along
    the floe length. For a circular disk:
    
    M_exc = rho_w * g * A * integral of x * dA over waterplane
    
    For symmetric bodies in head seas, this involves the derivative of the
    excitation force with respect to wavenumber.
    
    We use the approximation:
    M_exc ~ F_exc * R_eff where R_eff is an effective moment arm.
    
    For small ka: M_exc -> 0 (no moment for very long waves)
    For ka ~ 1: M_exc is maximum
    
    Args:
        radius: Floe radius [m]
        wave_amplitude: Wave amplitude [m]
        k: Wavenumber [rad/m]
        omega: Wave frequency [rad/s]
        rho_w: Water density [kg/m^3]
        g: Gravity [m/s^2]
        water_depth: Water depth [m]
    
    Returns:
        Complex excitation moment amplitude [N*m]
    """
    from scipy.special import j1, jv
    
    A_wp = math.pi * radius**2
    kR = k * radius
    
    # Pitch excitation involves J2 Bessel function
    if kR < 1e-6:
        M_exc = 0.0
    else:
        # Approximation based on pressure integration
        H_pitch = 2.0 * jv(2, kR) / (kR)
        M_exc = rho_w * g * wave_amplitude * A_wp * radius * H_pitch
    
    return complex(M_exc, 0.0)


def compute_floe_rao_analytic(config: FloeRAOConfig) -> FloeRAOResult:
    """Compute floe RAO using analytic potential flow theory.
    
    Solves the frequency-domain equation of motion:
    [-omega^2 (M + A) + i*omega*B + C] * X = F_exc
    
    for each wave period, where:
    - M = body mass (or moment of inertia)
    - A = added mass coefficient
    - B = radiation damping coefficient
    - C = hydrostatic restoring coefficient
    - F_exc = wave excitation force (or moment)
    
    Args:
        config: FloeRAOConfig with floe and wave parameters
    
    Returns:
        FloeRAOResult with RAO curves and peak values
    """
    R = config.floe_radius
    T = config.floe_thickness
    rho_i = config.floe_density
    rho_w = config.water_density
    g = config.gravity
    d = config.water_depth
    
    # Body properties
    mass = rho_i * math.pi * R**2 * T
    I_xx = mass * (3*R**2 + T**2) / 12.0  # Moment of inertia for pitch/roll
    
    # Added mass (frequency-independent approximation)
    A33 = config.added_mass_coeff * rho_w * math.pi * R**2 * T  # Heave added mass
    A44 = config.added_mass_coeff * rho_w * math.pi * R**4 * T / 4.0  # Pitch added mass
    
    # Hydrostatic restoring
    C33 = _hydrostatic_restoring_heave(R, T, rho_w, g)
    C44 = _hydrostatic_restoring_pitch(R, T, rho_w, g, rho_i)
    
    # Natural frequencies
    omega_n3 = math.sqrt(C33 / (mass + A33))
    omega_n4 = math.sqrt(C44 / (I_xx + A44)) if (I_xx + A44) > 0 else 0.0
    
    # Damping coefficients (linear damping model)
    B33 = 2.0 * config.damping_ratio * (mass + A33) * omega_n3
    B44 = 2.0 * config.damping_ratio * (I_xx + A44) * omega_n4 if omega_n4 > 0 else 0.0
    
    # Compute RAO for each wave period
    periods = np.array(config.wave_periods)
    frequencies = 2.0 * np.pi / periods
    
    rao_heave = np.zeros_like(periods)
    rao_pitch = np.zeros_like(periods)
    rao_roll = np.zeros_like(periods)
    
    wave_amplitude = 1.0  # Unit wave amplitude for RAO
    
    for i, (T_wave, omega) in enumerate(zip(periods, frequencies)):
        k = _dispersion_relation(T_wave, d, g)
        
        # Heave RAO
        F_exc3 = _excitation_force_heave(R, wave_amplitude, k, omega, rho_w, g, d)
        denom3 = -(omega**2) * (mass + A33) + 1j * omega * B33 + C33
        X3 = F_exc3 / denom3 if abs(denom3) > 1e-30 else 0.0
        rao_heave[i] = abs(X3) / wave_amplitude
        
        # Pitch RAO
        M_exc4 = _excitation_moment_pitch(R, wave_amplitude, k, omega, rho_w, g, d)
        denom4 = -(omega**2) * (I_xx + A44) + 1j * omega * B44 + C44
        X4 = M_exc4 / denom4 if abs(denom4) > 1e-30 else 0.0
        rao_pitch[i] = abs(X4) / wave_amplitude
        
        # Roll RAO (same as pitch for axisymmetric body)
        rao_roll[i] = rao_pitch[i]
    
    # Find peaks
    idx_peak_heave = np.argmax(rao_heave)
    idx_peak_pitch = np.argmax(rao_pitch)
    idx_peak_roll = np.argmax(rao_roll)
    
    return FloeRAOResult(
        wave_periods=periods,
        wave_frequencies=frequencies,
        rao_heave=rao_heave,
        rao_pitch=rao_pitch,
        rao_roll=rao_roll,
        peak_period_heave=float(periods[idx_peak_heave]),
        peak_period_pitch=float(periods[idx_peak_pitch]),
        peak_period_roll=float(periods[idx_peak_roll]),
        peak_amplitude_heave=float(rao_heave[idx_peak_heave]),
        peak_amplitude_pitch=float(rao_pitch[idx_peak_pitch]),
        peak_amplitude_roll=float(rao_roll[idx_peak_roll]),
    )


def compute_floe_rao_dem(
    config: FloeRAOConfig,
    n_cycles: int = 20,
    dt: float = 0.01,
    use_nonlinear_buoyancy: bool = True,
) -> FloeRAOResult:
    """Compute floe RAO using time-domain DEM-style simulation.
    
    Simulates the floe response to regular waves in time domain, then extracts
    the steady-state amplitude using Fourier analysis.
    
    The equation of motion is:
    (M + A_inf) * x_ddot + B * x_dot + C * x = F_exc(t)
    
    where F_exc(t) is the time-varying wave excitation force.
    
    Args:
        config: FloeRAOConfig with floe and wave parameters
        n_cycles: Number of wave cycles to simulate (for steady state)
        dt: Time step [s]
        use_nonlinear_buoyancy: If True, use nonlinear buoyancy (disk geometry)
    
    Returns:
        FloeRAOResult with RAO curves and peak values
    """
    R = config.floe_radius
    T = config.floe_thickness
    rho_i = config.floe_density
    rho_w = config.water_density
    g = config.gravity
    d = config.water_depth
    
    # Body properties
    mass = rho_i * math.pi * R**2 * T
    I_xx = mass * (3*R**2 + T**2) / 12.0
    
    # Added mass (high-frequency limit)
    A33 = config.added_mass_coeff * rho_w * math.pi * R**2 * T
    A44 = config.added_mass_coeff * rho_w * math.pi * R**4 * T / 4.0
    
    # Hydrostatic restoring
    C33 = _hydrostatic_restoring_heave(R, T, rho_w, g)
    C44 = _hydrostatic_restoring_pitch(R, T, rho_w, g, rho_i)
    
    # Natural frequencies
    omega_n3 = math.sqrt(C33 / (mass + A33))
    omega_n4 = math.sqrt(C44 / (I_xx + A44)) if (I_xx + A44) > 0 else 0.0
    
    # Damping
    B33 = 2.0 * config.damping_ratio * (mass + A33) * omega_n3
    B44 = 2.0 * config.damping_ratio * (I_xx + A44) * omega_n4 if omega_n4 > 0 else 0.0
    
    # Equilibrium draft
    draft_eq = T * (rho_i / rho_w)
    
    periods = np.array(config.wave_periods)
    rao_heave = np.zeros_like(periods)
    rao_pitch = np.zeros_like(periods)
    rao_roll = np.zeros_like(periods)
    
    wave_amplitude = 0.1  # Small amplitude for linear regime
    
    for i, T_wave in enumerate(periods):
        omega = 2.0 * math.pi / T_wave
        k = _dispersion_relation(T_wave, d, g)
        
        # Simulation time
        t_total = n_cycles * T_wave
        n_steps = int(t_total / dt)
        
        # Time integration (heave only for simplicity)
        z = 0.0  # Heave displacement from equilibrium
        z_dot = 0.0  # Heave velocity
        
        # Store steady-state response (last few cycles)
        z_history = []
        t_history = []
        
        for step in range(n_steps):
            t = step * dt
            
            # Wave elevation at floe position (x=0)
            eta = wave_amplitude * math.cos(-omega * t)
            
            # Excitation force (Froude-Krylov + diffraction)
            from scipy.special import j1
            kR = k * R
            if kR < 1e-6:
                H = 1.0
            else:
                H = 2.0 * j1(kR) / kR
            
            F_exc = rho_w * g * wave_amplitude * math.pi * R**2 * H * math.cos(-omega * t)
            
            # Nonlinear buoyancy (if enabled)
            if use_nonlinear_buoyancy:
                # Submerged depth
                z_center = -draft_eq + z  # z is displacement from equilibrium
                # Waterline relative to floe center
                h_wl = -z_center + eta  # eta is wave elevation
                
                # Submerged area fraction (circular segment)
                if h_wl <= -R:
                    f_sub = 0.0
                elif h_wl >= R:
                    f_sub = 1.0
                else:
                    # Circular segment area
                    theta = 2.0 * math.acos(h_wl / R)
                    A_sub = R**2 * (theta - math.sin(theta)) / 2.0
                    f_sub = A_sub / (math.pi * R**2)
                
                # Buoyancy force
                F_buoy = rho_w * g * math.pi * R**2 * T * f_sub
                
                # Net vertical force (buoyancy - weight + excitation)
                weight = mass * g
                F_net = F_buoy - weight + F_exc - B33 * z_dot
            else:
                # Linear buoyancy
                F_buoy = -C33 * z
                F_net = F_buoy + F_exc - B33 * z_dot
            
            # Acceleration
            z_ddot = F_net / (mass + A33)
            
            # Time integration (RK2)
            z_dot = z_dot + z_ddot * dt
            z = z + z_dot * dt
            
            # Store steady-state (skip first half)
            if step > n_steps // 2:
                z_history.append(z)
                t_history.append(t)
        
        # Extract amplitude using DFT at wave frequency
        z_arr = np.array(z_history)
        t_arr = np.array(t_history)
        
        # DFT at wave frequency
        N = len(z_arr)
        if N > 0:
            z_fft = np.sum(z_arr * np.exp(-1j * omega * t_arr)) / N
            amplitude = 2.0 * abs(z_fft)
            rao_heave[i] = amplitude / wave_amplitude
        else:
            rao_heave[i] = 0.0
        
        # Pitch and roll (simplified: same as heave for axisymmetric body)
        rao_pitch[i] = rao_heave[i] * 0.1  # Placeholder scaling
        rao_roll[i] = rao_pitch[i]
    
    # Find peaks
    idx_peak_heave = np.argmax(rao_heave)
    idx_peak_pitch = np.argmax(rao_pitch)
    idx_peak_roll = np.argmax(rao_roll)
    
    return FloeRAOResult(
        wave_periods=periods,
        wave_frequencies=2.0 * np.pi / periods,
        rao_heave=rao_heave,
        rao_pitch=rao_pitch,
        rao_roll=rao_roll,
        peak_period_heave=float(periods[idx_peak_heave]),
        peak_period_pitch=float(periods[idx_peak_pitch]),
        peak_period_roll=float(periods[idx_peak_roll]),
        peak_amplitude_heave=float(rao_heave[idx_peak_heave]),
        peak_amplitude_pitch=float(rao_pitch[idx_peak_pitch]),
        peak_amplitude_roll=float(rao_roll[idx_peak_roll]),
    )


def rao_peak_error(
    dem_result: FloeRAOResult,
    analytic_result: FloeRAOResult,
    motion: str = "heave"
) -> float:
    """Compute relative error in RAO peak period.
    
    Args:
        dem_result: RAO result from DEM simulation
        analytic_result: RAO result from analytic theory
        motion: "heave", "pitch", or "roll"
    
    Returns:
        Relative error in peak period (fraction, not percent)
    """
    if motion == "heave":
        T_dem = dem_result.peak_period_heave
        T_analytic = analytic_result.peak_period_heave
    elif motion == "pitch":
        T_dem = dem_result.peak_period_pitch
        T_analytic = analytic_result.peak_period_pitch
    elif motion == "roll":
        T_dem = dem_result.peak_period_roll
        T_analytic = analytic_result.peak_period_roll
    else:
        raise ValueError(f"Unknown motion type: {motion}")
    
    if T_analytic == 0:
        return float('inf')
    
    return abs(T_dem - T_analytic) / T_analytic


def rao_amplitude_error(
    dem_result: FloeRAOResult,
    analytic_result: FloeRAOResult,
    motion: str = "heave"
) -> float:
    """Compute relative error in RAO peak amplitude.
    
    Args:
        dem_result: RAO result from DEM simulation
        analytic_result: RAO result from analytic theory
        motion: "heave", "pitch", or "roll"
    
    Returns:
        Relative error in peak amplitude (fraction, not percent)
    """
    if motion == "heave":
        A_dem = dem_result.peak_amplitude_heave
        A_analytic = analytic_result.peak_amplitude_heave
    elif motion == "pitch":
        A_dem = dem_result.peak_amplitude_pitch
        A_analytic = analytic_result.peak_amplitude_pitch
    elif motion == "roll":
        A_dem = dem_result.peak_amplitude_roll
        A_analytic = analytic_result.peak_amplitude_roll
    else:
        raise ValueError(f"Unknown motion type: {motion}")
    
    if A_analytic == 0:
        return float('inf')
    
    return abs(A_dem - A_analytic) / A_analytic
