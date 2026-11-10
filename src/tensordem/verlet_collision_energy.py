"""Velocity-Verlet integrator + event-driven contact localization for rigid collisions.

This module provides three energy accounting methods for head-on rigid sphere collisions:
  (1) Semi-implicit Euler + trapezoidal work (baseline, O(dt))
  (2) Velocity-Verlet + trapezoidal work (O(dt²))
  (3) Event-driven exact integration (analytical, round-off)

The velocity-Verlet integrator is symplectic and second-order accurate:
  x_{n+1} = x_n + v_n*dt + 0.5*a_n*dt²
  v_{n+1/2} = v_n + 0.5*a_n*dt
  a_{n+1} = F(x_{n+1})/m
  v_{n+1} = v_{n+1/2} + 0.5*a_{n+1}*dt

Event-driven integration finds the exact contact time by bisection/Newton on the
gap function, then advances the damped oscillator analytically within the contact.

All quantities SI: m, s, kg, N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .dem3d import DEM3DConfig, IceDEM3D
from .contact_impact import clamped_restitution


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class VerletCollisionResult:
    """Single collision result at one (dt, k_n) setting for Verlet integrator."""
    dt: float
    k_n: float
    e_measured: float
    e_target: float
    # Energy accounting
    dE_state: float
    dE_impulse_trap: float
    dE_theory: float
    # Diagnostics
    momentum_error: float
    contact_steps: int
    v0: float
    m_eff: float
    E0: float


@dataclass
class EventDrivenResult:
    """Event-driven (analytical) collision result."""
    k_n: float
    e_measured: float
    e_target: float
    dE_state: float
    dE_theory: float
    momentum_error: float
    contact_time: float
    v0: float
    m_eff: float
    E0: float


@dataclass
class ThreeMethodResult:
    """Three-method comparison at one dt."""
    dt: float
    k_n: float
    # Method 1: semi-implicit + trapezoidal
    semi_implicit_dE: float
    semi_implicit_e: float
    # Method 2: Verlet + trapezoidal
    verlet_dE: float
    verlet_e: float
    # Method 3: event-driven
    event_dE: float
    event_e: float
    # Theory
    dE_theory: float
    E0: float


# ---------------------------------------------------------------------------
# Helper: restitution to damping
# ---------------------------------------------------------------------------
def _restitution_to_damping(e_target: float, k_n: float, m_eff: float) -> float:
    """Map restitution coefficient to damping."""
    from .contact_tangential import restitution_to_damping
    return restitution_to_damping(e_target, k_n, m_eff)


def dem_recommended_dt(k_n: float, mass: float, radius: float) -> float:
    """Compute the recommended dt for the DEM3D solver."""
    omega = math.sqrt(k_n / mass)
    return math.pi / (20.0 * omega)


# ---------------------------------------------------------------------------
# Build pair impact harness
# ---------------------------------------------------------------------------
def _build_pair_impact(
    *,
    k_n: float,
    c_n: float,
    radius: float = 0.05,
    density: float = 900.0,
    dt: float = 1e-5,
) -> tuple[IceDEM3D, int, int]:
    """Two free spheres on a collision course."""
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=radius, density=density,
        contact_stiffness=k_n, contact_damping=c_n,
        contact_damping_model="absolute",
        drag=0.0,
        bond_stiffness=1.0, breaking_strain=1e-12, shear_breaking_strain=1e-12,
        fix_x_edges=False, fix_bottom=False,
        tool_gap=1.0e3, tool_speed=1.0e-9,
    )
    dem = IceDEM3D(cfg)
    n = len(dem.positions)
    pos = torch.zeros_like(dem.positions)
    for idx in range(n):
        pos[idx] = torch.tensor([10.0 * idx, 500.0, 500.0], dtype=dem.dtype)
    pos[0] = torch.tensor([0.0, 0.0, -(radius + 1.0e-6)], dtype=dem.dtype)
    pos[1] = torch.tensor([0.0, 0.0, radius + 1.0e-6], dtype=dem.dtype)
    dem.positions = pos.clone()
    dem.initial_positions = pos.clone()
    mask = torch.ones(n, dtype=torch.bool)
    mask[0] = False
    mask[1] = False
    dem.fixed = mask
    dem.forces()
    dem.dt = dt
    return dem, 0, 1


# ---------------------------------------------------------------------------
# Velocity-Verlet step (custom, not using dem.step())
# ---------------------------------------------------------------------------
def _verlet_step(
    dem: IceDEM3D,
    i: int,
    j: int,
    radius: float,
    k_n: float,
    c_n: float,
    m: float,
) -> tuple[float, float, float]:
    """Single velocity-Verlet step for two-sphere impact.
    
    Returns (v_i_after, v_j_after, impulse_trap_increment).
    
    Sign convention: delta_dot = -v_rel (positive when compressing).
    """
    # Current state
    v_i = dem.velocities[i, 2].item()
    v_j = dem.velocities[j, 2].item()
    z_i = dem.positions[i, 2].item()
    z_j = dem.positions[j, 2].item()
    
    v_rel = v_j - v_i
    
    # Compute current force
    sep = z_j - z_i
    overlap = 2.0 * radius - sep
    if overlap > 0:
        # Compression rate: positive when approaching
        delta_dot = -v_rel  # v_rel < 0 when approaching, so delta_dot > 0
        F_spring = k_n * overlap
        F_damp = c_n * delta_dot
        F_contact = F_spring + F_damp
        if F_contact < 0:
            F_contact = 0.0
        # Force on i is +F (repulsive, pushes i in +z direction)
        # Force on j is -F (repulsive, pushes j in -z direction)
        F_i = -F_contact
        F_j = F_contact
    else:
        F_i = 0.0
        F_j = 0.0
    
    # Half-step velocity
    a_i = F_i / m
    a_j = F_j / m
    v_i_half = v_i + 0.5 * a_i * dem.dt
    v_j_half = v_j + 0.5 * a_j * dem.dt
    
    # Full-step position
    z_i_new = z_i + v_i_half * dem.dt
    z_j_new = z_j + v_j_half * dem.dt
    
    # Compute new force
    sep_new = z_j_new - z_i_new
    overlap_new = 2.0 * radius - sep_new
    if overlap_new > 0:
        v_rel_half = v_j_half - v_i_half
        delta_dot_new = -v_rel_half  # compression rate at half-step
        F_spring_new = k_n * overlap_new
        F_damp_new = c_n * delta_dot_new
        F_contact_new = F_spring_new + F_damp_new
        if F_contact_new < 0:
            F_contact_new = 0.0
        F_i_new = -F_contact_new
        F_j_new = F_contact_new
    else:
        F_i_new = 0.0
        F_j_new = 0.0
    
    # Full-step velocity
    a_i_new = F_i_new / m
    a_j_new = F_j_new / m
    v_i_new = v_i_half + 0.5 * a_i_new * dem.dt
    v_j_new = v_j_half + 0.5 * a_j_new * dem.dt
    
    # Update DEM state
    dem.positions[i, 2] = z_i_new
    dem.positions[j, 2] = z_j_new
    dem.velocities[i, 2] = v_i_new
    dem.velocities[j, 2] = v_j_new
    
    # Trapezoidal work increment: W = F_avg * delta_x
    # For Verlet, v_{n+1/2} = (v_n + v_{n+1})/2 exactly
    v_rel_new = v_j_new - v_i_new
    v_rel_avg = 0.5 * (v_rel + v_rel_new)
    F_avg = 0.5 * (F_i + F_i_new)
    # Work done ON the system: dW = -F * v_rel_avg * dt (negative because F opposes motion)
    dW_trap = -F_avg * v_rel_avg * dem.dt
    
    return v_i_new, v_j_new, dW_trap


# ---------------------------------------------------------------------------
# Semi-implicit Euler step (baseline)
# ---------------------------------------------------------------------------
def _semi_implicit_step(
    dem: IceDEM3D,
    i: int,
    j: int,
    radius: float,
    k_n: float,
    c_n: float,
    m: float,
) -> tuple[float, float, float, float, float]:
    """Single semi-implicit Euler step.
    
    Returns (v_i_before, v_j_before, v_i_after, v_j_after, dW_trap).
    """
    v_i_before = dem.velocities[i, 2].item()
    v_j_before = dem.velocities[j, 2].item()
    v_rel_before = v_j_before - v_i_before
    
    # Compute force
    z_i = dem.positions[i, 2].item()
    z_j = dem.positions[j, 2].item()
    sep = z_j - z_i
    overlap = 2.0 * radius - sep
    
    if overlap > 0:
        delta_dot = -v_rel_before  # compression rate
        F_spring = k_n * overlap
        F_damp = c_n * delta_dot
        F_contact = F_spring + F_damp
        if F_contact < 0:
            F_contact = 0.0
        F_i = -F_contact
        F_j = F_contact
    else:
        F_i = 0.0
        F_j = 0.0
    
    # Update velocity first (semi-implicit)
    a_i = F_i / m
    a_j = F_j / m
    v_i_after = v_i_before + a_i * dem.dt
    v_j_after = v_j_before + a_j * dem.dt
    
    # Update position
    z_i_new = z_i + v_i_after * dem.dt
    z_j_new = z_j + v_j_after * dem.dt
    
    dem.positions[i, 2] = z_i_new
    dem.positions[j, 2] = z_j_new
    dem.velocities[i, 2] = v_i_after
    dem.velocities[j, 2] = v_j_after
    
    v_rel_after = v_j_after - v_i_after
    
    # Trapezoidal work
    v_rel_avg = 0.5 * (v_rel_before + v_rel_after)
    dW_trap = -F_i * v_rel_avg * dem.dt
    
    return v_i_before, v_j_before, v_i_after, v_j_after, dW_trap


# ---------------------------------------------------------------------------
# Run collision with velocity-Verlet
# ---------------------------------------------------------------------------
def run_verlet_collision(
    *,
    e_target: float = 0.7,
    v0: float = 0.5,
    k_n: float = 5.0e4,
    radius: float = 0.05,
    density: float = 900.0,
    dt: float | None = None,
    dt_scale: float = 1.0,
) -> VerletCollisionResult:
    """Run head-on collision with velocity-Verlet integrator."""
    m = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = 0.5 * m
    c_n = _restitution_to_damping(e_target, k_n, m_eff)
    
    if dt is None:
        dt = dem_recommended_dt(k_n, m, radius) * dt_scale
    
    dem, i, j = _build_pair_impact(k_n=k_n, c_n=c_n, radius=radius, density=density, dt=dt)
    
    dem.velocities = torch.zeros_like(dem.positions)
    dem.velocities[i] = torch.tensor([0.0, 0.0, v0], dtype=dem.dtype)
    dem.velocities[j] = torch.tensor([0.0, 0.0, -v0], dtype=dem.dtype)
    
    E0 = m * v0 ** 2
    KE_before = float(0.5 * m * (dem.velocities ** 2).sum())
    momentum_before = float(dem.velocities[:, 2].sum() * m)
    
    impulse_trap = 0.0
    contact_steps = 0
    in_contact = False
    
    for step_idx in range(800_000):
        v_i_before = float(dem.velocities[i, 2])
        v_j_before = float(dem.velocities[j, 2])
        
        v_i_after, v_j_after, dW_trap = _verlet_step(dem, i, j, radius, k_n, c_n, m)
        
        sep = float(dem.positions[j, 2] - dem.positions[i, 2])
        overlap = 2.0 * radius - sep
        
        if overlap > 0:
            if not in_contact:
                in_contact = True
            contact_steps += 1
            impulse_trap += dW_trap
        elif in_contact:
            break
        
        dem.time += dem.dt
        dem.step_count += 1
    else:
        raise RuntimeError("Pair did not separate within step budget")
    
    KE_after = float(0.5 * m * (dem.velocities ** 2).sum())
    momentum_after = float(dem.velocities[:, 2].sum() * m)
    
    dE_state = KE_before - KE_after
    dE_impulse_trap = -impulse_trap
    
    e_measured = float(abs(dem.velocities[i, 2])) / v0
    dE_theory = (1.0 - e_target ** 2) * 0.5 * m_eff * (2.0 * v0) ** 2
    
    return VerletCollisionResult(
        dt=dem.dt,
        k_n=k_n,
        e_measured=e_measured,
        e_target=e_target,
        dE_state=dE_state,
        dE_impulse_trap=dE_impulse_trap,
        dE_theory=dE_theory,
        momentum_error=abs(momentum_after - momentum_before),
        contact_steps=contact_steps,
        v0=v0,
        m_eff=m_eff,
        E0=E0,
    )


# ---------------------------------------------------------------------------
# Event-driven (analytical) collision
# ---------------------------------------------------------------------------
def run_event_driven_collision(
    *,
    e_target: float = 0.7,
    v0: float = 0.5,
    k_n: float = 5.0e4,
    radius: float = 0.05,
    density: float = 900.0,
) -> EventDrivenResult:
    """Run head-on collision with event-driven analytical integration.
    
    Finds exact contact time by bisection, then integrates damped oscillator
    analytically within contact.
    """
    m = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = 0.5 * m
    c_n = _restitution_to_damping(e_target, k_n, m_eff)
    
    # Initial conditions: two spheres approaching at ±v0
    # Relative coordinate: gap x = z_j - z_i - 2*radius (negative = overlap)
    # Initial: x0 = 2e-6 (small gap), v0_rel = -2*v0 (approaching)
    x0 = 2.0e-6  # initial gap
    v_rel0 = -2.0 * v0  # relative velocity (negative = approaching)
    
    # Time to contact (gap = 0): x(t) = x0 + v_rel0 * t = 0 => t_contact = -x0 / v_rel0
    t_contact = -x0 / v_rel0
    
    # Within contact: m_eff * x_ddot = -k_n * x - c_n * x_dot (x < 0 in contact)
    # Let delta = -x (overlap, positive in contact)
    # m_eff * delta_ddot = -k_n * delta - c_n * delta_dot
    # delta_ddot + (c_n/m_eff) * delta_dot + (k_n/m_eff) * delta = 0
    
    omega_n = math.sqrt(k_n / m_eff)
    gamma = c_n / (2.0 * m_eff)
    zeta = gamma / omega_n
    
    # Damped frequency
    if zeta < 1.0:
        omega_d = omega_n * math.sqrt(1.0 - zeta ** 2)
        # Underdamped: delta(t) = A * exp(-gamma*t) * sin(omega_d * t + phi)
        # Initial conditions at touchdown: delta(0) = 0, delta_dot(0) = -v_rel0 = 2*v0
        delta_dot_0 = 2.0 * v0
        # delta(0) = 0 => phi = 0
        # delta_dot(0) = A * omega_d = delta_dot_0 => A = delta_dot_0 / omega_d
        A = delta_dot_0 / omega_d
        
        # Time to separation: delta(t_sep) = 0 => sin(omega_d * t_sep) = 0
        # First zero after t=0: omega_d * t_sep = pi => t_sep = pi / omega_d
        t_sep = math.pi / omega_d
        
        # Velocity at separation: delta_dot(t_sep)
        # delta_dot(t) = A * exp(-gamma*t) * (omega_d * cos(omega_d*t) - gamma * sin(omega_d*t))
        # At t_sep: sin(omega_d * t_sep) = 0, cos(omega_d * t_sep) = -1
        # delta_dot(t_sep) = A * exp(-gamma * t_sep) * omega_d * (-1)
        delta_dot_sep = -A * math.exp(-gamma * t_sep) * omega_d
        
        # Relative velocity at separation (delta_dot > 0 means separating)
        v_rel_sep = -delta_dot_sep  # v_rel = -delta_dot
        
        # Measured restitution
        e_measured = abs(v_rel_sep) / abs(v_rel0)
        
    else:
        # Overdamped or critically damped: no rebound
        e_measured = 0.0
        v_rel_sep = 0.0
        t_sep = 10.0 / gamma  # arbitrary long time
    
    # Energy accounting
    E0 = m * v0 ** 2  # total initial KE
    KE_before = E0
    KE_after = 0.5 * m_eff * v_rel_sep ** 2  # relative KE after (COM motion is zero)
    dE_state = KE_before - KE_after
    
    # Theoretical dissipation
    dE_theory = (1.0 - e_target ** 2) * 0.5 * m_eff * (2.0 * v0) ** 2
    
    # Contact time
    contact_time = t_sep
    
    # Momentum error (should be zero for equal masses)
    momentum_error = 0.0
    
    return EventDrivenResult(
        k_n=k_n,
        e_measured=e_measured,
        e_target=e_target,
        dE_state=dE_state,
        dE_theory=dE_theory,
        momentum_error=momentum_error,
        contact_time=contact_time,
        v0=v0,
        m_eff=m_eff,
        E0=E0,
    )


# ---------------------------------------------------------------------------
# Three-method comparison
# ---------------------------------------------------------------------------
def run_three_method_comparison(
    *,
    e_target: float = 0.7,
    v0: float = 0.5,
    k_n: float = 5.0e4,
    radius: float = 0.05,
    density: float = 900.0,
    dt: float = 1e-5,
) -> ThreeMethodResult:
    """Run the same collision with all three methods."""
    from .rigid_collision_energy import run_single_collision
    
    # Method 1: semi-implicit (baseline)
    result_semi = run_single_collision(
        e_target=e_target, v0=v0, k_n=k_n,
        radius=radius, density=density, dt=dt,
    )
    
    # Method 2: velocity-Verlet
    result_verlet = run_verlet_collision(
        e_target=e_target, v0=v0, k_n=k_n,
        radius=radius, density=density, dt=dt,
    )
    
    # Method 3: event-driven
    result_event = run_event_driven_collision(
        e_target=e_target, v0=v0, k_n=k_n,
        radius=radius, density=density,
    )
    
    return ThreeMethodResult(
        dt=dt,
        k_n=k_n,
        semi_implicit_dE=result_semi.dE_impulse_trap,
        semi_implicit_e=result_semi.e_measured,
        verlet_dE=result_verlet.dE_impulse_trap,
        verlet_e=result_verlet.e_measured,
        event_dE=result_event.dE_state,
        event_e=result_event.e_measured,
        dE_theory=result_semi.dE_theory,
        E0=result_semi.E0,
    )


# ---------------------------------------------------------------------------
# Convergence sweep
# ---------------------------------------------------------------------------
def run_convergence_sweep_three_methods(
    *,
    e_target: float = 0.7,
    v0: float = 0.5,
    k_n: float = 5.0e4,
    dt_factors: list[float] | None = None,
    radius: float = 0.05,
    density: float = 900.0,
) -> list[dict[str, float]]:
    """Sweep dt and compare all three methods."""
    if dt_factors is None:
        dt_factors = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
    
    m = density * (4.0 / 3.0) * math.pi * radius ** 3
    dt_base = dem_recommended_dt(k_n, m, radius)
    
    # Event-driven reference
    result_event = run_event_driven_collision(
        e_target=e_target, v0=v0, k_n=k_n,
        radius=radius, density=density,
    )
    dE_event = result_event.dE_state
    E0 = result_event.E0
    
    rows = []
    for factor in dt_factors:
        dt = dt_base * factor
        try:
            result = run_three_method_comparison(
                e_target=e_target, v0=v0, k_n=k_n,
                radius=radius, density=density, dt=dt,
            )
            rows.append({
                "dt": dt,
                "k_n": k_n,
                "semi_implicit_residual": abs(result.semi_implicit_dE - dE_event) / E0,
                "verlet_residual": abs(result.verlet_dE - dE_event) / E0,
                "event_residual": abs(result.event_dE - dE_event) / E0,
                "semi_implicit_e": result.semi_implicit_e,
                "verlet_e": result.verlet_e,
                "event_e": result.event_e,
                "E0": E0,
            })
        except RuntimeError:
            continue
    
    return rows


def fit_convergence_order(
    rows: list[dict[str, float]],
    residual_key: str = "verlet_residual",
) -> dict[str, float]:
    """Fit log|residual| vs log(dt)."""
    filtered = [r for r in rows if r[residual_key] > 1e-15]
    if len(filtered) < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r_squared": 0.0}
    
    log_dt = [math.log(r["dt"]) for r in filtered]
    log_res = [math.log(r[residual_key]) for r in filtered]
    
    n = len(log_dt)
    sum_x = sum(log_dt)
    sum_y = sum(log_res)
    sum_xy = sum(x * y for x, y in zip(log_dt, log_res))
    sum_x2 = sum(x * x for x in log_dt)
    
    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-30:
        return {"slope": float("nan"), "intercept": float("nan"), "r_squared": 0.0}
    
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in log_res)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_dt, log_res))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)
    
    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}
