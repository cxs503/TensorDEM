"""Verification Surface 1: Stokes terminal velocity.

A single sphere settles under gravity in a quiescent infinite fluid.
The equation of motion (1-D, vertical z, positive upward)::

    (m_p + C_a rho_f V_p) dw/dt = F_drag(w) + F_pressure - m_p g

where:
    F_drag = 0.5 rho_f C_D(Re) A_p |u_f - w| (u_f - w)
    F_pressure = -V_p grad(p) = V_p rho_f g  (hydrostatic)

In the Stokes limit (Re << 1), the terminal velocity is::

    v_t = (rho_p - rho_f) g d^2 / (18 mu)

With Schiller-Naumann drag correction, solve iteratively::

    0.5 rho_f C_D(Re_t) A_p v_t^2 = (rho_p - rho_f) V_p g

Test protocol
-------------
- Release particle from rest, integrate until steady state.
- Compare final velocity to analytical v_t (Stokes and SN).
- Compare relaxation curve to exponential with effective tau.
"""

from __future__ import annotations

import math
import numpy as np

from ..cfd_dem_dense.forces import (
    GRAVITY,
    ForceModel,
    sphere_volume,
    sphere_projected_area,
    schiller_naumann_cd,
)
from .particle_ode import ParticleState, FluidState, ParticleODE, MomentumLedger

__all__ = [
    "stokes_terminal_velocity",
    "schiller_naumann_terminal_velocity",
    "relaxation_time",
    "run_stokes_verification",
]


def stokes_terminal_velocity(
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    gravity: float = GRAVITY,
) -> float:
    """Analytical Stokes terminal velocity: v_t = (rho_p - rho_f) g d^2 / (18 mu).

    Positive downward (magnitude).
    """
    return (rho_particle - rho_fluid) * gravity * diameter**2 / (18.0 * mu)


def schiller_naumann_terminal_velocity(
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    gravity: float = GRAVITY,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Terminal velocity with Schiller-Naumann drag correction.

    Solve: 0.5 rho_f C_D(Re_t) A_p v_t^2 = (rho_p - rho_f) V_p g
    where C_D = (24/Re)(1 + 0.15 Re^0.687).
    """
    V_p = sphere_volume(diameter)
    A_p = sphere_projected_area(diameter)
    W = (rho_particle - rho_fluid) * gravity * V_p  # submerged weight

    # Initial guess: Stokes
    v = stokes_terminal_velocity(diameter, rho_particle, rho_fluid, mu, gravity)

    for _ in range(max_iter):
        Re = rho_fluid * v * diameter / mu
        if Re < 1e-15:
            break
        cd = 24.0 / Re * (1.0 + 0.15 * Re**0.687)
        f_drag = 0.5 * rho_fluid * cd * A_p * v**2
        residual = f_drag - W

        # Newton iteration
        dRe_dv = rho_fluid * diameter / mu
        correction = 1.0 + 0.15 * Re**0.687
        dcorrection_dv = 0.15 * 0.687 * Re**(-0.313) * dRe_dv
        df_dv = 12.0 * mu * A_p / diameter * (correction + v * dcorrection_dv)

        dv = residual / df_dv
        v = v - dv
        if abs(dv) < tol * v:
            break

    return v


def relaxation_time(
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    added_mass_coeff: float = 0.5,
) -> float:
    """Effective relaxation time including added mass.

    tau = (rho_p + C_a rho_f) d^2 / (18 mu)
    """
    return (rho_particle + added_mass_coeff * rho_fluid) * diameter**2 / (18.0 * mu)


def run_stokes_verification(
    diameter: float = 1.0e-4,
    rho_particle: float = 2500.0,
    rho_fluid: float = 1000.0,
    mu: float = 1.0e-3,
    gravity: float = GRAVITY,
    added_mass_coeff: float = 0.5,
    dt: float = 1e-5,
    t_max: float = 0.5,
) -> dict:
    """Run the Stokes terminal velocity verification."""
    model = ForceModel(
        drag_model="schiller_naumann",
        pressure_gradient=True,
        added_mass=True,
        added_mass_coefficient=added_mass_coeff,
        fluid_reaction=True,
    )
    ode = ParticleODE(diameter, rho_particle, rho_fluid, mu, model, gravity)

    # Analytical references
    v_t_stokes = stokes_terminal_velocity(diameter, rho_particle, rho_fluid, mu, gravity)
    v_t_sn = schiller_naumann_terminal_velocity(diameter, rho_particle, rho_fluid, mu, gravity)
    tau = relaxation_time(diameter, rho_particle, rho_fluid, mu, added_mass_coeff)

    # Initial state: particle at rest at origin
    state0 = ParticleState(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
    )

    # Quiescent fluid with hydrostatic pressure gradient
    def fluid_fn(t, x):
        return FluidState(
            velocity=np.zeros(3),
            material_acceleration=np.zeros(3),
            pressure_gradient=np.array([0.0, 0.0, -rho_fluid * gravity]),
            voidage=1.0,
        )

    # Integrate
    ledger = MomentumLedger()
    state_final, history, ledger = _integrate_settling(
        ode, state0, fluid_fn, t_max, dt, ledger
    )

    # Extract velocity magnitude (z-component, negative = settling)
    times = np.array([h["t"] for h in history])
    vz = np.array([h["velocity"][2] for h in history])
    speed = np.abs(vz)

    # Terminal velocity = final speed
    v_t_numerical = speed[-1]

    # Relative errors
    err_stokes = abs(v_t_numerical - v_t_stokes) / v_t_stokes
    err_sn = abs(v_t_numerical - v_t_sn) / v_t_sn

    # Fit relaxation: v(t) = v_t * (1 - exp(-t/tau_fit))
    mask = speed < 0.9 * v_t_numerical
    if np.sum(mask) > 10:
        t_fit = times[mask]
        v_fit = speed[mask]
        ratio = np.clip(1.0 - v_fit / v_t_numerical, 1e-15, 1.0)
        log_ratio = np.log(ratio)
        tau_fit = -np.sum(t_fit) / np.sum(log_ratio) if np.sum(log_ratio) != 0 else tau
    else:
        tau_fit = tau

    return {
        "v_t_stokes": v_t_stokes,
        "v_t_sn": v_t_sn,
        "v_t_numerical": v_t_numerical,
        "tau_analytical": tau,
        "tau_fit": tau_fit,
        "relative_error_stokes": err_stokes,
        "relative_error_sn": err_sn,
        "Re_terminal": rho_fluid * v_t_numerical * diameter / mu,
        "ledger_residual": ledger.residual,
        "times": times,
        "speed": speed,
        "history": history,
    }


def _integrate_settling(ode, state0, fluid_fn, t_max, dt, ledger):
    """Specialised integrator for 1-D settling (z-direction)."""
    state = state0.copy()
    history = []
    t = 0.0
    n_steps = int(math.ceil(t_max / dt))

    for i in range(n_steps):
        t_now = i * dt
        fluid = fluid_fn(t_now, state.position)
        accel, force_info = ode.compute_forces(state.velocity, fluid)
        reaction = ode.compute_reaction(force_info)

        # Ledger accumulation (inter-phase forces only)
        f_l1 = sum(np.linalg.norm(v) for v in
                   [force_info["drag"], force_info["pressure"],
                    force_info["added_mass_explicit"]])
        ledger.accumulate(
            drag_p=float(np.sum(force_info["drag"])),
            drag_f=float(-np.sum(force_info["drag"])),
            pressure_p=float(np.sum(force_info["pressure"])),
            pressure_f=float(-np.sum(force_info["pressure"])),
            added_mass_p=float(np.sum(force_info["added_mass_explicit"])),
            added_mass_f=float(-np.sum(force_info["added_mass_explicit"])),
            force_l1=f_l1,
        )

        history.append({
            "t": t_now,
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "acceleration": accel.copy(),
            "Re": force_info["Re"],
            "drag": force_info["drag"].copy(),
            "pressure": force_info["pressure"].copy(),
            "added_mass": force_info["added_mass_explicit"].copy(),
            "gravity": force_info["gravity"].copy(),
            "reaction": reaction.copy(),
        })

        # Forward Euler
        state.velocity = state.velocity + accel * dt
        state.position = state.position + state.velocity * dt

    return state, history, ledger
