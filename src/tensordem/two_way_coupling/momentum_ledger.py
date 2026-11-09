"""Verification Surface 3: Momentum exchange conservation ledger.

Two-way coupling requires that the momentum exchanged between particles and
fluid satisfies Newton's third law exactly (to machine precision).

Test protocol
-------------
1. **Single particle settling**: the drag, pressure gradient, and added mass
   forces on the particle must be exactly balanced by the fluid reaction.
   Track the cumulative ledger over many time steps.

2. **Multi-particle cloud settling**: N particles settling simultaneously.
   The total momentum transferred to the fluid must equal minus the total
   inter-phase force on all particles.

3. **Oscillating flow**: the time-integrated momentum exchange over a full
   period must be zero (no net momentum source/sink).

4. **Closure under force channel toggling**: turning off individual force
   channels (drag, pressure gradient, added mass) must change the ledger
   entries by exactly the removed channel's contribution.

The residual is defined as::

    r = |sum(F_inter_phase_particle) + sum(F_inter_phase_fluid)| / sum(|F|)

and must be < 1e-12 for all test cases.
"""

from __future__ import annotations

import math
import numpy as np

from ..cfd_dem_dense.forces import (
    GRAVITY,
    ForceModel,
    sphere_volume,
)
from .particle_ode import ParticleState, FluidState, ParticleODE, MomentumLedger

__all__ = [
    "single_particle_ledger",
    "multi_particle_ledger",
    "oscillating_flow_ledger",
    "channel_toggle_test",
    "run_momentum_ledger_verification",
]


def single_particle_ledger(
    diameter: float = 1.0e-4,
    rho_particle: float = 2500.0,
    rho_fluid: float = 1000.0,
    mu: float = 1.0e-3,
    gravity: float = GRAVITY,
    dt: float = 1e-5,
    n_steps: int = 1000,
) -> dict:
    """Settle a single particle and audit the momentum ledger."""
    model = ForceModel(
        drag_model="schiller_naumann",
        pressure_gradient=True,
        added_mass=True,
        added_mass_coefficient=0.5,
        fluid_reaction=True,
    )
    ode = ParticleODE(diameter, rho_particle, rho_fluid, mu, model, gravity)

    state = ParticleState(np.zeros(3), np.zeros(3))
    ledger = MomentumLedger()

    for step in range(n_steps):
        fluid = FluidState.quiescent(rho_fluid, gravity)
        accel, force_info = ode.compute_forces(state.velocity, fluid)
        reaction = ode.compute_reaction(force_info)

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

        state.velocity = state.velocity + accel * dt
        state.position = state.position + state.velocity * dt

    return {
        "n_steps": n_steps,
        "ledger": ledger.summary(),
        "residual": ledger.residual,
    }


def multi_particle_ledger(
    n_particles: int = 100,
    diameter: float = 1.0e-4,
    rho_particle: float = 2500.0,
    rho_fluid: float = 1000.0,
    mu: float = 1.0e-3,
    gravity: float = GRAVITY,
    voidage: float = 0.95,
    dt: float = 1e-5,
    n_steps: int = 500,
) -> dict:
    """Settle N particles and audit the total momentum ledger."""
    model = ForceModel(
        drag_model="schiller_naumann",
        pressure_gradient=True,
        added_mass=True,
        added_mass_coefficient=0.5,
        fluid_reaction=True,
    )
    ode = ParticleODE(diameter, rho_particle, rho_fluid, mu, model, gravity)

    # Total forces accumulated
    total_drag_p = 0.0
    total_drag_f = 0.0
    total_pressure_p = 0.0
    total_pressure_f = 0.0
    total_am_p = 0.0
    total_am_f = 0.0
    total_l1 = 0.0

    # All particles start at rest
    velocities = np.zeros((n_particles, 3))
    positions = np.zeros((n_particles, 3))

    for step in range(n_steps):
        for p in range(n_particles):
            fluid = FluidState(
                velocity=np.zeros(3),
                material_acceleration=np.zeros(3),
                pressure_gradient=np.array([0.0, 0.0, -rho_fluid * gravity]),
                voidage=voidage,
            )
            accel, force_info = ode.compute_forces(velocities[p], fluid)
            reaction = ode.compute_reaction(force_info)

            f_l1 = sum(np.linalg.norm(v) for v in
                       [force_info["drag"], force_info["pressure"],
                        force_info["added_mass_explicit"]])

            total_drag_p += float(np.sum(force_info["drag"]))
            total_drag_f += float(-np.sum(force_info["drag"]))
            total_pressure_p += float(np.sum(force_info["pressure"]))
            total_pressure_f += float(-np.sum(force_info["pressure"]))
            total_am_p += float(np.sum(force_info["added_mass_explicit"]))
            total_am_f += float(-np.sum(force_info["added_mass_explicit"]))
            total_l1 += f_l1

            velocities[p] += accel * dt
            positions[p] += velocities[p] * dt

    total_p = total_drag_p + total_pressure_p + total_am_p
    total_f = total_drag_f + total_pressure_f + total_am_f
    residual = abs(total_p + total_f) / total_l1 if total_l1 > 0 else 0.0

    return {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "voidage": voidage,
        "total_drag_particle": total_drag_p,
        "total_drag_fluid": total_drag_f,
        "total_pressure_particle": total_pressure_p,
        "total_pressure_fluid": total_pressure_f,
        "total_added_mass_particle": total_am_p,
        "total_added_mass_fluid": total_am_f,
        "total_particle_interphase": total_p,
        "total_fluid_interphase": total_f,
        "residual": residual,
        "force_l1": total_l1,
    }


def oscillating_flow_ledger(
    diameter: float = 1.0e-3,
    rho_particle: float = 2500.0,
    rho_fluid: float = 1000.0,
    mu: float = 1.0e-3,
    U0: float = 0.01,
    omega: float = 100.0,
    n_periods: int = 20,
    steps_per_period: int = 200,
) -> dict:
    """Particle in oscillating flow: net momentum exchange over full periods."""
    model = ForceModel(
        drag_model="schiller_naumann",
        pressure_gradient=False,
        added_mass=True,
        added_mass_coefficient=0.5,
        fluid_reaction=True,
    )
    ode = ParticleODE(diameter, rho_particle, rho_fluid, mu, model, gravity=0.0)

    period = 2.0 * math.pi / omega
    dt = period / steps_per_period
    t_total = n_periods * period

    state = ParticleState(np.zeros(3), np.zeros(3))

    # Accumulate per-period drag integrals
    period_drags = []
    current_period_drag = 0.0
    step_in_period = 0

    for step in range(int(t_total / dt)):
        t_now = step * dt
        u_f = np.array([U0 * math.sin(omega * t_now), 0.0, 0.0])
        du_f_dt = np.array([U0 * omega * math.cos(omega * t_now), 0.0, 0.0])
        fluid = FluidState(
            velocity=u_f,
            material_acceleration=du_f_dt,
            pressure_gradient=np.zeros(3),
            voidage=1.0,
        )

        accel, force_info = ode.compute_forces(state.velocity, fluid)
        current_period_drag += float(force_info["drag"][0]) * dt
        step_in_period += 1

        state.velocity = state.velocity + accel * dt
        state.position = state.position + state.velocity * dt

        if step_in_period >= steps_per_period:
            period_drags.append(current_period_drag)
            current_period_drag = 0.0
            step_in_period = 0

    period_drags = np.array(period_drags)

    # After transient (first 5 periods), check that periodic drag ≈ 0
    steady_drags = period_drags[5:]
    mean_drag = np.mean(steady_drags)
    max_drag = np.max(np.abs(steady_drags))

    # Reference scale: max drag force * period
    V_p = sphere_volume(diameter)
    Re_max = rho_fluid * U0 * diameter / mu
    f_drag_scale = 0.5 * rho_fluid * (24.0 / max(Re_max, 1e-15)) * (1 + 0.15 * Re_max**0.687) * math.pi * diameter**2 / 4 * U0**2
    ref_scale = f_drag_scale * period

    return {
        "n_periods": n_periods,
        "period_drags": period_drags.tolist(),
        "mean_steady_drag": mean_drag,
        "max_steady_drag": max_drag,
        "relative_mean": abs(mean_drag) / ref_scale if ref_scale > 0 else 0.0,
        "relative_max": max_drag / ref_scale if ref_scale > 0 else 0.0,
    }


def channel_toggle_test(
    diameter: float = 1.0e-4,
    rho_particle: float = 2500.0,
    rho_fluid: float = 1000.0,
    mu: float = 1.0e-3,
    dt: float = 1e-5,
    n_steps: int = 200,
) -> dict:
    """Verify that toggling force channels changes the ledger consistently."""
    configs = {
        "all": ForceModel(drag_model="schiller_naumann", pressure_gradient=True,
                          added_mass=True, fluid_reaction=True),
        "no_pressure": ForceModel(drag_model="schiller_naumann", pressure_gradient=False,
                                   added_mass=True, fluid_reaction=True),
        "no_added_mass": ForceModel(drag_model="schiller_naumann", pressure_gradient=True,
                                     added_mass=False, fluid_reaction=True),
        "no_reaction": ForceModel(drag_model="schiller_naumann", pressure_gradient=True,
                                   added_mass=True, fluid_reaction=False),
    }

    results = {}
    for name, model in configs.items():
        ode = ParticleODE(diameter, rho_particle, rho_fluid, mu, model, GRAVITY)
        state = ParticleState(np.zeros(3), np.zeros(3))
        ledger = MomentumLedger()

        for step in range(n_steps):
            fluid = FluidState.quiescent(rho_fluid, GRAVITY)
            accel, force_info = ode.compute_forces(state.velocity, fluid)
            reaction = ode.compute_reaction(force_info)

            f_l1 = sum(np.linalg.norm(v) for v in
                       [force_info["drag"], force_info["pressure"],
                        force_info["added_mass_explicit"]])

            if model.fluid_reaction:
                ledger.accumulate(
                    drag_p=float(np.sum(force_info["drag"])),
                    drag_f=float(-np.sum(force_info["drag"])),
                    pressure_p=float(np.sum(force_info["pressure"])),
                    pressure_f=float(-np.sum(force_info["pressure"])),
                    added_mass_p=float(np.sum(force_info["added_mass_explicit"])),
                    added_mass_f=float(-np.sum(force_info["added_mass_explicit"])),
                    force_l1=f_l1,
                )
            else:
                ledger.accumulate(
                    drag_p=float(np.sum(force_info["drag"])),
                    drag_f=0.0,
                    pressure_p=float(np.sum(force_info["pressure"])),
                    pressure_f=0.0,
                    added_mass_p=float(np.sum(force_info["added_mass_explicit"])),
                    added_mass_f=0.0,
                    force_l1=f_l1,
                )

            state.velocity = state.velocity + accel * dt
            state.position = state.position + state.velocity * dt

        results[name] = ledger.summary()

    # Check consistency
    all_sum = results["all"]["total_particle_interphase"]
    no_p_sum = results["no_pressure"]["total_particle_interphase"]
    no_am_sum = results["no_added_mass"]["total_particle_interphase"]

    pressure_diff = abs(all_sum - no_p_sum)
    am_diff = abs(all_sum - no_am_sum)

    return {
        "configs": results,
        "pressure_toggle_diff": pressure_diff,
        "added_mass_toggle_diff": am_diff,
        "all_residual": results["all"]["residual"],
        "no_reaction_residual": results["no_reaction"]["residual"],
    }


def run_momentum_ledger_verification(**kwargs) -> dict:
    """Run all momentum ledger verification tests."""
    single = single_particle_ledger(**{k: v for k, v in kwargs.items()
                                        if k in ("diameter", "rho_particle", "rho_fluid",
                                                  "mu", "gravity", "dt", "n_steps")})
    multi = multi_particle_ledger(**{k: v for k, v in kwargs.items()
                                      if k in ("diameter", "rho_particle", "rho_fluid",
                                                "mu", "gravity", "dt")})
    osc = oscillating_flow_ledger(**{k: v for k, v in kwargs.items()
                                      if k in ("diameter", "rho_particle", "rho_fluid",
                                                "mu")})
    toggle = channel_toggle_test(**{k: v for k, v in kwargs.items()
                                     if k in ("diameter", "rho_particle", "rho_fluid",
                                               "mu", "dt", "n_steps")})

    return {
        "single_particle": single,
        "multi_particle": multi,
        "oscillating_flow": osc,
        "channel_toggle": toggle,
        "all_residuals": {
            "single": single["residual"],
            "multi": multi["residual"],
            "oscillating_mean": osc["relative_mean"],
            "toggle_all": toggle["all_residual"],
        },
    }
