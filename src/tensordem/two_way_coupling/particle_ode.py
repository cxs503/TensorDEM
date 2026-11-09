"""Point-particle Lagrangian ODE integrator for two-way coupling.

Integrates the particle equation of motion::

    (m_p + C_a rho_f V_p) du_p/dt = F_drag + F_added_mass + F_pressure + F_gravity

where:
    F_drag        = 0.5 rho_f C_D(Re) A_p |u_f - u_p| (u_f - u_p) * eps_f^{-chi}
    F_added_mass  = C_a rho_f V_p Du_f/Dt   [explicit part; implicit on LHS]
    F_pressure    = -V_p grad(p)
    F_gravity     = -rho_f V_p g (Archimedes) - rho_p V_p g = -(rho_p - rho_f) V_p g

The effective mass on the LHS is ``m_eff = m_p + C_a rho_f V_p``.

Two-way coupling: the equal-and-opposite reaction is deposited onto the fluid
momentum equation as a volume force, tracked in a conservation ledger.

All quantities SI.  No domain-specific constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Reuse primitives from the existing dense coupling module
from ..cfd_dem_dense.forces import (
    GRAVITY,
    ForceModel,
    sphere_volume,
    sphere_projected_area,
    schiller_naumann_cd,
    per_particle_drag,
    pressure_gradient_force,
    added_mass_force,
    fluid_pressure_gradient,
    drag_density,
)

__all__ = [
    "ParticleState",
    "FluidState",
    "MomentumLedger",
    "ParticleODE",
    "integrate_particle",
]


@dataclass
class ParticleState:
    """State of a single point particle (1-D vertical for settling, or 3-D)."""
    position: np.ndarray   # shape (3,) or (n,3)
    velocity: np.ndarray   # shape (3,) or (n,3)

    def copy(self) -> "ParticleState":
        return ParticleState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
        )


@dataclass
class FluidState:
    """Local fluid state seen by the particle (or cell-averaged for coupling)."""
    velocity: np.ndarray        # fluid velocity at particle location
    material_acceleration: np.ndarray  # Du_f/Dt at particle location
    pressure_gradient: np.ndarray      # grad(p) at particle location
    voidage: float = 1.0               # local void fraction

    @staticmethod
    def quiescent(rho_fluid: float = 1000.0, gravity: float = GRAVITY) -> "FluidState":
        """Quiescent fluid with hydrostatic pressure gradient."""
        return FluidState(
            velocity=np.zeros(3),
            material_acceleration=np.zeros(3),
            pressure_gradient=np.array([0.0, 0.0, -rho_fluid * gravity]),
            voidage=1.0,
        )


@dataclass
class MomentumLedger:
    """Conservation ledger tracking momentum exchange between particles and fluid.

    Tracks inter-phase forces (drag, pressure gradient, added mass) which
    obey Newton's third law.  Gravity is an external force and has no
    fluid-side reaction.

    The residual is::

        r = |sum(F_inter_phase_particle) + sum(F_inter_phase_fluid)| / sum(|F|)

    and must be at machine precision for Newton's third law to hold.
    """
    # Cumulative inter-phase forces
    drag_particle: float = 0.0
    drag_fluid: float = 0.0
    pressure_particle: float = 0.0
    pressure_fluid: float = 0.0
    added_mass_particle: float = 0.0
    added_mass_fluid: float = 0.0
    n_steps: int = 0
    _force_l1_accum: float = 0.0

    def accumulate(self, *, drag_p: float, drag_f: float,
                   pressure_p: float, pressure_f: float,
                   added_mass_p: float, added_mass_f: float,
                   force_l1: float) -> None:
        """Accumulate one time step's inter-phase forces."""
        self.drag_particle += drag_p
        self.drag_fluid += drag_f
        self.pressure_particle += pressure_p
        self.pressure_fluid += pressure_f
        self.added_mass_particle += added_mass_p
        self.added_mass_fluid += added_mass_f
        self._force_l1_accum += force_l1
        self.n_steps += 1

    @property
    def total_particle_interphase(self) -> float:
        """Total inter-phase force on particle (excludes gravity)."""
        return (self.drag_particle + self.pressure_particle +
                self.added_mass_particle)

    @property
    def total_fluid_interphase(self) -> float:
        """Total inter-phase reaction on fluid."""
        return (self.drag_fluid + self.pressure_fluid +
                self.added_mass_fluid)

    @property
    def residual(self) -> float:
        """Normalised momentum conservation residual for inter-phase forces."""
        l1 = self._force_l1_accum
        if l1 < 1e-300:
            return 0.0
        return abs(self.total_particle_interphase + self.total_fluid_interphase) / l1

    def summary(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "drag_particle": self.drag_particle,
            "drag_fluid": self.drag_fluid,
            "pressure_particle": self.pressure_particle,
            "pressure_fluid": self.pressure_fluid,
            "added_mass_particle": self.added_mass_particle,
            "added_mass_fluid": self.added_mass_fluid,
            "total_particle_interphase": self.total_particle_interphase,
            "total_fluid_interphase": self.total_fluid_interphase,
            "residual": self.residual,
            "force_l1_accum": self._force_l1_accum,
        }


class ParticleODE:
    """Right-hand side of the particle equation of motion.

    Given the current particle state and the local fluid state, compute
    the net force and effective mass for the ODE::

        m_eff du_p/dt = F_net

    where m_eff = m_p + C_a rho_f V_p (added mass on LHS).
    """

    def __init__(
        self,
        diameter: float,
        rho_particle: float,
        rho_fluid: float,
        mu: float,
        model: ForceModel | None = None,
        gravity: float = GRAVITY,
    ):
        self.d = diameter
        self.rho_p = rho_particle
        self.rho_f = rho_fluid
        self.mu = mu
        self.model = model or ForceModel(
            drag_model="schiller_naumann",
            pressure_gradient=True,
            added_mass=True,
            added_mass_coefficient=0.5,
            fluid_reaction=True,
        )
        self.g = gravity
        self.V_p = sphere_volume(diameter)
        self.A_p = sphere_projected_area(diameter)
        self.m_p = rho_particle * self.V_p
        self.m_added = self.model.added_mass_coefficient * rho_fluid * self.V_p
        self.m_eff = self.m_p + self.m_added

    def compute_forces(
        self,
        v_p: np.ndarray,
        fluid: FluidState,
    ) -> tuple[np.ndarray, dict]:
        """Compute all force channels and return (acceleration, force_dict).

        force_dict contains each channel's contribution for the ledger.
        """
        eps = fluid.voidage
        u_f = fluid.velocity
        du_f_dt = fluid.material_acceleration
        grad_p = fluid.pressure_gradient

        # Relative velocity
        u_rel = u_f - v_p
        u_rel_mag = np.linalg.norm(u_rel)

        # Reynolds number
        Re = self.rho_f * u_rel_mag * self.d / self.mu

        # Drag force (Schiller-Naumann with optional voidage correction)
        if Re < 1e-12:
            f_drag = np.zeros(3)
        else:
            cd = float(schiller_naumann_cd(np.array([Re]))[0])
            cd *= self.model.cd_scale
            # Voidage correction
            if self.model.drag_model == "wen_yu":
                cd *= eps ** (-2.65)
            elif self.model.drag_model == "gidaspow" and eps < 0.8:
                # Ergun branch: use Ergun drag density
                drag_den = drag_density(u_rel_mag, eps, diameter=self.d,
                                        rho_fluid=self.rho_f, mu=self.mu,
                                        model=self.model)
                f_drag = drag_den * u_rel * self.V_p
                cd = None  # handled via Ergun
            # elif di_felice or schiller_naumann: no voidage correction for SN

            if cd is not None:
                f_drag_mag = 0.5 * self.rho_f * cd * self.A_p * u_rel_mag
                f_drag = f_drag_mag * u_rel
                if self.model.drag_model == "di_felice":
                    from ..cfd_dem_dense.forces import di_felice_chi
                    chi = float(di_felice_chi(np.array([Re]))[0])
                    f_drag *= eps ** (-chi)

        # Pressure gradient force
        f_pressure = -self.V_p * grad_p if self.model.pressure_gradient else np.zeros(3)

        # Added mass force: F_am = C_a rho_f V_p (Du_f/Dt - du_p/dt)
        # The explicit part is C_a rho_f V_p Du_f/Dt (implicit part on LHS)
        f_am_explicit = self.m_added * du_f_dt if self.model.added_mass else np.zeros(3)

        # Gravity (external force, no fluid reaction)
        f_gravity = np.array([0.0, 0.0, -self.m_p * self.g])

        # Net force (excluding added mass implicit part)
        f_net = f_drag + f_pressure + f_am_explicit + f_gravity

        # Acceleration: m_eff * a = f_net
        accel = f_net / self.m_eff

        # Force magnitudes for ledger
        force_info = {
            "drag": f_drag.copy(),
            "pressure": f_pressure.copy(),
            "added_mass_explicit": f_am_explicit.copy(),
            "gravity": f_gravity.copy(),
            "net": f_net.copy(),
            "Re": Re,
        }

        return accel, force_info

    def compute_reaction(self, force_info: dict) -> np.ndarray:
        """Compute the fluid-side reaction force (Newton's third law).

        Only inter-phase forces (drag, pressure, added mass) have reactions.
        Gravity is external and has no fluid-side reaction.
        """
        if not self.model.fluid_reaction:
            return np.zeros(3)
        reaction = -(force_info["drag"] + force_info["pressure"] +
                     force_info["added_mass_explicit"])
        return reaction


def integrate_particle(
    ode: ParticleODE,
    state0: ParticleState,
    fluid_fn: Callable[[float, np.ndarray], FluidState],
    t_end: float,
    dt: float,
    ledger: MomentumLedger | None = None,
) -> tuple[ParticleState, list[dict], MomentumLedger]:
    """Integrate the particle ODE with forward Euler.

    Parameters
    ----------
    ode : ParticleODE
        The force model.
    state0 : ParticleState
        Initial condition.
    fluid_fn : callable
        ``fluid_fn(t, x_p) -> FluidState`` giving the local fluid state.
    t_end : float
        Integration end time.
    dt : float
        Time step.
    ledger : MomentumLedger, optional
        Accumulator for momentum conservation audit.

    Returns
    -------
    state_final : ParticleState
    history : list of dicts with t, position, velocity, forces
    ledger : MomentumLedger
    """
    if ledger is None:
        ledger = MomentumLedger()

    state = state0.copy()
    history = []
    t = 0.0
    n_steps = int(math.ceil(t_end / dt))

    for i in range(n_steps):
        t_now = i * dt
        fluid = fluid_fn(t_now, state.position)

        accel, force_info = ode.compute_forces(state.velocity, fluid)
        reaction = ode.compute_reaction(force_info)

        # Accumulate ledger (inter-phase forces only)
        f_l1 = sum(np.linalg.norm(v) for v in
                   [force_info["drag"], force_info["pressure"],
                    force_info["added_mass_explicit"]])
        ledger.accumulate(
            drag_p=float(np.sum(force_info["drag"])),
            drag_f=float(np.sum(reaction) - float(np.sum(-(force_info["pressure"] +
                                                             force_info["added_mass_explicit"])))),
            pressure_p=float(np.sum(force_info["pressure"])),
            pressure_f=float(-np.sum(force_info["pressure"])) if ode.model.pressure_gradient else 0.0,
            added_mass_p=float(np.sum(force_info["added_mass_explicit"])),
            added_mass_f=float(-np.sum(force_info["added_mass_explicit"])) if ode.model.added_mass else 0.0,
            force_l1=f_l1,
        )

        # Record history
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

        # Forward Euler step
        state.velocity = state.velocity + accel * dt
        state.position = state.position + state.velocity * dt

    return state, history, ledger
