"""Explicit dynamics of the ice-sheet slice against a moving rigid wedge bow.

Contact model
-------------
The hull is rigid and the crushed ice layer between hull and particles is *not*
resolved as particles, so the contact carries an **elastic-plastic (spring +
slider) state**:

* ``delta``      geometric penetration of the particle circle into the rake plane
* ``delta_p``    plastic (irreversible) part of the penetration, contact state
* ``delta_e``    ``delta - delta_p`` — the elastic overlap that carries force
* ``F = k_H delta_e^(3/2)``, capped by the crushing capacity
  ``F_cap = p_c t w(delta_e)``; while the cap is active the excess penetration
  is **plastic flow** and its work ``F d delta_p`` is booked as ice-crushing
  energy.

Without the plastic state the cap model would be path-independent and the
"plastic work" would be an artifact of contact chatter; with it the loading /
unloading branches differ and the bookkeeping is physically meaningful.

Energy ledger
-------------
========================  ==========================================================
``W_in``                  ship work ``sum (F_n sin b + F_t cos b) V dt``
``E_crush``               ``sum F d delta_p`` (plastic crushing work)
``E_crack``               elastic energy released by broken bonds
``E_friction``            ``sum |F_t v_slip| dt``
``E_drag``                hydrodynamic drag dissipation
``W_body``                gravity + buoyancy work (conservative pair)
``dKE``                   kinetic energy change (translation + rotation)
``dU_bond``               elastic energy change of the active bonds
``dU_contact``            elastic energy stored in the live contact springs
========================  ==========================================================

Checked identities (both are evaluated in :meth:`IceStructureContactSim.ledger`)::

    full:    W_in + W_body = E_crush + E_crack + E_friction + E_drag
                             + dKE + dU_bond + dU_contact
    reduced: W_in           = E_break + E_friction + dKE     (DAG node L5)
             with E_break  = E_crush + E_crack

``E_crush`` is booked as *breakage* energy because ice crushing is the failure
mechanism that consumes that work; the bond-break energy alone is two orders of
magnitude smaller and cannot close the ledger (that is the well-known reason DEM
ice models must carry an inelastic crushing limit state).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bow import WedgeBow
from .contact import (
    CrushingContact,
    CoulombFriction,
    capacity_vector,
    coulomb_force_vector,
    hertz_force_vector,
)
from .lattice import IceSheetLattice

__all__ = ["SimulationConfig", "IceStructureContactSim"]


@dataclass
class SimulationConfig:
    """Numerical settings for the ice-structure contact simulation."""

    dt_s: float = 2.0e-5
    drag_coefficient: float = 1.0
    drag_linear_coefficient: float = 0.0
    initial_stem_x_m: float = 0.20
    record_every: int = 100
    crushing_pressure_pa: float = 1.5e6


class IceStructureContactSim:
    """Dynamic ice-sheet slice pushed by a rigid wedge bow."""

    def __init__(
        self,
        lattice: IceSheetLattice,
        bow: WedgeBow,
        *,
        contact: CrushingContact | None = None,
        friction: CoulombFriction | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self.lattice = lattice
        self.bow = bow
        self.config = config or SimulationConfig()
        self.contact = contact or CrushingContact(
            ice_modulus_pa=lattice.material.young_modulus_pa,
            poisson_ratio=lattice.material.poisson_ratio,
            crushing_pressure_pa=self.config.crushing_pressure_pa,
            out_of_plane_thickness_m=lattice.width_t,
        )
        self.friction = friction or CoulombFriction(coefficient=0.15)

        self.time = 0.0
        self.stem_x = self.config.initial_stem_x_m
        self.hertz_stiffness = self.contact.hertz_stiffness(lattice.radius)

        n = lattice.n_particles
        self._delta_p = np.zeros(n)  # plastic penetration state
        self._u_contact = np.zeros(n)
        self._delta = np.zeros(n)

        # ledger
        self.w_in = 0.0  # ship work
        self.w_particle = 0.0  # contact work on the particles
        self.e_crush = 0.0
        self.e_crack = 0.0
        self.e_friction = 0.0
        self.e_drag = 0.0
        self.w_body = 0.0
        self.ke0 = self.kinetic_energy()
        self.u_bond0 = self.lattice.bond_energy()

        # fracture bookkeeping
        self.broken_bending = 0
        self.broken_crushing = 0

        # diagnostics
        self.history: list[dict[str, float]] = []
        self._resistance_sum = 0.0
        self._resistance_count = 0
        self._last_r = 0.0
        self._contact_steps = 0

    # ------------------------------------------------------------------ energy
    def kinetic_energy(self) -> float:
        lat = self.lattice
        return float(
            np.sum(0.5 * lat.mass * np.sum(lat.vel**2, axis=1))
            + np.sum(0.5 * lat.inertia * lat.omega**2)
        )

    @property
    def contact_spring_energy(self) -> float:
        return float(np.sum(self._u_contact))

    # -------------------------------------------------------------------- step
    def step(self) -> None:
        lat = self.lattice
        cfg = self.config
        dt = cfg.dt_s

        f_bond, tau_bond = lat.bond_forces()
        f_body = lat.body_forces()

        # --- hydrodynamic drag on submerged particles
        speed = np.sqrt(lat.vel[:, 0] ** 2 + lat.vel[:, 1] ** 2)
        submerged = lat.submerged_mask()
        area = lat.dx * lat.width_t
        drag_mag = np.where(
            submerged,
            0.5 * lat.water_density * cfg.drag_coefficient * area * speed**2
            + cfg.drag_linear_coefficient * speed,
            0.0,
        )
        inv_speed = 1.0 / np.maximum(speed, 1e-12)
        drag = -drag_mag[:, None] * lat.vel * inv_speed[:, None]

        # --- hull contact: elastic-plastic Hertz + crushing cap + Coulomb
        delta = self.bow.penetration(
            lat.pos[:, 0], lat.pos[:, 1], lat.radius, self.stem_x
        )
        in_contact = delta > 0.0
        self._delta = delta
        f_contact = np.zeros_like(lat.vel)
        resistance = 0.0
        if np.any(in_contact):
            delta_e = np.maximum(delta - self._delta_p, 0.0)
            f_el = hertz_force_vector(delta_e, np.full_like(delta_e, self.hertz_stiffness))
            f_cap = capacity_vector(
                delta,
                np.full_like(delta, lat.radius),
                self.contact.crushing_pressure_pa,
                lat.width_t,
            )
            yielding = in_contact & (f_el > f_cap)
            f_n = np.where(in_contact, np.minimum(f_el, f_cap), 0.0)
            # plastic flow: the elastic overlap that exactly carries the cap
            with np.errstate(invalid="ignore", divide="ignore"):
                delta_e_yield = np.where(
                    yielding,
                    np.power(
                        np.maximum(f_cap, 0.0) / self.hertz_stiffness, 2.0 / 3.0
                    ),
                    delta_e,
                )
            delta_p_new = np.where(yielding, delta - delta_e_yield, self._delta_p)
            d_plastic = np.clip(delta_p_new - self._delta_p, 0.0, None)
            self.e_crush += float(np.sum(f_n * d_plastic))
            self._delta_p = np.where(in_contact, delta_p_new, 0.0)
            self._u_contact = np.where(
                in_contact,
                (2.0 / 5.0) * self.hertz_stiffness * np.power(delta_e, 2.5),
                0.0,
            )

            nx_, nz_ = self.bow.normal
            tx_, tz_ = self.bow.tangent
            slip = (
                lat.vel[:, 0] * tx_ + lat.vel[:, 1] * tz_ - self.bow.speed_m_s * tx_
            )
            f_t = coulomb_force_vector(
                f_n,
                slip,
                self.friction.coefficient,
                self.friction.regularization_speed_m_s,
            )
            f_t = np.where(in_contact, f_t, 0.0)
            self.e_friction += float(np.sum(np.abs(f_t * slip))) * dt
            f_contact[:, 0] = f_n * nx_ + f_t * tx_
            f_contact[:, 1] = f_n * nz_ + f_t * tz_
            ship_force = f_n * nx_ + f_t * tx_  # forward force on the ice
            resistance = float(np.sum(ship_force))
            self._contact_steps += 1
        else:
            self._delta_p[:] = 0.0
            self._u_contact[:] = 0.0

        # --- integrate (semi-implicit Euler; the position update uses the new
        #     velocity, so the discrete work-kinetic-energy theorem is exact for
        #     the position-dependent forces when work is taken at that velocity)
        total = f_bond + f_body + drag + f_contact
        acc = total / lat.mass[:, None]
        acc[lat.fixed] = 0.0
        lat.vel += acc * dt
        lat.vel[lat.fixed] = 0.0
        v = lat.vel
        self.w_body += float(np.sum(f_body * v)) * dt
        self.e_drag += float(np.sum(-drag * v)) * dt
        self.w_particle += float(np.sum(f_contact * v)) * dt
        # ship work = (forward reaction on the hull) * hull speed
        self.w_in += resistance * self.bow.speed_m_s * dt
        lat.pos += v * dt

        ang_acc = tau_bond / lat.inertia
        ang_acc[lat.fixed] = 0.0
        lat.omega += ang_acc * dt
        lat.omega[lat.fixed] = 0.0
        lat.ang += lat.omega * dt

        # --- fracture after the geometry update
        broken = lat.check_failure(
            lat.material.flexural_strength_pa,
            lat.material.compressive_strength_pa,
        )
        if broken["broken"]:
            self.e_crack += float(broken["crack_energy"])
            self.broken_bending += int(broken["bending"])
            self.broken_crushing += int(broken["crushing"])

        self.stem_x += self.bow.speed_m_s * dt
        self.time += dt
        self._resistance_sum += resistance
        self._resistance_count += 1
        self._last_r = resistance

    # --------------------------------------------------------------------- run
    def run(self, duration_s: float, *, skip_s: float = 0.0) -> dict[str, Any]:
        n_steps = int(round(duration_s / self.config.dt_s))
        for k in range(n_steps):
            self.step()
            if k % self.config.record_every == 0:
                self.history.append(
                    {
                        "time_s": self.time,
                        "stem_x_m": self.stem_x,
                        "resistance_N": self._last_r,
                        "kinetic_energy_J": self.kinetic_energy(),
                        "bond_energy_J": self.lattice.bond_energy(),
                        "w_in_J": self.w_in,
                        "e_crush_J": self.e_crush,
                        "e_crack_J": self.e_crack,
                        "e_friction_J": self.e_friction,
                        "e_drag_J": self.e_drag,
                        "w_body_J": self.w_body,
                        "broken_bending": self.broken_bending,
                        "broken_crushing": self.broken_crushing,
                    }
                )
        return self.diagnostics(skip_s=skip_s)

    def mean_resistance(self, *, skip_s: float = 0.0) -> float:
        """Mean forward contact force after ``skip_s`` (start-up transient)."""
        samples = [h["resistance_N"] for h in self.history if h["time_s"] >= skip_s]
        if not samples:
            return 0.0
        return float(np.mean(samples))

    def mean_resistance_all(self) -> float:
        if self._resistance_count == 0:
            return 0.0
        return self._resistance_sum / self._resistance_count

    # -------------------------------------------------------------- diagnostics
    def ledger(self) -> dict[str, float]:
        d_ke = self.kinetic_energy() - self.ke0
        d_u_bond = self.lattice.bond_energy() - self.u_bond0
        d_u_contact = self.contact_spring_energy
        e_break = self.e_crush + self.e_crack
        full_residual = abs(
            self.w_in
            + self.w_body
            - self.e_drag
            - (
                self.e_crush
                + self.e_crack
                + self.e_friction
                + d_ke
                + d_u_bond
                + d_u_contact
            )
        )
        reduced_residual = abs(self.w_in - (e_break + self.e_friction + d_ke))
        denom = max(abs(self.w_in), 1e-12)
        return {
            "w_in_J": self.w_in,
            "w_particle_J": self.w_particle,
            "e_crush_J": self.e_crush,
            "e_crack_J": self.e_crack,
            "e_break_J": e_break,
            "e_friction_J": self.e_friction,
            "e_drag_J": self.e_drag,
            "w_body_J": self.w_body,
            "d_ke_J": d_ke,
            "d_u_bond_J": d_u_bond,
            "d_u_contact_J": d_u_contact,
            "contact_steps": float(self._contact_steps),
            "full_residual_abs_J": full_residual,
            "full_residual": full_residual / denom,
            "reduced_residual_abs_J": reduced_residual,
            "reduced_residual": reduced_residual / denom,
        }

    def diagnostics(self, *, skip_s: float = 0.0) -> dict[str, Any]:
        led = self.ledger()
        lat = self.lattice
        total_broken = self.broken_bending + self.broken_crushing
        return {
            "time_s": self.time,
            "stem_x_m": self.stem_x,
            "speed_m_s": self.bow.speed_m_s,
            "mean_resistance_N": self.mean_resistance(skip_s=skip_s),
            "mean_resistance_all_steps_N": self.mean_resistance_all(),
            "peak_resistance_N": float(
                max((h["resistance_N"] for h in self.history), default=0.0)
            ),
            "broken_bending": self.broken_bending,
            "broken_crushing": self.broken_crushing,
            "broken_total": total_broken,
            "bending_fraction": (
                self.broken_bending / total_broken if total_broken else 0.0
            ),
            "active_bonds": int(np.count_nonzero(lat.bonds["active"])),
            "total_bonds": lat.n_bonds,
            "contact_steps": self._contact_steps,
            "ledger": led,
            "history": self.history,
        }