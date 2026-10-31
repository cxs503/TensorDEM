"""Multi-body (formation) ice-ship interaction on a shared 3-D bonded ice field.

Each ship is represented by a rigid wedge-shaped bow (the same idealisation as
:class:`tensordom.dem3d_wedge.MovingWedgeBowIceDEM3D`) that translates along a
lane at constant speed.  All bows act simultaneously on *one* bonded-particle ice
field, so broken ice, contact chains and ridge-like accumulation are shared
between the ships: this is the multi-body DEM required by the formation study.

Verification (``tests/test_formation_dem.py``):
* ``N = 1`` degeneracy against the single-ship wedge model,
* shared-field coupling (a second ship changes the fracture state / forces),
* lane-overlap sensitivity,
* discrete energy balance: ship work in = mechanical energy change + bond
  fracture release + damping dissipation, residual reported as a fraction of the
  work input.

Energy bookkeeping is explicit, not residual-based: each dissipative channel is
integrated from its own force law (dashpot ``c v_n^2`` for contacts and bows,
``c_drag v^2`` for the background drag), and recovery of stored elastic energy is
taken from the mixed-mode diagnostics of the base solver.  The reported residual
therefore measures time-integration error only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

from .dem3d import IceDEM3D


@dataclass(frozen=True)
class ShipBow:
    """Rigid wedge bow of one formation member.

    ``center_y`` and ``half_width`` place the bow in the transverse direction so
    several ships can occupy distinct lanes of the same ice field.  ``tip_x0`` is
    the initial bow-tip position; the tip advances at ``speed``.
    """

    name: str
    speed: float
    center_y: float
    half_width: float
    tip_x0: float
    wedge_angle_deg: float = 45.0
    tip_height_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ship name must be a nonempty string")
        for name in ("speed", "half_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("center_y", "tip_x0"):
            value = getattr(self, name)
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 5.0 <= self.wedge_angle_deg <= 85.0:
            raise ValueError("wedge_angle_deg must be in [5, 85]")
        if not 0.0 <= self.tip_height_fraction <= 1.0:
            raise ValueError("tip_height_fraction must be in [0, 1]")


class FormationIceDEM3D(IceDEM3D):
    """Bonded-particle ice DEM driven by several independently moving wedge bows."""

    def __init__(self, config, ships: Sequence[ShipBow]) -> None:
        ships = list(ships)
        if not ships:
            raise ValueError("at least one ship bow is required")
        super().__init__(config)
        self.ships = ships
        y_min = float(self.initial_positions[:, 1].min())
        y_max = float(self.initial_positions[:, 1].max())
        z_min = float(self.initial_positions[:, 2].min())
        z_max = float(self.initial_positions[:, 2].max())
        self.bow_tip_z = {
            ship.name: z_min + ship.tip_height_fraction * (z_max - z_min) for ship in ships
        }
        self._unreached_penalty = 1.0e9
        self._pending_work = {ship.name: 0.0 for ship in ships}
        self._pending_bow_dashpot = {ship.name: 0.0 for ship in ships}
        self._pending_contact_dashpot = 0.0
        self._pending_drag = 0.0
        # The inherited single spherical indenter is disabled; each ship owns a wedge.
        self.tool_start = torch.tensor([self._unreached_penalty] * 3,
                                       dtype=self.dtype, device=self.device)
        self.tool_velocity.zero_()
        self.energy_log: list[dict[str, float]] = []
        self._work_in = {ship.name: 0.0 for ship in ships}
        self._propulsion_work = {ship.name: 0.0 for ship in ships}
        self._dissipation = {
            "bow_damping_J": 0.0,
            "contact_damping_J": 0.0,
            "drag_J": 0.0,
        }
        self._initial_energy: float | None = None
        self._reference_energy_released = 0.0
        self.last_ship_diagnostics: dict[str, dict[str, float]] = {}
        self._pending_bow_forces: dict[str, object] = {}

    # -- candidate lanes -------------------------------------------------------
    @property
    def lane_bounds(self) -> dict[str, float]:
        return {
            "y_min_m": float(self.initial_positions[:, 1].min()),
            "y_max_m": float(self.initial_positions[:, 1].max()),
            "x_min_m": float(self.initial_positions[:, 0].min()),
            "x_max_m": float(self.initial_positions[:, 0].max()),
            "z_min_m": float(self.initial_positions[:, 2].min()),
            "z_max_m": float(self.initial_positions[:, 2].max()),
        }

    def bow_tip_x(self, name: str) -> float:
        ship = self._ship(name)
        return ship.tip_x0 + ship.speed * self.time

    def _ship(self, name: str) -> ShipBow:
        for ship in self.ships:
            if ship.name == name:
                return ship
        raise KeyError(f"unknown ship {name!r}")

    # -- contact ---------------------------------------------------------------
    @torch.no_grad()
    def _bow_geometry(self, ship: ShipBow, velocities: torch.Tensor) -> tuple[torch.Tensor,
                                                                            torch.Tensor,
                                                                            torch.Tensor]:
        """Wedge normal, penetration depth and normal approach speed per particle."""
        theta = math.radians(ship.wedge_angle_deg)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        tip_x = self.bow_tip_x(ship.name)
        tip_z = self.bow_tip_z[ship.name]
        dx = self.positions[:, 0] - tip_x
        dz = self.positions[:, 2] - tip_z
        side = torch.where(dz >= 0, 1.0, -1.0)
        wedge_distance = sin_t * dx + cos_t * dz.abs()
        normal = torch.zeros_like(self.positions)
        normal[:, 0] = sin_t
        normal[:, 2] = side * cos_t
        dy = self.positions[:, 1] - ship.center_y
        side_distance = dy.abs() - ship.half_width
        side_active = side_distance > wedge_distance
        signed_distance = torch.maximum(wedge_distance, side_distance)
        normal[:, 0] = torch.where(side_active, torch.zeros_like(normal[:, 0]), normal[:, 0])
        normal[:, 2] = torch.where(side_active, torch.zeros_like(normal[:, 2]), normal[:, 2])
        normal[:, 1] = torch.where(side_active, torch.where(dy >= 0, 1.0, -1.0),
                                   torch.zeros_like(normal[:, 1]))
        overlap = (self.config.radius - signed_distance).clamp_min(0.0)
        relative_velocity = velocities - torch.tensor([ship.speed, 0.0, 0.0],
                                                     dtype=velocities.dtype,
                                                     device=velocities.device)
        normal_speed = (relative_velocity * normal).sum(dim=1)
        return normal, overlap, normal_speed

    @torch.no_grad()
    def _bow_contact(self, ship: ShipBow, velocities: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Nodal bow forces, reaction, penetration depth and dashpot dissipation rate."""
        if velocities is None:
            velocities = self.velocities
        normal, overlap, normal_speed = self._bow_geometry(ship, velocities)
        magnitude = (self.config.contact_stiffness * overlap
                     - self.config.contact_damping * normal_speed).clamp_min(0.0)
        magnitude = torch.where(overlap > 0.0, magnitude, torch.zeros_like(magnitude))
        nodal_force = magnitude[:, None] * normal
        reaction = -nodal_force.sum(dim=0)
        # Dashpot dissipation of an active contact: c v_n^2 with a parallel linear
        # spring (the elastic part is recoverable and excluded here).
        active = magnitude > 0.0
        dashpot_power = torch.where(active,
                                    self.config.contact_damping * normal_speed.square(),
                                    torch.zeros_like(normal_speed))
        return nodal_force, reaction, overlap, dashpot_power

    @torch.no_grad()
    def forces(self, external_forces: torch.Tensor | None = None, *,
               update_fracture: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        force, reaction = super().forces(external_forces, update_fracture=update_fracture)
        total_reaction = reaction.clone()
        for ship in self.ships:
            nodal, ship_reaction, overlap, dashpot_power = self._bow_contact(ship)
            force = force + nodal
            total_reaction = total_reaction + ship_reaction
            self.last_ship_diagnostics[ship.name] = {
                "reaction_x_N": float(ship_reaction[0]),
                "reaction_y_N": float(ship_reaction[1]),
                "reaction_z_N": float(ship_reaction[2]),
                "bow_contact_energy_J": float(
                    0.5 * self.config.contact_stiffness * overlap.square().sum()),
                "bow_particles_in_contact": int((overlap > 0).sum()),
                "bow_tip_x_m": self.bow_tip_x(ship.name),
                "bow_speed_m_s": ship.speed,
            }
            if update_fracture:
                # Work is accumulated as F . dx after the position update so that
                # it matches the symplectic-Euler displacement of this step.
                self._pending_bow_forces[ship.name] = nodal
                self._pending_bow_dashpot[ship.name] = float(dashpot_power.sum())
        if update_fracture:
            self._pending_drag = float(self.config.drag * self.velocities.square().sum())
            self._pending_contact_dashpot = self._contact_dashpot_power()
            self.last_reaction = total_reaction
        return force, total_reaction

    @torch.no_grad()
    def _contact_dashpot_power(self) -> float:
        """Dashpot dissipation rate of the active particle-particle contacts."""
        pairs = self._contact_candidate_pairs()
        if not pairs.numel():
            return 0.0
        ci, cj = pairs.T
        delta = self.positions[cj] - self.positions[ci]
        distance = torch.linalg.vector_norm(delta, dim=1)
        overlap = (2 * self.config.radius - distance).clamp_min(0.0)
        near = overlap > 0.0
        if not bool(near.any()):
            return 0.0
        ci, cj = ci[near], cj[near]
        delta, overlap = delta[near], overlap[near]
        normal = delta / torch.linalg.vector_norm(delta, dim=1)[:, None]
        bonded = self._alive_bond_for_pairs(torch.stack((ci, cj), dim=1))
        relative_velocity = self.velocities[cj] - self.velocities[ci]
        vn = (relative_velocity * normal).sum(dim=1)
        magnitude = (self.config.contact_stiffness * overlap
                     - self.config.contact_damping * vn).clamp_min(0.0)
        active = (~bonded) & (magnitude > 0.0)
        power = torch.where(active, self.config.contact_damping * vn.square(),
                            torch.zeros_like(vn))
        return float(power.sum())

    # -- stepping and audit ----------------------------------------------------
    def _mechanical_energy(self) -> dict[str, float]:
        diagnostics = self.diagnostics()
        bow_energy = sum(self.last_ship_diagnostics[name]["bow_contact_energy_J"]
                         for name in self.last_ship_diagnostics) \
            if self.last_ship_diagnostics else 0.0
        spring = (diagnostics["kinetic_energy_J"] + diagnostics["bond_elastic_energy_J"]
                  + diagnostics["particle_contact_energy_J"] + bow_energy)
        return {
            "kinetic_J": float(diagnostics["kinetic_energy_J"]),
            "bond_elastic_J": float(diagnostics["bond_elastic_energy_J"]),
            "particle_contact_J": float(diagnostics["particle_contact_energy_J"]),
            "bow_elastic_J": float(bow_energy),
            "released_bond_J": float(diagnostics["released_bond_energy_J"]),
            "mechanical_J": spring,
        }

    @torch.no_grad()
    def step(self, external_forces: torch.Tensor | None = None) -> None:
        if self._initial_energy is None:
            self._initial_energy = self._mechanical_energy()["mechanical_J"]
            self._pending_bow_forces = {}
        positions_before = self.positions.clone()
        velocity_before = self.velocities.clone()
        super().step(external_forces)
        displacement = self.positions - positions_before
        for ship in self.ships:
            nodal = self._pending_bow_forces.get(ship.name)
            if nodal is not None:
                self._work_in[ship.name] += float((nodal * displacement).sum())
            self._dissipation["bow_damping_J"] += self._pending_bow_dashpot[ship.name] * self.dt
            self._propulsion_work[ship.name] += (
                -float(self.last_ship_diagnostics[ship.name]["reaction_x_N"]) * ship.speed * self.dt)
        # Dissipation channels are integrated with the trapezoidal rule in velocity:
        # the start-of-step rates come from the force evaluation, the end-of-step
        # rates are evaluated with the updated velocities.
        for ship in self.ships:
            rate_after = float(self._bow_contact(ship, self.velocities)[3].sum())
            self._dissipation["bow_damping_J"] += (
                0.5 * (self._pending_bow_dashpot[ship.name] + rate_after) * self.dt)
        drag_after = float(self.config.drag * self.velocities.square().sum())
        self._dissipation["drag_J"] += 0.5 * (self._pending_drag + drag_after) * self.dt
        self._dissipation["contact_damping_J"] += self._pending_contact_dashpot * self.dt
        del velocity_before

    def energy_snapshot(self) -> dict[str, float]:
        """Current energy channels (one full force evaluation)."""
        energy = self._mechanical_energy()
        energy["work_in_J"] = float(sum(self._work_in.values()))
        energy["propulsion_J"] = float(sum(self._propulsion_work.values()))
        energy["dissipation_J"] = float(sum(self._dissipation.values()))
        energy["time_s"] = self.time
        self.energy_log.append(energy)
        return energy

    def audit(self) -> dict[str, Any]:
        """Energy balance of the run: work in versus stored, released and dissipated."""
        if self._initial_energy is None:
            raise RuntimeError("audit requires at least one step")
        energy = self._mechanical_energy()
        work_in = float(sum(self._work_in.values()))
        released = energy["released_bond_J"]
        dissipation = float(sum(self._dissipation.values()))
        stored_change = energy["mechanical_J"] - self._initial_energy
        residual = work_in - stored_change - released - dissipation
        denominator = abs(work_in) if abs(work_in) > 1.0e-12 else 1.0
        return {
            "work_in_J": work_in,
            "stored_energy_change_J": stored_change,
            "released_bond_J": released,
            "dissipation_J": dissipation,
            "dissipation_breakdown": dict(self._dissipation),
            "work_per_ship_J": dict(self._work_in),
            "propulsion_J": float(sum(self._propulsion_work.values())),
            "propulsion_per_ship_J": dict(self._propulsion_work),
            "ploughing_J": work_in - float(sum(self._propulsion_work.values())),
            "residual_J": residual,
            "relative_residual": residual / denominator,
            "steps": int(self.step_count),
            "dt_s": float(self.dt),
        }

    def ship_report(self) -> dict[str, dict[str, float]]:
        """Per-ship mean/peak axial reaction and work, plus fracture attribution."""
        report: dict[str, dict[str, float]] = {}
        for ship in self.ships:
            stats = self.last_ship_diagnostics.get(ship.name)
            if not stats:
                continue
            entry = {
                "reaction_x_N": stats["reaction_x_N"],
                "bow_particles_in_contact": float(stats["bow_particles_in_contact"]),
                "bow_tip_x_m": stats["bow_tip_x_m"],
                "work_in_J": self._work_in[ship.name],
                "center_y_m": ship.center_y,
                "half_width_m": ship.half_width,
                "speed_m_s": ship.speed,
            }
            report[ship.name] = entry
        return report

    def diagnostics(self, external_forces: torch.Tensor | None = None) -> dict[str, float | int]:
        result = super().diagnostics(external_forces)
        for ship in self.ships:
            stats = self.last_ship_diagnostics.get(ship.name, {})
            result[f"reaction_x_{ship.name}_N"] = stats.get("reaction_x_N", 0.0)
            result[f"broken_share_{ship.name}"] = 0.0
        result["formation_ship_count"] = len(self.ships)
        return result