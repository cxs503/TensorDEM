"""Energy-accounting extension for the 3-D bonded-sphere DEM.

The base solver remains backward compatible. This audited subclass tracks work,
viscous dissipation, and the solver's existing bond-release estimate. The
balance residual is a discrete diagnostic, not a claim of exact conservation.
"""
from __future__ import annotations

from typing import Any

import torch

from .dem3d import DEM3DConfig, IceDEM3D
from . import hertz_mindlin
from .dem3d_energy_quadrature import platen_defect_components


class EnergyAuditedIceDEM3D(IceDEM3D):
    """3-D DEM with cumulative work/dissipation ledger for verification runs."""

    def __init__(self, config: DEM3DConfig = DEM3DConfig()) -> None:
        super().__init__(config)
        self.cumulative_tool_work_J = 0.0
        self.cumulative_external_work_J = 0.0
        self.cumulative_drag_dissipation_J = 0.0
        self.cumulative_particle_damping_J = 0.0
        self.cumulative_tool_damping_J = 0.0
        self.cumulative_fracture_release_J = 0.0
        # Tangential friction work (total work of the Coulomb spring force; negative
        # during stick as energy is stored in the tangential spring) and rolling-
        # resistance work (always <= 0, pure dissipation).
        self.cumulative_friction_work_J = 0.0
        self.cumulative_rolling_work_J = 0.0
        # Platen (top/bottom + lateral) actuator work and dashpot dissipation.
        self.cumulative_platen_work_J = 0.0
        self.cumulative_platen_damping_J = 0.0
        # Exact-quadrature ledger arms.  ``*_exact_J`` accumulate the work of a
        # kinematic boundary as the *potential increment at fixed particle
        # positions*, which is the exact discrete work of the penalty spring;
        # the historical ``*_work_J`` arms evaluate the power at the start of
        # the step (left-endpoint quadrature) and are kept for
        # backward-compatible reporting.  See
        # ``docs/dem3d_energy_quadrature_closure.md``.
        self.cumulative_platen_work_exact_J = 0.0
        self.cumulative_tool_work_exact_J = 0.0
        self.cumulative_external_work_exact_J = 0.0
        self.energy_balance_residual_exact_J = 0.0
        # Closed-form left-endpoint quadrature defect 1/2 k n dz^2 accumulated
        # per step (see ``dem3d_energy_quadrature``).  Diagnostics only: it
        # predicts ``cumulative_platen_work_exact_J - cumulative_platen_work_J``
        # for the spring part of the platen force.
        self.cumulative_platen_deficit_predicted_J = 0.0
        # ``cumulative_bond_dissipation_J`` is created by the base solver (it
        # stays 0.0 on the legacy linear bond); the ledger only reads it.
        self.reference_mechanical_energy_J = self._mechanical_energy()
        self.energy_balance_residual_J = 0.0

    @torch.no_grad()
    def _mechanical_energy(self) -> float:
        """Return kinetic plus currently stored elastic energy without mutating damage."""
        cfg = self.config
        kinetic = 0.5 * cfg.mass * self.velocities.square().sum()
        i, j = self.pairs.T
        distance = torch.linalg.vector_norm(self.positions[j] - self.positions[i], dim=1)
        extension = distance - self.rest_lengths
        # Constitutive stored energy: the identical expression for the legacy
        # linear bond, and it includes the Maxwell branch springs when the
        # rate-dependent law is active.
        bond = (self.bond_stored_energy(extension) * self.alive).sum()

        pairs = self._contact_candidate_pairs()
        particle_contact = torch.zeros((), dtype=self.dtype, device=self.device)
        if pairs.numel():
            ci, cj = pairs.T
            d = torch.linalg.vector_norm(self.positions[cj] - self.positions[ci], dim=1)
            overlap = (2 * cfg.radius - d).clamp_min(0)
            eligible = ~self._alive_bond_for_pairs(pairs)
            particle_contact = (0.5 * cfg.contact_stiffness * overlap.square() * eligible).sum()

        tool_contact = self._tool_contact_potential(self.positions, self.tool_position)
        # Tangential (Cundall-Strack / Hertz-Mindlin) spring energy of active, unbonded contacts.
        # The friction force is a spring force: during stick its work is recoverable,
        # so the stored energy belongs in the mechanical balance.
        # For Hertz contacts: k_t = 8 G* sqrt(R* delta) (Mindlin)
        # For linear contacts: k_t = contact_stiffness (constant)
        tangential = torch.zeros((), dtype=self.dtype, device=self.device)
        if pairs.numel() and cfg.friction_coefficient > 0.0:
            ci, cj = pairs.T
            d = torch.linalg.vector_norm(self.positions[cj] - self.positions[ci], dim=1)
            overlap = (2 * cfg.radius - d).clamp_min(0)
            active = (overlap > 0) & (~self._alive_bond_for_pairs(pairs))
            if bool(active.any()):
                cnormal = (self.positions[cj] - self.positions[ci]) / d.clamp_min(
                    torch.finfo(self.dtype).eps
                )[:, None]
                tang_disp = self._get_tangential_state(ci[active], cj[active], cnormal[active])
                if cfg.contact_model == "hertz":
                    e_star, r_eff = self._hertz_pair_params
                    k_t = hertz_mindlin.hertz_mindlin_tangential_stiffness(
                        overlap[active], e_star, self._hertz_g_star, r_eff)
                    tangential = (0.5 * k_t * tang_disp.square().sum(dim=1)).sum()
                else:
                    tangential = (0.5 * cfg.contact_stiffness * tang_disp.square().sum(dim=1)).sum()
        # Platen (top/bottom + lateral) penalty-spring energy belongs in the
        # mechanical balance: the actuator work is partitioned between the
        # particle system and these boundary springs.
        platen = self._platen_potential(
            self.positions,
            top_z=(self.top_platen_start_z + self.time * cfg.top_platen_velocity
                   if cfg.top_platen_enabled else None),
            bottom_z=(self.bottom_platen_start_z + self.time * cfg.bottom_platen_velocity
                      if cfg.bottom_platen_enabled else None),
            lateral_pos=self.lateral_platen_pos,
        )
        return float(kinetic + bond + particle_contact + tool_contact + tangential + platen)

    @torch.no_grad()
    def _platen_potential(self, positions, top_z: float | None = None,
                          bottom_z: float | None = None,
                          lateral_pos=None) -> float:
        """Stored elastic energy (J) of the penalty platens.

        Evaluated at *fixed particle positions* so the same expression serves
        both ``_mechanical_energy`` (state energy) and the exact per-step
        actuator work ``-(U(after) - U(before))``.  ``None`` boundary arguments
        mean the corresponding platen is absent from the configuration.
        """
        cfg = self.config
        total = 0.0
        if top_z is not None:
            overlap = (positions[:, 2] + cfg.radius - top_z).clamp_min(0.0)
            total += 0.5 * cfg.platen_stiffness * float(overlap.square().sum())
        if bottom_z is not None:
            overlap = (bottom_z + cfg.radius - positions[:, 2]).clamp_min(0.0)
            total += 0.5 * cfg.platen_stiffness * float(overlap.square().sum())
        if lateral_pos is not None and cfg.lateral_platens:
            k = (cfg.lateral_platen_stiffness if cfg.lateral_platen_stiffness is not None
                 else cfg.platen_stiffness)
            px, py = positions[:, 0], positions[:, 1]
            total_lateral = 0.0
            for idx in range(4):
                if idx == 0:
                    overlap = (lateral_pos[0] - px + cfg.radius).clamp_min(0.0)
                elif idx == 1:
                    overlap = (px + cfg.radius - lateral_pos[1]).clamp_min(0.0)
                elif idx == 2:
                    overlap = (lateral_pos[2] - py + cfg.radius).clamp_min(0.0)
                else:
                    overlap = (py + cfg.radius - lateral_pos[3]).clamp_min(0.0)
                total_lateral += float(overlap.square().sum())
            total += 0.5 * k * total_lateral
        return total

    @torch.no_grad()
    def _tool_contact_potential(self, positions, tool_pos) -> float:
        """Stored elastic energy (J) of the spherical tool contact.

        Uses the same linear penalty expression as ``_mechanical_energy``; on the
        ``contact_model="hertz"`` branch this is the ledger's linearised
        convention, not the Hertz potential ``4/5 k delta^(5/2)`` (documented
        limitation, unchanged by this fix).
        """
        cfg = self.config
        distance = torch.linalg.vector_norm(positions - tool_pos, dim=1)
        overlap = (cfg.radius + cfg.tool_radius - distance).clamp_min(0.0)
        return 0.5 * cfg.contact_stiffness * float(overlap.square().sum())

    @torch.no_grad()
    def _boundary_potentials_at(self, positions, top_z, bottom_z, lateral_pos,
                                tool_pos) -> tuple[float, float]:
        """(platen, tool) stored energies at an explicit boundary configuration."""
        return (self._platen_potential(positions, top_z=top_z, bottom_z=bottom_z,
                                       lateral_pos=lateral_pos),
                self._tool_contact_potential(positions, tool_pos))

    @torch.no_grad()
    def _dissipation_rates(self) -> tuple[float, float, float]:
        """Instantaneous drag, particle-contact, and tool-contact dissipation rates."""
        cfg = self.config
        drag_rate = cfg.drag * self.velocities.square().sum()
        particle_rate = torch.zeros((), dtype=self.dtype, device=self.device)
        pairs = self._contact_candidate_pairs()
        if pairs.numel():
            i, j = pairs.T
            delta = self.positions[j] - self.positions[i]
            distance = torch.linalg.vector_norm(delta, dim=1)
            overlap = 2 * cfg.radius - distance
            normal = delta / distance.clamp_min(torch.finfo(self.dtype).eps)[:, None]
            relative_v = self.velocities[j] - self.velocities[i]
            vn = (relative_v * normal).sum(dim=1)
            active = (overlap > 0) & (~self._alive_bond_for_pairs(pairs))
            active &= (cfg.contact_stiffness * overlap - cfg.contact_damping * vn) > 0
            particle_rate = (cfg.contact_damping * vn.square() * active).sum()

        delta = self.positions - self.tool_position
        distance = torch.linalg.vector_norm(delta, dim=1)
        overlap = cfg.radius + cfg.tool_radius - distance
        normal = delta / distance.clamp_min(torch.finfo(self.dtype).eps)[:, None]
        fallback = torch.zeros_like(normal)
        fallback[:, 2] = -1
        normal = torch.where((distance > torch.finfo(self.dtype).eps)[:, None], normal, fallback)
        relative_v = self.velocities - self.tool_velocity
        vn = (relative_v * normal).sum(dim=1)
        active = (overlap > 0) & ((cfg.contact_stiffness * overlap - cfg.contact_damping * vn) > 0)
        tool_rate = (cfg.contact_damping * vn.square() * active).sum()
        return float(drag_rate), float(particle_rate), float(tool_rate)

    @torch.no_grad()
    def step(self, external_forces: torch.Tensor | None = None) -> None:
        """Advance one step and accumulate discrete work and dissipation estimates."""
        energy_before = self._mechanical_energy()
        force, reaction = self.forces(external_forces)
        drag_rate, particle_rate, tool_rate = self._dissipation_rates()
        loads = self._loads(external_forces)
        tool_work = float((-reaction * self.tool_velocity).sum()) * self.dt
        external_work = 0.0 if loads is None else float((loads * self.velocities).sum()) * self.dt

        self.cumulative_tool_work_J += tool_work
        self.cumulative_external_work_J += external_work
        self.cumulative_drag_dissipation_J += drag_rate * self.dt
        self.cumulative_particle_damping_J += particle_rate * self.dt
        self.cumulative_tool_damping_J += tool_rate * self.dt
        self.cumulative_fracture_release_J = float(self.failure_energy_J.sum())
        self.cumulative_friction_work_J += self._last_friction_power * self.dt
        self.cumulative_rolling_work_J += self._last_rolling_power * self.dt
        self.cumulative_platen_work_J += self._platen_work() * self.dt
        self.cumulative_platen_damping_J += self._platen_damping_power() * self.dt

        # Exact discrete work of the kinematic boundaries over this step.  The
        # particles have not moved yet, so the boundary potential can be
        # evaluated at the same particle positions before and after the
        # boundary advance; its increment is the exact work of the penalty
        # spring (quadratic in the boundary position, so no quadrature error
        # and no dependence on the contact-activation history).
        cfg = self.config
        top_z0 = (self.top_platen_start_z + self.time * cfg.top_platen_velocity
                  if cfg.top_platen_enabled else None)
        top_z1 = (None if top_z0 is None else top_z0 + self.dt * cfg.top_platen_velocity)
        bottom_z0 = (self.bottom_platen_start_z + self.time * cfg.bottom_platen_velocity
                     if cfg.bottom_platen_enabled else None)
        bottom_z1 = (None if bottom_z0 is None
                     else bottom_z0 + self.dt * cfg.bottom_platen_velocity)
        lateral0 = self.lateral_platen_pos if cfg.lateral_platens else None
        lateral1 = (None if lateral0 is None
                    else lateral0 + self.dt * self.lateral_platen_velocity)
        tool0 = self.tool_position
        tool1 = self.tool_start + (self.time + self.dt) * self.tool_velocity
        platen_u0, tool_u0 = self._boundary_potentials_at(
            self.positions, top_z0, bottom_z0, lateral0, tool0)
        platen_u1, tool_u1 = self._boundary_potentials_at(
            self.positions, top_z1, bottom_z1, lateral1, tool1)
        self.cumulative_platen_work_exact_J += platen_u1 - platen_u0
        self.cumulative_tool_work_exact_J += tool_u1 - tool_u0
        lat_k = (cfg.lateral_platen_stiffness if cfg.lateral_platen_stiffness is not None
                 else cfg.platen_stiffness)
        self.cumulative_platen_deficit_predicted_J += platen_defect_components(
            self.positions, radius=cfg.radius, stiffness=cfg.platen_stiffness,
            top_z0=top_z0, top_z1=top_z1, bottom_z0=bottom_z0, bottom_z1=bottom_z1,
            lateral_pos0=lateral0, lateral_pos1=lateral1,
            lateral_stiffness=lat_k)["total"]

        self.velocities += self.dt * force / self.config.mass
        self.velocities[self.fixed] = 0
        self.positions += self.dt * self.velocities
        self.positions[self.fixed] = self.initial_positions[self.fixed]
        # Lateral servo platens integrate their own velocity here.  The base
        # solver advances them in ``IceDEM3D.step``; this override replays the
        # kinematic update and must keep that term, otherwise a driver that
        # sets ``lateral_platen_velocity`` (true-triaxial pressure servo) moves
        # a velocity that never displaces the platen, and the prescribed
        # confining pressure can never be reached.
        if self.config.lateral_platens:
            self.lateral_platen_pos += self.lateral_platen_velocity * self.dt
        if loads is not None:
            # Exact discrete work of a constant external load: F . dx with
            # dx = dt * v_new.  The historical arm above used v_old.
            self.cumulative_external_work_exact_J += float(
                (loads * self.velocities).sum()) * self.dt
        self.time += self.dt
        self.step_count += 1
        if not bool(torch.isfinite(self.positions).all() and torch.isfinite(self.velocities).all()):
            raise FloatingPointError("non-finite DEM3D state; reduce dt or contact stiffness")

        energy_after = self._mechanical_energy()
        dissipated = (self.cumulative_drag_dissipation_J
                      + self.cumulative_particle_damping_J
                      + self.cumulative_tool_damping_J
                      + self.cumulative_friction_work_J
                      + self.cumulative_rolling_work_J
                      + self.cumulative_platen_damping_J
                      + self.cumulative_bond_dissipation_J)
        input_work = (self.cumulative_tool_work_J + self.cumulative_external_work_J
                      + self.cumulative_platen_work_J)
        # ``dissipated`` is shared by both arms: the dissipative channels are
        # still integrated with left-endpoint rates (see the doc for the
        # measured size of that term).
        input_work_exact = (self.cumulative_tool_work_exact_J
                            + self.cumulative_external_work_exact_J
                            + self.cumulative_platen_work_exact_J)
        self.energy_balance_residual_J = (
            energy_after - self.reference_mechanical_energy_J - input_work
            + dissipated + self.cumulative_fracture_release_J
        )
        self.energy_balance_residual_exact_J = (
            energy_after - self.reference_mechanical_energy_J - input_work_exact
            + dissipated + self.cumulative_fracture_release_J
        )

    def _platen_work(self) -> float:
        """Instantaneous power (W) delivered to the specimen by all platens."""
        cfg = self.config
        power = 0.0
        if cfg.top_platen_enabled:
            power += float(-self.last_top_platen_reaction_z) * cfg.top_platen_velocity
        if cfg.bottom_platen_enabled:
            power += float(-self.last_bottom_platen_reaction_z) * cfg.bottom_platen_velocity
        if cfg.lateral_platens:
            for idx in range(4):
                power += float(-self.last_lateral_platen_reaction[idx]) \
                    * self.lateral_platen_velocity[idx]
        return power

    def _platen_damping_power(self) -> float:
        """Instantaneous dashpot dissipation (W, >= 0) in all platen contacts."""
        cfg = self.config
        power = 0.0
        if cfg.top_platen_enabled:
            top_z = self.top_platen_start_z + self.time * cfg.top_platen_velocity
            top_overlap = (self.positions[:, 2] + cfg.radius - top_z).clamp_min(0.0)
            rel = self.velocities[:, 2] - cfg.top_platen_velocity
            power += cfg.platen_damping * float((rel ** 2)[top_overlap > 0].sum())
        if cfg.bottom_platen_enabled:
            bottom_z = self.bottom_platen_start_z + self.time * cfg.bottom_platen_velocity
            bottom_overlap = (bottom_z + cfg.radius - self.positions[:, 2]).clamp_min(0.0)
            rel = self.velocities[:, 2] - cfg.bottom_platen_velocity
            power += cfg.platen_damping * float((rel ** 2)[bottom_overlap > 0].sum())
        if cfg.lateral_platens:
            lat_c = (cfg.lateral_platen_damping if cfg.lateral_platen_damping is not None
                     else cfg.platen_damping)
            px, py = self.positions[:, 0], self.positions[:, 1]
            vx, vy = self.velocities[:, 0], self.velocities[:, 1]
            ov0 = (self.lateral_platen_pos[0] - px + cfg.radius).clamp_min(0.0)
            power += lat_c * float(((vx - self.lateral_platen_velocity[0]) ** 2)[ov0 > 0].sum())
            ov1 = (px + cfg.radius - self.lateral_platen_pos[1]).clamp_min(0.0)
            power += lat_c * float(((vx - self.lateral_platen_velocity[1]) ** 2)[ov1 > 0].sum())
            ov2 = (self.lateral_platen_pos[2] - py + cfg.radius).clamp_min(0.0)
            power += lat_c * float(((vy - self.lateral_platen_velocity[2]) ** 2)[ov2 > 0].sum())
            ov3 = (py + cfg.radius - self.lateral_platen_pos[3]).clamp_min(0.0)
            power += lat_c * float(((vy - self.lateral_platen_velocity[3]) ** 2)[ov3 > 0].sum())
        return power

    @torch.no_grad()
    def diagnostics(self, external_forces: torch.Tensor | None = None) -> dict[str, float | int]:
        row = super().diagnostics(external_forces)
        # Base diagnostics evaluates forces and can detect damage at the sampled
        # configuration; synchronize the release ledger with that authoritative state.
        self.cumulative_fracture_release_J = float(self.failure_energy_J.sum())
        # Recompute the balance from the ledger (the step-owner's residual is the
        # authoritative one; this keeps a probe consistent with the last step).
        dissipated = (self.cumulative_drag_dissipation_J
                      + self.cumulative_particle_damping_J
                      + self.cumulative_tool_damping_J
                      + self.cumulative_friction_work_J
                      + self.cumulative_rolling_work_J
                      + self.cumulative_platen_damping_J
                      + self.cumulative_bond_dissipation_J)
        input_work = (self.cumulative_tool_work_J + self.cumulative_external_work_J
                      + self.cumulative_platen_work_J)
        # ``dissipated`` is shared by both arms: the dissipative channels are
        # still integrated with left-endpoint rates (see the doc for the
        # measured size of that term).
        input_work_exact = (self.cumulative_tool_work_exact_J
                            + self.cumulative_external_work_exact_J
                            + self.cumulative_platen_work_exact_J)
        self.energy_balance_residual_J = (
            self._mechanical_energy() - self.reference_mechanical_energy_J - input_work
            + dissipated + self.cumulative_fracture_release_J
        )
        self.energy_balance_residual_exact_J = (
            self._mechanical_energy() - self.reference_mechanical_energy_J - input_work_exact
            + dissipated + self.cumulative_fracture_release_J
        )
        row.update({
            "tool_work_cumulative_J": self.cumulative_tool_work_J,
            "external_work_cumulative_J": self.cumulative_external_work_J,
            "drag_dissipation_cumulative_J": self.cumulative_drag_dissipation_J,
            "particle_damping_cumulative_J": self.cumulative_particle_damping_J,
            "tool_damping_cumulative_J": self.cumulative_tool_damping_J,
            "fracture_release_cumulative_J": self.cumulative_fracture_release_J,
            "friction_work_cumulative_J": self.cumulative_friction_work_J,
            "rolling_work_cumulative_J": self.cumulative_rolling_work_J,
            "energy_balance_residual_J": self.energy_balance_residual_J,
            "energy_balance_residual_relative": self.energy_balance_residual_J / max(
                abs(self.reference_mechanical_energy_J)
                + abs(self.cumulative_tool_work_J)
                + abs(self.cumulative_external_work_J)
                + abs(self.cumulative_platen_work_J)
                + self.cumulative_fracture_release_J, 1e-30
            ),
            "platen_work_cumulative_J": self.cumulative_platen_work_J,
            "platen_damping_cumulative_J": self.cumulative_platen_damping_J,
            "bond_dissipation_cumulative_J": self.cumulative_bond_dissipation_J,
            "platen_work_exact_cumulative_J": self.cumulative_platen_work_exact_J,
            "tool_work_exact_cumulative_J": self.cumulative_tool_work_exact_J,
            "external_work_exact_cumulative_J": self.cumulative_external_work_exact_J,
            "energy_balance_residual_exact_J": self.energy_balance_residual_exact_J,
            "platen_deficit_predicted_cumulative_J": self.cumulative_platen_deficit_predicted_J,
            "platen_deficit_measured_cumulative_J": (
                self.cumulative_platen_work_exact_J - self.cumulative_platen_work_J),
        })
        return row

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["energy_audit"] = {
            "reference_mechanical_energy_J": self.reference_mechanical_energy_J,
            "cumulative_tool_work_J": self.cumulative_tool_work_J,
            "cumulative_external_work_J": self.cumulative_external_work_J,
            "cumulative_drag_dissipation_J": self.cumulative_drag_dissipation_J,
            "cumulative_particle_damping_J": self.cumulative_particle_damping_J,
            "cumulative_tool_damping_J": self.cumulative_tool_damping_J,
            "cumulative_fracture_release_J": self.cumulative_fracture_release_J,
            "cumulative_friction_work_J": self.cumulative_friction_work_J,
            "cumulative_rolling_work_J": self.cumulative_rolling_work_J,
            "cumulative_bond_dissipation_J": self.cumulative_bond_dissipation_J,
            "energy_balance_residual_J": self.energy_balance_residual_J,
            "cumulative_platen_work_exact_J": self.cumulative_platen_work_exact_J,
            "cumulative_tool_work_exact_J": self.cumulative_tool_work_exact_J,
            "cumulative_external_work_exact_J": self.cumulative_external_work_exact_J,
            "energy_balance_residual_exact_J": self.energy_balance_residual_exact_J,
            "cumulative_platen_deficit_predicted_J":
                self.cumulative_platen_deficit_predicted_J,
        }
        return state

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        audit = state.get("energy_audit")
        if audit is None:
            # Older base-solver checkpoints remain loadable; their past work
            # cannot be reconstructed, so start a clearly new accounting epoch.
            self.reference_mechanical_energy_J = self._mechanical_energy()
            self.cumulative_tool_work_J = 0.0
            self.cumulative_external_work_J = 0.0
            self.cumulative_drag_dissipation_J = 0.0
            self.cumulative_particle_damping_J = 0.0
            self.cumulative_tool_damping_J = 0.0
            self.cumulative_fracture_release_J = float(self.failure_energy_J.sum())
            self.cumulative_friction_work_J = 0.0
            self.cumulative_rolling_work_J = 0.0
            self.energy_balance_residual_J = 0.0
            self.cumulative_platen_deficit_predicted_J = 0.0
            return
        keys = (
            "reference_mechanical_energy_J", "cumulative_tool_work_J",
            "cumulative_external_work_J", "cumulative_drag_dissipation_J",
            "cumulative_particle_damping_J", "cumulative_tool_damping_J",
            "cumulative_fracture_release_J", "cumulative_friction_work_J",
            "cumulative_rolling_work_J", "cumulative_bond_dissipation_J",
            "energy_balance_residual_J", "cumulative_platen_work_exact_J",
            "cumulative_tool_work_exact_J", "cumulative_external_work_exact_J",
            "energy_balance_residual_exact_J",
            "cumulative_platen_deficit_predicted_J",
        )
        staged = {key: float(audit.get(key, 0.0)) for key in keys}
        if not all(torch.isfinite(torch.tensor(value)) for value in staged.values()):
            raise ValueError("checkpoint energy audit contains non-finite values")
        for key, value in staged.items():
            setattr(self, key, value)
