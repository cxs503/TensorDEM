"""Energy-accounting extension for the 3-D bonded-sphere DEM.

The base solver remains backward compatible. This audited subclass tracks work,
viscous dissipation, and the solver's existing bond-release estimate. The
balance residual is a discrete diagnostic, not a claim of exact conservation.
"""
from __future__ import annotations

from typing import Any

import torch

from .dem3d import DEM3DConfig, IceDEM3D


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
        bond = (0.5 * cfg.bond_stiffness * extension.square() * self.alive).sum()

        pairs = self._contact_candidate_pairs()
        particle_contact = torch.zeros((), dtype=self.dtype, device=self.device)
        if pairs.numel():
            ci, cj = pairs.T
            d = torch.linalg.vector_norm(self.positions[cj] - self.positions[ci], dim=1)
            overlap = (2 * cfg.radius - d).clamp_min(0)
            eligible = ~self._alive_bond_for_pairs(pairs)
            particle_contact = (0.5 * cfg.contact_stiffness * overlap.square() * eligible).sum()

        tool_distance = torch.linalg.vector_norm(self.positions - self.tool_position, dim=1)
        tool_overlap = (cfg.radius + cfg.tool_radius - tool_distance).clamp_min(0)
        tool_contact = (0.5 * cfg.contact_stiffness * tool_overlap.square()).sum()
        return float(kinetic + bond + particle_contact + tool_contact)

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

        self.velocities += self.dt * force / self.config.mass
        self.velocities[self.fixed] = 0
        self.positions += self.dt * self.velocities
        self.positions[self.fixed] = self.initial_positions[self.fixed]
        self.time += self.dt
        self.step_count += 1
        if not bool(torch.isfinite(self.positions).all() and torch.isfinite(self.velocities).all()):
            raise FloatingPointError("non-finite DEM3D state; reduce dt or contact stiffness")

        energy_after = self._mechanical_energy()
        dissipated = (self.cumulative_drag_dissipation_J
                      + self.cumulative_particle_damping_J
                      + self.cumulative_tool_damping_J)
        input_work = self.cumulative_tool_work_J + self.cumulative_external_work_J
        self.energy_balance_residual_J = (
            energy_after - self.reference_mechanical_energy_J - input_work
            + dissipated + self.cumulative_fracture_release_J
        )

    @torch.no_grad()
    def diagnostics(self, external_forces: torch.Tensor | None = None) -> dict[str, float | int]:
        row = super().diagnostics(external_forces)
        # Base diagnostics evaluates forces and can detect damage at the sampled
        # configuration; synchronize the release ledger with that authoritative state.
        self.cumulative_fracture_release_J = float(self.failure_energy_J.sum())
        energy = self._mechanical_energy()
        dissipated = (self.cumulative_drag_dissipation_J
                      + self.cumulative_particle_damping_J
                      + self.cumulative_tool_damping_J)
        input_work = self.cumulative_tool_work_J + self.cumulative_external_work_J
        self.energy_balance_residual_J = (
            energy - self.reference_mechanical_energy_J - input_work
            + dissipated + self.cumulative_fracture_release_J
        )
        row.update({
            "tool_work_cumulative_J": self.cumulative_tool_work_J,
            "external_work_cumulative_J": self.cumulative_external_work_J,
            "drag_dissipation_cumulative_J": self.cumulative_drag_dissipation_J,
            "particle_damping_cumulative_J": self.cumulative_particle_damping_J,
            "tool_damping_cumulative_J": self.cumulative_tool_damping_J,
            "fracture_release_cumulative_J": self.cumulative_fracture_release_J,
            "energy_balance_residual_J": self.energy_balance_residual_J,
            "energy_balance_residual_relative": self.energy_balance_residual_J / max(
                abs(self.reference_mechanical_energy_J)
                + abs(self.cumulative_tool_work_J)
                + abs(self.cumulative_external_work_J)
                + self.cumulative_fracture_release_J, 1e-30
            ),
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
            "energy_balance_residual_J": self.energy_balance_residual_J,
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
            self.energy_balance_residual_J = 0.0
            return
        keys = (
            "reference_mechanical_energy_J", "cumulative_tool_work_J",
            "cumulative_external_work_J", "cumulative_drag_dissipation_J",
            "cumulative_particle_damping_J", "cumulative_tool_damping_J",
            "cumulative_fracture_release_J", "energy_balance_residual_J",
        )
        staged = {key: float(audit[key]) for key in keys}
        if not all(torch.isfinite(torch.tensor(value)) for value in staged.values()):
            raise ValueError("checkpoint energy audit contains non-finite values")
        for key, value in staged.items():
            setattr(self, key, value)
