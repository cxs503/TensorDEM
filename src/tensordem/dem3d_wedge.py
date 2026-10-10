"""Prescribed moving wedge-bow contact for 3-D bonded-particle ice DEM.

The bow is an idealized 2-D wedge extruded uniformly across the y direction.
It is intended for engineering workflow development and sensitivity studies,
not as a validated ship-hull hydrodynamics or ice-resistance model.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from .dem3d import DEM3DConfig, IceDEM3D


class MovingWedgeBowIceDEM3D(IceDEM3D):
    """DEM3D with a rigid, horizontally translating wedge-shaped bow contact."""

    def __init__(
        self,
        config: DEM3DConfig,
        *,
        bow_speed: float = 0.2,
        wedge_angle_deg: float = 45.0,
        initial_gap: float = 0.0,
        tip_height_fraction: float = 0.5,
    ) -> None:
        for name, value in (
            ("bow_speed", bow_speed),
            ("wedge_angle_deg", wedge_angle_deg),
            ("initial_gap", initial_gap),
            ("tip_height_fraction", tip_height_fraction),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if bow_speed <= 0:
            raise ValueError("bow_speed must be positive")
        if not 5.0 <= wedge_angle_deg <= 85.0:
            raise ValueError("wedge_angle_deg must be in [5, 85]")
        if initial_gap < 0:
            raise ValueError("initial_gap must be nonnegative")
        if not 0.0 <= tip_height_fraction <= 1.0:
            raise ValueError("tip_height_fraction must be in [0, 1]")
        super().__init__(config)
        self.bow_speed = float(bow_speed)
        self.wedge_angle_deg = float(wedge_angle_deg)
        self.initial_gap = float(initial_gap)
        self.tip_height_fraction = float(tip_height_fraction)
        self.bow_start_x = float(self.initial_positions[:, 0].min()) - 3.0 * config.radius - initial_gap
        z_min = float(self.initial_positions[:, 2].min())
        z_max = float(self.initial_positions[:, 2].max())
        self.bow_tip_z = z_min + tip_height_fraction * (z_max - z_min)
        # The inherited spherical indenter is disabled without changing the
        # base configuration, preserving old checkpoints and solver semantics.
        self.tool_start = torch.tensor([1.0e9, 1.0e9, 1.0e9], dtype=self.dtype, device=self.device)
        self.tool_velocity.zero_()

    @property
    def bow_tip_x(self) -> float:
        return self.bow_start_x + self.bow_speed * self.time

    @torch.no_grad()
    def _wedge_contact(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return nodal bow forces, bow reaction, and per-particle overlap."""
        theta = math.radians(self.wedge_angle_deg)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        dx = self.positions[:, 0] - self.bow_tip_x
        dz = self.positions[:, 2] - self.bow_tip_z
        side = torch.where(dz >= 0, 1.0, -1.0)
        signed_distance = sin_t * dx + cos_t * dz.abs()
        overlap = (self.config.radius - signed_distance).clamp_min(0.0)
        normal = torch.zeros_like(self.positions)
        normal[:, 0] = sin_t
        normal[:, 2] = side * cos_t
        relative_velocity = self.velocities.clone()
        relative_velocity[:, 0] -= self.bow_speed
        normal_speed = (relative_velocity * normal).sum(dim=1)
        magnitude = (
            self.config.contact_stiffness * overlap
            - self.config.contact_damping * normal_speed
        ).clamp_min(0.0)
        magnitude = torch.where(overlap > 0.0, magnitude, torch.zeros_like(magnitude))
        nodal_force = magnitude[:, None] * normal
        reaction = -nodal_force.sum(dim=0)
        return nodal_force, reaction, overlap

    @torch.no_grad()
    def forces(
        self,
        external_forces: torch.Tensor | None = None,
        *,
        update_fracture: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        force, _ = super().forces(external_forces, update_fracture=update_fracture)
        bow_force, bow_reaction, _ = self._wedge_contact()
        force += bow_force
        self.last_reaction = bow_reaction
        return force, bow_reaction

    @torch.no_grad()
    def diagnostics(self, external_forces: torch.Tensor | None = None) -> dict[str, float | int]:
        result = super().diagnostics(external_forces)
        _, reaction, overlap = self._wedge_contact()
        bow_energy = 0.5 * self.config.contact_stiffness * overlap.square().sum()
        result.update({
            "bow_tip_x_m": self.bow_tip_x,
            "bow_tip_z_m": self.bow_tip_z,
            "bow_speed_m_s": self.bow_speed,
            "bow_reaction_x_N": float(reaction[0]),
            "bow_reaction_y_N": float(reaction[1]),
            "bow_reaction_z_N": float(reaction[2]),
            "bow_contact_energy_J": float(bow_energy),
            # The compatibility field is total rigid-boundary contact energy;
            # the explicit bow_contact_energy_J field disambiguates this model.
            "tool_contact_energy_J": float(bow_energy),
        })
        return result
