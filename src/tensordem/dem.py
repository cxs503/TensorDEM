"""Two-dimensional central-force bonded disks, using SI units."""

from dataclasses import dataclass
import math

import torch

from .hull_contact import polyline_contact_forces


@dataclass(frozen=True)
class DEMConfig:
    nx: int = 21
    ny: int = 5
    radius: float = 0.025
    thickness: float = 0.2
    density: float = 917.0
    bond_stiffness: float = 2_000.0
    contact_stiffness: float = 2_000.0
    breaking_strain: float = 0.015
    shear_breaking_strain: float = 0.03
    contact_damping: float = 5.0
    drag: float = 2.0
    tool_radius: float = 0.1
    tool_speed: float = 1.0
    tool_gap: float = 0.01
    dt: float | None = None
    device: str = "cpu"
    fix_edges: bool = True

    def __post_init__(self) -> None:
        for name in ("nx", "ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        for name in (
            "radius", "thickness", "density", "bond_stiffness",
            "contact_stiffness", "breaking_strain", "shear_breaking_strain",
            "tool_radius", "tool_speed",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("contact_damping", "drag", "tool_gap"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0):
            raise ValueError("dt must be finite and positive")

    @property
    def mass(self) -> float:
        return self.density * math.pi * self.radius**2 * self.thickness

    @property
    def recommended_dt(self) -> float:
        # Bound all possible pair contacts plus the moving tool, and dashpots.
        neighbors = self.nx * self.ny - 1
        stiffness = (2 * neighbors + 1) * max(
            self.bond_stiffness, self.contact_stiffness
        )
        damping = (2 * neighbors + 1) * self.contact_damping + self.drag
        elastic_dt = 0.2 * math.sqrt(self.mass / stiffness)
        damping_dt = 0.2 * self.mass / damping if damping else math.inf
        travel_dt = 0.1 * self.radius / self.tool_speed
        return min(elastic_dt, damping_dt, travel_dt)


class IceDEM:
    """A small ice lattice; only tensile axial strain breaks bonds.

    Positions, velocities and externally applied forces have shape (N, 2).
    External forces use SI newtons in the global x/y frame and are evaluated
    at the current step (not accumulated internally). Bond and contact forces
    are central and equal/opposite. Tensile failure uses axial bond strain;
    shear failure uses an objective local Green-Lagrange strain invariant.
    The model still has no particle rotations or explicit shear-bond moments.
    """

    def __init__(
        self,
        config: DEMConfig = DEMConfig(),
        *,
        hull_profile: torch.Tensor | None = None,
        hull_start: tuple[float, float] = (0.0, 0.0),
        hull_velocity: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA devices are supported")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.dt = config.dt if config.dt is not None else config.recommended_dt
        if self.dt > config.recommended_dt:
            raise ValueError(
                f"dt exceeds conservative limit {config.recommended_dt:.6g} s"
            )
        x = torch.arange(config.nx, dtype=torch.float64, device=self.device)
        y = torch.arange(config.ny, dtype=torch.float64, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.positions = torch.stack(
            (xx.flatten() - (config.nx - 1) / 2, yy.flatten()), dim=1
        ) * (2 * config.radius)
        self.initial_positions = self.positions.clone()
        self.velocities = torch.zeros_like(self.positions)
        self.fixed = torch.zeros(len(self.positions), dtype=torch.bool, device=self.device)
        if config.fix_edges:
            self.fixed = (xx.flatten() == 0) | (xx.flatten() == config.nx - 1)
        self.pairs = torch.triu_indices(
            len(self.positions), len(self.positions), offset=1, device=self.device
        ).T
        delta = self.positions[self.pairs[:, 1]] - self.positions[self.pairs[:, 0]]
        distances = torch.linalg.vector_norm(delta, dim=1)
        self.initial_normals = delta / distances[:, None]
        # Square-grid nearest neighbors and diagonals form the bonded lattice.
        self.bonded = distances <= 2 * config.radius * math.sqrt(2) * (1 + 1e-10)
        self.bond_indices = self.pairs[self.bonded]
        self.reference_bond_vectors = (
            self.initial_positions[self.bond_indices[:, 1]]
            - self.initial_positions[self.bond_indices[:, 0]]
        )
        self.rest_lengths = distances.clone()
        self.alive = self.bonded.clone()
        self.time = 0.0
        self.tool_start = torch.tensor(
            [0.0, (config.ny - 1) * 2 * config.radius
             + config.radius + config.tool_radius + config.tool_gap],
            dtype=torch.float64, device=self.device,
        )
        self.tool_velocity = torch.tensor(
            [0.0, -config.tool_speed], dtype=torch.float64, device=self.device
        )
        self.hull_profile = None
        self.hull_start = torch.tensor(hull_start, dtype=torch.float64, device=self.device)
        self.hull_velocity = torch.tensor(hull_velocity, dtype=torch.float64, device=self.device)
        if not bool(torch.isfinite(self.hull_start).all() and torch.isfinite(self.hull_velocity).all()):
            raise ValueError("hull_start and hull_velocity must be finite")
        if hull_profile is not None:
            if not isinstance(hull_profile, torch.Tensor):
                raise TypeError("hull_profile must be a torch.Tensor with shape (M, 2)")
            if hull_profile.ndim != 2 or hull_profile.shape[1] != 2 or hull_profile.shape[0] < 2:
                raise ValueError("hull_profile must have shape (M, 2), M >= 2")
            self.hull_profile = hull_profile.to(device=self.device, dtype=torch.float64).clone()
            if not bool(torch.isfinite(self.hull_profile).all()):
                raise ValueError("hull_profile must contain only finite values")
            if bool((torch.linalg.vector_norm(self.hull_profile[1:] - self.hull_profile[:-1], dim=1)
                     <= torch.finfo(self.hull_profile.dtype).eps).any()):
                raise ValueError("hull_profile must not contain zero-length segments")

    @property
    def tool_position(self) -> torch.Tensor:
        """Return the circular tool position or prescribed hull translation."""
        if self.hull_profile is not None:
            return self.hull_start + self.time * self.hull_velocity
        return self.tool_start + self.time * self.tool_velocity

    @property
    def broken_bonds(self) -> int:
        return int((self.bonded & ~self.alive).sum().item())

    @torch.no_grad()
    def _local_objective_shear_strain(self) -> torch.Tensor:
        """Return a rotation-invariant local shear strain estimate per particle.

        A least-squares deformation gradient is reconstructed from the initial
        bonded neighborhood. The Green-Lagrange strain removes rigid rotation;
        the equivalent in-plane shear measure is sqrt((E_xx-E_yy)^2 + 4 E_xy^2).
        """
        n = len(self.positions)
        i, j = self.bond_indices.T
        reference = self.reference_bond_vectors
        current = self.positions[j] - self.positions[i]
        reference_outer = reference[:, :, None] * reference[:, None, :]
        current_reference = current[:, :, None] * reference[:, None, :]
        covariance = torch.zeros((n, 2, 2), dtype=self.positions.dtype, device=self.device)
        cross = torch.zeros_like(covariance)
        covariance.index_add_(0, i, reference_outer)
        covariance.index_add_(0, j, reference_outer)
        cross.index_add_(0, i, current_reference)
        cross.index_add_(0, j, current_reference)
        deformation = cross @ torch.linalg.pinv(covariance)
        identity = torch.eye(2, dtype=self.positions.dtype, device=self.device)
        strain = 0.5 * (deformation.transpose(1, 2) @ deformation - identity)
        return torch.sqrt(
            (strain[:, 0, 0] - strain[:, 1, 1]).square()
            + 4.0 * strain[:, 0, 1].square()
        )

    def _validate_external_forces(
        self, external_forces: torch.Tensor | None
    ) -> torch.Tensor | None:
        if external_forces is None:
            return None
        if not isinstance(external_forces, torch.Tensor):
            raise TypeError("external_forces must be a torch.Tensor with shape (N, 2)")
        if tuple(external_forces.shape) != tuple(self.positions.shape):
            raise ValueError(
                f"external_forces must have shape {tuple(self.positions.shape)}"
            )
        loads = external_forces.to(device=self.device, dtype=self.positions.dtype)
        if not bool(torch.isfinite(loads).all()):
            raise ValueError("external_forces must contain only finite values")
        return loads

    @torch.no_grad()
    def forces(
        self, external_forces: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return total particle force and ice-on-tool reaction.

        external_forces is an optional instantaneous nodal load array in N.
        It is additive to DEM bond/contact forces and linear drag. The returned
        tool reaction retains its existing sign convention (ice on tool).
        """
        cfg = self.config
        loads = self._validate_external_forces(external_forces)
        i, j = self.pairs.T
        delta = self.positions[j] - self.positions[i]
        distance = torch.linalg.vector_norm(delta, dim=1)
        normal = delta / distance.clamp_min(torch.finfo(delta.dtype).eps)[:, None]
        normal = torch.where((distance > 0)[:, None], normal, self.initial_normals)
        extension = distance - self.rest_lengths
        tensile_strain = extension / self.rest_lengths
        local_shear = self._local_objective_shear_strain()
        i_bond, j_bond = self.bond_indices.T
        bond_shear = 0.5 * (local_shear[i_bond] + local_shear[j_bond])
        shear_failed = bond_shear > cfg.shear_breaking_strain
        tensile_failed = tensile_strain[self.bonded] > cfg.breaking_strain
        self.alive[self.bonded] &= ~(tensile_failed | shear_failed)
        normal_speed = ((self.velocities[j] - self.velocities[i]) * normal).sum(dim=1)
        overlap = (2 * cfg.radius - distance).clamp_min(0)
        repulsion = (
            cfg.contact_stiffness * overlap - cfg.contact_damping * normal_speed
        ).clamp_min(0)
        repulsion = torch.where(
            (overlap > 0) & ~self.alive, repulsion, torch.zeros_like(repulsion)
        )
        magnitude = torch.where(
            self.alive, cfg.bond_stiffness * extension, -repulsion
        )
        pair_force = magnitude[:, None] * normal
        force = -cfg.drag * self.velocities
        force.index_add_(0, i, pair_force)
        force.index_add_(0, j, -pair_force)

        if self.hull_profile is None:
            tool_delta = self.positions - self.tool_position
            tool_distance = torch.linalg.vector_norm(tool_delta, dim=1)
            tool_normal = tool_delta / tool_distance.clamp_min(
                torch.finfo(tool_delta.dtype).eps
            )[:, None]
            fallback = torch.zeros_like(tool_normal)
            fallback[:, 1] = -1
            tool_normal = torch.where((tool_distance > 0)[:, None], tool_normal, fallback)
            tool_overlap = (cfg.radius + cfg.tool_radius - tool_distance).clamp_min(0)
            tool_speed = ((self.velocities - self.tool_velocity) * tool_normal).sum(dim=1)
            tool_magnitude = (
                cfg.contact_stiffness * tool_overlap - cfg.contact_damping * tool_speed
            ).clamp_min(0)
            tool_magnitude = torch.where(
                tool_overlap > 0, tool_magnitude, torch.zeros_like(tool_magnitude)
            )
            tool_force = tool_magnitude[:, None] * tool_normal
            tool_reaction = -tool_force.sum(dim=0)
        else:
            vertices = self.hull_profile + self.tool_position
            tool_force, tool_reaction = polyline_contact_forces(
                self.positions,
                self.velocities,
                vertices,
                self.hull_velocity,
                particle_radius=cfg.radius,
                hull_radius=cfg.tool_radius,
                stiffness=cfg.contact_stiffness,
                damping=cfg.contact_damping,
            )
        force += tool_force
        if loads is not None:
            force += loads
        return force, tool_reaction

    @torch.no_grad()
    def step(self, external_forces: torch.Tensor | None = None) -> None:
        """Semi-implicit Euler; optional nodal loads are supplied in newtons."""
        force, _ = self.forces(external_forces=external_forces)
        self.velocities += self.dt * force / self.config.mass
        self.velocities[self.fixed] = 0
        self.positions += self.dt * self.velocities
        self.positions[self.fixed] = self.initial_positions[self.fixed]
        self.time += self.dt
        if not bool(torch.isfinite(self.positions).all()) or not bool(
            torch.isfinite(self.velocities).all()
        ):
            raise FloatingPointError("Non-finite simulation state; reduce dt")

    @torch.no_grad()
    def diagnostics(
        self, external_forces: torch.Tensor | None = None
    ) -> dict[str, float | int]:
        force, reaction = self.forces(external_forces=external_forces)
        boundary_reaction = -force[self.fixed].sum(dim=0)
        return {
            "time": self.time,
            "tool_y": float(self.tool_position[1].item()),
            "reaction_x": float(reaction[0].item()),
            "reaction_y": float(reaction[1].item()),
            "boundary_reaction_x": float(boundary_reaction[0].item()),
            "boundary_reaction_y": float(boundary_reaction[1].item()),
            "broken_bonds": self.broken_bonds,
            "kinetic_energy": float(
                (0.5 * self.config.mass * self.velocities.square().sum()).item()
            ),
        }
