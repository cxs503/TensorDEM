"""Two-dimensional central-force bonded disks, using SI units."""

from dataclasses import dataclass
import math

import torch


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
            "contact_stiffness", "breaking_strain", "tool_radius", "tool_speed",
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

    Positions and velocities have shape (N, 2). Bond and contact forces are
    central and equal/opposite. There are no rotational or shear degrees of
    freedom. Force evaluation irreversibly updates bond damage.
    """

    def __init__(self, config: DEMConfig = DEMConfig()) -> None:
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

    @property
    def tool_position(self) -> torch.Tensor:
        return self.tool_start + self.time * self.tool_velocity

    @property
    def broken_bonds(self) -> int:
        return int((self.bonded & ~self.alive).sum().item())

    @torch.no_grad()
    def forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return particle forces and ice-on-tool reaction at current time."""
        cfg = self.config
        i, j = self.pairs.T
        delta = self.positions[j] - self.positions[i]
        distance = torch.linalg.vector_norm(delta, dim=1)
        normal = delta / distance.clamp_min(torch.finfo(delta.dtype).eps)[:, None]
        normal = torch.where((distance > 0)[:, None], normal, self.initial_normals)
        extension = distance - self.rest_lengths
        self.alive &= extension / self.rest_lengths <= cfg.breaking_strain
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
        force += tool_force
        return force, -tool_force.sum(dim=0)

    @torch.no_grad()
    def step(self) -> None:
        """Semi-implicit Euler with hard constraints on the edge particles."""
        force, _ = self.forces()
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
    def diagnostics(self) -> dict[str, float | int]:
        _, reaction = self.forces()
        return {
            "time": self.time,
            "tool_y": float(self.tool_position[1].item()),
            "reaction_x": float(reaction[0].item()),
            "reaction_y": float(reaction[1].item()),
            "broken_bonds": self.broken_bonds,
            "kinetic_energy": float(
                (0.5 * self.config.mass * self.velocities.square().sum()).item()
            ),
        }
