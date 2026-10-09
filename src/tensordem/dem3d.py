"""Three-dimensional bonded-sphere DEM engine for ice fracture.

All quantities use SI units. This first 3-D solver is deliberately independent of
the established 2-D implementation so existing cases and output formats remain
unchanged. The lattice is a simple-cubic network with face/body diagonals inside
the initial cutoff; calibrate its directional stiffness before engineering use.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class DEM3DConfig:
    """Numerical and physical parameters for the 3-D bonded-sphere model."""
    nx: int = 9
    ny: int = 5
    nz: int = 3
    radius: float = 0.025
    density: float = 917.0
    bond_stiffness: float = 2_000.0
    contact_stiffness: float = 5_000.0
    breaking_strain: float = 0.015
    shear_breaking_strain: float = 0.03
    contact_damping: float = 5.0
    drag: float = 2.0
    tool_radius: float = 0.1
    tool_speed: float = 0.5
    tool_gap: float = 0.01
    dt: float | None = None
    device: str = "cpu"
    fix_x_edges: bool = True
    fix_bottom: bool = False

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        for name in ("radius", "density", "bond_stiffness", "contact_stiffness",
                     "breaking_strain", "shear_breaking_strain", "tool_radius", "tool_speed"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("contact_damping", "drag", "tool_gap"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0):
            raise ValueError("dt must be finite and positive")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")

    @property
    def mass(self) -> float:
        return self.density * (4.0 / 3.0) * math.pi * self.radius**3

    @property
    def recommended_dt(self) -> float:
        # Conservative explicit stability/travel bound; validate by convergence.
        n = self.nx * self.ny * self.nz
        degree_bound = min(n - 1, 26)
        k = (2 * degree_bound + 1) * max(self.bond_stiffness, self.contact_stiffness)
        c = (2 * degree_bound + 1) * self.contact_damping + self.drag
        elastic = 0.15 * math.sqrt(self.mass / k)
        damping = 0.15 * self.mass / c if c > 0 else math.inf
        travel = 0.1 * self.radius / self.tool_speed
        return min(elastic, damping, travel)


class IceDEM3D:
    """3-D bonded-sphere DEM with spherical prescribed indenter.

    State arrays have shape (N, 3); external forces are instantaneous nodal
    loads in newtons. The indenter translates downward at constant speed.
    Bond failure is irreversible. Pair interactions are currently O(N^2), so
    this engine is intended for verification and moderate-sized prototypes.
    """

    def __init__(self, config: DEM3DConfig = DEM3DConfig()) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.dtype = torch.float64
        self.dt = config.dt if config.dt is not None else config.recommended_dt
        if self.dt > config.recommended_dt:
            raise ValueError(f"dt exceeds conservative limit {config.recommended_dt:.6g} s")

        axes = [torch.arange(v, dtype=self.dtype, device=self.device) for v in
                (config.nx, config.ny, config.nz)]
        zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        grid = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
        self.positions = (grid - torch.tensor(
            [(config.nx - 1) / 2, (config.ny - 1) / 2, 0.0],
            dtype=self.dtype, device=self.device
        )) * (2 * config.radius)
        self.initial_positions = self.positions.clone()
        self.velocities = torch.zeros_like(self.positions)
        self.fixed = torch.zeros(len(grid), dtype=torch.bool, device=self.device)
        if config.fix_x_edges:
            self.fixed |= (xx.reshape(-1) == 0) | (xx.reshape(-1) == config.nx - 1)
        if config.fix_bottom:
            self.fixed |= zz.reshape(-1) == 0

        self.pairs = torch.triu_indices(len(grid), len(grid), offset=1, device=self.device).T
        i, j = self.pairs.T
        ref = self.initial_positions[j] - self.initial_positions[i]
        self.rest_lengths = torch.linalg.vector_norm(ref, dim=1)
        cutoff = 2 * config.radius * math.sqrt(3) * (1 + 1e-10)
        self.bonded = self.rest_lengths <= cutoff
        self.alive = self.bonded.clone()
        self.bond_indices = self.pairs[self.bonded]
        self.reference_bond_vectors = ref[self.bonded].clone()
        self.initial_normals = ref / self.rest_lengths.clamp_min(torch.finfo(self.dtype).eps)[:, None]
        self.failure_mode = torch.zeros(len(self.pairs), dtype=torch.uint8, device=self.device)
        self.failure_time_s = torch.full((len(self.pairs),), -1.0, dtype=self.dtype, device=self.device)
        self.failure_extension_m = torch.zeros(len(self.pairs), dtype=self.dtype, device=self.device)
        self.failure_energy_J = torch.zeros(len(self.pairs), dtype=self.dtype, device=self.device)

        self.time = 0.0
        self.step_count = 0
        self.tool_start = torch.tensor(
            [0.0, 0.0, (config.nz - 1) * 2 * config.radius + config.radius
             + config.tool_radius + config.tool_gap],
            dtype=self.dtype, device=self.device)
        self.tool_velocity = torch.tensor([0.0, 0.0, -config.tool_speed],
                                          dtype=self.dtype, device=self.device)
        self.last_reaction = torch.zeros(3, dtype=self.dtype, device=self.device)

    @property
    def tool_position(self) -> torch.Tensor:
        return self.tool_start + self.time * self.tool_velocity

    @property
    def broken_bonds(self) -> int:
        return int((self.bonded & ~self.alive).sum().item())

    @torch.no_grad()
    def _objective_shear_strain(self) -> torch.Tensor:
        """Least-squares local F and Green-Lagrange strain invariant in 3-D."""
        n = len(self.positions)
        i, j = self.bond_indices.T
        r = self.reference_bond_vectors
        q = self.positions[j] - self.positions[i]
        rr = r[:, :, None] * r[:, None, :]
        qr = q[:, :, None] * r[:, None, :]
        cov = torch.zeros((n, 3, 3), dtype=self.dtype, device=self.device)
        cross = torch.zeros_like(cov)
        cov.index_add_(0, i, rr); cov.index_add_(0, j, rr)
        cross.index_add_(0, i, qr); cross.index_add_(0, j, qr)
        f = cross @ torch.linalg.pinv(cov)
        eye = torch.eye(3, dtype=self.dtype, device=self.device)
        e = 0.5 * (f.transpose(1, 2) @ f - eye)
        dev = e - torch.diag_embed(e.diagonal(dim1=1, dim2=2).mean(dim=1).unsqueeze(1).expand(-1, 3))
        # sqrt(2/3 devE:devE), invariant under rigid rotation.
        return torch.sqrt((2.0 / 3.0) * dev.square().sum(dim=(1, 2)).clamp_min(0.0))

    def _loads(self, external_forces: torch.Tensor | None) -> torch.Tensor | None:
        if external_forces is None:
            return None
        if not isinstance(external_forces, torch.Tensor) or tuple(external_forces.shape) != tuple(self.positions.shape):
            raise ValueError(f"external_forces must be a tensor of shape {tuple(self.positions.shape)}")
        loads = external_forces.to(device=self.device, dtype=self.dtype)
        if not bool(torch.isfinite(loads).all()):
            raise ValueError("external_forces must contain only finite values")
        return loads

    @torch.no_grad()
    def forces(self, external_forces: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate nodal forces and the ice-on-tool reaction vector."""
        cfg = self.config
        loads = self._loads(external_forces)
        i, j = self.pairs.T
        delta = self.positions[j] - self.positions[i]
        distance = torch.linalg.vector_norm(delta, dim=1)
        eps = torch.finfo(self.dtype).eps
        normal = delta / distance.clamp_min(eps)[:, None]
        normal = torch.where((distance > eps)[:, None], normal, self.initial_normals)
        extension = distance - self.rest_lengths
        strain = extension / self.rest_lengths
        bi, bj = self.bond_indices.T
        shear = self._objective_shear_strain()
        shear_bond = 0.5 * (shear[bi] + shear[bj])
        tensile_fail = strain[self.bonded] > cfg.breaking_strain
        shear_fail = shear_bond > cfg.shear_breaking_strain
        old_alive = self.alive.clone()
        self.alive[self.bonded] &= ~(tensile_fail | shear_fail)
        newly = old_alive & ~self.alive
        modes_t = torch.zeros_like(self.bonded); modes_s = torch.zeros_like(self.bonded)
        modes_t[self.bonded] = tensile_fail
        modes_s[self.bonded] = shear_fail
        self.failure_mode[newly] = modes_t[newly].to(torch.uint8) + 2 * modes_s[newly].to(torch.uint8)
        self.failure_time_s[newly] = self.time
        self.failure_extension_m[newly] = extension[newly]
        self.failure_energy_J[newly] = 0.5 * cfg.bond_stiffness * extension[newly].square()

        relative_v = self.velocities[j] - self.velocities[i]
        vn = (relative_v * normal).sum(dim=1)
        overlap = (2 * cfg.radius - distance).clamp_min(0)
        contact_mag = (cfg.contact_stiffness * overlap - cfg.contact_damping * vn).clamp_min(0)
        contact_mag = torch.where((overlap > 0) & ~self.alive, contact_mag, torch.zeros_like(contact_mag))
        bond_mag = cfg.bond_stiffness * extension
        magnitude = torch.where(self.alive, bond_mag, -contact_mag)
        pair_force = magnitude[:, None] * normal
        force = -cfg.drag * self.velocities
        force.index_add_(0, i, pair_force)
        force.index_add_(0, j, -pair_force)

        tool_delta = self.positions - self.tool_position
        tool_dist = torch.linalg.vector_norm(tool_delta, dim=1)
        tool_normal = tool_delta / tool_dist.clamp_min(eps)[:, None]
        fallback = torch.zeros_like(tool_normal); fallback[:, 2] = -1
        tool_normal = torch.where((tool_dist > eps)[:, None], tool_normal, fallback)
        tool_overlap = (cfg.radius + cfg.tool_radius - tool_dist).clamp_min(0)
        tool_vn = ((self.velocities - self.tool_velocity) * tool_normal).sum(dim=1)
        tool_mag = (cfg.contact_stiffness * tool_overlap - cfg.contact_damping * tool_vn).clamp_min(0)
        tool_mag = torch.where(tool_overlap > 0, tool_mag, torch.zeros_like(tool_mag))
        tool_force = tool_mag[:, None] * tool_normal
        reaction = -tool_force.sum(dim=0)
        force += tool_force
        if loads is not None:
            force += loads
        self.last_reaction = reaction
        return force, reaction

    @torch.no_grad()
    def step(self, external_forces: torch.Tensor | None = None) -> None:
        force, _ = self.forces(external_forces)
        self.velocities += self.dt * force / self.config.mass
        self.velocities[self.fixed] = 0
        self.positions += self.dt * self.velocities
        self.positions[self.fixed] = self.initial_positions[self.fixed]
        self.time += self.dt
        self.step_count += 1
        if not bool(torch.isfinite(self.positions).all() and torch.isfinite(self.velocities).all()):
            raise FloatingPointError("non-finite DEM3D state; reduce dt or contact stiffness")

    @torch.no_grad()
    def diagnostics(self, external_forces: torch.Tensor | None = None) -> dict[str, float | int]:
        force, reaction = self.forces(external_forces)
        support = -force[self.fixed].sum(dim=0)
        return {
            "time": self.time,
            "tool_x": float(self.tool_position[0]), "tool_y": float(self.tool_position[1]),
            "tool_z": float(self.tool_position[2]),
            "reaction_x": float(reaction[0]), "reaction_y": float(reaction[1]),
            "reaction_z": float(reaction[2]),
            "boundary_reaction_x": float(support[0]), "boundary_reaction_y": float(support[1]),
            "boundary_reaction_z": float(support[2]),
            "broken_bonds": self.broken_bonds,
            "kinetic_energy_J": float(0.5 * self.config.mass * self.velocities.square().sum()),
            "particle_count": len(self.positions),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return restart-friendly state (tensors remain on the active device)."""
        return {
            "dimension": 3, "config": asdict(self.config), "dt": self.dt,
            "time": self.time, "step_count": self.step_count,
            "positions": self.positions.clone(), "velocities": self.velocities.clone(),
            "initial_positions": self.initial_positions.clone(), "fixed": self.fixed.clone(),
            "pairs": self.pairs.clone(), "bonded": self.bonded.clone(), "alive": self.alive.clone(),
            "failure_mode": self.failure_mode.clone(), "failure_time_s": self.failure_time_s.clone(),
            "failure_extension_m": self.failure_extension_m.clone(), "failure_energy_J": self.failure_energy_J.clone(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot created by :meth:`state_dict` after strict validation."""
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary returned by state_dict()")
        required = {
            "dimension", "config", "dt", "time", "step_count", "positions",
            "velocities", "initial_positions", "fixed", "pairs", "bonded",
            "alive", "failure_mode", "failure_time_s", "failure_extension_m",
            "failure_energy_J",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
        if state["dimension"] != 3:
            raise ValueError("checkpoint dimension must be 3")
        if state["config"] != asdict(self.config):
            raise ValueError("checkpoint config does not match this solver configuration")
        if not math.isfinite(float(state["dt"])) or float(state["dt"]) != self.dt:
            raise ValueError("checkpoint dt does not match this solver")
        time_value = float(state["time"])
        step_value = state["step_count"]
        if not math.isfinite(time_value) or time_value < 0:
            raise ValueError("checkpoint time must be finite and nonnegative")
        if isinstance(step_value, bool) or not isinstance(step_value, int) or step_value < 0:
            raise ValueError("checkpoint step_count must be a nonnegative integer")

        tensor_shapes = {
            "positions": tuple(self.positions.shape),
            "velocities": tuple(self.velocities.shape),
            "initial_positions": tuple(self.initial_positions.shape),
            "fixed": tuple(self.fixed.shape),
            "pairs": tuple(self.pairs.shape),
            "bonded": tuple(self.bonded.shape),
            "alive": tuple(self.alive.shape),
            "failure_mode": tuple(self.failure_mode.shape),
            "failure_time_s": tuple(self.failure_time_s.shape),
            "failure_extension_m": tuple(self.failure_extension_m.shape),
            "failure_energy_J": tuple(self.failure_energy_J.shape),
        }
        staged: dict[str, torch.Tensor] = {}
        for name, expected_shape in tensor_shapes.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                raise ValueError(f"checkpoint {name} must be a tensor with shape {expected_shape}")
            target = getattr(self, name)
            value = value.to(device=self.device, dtype=target.dtype)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"checkpoint {name} contains non-finite values")
            staged[name] = value
        if not torch.equal(staged["initial_positions"], self.initial_positions):
            raise ValueError("checkpoint reference geometry does not match this solver")
        if not torch.equal(staged["pairs"], self.pairs):
            raise ValueError("checkpoint pair topology does not match this solver")
        if not torch.equal(staged["bonded"], self.bonded):
            raise ValueError("checkpoint bond topology does not match this solver")
        if bool((staged["alive"] & ~staged["bonded"]).any()):
            raise ValueError("checkpoint marks a non-bond pair as alive")
        if not torch.equal(staged["fixed"], self.fixed):
            raise ValueError("checkpoint fixed-boundary mask does not match this solver")

        for name in ("positions", "velocities", "initial_positions", "fixed", "pairs",
                     "bonded", "alive", "failure_mode", "failure_time_s",
                     "failure_extension_m", "failure_energy_J"):
            getattr(self, name).copy_(staged[name])
        self.time = time_value
        self.step_count = step_value
        self.last_reaction.zero_()
