"""Vectorized DEM engine for large-scale granular simulations (10k–1M particles).

GPU-ready DEM engine using counting–prefix-sum–scatter cell-list pattern.
On loong64 (no CUDA), runs on CPU via PyTorch vectorized ops.

Contact models:
- Normal: linear spring-dashpot OR Hertz
- Tangential: simplified Coulomb friction
- Rolling resistance: T_r = mu_r * R_eff * |F_n|
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VecDEMConfig:
    """Configuration for the vectorized DEM engine."""
    n_particles: int = 1000
    radius: float = 0.01
    density: float = 1000.0
    contact_model: str = "linear"
    contact_stiffness: float = 1e5
    young_modulus: float = 1e9
    poisson_ratio: float = 0.3
    restitution_coefficient: float = 0.5
    friction_coefficient: float = 0.3
    rolling_resistance_coefficient: float = 0.0
    contact_damping: float = 0.0
    contact_damping_model: str = "restitution"
    gravity: float = 0.0
    dt: float | None = None
    domain_size: tuple[float, float, float] = (0.1, 0.1, 0.1)
    seed: int = 42
    device: str = "cpu"

    @property
    def mass(self) -> float:
        return self.density * (4.0 / 3.0) * math.pi * self.radius ** 3

    @property
    def recommended_dt(self) -> float:
        if self.mass <= 0 or self.radius <= 0:
            return 1e-6
        if self.contact_model == "hertz":
            E_star = self.young_modulus / (2.0 * (1.0 - self.poisson_ratio ** 2))
            R_star = self.radius / 2.0
            delta_ref = 1e-4 * self.radius
            k_eff = (4.0 / 3.0) * E_star * math.sqrt(R_star) * math.sqrt(delta_ref)
        else:
            k_eff = self.contact_stiffness
        m_eff = self.mass / 2.0
        omega = math.sqrt(k_eff / m_eff)
        return 0.1 / omega

    def effective_dt(self) -> float:
        return self.dt if self.dt is not None else self.recommended_dt


def build_cell_list(
    positions: torch.Tensor,
    cell_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sorted cell list. Returns (sorted_indices, sorted_keys, cell_starts, cell_coords)."""
    N = positions.shape[0]
    device = positions.device

    domain_min = positions.min(dim=0).values - cell_width
    cell_coords = torch.floor((positions - domain_min) / cell_width).to(torch.int64)

    offset = 1_000_000
    cell_keys = ((cell_coords[:, 0] + offset) * 2_000_003 +
                 (cell_coords[:, 1] + offset) * 2_011 +
                 (cell_coords[:, 2] + offset))

    sorted_keys, sorted_indices = torch.sort(cell_keys)

    if N == 0:
        cell_starts = torch.tensor([0], dtype=torch.long, device=device)
    else:
        changes = torch.ones(N, dtype=torch.bool, device=device)
        changes[1:] = sorted_keys[1:] != sorted_keys[:-1]
        cell_starts = torch.nonzero(changes, as_tuple=False).flatten()
        cell_starts = torch.cat([cell_starts, torch.tensor([N], device=device)])

    return sorted_indices, sorted_keys, cell_starts, cell_coords


def find_neighbor_pairs_vectorized(
    positions: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Find all overlapping pairs using cell list.

    Uses counting-prefix-sum-scatter pattern. Loops over cells (not particles),
    which is O(n_cells) where n_cells << N for dense packings.
    """
    N = positions.shape[0]
    device = positions.device
    cutoff = 2.0 * radius

    if N < 2:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    cell_width = cutoff
    sorted_indices, sorted_keys, cell_starts, cell_coords = build_cell_list(positions, cell_width)

    n_cells = len(cell_starts) - 1
    if n_cells == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    # Get cell coordinates for each cell (from first particle in cell)
    cell_coords_unique = cell_coords[sorted_indices[cell_starts[:n_cells]]]

    # Generate 27 offsets
    offsets = torch.tensor(
        [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=torch.int64, device=device
    )

    # For each cell, compute 27 neighbor keys
    neighbor_coords = cell_coords_unique.unsqueeze(1) + offsets.unsqueeze(0)  # (n_cells, 27, 3)
    offset_val = 1_000_000
    neighbor_keys = ((neighbor_coords[:, :, 0] + offset_val) * 2_000_003 +
                     (neighbor_coords[:, :, 1] + offset_val) * 2_011 +
                     (neighbor_coords[:, :, 2] + offset_val))

    # Look up neighbor keys in sorted_keys
    flat_nkeys = neighbor_keys.reshape(-1)
    start_pos = torch.searchsorted(sorted_keys, flat_nkeys)
    end_pos = torch.searchsorted(sorted_keys, flat_nkeys + 1)

    # Valid matches
    valid = (start_pos < N) & (end_pos > start_pos)
    safe_start = start_pos.clamp(max=N - 1)
    valid = valid & (sorted_keys[safe_start] == flat_nkeys)

    # Reshape back to (n_cells, 27)
    valid = valid.reshape(n_cells, 27)
    start_pos = start_pos.reshape(n_cells, 27)
    end_pos = end_pos.reshape(n_cells, 27)

    # Generate pairs by looping over cells (fast: n_cells << N)
    all_pairs = []
    for c in range(n_cells):
        c_start = cell_starts[c].item()
        c_end = cell_starts[c + 1].item()
        if c_start == c_end:
            continue
        c_particles = sorted_indices[c_start:c_end]

        for k in range(27):
            if not valid[c, k]:
                continue
            n_start = start_pos[c, k].item()
            n_end = end_pos[c, k].item()
            if n_start == n_end:
                continue
            n_particles = sorted_indices[n_start:n_end]

            # Generate pairs (i, j) with i < j
            i_grid = c_particles.unsqueeze(1).expand(len(c_particles), len(n_particles))
            j_grid = n_particles.unsqueeze(0).expand(len(c_particles), len(n_particles))
            mask = i_grid < j_grid
            if mask.any():
                pairs = torch.stack([i_grid[mask], j_grid[mask]], dim=1)
                all_pairs.append(pairs)

    if not all_pairs:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    pairs = torch.cat(all_pairs, dim=0)
    pairs = torch.unique(pairs, dim=0)

    # Distance filter
    diff = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    dist = torch.linalg.vector_norm(diff, dim=1)
    pairs = pairs[dist < cutoff]

    return pairs


def compute_contact_forces(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    pairs: torch.Tensor,
    radius: float,
    config: VecDEMConfig,
) -> torch.Tensor:
    """Compute contact forces. Returns (N, 3) force tensor."""
    N = positions.shape[0]
    device = positions.device
    dtype = positions.dtype
    mass = config.mass

    forces = torch.zeros((N, 3), dtype=dtype, device=device)
    if len(pairs) == 0:
        return forces

    ci, cj = pairs[:, 0], pairs[:, 1]
    delta_pos = positions[cj] - positions[ci]
    distance = torch.linalg.vector_norm(delta_pos, dim=1)
    overlap = (2.0 * radius - distance).clamp_min(0.0)

    contact_mask = overlap > 0.0
    if not contact_mask.any():
        return forces

    ci_c, cj_c = ci[contact_mask], cj[contact_mask]
    overlap_c = overlap[contact_mask]
    distance_c = distance[contact_mask]
    delta_pos_c = delta_pos[contact_mask]

    eps = 1e-15
    normal = delta_pos_c / distance_c.clamp_min(eps).unsqueeze(1)
    rel_v = velocities[cj_c] - velocities[ci_c]
    vn = (rel_v * normal).sum(dim=1)

    # Normal force
    if config.contact_model == "hertz":
        E_star = config.young_modulus / (2.0 * (1.0 - config.poisson_ratio ** 2))
        R_star = radius / 2.0
        k_local = (4.0 / 3.0) * E_star * math.sqrt(R_star) * torch.sqrt(overlap_c)
    else:
        k_local = torch.full_like(overlap_c, config.contact_stiffness)

    m_eff = mass / 2.0
    if config.contact_damping_model == "absolute":
        gamma_n = config.contact_damping
    elif config.restitution_coefficient < 1.0:
        ln_e = math.log(config.restitution_coefficient)
        zeta = -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)
        gamma_n = 2.0 * zeta * torch.sqrt(k_local * m_eff)
    else:
        gamma_n = 0.0

    fn_mag = (k_local * overlap_c - gamma_n * vn).clamp_min(0.0)
    fn_i = -fn_mag.unsqueeze(1) * normal

    # Tangential force
    ft_i = torch.zeros_like(fn_i)
    if config.friction_coefficient > 0.0:
        v_tangent = rel_v - vn.unsqueeze(1) * normal
        vt_mag = torch.linalg.vector_norm(v_tangent, dim=1)
        ft_limit = config.friction_coefficient * fn_mag
        if vt_mag.max() > eps:
            ft_dir = v_tangent / vt_mag.clamp_min(eps).unsqueeze(1)
            ft_mag = torch.minimum(config.contact_stiffness * config.effective_dt() * vt_mag, ft_limit)
            ft_i = -ft_mag.unsqueeze(1) * ft_dir

    f_total_i = fn_i + ft_i

    # Scatter-add
    forces.scatter_add_(0, ci_c.unsqueeze(1).expand(-1, 3), f_total_i)
    forces.scatter_add_(0, cj_c.unsqueeze(1).expand(-1, 3), -f_total_i)

    return forces


class VecDEM:
    """Vectorized DEM engine."""

    def __init__(self, config: VecDEMConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.float64
        self.dt = config.effective_dt()

        torch.manual_seed(config.seed)
        N = config.n_particles
        domain = torch.tensor(config.domain_size, dtype=self.dtype, device=self.device)

        n_side = math.ceil(N ** (1.0 / 3.0))
        spacing = 2.0 * config.radius * 1.05
        grid_coords = torch.arange(n_side, dtype=self.dtype, device=self.device)
        zz, yy, xx = torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
        grid_pos = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) * spacing
        grid_pos = grid_pos[:N]
        jitter = torch.randn_like(grid_pos) * 0.1 * config.radius
        self.positions = grid_pos + jitter
        center = self.positions.mean(dim=0)
        self.positions += (domain / 2.0 - center)

        self.velocities = torch.zeros_like(self.positions)
        self.forces = torch.zeros_like(self.positions)
        self.time = 0.0
        self.step_count = 0

    def step(self) -> None:
        self.velocities += 0.5 * self.forces / self.config.mass * self.dt
        self.positions += self.velocities * self.dt

        pairs = find_neighbor_pairs_vectorized(self.positions, self.config.radius)
        self.forces = compute_contact_forces(
            self.positions, self.velocities, pairs, self.config.radius, self.config
        )

        if self.config.gravity > 0.0:
            self.forces[:, 2] -= self.config.mass * self.config.gravity

        self.velocities += 0.5 * self.forces / self.config.mass * self.dt
        self.time += self.dt
        self.step_count += 1

    def kinetic_energy(self) -> float:
        return float(0.5 * self.config.mass * (self.velocities ** 2).sum())

    def potential_energy(self) -> float:
        if self.config.gravity <= 0.0:
            return 0.0
        return float(self.config.mass * self.config.gravity * self.positions[:, 2].sum())

    def total_momentum(self) -> torch.Tensor:
        return self.config.mass * self.velocities.sum(dim=0)

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()


class ReferenceDEM:
    """Reference CPU DEM with naive O(N²) neighbor search."""

    def __init__(self, config: VecDEMConfig):
        self.config = config
        self.dtype = torch.float64
        self.dt = config.effective_dt()

        torch.manual_seed(config.seed)
        N = config.n_particles
        domain = torch.tensor(config.domain_size, dtype=self.dtype)

        n_side = math.ceil(N ** (1.0 / 3.0))
        spacing = 2.0 * config.radius * 1.05
        grid_coords = torch.arange(n_side, dtype=self.dtype)
        zz, yy, xx = torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
        grid_pos = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) * spacing
        grid_pos = grid_pos[:N]
        jitter = torch.randn_like(grid_pos) * 0.1 * config.radius
        self.positions = grid_pos + jitter
        center = self.positions.mean(dim=0)
        self.positions += (domain / 2.0 - center)

        self.velocities = torch.zeros_like(self.positions)
        self.forces = torch.zeros_like(self.positions)
        self.time = 0.0
        self.step_count = 0

    def find_pairs_naive(self) -> torch.Tensor:
        N = len(self.positions)
        cutoff = 2.0 * self.config.radius
        pairs = []
        for i in range(N):
            diff = self.positions[i+1:] - self.positions[i]
            dist = torch.linalg.vector_norm(diff, dim=1)
            mask = dist < cutoff
            if mask.any():
                j_indices = torch.nonzero(mask, as_tuple=False).flatten() + i + 1
                i_indices = torch.full((len(j_indices),), i, dtype=torch.long)
                pairs.append(torch.stack([i_indices, j_indices], dim=1))
        if not pairs:
            return torch.empty((0, 2), dtype=torch.long)
        return torch.cat(pairs, dim=0)

    def step(self) -> None:
        self.velocities += 0.5 * self.forces / self.config.mass * self.dt
        self.positions += self.velocities * self.dt
        pairs = self.find_pairs_naive()
        self.forces = compute_contact_forces(
            self.positions, self.velocities, pairs, self.config.radius, self.config
        )
        if self.config.gravity > 0.0:
            self.forces[:, 2] -= self.config.mass * self.config.gravity
        self.velocities += 0.5 * self.forces / self.config.mass * self.dt
        self.time += self.dt
        self.step_count += 1

    def kinetic_energy(self) -> float:
        return float(0.5 * self.config.mass * (self.velocities ** 2).sum())

    def total_momentum(self) -> torch.Tensor:
        return self.config.mass * self.velocities.sum(dim=0)
