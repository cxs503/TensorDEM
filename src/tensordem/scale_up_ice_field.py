"""Scale-up broken ice field DEM simulation for 100k+ particles.

SIR-PC2|I02|G6: Large-scale broken ice field (100k+ particles) for ice
navigation campaign DAG node.  Provides a vectorized DEM engine that avoids
Python loops in bond construction and contact detection, enabling 100k+
particle simulations within the 60 s/step wall-clock budget on CPU.

Physical scenario
-----------------
A rectangular domain is filled with bonded-sphere ice particles on a
simple-cubic lattice.  A configurable fraction of bonds are pre-broken
to simulate a broken-ice field (separate floes).  Gravity acts downward
(z < 0); linear drag damps oscillations.  No ship hull is present in
this scale-up validation — the goal is to verify that the solver handles
100k+ particles with acceptable performance and exact mass conservation.

All quantities use SI units.
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScaleUpIceFieldConfig:
    """Numerical and physical parameters for the scale-up broken-ice field."""

    # Lattice dimensions (total particles = nx * ny * nz)
    nx: int = 100
    ny: int = 10
    nz: int = 100
    radius: float = 0.025          # particle radius [m]
    density: float = 917.0         # ice density [kg/m³]
    bond_stiffness: float = 2_000.0   # [N/m]
    contact_stiffness: float = 5_000.0  # [N/m]
    breaking_strain: float = 0.015
    shear_breaking_strain: float = 0.03
    contact_damping: float = 5.0   # [N·s/m]
    drag: float = 2.0              # linear drag [N·s/m]
    dt: float | None = None
    device: str = "cpu"
    gravity: float = 9.81          # [m/s²] downward
    pre_break_fraction: float = 0.3  # fraction of bonds pre-broken
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 2:
                raise ValueError(f"{name} must be int >= 2")
        for name in ("radius", "density", "bond_stiffness", "contact_stiffness",
                     "breaking_strain", "shear_breaking_strain"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite positive")
        if not math.isfinite(self.gravity) or self.gravity < 0:
            raise ValueError("gravity must be finite non-negative")
        for name in ("contact_damping", "drag", "pre_break_fraction"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite non-negative")
        if self.pre_break_fraction >= 1.0:
            raise ValueError("pre_break_fraction must be < 1")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")

    @property
    def particle_count(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def mass(self) -> float:
        return self.density * (4.0 / 3.0) * math.pi * self.radius ** 3

    @property
    def recommended_dt(self) -> float:
        n = self.particle_count
        degree_bound = min(n - 1, 26)
        k_max = max(self.bond_stiffness, self.contact_stiffness)
        k = (2 * degree_bound + 1) * k_max
        c = (2 * degree_bound + 1) * self.contact_damping + self.drag
        elastic = 0.15 * math.sqrt(self.mass / k)
        damping = 0.15 * self.mass / c if c > 0 else math.inf
        return min(elastic, damping)


# ---------------------------------------------------------------------------
# Vectorized cell-list contact detection
# ---------------------------------------------------------------------------
@torch.no_grad()
def _vectorized_cell_list_pairs(
    positions: torch.Tensor,
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build contact candidate pairs using a fully vectorized cell list.

    Returns sorted (i, j) pairs with i < j.  No Python loops over particles.
    """
    n = len(positions)
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    width = 2.0 * radius

    # Cell coordinates (integer)
    cell_coords = torch.floor(positions / width).to(torch.int64)

    # Shift to non-negative for hashing
    mins = cell_coords.min(dim=0).values
    shifted = cell_coords - mins
    dims = (shifted.max(dim=0).values + 1)
    dim_yz = int(dims[1].item()) * int(dims[2].item())
    dim_z = int(dims[2].item())

    # Linear hash
    cell_hash = shifted[:, 0] * dim_yz + shifted[:, 1] * dim_z + shifted[:, 2]

    # Sort by hash for O(1) cell lookup
    sorted_hash, sort_idx = torch.sort(cell_hash)
    sorted_positions = positions[sort_idx]

    # Unique cells and boundaries
    unique_hash, counts = torch.unique_consecutive(sorted_hash, return_counts=True)
    boundaries = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        torch.cumsum(counts, 0),
    ])

    # 13 half-space neighbor offsets (avoid duplicate pairs)
    offsets_13: list[tuple[int, int, int]] = []
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == dy == dz == 0:
                    continue
                if not (dz > 0 or (dz == 0 and dy > 0)
                        or (dz == 0 and dy == 0 and dx > 0)):
                    continue
                offsets_13.append((dx, dy, dz))

    all_i: list[torch.Tensor] = []
    all_j: list[torch.Tensor] = []

    for dx, dy, dz in offsets_13:
        # Neighbor cell hash for every particle
        nc = cell_coords.clone()
        nc[:, 0] += dx
        nc[:, 1] += dy
        nc[:, 2] += dz
        ns = nc - mins
        nhash = ns[:, 0] * dim_yz + ns[:, 1] * dim_z + ns[:, 2]

        # Search for neighbor hash in unique_hash
        idx = torch.searchsorted(unique_hash, nhash)
        valid = (idx < len(unique_hash))
        safe_idx = idx.clamp(max=len(unique_hash) - 1)
        valid = valid & (unique_hash[safe_idx] == nhash)

        # Cell boundaries for valid neighbor cells
        left = boundaries[safe_idx]
        right = boundaries[(safe_idx + 1).clamp(max=len(boundaries) - 1)]
        num_nb = (right - left) * valid.long()

        # Expand: for each particle with neighbors, create pairs
        active_mask = num_nb > 0
        if not active_mask.any():
            continue

        active_i = torch.nonzero(active_mask).flatten()
        active_left = left[active_mask]
        active_num = num_nb[active_mask]

        expanded_i = active_i.repeat_interleave(active_num)

        # Build absolute indices into sorted array
        max_nn = int(active_num.max().item())
        if max_nn == 0:
            continue
        offsets_arange = torch.arange(max_nn, device=device)
        abs_idx = active_left.unsqueeze(1) + offsets_arange.unsqueeze(0)
        mask = offsets_arange.unsqueeze(0) < active_num.unsqueeze(1)
        abs_idx_flat = abs_idx[mask]

        expanded_j = sort_idx[abs_idx_flat]

        all_i.append(expanded_i)
        all_j.append(expanded_j)

    if not all_i:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    pi = torch.cat(all_i)
    pj = torch.cat(all_j)

    # Keep only j > i (half-space dedup)
    keep = pj > pi
    pairs = torch.stack([pi[keep], pj[keep]], dim=1)

    # Sort for deterministic output
    order = torch.argsort(pairs[:, 0] * n + pairs[:, 1])
    return pairs[order]


# ---------------------------------------------------------------------------
# Vectorized bond construction
# ---------------------------------------------------------------------------
def _vectorized_bond_pairs(
    nx: int, ny: int, nz: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Build bond topology for a simple-cubic lattice with face/body diagonals.

    Returns (M, 2) long tensor of sorted unique pairs (i < j).
    No Python loops over grid points.
    """
    axes = [torch.arange(v, dtype=dtype, device=device) for v in (nx, ny, nz)]
    zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    grid = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)  # (N, 3)

    # 13 half-space neighbor offsets
    offsets_13: list[tuple[int, int, int]] = []
    for dz in range(0, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == dy == dz == 0:
                    continue
                if not (dz > 0 or (dz == 0 and dy > 0)
                        or (dz == 0 and dy == 0 and dx > 0)):
                    continue
                offsets_13.append((dx, dy, dz))

    all_pairs: list[torch.Tensor] = []
    for dx, dy, dz in offsets_13:
        shifted = grid + torch.tensor([dx, dy, dz], dtype=dtype, device=device)
        valid = ((shifted[:, 0] >= 0) & (shifted[:, 0] < nx)
                 & (shifted[:, 1] >= 0) & (shifted[:, 1] < ny)
                 & (shifted[:, 2] >= 0) & (shifted[:, 2] < nz))
        i_idx = torch.arange(len(grid), device=device)[valid]
        j_idx = (shifted[valid, 2].long() * ny + shifted[valid, 1].long()) * nx + shifted[valid, 0].long()
        all_pairs.append(torch.stack([i_idx, j_idx], dim=1))

    pairs = torch.cat(all_pairs, dim=0)
    n = len(grid)
    order = torch.argsort(pairs[:, 0] * n + pairs[:, 1])
    return pairs[order]


# ---------------------------------------------------------------------------
# Scale-up ice field engine
# ---------------------------------------------------------------------------
class ScaleUpIceField:
    """Vectorized DEM engine for large-scale broken ice field simulation.

    Key differences from :class:`IceDEM3D`:
    - Bond construction uses vectorized torch ops (no Python loops).
    - Contact detection uses a vectorized cell list (no Python loops).
    - Gravity is built-in (applied as external force on every particle).
    - Pre-breaking of bonds simulates a broken-ice field.
    - Mass conservation is tracked explicitly.
    """

    def __init__(self, config: ScaleUpIceFieldConfig = ScaleUpIceFieldConfig()) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.dtype = torch.float64
        self.dt = config.dt if config.dt is not None else config.recommended_dt
        if self.dt > config.recommended_dt:
            raise ValueError(
                f"dt={self.dt:.6g} exceeds conservative limit {config.recommended_dt:.6g} s")

        n = config.particle_count

        # ---- positions on a centred simple-cubic lattice ----
        axes = [torch.arange(v, dtype=self.dtype, device=self.device)
                for v in (config.nx, config.ny, config.nz)]
        zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        grid = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
        centre = torch.tensor(
            [(config.nx - 1) / 2, (config.ny - 1) / 2, (config.nz - 1) / 2],
            dtype=self.dtype, device=self.device,
        )
        self.positions = (grid - centre) * (2 * config.radius)
        self.initial_positions = self.positions.clone()
        self.velocities = torch.zeros_like(self.positions)

        # ---- fixed boundary: bottom layer ----
        self.fixed = torch.zeros(n, dtype=torch.bool, device=self.device)
        gz = grid[:, 2]
        self.fixed |= gz == 0

        # ---- vectorized bond construction ----
        self.pairs = _vectorized_bond_pairs(
            config.nx, config.ny, config.nz, self.device, self.dtype)
        i, j = self.pairs.T
        ref = self.initial_positions[j] - self.initial_positions[i]
        self.rest_lengths = torch.linalg.vector_norm(ref, dim=1)
        self.bonded = torch.ones(len(self.pairs), dtype=torch.bool, device=self.device)
        self.alive = self.bonded.clone()
        self._bond_keys = self.pairs[:, 0] * n + self.pairs[:, 1]

        # ---- pre-break bonds to create broken-ice floes ----
        if config.pre_break_fraction > 0:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(config.seed)
            rand = torch.rand(len(self.pairs), generator=rng)
            break_mask = rand < config.pre_break_fraction
            self.alive[break_mask] = False

        # ---- gravity force (constant, applied every step) ----
        self._gravity_force = torch.zeros_like(self.positions)
        self._gravity_force[:, 2] = -config.gravity * config.mass

        # ---- bookkeeping ----
        self.time = 0.0
        self.step_count = 0
        self._initial_total_mass = float(n * config.mass)
        self._wallclock_per_step: list[float] = []

    # ------------------------------------------------------------------
    @property
    def particle_count(self) -> int:
        return len(self.positions)

    @property
    def total_mass(self) -> float:
        return float(self.particle_count * self.config.mass)

    @property
    def mass_conservation_ratio(self) -> float:
        """|m_initial - m_current| / m_initial  (should be 0 for DEM)."""
        return abs(self._initial_total_mass - self.total_mass) / self._initial_total_mass

    @property
    def broken_bonds(self) -> int:
        return int((self.bonded & ~self.alive).sum().item())

    @property
    def last_wallclock_per_step(self) -> float:
        return self._wallclock_per_step[-1] if self._wallclock_per_step else 0.0

    @property
    def mean_wallclock_per_step(self) -> float:
        if not self._wallclock_per_step:
            return 0.0
        return sum(self._wallclock_per_step) / len(self._wallclock_per_step)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _alive_bond_for_pairs(self, pairs: torch.Tensor) -> torch.Tensor:
        if pairs.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.device)
        keys = pairs[:, 0] * len(self.positions) + pairs[:, 1]
        where = torch.searchsorted(self._bond_keys, keys)
        safe = where.clamp(max=max(len(self._bond_keys) - 1, 0))
        matched = (where < len(self._bond_keys)) & (self._bond_keys[safe] == keys)
        result = torch.zeros_like(matched)
        result[matched] = self.alive[safe[matched]]
        return result

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forces(self) -> torch.Tensor:
        """Evaluate all forces (bond + contact + gravity + drag)."""
        cfg = self.config
        eps = torch.finfo(self.dtype).eps
        n = len(self.positions)

        # Drag
        force = -cfg.drag * self.velocities

        # Gravity
        force = force + self._gravity_force

        # Bond forces
        i, j = self.pairs.T
        delta = self.positions[j] - self.positions[i]
        distance = torch.linalg.vector_norm(delta, dim=1)
        normal = delta / distance.clamp_min(eps)[:, None]
        extension = distance - self.rest_lengths
        bond_mag = cfg.bond_stiffness * extension
        bond_force = bond_mag[:, None] * normal
        active_bond_force = torch.where(
            self.alive[:, None], bond_force, torch.zeros_like(bond_force))
        force.index_add_(0, i, active_bond_force)
        force.index_add_(0, j, -active_bond_force)

        # Contact forces (vectorized cell list)
        contact_pairs = _vectorized_cell_list_pairs(
            self.positions, cfg.radius, self.device, self.dtype)
        if contact_pairs.numel():
            ci, cj = contact_pairs.T
            cdelta = self.positions[cj] - self.positions[ci]
            cdistance = torch.linalg.vector_norm(cdelta, dim=1)
            overlap = (2 * cfg.radius - cdistance).clamp_min(0)
            near = overlap > 0
            ci, cj = ci[near], cj[near]
            cdelta, cdistance, overlap = cdelta[near], cdistance[near], overlap[near]
            if ci.numel():
                cnormal = cdelta / cdistance.clamp_min(eps)[:, None]
                bond_alive = self._alive_bond_for_pairs(
                    torch.stack((ci, cj), dim=1))
                relative_v = self.velocities[cj] - self.velocities[ci]
                vn = (relative_v * cnormal).sum(dim=1)
                contact_mag = (cfg.contact_stiffness * overlap
                               - cfg.contact_damping * vn).clamp_min(0)
                contact_mag = torch.where(
                    ~bond_alive, contact_mag, torch.zeros_like(contact_mag))
                contact_force = -contact_mag[:, None] * cnormal
                force.index_add_(0, ci, contact_force)
                force.index_add_(0, cj, -contact_force)

        # Zero out forces on fixed particles
        force[self.fixed] = 0.0
        return force

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self) -> dict[str, float]:
        """Advance one time step.  Returns timing and diagnostics dict."""
        t0 = _time.perf_counter()

        force = self.forces()
        self.velocities += self.dt * force / self.config.mass
        self.velocities[self.fixed] = 0
        self.positions += self.dt * self.velocities
        self.positions[self.fixed] = self.initial_positions[self.fixed]
        self.time += self.dt
        self.step_count += 1

        if not bool(torch.isfinite(self.positions).all()
                    and torch.isfinite(self.velocities).all()):
            raise FloatingPointError(
                "non-finite state; reduce dt or stiffness")

        elapsed = _time.perf_counter() - t0
        self._wallclock_per_step.append(elapsed)

        return {
            "step": self.step_count,
            "time_s": self.time,
            "wallclock_s": elapsed,
            "mass_conservation": self.mass_conservation_ratio,
            "particle_count": self.particle_count,
        }

    # ------------------------------------------------------------------
    def run(self, n_steps: int) -> dict[str, Any]:
        """Run *n_steps* and return summary metrics."""
        last_diag = None
        for _ in range(n_steps):
            last_diag = self.step()

        summary = {
            "particle_count": self.particle_count,
            "mass_conservation": self.mass_conservation_ratio,
            "wallclock_per_step_last": self.last_wallclock_per_step,
            "wallclock_per_step_mean": self.mean_wallclock_per_step,
            "wallclock_per_step_max": max(self._wallclock_per_step) if self._wallclock_per_step else 0,
            "total_steps": self.step_count,
            "sim_time_s": self.time,
            "total_bonds": len(self.pairs),
            "broken_bonds": self.broken_bonds,
            "pre_break_fraction": self.config.pre_break_fraction,
            "dt": self.dt,
        }
        return summary

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "nx": self.config.nx, "ny": self.config.ny, "nz": self.config.nz,
                "radius": self.config.radius, "density": self.config.density,
                "particle_count": self.config.particle_count,
            },
            "dt": self.dt,
            "time": self.time,
            "step_count": self.step_count,
            "positions": self.positions.clone(),
            "velocities": self.velocities.clone(),
            "initial_positions": self.initial_positions.clone(),
            "fixed": self.fixed.clone(),
            "alive": self.alive.clone(),
        }


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------
def run_broken_ice_field(
    nx: int = 100,
    ny: int = 10,
    nz: int = 100,
    n_steps: int = 5,
    **kwargs,
) -> dict[str, Any]:
    """Create and run a broken-ice field simulation.

    Returns a results dict with all acceptance-criteria metrics.
    """
    cfg = ScaleUpIceFieldConfig(nx=nx, ny=ny, nz=nz, **kwargs)
    sim = ScaleUpIceField(cfg)
    summary = sim.run(n_steps)
    summary["config"] = {
        "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
        "radius": cfg.radius, "density": cfg.density,
        "particle_count": cfg.particle_count,
        "mass_per_particle_kg": cfg.mass,
        "gravity": cfg.gravity,
        "pre_break_fraction": cfg.pre_break_fraction,
        "dt": sim.dt,
        "device": cfg.device,
    }
    return summary
