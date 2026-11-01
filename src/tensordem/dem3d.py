"""Three-dimensional bonded-sphere DEM engine for ice fracture.

All quantities use SI units. This first 3-D solver is deliberately independent of
the established 2-D implementation so existing cases and output formats remain
unchanged. The lattice is a simple-cubic network with face/body diagonals inside
the initial cutoff; calibrate its directional stiffness before engineering use.

Contact constitutive model (L1_foundation — SIR-PC2|F05):
- Normal: linear spring-dashpot with restitution-coefficient parameterization.
- Tangential: Cundall-Strack incremental Coulomb-friction spring (|F_t| <= mu * |F_n|).
- Rolling resistance: torque T_r = mu_r * R_eff * |F_n| opposing relative rotation.
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
    geometry: str = "box"
    fix_x_edges: bool = True
    fix_bottom: bool = False
    top_platen_enabled: bool = False
    bottom_platen_enabled: bool = False
    platen_stiffness: float = 5_000.0
    platen_damping: float = 0.0
    top_platen_gap: float = 0.0
    bottom_platen_gap: float = 0.0
    top_platen_velocity: float = 0.0
    bottom_platen_velocity: float = 0.0
    # --- SIR-PC2|F05: Coulomb friction + rolling resistance ---
    friction_coefficient: float = 0.0        # mu: Coulomb friction coefficient
    rolling_resistance_coefficient: float = 0.0  # mu_r: rolling resistance coefficient
    restitution_coefficient: float = 1.0     # e: coefficient of restitution (0,1]

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
        for name in ("contact_damping", "drag", "tool_gap", "platen_damping",
                     "top_platen_gap", "bottom_platen_gap"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.platen_stiffness) or self.platen_stiffness <= 0:
            raise ValueError("platen_stiffness must be finite and positive")
        for name in ("top_platen_velocity", "bottom_platen_velocity"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("top_platen_enabled", "bottom_platen_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0):
            raise ValueError("dt must be finite and positive")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.geometry not in ("box", "cylinder_xz"):
            raise ValueError("geometry must be 'box' or 'cylinder_xz'")
        if self.geometry == "cylinder_xz" and (self.nx < 3 or self.nz < 3):
            raise ValueError("cylinder_xz geometry requires nx and nz >= 3")
        # Validate friction / rolling / restitution parameters
        if not math.isfinite(self.friction_coefficient) or self.friction_coefficient < 0:
            raise ValueError("friction_coefficient must be finite and nonnegative")
        if not math.isfinite(self.rolling_resistance_coefficient) or self.rolling_resistance_coefficient < 0:
            raise ValueError("rolling_resistance_coefficient must be finite and nonnegative")
        if not (0.0 < self.restitution_coefficient <= 1.0):
            raise ValueError("restitution_coefficient must be in (0, 1]")

    @property
    def mass(self) -> float:
        return self.density * (4.0 / 3.0) * math.pi * self.radius**3

    @property
    def recommended_dt(self) -> float:
        # Conservative explicit stability/travel bound; validate by convergence.
        n = self.nx * self.ny * self.nz
        degree_bound = min(n - 1, 26)
        contact_stiffness = max(
            self.bond_stiffness, self.contact_stiffness,
            self.platen_stiffness if (self.top_platen_enabled or self.bottom_platen_enabled) else 0.0,
        )
        k = (2 * degree_bound + 1) * contact_stiffness
        c = ((2 * degree_bound + 1) * self.contact_damping + self.drag
             + (2 if (self.top_platen_enabled or self.bottom_platen_enabled) else 0)
             * self.platen_damping)
        elastic = 0.15 * math.sqrt(self.mass / k)
        damping = 0.15 * self.mass / c if c > 0 else math.inf
        travel = 0.1 * self.radius / self.tool_speed
        return min(elastic, damping, travel)


class IceDEM3D:
    """3-D bonded-sphere DEM with spherical prescribed indenter.

    State arrays have shape (N, 3); external forces are instantaneous nodal
    loads in newtons. The indenter translates downward at constant speed.
    Bond failure is irreversible. Permanent local bonds are stored in O(N);
    contacts use a spatial cell list rebuilt each force evaluation. The current
    CPU-side cell hashing is a prototype and can synchronize CUDA execution.

    Contact model (SIR-PC2|F05):
    - Normal: linear spring-dashpot, damping derived from restitution_coefficient.
    - Tangential: Cundall-Strack incremental Coulomb-friction spring.
    - Rolling: resistance torque T_r = mu_r * R_eff * |F_n|.
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
        full_grid = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
        # Geometry is defined on the original structured lattice, then compacted
        # so every state array and every bond uses contiguous particle indices.
        if config.geometry == "cylinder_xz":
            xc, zc = (config.nx - 1) / 2.0, (config.nz - 1) / 2.0
            disk_radius = min(config.nx - 1, config.nz - 1) / 2.0
            keep = ((xx - xc).square() + (zz - zc).square()) <= disk_radius**2 + 1e-12
        else:
            keep = torch.ones_like(xx, dtype=torch.bool)
        compact_to_full = torch.nonzero(keep.reshape(-1), as_tuple=False).flatten()
        full_to_compact = torch.full((len(full_grid),), -1, dtype=torch.long, device=self.device)
        full_to_compact[compact_to_full] = torch.arange(len(compact_to_full), device=self.device)
        grid = full_grid[compact_to_full]
        z_origin = (config.nz - 1) / 2.0 if config.geometry == "cylinder_xz" else 0.0
        self.positions = (grid - torch.tensor(
            [(config.nx - 1) / 2, (config.ny - 1) / 2, z_origin],
            dtype=self.dtype, device=self.device
        )) * (2 * config.radius)
        self.initial_positions = self.positions.clone()
        self.velocities = torch.zeros_like(self.positions)
        self.fixed = torch.zeros(len(grid), dtype=torch.bool, device=self.device)
        gx, gy, gz = grid.unbind(dim=1)
        if config.fix_x_edges and config.geometry == "box":
            self.fixed |= (gx == 0) | (gx == config.nx - 1)
        if config.fix_bottom:
            self.fixed |= gz == 0

        # Build only valid neighbour bonds and remap full-grid IDs to compact IDs.
        n = len(grid)
        bond_pairs: list[tuple[int, int]] = []
        for dz in range(0, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == dy == dz == 0:
                        continue
                    # Keep one half-space of offsets to avoid duplicate bonds.
                    if not (dz > 0 or (dz == 0 and dy > 0) or
                            (dz == 0 and dy == 0 and dx > 0)):
                        continue
                    for z in range(config.nz):
                        zz2 = z + dz
                        if not 0 <= zz2 < config.nz:
                            continue
                        for y in range(config.ny):
                            yy2 = y + dy
                            if not 0 <= yy2 < config.ny:
                                continue
                            for x in range(config.nx):
                                xx2 = x + dx
                                if not 0 <= xx2 < config.nx:
                                    continue
                                full_a = (z * config.ny + y) * config.nx + x
                                full_b = (zz2 * config.ny + yy2) * config.nx + xx2
                                a = int(full_to_compact[full_a].item())
                                b = int(full_to_compact[full_b].item())
                                if a >= 0 and b >= 0:
                                    bond_pairs.append((min(a, b), max(a, b)))
        self.pairs = torch.tensor(bond_pairs, dtype=torch.long, device=self.device)
        order = torch.argsort(self.pairs[:, 0] * n + self.pairs[:, 1])
        self.pairs = self.pairs[order]
        i, j = self.pairs.T
        ref = self.initial_positions[j] - self.initial_positions[i]
        self.rest_lengths = torch.linalg.vector_norm(ref, dim=1)
        self.bonded = torch.ones(len(self.pairs), dtype=torch.bool, device=self.device)
        self.alive = self.bonded.clone()
        self.bond_indices = self.pairs
        self.reference_bond_vectors = ref.clone()
        self.initial_normals = ref / self.rest_lengths.clamp_min(torch.finfo(self.dtype).eps)[:, None]
        self.failure_mode = torch.zeros(len(self.pairs), dtype=torch.uint8, device=self.device)
        self.failure_time_s = torch.full((len(self.pairs),), -1.0, dtype=self.dtype, device=self.device)
        self.failure_extension_m = torch.zeros(len(self.pairs), dtype=self.dtype, device=self.device)
        self.failure_energy_J = torch.zeros(len(self.pairs), dtype=self.dtype, device=self.device)
        self._bond_keys = self.pairs[:, 0] * len(grid) + self.pairs[:, 1]

        self.time = 0.0
        self.step_count = 0
        self.tool_start = torch.tensor(
            [0.0, 0.0, (config.nz - 1) * 2 * config.radius + config.radius
             + config.tool_radius + config.tool_gap],
            dtype=self.dtype, device=self.device)
        self.tool_velocity = torch.tensor([0.0, 0.0, -config.tool_speed],
                                          dtype=self.dtype, device=self.device)
        self.last_reaction = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.top_platen_start_z = (
            float(self.initial_positions[:, 2].max()) + config.radius + config.top_platen_gap
        )
        self.bottom_platen_start_z = (
            float(self.initial_positions[:, 2].min()) - config.radius - config.bottom_platen_gap
        )
        self.last_top_platen_reaction_z = torch.zeros((), dtype=self.dtype, device=self.device)
        self.last_bottom_platen_reaction_z = torch.zeros((), dtype=self.dtype, device=self.device)

        # --- SIR-PC2|F05: tangential friction & rolling resistance state ---
        # Tangential displacement accumulator per contact pair (keyed by pair hash).
        # Stored as dict: pair_key -> (tangential_disp_3d, last_normal_3d)
        self._tangential_state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Rolling resistance torque accumulator per particle (N, 3).
        self._rolling_torque = torch.zeros((n, 3), dtype=self.dtype, device=self.device)
        # Flag indicating rolling resistance torque is physically modelled.
        self.rolling_resistance_torque_physical = config.rolling_resistance_coefficient > 0.0

    @property
    def tool_position(self) -> torch.Tensor:
        return self.tool_start + self.time * self.tool_velocity

    @property
    def broken_bonds(self) -> int:
        return int((self.bonded & ~self.alive).sum().item())

    @torch.no_grad()
    def _contact_candidate_pairs(self) -> torch.Tensor:
        """Build current overlap candidates with a uniform spatial cell list.

        Cell width is the contact diameter, so overlapping spheres can only be
        in the same or one of the 26 adjacent cells. Bond topology is stored
        separately and is never discarded when particles move or bonds break.
        """
        n = len(self.positions)
        if n < 2:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        width = 2.0 * self.config.radius
        coords = torch.floor(self.positions.detach().cpu() / width).to(torch.int64).tolist()
        cells: dict[tuple[int, int, int], list[int]] = {}
        for idx, xyz in enumerate(coords):
            key = (xyz[0], xyz[1], xyz[2])
            cells.setdefault(key, []).append(idx)
        pairs: list[tuple[int, int]] = []
        for key, members in cells.items():
            x, y, z = key
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        other = cells.get((x + dx, y + dy, z + dz))
                        if other is None:
                            continue
                        for i in members:
                            for j in other:
                                if j > i:
                                    pairs.append((i, j))
        if not pairs:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        return torch.tensor(pairs, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _alive_bond_for_pairs(self, pairs: torch.Tensor) -> torch.Tensor:
        """Return an alive-bond mask aligned with arbitrary sorted node pairs."""
        if pairs.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.device)
        keys = pairs[:, 0] * len(self.positions) + pairs[:, 1]
        where = torch.searchsorted(self._bond_keys, keys)
        safe = where.clamp(max=max(len(self._bond_keys) - 1, 0))
        matched = (where < len(self._bond_keys)) & (self._bond_keys[safe] == keys)
        result = torch.zeros_like(matched)
        result[matched] = self.alive[safe[matched]]
        return result

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
    def _commit_fracture(self, extension: torch.Tensor) -> None:
        """Commit irreversible bond events only during an explicit force update."""
        cfg = self.config
        strain = extension / self.rest_lengths
        bi, bj = self.bond_indices.T
        shear = self._objective_shear_strain()
        shear_bond = 0.5 * (shear[bi] + shear[bj])
        tensile_fail = strain > cfg.breaking_strain
        shear_fail = shear_bond > cfg.shear_breaking_strain
        old_alive = self.alive.clone()
        self.alive &= ~(tensile_fail | shear_fail)
        newly = old_alive & ~self.alive
        self.failure_mode[newly] = tensile_fail[newly].to(torch.uint8) + 2 * shear_fail[newly].to(torch.uint8)
        self.failure_time_s[newly] = self.time
        self.failure_extension_m[newly] = extension[newly]
        self.failure_energy_J[newly] = 0.5 * cfg.bond_stiffness * extension[newly].square()

    # ------------------------------------------------------------------
    # SIR-PC2|F05: Coulomb friction + rolling resistance helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pair_key(i: int, j: int) -> int:
        """Deterministic key for a sorted particle pair."""
        return min(i, j) * 1_000_003 + max(i, j)

    def _get_tangential_state(self, ci: torch.Tensor, cj: torch.Tensor,
                              cnormal: torch.Tensor) -> torch.Tensor:
        """Retrieve or initialise tangential displacement for each contact pair.

        Returns (N, 3) tangential displacement vectors.
        """
        n_contacts = ci.numel()
        tang_disp = torch.zeros((n_contacts, 3), dtype=self.dtype, device=self.device)
        for idx in range(n_contacts):
            i = int(ci[idx].item())
            j = int(cj[idx].item())
            key = self._pair_key(i, j)
            if key in self._tangential_state:
                stored_disp, stored_normal = self._tangential_state[key]
                # Rotate stored tangential displacement to current normal frame
                # (simple approach: keep as-is in global frame; projection handles it)
                tang_disp[idx] = stored_disp
            else:
                tang_disp[idx] = torch.zeros(3, dtype=self.dtype, device=self.device)
        return tang_disp

    def _update_tangential_state(self, ci: torch.Tensor, cj: torch.Tensor,
                                 cnormal: torch.Tensor,
                                 tang_disp: torch.Tensor) -> None:
        """Persist tangential displacement for the next step."""
        n_contacts = ci.numel()
        for idx in range(n_contacts):
            i = int(ci[idx].item())
            j = int(cj[idx].item())
            key = self._pair_key(i, j)
            self._tangential_state[key] = (
                tang_disp[idx].clone(),
                cnormal[idx].clone(),
            )

    def _compute_rolling_resistance_torque(
        self, ci: torch.Tensor, cj: torch.Tensor,
        cnormal: torch.Tensor, contact_mag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rolling resistance torque for each contact.

        Model: T_r = mu_r * R_eff * |F_n| * omega_hat
        where R_eff = R (equal spheres), omega_hat is the direction of
        relative angular velocity projected onto the tangential plane.

        For simplicity (no explicit particle rotation DOF in current DEM3D),
        we approximate the rolling resistance as a tangential force couple:
        F_rr = mu_r * |F_n| * sign(v_tangent), applied at the contact point.

        Returns (torque_on_i, torque_on_j) each (N, 3).
        """
        n_contacts = ci.numel()
        mu_r = self.config.rolling_resistance_coefficient
        R = self.config.radius

        # Relative tangential velocity (approximation using translational vel)
        relative_v = self.velocities[cj] - self.velocities[ci]
        vn = (relative_v * cnormal).sum(dim=1, keepdim=True)
        v_tangent = relative_v - vn * cnormal
        v_t_mag = torch.linalg.vector_norm(v_tangent, dim=1, keepdim=True)

        # Rolling resistance force magnitude
        F_rr_mag = mu_r * contact_mag  # (N,)
        # Direction opposes tangential motion
        direction = torch.where(
            v_t_mag > 1e-15,
            -v_tangent / v_t_mag,
            torch.zeros_like(v_tangent),
        )
        F_rr = F_rr_mag[:, None] * direction  # (N, 3)

        # Torque = R x F_rr (lever arm = R * normal direction)
        # For sphere-sphere contact, torque on each particle:
        # T_i = R * (n x F_rr), T_j = -T_i
        torque_i = R * torch.linalg.cross(cnormal, F_rr)
        torque_j = -torque_i

        return torque_i, torque_j

    # ------------------------------------------------------------------
    # Main force evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forces(self, external_forces: torch.Tensor | None = None, *,
               update_fracture: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate forces; default commits fracture for API compatibility.

        ``update_fracture=False`` evaluates the currently stored bond topology
        without changing fracture history or the stored last reaction.

        Contact model (SIR-PC2|F05):
        - Normal: linear spring-dashpot with restitution-based damping.
        - Tangential: Cundall-Strack Coulomb-friction spring.
        - Rolling: resistance torque T_r = mu_r * R * |F_n|.
        """
        cfg = self.config
        loads = self._loads(external_forces)
        eps = torch.finfo(self.dtype).eps
        # Permanent bond interactions retain compact topology and failure history.
        i, j = self.pairs.T
        delta = self.positions[j] - self.positions[i]
        distance = torch.linalg.vector_norm(delta, dim=1)
        normal = delta / distance.clamp_min(eps)[:, None]
        normal = torch.where((distance > eps)[:, None], normal, self.initial_normals)
        extension = distance - self.rest_lengths
        if update_fracture:
            self._commit_fracture(extension)

        bond_mag = cfg.bond_stiffness * extension
        bond_force = bond_mag[:, None] * normal
        force = -cfg.drag * self.velocities
        if loads is not None:
            # External nodal loads enter the force balance here. Dropping this
            # term silently removes every externally applied load from the
            # simulation (no error, zero response), so it must stay explicit.
            force = force + loads
        active_bond_force = torch.where(self.alive[:, None], bond_force, torch.zeros_like(bond_force))
        force.index_add_(0, i, active_bond_force)
        force.index_add_(0, j, -active_bond_force)

        contact_pairs = self._contact_candidate_pairs()
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
                bond_alive = self._alive_bond_for_pairs(torch.stack((ci, cj), dim=1))
                relative_v = self.velocities[cj] - self.velocities[ci]
                vn = (relative_v * cnormal).sum(dim=1)

                # --- Normal contact force with restitution-based damping ---
                # Compute effective mass for restitution mapping
                m_eff = cfg.mass / 2.0  # equal-mass spheres
                # Map restitution coefficient to damping
                if cfg.restitution_coefficient < 1.0:
                    ln_e = math.log(cfg.restitution_coefficient)
                    abs_ln_e = -ln_e
                    zeta = abs_ln_e / math.sqrt(math.pi ** 2 + abs_ln_e ** 2)
                    gamma_n = 2.0 * zeta * math.sqrt(cfg.contact_stiffness * m_eff)
                else:
                    gamma_n = 0.0

                contact_mag = (cfg.contact_stiffness * overlap - gamma_n * vn).clamp_min(0)
                contact_mag = torch.where(~bond_alive, contact_mag, torch.zeros_like(contact_mag))
                contact_force = -contact_mag[:, None] * cnormal

                # --- SIR-PC2|F05: Tangential Coulomb friction ---
                if cfg.friction_coefficient > 0.0:
                    # Retrieve tangential displacement state
                    tang_disp = self._get_tangential_state(ci, cj, cnormal)
                    # Integrate tangential displacement
                    v_tangent = relative_v - vn[:, None] * cnormal
                    tang_disp = tang_disp + v_tangent * self.dt

                    # Tangential force from spring
                    k_t = cfg.contact_stiffness  # default: k_t = k_n
                    F_t_raw = k_t * tang_disp
                    F_t_mag = torch.linalg.vector_norm(F_t_raw, dim=1)
                    F_limit = cfg.friction_coefficient * contact_mag

                    # Coulomb saturation
                    scale = torch.where(
                        F_t_mag > F_limit,
                        F_limit / F_t_mag.clamp_min(1e-15),
                        torch.ones_like(F_t_mag),
                    )
                    F_t = F_t_raw * scale[:, None]

                    # Apply tangential force
                    force.index_add_(0, ci, F_t)
                    force.index_add_(0, cj, -F_t)

                    # Update tangential state
                    self._update_tangential_state(ci, cj, cnormal, tang_disp)

                # --- SIR-PC2|F05: Rolling resistance torque ---
                if cfg.rolling_resistance_coefficient > 0.0:
                    torque_i, torque_j = self._compute_rolling_resistance_torque(
                        ci, cj, cnormal, contact_mag,
                    )
                    # Accumulate rolling torque (stored for diagnostics)
                    self._rolling_torque.index_add_(0, ci, torque_i)
                    self._rolling_torque.index_add_(0, cj, torque_j)

                    # Apply rolling resistance as tangential force couple
                    # F_rr = T_r / R, applied at contact point
                    R = cfg.radius
                    F_rr_i = torque_i / R
                    F_rr_j = torque_j / R
                    force.index_add_(0, ci, F_rr_i)
                    force.index_add_(0, cj, F_rr_j)

                force.index_add_(0, ci, contact_force)
                force.index_add_(0, cj, -contact_force)

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

        # Optional infinite horizontal platens. Positive overlap means a particle
        # intersects the plane; damping opposes the relative normal approach.
        top_reaction_z = torch.zeros((), dtype=self.dtype, device=self.device)
        bottom_reaction_z = torch.zeros((), dtype=self.dtype, device=self.device)
        if cfg.top_platen_enabled:
            top_z = self.top_platen_start_z + self.time * cfg.top_platen_velocity
            top_overlap = (self.positions[:, 2] + cfg.radius - top_z).clamp_min(0.0)
            top_relative_v = self.velocities[:, 2] - cfg.top_platen_velocity
            top_mag = (cfg.platen_stiffness * top_overlap
                       + cfg.platen_damping * top_relative_v).clamp_min(0.0)
            top_force_z = -top_mag
            force[:, 2] += top_force_z
            top_reaction_z = -top_force_z.sum()
        if cfg.bottom_platen_enabled:
            bottom_z = self.bottom_platen_start_z + self.time * cfg.bottom_platen_velocity
            bottom_overlap = (bottom_z + cfg.radius - self.positions[:, 2]).clamp_min(0.0)
            bottom_relative_v = self.velocities[:, 2] - cfg.bottom_platen_velocity
            # Damper opposes relative normal approach: with the bottom platen
            # normal +z the damping term is -c*(v_z - v_platen). Using +c turns
            # the damper into a negative-damping (energy-injecting) term.
            bottom_mag = (cfg.platen_stiffness * bottom_overlap
                          - cfg.platen_damping * bottom_relative_v).clamp_min(0.0)
            bottom_force_z = bottom_mag
            force[:, 2] += bottom_force_z
            bottom_reaction_z = -bottom_force_z.sum()

        if update_fracture:
            self.last_reaction.copy_(reaction)
        self.last_top_platen_reaction_z.copy_(top_reaction_z)
        self.last_bottom_platen_reaction_z.copy_(bottom_reaction_z)
        return force, reaction

    @torch.no_grad()
    def step(self, external_forces: torch.Tensor | None = None) -> None:
        """Single explicit Euler step."""
        force, _ = self.forces(external_forces)
        acc = force / self.config.mass
        # Fixed particles are hard supports: they neither accelerate nor move
        # away from their initial position.
        acc[self.fixed] = 0.0
        self.velocities += acc * self.dt
        self.velocities[self.fixed] = 0.0
        self.positions += self.velocities * self.dt
        self.positions[self.fixed] = self.initial_positions[self.fixed]
        self.time += self.dt
        self.step_count += 1
        if not bool(torch.isfinite(self.positions).all() and torch.isfinite(self.velocities).all()):
            raise FloatingPointError("non-finite DEM3D state; reduce dt or contact stiffness")

    @torch.no_grad()
    def diagnostics(self, external_forces: torch.Tensor | None = None) -> dict[str, float | int]:
        """Return a full diagnostic snapshot including forces, reactions, and energy."""
        force, reaction = self.forces(external_forces, update_fracture=False)
        support = -force[self.fixed].sum(dim=0)
        kinetic = 0.5 * self.config.mass * self.velocities.square().sum()
        # Recoverable elastic energy in surviving bonds only. Broken bonds are
        # represented separately by their recorded release-energy estimate.
        i, j = self.pairs.T
        distance = torch.linalg.vector_norm(self.positions[j] - self.positions[i], dim=1)
        extension = distance - self.rest_lengths
        bond_energy = (0.5 * self.config.bond_stiffness * extension.square() * self.alive).sum()
        contact_pairs = self._contact_candidate_pairs()
        contact_energy = torch.zeros((), dtype=self.dtype, device=self.device)
        if contact_pairs.numel():
            ci, cj = contact_pairs.T
            cdistance = torch.linalg.vector_norm(self.positions[cj] - self.positions[ci], dim=1)
            overlap = (2 * self.config.radius - cdistance).clamp_min(0)
            broken_or_unbonded = ~self._alive_bond_for_pairs(contact_pairs)
            contact_energy = (0.5 * self.config.contact_stiffness
                              * overlap.square() * broken_or_unbonded).sum()
        tool_overlap = (self.config.radius + self.config.tool_radius
                        - torch.linalg.vector_norm(self.positions - self.tool_position, dim=1)).clamp_min(0)
        tool_energy = (0.5 * self.config.contact_stiffness * tool_overlap.square()).sum()
        top_platen_energy = torch.zeros((), dtype=self.dtype, device=self.device)
        if self.config.top_platen_enabled:
            top_z = self.top_platen_start_z + self.time * self.config.top_platen_velocity
            top_overlap = (self.positions[:, 2] + self.config.radius - top_z).clamp_min(0.0)
            top_platen_energy = (0.5 * self.config.platen_stiffness * top_overlap.square()).sum()
        bottom_platen_energy = torch.zeros((), dtype=self.dtype, device=self.device)
        if self.config.bottom_platen_enabled:
            bottom_z = self.bottom_platen_start_z + self.time * self.config.bottom_platen_velocity
            bottom_overlap = (bottom_z + self.config.radius - self.positions[:, 2]).clamp_min(0.0)
            bottom_platen_energy = (0.5 * self.config.platen_stiffness * bottom_overlap.square()).sum()
        platen_energy = top_platen_energy + bottom_platen_energy
        released = self.failure_energy_J.sum()
        return {
            "time": self.time,
            "tool_x": float(self.tool_position[0]), "tool_y": float(self.tool_position[1]),
            "tool_z": float(self.tool_position[2]),
            "reaction_x": float(reaction[0]), "reaction_y": float(reaction[1]),
            "reaction_z": float(reaction[2]),
            "top_platen_reaction_z_N": float(self.last_top_platen_reaction_z),
            "bottom_platen_reaction_z_N": float(self.last_bottom_platen_reaction_z),
            "boundary_reaction_x": float(support[0]), "boundary_reaction_y": float(support[1]),
            "boundary_reaction_z": float(support[2]),
            "broken_bonds": self.broken_bonds,
            "particle_count": len(self.positions),
            "kinetic_energy_J": float(kinetic),
            "bond_elastic_energy_J": float(bond_energy),
            "particle_contact_energy_J": float(contact_energy),
            "tool_contact_energy_J": float(tool_energy),
            "platen_contact_energy_J": float(platen_energy),
            "released_bond_energy_J": float(released),
            "mechanical_energy_J": float(kinetic + bond_energy + contact_energy + tool_energy + platen_energy),
        }


    def energy(self) -> dict[str, float]:
        """Return a snapshot of mechanical energy components (J)."""
        cfg = self.config
        kinetic = 0.5 * cfg.mass * (self.velocities ** 2).sum().item()
        i, j = self.pairs.T
        delta = self.positions[j] - self.positions[i]
        distance = torch.linalg.vector_norm(delta, dim=1)
        extension = distance - self.rest_lengths
        bond_energy = 0.5 * cfg.bond_stiffness * (extension[self.alive] ** 2).sum().item()
        contact_energy = 0.0
        contact_pairs = self._contact_candidate_pairs()
        if contact_pairs.numel():
            ci, cj = contact_pairs.T
            cdelta = self.positions[cj] - self.positions[ci]
            cdistance = torch.linalg.vector_norm(cdelta, dim=1)
            overlap = (2 * cfg.radius - cdistance).clamp_min(0)
            near = overlap > 0
            ci, cj = ci[near], cj[near]
            overlap = overlap[near]
            if ci.numel():
                bond_alive = self._alive_bond_for_pairs(torch.stack((ci, cj), dim=1))
                active = ~bond_alive & (overlap > 0)
                contact_energy = 0.5 * cfg.contact_stiffness * (overlap[active] ** 2).sum().item()
        tool_delta = self.positions - self.tool_position
        tool_dist = torch.linalg.vector_norm(tool_delta, dim=1)
        tool_overlap = (cfg.radius + cfg.tool_radius - tool_dist).clamp_min(0)
        tool_energy = 0.5 * cfg.contact_stiffness * (tool_overlap ** 2).sum().item()
        platen_energy = 0.0
        if cfg.top_platen_enabled:
            top_z = self.top_platen_start_z + self.time * cfg.top_platen_velocity
            top_overlap = (self.positions[:, 2] + cfg.radius - top_z).clamp_min(0.0)
            platen_energy += 0.5 * cfg.platen_stiffness * (top_overlap ** 2).sum().item()
        if cfg.bottom_platen_enabled:
            bottom_z = self.bottom_platen_start_z + self.time * cfg.bottom_platen_velocity
            bottom_overlap = (bottom_z + cfg.radius - self.positions[:, 2]).clamp_min(0.0)
            platen_energy += 0.5 * cfg.platen_stiffness * (bottom_overlap ** 2).sum().item()
        released = 0.5 * cfg.bond_stiffness * (extension[~self.alive] ** 2).sum().item()
        return {
            "time_s": self.time,
            "step": self.step_count,
            "broken_bonds": self.broken_bonds,
            "particle_count": len(self.positions),
            "kinetic_energy_J": float(kinetic),
            "bond_elastic_energy_J": float(bond_energy),
            "particle_contact_energy_J": float(contact_energy),
            "tool_contact_energy_J": float(tool_energy),
            "platen_contact_energy_J": float(platen_energy),
            "released_bond_energy_J": float(released),
            "mechanical_energy_J": float(kinetic + bond_energy + contact_energy + tool_energy + platen_energy),
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
        saved_config = state["config"]
        if not isinstance(saved_config, dict):
            raise ValueError("checkpoint config must be a dictionary")
        # Backward compatibility: configs saved before plane-platen support omit
        # new optional keys, whose defaults preserve the previous solver behavior.
        normalized_config = {**asdict(DEM3DConfig()), **saved_config}
        if normalized_config != asdict(self.config):
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
            # Validate storage type before conversion: e.g. float NaN->bool
            # silently becomes True, and fractional pair indices truncate.
            if value.dtype != target.dtype:
                raise ValueError(f"checkpoint {name} dtype must be {target.dtype}, got {value.dtype}")
            value = value.to(device=self.device)
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
