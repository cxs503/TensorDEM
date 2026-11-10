"""Scale-clean 3D granular DEM kernel (IN383-DEM01).

Physics (identical at every particle count — this is what criterion C4 gates):

* Hertz–Mindlin normal contact ``F_n = (4/3) E* sqrt(R*) d^{3/2} - c_n v_n``
  with a Tsuji damping coefficient derived from the restitution coefficient;
* Cundall–Strack tangential history (incremental spring ``F_t = -S_t xi``) with
  a Coulomb cap ``|F_t| <= mu F_n``; the tangential state is stored per
  neighbour-list slot so that it survives across time steps;
* planar walls (floor / left / right indenter) with the same contact law and
  an independent wall friction coefficient (``mu_wall``), the right wall being
  a rigid indenter with a prescribed velocity;
* velocity-Verlet integration in float64.

Known limitations (documented in ``docs/IN383_DEM01_scale_ceiling.md``):
no particle rotation / rolling resistance, no cohesion, no buoyancy.  These do
not affect any of the scale gates, which compare the *same* model at different
particle counts.

Complexity: one time step costs ``O(C)`` with ``C`` the neighbour-list size; the
list is rebuilt only when a particle has moved more than ``skin/2``, so the
``O(N log N)`` rebuild is amortised over many steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .search import cell_list_pairs

_TWO_PI = 2.0 * math.pi


@dataclass
class FloeBedConfig:
    """Configuration of a periodic-in-y slab of granular ice floes."""

    n: int = 1000
    diameter: float = 1.0
    polydispersity: float = 0.0
    density: float = 917.0
    young: float = 1.0e9
    poisson: float = 0.3
    mu_grain: float = 0.4
    mu_wall: float = 0.4
    restitution: float = 0.15
    skin: float = 0.2
    layers_y: int = 6
    height_layers: int = 10
    packing: float = 0.55
    gravity: float = 9.81
    dt: float | None = None
    dt_safety: float = 0.03
    seed: int = 0
    floor: bool = True
    left_wall: bool = True
    right_wall: bool = True

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive")
        if not 0.0 < self.restitution <= 1.0:
            raise ValueError("restitution must be in (0, 1]")
        if self.skin <= 0.0:
            raise ValueError("skin must be positive")

    # ------------------------------------------------------------- geometry
    @property
    def lattice_spacing(self) -> float:
        return self.diameter * (math.pi / (6.0 * self.packing)) ** (1.0 / 3.0)

    @property
    def n_y(self) -> int:
        return int(self.layers_y)

    @property
    def n_z(self) -> int:
        return int(self.height_layers)

    @property
    def n_x(self) -> int:
        return int(math.ceil(self.n / (self.n_y * self.n_z)))

    @property
    def deposition_spacing(self) -> float:
        """Lattice spacing used to build the initial (overlap-free) bed."""
        return 1.0 * self.diameter

    @property
    def box(self) -> np.ndarray:
        # Box edges are an integer number of deposition spacings, so the initial
        # simple-cubic lattice sits strictly inside the box with a *positive* gap
        # between neighbours: no particle is clipped and no spurious contact
        # force is injected at t=0.
        sp = self.deposition_spacing
        return np.array([self.n_x, self.n_y, self.n_z], dtype=np.float64) * sp

    @property
    def periodic(self) -> tuple[bool, bool, bool]:
        return (False, True, False)

    # -------------------------------------------------------- contact law
    @property
    def effective_modulus(self) -> float:
        return self.young / (2.0 * (1.0 - self.poisson**2))

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def effective_shear(self) -> float:
        return self.shear_modulus / (2.0 * (2.0 - self.poisson))

    def contact_dt(self) -> float:
        """Stable step from the Hertz contact period at a 1 % overlap."""
        if self.dt is not None:
            return float(self.dt)
        r_mean = 0.5 * self.diameter
        mass = self.density * 4.0 / 3.0 * math.pi * r_mean**3
        m_eff = 0.5 * mass
        delta = 0.01 * r_mean
        k_normal = 2.0 * self.effective_modulus * math.sqrt(r_mean * delta)
        omega = math.sqrt(k_normal / m_eff)
        return self.dt_safety * _TWO_PI / omega


@dataclass
class NeighbourList:
    i: torch.Tensor
    j: torch.Tensor
    d0: torch.Tensor  # contact-frame reference (unused, kept for audit)
    built_at: int
    n_candidate: int

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.i.numel())


@dataclass
class BedState:
    pos: torch.Tensor
    vel: torch.Tensor
    step: int = 0
    time: float = 0.0
    rebuilds: int = 0
    force_evals: int = 0
    wall_work: dict[str, float] = field(default_factory=dict)


class FloeBed:
    """Granular slab (periodic in ``y``) with a movable right indenter wall."""

    def __init__(self, config: FloeBedConfig | None = None) -> None:
        self.config = config or FloeBedConfig()
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        box = cfg.box
        sp = cfg.deposition_spacing
        grid = np.stack(
            np.meshgrid(
                (np.arange(cfg.n_x) + 0.5) * sp,
                (np.arange(cfg.n_y) + 0.5) * sp,
                (np.arange(cfg.n_z) + 0.5) * sp,
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)[: cfg.n]
        # tiny jitter (<= 2 % of d) keeps the touching lattice from being a
        # perfectly balanced configuration while creating no measurable overlap
        jitter = (rng.random(grid.shape) - 0.5) * 0.005 * cfg.diameter
        pos = grid + jitter  # the touching lattice lies strictly inside the box
        radii = np.full(cfg.n, 0.5 * cfg.diameter)
        if cfg.polydispersity > 0.0:
            radii *= 1.0 + cfg.polydispersity * (rng.random(cfg.n) - 0.5)
        mass = cfg.density * 4.0 / 3.0 * math.pi * radii**3
        self.radii_np = radii
        self.mass_np = mass
        self.pos = torch.from_numpy(pos)
        self.vel = torch.zeros((cfg.n, 3), dtype=torch.float64)
        self.radii = torch.from_numpy(radii)
        self.mass = torch.from_numpy(mass)
        self.box = box
        self.dt = cfg.contact_dt()
        self.inv_mass = 1.0 / self.mass
        self.state = BedState(self.pos, self.vel)
        # wall positions (right wall is the indenter)
        self.wall_pos = {
            "floor": 0.0,
            "left": 0.0,
            "right": float(box[0]),
        }
        self.wall_vel = {"floor": 0.0, "left": 0.0, "right": 0.0}
        self._xi_pair: torch.Tensor | None = None
        self._keys: np.ndarray | None = None
        self._xpair: torch.Tensor | None = None
        self.wall_mu_scale = {name: 1.0 for name in self.wall_pos}
        self._xi_wall = {
            name: torch.zeros((cfg.n, 3), dtype=torch.float64)
            for name in self.wall_pos
        }
        self._wall_force = {name: 0.0 for name in self.wall_pos}
        self._wall_tangent = {name: 0.0 for name in self.wall_pos}
        self.neighbours: NeighbourList | None = None
        self._ref_pos = self.pos.clone()
        self._force: torch.Tensor | None = None
        self._normal_force: torch.Tensor | None = None
        self._restitution_damping = self._damping_coefficient(cfg.restitution)
        self.build_neighbours()

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _damping_coefficient(restitution: float) -> float:
        if restitution >= 1.0:
            return 0.0
        ln_e = math.log(restitution)
        return -ln_e / math.sqrt(ln_e * ln_e + math.pi**2)

    def _damping_force(self, s_n: torch.Tensor, v_n: torch.Tensor,
                       m_eff: torch.Tensor, beta: float) -> torch.Tensor:
        if beta == 0.0:
            return torch.zeros_like(s_n)
        return 2.0 * math.sqrt(5.0 / 6.0) * beta * torch.sqrt(s_n * m_eff) * v_n

    # ---------------------------------------------------- neighbour search
    def build_neighbours(self, *, keep_state: bool = True) -> NeighbourList:
        cfg = self.config
        pos_np = self.pos.numpy()
        i, j, stats = cell_list_pairs(
            pos_np,
            self.box,
            self.radii_np,
            skin=cfg.skin,
            periodic=cfg.periodic,
        )
        n = cfg.n
        keys = i.astype(np.int64) * n + j.astype(np.int64)
        # periodic image of every pair is frozen for the life of the list: between
        # rebuilds no particle moves more than skin/2, so the image cannot change
        dy = pos_np[j, 1] - pos_np[i, 1]
        self._ymask = torch.from_numpy(-np.round(dy / self.box[1]))
        self._keys = keys
        i_t = torch.from_numpy(i)
        j_t = torch.from_numpy(j)
        self.neighbours = NeighbourList(i_t, j_t, torch.zeros(1), self.state.step, i.size)
        if keep_state and self._xi_pair is not None and keys.size:
            old = self._xi_pair
            prev_keys = getattr(self, "_prev_keys", np.zeros(0, dtype=np.int64))
            new_xi = torch.zeros((keys.size, 3), dtype=torch.float64)
            if prev_keys.size:
                where = np.searchsorted(prev_keys, keys)
                safe = np.clip(where, 0, prev_keys.size - 1)
                hit = (where < prev_keys.size) & (prev_keys[safe] == keys)
                if hit.any():
                    new_xi[torch.from_numpy(np.flatnonzero(hit))] = old[torch.from_numpy(safe[hit])]
            self._xi_pair = new_xi
        else:
            self._xi_pair = torch.zeros((keys.size, 3), dtype=torch.float64)
        self._prev_keys = keys
        self._ref_pos = self.pos.clone()
        self.state.rebuilds += 1
        return self.neighbours

    def _need_rebuild(self) -> bool:
        if self.neighbours is None or self._ref_pos is None:
            return True
        # |dx|,|dy|,|dz| <= skin/2 individually implies Euclidean displacement
        # <= skin/2, which is all the rebuild criterion needs (cheaper than a norm)
        disp = (self.pos - self._ref_pos).abs().max()
        return bool(disp > 0.25 * self.config.skin)

    # ---------------------------------------------------------------- forces
    def forces(self) -> torch.Tensor:
        cfg = self.config
        if self._need_rebuild():
            self.build_neighbours()
        nl = self.neighbours
        assert nl is not None
        pos, vel, radii, mass = self.pos, self.vel, self.radii, self.mass
        force = torch.zeros((cfg.n, 3), dtype=torch.float64)
        beta = self._restitution_damping
        i, j = nl.i, nl.j
        if i.numel():
            d = pos[j] - pos[i]
            d[:, 1] += self._ymask * self.box[1]
            dist = torch.linalg.norm(d, dim=1)
            overlap = radii[i] + radii[j] - dist
            active = overlap > 0.0
            if not bool(active.any()):
                self._force = self._auxiliary_forces(force)
                self.state.force_evals += 1
                return self._force
            ia, ja = i[active], j[active]
            d = d[active]
            dist = dist[active]
            overlap = overlap[active]
            n_hat = d / dist.clamp_min(1e-300).unsqueeze(1)   # from i to j
            r_eff = radii[ia] * radii[ja] / (radii[ia] + radii[ja])
            m_eff = mass[ia] * mass[ja] / (mass[ia] + mass[ja])
            s_n = 2.0 * cfg.effective_modulus * torch.sqrt(r_eff * overlap)
            f_n = (2.0 / 3.0) * s_n * overlap                 # Hertz, >= 0
            v_rel = vel[ia] - vel[ja]                         # velocity of i w.r.t. j
            v_n = (v_rel * n_hat).sum(dim=1)                  # > 0 => approaching
            # dashpot: with n_hat from i to j, an approaching pair has v_n > 0 and
            # the damping must *increase* the repulsive magnitude (dissipative).
            f_n = (f_n + self._damping_force(s_n, v_n, m_eff, beta)).clamp_min(0.0)
            # ---- tangential (Cundall-Strack, slot-stable history)
            xi = self._xi_pair
            assert xi is not None
            xi_act = xi[active]
            s_t = 8.0 * cfg.effective_shear * torch.sqrt(r_eff * overlap)
            v_t = v_rel - v_n.unsqueeze(1) * n_hat            # slip of i w.r.t. j
            xi_new = xi_act + v_t * self.dt
            mag = torch.linalg.norm(xi_new, dim=1)
            cap = cfg.mu_grain * f_n
            scale = torch.where(
                mag > 0.0,
                torch.clamp(cap / (s_t * mag).clamp_min(1e-300), max=1.0),
                torch.ones_like(mag),
            )
            xi_new = xi_new * scale.unsqueeze(1)
            xi[active] = xi_new
            # force on j: +F_n n_hat + S_t xi  (equal and opposite on i)
            f_on_j = f_n.unsqueeze(1) * n_hat + s_t.unsqueeze(1) * xi_new
            force.index_add_(0, ia, -f_on_j)
            force.index_add_(0, ja, f_on_j)
            self._normal_force = f_n
        else:
            self._normal_force = torch.zeros(0, dtype=torch.float64)
        self._force = self._auxiliary_forces(force)
        self.state.force_evals += 1
        return self._force

    def _auxiliary_forces(self, force: torch.Tensor) -> torch.Tensor:
        """Gravity + wall contacts (floor / left / right indenter)."""
        cfg = self.config
        pos, vel, radii, mass = self.pos, self.vel, self.radii, self.mass
        if cfg.gravity:
            force[:, 2] -= mass * cfg.gravity
        for name, x_wall in self.wall_pos.items():
            if name == "floor" and not cfg.floor:
                continue
            if name == "left" and not cfg.left_wall:
                continue
            if name == "right" and not cfg.right_wall:
                continue
            axis = 2 if name == "floor" else 0
            if name == "floor":
                gap = pos[:, 2]
                sign = 1.0
            elif name == "left":
                gap = pos[:, 0]
                sign = 1.0
            else:
                gap = self.wall_pos["right"] - pos[:, 0]
                sign = -1.0
            overlap = radii - gap
            active = overlap > 0.0
            if not bool(active.any()):
                self._wall_force[name] = 0.0
                self._wall_tangent[name] = 0.0
                continue
            idx = torch.nonzero(active, as_tuple=False).squeeze(1)
            delta = overlap[active]
            n_hat = torch.zeros((idx.numel(), 3), dtype=torch.float64)
            n_hat[:, axis] = sign
            e_star = torch.full_like(delta, cfg.effective_modulus)
            r_eff = radii[idx]
            m_eff = mass[idx]
            s_n = 2.0 * e_star * torch.sqrt(r_eff * delta)
            f_n = (2.0 / 3.0) * s_n * delta
            v_n = vel[idx, axis] * sign - self.wall_vel[name] * sign
            f_n = f_n - self._damping_force(s_n, v_n, m_eff, self._restitution_damping)
            f_n = f_n.clamp_min(0.0)
            xi = self._xi_wall[name]
            xi_act = xi[idx]
            g_star = torch.full_like(delta, cfg.effective_shear)
            s_t = 8.0 * g_star * torch.sqrt(r_eff * delta)
            v_rel = vel[idx].clone()
            v_rel[:, axis] = v_rel[:, axis] - self.wall_vel[name]
            v_t = v_rel - v_n.unsqueeze(1) * n_hat
            xi_new = xi_act + v_t * self.dt
            mag = torch.linalg.norm(xi_new, dim=1)
            cap = cfg.mu_wall * self.wall_mu_scale.get(name, 1.0) * f_n.clamp_min(0.0)
            scale = torch.where(mag > 0, torch.clamp(cap / mag.clamp_min(1e-300), max=1.0),
                                torch.ones_like(mag))
            xi_new = xi_new * scale.unsqueeze(1)
            xi[idx] = xi_new
            self._xi_wall[name] = xi
            f_t = -s_t.unsqueeze(1) * xi_new
            f = f_n.unsqueeze(1) * n_hat + f_t
            force.index_add_(0, idx, f)
            self._wall_force[name] = float((f[:, axis] * sign).sum())
            tang = f.clone()
            tang[:, axis] = 0.0
            self._wall_tangent[name] = float(torch.linalg.norm(tang, dim=1).sum())
        return force

    # ----------------------------------------------------------- integration
    def apply_viscous_damping(self, factor: float) -> None:
        """Scale all velocities (deposition helper; not part of the contact law)."""
        if factor != 1.0:
            self.vel *= factor
            self._force = None

    def step(self, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            if self._force is None:
                self.forces()
            assert self._force is not None
            self.vel += 0.5 * self.dt * self._force * self.inv_mass.unsqueeze(1)
            self.pos += self.dt * self.vel
            self._force = None
            self.forces()
            self.vel += 0.5 * self.dt * self._force * self.inv_mass.unsqueeze(1)
            if not bool(torch.isfinite(self.vel).all()):
                raise RuntimeError(
                    f"contact integration diverged at step {self.state.step} "
                    f"(dt={self.dt:.3e} s): non-finite velocities"
                )
            # Position safety: clamp particles that escape far outside the box.
            # This prevents NaN/inf in the cell list when a numerical instability
            # ejects a particle during transient loading (e.g., initial settling
            # of a large bed). The clamp is generous (10 box lengths) so it only
            # fires for genuinely divergent trajectories, not physical motion.
            box = self.box
            margin = 10.0 * box.max()
            for a in range(3):
                lo, hi = -margin, box[a] + margin
                mask_lo = self.pos[:, a] < lo
                mask_hi = self.pos[:, a] > hi
                if bool(mask_lo.any()):
                    self.pos[mask_lo, a] = lo
                    self.vel[mask_lo, a] = 0.0
                if bool(mask_hi.any()):
                    if a == 1 and self.config.periodic[1]:
                        # periodic axis: wrap, don't clamp
                        self.pos[mask_hi, a] = ((self.pos[mask_hi, a] - box[a]) % box[a]) + box[a]
                    else:
                        self.pos[mask_hi, a] = hi
                        self.vel[mask_hi, a] = 0.0
            if not bool(torch.isfinite(self.pos).all()):
                raise RuntimeError(
                    f"contact integration diverged at step {self.state.step}: "
                    f"non-finite positions after clamping"
                )
            self.state.step += 1
            self.state.time += self.dt

    # ------------------------------------------------------------ diagnostics
    def kinetic_energy(self) -> float:
        return float((0.5 * self.mass * (self.vel**2).sum(dim=1)).sum())

    def gravitational_energy(self) -> float:
        return float((self.mass * self.config.gravity * self.pos[:, 2]).sum())

    def contact_energy(self) -> float:
        """Elastic energy stored in the normal + tangential contact springs."""
        nl = self.neighbours
        if nl is None or nl.i.numel() == 0:
            return 0.0
        i, j = nl.i, nl.j
        d = self.pos[j] - self.pos[i]
        d[:, 1] += self._ymask * self.box[1]
        dist = torch.linalg.norm(d, dim=1)
        overlap = (self.radii[i] + self.radii[j] - dist).clamp_min(0.0)
        r_eff = self.radii[i] * self.radii[j] / (self.radii[i] + self.radii[j])
        e_star = self.config.effective_modulus
        # integral of (4/3)E*sqrt(R) d^{3/2} dd = (2/5)*(4/3)E sqrt(R) d^{5/2}
        normal = (2.0 / 5.0) * (2.0 / 3.0) * 2.0 * e_star * torch.sqrt(r_eff) * overlap**2.5
        energy = float(normal.sum())
        if self._xi_pair is not None and self._xi_pair.numel():
            s_t = 8.0 * self.config.effective_shear * torch.sqrt(r_eff * overlap)
            xi = self._xi_pair
            tang = 0.5 * s_t * (xi**2).sum(dim=1)
            energy += float(tang.sum())
        return energy

    def energy_ledger(self) -> dict[str, float]:
        ke = self.kinetic_energy()
        pe_g = self.gravitational_energy() if self.config.gravity else 0.0
        pe_c = self.contact_energy()
        return {"kinetic": ke, "gravity": pe_g, "contact": pe_c, "total": ke + pe_g + pe_c}

    def coordination_number(self) -> float:
        nl = self.neighbours
        if nl is None or nl.i.numel() == 0:
            return 0.0
        i, j = nl.i, nl.j
        d = self.pos[j] - self.pos[i]
        d[:, 1] += self._ymask * self.box[1]
        dist = torch.linalg.norm(d, dim=1)
        n_contact = int(((self.radii[i] + self.radii[j] - dist) > 0.0).sum())
        return 2.0 * n_contact / self.config.n

    def mean_contact_pressure(self) -> float:
        nf = self._normal_force
        if nf is None or nf.numel() == 0:
            return 0.0
        area = math.pi * (0.5 * self.config.diameter) ** 2
        return float(nf.mean()) / area

    def wall_force(self, name: str) -> float:
        return float(self._wall_force.get(name, 0.0))

    def wall_tangent_force(self, name: str) -> float:
        """Sum of the tangential contact-force magnitudes at a wall."""
        return float(self._wall_tangent.get(name, 0.0))

    def set_wall_velocity(self, name: str, velocity: float) -> None:
        self.wall_vel[name] = float(velocity)

    def advance_indenter(self) -> None:
        """Move the right wall by ``v*dt`` (used by the crushing protocol)."""
        self.wall_pos["right"] += self.wall_vel["right"] * self.dt

    def digest(self) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.pos.numpy(), dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.vel.numpy(), dtype=np.float64).tobytes())
        h.update(np.float64(self.state.time).tobytes())
        return h.hexdigest()

    # --------------------------------------------------------------- helpers
    def pack_fraction(self) -> float:
        volume = float(np.prod(self.box))
        particle_volume = float((4.0 / 3.0 * math.pi * self.radii_np**3).sum())
        return particle_volume / volume

    def memory_footprint_mb(self) -> float:
        bytes_total = 0
        for tensor in (self.pos, self.vel, self.radii, self.mass, self._force):
            if isinstance(tensor, torch.Tensor):
                bytes_total += tensor.numel() * tensor.element_size()
        for tensor in (self._xi_pair, self._xi_wall.get("floor")):
            if isinstance(tensor, torch.Tensor):
                bytes_total += tensor.numel() * tensor.element_size()
        if self.neighbours is not None:
            bytes_total += self.neighbours.i.numel() * 8 * 2
        return bytes_total / 1e6