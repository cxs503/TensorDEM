"""Two-dimensional rigid-disk DEM kernel for ice-ridge formation and ship penetration.

Idealisation
------------
Plane strain, unit out-of-plane thickness.  Ice rubble is represented by
rotating disks (radius ``r``, density ``rho_ice``) in water of density
``rho_water``.  Every interaction contributes to a closed work ledger so that
``W_ext + W_buoy = dKE + dU_g + dE_spring + E_break + E_diss`` is checked
instead of assumed.

Interactions
------------
* unbonded contact: linear spring-dashpot normal law plus a Cundall-Strack
  tangential spring (incremental, capped at ``mu F_n``).  The disks are
  **rotationless** (central-force, like ``tensordem.ridge_dem``): the rotational
  degree of freedom was removed after it was measured to spin up without bound
  (a saturated sliding torque of ``r mu F_n`` has no counter-torque once the
  rolling resistance saturates, and disk spins of 2e4 rad/s were observed).
  Consequence: blocks slide instead of rolling, so the emergent angle of repose
  is an upper bound; the macro friction angle is *measured*, not assumed.
  The tangential spring stores energy (counted in the ledger) and slip past the
  cap is the dissipative channel;
* freeze bonds: elastic-brittle normal spring plus incrementally accumulated
  tangential spring in series with tensile ``sigma_t`` / shear ``tau_s``
  strength; a broken bond charges its released elastic energy to the fracture
  channel;
* buoyancy: exact submerged area of a disk cut by the waterline,
  ``A = pi r^2/2 + c sqrt(r^2-c^2) + r^2 asin(c/r)`` with ``c = -y_c``;
* translational rigid segments (drive platens, ship-bow faces) with their own
  velocity, a tangential spring per (segment, disk) and dashpot damping.

Boundary work is accumulated as ``F_on_particle . v_segment`` (power delivered
by the boundary to the ice) plus the tangential-spring energy rate, so the
dissipation generated at the boundary sits on the receipt side of the ledger.

Why a tangential spring: a velocity-regularised Coulomb law ``tanh(v_t/eps)``
has an effective tangential damping ``mu F_n / eps``; with ``F_n ~ 1e5 N`` and
``eps ~ 0.02 m/s`` that demands ``dt < 1e-5 s`` and the first implementation
blew up (disk speeds of 245 m/s were observed).  The spring is bounded by
``k_t`` instead, the usual DEM stability limit.

Scope limits are documented in ``docs/ICE_RIDGE_MECHANICS_DEM.md``: unit
out-of-plane thickness, disks carry no bending stiffness, water enters only as
exact buoyancy and linear drag (no pore pressure, no added mass), and rolling
resistance is viscous rather than a true rolling-friction spring.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

GRAVITY = 9.81

__all__ = [
    "GRAVITY",
    "IceRubbleMaterial",
    "WallSegment",
    "Assembly",
    "bond_components",
]


def _finite(name: str, value, *, positive: bool = False, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class IceRubbleMaterial:
    """Micro-parameters of the disk assembly (plane strain, unit thickness)."""

    particle_radius: float = 0.25
    ice_density: float = 917.0
    water_density: float = 1025.0
    gravity: float = GRAVITY
    contact_stiffness: float = 1.0e7
    tangential_stiffness: float = 5.0e6
    contact_damping: float = 2.0e4
    friction: float = 0.45
    rolling_friction: float = 0.01
    bond_stiffness: float = 3.0e6
    bond_shear_stiffness: float = 1.5e6
    tensile_strength: float = 3.0e5
    shear_strength: float = 6.0e5
    omega_eps: float = 1.0
    drag: float = 4.0e2

    def __post_init__(self) -> None:
        for name in ("particle_radius", "ice_density", "water_density",
                     "contact_stiffness", "tangential_stiffness",
                     "bond_stiffness", "bond_shear_stiffness",
                     "tensile_strength", "shear_strength"):
            _finite(name, getattr(self, name), positive=True)
        for name in ("contact_damping", "friction", "rolling_friction", "drag"):
            _finite(name, getattr(self, name), nonnegative=True)
        _finite("omega_eps", self.omega_eps, positive=True)
        if self.tangential_stiffness > self.contact_stiffness:
            raise ValueError("tangential_stiffness must not exceed contact_stiffness")

    @property
    def particle_mass(self) -> float:
        return self.ice_density * math.pi * self.particle_radius ** 2

    @property
    def bond_area(self) -> float:
        """Freeze-bond cross-section, m^2 (diameter x unit thickness)."""
        return 2.0 * self.particle_radius

    @property
    def critical_dt(self) -> float:
        stiffness = self.contact_stiffness + self.tangential_stiffness
        return 2.0 / math.sqrt(stiffness / self.particle_mass)

    def recommended_dt(self, safety: float = 0.15) -> float:
        _finite("safety", safety, positive=True)
        return safety * self.critical_dt


@dataclass
class WallSegment:
    """Translational rigid segment (platen face or ship-bow face)."""

    p0: torch.Tensor
    p1: torch.Tensor
    velocity: torch.Tensor
    friction: float = 0.2
    label: str = "wall"
    tang: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.p0 = torch.as_tensor(self.p0, dtype=torch.float64).reshape(2)
        self.p1 = torch.as_tensor(self.p1, dtype=torch.float64).reshape(2)
        self.velocity = torch.as_tensor(self.velocity, dtype=torch.float64).reshape(2)

    def advance(self, dt: float) -> None:
        self.p0 = self.p0 + self.velocity * dt
        self.p1 = self.p1 + self.velocity * dt

    def ensure_tang(self, count: int) -> torch.Tensor:
        if self.tang is None or self.tang.shape[0] != count or self.tang.dim() != 2:
            self.tang = torch.zeros((count, 2), dtype=torch.float64)
        return self.tang


class Assembly:
    """A 2D disk assembly with buoyancy, freeze bonds and rigid segments."""

    def __init__(
        self,
        positions,
        radii,
        material: IceRubbleMaterial,
        *,
        velocities=None,
        waterline: float = 0.0,
        drag: float | None = None,
        fixed=None,
    ) -> None:
        pos = torch.as_tensor(positions, dtype=torch.float64).reshape(-1, 2)
        count = int(pos.shape[0])
        if isinstance(radii, (int, float)):
            rad = torch.full((count,), float(radii), dtype=torch.float64)
        else:
            rad = torch.as_tensor(radii, dtype=torch.float64).reshape(-1)
        if rad.numel() != count:
            raise ValueError("radii and positions must have the same length")
        self.pos = pos.clone()
        self.radii = rad.clone()
        self.material = material
        self.waterline = float(waterline)
        self.drag = material.drag if drag is None else float(drag)
        self.mass = material.ice_density * math.pi * self.radii ** 2
        self.inertia = 0.5 * self.mass * self.radii ** 2
        self.vel = (torch.zeros((count, 2), dtype=torch.float64)
                    if velocities is None
                    else torch.as_tensor(velocities, dtype=torch.float64).reshape(-1, 2).clone())
        self.omega = torch.zeros(count, dtype=torch.float64)
        self.theta = torch.zeros(count, dtype=torch.float64)
        if fixed is None:
            self.fixed = torch.zeros(count, dtype=torch.bool)
        else:
            self.fixed = torch.as_tensor(fixed, dtype=torch.bool).reshape(-1).clone()
        self.bonds = torch.zeros((0, 2), dtype=torch.long)
        self.bond_rest = torch.zeros(0, dtype=torch.float64)
        self.bond_shear = torch.zeros((0, 2), dtype=torch.float64)
        self.bond_alive = torch.zeros(0, dtype=torch.bool)
        self.walls: list[WallSegment] = []
        self.work_boundaries = 0.0
        self.work_buoyancy = 0.0
        self.dissipation = {
            "contact_damping": 0.0,
            "contact_friction": 0.0,
            "rolling": 0.0,
            "wall_damping": 0.0,
            "wall_friction": 0.0,
            "drag": 0.0,
        }
        self.break_energy = 0.0
        self.broken_bonds = 0
        self.time = 0.0
        self.steps = 0
        self._pairs = torch.zeros((0, 2), dtype=torch.long)
        self._pair_keys = torch.zeros(0, dtype=torch.long)
        self._tang = torch.zeros((0, 2), dtype=torch.float64)
        self._rebuild_every = 4
        self._since_rebuild = 0
        self.initial_gravity_potential = self.gravity_potential()
        self.initial_spring = self.spring_energy()["total"]
        self.wall_reaction = torch.zeros((0, 2), dtype=torch.float64)
        self._bonded_key = torch.zeros(0, dtype=torch.long)
        self.max_penetration = 0.0
        self.external_force = torch.zeros((count, 2), dtype=torch.float64)
        self._rebuild_neighbours()

    # ------------------------------------------------------------------ helpers
    @property
    def count(self) -> int:
        return int(self.pos.shape[0])

    def add_bonds(self, pairs, rest_lengths=None) -> None:
        pairs = torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2)
        if rest_lengths is None:
            rest = torch.linalg.vector_norm(self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]], dim=1)
        else:
            rest = torch.as_tensor(rest_lengths, dtype=torch.float64).reshape(-1)
        self.bonds = torch.cat((self.bonds, pairs), dim=0)
        self.bond_rest = torch.cat((self.bond_rest, rest))
        self.bond_shear = torch.cat(
            (self.bond_shear, torch.zeros((pairs.shape[0], 2), dtype=torch.float64)), dim=0)
        self.bond_alive = torch.cat(
            (self.bond_alive, torch.ones(pairs.shape[0], dtype=torch.bool)))
        self._refresh_bond_keys()

    def _refresh_bond_keys(self) -> None:
        if self.bonds.numel() == 0:
            self._bonded_key = torch.zeros(0, dtype=torch.long)
            return
        lo = torch.minimum(self.bonds[:, 0], self.bonds[:, 1])
        hi = torch.maximum(self.bonds[:, 0], self.bonds[:, 1])
        self._bonded_key = lo * self.count + hi

    def add_wall(self, wall: WallSegment) -> WallSegment:
        self.walls.append(wall)
        return wall

    def advance_walls(self, dt: float) -> None:
        for wall in self.walls:
            wall.advance(dt)

    def kinetic_energy(self) -> float:
        translational = float(0.5 * (self.mass * (self.vel ** 2).sum(dim=1)).sum())
        rotational = float(0.5 * (self.inertia * self.omega ** 2).sum())
        return translational + rotational

    def gravity_potential(self) -> float:
        m = self.material
        return float((self.mass * m.gravity * (self.pos[:, 1] - self.waterline)).sum())

    def submerged_area(self) -> torch.Tensor:
        """Exact area of each disk below the waterline, m^2 per metre width."""
        r = self.radii
        wet = self.waterline - self.pos[:, 1]
        c = torch.clamp(wet, -r, r)
        root = torch.clamp(r * r - c * c, min=0.0).sqrt()
        area = (0.5 * math.pi * r * r + c * root
                + r * r * torch.asin(torch.clamp(c / r, -1.0, 1.0)))
        area = torch.where(wet >= r, math.pi * r * r, area)
        area = torch.where(wet <= -r, torch.zeros_like(area), area)
        return area.clamp_min(0.0)

    def max_overlap(self) -> float:
        if self._pairs.numel() == 0:
            return 0.0
        i, j = self._pairs[:, 0], self._pairs[:, 1]
        distance = torch.linalg.vector_norm(self.pos[j] - self.pos[i], dim=1)
        return float((self.radii[i] + self.radii[j] - distance).max())

    # -------------------------------------------------------------- neighbours
    def _rebuild_neighbours(self) -> None:
        self._since_rebuild = 0
        if self.count < 2:
            self._pairs = torch.zeros((0, 2), dtype=torch.long)
            self._pair_keys = torch.zeros(0, dtype=torch.long)
            self._tang = torch.zeros((0, 2), dtype=torch.float64)
            return
        cutoff = 2.0 * float(self.radii.max()) * 1.10
        tree = cKDTree(self.pos.numpy())
        pairs = tree.query_pairs(cutoff, output_type="ndarray").reshape(-1, 2)
        new_pairs = torch.as_tensor(pairs, dtype=torch.long)
        new_keys = torch.minimum(new_pairs[:, 0], new_pairs[:, 1]) * self.count \
            + torch.maximum(new_pairs[:, 0], new_pairs[:, 1])
        tang = torch.zeros((new_keys.numel(), 2), dtype=torch.float64)
        if self._pair_keys.numel() and new_keys.numel():
            order = torch.argsort(self._pair_keys)
            sorted_keys = self._pair_keys[order]
            slot = torch.searchsorted(sorted_keys, new_keys).clamp(max=sorted_keys.numel() - 1)
            found = sorted_keys[slot] == new_keys
            tang = torch.where(found[:, None], self._tang[order][slot], tang)
        self._pairs = new_pairs
        self._pair_keys = new_keys
        self._tang = tang

    def _bonded_mask(self, i, j) -> torch.Tensor:
        if self._bonded_key.numel() == 0:
            return torch.zeros(i.numel(), dtype=torch.bool)
        key = torch.minimum(i, j) * self.count + torch.maximum(i, j)
        return torch.isin(key, self._bonded_key)

    # ------------------------------------------------------------------ forces
    def _bond_forces(self, force, dt, update_memory: bool) -> None:
        """Elastic-brittle freeze bonds with a vector tangential spring."""
        if self.bonds.numel() == 0:
            return
        m = self.material
        if not bool(self.bond_alive.any()):
            return
        idx = torch.nonzero(self.bond_alive, as_tuple=False).flatten()
        i = self.bonds[idx, 0]
        j = self.bonds[idx, 1]
        delta = self.pos[j] - self.pos[i]
        length = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
        normal = delta / length[:, None]
        extension = length - self.bond_rest[idx]
        normal_force = m.bond_stiffness * extension
        rel = self.vel[i] - self.vel[j]
        rel_n = (rel * normal).sum(dim=1)
        v_t_vec = rel - rel_n[:, None] * normal
        # tangential spring stored as a vector and projected back onto the
        # contact plane.  A scalar stored along the tangent pours energy into
        # the lattice when the bond rotates: measured v_rel 10.4 m/s after
        # 0.5 s with a scalar, 0.012 m/s with the vector form.
        xi = self.bond_shear[idx]
        if update_memory and dt is not None:
            xi = xi + v_t_vec * dt
            xi = xi - (xi * normal).sum(dim=1)[:, None] * normal
        shear_vec = -m.bond_shear_stiffness * xi
        shear_mag = torch.linalg.vector_norm(shear_vec, dim=1)
        area = m.bond_area
        broken = ((normal_force > m.tensile_strength * area)
                  | (shear_mag > m.shear_strength * area))
        if bool(broken.any()):
            released = float((0.5 * m.bond_stiffness * extension[broken] ** 2
                              + 0.5 * m.bond_shear_stiffness
                              * (xi[broken] ** 2).sum(dim=1)).sum())
            self.bond_alive[idx[broken]] = False
            self.broken_bonds += int(broken.sum())
            self.break_energy += released
            self._refresh_bond_keys()
            keep = ~broken
            idx, i, j = idx[keep], i[keep], j[keep]
            normal = normal[keep]
            normal_force, shear_vec = normal_force[keep], shear_vec[keep]
            shear_mag = shear_mag[keep]
            xi = xi[keep]
        if i.numel() == 0:
            return
        pair_force = normal_force[:, None] * normal + shear_vec
        force.index_add_(0, i, pair_force)
        force.index_add_(0, j, -pair_force)
        if update_memory and dt is not None:
            self.bond_shear[idx] = xi

    def _contact_forces(self, dt, update_memory: bool):
        """Unbonded disk-disk contacts (spring-dashpot normal + capped tangential)."""
        m = self.material
        n = self.count
        force = torch.zeros((n, 2), dtype=torch.float64)
        torque = torch.zeros(n, dtype=torch.float64)
        store = {"contact_damping": 0.0, "contact_friction": 0.0, "rolling": 0.0}
        if self._pairs.numel() == 0:
            return force, torque, store, 0, torch.zeros(0)
        pairs = self._pairs
        i, j = pairs[:, 0], pairs[:, 1]
        bonded = self._bonded_mask(i, j)
        delta = self.pos[j] - self.pos[i]
        distance = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
        overlap = self.radii[i] + self.radii[j] - distance
        # bonded pairs are carried by the bond springs; _tang stays aligned with
        # self._pairs (bonded slots are kept at zero)
        active = (overlap > 0.0) & ~bonded
        if not bool(active.any()):
            if update_memory:
                self._tang = torch.zeros_like(self._tang)
            return force, torque, store, 0, torch.zeros(0)
        ai, aj = i[active], j[active]
        ov = overlap[active]
        # normal points from j to i, so +magnitude*normal is repulsive on i
        normal = -delta[active] / distance[active][:, None]
        rel = self.vel[ai] - self.vel[aj]
        v_n = (rel * normal).sum(dim=1)
        v_t_vec = rel - v_n[:, None] * normal
        normal_mag = (m.contact_stiffness * ov - m.contact_damping * v_n).clamp_min(0.0)
        old_xi = self._tang[active]
        xi = old_xi + v_t_vec * dt if (update_memory and dt is not None) else old_xi
        xi = xi - (xi * normal).sum(dim=1)[:, None] * normal
        shear_vec = -m.tangential_stiffness * xi
        shear_mag = torch.linalg.vector_norm(shear_vec, dim=1)
        cap = m.friction * normal_mag
        slip = shear_mag > cap
        scale = torch.where(slip, cap / shear_mag.clamp_min(1e-30), torch.ones_like(cap))
        shear_vec = shear_vec * scale[:, None]
        if update_memory:
            xi = xi * scale[:, None]
            self._tang[active] = xi
            self._tang[~active] = 0.0
        pair_force = normal_mag[:, None] * normal + shear_vec
        force.index_add_(0, ai, pair_force)
        force.index_add_(0, aj, -pair_force)
        loaded = (m.contact_stiffness * ov - m.contact_damping * v_n) > 0.0
        store["contact_damping"] = float((m.contact_damping * v_n ** 2 * loaded).sum())
        shear_power = float((shear_vec * v_t_vec).sum())
        spring_rate = float(0.5 * m.tangential_stiffness
                            * ((xi ** 2).sum(dim=1) - (old_xi ** 2).sum(dim=1)).sum()) \
            / max(dt or 1.0, 1e-12)
        # the spring stores part of the tangential work; the remainder (slip) is
        # the dissipated channel
        store["contact_friction"] = -shear_power - spring_rate
        store["rolling"] = 0.0
        return force, torque, store, int(ai.numel()), normal_mag

    def _wall_forces(self, force, torque, dt, update_memory: bool):
        m = self.material
        reactions = []
        store = {"wall_damping": 0.0, "wall_friction": 0.0}
        reaction_power = {}
        for wall in self.walls:
            a, b = wall.p0, wall.p1
            ab = b - a
            length2 = float((ab * ab).sum())
            tang_mem = wall.ensure_tang(self.count)
            reaction = torch.zeros(2, dtype=torch.float64)
            if length2 > 0.0:
                t_param = ((self.pos - a) @ ab / length2).clamp(0.0, 1.0)
                closest = a + t_param[:, None] * ab
                delta = self.pos - closest
                distance = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
                overlap = self.radii - distance
                hit = overlap > 0.0
                if bool(hit.any()):
                    hit_index = torch.nonzero(hit, as_tuple=False).flatten()
                    normal = delta[hit] / distance[hit][:, None]
                    rel = self.vel[hit] - wall.velocity
                    v_n = (rel * normal).sum(dim=1)
                    v_t_vec = rel - v_n[:, None] * normal
                    normal_mag = (m.contact_stiffness * overlap[hit]
                                  - m.contact_damping * v_n).clamp_min(0.0)
                    old_xi = tang_mem[hit_index]
                    xi = old_xi + v_t_vec * dt if (update_memory and dt is not None) else old_xi
                    xi = xi - (xi * normal).sum(dim=1)[:, None] * normal
                    shear_vec = -m.tangential_stiffness * xi
                    shear_mag = torch.linalg.vector_norm(shear_vec, dim=1)
                    cap = wall.friction * normal_mag
                    slip = shear_mag > cap
                    scale = torch.where(slip, cap / shear_mag.clamp_min(1e-30),
                                        torch.ones_like(cap))
                    shear_vec = shear_vec * scale[:, None]
                    if update_memory:
                        tang_mem[hit_index] = xi * scale[:, None]
                        tang_mem[~hit] = 0.0
                    pair_force = normal_mag[:, None] * normal + shear_vec
                    force.index_add_(0, hit_index, pair_force)
                    reaction -= pair_force.sum(dim=0)
                    loaded = (m.contact_stiffness * overlap[hit]
                              - m.contact_damping * v_n) > 0.0
                    store["wall_damping"] += float(
                        (m.contact_damping * v_n ** 2 * loaded).sum())
                    shear_power = float((shear_vec * v_t_vec).sum())
                    spring_rate = float(0.5 * m.tangential_stiffness
                                        * ((xi * scale[:, None]) ** 2
                                           - old_xi ** 2).sum()) / max(dt or 1.0, 1e-12)
                    store["wall_friction"] += -shear_power - spring_rate
                elif update_memory:
                    tang_mem[:] = 0.0
            reactions.append(reaction)
            # sign convention: the power delivered *by* the segment *to* the ice
            # is -F_on_segment . v_segment (the reaction on the segment is
            # opposite to the force the segment applies to the ice)
            reaction_power[wall.label] = -float((reaction * wall.velocity).sum())
        self.wall_reaction = (torch.stack(reactions) if reactions
                              else torch.zeros((0, 2), dtype=torch.float64))
        return store, reaction_power

    def forces(self, dt=None, *, update_memory: bool = True):
        m = self.material
        n = self.count
        force = torch.zeros((n, 2), dtype=torch.float64)
        torque = torch.zeros(n, dtype=torch.float64)
        self._bond_forces(force, dt, update_memory)
        cforce, ctorque, powers, contact_count, normal_force = self._contact_forces(
            dt, update_memory)
        force += cforce
        torque += ctorque
        wall_powers, reaction_power = self._wall_forces(force, torque, dt, update_memory)
        powers.update(wall_powers)
        gravity = torch.zeros((n, 2), dtype=torch.float64)
        gravity[:, 1] = -self.mass * m.gravity
        area = self.submerged_area()
        buoyancy = torch.zeros((n, 2), dtype=torch.float64)
        buoyancy[:, 1] = m.water_density * m.gravity * area
        force += gravity + buoyancy
        force += -self.drag * self.vel + self.external_force
        powers["drag"] = float((self.drag * (self.vel ** 2).sum(dim=1)).sum())
        return {
            "force": force,
            "torque": torque,
            "buoyancy": buoyancy,
            "powers": powers,
            "wall_power": reaction_power,
            "wall_reaction": self.wall_reaction,
            "contact_count": contact_count,
            "normal_force": normal_force,
        }

    # ------------------------------------------------------------------ energy
    def spring_energy(self) -> dict:
        m = self.material
        out = {"bonds": 0.0, "contacts": 0.0, "tangential": 0.0}
        if self.bonds.numel() and bool(self.bond_alive.any()):
            idx = torch.nonzero(self.bond_alive, as_tuple=False).flatten()
            i, j = self.bonds[idx, 0], self.bonds[idx, 1]
            length = torch.linalg.vector_norm(self.pos[j] - self.pos[i], dim=1)
            ext = length - self.bond_rest[idx]
            out["bonds"] = float((0.5 * m.bond_stiffness * ext ** 2).sum()
                                 + 0.5 * m.bond_shear_stiffness
                                 * (self.bond_shear[idx] ** 2).sum())
        if self._pairs.numel():
            i, j = self._pairs[:, 0], self._pairs[:, 1]
            bonded = self._bonded_mask(i, j)
            i, j = i[~bonded], j[~bonded]
            if i.numel():
                distance = torch.linalg.vector_norm(self.pos[j] - self.pos[i], dim=1)
                overlap = (self.radii[i] + self.radii[j] - distance).clamp_min(0.0)
                out["contacts"] = float((0.5 * m.contact_stiffness * overlap ** 2).sum())
            if self._tang.dim() == 2 and self._tang.shape[0] == self._pairs.shape[0]:
                out["tangential"] = float(0.5 * m.tangential_stiffness
                                          * (self._tang ** 2).sum())
        wall_spring = 0.0
        for wall in self.walls:
            if wall.tang is not None and wall.tang.numel():
                wall_spring += float(0.5 * m.tangential_stiffness
                                     * (wall.tang ** 2).sum())
        out["wall_tangential"] = wall_spring
        out["total"] = out["bonds"] + out["contacts"] + out["tangential"] + wall_spring
        return out

    # ------------------------------------------------------------- integration
    def step(self, dt: float, *, advance_walls: bool = True, update_memory: bool = True):
        _finite("dt", dt, positive=True)
        self._since_rebuild += 1
        if self._since_rebuild >= self._rebuild_every:
            self._rebuild_neighbours()
        out = self.forces(dt, update_memory=update_memory)
        force = out["force"].clone()
        torque = out["torque"].clone()
        if bool(self.fixed.any()):
            force[self.fixed] = 0.0
            torque[self.fixed] = 0.0
        acceleration = force / self.mass[:, None]
        angular = torque / self.inertia
        # symplectic Euler (Euler-Cromer); the half-kick variant
        # x += v dt + a dt^2/2 is unconditionally unstable (|lambda|^2 = 1+h^2/2)
        self.vel = self.vel + acceleration * dt
        self.omega = self.omega + angular * dt
        if bool(self.fixed.any()):
            self.vel[self.fixed] = 0.0
            self.omega[self.fixed] = 0.0
        self.pos = self.pos + self.vel * dt
        self.theta = self.theta + self.omega * dt
        penetration = float((self.radii[self._pairs[:, 0]] + self.radii[self._pairs[:, 1]]
                             - torch.linalg.vector_norm(
                                 self.pos[self._pairs[:, 1]] - self.pos[self._pairs[:, 0]],
                                 dim=1)).max()) if self._pairs.numel() else 0.0
        self.max_penetration = max(self.max_penetration, penetration)
        # work ledger (midpoint power = exact work for a symplectic-Euler step)
        for label, power in out["wall_power"].items():
            self.work_boundaries += power * dt
        # Euler-Cromer displaces by v_{n+1} dt, so F . v_{n+1} dt is the work
        self.work_buoyancy += float((out["buoyancy"] * self.vel).sum()) * dt
        for key, power in out["powers"].items():
            self.dissipation[key] = self.dissipation.get(key, 0.0) + power * dt
        if advance_walls:
            self.advance_walls(dt)
        self.time += dt
        self.steps += 1
        return out

    # ------------------------------------------------------------------ ledger
    def ledger(self) -> dict:
        ke = self.kinetic_energy()
        spring = self.spring_energy()["total"]
        gravity = self.gravity_potential()
        dissipation = sum(self.dissipation.values())
        received = self.work_boundaries + self.work_buoyancy
        accounted = ke + (gravity - self.initial_gravity_potential) \
            + (spring - self.initial_spring) + self.break_energy + dissipation
        residual = received - accounted
        scale = max(abs(received), abs(self.break_energy) + abs(dissipation), 1e-9)
        return {
            "work_boundaries": self.work_boundaries,
            "work_buoyancy": self.work_buoyancy,
            "work_external": received,
            "delta_kinetic": ke,
            "delta_gravity": gravity - self.initial_gravity_potential,
            "delta_spring": spring - self.initial_spring,
            "break_energy": self.break_energy,
            "dissipation": dict(self.dissipation),
            "dissipation_total": dissipation,
            "residual": residual,
            "relative_residual": residual / scale,
            "broken_bonds": self.broken_bonds,
            "max_penetration": self.max_penetration,
        }

    def bond_components(self) -> tuple[int, torch.Tensor]:
        """Connected components of the live bond graph (free disks are singletons)."""
        if not (self.bonds.numel() and bool(self.bond_alive.any())):
            return self.count, torch.arange(self.count, dtype=torch.long)
        live = self.bonds[self.bond_alive]
        graph = coo_matrix(
            (torch.ones(live.shape[0]), (live[:, 0].numpy(), live[:, 1].numpy())),
            shape=(self.count, self.count))
        number, labels = connected_components(graph, directed=False)
        return int(number), torch.as_tensor(labels, dtype=torch.long)


def bond_components(assembly: Assembly) -> tuple[int, torch.Tensor]:
    """Module-level alias so callers need not touch a method."""
    return assembly.bond_components()
