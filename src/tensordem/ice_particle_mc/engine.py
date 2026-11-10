"""A compact 3-D bonded-sphere engine for ice-particle Mohr-Coulomb verification.

Why this engine exists
----------------------
``tensordem.dem3d`` already provides bonded 3-D DEM, but its bond failure is
*strain based* (a single tensile/shear breaking strain), so it cannot express the
Mohr-Coulomb envelope ``tau = c + sigma_n tan(phi)`` that this verification
targets, and it builds specimens on a simple-cubic lattice (three bond
orientations only).  Rather than perturb a heavily regression-locked module,
this engine is a small, self-contained implementation of exactly the law under
test:

* bonded pair  : normal + tangential springs with a **Mohr-Coulomb rupture**
  criterion (normal tensile cut-off + ``tau <= c + sigma tan phi``), no parallel
  contact spring, so the bond stiffness is the only axial compliance -- which is
  what makes the elastic modulus predictable from the inputs;
* unbonded pair: Hertz-Mindlin normal law + incremental tangential spring with a
  Coulomb cap ``|F_t| <= mu F_n``;
* rigid plane walls (fixed floor, prescribed-velocity platen, masked punch) with
  optional Coulomb friction;
* a closed energy ledger: external work = kinetic + bond elastic + Hertz elastic
  + friction + fracture-release + viscous dissipation.

Everything is vectorised ``torch.float64`` and deterministic (no RNG in the
stepping loop).  All material and numerical parameters are inputs; no domain
constant is baked in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import torch

from .bond import (MohrCoulombStrength, bond_areas, bond_failure_mask,
                   normal_stiffness_from_modulus, shear_stiffness_from_normal,
                   weibull_scale_from_median)

__all__ = ["AssemblyConfig", "PlaneWall", "IceParticleAssembly", "candidate_pairs"]


def candidate_pairs(positions: torch.Tensor, radii: torch.Tensor,
                    reach_factor: float = 2.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Static candidate contact pairs within ``reach_factor * (r_i + r_j)``.

    The set is built once and kept for the whole run; a pair only becomes a
    contact when its *current* distance is below ``r_i + r_j``.  Reach 2.5 is
    ~1.25 lattice spacings of spare motion, which covers the deformation of every
    case in this verification; it is stated as a limitation (a pair that
    separates by more than that never re-contacts).
    """
    n = int(positions.shape[0])
    i, j = torch.triu_indices(n, n, offset=1)
    distance = (positions[j] - positions[i]).norm(dim=1)
    reach = (radii[i] + radii[j]) * reach_factor
    mask = distance <= reach
    return i[mask], j[mask]


@dataclass
class AssemblyConfig:
    """Bond + contact + wall parameters (SI units)."""

    # --- bond law -------------------------------------------------------
    bond_modulus: float = 9.0e9          # E_b: bond axial modulus (Pa)
    shear_stiffness_ratio: float = 0.4   # k_s / k_n
    cohesion_pa: float = 1.5e6
    friction_angle_deg: float = 30.0
    tensile_strength_pa: float = 1.0e6
    bond_strength_weibull_modulus: float = 0.0   # 0 = homogeneous
    bond_strength_seed: int = 0
    boundary_strengthening: float = 1.0  # multiplies strength in the end layers
    boundary_layer_thickness: float = 0.0  # m, 0 disables the boundary layer
    # --- unbonded contact law (after rupture / for granular beds) -------
    contact_young_modulus: float = 9.0e9
    contact_poisson_ratio: float = 0.33
    contact_friction: float = 0.2
    contact_damping: float = 0.0         # N s / m, normal dashpot
    # --- numerical ------------------------------------------------------
    density: float = 917.0
    density_scale: float = 1.0           # mass scaling for a larger stable dt
    time_step: float | None = None       # None -> stability estimate
    time_step_safety: float = 0.15
    viscous_damping: float = 0.0         # 1/s, global velocity damping
    pair_reach_factor: float = 2.5
    # --- diagnostics ----------------------------------------------------
    store_history: bool = True

    def __post_init__(self) -> None:
        if self.bond_modulus <= 0 or self.density <= 0:
            raise ValueError("bond_modulus and density must be positive")
        if self.bond_strength_weibull_modulus < 0:
            raise ValueError("weibull modulus must be >= 0")

    @property
    def strength(self) -> MohrCoulombStrength:
        return MohrCoulombStrength(self.cohesion_pa, self.friction_angle_deg,
                                   self.tensile_strength_pa)


@dataclass
class PlaneWall:
    """Rigid plane wall: the forbidden half-space is ``x . normal > offset``.

    ``axis == 2`` and ``normal = +1`` is a floor at ``height``; ``normal = -1`` is
    a descending platen.  ``mask_*`` optionally limits the wall to a rectangular
    patch on the other two axes (a flat punch of projected area ``width*depth``).
    """

    axis: int
    normal: float            # +1: material must stay below (x.axis < offset)
    offset: float            # plane position along the axis
    velocity: float = 0.0    # d offset / dt
    stiffness: float = 1.0e9
    damping: float = 0.0
    friction: float = 0.0
    label: str = "wall"
    mask_axes: tuple[int, int] | None = None
    mask_lo: tuple[float, float] = (-math.inf, -math.inf)
    mask_hi: tuple[float, float] = (math.inf, math.inf)

    def advance(self, dt: float) -> None:
        self.offset += self.velocity * dt

    def reach(self, positions: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """Particle mask on the wall patch (geometry only, no penetration test)."""
        if self.mask_axes is None:
            return torch.ones(positions.shape[0], dtype=torch.bool)
        a, b = self.mask_axes
        return ((positions[:, a] >= self.mask_lo[0]) & (positions[:, a] <= self.mask_hi[0])
                & (positions[:, b] >= self.mask_lo[1]) & (positions[:, b] <= self.mask_hi[1]))


class IceParticleAssembly:
    """Bonded (and, after rupture, frictional) particle assembly."""

    def __init__(self, specimen: Any, config: AssemblyConfig = AssemblyConfig(),
                 walls: Iterable[PlaneWall] = ()) -> None:
        self.config = config
        self.positions = specimen.positions.clone().to(torch.float64)
        self.radii = specimen.radii.clone().to(torch.float64)
        self.n = int(self.positions.shape[0])
        self.velocities = torch.zeros_like(self.positions)
        self.mass = config.density * config.density_scale * (4.0 / 3.0) * math.pi * self.radii.pow(3)
        self.volume = float(specimen.volume)

        self.bond_i = specimen.bond_i.clone()
        self.bond_j = specimen.bond_j.clone()
        self.bond_length0 = specimen.bond_length.clone().to(torch.float64)
        self.bond_area = bond_areas(self.volume, int(self.bond_i.numel()), self.bond_length0)
        kn = normal_stiffness_from_modulus(config.bond_modulus, self.bond_area,
                                           self.bond_length0)
        self.bond_kn = kn
        self.bond_ks = shear_stiffness_from_normal(kn, config.shear_stiffness_ratio)
        self.bond_shear = torch.zeros(int(self.bond_i.numel()), 3, dtype=torch.float64)
        self.bond_alive = torch.ones(int(self.bond_i.numel()), dtype=torch.bool)
        self.bond_strength_scale = self._strength_scale()
        self.bond_failure_mode = torch.zeros(int(self.bond_i.numel()), dtype=torch.int64)

        self.pair_i, self.pair_j = candidate_pairs(self.positions, self.radii,
                                                   config.pair_reach_factor)
        self.contact_shear = torch.zeros(int(self.pair_i.numel()), 3, dtype=torch.float64)
        self.bond_pair_index = self._match_bond_pairs()
        self.bonded_pair_mask = torch.zeros(int(self.pair_i.numel()), dtype=torch.bool)
        valid = self.bond_pair_index >= 0
        self.bonded_pair_mask[self.bond_pair_index[valid]] = True

        # kinematic (grip) particles: prescribed velocity, forces transmit but are
        # not integrated -- that is how the uniaxial test loads the specimen
        # without a penalty platen (which would limit the stable dt).
        self.kinematic = torch.zeros(self.n, dtype=torch.bool)
        self.kinematic_velocity = torch.zeros_like(self.positions)
        self.grip_work = 0.0
        self.walls = list(walls)
        self.wall_shear = [torch.zeros(self.n, 3, dtype=torch.float64) for _ in self.walls]

        self.time = 0.0
        self.dt = float(config.time_step or self._stable_dt())
        self.kinetic = 0.0
        self.bond_elastic = 0.0
        self.contact_elastic = 0.0
        self.fracture_dissipation = 0.0
        self.friction_dissipation = 0.0
        self.wall_friction_dissipation = 0.0
        self.viscous_dissipation = 0.0
        self.external_work = 0.0
        self.killed_bonds = 0
        self.history: list[dict[str, float]] = []
        self.wall_force = [torch.zeros(3, dtype=torch.float64) for _ in self.walls]
        self._last_forces = torch.zeros_like(self.positions)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _strength_scale(self) -> torch.Tensor:
        cfg = self.config
        m = int(self.bond_i.numel())
        scale = torch.ones(m, dtype=torch.float64)
        if cfg.bond_strength_weibull_modulus > 0.0:
            gen = torch.Generator().manual_seed(int(cfg.bond_strength_seed))
            u = torch.rand(m, generator=gen, dtype=torch.float64).clamp(1e-12, 1.0 - 1e-12)
            gamma = cfg.bond_strength_weibull_modulus
            # median-preserving: eps_i = (-ln(1-U))**(1/m) / ln(2)**(1/m)
            scale = ((-torch.log(1.0 - u)) ** (1.0 / gamma)) / (math.log(2.0) ** (1.0 / gamma))
        if cfg.boundary_layer_thickness > 0.0 and cfg.boundary_strengthening != 1.0:
            z = self.positions[:, 2]
            z_lo = float(z.min())
            z_hi = float(z.max())
            for idx, edge in ((0, z_lo + cfg.boundary_layer_thickness),
                              (1, z_hi - cfg.boundary_layer_thickness)):
                in_layer = ((z[self.bond_i] <= edge) if idx == 0
                            else (z[self.bond_i] >= edge)) | (
                    (z[self.bond_j] <= edge) if idx == 0 else (z[self.bond_j] >= edge))
                scale = torch.where(in_layer, scale * cfg.boundary_strengthening, scale)
        return scale

    def _match_bond_pairs(self) -> torch.Tensor:
        """Index of each bond in the candidate-pair list (-1 if absent)."""
        n = torch.tensor(self.n, dtype=torch.int64)
        pair_key = torch.minimum(self.pair_i, self.pair_j) * n + torch.maximum(self.pair_i, self.pair_j)
        bond_key = torch.minimum(self.bond_i, self.bond_j) * n + torch.maximum(self.bond_i, self.bond_j)
        order = torch.argsort(pair_key)
        sorted_key = pair_key[order]
        pos = torch.searchsorted(sorted_key, bond_key)
        pos = pos.clamp(max=max(int(sorted_key.numel()) - 1, 0))
        found = sorted_key[pos] == bond_key if sorted_key.numel() else torch.zeros_like(bond_key, dtype=torch.bool)
        index = torch.where(found, order[pos] if sorted_key.numel() else pos,
                            torch.full_like(bond_key, -1))
        return index

    def _stable_dt(self) -> float:
        """``dt = safety * min_i sqrt(m_i / sum_b (k_n + k_s))``.

        The bound must sum the stiffness of *all* springs attached to a particle
        (coordination matters): using only the largest single bond stiffness
        over-estimates the stable step by ``sqrt(z)`` -- with coordination z = 5
        that is 2.2x, and the run diverges.
        """
        if int(self.bond_i.numel()) == 0:
            return float(self.config.time_step_safety * 1e-6)
        stiffness = self.bond_kn + self.bond_ks
        per_particle = torch.zeros(self.n, dtype=torch.float64)
        per_particle.index_add_(0, self.bond_i, stiffness)
        per_particle.index_add_(0, self.bond_j, stiffness)
        per_particle = torch.where(per_particle > 0.0, per_particle,
                                   torch.full_like(per_particle, float(stiffness.max())))
        return float(self.config.time_step_safety
                     * torch.sqrt(self.mass / per_particle).min())

    # ------------------------------------------------------------------
    # forces
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _bond_forces(self, forces: torch.Tensor) -> None:
        if int(self.bond_i.numel()) == 0:
            return
        cfg = self.config
        i, j = self.bond_i, self.bond_j
        delta = self.positions[j] - self.positions[i]
        length = delta.norm(dim=1).clamp_min(1e-12)
        normal = delta / length[:, None]
        # compression-positive normal force
        f_n = self.bond_kn * (self.bond_length0 - length)
        # tangential spring, incremented by the relative tangential velocity
        v_rel = self.velocities[j] - self.velocities[i]
        v_t = v_rel - (v_rel * normal).sum(dim=1, keepdim=True) * normal
        shear = self.bond_shear + v_t * self.dt
        shear = shear - (shear * normal).sum(dim=1, keepdim=True) * normal
        self.bond_shear = torch.where(self.bond_alive[:, None], shear, self.bond_shear)
        # restoring: the force on j opposes the tangential displacement of j
        # relative to i (a +kq force would be a negative-stiffness runaway)
        f_t = -self.bond_ks[:, None] * self.bond_shear
        f_t_norm = f_t.norm(dim=1)
        area = self.bond_area
        scale = self.bond_strength_scale
        tensile_strength = cfg.tensile_strength_pa * scale
        cohesion = cfg.cohesion_pa * scale
        tan_phi = cfg.strength.tan_phi
        tensile = (-f_n) > tensile_strength * area          # tension-positive stress
        capacity = cohesion * area + torch.clamp(f_n, min=0.0) * tan_phi
        shear_fail = (f_t_norm > capacity) & ~tensile
        failed = (tensile | shear_fail) & self.bond_alive

        if bool(failed.any()):
            energy = 0.5 * self.bond_kn * (length - self.bond_length0).pow(2) \
                + 0.5 * self.bond_ks * self.bond_shear.pow(2).sum(dim=1)
            self.fracture_dissipation += float(energy[failed].sum())
            self.bond_alive = self.bond_alive & ~failed
            self.bond_failure_mode = torch.where(
                failed & tensile, torch.full_like(self.bond_failure_mode, 1),
                torch.where(failed & shear_fail,
                            torch.full_like(self.bond_failure_mode, 2),
                            self.bond_failure_mode))
            self.killed_bonds += int(failed.sum())
            self._sync_bonded_mask()

        alive = self.bond_alive
        f_n = torch.where(alive, f_n, torch.zeros_like(f_n))
        f_t = torch.where(alive[:, None], f_t, torch.zeros_like(f_t))
        total = f_n[:, None] * normal + f_t
        forces.index_add_(0, i, -total)
        forces.index_add_(0, j, total)
        self.bond_elastic = float(
            (0.5 * self.bond_kn * (length - self.bond_length0).pow(2)
             + 0.5 * self.bond_ks * self.bond_shear.pow(2).sum(dim=1))[alive].sum())

    @torch.no_grad()
    def _sync_bonded_mask(self) -> None:
        self.bonded_pair_mask[:] = False
        index = self.bond_pair_index
        valid = index >= 0
        alive_bond = self.bond_alive[valid]
        self.bonded_pair_mask[index[valid]] = alive_bond

    @torch.no_grad()
    def _contact_forces(self, forces: torch.Tensor) -> None:
        if int(self.pair_i.numel()) == 0:
            return
        cfg = self.config
        i, j = self.pair_i, self.pair_j
        delta = self.positions[j] - self.positions[i]
        distance = delta.norm(dim=1)
        overlap = self.radii[i] + self.radii[j] - distance
        active = (overlap > 0.0) & ~self.bonded_pair_mask
        if not bool(active.any()):
            self.contact_elastic = 0.0
            return
        normal = delta / distance.clamp_min(1e-12)[:, None]
        e_star = cfg.contact_young_modulus / (2.0 * (1.0 - cfg.contact_poisson_ratio ** 2))
        g_star = cfg.contact_young_modulus / (4.0 * (2.0 - cfg.contact_poisson_ratio)
                                              * (1.0 + cfg.contact_poisson_ratio))
        r_star = (self.radii[i] * self.radii[j]) / (self.radii[i] + self.radii[j])
        eff_overlap = torch.where(active, overlap, torch.zeros_like(overlap))
        f_n = (4.0 / 3.0) * e_star * r_star.sqrt() * eff_overlap.pow(1.5)
        # normal dashpot (approaching only), energy accounted below
        v_rel = self.velocities[j] - self.velocities[i]
        v_n = (v_rel * normal).sum(dim=1)
        f_damp = torch.where(active & (v_n < 0.0),
                             -cfg.contact_damping * v_n, torch.zeros_like(v_n))
        f_n = f_n + f_damp

        k_t = 8.0 * g_star * r_star.sqrt() * eff_overlap.clamp_min(1e-18).sqrt()
        v_t = v_rel - v_n[:, None] * normal
        shear = self.contact_shear + v_t * self.dt
        shear = torch.where(active[:, None], shear - (shear * normal).sum(dim=1, keepdim=True) * normal,
                            torch.zeros_like(shear))
        f_t = -k_t[:, None] * shear
        f_t_norm = f_t.norm(dim=1)
        cap = cfg.contact_friction * torch.clamp(f_n, min=0.0)
        capped = torch.clamp(f_t_norm, max=1.0)
        scale = torch.where(f_t_norm > cap, cap / f_t_norm.clamp_min(1e-30),
                            torch.ones_like(f_t_norm))
        f_t = f_t * scale[:, None]
        slip = torch.where(cap < f_t_norm, (f_t_norm - cap) / k_t.clamp_min(1e-30),
                           torch.zeros_like(f_t_norm))
        self.friction_dissipation += float((cap * slip)[active].sum())
        self.contact_shear = torch.where(active[:, None], shear * scale[:, None],
                                         self.contact_shear)
        total = f_n[:, None] * normal + f_t
        total = torch.where(active[:, None], total, torch.zeros_like(total))
        forces.index_add_(0, i, -total)
        forces.index_add_(0, j, total)
        self.contact_elastic = float(
            ((8.0 / 15.0) * e_star * r_star.sqrt() * eff_overlap.pow(2.5))[active].sum())
        self._last_contact_forces = (f_n[active].abs(), i[active], j[active])
        self._last_viscous = float((f_damp.abs() * v_n.abs())[active].sum())
        del capped

    @torch.no_grad()
    def _wall_forces(self, forces: torch.Tensor) -> None:
        for w_idx, wall in enumerate(self.walls):
            patch = wall.reach(self.positions, self.radii)
            coordinate = self.positions[:, wall.axis]
            penetration = (wall.offset - coordinate) * wall.normal - self.radii
            penetration = torch.where(patch, penetration, torch.full_like(penetration, -1.0))
            active = penetration > 0.0
            if not bool(active.any()):
                self.wall_force[w_idx] = torch.zeros(3, dtype=torch.float64)
                continue
            pen = torch.where(active, penetration, torch.zeros_like(penetration))
            v_axis = self.velocities[:, wall.axis]
            f_n = wall.stiffness * pen - wall.damping * v_axis * (-wall.normal)
            f_n = torch.where(active, torch.clamp(f_n, min=0.0), torch.zeros_like(f_n))
            normal_vec = torch.zeros(self.n, 3, dtype=torch.float64)
            normal_vec[:, wall.axis] = -wall.normal
            total = f_n[:, None] * normal_vec
            if wall.friction > 0.0:
                v_t = self.velocities - self.velocities[:, wall.axis:wall.axis + 1] * torch.eye(
                    3, dtype=torch.float64)[wall.axis]
                shear = self.wall_shear[w_idx] + v_t * self.dt
                shear[:, wall.axis] = 0.0
                k_t = wall.stiffness * 0.1
                f_t = k_t * shear
                f_t_norm = f_t.norm(dim=1)
                cap = wall.friction * f_n
                ratio = torch.where(f_t_norm > cap, cap / f_t_norm.clamp_min(1e-30),
                                    torch.ones_like(f_t_norm))
                slip = torch.where(cap < f_t_norm, (f_t_norm - cap) / k_t,
                                   torch.zeros_like(f_t_norm))
                self.wall_friction_dissipation += float((cap * slip)[active].sum())
                f_t = f_t * ratio[:, None]
                self.wall_shear[w_idx] = torch.where(active[:, None], shear * ratio[:, None],
                                                     torch.zeros_like(shear))
                total = total + f_t
            forces += torch.where(active[:, None], total, torch.zeros_like(total))
            # reaction on the wall: opposite sign of the force exerted on particles
            self.wall_force[w_idx] = -torch.where(active[:, None], total,
                                                  torch.zeros_like(total)).sum(dim=0)
            if wall.velocity != 0.0:
                # work done by the moving wall on the medium: F_on_particle . v_wall
                self.external_work += float(
                    (torch.where(active[:, None], total, torch.zeros_like(total))[:, wall.axis]
                     * wall.velocity).sum()) * self.dt
            del pen

    @torch.no_grad()
    def forces(self) -> torch.Tensor:
        forces = torch.zeros_like(self.positions)
        self._bond_forces(forces)
        self._contact_forces(forces)
        if self.config.viscous_damping > 0.0:
            damping = self.config.viscous_damping * self.mass[:, None] * self.velocities
            forces -= damping
            self.viscous_dissipation += float((damping * self.velocities).sum()) * self.dt
        self._wall_forces(forces)
        self.kinetic = float(0.5 * (self.mass[:, None] * self.velocities.pow(2)).sum())
        return forces

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self) -> dict[str, float]:
        forces = self.forces()
        if bool(self.kinematic.any()):
            # the grips do work on the specimen: W = sum F_grip . v_grip dt
            # work done by the actuator ON the medium = -(force on the particles
            # from the interior) . v_grip: the actuator supplies the reaction
            work = -float((forces[self.kinematic]
                           * self.kinematic_velocity[self.kinematic]).sum()) * self.dt
            self.grip_work += work
            self.external_work += work
        self.velocities += forces / self.mass[:, None] * self.dt
        if bool(self.kinematic.any()):
            self.velocities[self.kinematic] = self.kinematic_velocity[self.kinematic]
        self.positions += self.velocities * self.dt
        for wall in self.walls:
            wall.advance(self.dt)
        self.time += self.dt
        diagnostics = self.diagnostics()
        if self.config.store_history:
            self.history.append(diagnostics)
        return diagnostics

    def diagnostics(self) -> dict[str, float]:
        residual = self.external_work - (
            self.contact_kinetic() + self.bond_elastic + self.contact_elastic
            + self.fracture_dissipation + self.friction_dissipation
            + self.wall_friction_dissipation + self.viscous_dissipation)
        scale = max(abs(self.external_work), 1e-12)
        return {
            "time": self.time,
            "external_work": self.external_work,
            "kinetic": self.contact_kinetic(),
            "bond_elastic": self.bond_elastic,
            "contact_elastic": self.contact_elastic,
            "fracture_j": self.fracture_dissipation,
            "friction_j": self.friction_dissipation,
            "wall_friction_j": self.wall_friction_dissipation,
            "viscous_j": self.viscous_dissipation,
            "energy_residual": residual,
            "energy_residual_relative": residual / scale,
            "killed_bonds": float(self.killed_bonds),
            "max_velocity": float(self.velocities.norm(dim=1).max()),
        }

    def contact_kinetic(self) -> float:
        return float(0.5 * (self.mass[:, None] * self.velocities.pow(2)).sum())

    # ------------------------------------------------------------------
    # post-processing helpers
    # ------------------------------------------------------------------
    def bond_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(compression-positive normal force, |tangential force|) of live bonds."""
        i, j = self.bond_i[self.bond_alive], self.bond_j[self.bond_alive]
        if int(i.numel()) == 0:
            return torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
        delta = self.positions[j] - self.positions[i]
        length = delta.norm(dim=1)
        f_n = self.bond_kn[self.bond_alive] * (self.bond_length0[self.bond_alive] - length)
        f_t = (self.bond_ks[self.bond_alive, None]
               * self.bond_shear[self.bond_alive]).norm(dim=1)
        return f_n, f_t

    def contact_force_snapshot(self) -> dict[str, torch.Tensor]:
        """Forces of every live inter-particle interaction (bonds + Hertz contacts)."""
        f_n, f_t = self.bond_forces()
        i, j = self.bond_i[self.bond_alive], self.bond_j[self.bond_alive]
        magnitudes = [f_n.abs()]
        idx_i = [i]
        idx_j = [j]
        if hasattr(self, "_last_contact_forces"):
            contact_mag, ci, cj = self._last_contact_forces
            magnitudes.append(contact_mag)
            idx_i.append(ci)
            idx_j.append(cj)
        return {"force": torch.cat(magnitudes), "i": torch.cat(idx_i),
                "j": torch.cat(idx_j),
                "positions_mid": 0.5 * (self.positions[torch.cat(idx_i)]
                                        + self.positions[torch.cat(idx_j)])}

    def snapshot_state(self) -> dict[str, Any]:
        return {"positions": self.positions.clone(),
                "velocities": self.velocities.clone(),
                "bond_alive": self.bond_alive.clone()}
    # ------------------------------------------------------------------
    # control-section measurement (stress without a platen penalty spring)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def section_force(self, z_cut: float) -> float:
        """Compression-positive axial force transmitted across the plane z = z_cut.

        Every live bond whose two ends straddle the plane contributes the
        z-component of the force it transmits; the sum is the internal section
        force of the specimen, i.e. a *measured* stress boundary condition --
        nothing is prescribed.
        """
        i, j = self.bond_i[self.bond_alive], self.bond_j[self.bond_alive]
        if int(i.numel()) == 0:
            return 0.0
        xi, xj = self.positions[i], self.positions[j]
        lower_first = xi[:, 2] <= z_cut
        straddle = (torch.minimum(xi[:, 2], xj[:, 2]) <= z_cut) & (
            torch.maximum(xi[:, 2], xj[:, 2]) > z_cut)
        if not bool(straddle.any()):
            return 0.0
        delta = xj - xi
        length = delta.norm(dim=1).clamp_min(1e-12)
        normal = delta / length[:, None]
        f_n = self.bond_kn[self.bond_alive] * (self.bond_length0[self.bond_alive] - length)
        f_t = self.bond_ks[self.bond_alive, None] * self.bond_shear[self.bond_alive]
        total = f_n[:, None] * normal + f_t
        # force on the *lower* end is -total; its z-component, compression positive
        # when the lower part is pushed down (negative z), so negate
        # compression positive: the force pushing the *lower* part down is
        # -F_on_lower = +total_z (total is the force applied to j, the upper end)
        transmitted = torch.where(lower_first, total[:, 2], -total[:, 2])
        return float(torch.where(straddle, transmitted, torch.zeros_like(transmitted)).sum())

    @torch.no_grad()
    def set_kinematic(self, mask: torch.Tensor, velocity: torch.Tensor) -> None:
        """Fix a set of particles to a prescribed velocity (rigid grip)."""
        if mask.shape != self.kinematic.shape:
            raise ValueError("kinematic mask shape mismatch")
        self.kinematic = mask.clone()
        self.kinematic_velocity = velocity.expand(self.n, 3).clone() if velocity.dim() == 1 \
            else velocity.clone()
