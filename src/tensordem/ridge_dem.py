"""Particle (DEM) side of the ice-ridge - pipeline - seabed grounding problem.

Physical picture
----------------
An ice ridge is idealised as a two-dimensional front-normal section (plane
strain, unit out-of-plane thickness): a triangular keel of depth
``keel_depth`` and width ``keel_width`` at the mouldline, built from disks with
a prescribed porosity and a prescribed freeze-bond fraction.  The keel is
driven down (or sideways) by a kinematic platen representing the consolidated
layer and sail pushed by the surrounding ice sheet.  Ice is buoyant
(``rho_i < rho_w``), so a ridge cannot fall onto a pipeline by itself: the
grounding force must be supplied by the driving ice, which is what the platen
measures.

Sub-models
----------
* elastic-brittle bonds (normal spring + incrementally accumulated tangential
  spring, tensile and shear strength in Pa) for freeze-bonded rubble;
* unbonded contact: penalty normal force with damping and a regularised
  Coulomb tangential force ``mu F_n tanh(v_t/v_eps)``; disks carry no rotation,
  so the emerging macro friction angle is *measured* by
  :func:`calibrate_rubble_strength` instead of being assumed;
* a rigid cylinder (the pipeline) with ice-pipe Coulomb friction;
* rigid planes (drive platen, calibration walls), optionally servo-controlled;
* a Mohr-Coulomb subgrade reaction on particles that penetrate below the
  mudline, i.e. the keel gouging through the seabed on its way to the pipe
  (:class:`tensordem.ridge_geotech.MohrCoulombSoil`).

Scope limits (see docs/ice_ridge_pipeline_grounding.md): rotationless
central-force disks under-represent rolling, which is why the macro friction
angle is measured rather than assumed; there is no pore fluid inside the DEM
(water enters only as buoyancy and through the soil effective-stress model);
particles are monodisperse; free lateral boundaries are water without back
pressure.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from .ridge_geotech import GRAVITY, MohrCoulombSoil

__all__ = [
    "RidgeKeel",
    "RidgeMaterial",
    "Plane",
    "RidgeAssembly",
    "generate_keel_assembly",
    "calibrate_rubble_strength",
]


def _finite(name: str, value, *, positive: bool = False, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class RidgeKeel:
    """Geometry of a ridge keel cross-section (plane strain, unit thickness)."""

    keel_depth: float
    keel_width: float
    porosity: float = 0.30
    sail_height: float | None = None

    def __post_init__(self) -> None:
        _finite("keel_depth", self.keel_depth, positive=True)
        _finite("keel_width", self.keel_width, positive=True)
        _finite("porosity", self.porosity, nonnegative=True)
        if self.porosity >= 0.6:
            raise ValueError("porosity must be < 0.6 for a bonded rubble lattice")
        if self.sail_height is None:
            object.__setattr__(self, "sail_height", 0.2 * self.keel_depth)
        _finite("sail_height", self.sail_height, nonnegative=True)

    @property
    def angle_deg(self) -> float:
        """Keel flank angle measured from the vertical, degrees."""
        return math.degrees(math.atan(0.5 * self.keel_width / self.keel_depth))

    def half_width(self, height_above_tip: float) -> float:
        """Keel half width at ``height_above_tip`` above the keel tip, metres."""
        _finite("height_above_tip", height_above_tip)
        clamped = min(max(height_above_tip, 0.0), self.keel_depth)
        return 0.5 * self.keel_width * clamped / self.keel_depth

    @property
    def cross_section_area(self) -> float:
        """Envelope keel cross-section area, m^2."""
        return 0.5 * self.keel_width * self.keel_depth


@dataclass(frozen=True)
class RidgeMaterial:
    """Micro-parameters of the ridge rubble / freeze-bonded ice lattice."""

    particle_radius: float = 0.75
    ice_density: float = 917.0
    water_density: float = 1025.0
    gravity: float = GRAVITY
    contact_stiffness: float = 1.0e7
    contact_damping: float = 5.0e4
    friction_coefficient: float = 0.55
    pipe_friction_coefficient: float = 0.30
    soil_friction_coefficient: float = 0.50
    bond_stiffness: float = 5.0e6
    bond_shear_stiffness: float = 1.0e7
    bond_tensile_strength: float = 50.0e3
    bond_shear_strength: float = 100.0e3
    freeze_bond_fraction: float = 0.65
    drag: float = 2.0e3
    velocity_regularisation: float = 0.01
    dt: float | None = None
    seed: int = 20261031

    def __post_init__(self) -> None:
        for name in ("particle_radius", "ice_density", "water_density", "gravity",
                     "contact_stiffness", "bond_stiffness", "bond_shear_stiffness",
                     "bond_tensile_strength", "bond_shear_strength",
                     "velocity_regularisation"):
            _finite(name, getattr(self, name), positive=True)
        for name in ("contact_damping", "friction_coefficient",
                     "pipe_friction_coefficient", "soil_friction_coefficient", "drag"):
            _finite(name, getattr(self, name), nonnegative=True)
        _finite("freeze_bond_fraction", self.freeze_bond_fraction, nonnegative=True)
        if self.freeze_bond_fraction > 1.0:
            raise ValueError("freeze_bond_fraction must be <= 1")
        if self.ice_density > self.water_density:
            raise ValueError("ice must be buoyant: ice_density <= water_density")
        if self.dt is not None:
            _finite("dt", self.dt, positive=True)
        if self.bond_tensile_strength / self.bond_stiffness > 0.05:
            raise ValueError(
                "freeze bonds too ductile: bond_tensile_strength/bond_stiffness "
                "> 0.05 stores enough elastic energy to break explicit stability")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

    @property
    def particle_mass(self) -> float:
        """Mass of one disk of unit thickness, kg."""
        return self.ice_density * math.pi * self.particle_radius ** 2

    @property
    def buoyant_unit_force(self) -> float:
        """Net vertical body force per disk (+down in this module), N."""
        volume = math.pi * self.particle_radius ** 2
        return (self.ice_density - self.water_density) * volume * self.gravity

    @property
    def bond_failure_strain(self) -> float:
        """Implied tensile failure strain of a freeze bond, ``sigma_b t / k``.

        Very ductile bonds (large values) store ``0.5 k delta^2`` of energy each
        and release it on break, which destroys explicit stability; values above
        0.05 are therefore rejected by ``__post_init__``.
        """
        return self.bond_tensile_strength * self.particle_radius * 2.0 / (
            self.bond_stiffness * 2.0 * self.particle_radius) if self.bond_stiffness else math.inf

    @property
    def bond_area(self) -> float:
        """Contact/bond area of one disk for unit thickness, m^2."""
        return 2.0 * self.particle_radius

    @property
    def recommended_dt(self) -> float:
        """Conservative explicit time step from stiffness and damping limits."""
        if self.dt is not None:
            return self.dt
        neighbours = 13.0
        stiffness = neighbours * (self.contact_stiffness + 2.0 * self.bond_stiffness)
        damping = neighbours * self.contact_damping + self.drag
        elastic = 0.2 * math.sqrt(self.particle_mass / stiffness)
        viscous = 0.2 * self.particle_mass / damping if damping else math.inf
        return min(elastic, viscous)


@dataclass(frozen=True)
class Plane:
    """A rigid half-space wall with the solid on the ``-normal`` side.

    ``normal`` is a unit vector pointing from the wall *into* the material;
    ``point`` lies on the wall surface; ``velocity`` is the wall velocity along
    ``normal`` (positive pushes into the material).  ``servo_stress`` (Pa)
    turns the wall into a servo-controlled pressure boundary of width
    ``servo_width`` metres.
    """

    normal: tuple[float, float]
    point: tuple[float, float]
    velocity: float = 0.0
    stiffness: float = 2.0e6
    damping: float = 3.0e4
    friction: float = 0.3
    servo_stress: float | None = None
    servo_width: float = 1.0
    servo_gain: float = 1.0e-9
    servo_velocity_limit: float = 0.5

    def __post_init__(self) -> None:
        n = torch.tensor(self.normal, dtype=torch.float64)
        if n.shape != (2,) or not bool(torch.isfinite(n).all()):
            raise ValueError("normal must be a finite 2-vector")
        if abs(float(torch.linalg.vector_norm(n)) - 1.0) > 1e-9:
            raise ValueError("normal must be a unit vector")
        if len(self.point) != 2:
            raise ValueError("point must be a 2-vector")
        _finite("velocity", self.velocity)
        _finite("stiffness", self.stiffness, positive=True)
        _finite("damping", self.damping, nonnegative=True)
        _finite("friction", self.friction, nonnegative=True)
        if self.servo_stress is not None:
            _finite("servo_stress", self.servo_stress, positive=True)
            _finite("servo_width", self.servo_width, positive=True)
            _finite("servo_gain", self.servo_gain, positive=True)
            _finite("servo_velocity_limit", self.servo_velocity_limit, positive=True)

    def displaced(self, distance: float) -> "Plane":
        """Copy of the plane translated by ``distance`` along its normal."""
        point = (self.point[0] + distance * self.normal[0],
                 self.point[1] + distance * self.normal[1])
        return replace(self, point=point)


class RidgeAssembly:
    """Disks + elastic-brittle bonds + rigid obstacles + optional subgrade.

    ``body_force`` is the net vertical body force per particle in the y-up
    frame (for submerged ice this is positive, i.e. buoyant).  ``driven``
    marks kinematic particles whose velocity is imposed by the platen.  Plane
    and cylinder obstacles are supplied as :class:`Plane` and ``dict`` records.
    """

    def __init__(self, positions, radii, masses, bonds, material: RidgeMaterial,
                 *, planes=None, cylinders=None, soil: MohrCoulombSoil | None = None,
                 driven=None, body_force=None, thickness: float = 1.0,
                 mudline: float | None = None):
        self.material = material
        self.pos = torch.as_tensor(positions, dtype=torch.float64).reshape(-1, 2)
        self.radii = torch.as_tensor(radii, dtype=torch.float64).reshape(-1)
        self.mass = torch.as_tensor(masses, dtype=torch.float64).reshape(-1)
        count = int(self.pos.shape[0])
        if count < 1:
            raise ValueError("at least one particle is required")
        if self.radii.numel() != count or self.mass.numel() != count:
            raise ValueError("positions, radii and masses must have equal length")
        if not bool(torch.isfinite(self.pos).all()):
            raise ValueError("positions must be finite")
        if not bool((self.radii > 0).all()) or not bool((self.mass > 0).all()):
            raise ValueError("radii and masses must be positive")
        bond_pairs = torch.as_tensor(bonds, dtype=torch.long).reshape(-1, 2)
        if bond_pairs.numel() and (int(bond_pairs.max()) >= count or int(bond_pairs.min()) < 0):
            raise ValueError("bond indices out of range")
        self.bonds = bond_pairs
        self.bond_alive = torch.ones(bond_pairs.shape[0], dtype=torch.bool)
        delta = self.pos[self.bonds[:, 1]] - self.pos[self.bonds[:, 0]]
        self.bond_length0 = torch.linalg.vector_norm(delta, dim=1)
        unit = delta / self.bond_length0.clamp_min(1e-12)[:, None]
        self.bond_tangent0 = torch.stack((-unit[:, 1], unit[:, 0]), dim=1)
        self.bond_shear_disp = torch.zeros(bond_pairs.shape[0], dtype=torch.float64)
        self.vel = torch.zeros_like(self.pos)
        self.body_force = (torch.zeros(count, dtype=torch.float64) if body_force is None
                           else torch.as_tensor(body_force, dtype=torch.float64).reshape(count))
        self.driven = (torch.zeros(count, dtype=torch.bool) if driven is None
                       else torch.as_tensor(driven, dtype=torch.bool).reshape(-1))
        if self.driven.numel() != count:
            raise ValueError("driven must flag every particle")
        self.planes: list[Plane] = list(planes or [])
        self.cylinders: list[dict] = list(cylinders or [])
        self.soil = soil
        self.mudline = mudline
        self.thickness = float(thickness)
        self.broken_bonds = 0
        self.time = 0.0
        self.driven_velocity = torch.zeros(count, 2, dtype=torch.float64)
        # cached index tensors: rebuilding them every step dominates the cost of
        # a small 2-D assembly
        self._pair_i, self._pair_j = torch.triu_indices(count, count, offset=1)
        self._pair_radius_sum = self.radii[self._pair_i] + self.radii[self._pair_j]
        self._skin = 0.15 * float(self.radii.min())
        self._all = torch.arange(count)
        self._neighbour_step = 0
        self._rebuild_every = 25
        self._neigh_i = self._pair_i
        self._neigh_j = self._pair_j
        self._update_neighbours()
        self._bonded = torch.zeros(count * count, dtype=torch.bool)
        if self.bonds.numel():
            self._bonded[self.bonds[:, 0] * count + self.bonds[:, 1]] = True
            self._bonded[self.bonds[:, 1] * count + self.bonds[:, 0]] = True
        self._free = (~self.driven).to(torch.float64)

    # ------------------------------------------------------------------ helpers
    @property
    def count(self) -> int:
        return int(self.pos.shape[0])

    def set_platen_velocity(self, velocity) -> None:
        """Impose the velocity of the kinematic (driven) particles."""
        v = torch.as_tensor(velocity, dtype=torch.float64).reshape(2)
        self.driven_velocity = torch.zeros_like(self.pos)
        if bool(self.driven.any()):
            self.driven_velocity[self.driven] = v

    def kinetic_energy(self) -> float:
        return float(0.5 * (self.mass[:, None] * self.vel ** 2).sum())

    def _update_neighbours(self) -> None:
        """Rebuild the Verlet neighbour list (pairs within contact + skin)."""
        delta = self.pos[self._pair_j] - self.pos[self._pair_i]
        distance = (delta * delta).sum(dim=1).sqrt()
        keep = distance < self._pair_radius_sum + self._skin
        self._neigh_i = self._pair_i[keep]
        self._neigh_j = self._pair_j[keep]

    def _mark_bond_broken(self, mask: torch.Tensor) -> None:
        """Clear the cached bond lookup for bonds that just failed."""
        if not self.bonds.numel():
            return
        count = self.count
        index = self.bonds[mask]
        self._bonded[index[:, 0] * count + index[:, 1]] = False
        self._bonded[index[:, 1] * count + index[:, 0]] = False

    # ------------------------------------------------------------------- forces
    def forces(self, dt: float | None = None, *, update_memory: bool = True,
               break_bonds: bool = True) -> dict:
        """Assemble all particle forces and interaction diagnostics."""
        m = self.material
        count = self.count
        force = torch.zeros_like(self.pos)
        force[:, 1] += self.body_force
        # environmental/global drag damps free particles only; kinematic
        # platen particles are excluded so their reaction stays interpretable
        force -= m.drag * self.vel * self._free[:, None]

        # --- freeze bonds -------------------------------------------------
        if self.bonds.numel():
            alive = self.bond_alive
            i = self.bonds[:, 0]
            j = self.bonds[:, 1]
            delta = self.pos[j] - self.pos[i]
            length = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
            normal = delta / length[:, None]
            extension = length - self.bond_length0
            normal_force = m.bond_stiffness * extension
            tangent = self.bond_tangent0
            relative_velocity = self.vel[j] - self.vel[i]
            shear_rate = (relative_velocity * tangent).sum(dim=1)
            shear_force = -m.bond_shear_stiffness * self.bond_shear_disp
            area = m.bond_area * self.thickness
            broken = alive & ((normal_force > m.bond_tensile_strength * area)
                              | (shear_force.abs() > m.bond_shear_strength * area))
            if break_bonds and bool(broken.any()):
                self.bond_alive = alive & ~broken
                self.broken_bonds += int(broken.sum())
                self._mark_bond_broken(broken)
                alive = self.bond_alive
            pair_force = (normal_force[:, None] * normal
                          + shear_force[:, None] * tangent) * alive[:, None].to(torch.float64)
            force.index_add_(0, i, pair_force)
            force.index_add_(0, j, -pair_force)
            if update_memory and dt is not None:
                self.bond_shear_disp = self.bond_shear_disp + shear_rate * dt * alive.to(
                    torch.float64)

        # --- unbonded disk contacts ---------------------------------------
        index_i = self._neigh_i
        index_j = self._neigh_j
        delta = self.pos[index_j] - self.pos[index_i]
        distance = (delta * delta).sum(dim=1).sqrt()
        overlap = self.radii[index_i] + self.radii[index_j] - distance
        active = overlap > 0.0
        if self.bonds.numel():
            active = active & ~self._bonded[index_i * count + index_j]
        contact_count = int(active.sum())
        if contact_count:
            ai = index_i[active]
            aj = index_j[active]
            # outward normal from j towards i, so that +magnitude * normal is the
            # repulsive force on i (and its opposite acts on j)
            normal = -delta[active] / distance[active][:, None]
            relative_velocity = self.vel[ai] - self.vel[aj]
            v_n = (relative_velocity * normal).sum(dim=1)
            magnitude = (m.contact_stiffness * overlap[active]
                         - m.contact_damping * v_n).clamp_min(0.0)
            tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
            v_t = (relative_velocity * tangent).sum(dim=1)
            tangential = (-m.friction_coefficient * magnitude
                          * torch.tanh(v_t / m.velocity_regularisation))
            pair_force = (magnitude[:, None] * normal
                          + tangential[:, None] * tangent)
            force.index_add_(0, ai, pair_force)
            force.index_add_(0, aj, -pair_force)

        # --- rigid cylinders (pipeline) ------------------------------------
        pipe_force = torch.zeros(2, dtype=torch.float64)
        pipe_contacts = 0
        for body in self.cylinders:
            centre = body["centre"]
            radius = float(body["radius"])
            velocity = body.get("velocity", torch.zeros(2, dtype=torch.float64))
            separation = self.pos - centre
            dist = torch.linalg.vector_norm(separation, dim=1).clamp_min(1e-12)
            gap = radius + self.radii - dist
            hit = gap > 0.0
            pipe_contacts += int(hit.sum())
            if bool(hit.any()):
                normal = separation[hit] / dist[hit][:, None]
                relative_velocity = self.vel[hit] - velocity
                v_n = (relative_velocity * normal).sum(dim=1)
                magnitude = (m.contact_stiffness * gap[hit]
                             - m.contact_damping * v_n).clamp_min(0.0)
                tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
                v_t = (relative_velocity * tangent).sum(dim=1)
                tangential = (-m.pipe_friction_coefficient * magnitude
                              * torch.tanh(v_t / m.velocity_regularisation))
                on_particles = magnitude[:, None] * normal + tangential[:, None] * tangent
                force.index_add_(0, self._all[hit], on_particles)
                pipe_force -= on_particles.sum(dim=0)

        # --- rigid planes --------------------------------------------------
        plane_forces = []
        plane_contacts = []
        for plane in self.planes:
            normal = torch.tensor(plane.normal, dtype=torch.float64)
            point = torch.tensor(plane.point, dtype=torch.float64)
            signed = (self.pos - point) @ normal
            gap = self.radii - signed
            hit = gap > 0.0
            plane_contacts.append(int(hit.sum()))
            on_plane = torch.zeros(2, dtype=torch.float64)
            if bool(hit.any()):
                wall_velocity = plane.velocity * normal
                relative_velocity = self.vel[hit] - wall_velocity
                v_n = (relative_velocity * normal).sum(dim=1)
                magnitude = (plane.stiffness * gap[hit]
                             - plane.damping * v_n).clamp_min(0.0)
                tangent = torch.tensor((-normal[1], normal[0]), dtype=torch.float64)
                v_t = (relative_velocity * tangent).sum(dim=1)
                tangential = (-plane.friction * magnitude
                              * torch.tanh(v_t / m.velocity_regularisation))
                on_particles = magnitude[:, None] * normal + tangential[:, None] * tangent
                force.index_add_(0, self._all[hit], on_particles)
                on_plane = -on_particles.sum(dim=0)
            plane_forces.append(on_plane)

        # --- Mohr-Coulomb subgrade on particles below the mudline ----------
        soil_force = torch.zeros(2, dtype=torch.float64)
        soil_contacts = 0
        if self.soil is not None and self.mudline is not None:
            depth = self.mudline - self.pos[:, 1]
            below = depth > 0.0
            soil_contacts = int(below.sum())
            if soil_contacts:
                q_ult = torch.tensor([
                    self.soil.vertical_pressure_capacity(2.0 * float(r),
                                                         float(max(d, 1e-6)))
                    for r, d in zip(self.radii[below], depth[below])],
                    dtype=torch.float64)
                k_soil = torch.tensor(
                    [self.soil.subgrade_modulus(2.0 * float(r)) * 2.0 * float(r)
                     for r in self.radii[below]], dtype=torch.float64)
                area = 2.0 * self.radii[below] * self.thickness
                resistance = torch.minimum(k_soil * depth[below], q_ult) * area
                v_y = self.vel[below, 1]
                magnitude = (resistance + 2.0 * area * 1.0e4 * (-v_y).clamp_min(0.0))
                tangential = (-m.soil_friction_coefficient * magnitude
                              * torch.tanh(self.vel[below, 0] / m.velocity_regularisation))
                on_particles = torch.stack((tangential, magnitude), dim=1)
                force.index_add_(0, self._all[below], on_particles)
                soil_force = on_particles.sum(dim=0)

        # --- kinematic platen reaction -------------------------------------
        platen_force = torch.zeros(2, dtype=torch.float64)
        if bool(self.driven.any()):
            platen_force = -force[self.driven].sum(dim=0)

        return {
            "force": force,
            "platen_force": platen_force,
            "pipe_force": pipe_force,
            "plane_forces": plane_forces,
            "soil_force": soil_force,
            "contacts": contact_count,
            "pipe_contacts": pipe_contacts,
            "plane_contacts": plane_contacts,
            "soil_contacts": soil_contacts,
        }

    # ------------------------------------------------------------ integration
    def step(self, dt: float, *, update_memory: bool = True) -> dict:
        """One explicit (semi-implicit Euler) step; returns the force summary."""
        self._neighbour_step += 1
        if self._neighbour_step >= self._rebuild_every:
            self._neighbour_step = 0
            self._update_neighbours()
        out = self.forces(dt, update_memory=update_memory)
        force = out["force"]
        acceleration = force / self.mass[:, None]
        self.vel = self.vel + acceleration * dt
        if bool(self.driven.any()):
            self.vel[self.driven] = self.driven_velocity[self.driven]
        self.pos = self.pos + self.vel * dt
        self.time += dt
        return out

    def positions_of(self, mask) -> torch.Tensor:
        return self.pos[torch.as_tensor(mask, dtype=torch.bool)]


TRIANGULAR_PACKING_POROSITY = 1.0 - math.pi / (2.0 * math.sqrt(3.0))
"""Porosity of a triangular lattice of touching equal disks (0.0931)."""


def _bond_pairs(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    """All ``i < j`` pairs closer than ``cutoff``."""
    count = int(positions.shape[0])
    if count < 2:
        return torch.zeros((0, 2), dtype=torch.long)
    index_i, index_j = torch.triu_indices(count, count, offset=1)
    distance = torch.linalg.vector_norm(positions[index_j] - positions[index_i], dim=1)
    keep = distance < cutoff
    return torch.stack((index_i[keep], index_j[keep]), dim=1)


def _thin(positions: torch.Tensor, bonds: torch.Tensor, alive: torch.Tensor,
          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop removed particles and renumber the surviving bonds."""
    keep_index = torch.nonzero(alive, as_tuple=False).flatten()
    remap = torch.full((alive.numel(),), -1, dtype=torch.long)
    remap[keep_index] = torch.arange(keep_index.numel())
    survivors = alive[bonds[:, 0]] & alive[bonds[:, 1]]
    return positions[keep_index], remap[bonds[survivors]]


def _remove_initial_overlaps(positions: torch.Tensor, radius: float,
                             generator: torch.Generator) -> torch.Tensor:
    """Drop lattice points that would start already overlapping a kept disk.

    Ragged lattice edges (row offsets clipped at the keel flank) can otherwise
    create initial overlaps of a few centimetres, i.e. contact forces of order
    1e5 N, and the assembly explodes in the first few steps.  Points are
    visited in a random order and accepted only if they clear every accepted
    disk by ``2 r (1 + 1e-3)``.
    """
    count = int(positions.shape[0])
    order = torch.randperm(count, generator=generator)
    minimum = 2.0 * radius * (1.0 - 1e-6)
    kept: list[int] = []
    for index in order.tolist():
        point = positions[index]
        if kept:
            existing = positions[torch.tensor(kept, dtype=torch.long)]
            if float(torch.linalg.vector_norm(existing - point, dim=1).min()) < minimum:
                continue
        kept.append(index)
    return positions[torch.tensor(sorted(kept), dtype=torch.long)]


def _seed_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def generate_keel_assembly(keel: RidgeKeel, material: RidgeMaterial, *,
                           tip_height: float = 1.0, submerged: bool = True,
                           thickness: float = 1.0) -> dict:
    """Build the triangular keel as a bonded disk assembly.

    ``tip_height`` is the height of the keel tip above the mudline (y = 0) at
    the start of the run.  Particles are placed on a triangular lattice with
    spacing ``2 r`` and then randomly thinned to the requested porosity; the
    top-most row is kinematic and acts as the drive platen.

    Returns a dict with ``assembly``, ``info`` (particle count, porosity after
    thinning, bond count, driven count, keel tip position) and ``driven``.
    """
    if not isinstance(keel, RidgeKeel) or not isinstance(material, RidgeMaterial):
        raise TypeError("keel and material must be RidgeKeel / RidgeMaterial")
    _finite("tip_height", tip_height)
    radius = material.particle_radius
    spacing = 2.0 * radius
    row_spacing = spacing * math.sqrt(3.0) / 2.0
    rows = max(int(keel.keel_depth // row_spacing) + 1, 2)
    generator = _seed_generator(material.seed)
    points = []
    for row in range(rows):
        height = row * row_spacing
        if height > keel.keel_depth + 1e-9:
            break
        half = keel.half_width(height)
        offset = 0.5 * spacing if row % 2 else 0.0
        first = math.ceil((-half - offset) / spacing)
        last = math.floor((half - offset) / spacing)
        for column in range(first, last + 1):
            points.append((column * spacing + offset, tip_height + height))
    positions = torch.tensor(points, dtype=torch.float64)
    if positions.shape[0] < 8:
        raise ValueError("keel lattice too small: increase keel size or reduce radius")
    positions = _remove_initial_overlaps(positions, radius, generator)
    bonds = _bond_pairs(positions, 1.15 * spacing)
    remove_fraction = 1.0 - (1.0 - keel.porosity) / (1.0 - TRIANGULAR_PACKING_POROSITY)
    if remove_fraction < 0.0 or remove_fraction >= 1.0:
        raise ValueError("porosity outside the reachable range for this lattice")
    alive = torch.rand(positions.shape[0], generator=generator) >= remove_fraction
    positions, bonds = _thin(positions, bonds, alive)
    keep_bond = torch.rand(bonds.shape[0], generator=generator) < material.freeze_bond_fraction
    bonds = bonds[keep_bond]
    count = int(positions.shape[0])
    masses = torch.full((count,), material.particle_mass, dtype=torch.float64)
    body = torch.zeros(count, dtype=torch.float64)
    if submerged:
        body += (material.water_density - material.ice_density) * math.pi * radius ** 2 \
            * material.gravity
    top = float(positions[:, 1].max())
    driven = positions[:, 1] > top - 1.5 * row_spacing
    assembly = RidgeAssembly(positions, torch.full((count,), radius, dtype=torch.float64),
                             masses, bonds, material, driven=driven, body_force=body,
                             thickness=thickness, mudline=0.0)
    info = {
        "particles": count,
        "bonds": int(bonds.shape[0]),
        "driven_particles": int(driven.sum()),
        "porosity": float(1.0 - count * math.pi * radius ** 2
                          / max(keel.cross_section_area, 1e-12)),
        "requested_porosity": keel.porosity,
        "keel_tip_height": tip_height,
        "keel_top_height": top,
        "row_spacing": row_spacing,
        "particle_radius": radius,
    }
    return {"assembly": assembly, "info": info, "driven": driven, "positions": positions}


def _specimen_assembly(material: RidgeMaterial, *, columns: int, rows: int,
                       confining_stress: float, porosity: float = 0.05,
                       thickness: float = 1.0) -> tuple[RidgeAssembly, dict]:
    """Rectangular bonded specimen with servo lateral walls (biaxial cell).

    Walls sit one particle radius outside the outermost particle centres so the
    initial state is exactly in contact with zero force.
    """
    radius = material.particle_radius
    spacing = 2.0 * radius
    row_spacing = spacing * math.sqrt(3.0) / 2.0
    width = (columns - 1) * spacing
    height = (rows - 1) * row_spacing
    generator = _seed_generator(material.seed + 7)
    points = []
    for row in range(rows):
        offset = 0.5 * spacing if row % 2 else 0.0
        for column in range(columns):
            x = -0.5 * width + column * spacing + offset
            if abs(x) > 0.5 * width + 1e-9:
                continue
            points.append((x, -0.5 * height + row * row_spacing))
    positions = torch.tensor(points, dtype=torch.float64)
    positions = _remove_initial_overlaps(positions, radius, generator)
    bonds = _bond_pairs(positions, 1.15 * spacing)
    remove_fraction = max(0.0, 1.0 - (1.0 - porosity) / (1.0 - TRIANGULAR_PACKING_POROSITY))
    if remove_fraction > 0.0:
        alive = torch.rand(positions.shape[0], generator=generator) >= remove_fraction
        positions, bonds = _thin(positions, bonds, alive)
    keep_bond = torch.rand(bonds.shape[0], generator=generator) < material.freeze_bond_fraction
    bonds = bonds[keep_bond]
    count = int(positions.shape[0])
    half_w = 0.5 * width + radius
    half_h = 0.5 * height + radius
    wall_stiffness = 4.0 * material.contact_stiffness
    planes = [
        Plane((1.0, 0.0), (-half_w, 0.0), stiffness=wall_stiffness,
              damping=material.contact_damping, friction=material.friction_coefficient,
              servo_stress=confining_stress, servo_width=height,
              servo_gain=2.0e-7, servo_velocity_limit=0.5),
        Plane((-1.0, 0.0), (half_w, 0.0), stiffness=wall_stiffness,
              damping=material.contact_damping, friction=material.friction_coefficient,
              servo_stress=confining_stress, servo_width=height,
              servo_gain=2.0e-7, servo_velocity_limit=0.5),
        Plane((0.0, 1.0), (0.0, -half_h), stiffness=wall_stiffness,
              damping=material.contact_damping, friction=material.friction_coefficient),
        Plane((0.0, -1.0), (0.0, half_h), stiffness=wall_stiffness,
              damping=material.contact_damping, friction=material.friction_coefficient),
    ]
    assembly = RidgeAssembly(positions, torch.full((count,), radius, dtype=torch.float64),
                             torch.full((count,), material.particle_mass, dtype=torch.float64),
                             bonds, material, planes=planes, thickness=thickness)
    return assembly, {"width": width, "height": height, "particles": count,
                      "bonds": int(bonds.shape[0]), "confining_stress": confining_stress,
                      "top_plane_height": half_h, "wall_stiffness": wall_stiffness}


def _advance_planes(assembly: RidgeAssembly, dt: float) -> None:
    """Move every plane along its normal by ``velocity * dt`` and run servos."""
    for index, plane in enumerate(assembly.planes):
        normal = torch.tensor(plane.normal, dtype=torch.float64)
        velocity = plane.velocity
        if plane.servo_stress is not None:
            force = assembly.last_plane_forces[index]
            measured = -float(force @ normal) / (plane.servo_width * assembly.thickness)
            error = measured - plane.servo_stress
            # below the target pressure the wall must push *into* the specimen
            velocity = max(min(-plane.servo_gain * error, plane.servo_velocity_limit),
                           -plane.servo_velocity_limit)
        if velocity == 0.0:
            continue
        point = (plane.point[0] + velocity * dt * normal[0],
                 plane.point[1] + velocity * dt * normal[1])
        assembly.planes[index] = replace(plane, point=point, velocity=velocity)


def calibrate_rubble_strength(material: RidgeMaterial, *,
                              confining_stresses=(50.0e3, 150.0e3),
                              columns: int = 8, rows: int = 8,
                              loading_rate: float = 0.15, strain_target: float = 0.10,
                              consolidation_steps: int = 1500,
                              max_steps: int = 6000) -> dict:
    """Measure the macro cohesion and friction angle of the rubble material.

    Two constant-confining-pressure biaxial-shear runs (servo lateral walls,
    displacement-driven top platen) give one Mohr circle per confinement; the
    common tangent of the two circles gives ``(c, phi)``.  This is the
    micro -> macro bridge used by the continuum limit analysis, so ``phi`` and
    ``c`` are *measured* for this micro-parameter set rather than assumed.
    """
    if len(confining_stresses) != 2:
        raise ValueError("exactly two confining stresses are required")
    for value in confining_stresses:
        _finite("confining_stress", value, positive=True)
    dt = material.recommended_dt
    specimens = []
    circles = []
    for confining in confining_stresses:
        assembly, geometry = _specimen_assembly(material, columns=columns, rows=rows,
                                                confining_stress=confining)
        for _ in range(consolidation_steps):
            out = assembly.step(dt)
            assembly.last_plane_forces = out["plane_forces"]
            _advance_planes(assembly, dt)
        top_start = assembly.planes[3].point[1]
        normal_left = torch.tensor(assembly.planes[0].normal, dtype=torch.float64)
        normal_right = torch.tensor(assembly.planes[1].normal, dtype=torch.float64)
        deviator = []
        lateral = []
        axial = []
        strain = 0.0
        for _step in range(max_steps):
            assembly.planes[3] = replace(assembly.planes[3], velocity=loading_rate)
            out = assembly.step(dt)
            assembly.last_plane_forces = out["plane_forces"]
            _advance_planes(assembly, dt)
            sigma_v = -float(out["plane_forces"][2][1]) / (geometry["width"] * assembly.thickness)
            sigma_h = 0.5 * (-float(out["plane_forces"][0] @ normal_left)
                             - float(out["plane_forces"][1] @ normal_right)) \
                / (geometry["height"] * assembly.thickness)
            strain = (top_start - assembly.planes[3].point[1]) / geometry["height"]
            axial.append(sigma_v)
            lateral.append(sigma_h)
            deviator.append(sigma_v - sigma_h)
            if strain > strain_target:
                break
        deviator_t = torch.tensor(deviator, dtype=torch.float64)
        window = min(60, max(1, deviator_t.numel() // 10))
        if window > 1:
            kernel = torch.ones(window, dtype=torch.float64) / window
            left = window // 2
            padded = torch.nn.functional.pad(deviator_t.reshape(1, 1, -1),
                                             (left, window - 1 - left), mode="replicate")
            smoothed = torch.nn.functional.conv1d(padded, kernel.reshape(1, 1, -1)).reshape(-1)
        else:
            smoothed = deviator_t
        peak = float(smoothed.max())
        index = int(torch.argmax(smoothed))
        lateral_peak = lateral[min(index, len(lateral) - 1)]
        circles.append((confining, peak))
        specimens.append({
            "confining_stress": float(confining),
            "peak_deviator_stress": peak,
            "axial_stress_at_peak": float(axial[min(index, len(axial) - 1)]),
            "lateral_stress_at_peak": float(lateral_peak),
            "mean_lateral_stress": float(sum(lateral) / len(lateral)),
            "confining_relative_error": abs(sum(lateral) / len(lateral) - confining) / confining,
            "steps": len(deviator),
            "final_strain": strain,
            "smoothing_window": window,
            "specimen": geometry,
            "broken_bonds": assembly.broken_bonds,
        })
    (c1, d1), (c2, d2) = circles
    centre1 = c1 + 0.5 * d1
    centre2 = c2 + 0.5 * d2
    radius1 = 0.5 * d1
    radius2 = 0.5 * d2
    if abs(centre1 - centre2) < 1e-9:
        raise ValueError("confining stresses must differ")
    sin_phi = (radius1 - radius2) / (centre1 - centre2)
    sin_phi = max(min(sin_phi, 0.95), -0.95)
    phi_rad = math.asin(sin_phi)
    cohesion = max(radius1 / math.cos(phi_rad) - centre1 * math.tan(phi_rad), 0.0)
    return {
        "cohesion": cohesion,
        "friction_angle_deg": math.degrees(phi_rad),
        "circles_confining": [c1, c2],
        "circles_deviator": [d1, d2],
        "specimens": specimens,
        "secant_angles_deg": [
            math.degrees(math.asin(max(min(d / (2.0 * c + d), 0.99), -0.99)))
            for c, d in ((c1, d1), (c2, d2))],
        "dt": dt,
        "note": "constant-confining-pressure biaxial shear; self weight neglected",
    }
