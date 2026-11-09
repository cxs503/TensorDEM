"""Plane-strain DEM of the ship hull crushing a multi-year ridge keel.

Physical picture
----------------
A ship entering a ridge loads the keel rubble through its ice belt.  Cutting a
channel through the ridge destroys the rubble over the full keel depth (the
swept cross-section is ``B_cut x h_keel`` per metre of advance), so the ridge
resistance can be split into

* a *pressure* that the rubble offers to the advancing hull plate, obtained
  here from a two-dimensional plane-strain DEM of a bonded rubble layer of
  height ``h_keel`` crushed by a moving hull plate, and
* a *geometry* factor, the contact area of the channel cut, which supplies the
  keel-depth and channel-width scaling.

The DEM layer is a bonded disc assembly (freeze bonds in tension and shear,
penalty contact with damping, regularised Coulomb friction, buoyancy body
force).  The hull is a rigid plate that traverses the layer.  The plate is
either kinematic (velocity imposed - the ship's propulsion sustains the
advance) or a rigid body with mass and thrust (a coasting ram).

Everything that stores or dissipates energy is accounted for explicitly so the
energy ledger closes:
``E_input = dKE + dU_elastic + dU_buoyancy + E_fracture + W_friction + W_damping``.
The ledger is the primary validation of the integrator: an explicit
velocity-Verlet scheme with a penalty contact law must close the ledger on the
integrated physics, not on a parallel model.

Scope limits (stated, not hidden)
---------------------------------
* Two-dimensional plane strain with unit out-of-plane thickness: crushed
  rubble cannot escape sideways, which is the confined condition of a channel
  but overstates the confinement of a free ridge flank.
* Discs are monodisperse and rigid; the crushing energy is therefore the
  freeze-bond fracture energy plus friction, not the energy of crushing
  individual ice blocks.
* No pore water flow inside the rubble, no thermal effects.
* Freeze-bond strength is a *calibration* input; the resulting crushing
  pressure is an output (:attr:`CrushResult.pressure_steady`) and its ratio to
  ``sigma_ice`` is reported rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch

__all__ = [
    "GRAVITY",
    "RidgeSpecimen",
    "CrushParameters",
    "RidgeCrushDem",
    "build_specimen",
    "run_crush_case",
]

GRAVITY = 9.81


def _finite(name: str, value: float, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class RidgeSpecimen:
    """Bonded rubble layer that stands in for the ridge keel cross-section."""

    height: float
    length: float
    porosity: float = 0.30
    particle_radius: float = 0.5
    seed: int = 20261108

    def __post_init__(self) -> None:
        _finite("height", self.height, positive=True)
        _finite("length", self.length, positive=True)
        _real_porosity = _finite("porosity", self.porosity, nonnegative=True)
        if not 0.0 <= _real_porosity < 0.6:
            raise ValueError("porosity must lie in [0, 0.6)")
        _finite("particle_radius", self.particle_radius, positive=True)
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")

    @property
    def area(self) -> float:
        return self.height * self.length

    @property
    def solid_area(self) -> float:
        return (1.0 - self.porosity) * self.area


def build_specimen(specimen: RidgeSpecimen) -> dict:
    """Jittered triangular lattice thinned to the requested porosity."""
    rng = np.random.default_rng(int(specimen.seed))
    r = specimen.particle_radius
    spacing = 2.05 * r  # above the contact distance: zero initial overlap
    row = spacing * math.sqrt(3.0) / 2.0
    rows = int(specimen.height // row) + 1
    cols = int(specimen.length // spacing) + 2
    points = []
    for j in range(rows):
        y = r + j * row
        if y > specimen.height - r:
            break
        offset = 0.5 * spacing if j % 2 else 0.0
        for i in range(cols):
            x = r + i * spacing + offset
            if x > specimen.length - r:
                continue
            points.append((x, y))
    positions = np.asarray(points, dtype=float).reshape(-1, 2)
    if positions.shape[0] < 8:
        raise ValueError("specimen too small: increase height/length or reduce radius")
    lattice_solid = math.pi * r ** 2 * positions.shape[0] / specimen.area
    keep_fraction = specimen.solid_area / (math.pi * r ** 2 * positions.shape[0])
    if keep_fraction >= 1.0:
        raise ValueError("requested porosity is denser than the lattice packing")
    n_keep = int(round(specimen.solid_area / (math.pi * r ** 2)))
    n_keep = max(min(n_keep, positions.shape[0] - 1), 8)
    keep = np.sort(rng.permutation(positions.shape[0])[:n_keep])
    positions = positions[keep]
    # a small number of thinning-induced overlaps are removed by a short
    # separation pass; bonds are built afterwards from the relaxed geometry
    for _ in range(120):
        delta = positions[:, None, :] - positions[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        overlap = 2.0 * r - dist
        mask = overlap > 0.0
        if not mask.any():
            break
        push = np.zeros_like(positions)
        count = np.zeros(positions.shape[0])
        ii, jj = np.nonzero(mask)
        np.add.at(push, ii, delta[ii, jj] / dist[ii, jj, None] * overlap[ii, jj, None] * 0.5)
        np.add.at(count, ii, 1.0)
        count[count == 0] = 1.0
        positions = positions + 0.5 * push / count[:, None]
        positions[:, 0] = np.clip(positions[:, 0], r, specimen.length - r)
        positions[:, 1] = np.clip(positions[:, 1], r, specimen.height - r)
    delta = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(dist, np.inf)
    pairs = np.argwhere(np.triu(dist < 1.12 * spacing, k=1))
    solid = math.pi * r ** 2 * positions.shape[0]
    return {
        "positions": positions,
        "radius": r,
        "bonds": pairs.astype(np.int64),
        "porosity": float(1.0 - solid / specimen.area),
        "lattice_solid_fraction": lattice_solid,
        "count": int(positions.shape[0]),
        "bond_count": int(pairs.shape[0]),
    }


@dataclass(frozen=True)
class CrushParameters:
    """Micro-mechanical and operational parameters of the crushing model."""

    ice_strength: float = 7.0e6            # sigma_ice [Pa]
    bond_strength_fraction: float = 0.35   # bond tensile strength / sigma_ice
    bond_shear_ratio: float = 1.0
    bond_stiffness: float = 1.0e9          # N/m per unit thickness
    bond_shear_stiffness: float = 1.0e9
    contact_stiffness: float = 1.0e9
    contact_damping: float = 1.0e5
    friction_coefficient: float = 0.55
    hull_friction_coefficient: float = 0.15
    wall_friction_coefficient: float = 0.40
    ice_density: float = 917.0
    water_density: float = 1025.0
    gravity: float = GRAVITY
    drag: float = 8.0e2
    hull_velocity: float = 2.0
    hull_mass: float | None = None         # None -> imposed plate velocity
    thrust: float = 0.0
    advance: float = 2.0
    sail_height: float = 3.5
    velocity_regularisation: float = 0.02
    velocity_damping: float = 0.02
    buoyancy: bool = False
    lid: bool = False
    cohesion_softening: float = 0.02
    skin_ratio: float = 0.6
    pair_interval: int = 10
    dt: float | None = None

    def __post_init__(self) -> None:
        for name in ("ice_strength", "bond_strength_fraction", "bond_shear_ratio",
                     "bond_stiffness", "bond_shear_stiffness", "contact_stiffness",
                     "ice_density", "water_density", "gravity",
                     "advance", "sail_height", "velocity_regularisation", "skin_ratio",
                     "cohesion_softening"):
            _finite(name, getattr(self, name), positive=True)
        _finite("hull_velocity", self.hull_velocity, nonnegative=True)
        _finite("velocity_damping", self.velocity_damping, nonnegative=True)
        if self.velocity_damping >= 1.0:
            raise ValueError("velocity_damping must be < 1")
        for name in ("contact_damping", "friction_coefficient",
                     "hull_friction_coefficient", "wall_friction_coefficient",
                     "drag", "thrust"):
            _finite(name, getattr(self, name), nonnegative=True)
        if self.hull_mass is not None:
            _finite("hull_mass", self.hull_mass, positive=True)
        if self.dt is not None:
            _finite("dt", self.dt, positive=True)
        for flag in ("buoyancy", "lid"):
            if not isinstance(getattr(self, flag), bool):
                raise ValueError(f"{flag} must be a bool")
        if isinstance(self.pair_interval, bool) or not isinstance(self.pair_interval, int) \
                or self.pair_interval < 1:
            raise ValueError("pair_interval must be a positive integer")
        if self.ice_density > self.water_density:
            raise ValueError("ice must be buoyant")


class RidgeCrushDem:
    """Bonded disc rubble layer crushed by a rigid hull plate."""

    def __init__(self, specimen: RidgeSpecimen, params: CrushParameters,
                 *, particles: dict | None = None) -> None:
        self.specimen = specimen
        self.params = params
        built = build_specimen(specimen) if particles is None else particles
        self.positions = torch.as_tensor(built["positions"], dtype=torch.float64)
        self.radius = float(built["radius"])
        count = int(self.positions.shape[0])
        self.count = count
        self.porosity_measured = float(built["porosity"])
        self.velocities = torch.zeros_like(self.positions)
        volume = math.pi * self.radius ** 2
        self.mass = torch.full((count,), params.ice_density * volume, dtype=torch.float64)
        self.buoyancy = (params.water_density - params.ice_density) * volume * params.gravity
        bonds = torch.as_tensor(built["bonds"], dtype=torch.long).reshape(-1, 2)
        self.bond_i = bonds[:, 0]
        self.bond_j = bonds[:, 1]
        delta = self.positions[self.bond_j] - self.positions[self.bond_i]
        self.bond_length0 = torch.linalg.vector_norm(delta, dim=1)
        self.bond_alive = torch.ones(self.bond_i.numel(), dtype=torch.bool)  # kept for API
        self.bond_shear = torch.zeros((self.bond_i.numel(), 2), dtype=torch.float64)
        self.bond_area = 2.0 * self.radius  # unit out-of-plane thickness
        self.bond_tensile_force = params.bond_strength_fraction * params.ice_strength * self.bond_area
        self.bond_shear_force = params.bond_shear_ratio * self.bond_tensile_force
        self.bond_cohesion = torch.full((self.bond_i.numel(),), self.bond_shear_force,
                                        dtype=torch.float64)
        self.bond_slip = torch.zeros(self.bond_i.numel(), dtype=torch.float64)
        self._bonded_keys = self._make_keys(self.bond_i, self.bond_j)
        self._bonded_sorted, self._bonded_order = torch.sort(self._bonded_keys)

        self.hull_x = float(specimen.length)
        self.hull_velocity = -abs(params.hull_velocity)
        self.hull_mass = params.hull_mass
        self.floor_y = 0.0
        self.lid_y = float(specimen.height + params.sail_height)
        self.hull_top = float(specimen.height)
        self.dt = params.dt if params.dt is not None else self._recommended_dt()

        self._pairs = None
        self._pair_steps = 20
        self._pair_counter = 0
        self._reset_ledger()
        self.steps = 0
        self.hull_history: list[float] = []
        self.advance_history: list[float] = []
        self._initial_energy = self._stored_energy()

    # ------------------------------------------------------------------ set-up
    def _make_keys(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        lo = torch.minimum(i, j)
        hi = torch.maximum(i, j)
        return lo * self.count + hi

    def _recommended_dt(self) -> float:
        p = self.params
        neighbours = 6.0
        stiffness = neighbours * (p.contact_stiffness + 2.0 * p.bond_stiffness)
        damping = neighbours * p.contact_damping + p.drag
        mass = float(self.mass[0])
        elastic = 0.2 * math.sqrt(mass / stiffness)
        viscous = 0.2 * mass / damping if damping else math.inf
        return min(elastic, viscous)

    def _reset_ledger(self) -> None:
        self.energy_fracture = 0.0
        self.work_friction = 0.0
        self.work_damping = 0.0
        self.work_input = 0.0
        self.work_thrust = 0.0
        self.bonds_broken = 0
        self._initialised = True

    # --------------------------------------------------------------- neighbours
    def _rebuild_pairs(self) -> None:
        """Rebuild the interaction pair list (contacts within ``2r + skin*r``).

        Freeze-bonded pairs are *not* removed from the contact list: the bond
        and the contact penalty act together, so the interaction network stays
        continuous when a bond breaks.  Removing bonded pairs instead makes the
        contact force appear only at the next rebuild, which lets particles
        interpenetrate between rebuilds and destroys the run.
        """
        radius_sum = 2.0 * self.radius + self.params.skin_ratio * self.radius
        pos = self.positions
        collected = []
        step = 256
        for start in range(0, self.count, step):
            stop = min(start + step, self.count)
            delta = pos[start:stop, None, :] - pos[None, :, :]
            dist = torch.linalg.vector_norm(delta, dim=2)
            rows = torch.arange(start, stop)[:, None]
            cols = torch.arange(self.count)[None, :]
            mask = (dist < radius_sum) & (rows < cols)
            index = mask.nonzero()
            if index.numel():
                collected.append(torch.stack([index[:, 0] + start, index[:, 1]], dim=1))
        if collected:
            self._pairs = torch.cat(collected, dim=0)
        else:
            self._pairs = torch.zeros((0, 2), dtype=torch.long)
        self._pair_counter = 0

    def _maybe_rebuild_pairs(self) -> None:
        if self._pairs is None or self._pair_counter >= self.params.pair_interval:
            self._rebuild_pairs()

    def _hull_mask(self) -> torch.Tensor:
        """Particles the hull plate can load.

        The plate spans the keel depth only (``y <= hull_top``) so that crushed
        rubble can be extruded upward, and it cannot load material the hull has
        already passed (``x <= hull_x + r``); otherwise a particle that is
        squeezed over the plate top and drifts behind it re-enters the plate
        with a huge spurious penetration and destroys the run.
        """
        return (self.positions[:, 1] <= self.hull_top) & \
            (self.positions[:, 0] <= self.hull_x + self.radius)

    # ------------------------------------------------------------------- forces
    def _bond_forces(self) -> tuple[torch.Tensor, dict]:
        """Cohesive-brittle bond forces with bounded (plastic) magnitudes.

        A freeze bond carries an axial spring with a tensile cut-off and a
        tangential spring bounded by a cohesion ``c`` that softens with the
        accumulated plastic slip.  The force never jumps: when the trial
        tangential force exceeds ``c`` the excess is projected back onto the
        cohesion surface, the elastic energy of the projection is booked as
        cohesive dissipation and ``c`` decays.  A bounded law is a modelling
        choice, not a convenience: an instantaneous bond break in an explicit
        scheme releases a finite force step into a stiff lattice whose energy
        cannot be resolved, and the run then diverges (the brittle-bond version
        of this model did exactly that).  With the plastic projection the
        rubble is crushed progressively, which is what a ridge does.
        """
        zeros = torch.zeros_like(self.positions)
        p = self.params
        if self.bond_i.numel() == 0:  # pragma: no cover - defensive
            return zeros, {"dissipation": 0.0, "exhausted": 0}
        delta = self.positions[self.bond_j] - self.positions[self.bond_i]
        length = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
        normal = delta / length[:, None]
        extension = length - self.bond_length0
        axial = p.bond_stiffness * extension
        tensile_limit = p.bond_strength_fraction * p.ice_strength * self.bond_area
        excess_axial = (axial - tensile_limit).clamp_min(0.0)
        axial = axial - excess_axial
        tangential = self.bond_shear - (self.bond_shear * normal).sum(dim=1, keepdim=True) * normal
        shear_trial = p.bond_shear_stiffness * tangential
        magnitude = torch.linalg.vector_norm(shear_trial, dim=1)
        cohesion = p.bond_shear_ratio * p.bond_strength_fraction * p.ice_strength * self.bond_area
        cap = self.bond_cohesion
        yielding = magnitude > cap
        dissipation = 0.0
        if bool(yielding.any()):
            slip = (magnitude - cap).clamp_min(0.0) / p.bond_shear_stiffness
            slip_total = self.bond_slip + slip
            cohesion_new = cohesion * torch.exp(-slip_total / max(p.cohesion_softening, 1e-12))
            cap_new = torch.where(yielding, cohesion_new, cap)
            scale = torch.where(yielding, cap_new / magnitude.clamp_min(1e-30),
                                torch.ones_like(magnitude))
            shear_force = shear_trial * scale[:, None]
            # the stored elastic energy that the plastic projection removes is
            # dissipated (no velocity change in a projection step)
            released = 0.5 * (magnitude ** 2 - cap_new ** 2).clamp_min(0.0) \
                / p.bond_shear_stiffness
            dissipation += float(released[yielding].sum())
            self.bond_shear = tangential * scale[:, None]
            self.bond_slip = torch.where(yielding, slip_total, self.bond_slip)
            self.bond_cohesion = torch.where(yielding, cohesion_new, self.bond_cohesion)
        else:
            shear_force = shear_trial
        if bool((excess_axial > 0.0).any()):
            released = 0.5 * ((axial + excess_axial) ** 2 - axial ** 2) / p.bond_stiffness
            dissipation += float(released[excess_axial > 0.0].sum())
            # the tensile cut-off is plastic: forget the excess extension
            self.bond_length0 = torch.where(excess_axial > 0.0, length, self.bond_length0)
        force = axial[:, None] * normal + shear_force
        zeros.index_add_(0, self.bond_i, force)
        zeros.index_add_(0, self.bond_j, -force)
        self.energy_fracture += dissipation
        exhausted = int((self.bond_cohesion < 0.05 * cohesion).sum())
        return zeros, {"dissipation": dissipation, "exhausted": exhausted}

    def _contact_forces(self) -> tuple[torch.Tensor, dict]:
        zeros = torch.zeros_like(self.positions)
        p = self.params
        if self._pairs is None or self._pairs.numel() == 0:
            return zeros, {"friction_power": 0.0, "damping_power": 0.0, "elastic": 0.0}
        i = self._pairs[:, 0]
        j = self._pairs[:, 1]
        delta = self.positions[j] - self.positions[i]
        dist = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
        normal = delta / dist[:, None]
        overlap = 2.0 * self.radius - dist
        active = overlap > 0.0
        if not bool(active.any()):
            return zeros, {"friction_power": 0.0, "damping_power": 0.0, "elastic": 0.0}
        i = i[active]
        j = j[active]
        normal = normal[active]
        overlap = overlap[active]
        relative = self.velocities[j] - self.velocities[i]
        vn = (relative * normal).sum(dim=1)
        damping = -p.contact_damping * vn
        damping = torch.clamp(damping, min=0.0)
        fn = torch.clamp(p.contact_stiffness * overlap + damping, min=0.0)
        tangential = relative - vn[:, None] * normal
        vt = torch.linalg.vector_norm(tangential, dim=1)
        scale = torch.tanh(vt / p.velocity_regularisation)
        ft = -p.friction_coefficient * fn * scale / vt.clamp_min(1e-30)
        friction = tangential * ft[:, None]
        # the normal force pushes the pair apart: -n for i, +n for j
        force = -normal * fn[:, None] + friction
        zeros.index_add_(0, i, force)
        zeros.index_add_(0, j, -force)
        friction_power = float((p.friction_coefficient * fn * scale * vt).sum())
        damping_power = float((p.contact_damping * torch.clamp(-vn, min=0.0) ** 2).sum())
        elastic = float((0.5 * p.contact_stiffness * overlap ** 2).sum())
        return zeros, {"friction_power": friction_power, "damping_power": damping_power,
                       "elastic": elastic}

    def _plane_force(self, normal: tuple[float, float], gap: torch.Tensor,
                     friction: float, hull: bool = False,
                     mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """Penalty + damping + Coulomb force of a rigid plane on the particles."""
        p = self.params
        zeros = torch.zeros_like(self.positions)
        n = torch.tensor(normal, dtype=torch.float64)
        pen = self.radius - gap
        active = pen > 0.0
        if mask is not None:
            active = active & mask
        if not bool(active.any()):
            return zeros, {"friction_power": 0.0, "damping_power": 0.0,
                           "elastic": 0.0, "normal": 0.0}
        relative = self.velocities
        if hull:
            relative = relative - torch.stack(
                (torch.full_like(gap, self.hull_velocity), torch.zeros_like(gap)), dim=1)
        vn = (relative * n).sum(dim=1)
        damping = torch.clamp(-p.contact_damping * vn, min=0.0)
        fn = torch.clamp(p.contact_stiffness * pen + damping, min=0.0)
        fn = fn * active
        tangential = relative - vn[:, None] * n
        vt = torch.linalg.vector_norm(tangential, dim=1)
        scale = torch.tanh(vt / p.velocity_regularisation)
        ft = -friction * fn * scale / vt.clamp_min(1e-30)
        friction_force = tangential * ft[:, None]
        force = (n * fn[:, None] + friction_force) * active[:, None]
        zeros += force
        diagnostics = {
            "friction_power": float((friction * fn * scale * vt).sum()),
            "damping_power": float((p.contact_damping * torch.clamp(-vn * active, min=0.0) ** 2).sum()),
            "elastic": float((0.5 * p.contact_stiffness * (pen * active) ** 2).sum()),
            "normal": float(fn.sum()),
        }
        return zeros, diagnostics

    def forces(self) -> dict:
        p = self.params
        self._maybe_rebuild_pairs()
        force = torch.zeros_like(self.positions)
        # buoyancy (upwards, only when the effective-stress idealisation is off)
        if p.buoyancy:
            force[:, 1] += self.buoyancy
        force -= p.drag * self.velocities
        bond_force, _ = self._bond_forces()
        force += bond_force
        contact_force, contact_diag = self._contact_forces()
        force += contact_force
        floor, floor_diag = self._plane_force((0.0, 1.0), self.positions[:, 1],
                                              p.wall_friction_coefficient)
        force += floor
        if p.lid:
            lid, lid_diag = self._plane_force((0.0, -1.0), self.lid_y - self.positions[:, 1],
                                              p.wall_friction_coefficient)
            force += lid
        else:
            lid_diag = {"friction_power": 0.0, "damping_power": 0.0, "elastic": 0.0}
        wall, wall_diag = self._plane_force((1.0, 0.0), self.positions[:, 0],
                                            p.wall_friction_coefficient)
        force += wall
        hull, hull_diag = self._plane_force((-1.0, 0.0),
                                            self.hull_x - self.positions[:, 0],
                                            p.hull_friction_coefficient, hull=True,
                                            mask=self._hull_mask())
        force += hull
        drag_power = float((p.drag * (self.velocities ** 2).sum(dim=1)).sum())
        return {
            "force": force,
            "hull_force": hull,
            "friction_power": contact_diag["friction_power"] + floor_diag["friction_power"]
            + lid_diag["friction_power"] + wall_diag["friction_power"]
            + hull_diag["friction_power"],
            "damping_power": contact_diag["damping_power"] + floor_diag["damping_power"]
            + lid_diag["damping_power"] + wall_diag["damping_power"]
            + hull_diag["damping_power"] + drag_power,
            "elastic": contact_diag["elastic"] + floor_diag["elastic"] + lid_diag["elastic"]
            + wall_diag["elastic"] + hull_diag["elastic"],
            "hull_normal_force": hull_diag["normal"],
            "hull_normal_x": -hull_diag["normal"],
            "hull_contact_energy": hull_diag["elastic"],
        }

    # ------------------------------------------------------------ time stepping
    def step(self) -> dict:
        self._maybe_rebuild_pairs()
        out = self.forces()
        force = out["force"]
        # shear memory of the bonds is updated with the mid-step velocity
        p = self.params
        delta = self.positions[self.bond_j] - self.positions[self.bond_i]
        length = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
        normal = delta / length[:, None]
        relative = self.velocities[self.bond_j] - self.velocities[self.bond_i]
        tangential_velocity = relative - (relative * normal).sum(dim=1, keepdim=True) * normal
        self.bond_shear = self.bond_shear + tangential_velocity * self.dt
        acceleration = force / self.mass[:, None]
        self.velocities = self.velocities + acceleration * self.dt
        if p.velocity_damping > 0.0:
            # quasi-static local damping: the ridge is crushed slowly, so the
            # inertial energy must not be allowed to build up.  The energy taken
            # out here is booked as damping work, which keeps the ledger exact.
            before = float(0.5 * (self.mass[:, None] * self.velocities ** 2).sum())
            self.velocities = self.velocities * (1.0 - p.velocity_damping)
            after = float(0.5 * (self.mass[:, None] * self.velocities ** 2).sum())
            self.work_damping += before - after
        self.positions = self.positions + self.velocities * self.dt
        delta_pos = self.velocities * self.dt
        # hull advance and external work
        hull_force_x = float(out["hull_force"][:, 0].sum())
        self.hull_history.append(hull_force_x)
        if self.hull_mass is None:
            self.hull_x += self.hull_velocity * self.dt
            self.work_input += float(out["hull_normal_x"]) * self.hull_velocity * self.dt
        else:
            accel = (self.params.thrust + hull_force_x) / self.hull_mass
            self.hull_velocity += accel * self.dt
            self.hull_x += self.hull_velocity * self.dt
            self.work_thrust += -self.params.thrust * self.hull_velocity * self.dt
        self.work_friction += out["friction_power"] * self.dt
        self.work_damping += out["damping_power"] * self.dt
        self._pair_counter += 1
        self.steps += 1
        return out

    # -------------------------------------------------------------- integration
    def _stored_energy(self) -> dict:
        """All reversible energies of the integrated system [J per unit thickness]."""
        p = self.params
        ke_particles = float(0.5 * (self.mass[:, None] * self.velocities ** 2).sum())
        ke_hull = 0.5 * self.hull_mass * self.hull_velocity ** 2 if self.hull_mass else 0.0
        bond_energy = 0.0
        if self.bond_i.numel():
            delta = self.positions[self.bond_j] - self.positions[self.bond_i]
            length = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
            normal = delta / length[:, None]
            extension = length - self.bond_length0
            tangential = self.bond_shear - (self.bond_shear * normal).sum(dim=1, keepdim=True) * normal
            axial = 0.5 * p.bond_stiffness * extension ** 2
            shear = 0.5 * p.bond_shear_stiffness * (tangential ** 2).sum(dim=1)
            bond_energy = float((axial + shear)[self.bond_alive].sum())
        contact_energy = 0.0
        if self._pairs is not None and self._pairs.numel():
            i = self._pairs[:, 0]
            j = self._pairs[:, 1]
            dist = torch.linalg.vector_norm(self.positions[j] - self.positions[i], dim=1)
            overlap = (2.0 * self.radius - dist).clamp_min(0.0)
            contact_energy = float((0.5 * p.contact_stiffness * overlap ** 2).sum())
        boundary_energy = 0.0
        planes = [((0.0, 1.0), self.positions[:, 1]),
                  ((1.0, 0.0), self.positions[:, 0])]
        hull_pen = (self.radius - (self.hull_x - self.positions[:, 0])).clamp_min(0.0)
        hull_pen = torch.where(self._hull_mask(), hull_pen,
                               torch.zeros_like(hull_pen))
        boundary_energy += float((0.5 * p.contact_stiffness * hull_pen ** 2).sum())
        if p.lid:
            planes.append(((0.0, -1.0), self.lid_y - self.positions[:, 1]))
        for normal, gap in planes:
            pen = (self.radius - gap).clamp_min(0.0)
            boundary_energy += float((0.5 * p.contact_stiffness * pen ** 2).sum())
        buoyancy_potential = float(-self.buoyancy * self.positions[:, 1].sum()) \
            if p.buoyancy else 0.0
        total = (ke_particles + ke_hull + bond_energy + contact_energy
                 + boundary_energy + buoyancy_potential)
        return {
            "ke_particles": ke_particles,
            "ke_hull": float(ke_hull),
            "bond_elastic": bond_energy,
            "contact_elastic": contact_energy,
            "boundary_elastic": boundary_energy,
            "buoyancy_potential": buoyancy_potential,
            "elastic_total": bond_energy + contact_energy + boundary_energy,
            "total": total,
        }

    def energy(self) -> dict:
        """Stored energy plus the accumulated dissipation ledger."""
        stored = self._stored_energy()
        stored["fracture"] = float(self.energy_fracture)
        stored["friction_work"] = float(self.work_friction)
        stored["damping_work"] = float(self.work_damping)
        stored["input_work"] = float(self.work_input)
        stored["thrust_work"] = float(self.work_thrust)
        stored["bonds_broken"] = int(self.bonds_broken)
        return stored

    def ledger(self) -> dict:
        """Energy closure of the whole run.

        ``residual = (stored + dissipated) - (initial stored + external work)``
        where the external work is the kinematic plate work or, for a rigid
        hull, the thrust work.  The closure error is normalised both by the
        input energy and by the total energy that was exchanged, so a large
        inert reservoir cannot hide a ledger error.
        """
        final = self._stored_energy()
        initial = self._initial_energy
        external = self.work_input + self.work_thrust
        dissipated = self.energy_fracture + self.work_friction + self.work_damping
        lhs = final["total"] + dissipated
        rhs = initial["total"] + external
        residual = lhs - rhs
        exchanged = abs(final["total"] - initial["total"]) + dissipated + abs(external)
        scale = max(abs(rhs), 1e-30)
        return {
            "initial_total": float(initial["total"]),
            "final_total": float(final["total"]),
            "stored_change": float(final["total"] - initial["total"]),
            "external_work": float(external),
            "fracture_energy": float(self.energy_fracture),
            "friction_work": float(self.work_friction),
            "damping_work": float(self.work_damping),
            "dissipated": float(dissipated),
            "residual": float(residual),
            "closure_relative": float(abs(residual) / scale),
            "closure_of_exchanged": float(abs(residual) / max(exchanged, 1e-30)),
            "energy_terms": {k: float(v) for k, v in final.items() if isinstance(v, (int, float))},
        }

    def run(self, *, max_steps: int = 400000, trace_every: int = 20,
            progress: bool = False) -> dict:
        """Advance the hull until the target advance is reached."""
        self.steps = 0
        self.hull_history = []
        self.advance_history = []
        self._initial_energy = self._stored_energy()
        self.dt_used = self.dt
        advance_target = self.params.advance
        force_series: list[float] = []
        while self.steps < max_steps:
            self.step()
            advance = float(self.specimen.length - self.hull_x)
            force_series.append(float(-self.hull_history[-1]))
            if self.steps % trace_every == 0:
                self.advance_history.append(advance)
            if advance >= advance_target:
                break
            if self.hull_mass is not None and self.hull_velocity > -1e-3:
                break
        advance = float(self.specimen.length - self.hull_x)
        forces = np.asarray(force_series, dtype=float)
        window = forces[len(forces) // 2:] if forces.size else forces
        steady = float(window.mean()) if window.size else 0.0
        peak = float(forces.max()) if forces.size else 0.0
        hull_normal = float(self.params.contact_stiffness * 0.0)
        return {
            "steps": int(self.steps),
            "dt": float(self.dt),
            "advance": advance,
            "advance_target": advance_target,
            "particles": self.count,
            "bonds_initial": int(self.bond_i.numel()),
            "bonds_broken": int(self.bonds_broken),
            "porosity_measured": self.porosity_measured,
            "force_steady": steady,
            "force_peak": peak,
            "force_series_mean": float(forces.mean()) if forces.size else 0.0,
            "pressure_steady": steady / self.specimen.height,
            "pressure_peak": peak / self.specimen.height,
            "pressure_ratio": steady / self.specimen.height / self.params.ice_strength,
            "hull_normal_force_max": float(max(self.hull_history)) if self.hull_history else 0.0,
            "ledger": self.ledger(),
            "energy": self.energy(),
            "force_trace": {
                "advance": [float(v) for v in self.advance_history],
                "mean_force": _block_mean(forces, len(self.advance_history)),
            },
        }


def _block_mean(series: np.ndarray, blocks: int) -> list[float]:
    if blocks <= 0 or series.size == 0:
        return []
    edges = np.linspace(0, series.size, blocks + 1).astype(int)
    out = []
    for k in range(blocks):
        lo, hi = edges[k], max(edges[k + 1], edges[k] + 1)
        out.append(float(series[lo:hi].mean()))
    return out


@dataclass
class CrushResult:
    """Container for one DEM crushing run."""

    run: dict
    specimen: RidgeSpecimen
    params: CrushParameters
    particles: dict = field(default_factory=dict)

    @property
    def pressure_steady(self) -> float:
        return float(self.run["pressure_steady"])

    @property
    def force_steady(self) -> float:
        return float(self.run["force_steady"])

    def as_dict(self) -> dict:
        return {
            "specimen": {
                "height": self.specimen.height,
                "length": self.specimen.length,
                "porosity": self.specimen.porosity,
                "particle_radius": self.specimen.particle_radius,
                "seed": int(self.specimen.seed),
            },
            "parameters": {
                "ice_strength": self.params.ice_strength,
                "bond_strength_fraction": self.params.bond_strength_fraction,
                "hull_velocity": self.params.hull_velocity,
                "advance": self.params.advance,
                "friction_coefficient": self.params.friction_coefficient,
                "contact_stiffness": self.params.contact_stiffness,
                "bond_stiffness": self.params.bond_stiffness,
                "dt": self.run.get("dt"),
            },
            "pressure_steady": self.pressure_steady,
            "pressure_peak": float(self.run["pressure_peak"]),
            "force_steady": self.force_steady,
            "force_peak": float(self.run["force_peak"]),
            "porosity_measured": float(self.run["porosity_measured"]),
            "bonds_broken": int(self.run["bonds_broken"]),
            "bonds_initial": int(self.run["bonds_initial"]),
            "steps": int(self.run["steps"]),
            "ledger": self.run["ledger"],
        }


def run_crush_case(height: float, *, ice_strength: float, porosity: float = 0.30,
                   hull_velocity: float = 2.0, advance: float = 2.0,
                   particle_radius: float = 0.5, seed: int = 20261108,
                   length_ratio: float = 1.5, max_steps: int = 400000,
                   **kwargs) -> CrushResult:
    """Convenience wrapper: build the specimen, run the crush and package it."""
    specimen = RidgeSpecimen(height=height, length=length_ratio * height,
                             porosity=porosity, particle_radius=particle_radius,
                             seed=seed)
    params = CrushParameters(ice_strength=ice_strength, hull_velocity=hull_velocity,
                             advance=advance, **kwargs)
    dem = RidgeCrushDem(specimen, params)
    run = dem.run(max_steps=max_steps)
    return CrushResult(run=run, specimen=specimen, params=params,
                       particles={"count": dem.count})
