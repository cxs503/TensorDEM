"""Soft-sphere DEM mechanics for the Lagrangian bed-fluid coupling.

Monodisperse (optionally polydisperse) spheres integrated with a
velocity-Verlet / semi-implicit scheme and a linear spring-dashpot normal
contact plus rate-limited Coulomb friction:

    F_n = k_n delta - gamma_n (v_rel . n)        (F_n >= 0)
    F_t = -min(mu |F_n|, k_t |v_t| dt) v_t / |v_t|

Wall planes (floor, side walls, optional ceiling) use the same normal law with
the plane as a body of infinite mass.  The contact stiffness is a *numerical*
parameter chosen so that the contact time is resolved by ~50 time steps; this
is the standard soft-sphere compromise and is documented as a limitation in the
acceptance report (contact stiffness is not calibrated to a material).

All force sums are deterministic; the only stochastic input is the initial
packing (seeded), so two runs with the same seed are bit-identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .closures import GRAVITY, sphere_volume


@dataclass
class ContactParameters:
    """Numerical contact parameters (soft-sphere compromise)."""

    k_n: float = 4.0e0       # normal stiffness [N/m] (static overlap ~2 % of d)
    k_t: float = 1.6e0       # tangential stiffness [N/m]
    gamma_n: float = 6.8e-4  # normal damping [N s/m] (restitution ratio ~0.15)
    mu_friction: float = 0.3
    friction_rate_limit: bool = True

    def contact_time(self, mass: float) -> float:
        return float(np.pi * np.sqrt(mass / self.k_n))


@dataclass
class Box:
    """Column/box geometry: periodic in x,y or bounded by side walls."""

    size_xy: float
    height: float
    floor_z: float = 0.0
    ceil_z: float | None = None
    periodic_xy: bool = True
    wall_margin: float = 0.0

    @property
    def area(self) -> float:
        return self.size_xy**2


@dataclass
class ParticleState:
    pos: np.ndarray
    vel: np.ndarray
    radius: np.ndarray
    density: float

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    @property
    def mass(self) -> np.ndarray:
        return self.density * 4.0 / 3.0 * np.pi * self.radius**3


def make_particles(pos, radius, density, vel=None) -> ParticleState:
    pos = np.asarray(pos, dtype=float)
    rad = np.broadcast_to(np.asarray(radius, dtype=float), (pos.shape[0],)).copy()
    vel = np.zeros_like(pos) if vel is None else np.asarray(vel, dtype=float).copy()
    return ParticleState(pos=pos, vel=vel, radius=rad, density=float(density))


_TRIU_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _triu(n: int):
    cached = _TRIU_CACHE.get(n)
    if cached is None:
        cached = np.triu_indices(n, k=1)
        _TRIU_CACHE[n] = cached
    return cached


def pair_contacts(state: ParticleState, box: Box):
    """Return (i, j, normal, overlap) for all overlapping pairs (minimum image in xy)."""
    pos = state.pos
    rad = state.radius
    n = pos.shape[0]
    if n < 2:
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros((0, 3)), np.zeros(0)
    delta = pos[None, :, :] - pos[:, None, :]
    if box.periodic_xy:
        delta[:, :, 0] -= box.size_xy * np.round(delta[:, :, 0] / box.size_xy)
        delta[:, :, 1] -= box.size_xy * np.round(delta[:, :, 1] / box.size_xy)
    dist = np.linalg.norm(delta, axis=2)
    rsum = rad[:, None] + rad[None, :]
    iu, ju = _triu(n)
    overlap = rsum[iu, ju] - dist[iu, ju]
    mask = overlap > 0.0
    iu, ju, overlap = iu[mask], ju[mask], overlap[mask]
    d = dist[iu, ju]
    normal = delta[iu, ju] / np.maximum(d, 1e-30)[:, None]
    return iu, ju, normal, overlap


def contact_forces(state: ParticleState, box: Box, params: ContactParameters, dt: float):
    """Net contact (particle-particle + wall) force [N] per particle."""
    pos, vel, rad = state.pos, state.vel, state.radius
    force = np.zeros_like(pos)

    iu, ju, normal, overlap = pair_contacts(state, box)
    if iu.size:
        v_rel = vel[iu] - vel[ju]
        vn = np.einsum("ij,ij->i", v_rel, normal)   # >0 = approaching
        # Kelvin-Voigt: F_n = k_n * overlap + gamma_n * (d overlap/dt), with
        # d(overlap)/dt = vn.  (+ gamma, NOT - gamma: the minus sign is
        # anti-damping and pumps energy -- verified against a restitution test.)
        fn = np.maximum(params.k_n * overlap + params.gamma_n * vn, 0.0)
        # normal points from i to j, so the force on i is repulsive (-n)
        f_on_i = -fn[:, None] * normal
        # tangential (rate-limited Coulomb), opposing the slip of i w.r.t. j
        vt = v_rel - vn[:, None] * normal
        vt_mag = np.linalg.norm(vt, axis=1)
        ft_mag = np.minimum(params.mu_friction * fn, params.k_t * vt_mag * dt)
        f_on_i += -(ft_mag / np.maximum(vt_mag, 1e-30))[:, None] * vt
        np.add.at(force, iu, f_on_i)
        np.add.at(force, ju, -f_on_i)

    # floor
    pen = rad - (pos[:, 2] - box.floor_z)
    below = pen > 0.0
    if np.any(below):
        # normal +z (wall -> particle); damping opposes the relative normal velocity
        fn_floor = np.maximum(params.k_n * pen[below] - params.gamma_n * vel[below, 2], 0.0)
        force[below, 2] += fn_floor
        vt = vel[below, :2].copy()
        vt_mag = np.linalg.norm(vt, axis=1)
        ft_mag = np.minimum(params.mu_friction * fn_floor, params.k_t * vt_mag * dt)
        force[below, :2] += -(ft_mag / np.maximum(vt_mag, 1e-30))[:, None] * vt

    # ceiling (compaction piston)
    if box.ceil_z is not None:
        pen_c = (pos[:, 2] + rad) - box.ceil_z
        above = pen_c > 0.0
        if np.any(above):
            fn_ceil = np.maximum(params.k_n * pen_c[above] + params.gamma_n * vel[above, 2], 0.0)
            force[above, 2] -= fn_ceil

    # side walls
    if not box.periodic_xy:
        lo = box.wall_margin
        hi = box.size_xy - box.wall_margin
        for axis in (0, 1):
            pen_lo = lo + rad - pos[:, axis]
            m = pen_lo > 0
            if np.any(m):
                force[m, axis] += np.maximum(params.k_n * pen_lo[m], 0.0)
            pen_hi = pos[:, axis] + rad - hi
            m = pen_hi > 0
            if np.any(m):
                force[m, axis] -= np.maximum(params.k_n * pen_hi[m], 0.0)
    return force


def total_gravity_force(state: ParticleState, gravity=GRAVITY) -> np.ndarray:
    f = np.zeros_like(state.pos)
    f[:, 2] = -state.mass * gravity
    return f


def step(
    state: ParticleState,
    box: Box,
    params: ContactParameters,
    dt: float,
    external_force=None,
    gravity: float = GRAVITY,
    max_speed: float | None = None,
) -> ParticleState:
    """Advance one step (semi-implicit Euler with contact + external forcing)."""
    force = contact_forces(state, box, params, dt) + total_gravity_force(state, gravity)
    if external_force is not None:
        force = force + external_force
    acc = force / state.mass[:, None]
    vel = state.vel + acc * dt
    if max_speed is not None:
        speed = np.linalg.norm(vel, axis=1)
        scale = np.minimum(1.0, max_speed / np.maximum(speed, 1e-30))
        vel = vel * scale[:, None]
    pos = state.pos + vel * dt
    if box.periodic_xy:
        pos[:, 0] = np.mod(pos[:, 0], box.size_xy)
        pos[:, 1] = np.mod(pos[:, 1], box.size_xy)
    return ParticleState(pos=pos, vel=vel, radius=state.radius, density=state.density)


def kinetic_energy(state: ParticleState) -> float:
    return float(0.5 * np.sum(state.mass * np.sum(state.vel**2, axis=1)))


def wall_reaction_z(state: ParticleState, box: Box, params: ContactParameters) -> float:
    """Vertical reaction force carried by the floor (sum of floor normals)."""
    pen = state.radius - (state.pos[:, 2] - box.floor_z)
    below = pen > 0.0
    if not np.any(below):
        return 0.0
    return float(
        np.sum(np.maximum(params.k_n * pen[below] + params.gamma_n * state.vel[below, 2], 0.0))
    )