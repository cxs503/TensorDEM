"""Rigid-body DEM of floating polyhedral floes: dynamics, buoyancy and ledgers.

The 2-D floe field moves in the plane and has one rotational degree of freedom.
A polyhedral floe has six: it can be tilted by a contact, it has a full inertia
tensor, and the buoyancy force acts at the centre of buoyancy, which is not the
centre of mass.  This module integrates Newton-Euler for that body,

.. math::
    m \\dot v = \\sum f, \\qquad
    I \\dot\\omega = \\sum \\tau - \\omega\\times I\\omega, \\qquad
    \\dot R = [\\omega]_\\times R,

with contacts from :mod:`~tensordem.polyhedron_floe3d.contact`, gravity acting at
the centroid and buoyancy acting at the centre of buoyancy.

Buoyancy of a partially submerged polyhedron
--------------------------------------------
The submerged volume and its centroid are computed **exactly** from the body's
tetrahedral decomposition: for each tetrahedron of a fan from the centroid the
part below the water plane is either empty/full or a smaller tetrahedron
(one wet vertex), the complement of a small tetrahedron (three wet vertices) or a
triangular prism (two wet vertices).  Each case has a closed-form volume and
centroid, so neither :math:`V_{sub}` nor the metacentric restoring moment is a
quadrature or a raster approximation.  A unit test cross-checks the routine
against an independent convex-hull clipping reference on random tetrahedra.

Integrators and what "energy conservation" means here
----------------------------------------------------
``integrator="leapfrog"`` (default, one force evaluation per step) is symplectic:
the *discrete* (shadow) energy
:math:`\\tfrac12 m v_{n+1/2}\\!\\cdot\\! v_{n+3/2} + U_{n+1/2} + \\tfrac12 k\\,\\delta_n \\delta_{n+1}`
of a linearly-spring contact is conserved to machine precision, while the
*continuous* energy shows the usual bounded :math:`O((\\omega\\Delta t)^2)`
oscillation.  Both are reported (``energy_residual_shadow`` /
``energy_residual_true``), because quoting only one of them would hide either the
scheme's accuracy or its stability.  ``integrator="verlet"`` (kick-drift-kick,
two force evaluations) is available for comparisons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contact import ContactLaw, PairContact, sat_contact, wall_contact
from .geometry import ConvexPolyhedron

__all__ = [
    "FloeBody",
    "Wall",
    "DEMSystem3D",
    "submerged_tet_moments",
    "body_tetrahedra",
]

_TINY = 1.0e-300


def _cross3(a, b):
    """Cross product of two 3-vectors without numpy's axis bookkeeping."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


# ---------------------------------------------------------------------------
# exact buoyancy of a partially submerged polyhedron
# ---------------------------------------------------------------------------
def body_tetrahedra(shape: ConvexPolyhedron) -> np.ndarray:
    """Tetrahedral fan of a convex body (indices into ``shape.vertices``).

    The centroid of a convex hull is interior, so every tetrahedron
    ``(centroid, a, b, c)`` of the fan has a positive volume and the fan tiles
    the body exactly.
    """
    tets = np.asarray(shape.faces, dtype=int)
    pts = shape.vertices[tets]
    centro = pts.mean(axis=1)                     # circumcentre of each facet
    # the fan apex is the body centroid == origin in the body frame
    apex = np.zeros((tets.shape[0], 1), dtype=int)
    signed = np.einsum("ij,ij->i",
                       pts[:, 0] - 0.0,
                       np.cross(pts[:, 1] - 0.0, pts[:, 2] - 0.0))
    # orient every facet so that (a,b,c) points outwards from the centroid
    flip = signed < 0.0
    tets = tets.copy()
    tets[flip] = tets[flip][:, [0, 2, 1]]
    # store local coordinates explicitly with the apex as vertex 0
    apex_coords = np.zeros((tets.shape[0], 1, 3))
    return np.concatenate([apex_coords, shape.vertices[tets]], axis=1)


def submerged_tet_moments(points: np.ndarray, water_level: float,
                          axis: int = 2
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volume, first moment and centroid of the part of each tetrahedron below a plane.

    ``points`` is ``(n, 4, 3)`` world-frame tetrahedra (vertex 0 is the fan apex),
    ``water_level`` the plane ``x_axis = water_level``; the region
    ``x_axis <= water_level`` is kept.  ``axis = 2`` (default) is the waterline
    clipping used for buoyancy; ``axis = 0`` / ``1`` clip against a vertical
    plane, which is what the exact slab (channel-lane) volume uses.
    Returns ``(volume, moment, centroid)`` with ``moment`` the first moment
    :math:`\\int x\\,dV` in the *original* coordinate frame.
    """
    points = np.asarray(points, dtype=np.float64)
    if axis != 2:
        # clip along another axis by permuting that axis into the z slot, run the
        # (vectorised) z clipper, then permute the moments back
        perm = [0, 1, 2]
        perm[axis], perm[2] = perm[2], perm[axis]
        volume, moment, centroid = submerged_tet_moments(points[..., perm],
                                                         water_level, axis=2)
        moment_out = np.zeros_like(moment)
        centroid_out = np.zeros_like(centroid)
        moment_out[..., perm] = moment
        centroid_out[..., perm] = centroid
        return volume, moment_out, centroid_out
    n = points.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
    level = points[:, :, 2] - float(water_level)     # (n, 4), < 0 => submerged
    wet = level < 0.0
    count = wet.sum(axis=1)

    p0, p1, p2, p3 = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    full_volume = np.abs(np.einsum("ij,ij->i", p1 - p0,
                                   np.cross(p2 - p0, p3 - p0))) / 6.0
    volume = np.zeros(n)
    moment = np.zeros((n, 3))

    def add(mask, vol, centroid):
        volume[mask] += vol[mask]
        moment[mask] += vol[mask, None] * centroid[mask]

    # ---- k = 0 (dry) and k = 4 (fully submerged)
    all_dry = count == 0
    all_wet = count == 4
    add(all_wet, full_volume, points.mean(axis=1))

    def _cut_subset(rows, ref_index, other_index, points, level, full_volume):
        """Small tetrahedron between a reference vertex and 3 'other' vertices."""
        sub = points[rows]
        sub_level = level[rows]
        inner = np.arange(rows.size)
        ref = sub[inner, ref_index]
        others = sub[inner[:, None], other_index]
        s_ref = sub_level[inner, ref_index][:, None]
        s_oth = sub_level[inner[:, None], other_index]
        denom = s_ref - s_oth
        safe = np.where(np.abs(denom) > _TINY, denom, 1.0)
        lam = np.clip(s_ref / safe, 0.0, 1.0)             # in [0, 1]
        cross = ref[:, None, :] + lam[..., None] * (others - ref[:, None, :])
        frac = np.prod(lam, axis=1)
        centroid = 0.25 * (ref + cross.sum(axis=1))
        return frac * full_volume[rows], centroid

    # ---- k = 1 : one wet vertex, a smaller tetrahedron from that vertex
    if np.any(count == 1):
        sel = count == 1
        order = np.argsort(~wet[sel], axis=1, kind="stable")   # wet vertex first
        rows = np.flatnonzero(sel)
        vol_s, centroid_s = _cut_subset(rows, order[:, 0], order[:, 1:],
                                        points, level, full_volume)
        volume[rows] += vol_s
        moment[rows] += vol_s[:, None] * centroid_s

    # ---- k = 3 : one dry vertex -> complement of a small tetrahedron
    if np.any(count == 3):
        sel = count == 3
        order = np.argsort(wet[sel], axis=1, kind="stable")    # dry vertex first
        vol_dry, centroid_dry = _cut_subset(np.flatnonzero(sel), order[:, 0],
                                            order[:, 1:], points, level, full_volume)
        vol_dry_full = np.zeros(n)
        centroid_dry_full = np.zeros((n, 3))
        vol_dry_full[sel] = vol_dry
        centroid_dry_full[sel] = centroid_dry
        vol = full_volume - vol_dry_full
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            centroid = (full_volume[:, None] * points.mean(axis=1)
                        - vol_dry_full[:, None] * centroid_dry_full) / np.where(vol > _TINY, vol, 1.0)[:, None]
        add(sel, vol, centroid)

    # ---- k = 2 : a triangular prism between the two wet vertices
    if np.any(count == 2):
        sel = count == 2
        idx = np.flatnonzero(sel)
        order = np.argsort(wet[idx], axis=1, kind="stable")   # two dry, two wet
        dry = order[:, :2]
        weti = order[:, 2:]
        pts = points[idx]
        lv = level[idx]
        rows = np.arange(idx.size)

        def cross_point(wi, di):
            s_w = lv[rows, wi]
            s_d = lv[rows, di]
            lam = (s_w / (s_w - s_d))[:, None]
            return pts[rows, wi] + lam * (pts[rows, di] - pts[rows, wi])

        a = pts[rows, weti[:, 0]]
        b = pts[rows, weti[:, 1]]
        q_ac = cross_point(weti[:, 0], dry[:, 0])
        q_ad = cross_point(weti[:, 0], dry[:, 1])
        q_bc = cross_point(weti[:, 1], dry[:, 0])
        q_bd = cross_point(weti[:, 1], dry[:, 1])
        vol_sum = np.zeros(idx.size)
        mom_sum = np.zeros((idx.size, 3))
        for tet in ((a, q_ac, q_ad, b), (q_ac, q_ad, b, q_bc), (q_ad, b, q_bc, q_bd)):
            q0, q1, q2, q3 = tet
            v = np.abs(np.einsum("ij,ij->i", q1 - q0,
                                 np.cross(q2 - q0, q3 - q0))) / 6.0
            vol_sum += v
            mom_sum += v[:, None] * 0.25 * (q0 + q1 + q2 + q3)
        volume[idx] += vol_sum
        moment[idx] += mom_sum

    centroid = moment / np.maximum(volume, _TINY)[:, None]
    return volume, moment, centroid


# ---------------------------------------------------------------------------
# bodies and walls
# ---------------------------------------------------------------------------
@dataclass
class Wall:
    """Half-space ``{x : normal . x >= offset}`` with an optional velocity."""

    normal: np.ndarray
    offset: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    label: str = ""

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=np.float64)
        self.normal = n / np.linalg.norm(n)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)


class FloeBody:
    """A rigid polyhedral floe with six degrees of freedom."""

    __slots__ = ("shape", "density", "position", "velocity", "rotation",
                 "omega", "tets", "mass", "inertia_diag", "label", "strength",
                 "external_force", "external_torque")

    def __init__(self, shape: ConvexPolyhedron, density: float,
                 position: Sequence[float] = (0.0, 0.0, 0.0),
                 rotation: Optional[np.ndarray] = None,
                 velocity: Sequence[float] = (0.0, 0.0, 0.0),
                 omega: Sequence[float] = (0.0, 0.0, 0.0),
                 label: str = "", strength: float = math.inf):
        self.shape = shape
        self.density = float(density)
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        self.omega = np.asarray(omega, dtype=np.float64)
        self.tets = body_tetrahedra(shape)               # (n_tet, 4, 3) local
        self.mass = float(density) * shape.volume
        self.inertia_diag = float(density) * shape.inertia_diag
        self.label = label
        self.strength = float(strength)
        self.external_force = np.zeros(3)
        self.external_torque = np.zeros(3)

    # -- derived state -----------------------------------------------------
    def world_vertices(self) -> np.ndarray:
        return self.shape.vertices @ self.rotation.T + self.position

    def world_normals(self) -> np.ndarray:
        return self.shape.normals @ self.rotation.T

    def world_edges(self) -> np.ndarray:
        e = self.shape.vertices[self.shape.edges[:, 1]] - self.shape.vertices[self.shape.edges[:, 0]]
        return e @ self.rotation.T

    def world_tets(self) -> np.ndarray:
        return self.tets @ self.rotation.T + self.position

    def inertia_world(self) -> np.ndarray:
        return self.rotation @ np.diag(self.inertia_diag) @ self.rotation.T

    def inverse_inertia_world(self) -> np.ndarray:
        inv = 1.0 / np.maximum(self.inertia_diag, _TINY)
        return self.rotation @ np.diag(inv) @ self.rotation.T

    @property
    def radius(self) -> float:
        return self.shape.bounding_radius

    def kinetic_energy(self) -> Tuple[float, float]:
        ke_t = 0.5 * self.mass * float(self.velocity @ self.velocity)
        inertia = self.inertia_world()
        ke_r = 0.5 * float(self.omega @ inertia @ self.omega)
        return ke_t, ke_r

    def rotate(self, delta_rotation: np.ndarray) -> None:
        rot = delta_rotation @ self.rotation
        u, _, vt = np.linalg.svd(rot)
        self.rotation = u @ vt


def _expm_skew(omega: np.ndarray, dt: float) -> np.ndarray:
    """Rodrigues formula for ``exp([omega]_x dt)``."""
    theta = float(np.linalg.norm(omega)) * dt
    if theta < 1.0e-12:
        k = np.array([[0.0, -omega[2], omega[1]],
                      [omega[2], 0.0, -omega[0]],
                      [-omega[1], omega[0], 0.0]]) * dt
        return np.eye(3) + k + 0.5 * k @ k
    axis = omega / np.linalg.norm(omega)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def _project_to_plane(body: FloeBody, z: float = 0.0) -> None:
    """Freeze the vertical and tilt degrees of freedom (the 2-D limit)."""
    body.position[2] = z
    body.velocity[2] = 0.0
    body.omega[0] = 0.0
    body.omega[1] = 0.0
    rotation = body.rotation
    angle = math.atan2(rotation[1, 0], rotation[0, 0])
    c, s = math.cos(angle), math.sin(angle)
    body.rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# the system
# ---------------------------------------------------------------------------
class DEMSystem3D:
    """Rigid-body DEM of polyhedral floes with contacts, buoyancy and ledgers."""

    def __init__(self, bodies: Sequence[FloeBody], *, law: Optional[ContactLaw] = None,
                 gravity: float = 9.81, water_level: Optional[float] = None,
                 water_density: float = 1025.0, dt: float = 2.0e-4,
                 integrator: str = "leapfrog", record: bool = True,
                 constrain_to_plane: bool = False,
                 linear_drag: float = 0.0, angular_drag: float = 0.0):
        self.bodies: List[FloeBody] = list(bodies)
        self.law = ContactLaw() if law is None else law
        self.gravity = float(gravity)
        self.water_level = None if water_level is None else float(water_level)
        self.water_density = float(water_density)
        self.dt = float(dt)
        self.integrator = integrator
        self.record = bool(record)
        #: freeze the vertical / tilt degrees of freedom (the 2-D limit model)
        self.constrain_to_plane = bool(constrain_to_plane)
        #: hydrodynamic drag closures [N s / m] and [N m s / rad]; ``F = -c v`` is
        #: the Stokes-like linear closure, used for the floating-equilibrium case.
        #: A modelling choice, not a measured material constant: it sets how fast a
        #: released floe relaxes to its hydrostatic draft, not where that draft is.
        self.linear_drag = float(linear_drag)
        self.angular_drag = float(angular_drag)
        self.walls: List[Wall] = []
        self.friction_state: Dict[Tuple[int, int], np.ndarray] = {}
        self.wall_friction_state: Dict[Tuple[int, int], np.ndarray] = {}
        self.rolling_state: Dict[Tuple[int, int], np.ndarray] = {}

        # ledgers
        self.ke_trans = 0.0
        self.ke_rot = 0.0
        self.pe_gravity = 0.0
        self.pe_spring = 0.0
        self.dissipated = 0.0
        self.work_external = 0.0
        #: buoyancy force per body for the current step; the work it does is booked
        #: in :meth:`step` with the *midpoint* velocity of the kick (see the
        #: ledger note in :meth:`energy_residual`), never as a power.
        self._buoyancy_force = np.zeros((0, 3))
        self.spring_pairs = np.zeros(0)      # 0.5 * kn * delta^2 per contact
        self.spring_deltas = np.zeros(0)     # delta per contact
        self.history: List[Dict[str, object]] = []
        self.time = 0.0
        self.last_pairs: List[PairContact] = []
        self.last_pair_forces: np.ndarray = np.zeros((0, 3))
        self.last_wall_forces: List[Tuple[int, int, np.ndarray]] = []
        self.wall_force_by_label: Dict[str, np.ndarray] = {}
        self.n_pair_contacts = 0
        self.n_wall_contacts = 0
        self.max_speed = 20.0
        self.max_spin = 200.0
        self.safety_clamps = 0
        self._initial_energy = 0.0
        self.mass_initial = sum(b.mass for b in self.bodies)
        self.refresh_energy()

    # -- setup -------------------------------------------------------------
    def add_wall(self, normal: Sequence[float], offset: float,
                 velocity: Sequence[float] = (0.0, 0.0, 0.0), label: str = "") -> Wall:
        wall = Wall(np.asarray(normal, dtype=np.float64), float(offset),
                    np.asarray(velocity, dtype=np.float64), label)
        self.walls.append(wall)
        return wall

    def add_box(self, centre: Sequence[float], half: Sequence[float], *,
                bottom: bool = True, sides: bool = True, top: bool = False) -> None:
        c = np.asarray(centre, dtype=np.float64)
        h = np.asarray(half, dtype=np.float64)
        if bottom:
            self.add_wall([0, 0, 1], c[2] - h[2], label="floor")
        if top:
            self.add_wall([0, 0, -1], -(c[2] + h[2]), label="roof")
        if sides:
            self.add_wall([1, 0, 0], c[0] - h[0], label="x-")
            self.add_wall([-1, 0, 0], -(c[0] + h[0]), label="x+")
            self.add_wall([0, 1, 0], c[1] - h[1], label="y-")
            self.add_wall([0, -1, 0], -(c[1] + h[1]), label="y+")

    # -- diagnostics -------------------------------------------------------
    def record_initial_energy(self) -> None:
        self.refresh_energy()
        self._initial_energy = (self.ke_trans + self.ke_rot + self.pe_gravity
                                + self.pe_spring)

    @property
    def total_kinetic(self) -> float:
        return self.ke_trans + self.ke_rot

    def refresh_energy(self) -> None:
        ke_t = 0.0
        ke_r = 0.0
        for body in self.bodies:
            t, r = body.kinetic_energy()
            ke_t += t
            ke_r += r
        self.ke_trans = ke_t
        self.ke_rot = ke_r
        self.pe_gravity = float(sum(b.mass * self.gravity * b.position[2]
                                    for b in self.bodies))

    def energy(self) -> float:
        return self.ke_trans + self.ke_rot + self.pe_gravity + self.pe_spring

    def energy_scale(self) -> float:
        """Reference energy used to normalise the ledger residual.

        A field that starts at rest has ``E(0) = 0`` (gravity balanced by
        buoyancy), so normalising by ``E(0)`` would divide by zero; the weight
        times a floe radius is the natural scale of the work done in that case.
        """
        scale = float(sum(b.mass * self.gravity * b.radius for b in self.bodies))
        return max(scale, 1.0e-30)

    def energy_residual(self) -> float:
        """|E(t) - E(0) - W_ext + Dissipation| / max(|E(0)|, weight*radius)."""
        reference = max(abs(self._initial_energy), self.energy_scale())
        drift = self.energy() - self._initial_energy - self.work_external + self.dissipated
        return abs(drift) / reference

    def mass_residual(self) -> float:
        total = sum(b.mass for b in self.bodies)
        reference = max(abs(self.mass_initial), 1.0e-30)
        return abs(total - self.mass_initial) / reference

    def total_momentum(self) -> np.ndarray:
        return sum((b.mass * b.velocity for b in self.bodies), np.zeros(3))

    def total_angular_momentum(self, about: Optional[Sequence[float]] = None) -> np.ndarray:
        about = np.zeros(3) if about is None else np.asarray(about, dtype=np.float64)
        total = np.zeros(3)
        for b in self.bodies:
            r = b.position - about
            inertia = b.inertia_world()
            total += b.mass * np.cross(r, b.velocity) + inertia @ b.omega
        return total

    def porosity(self, box_min: Sequence[float], box_max: Sequence[float]) -> float:
        """Solid fraction complement of polyhedral floes whose centroid is in the box."""
        lo = np.asarray(box_min, dtype=np.float64)
        hi = np.asarray(box_max, dtype=np.float64)
        volume = float(np.prod(hi - lo))
        if volume <= 0.0:
            return float("nan")
        solid = 0.0
        for b in self.bodies:
            if np.all(b.position >= lo) and np.all(b.position <= hi):
                solid += b.shape.volume
        return 1.0 - solid / volume

    def solid_volume_slab(self, axis: int, lo: float, hi: float) -> float:
        """Exact solid volume of the floes inside the slab ``lo <= x_axis <= hi``.

        Unlike :meth:`porosity` -- which counts a body in full as soon as its
        *centroid* falls inside the probe box -- this clips every tetrahedron of
        every floe against the two slab faces, so a floe half-way across a face
        contributes exactly half.  The comparison is what makes the channel
        concentration curve a measurement rather than a step function.

        Implemented as ``V(x <= hi) - V(x <= lo)`` with the same plane clipper
        used for the waterline (:func:`submerged_tet_moments`), so the two
        quantities cannot drift apart numerically.
        """
        total = 0.0
        for body in self.bodies:
            tets = body.world_tets()
            above = float(np.sum(submerged_tet_moments(tets, hi, axis=int(axis))[0]))
            below = float(np.sum(submerged_tet_moments(tets, lo, axis=int(axis))[0]))
            total += max(above - below, 0.0)
        return total

    # -- force assembly ----------------------------------------------------
    def _broad_phase(self, reaches: np.ndarray) -> List[Tuple[int, int]]:
        n = len(self.bodies)
        pairs: List[Tuple[int, int]] = []
        if n < 2:
            return pairs
        centres = np.array([b.position for b in self.bodies])
        limits = np.array([b.shape.bounding_radius for b in self.bodies]) + reaches
        for i in range(n):
            delta = centres[i + 1:] - centres[i]
            dist = np.linalg.norm(delta, axis=1)
            hit = np.flatnonzero(dist <= limits[i] + limits[i + 1:])
            pairs.extend((i, i + 1 + int(j)) for j in hit)
        return pairs

    def _buoyancy(self) -> None:
        """Exact buoyancy of every partially submerged body (one batched call)."""
        if self.water_level is None or not self.bodies:
            return
        sizes = [b.tets.shape[0] for b in self.bodies]
        tets = np.concatenate([b.world_tets() for b in self.bodies], axis=0)
        volume, moment, _centroid = submerged_tet_moments(tets, self.water_level)
        index = np.repeat(np.arange(len(self.bodies)), sizes)
        n_bodies = len(self.bodies)
        total = np.bincount(index, weights=volume, minlength=n_bodies)
        moments = np.stack([np.bincount(index, weights=moment[:, k], minlength=n_bodies)
                            for k in range(3)], axis=1)
        buoyancy = np.zeros((len(self.bodies), 3))
        for index, (body, vol, mom) in enumerate(zip(self.bodies, total, moments)):
            if vol <= 0.0:
                continue
            centre = mom / vol
            force = np.array([0.0, 0.0, self.water_density * self.gravity * vol])
            body.external_force += force
            body.external_torque += _cross3(centre - body.position, force)
            buoyancy[index] = force
        self._buoyancy_force = buoyancy

    def _apply_pair(self, i: int, j: int, geom: PairContact,
                    bodies: List[FloeBody]) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Contact force on body i and j, stored spring energy, dissipation."""
        body_a, body_b = bodies[i], bodies[j]
        normal = geom.normal
        depth = geom.depth
        point = geom.point
        ra = point - body_a.position
        rb = point - body_b.position
        va = body_a.velocity + _cross3(body_a.omega, ra)
        vb = body_b.velocity + _cross3(body_b.omega, rb)
        v_rel = vb - va
        v_n = float(v_rel @ normal)
        m_eff = body_a.mass * body_b.mass / (body_a.mass + body_b.mass)
        law = self.law
        # ``normal`` points from body a to body b, so ``v_n < 0`` is an approach and
        # the dashpot must *add* to the repulsion: f_n = k_n delta - c_n v_n.  The
        # opposite sign is negative damping: it feeds energy in on every separating
        # step and makes a confined constant-speed drive (the channel case) heat up.
        f_n = law.kn * depth - law.normal_damping(m_eff) * v_n
        if f_n < 0.0:
            f_n = 0.0
        v_t = v_rel - v_n * normal
        key = (i, j)
        xi = self.friction_state.get(key)
        if xi is None:
            xi = np.zeros(3)
        xi = xi + v_t * self.dt
        f_t = -law.kt * xi
        cap = law.mu * f_n
        mag = float(np.linalg.norm(f_t))
        if mag > cap:
            scale = cap / mag if mag > 0.0 else 0.0
            xi = xi * scale
            f_t = f_t * scale
            mag = cap
        self.friction_state[key] = xi

        force_b = f_n * normal + f_t
        force_a = -force_b
        torque_a = _cross3(ra, force_a)
        torque_b = _cross3(rb, force_b)

        roll_key = (i, j)
        omega_rel = body_b.omega - body_a.omega
        omega_rel_t = omega_rel - float(omega_rel @ normal) * normal
        spin = float(np.linalg.norm(omega_rel_t))
        if spin > 0.0 and law.mu_rolling > 0.0:
            r_eff = 0.5 * (body_a.radius + body_b.radius)
            cap_r = law.mu_rolling * r_eff * f_n
            torque_r = -cap_r * omega_rel_t / spin
            torque_a += torque_r
            torque_b -= torque_r
            self.dissipated += float(cap_r * spin * self.dt)

        spring = 0.5 * law.kn * depth * depth + 0.5 * law.kt * float(xi @ xi)
        # the dashpot only does work while the contact is loaded: counting it on a
        # separating (f_n = 0) step would create dissipation out of nothing
        diss = (law.normal_damping(m_eff) * v_n * v_n * self.dt if f_n > 0.0 else 0.0) \
            + mag * float(np.linalg.norm(v_t)) * self.dt
        self.dissipated += diss
        return force_a, force_b, spring, diss, torque_a, torque_b, depth

    def _apply_wall(self, index: int, wall: Wall, body: FloeBody,
                    vertices: Optional[np.ndarray] = None):
        if vertices is None:
            vertices = body.world_vertices()
        hit = wall_contact(vertices, wall.normal, wall.offset)
        if hit is None:
            self.wall_friction_state.pop((index, id(wall)), None)
            return None
        depth, point = hit
        ra = point - body.position
        v_point = body.velocity + _cross3(body.omega, ra)
        v_rel = v_point - wall.velocity
        n = wall.normal
        v_n = float(v_rel @ n)                     # < 0 : approaching the wall
        law = self.law
        f_n = law.kn * depth - law.normal_damping(body.mass) * v_n
        if f_n < 0.0:
            f_n = 0.0
        v_t = v_rel - v_n * n
        key = (index, id(wall))
        xi = self.wall_friction_state.get(key)
        if xi is None:
            xi = np.zeros(3)
        xi = xi + v_t * self.dt
        f_t = -law.kt * xi
        cap = law.mu * f_n
        mag = float(np.linalg.norm(f_t))
        if mag > cap:
            scale = cap / mag if mag > 0.0 else 0.0
            xi = xi * scale
            f_t = f_t * scale
            mag = cap
        self.wall_friction_state[key] = xi
        force = f_n * n + f_t
        torque = _cross3(ra, force)
        spring = 0.5 * law.kn * depth * depth + 0.5 * law.kt * float(xi @ xi)
        self.dissipated += (law.normal_damping(body.mass) * v_n * v_n * self.dt
                            if f_n > 0.0 else 0.0) \
            + mag * float(np.linalg.norm(v_t)) * self.dt
        # work done *by* the moving wall on the body over this step [J]
        self.work_external += float(force @ wall.velocity) * self.dt
        return force, torque, spring, depth

    # -- one step ----------------------------------------------------------
    def step(self) -> Dict[str, float]:
        """Advance one time step; returns the per-step diagnostics."""
        dt = self.dt
        bodies = self.bodies
        for body in bodies:
            body.external_force[:] = 0.0
            body.external_torque[:] = 0.0
            body.external_force[2] -= body.mass * self.gravity
        self._buoyancy()

        self.ke_trans = 0.0
        self.ke_rot = 0.0
        self.pe_gravity = 0.0
        for body in bodies:
            t, r = body.kinetic_energy()
            self.ke_trans += t
            self.ke_rot += r
            self.pe_gravity += body.mass * self.gravity * body.position[2]

        forces = [b.external_force.copy() for b in bodies]
        torques = [b.external_torque.copy() for b in bodies]
        spring_total = 0.0
        spring_values: List[float] = []
        deltas: List[float] = []
        self.last_pairs = []
        self.last_wall_forces = []
        self.wall_force_by_label = {}
        self.n_pair_contacts = 0
        self.n_wall_contacts = 0
        self.max_speed = 20.0
        self.max_spin = 200.0
        self.safety_clamps = 0

        # World-frame geometry is built *once* per step and shared by the pair
        # SAT tests and the wall tests: ``world_vertices`` used to be recomputed
        # inside every ``_apply_wall`` call (n_bodies x n_walls times per step),
        # which was the single largest cost of a step for wall-bounded cases.
        world = None
        if len(bodies) > 1 or self.walls:
            world = [(b.world_vertices(), b.world_normals(), b.world_edges(),
                      None if b.shape.extrusion_axis is None
                      else b.rotation @ b.shape.extrusion_axis) for b in bodies]
        if len(bodies) > 1:
            for (i, j) in self._broad_phase(np.zeros(len(bodies))):
                va, na, ea, xa = world[i]
                vb, nb, eb, xb = world[j]
                if not (np.all(va.min(0) <= vb.max(0) + 1e-9) and np.all(vb.min(0) <= va.max(0) + 1e-9)):
                    continue
                geom = sat_contact(va, na, ea, vb, nb, eb,
                                   extrusion_a=xa, extrusion_b=xb)
                if geom.gap > 0.0:
                    self.friction_state.pop((i, j), None)
                    continue
                fa, fb, spring, _diss, ta, tb, depth = self._apply_pair(i, j, geom, bodies)
                forces[i] += fa
                forces[j] += fb
                torques[i] += ta
                torques[j] += tb
                spring_total += spring
                spring_values.append(0.5 * self.law.kn * depth * depth)
                deltas.append(depth)
                self.last_pairs.append(geom)
                self.n_pair_contacts += 1

        for index, body in enumerate(bodies):
            vertices = world[index][0] if world is not None else body.world_vertices()
            for wall in self.walls:
                # wall broad phase: a half-space cannot touch a body whose bounding
                # sphere stays on the inside (``min_v n.v >= n.c - R_b``).  The
                # friction accumulator is dropped on the skip exactly as it is on
                # a "tested but not penetrating" wall, so no stale tangential
                # spring can be revived when the body comes back.
                if float(wall.normal @ body.position) - body.shape.bounding_radius >= wall.offset:
                    self.wall_friction_state.pop((index, id(wall)), None)
                    continue
                result = self._apply_wall(index, wall, body, vertices)
                if result is None:
                    continue
                force, torque, spring, depth = result
                forces[index] += force
                torques[index] += torque
                spring_total += spring
                spring_values.append(0.5 * self.law.kn * depth * depth)
                deltas.append(depth)
                self.wall_force_by_label[wall.label] = \
                    self.wall_force_by_label.get(wall.label, np.zeros(3)) + force
                self.n_wall_contacts += 1
                if wall.label != "floor":
                    self.last_wall_forces.append((index, id(wall), force.copy()))

        self.pe_spring = spring_total
        self.spring_pairs = np.asarray(spring_values, dtype=np.float64)
        self.spring_deltas = np.asarray(deltas, dtype=np.float64)

        # ---- hydrodynamic drag (optional linear closure) -----------------
        drag_forces = [np.zeros(3) for _ in bodies]
        drag_torques = [np.zeros(3) for _ in bodies]
        if self.linear_drag > 0.0 or self.angular_drag > 0.0:
            for index, body in enumerate(bodies):
                if self.linear_drag > 0.0:
                    drag_force = -self.linear_drag * body.velocity
                    forces[index] += drag_force
                    drag_forces[index] = drag_force
                if self.angular_drag > 0.0:
                    drag_torque = -self.angular_drag * body.omega
                    torques[index] += drag_torque
                    drag_torques[index] = drag_torque

        # ---- integration (leapfrog in velocity = kick-drift) -------------
        # velocities before the kick, so the hydrodynamic ledger below can use the
        # midpoint velocity: dE = F . (v_n + v_{n+1})/2 dt is the exact kinetic
        # energy increment of a kick, F . v_n dt is not.
        v_before = [body.velocity.copy() for body in bodies]
        w_before = [body.omega.copy() for body in bodies]
        for body, force, torque in zip(bodies, forces, torques):
            inv_inertia = body.inverse_inertia_world()
            inertia = body.inertia_world()
            alpha = inv_inertia @ (torque - _cross3(body.omega, inertia @ body.omega))
            body.velocity = body.velocity + dt * force / body.mass
            body.omega = body.omega + dt * alpha
            # numerical guard: a floe cannot physically exceed these magnitudes in
            # a channel; clipping keeps a marginal time step from turning into a
            # NaN trajectory and the event is counted in ``safety_clamps``
            speed = float(np.linalg.norm(body.velocity))
            if speed > self.max_speed:
                body.velocity *= self.max_speed / speed
                self.safety_clamps += 1
            spin = float(np.linalg.norm(body.omega))
            if spin > self.max_spin:
                body.omega *= self.max_spin / spin
                self.safety_clamps += 1
            body.position = body.position + dt * body.velocity
            body.rotate(_expm_skew(body.omega, dt))
            if self.constrain_to_plane:
                _project_to_plane(body)
        # ---- hydrostatic / hydrodynamic energy ledger --------------------
        # Book work (J) and dissipation (J) over the step just integrated, with the
        # midpoint velocity of the kick.  Both closures are centred on the body, so
        # a single midpoint velocity is exact for them.
        if len(self._buoyancy_force) != len(bodies):
            self._buoyancy_force = np.zeros((len(bodies), 3))
        for index, body in enumerate(bodies):
            v_mid = 0.5 * (v_before[index] + body.velocity)
            w_mid = 0.5 * (w_before[index] + body.omega)
            self.work_external += float(self._buoyancy_force[index] @ v_mid) * dt
            if self.linear_drag > 0.0:
                self.dissipated += float(-drag_forces[index] @ v_mid) * dt
            if self.angular_drag > 0.0:
                self.dissipated += float(-drag_torques[index] @ w_mid) * dt
        self.time += dt

        if self.record:
            self.history.append({
                "t": self.time,
                "ke_trans": self.ke_trans,
                "ke_rot": self.ke_rot,
                "pe_gravity": self.pe_gravity,
                "pe_spring": self.pe_spring,
                "dissipated": self.dissipated,
                "work_external": self.work_external,
                "spring_values": self.spring_pairs.copy(),
                "spring_deltas": self.spring_deltas.copy(),
                "velocity": np.array([b.velocity.copy() for b in bodies]),
                "position": np.array([b.position.copy() for b in bodies]),
                "omega": np.array([b.omega.copy() for b in bodies]),
            })
        return {
            "time": self.time,
            "ke": self.ke_trans + self.ke_rot,
            "pe_spring": self.pe_spring,
            "dissipated": self.dissipated,
            "energy_residual": self.energy_residual(),
        }

    def run(self, n_steps: int, *, callback=None) -> None:
        for _ in range(int(n_steps)):
            self.step()
            if callback is not None:
                callback(self)