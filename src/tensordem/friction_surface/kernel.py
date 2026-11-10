"""Vectorised 3-D soft-sphere DEM kernel for the friction-contact verification surface.

Why a separate kernel
---------------------
``tensordem.granular3d_core`` carries the per-pair tangential/rolling state in
Python dicts and loops over the active pairs every step, and it has **no
tangential/rolling damping** (the declared ``F_t = min(mu F_n, k_t delta_t) +
c_t v_t`` needs ``c_t``).  The friction-contact verification surface needs

* the declared two-term tangential law with ``c_t`` and an explicit
  cone-projection switch,
* rolling resistance with the same damping/cap structure,
* friction *and* rolling resistance on walls (a pile of ideal spheres cannot
  carry a slope without wall rolling resistance either),
* enough speed to run repose / shear-box / packing campaigns (thousands of
  particles, tens of thousands of steps),

so the per-pair state is kept in numpy arrays keyed by the stable integer
``i*n + j`` and carried between steps with ``searchsorted`` (fully vectorised,
O(M log M) per step, no per-pair Python loop).  The contact laws themselves live
in :mod:`tensordem.friction_surface.laws` and are shared with the analytic
anchors, so the anchored law and the simulated law are literally the same code.

Conventions
-----------
* ``n`` points from body *i* to body *j*: ``n = (x_j - x_i)/|x_j - x_i|``.
* Normal force is repulsive only (no tension).
* Tangential displacement is the accumulated *surface* tangential relative
  displacement of the contact point (Cundall-Strack with rotation).
* All quantities SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .laws import damping_ratio, normal_force, rolling_torque, tangential_force

__all__ = [
    "ContactModel",
    "PairState",
    "Wall",
    "PlaneWall",
    "CylinderWall",
    "SplitBox",
    "kinetic_energy",
    "build_pairs",
    "pair_forces",
    "apply_walls",
    "integrate",
]


@dataclass(frozen=True)
class ContactModel:
    """Monodisperse-sphere material plus the control switches of the laws."""

    radius: float = 5.0e-3
    density: float = 2500.0
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    kr_ratio: float = 0.1          # k_r = kr_ratio * k_t * R^2
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    ct_ratio: float = 0.0          # c_t = ct_ratio * 2 sqrt(k_t m_eff)
    cr_ratio: float = 0.0          # c_r = cr_ratio * 2 sqrt(k_r I_eff)
    cap_total: bool = True         # project F_t / M_r back onto the friction cone
    gravity: float = 9.81

    # ---- derived ---------------------------------------------------------
    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def kr(self) -> float:
        return self.kr_ratio * self.kt * self.radius ** 2

    @property
    def cn(self) -> float:
        """Normal dashpot for a pair (effective mass m/2)."""
        m_eff = 0.5 * self.mass
        return 2.0 * damping_ratio(self.restitution) * math.sqrt(self.kn * m_eff)

    @property
    def ct(self) -> float:
        m_eff = 0.5 * self.mass
        return 2.0 * self.ct_ratio * math.sqrt(self.kt * m_eff)

    @property
    def cr(self) -> float:
        i_eff = 0.5 * self.inertia
        return 2.0 * self.cr_ratio * math.sqrt(self.kr * i_eff)

    @property
    def timestep(self) -> float:
        """20 steps per period of the stiffest (normal) contact oscillation."""
        m_eff = 0.5 * self.mass
        return 2.0 * math.pi * math.sqrt(m_eff / self.kn) / 20.0


class PairState:
    """Per-pair tangential displacement and rolling rotation, keyed by ``i*n+j``."""

    __slots__ = ("keys", "tang", "roll")

    def __init__(self) -> None:
        self.keys = np.zeros(0, dtype=np.int64)
        self.tang = np.zeros((0, 3))
        self.roll = np.zeros((0, 3))

    def carry(self, new_keys: np.ndarray) -> None:
        """Carry the state of persisting pairs onto ``new_keys`` (ascending)."""
        old_keys = self.keys
        if new_keys.size == 0:
            self.keys = new_keys
            self.tang = np.zeros((0, 3))
            self.roll = np.zeros((0, 3))
            return
        if old_keys.size == 0:
            self.keys = new_keys
            self.tang = np.zeros((new_keys.size, 3))
            self.roll = np.zeros((new_keys.size, 3))
            return
        idx = np.searchsorted(old_keys, new_keys)
        np.clip(idx, 0, old_keys.size - 1, out=idx)
        hit = old_keys[idx] == new_keys
        tang = np.where(hit[:, None], self.tang[idx], 0.0)
        roll = np.where(hit[:, None], self.roll[idx], 0.0)
        self.keys = new_keys
        self.tang = tang
        self.roll = roll



def _scatter_add(index: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Fast ``np.add.at`` replacement for (M,3) increments (bincount per axis)."""
    out = np.empty((n, 3))
    for c in range(3):
        out[:, c] = np.bincount(index, weights=values[:, c], minlength=n)
    return out

def build_pairs(pos: np.ndarray, radius: float, margin: float = 0.0,
                n_keys: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate pairs within ``2r + margin``; keys ``i*n_keys + j`` ascending.

    ``n_keys`` must be a *stable* upper bound on the particle index when the
    active particle set grows (deposition runs).  Using the current count
    ``len(pos)`` silently aliases keys across steps -- e.g. ``(1,5)`` at n=10 and
    ``(0,15)`` at n=20 both map to 15 -- so a new contact inherits an unrelated
    pair's tangential displacement and the run is injected with spurious force
    (measured: particles thrown 1-3 m and a "heap" that never forms).
    """
    if n_keys is None:
        n_keys = len(pos)
    n = len(pos)
    if n < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    tree = cKDTree(pos)
    pairs = tree.query_pairs(2.0 * radius + margin, output_type="ndarray")
    if pairs.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    iu = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    ju = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    keys = iu * n_keys + ju
    order = np.argsort(keys)
    return iu[order], ju[order], keys[order]


def pair_forces(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
                iu: np.ndarray, ju: np.ndarray, ps: PairState,
                model: ContactModel, dt: float, *, contacts: dict | None = None,
                n_keys: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Particle-particle contact forces/torques; updates ``ps`` in place.

    If ``contacts`` is a dict it is filled with the *active* contacts
    (``i``, ``j``, unit normal, normal force, tangential force vector) so that a
    macroscopic stress can be measured across an arbitrary plane without
    re-computing the forces.
    """
    n = len(pos)
    if n_keys is None:
        n_keys = n
    force = np.zeros((n, 3))
    torque = np.zeros((n, 3))
    if contacts is not None:
        contacts.clear()
    if iu.size == 0:
        return force, torque

    r = model.radius
    dvec = pos[ju] - pos[iu]
    dist = np.linalg.norm(dvec, axis=1)
    overlap = 2.0 * r - dist
    active = overlap > 0.0
    if not active.any():
        ps.carry(np.zeros(0, dtype=np.int64))
        return force, torque

    iu_a, ju_a = iu[active], ju[active]
    dvec_a, dist_a, overlap_a = dvec[active], dist[active], overlap[active]
    nvec = dvec_a / np.maximum(dist_a, 1e-300)[:, None]

    v_rel = vel[ju_a] - vel[iu_a]
    v_n = (v_rel * nvec).sum(axis=1)
    f_n = normal_force(overlap_a, v_n, model.kn, model.cn)

    # surface relative velocity at the contact point (rotation included)
    surf_i = np.cross(omega[iu_a], nvec * r)
    surf_j = np.cross(omega[ju_a], -nvec * r)
    v_t = (v_rel - v_n[:, None] * nvec) + (surf_j - surf_i)

    ps.carry(iu_a * n_keys + ju_a)
    # incremental tangential displacement (rotation of the pair frame ignored:
    # standard Cundall-Strack "no memory rotation" variant, documented)
    ps.tang = ps.tang + v_t * dt
    f_t = tangential_force(ps.tang, v_t, f_n, kt=model.kt, mu=model.mu,
                           ct=model.ct, cap_total=model.cap_total)

    f_total = f_n[:, None] * nvec + f_t
    if contacts is not None:
        contacts["i"] = iu_a
        contacts["j"] = ju_a
        contacts["normal"] = nvec
        contacts["f_n"] = f_n
        contacts["f_t"] = f_t
        contacts["f_total"] = f_total
    force = _scatter_add(iu_a, -f_total, n) + _scatter_add(ju_a, f_total, n)
    t_i = np.cross(-nvec * r, f_t)
    torque = _scatter_add(iu_a, t_i, n) + _scatter_add(ju_a, t_i, n)

    if model.mu_rolling > 0.0:
        spin = omega[ju_a] - omega[iu_a]
        ps.roll = ps.roll + spin * dt
        m_r = rolling_torque(ps.roll, spin, f_n, kr=model.kr,
                             mu_r=model.mu_rolling, r_eff=r,
                             cr=model.cr, cap_total=model.cap_total)
        # the resisting couple must *reduce* the relative spin: with
        # theta_r = int(omega_j - omega_i) and m_r = -kr theta_r the torque on j is
        # m_r and on i is -m_r.  Applying them the other way round is an
        # angular anti-damper: the pair spins up without bound and the friction
        # converts that spin into translational motion (measured: single
        # particles thrown metres away, "heaps" that never form).
        torque += _scatter_add(iu_a, -m_r, n) + _scatter_add(ju_a, m_r, n)
    return force, torque


# ---------------------------------------------------------------------------
# Walls
# ---------------------------------------------------------------------------

class Wall:
    """Base class: a wall keeps its own per-particle tangential/rolling state.

    ``mu``/``mu_rolling`` default to the particle values (``None`` means inherit).
    Every wall records ``traction`` = total x-force it applies, which is how the
    direct-shear box measures the shear force on its half-box.
    """

    def __init__(self, mu: float | None = None, mu_rolling: float | None = None,
                 kn_factor: float = 1.0) -> None:
        self.mu = mu
        self.mu_rolling = mu_rolling
        self.kn_factor = kn_factor
        self._tang: np.ndarray | None = None
        self._roll: np.ndarray | None = None
        self.traction = np.zeros(3)

    def _ensure(self, n: int) -> None:
        if self._tang is None:
            self._tang = np.zeros((n, 3))
            self._roll = np.zeros((n, 3))
        elif self._tang.shape[0] < n:
            grow = n - self._tang.shape[0]
            self._tang = np.vstack([self._tang, np.zeros((grow, 3))])
            self._roll = np.vstack([self._roll, np.zeros((grow, 3))])

    def contact_geometry(self, pos: np.ndarray, radius: float, t: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(mask, inward_normal, overlap)`` for the current state."""
        raise NotImplementedError

    def surface_velocity(self, pos: np.ndarray, t: float) -> np.ndarray:
        """Velocity of the wall material at the particle positions."""
        return np.zeros_like(pos)

    def apply(self, pos, vel, omega, model, dt, t):
        mask, normal, overlap = self.contact_geometry(pos, model.radius, t)
        n = len(pos)
        force = np.zeros((n, 3))
        torque = np.zeros((n, 3))
        self.traction = np.zeros(3)
        self._ensure(n)
        self._tang[~mask] = 0.0
        self._roll[~mask] = 0.0
        if not mask.any():
            return force, torque
        idx = np.where(mask)[0]
        nvec = normal[idx]
        r = model.radius
        kn = model.kn * self.kn_factor
        v_wall = self.surface_velocity(pos, t)[idx]
        v_rel = vel[idx] - v_wall
        v_n = (v_rel * nvec).sum(axis=1)
        f_n = normal_force(overlap[idx], v_n, kn, model.cn)
        surf = np.cross(omega[idx], -nvec * r)
        v_t = (v_rel - v_n[:, None] * nvec) + surf
        self._tang[idx] = self._tang[idx] + v_t * dt
        mu = model.mu if self.mu is None else self.mu
        f_t = tangential_force(self._tang[idx], v_t, f_n,
                               kt=model.kt * self.kn_factor, mu=mu,
                               ct=model.ct, cap_total=model.cap_total)
        f_total = f_n[:, None] * nvec + f_t
        force[idx] = f_total
        torque[idx] = np.cross(-nvec * r, f_t)
        mu_r = model.mu_rolling if self.mu_rolling is None else self.mu_rolling
        if mu_r > 0.0:
            self._roll[idx] = self._roll[idx] + omega[idx] * dt
            torque[idx] += rolling_torque(self._roll[idx], omega[idx], f_n,
                                          kr=model.kr, mu_r=mu_r, r_eff=r,
                                          cr=model.cr,
                                          cap_total=model.cap_total)
        self.traction = f_total.sum(axis=0)
        return force, torque


class PlaneWall(Wall):
    """Half space ``{x : (x - point) . normal < 0}`` is wall material."""

    def __init__(self, point, normal, velocity=(0.0, 0.0, 0.0), **kwargs) -> None:
        super().__init__(**kwargs)
        nv = np.asarray(normal, dtype=float)
        self.point = np.asarray(point, dtype=float)
        self.normal = nv / np.linalg.norm(nv)
        self.velocity = np.asarray(velocity, dtype=float)

    def move_to(self, z: float) -> None:
        """Translate the plane along its unit normal to signed offset ``z``."""
        self.point = self.point + self.normal * (z - float(self.point @ self.normal))

    def surface_velocity(self, pos, t):
        return np.tile(self.velocity, (len(pos), 1))

    def contact_geometry(self, pos, radius, t):
        signed = (pos - self.point) @ self.normal
        overlap = radius - signed
        mask = overlap > 0.0
        normal = np.tile(self.normal, (len(pos), 1))
        normal[~mask] = 0.0
        return mask, normal, np.maximum(overlap, 0.0)


class CylinderWall(Wall):
    """Bore of a cylinder around the z axis; optionally present only above z_floor.

    ``z_floor`` may be a float or a callable ``t -> z``; the lifting-cylinder
    repose experiment uses ``z_floor`` rising linearly in time.
    """

    def __init__(self, center_xy, radius: float, z_floor=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.center = np.asarray(center_xy, dtype=float)
        self.radius = float(radius)
        self.z_floor = z_floor

    def contact_geometry(self, pos, radius, t):
        d = pos[:, :2] - self.center
        rho = np.linalg.norm(d, axis=1)
        overlap = rho + radius - self.radius
        mask = overlap > 0.0
        if self.z_floor is not None:
            zf = self.z_floor(t) if callable(self.z_floor) else self.z_floor
            mask &= pos[:, 2] > zf
        normal = np.zeros_like(pos)
        with np.errstate(invalid="ignore", divide="ignore"):
            normal[:, :2] = -d / np.maximum(rho, 1e-300)[:, None]
        normal[~mask] = 0.0
        return mask, normal, np.maximum(overlap, 0.0)


class SplitBox(Wall):
    """Direct-shear split box: the upper half is displaced by ``u(t)`` in +x.

    Four vertical planes plus the floor, handled in one call so the half-box
    traction is a single force sum.  ``displacement(t)`` is the imposed
    horizontal offset of the upper half (the shear-box lid motion).
    """

    def __init__(self, width: float, depth: float, z_split: float,
                 displacement=lambda t: 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.width = float(width)
        self.depth = float(depth)
        self.z_split = float(z_split)
        self.displacement = displacement
        self._floor = PlaneWall((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), **kwargs)

    def speed(self, t: float, eps: float = 1e-6) -> float:
        return (self.displacement(t + eps) - self.displacement(t - eps)) / (2 * eps)

    def contact_geometry(self, pos, radius, t):
        u = self.displacement(t)
        upper = pos[:, 2] >= self.z_split
        xlo = np.where(upper, u, 0.0) + radius
        xhi = np.where(upper, self.width + u, self.width) - radius
        ylo = radius
        yhi = self.depth - radius
        overlap = np.maximum(xlo - pos[:, 0], 0.0)
        normal = np.zeros_like(pos)
        normal[:, 0] = 1.0
        for pen, vec in ((np.maximum(pos[:, 0] - xhi, 0.0), (-1.0, 0.0, 0.0)),
                         (np.maximum(ylo - pos[:, 1], 0.0), (0.0, 1.0, 0.0)),
                         (np.maximum(pos[:, 1] - yhi, 0.0), (0.0, -1.0, 0.0))):
            better = pen > overlap
            overlap = np.where(better, pen, overlap)
            normal[better] = vec
        # floor
        floor_overlap = radius - pos[:, 2]
        floor_better = floor_overlap > overlap
        overlap = np.where(floor_better, floor_overlap, overlap)
        normal[floor_better] = (0.0, 0.0, 1.0)
        mask = overlap > 0.0
        normal[~mask] = 0.0
        return mask, normal, np.maximum(overlap, 0.0)

    def surface_velocity(self, pos, t):
        v = np.zeros_like(pos)
        v[pos[:, 2] >= self.z_split, 0] = self.speed(t)
        return v


def apply_walls(walls, pos, vel, omega, model, dt, t):
    force = np.zeros_like(pos)
    torque = np.zeros_like(pos)
    for wall in walls:
        f, tq = wall.apply(pos, vel, omega, model, dt, t)
        force += f
        torque += tq
    return force, torque


def integrate(pos, vel, omega, force, torque, model, dt):
    """Velocity-Verlet-style DEM update (force evaluated once per step)."""
    acc = force / model.mass
    ang = torque / model.inertia
    pos += vel * dt + 0.5 * acc * dt ** 2
    vel += acc * dt
    omega += ang * dt
    return pos, vel, omega


def kinetic_energy(vel, omega, model) -> float:
    return float(0.5 * model.mass * np.sum(vel ** 2)
                 + 0.5 * model.inertia * np.sum(omega ** 2))