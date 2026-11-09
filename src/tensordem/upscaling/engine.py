"""Vectorised 3-D DEM time integrator used by the upscaling campaign.

The contact *physics* is identical to :mod:`tensordem.granular3d_core`
(linear spring-dashpot normal, Cundall–Strack incremental tangential spring
saturated at ``mu F_n``, isotropic rolling spring saturated at
``mu_r R F_n``).  The kernel here is re-implemented with vectorised
per-pair *history remapping* (sorted integer keys + ``searchsorted``)
instead of a Python dict loop, because the coarse-graining campaign runs
~10^4 steps x 24 configurations and the dict loop dominates the runtime.
Consistency with the reference kernel is asserted by
``tests/test_upscaling_scaling.py::test_pair_kernel_matches_reference``
to machine precision.

Bodies: monodisperse spheres (single radius), walls are analytic surfaces
(planes and a rotating cylinder) with the same friction/restitution law as
particle-particle contacts.

All quantities SI (m, kg, s, N, rad).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised 3-D cross product (faster than np.cross)."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def _accum(out: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """Scatter-add rows of ``vals`` into ``out[idx]`` (bincount; ~10x add.at)."""
    if len(idx) == 0:
        return
    m = out.shape[0]
    for c in range(3):
        out[:, c] += np.bincount(idx, weights=vals[:, c], minlength=m)


# ---------------------------------------------------------------------------
# Pair history store (vectorised replacement of the dict-based ContactState)
# ---------------------------------------------------------------------------


class PairHistory:
    """Per-pair tangential / rolling displacement, keyed by ``i*n + j`` (i<j)."""

    __slots__ = ("keys", "tang", "roll")

    def __init__(self) -> None:
        self.keys = np.zeros(0, dtype=np.int64)
        self.tang = np.zeros((0, 3), dtype=np.float64)
        self.roll = np.zeros((0, 3), dtype=np.float64)

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (tang, roll) arrays aligned with the (sorted) ``keys``."""
        m = len(keys)
        tang = np.zeros((m, 3), dtype=np.float64)
        roll = np.zeros((m, 3), dtype=np.float64)
        if len(self.keys) and m:
            idx = np.searchsorted(self.keys, keys)
            idx = np.clip(idx, 0, len(self.keys) - 1)
            hit = self.keys[idx] == keys
            tang[hit] = self.tang[idx[hit]]
            roll[hit] = self.roll[idx[hit]]
        return tang, roll

    def commit(self, keys: np.ndarray, tang: np.ndarray, roll: np.ndarray) -> None:
        self.keys = keys
        self.tang = tang
        self.roll = roll


# ---------------------------------------------------------------------------
# Pair (particle-particle) contacts
# ---------------------------------------------------------------------------


def build_pairs(pos: np.ndarray, radius: float,
                box: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Candidate pairs within 2R (``box`` = periodic lengths, or None)."""
    n = len(pos)
    if n < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    tree = cKDTree(pos, boxsize=box)
    pairs = tree.query_pairs(2.0 * radius, output_type="ndarray")
    if len(pairs) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    iu = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    ju = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    return iu, ju


def pair_forces(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
                iu: np.ndarray, ju: np.ndarray,
                mat, dt: float, history: PairHistory,
                box: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """Particle-particle contact forces.

    Returns ``(force, torque, elastic_energy)``; updates ``history``.
    """
    n = len(pos)
    force = np.zeros((n, 3), dtype=np.float64)
    torque = np.zeros((n, 3), dtype=np.float64)
    if len(iu) == 0:
        history.commit(np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3)))
        return force, torque, 0.0

    dvec = pos[ju] - pos[iu]
    if box is not None:
        # minimum image (only the periodic axes have box > 0)
        with np.errstate(invalid="ignore"):
            shift = np.where(box > 0.0, np.rint(dvec / np.where(box > 0.0, box, 1.0)), 0.0)
        dvec = dvec - shift * box
    dist = np.linalg.norm(dvec, axis=1)
    overlap = 2.0 * mat.radius - dist
    active = overlap > 0.0
    if not active.any():
        history.commit(np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3)))
        return force, torque, 0.0

    iu_a, ju_a = iu[active], ju[active]
    dvec_a, overlap_a = dvec[active], overlap[active]
    dist_a = np.maximum(dist[active], 1e-30)
    nvec = dvec_a / dist_a[:, None]

    r = mat.radius
    kn, kt, kr = mat.kn, mat.kt, mat.kr
    mu, mu_r = mat.mu, mat.mu_rolling
    gamma = mat.gamma_pair

    v_rel = vel[ju_a] - vel[iu_a]
    v_n = (v_rel * nvec).sum(axis=1)
    f_n = np.maximum(kn * overlap_a - gamma * v_n, 0.0)

    keys = iu_a * n + ju_a
    order = np.argsort(keys)
    keys = keys[order]
    iu_s, ju_s, nvec_s, f_n_s = iu_a[order], ju_a[order], nvec[order], f_n[order]
    v_rel_s = v_rel[order]
    v_n_s = v_n[order]

    tang, roll = history.lookup(keys)

    # ---- tangential (Cundall-Strack) -----------------------------------
    if mu > 0.0:
        v_t = v_rel_s - v_n_s[:, None] * nvec_s
        surf_i = _cross(omega[iu_s], nvec_s * r)
        surf_j = _cross(omega[ju_s], -nvec_s * r)
        v_t_surf = v_t + (surf_j - surf_i)
        tang = tang + v_t_surf * dt
        tnorm = np.linalg.norm(tang, axis=1)
        cap = mu * f_n_s / kt
        scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
        tang = tang * scale[:, None]
        f_t = -kt * tang
    else:
        f_t = np.zeros_like(nvec_s)

    f_total = f_n_s[:, None] * nvec_s + f_t
    _accum(force, iu_s, -f_total)
    _accum(force, ju_s, f_total)

    if mu > 0.0:
        t_contact = _cross(-nvec_s * r, f_t)
        _accum(torque, iu_s, t_contact)
        _accum(torque, ju_s, t_contact)

    # ---- rolling resistance -------------------------------------------
    if mu_r > 0.0:
        spin = omega[ju_s] - omega[iu_s]
        roll = roll + spin * dt
        rnorm = np.linalg.norm(roll, axis=1)
        cap_r = (mu_r * r * f_n_s) / kr
        scale_r = np.where(rnorm > cap_r, cap_r / np.maximum(rnorm, 1e-30), 1.0)
        roll = roll * scale_r[:, None]
        torque_r = -kr * roll
        _accum(torque, iu_s, -torque_r)
        _accum(torque, ju_s, torque_r)
    else:
        roll = np.zeros_like(roll)

    history.commit(keys, tang, roll)
    elastic = float(0.5 * kn * np.sum(overlap_a ** 2))
    return force, torque, elastic


# ---------------------------------------------------------------------------
# Walls
# ---------------------------------------------------------------------------


class PlaneWall:
    """Infinite plane wall; ``normal`` points into the particle domain."""

    def __init__(self, point: Sequence[float], normal: Sequence[float],
                 velocity: Sequence[float] = (0.0, 0.0, 0.0),
                 name: str = "plane") -> None:
        self.point = np.asarray(point, dtype=np.float64)
        n = np.asarray(normal, dtype=np.float64)
        self.normal = n / np.linalg.norm(n)
        self.velocity = velocity if callable(velocity) else np.asarray(velocity, dtype=np.float64)
        self.name = name
        self.n = 0  # particle count, set by Sim

    def gap(self, pos: np.ndarray, radius: float) -> np.ndarray:
        return (pos - self.point) @ self.normal - radius

    def normal_at(self, pos: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.normal, pos.shape).copy()

    def surface_velocity(self, contact_points: np.ndarray) -> np.ndarray:
        if callable(self.velocity):
            return np.asarray(self.velocity(contact_points), dtype=np.float64)
        return np.broadcast_to(self.velocity, contact_points.shape).copy()


class CylinderWall:
    """Cylinder about the y axis, particles *inside*; ``omega`` about +y."""

    def __init__(self, center: Sequence[float], radius: float, omega: float = 0.0,
                 name: str = "cylinder") -> None:
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.omega = float(omega)
        self.name = name

    def gap(self, pos: np.ndarray, radius: float) -> np.ndarray:
        d = np.hypot(pos[:, 0] - self.center[0], pos[:, 2] - self.center[2])
        return (self.radius - d) - radius

    def normal_at(self, pos: np.ndarray) -> np.ndarray:
        d = np.hypot(pos[:, 0] - self.center[0], pos[:, 2] - self.center[2])
        d = np.maximum(d, 1e-30)
        n = np.zeros_like(pos)
        n[:, 0] = -(pos[:, 0] - self.center[0]) / d
        n[:, 2] = -(pos[:, 2] - self.center[2]) / d
        return n

    def surface_velocity(self, contact_points: np.ndarray) -> np.ndarray:
        w = np.array([0.0, self.omega, 0.0])
        rel = contact_points - self.center
        return _cross(np.broadcast_to(w, rel.shape), rel)


def wall_forces(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
                walls: Sequence[object], mat, dt: float,
                wall_history: dict[str, PairHistory]
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """Sum contact forces/torques from all analytic walls."""
    n = len(pos)
    force = np.zeros((n, 3), dtype=np.float64)
    torque = np.zeros((n, 3), dtype=np.float64)
    elastic = 0.0
    r = mat.radius
    for widx, wall in enumerate(walls):
        gap = wall.gap(pos, r)
        active = gap < 0.0
        if not active.any():
            wall_history[wall.name].commit(np.zeros(0, np.int64),
                                           np.zeros((0, 3)), np.zeros((0, 3)))
            continue
        idx = np.nonzero(active)[0]
        nvec = wall.normal_at(pos[idx])
        overlap = -gap[idx]
        cpoint = pos[idx] + nvec * r
        v_wall = wall.surface_velocity(cpoint)
        v_rel = vel[idx] - v_wall
        v_n = (v_rel * nvec).sum(axis=1)
        f_n = np.maximum(mat.kn * overlap - mat.gamma_wall * v_n, 0.0)

        keys = widx * n + idx
        order = np.argsort(keys)
        keys, idx_s = keys[order], idx[order]
        nvec_s, f_n_s = nvec[order], f_n[order]
        v_rel_s, v_n_s = v_rel[order], v_n[order]

        hist = wall_history[wall.name]
        tang, _ = hist.lookup(keys)
        if mat.mu > 0.0:
            v_t = v_rel_s - v_n_s[:, None] * nvec_s
            surf = _cross(omega[idx_s], nvec_s * r)
            v_t_surf = v_t - surf
            tang = tang + v_t_surf * dt
            tnorm = np.linalg.norm(tang, axis=1)
            cap = mat.mu * f_n_s / mat.kt
            scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
            tang = tang * scale[:, None]
            f_t = -mat.kt * tang
        else:
            f_t = np.zeros_like(nvec_s)
        f_total = f_n_s[:, None] * nvec_s + f_t
        _accum(force, idx_s, f_total)
        _accum(torque, idx_s, _cross(-nvec_s * r, f_t))
        hist.commit(keys, tang, np.zeros_like(tang))
        elastic += float(0.5 * mat.kn * np.sum(overlap ** 2))
    return force, torque, elastic


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class Sim:
    """Monodisperse sphere DEM in a box (optionally periodic) with walls."""

    pos: np.ndarray
    mat: object
    box: np.ndarray | None = None            # periodic lengths (0 = non-periodic)
    walls: list = field(default_factory=list)
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    dt: float = 1e-4
    vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    omega: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    t: float = 0.0
    max_velocity: float = 0.0
    elastic_energy: float = 0.0

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float64)
        n = len(self.pos)
        if self.vel.shape != (n, 3):
            self.vel = np.zeros((n, 3), dtype=np.float64)
        if self.omega.shape != (n, 3):
            self.omega = np.zeros((n, 3), dtype=np.float64)
        self.gravity = np.asarray(self.gravity, dtype=np.float64)
        self.pair_history = PairHistory()
        self.wall_history = {w.name: PairHistory() for w in self.walls}
        self._box = np.asarray(self.box if self.box is not None else np.zeros(3),
                               dtype=np.float64)
        self._pair_list = None
        self._pair_age = 0
        self._pair_interval = 8      # rebuild the Verlet list every 8 steps
        self._skin = 0.5             # skin = 0.5 R (rebuild before contacts enter)

    # -- diagnostics -----------------------------------------------------
    def kinetic_energy(self) -> float:
        m, I = self.mat.mass, self.mat.inertia
        return float(0.5 * m * np.sum(self.vel ** 2) + 0.5 * I * np.sum(self.omega ** 2))

    def potential_energy(self) -> float:
        return float(-self.mat.mass * np.sum(self.gravity * self.pos))

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy() + self.elastic_energy

    # -- stepping --------------------------------------------------------
    def _update_pair_list(self) -> None:
        """Rebuild the candidate-pair list with a Verlet skin every N steps."""
        self._pair_age += 1
        if self._pair_list is not None and self._pair_age < self._pair_interval:
            return
        skin = self._skin * self.mat.radius
        box = self._box if np.any(self._box > 0) else None
        iu, ju = build_pairs(self.pos, self.mat.radius + 0.5 * skin, box=box)
        self._pair_list = (iu, ju)
        self._pair_age = 0

    def step(self) -> None:
        self._update_pair_list()
        iu, ju = self._pair_list
        f_pair, t_pair, e_pair = pair_forces(self.pos, self.vel, self.omega,
                                             iu, ju, self.mat, self.dt,
                                             self.pair_history,
                                             box=self._box if np.any(self._box > 0) else None)
        f_wall, t_wall, e_wall = wall_forces(self.pos, self.vel, self.omega,
                                             self.walls, self.mat, self.dt,
                                             self.wall_history)
        force = f_pair + f_wall + self.mat.mass * self.gravity
        torque = t_pair + t_wall

        self.vel += force / self.mat.mass * self.dt
        self.omega += torque / self.mat.inertia * self.dt
        self.pos += self.vel * self.dt
        if np.any(self._box > 0):
            periodic = self._box > 0
            self.pos[:, periodic] = np.mod(self.pos[:, periodic], self._box[periodic])
        self.elastic_energy = e_pair + e_wall
        self.t += self.dt
        vmax = float(np.max(np.linalg.norm(self.vel, axis=1))) if len(self.vel) else 0.0
        self.max_velocity = max(self.max_velocity, vmax)

    def run(self, n_steps: int, on_step: Callable[["Sim"], None] | None = None) -> None:
        for _ in range(int(n_steps)):
            self.step()
            if on_step is not None:
                on_step(self)


# ---------------------------------------------------------------------------
# Packing helpers (geometry-scaled HCP fill, reused by the campaign cases)
# ---------------------------------------------------------------------------


def hcp_fill_box(radius: float, lx: float, ly: float, lz: float,
                 seed: int = 1, jitter: float = 0.02,
                 z_offset: float = 0.0) -> np.ndarray:
    """HCP (ABAB) fill of a box ``lx x ly x lz`` with spheres of ``radius``.

    Proper hexagonal close packing: in-plane triangular lattice with lattice
    constant ``2R`` and layers spaced ``2R sqrt(2/3)`` with the HCP lateral
    offset, so nearest neighbours exactly touch (no built-in overlap; only the
    ``jitter`` introduces a small pre-stress, which matters for zero-gravity
    protocols).
    """
    a = 2.0 * radius
    dz = a * math.sqrt(2.0 / 3.0)
    dy = a * math.sqrt(3.0) / 2.0
    pts = []
    nz = int(max(1, math.floor((lz - 2.0 * radius) / dz))) + 1
    ny = int(max(1, math.floor((ly - 2.0 * radius) / dy))) + 1
    nx = int(max(1, math.floor((lx - 2.0 * radius) / a))) + 1
    for k in range(nz):
        z = radius + k * dz + z_offset
        su = (k % 2) * 0.5 * a
        sv = (k % 2) * a * math.sqrt(3.0) / 6.0
        for j in range(ny):
            y = radius + sv + j * dy
            if y > ly - radius + 1e-12:
                continue
            for i in range(nx):
                x = radius + su + i * a + (0.5 * a if j % 2 else 0.0)
                if x > lx - radius + 1e-12:
                    continue
                pts.append((x, y, z))
    pos = np.asarray(pts, dtype=np.float64)
    if len(pos) == 0:
        raise ValueError("empty packing")
    if jitter:
        rng = np.random.default_rng(seed)
        pos = pos + rng.standard_normal(pos.shape) * (jitter * radius)
    pos[:, 0] = np.clip(pos[:, 0], radius, lx - radius)
    pos[:, 1] = np.clip(pos[:, 1], radius, ly - radius)
    pos[:, 2] = np.clip(pos[:, 2], radius + z_offset, lz - radius + z_offset)
    return pos


def hcp_fill_cylinder(radius: float, axis_center: tuple[float, float],
                      cylinder_radius: float, width: float,
                      fill_fraction: float = 0.45, seed: int = 1,
                      jitter: float = 0.02) -> np.ndarray:
    """HCP fill of the lower part of a y-axis cylinder (drum)."""
    lx = 2.0 * cylinder_radius
    lz = 2.0 * cylinder_radius * fill_fraction
    pos = hcp_fill_box(radius, lx, width, lz, seed=seed, jitter=jitter,
                       z_offset=0.0)
    # shift so that the cylinder axis is at (cx, cz)
    pos[:, 0] += axis_center[0] - cylinder_radius
    pos[:, 2] += axis_center[1] - cylinder_radius
    pos[:, 1] += 0.0
    # keep only particles inside the cylinder
    d = np.hypot(pos[:, 0] - axis_center[0], pos[:, 2] - axis_center[1])
    pos = pos[d <= cylinder_radius - radius][:]
    return pos