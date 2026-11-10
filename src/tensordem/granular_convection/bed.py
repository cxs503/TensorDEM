"""A deterministic 2-D rigid-disk bed with per-particle density.

Why a new engine
----------------
``tensordem.granular2d.engine.DiskSystem`` derives the mass of every disk from a
single ``DiskConfig.density``.  The reverse Brazil-nut effect is driven by the
*competition* between a size (percolation) and a density (condensation) contrast
(Hong, Quinn & Luding 2001), so the two species must be allowed to carry
different densities.  This module therefore keeps its own small kernel: same
contact law family (linear spring-dashpot normal, Cundall-Strack style
tangential with a Coulomb cap), but per-particle mass and inertia.

Numerics
--------
* Contact search: :class:`scipy.spatial.cKDTree` candidate pairs rebuilt every
  ``relink`` steps with a skin that covers the maximum relative displacement.
* Tangential state: an incremental spring per pair, stored in a sorted-key
  array (``searchsorted``), so no Python-level per-contact bookkeeping.
* Time integration: semi-implicit (symplectic) Euler, ``dt = 0.1 * t_c`` of the
  stiffest pair by default, which is well inside the linear-stability limit of
  the damped spring.

Everything is float64 numpy and deterministic for a fixed seed: there are no
random numbers in the time loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "BedConfig",
    "BedSystem",
    "build_bed",
    "segregate_direction",
    "packing_lattice",
]


@dataclass(frozen=True)
class BedConfig:
    """Configuration of the vibrated bed."""

    width: float = 0.140            # cell width [m]
    bed_height: float = 0.100       # target height of the initial bed [m]
    height: float = 0.200           # cell height (only the top wall is passive) [m]
    d_small: float = 0.0060         # small-species diameter [m]
    d_large: float = 0.0120         # large-species diameter [m]
    rho_small: float = 2500.0       # small-species density [kg/m^3]
    rho_large: float = 2500.0       # large-species density [kg/m^3]
    area_fraction_large: float = 0.40
    solid_fraction: float = 0.80    # of the packing fraction 0.88 (hexagonal limit)
    thickness: float = 0.010        # out-of-plane slice thickness (mass only) [m]
    kn: float = 5.0e4               # normal stiffness [N/m]
    kt_ratio: float = 0.5
    zeta: float = 0.30              # normal dashpot ratio
    mu: float = 0.40                # particle-particle and particle-wall friction
    mu_roll: float = 0.10           # rolling-resistance coefficient
    gravity: float = 9.81
    relink: int = 25
    skin: float = 0.35              # skin as a fraction of the small diameter
    max_particles: int = 900
    seed: int = 7006

    def __post_init__(self) -> None:
        for name in ("width", "height", "d_small", "d_large", "rho_small",
                     "rho_large", "kn", "thickness", "gravity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.d_large < self.d_small:
            raise ValueError("d_large must be >= d_small")
        if not (0.0 <= self.area_fraction_large < 1.0):
            raise ValueError("area_fraction_large must lie in [0, 1)")
        if not (0.0 < self.solid_fraction < 1.0):
            raise ValueError("solid_fraction must lie in (0, 1)")
        if self.mu < 0.0 or self.zeta < 0.0:
            raise ValueError("mu and zeta must be non-negative")


def packing_lattice(config: BedConfig, *, stratified: bool = True,
                    seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Dense bimodal fill from a triangular template lattice.

    A triangular template with pitch ``2 r_small * 1.03`` is scanned row by row.
    A site is assigned a *large* disk (drawn from the interleaved or random plan)
    when every template site within ``r_large + r_small`` is still free -- those
    sites are then blocked -- otherwise a small disk is placed.  The result is a
    strictly non-overlapping, well-mixed packing at the hexagonal limit of the
    requested mixture, which the gravity settle then relaxes to its natural
    solid fraction.  (Random sequential adsorption is *not* used: it jams at
    ~0.5 solid fraction and leaves a strongly heterogeneous mixture.)
    """
    rng = np.random.default_rng(config.seed if seed is None else seed)
    rs = 0.5 * config.d_small
    rl = 0.5 * config.d_large
    f = config.area_fraction_large
    pitch = 2.0 * rs * 1.03
    rows = int((config.bed_height - rs) / (pitch * math.sqrt(3.0) / 2.0)) + 1
    cols = int((config.width - rs) / pitch) + 1
    sites: list[tuple[float, float]] = []
    for j in range(rows):
        y = rs + j * pitch * math.sqrt(3.0) / 2.0
        for i in range(cols):
            x = rs + i * pitch + (0.5 * pitch if j % 2 else 0.0)
            if x < config.width - rs and y < config.bed_height:
                sites.append((x, y))
    sites_arr = np.asarray(sites, dtype=np.float64)
    if len(sites_arr) == 0:
        raise ValueError("the requested cell is smaller than one particle")
    site_tree = cKDTree(sites_arr)
    free = np.ones(len(sites_arr), dtype=bool)
    pos: list[tuple[float, float]] = []
    radii: list[float] = []
    # -- phase 1: the large species, on a coarse sub-lattice of the template ---
    n_large_target = 0
    if config.d_large > config.d_small:
        area_fill = 0.85 * config.width * config.bed_height
        n_large_target = int(round(f * area_fill / (math.pi * rl**2)))
    # A *spread* candidate order is essential: a row-major order would put every
    # large disk in the bottom rows and build a spurious initial gradient (it did:
    # dz_initial = -15 mm on a 74 mm bed).  ``stratified`` therefore only keeps
    # the species plan interleaved; the spatial order is always shuffled.
    order = rng.permutation(len(sites_arr))
    n_large = 0
    for k in order:
        if n_large >= n_large_target:
            break
        if not free[k]:
            continue
        x, y = sites_arr[k]
        if x - rl < 0.0 or x + rl > config.width or y - rl < 0.0:
            continue
        near = site_tree.query_ball_point(sites_arr[k], rl + rs)
        # keep the large species apart (no two large disks may touch a common small)
        if not all(free[i] for i in near):
            continue
        pos.append(tuple(sites_arr[k]))
        radii.append(rl)
        free[near] = False
        n_large += 1
    # -- phase 2: fill every remaining free site with a small disk -------------
    for k in range(len(sites_arr)):
        if not free[k]:
            continue
        x, y = sites_arr[k]
        if x - rs < 0.0 or x + rs > config.width or y - rs < 0.0:
            continue
        pos.append((x, y))
        radii.append(rs)
        free[k] = False
    return np.asarray(pos, dtype=np.float64), np.asarray(radii, dtype=np.float64)


class BedSystem:
    """A rigid-disk assembly with per-particle density, walls and a shaker frame."""

    def __init__(self, positions, radii, densities, config: BedConfig | None = None,
                 *, dt: float | None = None) -> None:
        self.config = config or BedConfig()
        pos = np.array(positions, dtype=np.float64)
        rad = np.array(radii, dtype=np.float64).reshape(-1)
        rho = np.array(densities, dtype=np.float64).reshape(-1)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        if not (len(pos) == len(rad) == len(rho)):
            raise ValueError("positions, radii and densities must have equal length")
        if not np.isfinite(pos).all() or not np.isfinite(rad).all():
            raise ValueError("positions and radii must be finite")
        if (rad <= 0.0).any() or (rho <= 0.0).any():
            raise ValueError("radii and densities must be positive")
        self.pos = pos
        self.radii = rad
        self.densities = rho
        cfg = self.config
        self.mass = rho * math.pi * rad**2 * cfg.thickness
        self.inertia = 0.5 * self.mass * rad**2
        self.vel = np.zeros_like(pos)
        self.theta = np.zeros(len(rad))
        self.omega = np.zeros(len(rad))
        self.time = 0.0
        self.steps = 0
        self.dt = dt if dt is not None else self.recommended_dt
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.time_force = None  # callable(t) -> (N,2) extra body accelerations
        self._pairs = np.zeros((0, 2), dtype=np.int64)
        self._keys = np.zeros(0, dtype=np.int64)
        self._xi = np.zeros((0, 2))
        self._wall_keys = np.zeros(0, dtype=np.int64)
        self._wall_xi = np.zeros((0, 2))
        self.last_contact_count = 0
        self.wall_force = np.zeros(2)

    # -- numerics -------------------------------------------------------
    @property
    def recommended_dt(self) -> float:
        cfg = self.config
        m_min = float(np.min(self.mass))
        omega = math.sqrt(cfg.kn / m_min)
        return 0.1 * (2.0 * math.pi / omega)

    def _relink(self) -> None:
        cfg = self.config
        r_max = float(np.max(self.radii))
        reach = 2.0 * r_max + cfg.skin * cfg.d_small
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(reach, output_type="ndarray")
        if len(pairs):
            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        self._pairs = pairs.astype(np.int64)
        keys = pairs[:, 0] * len(self.pos) + pairs[:, 1]
        self._keys = keys.astype(np.int64)
        self._xi = np.zeros((len(pairs), 2))
        self._x_wall = np.where(
            self.pos[:, 0] < 0.5 * cfg.width, 0, 1).astype(np.int64)
        self._wall_keys = np.arange(len(self.pos), dtype=np.int64)
        self._wall_xi = np.zeros((len(self.pos), 2))

    def _pairs_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Contact forces and torques from the tracked candidate pairs.

        Law: linear spring-dashpot normal (``kn``, Coulomb damping ratio
        ``zeta``); tangential friction is a *regularised Coulomb* law --
        ``f_t = -min(c_t |v_t|, mu f_n) v_t/|v_t|`` -- where ``v_t`` is the
        relative tangential velocity at the contact point (including spin).
        A regularised law is used instead of an incremental Cundall-Strack
        spring because the tangential spring at the small-particle radius makes
        the explicit rotational integration marginally unstable at the dt this
        bed needs; the regularised form is unconditionally dissipative and is a
        standard DEM option.  A rolling-resistance torque
        ``min(c_roll |w_rel|, mu_roll r_eff f_n)`` is added for the same reason
        (it also represents a real ice/grain rolling resistance).
        """
        cfg = self.config
        n = len(self.pos)
        f = np.zeros((n, 2))
        t = np.zeros(n)
        if len(self._pairs) == 0:
            self.last_contact_count = 0
            return f, t
        i, j = self._pairs[:, 0], self._pairs[:, 1]
        d = self.pos[j] - self.pos[i]
        dist = np.hypot(d[:, 0], d[:, 1])
        overlap = self.radii[i] + self.radii[j] - dist
        active = overlap > 0.0
        self.last_contact_count = int(np.count_nonzero(active))
        if self.last_contact_count == 0:
            return f, t
        idx = np.nonzero(active)[0]
        ii, jj = i[idx], j[idx]
        dist_c = np.maximum(dist[idx], 1e-12)
        normal = d[idx] / dist_c[:, None]
        ov = overlap[idx]
        tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
        # contact-point velocities (translational + spin)
        r_i = self.radii[ii]
        r_j = self.radii[jj]
        w_rel = self.omega[jj] - self.omega[ii]
        # contact point velocity of body i: v_i + omega_i x (n r_i)
        cp_i = np.stack([-self.omega[ii] * (normal[:, 1] * r_i),
                         self.omega[ii] * (normal[:, 0] * r_i)], axis=1)
        cp_j = np.stack([-self.omega[jj] * (-normal[:, 1] * r_j),
                         self.omega[jj] * (-normal[:, 0] * r_j)], axis=1)
        v_c = (self.vel[jj] + cp_j) - (self.vel[ii] + cp_i)
        v_n = np.einsum("ab,ab->a", v_c, normal)
        v_t = v_c - v_n[:, None] * normal
        v_t_mag = np.hypot(v_t[:, 0], v_t[:, 1])
        m_eff = 1.0 / (1.0 / self.mass[ii] + 1.0 / self.mass[jj])
        c_n = 2.0 * cfg.zeta * np.sqrt(cfg.kn * m_eff)
        f_n = np.maximum(cfg.kn * ov - c_n * v_n, 0.0)
        c_t = 2.0 * cfg.zeta * np.sqrt(cfg.kn * cfg.kt_ratio * m_eff)
        f_t_mag = np.minimum(c_t * v_t_mag, cfg.mu * f_n)
        safe = np.where(v_t_mag > 0.0, f_t_mag / np.maximum(v_t_mag, 1e-30), 0.0)
        f_pair = -normal * f_n[:, None] + tangent * safe[:, None]
        fx, fy = f_pair[:, 0], f_pair[:, 1]
        f[:, 0] += np.bincount(ii, weights=fx, minlength=n) - np.bincount(jj, weights=fx, minlength=n)
        f[:, 1] += np.bincount(ii, weights=fy, minlength=n) - np.bincount(jj, weights=fy, minlength=n)
        # torque on i from the contact force applied at +n r_i, on j at -n r_j
        tq_i = (normal[:, 0] * r_i) * fy - (normal[:, 1] * r_i) * fx
        tq_j = (-normal[:, 0] * r_j) * (-fy) - (-normal[:, 1] * r_j) * (-fx)
        t += np.bincount(ii, weights=tq_i, minlength=n) + np.bincount(jj, weights=tq_j, minlength=n)
        # rolling resistance
        r_eff = 0.5 * (r_i + r_j)
        i_eff = 0.5 * (self.inertia[ii] + self.inertia[jj])
        c_roll = 0.25 * np.sqrt(cfg.kn * cfg.kt_ratio * i_eff)
        t_roll = np.minimum(c_roll * np.abs(w_rel), cfg.mu_roll * r_eff * f_n)
        t_roll = -np.sign(w_rel) * t_roll
        t += np.bincount(ii, weights=-t_roll, minlength=n) + np.bincount(jj, weights=t_roll, minlength=n)
        return f, t

    def _wall_forces(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        n = len(self.pos)
        f = np.zeros((n, 2))
        t = np.zeros(n)
        r = self.radii
        # left / right walls: signed penetration
        pen_left = r - self.pos[:, 0]
        pen_right = self.pos[:, 0] + r - cfg.width
        pen_bottom = r - self.pos[:, 1]
        m_eff = self.mass
        c_n = 2.0 * cfg.zeta * np.sqrt(cfg.kn * m_eff)
        pen_top = self.pos[:, 1] + r - cfg.height
        for pen, nrm in ((pen_left, np.array([1.0, 0.0])),
                         (pen_right, np.array([-1.0, 0.0])),
                         (pen_bottom, np.array([0.0, 1.0])),
                         (pen_top, np.array([0.0, -1.0]))):
            act = pen > 0.0
            if not act.any():
                continue
            v_n = self.vel[act] @ nrm
            f_n = np.maximum(cfg.kn * pen[act] - c_n[act] * v_n, 0.0)
            f_here = nrm[None, :] * f_n[:, None]
            # tangential viscous friction, capped
            tangent = np.array([-nrm[1], nrm[0]])
            v_t = self.vel[act] @ tangent
            c_t = 2.0 * cfg.zeta * np.sqrt(cfg.kn * cfg.kt_ratio * self.mass[act])
            f_t = -np.minimum(c_t * np.abs(v_t), cfg.mu * f_n) * np.sign(v_t)
            f_here += tangent[None, :] * f_t[:, None]
            f[act] += f_here
            # torque about the disk centre: contact point is r along -nrm
            rc = -nrm[None, :] * r[act][:, None]
            t[act] += rc[:, 0] * f_here[:, 1] - rc[:, 1] * f_here[:, 0]
            self.wall_force += f_here.sum(axis=0)
        return f, t

    # -- public API -----------------------------------------------------
    def step(self, n_steps: int = 1) -> None:
        cfg = self.config
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        for k in range(n_steps):
            if k % cfg.relink == 0:
                self._relink()
            fp_self, tp_self = self._pairs_forces()
            fw, tw = self._wall_forces()
            acc = (fp_self + fw) / self.mass[:, None]
            acc[:, 1] -= cfg.gravity
            if self.time_force is not None:
                acc += self.time_force(self.time)
            self.vel += acc * self.dt
            self.omega += (tp_self + tw) / self.inertia * self.dt
            self.pos += self.vel * self.dt
            self.theta += self.omega * self.dt
            self.time += self.dt
            self.steps += 1

    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.mass * np.sum(self.vel**2, axis=1))
                     + 0.5 * np.sum(self.inertia * self.omega**2))

    def gravity_centre(self, mask=None) -> tuple[float, float]:
        m = self.mass if mask is None else self.mass[mask]
        p = self.pos if mask is None else self.pos[mask]
        if m.sum() <= 0.0:
            return (float("nan"), float("nan"))
        return (float(np.sum(m * p[:, 0]) / m.sum()),
                float(np.sum(m * p[:, 1]) / m.sum()))

    def snapshots(self) -> dict:
        return {"pos": self.pos.copy(), "radii": self.radii.copy(),
                "densities": self.densities.copy(), "time": float(self.time)}


def build_bed(config: BedConfig, *, stratified: bool = True,
              settle_steps: int = 4000, seed: int | None = None) -> tuple[BedSystem, np.ndarray]:
    """Build and settle a bimodal bed; returns ``(system, is_large_mask)``."""
    pos, radii = packing_lattice(config, stratified=stratified, seed=seed)
    is_large = radii > 0.25 * (config.d_small + config.d_large)
    densities = np.where(is_large, config.rho_large, config.rho_small)
    system = BedSystem(pos, radii, densities, config)
    # settle: gravity only, no shaking
    system.step(settle_steps)
    # strip residual kinetic energy so the shake starts from rest
    system.vel[:] = 0.0
    system.omega[:] = 0.0
    return system, is_large


def segregate_direction(z_large: float, z_small: float) -> int:
    """+1 Brazil-nut (large on top), -1 reverse Brazil-nut, 0 unresolved."""
    dz = z_large - z_small
    if dz > 0.0:
        return 1
    if dz < 0.0:
        return -1
    return 0