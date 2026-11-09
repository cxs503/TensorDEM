"""Bidisperse granular segregation: DEM kernel + measurable segregation indices.

Physical background
-------------------
A bed of two sphere sizes driven by vertical shaking spontaneously unmixes: the
large particles rise to the free surface (the *Brazil-nut* effect) while the
fines percolate downwards through the voids opened during the dilation phase of
each shake cycle.  The same size-driven mechanism acts when a heap is built by
pouring: the coarse fraction rolls down the free surface to the periphery.

This module is TensorDEM's verification face for that physics.  It owns

* a **bidisperse** DEM kernel (per-particle radii; ``granular3d_core`` is
  monodisperse by construction) with linear spring-dashpot normal contact,
  Cundall-Strack incremental tangential friction, rate-limited rolling
  resistance and wall contacts;
* the two standard drives: sinusoidal vertical shaking of the floor
  (``Gamma = A omega^2 / g``) and gravity pouring from a nozzle;
* measurable, *not injected*, segregation indices: the top-layer coarse volume
  fraction excess ``S`` and the Brazil-nut rise velocity ``v_up`` of the coarse
  centre of mass;
* bookkeeping audits (mass ledger, centre-of-mass consistency, maximum
  interpenetration, floor load balance) so a "segregation" result cannot be
  produced by an integration artefact.

Sign convention
---------------
``S = (f_coarse,top - phi_coarse) / phi_coarse`` where ``f_coarse,top`` is the
*measured* coarse volume fraction in the top layer of the bed.  Positive ``S``
means "coarse enriched at the surface", i.e. the Brazil-nut direction, and it
increases monotonically with the size ratio.  The mirror-image fine-particle
index written in the campaign node text,
``S_fine = -phi_coarse / (1 - phi_coarse) * S``, is reported as well.

Equal-size negative control
---------------------------
With ``size_ratio == 1`` the two labels are physically identical, so any
non-zero ``S`` or ``v_up`` is an algorithm artefact (label-dependent force,
radius-indexed bug, drift).  Both labels run through exactly the same code
path (two radius classes of equal radius), which makes the control a real
regression test.

Limitations (see ``docs/bidisperse_segregation.md``)
----------------------------------------------------
* Contact stiffness is chosen from a *target static interpenetration* (~2 % of
  the small radius): this is a soft-sphere DEM.  Stiffness scaling changes the
  contact time but not the dimensionless shaking parameters (``Gamma``,
  ``A/d``, size ratio, volume fraction) in which the scaling laws are stated.
* The quasi-2D configuration is a *monolayer slab of spheres* (depth
  ``1.15 d_large``), not a true 2-D disc system.
* Wall rolling resistance is disabled (wall friction only).

All quantities SI (m, kg, s, N, Pa).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

G = 9.81

DEBUG = bool(int(os.environ.get("TENSORDEM_DEBUG", "0")))

#: contact-law constants (same conventions as ``granular3d_core``)
KT_RATIO = 0.5
OVERLAP_TARGET = 0.02     # allowed static interpenetration / r_small
DT_SAFETY = 0.06          # dt / contact time


def damping_ratio(restitution: float) -> float:
    """zeta = |ln e| / sqrt(pi^2 + ln^2 e) (dashpot from restitution)."""
    e = max(min(restitution, 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


def sphere_volume(radius):
    return 4.0 / 3.0 * math.pi * np.asarray(radius, dtype=np.float64) ** 3


@dataclass(frozen=True)
class BedConfig:
    """Case definition for the bidisperse segregation bed (SI)."""

    size_ratio: float = 2.0
    large_volume_fraction: float = 0.2
    n_particles: int = 20000
    d_small: float = 0.004
    density: float = 2500.0
    mu: float = 0.4
    mu_wall: float = 0.35
    mu_rolling: float = 0.02
    restitution: float = 0.3
    geometry: str = "box3d"          # "box3d" | "quasi2d"
    width: float = 0.10              # x extent (snapped to the lattice)
    depth: Optional[float] = None    # y extent; quasi2d default 1.15 d_large
    packing_fraction: float = 0.60   # target packing, used only for box sizing
    headroom: float = 0.8            # free board above the packed bed / H
    seed: int = 1
    kn: Optional[float] = None       # auto if None
    dt: Optional[float] = None       # auto if None
    dt_safety: float = DT_SAFETY
    lattice_margin: float = 0.08     # extra centre distance, in units of r_small

    @property
    def radius_small(self) -> float:
        return 0.5 * self.d_small

    @property
    def radius_large(self) -> float:
        return 0.5 * self.d_small * self.size_ratio

    @property
    def d_large(self) -> float:
        return 2.0 * self.radius_large

    @property
    def counts(self) -> tuple[int, int]:
        """(n_large, n_small) reproducing ``large_volume_fraction`` by volume."""
        s3 = self.size_ratio ** 3
        phi = self.large_volume_fraction
        n_large = int(round(self.n_particles / (1.0 + s3 * (1.0 - phi) / phi)))
        return n_large, self.n_particles - n_large

    def resolved(self) -> "BedConfig":
        depth = self.depth
        if depth is None and self.geometry == "quasi2d":
            depth = 1.15 * self.d_large
        return replace(self, depth=depth)


def _lattice_sites(nx: int, ny: int, k0: int, k1: int, a: float,
                   inset: float = 0.0) -> np.ndarray:
    """Cubic lattice site centres for layers ``k0 .. k1-1``.

    ``inset`` (the coarse radius) keeps every site at least one coarse radius
    away from the walls and the floor, so no particle has to be moved after
    placement (a post-hoc clamp or lift would introduce overlaps).
    """
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny),
                             np.arange(k0, k1), indexing="ij")
    pts = np.empty((ii.size, 3), dtype=np.float64)
    pts[:, 0] = inset + (ii.ravel() + 0.5) * a
    pts[:, 1] = inset + (jj.ravel() + 0.5) * a
    pts[:, 2] = inset + 0.5 * a + kk.ravel() * a
    return pts


def build_packing(cfg: BedConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Mixed-size lattice packing with per-particle radii.

    Two-phase construction on one fine lattice (spacing ``2 r_s (1+margin)``):

    1. the coarse particles are dropped on a *random* subset of lattice sites,
       so their spatial distribution carries no vertical gradient;
    2. the fines fill the remaining sites that are clear of coarse particles.

    The lattice is sized for the site budget the coarse particles consume
    (``~ s^3`` sites each) plus one site per fine particle, so the resulting
    bed is *looser* than the target packing fraction and is compacted by
    ``BidisperseBed.settle``.  Returns ``(pos, radius, Lx, Ly)``.
    """
    rng = np.random.default_rng(cfg.seed)
    r_s, r_l = cfg.radius_small, cfg.radius_large
    n_l, n_s = cfg.counts
    a = 2.0 * r_s * (1.0 + cfg.lattice_margin)
    Lx = max(2.0 * a, round(cfg.width / a) * a) + 2.0 * r_l
    depth = float(cfg.depth) if cfg.depth is not None else Lx
    Ly = max(2.0 * a, round((depth - 2.0 * r_l) / a) * a) + 2.0 * r_l
    nx = int(round((Lx - 2.0 * r_l) / a))
    ny = int(round((Ly - 2.0 * r_l) / a))

    sites_per_large = int(math.ceil((2.0 * r_l / a) ** 3)) + 1
    n_sites = n_s + n_l * sites_per_large + 2 * nx * ny
    n_layers = int(math.ceil(n_sites / (nx * ny)))

    occ: dict[tuple[int, int, int], float] = {}
    pos_list: list[tuple[float, float, float]] = []
    rad_list: list[float] = []
    coarse_list: list[bool] = []
    reach_l = int(math.ceil(2.0 * r_l / a)) + 1
    reach_s = int(math.ceil((r_l + r_s) / a)) + 1
    margin = cfg.lattice_margin * r_s

    def free(ix, iy, iz, radius, reach):
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                for dk in range(-reach, reach + 1):
                    r_other = occ.get((ix + di, iy + dj, iz + dk))
                    if r_other is None:
                        continue
                    if a * math.sqrt(di * di + dj * dj + dk * dk) < radius + r_other + margin:
                        return False
        return True

    placed_l = 0
    while placed_l < n_l:                    # phase 1: coarse, random sites
        sites = _lattice_sites(nx, ny, 0, n_layers, a, r_l)
        for idx in rng.permutation(len(sites)):
            if placed_l >= n_l:
                break
            x, y, z = sites[idx]
            key = (int(x // a), int(y // a), int(z // a))
            if key in occ or not free(*key, r_l, reach_l):
                continue
            occ[key] = r_l
            pos_list.append((x, y, z))
            rad_list.append(r_l)
            coarse_list.append(True)
            placed_l += 1
        n_layers += max(2, int(round(2.0 * r_l / a)))   # extend if too tight
        if n_layers > 100000:                            # pragma: no cover
            raise RuntimeError("coarse packing did not converge")

    placed_s = 0
    n_layers = int(math.ceil((n_s + n_l * sites_per_large + 2 * nx * ny) / (nx * ny)))
    while placed_s < n_s:                    # phase 2: fines in the gaps
        sites = _lattice_sites(nx, ny, 0, n_layers, a, r_l)
        for idx in rng.permutation(len(sites)):
            if placed_s >= n_s:
                break
            x, y, z = sites[idx]
            key = (int(x // a), int(y // a), int(z // a))
            if key in occ or not free(*key, r_s, reach_s):
                continue
            occ[key] = r_s
            pos_list.append((x, y, z))
            rad_list.append(r_s)
            coarse_list.append(False)
            placed_s += 1
        n_layers += 2
        if n_layers > 100000:                            # pragma: no cover
            raise RuntimeError("fine packing did not converge")

    pos = np.asarray(pos_list, dtype=np.float64)
    rad = np.asarray(rad_list, dtype=np.float64)
    coarse = np.asarray(coarse_list, dtype=bool)
    pos += rng.standard_normal(pos.shape) * 0.005 * a
    perm = rng.permutation(len(pos))       # de-correlate index order from class
    return pos[perm], rad[perm], coarse[perm], Lx, Ly


def _accum(target: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """target[idx] += vals, vectorised per component (np.add.at is slow)."""
    for k in range(3):
        target[:, k] += np.bincount(idx, weights=vals[:, k], minlength=len(target))


class BidisperseBed:
    """Bidisperse sphere bed in a closed box with a kinematically driven floor."""

    def __init__(self, cfg: BedConfig):
        self.cfg = cfg = cfg.resolved()
        pos, rad, coarse, Lx, Ly = build_packing(cfg)
        self.pos = pos
        self.r = rad
        self.Lx, self.Ly = Lx, Ly
        n = len(rad)
        self.n = n
        self.cls = coarse.astype(np.int64)
        self.volume = sphere_volume(rad)
        self.mass = cfg.density * self.volume
        self.inertia = 0.4 * self.mass * rad ** 2
        self.inv_m = 1.0 / self.mass
        self.inv_i = 1.0 / self.inertia
        self.frozen = np.zeros(n, dtype=bool)

        self.vel = np.zeros((n, 3))
        self.omega = np.zeros((n, 3))

        self.bed_height_packed = float(np.max(pos[:, 2] + rad))
        self.Lz = self.bed_height_packed * (1.0 + cfg.headroom)

        self.zeta = damping_ratio(cfg.restitution)
        self.kn = float(cfg.kn) if cfg.kn is not None else self._auto_kn()
        self.dt = float(cfg.dt) if cfg.dt is not None else self._auto_dt(self.kn)

        self._tang_keys = np.zeros(0, dtype=np.int64)
        self._tang = np.zeros((0, 3))
        self._roll_keys = np.zeros(0, dtype=np.int64)
        self._roll = np.zeros((0, 3))
        self.wall_tang = [np.zeros((n, 3)) for _ in range(5)]
        self.t = 0.0
        self.extra_damping = 0.0
        self.shake = (0.0, 0.0)          # (amplitude, omega)
        self.last_floor_force = np.zeros(n)
        self.last_max_overlap = 0.0
        self.released = np.ones(n, dtype=bool)

    # --------------------------------------------------------------- setup --
    def _auto_kn(self) -> float:
        """Stiffness from a target static interpenetration at the bed base."""
        cfg = self.cfg
        rho_b = cfg.packing_fraction * cfg.density
        f_ref = rho_b * G * self.bed_height_packed * (2.0 * cfg.radius_small) ** 2
        return float(max(f_ref / (OVERLAP_TARGET * cfg.radius_small), 50.0))

    def _auto_dt(self, kn: float) -> float:
        cfg = self.cfg
        m_small = cfg.density * float(sphere_volume(cfg.radius_small))
        t_c = math.pi * math.sqrt(m_small / (2.0 * kn))
        return float(cfg.dt_safety * t_c)

    @property
    def kt(self) -> float:
        return KT_RATIO * self.kn

    def floor_offset(self, t: float) -> float:
        amp, om = self.shake
        return amp * math.sin(om * t)

    def floor_velocity(self, t: float) -> float:
        amp, om = self.shake
        return amp * om * math.cos(om * t)

    # ---------------------------------------------------------- neighbours --
    def _pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Contact pairs (i < j), radius-class aware and pre-filtered.

        Each radius class is queried with its *own* cut-off (small-small,
        coarse-coarse) and the cross class with ``r_small + r_large``, so the
        candidate list stays within a few percent of the true contact list (a
        single ``2 r_max`` cut-off would inflate it by ``(r_max/r_typ)^3``).
        Pairs are pre-filtered to actual overlaps here, which keeps every
        downstream array proportional to the real contact number.
        """
        pos, r, cls = self.pos, self.r, self.cls
        pts = pos[:, [0, 2]] if self.cfg.geometry == "quasi2d" else pos
        idx_s = np.flatnonzero(cls == 0)
        idx_l = np.flatnonzero(cls == 1)
        r_s = float(r[idx_s[0]]) if len(idx_s) else 0.0
        r_l = float(r[idx_l[0]]) if len(idx_l) else 0.0
        out_i: list[np.ndarray] = []
        out_j: list[np.ndarray] = []
        ts = cKDTree(pts[idx_s]) if len(idx_s) > 1 else None
        tl = cKDTree(pts[idx_l]) if len(idx_l) > 1 else None
        if ts is not None:
            sub = ts.query_pairs(2.0 * r_s * 1.02, output_type="ndarray")
            if len(sub):
                out_i.append(idx_s[sub[:, 0]])
                out_j.append(idx_s[sub[:, 1]])
        if tl is not None:
            sub = tl.query_pairs(2.0 * r_l * 1.02, output_type="ndarray")
            if len(sub):
                out_i.append(idx_l[sub[:, 0]])
                out_j.append(idx_l[sub[:, 1]])
        if ts is not None and tl is not None:
            sm = tl.sparse_distance_matrix(ts, (r_s + r_l) * 1.02,
                                           output_type="ndarray")
            if len(sm):
                out_i.append(idx_l[sm["i"].astype(np.int64)])
                out_j.append(idx_s[sm["j"].astype(np.int64)])
        if not out_i:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        iu = np.concatenate(out_i)
        ju = np.concatenate(out_j)
        d = np.linalg.norm(pos[ju] - pos[iu], axis=1)
        keep = d < r[iu] + r[ju]
        return iu[keep], ju[keep]

    # ------------------------------------------------------------ dynamics --
    def _particle_forces(self, iu, ju, dt):
        n = self.n
        force = np.zeros((n, 3))
        torque = np.zeros((n, 3))
        if len(iu) == 0:
            return force, torque
        pos, r = self.pos, self.r
        dvec = pos[ju] - pos[iu]
        dist = np.linalg.norm(dvec, axis=1)
        overlap = r[iu] + r[ju] - dist
        active = overlap > 0.0
        if not active.any():
            return force, torque
        iu, ju, overlap = iu[active], ju[active], overlap[active]
        dist = np.maximum(dist[active], 1e-30)
        nvec = dvec[active] / dist[:, None]
        self.last_max_overlap = max(
            self.last_max_overlap,
            float(np.max(overlap / np.minimum(r[iu], r[ju]))))

        kn, kt = self.kn, self.kt
        v_rel = self.vel[ju] - self.vel[iu]
        v_n = np.einsum("ij,ij->i", v_rel, nvec)
        m_eff = self.mass[iu] * self.mass[ju] / (self.mass[iu] + self.mass[ju])
        gamma = 2.0 * self.zeta * np.sqrt(kn * m_eff)
        f_n = np.maximum(kn * overlap - gamma * v_n, 0.0)
        if DEBUG:
            for _n, _a in (("overlap", overlap), ("v_n", v_n), ("gamma", gamma),
                           ("f_n", f_n), ("omega", self.omega), ("vel", self.vel)):
                if not np.isfinite(_a).all():
                    raise RuntimeError(f"non-finite {_n} in particle_forces")

        n_pairs = len(iu)
        keys = iu.astype(np.int64) * n + ju.astype(np.int64)
        order = np.argsort(keys, kind="stable")
        keys_s = keys[order]
        tang = self._lookup(self._tang_keys, self._tang, keys_s)
        surf_i = np.cross(self.omega[iu], nvec * r[iu][:, None])
        surf_j = np.cross(self.omega[ju], -nvec * r[ju][:, None])
        v_t = (v_rel - v_n[:, None] * nvec) + (surf_j - surf_i)
        tang = tang + v_t[order] * dt
        cap = (self.cfg.mu * f_n[order] / kt)[:, None]
        tnorm = np.linalg.norm(tang, axis=1)[:, None]
        tang *= np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
        self._tang_keys, self._tang = keys_s, tang
        f_t = np.empty((n_pairs, 3))
        f_t[order] = -kt * tang

        f_total = f_n[:, None] * nvec + f_t
        tc = -np.cross(nvec, f_t)
        _accum(force, iu, -f_total)
        _accum(force, ju, f_total)
        _accum(torque, iu, tc * r[iu][:, None])
        _accum(torque, ju, tc * r[ju][:, None])

        if self.cfg.mu_rolling > 0.0:
            r_eff = 2.0 * r[iu] * r[ju] / (r[iu] + r[ju])
            kr = kt * r_eff ** 2
            spin = (self.omega[ju] - self.omega[iu])[order]
            roll = self._lookup(self._roll_keys, self._roll, keys_s) + spin * dt
            cap_r = (self.cfg.mu_rolling * r_eff[order] * f_n[order] / kr)[:, None]
            rnorm = np.linalg.norm(roll, axis=1)[:, None]
            roll *= np.where(rnorm > cap_r, cap_r / np.maximum(rnorm, 1e-30), 1.0)
            self._roll_keys, self._roll = keys_s, roll
            torque_r = np.empty((n_pairs, 3))
            torque_r[order] = -kr[:, None] * roll
            _accum(torque, iu, -torque_r)
            _accum(torque, ju, torque_r)
        return force, torque

    @staticmethod
    def _lookup(prev_keys, prev_vals, keys_sorted):
        """Gather the per-pair state of the previous step (0 where new)."""
        if len(prev_keys) == 0 or len(keys_sorted) == 0:
            return np.zeros((len(keys_sorted), 3))
        slot = np.minimum(np.searchsorted(prev_keys, keys_sorted),
                          max(len(prev_keys) - 1, 0))
        found = prev_keys[slot] == keys_sorted
        vals = np.zeros((len(keys_sorted), 3))
        vals[found] = prev_vals[slot[found]]
        return vals

    def _wall_contacts(self, t, dt):
        """Floor + four side walls + lid, all as plane frictional contacts."""
        n = self.n
        pos, r, vel, om = self.pos, self.r, self.vel, self.omega
        force = np.zeros((n, 3))
        torque = np.zeros((n, 3))
        kn, kt, mu_w = self.kn, self.kt, self.cfg.mu_wall
        zf = self.floor_offset(t)
        vzf = self.floor_velocity(t)

        walls = [
            # (id, axis, normal sign, wall coordinate, wall velocity vector)
            (0, 2, +1.0, zf, np.array([0.0, 0.0, vzf])),
            (1, 0, +1.0, 0.0, np.zeros(3)),
            (2, 0, -1.0, self.Lx, np.zeros(3)),
            (3, 1, +1.0, 0.0, np.zeros(3)),
            (4, 1, -1.0, self.Ly, np.zeros(3)),
        ]
        floor_fz = np.zeros(n)
        for wid, axis, sign, coord, v_wall in walls:
            nvec = np.zeros(3)
            nvec[axis] = sign
            pen = -sign * (pos[:, axis] - coord) + r   # > 0 inside the wall
            active = pen > 0.0
            if not active.any():
                self.wall_tang[wid][:] = 0.0
                continue
            idx = np.flatnonzero(active)
            ov = pen[idx]
            self.last_max_overlap = max(
                self.last_max_overlap, float(np.max(ov / r[idx])))
            v_rel = vel[idx] - v_wall
            v_n = v_rel @ nvec
            gamma = 2.0 * self.zeta * np.sqrt(kn * self.mass[idx])
            f_n = np.maximum(kn * ov - gamma * v_n, 0.0)
            surf = np.cross(om[idx], -nvec[None, :] * r[idx][:, None])
            v_t = v_rel - v_n[:, None] * nvec[None, :] + surf
            tang = self.wall_tang[wid]
            new_tang = np.zeros_like(tang)
            new_tang[idx] = tang[idx] + v_t * dt
            cap = (mu_w * f_n / kt)[:, None]
            tnorm = np.linalg.norm(new_tang[idx], axis=1)[:, None]
            new_tang[idx] *= np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
            self.wall_tang[wid] = new_tang
            f_t = -kt * new_tang[idx]
            if DEBUG and not (np.isfinite(f_t).all() and np.isfinite(f_n).all()):
                raise RuntimeError("non-finite wall contact force")
            _accum(force, idx, f_n[:, None] * nvec[None, :] + f_t)
            _accum(torque, idx, np.cross(-nvec[None, :] * r[idx][:, None], f_t))
            if wid == 0:
                floor_fz += np.bincount(idx, weights=f_n, minlength=n)
        return force, torque, floor_fz

    def step(self, dt: Optional[float] = None):
        dt = self.dt if dt is None else dt
        iu, ju = self._pairs()
        self.last_max_overlap = 0.0
        f_p, tq_p = self._particle_forces(iu, ju, dt)
        f_w, tq_w, floor_fz = self._wall_contacts(self.t, dt)
        acc = (f_p + f_w) * self.inv_m[:, None]
        acc[:, 2] -= G
        acc[self.frozen] = 0.0
        self.vel += acc * dt
        self.omega += (tq_p + tq_w) * self.inv_i[:, None] * dt
        if self.extra_damping > 0.0:
            decay = math.exp(-self.extra_damping * dt)
            self.vel *= decay
            self.omega *= decay
        self.vel[self.frozen] = 0.0
        self.omega[self.frozen] = 0.0
        self.pos += self.vel * dt
        self.t += dt
        self.last_floor_force = floor_fz
        overlap = self.last_max_overlap
        self.last_max_overlap = 0.0
        return {"max_overlap": overlap}

    def settle(self, duration: float, damping: float = 120.0):
        """Relax the freshly built packing under gravity (lattice -> random)."""
        self.extra_damping = damping
        self.shake = (0.0, 0.0)
        steps = int(round(duration / self.dt))
        for _ in range(steps):
            self.step(self.dt)
        self.vel[:] = 0.0
        self.omega[:] = 0.0
        self.extra_damping = 0.0
        self.bed_height_settled = float(np.max(self.pos[:, 2] + self.r))
        self.Lz = self.bed_height_settled * (1.0 + self.cfg.headroom)
        self._tang_keys = np.zeros(0, dtype=np.int64)
        self._roll_keys = np.zeros(0, dtype=np.int64)
        self.wall_tang = [np.zeros((self.n, 3)) for _ in range(5)]

    # -------------------------------------------------------- measurements --
    @property
    def total_mass(self) -> float:
        return float(np.sum(self.mass))

    def measure(self, top_fraction: float = 0.2, x_bins: int = 8) -> dict:
        """Instantaneous segregation / bookkeeping observables."""
        z = self.pos[:, 2]
        r, vol, m = self.r, self.volume, self.mass
        coarse = self.cls == 1
        z_floor = self.floor_offset(self.t)
        z_top = float(np.max(z + r))
        h = max(z_top - z_floor, 1e-12)
        cut = z_top - top_fraction * h
        in_top = (z + r) >= cut
        v_top = float(np.sum(vol[in_top]))
        f_coarse_top = float(np.sum(vol[in_top & coarse]) / v_top) if v_top > 0 else 0.0
        phi = self.cfg.large_volume_fraction
        S = (f_coarse_top - phi) / phi
        S_fine = (f_coarse_top - phi) * phi / ((1.0 - phi) * phi) * -1.0
        # thin-slab variant (3 coarse diameters below the surface)
        cut3 = z_top - 3.0 * self.cfg.d_large
        in3 = (z + r) >= cut3
        v3 = float(np.sum(vol[in3]))
        f3 = float(np.sum(vol[in3 & coarse]) / v3) if v3 > 0 else 0.0
        m_tot = self.total_mass
        z_com = float(np.sum(m * z) / m_tot)
        z_coarse = float(np.sum(m[coarse] * z[coarse]) / np.sum(m[coarse]))
        x_com = float(np.sum(m * self.pos[:, 0]) / m_tot)
        return {
            "t": self.t,
            "S": S,
            "S_fine": S_fine,
            "S_thin": (f3 - phi) / phi,
            "f_coarse_top": f_coarse_top,
            "z_top": z_top,
            "bed_height": h,
            "z_com": z_com,
            "z_coarse": z_coarse,
            "x_com": x_com,
            "segregation_distance": z_coarse - z_com,
            "mass_total": m_tot,
            "n_particles": self.n,
            "max_overlap": self.last_max_overlap,
            "floor_force_total": float(np.sum(self.last_floor_force)),
            "floor_pressure_bins": self._floor_pressure_bins(x_bins),
        }

    def _floor_pressure_bins(self, n_bins: int) -> list[float]:
        x = self.pos[:, 0]
        idx = np.clip((x / self.Lx * n_bins).astype(np.int64), 0, n_bins - 1)
        area = (self.Lx / n_bins) * self.Ly
        force = np.bincount(idx, weights=self.last_floor_force, minlength=n_bins)
        return (force / area).tolist()

    def packing_state(self) -> dict:
        """Bulk quantities used by the stability mapping."""
        return {
            "mass_total": self.total_mass,
            "solid_volume": float(np.sum(self.volume)),
            "bed_height": float(np.max(self.pos[:, 2] + self.r)) - self.floor_offset(self.t),
            "Lx": self.Lx,
            "Ly": self.Ly,
            "Lz": self.Lz,
            "n_large": int(np.sum(self.cls == 1)),
            "n_small": int(np.sum(self.cls == 0)),
            "kn": self.kn,
            "dt": self.dt,
            "kn_overlap_ratio": self.last_max_overlap,
        }

    # --------------------------------------------------------------- drives --
    def run_shaking(self, duration: float, amplitude: float, frequency: float,
                    sample_dt: float = 0.01, top_fraction: float = 0.2,
                    x_bins: int = 8) -> dict:
        """Sinusoidal vertical shaking; returns the sampled history."""
        omega = 2.0 * math.pi * frequency
        self.shake = (amplitude, omega)
        steps = int(round(duration / self.dt))
        sample_every = max(1, int(round(sample_dt / self.dt)))
        history = []
        initial = self.measure(top_fraction, x_bins)
        diverged = False
        for step in range(1, steps + 1):
            self.step()
            if not np.isfinite(self.pos).all():
                diverged = True
                break
            if step % sample_every == 0 or step == steps:
                rec = self.measure(top_fraction, x_bins)
                rec["step"] = step
                rec["gamma"] = amplitude * omega ** 2 / G
                history.append(rec)
        return {"history": history, "initial": initial, "diverged": diverged,
                "amplitude": amplitude, "frequency": frequency,
                "gamma": amplitude * omega ** 2 / G, "duration": duration,
                "n_steps": steps,
                "d_small": 2.0 * self.cfg.radius_small}

    def run_pour(self, n_steps: int, nozzle_width: float, release_steps: int,
                 sample_dt: float = 0.01, top_fraction: float = 0.2) -> dict:
        """Heap building by releasing particles from a nozzle at the top."""
        rng = np.random.default_rng(self.cfg.seed + 991)
        idx = np.arange(self.n)
        x_mid = 0.5 * self.Lx
        # park every particle in the nozzle stack, frozen
        order = rng.permutation(idx)
        per_column = max(1, int(nozzle_width / (2.0 * self.r.max())))
        park_z0 = self.Lz * 0.98
        for k, i in enumerate(order):
            col = k % per_column
            layer = k // per_column
            self.pos[i, 0] = x_mid + (col + 0.5) * nozzle_width / per_column
            self.pos[i, 1] = 0.5 * self.Ly
            self.pos[i, 2] = park_z0 - layer * 2.02 * self.r.max()
        self.frozen[:] = True
        self.vel[:] = 0.0
        self.omega[:] = 0.0
        self.released[:] = False
        release_rate = self.n / max(release_steps, 1)
        history = []
        released_count = 0
        for step in range(1, n_steps + 1):
            target = min(self.n, int(step * release_rate))
            while released_count < target:
                i = order[released_count]
                self.frozen[i] = False
                self.released[i] = True
                self.vel[i] = (0.0, 0.0, -0.05)
                released_count += 1
            self.step()
            if step % max(1, int(round(sample_dt / self.dt))) == 0 or step == n_steps:
                history.append(self._heap_measure(top_fraction))
        return {"history": history, "n_steps": n_steps,
                "released": released_count,
                "mass_injected": float(np.sum(self.mass[self.released]))}

    def _heap_measure(self, top_fraction: float) -> dict:
        z = self.pos[:, 2]
        coarse = self.cls == 1
        vol = self.volume
        m = self.mass
        z_top = float(np.max(z + self.r))
        apex = self.pos[np.argmax(z + self.r)]
        dx = np.abs(self.pos[:, 0] - apex[0])
        half = max(np.max(dx), 1e-9)
        base = z <= (z_top * 0.35)          # lower third of the heap
        outer = base & (dx > 0.5 * half)
        inner = base & (dx <= 0.5 * half)
        phi = self.cfg.large_volume_fraction
        f_outer = float(np.sum(vol[outer & coarse]) / np.sum(vol[outer])) if outer.any() else 0.0
        f_inner = float(np.sum(vol[inner & coarse]) / np.sum(vol[inner])) if inner.any() else 0.0
        return {
            "t": self.t,
            "S_pour": (f_outer - phi) / phi,
            "S_pour_core": (f_inner - phi) / phi,
            "f_coarse_outer": f_outer,
            "f_coarse_inner": f_inner,
            "z_top": z_top,
            "z_com": float(np.sum(m * z) / np.sum(m)),
            "mass_released": float(np.sum(m[self.released])),
            "n_released": int(np.sum(self.released)),
            "max_overlap": self.last_max_overlap,
        }


# ------------------------------------------------------------------ analysis --
def _fit_slope(t, y):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(t) < 3:
        return float("nan")
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0])


def rise_velocity(history: list[dict], window: float = 0.6) -> dict:
    """Brazil-nut rise velocity of the coarse COM relative to the bed COM.

    The slope is fitted over the last ``window`` fraction of the record so that
    the transient compaction phase after start-up is excluded.
    """
    if len(history) < 4:
        return {"v_up": float("nan"), "rise_per_cycle": float("nan")}
    t = np.array([h["t"] for h in history])
    d = np.array([h["segregation_distance"] for h in history])
    t0 = t[0] + window * (t[-1] - t[0])
    mask = t >= t0
    v_up = _fit_slope(t[mask], d[mask])
    return {"v_up": v_up, "rise_per_cycle": float("nan"),
            "segregation_distance": float(d[-1] - d[0]),
            "window_start": float(t0)}


def summarize_shake(result: dict, top_fraction: float = 0.2) -> dict:
    """Steady-state segregation index, rise velocity and audit trail."""
    hist = result["history"]
    t = np.array([h["t"] for h in hist])
    S = np.array([h["S"] for h in hist])
    # steady window: last 40 % of the record
    t0 = t[0] + 0.6 * (t[-1] - t[0])
    mask = t >= t0
    S_mean = float(np.mean(S[mask]))
    S_std = float(np.std(S[mask], ddof=1)) if np.sum(mask) > 2 else float("nan")
    n_eff = max(int(np.sum(mask)), 1)
    rise = rise_velocity(hist, window=0.6)
    freq = result["frequency"]
    v_up = rise["v_up"]
    out = {
        "S": S_mean,
        "S_std": S_std,
        "S_stderr": S_std / math.sqrt(n_eff),
        "S_initial": float(hist[0]["S"]),
        "S_fine_steady": float(np.mean([h["S_fine"] for h in hist if h["t"] >= t0])),
        "S_thin": float(np.mean([h["S_thin"] for h in hist if h["t"] >= t0])),
        "v_up": v_up,
        "rise_per_cycle": v_up / freq if freq else float("nan"),
        "segregation_distance": rise.get("segregation_distance", float("nan")),
        "gamma": result["gamma"],
        "amplitude": result["amplitude"],
        "frequency": freq,
        "amplitude_over_d_small": result["amplitude"] / result["d_small"],
        "mass_total": float(hist[-1]["mass_total"]),
        "mass_total_initial": float(result["initial"]["mass_total"]),
        "mass_rel_error": abs(hist[-1]["mass_total"] - result["initial"]["mass_total"])
        / result["initial"]["mass_total"],
        "z_com_initial": float(result["initial"]["z_com"]),
        "z_com_final": float(hist[-1]["z_com"]),
        "max_overlap": float(np.max([h["max_overlap"] for h in hist])),
        "floor_pressure_peak_bins": np.max(np.array(
            [h["floor_pressure_bins"] for h in hist]), axis=0).tolist(),
        "floor_pressure_mean_bins": np.mean(np.array(
            [h["floor_pressure_bins"] for h in hist]), axis=0).tolist(),
        "bed_height_final": float(hist[-1]["bed_height"]),
        "bed_height_initial": float(result["initial"]["bed_height"]),
        "n_steps": result["n_steps"],
    }
    return out


def fit_size_exponent(rows: list[dict]) -> dict:
    """Fit v_up ∝ (size_ratio - 1)^p by least squares on the log-log line."""
    pts = [(r["size_ratio"], r["v_up"]) for r in rows
           if r.get("v_up", 0) > 0 and r["size_ratio"] > 1.0]
    if len(pts) < 2:
        return {"p": float("nan"), "p_stderr": float("nan"), "n": len(pts)}
    x = np.log(np.array([p[0] for p in pts]) - 1.0)
    y = np.log(np.array([p[1] for p in pts]))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    p = float(coef[0])
    dof = max(len(x) - 2, 1)
    ss = float(res[0]) if len(res) else 0.0
    sigma2 = ss / dof if dof else 0.0
    cov = sigma2 * np.linalg.inv(A.T @ A)
    return {"p": p, "p_stderr": float(math.sqrt(max(cov[0, 0], 0.0))),
            "log_a": float(coef[1]), "n": len(pts),
            "size_ratios": [p[0] for p in pts]}


def fit_gamma_threshold(rows: list[dict]) -> dict:
    """Michaelis-Menten style onset fit: S(Gamma) = S_sat (G-Gc)/((G-Gc)+G0).

    Returns the fitted threshold ``Gamma_c`` (where S starts to grow), the
    saturation value ``S_sat`` and the half-saturation offset ``G0``.
    """
    rows = sorted([r for r in rows if np.isfinite(r.get("S", np.nan))],
                  key=lambda r: r["gamma"])
    if len(rows) < 4:
        return {"gamma_c": float("nan"), "S_sat": float("nan"), "S_half": float("nan")}
    g = np.array([r["gamma"] for r in rows])
    S = np.array([r["S"] for r in rows])
    # Denjoy-style least squares on a fixed grid of (Gamma_c, G0), linear in S_sat
    best = None
    for gc in np.linspace(-0.5, min(g.max() * 0.9, 2.5), 121):
        x = np.maximum(g - gc, 0.0)
        for g0 in np.linspace(0.05, 3.0, 60):
            shape = x / (x + g0)
            denom = float(shape @ shape)
            if denom < 1e-12:
                continue
            Ssat = float(shape @ S) / denom
            err = float(np.sum((S - Ssat * shape) ** 2))
            if best is None or err < best[0]:
                best = (err, float(gc), float(g0), Ssat)
    err, gc, g0, Ssat = best
    # onset threshold from the fitted curve at 10 % of saturation
    return {"gamma_c": gc, "gamma0": g0, "S_sat": Ssat,
            "gamma_onset_10pct": gc + 0.1 * g0 / 0.9,
            "rmse": math.sqrt(err / len(g)), "n": len(g),
            "gamma_points": g.tolist(), "S_points": S.tolist()}


def coefficient_of_variation(values) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(v) < 2 or abs(np.mean(v)) < 1e-30:
        return float("nan")
    return float(np.std(v, ddof=1) / abs(np.mean(v)))