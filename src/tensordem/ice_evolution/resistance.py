"""Minimal 2-D bonded-disk DEM: a rigid platen (ship bow) drives through a
rubble bed (campaign IN155-N09).

The platen is a rigid vertical wall moving at a constant speed through a bed
of bidisperse ice disks.  Disks that were in contact at ``t = 0`` carry a
sintered bond (stiffness ``k_bond``, failure force ``F_bond``) supplied by the
sintering module.  As the platen advances it breaks bonds and compacts the
rubble; the time history of the platen force is the ice-resistance history.

Model
-----
* rigid disks, mass ``m = rho pi r^2`` (per unit thickness);
* disk-disk contact: linear spring-dashpot normal law + rate-independent
  Coulomb tangential friction (``mu F_n``);
* sintered bond: two-way linear spring of rest length ``L_0`` (the initial
  centre distance), stiffness ``k_bond``; it fails when the tensile force
  exceeds ``F_bond``;
* platen: a rigid vertical wall at ``x = x_p(t) = x_p0 + v_p t``; a disk
  overlapping the face receives a normal spring-dashpot push in ``+x``;
* integration: semi-implicit (symplectic) Euler; top/bottom are rigid walls.

The resistance is the time-averaged platen force during the transit.  The
calibration ``R(sigma_bond)`` is what the channel driver interpolates.

Units are SI (m, s, N, kg).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RHO_ICE = 917.0
GRAVITY = 9.81


@dataclass(frozen=True)
class PlatenConfig:
    """Configuration of the platen-through-rubble DEM."""
    n_particles: int = 160
    mean_radius: float = 0.012
    box_width: float = 1.6        # m, x extent
    box_height: float = 0.40      # m, y extent
    platen_speed: float = 0.4     # m/s
    duration_s: float = 4.5
    mu: float = 0.3
    kn: float = 2.0e5             # N/m, contact normal stiffness
    damping_ratio: float = 0.3
    dt_s: float = 2.0e-4
    seed: int = 0
    # sintered bond (supplied by the sintering module)
    k_bond: float = 0.0           # N/m  (0 -> no bonds, loose rubble)
    f_bond: float = 0.0           # N
    thickness_m: float = 0.2      # m, for per-area resistance

    def __post_init__(self) -> None:
        if self.n_particles < 8:
            raise ValueError("n_particles must be >= 8")
        if self.mean_radius <= 0.0 or self.box_width <= 0.0 or self.box_height <= 0.0:
            raise ValueError("radii and box extents must be positive")
        if self.platen_speed <= 0.0 or self.duration_s <= 0.0:
            raise ValueError("platen_speed and duration_s must be positive")


class PlatenDEM:
    """A rubble bed with a moving platen; see the module docstring."""

    def __init__(self, cfg: PlatenConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        n = cfg.n_particles
        # bidisperse radii (30 % large, ratio 1.4), normalised to mean_radius
        big = rng.random(n) < 0.3
        radii = cfg.mean_radius * np.where(big, 1.4, 1.0)
        radii = radii * (cfg.mean_radius / radii.mean())
        # scale to a target area fraction (0.5)
        area = cfg.box_width * cfg.box_height
        radii = radii * math.sqrt(0.5 * area / (math.pi * float((radii**2).sum())))
        self.radii = radii
        self.mass = RHO_ICE * math.pi * radii**2

        # jittered-grid initial packing
        cols = int(math.ceil(math.sqrt(n * cfg.box_width / cfg.box_height)))
        rows = int(math.ceil(n / cols))
        gx = cfg.box_width / cols
        gy = cfg.box_height / rows
        pos = np.zeros((n, 2))
        idx = 0
        for rr in range(rows):
            for cc in range(cols):
                if idx >= n:
                    break
                pos[idx, 0] = (cc + 0.5) * gx + rng.uniform(-0.25, 0.25) * gx
                pos[idx, 1] = (rr + 0.5) * gy + rng.uniform(-0.25, 0.25) * gy
                idx += 1
        self.pos = pos
        self.vel = np.zeros((n, 2))

        # bonds between initially touching disks
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        touch = (d < (radii[:, None] + radii[None, :]) * 1.05)
        iu = np.triu_indices(n, k=1)
        touch = touch[iu]
        self.bond_i = iu[0][touch]
        self.bond_j = iu[1][touch]
        self.bond_L0 = d[iu][touch]
        self.bond_alive = np.ones(len(self.bond_i), dtype=bool)

        self.x_p0 = 0.0
        self.v_p = cfg.platen_speed
        self.time = 0.0
        self.f_platen = 0.0
        self.n_broken = 0

        self.dt = cfg.dt_s
        self._n = n

    # ------------------------------------------------------------------ step
    def step(self) -> None:
        cfg = self.cfg
        n = self._n
        p = self.pos
        v = self.vel
        r = self.radii
        m = self.mass
        F = np.zeros((n, 2))

        # ---- disk-disk contacts (vectorised O(N^2))
        d = p[:, None, :] - p[None, :, :]          # (N,N,2), d[i,j] = p_i - p_j
        dist = np.linalg.norm(d, axis=2)           # (N,N)
        sum_r = r[:, None] + r[None, :]
        delta = sum_r - dist
        iu = np.triu_indices(n, k=1)
        i_idx, j_idx = iu
        dist_p = dist[i_idx, j_idx]
        delta_p = delta[i_idx, j_idx]
        d_p = d[i_idx, j_idx]
        nvec = d_p / np.maximum(dist_p, 1e-12)[:, None]
        vrel = v[i_idx] - v[j_idx]
        vnorm = np.sum(vrel * nvec, axis=1)
        m_eff = m[i_idx] * m[j_idx] / (m[i_idx] + m[j_idx])
        c_n = 2.0 * cfg.damping_ratio * np.sqrt(cfg.kn * m_eff)
        overlap = delta_p > 0.0
        f_n = (cfg.kn * delta_p + c_n * np.maximum(-vnorm, 0.0)) * overlap
        fvec = f_n[:, None] * nvec
        F[i_idx] += fvec
        F[j_idx] -= fvec
        # Coulomb tangential friction
        v_t = vrel - vnorm[:, None] * nvec
        vt_norm = np.linalg.norm(v_t, axis=1)
        f_t_mag = np.where(vt_norm > 1e-12, cfg.mu * np.abs(f_n), 0.0)
        f_t = np.where(vt_norm[:, None] > 1e-12,
                       f_t_mag[:, None] * v_t / np.maximum(vt_norm, 1e-12)[:, None],
                       0.0)
        F[i_idx] += f_t
        F[j_idx] -= f_t

        # ---- sintered bonds
        alive = self.bond_alive
        if alive.any() and cfg.k_bond > 0.0:
            bi = self.bond_i[alive]
            bj = self.bond_j[alive]
            db = p[bi] - p[bj]
            dist_b = np.linalg.norm(db, axis=1)
            n_b = db / np.maximum(dist_b, 1e-12)[:, None]
            strain = dist_b - self.bond_L0[alive]
            f_b = cfg.k_bond * strain
            break_mask = (f_b > 0.0) & (f_b > cfg.f_bond)
            if break_mask.any():
                alive_idx = np.nonzero(alive)[0]
                self.bond_alive[alive_idx[break_mask]] = False
                self.n_broken += int(break_mask.sum())
            fvec = f_b[:, None] * n_b
            F[bi] -= fvec
            F[bj] += fvec

        # ---- gravity
        F[:, 1] -= m * GRAVITY

        # ---- platen force on disks
        x_p = self.x_p0 + self.v_p * self.time
        pen = x_p - (p[:, 0] - r)
        vrel = self.v_p - v[:, 0]
        c_p = 2.0 * cfg.damping_ratio * math.sqrt(cfg.kn * float(m.mean()))
        f_pl = (cfg.kn * pen + c_p * np.maximum(vrel, 0.0)) * (pen > 0.0)
        F[:, 0] += f_pl
        self.f_platen = float(f_pl.sum())

        # ---- integrate (semi-implicit Euler)
        v += (F / m[:, None]) * self.dt
        p += v * self.dt
        # top/bottom walls
        p[:, 1] = np.clip(p[:, 1], r, cfg.box_height - r)
        # keep disks to the right of the platen face
        p[:, 0] = np.maximum(p[:, 0], x_p + r)
        self.time += self.dt

    def run(self, n_steps: int, record_every: int = 10) -> dict:
        fx = []
        tt = []
        for s in range(n_steps):
            self.step()
            if s % record_every == 0:
                fx.append(self.f_platen)
                tt.append(self.time)
        fx = np.asarray(fx)
        return {
            "t": np.asarray(tt),
            "f_platen": fx,
            "n_broken": self.n_broken,
            "n_bonds": int(len(self.bond_i)),
            "resistance_mean_n": float(fx.mean()) if len(fx) else 0.0,
            "resistance_peak_n": float(fx.max()) if len(fx) else 0.0,
        }


def run_platen(cfg: PlatenConfig) -> dict:
    """Run the platen transit and return the resistance summary + history."""
    dem = PlatenDEM(cfg)
    n_steps = int(round(cfg.duration_s / dem.dt))
    out = dem.run(n_steps, record_every=max(1, int(round(0.02 / dem.dt))))
    out["resistance_stress_pa"] = out["resistance_mean_n"] / (cfg.box_height * cfg.thickness_m)
    out["dt"] = dem.dt
    out["n_steps"] = n_steps
    return out
