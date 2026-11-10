"""NSCD (Non-Smooth Contact Dynamics) / Moreau time stepping for rigid spheres.

Motivation
----------
A soft-sphere (penalty) DEM resolves a contact through a compliant spring, so
the geometric overlap ``delta`` is a *state variable*: ``delta = F_n/k_n``.  In a
rigid (``phi = 0.85``) packing under load the overlap is not a physical quantity
yet it changes the packing geometry, the effective solid fraction and the
momentum path.  NSCD instead treats contacts as unilateral rigid constraints
solved at the impulse level (Moreau 1988; Jean & Moreau 1992; Radjai & Dubois
2011):

* free flight over a step ``dt``:  ``v_free = v + dt (f/m)``
* the contact impulse ``p`` must satisfy the complementarity pair

      0 <= u_n^+ + e u_n^-   _|_   p_n >= 0                  (Newton restitution)
      p_t = argmin_{|p_t| <= mu p_n} |u_t^+ + p_t / m_t|      (Coulomb friction)

  with ``u_n`` the normal relative velocity at the contact, ``u^-``/``u^+`` the
  pre/post-step values and ``u^+ = u^- + W p`` the local velocity update (``W``
  the Delassus operator).
* the velocities are integrated: ``x^+ = x + dt v^+``.

Because the constraint is imposed on the *predicted* gap,
``g_pred = g + dt u_n^- < 0`` (an anticipated contact), and the post-step normal
velocity satisfies ``u_n^+ >= -e u_n^- >= 0``, the flat-wall gap update
``g^+ = g + dt u_n^+ >= 0`` gives **zero penetration by construction**.  For
curved sphere-sphere contacts the linearised gap leaves an ``O(dt^2)`` curvature
remainder which the optional position-level projection removes.

The complementarity problem is solved with **NLGS** (Non-Linear Gauss-Seidel):
sweeps of local projected sub-solves, each followed by the corresponding
velocity update.  Two backends are provided:

``sweep='gauss_seidel'``  sequential contact-by-contact NLGS (textbook; used for
    the verification tiers, where exactness matters);
``sweep='jacobi'``  fully vectorised, all contacts updated from the same
    velocity snapshot (needed for thousand-particle packings).

Units: SI (m, kg, s, N).  A sphere has ``I = 2/5 m r^2`` unless given.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


class NSCDError(ValueError):
    """Base class for NSCD errors."""


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Wall:
    """Planar wall: the material lies where ``side * x[axis] >= side * coord``."""

    axis: int
    coord: float
    side: int = 1                       # +1: material above, -1: material below
    velocity: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise NSCDError("wall axis must be 0, 1 or 2")
        if self.side not in (-1, 1):
            raise NSCDError("wall side must be -1 or +1")

    @property
    def normal(self) -> np.ndarray:
        n = np.zeros(3)
        n[self.axis] = float(self.side)
        return n


@dataclass
class NSCDStepInfo:
    n_candidates: int
    n_active: int
    iterations: int
    impulse_change: float
    max_overlap: float
    kinetic_energy: float
    wall_impulse: np.ndarray


class NSCDWorld:
    """Rigid-sphere NSCD world (translation + rotation, planar walls)."""

    def __init__(self, radius, *, mass=None, density: float = 2500.0,
                 mu: float = 0.5, restitution: float = 0.5,
                 gravity: Sequence[float] = (0.0, 0.0, -9.81),
                 walls: Sequence[Wall] = (), dt: float = 1e-4,
                 tol: float = 1e-10, max_iter: int = 400, sweep: str = "jacobi",
                 position_correction: bool = True, omega_relax: float = 1.0,
                 contact_margin: float = 0.0):
        self.radius = np.atleast_1d(np.asarray(radius, dtype=float))
        self.n = self.radius.size
        if np.any(self.radius <= 0.0):
            raise NSCDError("radii must be positive")
        if mass is None:
            self.mass = density * 4.0 / 3.0 * np.pi * self.radius ** 3
        else:
            self.mass = np.broadcast_to(np.asarray(mass, dtype=float), (self.n,)).copy()
        self.inertia = 0.4 * self.mass * self.radius ** 2
        if mu < 0.0:
            raise NSCDError("friction coefficient must be nonnegative")
        if not 0.0 <= restitution <= 1.0:
            raise NSCDError("restitution must lie in [0, 1]")
        if dt <= 0.0:
            raise NSCDError("dt must be positive")
        if sweep not in ("jacobi", "gauss_seidel", "colored"):
            raise NSCDError("sweep must be 'jacobi', 'gauss_seidel' or 'colored'")
        self.mu = float(mu)
        self.restitution = float(restitution)
        self.gravity = np.asarray(gravity, dtype=float)
        self.walls = list(walls)
        self.dt = float(dt)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.sweep = sweep
        self.position_correction = bool(position_correction)
        self.omega_relax = float(omega_relax)
        self.contact_margin = float(contact_margin)
        self.pos = np.zeros((self.n, 3))
        self.vel = np.zeros((self.n, 3))
        self.omega = np.zeros((self.n, 3))
        self.time = 0.0
        self.last: NSCDStepInfo | None = None
        self.last_wall_impulse = np.zeros((max(len(self.walls), 1), 3))

    # -- state ---------------------------------------------------------
    def set_state(self, pos, vel=None, omega=None) -> None:
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (self.n, 3):
            raise NSCDError(f"positions must have shape ({self.n}, 3)")
        self.pos = pos.copy()
        self.vel = np.zeros((self.n, 3)) if vel is None else np.asarray(vel, dtype=float).copy()
        self.omega = np.zeros((self.n, 3)) if omega is None else np.asarray(omega, dtype=float).copy()

    # -- contact detection ---------------------------------------------
    def _detect(self) -> dict:
        pos, rad, dt = self.pos, self.radius, self.dt
        vmax = float(np.max(np.linalg.norm(self.vel, axis=1))) if self.n else 0.0
        margin = dt * vmax * 1.05 + self.contact_margin + 1e-15
        pairs = cKDTree(pos).query_pairs(2.0 * float(rad.max()) + margin,
                                         output_type="ndarray")
        ii, jj, nrm, gap, wall_idx = [], [], [], [], []
        if pairs.size:
            d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
            dist = np.linalg.norm(d, axis=1)
            ok = (dist < rad[pairs[:, 0]] + rad[pairs[:, 1]] + margin) & (dist > 0.0)
            pairs, d, dist = pairs[ok], d[ok], dist[ok]
            nrm.append(d / dist[:, None])
            gap.append(dist - rad[pairs[:, 0]] - rad[pairs[:, 1]])
            ii.append(pairs[:, 0])
            jj.append(pairs[:, 1])
            wall_idx.append(-np.ones(pairs.shape[0], dtype=int))
        for w, wall in enumerate(self.walls):
            g = wall.side * (pos[:, wall.axis] - wall.coord) - rad
            take = np.nonzero(g < margin)[0]
            if take.size:
                ii.append(take)
                jj.append(-np.ones(take.size, dtype=int))
                # stored normal points from the particle (body i) to the wall
                # (body j) so that all contacts share the same sign convention
                nrm.append(np.repeat((-wall.normal)[None, :], take.size, axis=0))
                gap.append(g[take])
                wall_idx.append(np.full(take.size, w, dtype=int))
        if not ii:
            return dict(ii=np.zeros(0, int), jj=np.zeros(0, int),
                        n=np.zeros((0, 3)), gap=np.zeros(0),
                        wall=np.zeros(0, int))
        return dict(ii=np.concatenate(ii), jj=np.concatenate(jj),
                    n=np.concatenate(nrm), gap=np.concatenate(gap),
                    wall=np.concatenate(wall_idx))

    def _arms(self, c):
        jj = c["jj"]
        arm_i = c["n"] * self.radius[c["ii"]][:, None]
        arm_j = -c["n"] * np.where(jj >= 0, self.radius[np.maximum(jj, 0)], 0.0)[:, None]
        return arm_i, arm_j

    def _rel_vel(self, c, arm_i, arm_j):
        ii, jj, wall = c["ii"], c["jj"], c["wall"]
        v_i = self.vel[ii] + np.cross(self.omega[ii], arm_i)
        safe_j = np.maximum(jj, 0)
        v_j = self.vel[safe_j] + np.cross(self.omega[safe_j], arm_j)
        if wall.size:
            wmask = wall >= 0
            if wmask.any():
                wv = np.array([self.walls[w].velocity for w in wall[wmask]], dtype=float)
                v_j = v_j.copy()
                v_j[wmask] = wv
        return v_j - v_i

    def _masses(self, c):
        jj = c["jj"]
        rad, mass, inertia = self.radius, self.mass, self.inertia
        safe_j = np.maximum(jj, 0)
        has_j = jj >= 0
        inv_mn = 1.0 / mass[c["ii"]] + np.where(has_j, 1.0 / mass[safe_j], 0.0)
        inv_mt = (inv_mn
                  + rad[c["ii"]] ** 2 / inertia[c["ii"]]
                  + np.where(has_j, rad[safe_j] ** 2 / inertia[safe_j], 0.0))
        return 1.0 / inv_mn, 1.0 / inv_mt

    def _apply(self, c, arm_i, arm_j, impulse):
        ii, jj = c["ii"], c["jj"]
        np.add.at(self.vel, ii, -impulse / self.mass[ii][:, None])
        np.add.at(self.omega, ii, np.cross(arm_i, -impulse) / self.inertia[ii][:, None])
        mask = jj >= 0
        if mask.any():
            jjv = jj[mask]
            Jj = impulse[mask]
            np.add.at(self.vel, jjv, Jj / self.mass[jjv][:, None])
            np.add.at(self.omega, jjv, np.cross(arm_j[mask], Jj) / self.inertia[jjv][:, None])

    # -- NLGS solvers ---------------------------------------------------
    def _solve_jacobi(self, c, arm_i, arm_j, un0, m_n, m_t):
        nrm, e, mu = c["n"], self.restitution, self.mu
        p_n = np.zeros(un0.size)
        p_t = np.zeros((un0.size, 3))
        scale = max(float(np.max(m_n * np.abs(un0))), 1e-30)
        change, it = np.inf, 0
        for it in range(1, self.max_iter + 1):
            u = self._rel_vel(c, arm_i, arm_j)
            u_n = np.einsum("ij,ij->i", u, nrm)
            p_n_new = np.maximum(p_n - m_n * (u_n + e * un0), 0.0)
            dp_n = (p_n_new - p_n) * self.omega_relax
            u_t = u - u_n[:, None] * nrm
            p_t_try = p_t - m_t[:, None] * u_t * self.omega_relax
            mag = np.linalg.norm(p_t_try, axis=1)
            lim = mu * p_n_new
            shrink = np.where(mag > lim, lim / np.maximum(mag, 1e-300), 1.0)
            p_t_new = p_t_try * shrink[:, None]
            dp_t = p_t_new - p_t
            self._apply(c, arm_i, arm_j, dp_n[:, None] * nrm + dp_t)
            p_n, p_t = p_n_new, p_t_new
            change = float(np.max(np.abs(np.concatenate(
                [dp_n, np.linalg.norm(dp_t, axis=1)]))))
            if not np.isfinite(change):
                raise NSCDError("NLGS diverged (non-finite impulse); "
                                "reduce omega_relax or use sweep='colored'")
            if change <= self.tol * scale:
                break
        return p_n, p_t, it, change

    @staticmethod
    def _color_contacts(c, n_particles: int) -> np.ndarray:
        """Greedy colouring of the contact graph (no two neighbours share a colour).

        Contacts of one colour touch disjoint particles, so a *vectorised* update
        over a colour is exactly a sequential Gauss-Seidel pass over those
        contacts — this is what makes the sequential NLGS affordable at
        thousand-contact scale.
        """
        ii, jj = c["ii"], c["jj"]
        colors = np.zeros(ii.size, dtype=int)
        used: list = [set() for _ in range(n_particles)]
        for k in range(ii.size):
            i = int(ii[k])
            j = int(jj[k])
            forbidden = used[i] if j < 0 else used[i] | used[j]
            col = 0
            while col in forbidden:
                col += 1
            colors[k] = col
            used[i].add(col)
            if j >= 0:
                used[j].add(col)
        return colors

    def _solve_colored(self, c, arm_i, arm_j, un0, m_n, m_t):
        """Colour-partitioned vectorised NLGS (equivalent to sequential GS)."""
        nrm, e, mu = c["n"], self.restitution, self.mu
        colors = self._color_contacts(c, self.n)
        order = [np.nonzero(colors == col)[0] for col in np.unique(colors)]
        na = un0.size
        p_n = np.zeros(na)
        p_t = np.zeros((na, 3))
        scale = max(float(np.max(m_n * np.abs(un0))), 1e-30)
        change, it = np.inf, 0
        for it in range(1, self.max_iter + 1):
            change = 0.0
            for idx in order:
                ci = {key: (val[idx] if isinstance(val, np.ndarray) else val)
                      for key, val in c.items()}
                ai, aj = arm_i[idx], arm_j[idx]
                u = self._rel_vel(ci, ai, aj)
                u_n = np.einsum("ij,ij->i", u, nrm[idx])
                p_n_new = np.maximum(p_n[idx] - m_n[idx] * (u_n + e * un0[idx]), 0.0)
                dp_n = p_n_new - p_n[idx]
                u_t = u - u_n[:, None] * nrm[idx]
                p_t_try = p_t[idx] - m_t[idx][:, None] * u_t
                mag = np.linalg.norm(p_t_try, axis=1)
                lim = mu * p_n_new
                shrink = np.where(mag > lim, lim / np.maximum(mag, 1e-300), 1.0)
                p_t_new = p_t_try * shrink[:, None]
                dp_t = p_t_new - p_t[idx]
                self._apply(ci, ai, aj, dp_n[:, None] * nrm[idx] + dp_t)
                p_n[idx] = p_n_new
                p_t[idx] = p_t_new
                change = max(change, float(np.max(np.abs(dp_n))) if dp_n.size else 0.0,
                             float(np.max(np.linalg.norm(dp_t, axis=1))) if dp_t.size else 0.0)
            if change <= self.tol * scale:
                break
        return p_n, p_t, it, change

    def _solve_gs(self, c, arm_i, arm_j, un0, m_n, m_t):
        """Sequential (contact-by-contact) non-linear Gauss-Seidel."""
        nrm, e, mu = c["n"], self.restitution, self.mu
        na = un0.size
        p_n = np.zeros(na)
        p_t = np.zeros((na, 3))
        scale = max(float(np.max(m_n * np.abs(un0))), 1e-30)
        change, it = np.inf, 0
        for it in range(1, self.max_iter + 1):
            change = 0.0
            for k in range(na):
                ck = {key: (val[k:k + 1] if isinstance(val, np.ndarray) else val)
                      for key, val in c.items()}
                ai, aj = arm_i[k:k + 1], arm_j[k:k + 1]
                u = self._rel_vel(ck, ai, aj)[0]
                u_n = float(u @ nrm[k])
                p_n_new = max(p_n[k] - m_n[k] * (u_n + e * un0[k]), 0.0)
                dp_n = p_n_new - p_n[k]
                u_t = u - u_n * nrm[k]
                p_t_try = p_t[k] - m_t[k] * u_t
                mag = float(np.linalg.norm(p_t_try))
                lim = mu * p_n_new
                if mag > lim:
                    p_t_try = p_t_try * (lim / max(mag, 1e-300))
                dp_t = p_t_try - p_t[k]
                self._apply(ck, ai, aj, (dp_n * nrm[k] + dp_t)[None, :])
                p_n[k] = p_n_new
                p_t[k] = p_t_try
                change = max(change, abs(dp_n), float(np.linalg.norm(dp_t)))
            if change <= self.tol * scale:
                break
        return p_n, p_t, it, change

    # -- one time step --------------------------------------------------
    def step(self) -> NSCDStepInfo:
        dt = self.dt
        self.vel = self.vel + dt * self.gravity[None, :]
        c = self._detect()
        n_cand = c["ii"].size
        arm_i, arm_j = self._arms(c)
        u = self._rel_vel(c, arm_i, arm_j)
        u_n0 = np.einsum("ij,ij->i", u, c["n"]) if n_cand else np.zeros(0)
        active = (c["gap"] + dt * u_n0 < 0.0) if n_cand else np.zeros(0, dtype=bool)
        wall_imp = np.zeros((max(len(self.walls), 1), 3))
        n_active = int(active.sum())
        it, change, p_n = 0, 0.0, np.zeros(0)
        if n_active:
            idx = np.nonzero(active)[0]
            ca = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in c.items()}
            ai, aj = arm_i[idx], arm_j[idx]
            un0 = u_n0[idx]
            m_n, m_t = self._masses(ca)
            solver = {"gauss_seidel": self._solve_gs,
                      "colored": self._solve_colored}.get(self.sweep,
                                                          self._solve_jacobi)
            p_n, p_t, it, change = solver(ca, ai, aj, un0, m_n, m_t)
            # total impulse per contact (for wall force bookkeeping)
            J = p_n[:, None] * ca["n"] + p_t
            wmask = ca["wall"] >= 0
            for w in np.unique(ca["wall"][wmask]):
                wall_imp[int(w)] += J[(ca["wall"] == w)].sum(axis=0)
        self.pos = self.pos + dt * self.vel
        if self.position_correction:
            self._project_positions()
        info = NSCDStepInfo(n_candidates=n_cand, n_active=n_active,
                            iterations=int(it), impulse_change=float(change),
                            max_overlap=self._max_overlap(),
                            kinetic_energy=self.total_kinetic_energy(),
                            wall_impulse=wall_imp)
        self.last = info
        self.last_wall_impulse = wall_imp
        self.time += dt
        return info

    # -- diagnostics ----------------------------------------------------
    def _max_overlap(self) -> float:
        if not self.n:
            return 0.0
        worst = 0.0
        pairs = cKDTree(self.pos).query_pairs(2.0 * float(self.radius.max()),
                                              output_type="ndarray")
        if pairs.size:
            d = np.linalg.norm(self.pos[pairs[:, 1]] - self.pos[pairs[:, 0]], axis=1)
            worst = float(np.max(self.radius[pairs[:, 0]] + self.radius[pairs[:, 1]] - d))
        for wall in self.walls:
            g = wall.side * (self.pos[:, wall.axis] - wall.coord) - self.radius
            worst = max(worst, float(-np.min(g)))
        return max(worst, 0.0)

    def _project_positions(self, n_iter: int = 4) -> None:
        """Remove residual overlaps by splitting them along the contact normal."""
        for _ in range(n_iter):
            moved = False
            pairs = cKDTree(self.pos).query_pairs(2.0 * float(self.radius.max()),
                                                  output_type="ndarray")
            if pairs.size:
                d = self.pos[pairs[:, 1]] - self.pos[pairs[:, 0]]
                dist = np.linalg.norm(d, axis=1)
                good = dist > 0.0
                pairs, d, dist = pairs[good], d[good], dist[good]
                overlap = self.radius[pairs[:, 0]] + self.radius[pairs[:, 1]] - dist
                hit = overlap > 0.0
                if hit.any():
                    ii, jj = pairs[hit, 0], pairs[hit, 1]
                    nvec = d[hit] / dist[hit][:, None]
                    corr = overlap[hit][:, None] * nvec
                    np.add.at(self.pos, ii, -0.5 * corr)
                    np.add.at(self.pos, jj, 0.5 * corr)
                    moved = True
            for wall in self.walls:
                g = wall.side * (self.pos[:, wall.axis] - wall.coord) - self.radius
                pen = g < 0.0
                if pen.any():
                    self.pos[pen, wall.axis] += wall.side * (-g[pen])
                    moved = True
            if not moved:
                return

    def run(self, steps: int, callback=None) -> list:
        out = []
        for _ in range(int(steps)):
            info = self.step()
            out.append(info)
            if callback is not None:
                callback(self, info)
        return out

    # -- aggregate observables -----------------------------------------
    def total_momentum(self) -> np.ndarray:
        return (self.mass[:, None] * self.vel).sum(axis=0)

    def total_kinetic_energy(self) -> float:
        return float(np.sum(0.5 * self.mass * np.sum(self.vel ** 2, axis=1)
                            + 0.5 * self.inertia * np.sum(self.omega ** 2, axis=1)))

    def wall_force(self, wall_index: int) -> np.ndarray:
        """Impulse rate exerted by the material on wall ``wall_index`` (N)."""
        last = self.last
        if last is None:
            raise NSCDError("step() must be called before wall_force()")
        return last.wall_impulse[wall_index] / self.dt


__all__ = ["NSCDError", "Wall", "NSCDWorld", "NSCDStepInfo"]