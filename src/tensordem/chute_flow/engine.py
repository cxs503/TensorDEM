"""Periodic inclined-plane (chute) DEM engine on top of the shared contact kernel.

Geometry
--------
* ``x``  flow direction, ``y``  spanwise -- **periodic**
* ``z``  wall normal, gravity tilted by ``theta`` from the wall normal
* rough base: a set of *fixed* spheres of the same radius (see
  :func:`tensordem.chute_flow.spec.make_base`), handled as immobile particles in
  the pair list so that the grain-base contact law is literally the same code as
  the grain-grain contact law (no separate wall model, no unit mismatch).

Numerics
--------
Velocity-Verlet with the dissipative (velocity dependent) forces evaluated at
the half-step velocity, i.e. the standard predictor half-kick:

``v* = v + dt/2 (F/m + g)``; ``x' = x + dt v*``; ``F' = F(x', v*)``;
``v' = v* + dt/2 (F'/m + g)``.

The tangential/rolling springs are advanced exactly once per step inside
:func:`tensordem.granular_column.kernel.pair_contact_forces`, so the per-step
slip residual is the physically dissipated work.  Minimum image is applied to
the pair vector through the kernel's ``box=`` argument, and candidate pairs are
found by a 3x3 image tiling of the periodic cell.

The contact law is ``contact_mode="reference"`` -- the linear spring-dashpot
convention of :mod:`tensordem.granular3d_core` (tension cutoff at zero), which
is the convention reproduced by LIGGGHTS ``pair_style gran/hooke/history``
apart from the one-sided dashpot (documented spec difference).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from ..granular3d_core import Material3D
from ..granular_column.kernel import PairSpringState, pair_contact_forces
from .spec import ChuteCase

__all__ = ["ChuteEngine", "periodic_pair_candidates"]


def periodic_pair_candidates(pos: np.ndarray, n_mobile: int,
                             box: tuple[float, float], rcut: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Candidate pairs ``(i, j), i < j`` within ``rcut`` under minimum image.

    ``pos`` stacks the mobile particles first (``0..n_mobile-1``) and the fixed
    base spheres after them.  Base-base pairs are dropped (both immobile), and
    each unordered pair is returned once.
    """
    n_all = pos.shape[0]
    tiled = np.empty((9 * n_all, 3), dtype=np.float64)
    k = 0
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            block = tiled[k:k + n_all]
            block[:] = pos
            if sx:
                block[:, 0] += sx * box[0]
            if sy:
                block[:, 1] += sy * box[1]
            k += n_all
    tree = cKDTree(tiled)
    pairs = tree.query_pairs(rcut, output_type="ndarray")
    if pairs.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    a, b = pairs[:, 0], pairs[:, 1]
    centre = 4 * n_all                      # central tile occupies [4N, 5N)
    keep = (a >= centre) & (a < centre + n_all)
    a, b = a[keep], b[keep]
    i = a - centre
    j = b % n_all
    act = i != j
    i, j = i[act], j[act]
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    key = lo.astype(np.int64) * n_all + hi
    _, uniq = np.unique(key, return_index=True)
    lo, hi = lo[uniq], hi[uniq]
    m = (lo < n_mobile) | (hi < n_mobile)
    return lo[m].astype(np.int64), hi[m].astype(np.int64)


class ChuteEngine:
    """Stateful chute-flow integrator (numpy, no torch dependency)."""

    def __init__(self, case: ChuteCase, init_pos: np.ndarray,
                 base_pos: np.ndarray, *, contact_mode: str = "reference"):
        self.case = case
        self.mat = Material3D(radius=case.radius, density=case.density,
                              mu=case.mu, mu_rolling=case.mu_rolling,
                              restitution=case.restitution, kn=case.kn,
                              kt_ratio=case.kt_ratio, gravity=case.gravity)
        self.contact_mode = contact_mode
        self.n = init_pos.shape[0]
        self.pos = np.vstack([init_pos, base_pos]).astype(np.float64)
        self.vel = np.zeros_like(self.pos)
        self.omega = np.zeros_like(self.pos)
        self.box = (case.lx, case.ly)
        self.gvec = np.asarray(case.gravity_vector, dtype=np.float64)
        self.mass = case.mass
        self.inertia = case.inertia
        self.dt = case.dt
        self.t = 0.0
        self.step_count = 0
        self.state = PairSpringState()
        self.npairs = 0
        self._res = self._forces()

    # ---------------- internals ----------------
    def _forces(self) -> dict:
        iu, ju = periodic_pair_candidates(self.pos, self.n, self.box,
                                          2.0 * self.case.radius + 1e-12)
        self.npairs = int(iu.size)
        res = pair_contact_forces(self.pos, self.vel, self.omega, iu, ju,
                                  self.state, self.mat, self.dt,
                                  update_state=True,
                                  contact_mode=self.contact_mode,
                                  box=self.box)
        return res

    # ---------------- integration ----------------
    def step(self) -> None:
        n, dt = self.n, self.dt
        inv_m = 1.0 / self.mass
        inv_i = 1.0 / self.inertia
        self.vel[:n] += 0.5 * dt * (self._res["force"][:n] * inv_m + self.gvec)
        self.omega[:n] += 0.5 * dt * self._res["torque"][:n] * inv_i
        self.pos[:n] += dt * self.vel[:n]
        self.pos[:n, 0] %= self.box[0]
        self.pos[:n, 1] %= self.box[1]
        self._res = self._forces()
        self.vel[:n] += 0.5 * dt * (self._res["force"][:n] * inv_m + self.gvec)
        self.omega[:n] += 0.5 * dt * self._res["torque"][:n] * inv_i
        self.t += dt
        self.step_count += 1

    def run(self, n_steps: int) -> None:
        for _ in range(int(n_steps)):
            self.step()

    # ---------------- diagnostics ----------------
    def diagnostics(self) -> dict:
        v = self.vel[:self.n]
        speed2 = np.einsum("ij,ij->i", v, v)
        z = self.pos[:self.n, 2]
        overlap = 0.0
        if self._res["ia"].size:
            ia = self._res["ia"]
            dist = np.zeros(0)
            overlap = float(np.max(2.0 * self.case.radius -
                                   np.linalg.norm(
                                       self.pos[self._res["ja"]] -
                                       self.pos[ia], axis=1)))
        return {
            "t": self.t,
            "t_over_tref": self.t / self.case.t_ref,
            "steps": self.step_count,
            "n_contacts": int(self._res["n_contacts"]),
            "v_rms_over_vref": float(math.sqrt(speed2.mean()) / self.case.v_ref),
            "vx_mean_over_vref": float(v[:, 0].mean() / self.case.v_ref),
            "ke_per_particle_over_mgd": float(
                0.5 * self.mass * speed2.mean() /
                (self.mass * self.case.gravity * self.case.d)),
            "z95_over_d": float(np.percentile(z, 95) / self.case.d),
            "zmax_over_d": float(z.max() / self.case.d),
            "max_overlap_over_d": overlap / self.case.d,
        }

    def snapshot(self) -> dict:
        """Particle state in the tool-neutral dump schema."""
        v = self.vel[:self.n]
        return {
            "t": self.t,
            "x": self.pos[:self.n, 0].copy(),
            "y": self.pos[:self.n, 1].copy(),
            "z": self.pos[:self.n, 2].copy(),
            "vx": v[:, 0].copy(),
            "vy": v[:, 1].copy(),
            "vz": v[:, 2].copy(),
        }