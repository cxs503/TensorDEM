"""Array-native sphere-pack DEM for large-N heap / column-collapse studies.

Why this module exists (P165-W2-N06, criterion (f))
---------------------------------------------------
:class:`tensordem.superquadric_dem.SQSystem` keeps one Python object per body.
That design is right for mixed superquadric/clump scenes, but at N >= 10^4 the
per-body attribute traffic (state read/write, pair-history bookkeeping) costs
more than the contact algebra itself: measured 0.30 s/step at N = 2 048 and
0.88 s/step at N = 4 000, i.e. a *particle-number convergence study up to
N = 32 768 is not reachable* in a 2 h job.  This module is the array-native
engine for the exact-sphere subset:

* state is ``(N, 3)`` / ``(N,)`` numpy arrays (no Python body objects),
* candidate pairs come from a Verlet neighbour list (KD-tree, rebuilt only when
  a particle has moved further than the skin), so the neighbour search is
  amortised over ~10 steps,
* the pair history (Mindlin tangential displacement) is one ``(P, 3)`` array
  aligned with the neighbour list instead of a Python dict,
* rotations are not integrated: for a *sphere* the contact geometry does not
  depend on orientation and the inertia tensor is isotropic, so the world-frame
  angular velocity obeys ``dω/dt = τ/I`` directly.  This is exact, not an
  approximation (verified against :class:`SQSystem` in
  ``tests/test_sphere_pack_dem.py``, agreement ~1e-14 over 200 steps).

The force law is *not* re-derived here: normal (Hertz / flat-punch + dashpot),
Mindlin tangential and Coulomb rolling resistance are the validated
:mod:`tensordem.superquadric_kernel` functions, called with the same arguments
as ``SQSystem._sq_sq_contacts_spheres`` / ``SQSystem._wall_contacts``.  The
velocity-Verlet (KDK) scheme with contact-force half-step position correction
mirrors ``SQSystem._step_velocity_verlet``.  Equivalence is a test, not a claim.

Wall convention is the parent's: the half-space kept free of particles is
``n . x >= offset``; a wall contact carries *no* rolling resistance (parent
behaviour, pinned by the parent test suite).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .superquadric_kernel import (
    ContactMaterial,
    mindlin_tangential_update,
    normal_force,
    rolling_resistance_torque,
)

GRAVITY = np.array([0.0, 0.0, -9.81])
DTYPE = np.float64


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SphereWall:
    """Planar half-space ``normal . x >= offset`` with Coulomb friction.

    Same convention as :class:`tensordem.superquadric_dem.Wall`; ``active`` is
    switched off to release a gate without rebuilding the system.
    """
    normal: np.ndarray
    offset: float = 0.0
    mu: Optional[float] = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=DTYPE)
        self.normal = self.normal / float(np.linalg.norm(self.normal))


@dataclass
class SpherePackConfig:
    material: ContactMaterial = field(default_factory=ContactMaterial)
    gravity: np.ndarray = field(default_factory=lambda: GRAVITY.copy())
    dt: Optional[float] = None
    safety_factor: float = 0.05
    reference_overlap: float = 1e-3
    integrator: str = "velocity_verlet"     # "velocity_verlet" | "euler" (negative control)
    velocity_predictor: bool = True
    skin_factor: float = 0.35               # Verlet skin, in units of the max radius
    rebuild_every: int = 12                 # max steps between neighbour rebuilds
    damping_global: float = 0.0             # 1/s background viscous drag (quasi-static)
    damping_rotational: float = 0.0         # 1/s background spin damping
    spin_slip: bool = False                 # True: Cundall-Strack slip at the *contact
    #                                         point* (includes the omega x r term).
    #                                         False reproduces the parent sphere path,
    #                                         which uses centre velocities only and
    #                                         therefore cannot see that a rolling
    #                                         grain has already stopped slipping
    #                                         (measured: unbounded spin-up, w -> 600
    #                                         rad/s).  See the P165-W2-N06 report.


def estimate_dt(radii: Sequence[float], density: float,
                material: ContactMaterial,
                safety_factor: float = 0.05,
                reference_overlap: float = 1e-3) -> float:
    """Stability step from the stiffest reference contact.

    Identical formula to :meth:`tensordem.superquadric_dem.SQSystem.estimate_dt`
    with ``r_ref`` the smallest body length, so the fast engine and the parent
    run the same time step for the same packing.
    """
    r_ref = float(np.min(radii))
    m_min = density * 4.0 / 3.0 * math.pi * r_ref ** 3
    delta = reference_overlap * 2.0 * r_ref
    k_hertz = (4.0 / 3.0) * material.e_star * math.sqrt(r_ref / 2.0) * math.sqrt(delta)
    # area-equivalent flat-punch stiffness (same as flat_punch / SQSystem)
    r_area = math.sqrt(math.pi * r_ref ** 2 / math.pi)
    k_flat = 2.0 * material.e_star * r_area       # punch stiffness = 2 E* a
    k_max = max(k_hertz, k_flat, 1e-12)
    return float(safety_factor * 2.0 * math.pi * math.sqrt(m_min / k_max))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SpherePack:
    """Monodisperse or polydisperse sphere pack, velocity-Verlet, pairwise DEM."""

    def __init__(self, positions: np.ndarray, radii: np.ndarray, density: float,
                 walls: Sequence[SphereWall], config: Optional[SpherePackConfig] = None,
                 fixed: Optional[np.ndarray] = None) -> None:
        pos = np.array(positions, dtype=DTYPE, order="C")
        rad = np.array(radii, dtype=DTYPE, order="C")
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must be (N, 3)")
        if rad.shape == ():
            rad = np.full(len(pos), float(rad))
        if len(rad) != len(pos):
            raise ValueError("radii must be scalar or one per body")
        self.config = config or SpherePackConfig()
        self.material = self.config.material
        self.gravity = np.asarray(self.config.gravity, dtype=DTYPE)
        self.pos = pos
        self.vel = np.zeros_like(pos)
        self.omega = np.zeros_like(pos)
        self.rad = rad
        self.fixed = np.zeros(len(pos), dtype=bool) if fixed is None \
            else np.asarray(fixed, dtype=bool)
        self.moving = ~self.fixed
        self.density = float(density)
        self.mass = self.density * 4.0 / 3.0 * math.pi * rad ** 3
        # isotropic sphere inertia 2/5 m R^2 (exact sphere limit of the kernel)
        self.inertia = 0.4 * self.mass * rad ** 2
        self.walls: List[SphereWall] = list(walls)
        self.time = 0.0
        self.n_steps = 0
        self.max_penetration_ratio = 0.0
        self.dt = float(self.config.dt) if self.config.dt else estimate_dt(
            rad, self.density, self.material,
            safety_factor=self.config.safety_factor,
            reference_overlap=self.config.reference_overlap)
        self._candidate_i = np.zeros(0, dtype=np.int64)
        self._candidate_j = np.zeros(0, dtype=np.int64)
        self._candidate_keys = np.zeros(0, dtype=np.int64)
        self._xi = np.zeros((0, 3), dtype=DTYPE)
        self._cand_age = 10 ** 9
        self._last_pos = None
        self._build_neighbour_list()

    # -- neighbour list -----------------------------------------------------
    def _build_neighbour_list(self) -> None:
        """Verlet candidate pairs, built per radius class.

        The cutoff must not be taken from the *largest* body when a drive body
        (indenter, wall sphere, mill charge) is much larger than the grains:
        a single global ``2 * r_max`` cutoff then admits every grain within a
        drive-body diameter, i.e. O(N) candidates per grain and an effectively
        O(N^2) force loop (measured: 0.014 s/step instead of 0.0036 s/step at
        N = 1260 with R/d = 3).  Free-free pairs keep the global grain cutoff;
        each fixed body gets its own ``r_body + r_grain`` cutoff.  The
        candidate set is still a superset of the true contact set, so the
        physics is untouched (pinned by ``tests/test_sphere_pack_dem.py``).
        """
        from scipy.spatial import cKDTree

        n = len(self.pos)
        if n < 2:
            self._candidate_i = np.zeros(0, dtype=np.int64)
            self._candidate_j = np.zeros(0, dtype=np.int64)
            self._candidate_keys = np.zeros(0, dtype=np.int64)
            self._xi = np.zeros((0, 3), dtype=DTYPE)
            return
        skin = float(self.config.skin_factor)
        free = self.moving
        parts = []
        idx_free = np.nonzero(free)[0]
        if len(idx_free):
            r_free = float(np.max(self.rad[idx_free]))
            tree_free = cKDTree(self.pos[idx_free])
            pf = tree_free.query_pairs(2.0 * r_free * (1.0 + skin) + 1e-15,
                                       output_type="ndarray")
            if len(pf):
                parts.append(np.column_stack([idx_free[pf[:, 0]],
                                              idx_free[pf[:, 1]]]))
            for f in np.nonzero(self.fixed)[0]:
                rng_f = (float(self.rad[f]) + r_free) * (1.0 + skin) + 1e-15
                nb = np.asarray(tree_free.query_ball_point(self.pos[f], rng_f),
                                dtype=np.int64)
                if len(nb):
                    parts.append(np.column_stack(
                        [np.full(len(nb), int(f), dtype=np.int64), idx_free[nb]]))
        if not parts:
            self._candidate_i = np.zeros(0, dtype=np.int64)
            self._candidate_j = np.zeros(0, dtype=np.int64)
            self._candidate_keys = np.zeros(0, dtype=np.int64)
            self._xi = np.zeros((0, 3), dtype=DTYPE)
            self._cand_age = 0
            self._last_pos = self.pos.copy()
            return
        pairs = np.vstack(parts)
        pairs.sort(axis=1)                       # canonical i < j ordering
        i = pairs[:, 0].astype(np.int64)
        j = pairs[:, 1].astype(np.int64)
        keys = i * n + j
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        i = i[order]
        j = j[order]
        xi = np.zeros((len(keys), 3), dtype=DTYPE)
        if len(self._candidate_keys):
            # carry the tangential history across the rebuild
            loc = np.searchsorted(self._candidate_keys, keys)
            loc_c = np.clip(loc, 0, len(self._candidate_keys) - 1)
            hit = self._candidate_keys[loc_c] == keys
            xi[hit] = self._xi[loc_c[hit]]
        self._candidate_keys = keys
        self._candidate_i = i
        self._candidate_j = j
        self._xi = xi
        self._cand_age = 0
        self._last_pos = self.pos.copy()

    def _neighbour_refresh_due(self) -> bool:
        if self._cand_age >= self.config.rebuild_every or self._last_pos is None:
            return True
        moved = np.linalg.norm(self.pos - self._last_pos, axis=1)
        # skin of the free-free class: a large fixed body must not relax
        # this criterion (it would allow grains to move further than the
        # grain-grain candidate margin before a rebuild).
        r_scale = float(np.max(self.rad[self.moving])) if np.any(self.moving) \
            else float(np.max(self.rad))
        skin_half = 0.5 * self.config.skin_factor * r_scale
        return bool(np.max(moved) > skin_half)

    # -- forces -------------------------------------------------------------
    def _wall_contacts(self, force: np.ndarray, torque: np.ndarray) -> None:
        mat = self.material
        n = len(self.pos)
        # One (N, 3) history array per wall rather than a dict keyed by particle
        # id: the dict version needs one python-level get/set per contacting
        # particle per wall per *force evaluation*, which is O(1e3) interpreted
        # operations in a settled bed and made it the dominant per-step cost
        # (measured: 20 ms/step at N = 1944, ~90 % of it in this loop).  The
        # arithmetic below is unchanged -- only the storage of xi.
        hist = getattr(self, "_wall_xi", None)
        if hist is None or len(hist) != len(self.walls) \
                or (len(hist) and hist[0].shape[0] != n):
            hist = [np.zeros((n, 3), dtype=DTYPE) for _ in self.walls]
            self._wall_xi = hist
        for wi, wall in enumerate(self.walls):
            if not wall.active:
                continue
            mu = mat.mu if wall.mu is None else wall.mu
            nrm = wall.normal
            gap = self.pos @ nrm - wall.offset - self.rad
            sel = np.nonzero((gap < 0.0) & self.moving)[0]
            if len(sel) == 0:
                continue
            delta = -gap[sel]
            r_s = self.rad[sel]
            p_world = self.pos[sel] - r_s[:, None] * nrm
            r_vec = p_world - self.pos[sel]
            v_contact = self.vel[sel] + np.cross(self.omega[sel], r_vec)
            v_n = (v_contact - wall.velocity) @ nrm
            f_n, k_n, branch = normal_force(
                delta, r_s, np.zeros(len(sel)), v_n, self.mass[sel], mat)
            store = hist[wi]
            xi = store[sel]
            v_rel = v_contact - wall.velocity
            v_t = v_rel - np.sum(v_rel * nrm, axis=1)[:, None] * nrm
            f_t, xi_new, _ = mindlin_tangential_update(
                f_n, 0.5 * np.maximum(k_n, 1e-12), np.full(len(sel), mu), xi,
                v_t=v_t, dt=self.dt, normal=np.broadcast_to(nrm, v_t.shape))
            store[sel] = xi_new
            f_vec = f_n[:, None] * nrm + f_t
            self._scatter(force, sel, f_vec)
            # NOTE: no rolling resistance at wall contacts -- parent behaviour
            self._scatter(torque, sel, np.cross(r_vec, f_vec))
            ratio = delta / np.maximum(2.0 * r_s, 1e-30)
            self.max_penetration_ratio = max(self.max_penetration_ratio,
                                             float(np.max(ratio)))

    @staticmethod
    def _scatter(target: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
        n = target.shape[0]
        for d in range(3):
            target[:, d] += np.bincount(idx, weights=vals[:, d], minlength=n)

    def contact_forces(self):
        """Assemble contact forces/torques for the current state."""
        if self._neighbour_refresh_due():
            self._build_neighbour_list()
        mat = self.material
        n = len(self.pos)
        force = np.zeros_like(self.pos)
        torque = np.zeros_like(self.pos)
        n_contacts = 0
        if len(self._candidate_keys):
            i = self._candidate_i
            j = self._candidate_j
            d = self.pos[j] - self.pos[i]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            r_sum = self.rad[i] + self.rad[j]
            delta_all = r_sum - dist
            act = np.nonzero(delta_all > 0.0)[0]
            if len(act):
                ii, jj = i[act], j[act]
                dist_a = dist[act]
                normal = d[act] / dist_a[:, None]
                delta = delta_all[act]
                r_eff = self.rad[ii] * self.rad[jj] / r_sum[act]
                v_rel = self.vel[jj] - self.vel[ii]
                v_n = np.einsum("ij,ij->i", v_rel, normal)
                if self.config.spin_slip:
                    # contact-point velocities: v + omega x (r n) -- the spin term
                    # is tangential, so v_n and the normal force are unchanged
                    v_rel = v_rel + np.cross(self.omega[jj], -self.rad[jj, None] * normal) \
                        - np.cross(self.omega[ii], self.rad[ii, None] * normal)
                    v_t_slip = v_rel - np.einsum("ij,ij->i", v_rel, normal)[:, None] * normal
                else:
                    v_t_slip = v_rel - v_n[:, None] * normal
                m_eff = self.mass[ii] * self.mass[jj] / (self.mass[ii] + self.mass[jj])
                f_n, k_n, branch = normal_force(
                    delta, r_eff, np.zeros_like(delta), v_n, m_eff, mat)
                xi = self._xi[act]
                f_t, xi_new, _ = mindlin_tangential_update(
                    f_n, 0.5 * np.maximum(k_n, 1e-12),
                    np.full(len(act), mat.mu), xi,
                    v_t=v_t_slip, dt=self.dt, normal=normal)
                self._xi[act] = xi_new
                f_vec = f_n[:, None] * normal + f_t
                self._scatter(force, ii, -f_vec)
                self._scatter(force, jj, f_vec)
                r_b = np.minimum(self.rad[ii], self.rad[jj])
                # torque from the tangential force acts at the contact point on
                # the sphere surface: identical to np.cross(pa - x_i, -f_vec)
                torque_i = -np.cross(self.rad[ii, None] * normal, f_vec)
                torque_j = np.cross(self.rad[jj, None] * normal, f_vec)
                m_roll = rolling_resistance_torque(
                    f_n, r_eff, self.omega[jj] - self.omega[ii],
                    np.full(len(act), mat.mu_rolling), r_b)
                self._scatter(torque, ii, torque_i - m_roll)
                self._scatter(torque, jj, torque_j + m_roll)
                ratio = delta / np.maximum(2.0 * r_b, 1e-30)
                self.max_penetration_ratio = max(self.max_penetration_ratio,
                                                 float(np.max(ratio)))
                n_contacts += int(len(act))
        self._wall_contacts(force, torque)
        self.n_contacts = n_contacts
        return force, torque

    # -- integration --------------------------------------------------------
    def _angular_accel(self, torque: np.ndarray) -> np.ndarray:
        alpha = torque / self.inertia[:, None]
        if self.config.damping_rotational > 0.0:
            alpha = alpha - self.config.damping_rotational * self.omega
        return alpha

    def _translate_accel(self, force: np.ndarray) -> np.ndarray:
        f = force
        if self.config.damping_global > 0.0:
            f = f - self.config.damping_global * self.mass[:, None] * self.vel
        return f / self.mass[:, None] + self.gravity

    def step(self) -> None:
        dt = self.dt
        mod = self.config.integrator
        if mod == "euler":
            return self._step_euler()
        if mod not in ("velocity_verlet",):
            raise ValueError(f"unknown integrator {mod!r}")
        force0, torque0 = self.contact_forces()
        a0 = self._translate_accel(force0)
        a0[self.fixed] = 0.0
        alpha0 = self._angular_accel(torque0)
        alpha0[self.fixed] = 0.0
        v_half = self.vel + 0.5 * dt * a0
        omega_half = self.omega + 0.5 * dt * alpha0
        pos1 = self.pos + dt * v_half
        v_eval = v_half + 0.5 * dt * a0 if self.config.velocity_predictor else v_half
        # forces at the drifted position (the "half-step position correction")
        self.pos = pos1
        self.vel = v_eval
        force1, torque1 = self.contact_forces()
        a1 = self._translate_accel(force1)
        a1[self.fixed] = 0.0
        alpha1 = self._angular_accel(torque1)
        alpha1[self.fixed] = 0.0
        self.vel = v_half + 0.5 * dt * a1
        self.omega = omega_half + 0.5 * dt * alpha1
        self.time += dt
        self.n_steps += 1

    def _step_euler(self) -> None:
        force, torque = self.contact_forces()
        dt = self.dt
        accel = self._translate_accel(force)
        accel[self.fixed] = 0.0
        alpha = self._angular_accel(torque)
        alpha[self.fixed] = 0.0
        self.vel = self.vel + dt * accel
        self.pos = self.pos + dt * self.vel
        self.omega = self.omega + dt * alpha
        self.time += dt
        self.n_steps += 1

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    # -- measurement --------------------------------------------------------
    def positions(self) -> np.ndarray:
        return self.pos.copy()

    def energy(self) -> float:
        """Translational + rotational kinetic energy plus gravitational potential."""
        kinetic = 0.5 * float(np.sum(self.mass * np.einsum("ij,ij->i", self.vel, self.vel))) \
            + 0.5 * float(np.sum(self.inertia[:, None] * self.omega ** 2))
        potential = -float(np.sum(self.mass * (self.pos @ self.gravity)))
        return kinetic + potential

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self.mass * np.einsum("ij,ij->i", self.vel, self.vel))) \
            + 0.5 * float(np.sum(self.inertia[:, None] * self.omega ** 2))