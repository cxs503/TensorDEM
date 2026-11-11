"""Minimal rigid-body clump DEM engine plus the two macroscopic protocols.

Engine
------
:class:`ClumpDEM` integrates rigid clumps (translation + quaternion rotation)
with member-sphere contacts:

* narrow phase: KD-tree member-pair candidates -> junction merging
  (:func:`tensordem.contact.clump_contact.member_pair_candidates` /
  :func:`~tensordem.contact.clump_contact.merge_junction_candidates`), plus
  member/box contacts for the floor, walls and piston;
* force law: linear spring-dashpot normal + Cundall-Strack tangential +
  rolling-resistance torque (:func:`~tensordem.contact.clump_contact.evaluate_contact_law`);
* semi-implicit Euler for translation, body-frame quaternion update for rotation;
* per-step energy bookkeeping (:class:`~tensordem.contact.clump_contact.EnergyLedger`),
  including the tangential energy dropped when a contact detaches.

Protocols
---------
``lifted_box_repose``
    Standard angle-of-repose measurement: a bed is settled inside a box that is
    then lifted vertically, the grains flow out and form a heap; the repose angle
    is read from the heap silhouette (convex hull of ``(|x|, z)``) and
    cross-checked with a fit to the upper-envelope profile.  A quasi-2D slab
    (frictionless side walls) is used so the flank is a clean line and the run
    stays affordable.

``piston_compression``
    A settled bed is compressed by a slowly moving piston; the contact-force
    distribution and force-chain topology are recorded at the end of the stroke
    together with the full energy balance (external work vs kinetic + elastic +
    dissipated).  This is the protocol behind the C2 (force-chain width) and C6
    (energy) gates.

Both protocols take a *particle factory* (``Callable[[], ClumpParticle]``) so
that the identical code path handles the spherical baseline (a one-member clump)
and the non-spherical clump -- that is what makes the comparison controlled.

Units: SI.  The parameters used by the validation face are pipeable from
``scripts/clump_polyhedron_study.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from ..contact.clump_contact import (
    BoxObstacle,
    ContactHistory,
    ContactParams,
    EnergyLedger,
    evaluate_contact_law,
    force_chain_clusters,
    force_distribution_stats,
    member_pair_candidates,
    merge_junction_candidates,
    sphere_box_contacts,
)
from ..contact import rolling_static
from .clump import ClumpParticle, quat_integrate, quat_to_matrix

__all__ = [
    "ClumpDEM",
    "lattice_packing",
    "heap_case",
    "lifted_box_repose",
    "measure_repose_angle",
    "piston_case",
    "piston_compression",
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ClumpDEM:
    """Rigid multi-sphere DEM with axis-aligned box obstacles."""

    def __init__(
        self,
        particles: Sequence[ClumpParticle],
        positions: Optional[np.ndarray] = None,
        quaternions: Optional[np.ndarray] = None,
        obstacles: Sequence[BoxObstacle] = (),
        *,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        params: Optional[ContactParams] = None,
        dt: float = 5e-5,
    ) -> None:
        self.particles = list(particles)
        n = len(self.particles)
        if n == 0:
            raise ValueError("the simulation needs at least one particle")
        self.n_bodies = n
        self.obstacles = list(obstacles)
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        self.params = params or ContactParams()
        self.dt = float(dt)

        self.local_centers = np.concatenate([p.local_centers for p in self.particles])
        self.member_radius = np.concatenate([p.radii for p in self.particles])
        self.body_of_member = np.concatenate(
            [np.full(p.n_members, k, dtype=int) for k, p in enumerate(self.particles)]
        )
        self.n_members = len(self.member_radius)
        # member mass for the contact effective mass (member volume split -- an
        # approximation for overlapping members, documented in the study notes)
        dens = np.concatenate([np.full(p.n_members, p.density) for p in self.particles])
        self.member_mass = dens * 4.0 / 3.0 * math.pi * self.member_radius ** 3

        self.mass = np.array([p.mass for p in self.particles])
        self.inv_mass = 1.0 / self.mass
        self.inertia_body = np.stack([p.inertia for p in self.particles])
        self.inv_inertia_body = np.linalg.inv(self.inertia_body)
        #: Mean principal moment of inertia per body (trace/3).  Used only to
        #: size the *static* rolling branch (the angular impulse that arrests the
        #: relative spin in one step); for a sphere it is exactly 2/5 m r^2.
        self.mean_inertia = np.trace(self.inertia_body, axis1=1, axis2=2) / 3.0

        self.pos = (
            np.zeros((n, 3))
            if positions is None
            else np.asarray(positions, dtype=float).reshape(n, 3).copy()
        )
        if quaternions is None:
            self.quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        else:
            q = np.asarray(quaternions, dtype=float).reshape(n, 4).copy()
            self.quat = q / np.linalg.norm(q, axis=1, keepdims=True)
        self.vel = np.zeros((n, 3))
        self.omega = np.zeros((n, 3))

        self.history_pair = ContactHistory(capacity=max(2048, 16 * self.n_members))
        self.history_wall = ContactHistory(capacity=max(2048, 4 * self.n_members))
        self.ledger = EnergyLedger()
        self.time = 0.0
        self.step_count = 0
        self._pe_contact = 0.0
        self._pe_tangential = 0.0

        self.last_contacts: Dict[str, np.ndarray] = {}
        self.energy_initial: Optional[float] = self.total_energy()

    # -- geometry helpers --------------------------------------------------
    def rotation_matrices(self) -> np.ndarray:
        return quat_to_matrix(self.quat)

    def member_centers(self) -> np.ndarray:
        rot = self.rotation_matrices()[self.body_of_member]      # (n_members, 3, 3)
        return self.pos[self.body_of_member] + np.einsum(
            "mij,mj->mi", rot, self.local_centers
        )

    def _vel_at(self, xc: np.ndarray, body: np.ndarray) -> np.ndarray:
        return self.vel[body] + np.cross(self.omega[body], xc - self.pos[body])

    def _inertia_world(self) -> Tuple[np.ndarray, np.ndarray]:
        """World-frame inertia and its inverse: ``R I R^T`` and ``R I^-1 R^T``.

        The inverse is rotated as well (instead of inverting the world tensor),
        which keeps the hot loop free of batched matrix inversions.
        """
        rot = self.rotation_matrices()
        rot_t = rot.transpose(0, 2, 1)
        inertia = np.einsum("bij,bjk,bkl->bil", rot, self.inertia_body, rot_t)
        inv_inertia = np.einsum("bij,bjk,bkl->bil", rot, self.inv_inertia_body, rot_t)
        return inertia, inv_inertia

    def kinetic_energy(self) -> float:
        inertia_world, _ = self._inertia_world()
        ke_t = 0.5 * float(np.sum(self.mass * np.sum(self.vel ** 2, axis=1)))
        ke_r = 0.5 * float(
            np.einsum("bi,bij,bj->b", self.omega, inertia_world, self.omega).sum()
        )
        return ke_t + ke_r

    def gravitational_energy(self) -> float:
        # PE = -m g . x  =>  F = -dPE/dx = m g
        return float(-np.sum(self.mass[:, None] * (self.gravity * self.pos)))

    def total_energy(self) -> float:
        return (
            self.kinetic_energy()
            + self.gravitational_energy()
            + self._pe_contact
            + self._pe_tangential
        )

    def close_energy(self, initial_total: Optional[float] = None) -> dict:
        init = self.energy_initial if initial_total is None else initial_total
        return self.ledger.close(self.total_energy(), init)

    # -- one step ----------------------------------------------------------
    def step(self) -> Dict[str, np.ndarray]:
        """Advance one time step (semi-implicit Euler + quaternion update).

        Contact forces are evaluated at the beginning of the step; the energy
        terms are evaluated *after* the velocity update on the actual relative
        displacements ``Delta x = v_new dt`` (semi-implicit quadrature), which is
        what makes :class:`EnergyLedger` close to discretisation accuracy instead
        of O(dt).
        """
        p = self.params
        dt = self.dt
        mc = self.member_centers()
        bi_all = self.body_of_member

        # ---- clump/clump narrow phase ------------------------------------ #
        cand = member_pair_candidates(mc, self.member_radius, bi_all)
        group_keys = (
            bi_all[cand["i"]].astype(np.int64) * self.n_bodies + bi_all[cand["j"]]
            if len(cand["i"])
            else None
        )
        kept, _ = merge_junction_candidates(
            cand,
            self.member_radius,
            merge_angle_deg=p.merge_angle_deg,
            group_keys=group_keys,
        )
        mi, mj = kept["i"], kept["j"]
        n_pair = len(mi)
        if n_pair:
            pen_p = kept["penetration"]
            nrm_p = kept["normal"]
            xc_p = kept["point"]
            b_i = bi_all[mi]
            b_j = bi_all[mj]
            r_i = self.member_radius[mi]
            r_j = self.member_radius[mj]
            r_eff_p = r_i * r_j / (r_i + r_j)
            m_eff_p = self.member_mass[mi] * self.member_mass[mj] / (
                self.member_mass[mi] + self.member_mass[mj]
            )
            keys_p = mi.astype(np.int64) * self.n_members + mj.astype(np.int64)
            v_rel_p = self._vel_at(xc_p, b_i) - self._vel_at(xc_p, b_j)
        else:
            pen_p = nrm_p = xc_p = np.zeros((0, 3))
            b_i = b_j = np.zeros(0, dtype=int)
            r_eff_p = m_eff_p = np.zeros(0)
            keys_p = np.zeros(0, dtype=np.int64)
            v_rel_p = np.zeros((0, 3))

        # ---- clump/obstacle narrow phase --------------------------------- #
        wall_blocks = []
        for k, obs in enumerate(self.obstacles):
            c = sphere_box_contacts(mc, self.member_radius, obs)
            if len(c["penetration"]) == 0:
                continue
            m_ids = c["member"]
            wall_blocks.append(
                {
                    "obs": obs,
                    "obs_id": k,
                    "member": m_ids,
                    "pen": c["penetration"],
                    "normal_out": c["normal"],
                    "point": c["point"],
                    "body": bi_all[m_ids],
                    "keys": k * self.n_members + m_ids.astype(np.int64),
                }
            )

        # ---- tangential history ------------------------------------------ #
        self.history_pair.begin_step()
        self.history_wall.begin_step()
        if n_pair:
            slots_p = self.history_pair.slots(keys_p)
            self.history_pair.touch(slots_p)
            xi_p = self.history_pair.xi[slots_p].copy()
        else:
            slots_p = np.zeros(0, dtype=np.int64)
            xi_p = np.zeros((0, 3))
        wall_keys = (
            np.concatenate([b["keys"] for b in wall_blocks])
            if wall_blocks
            else np.zeros(0, dtype=np.int64)
        )
        if len(wall_keys):
            slots_w = self.history_wall.slots(wall_keys)
            self.history_wall.touch(slots_w)
            xi_w = self.history_wall.xi[slots_w].copy()
        else:
            slots_w = np.zeros(0, dtype=np.int64)
            xi_w = np.zeros((0, 3))

        # ---- force law ---------------------------------------------------- #
        force = np.zeros((self.n_bodies, 3))
        torque = np.zeros((self.n_bodies, 3))
        pe_c = pe_t = 0.0
        rec = {
            "bi": [], "bj": [], "point": [], "normal": [], "fn": [], "ft": [],
            "cn": [], "vn_old": [], "tau_roll": [], "vobs": [],
            "kind": [], "penetration": [], "sliding": [],
        }

        self._e_dash_pre = 0.0
        if n_pair:
            res_p = evaluate_contact_law(pen_p, nrm_p, v_rel_p, m_eff_p, p, xi_p, dt)
            self.history_pair.xi[slots_p] = res_p.xi
            f_on_i = -res_p.fn[:, None] * nrm_p + res_p.ft
            f_on_j = -f_on_i
            np.add.at(force, b_i, f_on_i)
            np.add.at(force, b_j, f_on_j)
            np.add.at(torque, b_i, np.cross(xc_p - self.pos[b_i], f_on_i))
            np.add.at(torque, b_j, np.cross(xc_p - self.pos[b_j], f_on_j))
            w_rel = self.omega[b_i] - self.omega[b_j]
            w_mag = np.linalg.norm(w_rel, axis=1)
            lever_p = rolling_static.rolling_lever_arm(
                xc_p, self.pos[b_i], r_eff_p,
                getattr(p, "rolling_lever", "member"),
            )
            limit_p = rolling_static.rolling_limit(p.mu_roll, lever_p, res_p.fn)
            inert_p = 1.0 / (1.0 / self.mean_inertia[b_i] + 1.0 / self.mean_inertia[b_j])
            tau_roll = rolling_static.rolling_torque(
                w_rel, limit_p, inert_p, dt,
                model=getattr(p, "rolling_model", "viscous"), eps_w=p.eps_w,
            )
            np.add.at(torque, b_i, tau_roll)
            np.add.at(torque, b_j, -tau_roll)
            pe_c += float(res_p.pe_normal.sum())
            pe_t += float(res_p.pe_tangential.sum())
            vn_old = np.sum(v_rel_p * nrm_p, axis=1)
            cn = 2.0 * p.zeta * np.sqrt(m_eff_p * p.kn)
            rec["bi"].append(b_i)
            rec["bj"].append(b_j)
            rec["point"].append(xc_p)
            rec["normal"].append(nrm_p)
            rec["fn"].append(res_p.fn)
            rec["ft"].append(res_p.ft)
            rec["cn"].append(cn)
            rec["vn_old"].append(vn_old)
            rec["tau_roll"].append(tau_roll)
            rec["vobs"].append(np.zeros((n_pair, 3)))
            rec["kind"].append(np.zeros(n_pair, dtype=int))
            rec["penetration"].append(pen_p)
            ft_mag = np.linalg.norm(res_p.ft, axis=1)
            rec["sliding"].append(ft_mag >= p.mu * res_p.fn - 1e-15)

        ext_work = 0.0
        wall_fn: List[np.ndarray] = []
        wall_bi: List[np.ndarray] = []
        wall_bj: List[np.ndarray] = []
        wall_pt: List[np.ndarray] = []
        wall_n: List[np.ndarray] = []
        wall_pen: List[np.ndarray] = []
        wall_kind: List[np.ndarray] = []
        if wall_blocks:
            off = 0
            for blk in wall_blocks:
                k_n = len(blk["pen"])
                sl = slice(off, off + k_n)
                off += k_n
                m_ids = blk["member"]
                b_ids = blk["body"]
                r_i = self.member_radius[m_ids]
                m_eff = self.member_mass[m_ids]
                normal_in = -blk["normal_out"]      # particle -> wall
                xc = blk["point"]
                obs = blk["obs"]
                mu = p.mu if obs.mu is None else obs.mu
                mu_roll = p.mu_roll if obs.mu_roll is None else obs.mu_roll
                local = p.copy(mu=mu, mu_roll=mu_roll)
                v_rel = self._vel_at(xc, b_ids) - obs.velocity
                res_w = evaluate_contact_law(
                    blk["pen"], normal_in, v_rel, m_eff, local, xi_w[sl], dt
                )
                self.history_wall.xi[slots_w[sl]] = res_w.xi
                f_on_sphere = -res_w.fn[:, None] * normal_in + res_w.ft
                np.add.at(force, b_ids, f_on_sphere)
                np.add.at(torque, b_ids, np.cross(xc - self.pos[b_ids], f_on_sphere))
                w_rel = self.omega[b_ids]
                w_mag = np.linalg.norm(w_rel, axis=1)
                lever_w = rolling_static.rolling_lever_arm(
                    xc, self.pos[b_ids], r_i,
                    getattr(local, "rolling_lever", "member"),
                )
                limit_w = rolling_static.rolling_limit(mu_roll, lever_w, res_w.fn)
                tau_roll = rolling_static.rolling_torque(
                    w_rel, limit_w, self.mean_inertia[b_ids], dt,
                    model=getattr(local, "rolling_model", "viscous"), eps_w=p.eps_w,
                )
                np.add.at(torque, b_ids, tau_roll)
                ext_work += float(np.sum(f_on_sphere * obs.velocity) * dt)
                pe_c += float(res_w.pe_normal.sum())
                pe_t += float(res_w.pe_tangential.sum())
                vn_old = np.sum(v_rel * normal_in, axis=1)
                cn = 2.0 * p.zeta * np.sqrt(m_eff * p.kn)
                rec["bi"].append(b_ids)
                rec["bj"].append(np.full(k_n, -1, dtype=int))
                rec["point"].append(xc)
                rec["normal"].append(normal_in)
                rec["fn"].append(res_w.fn)
                rec["ft"].append(res_w.ft)
                rec["cn"].append(cn)
                rec["vn_old"].append(vn_old)
                rec["tau_roll"].append(tau_roll)
                rec["vobs"].append(np.tile(obs.velocity, (k_n, 1)))
                rec["kind"].append(np.full(k_n, 1 + int(blk["obs_id"]), dtype=int))
                rec["penetration"].append(blk["pen"])
                ft_mag = np.linalg.norm(res_w.ft, axis=1)
                rec["sliding"].append(ft_mag >= mu * res_w.fn - 1e-15)
                wall_fn.append(res_w.fn)
                wall_bi.append(b_ids)
                wall_bj.append(np.full(k_n, -1, dtype=int))
                wall_pt.append(xc)
                wall_n.append(blk["normal_out"])
                wall_pen.append(blk["pen"])
                wall_kind.append(np.full(k_n, 1 + int(blk["obs_id"]), dtype=int))

        # ---- retire detached contacts ------------------------------------- #
        e_detach = 0.0
        for hist in (self.history_pair, self.history_wall):
            dead = np.nonzero((hist._stamp > 0) & (hist._stamp != hist.step_id))[0]
            if len(dead):
                e_detach += float(0.5 * p.kt * np.sum(hist.xi[dead] ** 2, axis=1).sum())
                hist.xi[dead] = 0.0
                hist._stamp[dead] = 0

        # ---- integrate ----------------------------------------------------- #
        self.vel += (force * self.inv_mass[:, None] + self.gravity) * dt
        self.pos += self.vel * dt

        inertia_world, inv_inertia_world = self._inertia_world()
        gyro = np.cross(self.omega, np.einsum("bij,bj->bi", inertia_world, self.omega))
        omega_dot = np.einsum("bij,bj->bi", inv_inertia_world, torque - gyro)
        self.omega += omega_dot * dt
        self.quat = quat_integrate(self.quat, self.omega, dt)
        for obs in self.obstacles:
            obs.advance(dt)
        self.time += dt
        self.step_count += 1

        # ---- energy accounting on the actual displacements ------------------ #
        e_dash = e_fric = e_roll = 0.0
        if rec["fn"]:
            bi = np.concatenate(rec["bi"])
            bj = np.concatenate(rec["bj"])
            point = np.concatenate(rec["point"])
            normal = np.concatenate(rec["normal"])
            fn = np.concatenate(rec["fn"])
            ft = np.concatenate(rec["ft"])
            cn = np.concatenate(rec["cn"])
            vn_old = np.concatenate(rec["vn_old"])
            tau_roll = np.concatenate(rec["tau_roll"])
            vobs = np.concatenate(rec["vobs"])
            sliding = np.concatenate(rec["sliding"])
            live = fn > 0.0

            v_i = self.vel[bi] + np.cross(self.omega[bi], point - self.pos[bi])
            has_j = bj >= 0
            v_j = np.where(
                has_j[:, None],
                self.vel[np.maximum(bj, 0)]
                + np.cross(self.omega[np.maximum(bj, 0)], point - self.pos[np.maximum(bj, 0)]),
                vobs,
            )
            v_rel = v_i - v_j
            vn_new = np.sum(v_rel * normal, axis=1)
            vt_new = v_rel - vn_new[:, None] * normal
            # dashpot: force fixed at c_n * vn_old along normal, displacement vn_new dt
            e_dash = float(np.sum(np.where(live, cn * vn_old * vn_new * dt, 0.0)))
            # tangential work on the actual displacement (sticking contacts exchange
            # this energy with the tangential spring, sliding contacts dissipate it)
            e_fric = float(-np.sum(np.sum(ft * vt_new, axis=1) * dt))
            w_rel = self.omega[bi] - np.where(has_j[:, None], self.omega[np.maximum(bj, 0)], 0.0)
            e_roll = float(-np.sum(np.sum(tau_roll * w_rel, axis=1) * dt))

        self._pe_contact = pe_c
        self._pe_tangential = pe_t
        self.ledger.external_work += ext_work
        self.ledger.dissipated_normal += e_dash
        self.ledger.dissipated_friction += e_fric
        self.ledger.dissipated_rolling += e_roll
        self.ledger.dissipated_detach += e_detach

        if rec["fn"]:
            fn_all = np.concatenate(rec["fn"])
            self.last_contacts = {
                "fn": fn_all,
                "body_i": np.concatenate(rec["bi"]),
                "body_j": np.concatenate(rec["bj"]),
                "point": np.concatenate(rec["point"]),
                "normal": np.concatenate(rec["normal"]),
                "penetration": np.concatenate(rec["penetration"]),
                "kind": np.concatenate(rec["kind"]),
            }
        else:
            self.last_contacts = {
                "fn": np.zeros(0),
                "body_i": np.zeros(0, dtype=int),
                "body_j": np.zeros(0, dtype=int),
                "point": np.zeros((0, 3)),
                "normal": np.zeros((0, 3)),
                "penetration": np.zeros(0),
                "kind": np.zeros(0, dtype=int),
            }
        return self.last_contacts

    def run(self, n_steps: int) -> Dict[str, np.ndarray]:
        for _ in range(int(n_steps)):
            self.step()
        return self.last_contacts

    def snapshot(self) -> dict:
        return {
            "positions": self.pos.copy(),
            "quaternions": self.quat.copy(),
            "velocities": self.vel.copy(),
            "omegas": self.omega.copy(),
            "time": self.time,
            "steps": self.step_count,
        }

    def velocity_spread(self) -> dict:
        speed = np.linalg.norm(self.vel, axis=1)
        om = np.linalg.norm(self.omega, axis=1)
        return {
            "max_speed": float(speed.max()),
            "mean_speed": float(speed.mean()),
            "max_omega": float(om.max()),
            "mean_omega": float(om.mean()),
            "median_omega": float(np.median(om)),
        }

    def rolling_torque_saturation(self, eps_w: Optional[float] = None) -> dict:
        """How close the rolling-resistance torque is to its Coulomb limit.

        The contact law applies ``tau = -mu_roll * r * |Fn| * w_rel / (|w_rel| + eps_w)``
        (``clump_dem.step``), so the delivered torque is the Coulomb limit scaled by
        ``s = |w| / (|w| + eps_w)``.  ``s -> 1`` is genuine Coulomb rolling
        resistance; ``s << 1`` means the branch is acting as a *viscous* torque of
        coefficient ``mu_roll r |Fn| / eps_w`` instead.  Reported per body (the
        wall branch uses the body spin directly; the pair branch uses ``w_rel``,
        for which the body spin is an upper bound).
        """
        eps = float(self.params.eps_w if eps_w is None else float(eps_w))
        if eps <= 0.0:
            raise ValueError("eps_w must be positive")
        w = np.linalg.norm(self.omega, axis=1)
        model = getattr(self.params, "rolling_model", "viscous")
        if model == "coulomb":
            # static branch: saturation = min(1, I|w|/dt / (mu_roll R |Fn|)) per
            # body, evaluated with the body's own spin and the largest lever arm
            # available to it (contact-level values live in ``last_contacts``).
            from ..contact.rolling_static import stopping_speed
            lever = np.array([p.bounding_radius for p in self.particles], dtype=float)
            limit = self.params.mu_roll * lever * self.mass * 9.81
            sat = stopping_speed(limit, self.mean_inertia, self.dt) / np.maximum(w, 1e-30)
            s = np.minimum(1.0, sat)
        else:
            s = w / (w + eps)
        return {
            "eps_w": eps,
            "model": model,
            "mean": float(s.mean()),
            "median": float(np.median(s)),
            "p10": float(np.percentile(s, 10)),
            "p90": float(np.percentile(s, 90)),
            "max": float(s.max()),
            "mean_omega": float(w.mean()),
        }


# ---------------------------------------------------------------------------
# Packing / case builders
# ---------------------------------------------------------------------------
def lattice_packing(
    particle: ClumpParticle,
    lo: Sequence[float],
    hi: Sequence[float],
    *,
    spacing: Optional[Sequence[float]] = None,
    jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aligned-orientation lattice filling of the box ``[lo, hi]``.

    ``spacing`` defaults to ``2 * (a, b, b) * 1.02`` from the particle extents,
    i.e. the members do not overlap at t=0 (a clean initial state).  A tiny
    random jitter and a small random rotation break the lattice symmetry.
    """
    rng = rng or np.random.default_rng(0)
    c = particle.local_centers
    r = particle.radii
    ext = (c + r[:, None]).max(axis=0) - (c - r[:, None]).min(axis=0)
    sp = np.asarray(spacing, dtype=float) if spacing is not None else 1.02 * ext
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    # the bounds describe the *container*; the centres are inset by half a
    # particle so that no member pokes through a wall at t=0
    lo_c = lo + 0.505 * ext
    hi_c = hi - 0.505 * ext
    span = np.maximum(hi_c - lo_c, 1e-12)
    counts = np.maximum(1, np.floor(span / sp).astype(int))
    axes = [lo_c[k] + (np.arange(counts[k]) + 0.5) * sp[k] for k in range(3)]
    # keep the filling centred on the box midpoint
    for k in range(3):
        used = counts[k] * sp[k]
        axes[k] = axes[k] + 0.5 * (span[k] - used)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    n = len(grid)
    if jitter > 0.0:
        grid = grid + rng.normal(scale=jitter, size=grid.shape)
    # small random rotation about a random axis (keeps the packing non-overlapping
    # only if jitter is small; the soft contact resolves the rest)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    if jitter > 0.0:
        axis = rng.normal(size=(n, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        ang = rng.uniform(0.0, 0.25, size=n)
        quats = np.concatenate(
            [np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None] * axis], axis=1
        )
    return grid, quats


def heap_case(
    particle_factory: Callable[[], ClumpParticle],
    *,
    box_half_x: float = 0.020,
    slab_half_y: float = 0.0065,
    fill_height: float = 0.095,
    wall_height: float = 0.10,
    floor_half_x: float = 0.10,
    lift_speed: float = 0.30,
    params: Optional[ContactParams] = None,
    dt: float = 5e-5,
    seed: int = 0,
    wall_mu: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
) -> Tuple[ClumpDEM, List[BoxObstacle]]:
    """Quasi-2D lifted-box heap: returns the simulation and the two lift walls.

    ``spacing`` is forwarded to :func:`lattice_packing`.  Exposing it lets a
    caller *match* the fill between particle shapes (the default spacing is
    derived from the particle extents, so a longer clump is packed with fewer
    particles in the same box -- a shape/particle-count confound).
    """
    params = params or ContactParams()
    rng = np.random.default_rng(seed)
    p0 = particle_factory()
    floor = BoxObstacle(
        center=(0.0, 0.0, -0.001), half=(floor_half_x, slab_half_y + 0.002, 0.001),
        mu=wall_mu, name="floor",
    )
    ywall_lo = BoxObstacle(
        center=(0.0, -(slab_half_y + 0.0005), wall_height / 2.0),
        half=(floor_half_x, 0.0005, wall_height / 2.0),
        mu=0.0, mu_roll=0.0, name="ywall_lo",
    )
    ywall_hi = BoxObstacle(
        center=(0.0, +(slab_half_y + 0.0005), wall_height / 2.0),
        half=(floor_half_x, 0.0005, wall_height / 2.0),
        mu=0.0, mu_roll=0.0, name="ywall_hi",
    )
    lift_lo = BoxObstacle(
        center=(-(box_half_x + 0.001), 0.0, wall_height / 2.0),
        half=(0.001, slab_half_y + 0.001, wall_height / 2.0),
        velocity=(0.0, 0.0, 0.0), name="lift_lo",
    )
    lift_hi = BoxObstacle(
        center=(+(box_half_x + 0.001), 0.0, wall_height / 2.0),
        half=(0.001, slab_half_y + 0.001, wall_height / 2.0),
        velocity=(0.0, 0.0, 0.0), name="lift_hi",
    )
    positions, quats = lattice_packing(
        p0,
        (-box_half_x, -slab_half_y, 0.0),
        (box_half_x, slab_half_y, fill_height),
        spacing=spacing,
        jitter=2e-5,
        rng=rng,
    )
    particles = [particle_factory() for _ in range(len(positions))]
    sim = ClumpDEM(
        particles,
        positions,
        quats,
        obstacles=[floor, ywall_lo, ywall_hi, lift_lo, lift_hi],
        params=params,
        dt=dt,
    )
    return sim, [lift_lo, lift_hi]


def measure_repose_angle(
    positions: np.ndarray,
    *,
    n_bins: int = 20,
    flank_lo: float = 0.10,
    flank_hi: float = 0.92,
    hull_consistency_deg: float = 6.0,
) -> dict:
    """Repose angle from a heap silhouette (two independent estimators).

    Primary estimator (``angle_profile_deg``): least-squares fit to the *upper
    envelope* of the ``(|x|, z)`` point cloud (``max z`` per radial bin), restricted
    to the flank between ``flank_lo`` and ``flank_hi`` of the heap radius and to
    bins above 8 % of the peak height, with one 2-sigma rejection pass.  The
    envelope ignores single flying particles and the crest/toe curvature.

    Cross-check (``angle_hull_deg``): slope of the **longest** slanted edge of the
    convex hull of the same cloud (the pile silhouette).  Choosing the longest
    edge -- not the steepest -- keeps the estimator from locking onto a short
    steep segment of the hull.

    ``angle_profile_deg`` is ``nan`` when the profile is not a proper flank (e.g.
    no gravity: the cloud is not a heap).
    """
    pos = np.asarray(positions, dtype=float)
    x = pos[:, 0]
    z = pos[:, 2] - pos[:, 2].min()
    r = np.abs(x)
    out = {
        "angle_profile_deg": float("nan"),
        "angle_hull_deg": float("nan"),
        "peak_height": float(z.max()) if len(z) else float("nan"),
        "max_radius": float(r.max()) if len(r) else float("nan"),
        "flank_points": 0,
        "hull_edge_span": 0.0,
        "hull_consistency_deg": float("nan"),
    }
    if len(r) < 8 or r.max() <= 0:
        return out
    pts = np.stack([r, z], axis=1)
    r_span = float(r.max())
    z_peak = float(z.max())

    # ---- primary: upper-envelope fit ------------------------------------- #
    edges = np.linspace(0.0, r_span, n_bins + 1)
    centres, tops = [], []
    for k in range(n_bins):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if np.any(m):
            centres.append(0.5 * (edges[k] + edges[k + 1]))
            tops.append(z[m].max())
    centres = np.asarray(centres)
    tops = np.asarray(tops)
    sel = (
        (centres >= flank_lo * r_span)
        & (centres <= flank_hi * r_span)
        & (tops > 0.08 * z_peak)
    )
    if sel.sum() >= 3:
        cx, cz = centres[sel], tops[sel]
        coef = np.polyfit(cx, cz, 1)
        for _ in range(2):                       # 2-sigma rejection passes
            pred = np.polyval(coef, cx)
            resid = cz - pred
            sigma = resid.std() or 1e-12
            keep = np.abs(resid) < 2.0 * sigma
            if keep.sum() < 3:
                break
            cx, cz = cx[keep], cz[keep]
            coef = np.polyfit(cx, cz, 1)
        if coef[0] < 0:
            out["angle_profile_deg"] = float(math.degrees(math.atan(-coef[0])))
        out["flank_points"] = int(len(cx))

    # ---- cross-check: longest slanted convex-hull edge -------------------- #
    try:
        verts = pts[ConvexHull(pts).vertices]
    except Exception:
        return out
    best = (0.0, 0.0)
    for k in range(len(verts)):
        p1 = verts[k]
        p2 = verts[(k + 1) % len(verts)]
        dr = p2[0] - p1[0]
        if dr <= 0.25 * r_span:                  # only long outward-going edges
            continue
        slope = (p2[1] - p1[1]) / dr
        if slope < 0 and dr > best[1]:
            best = (slope, dr)
    if best[1] > 0:
        out["angle_hull_deg"] = float(math.degrees(math.atan(-best[0])))
        out["hull_edge_span"] = float(best[1])
    if np.isfinite(out["angle_hull_deg"]) and np.isfinite(out["angle_profile_deg"]):
        out["hull_consistency_deg"] = abs(
            out["angle_hull_deg"] - out["angle_profile_deg"]
        )
    return out


def lifted_box_repose(
    particle_factory: Callable[[], ClumpParticle],
    *,
    seed: int = 0,
    dt: float = 5e-5,
    params: Optional[ContactParams] = None,
    settle_steps: int = 900,
    lift_steps: int = 2000,
    tail_steps: int = 1500,
    lift_speed: float = 0.30,
    gravity: Sequence[float] = (0.0, 0.0, -9.81),
    wall_mu: Optional[float] = None,
    min_tail_steps: Optional[int] = None,
    max_tail_steps: Optional[int] = None,
    quiescence_mean_speed: Optional[float] = None,
    probe_every: int = 100,
    **case_kw,
) -> dict:
    """Run the lifted-box repose protocol and return the measurement.

    Quiescence gate (optional, off by default so round-1 numbers reproduce
    bit-for-bit): with ``quiescence_mean_speed`` set, the tail is integrated in
    chunks of ``probe_every`` steps until the *mean body speed* drops to the
    tolerance, starting from ``min_tail_steps`` and giving up at
    ``max_tail_steps``.  Without the gate a fixed ``tail_steps`` is used and the
    tail may still be flowing, which is exactly what the reported
    ``tail_spread`` / ``quiesced`` fields let a caller detect.
    """
    sim, walls = heap_case(
        particle_factory,
        lift_speed=lift_speed,
        params=params,
        dt=dt,
        seed=seed,
        wall_mu=wall_mu,
        **case_kw,
    )
    sim.gravity = np.asarray(gravity, dtype=float).reshape(3)
    sim.run(settle_steps)
    settle_spread = sim.velocity_spread()
    for w in walls:
        w.velocity[:] = (0.0, 0.0, lift_speed)
    sim.run(lift_steps)
    for w in walls:
        w.velocity[:] = (0.0, 0.0, 0.0)
    flow_spread = sim.velocity_spread()
    flow_saturation = sim.rolling_torque_saturation()
    if quiescence_mean_speed is None:
        sim.run(tail_steps)
        tail_used = int(tail_steps)
        quiesced: Optional[bool] = None
    else:
        lo = int(min_tail_steps if min_tail_steps is not None else 0)
        hi = int(max_tail_steps if max_tail_steps is not None else max(tail_steps, 1))
        hi = max(hi, lo)
        sim.run(lo)
        tail_used = lo
        quiesced = False
        while tail_used < hi:
            chunk = min(max(1, probe_every), hi - tail_used)
            sim.run(chunk)
            tail_used += chunk
            if sim.velocity_spread()["mean_speed"] <= quiescence_mean_speed:
                quiesced = True
                break
    meas = measure_repose_angle(sim.pos)
    tail_spread = sim.velocity_spread()
    return {
        "measurement": meas,
        "positions": sim.pos.copy(),
        "n_particles": sim.n_bodies,
        "n_members": sim.n_members,
        "solid_volume": float(sum(p.volume for p in sim.particles)),
        "mass": float(sim.mass.sum()),
        "time": sim.time,
        "steps": sim.step_count,
        "settle_spread": settle_spread,
        "tail_spread": tail_spread,
        "flow_spread": flow_spread,
        "flow_rolling_saturation": flow_saturation,
        "tail_steps_used": tail_used,
        "quiesced": quiesced,
        "quiescence_mean_speed": quiescence_mean_speed,
        "rolling_saturation": sim.rolling_torque_saturation(),
        "energy": sim.close_energy(),
        "sim": sim,
    }


# ---------------------------------------------------------------------------
# Compression / force chain protocol
# ---------------------------------------------------------------------------
def piston_case(
    particle_factory: Callable[[], ClumpParticle],
    *,
    box_half_x: float = 0.030,
    slab_half_y: float = 0.008,
    fill_height: float = 0.060,
    params: Optional[ContactParams] = None,
    dt: float = 5e-5,
    seed: int = 0,
    piston_speed: float = 0.02,
) -> Tuple[ClumpDEM, BoxObstacle]:
    """Confined compression cell with a piston moving in +x."""
    params = params or ContactParams()
    rng = np.random.default_rng(seed)
    p0 = particle_factory()
    floor = BoxObstacle(
        center=(0.0, 0.0, -0.001), half=(box_half_x + 0.01, slab_half_y + 0.01, 0.001),
        name="floor",
    )
    y_lo = BoxObstacle(
        center=(0.0, -(slab_half_y + 0.0005), fill_height / 2.0),
        half=(box_half_x + 0.01, 0.0005, fill_height / 2.0), name="y_lo",
    )
    y_hi = BoxObstacle(
        center=(0.0, +(slab_half_y + 0.0005), fill_height / 2.0),
        half=(box_half_x + 0.01, 0.0005, fill_height / 2.0), name="y_hi",
    )
    back = BoxObstacle(
        center=(-(box_half_x + 0.001), 0.0, fill_height / 2.0),
        half=(0.001, slab_half_y + 0.001, fill_height / 2.0), name="back_wall",
    )
    piston = BoxObstacle(
        center=((box_half_x + 0.001), 0.0, fill_height / 2.0),
        half=(0.001, slab_half_y + 0.001, fill_height / 2.0),
        velocity=(-piston_speed, 0.0, 0.0), name="piston",
    )
    positions, quats = lattice_packing(
        p0,
        (-box_half_x, -slab_half_y, 0.0),
        (box_half_x, slab_half_y, fill_height),
        spacing=spacing,
        jitter=2e-5,
        rng=rng,
    )
    particles = [particle_factory() for _ in range(len(positions))]
    sim = ClumpDEM(
        particles,
        positions,
        quats,
        obstacles=[floor, y_lo, y_hi, back, piston],
        params=params,
        dt=dt,
    )
    return sim, piston


def piston_compression(
    particle_factory: Callable[[], ClumpParticle],
    *,
    seed: int = 0,
    dt: float = 5e-5,
    params: Optional[ContactParams] = None,
    settle_steps: int = 1200,
    compress_steps: int = 2000,
    piston_speed: float = 0.02,
    force_avg_steps: int = 40,
    **case_kw,
) -> dict:
    """Settle, compress with the piston, then measure the force network and energy.

    The contact forces used for the force-chain statistics are averaged over the
    last ``force_avg_steps`` steps of the stroke (the assembly is quasi-static, so
    this only removes the residual vibration noise of the last few contacts).
    """
    sim, piston = piston_case(
        particle_factory,
        params=params,
        dt=dt,
        seed=seed,
        piston_speed=piston_speed,
        **case_kw,
    )
    # settle until the assembly is at rest: a loose lattice keeps compacting for
    # much longer than a fixed step count, and starting the stroke on a flowing
    # bed would make the energy audit meaningless
    for k in range(int(settle_steps)):
        sim.step()
        if k > 400 and k % 100 == 0 and sim.velocity_spread()["mean_speed"] < 1e-3:
            break
    settle_used = sim.step_count
    piston.velocity[:] = (0.0, 0.0, 0.0)
    sim.ledger = EnergyLedger()
    sim.energy_initial = sim.total_energy()
    sim.ledger.initial_total = sim.energy_initial
    piston.velocity[:] = (-piston_speed, 0.0, 0.0)
    acc = None
    for _ in range(compress_steps):
        c = sim.step()
        if _ >= compress_steps - force_avg_steps:
            acc = c if acc is None else {
                k: np.concatenate([acc[k], c[k]], axis=0) for k in c
            }
    piston.velocity[:] = (0.0, 0.0, 0.0)
    sim.run(20)
    energy = sim.close_energy()
    contacts = acc if acc is not None else sim.last_contacts
    stats = force_distribution_stats(contacts["fn"])
    chains = force_chain_clusters(
        contacts["fn"], contacts["body_i"], contacts["body_j"], sim.n_bodies
    )
    wall_mask = contacts["body_j"] < 0
    stats_walls = force_distribution_stats(contacts["fn"][wall_mask])
    return {
        "force_stats": stats,
        "force_stats_internal": force_distribution_stats(contacts["fn"][~wall_mask]),
        "force_stats_wall": stats_walls,
        "chains": chains,
        "energy": energy,
        "kinetic_energy": sim.kinetic_energy(),
        "n_particles": sim.n_bodies,
        "n_members": sim.n_members,
        "piston_displacement": piston_speed * compress_steps * dt,
        "settle_steps_used": settle_used,
        "time": sim.time,
        "steps": sim.step_count,
        "contacts": contacts,
        "sim": sim,
    }