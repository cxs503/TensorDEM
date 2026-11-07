"""Six-degree-of-freedom DEM driver for superquadric particles and sphere clumps.

Why this module exists
----------------------
:mod:`tensordem.superquadric` (prototype) stops at a single contact evaluation,
and :mod:`tensordem.shape_error_budget` *estimated* the macroscopic consequence of
a sphere-clump approximation with a hand-written angularity formula.  Neither can
answer the engineering question "for which shapes may the true shape be replaced
by an N-sphere clump?" with a number that a third party can recompute.  This
module supplies the missing layer: an actual time-stepping granular solver, so
angles of repose, shear-cell friction and force-chain statistics are *measured*
rather than asserted.

Model
-----
* Bodies are **superquadrics** (``a, b, c, n1, n2`` in the body frame) or rigid
  **sphere clumps** (body-frame sphere offsets + radii).
* Contacts: superquadric-superquadric through the vectorised kernel
  :func:`tensordem.superquadric_kernel.contact_geometry_batch`;
  sphere-superquadric through the exact closest-point signed distance of the
  sphere centre; sphere-sphere analytically.  Each clump sphere contributes its
  force at its own contact point, so a clump transmits torque like a genuine
  multi-sphere rigid body.
* Force laws: automatic Hertz / Boussinesq flat-punch normal branch, Mindlin
  tangential spring with Coulomb saturation, Coulomb rolling resistance.
* Boundaries: planar half-spaces with friction and an optional velocity (the
  moving floor of the shear cell).
* Integration: semi-implicit (symplectic) Euler in translation, body-frame Euler
  equations with a Rodrigues rotation update, step size from a contact-stiffness
  stability estimate.
* Ledgers: total mechanical energy, maximum penetration ratio ``delta / D`` and
  the full contact list (normals + forces) for stress and force-chain
  statistics.

All quantities are SI (m, kg, s, N, Pa, J).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .flat_punch import area_equivalent_punch_stiffness
from .superquadric_kernel import (
    ContactMaterial,
    closest_point_batch,
    contact_geometry_batch,
    effective_radius_hertz,
    facet_patch_area,
    mindlin_tangential_update,
    normal_force,
    principal_curvatures_local,
    rolling_resistance_torque,
    signed_distance_batch,
    superquadric_inertia,
)

__all__ = [
    "SQBody",
    "Wall",
    "SQSystemConfig",
    "SQSystem",
    "ColumnCollapseResult",
    "ShearCellResult",
    "make_superquadric_body",
    "make_clump_body",
    "run_column_collapse",
    "run_shear_cell",
    "initial_column_packing",
    "repose_angle_from_cloud",
    "force_chain_statistics",
    "stress_tensor_from_contacts",
    "energy_drift_per_1000",
    "rodrigues",
]

GRAVITY = np.array([0.0, 0.0, -9.81])


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (Rodrigues' formula)."""
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-15:
        return np.eye(3)
    k = axis_angle / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


def _orthonormalise_batch(r: np.ndarray) -> np.ndarray:
    """Batched Gram-Schmidt re-orthonormalisation of ``(M,3,3)`` rotation matrices."""
    x = r[:, :, 0].copy()
    y = r[:, :, 1].copy()
    x /= np.maximum(np.linalg.norm(x, axis=1), 1e-30)[:, None]
    y -= np.sum(x * y, axis=1)[:, None] * x
    y /= np.maximum(np.linalg.norm(y, axis=1), 1e-30)[:, None]
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=2)


def _orthonormalise(r: np.ndarray) -> np.ndarray:
    """Gram-Schmidt re-orthonormalisation of a rotation matrix (~10x faster than SVD)."""
    x = r[:, 0].copy()
    y = r[:, 1].copy()
    x /= max(np.linalg.norm(x), 1e-30)
    y -= np.dot(x, y) * x
    y /= max(np.linalg.norm(y), 1e-30)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Bodies, walls, configuration
# ---------------------------------------------------------------------------

@dataclass
class SQBody:
    """A rigid body: superquadric shape and/or rigid sphere clump.

    Attributes:
        shape: ``(a, b, c, n1, n2)`` (body frame, metres and exponents).
        position: centre of mass (world frame).
        rotation: ``(3,3)`` body -> world rotation matrix.
        velocity, omega: linear velocity (m/s) and angular velocity (rad/s).
        mass, inertia: mass (kg) and diagonal body-frame inertia (kg m^2).
        clump: optional ``(offsets (K,3), radii (K,))`` rigid sphere clump.
        fixed: immovable body.
    """
    shape: Tuple[float, float, float, float, float]
    position: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.full(3, 1.0))
    clump: Optional[Tuple[np.ndarray, np.ndarray]] = None
    fixed: bool = False
    label: str = ""

    @property
    def is_clump(self) -> bool:
        return self.clump is not None

    @property
    def bounding_radius(self) -> float:
        if self.clump is not None:
            offs, radii = self.clump
            return float(np.max(np.linalg.norm(offs, axis=1) + radii))
        a, b, c = self.shape[0], self.shape[1], self.shape[2]
        return float(math.sqrt(a * a + b * b + c * c))


@dataclass
class Wall:
    """Planar half-space boundary ``normal . x >= offset`` with friction."""
    normal: np.ndarray
    offset: float = 0.0
    mu: Optional[float] = None
    mu_rolling: Optional[float] = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active: bool = True
    label: str = ""

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal - self.offset


@dataclass
class SQSystemConfig:
    """Solver configuration (SI)."""
    material: ContactMaterial = field(default_factory=ContactMaterial)
    dt: Optional[float] = None
    gravity: np.ndarray = field(default_factory=lambda: GRAVITY.copy())
    safety_factor: float = 0.05
    reference_overlap: float = 1e-3
    energy_every: int = 1
    damping_global: float = 0.0   # 1/s background viscous damping (quasi-static runs)


# ---------------------------------------------------------------------------
# Body factories
# ---------------------------------------------------------------------------

def make_superquadric_body(
    a: float, b: float, c: float, n1: float, n2: float,
    position: Sequence[float], density: float, *, rotation: Optional[np.ndarray] = None,
    label: str = "",
) -> SQBody:
    """Build a superquadric body with mass and inertia from voxel integration."""
    mass, inertia = superquadric_inertia(a, b, c, n1, n2, density, n_grid=24)
    return SQBody(
        shape=(float(a), float(b), float(c), float(n1), float(n2)),
        position=np.asarray(position, dtype=float),
        rotation=np.eye(3) if rotation is None else np.asarray(rotation, dtype=float),
        mass=mass,
        inertia=inertia,
        label=label,
    )


def make_clump_body(
    offsets: np.ndarray, radii: np.ndarray, position: Sequence[float], density: float,
    *, rotation: Optional[np.ndarray] = None, label: str = "",
) -> SQBody:
    """Build a rigid sphere-clump body (mass and inertia from the sphere set)."""
    offs = np.asarray(offsets, dtype=float)
    rr = np.asarray(radii, dtype=float)
    mass = density * float(np.sum(4.0 / 3.0 * math.pi * rr ** 3))
    inertia = np.zeros(3)
    for o, r in zip(offs, rr):
        mi = density * 4.0 / 3.0 * math.pi * r ** 3
        i0 = 0.4 * mi * r * r
        inertia += np.array([i0 + mi * (o[1] ** 2 + o[2] ** 2),
                             i0 + mi * (o[0] ** 2 + o[2] ** 2),
                             i0 + mi * (o[0] ** 2 + o[1] ** 2)])
    r_bind = float(np.max(np.linalg.norm(offs, axis=1) + rr))
    return SQBody(
        shape=(r_bind, r_bind, r_bind, 2.0, 2.0),
        position=np.asarray(position, dtype=float),
        rotation=np.eye(3) if rotation is None else np.asarray(rotation, dtype=float),
        mass=mass,
        inertia=inertia,
        clump=(offs, rr),
        label=label,
    )


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class SQSystem:
    """Granular system with contact detection, force assembly and time stepping."""

    def __init__(
        self,
        bodies: Sequence[SQBody],
        walls: Sequence[Wall] = (),
        config: Optional[SQSystemConfig] = None,
    ) -> None:
        self.bodies: List[SQBody] = list(bodies)
        self.walls: List[Wall] = list(walls)
        self.config = config or SQSystemConfig()
        self.material = self.config.material
        self.time = 0.0
        self._history: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.dt = self.config.dt if self.config.dt is not None else self.estimate_dt()
        self.energy_log: List[float] = []
        self.max_penetration_ratio = 0.0
        self.last_contacts: List[Dict[str, object]] = []
        self._clump_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._step_count = 0

    # -- geometry -----------------------------------------------------------
    def _sq_indices(self) -> List[int]:
        return [i for i, b in enumerate(self.bodies) if b.clump is None]

    def characteristic_length(self) -> float:
        lens = [b.shape[0] for b in self.bodies if not b.is_clump] or [1.0]
        return float(min(lens))

    def estimate_dt(self) -> float:
        """Stability time step from the stiffest reference contact."""
        masses = [b.mass for b in self.bodies if not b.fixed] or [1.0]
        m_min = max(min(masses), 1e-30)
        lens = [b.shape[0] for b in self.bodies] or [1.0]
        r_ref = max(min(lens), 1e-6)
        delta = self.config.reference_overlap * 2.0 * r_ref
        k_hertz = (4.0 / 3.0) * self.material.e_star * math.sqrt(r_ref / 2.0) * math.sqrt(delta)
        k_flat = area_equivalent_punch_stiffness(math.pi * r_ref ** 2, self.material.e_star)
        k_max = max(k_hertz, k_flat, 1e-12)
        return float(self.config.safety_factor * 2.0 * math.pi * math.sqrt(m_min / k_max))

    def _clump_spheres(self, refresh: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not refresh and self._clump_cache is not None:
            return self._clump_cache
        centres, radii, owners = [], [], []
        for idx, body in enumerate(self.bodies):
            if body.clump is None:
                continue
            offs, rr = body.clump
            centres.append(body.position + offs @ body.rotation.T)
            radii.append(rr)
            owners.append(np.full(len(rr), idx))
        if not centres:
            self._clump_cache = (np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int))
        else:
            self._clump_cache = (np.vstack(centres), np.concatenate(radii),
                                 np.concatenate(owners))
        return self._clump_cache

    def _candidate_pairs(self, pad: float) -> List[Tuple[int, int]]:
        idx = self._sq_indices()
        if len(idx) < 2:
            return []
        pos = np.array([self.bodies[i].position for i in idx])
        rad = np.array([self.bodies[i].bounding_radius for i in idx])
        tree = cKDTree(pos)
        pairs = tree.query_pairs(float(np.max(rad)) * 2.0 + pad, output_type="ndarray")
        return [(idx[int(p)], idx[int(q)]) for p, q in pairs
                if np.linalg.norm(pos[p] - pos[q]) <= rad[p] + rad[q] + pad]

    def _pair_geometry(self, i: int, j: int):
        bi, bj = self.bodies[i], self.bodies[j]
        return contact_geometry_batch(
            bi.position.reshape(1, 3), bi.rotation[None], tuple(bi.shape),
            bj.position.reshape(1, 3), bj.rotation[None], tuple(bj.shape),
        )

    def _facet_patch(self, shape, normal_local: np.ndarray, delta: float) -> float:
        """Flat-facet patch area for the punch branch (0 for curved shapes)."""
        if min(shape[3], shape[4]) < 8.0:
            return 0.0
        axis = int(np.argmax(np.abs(normal_local)))
        half = shape[:3]
        perm = [0, 1, 2]
        perm[axis], perm[2] = 2, axis
        a_, b_, c_ = half[perm[0]], half[perm[1]], half[perm[2]]
        n1_, n2_ = (shape[3], shape[4]) if axis == 2 else (shape[4], shape[3])
        return facet_patch_area(a_, b_, c_, n1_, n2_, delta)

    # -- force assembly -----------------------------------------------------
    def contact_forces(self) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
        self._clump_cache = None
        force = np.zeros((len(self.bodies), 3))
        torque = np.zeros((len(self.bodies), 3))
        force, torque, c1 = self._sq_sq_contacts(force, torque)
        force, torque, c2 = self._clump_contacts(force, torque)
        force, torque, c3 = self._wall_contacts(force, torque)
        self.last_contacts = c1 + c2 + c3
        return force, torque, self.last_contacts

    def _sq_sq_contacts(self, force, torque):
        mat = self.material
        contacts: List[Dict[str, object]] = []
        pairs = self._candidate_pairs(0.25 * self.characteristic_length())
        if not pairs:
            return force, torque, contacts
        idx_i = np.array([p[0] for p in pairs])
        idx_j = np.array([p[1] for p in pairs])
        m = len(pairs)
        pos_i = np.array([self.bodies[i].position for i in idx_i])
        pos_j = np.array([self.bodies[i].position for i in idx_j])
        rot_i = np.array([self.bodies[i].rotation for i in idx_i])
        rot_j = np.array([self.bodies[i].rotation for i in idx_j])
        sh_i = np.array([self.bodies[i].shape for i in idx_i], dtype=float)
        sh_j = np.array([self.bodies[i].shape for i in idx_j], dtype=float)
        # one batched geometry call for all candidate pairs (no python loop)
        geo = contact_geometry_batch(
            pos_i, rot_i, tuple(sh_i[:, k] for k in range(5)),
            pos_j, rot_j, tuple(sh_j[:, k] for k in range(5)),
        )
        delta = geo.overlap
        normal = geo.normal
        pa = geo.point_a
        pb = geo.point_b
        la = np.einsum("mji,mj->mi", rot_i, pa - pos_i)
        lb = np.einsum("mji,mj->mi", rot_j, pb - pos_j)
        k1a, k2a = principal_curvatures_local(la, sh_i[:, 0], sh_i[:, 1], sh_i[:, 2],
                                              sh_i[:, 3], sh_i[:, 4])
        k1b, k2b = principal_curvatures_local(lb, sh_j[:, 0], sh_j[:, 1], sh_j[:, 2],
                                              sh_j[:, 3], sh_j[:, 4])
        r_eff = effective_radius_hertz(k1a, k2a, k1b, k2b)
        flat = np.minimum(sh_i[:, 3], sh_j[:, 3]) >= 8.0
        patch = np.zeros(m)
        for k in np.nonzero(flat)[0]:
            n_local = rot_i[k].T @ normal[k]
            patch[k] = self._facet_patch(self.bodies[idx_i[k]].shape, n_local, float(delta[k]))

        vel_i = np.array([self.bodies[i].velocity for i in idx_i])
        vel_j = np.array([self.bodies[i].velocity for i in idx_j])
        v_n = np.sum((vel_j - vel_i) * normal, axis=1)
        m_i = np.array([self.bodies[i].mass for i in idx_i])
        m_j = np.array([self.bodies[i].mass for i in idx_j])
        f_n, k_n, branch = normal_force(
            delta, r_eff, patch, v_n, m_i * m_j / np.maximum(m_i + m_j, 1e-30), mat)

        keys = [(min(int(a), int(b)), max(int(a), int(b)), 0) for a, b in zip(idx_i, idx_j)]
        xi = np.array([self._history.get(k, np.zeros(3)) for k in keys])
        v_rel = vel_j - vel_i
        v_t = v_rel - np.sum(v_rel * normal, axis=1)[:, None] * normal
        f_t, xi_new, _ = mindlin_tangential_update(
            f_n, 0.5 * np.maximum(k_n, 1e-12), np.full(m, mat.mu), xi,
            v_t=v_t, dt=self.dt, normal=normal)
        for k, key in enumerate(keys):
            self._history[key] = xi_new[k]
        f_vec = f_n[:, None] * normal + f_t
        np.add.at(force, idx_i, -f_vec)
        np.add.at(force, idx_j, f_vec)
        torque_i = np.cross(pa - pos_i, -f_vec)
        torque_j = np.cross(pb - pos_j, f_vec)
        rb = np.minimum(np.array([self.bodies[i].bounding_radius for i in idx_i]),
                        np.array([self.bodies[i].bounding_radius for i in idx_j]))
        omega_rel = np.array([self.bodies[i].omega for i in idx_j]) - \
            np.array([self.bodies[i].omega for i in idx_i])
        r_roll = np.where(np.isfinite(r_eff), r_eff, rb)
        m_roll = rolling_resistance_torque(
            f_n, r_roll, omega_rel, np.full(m, mat.mu_rolling), rb)
        np.add.at(torque, idx_i, torque_i - m_roll)
        np.add.at(torque, idx_j, torque_j + m_roll)

        L = np.minimum(sh_i[:, 0], sh_j[:, 0])
        ratio = delta / np.maximum(2.0 * L, 1e-30)
        self.max_penetration_ratio = max(self.max_penetration_ratio, float(np.max(ratio)))
        for k in range(m):
            contacts.append({"kind": "sq-sq", "i": int(idx_i[k]), "j": int(idx_j[k]),
                             "overlap": float(delta[k]), "fn": float(f_n[k]),
                             "normal": normal[k].copy(), "branch": int(branch[k]),
                             "point": pa[k].copy()})
        return force, torque, contacts

    def _clump_contacts(self, force, torque):
        mat = self.material
        contacts: List[Dict[str, object]] = []
        sph_c, sph_r, owner = self._clump_spheres()
        if len(sph_r) == 0:
            return force, torque, contacts

        # --- sphere-sphere between different bodies ---
        tree = cKDTree(sph_c)
        pairs = tree.query_pairs(2.0 * float(np.max(sph_r)) + 1e-12, output_type="ndarray")
        if len(pairs):
            pairs = pairs[owner[pairs[:, 0]] != owner[pairs[:, 1]]]
        if len(pairs):
            keep = (sph_r[pairs[:, 0]] + sph_r[pairs[:, 1]]
                    - np.linalg.norm(sph_c[pairs[:, 1]] - sph_c[pairs[:, 0]], axis=1)) > 0
            pairs = pairs[keep]
        if len(pairs):
            d = sph_c[pairs[:, 1]] - sph_c[pairs[:, 0]]
            dist = np.linalg.norm(d, axis=1)
            normal = d / np.maximum(dist, 1e-30)[:, None]
            delta = sph_r[pairs[:, 0]] + sph_r[pairs[:, 1]] - dist
            pa = sph_c[pairs[:, 0]] + sph_r[pairs[:, 0]][:, None] * normal
            bi = owner[pairs[:, 0]]
            bj = owner[pairs[:, 1]]
            vi = np.array([self.bodies[b].velocity for b in bi])
            vj = np.array([self.bodies[b].velocity for b in bj])
            v_n = np.sum((vj - vi) * normal, axis=1)
            r_eff = (sph_r[pairs[:, 0]] * sph_r[pairs[:, 1]]
                     / (sph_r[pairs[:, 0]] + sph_r[pairs[:, 1]]))
            mi = np.array([self.bodies[b].mass for b in bi])
            mj = np.array([self.bodies[b].mass for b in bj])
            f_n, k_n, branch = normal_force(
                delta, r_eff, np.zeros(len(delta)), v_n,
                mi * mj / np.maximum(mi + mj, 1e-30), mat)
            keys = [(int(min(a_, b_)), int(max(a_, b_)), 1) for a_, b_ in pairs]
            xi = np.array([self._history.get(k, np.zeros(3)) for k in keys])
            pos_bi = np.array([self.bodies[b].position for b in bi])
            pos_bj = np.array([self.bodies[b].position for b in bj])
            om_bi = np.array([self.bodies[b].omega for b in bi])
            om_bj = np.array([self.bodies[b].omega for b in bj])
            # relative velocity *at the contact point* (includes the spin terms,
            # which are what make rolling friction and tangential slip distinct)
            v_rel = (vj + np.cross(om_bj, pa - pos_bj)) - (vi + np.cross(om_bi, pa - pos_bi))
            v_t = v_rel - np.sum(v_rel * normal, axis=1)[:, None] * normal
            f_t, xi_new, _ = mindlin_tangential_update(
                f_n, 0.5 * np.maximum(k_n, 1e-12), np.full(len(delta), mat.mu), xi,
                v_t=v_t, dt=self.dt, normal=normal)
            for k, key in enumerate(keys):
                self._history[key] = xi_new[k]
            f_vec = f_n[:, None] * normal + f_t
            np.add.at(force, bi, -f_vec)
            np.add.at(force, bj, f_vec)
            np.add.at(torque, bi, np.cross(pa - pos_bi, -f_vec))
            np.add.at(torque, bj, np.cross(pa - pos_bj, f_vec))
            # Coulomb rolling resistance on sphere-sphere contacts (its absence is
            # why a pile of ideal spheres cannot sustain any slope)
            m_roll = rolling_resistance_torque(
                f_n, r_eff, om_bj - om_bi, np.full(len(delta), mat.mu_rolling), r_eff)
            np.add.at(torque, bj, m_roll)
            np.add.at(torque, bi, -m_roll)
            contacts.extend([{"kind": "sphere-sphere", "i": int(bi[k]), "j": int(bj[k]),
                              "overlap": float(delta[k]), "fn": float(f_n[k]),
                              "normal": normal[k].copy(), "branch": int(branch[k]),
                              "point": pa[k].copy()} for k in range(len(delta))])

        # --- sphere vs superquadric (exact closest-point signed distance) ---
        sq_idx = self._sq_indices()
        if sq_idx:
            sq_pos = np.array([self.bodies[i].position for i in sq_idx])
            sq_rad = np.array([self.bodies[i].bounding_radius for i in sq_idx])
            stree = cKDTree(sq_pos)
            lists = stree.query_ball_point(
                sph_c, float(np.max(sq_rad)) + float(np.max(sph_r)) * 2.0)
            for si, cand in enumerate(lists):
                for c in cand:
                    i = sq_idx[c]
                    body = self.bodies[i]
                    centre = sph_c[si]
                    r = float(sph_r[si])
                    q_local = body.rotation.T @ (centre - body.position)
                    gap = float(signed_distance_batch(q_local.reshape(1, 3), *body.shape)[0]) - r
                    if gap >= 0.0:
                        continue
                    delta = -gap
                    p_local, _ = closest_point_batch(q_local.reshape(1, 3), *body.shape)
                    p_world = body.rotation @ p_local[0] + body.position
                    normal = p_world - centre
                    nrm = float(np.linalg.norm(normal))
                    normal = normal / nrm if nrm > 1e-15 else np.array([0.0, 0.0, 1.0])
                    k1, k2 = principal_curvatures_local(p_local, *body.shape)
                    r_eff = float(effective_radius_hertz(1.0 / r, 1.0 / r, k1, k2)[0])
                    patch = self._facet_patch(body.shape, body.rotation.T @ (-normal), delta)
                    bi, bj = int(owner[si]), i
                    v_n = float(np.dot(self.bodies[bj].velocity - self.bodies[bi].velocity,
                                       normal))
                    m_eff = (self.bodies[bi].mass * self.bodies[bj].mass
                             / max(self.bodies[bi].mass + self.bodies[bj].mass, 1e-30))
                    f_n, k_n, branch = normal_force(
                        np.array([delta]), np.array([r_eff]), np.array([patch]),
                        np.array([v_n]), np.array([m_eff]), mat)
                    key = (min(si, 100000 + i), max(si, 100000 + i), 2)
                    xi = self._history.get(key, np.zeros(3)).reshape(1, 3)
                    v_rel = self.bodies[bj].velocity - self.bodies[bi].velocity
                    v_t = v_rel - float(np.dot(v_rel, normal)) * normal
                    f_t, xi_new, _ = mindlin_tangential_update(
                        f_n, 0.5 * np.maximum(k_n, 1e-12), np.array([mat.mu]), xi,
                        v_t=v_t.reshape(1, 3), dt=self.dt, normal=normal.reshape(1, 3))
                    self._history[key] = xi_new[0]
                    f_vec = f_n[0] * normal + f_t[0]
                    force[bi] += -f_vec
                    force[bj] += f_vec
                    torque[bi] += np.cross(centre - self.bodies[bi].position, -f_vec)
                    torque[bj] += np.cross(p_world - self.bodies[bj].position, f_vec)
                    L = min(self.bodies[bi].shape[0], self.bodies[bj].shape[0])
                    self.max_penetration_ratio = max(
                        self.max_penetration_ratio, float(delta / (2.0 * L)))
                    contacts.append({"kind": "sphere-sq", "i": bi, "j": bj,
                                     "overlap": float(delta), "fn": float(f_n[0]),
                                     "normal": normal.copy(), "branch": int(branch[0]),
                                     "point": p_world.copy()})
        return force, torque, contacts

    def _wall_contacts(self, force, torque):
        mat = self.material
        contacts: List[Dict[str, object]] = []
        sph_c, sph_r, owner = self._clump_spheres()
        sq_idx = self._sq_indices()
        sq_pos = np.array([self.bodies[i].position for i in sq_idx]) if sq_idx else np.zeros((0, 3))
        sq_rot = np.array([self.bodies[i].rotation for i in sq_idx]) if sq_idx else np.zeros((0, 3, 3))
        sq_sh = np.array([self.bodies[i].shape for i in sq_idx], dtype=float) if sq_idx \
            else np.zeros((0, 5))
        sq_rad = np.array([self.bodies[i].bounding_radius for i in sq_idx]) if sq_idx else np.zeros(0)
        for wi, wall in enumerate(self.walls):
            if not wall.active:
                continue
            mu = mat.mu if wall.mu is None else wall.mu
            mu_r = mat.mu_rolling if wall.mu_rolling is None else wall.mu_rolling

            # ---- sphere clump contacts (vectorised) ----
            if len(sph_r):
                gap = sph_c @ wall.normal - wall.offset - sph_r
                sel = np.nonzero(gap < 0.0)[0]
                if len(sel):
                    delta = -gap[sel]
                    p_world = sph_c[sel] - sph_r[sel, None] * wall.normal
                    bodies_idx = owner[sel]
                    bod_pos = np.array([self.bodies[b].position for b in bodies_idx])
                    bod_mass = np.array([self.bodies[b].mass for b in bodies_idx])
                    bod_vel = np.array([self.bodies[b].velocity for b in bodies_idx])
                    bod_omega = np.array([self.bodies[b].omega for b in bodies_idx])
                    r_vec = p_world - bod_pos
                    v_contact = bod_vel + np.cross(bod_omega, r_vec)
                    v_n = (v_contact - wall.velocity) @ wall.normal
                    f_n, k_n, branch = normal_force(
                        delta, sph_r[sel], np.zeros(len(sel)), v_n, bod_mass, mat)
                    keys = [(int(b), -1 - wi, 100 + int(s)) for b, s in zip(bodies_idx, sel)]
                    xi = np.array([self._history.get(k, np.zeros(3)) for k in keys])
                    v_rel = v_contact - wall.velocity
                    v_t = v_rel - np.sum(v_rel * wall.normal, axis=1)[:, None] * wall.normal
                    f_t, xi_new, _ = mindlin_tangential_update(
                        f_n, 0.5 * np.maximum(k_n, 1e-12), np.full(len(sel), mu), xi,
                        v_t=v_t, dt=self.dt,
                        normal=np.broadcast_to(wall.normal, v_t.shape))
                    for k, key in enumerate(keys):
                        self._history[key] = xi_new[k]
                    f_vec = f_n[:, None] * wall.normal + f_t
                    np.add.at(force, bodies_idx, f_vec)
                    np.add.at(torque, bodies_idx, np.cross(r_vec, f_vec))
                    m_roll = rolling_resistance_torque(
                        f_n, sph_r[sel], bod_omega, np.full(len(sel), mu_r), sph_r[sel])
                    np.add.at(torque, bodies_idx, m_roll)
                    ratio = delta / np.maximum(2.0 * sph_r[sel], 1e-30)
                    self.max_penetration_ratio = max(self.max_penetration_ratio,
                                                     float(np.max(ratio)))
                    for k in range(len(sel)):
                        contacts.append({"kind": "wall", "i": int(bodies_idx[k]), "wall": wi,
                                         "overlap": float(delta[k]), "fn": float(f_n[k]),
                                         "normal": wall.normal.copy(),
                                         "branch": int(branch[k]),
                                         "fvec": f_vec[k].copy(),
                                         "point": p_world[k].copy()})

            # ---- superquadric contacts (batched projection) ----
            if len(sq_idx):
                centre_dist = sq_pos @ wall.normal - wall.offset
                # keep bodies whose bounding sphere can reach the wall
                cand = np.nonzero(centre_dist - sq_rad < 0.0)[0]
                if len(cand):
                    q_world = (sq_pos[cand]
                               - (sq_pos[cand] @ wall.normal - wall.offset)[:, None] * wall.normal
                               - 1e-3 * sq_rad[cand, None] * wall.normal)
                    q_local = np.einsum("mji,mj->mi", sq_rot[cand], q_world - sq_pos[cand])
                    shi = sq_sh[cand]
                    p_local, _ = closest_point_batch(q_local, shi[:, 0], shi[:, 1], shi[:, 2],
                                                    shi[:, 3], shi[:, 4])
                    p_world = np.einsum("mij,mj->mi", sq_rot[cand], p_local) + sq_pos[cand]
                    gap = p_world @ wall.normal - wall.offset
                    sel = np.nonzero(gap < 0.0)[0]
                    if len(sel):
                        delta = -gap[sel]
                        bodies_idx = np.array(sq_idx)[cand[sel]]
                        k1, k2 = principal_curvatures_local(p_local[sel], shi[sel, 0],
                                                            shi[sel, 1], shi[sel, 2],
                                                            shi[sel, 3], shi[sel, 4])
                        r_eff = effective_radius_hertz(k1, k2, np.zeros(len(sel)),
                                                       np.zeros(len(sel)))
                        pos_sel = p_world[sel]
                        bod_pos = sq_pos[cand[sel]]
                        bod_vel = np.array([self.bodies[b].velocity for b in bodies_idx])
                        bod_omega = np.array([self.bodies[b].omega for b in bodies_idx])
                        bod_mass = np.array([self.bodies[b].mass for b in bodies_idx])
                        r_vec = pos_sel - bod_pos
                        v_contact = bod_vel + np.cross(bod_omega, r_vec)
                        v_n = (v_contact - wall.velocity) @ wall.normal
                        patch = np.zeros(len(sel))
                        for k, b in enumerate(bodies_idx):
                            if min(self.bodies[b].shape[3], self.bodies[b].shape[4]) >= 8.0:
                                n_local = self.bodies[b].rotation.T @ (-wall.normal)
                                patch[k] = self._facet_patch(self.bodies[b].shape, n_local,
                                                             float(delta[k]))
                        f_n, k_n, branch = normal_force(
                            delta, np.where(np.isfinite(r_eff), r_eff, 0.0) + 0.0,
                            patch, v_n, bod_mass, mat)
                        keys = [(int(b), -1 - wi, 3) for b in bodies_idx]
                        xi = np.array([self._history.get(k, np.zeros(3)) for k in keys])
                        v_rel = v_contact - wall.velocity
                        v_t = v_rel - np.sum(v_rel * wall.normal, axis=1)[:, None] * wall.normal
                        f_t, xi_new, _ = mindlin_tangential_update(
                            f_n, 0.5 * np.maximum(k_n, 1e-12), np.full(len(sel), mu), xi,
                            v_t=v_t, dt=self.dt,
                            normal=np.broadcast_to(wall.normal, v_t.shape))
                        for k, key in enumerate(keys):
                            self._history[key] = xi_new[k]
                        f_vec = f_n[:, None] * wall.normal + f_t
                        np.add.at(force, bodies_idx, f_vec)
                        np.add.at(torque, bodies_idx, np.cross(r_vec, f_vec))
                        extents = np.array([min(self.bodies[b].shape[:3])
                                            for b in bodies_idx])
                        ratio = delta / np.maximum(2.0 * extents, 1e-30)
                        self.max_penetration_ratio = max(self.max_penetration_ratio,
                                                         float(np.max(ratio)))
                        for k in range(len(sel)):
                            contacts.append({"kind": "wall", "i": int(bodies_idx[k]), "wall": wi,
                                             "overlap": float(delta[k]), "fn": float(f_n[k]),
                                             "normal": wall.normal.copy(),
                                             "branch": int(branch[k]),
                                             "fvec": f_vec[k].copy(),
                                             "point": pos_sel[k].copy()})
        return force, torque, contacts

    # -- integration --------------------------------------------------------
    def energy(self) -> float:
        """Total mechanical energy (translational + rotational + gravity + springs)."""
        bodies = self.bodies
        if not bodies:
            return 0.0
        moving = np.array([not b.fixed for b in bodies])
        m = np.array([b.mass for b in bodies])
        inertia = np.array([b.inertia for b in bodies])
        r = np.array([b.rotation for b in bodies])
        v = np.array([b.velocity for b in bodies])
        omega = np.array([b.omega for b in bodies])
        x = np.array([b.position for b in bodies])
        w_body = np.einsum("mji,mj->mi", r, omega)
        kinetic = 0.5 * float(np.sum(m * moving * np.sum(v * v, axis=1))) \
            + 0.5 * float(np.sum(inertia * w_body ** 2 * moving[:, None]))
        potential = -float(np.sum(m * moving * (x @ self.config.gravity)))
        elastic = 0.0
        if self.last_contacts:
            fn = np.array([float(c.get("fn", 0.0)) for c in self.last_contacts])
            delta = np.array([float(c.get("overlap", 0.0)) for c in self.last_contacts])
            branch = np.array([int(c.get("branch", 0)) for c in self.last_contacts])
            elastic = float(np.sum(np.where(branch == 1, 0.5 * fn * delta, 0.4 * fn * delta)))
        return kinetic + potential + elastic

    def step(self) -> None:
        force, torque, _ = self.contact_forces()
        dt = self.dt
        bodies = self.bodies
        moving = np.array([not b.fixed for b in bodies])
        mass = np.array([b.mass for b in bodies])[:, None]
        inertia = np.array([b.inertia for b in bodies])
        rot = np.array([b.rotation for b in bodies])
        vel = np.array([b.velocity for b in bodies])
        pos = np.array([b.position for b in bodies])
        omega = np.array([b.omega for b in bodies])

        force_total = force
        if self.config.damping_global > 0.0:
            force_total = force - self.config.damping_global * mass * vel
        vel = np.where(moving[:, None],
                       vel + dt * (force_total / mass + self.config.gravity), vel)
        pos = np.where(moving[:, None], pos + dt * vel, pos)
        tau_body = np.einsum("mji,mj->mi", rot, torque)
        omega_body = np.einsum("mji,mj->mi", rot, omega)
        gyro = np.cross(omega_body, inertia * omega_body)
        alpha = (tau_body - gyro) / np.maximum(inertia, 1e-30)
        omega_body = omega_body + dt * alpha
        omega = np.einsum("mij,mj->mi", rot, omega_body)

        axis = omega * dt
        theta = np.linalg.norm(axis, axis=1)
        safe = np.maximum(theta, 1e-15)
        kx = np.zeros((len(bodies), 3, 3))
        kx[:, 0, 1] = -axis[:, 2]
        kx[:, 0, 2] = axis[:, 1]
        kx[:, 1, 0] = axis[:, 2]
        kx[:, 1, 2] = -axis[:, 0]
        kx[:, 2, 0] = -axis[:, 1]
        kx[:, 2, 1] = axis[:, 0]
        coeff_s = np.where(theta > 1e-15, np.sin(theta) / safe, 1.0)[:, None, None]
        coeff_c = np.where(theta > 1e-15, (1.0 - np.cos(theta)) / safe ** 2, 0.5)[:, None, None]
        delta_rot = coeff_s * np.einsum("mij,mjk->mik", kx, rot) \
            + coeff_c * np.einsum("mij,mjk,mkl->mil", kx, kx, rot)
        rot = _orthonormalise_batch(rot + delta_rot)

        for i, body in enumerate(bodies):
            if body.fixed:
                continue
            body.velocity = vel[i]
            body.position = pos[i]
            body.omega = omega[i]
            body.rotation = rot[i]
        self.time += dt
        self._step_count += 1
        if self._step_count % self.config.energy_every == 0:
            self.energy_log.append(self.energy())

    def run(self, n_steps: int, callback=None) -> None:
        for _ in range(n_steps):
            self.step()
            if callback is not None and callback(self) is False:
                break
        self.contact_forces()

    # -- measurement --------------------------------------------------------
    def positions(self) -> np.ndarray:
        return np.array([b.position for b in self.bodies], dtype=float)

    def stress_tensor(self, volume: float) -> np.ndarray:
        """Virial (Cauchy) stress from the current contact set."""
        sigma = np.zeros((3, 3))
        for c in self.last_contacts:
            i = int(c["i"])
            point = np.asarray(c["point"], dtype=float)
            f = float(c["fn"]) * np.asarray(c["normal"], dtype=float)
            sigma += np.outer(point - self.bodies[i].position, f)
        return sigma / max(volume, 1e-30)


# ---------------------------------------------------------------------------
# Macroscopic experiments and statistics
# ---------------------------------------------------------------------------

@dataclass
class ColumnCollapseResult:
    """Outcome of a column-collapse (heap) experiment."""
    repose_angle_deg: float
    surface_profile: List[Tuple[float, float]]
    max_penetration_ratio: float
    steady_penetration_ratio: float
    energy_drift_per_1000: float
    n_contacts: int
    force_chain_angle_deg: float
    force_chain_strength_ratio: float
    steps: int
    dt: float
    wall_clock_s: float
    final_positions: Optional[np.ndarray] = None
    contact_normals: Optional[np.ndarray] = None
    contact_forces: Optional[np.ndarray] = None


@dataclass
class ShearCellResult:
    """Outcome of the moving-floor direct-shear proxy."""
    mu_apparent: float
    sigma_normal_pa: float
    tau_shear_pa: float
    max_penetration_ratio: float
    steps: int
    wall_clock_s: float
    n_samples: int = 0


def repose_angle_from_cloud(
    positions: np.ndarray, axis_xy: Tuple[float, float] = (0.0, 0.0),
    n_bins: int = 10, min_count: int = 3, use_quantile: float = 1.0,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Angle of repose from the free surface of a heap (degrees).

    The heap is binned radially around ``axis_xy``; each annulus contributes the
    ``use_quantile`` quantile of its particle heights (robust against ejecta).
    A straight line is fitted to the flank (25%-90% of the maximum radius) and
    ``theta_r = atan(-dz/dr)``.
    """
    pts = np.asarray(positions, dtype=float)
    r = np.linalg.norm(pts[:, :2] - np.asarray(axis_xy), axis=1)
    if len(r) < 8:
        return 0.0, []
    edges = np.linspace(0.0, float(np.max(r)) * 1.001, n_bins + 1)
    profile: List[Tuple[float, float]] = []
    for k in range(n_bins):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if int(np.sum(m)) < min_count:
            continue
        profile.append((0.5 * (edges[k] + edges[k + 1]),
                        float(np.quantile(pts[m, 2], use_quantile))))
    if len(profile) < 3:
        return 0.0, profile
    prof = np.array(profile)
    lo = 0.25 * float(np.max(prof[:, 0]))
    hi = 0.90 * float(np.max(prof[:, 0]))
    m = (prof[:, 0] >= lo) & (prof[:, 0] <= hi)
    if int(np.sum(m)) < 2:
        m = np.ones(len(prof), dtype=bool)
    slope = float(np.polyfit(prof[m, 0], prof[m, 1], 1)[0])
    return float(math.degrees(math.atan(-slope))), profile


def force_chain_statistics(contacts: Iterable[Dict[str, object]]) -> Dict[str, float]:
    """Orientation and heterogeneity statistics of a contact force network.

    * ``angle_deg``: force-weighted mean angle of the strong contact normals
      from the vertical (all normals oriented into the upper half-space first).
    * ``strength_ratio``: ``mean strong force / mean force`` (Radjai's
      ``f_strong / f``), a scale-free measure of how concentrated the load is on
      force chains.
    """
    items = [(np.asarray(c["normal"], dtype=float), float(c["fn"]))
             for c in contacts if float(c.get("fn", 0.0)) > 0.0]
    if not items:
        return {"angle_deg": 0.0, "strength_ratio": 0.0, "n": 0.0}
    f = np.array([i[1] for i in items])
    mean_f = float(np.mean(f))
    strong = [(n, fi) for n, fi in items if fi >= mean_f] or items
    weights = np.array([s[1] for s in strong])
    normals = np.array([s[0] for s in strong])
    sign = np.sign(normals[:, 2])
    sign[sign == 0] = 1.0
    aligned = normals * sign[:, None]
    mean_vec = (weights[:, None] * aligned).sum(axis=0) / float(np.sum(weights))
    nrm = float(np.linalg.norm(mean_vec))
    angle = 0.0 if nrm < 1e-12 else math.degrees(
        math.acos(min(1.0, abs(mean_vec[2] / nrm))))
    return {"angle_deg": angle,
            "strength_ratio": float(np.mean(weights)) / max(mean_f, 1e-30),
            "n": float(len(items))}


def stress_tensor_from_contacts(
    contacts: Iterable[Dict[str, object]], positions: np.ndarray, volume: float,
) -> np.ndarray:
    """Virial stress tensor from a contact list."""
    sigma = np.zeros((3, 3))
    for c in contacts:
        i = int(c["i"])
        point = np.asarray(c["point"], dtype=float)
        f = float(c["fn"]) * np.asarray(c["normal"], dtype=float)
        sigma += np.outer(point - positions[i], f)
    return sigma / max(volume, 1e-30)


def energy_drift_per_1000(energy: Sequence[float], n_steps: int) -> float:
    """Relative energy drift per 1000 steps, ``|E_end - E_start| / |E_start|``."""
    if len(energy) < 2 or n_steps <= 0:
        return 0.0
    e0 = float(energy[0])
    if abs(e0) < 1e-30:
        return 0.0
    return abs(float(energy[-1]) - e0) / abs(e0) * (1000.0 / n_steps)


def initial_column_packing(
    n_particles: int, radius: float, column_half_width: float, z0: float,
    *, spacing_factor: float = 2.1, seed: int = 0,
) -> np.ndarray:
    """Deterministic loose cubic packing of a square column (particle centres).

    The grid is *centred* inside the gate half-width so that no particle starts
    overlapping a gate (an initially penetrating particle injects energy and the
    run explodes; this was the first bug found while validating the driver).
    """
    rng = np.random.default_rng(seed)
    spacing = spacing_factor * radius
    per_side = max(int(math.floor(2.0 * column_half_width / spacing)), 1)
    per_layer = per_side ** 2
    n_layers = int(math.ceil(n_particles / per_layer))
    offset = -0.5 * (per_side - 1) * spacing
    positions = []
    for kz in range(n_layers):
        for ky in range(per_side):
            for kx in range(per_side):
                positions.append([offset + kx * spacing,
                                  offset + ky * spacing,
                                  z0 + kz * spacing])
    pos = np.array(positions[:n_particles], dtype=float)
    pos += 0.02 * radius * rng.standard_normal(pos.shape)
    return pos


def run_column_collapse(
    bodies: Sequence[SQBody],
    walls: Sequence[Wall],
    config: SQSystemConfig,
    *,
    hold_steps: int,
    settle_steps: int,
    gate_walls: Sequence[int] = (),
    axis_xy: Tuple[float, float] = (0.0, 0.0),
) -> ColumnCollapseResult:
    """Column-collapse repose experiment.

    The gates (indices into ``walls``) hold the column for ``hold_steps``; they
    are then switched off and the material is given ``settle_steps`` to deposit.
    Repose angle, penetration, energy drift and force-chain statistics are
    measured on the final state.
    """
    t0 = time.perf_counter()
    sysm = SQSystem(bodies, walls, config)
    sysm.run(hold_steps)
    for wi in gate_walls:
        sysm.walls[wi].active = False
    start = len(sysm.energy_log)
    sysm.run(settle_steps)
    e_window = sysm.energy_log[start:]
    drift = energy_drift_per_1000(e_window, len(e_window))
    pos = sysm.positions()
    angle, profile = repose_angle_from_cloud(pos, axis_xy=axis_xy)
    chain = force_chain_statistics(sysm.last_contacts)
    normals = (np.array([c["normal"] for c in sysm.last_contacts])
               if sysm.last_contacts else np.zeros((0, 3)))
    forces = (np.array([c["fn"] for c in sysm.last_contacts])
              if sysm.last_contacts else np.zeros(0))
    steady_pen = 0.0
    for c in sysm.last_contacts:
        ov = float(c.get("overlap", 0.0))
        i = int(c["i"])
        D = 2.0 * min(sysm.bodies[i].shape[:3])
        if D > 0:
            steady_pen = max(steady_pen, ov / D)
    return ColumnCollapseResult(
        repose_angle_deg=angle,
        surface_profile=profile,
        max_penetration_ratio=sysm.max_penetration_ratio,
        steady_penetration_ratio=steady_pen,
        energy_drift_per_1000=drift,
        n_contacts=len(sysm.last_contacts),
        force_chain_angle_deg=chain["angle_deg"],
        force_chain_strength_ratio=chain["strength_ratio"],
        steps=hold_steps + settle_steps,
        dt=sysm.dt,
        wall_clock_s=time.perf_counter() - t0,
        final_positions=pos,
        contact_normals=normals,
        contact_forces=forces,
    )


def run_shear_cell(
    bodies: Sequence[SQBody],
    walls: Sequence[Wall],
    config: SQSystemConfig,
    *,
    shear_speed: float,
    normalise_steps: int,
    shear_steps: int,
    floor_wall: int = 0,
    area: float = 1.0,
    sample_every: int = 10,
    settle_steps: int = 0,
) -> ShearCellResult:
    """Moving-floor direct-shear proxy: apparent friction ``mu = tau / sigma``.

    The floor (``walls[floor_wall]``) is driven at ``shear_speed`` in its own
    plane.  ``sigma`` is the mean normal traction on the floor and ``tau`` the
    mean tangential traction magnitude, both averaged over the shearing window.
    For a cohesionless assembly the ratio saturates at the *effective internal*
    friction coefficient because the shear band localises inside the sample when
    the wall is rougher than the material, which is what makes this a usable
    shear-box analogue (Jenike-cell style, simplified).
    """
    t0 = time.perf_counter()
    sysm = SQSystem(bodies, walls, config)
    sysm.walls[floor_wall].velocity = np.asarray(walls[floor_wall].velocity, dtype=float)
    if settle_steps:
        sysm.run(settle_steps)
    sysm.run(normalise_steps)
    floor = sysm.walls[floor_wall]
    tangent = floor.velocity - np.dot(floor.velocity, floor.normal) * floor.normal
    t_hat = tangent / np.linalg.norm(tangent)
    sigma_acc, tau_acc, n_samp = 0.0, 0.0, 0
    for k in range(shear_steps):
        sysm.step()
        if k % sample_every != 0 and k != shear_steps - 1:
            continue
        contacts = sysm.contact_forces()[2]
        fn_tot = 0.0
        ft_tot = 0.0
        for c in contacts:
            if c["kind"] != "wall" or int(c.get("wall", -1)) != floor_wall:
                continue
            f_vec = float(c["fn"]) * np.asarray(c["normal"], dtype=float)
            fn_tot += float(np.dot(f_vec, floor.normal))
            ft_vec = np.asarray(c.get("fvec", f_vec), dtype=float) - f_vec
            ft_tot += float(abs(np.dot(ft_vec, t_hat)))
        sigma_acc += fn_tot
        tau_acc += ft_tot
        n_samp += 1
    sigma = sigma_acc / max(n_samp, 1) / area
    tau = tau_acc / max(n_samp, 1) / area
    return ShearCellResult(
        mu_apparent=float(tau / sigma) if sigma > 0 else 0.0,
        sigma_normal_pa=float(sigma),
        tau_shear_pa=float(tau),
        max_penetration_ratio=sysm.max_penetration_ratio,
        steps=normalise_steps + shear_steps + settle_steps,
        wall_clock_s=time.perf_counter() - t0,
        n_samples=n_samp,
    )