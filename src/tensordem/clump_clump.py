"""Clump-clump contact: geometric junction merging + Hertz-Mindlin pair forces.

The multi-sphere (clump) path of this package previously stopped at clump-wall
(:class:`tensordem.clump_dynamics.PlaneContact`) and clump-sphere
(:class:`tensordem.clump_dynamics.SphereObstacle`) contacts; clump-clump was
``NOT_SUPPORTED`` (``docs/clump_multisphere_particles.md``).  This module closes
that gap with a member-summing narrow phase plus the geometric merging step that
a naive member sum is missing.

Why merging is not optional
---------------------------
A clump is a rigid union of overlapping spheres.  When two clumps approach it is
possible for several member pairs ``(i in A, j in B)`` to indent the *same*
physical junction: the two unions touch over one lens-shaped region which
happens to be sampled by two (or four) member pairs.  Summing the per-pair
forces then counts the same contact twice, so the repulsion becomes a
non-physical multiple of the true junction force -- exactly ``2x`` for two
symmetric duplicate pairs.  This module therefore

1. enumerates the member pairs and keeps those with positive penetration
   (:func:`member_pair_candidates`);
2. clusters candidates that describe the same junction using a *contact-patch*
   criterion: two candidates are the same junction when their contact points are
   closer than the sum of their Hertz patch radii ``a = sqrt(R* delta)`` and
   their normals agree within ``merge_angle`` (:func:`merge_candidates`,
   single-linkage, so the result does not depend on the pair order);
3. carries **one** force per junction, at the representative point/normal of the
   cluster and with the *largest* penetration in the cluster (the deepest
   indentation of that junction; adding the cluster's penetrations would
   re-introduce the double counting inside the force law).

Contact law (Hertz-Mindlin)
---------------------------
For a junction with penetration ``delta``, effective radius
``R* = r_i r_j / (r_i + r_j)`` and reduced modulus
``1/E* = (1 - nu_A^2)/E_A + (1 - nu_B^2)/E_B``:

.. code-block:: text

   F_n^elastic = (4/3) E* sqrt(R*) delta^{3/2}          (Hertz)
   k_n(delta)  = dF/ddelta = 2 E* sqrt(R* delta)        (tangent stiffness)
   c_n         = 2 zeta sqrt(m_eff k_n(delta))          (viscous dashpot)
   F_n         = max(F_n^elastic - c_n v_n, 0)
   k_t(delta)  = 8 G* sqrt(R* delta)                    (Mindlin)
   F_t         = -k_t xi,   |F_t| <= mu F_n             (Cundall-Strack history)

``G*`` uses ``1/G* = (2 - nu_A)/G_A + (2 - nu_B)/G_B`` and ``zeta`` follows from
the requested coefficient of restitution through the log-decrement law of
:func:`tensordem.clump_dynamics.restitution_to_damping_ratio`.  Both reduced
moduli are the *same* formulas as
:func:`tensordem.hertz_mindlin.hertz_reduced_shear_modulus` /
``tensordem.hertz_mindlin.mindlin_stiffness_ratio``; they are evaluated in NumPy
here so the clump path stays free of the torch dependency, and
``tests/test_clump_clump_contact.py`` pins the numerical equality against the
torch implementation.

Third law and angular momentum
------------------------------
Every junction publishes one contact point ``x`` and one force ``F`` on body B;
body A receives ``-F`` and the torques are ``(x - x_A) x (-F)`` and
``(x - x_B) x F``.  The two moment arms differ by exactly the vector between the
two centres of mass, so the pair contributes

.. code-block:: text

   x_A x F_A + tau_A + x_B x F_B + tau_B = x x F - x x F = 0

to the total moment about the origin: linear and angular momentum are conserved
to round-off, not merely to truncation-error order.  Rolling/drilling resistance
(as implemented for clump-wall in :mod:`tensordem.clump_rolling`) is
deliberately *not* applied here because a per-member internal couple has no
third-law partner and would break the angular-momentum identity; this is
documented as a limitation in ``docs/clump_clump_contact.md``.

References
----------
* Johnson, *Contact Mechanics* (1985), ch. 4 (Hertz) and eq. 3.39 (reduced shear
  modulus).
* Mindlin & Deresiewicz, *J. Appl. Mech.* 20 (1953) -- no-slip tangential
  stiffness ``k_t = 8 G* a``.
* Cundall & Strack, *Geotechnique* 29 (1979) -- hysteretic tangential spring with
  friction-cone projection.
* Bourrier et al., *Comput. Methods Appl. Mech. Engrg.* 267 (2013) -- contact
  clustering for clump/polyhedron narrow phases (the merging criterion used
  here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np

from .clump_dynamics import (
    ClumpBody,
    ClumpContact,
    quaternion_from_axis_angle,
    quaternion_multiply,
    restitution_to_damping_ratio,
)

__all__ = [
    "ClumpClumpConfig",
    "ClumpClumpContact",
    "ClumpClumpError",
    "ClumpClumpResponse",
    "ContactJoint",
    "MemberPairCandidate",
    "clump_clump_joints",
    "duplicate_force_ratio",
    "hertz_normal_force",
    "member_pair_candidates",
    "merge_candidates",
    "pair_material_constants",
    "pair_step",
    "total_angular_momentum",
    "total_linear_momentum",
]

# A junction needs strictly positive penetration; the tolerance only guards a
# contact point that sits on the surface within round-off (length, metres).
_GEOM_TOL = 0.0


class ClumpClumpError(ValueError):
    """Invalid clump-clump contact configuration or call."""


# --------------------------------------------------------------------------
# Small vector helpers (np.cross on length-3 arrays is surprisingly slow)
# --------------------------------------------------------------------------
def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array((a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]), dtype=np.float64)


def _effective_mass_along(clump, arm: np.ndarray, normal: np.ndarray,
                           inverse_inertia: np.ndarray) -> float:
    """Fast copy of :func:`tensordem.clump_dynamics.effective_mass_along`.

    ``1/m_eff = 1/m + n . [(I^-1 (a x n)) x a]``.  Bitwise equality with the
    public helper is pinned in ``tests/test_clump_clump_contact.py``.
    """
    rotational = _cross(inverse_inertia @ _cross(arm, normal), arm)
    inverse = 1.0 / clump.mass + float(normal @ rotational)
    if inverse <= 0.0:
        return math.inf
    return 1.0 / inverse


def _angular_acceleration(body: ClumpBody, rotation: np.ndarray,
                          torque_world: np.ndarray) -> np.ndarray:
    """Fast copy of :meth:`tensordem.clump_dynamics.ClumpBody.angular_acceleration_body`."""
    torque_body = rotation.T @ torque_world
    omega = body.angular_velocity
    gyroscopic = _cross(omega, body.inertia_body @ omega)
    return body.inverse_inertia_body @ (torque_body - gyroscopic)


# --------------------------------------------------------------------------
# Material constants
# --------------------------------------------------------------------------
def pair_material_constants(young_modulus_a: float, poisson_a: float,
                            young_modulus_b: float, poisson_b: float
                            ) -> tuple[float, float]:
    """``(E*, G*)`` of a two-material Hertz-Mindlin contact (Pa).

    ``1/E* = (1 - nu_a^2)/E_a + (1 - nu_b^2)/E_b`` and
    ``1/G* = (2 - nu_a)/G_a + (2 - nu_b)/G_b`` with ``G = E / (2 (1 + nu))``.
    """
    for name, value in (("young_modulus_a", young_modulus_a),
                        ("young_modulus_b", young_modulus_b)):
        if not math.isfinite(value) or value <= 0.0:
            raise ClumpClumpError(f"{name} must be finite and positive")
    for name, value in (("poisson_a", poisson_a), ("poisson_b", poisson_b)):
        if not (math.isfinite(value) and 0.0 <= value < 0.5):
            raise ClumpClumpError(f"{name} must be in [0, 0.5)")
    shear_a = young_modulus_a / (2.0 * (1.0 + poisson_a))
    shear_b = young_modulus_b / (2.0 * (1.0 + poisson_b))
    inverse_e = ((1.0 - poisson_a ** 2) / young_modulus_a
                 + (1.0 - poisson_b ** 2) / young_modulus_b)
    inverse_g = (2.0 - poisson_a) / shear_a + (2.0 - poisson_b) / shear_b
    return 1.0 / inverse_e, 1.0 / inverse_g


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ClumpClumpConfig:
    """Hertz-Mindlin parameters and geometric-merge tolerances (SI units).

    ``young_modulus`` / ``poisson_ratio`` describe the clump material; both
    bodies share it unless ``young_modulus_b`` / ``poisson_ratio_b`` are given
    (bimaterial pair).

    ``merge_factor`` scales the contact-patch size used as the merge length: two
    member-pair candidates are the same junction when
    ``|x_p - x_q| <= merge_factor * (a_p + a_q)`` with ``a = sqrt(R* delta)``.
    ``merge_factor = 0`` disables merging geometrically (``merging=False`` does
    the same explicitly); ``merge_factor >> 1`` over-merges and is only useful as
    a negative control.

    ``merge_angle`` is the largest angle (radians) between the two candidate
    normals for them to count as the same junction.
    """

    young_modulus: float = 1.0e8
    poisson_ratio: float = 0.3
    young_modulus_b: float | None = None
    poisson_ratio_b: float | None = None
    restitution: float = 0.9
    friction: float = 0.3
    merging: bool = True
    merge_factor: float = 1.0
    merge_angle: float = math.radians(30.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.young_modulus) or self.young_modulus <= 0.0:
            raise ClumpClumpError("young_modulus must be finite and positive")
        if not (math.isfinite(self.poisson_ratio) and 0.0 <= self.poisson_ratio < 0.5):
            raise ClumpClumpError("poisson_ratio must be in [0, 0.5)")
        if self.young_modulus_b is not None:
            if not math.isfinite(self.young_modulus_b) or self.young_modulus_b <= 0.0:
                raise ClumpClumpError("young_modulus_b must be finite and positive")
        if self.poisson_ratio_b is not None:
            if not (math.isfinite(self.poisson_ratio_b)
                    and 0.0 <= self.poisson_ratio_b < 0.5):
                raise ClumpClumpError("poisson_ratio_b must be in [0, 0.5)")
        if not (0.0 < self.restitution <= 1.0):
            raise ClumpClumpError("restitution must be in (0, 1]")
        if self.friction < 0.0:
            raise ClumpClumpError("friction must be non-negative")
        if self.merge_factor < 0.0:
            raise ClumpClumpError("merge_factor must be non-negative")
        if not (0.0 < self.merge_angle <= math.pi):
            raise ClumpClumpError("merge_angle must be in (0, pi]")

    @property
    def young_modulus_b_value(self) -> float:
        return self.young_modulus if self.young_modulus_b is None else float(
            self.young_modulus_b)

    @property
    def poisson_ratio_b_value(self) -> float:
        return self.poisson_ratio if self.poisson_ratio_b is None else float(
            self.poisson_ratio_b)

    @property
    def e_star(self) -> float:
        return pair_material_constants(
            self.young_modulus, self.poisson_ratio,
            self.young_modulus_b_value, self.poisson_ratio_b_value)[0]

    @property
    def g_star(self) -> float:
        return pair_material_constants(
            self.young_modulus, self.poisson_ratio,
            self.young_modulus_b_value, self.poisson_ratio_b_value)[1]

    @property
    def damping_ratio(self) -> float:
        return restitution_to_damping_ratio(self.restitution)


# --------------------------------------------------------------------------
# Geometry: member-pair candidates and junction merging
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MemberPairCandidate:
    """One overlapping member pair of two clumps (pure geometry)."""

    index_a: int
    index_b: int
    penetration: float
    point: np.ndarray
    normal: np.ndarray
    radius_eff: float
    patch_radius: float

    def as_dict(self) -> dict[str, object]:
        return {
            "pair": (self.index_a, self.index_b),
            "penetration": self.penetration,
            "point": self.point.tolist(),
            "normal": self.normal.tolist(),
            "radius_eff": self.radius_eff,
            "patch_radius": self.patch_radius,
        }


@dataclass(frozen=True)
class ContactJoint:
    """One geometric junction: the force is applied here exactly once."""

    members: tuple[tuple[int, int], ...]
    penetration: float
    point: np.ndarray
    normal: np.ndarray
    radius_eff: float
    patch_radius: float

    @property
    def key(self) -> tuple[int, int]:
        """History key: the deepest member pair of the junction."""
        return self.members[0]

    @property
    def n_members(self) -> int:
        return len(self.members)

    def as_dict(self) -> dict[str, object]:
        return {
            "members": [list(pair) for pair in self.members],
            "penetration": self.penetration,
            "point": self.point.tolist(),
            "normal": self.normal.tolist(),
            "radius_eff": self.radius_eff,
            "patch_radius": self.patch_radius,
        }


@dataclass
class _Kinematics:
    """Per-body quantities cached for one force evaluation.

    Reproduces exactly what :class:`~tensordem.clump_dynamics.ClumpBody` exposes
    (:meth:`~ClumpBody.member_centers`, ``rotation_matrix``,
    ``inverse_inertia_world``, ``angular_velocity_world`` and
    ``point_velocity``) without rebuilding the rotation matrix five times per
    contact; ``tests/test_clump_clump_contact.py`` pins the bitwise equivalence.
    """

    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    omega_world: np.ndarray
    inverse_inertia: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    mass: float

    @classmethod
    def of(cls, body: ClumpBody) -> "_Kinematics":
        rotation = body.rotation_matrix()
        return cls(
            position=np.asarray(body.position, dtype=np.float64),
            velocity=np.asarray(body.velocity, dtype=np.float64),
            rotation=rotation,
            omega_world=rotation @ body.angular_velocity,
            inverse_inertia=rotation @ body.inverse_inertia_body @ rotation.T,
            centers=body.position + body.clump.local_centers @ rotation.T,
            radii=np.asarray(body.clump.radii, dtype=np.float64),
            mass=float(body.clump.mass),
        )

    def point_velocity(self, point: np.ndarray) -> np.ndarray:
        return self.velocity + _cross(self.omega_world, point - self.position)


def _candidates_from(kin_a: _Kinematics, kin_b: _Kinematics
                     ) -> list[MemberPairCandidate]:
    out: list[MemberPairCandidate] = []
    for i in range(kin_a.centers.shape[0]):
        for j in range(kin_b.centers.shape[0]):
            delta_vec = kin_b.centers[j] - kin_a.centers[i]
            distance = math.sqrt(float(delta_vec @ delta_vec))
            penetration = float(kin_a.radii[i] + kin_b.radii[j] - distance)
            if penetration <= _GEOM_TOL:
                continue
            if distance > 0.0:
                normal = delta_vec / distance
            else:  # coincident member centres: any axis is equally valid
                normal = np.array([0.0, 0.0, 1.0])
            point = kin_a.centers[i] + (kin_a.radii[i] - 0.5 * penetration) * normal
            radius_eff = float(kin_a.radii[i] * kin_b.radii[j]
                               / (kin_a.radii[i] + kin_b.radii[j]))
            out.append(MemberPairCandidate(
                index_a=i, index_b=j, penetration=penetration, point=point,
                normal=normal, radius_eff=radius_eff,
                patch_radius=math.sqrt(radius_eff * penetration)))
    return out


def member_pair_candidates(body_a: ClumpBody, body_b: ClumpBody
                           ) -> list[MemberPairCandidate]:
    """All member pairs with strictly positive penetration (no merging).

    The contact point is the mid-point of the overlap along the line of centres,
    ``x = c_i + (r_i - delta/2) n`` with ``n`` the unit vector from member ``i``
    of A to member ``j`` of B, so the normal points from A into B and the force
    on B is ``+F_n n``.
    """
    return _candidates_from(_Kinematics.of(body_a), _Kinematics.of(body_b))


def _same_junction(p: MemberPairCandidate, q: MemberPairCandidate,
                   merge_factor: float, cos_angle_limit: float) -> bool:
    offset = p.point - q.point
    if math.sqrt(float(offset @ offset)) > merge_factor * (
            p.patch_radius + q.patch_radius):
        return False
    return float(p.normal @ q.normal) >= cos_angle_limit


def merge_candidates(candidates: Sequence[MemberPairCandidate],
                     config: ClumpClumpConfig | None = None
                     ) -> list[ContactJoint]:
    """Cluster candidates that describe the same junction (single linkage).

    The cluster representative uses the deepest candidate (largest penetration)
    for the force magnitude and radius, plus the penetration-weighted mean point
    and normal, so a symmetric duplicate pair yields the symmetry axis exactly.
    """
    cfg = config or ClumpClumpConfig()
    items = list(candidates)
    n = len(items)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cos_limit = math.cos(cfg.merge_angle)
    for p in range(n):
        for q in range(p + 1, n):
            if _same_junction(items[p], items[q], cfg.merge_factor, cos_limit):
                root_p, root_q = find(p), find(q)
                if root_p != root_q:
                    parent[max(root_p, root_q)] = min(root_p, root_q)

    clusters: dict[int, list[int]] = {}
    for index in range(n):
        clusters.setdefault(find(index), []).append(index)

    joints: list[ContactJoint] = []
    for members in clusters.values():
        group = [items[k] for k in members]
        weights = np.array([c.penetration for c in group], dtype=np.float64)
        total = float(weights.sum())
        point = np.zeros(3)
        normal = np.zeros(3)
        for weight, candidate in zip(weights, group):
            point += weight * candidate.point
            normal += weight * candidate.normal
        point /= total
        norm = math.sqrt(float(normal @ normal))
        normal = normal / norm if norm > 0.0 else group[0].normal.copy()
        deepest = max(group, key=lambda c: c.penetration)
        pairs = tuple(sorted((c.index_a, c.index_b) for c in group))
        deepest_pair = (deepest.index_a, deepest.index_b)
        ordered = (deepest_pair,) + tuple(p for p in pairs if p != deepest_pair)
        joints.append(ContactJoint(
            members=ordered,
            penetration=float(deepest.penetration),
            point=point,
            normal=normal,
            radius_eff=deepest.radius_eff,
            patch_radius=deepest.patch_radius))
    joints.sort(key=lambda j: j.members)
    return joints


def _joints_from(candidates: Sequence[MemberPairCandidate],
                 config: ClumpClumpConfig) -> list[ContactJoint]:
    if not config.merging or config.merge_factor == 0.0:
        joints = [ContactJoint(members=((c.index_a, c.index_b),),
                               penetration=c.penetration, point=c.point,
                               normal=c.normal, radius_eff=c.radius_eff,
                               patch_radius=c.patch_radius)
                  for c in candidates]
        joints.sort(key=lambda j: j.members)
        return joints
    return merge_candidates(candidates, config)


def clump_clump_joints(body_a: ClumpBody, body_b: ClumpBody,
                       config: ClumpClumpConfig | None = None
                       ) -> list[ContactJoint]:
    """Junctions used by the force law; merged unless ``config.merging`` is off."""
    cfg = config or ClumpClumpConfig()
    return _joints_from(member_pair_candidates(body_a, body_b), cfg)


# --------------------------------------------------------------------------
# Force law
# --------------------------------------------------------------------------
def hertz_normal_force(penetration: float, radius_eff: float, e_star: float,
                       velocity_normal: float = 0.0, mass_eff: float = math.inf,
                       damping_ratio: float = 0.0) -> float:
    """Hertz repulsion with the viscous dashpot: ``max(F_elastic - c_n v_n, 0)``.

    ``velocity_normal`` is ``(v_B - v_A) . n``, so a closing contact has a
    negative value and the dashpot adds to the repulsion.
    """
    if penetration <= 0.0:
        return 0.0
    elastic = (4.0 / 3.0) * e_star * math.sqrt(radius_eff) * penetration ** 1.5
    if damping_ratio <= 0.0 or not math.isfinite(mass_eff):
        return elastic
    tangent = 2.0 * e_star * math.sqrt(radius_eff * penetration)
    c_n = 2.0 * damping_ratio * math.sqrt(mass_eff * tangent)
    return max(elastic - c_n * velocity_normal, 0.0)


@dataclass
class ClumpClumpResponse:
    """Forces/torques of one clump-clump evaluation (SI)."""

    force_a: np.ndarray
    torque_a: np.ndarray
    force_b: np.ndarray
    torque_b: np.ndarray
    joints: list[ContactJoint]
    normal_forces: np.ndarray
    tangential_forces: list[np.ndarray]
    candidates: int

    @property
    def force_magnitude(self) -> float:
        return float(np.linalg.norm(self.force_a))

    def as_dict(self) -> dict[str, object]:
        return {
            "force_a": self.force_a.tolist(),
            "torque_a": self.torque_a.tolist(),
            "force_b": self.force_b.tolist(),
            "torque_b": self.torque_b.tolist(),
            "force_magnitude": self.force_magnitude,
            "joints": len(self.joints),
            "candidates": self.candidates,
            "normal_forces": self.normal_forces.tolist(),
            "tangential_forces": [f.tolist() for f in self.tangential_forces],
            "joints_detail": [j.as_dict() for j in self.joints],
        }


class ClumpClumpContact:
    """Bidirectional clump-clump interaction: Hertz-Mindlin with merged junctions.

    The object owns the per-junction tangential history, so the same instance has
    to be used across the time steps of a collision.  ``dt`` enters only through
    the tangential slip increment ``xi += v_t dt`` and the Coulomb projection.
    """

    def __init__(self, config: ClumpClumpConfig | None = None) -> None:
        self.config = config or ClumpClumpConfig()
        self._tangential: dict[tuple[int, int], np.ndarray] = {}
        self._last: ClumpClumpResponse | None = None
        self._elastic_energy = 0.0

    # -- state -----------------------------------------------------------
    def reset(self) -> None:
        self._tangential.clear()
        self._last = None
        self._elastic_energy = 0.0

    @property
    def last(self) -> ClumpClumpResponse:
        if self._last is None:
            raise ClumpClumpError("no evaluation has been performed yet")
        return self._last

    def joints(self, body_a: ClumpBody, body_b: ClumpBody) -> list[ContactJoint]:
        return clump_clump_joints(body_a, body_b, self.config)

    def stored_energy(self) -> float:
        """Reversible elastic energy of the junctions (Hertz + Mindlin) [J].

        Add it to ``body.energy()`` to close the mechanical energy budget: the
        dashpot and the Coulomb projection are the only dissipative terms, so
        ``KE + gravity + stored_energy`` must be non-increasing.
        """
        return float(self._elastic_energy)

    # -- forces ----------------------------------------------------------
    def forces(self, body_a: ClumpBody, body_b: ClumpBody, *, dt: float = 0.0,
               commit: bool = True) -> ClumpClumpResponse:
        """Evaluate every junction; ``force_b``/``torque_b`` are the partners.

        ``force_b == -force_a`` and the two torques are taken about the *same*
        contact points, so the pair contributes zero net force and zero net
        moment about any origin (Newton's third law by construction).
        """
        if dt < 0.0:
            raise ClumpClumpError("dt must be non-negative")
        return self._forces_from(body_a, body_b,
                                 _Kinematics.of(body_a), _Kinematics.of(body_b),
                                 dt=dt, commit=commit)

    def _forces_from(self, body_a: ClumpBody, body_b: ClumpBody,
                     kin_a: _Kinematics, kin_b: _Kinematics, *, dt: float,
                     commit: bool) -> "ClumpClumpResponse":
        cfg = self.config
        e_star, g_star = cfg.e_star, cfg.g_star
        zeta, mu = cfg.damping_ratio, cfg.friction
        candidates = _candidates_from(kin_a, kin_b)
        joints = _joints_from(candidates, cfg)

        force_a = np.zeros(3)
        torque_a = np.zeros(3)
        torque_b = np.zeros(3)
        elastic_energy = 0.0
        normal_forces = np.zeros(len(joints))
        tangential_forces: list[np.ndarray] = []
        seen: set[tuple[int, int]] = set()

        for index, joint in enumerate(joints):
            normal = joint.normal
            point = joint.point
            arm_a = point - kin_a.position
            arm_b = point - kin_b.position
            relative = kin_b.point_velocity(point) - kin_a.point_velocity(point)
            velocity_normal = float(relative @ normal)
            if zeta > 0.0:
                mass_eff = 1.0 / (1.0 / _effective_mass_along(
                    body_a.clump, arm_a, normal, kin_a.inverse_inertia)
                    + 1.0 / _effective_mass_along(
                        body_b.clump, arm_b, normal, kin_b.inverse_inertia))
            else:
                mass_eff = math.inf
            elastic_normal = hertz_normal_force(joint.penetration, joint.radius_eff,
                                                e_star)
            f_n = hertz_normal_force(joint.penetration, joint.radius_eff, e_star,
                                     velocity_normal, mass_eff, zeta)

            key = joint.key
            xi = self._tangential.get(key, np.zeros(3))
            if dt > 0.0:
                xi = xi + (relative - velocity_normal * normal) * dt
            k_t = 8.0 * g_star * math.sqrt(
                joint.radius_eff * max(joint.penetration, 0.0))
            f_t = -k_t * xi if k_t > 0.0 else np.zeros(3)
            magnitude = math.sqrt(float(f_t @ f_t))
            limit = mu * f_n
            if magnitude > limit and magnitude > 0.0:
                f_t = f_t * (limit / magnitude)
                xi = -f_t / k_t if k_t > 0.0 else np.zeros(3)
            if commit:
                self._tangential[key] = xi
            seen.add(key)

            # Hertz elastic energy U = (2/5) F_elastic delta, Mindlin spring
            # (1/2) k_t |xi|^2; damping dissipates, so it is not included here
            elastic_energy += 0.4 * elastic_normal * joint.penetration
            elastic_energy += 0.5 * k_t * float(xi @ xi)

            force_on_b = f_n * normal + f_t
            force_on_a = -force_on_b
            force_a += force_on_a
            torque_a += _cross(arm_a, force_on_a)
            torque_b += _cross(arm_b, force_on_b)
            normal_forces[index] = f_n
            tangential_forces.append(f_t)

        self._elastic_energy = elastic_energy
        if commit:
            for stale in set(self._tangential) - seen:
                del self._tangential[stale]

        response = ClumpClumpResponse(
            force_a=force_a, torque_a=torque_a,
            force_b=-force_a, torque_b=torque_b,
            joints=joints, normal_forces=normal_forces,
            tangential_forces=tangential_forces, candidates=len(candidates))
        self._last = response
        return response

    # -- time stepping ---------------------------------------------------
    def step(self, body_a: ClumpBody, body_b: ClumpBody, dt: float,
             extra_a: Sequence["ClumpContact"] = (),
             extra_b: Sequence["ClumpContact"] = ()) -> ClumpClumpResponse:
        """One velocity-Verlet step for the pair (same scheme as ClumpBody.step).

        ``extra_a`` / ``extra_b`` are optional one-body contacts (e.g.
        :class:`tensordem.clump_dynamics.PlaneContact` for a floor) evaluated for
        that body only; they are folded into the same force evaluations as the
        pair contact, so gravity, the floor and the pair stay consistent inside
        one Verlet step.
        """
        if dt <= 0.0:
            raise ClumpClumpError("dt must be positive")
        return self._verlet(body_a, body_b, dt, extra_a, extra_b)

    def _verlet(self, body_a: ClumpBody, body_b: ClumpBody, dt: float,
                extra_a: Sequence["ClumpContact"] = (),
                extra_b: Sequence["ClumpContact"] = ()) -> ClumpClumpResponse:
        kin_a = _Kinematics.of(body_a)
        kin_b = _Kinematics.of(body_b)
        response = self._forces_from(body_a, body_b, kin_a, kin_b, dt=dt,
                                     commit=False)
        force_a, torque_a = self._total(body_a, response.force_a,
                                       response.torque_a, extra_a, dt, False)
        force_b, torque_b = self._total(body_b, response.force_b,
                                       response.torque_b, extra_b, dt, False)
        velocity_a_half = body_a.velocity + 0.5 * dt * force_a / body_a.clump.mass
        velocity_b_half = body_b.velocity + 0.5 * dt * force_b / body_b.clump.mass
        rotation_a, rotation_b = kin_a.rotation, kin_b.rotation
        omega_a_half = body_a.angular_velocity + 0.5 * dt * _angular_acceleration(
            body_a, rotation_a, torque_a)
        omega_b_half = body_b.angular_velocity + 0.5 * dt * _angular_acceleration(
            body_b, rotation_b, torque_b)
        body_a.position = body_a.position + dt * velocity_a_half
        body_b.position = body_b.position + dt * velocity_b_half
        for body, omega in ((body_a, omega_a_half), (body_b, omega_b_half)):
            spin = float(np.linalg.norm(omega))
            if spin > 0.0:
                increment = quaternion_from_axis_angle(omega, spin * dt)
                body.orientation = quaternion_multiply(body.orientation, increment)
                body.orientation /= float(np.linalg.norm(body.orientation))
        body_a.velocity = velocity_a_half
        body_b.velocity = velocity_b_half
        body_a.angular_velocity = omega_a_half
        body_b.angular_velocity = omega_b_half

        kin_a = _Kinematics.of(body_a)
        kin_b = _Kinematics.of(body_b)
        response = self._forces_from(body_a, body_b, kin_a, kin_b, dt=dt,
                                     commit=True)
        force_a, torque_a = self._total(body_a, response.force_a,
                                       response.torque_a, extra_a, dt, True)
        force_b, torque_b = self._total(body_b, response.force_b,
                                       response.torque_b, extra_b, dt, True)
        body_a.velocity = velocity_a_half + 0.5 * dt * force_a / body_a.clump.mass
        body_b.velocity = velocity_b_half + 0.5 * dt * force_b / body_b.clump.mass
        rotation_a, rotation_b = kin_a.rotation, kin_b.rotation
        body_a.angular_velocity = omega_a_half + 0.5 * dt * _angular_acceleration(
            body_a, rotation_a, torque_a)
        body_b.angular_velocity = omega_b_half + 0.5 * dt * _angular_acceleration(
            body_b, rotation_b, torque_b)
        return response

    @staticmethod
    def _total(body: ClumpBody, pair_force: np.ndarray, pair_torque: np.ndarray,
               extras: Sequence["ClumpContact"], dt: float, commit: bool
               ) -> tuple[np.ndarray, np.ndarray]:
        force = pair_force + body.clump.mass * body.gravity
        torque = pair_torque.copy()
        for contact in extras:
            extra_force, extra_torque = contact.forces(body, dt=dt, commit=commit)
            force = force + extra_force
            torque = torque + extra_torque
        return force, torque

    def run(self, body_a: ClumpBody, body_b: ClumpBody, n_steps: int, dt: float,
            record_every: int = 1,
            extra_a: Sequence["ClumpContact"] = (),
            extra_b: Sequence["ClumpContact"] = ()) -> dict[str, np.ndarray]:
        """Integrate the pair and return a monitor history (SI units)."""
        if n_steps <= 0:
            raise ClumpClumpError("n_steps must be positive")
        record_every = max(int(record_every), 1)
        keys = ("time", "position_a", "position_b", "velocity_a", "velocity_b",
                "energy", "momentum", "angular_momentum", "force_magnitude",
                "joints", "candidates")
        history: dict[str, list] = {key: [] for key in keys}
        for step in range(n_steps + 1):
            if step % record_every == 0:
                history["time"].append(step * dt)
                history["position_a"].append(body_a.position.copy())
                history["position_b"].append(body_b.position.copy())
                history["velocity_a"].append(body_a.velocity.copy())
                history["velocity_b"].append(body_b.velocity.copy())
                history["energy"].append(body_a.energy() + body_b.energy())
                history["momentum"].append(total_linear_momentum(body_a, body_b))
                history["angular_momentum"].append(
                    total_angular_momentum(body_a, body_b))
                if self._last is not None:
                    history["force_magnitude"].append(self._last.force_magnitude)
                    history["joints"].append(len(self._last.joints))
                    history["candidates"].append(self._last.candidates)
                else:
                    history["force_magnitude"].append(0.0)
                    history["joints"].append(0)
                    history["candidates"].append(0)
            if step == n_steps:
                break
            self.step(body_a, body_b, dt, extra_a, extra_b)
        return {key: np.asarray(value) for key, value in history.items()}


# --------------------------------------------------------------------------
# Pair integrator (module-level convenience)
# --------------------------------------------------------------------------
def pair_step(contact: ClumpClumpContact, body_a: ClumpBody, body_b: ClumpBody,
              dt: float, extra_a: Sequence["ClumpContact"] = (),
              extra_b: Sequence["ClumpContact"] = ()) -> ClumpClumpResponse:
    """Advance both bodies by one velocity-Verlet step under a pair contact."""
    return contact.step(body_a, body_b, dt, extra_a, extra_b)


def total_linear_momentum(body_a: ClumpBody, body_b: ClumpBody) -> np.ndarray:
    """Total linear momentum of the pair (zero-gravity convention preserved)."""
    return (body_a.clump.mass * body_a.velocity
            + body_b.clump.mass * body_b.velocity)


def total_angular_momentum(body_a: ClumpBody, body_b: ClumpBody,
                          origin: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Total angular momentum about ``origin``: ``sum m x^v + I omega``."""
    base = np.asarray(origin, dtype=np.float64)
    total = np.zeros(3)
    for body in (body_a, body_b):
        arm = body.position - base
        total += body.clump.mass * _cross(arm, body.velocity)
        total += body.angular_momentum_world()
    return total


# --------------------------------------------------------------------------
# Duplicate-counting audit
# --------------------------------------------------------------------------
def duplicate_force_ratio(body_a: ClumpBody, body_b: ClumpBody,
                          config: ClumpClumpConfig | None = None,
                          *, dt: float = 0.0) -> dict[str, float]:
    """Ratio of the naive member-sum force to the merged junction force.

    ``1.0`` means merging changed nothing (no duplicate sampling); ``2.0`` means
    the naive member sum would have doubled a single physical contact.  Both
    evaluations start from rest with fresh history, so the ratio isolates the
    geometric effect.
    """
    cfg = config or ClumpClumpConfig()
    merged_cfg = ClumpClumpConfig(**{**cfg.__dict__, "merging": True})
    naive_cfg = ClumpClumpConfig(**{**cfg.__dict__, "merging": False})
    merged = ClumpClumpContact(merged_cfg).forces(body_a, body_b, dt=dt,
                                                  commit=False)
    naive = ClumpClumpContact(naive_cfg).forces(body_a, body_b, dt=dt,
                                                commit=False)
    merged_magnitude = float(np.linalg.norm(merged.force_a))
    naive_magnitude = float(np.linalg.norm(naive.force_a))
    return {
        "merged_force": merged_magnitude,
        "unmerged_force": naive_magnitude,
        "ratio": (naive_magnitude / merged_magnitude
                  if merged_magnitude > 0.0 else math.nan),
        "joints": float(len(merged.joints)),
        "candidates": float(len(naive.joints)),
    }