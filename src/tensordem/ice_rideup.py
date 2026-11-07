"""Three-dimensional floe-edge contact with an inclined bow, bow ride-up
dynamics, and the ``R_break`` / ``R_clear`` / ``R_up`` resistance split.

What this module adds
---------------------
The pre-existing channel model (:mod:`tensordem.ice_channel`) treats a floe as a
*point with a radius* in the horizontal plane and gives the hull a rectangular
footprint; :mod:`tensordem.hull_mesh_contact` resolves sphere-triangle contacts.
Neither can express what a floe does when the hull **wedges underneath it**: the
floe is a body with a finite thickness and an attitude (quaternion), the contact
is between the floe's **rim circle** and a hull facet, the normal force carries a
**vertical component**, and the floe slides *up* the bow slope while friction
opposes that climb.  This module models that state and derives the resistance
bookkeeping from it.

Geometry (SI units, ship frame)
-------------------------------
* Ship frame: the ship advances along ``+x``; the ice is advected towards ``-x``
  at the ship speed ``V``, so the hull is at rest and the far-field ice velocity
  is ``(-V, 0, 0)``.  The waterline is ``z = 0``.
* Bow slope: a rectangular facet inclined at ``alpha`` to the horizontal whose
  outward normal points **up and forward**, ``n = (sin a, 0, cos a)``: the hull
  pushes the floe up (``n_z = cos a > 0``) and the reaction on the hull is a
  resistance (reaction ``x`` component is negative).  The facet starts at the
  lower (forward) edge ``(x_lower, z_lower)`` and rises *aft*,
  ``p(s) = p0 + s*u`` with ``u = (-cos a, 0, sin a)``.  This is the idealised
  wedge bow: the forefoot is the deepest point, the shell rises to the top edge
  ``z_upper`` above the waterline, and that top edge is where floes come out of
  the water.
* Stacking shelf: a horizontal facet at ``z_upper`` aft of the slope's top edge;
  at most one floe per ``(x, y)`` bin may rest on it (first-order stacking, the
  state requested by the campaign brief).

Contact kernel (closed form)
----------------------------
For each of the two rim circles of the cylinder (radius ``r``, half thickness
``t/2``) the closest point on the *unbounded* plane is analytic: with in-plane
basis ``u1, u2`` of the rim the signed distance of the rim to the plane is
``n.(c_rim - p0) - r*sqrt((n.u1)^2 + (n.u2)^2)`` -- a linear functional over a
circle attains its minimum on the rim in closed form, and for a convex cylinder
the global minimum over the whole lateral surface is always attained on one of
the two rims.  If that minimising rim point falls outside the rectangular facet
the closest feature pair becomes *rim circle vs facet edge* (or corner); that
distance is again closed form (point-to-circle distance of the clamped facet
point), with the facet boundary carrying a small radius
:data:`HULL_EDGE_RADIUS` so that an edge contact has a well defined penetration.
Force is Kelvin-Voigt normal (spring + dashpot) plus a Coulomb-cone tangential
law applied at the contact point, so the floe receives both force and torque
``tau = (x_contact - x_com) x F``.

Evidence level
--------------
* **REAL** -- contact geometry (closed form; verified against brute-force
  sampling and against analytic sphere-plane / cylinder-incline solutions), the
  penalty + Coulomb force law, rigid-body kinematics with quaternions and the
  Archimedes buoyancy of a flat disk (static vertical residual is machine zero).
* **REAL (reduced model)** -- the ride-up dynamics: floes are non-interacting
  (dilute channel, independent lanes), water is a linear damper plus a
  metacentric righting moment, buoyancy uses the flat-disk (near-level) form by
  default and
  the continuity of the intact sheet is one brittle elastic anchor per bonded
  floe instead of a bond network.  Those assumptions are listed in
  ``docs/ICE_RIDEUP_3D.md``.  No channel of the resistance split is prescribed:
  all three are measured from simulated contact forces and the split is a
  partition of the measured hull force whose residual is *checked*.
* **Righting moment, two models** -- ``righting_model="near_level"`` (default,
  the historical kernel) uses ``M = rho_w g V GM sin(phi)`` with the level-disk
  ``GM = R^2/(4d)`` and the keel-to-gravity term dropped; ``righting_model=
  "exact"`` uses the large-angle heave-relaxed hydrostatics of
  :func:`tensordem.floe_6dof.submerged_hydrostatics` for both the vertical force
  and the moment.  For the ride-up floe geometry (``R/t = 8``) the near-level
  law is already outside its small-angle envelope above ``arctan(t/2R)`` and
  overstates the restoring moment by up to a factor of seven in the heel band
  the bow imposes; the measured envelope is in
  ``docs/ice_rideup_righting_moment_envelope.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .floe_6dof import submerged_hydrostatics
from .ice_channel import WaterEnvironment

RIGHTING_MODEL_NEAR_LEVEL = "near_level"
RIGHTING_MODEL_EXACT = "exact"
RIGHTING_MODELS = (RIGHTING_MODEL_NEAR_LEVEL, RIGHTING_MODEL_EXACT)

__all__ = [
    "GRAVITY", "RHO_ICE", "RHO_WATER", "HULL_EDGE_RADIUS",
    "RIGHTING_MODEL_EXACT", "RIGHTING_MODEL_NEAR_LEVEL", "RIGHTING_MODELS",
    "BowSlope", "RectPatch", "ContactParams", "RideUpResult", "RideUpSimulation",
    "quat_normalize", "quat_multiply", "quat_to_rotmat", "quat_rotate",
    "quat_integrate", "perp_basis",
    "sphere_plane_contact", "sphere_plane_force",
    "rim_patch_contact", "disk_patch_contact",
    "floe_mass", "archimedes_buoyancy", "hydrostatic_lift",
    "ride_up_resistance", "run_case", "sweep", "monotonicity_slope",
]

GRAVITY = 9.80665
RHO_ICE = 917.0
RHO_WATER = 1025.0
#: Radius given to the facet boundary of the hull, m.  An ideal zero-thickness
#: facet has no penetration measure for an edge contact, so the boundary is
#: modelled as a thin cylinder of this radius (a rounding of the chine/stem).
HULL_EDGE_RADIUS = 0.05


# ---------------------------------------------------------------------------
# quaternion helpers
# ---------------------------------------------------------------------------

def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-300)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a * b`` for ``(..., 4)`` tensors ordered ``(w, x, y, z)``."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    r0 = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), dim=-1)
    r1 = torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), dim=-1)
    r2 = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=-1)
    return torch.stack((r0, r1, r2), dim=-2)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    R = quat_to_rotmat(q)
    return (R @ v.unsqueeze(-1)).squeeze(-1)


def quat_integrate(q: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
    """Advance a quaternion by body-frame ``omega`` over ``dt`` (exponential map)."""
    theta = omega.norm(dim=-1)
    half = 0.5 * theta * dt
    tiny = theta < 1e-12
    scale = torch.where(tiny, torch.full_like(theta, 0.5 * dt),
                        torch.sin(half) / theta.clamp_min(1e-300))
    dq = torch.cat((torch.cos(half).unsqueeze(-1), omega * scale.unsqueeze(-1)), dim=-1)
    return quat_normalize(quat_multiply(dq, q))


def perp_basis(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two unit vectors spanning the plane perpendicular to ``a`` (``(..., 3)``)."""
    ez = torch.zeros_like(a)
    ez[..., 2] = 1.0
    ex = torch.zeros_like(a)
    ex[..., 0] = 1.0
    seed = torch.where((a[..., 2].abs() > 0.9).unsqueeze(-1), ex, ez)
    u1 = torch.cross(a, seed, dim=-1)
    u1 = u1 / u1.norm(dim=-1, keepdim=True).clamp_min(1e-300)
    u2 = torch.cross(a, u1, dim=-1)
    return u1, u2


def _t3(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float64)
    return torch.tensor([float(v) for v in value], dtype=torch.float64)


def _broadcast(value, n: int, dtype=torch.float64, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device).expand(n)
    return torch.as_tensor(float(value), dtype=dtype, device=device).expand(n)


# ---------------------------------------------------------------------------
# hull facets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RectPatch:
    """Rectangular planar facet; ``normal`` is the outward normal (into the ice)."""

    origin: Tuple[float, float, float]
    e1: Tuple[float, float, float]
    length1: float
    e2: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    length2: float = 12.0
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    name: str = "patch"

    @property
    def origin_t(self) -> torch.Tensor:
        return _t3(self.origin)

    @property
    def e1_t(self) -> torch.Tensor:
        return _t3(self.e1)

    @property
    def e2_t(self) -> torch.Tensor:
        return _t3(self.e2)

    @property
    def normal_t(self) -> torch.Tensor:
        return _t3(self.normal)

    def point_at(self, s1: float, s2: float = 0.0) -> torch.Tensor:
        return self.origin_t + self.e1_t * s1 + self.e2_t * s2


@dataclass(frozen=True)
class BowSlope:
    """Idealised wedge bow: lower edge, V-shaped slope (two facets), shelf.

    ``angle_deg`` is the slope angle ``alpha`` to the horizontal (fore-aft
    inclination) and ``plan_half_angle_deg`` the deadrise ``beta`` of the two
    shell facets about the centreline -- the plan-view V that lets a floe escape
    outboard instead of being lifted.  ``beta = 0`` degenerates to the
    two-dimensional wedge used by the analytic verification tests.
    """

    angle_deg: float = 30.0
    x_lower: float = 0.0
    z_lower: float = -0.8
    z_upper: float = 1.0
    y_half: float = 6.0
    shelf_length: float = 4.0
    plan_half_angle_deg: float = 30.0
    side_length: float = 20.0
    side_draft: float = 2.0

    @property
    def alpha(self) -> float:
        return math.radians(self.angle_deg)

    @property
    def beta(self) -> float:
        return math.radians(self.plan_half_angle_deg)

    def _normal(self, side: float) -> torch.Tensor:
        a, b = self.alpha, self.beta
        n = torch.tensor([math.sin(a) * math.cos(b), side * math.sin(b), math.cos(a)],
                         dtype=torch.float64)
        return n / n.norm()

    @property
    def normal(self) -> torch.Tensor:
        """Normal of the centreline plane (``beta = 0`` case)."""
        return torch.tensor([math.sin(self.alpha), 0.0, math.cos(self.alpha)], dtype=torch.float64)

    @property
    def upslope(self) -> torch.Tensor:
        n = self.normal
        u = torch.tensor([-float(n[2]), 0.0, float(n[0])], dtype=torch.float64)
        return u / u.norm()

    def _facet(self, side: float) -> RectPatch:
        n = self._normal(side)
        e1 = torch.tensor([-float(n[2]), 0.0, float(n[0])], dtype=torch.float64)
        e1 = e1 / e1.norm()
        e2 = torch.cross(n, e1, dim=-1)
        rise = self.z_upper - self.z_lower
        length1 = rise / float(e1[2])
        return RectPatch(
            origin=(self.x_lower, 0.0, self.z_lower),
            e1=tuple(float(v) for v in e1),
            length1=length1,
            e2=tuple(float(v) for v in e2),
            length2=2.0 * self.y_half,
            normal=tuple(float(v) for v in n),
            name="slope_port" if side > 0 else "slope_starboard",
        )

    @property
    def slope_patch_port(self) -> RectPatch:
        return self._facet(+1.0)

    @property
    def slope_patch_starboard(self) -> RectPatch:
        return self._facet(-1.0)

    @property
    def slope_patch(self) -> RectPatch:
        """Centreline (``beta = 0``) facet -- the analytic-verification geometry."""
        return RectPatch(
            origin=(self.x_lower, 0.0, self.z_lower),
            e1=tuple(float(v) for v in self.upslope),
            length1=(self.z_upper - self.z_lower) / math.sin(self.alpha),
            e2=(0.0, 1.0, 0.0),
            length2=2.0 * self.y_half,
            normal=tuple(float(v) for v in self.normal),
            name="slope",
        )

    @property
    def shelf_patch(self) -> RectPatch:
        return RectPatch(
            origin=(self.x_upper, 0.0, self.z_upper),
            e1=(-1.0, 0.0, 0.0),
            length1=self.shelf_length,
            e2=(0.0, 1.0, 0.0),
            length2=2.0 * self.y_half,
            normal=(0.0, 0.0, 1.0),
            name="shelf",
        )

    def _side(self, side: float) -> RectPatch:
        return RectPatch(
            origin=(self.x_upper, side * self.y_half, -0.5 * self.side_draft),
            e1=(-1.0, 0.0, 0.0),
            length1=self.side_length,
            e2=(0.0, 0.0, 1.0),
            length2=self.side_draft,
            normal=(0.0, side, 0.0),
            name="side_port" if side > 0 else "side_starboard",
        )

    @property
    def side_patch_port(self) -> RectPatch:
        return self._side(+1.0)

    @property
    def side_patch_starboard(self) -> RectPatch:
        return self._side(-1.0)

    @property
    def patches(self) -> Tuple[RectPatch, ...]:
        """(port slope, starboard slope, port side, starboard side, shelf)."""
        return (self.slope_patch_port, self.slope_patch_starboard,
                self.side_patch_port, self.side_patch_starboard, self.shelf_patch)

    @property
    def n_slope_facets(self) -> int:
        return 2

    @property
    def slope_length(self) -> float:
        return (self.z_upper - self.z_lower) / math.sin(self.alpha)

    @property
    def x_upper(self) -> float:
        return self.x_lower - (self.z_upper - self.z_lower) / math.tan(self.alpha)


# ---------------------------------------------------------------------------
# analytic contact primitives
# ---------------------------------------------------------------------------

def sphere_plane_contact(center, radius: float, plane: RectPatch) -> Dict[str, torch.Tensor]:
    """Sphere vs the unbounded plane of ``plane`` (closed form)."""
    c = _t3(center)
    n = plane.normal_t
    p0 = plane.origin_t
    signed = float(((c - p0) * n).sum())
    return {
        "signed_distance": torch.tensor(signed, dtype=torch.float64),
        "penetration": torch.tensor(max(0.0, radius - signed), dtype=torch.float64),
        "normal": n.clone(),
        "point_hull": c - n * signed,
        "point_sphere": c - n * radius,
    }


def _kelvin_voigt(pen: float, v_rel: torch.Tensor, normal: torch.Tensor, *,
                  stiffness: float, damping: float, friction: float,
                  tangential_damping: float) -> Dict[str, torch.Tensor]:
    v_n = float((v_rel * normal).sum())
    f_n = max(0.0, stiffness * pen - damping * v_n)
    v_t = v_rel - normal * v_n
    slip = float(v_t.norm())
    if slip > 1e-12:
        f_t_mag = min(tangential_damping * slip, friction * f_n)
        f_t = -v_t / slip * f_t_mag
    else:
        f_t = torch.zeros(3, dtype=torch.float64)
    return {
        "normal_force": torch.tensor(f_n, dtype=torch.float64),
        "tangential_force": f_t,
        "force": normal * f_n + f_t,
        "slip_speed": torch.tensor(slip, dtype=torch.float64),
        "v_normal": torch.tensor(v_n, dtype=torch.float64),
    }


def sphere_plane_force(center, velocity, radius: float, plane: RectPatch, *,
                       stiffness: float, damping: float = 0.0,
                       friction: float = 0.0, tangential_damping: float = 0.0
                       ) -> Dict[str, torch.Tensor]:
    """Penalty + Coulomb contact of one sphere with a plane."""
    geom = sphere_plane_contact(center, radius, plane)
    out = _kelvin_voigt(float(geom["penetration"]), _t3(velocity), geom["normal"],
                        stiffness=stiffness, damping=damping, friction=friction,
                        tangential_damping=tangential_damping)
    out.update({k: geom[k] for k in ("penetration", "normal", "point_hull")})
    return out


def _patch_arrays(patches: Sequence[RectPatch], dtype=torch.float64, device=None):
    """Stack a facet set into tensors: origins (P,3), e1, e2, normals, lengths."""
    P = len(patches)
    origin = torch.stack([pt.origin_t.to(dtype=dtype, device=device) for pt in patches])
    e1 = torch.stack([pt.e1_t.to(dtype=dtype, device=device) for pt in patches])
    e2 = torch.stack([pt.e2_t.to(dtype=dtype, device=device) for pt in patches])
    nrm = torch.stack([pt.normal_t.to(dtype=dtype, device=device) for pt in patches])
    len1 = torch.tensor([pt.length1 for pt in patches], dtype=dtype, device=device)
    len2 = torch.tensor([pt.length2 for pt in patches], dtype=dtype, device=device)
    return origin, e1, e2, nrm, len1, len2


def patch_set_contact(centers: torch.Tensor, axes: torch.Tensor, radius, thickness,
                      patches: Sequence[RectPatch], *, n_samples: int = 48,
                      edge_radius: float = HULL_EDGE_RADIUS) -> Dict[str, torch.Tensor]:
    """Closest-feature contact of each floe (cylinder) with a set of facets.

    The rim circle is sampled at ``n_samples`` equally spaced points; for each
    sample the depth behind the facet plane is computed in closed form and, when
    the sample falls outside the rectangle, the distance to the clamped boundary
    point gives the *edge* candidate.  Taking the deepest candidate over samples,
    rims and facets yields a penetration measure that is continuous in the floe
    pose (the analytic "depth of the deepest rim point" is recovered to
    ``O(r (pi/n_samples)^2)`` -- 6 mm for ``r = 4 m`` and 48 samples -- and the
    facet boundary is bridged at :data:`HULL_EDGE_RADIUS`).
    """
    N = centers.shape[0]
    dev, dtype = centers.device, centers.dtype
    r = _broadcast(radius, N, dtype, dev)
    t = _broadcast(thickness, N, dtype, dev)
    origin, e1, e2, nrm, len1, len2 = _patch_arrays(patches, dtype, dev)
    P = origin.shape[0]

    half = 0.5 * t
    rim_c = torch.stack((centers + axes * half.unsqueeze(-1),
                         centers - axes * half.unsqueeze(-1)), dim=1)      # (N,2,3)
    a = axes.unsqueeze(1).expand(N, 2, 3)
    u1, u2 = perp_basis(a)                                                # (N,2,3)

    theta = torch.arange(n_samples, dtype=dtype, device=dev) * (2.0 * math.pi / n_samples)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    ring = (torch.cos(theta).view(1, 1, -1, 1) * u1.unsqueeze(2)
            + torch.sin(theta).view(1, 1, -1, 1) * u2.unsqueeze(2))       # (N,2,K,3)
    p = rim_c.unsqueeze(2) + ring * r.view(N, 1, 1, 1)                    # (N,2,K,3)

    rel = p.unsqueeze(3) - origin.view(1, 1, 1, P, 3)                     # (N,2,K,P,3)
    depth = -(rel * nrm.view(1, 1, 1, P, 3)).sum(-1)                      # (N,2,K,P)
    s1 = (rel * e1.view(1, 1, 1, P, 3)).sum(-1)
    s2 = (rel * e2.view(1, 1, 1, P, 3)).sum(-1)
    L1 = len1.view(1, 1, 1, P)
    L2 = len2.view(1, 1, 1, P)
    zero = torch.zeros_like(L1)
    s1c = torch.maximum(torch.minimum(s1, L1), zero)
    s2c = torch.maximum(torch.minimum(s2, 0.5 * L2), -0.5 * L2)
    inside = ((s1c - s1).abs() + (s2c - s2).abs()) < 1e-9

    Q = (origin.view(1, 1, 1, P, 3)
         + e1.view(1, 1, 1, P, 3) * s1c.unsqueeze(-1)
         + e2.view(1, 1, 1, P, 3) * s2c.unsqueeze(-1))                    # (N,2,K,P,3)
    edge_vec = p.unsqueeze(3) - Q
    edge_dist = edge_vec.norm(dim=-1)

    pen_face = torch.where(inside, depth, torch.full_like(depth, -1e30))
    pen_edge = edge_radius - edge_dist
    take_face = pen_face >= pen_edge
    pen = torch.where(take_face, pen_face, pen_edge)                      # (N,2,K,P)
    normal = torch.where(take_face.unsqueeze(-1), nrm.view(1, 1, 1, P, 3),
                         edge_vec / edge_dist.clamp_min(1e-300).unsqueeze(-1))

    pen_flat = pen.reshape(N, 2 * n_samples * P)
    best = pen_flat.argmax(dim=1)
    per_rim = n_samples * P
    rim_idx = torch.div(best, per_rim, rounding_mode="floor")
    rem = best - rim_idx * per_rim
    patch_idx = torch.div(rem, n_samples, rounding_mode="floor")
    sample_idx = rem - patch_idx * n_samples

    gather_p = normal.reshape(N, 2 * n_samples * P, 3)
    gather_pt = p.unsqueeze(3).expand(N, 2, n_samples, P, 3).reshape(N, 2 * n_samples * P, 3)
    bi = best.view(N, 1, 1).expand(N, 1, 3)
    best_pen = pen_flat.gather(1, best.view(N, 1)).squeeze(1)
    best_normal = gather_p.gather(1, bi).squeeze(1)
    best_point = gather_pt.gather(1, bi).squeeze(1)

    return {
        "penetration": best_pen,
        "active": best_pen > 0.0,
        "normal": best_normal,
        "point_floe": best_point,
        "point_hull": best_point - best_normal * best_pen.unsqueeze(-1),
        "patch_index": patch_idx,
        "sample_index": sample_idx,
        "rim_index": rim_idx,
    }


def rim_patch_contact(centers: torch.Tensor, axes: torch.Tensor, radius,
                      thickness, patch: RectPatch, *, n_samples: int = 256
                      ) -> Dict[str, torch.Tensor]:
    """Single-facet variant (dense sampling) -- used by the analytic tests."""
    return patch_set_contact(centers, axes, radius, thickness, (patch,),
                             n_samples=n_samples)


def disk_patch_contact(centers: torch.Tensor, quats: torch.Tensor, radius,
                       thickness, patches: Sequence[RectPatch], *, n_samples: int = 48
                       ) -> Dict[str, torch.Tensor]:
    """Best (deepest) contact of each disk with a set of facets."""
    axes = quat_rotate(quats, torch.tensor([0.0, 0.0, 1.0], dtype=centers.dtype,
                                           device=centers.device).expand_as(centers))
    out = patch_set_contact(centers, axes, radius, thickness, patches, n_samples=n_samples)
    out["axis"] = axes
    return out


# ---------------------------------------------------------------------------
# hydrostatics
# ---------------------------------------------------------------------------

def floe_mass(radius: torch.Tensor, thickness: torch.Tensor, rho_ice: float = RHO_ICE) -> torch.Tensor:
    """Mass of a circular floe, kg."""
    return rho_ice * math.pi * radius * radius * thickness


def archimedes_buoyancy(center_z: torch.Tensor, radius: torch.Tensor, thickness: torch.Tensor, *,
                        rho_water: float = RHO_WATER, rho_ice: float = RHO_ICE,
                        gravity: float = GRAVITY, waterline: float = 0.0) -> Dict[str, torch.Tensor]:
    """Buoyancy of a flat-lying disk and the static vertical residual ``B - W``.

    ``B = rho_w g A_wp d_sub`` with ``d_sub = clamp(waterline - (z - t/2), 0, t)``
    (Archimedes for a flat disk, the kernel used by :mod:`tensordem.ice_channel`).
    The residual is evaluated as a single expression so that at the analytic
    equilibrium draught ``d = (rho_i/rho_w) t`` it cancels to machine precision.
    """
    area = math.pi * radius * radius
    sub = (waterline - (center_z - 0.5 * thickness)).clamp_min(0.0)
    sub = torch.minimum(sub, thickness)
    buoy = area * rho_water * gravity * sub
    weight = area * rho_ice * gravity * thickness
    return {
        "buoyancy": buoy,
        "weight": weight,
        "residual": buoy - weight,
        "submerged_thickness": sub,
        "waterplane_area": area,
    }


def hydrostatic_lift(center_z: torch.Tensor, radius: torch.Tensor, thickness: torch.Tensor,
                     contact_normal_force: torch.Tensor, *, normal_z: float,
                     rho_water: float = RHO_WATER, rho_ice: float = RHO_ICE,
                     gravity: float = GRAVITY) -> torch.Tensor:
    """Vertical force balance of a floe: ``N * n_z + B - W`` (N, positive = up)."""
    b = archimedes_buoyancy(center_z, radius, thickness, rho_water=rho_water,
                            rho_ice=rho_ice, gravity=gravity)
    return contact_normal_force * normal_z + b["residual"]


def ride_up_resistance(lift_force_z, angle_deg: float, *, convention: str = "momentum"):
    """Ride-up resistance from a vertical net force (two documented conventions).

    ``convention='momentum'``: the exact geometry/energy relation for a floe
    pushed up a slope of angle ``a`` -- the horizontal reaction belonging to the
    lift is ``R = F_z * tan(a)`` (work by the lift over one run of the slope).

    ``convention='channel'``: the convention written in the campaign brief,
    ``R = F_z * sin(a)``.  They differ by ``cos(a)`` (0.71-0.94 over
    ``a in [20, 45] deg``); the sweep reports both so the choice cannot hide a
    factor.
    """
    if convention not in ("momentum", "channel"):
        raise ValueError("convention must be 'momentum' or 'channel'")
    a = math.radians(float(angle_deg))
    factor = math.tan(a) if convention == "momentum" else math.sin(a)
    return lift_force_z * factor

# ---------------------------------------------------------------------------
# contact / fracture parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContactParams:
    """Contact, fracture and damping parameters (all SI)."""

    crushing_strength: float = 1.5e6        # Pa, Keinonen / ISO-19906 order
    reference_penetration_fraction: float = 0.02
    normal_damping_ratio: float = 0.3
    tangential_damping_ratio: float = 0.3
    friction: float = 0.15                  # ice-steel, cf. tensordem.ice_steel_friction
    shear_strength: float = 0.5e6           # Pa, floe-boundary shear strength
    ice_modulus: float = 9.0e9              # Pa, sheet elastic modulus
    bonded_fraction: float = 0.20           # share of the field still in the sheet
    crush_displacement: float = 0.05         # m of ice crushing before a floe tears free
    water_damping: float = 0.6              # 1/s, linear translational damper
    angular_damping: float = 3.0            # 1/s
    edge_radius: float = HULL_EDGE_RADIUS

    def normal_stiffness(self, radius: float, thickness: float) -> float:
        """``k_n = 2 sigma_c t sqrt(2 R / delta_ref)`` (anchored like ice_channel)."""
        delta_ref = self.reference_penetration_fraction * 2.0 * radius
        return 2.0 * self.crushing_strength * thickness * math.sqrt(2.0 * radius / delta_ref)


@dataclass
class FloeField:
    """State of the floe field (float64 torch tensors)."""

    centers: torch.Tensor
    velocities: torch.Tensor
    quats: torch.Tensor
    omegas: torch.Tensor
    radius: torch.Tensor
    thickness: torch.Tensor
    bonded: torch.Tensor
    anchor: torch.Tensor

    @property
    def n(self) -> int:
        return self.centers.shape[0]

    def mass(self, rho_ice: float = RHO_ICE) -> torch.Tensor:
        return floe_mass(self.radius, self.thickness, rho_ice)

    def inertia(self, rho_ice: float = RHO_ICE) -> torch.Tensor:
        m = self.mass(rho_ice)
        return m * (0.25 * self.radius ** 2 + self.thickness ** 2 / 12.0)


@dataclass
class RideUpResult:
    """Time-averaged outcome of one run."""

    fr: float
    thickness: float
    angle_deg: float
    seed: int
    speed: float
    r_break: float
    r_clear: float
    r_up: float
    r_total: float
    r_up_channel: float
    lift_force_z: float
    n_climbing: float
    n_contact: float
    n_stacked: float
    fracture_energy_rate: float
    r_break_energy: float
    steps: int
    dt: float

    @property
    def closure_residual(self) -> float:
        if self.r_total <= 0:
            return float("nan")
        return abs(self.r_break + self.r_clear + self.r_up - self.r_total) / self.r_total

    @property
    def ratio_up(self) -> float:
        return self.r_up / self.r_total if self.r_total > 0 else float("nan")

    @property
    def ratio_up_channel(self) -> float:
        return self.r_up_channel / self.r_total if self.r_total > 0 else float("nan")

    def as_dict(self) -> Dict[str, float]:
        return {
            "Fr": self.fr, "thickness_m": self.thickness, "angle_deg": self.angle_deg,
            "seed": self.seed, "speed_m_per_s": self.speed,
            "R_break_N": self.r_break, "R_clear_N": self.r_clear, "R_up_N": self.r_up,
            "R_total_N": self.r_total, "closure_residual": self.closure_residual,
            "R_up_channel_N": self.r_up_channel, "lift_force_z_N": self.lift_force_z,
            "n_climbing": self.n_climbing, "n_contact": self.n_contact,
            "n_stacked": self.n_stacked,
            "fracture_energy_rate_W": self.fracture_energy_rate,
            "R_break_energy_N": self.r_break_energy,
            "R_up_over_R_total": self.ratio_up,
            "R_up_channel_over_R_total": self.ratio_up_channel,
            "steps": self.steps, "dt_s": self.dt,
        }


class RideUpSimulation:
    """Floe field advected into an idealised wedge bow, in the ship frame."""

    def __init__(self, *, fr: float = 0.3, thickness: float = 0.5,
                 slope: Optional[BowSlope] = None, seed: int = 0,
                 n_floes: int = 40, ship_length: float = 10.0,
                 radius_mean: float = 4.0, radius_spread: float = 0.30,
                 contact: ContactParams = ContactParams(),
                 dt: float = 0.01, steps: int = 1800, warmup_steps: int = 400,
                 n_samples: int = 24, patches=None, n_slope_facets: Optional[int] = None,
                 rho_ice: float = RHO_ICE, rho_water: float = RHO_WATER,
                 gravity: float = GRAVITY,
                 righting_model: str = RIGHTING_MODEL_NEAR_LEVEL) -> None:
        self.fr = float(fr)
        self.thickness = float(thickness)
        self.slope = slope or BowSlope()
        self.seed = int(seed)
        self.dt = float(dt)
        self.steps = int(steps)
        self.warmup_steps = int(warmup_steps)
        self.n_samples = int(n_samples)
        self.contact = contact
        self.rho_ice = float(rho_ice)
        self.rho_water = float(rho_water)
        self.gravity = float(gravity)
        if righting_model not in RIGHTING_MODELS:
            raise ValueError(
                f"righting_model must be one of {RIGHTING_MODELS}, got {righting_model!r}")
        self.righting_model = str(righting_model)
        # hydrostatic environment for the exact (large-angle) branch
        self._water = WaterEnvironment(water_density=self.rho_water,
                                       ice_density=self.rho_ice,
                                       gravity=self.gravity)
        # Froude reference length: Fr = V / sqrt(g L_ref), L_ref = 10 m, so that
        # Fr in [0.1, 0.5] maps to V in [1.0, 5.0] m/s, the icebreaking range.
        self.froude_length = float(ship_length)
        self.speed = self.fr * math.sqrt(self.gravity * self.froude_length)
        self.dtype = torch.float64
        self.gen = torch.Generator().manual_seed(self.seed)
        self._radius_mean = float(radius_mean)

        sl = self.slope
        self.patches = tuple(patches) if patches is not None else sl.patches
        self.n_slope_facets = (sl.n_slope_facets if n_slope_facets is None
                               else int(n_slope_facets))
        self.x_inflow = sl.x_lower + 2.0 + 1.5 * radius_mean
        self.x_exit = sl.x_upper - sl.shelf_length - 1.0
        self._span = self.x_inflow - self.x_exit

        n = int(n_floes)
        r = radius_mean * (1.0 + radius_spread * (2.0 * torch.rand(n, generator=self.gen,
                                                                  dtype=self.dtype) - 1.0))
        center = torch.zeros(n, 3, dtype=self.dtype)
        # spread the field over the whole reach so the stream starts in steady
        # state; floes that would be seeded inside the bow footprint are pushed
        # upstream (a floe cannot be initialised overlapping the hull)
        center[:, 0] = self.x_exit + self._span * torch.rand(n, generator=self.gen, dtype=self.dtype)
        in_bow = (center[:, 0] > sl.x_upper - sl.shelf_length) & (center[:, 0] < sl.x_lower + 2.0)
        k_bow = int(in_bow.sum())
        if k_bow:
            center[in_bow, 0] = (sl.x_lower + 2.0) + 2.0 * torch.rand(
                k_bow, generator=self.gen, dtype=self.dtype)
        center[:, 1] = (2.0 * torch.rand(n, generator=self.gen, dtype=self.dtype) - 1.0) * sl.y_half
        center[:, 2] = self.equilibrium_height(thickness)
        v = torch.zeros(n, 3, dtype=self.dtype)
        v[:, 0] = -self.speed
        yaw = 2.0 * math.pi * torch.rand(n, generator=self.gen, dtype=self.dtype)
        tilt = 0.02 * (2.0 * torch.rand(n, generator=self.gen, dtype=self.dtype) - 1.0)
        q = torch.stack((torch.cos(tilt / 2), torch.sin(tilt / 2),
                         torch.zeros(n, dtype=self.dtype), torch.zeros(n, dtype=self.dtype)), dim=1)
        qy = torch.stack((torch.cos(yaw / 2), torch.zeros(n, dtype=self.dtype),
                          torch.zeros(n, dtype=self.dtype), torch.sin(yaw / 2)), dim=1)
        q = quat_normalize(quat_multiply(qy, q))
        omega = torch.zeros(n, 3, dtype=self.dtype)
        bonded = torch.rand(n, generator=self.gen, dtype=self.dtype) < contact.bonded_fraction
        self.field = FloeField(center, v, q, omega, r,
                               torch.full((n,), float(thickness), dtype=self.dtype),
                               bonded, center.clone())

        # per-floe contact stiffness and damping
        kn = [self.contact.normal_stiffness(float(rr), self.thickness) for rr in r]
        self.k_n = torch.tensor(kn, dtype=self.dtype)
        m = self.field.mass(self.rho_ice)
        self.c_n = 2.0 * self.contact.normal_damping_ratio * torch.sqrt(self.k_n * m)
        self.c_t = 2.0 * self.contact.tangential_damping_ratio * torch.sqrt(self.k_n * m)
        self.mass = m
        self.inertia = self.field.inertia(self.rho_ice)

        # sheet anchor: brittle elastic restraint to the pristine position
        # floe-boundary strength: shear over the 2r x t boundary plane, released
        # after a documented crushing displacement (the brittle-elastic anchor
        # therefore stores 0.5 F_a delta_c before it snaps)
        area_bond = 2.0 * r * self.thickness
        self.anchor_strength = self.contact.shear_strength * area_bond
        self.anchor_stiffness = self.anchor_strength / self.contact.crush_displacement

        self._snapped = torch.zeros(self.field.n, dtype=torch.bool)
        self.series: Dict[str, List[float]] = {}
        self._reset_accumulators()
        self._bin_count = 0

    # -- helpers -----------------------------------------------------------
    def equilibrium_height(self, thickness: Optional[float] = None) -> float:
        """Centre height at which a flat floe is in buoyancy/weight balance, m.

        ``B = W`` requires the submerged thickness ``d = (rho_i/rho_w) t``, i.e.
        the centre sits ``t/2`` above the keel plane, which is ``d`` below the
        waterline, so ``z = t/2 - (rho_i/rho_w) t`` (negative: 89 % submerged).
        """
        t = self.thickness if thickness is None else float(thickness)
        return 0.5 * t - (self.rho_ice / self.rho_water) * t

    def _reset_accumulators(self) -> None:
        self.acc = {k: 0.0 for k in ("r_break", "r_clear", "r_up", "r_total", "lift", "channel")}
        self.acc_climb = 0.0
        self.acc_contact = 0.0
        self.acc_stacked = 0.0
        self.acc_fracture = 0.0
        self.acc_up_state = 0.0
        self.samples = 0
        self.t = 0.0
        self.series = {k: [] for k in ("t", "R_break", "R_clear", "R_up", "R_total",
                                       "n_climbing", "n_contact", "lift_force_z")}

    # -- forces ------------------------------------------------------------
    def _hull_contacts(self) -> Dict[str, torch.Tensor]:
        f = self.field
        con = disk_patch_contact(f.centers, f.quats, f.radius, f.thickness, self.patches,
                                n_samples=self.n_samples)
        n = con["normal"]
        arm = con["point_floe"] - f.centers
        v_pt = f.velocities + torch.cross(f.omegas, arm, dim=-1)
        v_n = (v_pt * n).sum(-1)
        pen = con["penetration"].clamp_min(0.0)
        active = con["active"]
        f_n = torch.where(active, (self.k_n * pen - self.c_n * v_n).clamp_min(0.0),
                          torch.zeros_like(pen))
        v_t = v_pt - n * v_n.unsqueeze(-1)
        slip = v_t.norm(dim=-1)
        f_t_mag = torch.minimum(self.c_t * slip, self.contact.friction * f_n)
        f_t = -v_t / slip.clamp_min(1e-300).unsqueeze(-1) * f_t_mag.unsqueeze(-1)
        force = n * f_n.unsqueeze(-1) + f_t
        torque = torch.cross(arm, force, dim=-1)
        return {"force": force, "torque": torque, "normal_force": f_n, "normal": n,
                "penetration": pen, "active": active,
                "patch_index": con["patch_index"], "point": con["point_floe"]}

    def _anchor_forces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.field
        # the anchor is a material point of the sheet: it advects with the far
        # field (updated in :meth:`step` together with the positions, so a floe
        # travelling with the field carries zero anchor strain)
        disp = f.centers - f.anchor
        elastic = -self.anchor_stiffness.unsqueeze(-1) * disp
        mag = elastic.norm(dim=-1)
        snap = f.bonded & (mag > self.anchor_strength)
        self._snapped = snap.clone()
        if bool(snap.any()):
            e_break = 0.5 * self.anchor_strength[snap] ** 2 / self.anchor_stiffness[snap]
            self.acc_fracture += float(e_break.sum())
            f.bonded = f.bonded & ~snap
        force = torch.where(f.bonded.unsqueeze(-1), elastic, torch.zeros_like(elastic))
        return force, torch.zeros_like(force)

    def _hydrostatic_forces_exact(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Large-angle hydrostatics: exact buoyancy force **and** moment.

        The floe is the cylinder ``rho <= R``, ``|zeta| <= t/2``; the waterline
        cuts it at the floe's *current* attitude and centre height, so the
        submerged volume, the centre of buoyancy and the resulting moment about
        the centre of mass are all exact for that attitude (no small-angle
        linearisation, no level-waterplane assumption).  Costs one 24x32 disk
        quadrature per floe per step.
        """
        f = self.field
        state = submerged_hydrostatics(f.radius, f.thickness, f.quats,
                                       f.centers[:, 2], self._water)
        force = state.weight.clone()
        force[:, 2] = force[:, 2] + state.buoyancy[:, 2]
        return force, state.moment

    def _hydrostatic_forces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.righting_model == RIGHTING_MODEL_EXACT:
            return self._hydrostatic_forces_exact()
        f = self.field
        b = archimedes_buoyancy(f.centers[:, 2], f.radius, f.thickness,
                               rho_water=self.rho_water, rho_ice=self.rho_ice,
                               gravity=self.gravity)
        force = torch.zeros_like(f.centers)
        force[:, 2] = b["residual"]
        axis = quat_rotate(f.quats, torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype,
                                                 device=f.centers.device).expand_as(f.centers))
        cos_tilt = axis[:, 2].abs().clamp(max=1.0)
        tilt = torch.acos(cos_tilt)
        d_sub = b["submerged_thickness"].clamp_min(1e-3)
        gm = f.radius ** 2 / (4.0 * d_sub)                     # BM for a disk, BG neglected
        v_sub = math.pi * f.radius ** 2 * b["submerged_thickness"]
        m_rest = self.rho_water * self.gravity * v_sub * gm * torch.sin(tilt)
        zhat = torch.zeros_like(axis)
        zhat[:, 2] = 1.0
        horiz = torch.cross(zhat, axis, dim=-1)
        hn = horiz.norm(dim=-1)
        m_rest = torch.where(hn > 1e-12, m_rest, torch.zeros_like(m_rest))
        torque = -horiz / hn.clamp_min(1e-300).unsqueeze(-1) * m_rest.unsqueeze(-1)
        return force, torque

    def step(self) -> None:
        f = self.field
        hull = self._hull_contacts()
        anchor_f, anchor_t = self._anchor_forces()
        hydro_f, hydro_t = self._hydrostatic_forces()

        v_amb = torch.zeros_like(f.velocities)
        v_amb[:, 0] = -self.speed
        damp_f = -self.contact.water_damping * self.mass.unsqueeze(-1) * (f.velocities - v_amb)
        damp_t = -self.contact.angular_damping * self.inertia.unsqueeze(-1) * f.omegas

        total_f = hull["force"] + anchor_f + hydro_f + damp_f
        total_t = hull["torque"] + anchor_t + hydro_t + damp_t

        f.velocities = f.velocities + total_f / self.mass.unsqueeze(-1) * self.dt
        f.centers = f.centers + f.velocities * self.dt
        f.omegas = f.omegas + total_t / self.inertia.unsqueeze(-1) * self.dt
        f.quats = quat_integrate(f.quats, f.omegas, self.dt)
        # material advection of the sheet reference (same displacement as a free floe)
        f.anchor[:, 0] = f.anchor[:, 0] - self.speed * self.dt

        self._account(hull)
        self._stacking()
        self._recycle()
        self.t += self.dt

    # -- resistance channels ----------------------------------------------
    def _account(self, hull: Dict[str, torch.Tensor]) -> None:
        f = self.field
        fx = hull["force"][:, 0]
        active = hull["active"]
        sin_a = math.sin(self.slope.alpha)
        on_slope = active & (hull["patch_index"] < self.n_slope_facets)
        lift = hull["normal_force"] * hull["normal"][:, 2]
        b = archimedes_buoyancy(f.centers[:, 2], f.radius, f.thickness,
                               rho_water=self.rho_water, rho_ice=self.rho_ice,
                               gravity=self.gravity)
        deficit = b["weight"] - b["buoyancy"]
        # ride-up state: (i) the contact is on a bow-slope facet, (ii) the contact
        # lifts (n_z > 0), (iii) the lift overcomes the buoyant deficit, and
        # (iv) the floe is carried *above* its hydrostatic draught -- a floe that
        # is merely pushed along at its floating draught is being cleared, not
        # ridden up.  Criterion (iv) is what ties R_up to the physical state.
        d_eq = (self.rho_ice / self.rho_water) * f.thickness
        lifted = b["submerged_thickness"] < 0.9 * d_eq
        # disjoint partition of the measured hull force: a floe that is still
        # part of the sheet loads the breaking channel first (its contact force
        # is the force needed to tear it free), otherwise it either rides up or
        # is being cleared
        breaking = active & (f.bonded | self._snapped)
        climbing = on_slope & (lift > deficit) & lifted & ~breaking
        clearing = active & ~breaking & ~climbing

        self.acc["r_total"] += float(fx[active].sum())
        self.acc["r_break"] += float(fx[breaking].sum())
        self.acc["r_up"] += float(fx[climbing].sum())
        self.acc["r_clear"] += float(fx[clearing].sum())
        self.acc["lift"] += float(lift[climbing].sum())
        self.acc["channel"] += float((lift[climbing] * sin_a).sum())
        self.acc_climb += float(climbing.sum())
        self.acc_contact += float(active.sum())
        self.samples += 1

        self.series["t"].append(self.t)
        self.series["R_break"].append(float(fx[breaking].sum()))
        self.series["R_clear"].append(float(fx[clearing].sum()))
        self.series["R_up"].append(float(fx[climbing].sum()))
        self.series["R_total"].append(float(fx[active].sum()))
        self.series["n_climbing"].append(float(climbing.sum()))
        self.series["n_contact"].append(float(active.sum()))
        self.series["lift_force_z"].append(float(lift[climbing].sum()))

    # -- field maintenance -------------------------------------------------
    def _recycle(self) -> None:
        """Return floes that left the contact zone to the inflow line.

        A floe leaves when it is advected past ``x_exit`` (the ship has passed
        it), or when the bow has swept it clear of the reach -- forward of
        ``x_inflow`` (a plow-shaped bow pushes ice ahead of itself) or outboard of
        the channel half width.  Recycling keeps the field statistically
        stationary, which is the assumption every time average in this module
        rests on; it is not a physical mechanism.
        """
        f = self.field
        rmax = float(f.radius.max())
        out = ((f.centers[:, 0] < self.x_exit)
               | (f.centers[:, 0] > self.x_inflow + 2.0 * rmax)
               | (f.centers[:, 1].abs() > self.slope.y_half + 2.0 * rmax))
        k = int(out.sum())
        if k == 0:
            return
        sl = self.slope
        f.centers[out, 0] = self.x_inflow - 2.0 * torch.rand(k, generator=self.gen, dtype=self.dtype)
        f.centers[out, 1] = (2.0 * torch.rand(k, generator=self.gen, dtype=self.dtype) - 1.0) * sl.y_half
        f.centers[out, 2] = self.equilibrium_height()
        f.velocities[out] = 0.0
        f.velocities[out, 0] = -self.speed
        f.omegas[out] = 0.0
        yaw = 2.0 * math.pi * torch.rand(k, generator=self.gen, dtype=self.dtype)
        one = torch.zeros(k, dtype=self.dtype)
        f.quats[out] = quat_normalize(quat_multiply(
            torch.stack((torch.cos(yaw / 2), one, torch.zeros(k, dtype=self.dtype),
                         torch.sin(yaw / 2)), dim=1),
            torch.stack((torch.ones(k, dtype=self.dtype), one, one, one), dim=1)))
        f.anchor[out] = f.centers[out]
        f.bonded[out] = torch.rand(k, generator=self.gen, dtype=self.dtype) < self.contact.bonded_fraction

    def _stacking(self) -> None:
        """First-order stacking on the shelf: at most one floe per ``(x, y)`` bin."""
        f = self.field
        sl = self.slope
        bottom = f.centers[:, 2] - 0.5 * self.thickness
        on_shelf = ((f.centers[:, 0] <= sl.x_upper) & (f.centers[:, 0] >= sl.x_upper - sl.shelf_length)
                    & ((bottom - sl.z_upper).abs() < 0.30))
        occupied: Dict[Tuple[int, int], int] = {}
        for i in torch.nonzero(on_shelf, as_tuple=False).flatten().tolist():
            xb = 0 if float(f.centers[i, 0]) > sl.x_upper - 0.5 * sl.shelf_length else 1
            yb = int(min(max((float(f.centers[i, 1]) + sl.y_half) / (2.0 * sl.y_half) * 8.0, 0.0), 7.0))
            key = (xb, yb)
            if key in occupied:
                # a second floe for the same bin: no second layer, push it aside
                sign = 1.0 if float(f.centers[i, 1]) >= 0.0 else -1.0
                f.velocities[i, 1] += sign * 0.5
            else:
                occupied[key] = i
        self._bin_count = len(occupied)
        self.acc_stacked = max(self.acc_stacked, float(self._bin_count))
        self.acc_up_state += float(on_shelf.sum())

    # -- driver ------------------------------------------------------------
    def run(self) -> RideUpResult:
        for _ in range(self.warmup_steps):
            self.step()
        self.acc_fracture = 0.0
        self._reset_accumulators()
        for _ in range(self.steps):
            self.step()
        s = max(self.samples, 1)
        duration = self.steps * self.dt
        energy_rate = self.acc_fracture / duration if duration > 0 else 0.0
        return RideUpResult(
            fr=self.fr, thickness=self.thickness, angle_deg=self.slope.angle_deg,
            seed=self.seed, speed=self.speed,
            r_break=self.acc["r_break"] / s, r_clear=self.acc["r_clear"] / s,
            r_up=self.acc["r_up"] / s, r_total=self.acc["r_total"] / s,
            r_up_channel=self.acc["channel"] / s, lift_force_z=self.acc["lift"] / s,
            n_climbing=self.acc_climb / s, n_contact=self.acc_contact / s,
            n_stacked=self.acc_stacked, fracture_energy_rate=energy_rate,
            r_break_energy=(energy_rate / self.speed) if self.speed > 0 else float("nan"),
            steps=self.steps, dt=self.dt,
        )


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def run_case(*, fr: float, thickness: float, angle_deg: float, seed: int = 0,
             steps: int = 1800, dt: float = 0.01, n_floes: int = 40, warmup_steps: int = 400,
             n_samples: int = 24, patches=None, n_slope_facets=None,
             contact: ContactParams = ContactParams(), ship_length: float = 10.0,
             radius_mean: float = 4.0, return_sim: bool = False,
             righting_model: str = RIGHTING_MODEL_NEAR_LEVEL):
    """Run one parameter point and return a :class:`RideUpResult` (or the sim)."""
    sim = RideUpSimulation(fr=fr, thickness=thickness, slope=BowSlope(angle_deg=angle_deg),
                           seed=seed, n_floes=n_floes, contact=contact, dt=dt,
                           steps=steps, warmup_steps=warmup_steps, n_samples=n_samples,
                           patches=patches, n_slope_facets=n_slope_facets,
                           ship_length=ship_length, radius_mean=radius_mean,
                           righting_model=righting_model)
    result = sim.run()
    return (result, sim) if return_sim else result


def monotonicity_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Least-squares slope of ``y`` on ``x`` (machine decision for criterion (b))."""
    xs = torch.tensor([float(v) for v in x], dtype=torch.float64)
    ys = torch.tensor([float(v) for v in y], dtype=torch.float64)
    xm = xs.mean()
    ym = ys.mean()
    denom = float(((xs - xm) ** 2).sum())
    if denom <= 0.0:
        return float("nan")
    return float(((xs - xm) * (ys - ym)).sum()) / denom


def sweep(*, fr_values: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
          thickness_values: Sequence[float] = (0.3, 0.5, 0.9, 1.2, 1.5),
          angle_values: Sequence[float] = (20.0, 30.0, 40.0, 45.0),
          seeds: Sequence[int] = (0, 1, 2), steps: int = 1800, dt: float = 0.01,
          n_floes: int = 40, contact: ContactParams = ContactParams(),
          warmup_steps: int = 400, n_samples: int = 24) -> List[RideUpResult]:
    """Full parameter sweep; one :class:`RideUpResult` per (Fr, h, alpha, seed)."""
    out: List[RideUpResult] = []
    for fr in fr_values:
        for h in thickness_values:
            for ang in angle_values:
                for sd in seeds:
                    out.append(run_case(fr=fr, thickness=h, angle_deg=ang, seed=sd,
                                        steps=steps, dt=dt, n_floes=n_floes, contact=contact,
                                        warmup_steps=warmup_steps, n_samples=n_samples))
    return out
