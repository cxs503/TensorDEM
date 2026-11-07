"""Six-degree-of-freedom capsize dynamics of disk floes (IN22-N02).

The H1 channel engine (:mod:`tensordem.ice_channel`) treats a floe as a
flat-lying disk with a *locked* attitude: heave ``z`` and yaw ``psi`` are
integrated, and the tilt mode is a passive single-axis pendulum driven by the
metacentric righting moment ``M = -rho_w g V_sub GM sin(phi)``.  Because the
tilt never feeds back into the hydrostatics or into the contact geometry, a
floe cannot lose stability in that model and the cleared channel width is
insensitive to the roll/pitch degree of freedom.

This module lifts the same floe population to a genuine **six-degree-of-freedom
rigid body** (``x, y, z`` translation and roll, pitch, yaw rotation):

* Orientation is a unit quaternion ``q`` (body -> world) integrated with the
  exponential map of the body-frame angular velocity,
  ``q <- q (x) exp(omega_b dt / 2)``, renormalised every step.  The norm stays
  within a few ulp of unity for arbitrarily long runs (:func:`quat_norm_error`).
* Hydrostatics are evaluated on the **tilted** cylinder by deterministic
  Gauss-Legendre quadrature over the footprint: a waterplane column at body
  position ``(xi, eta)`` is submerged over an axis length
  ``L = clamp(zeta_lim + t/2, 0, t)``, which gives the submerged volume ``V``,
  the centre of buoyancy ``r_cb`` in body axes, the buoyancy potential energy
  and therefore the full buoyancy moment ``(R r_cb) x F_b`` about the centre of
  mass.  Gravity acts at the centre of mass, so the buoyancy-gravity couple is
  the only attitude restoring mechanism -- exactly the physics that the locked
  model parameterises away.
* The attitude potential ``U(q)`` has a stable well at the flat pose and, for a
  disk, a secondary well at the edge-on pose.  The upright well loses its
  positive-definite Hessian at the *fold* of the righting-moment curve
  ``M(phi)``; that fold is the capsize angle ``theta_s``.

Two independent numerical routes to ``theta_s`` are provided and cross-checked:

1. :func:`capsize_angle_from_moment` -- the quasi-static loss of equilibrium
   ``argmax M_r(phi)`` from the **force route** (centre of buoyancy times
   buoyancy force, heave relaxed by a bisection solve);
2. :func:`capsize_angle_from_hessian` -- the first zero of
   ``d^2 U_c / d phi^2`` from the **energy route** (finite differences of the
   heave-relaxed potential, including the heave-attitude coupling).

The small-angle limit is pinned to the closed form ``k_phi = rho_w g V GM``
with ``GM = R^2/(4d) + d/2 - t/2``.

Simulation modes
----------------

``SixDofFloeSimulation(mode="tilt_locked")``
    Delegates every step to :class:`IceChannelSimulation` and asserts that the
    rotational state is exactly at rest, so the run is bit-for-bit the H1 run
    (regression lock through ``state_digest``).

``SixDofFloeSimulation(mode="six_dof")``
    Integrates the full attitude.  In-plane contact / hull / drag forces are
    taken from the H1 surface; the vertical and rotational channels add
    (a) tilted buoyancy with centre-of-buoyancy moments, (b) contact and hull
    moments about the waterline -- the centre of mass of a floe of density
    ratio ``rho_i/rho_w`` sits ``(1 - rho_i/rho_w) t/2`` below the waterline, so
    horizontal contact forces heel the floe, (c) the wind moment acting above
    the centre of mass, and (d) an optional hull-shoulder submersion term
    (``shoulder_submergence``, the ``tan`` of the shoulder angle) that pushes a
    contacted floe down at its rim, which is the mechanism that actually drives
    floe subduction under a hull.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .ice_channel import (
    ContactParameters,
    HullGeometry,
    IceChannelSimulation,
    WaterEnvironment,
    _DTYPE,
    submerged_draught,
)

__all__ = [
    "CapsizeParameters",
    "CapsizeCurve",
    "DiskQuadrature",
    "HydrostaticState",
    "SixDofFloeSimulation",
    "disk_quadrature",
    "analytic_metacentric_height",
    "analytic_roll_stiffness",
    "submerged_hydrostatics",
    "heave_equilibrium_height",
    "attitude_potential",
    "righting_moment",
    "capsize_curve",
    "capsize_angle_from_moment",
    "capsize_angle_from_hessian",
    "quat_identity",
    "quat_normalize",
    "quat_multiply",
    "quat_conjugate",
    "quat_from_axis_angle",
    "quat_from_euler",
    "quat_to_matrix",
    "integrate_quaternion",
    "quat_norm_error",
    "quat_tilt_angle",
]


# ---------------------------------------------------------------------------
# Quaternions (scalar-first: q = (w, x, y, z))
# ---------------------------------------------------------------------------

def quat_identity(n: int = 1, dtype: torch.dtype = _DTYPE) -> torch.Tensor:
    if int(n) <= 0:
        raise ValueError("quaternion batch size must be positive")
    q = torch.zeros((int(n), 4), dtype=dtype)
    q[:, 0] = 1.0
    return q


def quat_normalize(q: torch.Tensor, eps: float = 1e-14) -> torch.Tensor:
    """Return the unit quaternion; a vanishing quaternion is rejected."""
    q = torch.as_tensor(q, dtype=_DTYPE)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("quaternion array must have shape (N, 4)")
    if not torch.isfinite(q).all():
        raise ValueError("quaternion contains non-finite entries")
    norm = torch.linalg.norm(q, dim=1, keepdim=True)
    if bool((norm < float(eps)).any()):
        raise ValueError("quaternion norm is numerically zero; cannot normalise")
    return q / norm


def quat_norm_error(q: torch.Tensor) -> torch.Tensor:
    """Largest ``| |q| - 1 |`` over the batch."""
    q = torch.as_tensor(q, dtype=_DTYPE)
    return (torch.linalg.norm(q, dim=1) - 1.0).abs().max()


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a (x) b`` (as rotations: ``b`` is applied first)."""
    a = torch.as_tensor(a, dtype=_DTYPE)
    b = torch.as_tensor(b, dtype=_DTYPE)
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    q = torch.as_tensor(q, dtype=_DTYPE)
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=1)


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = torch.as_tensor(axis, dtype=_DTYPE)
    angle = torch.as_tensor(angle, dtype=_DTYPE).reshape(-1)
    if axis.ndim == 1:
        axis = axis.unsqueeze(0).expand(angle.numel(), 3)
    norm = torch.linalg.norm(axis, dim=1, keepdim=True)
    if bool((norm < 1e-15).any()):
        raise ValueError("rotation axis must be non-zero")
    unit = axis / norm
    half = 0.5 * angle
    return torch.cat([torch.cos(half).unsqueeze(1),
                      torch.sin(half).unsqueeze(1) * unit], dim=1)


def quat_from_euler(roll, pitch, yaw) -> torch.Tensor:
    """Intrinsic 3-2-1 (yaw-pitch-roll) angles -> unit quaternion."""
    roll = torch.as_tensor(roll, dtype=_DTYPE).reshape(-1)
    pitch = torch.as_tensor(pitch, dtype=_DTYPE).reshape(-1)
    yaw = torch.as_tensor(yaw, dtype=_DTYPE).reshape(-1)
    if not (roll.numel() == pitch.numel() == yaw.numel()):
        raise ValueError("roll, pitch and yaw must have the same length")
    n = int(roll.numel())
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=_DTYPE).expand(n, 3)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=_DTYPE).expand(n, 3)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=_DTYPE).expand(n, 3)
    qx = quat_from_axis_angle(ex, roll)
    qy = quat_from_axis_angle(ey, pitch)
    qz = quat_from_axis_angle(ez, yaw)
    return quat_normalize(quat_multiply(qz, quat_multiply(qy, qx)))


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Body -> world rotation matrix of a batch of unit quaternions."""
    q = quat_normalize(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    r[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    r[:, 0, 1] = 2.0 * (x * y - w * z)
    r[:, 0, 2] = 2.0 * (x * z + w * y)
    r[:, 1, 0] = 2.0 * (x * y + w * z)
    r[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    r[:, 1, 2] = 2.0 * (y * z - w * x)
    r[:, 2, 0] = 2.0 * (x * z - w * y)
    r[:, 2, 1] = 2.0 * (y * z + w * x)
    r[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return r


def integrate_quaternion(q: torch.Tensor, omega_body: torch.Tensor,
                         dt: float) -> torch.Tensor:
    """Exact exponential-map update ``q <- q (x) exp(omega_b dt / 2)``."""
    dt = float(dt)
    if not math.isfinite(dt):
        raise ValueError("dt must be finite")
    q = quat_normalize(q)
    omega_body = torch.as_tensor(omega_body, dtype=_DTYPE).reshape(-1, 3)
    if q.shape[0] != omega_body.shape[0]:
        raise ValueError("quaternion and angular velocity batches differ in size")
    speed = torch.linalg.norm(omega_body, dim=1)
    half = 0.5 * speed * dt
    # sin(half)/speed -> dt/2 in the zero-rotation limit (no division by zero)
    ratio = torch.where(speed > 1e-12, torch.sin(half) / torch.clamp(speed, min=1e-30),
                        torch.full_like(speed, 0.5 * dt))
    delta = torch.cat([torch.cos(half).unsqueeze(1),
                       omega_body * ratio.unsqueeze(1)], dim=1)
    return quat_normalize(quat_multiply(q, delta))


def quat_tilt_angle(q: torch.Tensor) -> torch.Tensor:
    """Angle [rad] between the floe symmetry axis (body ``e3``) and vertical."""
    q = quat_normalize(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    cos_gamma = 1.0 - 2.0 * (x * x + y * y)
    return torch.acos(torch.clamp(cos_gamma, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Quadrature over the disk footprint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiskQuadrature:
    """Gauss-Legendre (radial) x midpoint (azimuth) rule on the unit disk."""

    points: torch.Tensor      # (M, 2) nodes inside the unit disk
    weights: torch.Tensor     # (M,)  int f s ds dpsi (area element carries R^2)
    radial_order: int
    azimuth_order: int

    @property
    def size(self) -> int:
        return int(self.points.shape[0])


def disk_quadrature(radial_order: int = 24, azimuth_order: int = 32,
                    dtype: torch.dtype = _DTYPE) -> DiskQuadrature:
    radial_order = int(radial_order)
    azimuth_order = int(azimuth_order)
    if radial_order < 2 or azimuth_order < 4:
        raise ValueError("quadrature needs radial_order >= 2 and azimuth_order >= 4")
    s, ws = np.polynomial.legendre.leggauss(radial_order)
    s = 0.5 * (s + 1.0)
    ws = 0.5 * ws
    psi = 2.0 * np.pi * (np.arange(azimuth_order) + 0.5) / azimuth_order
    wpsi = 2.0 * np.pi / azimuth_order
    xs = s[:, None] * np.cos(psi)[None, :]
    ys = s[:, None] * np.sin(psi)[None, :]
    weights = np.broadcast_to(ws[:, None] * wpsi * s[:, None],
                              (radial_order, azimuth_order)).reshape(-1)
    points = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)
    return DiskQuadrature(points=torch.as_tensor(points, dtype=dtype),
                          weights=torch.as_tensor(weights, dtype=dtype),
                          radial_order=radial_order,
                          azimuth_order=azimuth_order)


_DEFAULT_QUAD = disk_quadrature(24, 32)


# ---------------------------------------------------------------------------
# Hydrostatics of a tilted disk
# ---------------------------------------------------------------------------

@dataclass
class HydrostaticState:
    """Output of :func:`submerged_hydrostatics` (batches of length N)."""

    volume: torch.Tensor          # m^3
    centroid_body: torch.Tensor   # (N, 3) centre of buoyancy in body axes
    centroid_world: torch.Tensor  # (N, 3) centre of buoyancy in world axes
    buoyancy: torch.Tensor        # (N, 3) world force
    weight: torch.Tensor          # (N, 3) world force
    moment: torch.Tensor          # (N, 3) world moment about the centre of mass
    energy: torch.Tensor          # J, floe + displaced-water potential energy
    projected_area: torch.Tensor  # m^2, waterplane projection pi R^2 |n_z|
    mass: torch.Tensor
    thickness: torch.Tensor
    radius: torch.Tensor


def _batch(values, dtype: torch.dtype = _DTYPE) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().to(dtype=dtype).reshape(-1)
    if isinstance(values, (list, tuple)):
        return torch.as_tensor(list(values), dtype=dtype).reshape(-1)
    return torch.as_tensor([values], dtype=dtype).reshape(-1)


def submerged_hydrostatics(radius, thickness, quat, centre_height,
                           water: WaterEnvironment,
                           quad: Optional[DiskQuadrature] = None) -> HydrostaticState:
    """Volume, centre of buoyancy, forces, moment and potential energy.

    The floe is the cylinder ``rho <= R``, ``|zeta| <= t/2`` in body axes.  With
    the body-frame waterline normal ``n_b = R_wb^T e_z`` and the waterline
    offset ``h = -z_cg``, the column at body position ``(xi, eta)`` is submerged
    for ``zeta < (h - n_bx xi - n_by eta) / n_bz``, i.e. the submerged axis
    length is ``L = clamp(zeta_lim + t/2, 0, t)``.
    """
    quad = quad or _DEFAULT_QUAD
    r = _batch(radius)
    t = _batch(thickness)
    z = _batch(centre_height)
    if not (r.numel() == t.numel() == z.numel()):
        raise ValueError("radius, thickness and centre_height must have the same length")
    if bool((r <= 0.0).any()) or bool((t <= 0.0).any()):
        raise ValueError("radius and thickness must be strictly positive")
    if not torch.isfinite(z).all():
        raise ValueError("centre height must be finite")
    n = int(r.numel())
    q = quat_normalize(torch.as_tensor(quat, dtype=_DTYPE).reshape(n, 4))
    rmat = quat_to_matrix(q)                     # body -> world
    n_b = rmat[:, 2, :]                          # R^T e_z == third row

    pts = quad.points.to(dtype=_DTYPE)
    wts = quad.weights.to(dtype=_DTYPE)
    area_weight = (r * r).unsqueeze(1) * wts.unsqueeze(0)          # (n, M)
    # the quadrature nodes live on the unit disk; the waterline variation of a
    # column must be evaluated at the physical in-plane coordinate R * xi
    const = r.unsqueeze(1) * (n_b[:, 0:1] * pts[:, 0].unsqueeze(0)
                              + n_b[:, 1:2] * pts[:, 1].unsqueeze(0))
    h = -z
    nz = n_b[:, 2:3]
    tiny = nz.abs() < 1e-12
    safe = torch.where(tiny, torch.full_like(nz, 1e-12), nz)
    zeta_lim = (h.unsqueeze(1) - const) / safe
    lower = -0.5 * t.unsqueeze(1)
    upper = 0.5 * t.unsqueeze(1)
    thickness_col = t.unsqueeze(1)
    # n_bz > 0: the submerged part is zeta < zeta_lim; n_bz < 0 flips it, so a
    # floe that has rolled past 90 deg is handled by the mirrored branch
    upside = nz < 0.0
    length_down = torch.clamp(zeta_lim - lower, min=torch.zeros_like(lower))
    length_up = torch.clamp(upper - zeta_lim, min=torch.zeros_like(upper))
    length = torch.where(upside, length_up, length_down)
    length = torch.minimum(length, thickness_col)
    if bool(tiny.any()):
        full = torch.where(const < h.unsqueeze(1), thickness_col,
                           torch.zeros_like(length))
        length = torch.where(tiny.expand_as(length), full, length)
    zeta_c = torch.where(upside, upper - 0.5 * length, lower + 0.5 * length)

    volume = (area_weight * length).sum(dim=1)
    if not torch.isfinite(volume).all() or bool((volume < 0.0).any()):
        raise ValueError("illegal submerged volume (negative or non-finite)")
    # in-plane first moment: the quadrature nodes live on the unit disk, so the
    # physical lever arm carries one factor of R
    m_xy = torch.stack([
        (area_weight * length * pts[:, 0].unsqueeze(0)).sum(dim=1),
        (area_weight * length * pts[:, 1].unsqueeze(0)).sum(dim=1)], dim=1) * r.unsqueeze(1)
    m_z = (area_weight * length * zeta_c).sum(dim=1)
    safe_volume = torch.clamp(volume, min=1e-30)
    centroid_body = torch.cat([m_xy / safe_volume.unsqueeze(1),
                               (m_z / safe_volume).unsqueeze(1)], dim=1)
    centre = torch.stack([torch.zeros(n, dtype=_DTYPE),
                          torch.zeros(n, dtype=_DTYPE), z], dim=1)
    centroid_world = torch.einsum('nij,nj->ni', rmat, centroid_body) + centre

    mass = water.ice_density * math.pi * r * r * t
    f_buoy = water.water_density * water.gravity * volume
    buoyancy = torch.stack([torch.zeros(n, dtype=_DTYPE),
                            torch.zeros(n, dtype=_DTYPE), f_buoy], dim=1)
    weight = torch.stack([torch.zeros(n, dtype=_DTYPE),
                          torch.zeros(n, dtype=_DTYPE),
                          -mass * water.gravity], dim=1)
    moment = torch.linalg.cross(centroid_world - centre, buoyancy)

    moment_along_normal = (n_b[:, 0] * m_xy[:, 0] + n_b[:, 1] * m_xy[:, 1]
                           + n_b[:, 2] * m_z)
    energy = (mass * water.gravity * z
              + water.water_density * water.gravity
              * (-z * volume - moment_along_normal))
    projected_area = math.pi * r * r * n_b[:, 2].abs()
    return HydrostaticState(volume=volume, centroid_body=centroid_body,
                            centroid_world=centroid_world, buoyancy=buoyancy,
                            weight=weight, moment=moment, energy=energy,
                            projected_area=projected_area, mass=mass,
                            thickness=t, radius=r)


def analytic_metacentric_height(radius: float, thickness: float, rho_ice: float,
                                rho_water: float) -> float:
    """``GM = KB + BM - KG`` of a flat disk measured from the keel [m].

    ``KB = d/2``, ``BM = I_wl/V = R^2/(4d)`` with ``I_wl = pi R^4/4``,
    ``KG = t/2`` and ``d = (rho_i/rho_w) t``.
    """
    radius = float(radius)
    thickness = float(thickness)
    if radius <= 0.0 or thickness <= 0.0:
        raise ValueError("radius and thickness must be positive")
    if not 0.0 < float(rho_ice) < float(rho_water):
        raise ValueError("need 0 < rho_ice < rho_water")
    draught = (float(rho_ice) / float(rho_water)) * thickness
    return 0.5 * draught + radius * radius / (4.0 * draught) - 0.5 * thickness


def analytic_roll_stiffness(radius: float, thickness: float, rho_ice: float,
                            rho_water: float, gravity: float) -> float:
    """Small-angle roll stiffness ``k_phi = rho_w g V GM`` [N m / rad]."""
    radius = float(radius)
    thickness = float(thickness)
    gm = analytic_metacentric_height(radius, thickness, rho_ice, rho_water)
    draught = (float(rho_ice) / float(rho_water)) * thickness
    volume = math.pi * radius * radius * draught
    return float(rho_water) * float(gravity) * volume * gm


def _heave_equilibrium_batch(radius: torch.Tensor, thickness: torch.Tensor,
                             quats: torch.Tensor, water: WaterEnvironment,
                             quad: Optional[DiskQuadrature] = None,
                             iterations: int = 45) -> torch.Tensor:
    """Vectorised bisection for ``rho_w V(z) = m`` at a batch of attitudes."""
    r = _batch(radius)
    t = _batch(thickness)
    if not (r.numel() == t.numel() == quats.shape[0]):
        raise ValueError("radius, thickness and quaternion batches must agree")
    if bool((r <= 0.0).any()) or bool((t <= 0.0).any()):
        raise ValueError("radius and thickness must be strictly positive")
    target = (water.ice_density / water.water_density) * math.pi * r * r * t
    lo = -r - t
    hi = r + t
    vol_lo = submerged_hydrostatics(r, t, quats, lo, water, quad).volume
    vol_hi = submerged_hydrostatics(r, t, quats, hi, water, quad).volume
    if bool((vol_lo < target).any()) or bool((vol_hi > target).any()):
        raise ValueError("attitude is not floatable in the bracketed height range")
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        volume = submerged_hydrostatics(r, t, quats, mid, water, quad).volume
        above = volume > target
        lo = torch.where(above, mid, lo)
        hi = torch.where(above, hi, mid)
        if float((hi - lo).max()) < 1e-14:
            break
    return 0.5 * (lo + hi)


def heave_equilibrium_height(radius: float, thickness: float, quat,
                             water: WaterEnvironment,
                             quad: Optional[DiskQuadrature] = None,
                             iterations: int = 45) -> float:
    """Centre height ``z*`` that satisfies ``rho_w V(z) = m`` for a fixed attitude.

    ``V`` decreases monotonically with ``z``; the batched bisection converges to
    ~1e-14 m, so the residual buoyancy/gravity imbalance is at round-off level.
    """
    q = torch.as_tensor(quat, dtype=_DTYPE).reshape(-1, 4)
    if q.shape[0] != 1:
        raise ValueError("heave_equilibrium_height takes a single attitude")
    z = _heave_equilibrium_batch(torch.tensor([float(radius)], dtype=_DTYPE),
                                 torch.tensor([float(thickness)], dtype=_DTYPE),
                                 q, water, quad, iterations)
    return float(z[0])


def attitude_potential_batch(radius: float, thickness: float, tilts,
                             water: WaterEnvironment,
                             quad: Optional[DiskQuadrature] = None) -> Dict[str, torch.Tensor]:
    """Heave-relaxed attitude scan at roll angles ``tilts`` (rad)."""
    tilts = torch.as_tensor(tilts, dtype=_DTYPE).reshape(-1)
    n = int(tilts.numel())
    if n == 0:
        raise ValueError("empty attitude scan")
    zeros = torch.zeros(n, dtype=_DTYPE)
    quats = quat_from_euler(tilts, zeros, zeros)
    r = torch.full((n,), float(radius), dtype=_DTYPE)
    t = torch.full((n,), float(thickness), dtype=_DTYPE)
    heights = _heave_equilibrium_batch(r, t, quats, water, quad)
    state = submerged_hydrostatics(r, t, quats, heights, water, quad)
    return {"tilt": tilts, "height": heights, "energy": state.energy,
            "moment": -state.moment[:, 0], "volume": state.volume,
            "buoyancy": state.buoyancy[:, 2]}


def attitude_potential(radius: float, thickness: float, tilt: float,
                       water: WaterEnvironment,
                       quad: Optional[DiskQuadrature] = None) -> Dict[str, float]:
    """Heave-relaxed attitude energy, righting moment and state at angle ``tilt``.

    ``tilt`` is a roll about the world x axis.  ``moment`` is the *righting*
    moment ``M_r = -tau_x`` (positive = restoring).
    """
    out = attitude_potential_batch(radius, thickness, [float(tilt)], water, quad)
    return {key: float(value[0]) for key, value in out.items()}


def righting_moment(radius: float, thickness: float, tilt: float,
                    water: WaterEnvironment,
                    quad: Optional[DiskQuadrature] = None) -> float:
    """Righting moment [N m] of one floe at roll angle ``tilt`` (heave relaxed)."""
    return float(attitude_potential(radius, thickness, tilt, water, quad)["moment"])


@dataclass
class CapsizeCurve:
    """Attitude scan of one floe: energy, righting moment, stiffness."""

    tilt: torch.Tensor          # rad
    energy: torch.Tensor        # J
    moment: torch.Tensor        # N m (righting, positive = restoring)
    stiffness: torch.Tensor     # d^2 U / d phi^2 [N m / rad]
    height: torch.Tensor        # m, relaxed heave
    volume: torch.Tensor        # m^3
    radius: float
    thickness: float
    metacentric_height: float
    analytic_stiffness: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "radius": self.radius,
            "thickness": self.thickness,
            "metacentric_height": self.metacentric_height,
            "analytic_roll_stiffness": self.analytic_stiffness,
            "tilt_deg": [float(v) for v in torch.rad2deg(self.tilt)],
            "energy_j": [float(v) for v in self.energy],
            "righting_moment_nm": [float(v) for v in self.moment],
            "stiffness_nm_per_rad": [float(v) for v in self.stiffness],
            "equilibrium_height_m": [float(v) for v in self.height],
            "submerged_volume_m3": [float(v) for v in self.volume],
        }


def capsize_curve(radius: float, thickness: float, water: WaterEnvironment,
                  tilt_count: int = 181, tilt_max: float = math.pi,
                  quad: Optional[DiskQuadrature] = None) -> CapsizeCurve:
    """Scan the attitude potential over ``[0, tilt_max]`` with relaxed heave."""
    if int(tilt_count) < 5:
        raise ValueError("tilt_count must be at least 5")
    if not (0.0 < float(tilt_max) <= math.pi):
        raise ValueError("tilt_max must lie in (0, pi]")
    tilts = torch.linspace(0.0, float(tilt_max), int(tilt_count), dtype=_DTYPE)
    scan = attitude_potential_batch(radius, thickness, tilts, water, quad)
    moment = scan["moment"]
    energy = scan["energy"]
    step = float(tilts[1] - tilts[0])
    stiffness = torch.zeros_like(energy)
    # k_phi = -dM_r/dphi = d^2U/dphi^2
    stiffness[1:-1] = (-moment[2:] + moment[:-2]) / (2.0 * step)
    stiffness[0] = (-moment[1] + moment[0]) / step
    stiffness[-1] = (-moment[-1] + moment[-2]) / step
    return CapsizeCurve(tilt=tilts, energy=energy, moment=moment,
                        stiffness=stiffness, height=scan["height"],
                        volume=scan["volume"], radius=float(radius),
                        thickness=float(thickness),
                        metacentric_height=analytic_metacentric_height(
                            radius, thickness, water.ice_density, water.water_density),
                        analytic_stiffness=analytic_roll_stiffness(
                            radius, thickness, water.ice_density,
                            water.water_density, water.gravity))


def capsize_angle_from_moment(radius: float, thickness: float,
                              water: WaterEnvironment,
                              quad: Optional[DiskQuadrature] = None,
                              upper: float = 0.5 * math.pi,
                              scan_points: int = 181) -> Dict[str, float]:
    """Capsize angle as the fold ``argmax M_r(phi)`` (force route).

    The upright branch of equilibria under a sustained external heeling moment
    ``m`` exists while ``m <= max_phi M_r(phi)``; the fold is the loss of
    equilibrium and therefore the capsize angle.  A coarse scan is refined by a
    golden-section search around the discrete maximum.
    """
    upper = float(upper)
    if not (0.0 < upper <= math.pi):
        raise ValueError("upper scan bound must lie in (0, pi]")
    if int(scan_points) < 9:
        raise ValueError("scan_points must be at least 9")
    tilts = torch.linspace(0.0, upper, int(scan_points), dtype=_DTYPE)
    scan = attitude_potential_batch(radius, thickness, tilts, water, quad)
    values = scan["moment"]
    best = int(torch.argmax(values))
    if best == 0:
        raise ValueError("righting moment does not peak inside the scan window")
    lo = float(tilts[max(best - 1, 0)])
    hi = float(tilts[min(best + 1, int(scan_points) - 1)])
    golden = 0.5 * (math.sqrt(5.0) - 1.0)
    a, b = lo, hi
    c = b - golden * (b - a)
    d = a + golden * (b - a)
    fc = righting_moment(radius, thickness, c, water, quad)
    fd = righting_moment(radius, thickness, d, water, quad)
    for _ in range(60):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - golden * (b - a)
            fc = righting_moment(radius, thickness, c, water, quad)
        else:
            a, c, fc = c, d, fd
            d = a + golden * (b - a)
            fd = righting_moment(radius, thickness, d, water, quad)
    phi = 0.5 * (a + b)
    peak = righting_moment(radius, thickness, phi, water, quad)
    return {"capsize_angle": float(phi), "peak_righting_moment": float(peak),
            "scan_bound": upper, "scan_points": int(scan_points)}


def capsize_angle_from_hessian(radius: float, thickness: float,
                               water: WaterEnvironment,
                               quad: Optional[DiskQuadrature] = None,
                               upper: float = 0.5 * math.pi,
                               scan_points: int = 721) -> Dict[str, float]:
    """Capsize angle as the first zero of ``d^2 U_c / d phi^2`` (energy route).

    ``U_c`` is the heave-relaxed potential on a uniform grid; the Hessian comes
    from a five-point finite difference of ``U_c`` -- a different numerical path
    from the centre-of-buoyancy moment used by
    :func:`capsize_angle_from_moment`.  The first positive-to-negative sign
    change is bracketed on the grid and bisected with the same operator.
    """
    upper = float(upper)
    if not (0.0 < upper <= math.pi):
        raise ValueError("upper scan bound must lie in (0, pi]")
    if int(scan_points) < 21:
        raise ValueError("scan_points must be at least 21")
    tilts = torch.linspace(0.0, upper, int(scan_points), dtype=_DTYPE)
    step = float(tilts[1] - tilts[0])
    energies = attitude_potential_batch(radius, thickness, tilts, water, quad)["energy"]
    # five-point second derivative on the interior; d^2U/dphi^2 = -dM_r/dphi
    hess = torch.zeros_like(energies)
    hess[2:-2] = (-energies[4:] + 16.0 * energies[3:-1] - 30.0 * energies[2:-2]
                  + 16.0 * energies[1:-3] - energies[:-4]) / (12.0 * step * step)
    hess[0] = (energies[0] - 2.0 * energies[1] + energies[2]) / (step * step)
    hess[1] = hess[0]
    hess[-1] = hess[-2] = hess[-3]
    if float(hess[2]) <= 0.0:
        raise ValueError("Hessian is not positive at the start of the scan")
    index = None
    for i in range(3, int(scan_points) - 2):
        if float(hess[i]) <= 0.0:
            index = i
            break
    if index is None:
        raise ValueError("no Hessian sign change inside the scan window")

    def hessian_at(phi: float) -> float:
        h = 4.0 * step
        energies_local = attitude_potential_batch(
            radius, thickness, [phi - 2 * h, phi - h, phi, phi + h, phi + 2 * h],
            water, quad)["energy"]
        return float((-energies_local[4] + 16.0 * energies_local[3]
                      - 30.0 * energies_local[2] + 16.0 * energies_local[1]
                      - energies_local[0]) / (12.0 * h * h))

    lo = float(tilts[index - 1])
    hi = float(tilts[index])
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if hessian_at(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    phi = 0.5 * (lo + hi)
    return {"capsize_angle": float(phi), "hessian_at_angle": float(hessian_at(phi)),
            "scan_bound": upper, "scan_points": int(scan_points)}

# ---------------------------------------------------------------------------
# 6-DOF channel simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapsizeParameters:
    """Sub-model parameters of the 6-DOF extension (all neutral when locked)."""

    shoulder_submergence: float = 0.0      # tan of the hull shoulder angle
    hull_pressure_height_fraction: float = 0.5   # fraction of draught below WL
    sail_pressure_height_fraction: float = 0.5   # fraction of freeboard above WL
    roll_damping_ratio: float = 0.1
    use_projected_footprint: bool = True
    quad_radial_order: int = 10
    quad_azimuth_order: int = 16

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.shoulder_submergence) < 3.0):
            raise ValueError("shoulder_submergence must lie in [0, 3)")
        for name in ("hull_pressure_height_fraction", "sail_pressure_height_fraction"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1]")
        if not (0.0 <= float(self.roll_damping_ratio) < 1.0):
            raise ValueError("roll_damping_ratio must lie in [0, 1)")
        if int(self.quad_radial_order) < 2 or int(self.quad_azimuth_order) < 4:
            raise ValueError("quadrature orders are too small")


class SixDofFloeSimulation(IceChannelSimulation):
    """Disk-floe DEM with a full attitude (quaternion) per floe.

    ``mode="tilt_locked"`` reproduces the H1 engine step-for-step (the extra
    rotational state stays exactly at rest and ``state_digest`` delegates to the
    parent implementation).  ``mode="six_dof"`` integrates roll and pitch
    together with the tilted hydrostatics.
    """

    def __init__(self, radius: Sequence[float], thickness: Sequence[float], *,
                 mode: str = "six_dof",
                 caps_params: Optional[CapsizeParameters] = None,
                 capsize_angles: Optional[Sequence[float]] = None,
                 water: Optional[WaterEnvironment] = None,
                 contact: Optional[ContactParameters] = None,
                 hull: Optional[HullGeometry] = None,
                 dt: float = 5.0e-3,
                 integrator: str = "euler",
                 dtype: torch.dtype = _DTYPE) -> None:
        super().__init__(radius, thickness, water=water, contact=contact,
                         hull=hull, dt=dt, integrator=integrator, dtype=dtype)
        if mode not in ("six_dof", "tilt_locked"):
            raise ValueError("mode must be 'six_dof' or 'tilt_locked'")
        self.mode = mode
        self.caps = caps_params or CapsizeParameters()
        self.quad = disk_quadrature(self.caps.quad_radial_order,
                                    self.caps.quad_azimuth_order, dtype=dtype)
        self.quat = quat_identity(self.n, dtype=dtype)
        self.omega_body = torch.zeros((self.n, 3), dtype=dtype)
        # principal moments of a solid cylinder about (e1, e2, e3 = symmetry axis)
        self.inertia_body = torch.stack([
            self.mass * (3.0 * self.radius ** 2 + self.thickness ** 2) / 12.0,
            self.mass * (3.0 * self.radius ** 2 + self.thickness ** 2) / 12.0,
            0.5 * self.mass * self.radius ** 2], dim=1)
        self.tilt_angle = torch.zeros(self.n, dtype=dtype)
        self.max_tilt_angle = torch.zeros(self.n, dtype=dtype)
        self.max_roll_rate = torch.zeros(self.n, dtype=dtype)
        self.capsize_flags = torch.zeros(self.n, dtype=torch.bool)
        self.capsize_time = torch.full((self.n,), float("nan"), dtype=dtype)
        self.worst_quat_error = 0.0
        self.worst_balance_error = 0.0
        self.worst_balance_error = 0.0
        self._gm_analytic = torch.tensor(
            [analytic_metacentric_height(float(self.radius[i]),
                                         float(self.thickness[i]),
                                         self.water.ice_density,
                                         self.water.water_density)
             for i in range(self.n)], dtype=dtype)
        if capsize_angles is None and mode == "tilt_locked":
            # the locked mode never rotates, so no capsize threshold is needed
            self._capsize_angles = torch.full((self.n,), 0.5 * math.pi,
                                              dtype=dtype)
        elif capsize_angles is None:
            cache: Dict[float, float] = {}
            values: List[float] = []
            for i in range(self.n):
                key = round(float(self.thickness[i]), 12)
                if key not in cache:
                    cache[key] = float(capsize_angle_from_moment(
                        float(np.mean(self.radius.numpy())), key, self.water,
                        self.quad, scan_points=121)["capsize_angle"])
                values.append(cache[key])
            self._capsize_angles = torch.tensor(values, dtype=dtype)
        else:
            given = torch.as_tensor(list(capsize_angles), dtype=dtype).reshape(-1)
            if given.numel() != self.n:
                raise ValueError("capsize_angles must have one entry per floe")
            if bool((given <= 0.0).any()):
                raise ValueError("capsize_angles must be positive")
            self._capsize_angles = given

    # -- diagnostics -------------------------------------------------------
    @property
    def capsize_angles(self) -> torch.Tensor:
        return self._capsize_angles

    def state_digest(self) -> str:
        """SHA-256 over the state; identical to H1 in the locked mode."""
        if self.mode == "tilt_locked":
            if not self._locked_state_is_exact():
                raise RuntimeError("tilt_locked mode has non-trivial rotational state")
            return super().state_digest()
        hasher = hashlib.sha256()
        for tensor in (self.pos, self.vel, self.yaw, self.omega, self.z, self.vz,
                       self.tilt, self.tilt_rate, self.quat, self.omega_body):
            hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        hasher.update(str(self.steps).encode())
        return hasher.hexdigest()

    def _locked_state_is_exact(self) -> bool:
        identity = quat_identity(self.n, dtype=self.dtype)
        return bool(torch.equal(self.quat, identity)) and bool(
            torch.equal(self.omega_body, torch.zeros_like(self.omega_body)))

    def set_flat_equilibrium(self) -> None:
        """Place every floe at its flat hydrostatic equilibrium height.

        For the flat pose the Archimedes draught is ``d = (rho_i/rho_w) t``, so
        ``z* = t/2 - d`` in closed form; the placement is verified against the
        quadrature hydrostatics by the buoyancy/gravity residual.
        """
        quats = quat_identity(self.n, dtype=self.dtype)
        heights = 0.5 * self.thickness - self.water.density_ratio * self.thickness
        self.z = heights.to(dtype=self.dtype)
        self.vz = torch.zeros_like(self.z)
        self.tilt = torch.zeros_like(self.tilt)
        self.tilt_rate = torch.zeros_like(self.tilt_rate)
        self.quat = quats
        self.omega_body = torch.zeros_like(self.omega_body)
        self.worst_balance_error = self.hydrostatic_balance_error()
        self.initial_balance_error = self.worst_balance_error

    def quaternion_norm_error(self) -> float:
        return float(quat_norm_error(self.quat))

    def hydrostatic_balance_error(self) -> float:
        """Largest ``|F_b + W| / |W|`` over the population."""
        state = submerged_hydrostatics(self.radius, self.thickness, self.quat,
                                       self.z, self.water, self.quad)
        net = (state.buoyancy[:, 2] + state.weight[:, 2]).abs()
        scale = torch.clamp(state.weight[:, 2].abs(), min=1e-30)
        return float((net / scale).max())

    def capsized_fraction(self) -> float:
        return float(self.capsize_flags.to(self.dtype).mean())

    def cleared_width(self, behind_length: Optional[float] = None,
                      lateral_limit: Optional[float] = None) -> float:
        """Cleared channel width, using the projected footprint in 6-DOF mode.

        A tilted disk projects onto the waterplane as an ellipse of half-width
        ``R |cos gamma|`` plus the thickness contribution ``t/2 |sin gamma|``;
        the corridor clearance uses that projected radius so that an edge-on
        floe is treated with its true silhouette.
        """
        if self.mode != "six_dof" or not self.caps.use_projected_footprint:
            return super().cleared_width(behind_length=behind_length,
                                         lateral_limit=lateral_limit)
        gamma = self.tilt_angle
        projected = (self.radius * torch.cos(gamma)
                     + 0.5 * self.thickness * torch.sin(gamma))
        x_ship = self.hull.x_centre(self.time)
        behind = float(behind_length if behind_length is not None
                       else self.hull.length)
        y_ship = float(self.hull.y_position)
        in_corridor = (self.pos[:, 0] >= x_ship - behind) & (self.pos[:, 0] <= x_ship)
        if not bool(in_corridor.any()):
            length, width = getattr(self, "domain", (float("nan"),
                                                     2.0 * float(self.hull.beam)))
            return float(width)
        clearance = (self.pos[in_corridor, 1] - y_ship).abs() - projected[in_corridor]
        half = float(clearance.min().clamp(min=0.0))
        width = 2.0 * half
        if lateral_limit is not None:
            width = min(width, 2.0 * float(lateral_limit))
        return width

    def ice_resistance(self, t: Optional[float] = None) -> float:
        """Instantaneous hull resistance [N] (x reaction of the hull contacts)."""
        time = self.time if t is None else float(t)
        hull_force, _ = self._hull_force(time, self.pos)
        return float(hull_force[:, 0].sum())

    # -- force assembly ----------------------------------------------------
    def net_forces_6dof(self, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """World force ``(N, 3)`` and moment about the centre of mass ``(N, 3)``."""
        water = self.water
        n = self.n
        force = torch.zeros((n, 3), dtype=self.dtype)
        moment = torch.zeros((n, 3), dtype=self.dtype)

        # --- floe-floe contacts, applied at the waterline height -----------
        idx, f_pair, arm_i, arm_j, _ = self._floe_floe_contacts()
        if idx.numel():
            i, j = idx[:, 0], idx[:, 1]
            force.index_add_(0, i, torch.stack(
                [f_pair[:, 0], f_pair[:, 1], torch.zeros_like(f_pair[:, 0])], dim=1))
            force.index_add_(0, j, torch.stack(
                [-f_pair[:, 0], -f_pair[:, 1], torch.zeros_like(f_pair[:, 0])], dim=1))
            lever_i = -self.z[i]
            lever_j = -self.z[j]
            moment[:, 2].index_add_(0, i, arm_i[:, 0] * f_pair[:, 1]
                                    - arm_i[:, 1] * f_pair[:, 0])
            moment[:, 2].index_add_(0, j, -arm_j[:, 0] * f_pair[:, 1]
                                    + arm_j[:, 1] * f_pair[:, 0])
            moment[:, 0].index_add_(0, i, -lever_i * f_pair[:, 1])
            moment[:, 1].index_add_(0, i, lever_i * f_pair[:, 0])
            moment[:, 0].index_add_(0, j, lever_j * f_pair[:, 1])
            moment[:, 1].index_add_(0, j, -lever_j * f_pair[:, 0])

        # --- hull contact --------------------------------------------------
        hull_force, hull_torque = self._hull_force(t, self.pos)
        force[:, :2] = force[:, :2] + hull_force
        draught = submerged_draught(self.z, self.thickness)
        level_hull = (-float(self.caps.hull_pressure_height_fraction) * draught
                      - self.z)
        moment[:, 2] = moment[:, 2] + hull_torque
        moment[:, 0] = moment[:, 0] - level_hull * hull_force[:, 1]
        moment[:, 1] = moment[:, 1] + level_hull * hull_force[:, 0]

        # --- environmental drag -------------------------------------------
        freeboard = self.thickness - draught
        sail = 2.0 * self.radius * torch.clamp(freeboard, min=0.0) * water.sail_area_fraction
        side = 2.0 * self.radius * draught
        wind_rel = self.vel - torch.tensor(list(water.wind_velocity), dtype=self.dtype)
        cur_rel = self.vel - torch.tensor(list(water.current_velocity), dtype=self.dtype)
        wind_speed = torch.linalg.norm(wind_rel, dim=1, keepdim=True)
        cur_speed = torch.linalg.norm(cur_rel, dim=1, keepdim=True)
        wind_force = (-0.5 * water.air_density * water.wind_drag_coefficient
                      * sail[:, None] * wind_speed * wind_rel)
        current_force = (-0.5 * water.water_density * water.water_drag_coefficient
                         * side[:, None] * cur_speed * cur_rel)
        force[:, :2] = force[:, :2] + wind_force + current_force
        sail_level = (float(self.caps.sail_pressure_height_fraction)
                      * torch.clamp(freeboard, min=0.0)
                      - (1.0 - water.density_ratio) * 0.5 * self.thickness)
        current_level = -0.5 * draught - self.z
        moment[:, 0] = moment[:, 0] - sail_level * wind_force[:, 1] \
            - current_level * current_force[:, 1]
        moment[:, 1] = moment[:, 1] + sail_level * wind_force[:, 0] \
            + current_level * current_force[:, 0]

        # --- tilted hydrostatics ------------------------------------------
        state = submerged_hydrostatics(self.radius, self.thickness, self.quat,
                                       self.z, self.water, self.quad)
        force[:, 2] = state.buoyancy[:, 2] + state.weight[:, 2]
        moment = moment + state.moment

        # --- optional hull-shoulder submersion ----------------------------
        if self.caps.shoulder_submergence > 0.0:
            hull_norm = torch.linalg.norm(hull_force, dim=1)
            down = -float(self.caps.shoulder_submergence) * hull_norm
            force[:, 2] = force[:, 2] + down
            direction = torch.where(hull_norm.unsqueeze(1) > 0.0,
                                    hull_force / torch.clamp(hull_norm.unsqueeze(1),
                                                             min=1e-30),
                                    torch.zeros_like(hull_force))
            rim = torch.cat([-direction * (0.5 * self.radius).unsqueeze(1),
                             torch.zeros(n, 1, dtype=self.dtype)], dim=1)
            rim_force = torch.stack([torch.zeros(n, dtype=self.dtype),
                                     torch.zeros(n, dtype=self.dtype), down], dim=1)
            moment = moment + torch.linalg.cross(rim, rim_force)

        # --- damping ------------------------------------------------------
        k_heave = water.water_density * water.gravity * state.projected_area
        c_heave = 2.0 * water.heave_damping_ratio * torch.sqrt(self.mass * k_heave)
        force[:, 2] = force[:, 2] - c_heave * self.vz

        volume = torch.clamp(state.volume, min=1e-30)
        k_roll = (water.water_density * water.gravity * volume
                  * torch.clamp(self._gm_analytic, min=0.0))
        c_roll = 2.0 * float(self.caps.roll_damping_ratio) * torch.sqrt(
            torch.clamp(k_roll * self.inertia_body[:, 0], min=0.0))
        omega_world = torch.einsum('nij,nj->ni', quat_to_matrix(self.quat),
                                   self.omega_body)
        moment[:, 0] = moment[:, 0] - c_roll * omega_world[:, 0]
        moment[:, 1] = moment[:, 1] - c_roll * omega_world[:, 1]
        return force, moment

    # -- integration -------------------------------------------------------
    def step(self) -> None:
        if self.mode == "tilt_locked":
            super().step()
            return

        dt = self.dt
        force, moment = self.net_forces_6dof(self.time)
        rmat = quat_to_matrix(self.quat)

        acc = force / self.mass.unsqueeze(1)
        self.vel = self.vel + dt * acc[:, :2]
        self.vz = self.vz + dt * acc[:, 2]
        self.pos = self.pos + dt * self.vel
        self.yaw = self.yaw + dt * self.omega
        self.z = self.z + dt * self.vz
        self.time += dt

        moment_body = torch.einsum('nji,nj->ni', rmat, moment)
        inertia = self.inertia_body
        gyro = torch.linalg.cross(self.omega_body, inertia * self.omega_body)
        omega_dot = (moment_body - gyro) / inertia
        self.omega_body = self.omega_body + dt * omega_dot
        self.quat = integrate_quaternion(self.quat, self.omega_body, dt)
        self.worst_quat_error = max(self.worst_quat_error, self.quaternion_norm_error())

        self.tilt_angle = quat_tilt_angle(self.quat)
        self.max_tilt_angle = torch.maximum(self.max_tilt_angle, self.tilt_angle)
        roll_speed = torch.linalg.norm(self.omega_body[:, :2], dim=1)
        self.max_roll_rate = torch.maximum(self.max_roll_rate, roll_speed)
        newly = (self.tilt_angle > self._capsize_angles) & (~self.capsize_flags)
        if bool(newly.any()):
            self.capsize_time = torch.where(newly,
                                            torch.full_like(self.capsize_time, self.time),
                                            self.capsize_time)
        self.capsize_flags = self.capsize_flags | (self.tilt_angle > self._capsize_angles)
        self.steps += 1

    # -- transit driver ----------------------------------------------------
    def run_transit(self, duration: float, record_every: int = 10,
                    settle_steps: int = 0) -> Dict[str, object]:
        """Advance ``duration`` seconds (or ``settle_steps`` if given) and report.

        Returns the cleared width, hull resistance, capsize bookkeeping and the
        numerical invariants used by the acceptance criteria.
        """
        if self.mode == "tilt_locked":
            steps = int(settle_steps) if settle_steps else int(round(duration / self.dt))
        else:
            steps = int(settle_steps) if settle_steps else int(round(duration / self.dt))
        if steps <= 0:
            raise ValueError("duration/duration steps must be positive")
        record_every = max(1, int(record_every))
        width_series: List[float] = []
        resistance_series: List[float] = []
        tilt_series: List[float] = []
        time_series: List[float] = []
        for step in range(steps):
            self.step()
            if (step + 1) % record_every == 0:
                width_series.append(self.cleared_width())
                time_series.append(self.time)
                tilt_series.append(float(self.max_tilt_angle.max()))
                resistance_series.append(self.ice_resistance())
        tail = max(1, len(width_series) // 4)
        return {
            "mode": self.mode,
            "steps": self.steps,
            "time": self.time,
            "cleared_width_final": float(width_series[-1]) if width_series else float("nan"),
            "cleared_width_tail_mean": float(np.mean(width_series[-tail:]))
            if width_series else float("nan"),
            "cleared_width_min": float(np.min(width_series)) if width_series else float("nan"),
            "resistance_tail_mean": float(np.mean(resistance_series[-tail:]))
            if resistance_series else float("nan"),
            "capsized_fraction": self.capsized_fraction(),
            "max_tilt_deg": float(torch.rad2deg(self.max_tilt_angle).max()),
            "tilt_p95_deg": float(torch.rad2deg(self.max_tilt_angle).quantile(0.95)),
            "max_roll_rate": float(self.max_roll_rate.max()),
            "capsize_angle_deg_mean": float(torch.rad2deg(self._capsize_angles).mean()),
            "quaternion_norm_error": self.quaternion_norm_error(),
            "worst_quaternion_norm_error": float(self.worst_quat_error),
            "worst_hydrostatic_balance_error": float(self.worst_balance_error),
            "initial_hydrostatic_balance_error": float(
                getattr(self, "initial_balance_error", self.worst_balance_error)),
            "state_digest": self.state_digest(),
            "time_series": time_series,
            "cleared_width_series": width_series,
            "max_tilt_series_deg": tilt_series,
        }