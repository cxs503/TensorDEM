"""Spherical-particle buoyancy and wet resistance for the 3-D bonded-sphere DEM.

The 2-D module :mod:`tensordem.buoyancy_wet` treats an extruded disk: its
submerged *area* fraction comes from circular-segment geometry and is then
multiplied by an extrusion thickness.  A 3-D bonded-sphere particle is a
**sphere**, whose submerged *volume* fraction follows the spherical-cap law.
The two geometry kernels are not interchangeable.  At ``h = +R/2`` (a
quarter-depth cap) the disk area fraction is 0.19550 against a sphere volume
fraction of 0.15625, a 25.1 % relative difference, so reusing the 2-D kernel for
a sphere fails the 15 % accuracy gate exactly where shallowly immersed particles
live.  The error is asymmetric: at ``h = -R/2`` the two kernels agree to 4.7 %.
Propagated to the free-floating draft of an ice sphere (rho_i/rho_w = 0.8946)
the 2-D kernel gives a draft 4.9 % too deep.  This module supplies the exact 3-D
kernel and the closed-form anchors needed to verify it.

Geometry (SI units)
-------------------
Let ``h`` be the particle-centre elevation above the free surface (``h > 0``
means the centre is out of the water) and ``R`` the particle radius.

The sphere centre sits at elevation ``h`` above the free surface, so the
sphere occupies ``z in [h-R, h+R]`` and the submerged part is ``z <= 0``.
With ``q = z - h`` (offset from the centre) the water surface is ``q = -h`` and
the submerged volume is the cap ``q in [-R, -h]``::

    V_sub(h) = int_{-R}^{-h} pi (R^2 - q^2) dq
             = pi * (2 R^3/3 - R^2 h + h^3/3),            -R <= h <= R
             = 0,                                          h >= R
             = 4/3 pi R^3,                                 h <= -R

which factors identically as ``V_sub = pi (R - h)^2 (2R + h) / 3``.  The
volume fraction is ``f = V_sub / (4/3 pi R^3)``, i.e.::

    f(h) = (R - h)^2 (2R + h) / (4 R^3)

Its derivative is the waterplane-area identity (``f`` decreases with ``h``)

    -dV_sub/dh = pi (R^2 - h^2) = A_wp(h)  > 0

so the hydrostatic restoring stiffness of a floating sphere is

    K = -dF_b/dh = rho_w * g * A_wp(h) = rho_w * g * pi (R^2 - h^2)  [N/m]

(positive: a sphere pushed down feels a larger upward force).

Archimedes equilibrium
----------------------
A free-floating particle of density ``rho_p`` in water of density ``rho_w``
satisfies ``f(h_eq) = rho_p / rho_w``.  With ``r = rho_p / rho_w`` and
``u = h_eq / R`` this is the cubic ``u^3 - 3u + 2 - 4r = 0``.  Since
``d/du (u^3 - 3u) = 3u^2 - 3 <= 0`` on ``[-1, 1]`` the root there is unique for
``0 <= r <= 1``, which makes a bracketed bisection exact and branch-free.  The
independent polynomial root (see the tests) is used as a cross-check.

All quantities use SI units.  Forces are returned with the same device/dtype as
the input tensors.
"""
from __future__ import annotations

import math

import torch

from .buoyancy_wet import WetCouplingConfig

__all__ = [
    "submerged_volume_sphere",
    "submerged_fraction_sphere",
    "submerged_fraction_sphere_tensor",
    "waterplane_area_sphere",
    "hydrostatic_stiffness_sphere",
    "equilibrium_centre_height",
    "equilibrium_draft",
    "compute_buoyancy_force_3d",
    "compute_wet_drag_force_3d",
    "compute_wet_forces_3d",
]


# ---------------------------------------------------------------------------
# Geometry kernel (scalars, CPU) — the verification anchor
# ---------------------------------------------------------------------------

def submerged_volume_sphere(h: float, R: float) -> float:
    """Submerged volume of a sphere of radius *R* whose centre is *h* above the
    free surface (``h < 0`` = centre below the surface).  SI, m^3.

    Integrated cap form ``pi (2 R^3/3 - R^2 h + h^3/3)``: the submerged cap is
    ``q in [-R, -h]`` with ``q`` measured from the sphere centre.
    """
    if R <= 0.0 or not math.isfinite(R):
        raise ValueError("radius must be finite and positive")
    if not math.isfinite(h):
        raise ValueError("centre height must be finite")
    if h >= R:
        return 0.0
    if h <= -R:
        return 4.0 / 3.0 * math.pi * R ** 3
    return math.pi * (2.0 * R ** 3 / 3.0 - R * R * h + h ** 3 / 3.0)


def submerged_fraction_sphere(h: float, R: float) -> float:
    """Submerged volume fraction ``f = V_sub / (4/3 pi R^3)`` in ``[0, 1]``."""
    return submerged_volume_sphere(h, R) / (4.0 / 3.0 * math.pi * R ** 3)


def submerged_fraction_sphere_factored(h: float, R: float) -> float:
    """Equivalent factored form ``(R-h)^2 (2R+h) / (4 R^3)``.

    Kept as an independent algebraic route: the tests cross-check it against
    :func:`submerged_fraction_sphere`, so an algebra slip in either one shows up.
    """
    if h >= R:
        return 0.0
    if h <= -R:
        return 1.0
    return (R - h) ** 2 * (2.0 * R + h) / (4.0 * R ** 3)


def waterplane_area_sphere(h: float, R: float) -> float:
    """Waterplane (intersection) area ``pi (R^2 - h^2)``, m^2; 0 outside [-R, R]."""
    if abs(h) >= R:
        return 0.0
    return math.pi * (R * R - h * h)


def hydrostatic_stiffness_sphere(
    h: float, R: float, water_density: float = 1025.0, gravity: float = 9.81
) -> float:
    """Analytic restoring stiffness ``K = -dF_b/dh = rho_w g A_wp(h)``, N/m."""
    return water_density * gravity * waterplane_area_sphere(h, R)


def submerged_fraction_sphere_tensor(
    y_centre: torch.Tensor, radius: float, water_level: float = 0.0
) -> torch.Tensor:
    """Vectorised spherical-cap submerged volume fraction for a batch of centres.

    Args:
        y_centre: (N,) vertical particle-centre coordinates.
        radius:   particle radius (scalar, m).
        water_level: free-surface elevation (scalar, m).

    Returns:
        (N,) volume fractions in ``[0, 1]``.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    h = y_centre - water_level
    h_clamped = torch.clamp(h, -radius, radius)
    f = (radius - h_clamped) ** 2 * (2.0 * radius + h_clamped) / (4.0 * radius ** 3)
    f = torch.clamp(f, 0.0, 1.0)
    f = torch.where(h >= radius, torch.zeros_like(f), f)
    f = torch.where(h <= -radius, torch.ones_like(f), f)
    return f


# ---------------------------------------------------------------------------
# Archimedes equilibrium
# ---------------------------------------------------------------------------

def equilibrium_centre_height(
    radius: float,
    particle_density: float,
    water_density: float = 1025.0,
    tol: float = 1e-14,
) -> float:
    """Centre elevation ``h_eq`` (m) of a freely floating sphere.

    Solves ``f(h_eq) = rho_p / rho_w`` by bracketed bisection on the unique
    root in ``[-R, R]``.  ``h_eq < 0`` means the centre sits below the surface,
    so the draft is ``R - h_eq``.
    """
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("radius must be finite and positive")
    if particle_density <= 0.0 or water_density <= 0.0:
        raise ValueError("densities must be positive")
    r = particle_density / water_density
    if r >= 1.0:
        # Denser than water: sinks; equilibrium does not exist in [-R, R].
        raise ValueError("particle_density must be below water_density to float")
    lo, hi = -radius, radius
    # f(lo) = 1 > r >= f(hi) = 0 — monotone decreasing, unique root in [-R, R].
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi or hi - lo <= tol * radius:
            break
        if submerged_fraction_sphere(mid, radius) > r:
            lo = mid          # centre is too deep: root lies above mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def equilibrium_draft(
    radius: float,
    particle_density: float,
    water_density: float = 1025.0,
    tol: float = 1e-14,
) -> float:
    """Draft ``d = R - h_eq`` (m below the free surface of the lowest point)."""
    return radius - equilibrium_centre_height(
        radius, particle_density, water_density, tol
    )


# ---------------------------------------------------------------------------
# Force evaluation (tensor, any device/dtype) — dropdown for IceDEM3D
# ---------------------------------------------------------------------------

def compute_buoyancy_force_3d(
    positions: torch.Tensor,
    radius: float,
    cfg: WetCouplingConfig,
) -> torch.Tensor:
    """Buoyancy force ``F_b = rho_w g V_sub(h)`` in +y for each sphere.

    Args:
        positions: (N, 3) particle centres ``(x, y, z)``.
        radius:    sphere radius, m.
        cfg:       :class:`~tensordem.buoyancy_wet.WetCouplingConfig`; the
                   ``thickness`` concept of the 2-D model does not apply here.

    Returns:
        (N, 3) force tensor, zero when ``cfg.enabled`` is False.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if not cfg.enabled:
        return torch.zeros_like(positions)
    f_sub = submerged_fraction_sphere_tensor(positions[:, 1], radius, cfg.water_level)
    vol_sub = 4.0 / 3.0 * math.pi * radius ** 3 * f_sub
    force = torch.zeros_like(positions)
    force[:, 1] = cfg.water_density * cfg.gravity * vol_sub
    return force


def compute_wet_drag_force_3d(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    radius: float,
    cfg: WetCouplingConfig,
) -> torch.Tensor:
    """Linear wet damping ``F = -C_wet f_sub v`` for each sphere, shape (N, 3)."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions shape")
    if not cfg.enabled or cfg.wet_drag_coeff <= 0.0:
        return torch.zeros_like(velocities)
    f_sub = submerged_fraction_sphere_tensor(positions[:, 1], radius, cfg.water_level)
    return -cfg.wet_drag_coeff * f_sub[:, None] * velocities


def compute_wet_forces_3d(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    radius: float,
    cfg: WetCouplingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combined ``(buoyancy, wet_drag)`` for 3-D spheres, each (N, 3)."""
    return (
        compute_buoyancy_force_3d(positions, radius, cfg),
        compute_wet_drag_force_3d(positions, velocities, radius, cfg),
    )