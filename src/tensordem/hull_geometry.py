"""Circular-bow hull geometry: closed-form arc contact and polyline error bounds.

Motivation
----------
``tensordem.hull_contact.polyline_contact_forces`` discretises a ship bow as a
polyline and uses the unsigned distance to the nearest chord as the penalty
overlap.  A chord is a *secant* of the true bow arc, so the distance measured
from a chord is always larger than the distance from the arc: the polyline hull
systematically **under-estimates** penetration, and therefore ice load.  This
module provides

1. ``CircularArc`` / ``arc_hull_contact_forces`` — an exact piecewise-circular
   hull kernel.  Distance to a circular arc is analytic (``|r - R|`` inside the
   angular span, endpoint distance outside), so a bow built from arc pieces has
   no geometric discretisation error at all and keeps the same SI conventions
   and penalty law as the polyline kernel.
2. ``sagitta`` / ``polyline_error_bounds`` — closed-form bounds for the secant
   discretisation error.  With ``R`` the bow radius and ``dtheta`` the angle
   subtended by one segment, the mid-chord sagitta is

       s = R * (1 - cos(dtheta / 2)) = 2 * R * sin(dtheta / 4)**2 .

   For a disk whose centre lies on the chord normal (the worst case), the chord
   penetration is exactly ``delta_chord = delta_arc - s``, independent of the
   particle radius and of the penetration depth.  The consequences are exact,
   not asymptotic:

   * the polyline contact force is ``F_chord = k * (delta - s)`` instead of
     ``k * delta``, i.e. a relative error of ``-s / delta``;
   * for ``delta <= s`` the chord sees **no contact at all** (a dropout), so
     shallow penalty contacts can be missed entirely;
   * halving ``dtheta`` divides the error by four (second order in the segment
     angle, first order in the segment count).

   The direction error is bounded by ``dtheta / 2`` and vanishes on the chord
   normal and at the vertices, giving a sawtooth ("washboard") modulation of
   the contact force as ice slides along a discretised bow.
3. ``sagitta_corrected_polyline`` — the cheapest mitigation: scale the vertices
   outwards so that each chord sits at the arc apothem plus one sagitta.  It
   removes the mid-chord bias but over-shoots near the vertices; the exact arc
   kernel is still required for <3% error at shallow overlap.

Everything here is dimensionless-safe, uses SI units (m, m/s, N/m, N*s/m, N)
and float64 by default.  No assumption is made about an underlying ice model:
this is contact geometry only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Closed-form polyline (secant) error bounds
# ---------------------------------------------------------------------------

def sagitta(radius: float, span: float, n_segments: int) -> float:
    """Mid-chord sagitta of a circle approximated by ``n_segments`` chords.

    Args:
        radius: arc radius (m), positive.
        span: total subtended angle (rad), positive and <= 2*pi.
        n_segments: number of equal chords, >= 1.

    Returns:
        s = R * (1 - cos(dtheta/2)) in metres, with dtheta = span / n_segments.
    """
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if not math.isfinite(span) or span <= 0.0 or span > _TWO_PI + 1e-12:
        raise ValueError("span must be finite, in (0, 2*pi]")
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    dtheta = span / n_segments
    return radius * (1.0 - math.cos(0.5 * dtheta))


def polyline_error_bounds(
    radius: float,
    span: float,
    n_segments: int,
    penetration: float,
) -> dict[str, float]:
    """Closed-form geometry error of a secant bow at a given penetration.

    All quantities are relative to the exact circular-arc contact.

    Args:
        radius: bow radius R (m).
        span: subtended angle of the discretised bow (rad).
        n_segments: number of chords.
        penetration: exact arc penetration delta (m), positive.

    Returns:
        dict with ``segment_angle`` (rad), ``sagitta`` (m), ``course_ratio``
        (s / R, dimensionless measure of how coarse the discretisation is),
        ``force_ratio`` (chord/arc force at the chord normal, clipped at 0),
        ``relative_force_error`` (``force_ratio - 1``, negative = under-load),
        ``dropout`` (1.0 when ``penetration <= sagitta``),
        ``direction_error_bound`` (rad, = dtheta/2) and
        ``max_error_free_penetration`` (penetration needed for a given
        tolerance is a separate query, see ``segments_for_tolerance``).
    """
    if not math.isfinite(penetration) or penetration < 0.0:
        raise ValueError("penetration must be finite and nonnegative")
    s = sagitta(radius, span, n_segments)
    dtheta = span / n_segments
    if penetration <= 0.0:
        force_ratio = 0.0
    else:
        force_ratio = max(0.0, (penetration - s) / penetration)
    return {
        "segment_angle": dtheta,
        "sagitta": s,
        "course_ratio": s / radius,
        "force_ratio": force_ratio,
        "relative_force_error": force_ratio - 1.0,
        "dropout": 1.0 if penetration <= s else 0.0,
        "direction_error_bound": 0.5 * dtheta,
        "chord_penetration": max(0.0, penetration - s),
    }


def segments_for_tolerance(
    radius: float,
    span: float,
    penetration: float,
    tolerance: float,
) -> int:
    """Smallest chord count whose worst-case force error is within tolerance.

    Inverts ``s = delta * tolerance`` exactly:
    ``n >= span / (2 * acos(1 - delta*tolerance/R))``.

    Args:
        radius: bow radius R (m).
        span: subtended angle (rad).
        penetration: worst-case (smallest) working penetration delta (m).
        tolerance: allowed relative force error, e.g. 0.03 for 3%.
    """
    if not (math.isfinite(tolerance) and 0.0 < tolerance < 1.0):
        raise ValueError("tolerance must be in (0, 1)")
    if not (math.isfinite(penetration) and penetration > 0.0):
        raise ValueError("penetration must be finite and positive")
    target = penetration * tolerance
    if target >= radius:
        return 1
    half_angle = math.acos(1.0 - target / radius)
    if half_angle <= 0.0:  # target below double precision resolution
        return int(math.ceil(span / (2.0 * math.acos(1.0 - target / radius))))
    return int(math.ceil(span / (2.0 * half_angle)))


# ---------------------------------------------------------------------------
# Polyline construction helpers
# ---------------------------------------------------------------------------

def arc_polyline(
    radius: float,
    span: float,
    n_segments: int,
    *,
    start_angle: float = -0.5 * math.pi,
    center: Sequence[float] = (0.0, 0.0),
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Vertices of an ``n_segments``-chord polyline inscribed in a circular arc.

    The first and last vertices lie exactly on the arc; the intermediate
    vertices are the chord joints.  The profile is produced counter-clockwise
    from ``start_angle``.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    sagitta(radius, span, n_segments)  # argument validation
    angles = torch.linspace(
        start_angle, start_angle + span, n_segments + 1, dtype=dtype
    )
    cx, cy = float(center[0]), float(center[1])
    return torch.stack(
        (radius * torch.cos(angles) + cx, radius * torch.sin(angles) + cy), dim=1
    )


def sagitta_corrected_polyline(
    radius: float,
    span: float,
    n_segments: int,
    *,
    start_angle: float = -0.5 * math.pi,
    center: Sequence[float] = (0.0, 0.0),
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Polyline whose chord apothem equals ``R*cos(dtheta/2) + s``.

    Vertices are pushed radially outwards to ``R' = (R cos(dtheta/2) + s) /
    cos(dtheta/2)``, which makes the mid-chord penetration exact.  The vertices
    then protrude ``s / cos(dtheta/2)`` beyond the true arc, so this is a
    partial mitigation only (see ``docs/dem2d_hull_polyline_geometry_audit.md``).
    """
    s = sagitta(radius, span, n_segments)
    dtheta = span / n_segments
    radius_eff = (radius * math.cos(0.5 * dtheta) + s) / math.cos(0.5 * dtheta)
    return arc_polyline(
        radius_eff, span, n_segments,
        start_angle=start_angle, center=center, dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Exact circular-arc hull kernel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CircularArc:
    """One convex hull arc piece.

    Attributes:
        center: (cx, cy) arc centre (m).
        radius: arc radius R (m), positive.
        start_angle: angle of the first endpoint (rad), measured from +x.
        span: subtended angle (rad), positive, traversed counter-clockwise.
    """

    center: tuple[float, float]
    radius: float
    start_angle: float
    span: float

    def __post_init__(self) -> None:
        for name, value in (
            ("center x", self.center[0]), ("center y", self.center[1]),
            ("radius", self.radius), ("start_angle", self.start_angle),
            ("span", self.span),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.span <= 0.0 or self.span > _TWO_PI + 1e-12:
            raise ValueError("span must be in (0, 2*pi]")

    def endpoint(self, which: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Endpoint coordinate (``which`` either 0 or 1) as a length-2 tensor."""
        if which not in (0, 1):
            raise ValueError("which must be 0 or 1")
        angle = self.start_angle + (self.span if which else 0.0)
        return torch.tensor(
            [self.center[0] + self.radius * math.cos(angle),
             self.center[1] + self.radius * math.sin(angle)],
            dtype=dtype,
        )


def arc_closest_points(
    positions: torch.Tensor,
    arc: CircularArc,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unsigned distance and closest point from each disk centre to one arc.

    Returns:
        (distance, closest_point, radial_direction) where ``radial_direction``
        is the outward unit radial vector of the arc at the particle angle.  It
        is used as the contact normal fallback when the centre sits exactly on
        the arc (zero distance).
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    dtype = positions.dtype
    center = torch.tensor(arc.center, dtype=dtype, device=positions.device)
    rel = positions - center
    radius = torch.linalg.vector_norm(rel, dim=1)
    eps = torch.finfo(dtype).tiny
    radial = rel / radius.clamp_min(eps)[:, None]
    beta = torch.atan2(rel[:, 1], rel[:, 0])
    offset = torch.remainder(beta - arc.start_angle, _TWO_PI)
    inside = offset <= arc.span

    closest_arc = center + arc.radius * radial
    start = arc.endpoint(0, dtype=dtype).to(positions.device)
    end = arc.endpoint(1, dtype=dtype).to(positions.device)
    to_start = torch.linalg.vector_norm(positions - start, dim=1)
    to_end = torch.linalg.vector_norm(positions - end, dim=1)
    at_start = to_start <= to_end
    endpoint_distance = torch.where(at_start, to_start, to_end)
    endpoint = torch.where(at_start[:, None], start[None, :], end[None, :])

    distance_arc = (radius - arc.radius).abs()
    distance = torch.where(inside, distance_arc, endpoint_distance)
    closest = torch.where(inside[:, None], closest_arc, endpoint)
    return distance, closest, radial


def arc_hull_contact_forces(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    arcs: Sequence[CircularArc],
    hull_velocity: torch.Tensor | Sequence[float],
    *,
    particle_radius: float,
    stiffness: float,
    hull_radius: float = 0.0,
    damping: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalty contact between DEM disks and a piecewise-circular hull.

    Same SI conventions and penalty law as
    ``tensordem.hull_contact.polyline_contact_forces``: the force on each disk
    is ``(k * overlap - c * normal_speed) * n`` clamped at zero, with ``n``
    pointing from the hull towards the disk and ``overlap = r_p + hull_radius -
    distance``.  The only difference is that the distance is measured to exact
    circular arcs, so no secant sagitta bias exists.

    Returns:
        (forces (N,2), hull_reaction (2,)) with the reaction equal and opposite.
    """
    if not arcs:
        raise ValueError("at least one CircularArc is required")
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if not isinstance(velocities, torch.Tensor) or velocities.shape != positions.shape:
        raise ValueError("velocities must have the same shape as positions")
    for name, value in (("particle_radius", particle_radius), ("stiffness", stiffness)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("hull_radius", hull_radius), ("damping", damping)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    dtype, device = positions.dtype, positions.device
    hv = torch.as_tensor(hull_velocity, dtype=dtype, device=device).reshape(2)
    if not bool(torch.isfinite(positions).all() and torch.isfinite(velocities).all()
                and torch.isfinite(hv).all()):
        raise ValueError("inputs must contain only finite values")

    best_distance = torch.full(
        (positions.shape[0],), float("inf"), dtype=dtype, device=device
    )
    best_delta = torch.zeros_like(positions)
    best_radial = torch.zeros_like(positions)
    for arc in arcs:
        distance, closest, radial = arc_closest_points(positions, arc)
        closer = distance < best_distance
        best_distance = torch.where(closer, distance, best_distance)
        best_delta = torch.where(closer[:, None], positions - closest, best_delta)
        best_radial = torch.where(closer[:, None], radial, best_radial)

    eps = torch.finfo(dtype).eps
    distance_safe = best_distance.clamp_min(eps)
    degenerate = best_distance <= eps
    normal = torch.where(
        degenerate[:, None], best_radial, best_delta / distance_safe[:, None]
    )
    overlap = (particle_radius + hull_radius - best_distance).clamp_min(0.0)
    normal_speed = ((velocities - hv) * normal).sum(dim=1)
    magnitude = (stiffness * overlap - damping * normal_speed).clamp_min(0.0)
    magnitude = torch.where(overlap > 0.0, magnitude, torch.zeros_like(magnitude))
    forces = magnitude[:, None] * normal
    return forces, -forces.sum(dim=0)


def analytic_arc_penetration(
    radius: float,
    particle_radius: float,
    radial_position: float,
) -> float:
    """Closed-form penetration of a disk against a bow circle (m).

    ``delta = r_p + R - r`` for a disk centre at radial distance ``r`` from the
    bow centre, clamped at zero.  This is the independent reference used by the
    audit script; it does not call the kernel.
    """
    if radial_position < 0.0:
        raise ValueError("radial_position must be nonnegative")
    return max(0.0, particle_radius + radius - radial_position)