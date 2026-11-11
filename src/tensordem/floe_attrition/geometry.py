"""Exact Wadell roundness geometry for DEM floe shapes (node p472w2-N05).

Why this module exists
----------------------
Broken-ice fields ("brash ice", frozen rubble) are made of *angular* floes that
get **rounded** by repeated hull/ice sliding.  Roundness is therefore a
first-class DEM state variable, not a post-processing descriptor.  TensorDEM had
shape descriptors only as *measurements* of a vertex cloud
(:mod:`tensordem.polyhedra_shapes`: ``sphericity = pi^(1/3)(6V)^(2/3)/A``) and
no shape family in which (a) roundness is an exact analytic function of a single
wear state variable and (b) that state variable can be driven by a wear law.

This module supplies that family:

* 2D -- ``RoundedPolygon``: a regular ``n``-gon of edge ``a`` whose corners are
  filleted with radius ``r`` (the DEM-representable idealisation of a blunted
  floe).  Closed forms for area, perimeter, **Wadell roundness**
  ``R = 4 pi A / P^2``, the corner-radius ratio ``r / R_ins``, the circumradius
  ``R_c(r)``, and the geometric interlocking index ``I = R_c / R_ins - 1``.
* 3D -- ``RoundedCube``: the Minkowski sum ``cube(a - 2r) (+) ball(r)``, i.e. a
  cube with all edges and corners rounded by radius ``r``.  Volume and surface
  area follow from Steiner's formula in closed form, hence a closed-form
  sphericity ``Psi(r)`` and a 3D Wadell roundness ``r / R_ins``.

All families are *single-parameter* (``r``) and analytic, so a wear law can be
integrated by inverting ``worn_volume(r)`` exactly -- the property the attrition
chain (:mod:`tensordem.floe_attrition.wear`) is built on.

Falsifiable identities (verified in ``tests/test_floe_attrition.py``)

1. *irreducible limit*: at ``r = r_max`` the 2D family is exactly a circle and
   ``4 pi A / P^2 == 1``; in 3D ``Psi == 1`` and ``V == 4/3 pi r^3``.
2. closed forms agree with an **independent** boundary integral (Green's theorem
   + composite trapezoid on the arcs) to ``<= 1e-12`` relative.
3. 3D sphericity closed form agrees with the existing vertex-cloud descriptor
   ``polyhedra_shapes.compute_shape_descriptor`` (ConvexHull based) to
   ``<= 1e-12`` for the Platonic solids.
4. the DEM-ready vertex discretisation converges to the analytic roundness as
   ``O(m^-2)`` in the arc-point count ``m``.

Conventions: SI, angles in radians internally, polygons centred at the origin,
one orientation (first corner bisector on the +x axis).  ``worn_area`` /
``worn_volume`` are the material removed relative to the *sharp* shape at
``r = 0``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "wadell_roundness_2d",
    "sphericity_3d",
    "wadell_roundness_3d",
    "RoundedPolygon",
    "RoundedCube",
    "rounded_polygon_vertices",
    "polygon_metrics_boundary_integral",
    "fillet_radius_for_roundness",
]


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def wadell_roundness_2d(area: float, perimeter: float) -> float:
    """Wadell (1932) 2D roundness ``R = 4 pi A / P^2``.

    ``R = 1`` for a circle and ``R < 1`` for every other convex shape
    (isoperimetric inequality); a smooth square gives ``pi/4``.
    """
    area = _positive("area", area)
    perimeter = _positive("perimeter", perimeter)
    return 4.0 * math.pi * area / (perimeter * perimeter)


def sphericity_3d(volume: float, surface_area: float) -> float:
    """Wadell sphericity ``Psi = pi^(1/3) (6V)^(2/3) / A`` (1 for a sphere)."""
    volume = _positive("volume", volume)
    surface_area = _positive("surface_area", surface_area)
    return math.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / surface_area


def wadell_roundness_3d(corner_radius: float, inscribed_radius: float) -> float:
    """Wadell 3D roundness: corner radius over the inscribed radius.

    ``R_3d = r_corner / R_inscribed`` in ``[0, 1]``; 1 for a sphere, 0 for a shape
    with a sharp corner.  For a set of corners the classical Wadell definition is
    the mean ``sum(r_i) / (N R_ins)``; the ``RoundedCube`` family has identical
    corners, so the two coincide.
    """
    corner_radius = float(corner_radius)
    inscribed_radius = _positive("inscribed_radius", inscribed_radius)
    if not math.isfinite(corner_radius) or corner_radius < 0.0:
        raise ValueError("corner_radius must be finite and non-negative")
    if corner_radius > inscribed_radius:
        raise ValueError("corner radius cannot exceed the inscribed radius")
    return corner_radius / inscribed_radius


@dataclass(frozen=True)
class RoundedPolygon:
    """Regular ``n``-gon (edge ``a``) with corner fillets of radius ``r``.

    ``r = 0`` is the sharp polygon, ``r = r_max = a cot(pi/n) / 2`` is the
    limiting circle (all edges consumed by the fillets).  ``r`` is the single
    wear state variable of the attrition chain.
    """

    n_sides: int
    edge_length: float
    fillet_radius: float

    def __post_init__(self) -> None:
        n = int(self.n_sides)
        if n < 3:
            raise ValueError("n_sides must be >= 3")
        object.__setattr__(self, "n_sides", n)
        object.__setattr__(self, "edge_length",
                           _positive("edge_length", self.edge_length))
        r = float(self.fillet_radius)
        if not math.isfinite(r) or r < 0.0:
            raise ValueError("fillet_radius must be finite and non-negative")
        r_max = self.max_fillet_radius
        if r > r_max * (1.0 + 1e-12):
            raise ValueError(f"fillet_radius {r} exceeds the geometric maximum {r_max}")
        object.__setattr__(self, "fillet_radius", min(r, r_max))

    # -- angles and lengths ------------------------------------------------
    @property
    def interior_angle(self) -> float:
        """Interior angle ``alpha = (n - 2) pi / n``."""
        return (self.n_sides - 2) * math.pi / self.n_sides

    @property
    def corner_half_angle(self) -> float:
        """``alpha / 2``; the fillet consumes ``r cot(alpha/2)`` per edge end."""
        return self.interior_angle / 2.0

    @property
    def arc_turn(self) -> float:
        """Turning angle of a corner arc, ``pi - alpha = 2 pi / n``."""
        return math.pi - self.interior_angle

    @property
    def tangent_length(self) -> float:
        """``t = r cot(alpha/2)``: distance from the corner tip to tangency."""
        return self.fillet_radius / math.tan(self.corner_half_angle)

    @property
    def apothem(self) -> float:
        """Inscribed-circle radius (unchanged by filleting while facets remain)."""
        return self.edge_length / (2.0 * math.tan(math.pi / self.n_sides))

    @property
    def inscribed_radius(self) -> float:
        return self.apothem

    @property
    def circumradius_sharp(self) -> float:
        return self.edge_length / (2.0 * math.sin(math.pi / self.n_sides))

    @property
    def max_fillet_radius(self) -> float:
        """``a cot(pi/n) / 2`` -- the fillet radius that turns the shape into a circle."""
        return self.edge_length / (2.0 * math.tan(math.pi / self.n_sides))

    @property
    def facet_length(self) -> float:
        """Remaining flat edge length ``a - 2 r cot(alpha/2)`` (>= 0)."""
        return max(0.0, self.edge_length - 2.0 * self.tangent_length)

    @property
    def circumradius(self) -> float:
        """Distance from the centre to the farthest point of the shape."""
        centre_offset = self.circumradius_sharp - self.fillet_radius / math.cos(math.pi / self.n_sides)
        return centre_offset + self.fillet_radius

    # -- areas, perimeters, roundness --------------------------------------
    @property
    def sharp_area(self) -> float:
        return self.n_sides * self.edge_length ** 2 / (4.0 * math.tan(math.pi / self.n_sides))

    @property
    def area(self) -> float:
        corner_loss = 1.0 / math.tan(self.corner_half_angle) - 0.5 * self.arc_turn
        return self.sharp_area - self.n_sides * self.fillet_radius ** 2 * corner_loss

    @property
    def perimeter(self) -> float:
        return (self.n_sides * self.edge_length
                - 2.0 * self.n_sides * self.tangent_length
                + self.n_sides * self.fillet_radius * self.arc_turn)

    @property
    def wadell_roundness(self) -> float:
        """``R = 4 pi A / P^2`` in ``[4 pi A_0 / P_0^2, 1]``."""
        return wadell_roundness_2d(self.area, self.perimeter)

    @property
    def corner_radius_ratio(self) -> float:
        """``r / R_inscribed`` -- the Wadell corner-radius roundness."""
        return self.fillet_radius / self.inscribed_radius

    @property
    def interlocking_index(self) -> float:
        """``I = R_c / R_ins - 1``: size-normalised angularity, exactly 0 at ``R = 1``.

        At ``r = 0`` it is ``sec(pi/n) - 1`` (0.414 for a square, 1.0 for a
        triangle): the sharper the corner, the larger the radial excess of the
        boundary over the inscribed circle, i.e. the larger the geometric barrier
        a neighbouring floe has to ride over.  Monotone decreasing in ``r``.
        """
        return self.circumradius / self.inscribed_radius - 1.0

    @property
    def contact_patch_width(self) -> float:
        """Boundary length of one corner patch, ``2 t + r (pi - alpha)``."""
        return 2.0 * self.tangent_length + self.fillet_radius * self.arc_turn

    # -- wear bookkeeping --------------------------------------------------
    @property
    def worn_area(self) -> float:
        """Material removed relative to the sharp polygon (per unit depth)."""
        return self.sharp_area - self.area

    def with_fillet(self, fillet_radius: float) -> "RoundedPolygon":
        return RoundedPolygon(self.n_sides, self.edge_length, fillet_radius)

    def as_dict(self) -> dict:
        return {
            "n_sides": self.n_sides,
            "edge_length": self.edge_length,
            "fillet_radius": self.fillet_radius,
            "max_fillet_radius": self.max_fillet_radius,
            "area": self.area,
            "perimeter": self.perimeter,
            "wadell_roundness": self.wadell_roundness,
            "corner_radius_ratio": self.corner_radius_ratio,
            "interlocking_index": self.interlocking_index,
            "inscribed_radius": self.inscribed_radius,
            "circumradius": self.circumradius,
            "facet_length": self.facet_length,
            "worn_area": self.worn_area,
        }


@dataclass(frozen=True)
class RoundedCube:
    """Cube of full size ``a`` with all edges/corners rounded by radius ``r``.

    The body is ``cube(L) (+) ball(r)`` with ``L = a - 2r`` (Steiner/Minkowski
    parallel body).  ``V`` and ``A`` are closed form::

        V(r) = L^3 + 6 L^2 r + 3 pi L r^2 + (4/3) pi r^3
        A(r) = 6 L^2 + 6 pi L r + 4 pi r^2        ( = dV/dr )

    At ``r = a/2`` the body is exactly the sphere of radius ``a/2``.
    """

    size: float
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", _positive("size", self.size))
        r = float(self.radius)
        if not math.isfinite(r) or r < 0.0:
            raise ValueError("radius must be finite and non-negative")
        if r > self.size / 2.0 * (1.0 + 1e-12):
            raise ValueError("radius cannot exceed size/2 (the inscribed sphere)")
        object.__setattr__(self, "radius", min(r, self.size / 2.0))

    @property
    def max_radius(self) -> float:
        return self.size / 2.0

    @property
    def flat_size(self) -> float:
        """Edge length of the inner cube, ``L = a - 2 r``."""
        return self.size - 2.0 * self.radius

    @property
    def volume(self) -> float:
        L, r = self.flat_size, self.radius
        return (L ** 3 + 6.0 * L ** 2 * r + 3.0 * math.pi * L * r ** 2
                + 4.0 / 3.0 * math.pi * r ** 3)

    @property
    def surface_area(self) -> float:
        """``A = dV/dr`` (Steiner): the derivative of the parallel-body volume."""
        L, r = self.flat_size, self.radius
        return 6.0 * L ** 2 + 6.0 * math.pi * L * r + 4.0 * math.pi * r ** 2

    @property
    def sphericity(self) -> float:
        return sphericity_3d(self.volume, self.surface_area)

    @property
    def wadell_roundness(self) -> float:
        """``r / R_inscribed``; ``R_inscribed = a/2`` is invariant in ``r``."""
        return wadell_roundness_3d(self.radius, self.size / 2.0)

    @property
    def interlocking_index(self) -> float:
        """``R_c / R_ins - 1`` with ``R_c = sqrt(3) L / 2 + r`` (0 for the sphere)."""
        L, r = self.flat_size, self.radius
        return (math.sqrt(3.0) * 0.5 * L + r) / (self.size / 2.0) - 1.0

    @property
    def sharp_volume(self) -> float:
        return self.size ** 3

    @property
    def worn_volume(self) -> float:
        """Material removed relative to the sharp cube."""
        return self.sharp_volume - self.volume

    @property
    def contact_patch_area(self) -> float:
        """Corner contact patch area ``pi r^2`` (the projected corner cap)."""
        return math.pi * self.radius ** 2

    def with_radius(self, radius: float) -> "RoundedCube":
        return RoundedCube(self.size, radius)

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "radius": self.radius,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "sphericity": self.sphericity,
            "wadell_roundness": self.wadell_roundness,
            "interlocking_index": self.interlocking_index,
            "worn_volume": self.worn_volume,
        }


def _corner_geometry(poly: RoundedPolygon):
    """Yield ``(tip, centre, t_prev, t_next, a_prev, a_next, delta)`` per corner.

    ``delta`` is the CCW turning of the arc (signed, ``|delta| == 2 pi / n``);
    ``r == 0`` corners have ``centre = t_prev = t_next = tip`` and ``delta = 0``.
    """
    n = poly.n_sides
    r = poly.fillet_radius
    rc0 = poly.circumradius_sharp
    tips = np.array([[rc0 * math.cos(2.0 * math.pi * k / n),
                      rc0 * math.sin(2.0 * math.pi * k / n)] for k in range(n)])
    for k in range(n):
        tip = tips[k]
        if r == 0.0:
            yield tip, tip, tip, tip, 0.0, 0.0, 0.0
            continue
        u = tip / rc0                                   # outward corner bisector
        nxt, prv = tips[(k + 1) % n], tips[(k - 1) % n]
        e_next = (nxt - tip) / np.linalg.norm(nxt - tip)
        e_prev = (prv - tip) / np.linalg.norm(prv - tip)
        t_next = tip + poly.tangent_length * e_next
        t_prev = tip + poly.tangent_length * e_prev
        centre = (rc0 - r / math.cos(math.pi / n)) * u
        a_prev = math.atan2(*(t_prev - centre)[::-1])
        a_next = math.atan2(*(t_next - centre)[::-1])
        a_ref = math.atan2(u[1], u[0])
        delta = math.remainder(a_next - a_prev, 2.0 * math.pi)
        alt = delta - 2.0 * math.pi * (1.0 if delta > 0 else -1.0)
        if math.cos(a_prev + 0.5 * alt - a_ref) > math.cos(a_prev + 0.5 * delta - a_ref):
            delta = alt
        yield tip, centre, t_prev, t_next, a_prev, a_prev + delta, delta


def rounded_polygon_vertices(poly: RoundedPolygon, arc_samples: int = 8) -> np.ndarray:
    """Vertex cloud of the rounded polygon, ready to be fed to a polygon DEM.

    Each corner arc contributes ``arc_samples + 1`` points (both tangency points
    included); consecutive arcs are joined by the remaining flat facet, whose end
    points are exactly those tangency points.  For ``r == 0`` the sharp vertices
    are returned.  CCW order, shape ``(n_points, 2)``.
    """
    if arc_samples < 1:
        raise ValueError("arc_samples must be >= 1")
    verts = []
    for tip, centre, _, _, a_prev, a_next, delta in _corner_geometry(poly):
        if poly.fillet_radius == 0.0:
            verts.append(tip)
            continue
        for a in a_prev + np.linspace(0.0, delta, arc_samples + 1):
            verts.append(centre + poly.fillet_radius * np.array([math.cos(a), math.sin(a)]))
    return np.asarray(verts, dtype=float)


def polygon_metrics_boundary_integral(poly: RoundedPolygon) -> tuple:
    """Area and perimeter from an **exact** boundary integral (Green's theorem).

    Independent of the corner-subtraction formulas used by ``area`` /
    ``perimeter``: the straight facets contribute the shoelace term and every arc
    the closed-form line integral

    .. math:: \\tfrac12 \\int (x\\,dy - y\\,dx) = \\tfrac12 [ r c_x \\Delta\\sin
              - r c_y \\Delta\\cos + r^2 \\Delta ]

    so the comparison of the two routes is a genuine cross-check (agreement at
    machine precision, not at a discretisation error).
    """
    n = poly.n_sides
    r = poly.fillet_radius
    area2 = 0.0
    perimeter = 0.0
    corners = list(_corner_geometry(poly))
    for k, (tip, centre, t_prev, t_next, a_prev, a_next, delta) in enumerate(corners):
        # flat facet from this corner's t_next to the next corner's t_prev
        p0 = t_next
        p1 = corners[(k + 1) % n][2]
        area2 += p0[0] * p1[1] - p1[0] * p0[1]
        perimeter += float(np.hypot(*(p1 - p0)))
        if r == 0.0:
            continue
        cx, cy = centre
        area2 += (r * cx * (math.sin(a_next) - math.sin(a_prev))
                  - r * cy * (math.cos(a_next) - math.cos(a_prev))
                  + r * r * delta)
        perimeter += r * abs(delta)
    return 0.5 * abs(area2), perimeter


def fillet_radius_for_roundness(poly: RoundedPolygon, target_roundness: float,
                                *, xtol: float = 1e-14) -> float:
    """Invert ``R(r)`` on ``[0, r_max]`` (monotone) by bisection.

    Used to build design-of-experiment shapes at prescribed roundness.  Raises if
    the target is outside ``[R(0), 1]``.
    """
    target = float(target_roundness)
    lo, hi = 0.0, poly.max_fillet_radius
    r_lo = poly.with_fillet(lo).wadell_roundness
    if target < r_lo - 1e-12 or target > 1.0 + 1e-12:
        raise ValueError(f"target roundness {target} outside [{r_lo}, 1]")
    if target <= r_lo:
        return 0.0
    if target >= 1.0:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hi - lo < xtol * max(1.0, poly.edge_length):
            break
        if poly.with_fillet(mid).wadell_roundness < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)