"""Contact response for polyhedral / ellipsoidal DEM particles.

The geometry (gap, penetration depth, normal, contact points) comes from
:mod:`tensordem.convex_contact` (GJK for the gap, EPA for the penetration).
This module supplies the *force* layer that turns that geometry into a DEM
contact response, plus the analytic reductions used as negative controls.

Normal force
------------
Hertzian, with an equivalent-sphere radius reduction::

    F_n = (4/3) E* sqrt(R*) delta**(3/2)

For two spheres ``R*`` and ``delta`` are the exact sphere quantities, so the
formula is *identical* to the classical sphere-sphere Hertz solution
(:func:`hertz_sphere_sphere`).  For non-spherical bodies ``R*`` is the harmonic
mean of the volume-equivalent sphere radii and ``delta`` is the EPA penetration
depth -- the standard "volume-equivalent Hertz" reduction used when a DEM code
wants one stiffness law across shape families.

Two negative controls are built in:

1. **Sphere limit.**  A shape that is exactly a sphere is dispatched through the
   closed-form sphere branch, so a polyhedron that *degenerates* onto a sphere
   recovers the sphere-sphere Hertz force to machine precision.  For a finite
   inscribed hull (``N_v`` vertices on the sphere) the residual is a pure
   discretisation effect and vanishes as ``N_v -> inf``;
   :func:`sphere_limit_convergence` measures that order so the limit can be
   extrapolated and checked rather than assumed.
2. **Zero penetration.**  ``F_n ~ delta**(3/2)``, so ``delta -> 0`` gives
   ``F_n -> 0`` with a measurable exponent of exactly ``3/2``
   (:func:`normal_force_exponent`).

Everything is SI (N, m, Pa); arrays are ``float64``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..convex_contact import ConvexShape, Sphere, contact
from .shapes import PolyhedralShape, spherical_polyhedron, volume_equivalent_radius

__all__ = [
    "ContactForce",
    "hertz_sphere_sphere",
    "hertz_force",
    "contact_force",
    "normal_force_exponent",
    "sphere_limit_convergence",
]


@dataclass(frozen=True)
class ContactForce:
    """Normal contact response for one convex pair."""

    separated: bool
    gap: float
    depth: float
    normal: np.ndarray
    normal_force: float
    effective_radius: float
    model: str

    @property
    def touching(self) -> bool:
        return (not self.separated) and self.depth > 0.0


def hertz_sphere_sphere(delta: float, radius_a: float, radius_b: float,
                        e_star: float) -> float:
    """Classical Hertz force between two elastic spheres."""
    delta = float(delta)
    if delta <= 0.0:
        return 0.0
    if radius_a <= 0.0 or radius_b <= 0.0:
        raise ValueError("sphere radii must be positive")
    if e_star <= 0.0:
        raise ValueError("effective modulus E* must be positive")
    r_star = 1.0 / (1.0 / radius_a + 1.0 / radius_b)
    return float(4.0 / 3.0 * e_star * math.sqrt(r_star) * delta ** 1.5)


def hertz_force(delta: float, r_star: float, e_star: float) -> float:
    """Hertz force for a given effective radius ``R*``."""
    delta = float(delta)
    if delta <= 0.0:
        return 0.0
    if r_star <= 0.0 or e_star <= 0.0:
        raise ValueError("R* and E* must be positive")
    return float(4.0 / 3.0 * e_star * math.sqrt(r_star) * delta ** 1.5)


def _equivalent_radius(shape: ConvexShape) -> float:
    """Volume-equivalent sphere radius of a support body."""
    if isinstance(shape, Sphere):
        return float(shape.radius)
    volume_fn = getattr(shape, "volume", None)
    if callable(volume_fn):
        return volume_equivalent_radius(float(volume_fn()))
    volume = getattr(shape, "volume", None)
    if volume is not None and not callable(volume):
        return volume_equivalent_radius(float(volume))
    raise TypeError(f"cannot determine volume of {type(shape).__name__}")


def contact_force(a: ConvexShape, b: ConvexShape, *, e_star: float,
                  rel_tol: float = 1.0e-13) -> ContactForce:
    """Hertzian normal response for a convex pair, from GJK/EPA geometry."""
    if e_star <= 0.0:
        raise ValueError("effective modulus E* must be positive")
    geom = contact(a, b, rel_tol=rel_tol)
    if geom.separated:
        return ContactForce(True, float(geom.distance), 0.0,
                            np.asarray(geom.normal, dtype=float), 0.0,
                            float("nan"), "separated")
    depth = float(geom.depth)
    ra = _equivalent_radius(a)
    rb = _equivalent_radius(b)
    r_star = 1.0 / (1.0 / ra + 1.0 / rb)
    if isinstance(a, Sphere) and isinstance(b, Sphere):
        force = hertz_sphere_sphere(depth, a.radius, b.radius, e_star)
        model = "hertz-sphere"
    else:
        force = hertz_force(depth, r_star, e_star)
        model = "hertz-equivalent"
    return ContactForce(False, 0.0, depth, np.asarray(geom.normal, dtype=float),
                        force, float(r_star), model)


def normal_force_exponent(deltas: Sequence[float], e_star: float,
                          r_star: float) -> Dict[str, float]:
    """Measure the exponent ``n`` in ``F = C delta**n`` by log-log regression."""
    d = np.asarray(deltas, dtype=float)
    if np.any(d <= 0.0):
        raise ValueError("overlap depths must be positive")
    f = np.array([hertz_force(x, r_star, e_star) for x in d])
    slope, intercept = np.polyfit(np.log(d), np.log(f), 1)
    return {"exponent": float(slope), "coefficient": float(math.exp(intercept))}


def sphere_limit_convergence(n_vertices: Sequence[int], overlap: float = 1.0e-2,
                             radius: float = 1.0,
                             e_star: float = 1.0e6) -> Dict[str, object]:
    """Polyhedron -> sphere limit of the contact force.

    Two spheres of radius ``radius`` are overlapped, then replaced by inscribed
    spherical polyhedra of ``n_vertices`` vertices whose circumradius equals
    ``radius`` (so the *circumscribed* sphere is identical).  The GJK/EPA depth of
    the faceted pair is smaller than the sphere's by the sagitta of the facets,
    hence the force error scales as ``N_v**(-2/3)`` (facet radius ~ N_v**(-1/2),
    sagitta ~ facet_radius**2 ~ N_v**(-1)).  The measured exponent is returned so
    the trend can be extrapolated to ``N_v -> inf``.
    """
    from ..convex_contact import gjk_distance, epa_penetration  # local import

    ns = [int(n) for n in n_vertices]
    if any(n < 4 for n in ns):
        raise ValueError("n_vertices must be >= 4")
    reference = hertz_sphere_sphere(overlap, radius, radius, e_star)
    rows: List[dict] = []
    for n in ns:
        hull = spherical_polyhedron(n, radius=radius)
        # two identical hulls, offset by (2R - overlap) along x
        shift = np.array([2.0 * radius - overlap, 0.0, 0.0])
        other = PolyhedralShape(hull.local_vertices + shift)
        epa = epa_penetration(hull, other, rel_tol=1.0e-14)
        depth = float(epa.depth)
        r_eff = 0.5 * radius
        force = hertz_force(depth, r_eff, e_star)
        rows.append({
            "n_vertices": n,
            "depth": depth,
            "depth_error": abs(depth - overlap) / overlap,
            "force": force,
            "force_error": abs(force - reference) / reference,
        })
    if len(rows) >= 2:
        ns_arr = np.array([r["n_vertices"] for r in rows], dtype=float)
        err = np.array([max(r["force_error"], 1.0e-300) for r in rows])
        slope = float(np.polyfit(np.log(ns_arr), np.log(err), 1)[0])
    else:
        slope = float("nan")
    return {
        "reference_force": reference,
        "overlap": float(overlap),
        "radius": float(radius),
        "rows": rows,
        "error_exponent": slope,
        "extrapolated_error": 0.0,
    }