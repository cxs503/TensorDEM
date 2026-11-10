"""Three-dimensional brash-ice fields: geometry, thickness statistics, porosity.

The 2-D brash-ice field of m7768 is a Laguerre tessellation of the channel
covered by convex polygons with a prescribed floe-size distribution.  The 3-D
field used here keeps that in-plane topology (the same Voronoi/Laguerre family)
and adds the two dimensions a channel model does not resolve:

* a **thickness** per floe, drawn from a prescribed distribution, which turns
  every polygon into a plate (the "extrude" lift) and makes the floe volume, the
  draft and the displaced volume physical;
* a **strength** per floe from the Weibull weakest-link law at the floe volume,
  so that a loaded floe field has a distribution of strengths rather than one
  calibrated number.

Porosity is *measured*, never imposed: ``phi = 1 - sum(V_floe) / V_box`` over the
floes whose centroid lies in the measurement box.  Keeping the measured value
instead of the target value is what makes the porosity evolution of a cleared
channel a prediction rather than a re-statement of the input.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import Voronoi

from .dynamics import FloeBody
from .geometry import (ConvexPolyhedron, extrude_polygon, polygon_area,
                       regular_polygon)
from .weibull import sample_strengths, weakest_link_scale

__all__ = [
    "voronoi_polygons_2d",
    "BrashIceField",
    "build_brash_ice_field",
    "build_disc_reference_field",
    "lognormal_thicknesses",
]

_TINY = 1.0e-12


def voronoi_polygons_2d(sites: np.ndarray, box: Sequence[float],
                        mirror: float = 1.05) -> List[np.ndarray]:
    """Voronoi cells of planar ``sites`` clipped to the rectangle ``box``.

    ``box = (x0, y0, x1, y1)``.  Boundary sites have unbounded cells, so the point
    set is mirrored across all four sides; the mirrored points only enter as
    clipping planes and no cell of the original set is affected.  ``mirror`` is
    measured in box widths and must exceed 1, otherwise the mirror images land
    *inside* the box and spuriously subdivide the cells.  The
    polygon of every original site is returned as an ``(m, 2)`` counter-
    clockwise vertex array.
    """
    sites = np.asarray(sites, dtype=np.float64)
    x0, y0, x1, y1 = [float(v) for v in box]
    shift = [(0, 0)]
    if mirror > 0.0:
        dx = (x1 - x0) * mirror
        dy = (y1 - y0) * mirror
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                if sx == 0 and sy == 0:
                    continue
                shift.append((sx * dx, sy * dy))
    pts = np.vstack([sites + np.array(s) for s in shift])
    vor = Voronoi(pts)
    polygons: List[np.ndarray] = []
    for index in range(sites.shape[0]):
        region = vor.regions[vor.point_region[index]]
        if not region or -1 in region:
            polygons.append(np.zeros((0, 2)))
            continue
        poly = vor.vertices[region]
        poly = _clip_rect(poly, x0, y0, x1, y1)
        polygons.append(poly)
    return polygons


def _clip_rect(poly: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon to an axis-aligned box."""
    out = np.asarray(poly, dtype=np.float64)
    for axis, bound, keep_above in ((0, x0, True), (0, x1, False),
                                    (1, y0, True), (1, y1, False)):
        if out.shape[0] == 0:
            return out
        values = out[:, axis] * (1.0 if keep_above else -1.0)
        limit = bound * (1.0 if keep_above else -1.0)
        inside = values >= limit
        if np.all(inside):
            continue
        if not np.any(inside):
            return np.zeros((0, 2))
        nxt = np.roll(np.arange(out.shape[0]), -1)
        result = []
        for i in range(out.shape[0]):
            j = nxt[i]
            a, b = out[i], out[j]
            ia, ib = inside[i], inside[j]
            if ia:
                result.append(a)
            if ia != ib:
                t = (limit - values[i]) / (values[j] - values[i])
                result.append(a + t * (b - a))
        out = np.array(result, dtype=np.float64)
    return out


def _convex_ccw(poly: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull
    if poly.shape[0] < 3:
        return poly
    hull = ConvexHull(poly)
    return poly[hull.vertices]


def lognormal_thicknesses(n: int, median: float, sigma_log: float,
                          rng: np.random.Generator) -> np.ndarray:
    if sigma_log <= 0.0:
        return np.full(int(n), float(median))
    return np.exp(np.log(float(median)) + sigma_log * rng.standard_normal(int(n)))


@dataclass
class BrashIceField:
    """A 3-D brash-ice field: polygonal footprints, thicknesses and strengths."""

    bodies: List[FloeBody]
    polygons: List[np.ndarray]
    thickness: np.ndarray
    box: Tuple[float, float, float, float]
    water_level: float
    ice_density: float
    water_density: float
    kind: str = "polyhedron"
    meta: Dict[str, float] = dc_field(default_factory=dict)

    # -- measured quantities ----------------------------------------------
    @property
    def total_volume(self) -> float:
        return float(sum(b.shape.volume for b in self.bodies))

    @property
    def mass(self) -> float:
        return float(sum(b.mass for b in self.bodies))

    def porosity(self, region: Optional[Sequence[float]] = None) -> float:
        """Measured porosity ``1 - V_solid / V_region`` (default: the whole box)."""
        x0, y0, x1, y1 = self.box
        if region is None:
            # the brash-ice *layer*: the floes only occupy the draft plus a little
            # freeboard, so measuring over an unbounded vertical range would report
            # a porosity of 1 by construction
            lo = np.array([x0, y0, self.water_level - 0.18])
            hi = np.array([x1, y1, self.water_level + 0.06])
        else:
            lo = np.asarray(region[:3], dtype=np.float64)
            hi = np.asarray(region[3:6], dtype=np.float64)
        volume = float(np.prod(hi - lo))
        solid = sum(b.shape.volume for b in self.bodies
                    if np.all(b.position >= lo) and np.all(b.position <= hi))
        return 1.0 - solid / volume if volume > 0.0 else float("nan")

    def statistics(self) -> Dict[str, float]:
        areas = np.array([b.shape.area for b in self.bodies])
        sides = np.array([b.shape.n_vertices for b in self.bodies])
        return {
            "n_floes": len(self.bodies),
            "kind": self.kind,
            "total_solid_volume": self.total_volume,
            "total_mass": self.mass,
            "porosity": self.porosity(),
            "mean_thickness": float(np.mean(self.thickness)),
            "std_thickness": float(np.std(self.thickness)),
            "mean_floe_volume": float(np.mean([b.shape.volume for b in self.bodies])),
            "mean_vertices": float(np.mean(sides)) if sides.size else 0.0,
            "mean_area": float(np.mean(areas)) if areas.size else 0.0,
        }


def _equilibrium_height(volume: float, thickness: float, water_level: float,
                        ice_density: float, water_density: float) -> float:
    """Vertical centroid position of a floating plate (draft = rho_i/rho_w * t)."""
    footprint = volume / thickness
    draft = min(ice_density / water_density * thickness, thickness)
    # submerged slab of depth `draft` measured from the floe bottom
    bottom = water_level - draft
    return bottom + 0.5 * thickness


def build_brash_ice_field(n_sites: int, box: Sequence[float], *, median_thickness: float,
                          thickness_sigma_log: float = 0.0, ice_density: float = 900.0,
                          water_density: float = 1025.0, water_level: float = 0.0,
                          modulus: float = 3.5, strength_scale: float = 1.2e6,
                          strength_reference_volume: float = 1.0,
                          seed: int = 0, kind: str = "polyhedron",
                          disc_sides: int = 24, jitter: bool = True) -> BrashIceField:
    """Build a 3-D brash-ice field by extruding a planar Laguerre tessellation.

    ``kind="polyhedron"`` keeps the tessellation polygons (irregular convex
    plates); ``kind="disc"`` replaces every footprint by a regular polygon of
    the **same area** (the disc-plate reference of the shape-effect study).  The
    two fields therefore share the floe volumes, thicknesses and centres and
    differ only in plan form.
    """
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = [float(v) for v in box]
    area = (x1 - x0) * (y1 - y0)
    sites = np.column_stack([rng.uniform(x0, x1, n_sites), rng.uniform(y0, y1, n_sites)])
    polygons = [_convex_ccw(p) for p in voronoi_polygons_2d(sites, box)]
    thickness = lognormal_thicknesses(n_sites, median_thickness, thickness_sigma_log, rng)

    bodies: List[FloeBody] = []
    kept_polygons: List[np.ndarray] = []
    kept_thickness: List[float] = []
    for poly, t in zip(polygons, thickness):
        if poly.shape[0] < 3:
            continue
        a = polygon_area(poly)
        if a <= 1.0e-9 * area:
            continue
        if kind == "disc":
            centre = poly.mean(axis=0)
            poly_use = regular_polygon(a, disc_sides, phase=rng.uniform(0, 2 * math.pi),
                                       centre=centre)
        else:
            poly_use = poly
        shrunk = _shrink(poly_use, 0.985)     # a hair of clearance between floes
        shape = extrude_polygon(shrunk, t)
        volume = shape.volume
        z = _equilibrium_height(volume, t, water_level, ice_density, water_density)
        strength = float(sample_strengths(1, modulus,
                                         weakest_link_scale(strength_scale, strength_reference_volume,
                                                            volume, modulus), rng)[0])
        # place the body by its *area* centroid (not the vertex mean) so that the
        # tessellation is reproduced exactly and no floe overlaps a wall
        # the body is stored in its principal frame, so the world orientation must
        # undo that rotation: otherwise the (irregular) footprint would be rotated
        # about its centroid and neighbouring floes would interpenetrate
        bodies.append(FloeBody(shape, ice_density,
                               position=[float(shape.centroid_in_planes[0]),
                                         float(shape.centroid_in_planes[1]), z],
                               rotation=shape.principal_axes,
                               label=f"{kind[0]}{len(bodies)}", strength=strength))
        kept_polygons.append(shrunk)
        kept_thickness.append(float(t))

    meta = {"target_porosity": float("nan"), "medians": float(np.median(kept_thickness))}
    field = BrashIceField(bodies, kept_polygons, np.array(kept_thickness), (x0, y0, x1, y1),
                          float(water_level), float(ice_density), float(water_density),
                          kind=kind, meta=meta)
    meta["target_porosity"] = field.porosity()
    return field


def build_disc_reference_field(field: BrashIceField, disc_sides: int = 24) -> BrashIceField:
    """Disc-plate twin of an existing field (identical volume, thickness, centre).

    Only the plan form changes: every irregular tessellation plate is replaced by
    a regular ``disc_sides``-gon of exactly the same footprint area and the same
    thickness, so the mass, the draft and the buoyancy are unchanged and any
    difference in clearing resistance is a pure shape effect.
    """
    bodies: List[FloeBody] = []
    polygons: List[np.ndarray] = []
    for body, poly in zip(field.bodies, field.polygons):
        area = polygon_area(poly)
        thickness = body.shape.volume / area
        disc = regular_polygon(area, disc_sides, centre=poly.mean(axis=0))
        shape = extrude_polygon(disc, thickness)
        bodies.append(FloeBody(shape, body.density,
                               position=body.position.copy(),
                               rotation=shape.principal_axes,
                               label=body.label, strength=body.strength))
        polygons.append(disc)
    return BrashIceField(bodies, polygons, field.thickness.copy(), field.box,
                         field.water_level, field.ice_density, field.water_density,
                         kind="disc",
                         meta={"target_porosity": field.meta.get("target_porosity", float("nan"))})


def _shrink(poly: np.ndarray, factor: float) -> np.ndarray:
    centre = poly.mean(axis=0)
    return centre + (poly - centre) * float(factor)