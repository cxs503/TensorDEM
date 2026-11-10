"""Plan-view rasteriser used to *measure* how well a DEM body reproduces a floe.

Why a raster and not a polygon library
--------------------------------------
The handoff of campaign ``p302w1-n03`` verifies the floe partition by area and
perimeter bookkeeping.  Those are exact, but they cannot answer the question the
DEM side actually cares about: *does the discrete body occupy the same ground as
the continuum floe?*  A single disc of equivalent area placed at the centroid can
sit half outside a non-convex floe and still pass an area check.  Measuring the
symmetric difference therefore needs a geometric ground truth.

No polygon-clipping library is a dependency of this package, so the intersection
of a polygon with a union of discs is measured on a raster.  Two properties make
that honest rather than sloppy:

* the floe polygons are edge-aligned to the FEM lattice, so with a raster that is
  an integer refinement of the element size every cell centre is unambiguously
  inside or outside and the raster area of a floe equals its shoelace area to
  round-off (asserted in the tests);
* every metric is reported together with the raster resolution and the
  resolution error estimated by halving ``h`` (:func:`resolution_check`), so the
  number quoted in the report is the converged one, not an artefact of ``h``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

from ..fem_dem_fracture.crack_network import polygon_area

__all__ = [
    "Raster",
    "polygon_mask",
    "disc_mask",
    "footprint_metrics",
    "resolution_check",
]

# A point this far inside the ring counts as inside.  Cell centres are placed at
# half-integers of the raster, so for an edge-aligned polygon no centre lies on
# the boundary and the radius only guards against floating-point noise.
_EPS = 1e-12


@dataclass(frozen=True)
class Raster:
    """Axis-aligned sampling window with square cells of side ``h`` [m]."""

    x0: float
    y0: float
    x1: float
    y1: float
    h: float

    def __post_init__(self) -> None:
        if self.h <= 0.0:
            raise ValueError("raster cell size must be positive")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("raster window must have positive extent")

    @property
    def nx(self) -> int:
        return int(round((self.x1 - self.x0) / self.h))

    @property
    def ny(self) -> int:
        return int(round((self.y1 - self.y0) / self.h))

    @property
    def cell_area(self) -> float:
        return self.h * self.h

    @property
    def area(self) -> float:
        return self.nx * self.ny * self.cell_area

    def centres(self) -> np.ndarray:
        """``(ny, nx, 2)`` coordinates of the cell centres."""
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.h
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.h
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        return np.stack([gx, gy], axis=-1)

    def refined(self, factor: int = 2) -> "Raster":
        """Same window sampled ``factor`` times finer (resolution check)."""
        return Raster(self.x0, self.y0, self.x1, self.y1, self.h / float(factor))

    def mask_area(self, mask: np.ndarray) -> float:
        return float(np.count_nonzero(mask)) * self.cell_area


def _ring_path(ring) -> Path:
    return Path(np.asarray(ring, dtype=float))


def polygon_mask(raster: Raster, rings) -> np.ndarray:
    """Boolean mask of the cells whose centre is inside ``rings``.

    ``rings`` is the floe polygon representation used by the handoff artefact:
    the outer ring first, then zero or more holes.
    """
    rings = list(rings)
    if not rings:
        raise ValueError("a polygon needs at least one ring")
    pts = raster.centres().reshape(-1, 2)
    inside = _ring_path(rings[0]).contains_points(pts, radius=_EPS)
    for hole in rings[1:]:
        inside &= ~_ring_path(hole).contains_points(pts, radius=_EPS)
    return inside.reshape(raster.ny, raster.nx)


def disc_mask(raster: Raster, centres, radii) -> np.ndarray:
    """Boolean mask of the cells covered by the union of the discs."""
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if centres.shape[0] != radii.size:
        raise ValueError("one radius per centre is required")
    out = np.zeros((raster.ny, raster.nx), dtype=bool)
    if centres.shape[0] == 0:
        return out
    cx = raster.x0 + (np.arange(raster.nx) + 0.5) * raster.h
    cy = raster.y0 + (np.arange(raster.ny) + 0.5) * raster.h
    # Bucket the discs by their column range so each one only touches the cells
    # it can possibly reach; with O(10..100) discs this keeps the pass cheap.
    for (x, y), r in zip(centres, radii):
        i0 = max(0, int(np.floor((x - r - raster.x0) / raster.h)) - 1)
        i1 = min(raster.nx, int(np.ceil((x + r - raster.x0) / raster.h)) + 2)
        j0 = max(0, int(np.floor((y - r - raster.y0) / raster.h)) - 1)
        j1 = min(raster.ny, int(np.ceil((y + r - raster.y0) / raster.h)) + 2)
        if i0 >= i1 or j0 >= j1:
            continue
        dx = cx[i0:i1] - x
        dy = cy[j0:j1] - y
        hit = (dx[None, :] ** 2 + dy[:, None] ** 2) <= r * r
        out[j0:j1, i0:i1] |= hit
    return out


def footprint_metrics(raster: Raster, rings, centres, radii) -> dict:
    """Coverage / spill / IoU of a disc set against its floe polygon.

    * ``polygon_area``   -- shoelace area of the rings (independent of the raster)
    * ``polygon_area_raster`` -- raster area, i.e. the area the sampler sees
    * ``disc_area``      -- analytic area of the discs (exact if they do not
      overlap; overlap is audited separately per floe)
    * ``covered_area``   -- raster area of ``disc union INTERSECT polygon``
    * ``spill_area``     -- raster area of ``disc union MINUS polygon``
    * ``coverage``       -- ``covered_area / polygon_area`` (how much of the floe
      is represented by material)
    * ``iou``            -- ``covered / (covered + spill + uncovered)``, the
      symmetric-difference score: 1 means the body occupies exactly the floe
    """
    poly = polygon_mask(raster, rings)
    discs = disc_mask(raster, centres, radii)
    poly_area_raster = raster.mask_area(poly)
    covered = raster.mask_area(discs & poly)
    spill = raster.mask_area(discs & ~poly)
    uncovered = raster.mask_area(~discs & poly)
    union = covered + spill + uncovered
    outer = polygon_area(np.asarray(rings[0], dtype=float))
    holes = sum(abs(polygon_area(np.asarray(h, dtype=float))) for h in list(rings)[1:])
    poly_area = abs(outer) - holes
    radii = np.asarray(radii, dtype=float).reshape(-1)
    return {
        "polygon_area": float(poly_area),
        "polygon_area_raster": float(poly_area_raster),
        "polygon_area_raster_error": float(poly_area_raster - poly_area),
        "disc_area": float(np.sum(np.pi * radii * radii)),
        "covered_area": float(covered),
        "spill_area": float(spill),
        "uncovered_area": float(uncovered),
        "coverage": float(covered / poly_area) if poly_area > 0 else 0.0,
        "iou": float(covered / union) if union > 0 else 0.0,
        "raster_h": float(raster.h),
    }


def resolution_check(raster: Raster, rings, build, *, factor: int = 2) -> dict:
    """Measure the footprint metrics at ``h`` and at ``h / factor``.

    ``build`` is called with the raster and must return ``(centres, radii)`` of
    the body under test; it is re-run on the finer raster so that the comparison
    isolates the *measurement* resolution and not the seeding rule.  The return
    value carries both metric sets and the largest relative change of the shape
    scores, so a report can state the discretisation error instead of hiding it.
    """
    coarse = footprint_metrics(raster, rings, *build(raster))
    fine_raster = raster.refined(factor)
    fine = footprint_metrics(fine_raster, rings, *build(fine_raster))
    deltas = {}
    for key in ("coverage", "iou", "spill_area", "covered_area"):
        denom = max(abs(coarse[key]), abs(fine[key]), 1e-30)
        deltas[key] = float(abs(fine[key] - coarse[key]) / denom)
    return {
        "coarse_h": float(coarse["raster_h"]),
        "fine_h": float(fine["raster_h"]),
        "coarse": coarse,
        "fine": fine,
        "relative_change": deltas,
        "max_relative_change": float(max(deltas.values())),
    }
