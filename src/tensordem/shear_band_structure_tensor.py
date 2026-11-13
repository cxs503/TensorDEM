"""Structure-tensor (image-gradient) measurement of a shear-band inclination.

Why a second operator
---------------------
``tensordem.biaxial_shear_band.radon_band_angle`` orients the *ridge* of a
scattered local-strain field by a 1-D projection focus.  It is exact on
synthetic ridges and its aspect-ratio bias was removed in R2, but it is a
1-D statistic: it only sees the projection coordinate, so a noisy field with
several competing structures (band + servo compaction front + boundary bands)
can move its maximum.  The experimental literature measures the same quantity
on a **gridded strain field** with the structure tensor (image gradient
tensor), e.g. Desrues & Viggiani (2004, *Int. J. Numer. Anal. Methods
Geomech.* 28:279-321) who compute strain fields on a mesh and read the band
orientation from the direction of maximum strain gradient.  This module
implements that route so the case can be measured twice, by two independent
operators, on the *same* particle snapshots:

1. scatter the per-particle local maximum shear strain ``gamma`` onto a
   regular grid whose cell size is the mean particle spacing of the window
   (``cell = sqrt(area / N)``; no hand-tuned resolution);
2. Gaussian-smooth the grid (separable, ``sigma`` in cells);
3. build the structure tensor ``J = <grad g (x) grad g>`` over the populated
   cells, weight = occupancy (so the free surface / wall gaps carry no weight);
4. the band *normal* is the eigenvector of the **largest** eigenvalue of ``J``
   (the strain gradient is across the band), the band inclination is the
   perpendicular direction, folded into ``[0, 90]`` degrees from ``sigma_3``;
5. ``coherence = (l1 - l2) / (l1 + l2)`` is the localisation strength of the
   orientation (0 = isotropic field, 1 = perfectly planar ridge).

Everything is pure ``numpy``; no scipy dependency is introduced.  The operator
is validated on synthetic bands of known inclination injected into the exact
case lattice (``benchmarks/biaxial_shear_band/run_r3_estimator_audit.py``,
``tests/test_shear_band_structure_tensor.py``).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .biaxial_shear_band import (
    BandMeasurementError,
    fold_angle_deg,
    local_shear_strain,
)

__all__ = [
    "grid_scalar_field",
    "smooth_grid",
    "structure_tensor_band_angle",
    "measure_band_structure_tensor",
]


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / float(sigma)) ** 2)
    return kernel / kernel.sum()


def _separable_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-order-hold Gaussian blur with edge replication (numpy only)."""
    if not sigma > 0.0:
        return field.copy()
    kernel = _gaussian_kernel(sigma)
    radius = (kernel.size - 1) // 2
    out = field
    for axis in (0, 1):
        padded = np.pad(out, [(radius, radius) if a == axis else (0, 0)
                              for a in (0, 1)], mode="edge")
        acc = np.zeros_like(out)
        for offset, weight in enumerate(kernel):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(offset, offset + out.shape[axis])
            acc += weight * padded[tuple(sl)]
        out = acc
    return out


def grid_scalar_field(x: Sequence[float], y: Sequence[float],
                      value: Sequence[float], *, cell: float | None = None,
                      bounds: Sequence[float] | None = None
                      ) -> dict[str, Any]:
    """Scatter ``value`` onto a regular grid of square cells of size ``cell``.

    Empty cells are filled by repeated neighbour averaging (a cheap harmonic
    fill) so that gradients across the sparse-particle gaps stay finite; the
    returned ``occupancy`` records which cells carried data at all.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    vals = np.asarray(value, dtype=np.float64).ravel()
    if not (xs.size == ys.size == vals.size):
        raise BandMeasurementError("x, y and value must have the same length")
    if xs.size < 30:
        raise BandMeasurementError("at least 30 samples are needed")
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise BandMeasurementError("coordinates must be finite")
    if bounds is None:
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
    else:
        xmin, xmax, ymin, ymax = (float(v) for v in bounds)
    width, height = xmax - xmin, ymax - ymin
    if not (width > 0.0 and height > 0.0):
        raise BandMeasurementError("degenerate measurement window")
    if cell is None:
        cell = math.sqrt(width * height / xs.size)      # mean particle spacing
    if not cell > 0.0:
        raise BandMeasurementError("cell size must be positive")
    nx = max(4, int(round(width / cell)))
    ny = max(4, int(round(height / cell)))
    ix = np.clip(((xs - xmin) / width * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip(((ys - ymin) / height * ny).astype(np.int64), 0, ny - 1)
    flat = ix * ny + iy
    total = np.bincount(flat, weights=vals, minlength=nx * ny)
    count = np.bincount(flat, minlength=nx * ny).astype(np.float64)
    field = np.zeros(nx * ny, dtype=np.float64)
    occupied = count > 0.0
    field[occupied] = total[occupied] / count[occupied]
    field = field.reshape(nx, ny)
    filled = occupied.reshape(nx, ny).copy()

    guard = 0
    while not filled.all() and guard < 4 * (nx + ny):     # neighbour fill
        guard += 1
        padded = np.pad(np.where(filled, field, 0.0), 1, mode="edge")
        weight = np.pad(filled.astype(np.float64), 1, mode="edge")
        acc = np.zeros((nx, ny))
        wacc = np.zeros((nx, ny))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                acc += padded[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny]
                wacc += weight[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny]
        newly = (~filled) & (wacc > 0.0)
        field[newly] = acc[newly] / wacc[newly]
        filled |= newly
    return {"field": field, "occupancy": occupied.reshape(nx, ny),
            "cell": float(cell), "cells": (nx, ny),
            "bounds": (xmin, xmax, ymin, ymax),
            "width": width, "height": height}


def smooth_grid(field: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian smoothing of a 2-D field (``sigma`` in cells)."""
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2:
        raise BandMeasurementError("field must be 2-D")
    return _separable_blur(arr, float(sigma))


def structure_tensor_band_angle(field: np.ndarray, occupancy: np.ndarray, *,
                                sigma: float = 1.5,
                                weight: Sequence[float] | None = None
                                ) -> dict[str, Any]:
    """Band inclination (deg from the horizontal ``sigma_3`` axis) of a grid.

    ``field`` is the scattered scalar (typically the local maximum shear
    strain), ``occupancy`` the mask of cells that hold particles.  Gradients
    are central differences of the smoothed field, in cell units; ``J`` is
    accumulated over populated cells only.
    """
    arr = np.asarray(field, dtype=np.float64)
    mask = np.asarray(occupancy, dtype=bool)
    if arr.shape != mask.shape or arr.ndim != 2:
        raise BandMeasurementError("field and occupancy must be equal 2-D shapes")
    smooth = smooth_grid(arr, sigma)
    gx, gy = np.gradient(smooth)
    w = mask.astype(np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    if w.shape != arr.shape:
        raise BandMeasurementError("weight must match field shape")
    w = w * mask
    if w.sum() <= 0.0:
        raise BandMeasurementError("no populated cells to integrate over")
    jxx = float(np.sum(w * gx * gx))
    jxy = float(np.sum(w * gx * gy))
    jyy = float(np.sum(w * gy * gy))
    trace = jxx + jyy
    if not trace > 0.0:
        raise BandMeasurementError("degenerate (constant) field: no gradient")
    disc = math.sqrt(max(0.25 * (jxx - jyy) ** 2 + jxy * jxy, 0.0))
    l1 = 0.5 * trace + disc          # gradient variance across the band normal
    l2 = 0.5 * trace - disc
    # eigenvector of l1 = band normal; band tangent is perpendicular to it
    if abs(jxy) > 1e-15:
        normal = np.array([l1 - jyy, jxy], dtype=np.float64)
    else:                              # axis-aligned field
        normal = np.array([1.0, 0.0]) if jxx >= jyy else np.array([0.0, 1.0])
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    angle = math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))
    return {
        "angle_deg": float(angle) % 180.0,
        "folded_angle_deg": fold_angle_deg(angle),
        "coherence": float((l1 - l2) / l1) if l1 > 0.0 else 0.0,
        "lambda_max": float(l1),
        "lambda_min": float(l2),
        "sigma_cells": float(sigma),
        "cells_populated": int(mask.sum()),
    }


def measure_band_structure_tensor(x_ref: np.ndarray, x_now: np.ndarray, *,
                                  bounds: Sequence[float] | None = None,
                                  k: int = 8, cell: float | None = None,
                                  sigma: float = 1.5) -> dict[str, Any]:
    """Measure the shear-band inclination with the structure tensor.

    Same input contract as :func:`tensordem.biaxial_shear_band.measure_band`
    (``bounds`` = ``(xmin, xmax, ymin, ymax)`` measurement window), so both
    operators can be applied to one snapshot pair and compared directly.
    """
    ref = np.asarray(x_ref, dtype=np.float64)
    now = np.asarray(x_now, dtype=np.float64)
    if bounds is not None:
        xmin, xmax, ymin, ymax = (float(v) for v in bounds)
        inside = ((ref[:, 0] >= xmin) & (ref[:, 0] <= xmax)
                  & (ref[:, 1] >= ymin) & (ref[:, 1] <= ymax))
    else:
        inside = np.ones(ref.shape[0], dtype=bool)
    if int(inside.sum()) < 30:
        raise BandMeasurementError("too few particles inside the measurement window")
    gamma = local_shear_strain(ref[inside], now[inside], k=k)
    # a homogeneous deformation leaves only the float round-off of the local
    # affine fit (|gamma| ~ 1e-16 in double precision), so a field whose
    # amplitude is below 1e-10 carries no band at all: refuse instead of
    # orienting numerical noise
    if (not np.isfinite(gamma).all() or gamma.mean() <= 0.0
            or float(np.max(np.abs(gamma))) < 1e-10):
        raise BandMeasurementError("local strain field is degenerate")
    window = (None if bounds is None else
              (xmin, xmax, ymin, ymax))
    grid = grid_scalar_field(ref[inside, 0], ref[inside, 1], gamma,
                             cell=cell, bounds=window)
    out = structure_tensor_band_angle(grid["field"], grid["occupancy"], sigma=sigma)
    out.update({"cell": grid["cell"], "cells": grid["cells"],
                "particles": int(inside.sum()),
                "localisation_index": float(gamma.std() / gamma.mean())})
    return out