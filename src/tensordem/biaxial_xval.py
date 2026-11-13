"""Cross-validation of the biaxial-cell measurement operators against OSS practice.

The case ``CASE-BIAXIAL-SHEAR-BAND`` (``docs/BIAXIAL_SHEAR_BAND.md``,
``docs/CASE_BIAXIAL_SHEAR_BAND_R2.md``) returns an **honest FAIL**: the band
inclination measured by the in-house Radon-focus operator sits ~7 % (primary
resolved run) to ~20 % (paired re-measurement) away from the 60 deg dense-sand
anchor, and one of two resolved runs is outside the 15 % tolerance.  A negative
result is only worth recording if the *measurement* that produces it is trusted,
so this module re-derives the two decisive quantities with **independent,
mainstream implementations**:

1. band inclination -- our production operator is a Radon/projection focus
   (:func:`tensordem.biaxial_shear_band.radon_band_angle`).  The community
   standard for localisation bands (Desrues & Viggiani 2004, *Int. J. Numer.
   Anal. Methods Geomech.* 28:279-321) is the **structure tensor** of the
   strain field: gradients of the (scalar) maximum-shear-strain field are
   smoothed into ``J = G_rho * (grad g (x) grad g)`` and the band direction is
   the eigenvector of the *minor* eigenvalue, i.e. the direction of slowest
   variation across a ridge.  That operator is provided by mature OSS
   (``skimage.feature.structure_tensor``; the same construction in the SPAM /
   muDIC DIC stacks) and is re-implemented here with the ``scipy.ndimage``
   primitives it is built from, because the loongarch64 ``scikit-image`` wheel
   in this environment is ABI-incompatible with the installed ``numpy`` 2.4
   (see ``docs/XVAL_DEM_BIAXIAL_CELL_NEGATIVE_RESULT.md`` section 5.1).
   A second, structurally different OSS-route estimator (second-moment / PCA of
   the high-strain particle set, the ``skimage.measure.regionprops``
   orientation) is reported alongside it so that agreement is not an artefact
   of one implementation.
2. stress -- the case obtains ``sigma_1`` / ``sigma_3`` from **wall tractions**
   (``F_wall / A``).  OSS DEM codes (YADE ``stressTensor``, LAMMPS
   ``compute stress/atom``) report the **internal** Love-Weber (virial) average
   instead: ``sigma = -1/V * sum_c f_ij (x) (x_j - x_i)``.  Both are now
   recorded by :func:`tensordem.biaxial_shear_band.run_biaxial_shear_band`
   (``sigma_*_internal`` in ``stress_path`` and ``stress_definitions`` in the
   result), so the friction angle that drives the negative result can be
   audited against the OSS convention.

Both comparisons are *controlled*: the two orientation estimators consume the
**same** local shear-strain field on the **same** point set, so the only
difference is the orientation operator; the two stress measures are evaluated at
the **same** integration step of the **same** DEM run, so the only difference is
the stress definition.

Nothing here steps the DEM.  Everything is SI.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import interpolate, ndimage

from .biaxial_shear_band import (
    BandMeasurementError,
    fold_angle_deg,
    local_shear_strain,
    radon_band_angle,
)

__all__ = [
    "strain_field_on_grid",
    "structure_tensor_angle",
    "second_moment_angle",
    "orientation_estimators",
    "cross_validate_synthetic",
    "XVAL_TOLERANCE",
]

#: acceptance tolerance of the cross-validation (owner precision standard)
XVAL_TOLERANCE = 0.15


# ---------------------------------------------------------------------------
# gridding (OSS: scipy.interpolate)
# ---------------------------------------------------------------------------
def strain_field_on_grid(x: Sequence[float], y: Sequence[float],
                         values: Sequence[float], *,
                         cells: int = 40) -> dict[str, Any]:
    """Rasterise a particle-borne scalar field onto a regular grid.

    ``scipy.interpolate.griddata`` (linear, nearest fill for the convex-hull
    gaps and the ragged specimen edges) -- the same two-step used by every DIC
    post-processor before a structure tensor is formed.  The grid is square
    (``cells`` in the short direction) so that the operator stays isotropic:
    an anisotropic grid would re-introduce exactly the aspect-ratio bias that
    invalidated the pre-R2 Radon estimator.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if not (xs.shape == ys.shape == v.shape):
        raise BandMeasurementError("x, y and values must have the same length")
    if xs.size < 20:
        raise BandMeasurementError("at least 20 samples are needed")
    if not (np.isfinite(xs).all() and np.isfinite(ys).all() and np.isfinite(v).all()):
        raise BandMeasurementError("samples must be finite")
    width = float(np.ptp(xs))
    height = float(np.ptp(ys))
    if not (width > 0.0 and height > 0.0):
        raise BandMeasurementError("degenerate point set")
    spacing = min(width, height) / max(int(cells), 4)
    nx = max(8, int(round(width / spacing)) + 1)
    ny = max(8, int(round(height / spacing)) + 1)
    gx = np.linspace(xs.min(), xs.max(), nx)
    gy = np.linspace(ys.min(), ys.max(), ny)
    grid_x, grid_y = np.meshgrid(gx, gy)          # row = y, column = x
    field = interpolate.griddata(np.column_stack((xs, ys)), v,
                                 (grid_x, grid_y), method="linear")
    holes = ~np.isfinite(field)
    if holes.any():
        nearest = interpolate.griddata(np.column_stack((xs, ys)), v,
                                      (grid_x[holes], grid_y[holes]),
                                      method="nearest")
        field[holes] = nearest
    if not np.isfinite(field).all():                       # pragma: no cover
        raise BandMeasurementError("could not fill the grid")
    return {"field": field, "x": gx, "y": gy, "spacing": spacing,
            "nx": nx, "ny": ny}


# ---------------------------------------------------------------------------
# (1a) structure-tensor orientation (Desrues & Viggiani / OSS route)
# ---------------------------------------------------------------------------
def structure_tensor_angle(x: Sequence[float], y: Sequence[float],
                           values: Sequence[float], *,
                           cells: int = 40, sigma_gradient: float = 1.0,
                           sigma_tensor: float = 2.0,
                           weight_quantile: float = 0.80) -> dict[str, Any]:
    """Band inclination (deg from the horizontal ``sigma_3`` axis) from ``J``.

    Steps (each one an OSS ``scipy.ndimage`` primitive, i.e. exactly what
    ``skimage.feature.structure_tensor`` wraps):

    * ``gaussian_filter`` -- remove the particle-scale scatter of the field;
    * ``sobel`` gradient of the smoothed field;
    * ``J = G * (grad g (x) grad g)`` -- the structure tensor, smoothed with
      ``sigma_tensor`` cells so that it describes a *band*, not a particle;
    * per-cell eigen-decomposition: the major eigenvector of ``J`` is the
      gradient direction, i.e. the **band normal**; the band direction is its
      perpendicular;
    * the reported angle is the median over the cells whose tensor energy
      (``trace J``) is above the ``weight_quantile`` quantile, folded into
      ``[0, 90]`` so that conjugate bands report the same inclination.
    """
    gridded = strain_field_on_grid(x, y, values, cells=cells)
    field = ndimage.gaussian_filter(gridded["field"], sigma_gradient,
                                    mode="nearest")
    grad_y = ndimage.sobel(field, axis=0, mode="nearest")
    grad_x = ndimage.sobel(field, axis=1, mode="nearest")
    jxx = ndimage.gaussian_filter(grad_x * grad_x, sigma_tensor, mode="nearest")
    jxy = ndimage.gaussian_filter(grad_x * grad_y, sigma_tensor, mode="nearest")
    jyy = ndimage.gaussian_filter(grad_y * grad_y, sigma_tensor, mode="nearest")
    energy = jxx + jyy
    finite = np.isfinite(energy)
    if not finite.any() or float(np.nanmax(energy)) <= 0.0:
        raise BandMeasurementError("degenerate structure tensor")
    threshold = float(np.quantile(energy[finite], weight_quantile))
    keep = finite & (energy >= threshold) & (energy > 0.0)
    if int(keep.sum()) < 5:                                # pragma: no cover
        raise BandMeasurementError("too few energetic cells")
    a = jxx[keep]
    b = jxy[keep]
    c = jyy[keep]
    half_trace = 0.5 * (a + c)
    disc = np.sqrt(np.maximum((0.5 * (a - c)) ** 2 + b * b, 0.0))
    lambda_major = half_trace + disc
    # major eigenvector (band normal); degenerate cells fall back to the axis
    use_b = np.abs(b) > 1e-30
    vx = np.where(use_b, b, 0.0)
    vy = np.where(use_b, lambda_major - a, 1.0)
    vx = np.where(use_b, vx, 1.0)
    norm = np.hypot(vx, vy)
    norm[norm == 0.0] = 1.0
    vx, vy = vx / norm, vy / norm
    # band direction = normal rotated by 90 deg
    tx, ty = -vy, vx
    angle = np.degrees(np.arctan2(ty, tx)) % 180.0
    angle = np.where(angle > 90.0, 180.0 - angle, angle)
    median = float(np.median(angle))
    return {
        "angle_deg": median,
        "folded_angle_deg": fold_angle_deg(median),
        "spread_deg": float(np.percentile(angle, 75) - np.percentile(angle, 25)),
        "cells_used": int(keep.sum()),
        "cells_total": int(finite.sum()),
        "grid": {"nx": gridded["nx"], "ny": gridded["ny"],
                 "spacing": gridded["spacing"]},
        "sigma_gradient": float(sigma_gradient),
        "sigma_tensor": float(sigma_tensor),
        "weight_quantile": float(weight_quantile),
        "estimator": "structure_tensor(scipy.ndimage: gaussian_filter+sobel+eigh)",
    }


# ---------------------------------------------------------------------------
# (1b) second-moment orientation of the high-strain set (OSS regionprops route)
# ---------------------------------------------------------------------------
def second_moment_angle(x: Sequence[float], y: Sequence[float],
                        values: Sequence[float], *,
                        weight_quantile: float = 0.80) -> dict[str, Any]:
    """Band inclination from the second moment of the high-strain particles.

    ``skimage.measure.regionprops``-style orientation: take the particles whose
    shear strain exceeds the ``weight_quantile`` quantile, form the (weighted)
    coordinate covariance and report the major axis.  Unlike the structure
    tensor this operator never grids the field and never differentiates it, so
    it is structurally independent of both the Radon focus and ``J``.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if not (xs.shape == ys.shape == v.shape):
        raise BandMeasurementError("x, y and values must have the same length")
    threshold = float(np.quantile(v, weight_quantile))
    keep = v >= threshold
    if int(keep.sum()) < 10:
        raise BandMeasurementError("too few particles above the strain quantile")
    px, py = xs[keep], ys[keep]
    w = v[keep] ** 2
    w = w / w.sum()
    mx = float((w * px).sum())
    my = float((w * py).sum())
    dx, dy = px - mx, py - my
    cov = np.array([[float((w * dx * dx).sum()), float((w * dx * dy).sum())],
                    [float((w * dx * dy).sum()), float((w * dy * dy).sum())]])
    eigval, eigvec = np.linalg.eigh(cov)
    major = eigvec[:, int(np.argmax(eigval))]
    angle = math.degrees(math.atan2(major[1], major[0])) % 180.0
    if angle > 90.0:
        angle = 180.0 - angle
    return {
        "angle_deg": float(angle),
        "folded_angle_deg": fold_angle_deg(float(angle)),
        "particles_used": int(keep.sum()),
        "anisotropy": (float(eigval[1] / eigval[0]) if eigval[0] > 0 else math.inf),
        "weight_quantile": float(weight_quantile),
        "estimator": "second_moment(numpy.linalg.eigh of weighted covariance)",
    }


# ---------------------------------------------------------------------------
# (1c) the two estimators on one shared strain field
# ---------------------------------------------------------------------------
def orientation_estimators(x_ref: np.ndarray, x_now: np.ndarray, *,
                           bounds: Sequence[float] | None = None,
                           k: int = 8, cells: int = 40) -> dict[str, Any]:
    """Run our Radon operator and both OSS-route operators on *one* field."""
    ref = np.asarray(x_ref, dtype=np.float64)
    now = np.asarray(x_now, dtype=np.float64)
    if bounds is not None:
        xmin, xmax, ymin, ymax = (float(v) for v in bounds)
        inside = ((ref[:, 0] >= xmin) & (ref[:, 0] <= xmax)
                  & (ref[:, 1] >= ymin) & (ref[:, 1] <= ymax))
    else:
        inside = np.ones(ref.shape[0], dtype=bool)
    if int(inside.sum()) < 40:
        raise BandMeasurementError("too few particles inside the window")
    gamma = local_shear_strain(ref[inside], now[inside], k=k)
    if not np.isfinite(gamma).all() or float(gamma.mean()) <= 0.0:
        raise BandMeasurementError("local strain field is degenerate")
    px = ref[inside, 0]
    py = ref[inside, 1]
    ours = radon_band_angle(px, py, gamma)
    tensor = structure_tensor_angle(px, py, gamma, cells=cells)
    moment = second_moment_angle(px, py, gamma)
    return {
        "particles": int(inside.sum()),
        "gamma_mean": float(gamma.mean()),
        "localisation_index": float(gamma.std() / gamma.mean()),
        "ours": {"angle_deg": float(ours["angle_deg"]),
                 "folded_angle_deg": float(ours["folded_angle_deg"]),
                 "eta2": float(ours["eta2"]),
                 "z_score": float(ours["z_score"]),
                 "bins": int(ours["bins"])},
        "structure_tensor": tensor,
        "second_moment": moment,
    }


def cross_validate_synthetic(pos: np.ndarray, *,
                             angles_deg: Sequence[float] = (30, 35, 40, 45, 50,
                                                            55, 60, 65, 70, 75),
                             slip: float | None = None,
                             band_width: float | None = None,
                             volumetric_strain: float = 0.0,
                             measurement_window: Sequence[float] | None = None,
                             k: int = 8, cells: int = 40) -> dict[str, Any]:
    """Feed synthetic bands of *known* inclination to every operator.

    The synthetic configuration comes from
    :func:`tensordem.biaxial_band_diagnostics.synthetic_band_configuration`
    (homogeneous isochoric background + finite-width ``tanh`` slip band, i.e. a
    test configuration, not a measurement), so the ground truth is exact and
    the errors below are pure operator errors.
    """
    from .biaxial_band_diagnostics import synthetic_band_configuration

    points = np.asarray(pos, dtype=np.float64)
    if measurement_window is not None:
        xmin, xmax, ymin, ymax = (float(v) for v in measurement_window)
        inside = ((points[:, 0] >= xmin) & (points[:, 0] <= xmax)
                  & (points[:, 1] >= ymin) & (points[:, 1] <= ymax))
        window_points = points[inside]
    else:
        window_points = points
    if window_points.shape[0] < 40:
        raise BandMeasurementError("too few particles in the measurement window")
    spacing = float(np.median(np.sort(np.linalg.norm(
        window_points[:, None, :] - window_points[None, :, :], axis=2), axis=1)[:, 1]))
    height = float(np.ptp(window_points[:, 1]))
    width_band = float(band_width) if band_width is not None else 3.0 * spacing
    slip_value = float(slip) if slip is not None else 0.04 * height

    rows = []
    for angle in angles_deg:
        moved = synthetic_band_configuration(
            points, angle_deg=float(angle), slip=slip_value,
            band_width=width_band, volumetric_strain=volumetric_strain)
        try:
            got = orientation_estimators(points, moved, bounds=measurement_window,
                                         k=k, cells=cells)
        except BandMeasurementError as exc:                # pragma: no cover
            rows.append({"injected_deg": float(angle), "error": str(exc)})
            continue
        ours = got["ours"]["folded_angle_deg"]
        tensor = got["structure_tensor"]["folded_angle_deg"]
        moment = got["second_moment"]["folded_angle_deg"]
        rows.append({
            "injected_deg": float(angle),
            "ours_deg": ours,
            "structure_tensor_deg": tensor,
            "second_moment_deg": moment,
            "ours_abs_error_deg": abs(ours - float(angle)),
            "structure_tensor_abs_error_deg": abs(tensor - float(angle)),
            "second_moment_abs_error_deg": abs(moment - float(angle)),
            "ours_rel_error": abs(ours - float(angle)) / float(angle),
            "structure_tensor_rel_error": abs(tensor - float(angle)) / float(angle),
            "second_moment_rel_error": abs(moment - float(angle)) / float(angle),
            "ours_vs_tensor_rel": (abs(ours - tensor) / tensor if tensor else math.nan),
            "ours_vs_moment_rel": (abs(ours - moment) / moment if moment else math.nan),
            "eta2": got["ours"]["eta2"],
            "z_score": got["ours"]["z_score"],
            "localisation_index": got["localisation_index"],
            "structure_tensor_spread_deg": got["structure_tensor"]["spread_deg"],
            "particles": got["particles"],
        })

    def worst(key: str) -> float:
        vals = [row[key] for row in rows if np.isfinite(row.get(key, math.nan))]
        return float(max(vals)) if vals else math.nan

    summary = {
        "angles_deg": [float(a) for a in angles_deg],
        "rows": rows,
        "slip": slip_value,
        "band_width": width_band,
        "particle_spacing": spacing,
        "measurement_window": list(measurement_window) if measurement_window else None,
        "worst": {
            "ours_abs_error_deg": worst("ours_abs_error_deg"),
            "structure_tensor_abs_error_deg": worst("structure_tensor_abs_error_deg"),
            "second_moment_abs_error_deg": worst("second_moment_abs_error_deg"),
            "ours_rel_error": worst("ours_rel_error"),
            "structure_tensor_rel_error": worst("structure_tensor_rel_error"),
            "second_moment_rel_error": worst("second_moment_rel_error"),
            "ours_vs_tensor_rel": worst("ours_vs_tensor_rel"),
            "ours_vs_moment_rel": worst("ours_vs_moment_rel"),
        },
        "tolerance": XVAL_TOLERANCE,
    }
    onset = summary["worst"]
    summary["criterion"] = {
        "structure_tensor_vs_truth_pass": bool(
            math.isfinite(onset["structure_tensor_rel_error"])
            and onset["structure_tensor_rel_error"] <= XVAL_TOLERANCE),
        "second_moment_vs_truth_pass": bool(
            math.isfinite(onset["second_moment_rel_error"])
            and onset["second_moment_rel_error"] <= XVAL_TOLERANCE),
        "ours_vs_structure_tensor_pass": bool(
            math.isfinite(onset["ours_vs_tensor_rel"])
            and onset["ours_vs_tensor_rel"] <= XVAL_TOLERANCE),
        "ours_vs_second_moment_pass": bool(
            math.isfinite(onset["ours_vs_moment_rel"])
            and onset["ours_vs_moment_rel"] <= XVAL_TOLERANCE),
    }
    return summary