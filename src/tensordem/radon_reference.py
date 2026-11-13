"""Independent, OSS-derived reference implementation of the Radon projection.

Why this module exists
----------------------
TensorDEM measures a shear-band inclination with a *scatter-point* Radon
projection (:func:`tensordem.biaxial_shear_band.radon_band_angle`): for every
candidate normal the particle weights are projected onto the normal and binned,
and the between-bin weighted variance fraction ``eta2`` is maximised.  That
estimator has no external counterpart in the repository, so an error there is
invisible to every self-comparison.

This module provides the *other* well-established way of doing the same
measurement, taken from a mature open-source implementation:

    scikit-image ``skimage.transform.radon`` (BSD-3-Clause)
    file  : ``skimage/transform/radon_transform.py``
    sha256: ``a821943a3cd1a7d1b992f38d3a25708bdba25b96b172b561dafc4f9c83f91040``
    code  : ``radon(image, theta, circle=False)`` pads the image to a square of
            the diagonal size, then for every angle ``theta``:
            ``R = [[cos, sin, -c*(cos+sin-1)], [-sin, cos, -c*(cos-sin-1)], [0,0,1]]``
            ``rotated = warp(padded_image, R, clip=False)``
            ``radon_image[:, i] = rotated.sum(0)``

i.e. *rasterise → rotate by spline interpolation about the image centre →
sum along one axis → one projection profile per angle*.  The orientation is
then read off the sinogram with the classical projection-focus criterion
(maximum variance / minimum entropy of the projection profile; the same family
as our ``eta2``, see ``desrues_viggiani_projection_criterion``).

Boundary conditions that this reproduction keeps from the OSS source
--------------------------------------------------------------------
* the padding is ``circle=False`` (pad to ``ceil(sqrt(2)*max(shape))``), so the
  projection length is the same for every angle and ``var(projection)`` is
  comparable across angles;
* the rotation centre is the image centre and the image is sampled on an
  **isotropic** grid (one ``pixel_size`` in *both* axes).  This is a hard
  requirement: an anisotropic pixel grid is exactly the defect that
  ``radon_band_angle`` documents in its own history (an independent min-max
  rescaling of x and y reports ``tan(theta) = (W/H) tan(theta_true)``);
* linear (order=1) interpolation, which is what ``skimage.transform.warp``
  uses by default for float images.

Local deviation from the OSS binary (recorded, not hidden)
---------------------------------------------------------
``import skimage`` fails in this environment (``numpy.dtype size changed``,
skimage's compiled extensions are incompatible with the installed numpy), so
``skimage.transform.warp`` cannot be called.  :func:`radon_sinogram` therefore
applies the *same* matrix ``R`` with ``scipy.ndimage.affine_transform``
(``order=1``, ``mode='constant'``), which is the same affine-spline resampling
``warp`` delegates to.  Equivalence is checked in
``tests/test_radon_reference.py`` against exact sinograms (a centred delta and
a disc, whose projections are known in closed form) and against the analytic
rotation invariance.

Angle convention
----------------
``radon_projection_angle`` returns the inclination, in degrees from the +x axis
of the *band ridge*, using the same convention as
:func:`tensordem.biaxial_shear_band.radon_band_angle`; the mapping between the
sinogram column index (the angle actually passed to the transform) and the ridge
inclination is fixed in :func:`_ridge_angle_from_sinogram_angle` and is verified
by ``tests/test_radon_reference.py`` on a synthetic stripe of known inclination.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

__all__ = [
    "OSS_SOURCE",
    "rasterize_weights",
    "radon_sinogram",
    "projection_scores",
    "all_projection_scores",
    "radon_projection_angle",
    "desrues_viggiani_projection_criterion",
]

OSS_SOURCE = {
    "package": "scikit-image",
    "module": "skimage.transform.radon_transform.radon",
    "license": "BSD-3-Clause",
    "file": "skimage/transform/radon_transform.py",
    "sha256": "a821943a3cd1a7d1b992f38d3a25708bdba25b96b172b561dafc4f9c83f91040",
    "construction": "pad-to-diagonal square, rotate about centre (spline/linear), sum axis 0",
    "local_substitution": "scipy.ndimage.affine_transform(order=1) instead of skimage.transform.warp "
                          "(skimage C extension binary-incompatible with installed numpy)",
}


# ---------------------------------------------------------------------------
# particle cloud -> isotropic raster
# ---------------------------------------------------------------------------
def rasterize_weights(x: Sequence[float], y: Sequence[float],
                      weight: Sequence[float], *, pixel_size: float,
                      pad_pixels: int = 2,
                      deposit_sigma: float = 0.0) -> dict[str, Any]:
    """Splat a weighted point cloud onto an isotropic raster (bilinear).

    The raster is sampled with the *same* ``pixel_size`` in both axes, which is
    the boundary condition the OSS Radon transform needs.  Bilinear splatting
    (each sample is split over its 4 surrounding pixel centres) is the standard
    particle-to-grid deposition and is the only interpolation this reference
    adds before the transform; the transform itself then resamples with linear
    interpolation, i.e. twice linear.  That double smoothing is precisely the
    cost of the image-based route and is quantified in the XVAL report.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    ws = np.asarray(weight, dtype=np.float64).ravel()
    if not (xs.shape == ys.shape == ws.shape):
        raise ValueError("x, y, weight must have the same length")
    h = float(pixel_size)
    if not (h > 0.0 and math.isfinite(h)):
        raise ValueError("pixel_size must be positive and finite")
    x0, y0 = float(xs.min()), float(ys.min())
    nx = int(math.ceil((xs.max() - x0) / h)) + 1 + 2 * int(pad_pixels)
    ny = int(math.ceil((ys.max() - y0) / h)) + 1 + 2 * int(pad_pixels)
    fx = (xs - x0) / h + int(pad_pixels)
    fy = (ys - y0) / h + int(pad_pixels)
    ix = np.floor(fx)
    iy = np.floor(fy)
    tx = fx - ix
    ty = fy - iy
    image = np.zeros((ny, nx), dtype=np.float64)
    for dx, dy, f in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                      (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
        np.add.at(image, ((iy + dy).astype(np.int64), (ix + dx).astype(np.int64)),
                  ws * f)
    if deposit_sigma and float(deposit_sigma) > 0.0:
        image = ndimage.gaussian_filter(image, sigma=float(deposit_sigma) / h,
                                        mode="constant", truncate=4.0)
    return {
        "image": image,
        "pixel_size": h,
        "origin": (x0 - int(pad_pixels) * h, y0 - int(pad_pixels) * h),
        "deposit_sigma": float(deposit_sigma),
        "extent_physical": ((nx - 1) * h, (ny - 1) * h),
    }


# ---------------------------------------------------------------------------
# OSS Radon transform
# ---------------------------------------------------------------------------
def radon_sinogram(image: np.ndarray, theta_deg: Sequence[float]) -> np.ndarray:
    """Sinogram of ``image`` — faithful reproduction of ``skimage.transform.radon``.

    ``image`` is ``(rows=y, cols=x)`` with isotropic sampling.  Returns an array
    of shape ``(padded_side, len(theta_deg))``; column ``i`` is
    ``rotate(image, theta_i).sum(axis=0)`` as in the OSS source.

    Padding follows ``circle=False``: pad to a square of side
    ``ceil(sqrt(2) * max(shape))``, which makes every projection the same
    length (otherwise ``var(projection)`` would not be comparable across
    angles).
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image must be 2-D")
    sides = np.array(img.shape, dtype=np.int64)
    diagonal = math.sqrt(2.0) * float(sides.max())
    pad = [int(math.ceil(diagonal - s)) for s in sides]
    new_center = [(s + p) // 2 for s, p in zip(sides, pad)]
    old_center = [s // 2 for s in sides]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    padded = np.pad(img, pad_width, mode="constant", constant_values=0.0)
    if padded.shape[0] != padded.shape[1]:
        raise ValueError("padded image must be square")
    centre = padded.shape[0] // 2
    theta = np.asarray(theta_deg, dtype=np.float64).ravel()
    out = np.zeros((padded.shape[0], theta.size), dtype=np.float64)
    ones = np.ones(padded.shape, dtype=np.float64)
    for i, ang in enumerate(np.deg2rad(theta)):
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        # identical matrix to the OSS source: output coords -> input coords
        matrix = np.array([
            [cos_a, sin_a, -centre * (cos_a + sin_a - 1.0)],
            [-sin_a, cos_a, -centre * (cos_a - sin_a - 1.0)],
        ])
        rotated = ndimage.affine_transform(
            padded, matrix[:2, :2], offset=matrix[:2, 2], order=1,
            mode="constant", cval=0.0, output_shape=padded.shape)
        out[:, i] = rotated.sum(axis=0)
    return out


def projection_scores(sinogram: np.ndarray, criterion: str = "energy") -> np.ndarray:
    """Per-angle projection-focus score of a sinogram (larger = better focus).

    ``energy``    (default) ``sum(p_hat**2)`` of the *normalised* profile — the
                  squared L2 norm of the density; a planar band collapses into
                  a few bins and concentrates the profile, giving the largest
                  value.  This is the scale-free analogue of our ``eta2``
                  (both are "how much of the field can be explained by a single
                  projection bin"); unlike ``var(projection)`` it does not grow
                  with the *spread* of the profile.
    ``entropy``   *negative* Shannon entropy of the normalised profile, so that
                  argmax = minimum entropy ("minimum-entropy" convention).
    ``peak``      maximum of the normalised profile.
    ``variance``  ``var(projection)`` in absolute units.  Kept for the record:
                  it is **not** scale free — a fixed mass spread over a wider
                  support has a larger variance, so the criterion rewards
                  smearing rather than collapse (measured in the XVAL report).
    """
    p = np.asarray(sinogram, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("sinogram must be 2-D (projection, angle)")
    total = p.sum(axis=0, keepdims=True)
    norm = np.divide(p, total, out=np.zeros_like(p), where=total > 0.0)
    if criterion == "energy":
        return np.sum(norm ** 2, axis=0)
    if criterion == "entropy":
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(norm > 0.0, np.log(np.where(norm > 0.0, norm, 1.0)), 0.0)
        return np.sum(norm * logp, axis=0)          # = -H, so argmax minimises H
    if criterion == "peak":
        return norm.max(axis=0)
    if criterion == "variance":
        return p.var(axis=0)
    raise ValueError(f"unknown criterion {criterion!r}")


def all_projection_scores(sinogram: np.ndarray) -> dict[str, np.ndarray]:
    """All four criteria in one pass, so a sensitivity study costs one transform."""
    return {name: projection_scores(sinogram, name)
            for name in ("energy", "entropy", "peak", "variance")}


def _ridge_angle_from_sinogram_angle(theta_deg: float) -> float:
    """Map the sinogram angle to the ridge inclination (deg from +x).

    The OSS transform sums along rows, i.e. after rotating the raster by
    ``theta`` it integrates along the rotated y-axis and keeps the profile as a
    function of the rotated x-axis.  The profile therefore collapses when the
    rotated x-axis lies along the band *normal*, i.e. when
    ``theta_sinogram = theta_ridge + 90`` -- the measured convention (synthetic
    stripes of 30/45/60/70/80 deg -> sinogram peaks at 120/135/150/160/170) is
    asserted in ``tests/test_radon_reference.py`` rather than trusted.
    """
    return (float(theta_deg) - 90.0) % 180.0


def radon_projection_angle(x: Sequence[float], y: Sequence[float],
                           weight: Sequence[float], *, pixel_size: float,
                           resolution_deg: float = 1.0,
                           criterion: str = "energy",
                           deposit_sigma: float = 0.0) -> dict[str, Any]:
    """Band-ridge inclination from the OSS image-domain Radon projection.

    Returns ``angle_deg`` (ridge inclination from +x, folded to ``[0, 180)``),
    the winning score, the whole score curve and the raster metadata so a run
    can be audited without re-deriving the grid.
    """
    raster = rasterize_weights(x, y, weight, pixel_size=pixel_size,
                               deposit_sigma=deposit_sigma)
    theta = np.arange(0.0, 180.0, float(resolution_deg))
    sino = radon_sinogram(raster["image"], theta)
    table = all_projection_scores(sino)
    scores = table[criterion]
    best = int(np.argmax(scores))
    return {
        "angle_deg": _ridge_angle_from_sinogram_angle(float(theta[best])),
        "sinogram_angle_deg": float(theta[best]),
        "score": float(scores[best]),
        "criterion": criterion,
        "criteria_angles_deg": {
            name: _ridge_angle_from_sinogram_angle(float(theta[int(np.argmax(v))]))
            for name, v in table.items()},
        "resolution_deg": float(resolution_deg),
        "pixel_size": float(raster["pixel_size"]),
        "deposit_sigma": float(deposit_sigma),
        "raster_shape": list(raster["image"].shape),
        "theta_deg": theta.tolist(),
        "scores": scores.tolist(),
        "oss_source": OSS_SOURCE,
    }


def desrues_viggiani_projection_criterion(theta_deg: float, eta2: float) -> dict[str, float]:
    """Diagnostic bridge between the two estimators' focus functionals.

    ``eta2`` (our estimator) is the *between-bin* weighted variance fraction of
    the projection, ``var(projection)`` (reference) is the *total* variance of
    the interpolated projection; by the law of total variance
    ``var_total = var_between + var_within`` restricted to a fixed binning, so
    both criteria peak at the same angle when the bin width is small compared
    with the ridge width and the occupancy is angle independent.  This helper
    only exposes the reference values, it does not make any measurement.
    """
    return {"angle_deg": float(theta_deg), "eta2": float(eta2)}