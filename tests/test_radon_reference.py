"""Tests for the OSS-derived reference Radon projection measurement.

These tests do four things, in increasing order of strength:

1. *closed form* -- the sinogram of a centred delta is a delta at the image
   centre for every angle, and a disc projects to a profile whose width is
   angle independent (both are exactly true of the continuous Radon transform,
   so they catch a wrong matrix, a wrong padding or a wrong summation axis);
2. *convention* -- a ridge of known inclination peaks at ``theta + 90`` in the
   sinogram and is reported back as the ridge inclination;
3. *agreement* -- on the identical particle field the reference and the
   production scatter-point estimator
   (:func:`tensordem.biaxial_shear_band.radon_band_angle`) return the same
   inclination;
4. *recorded failure mode* -- with no deposition kernel and a raster finer than
   the particle spacing the image route aliases onto the triangular-lattice
   directions (45 deg -> 60 deg).  The test asserts the failure, so nobody can
   "fix" it by silently coarsening something else.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.biaxial_shear_band import radon_band_angle
from tensordem.radon_reference import (
    OSS_SOURCE,
    radon_projection_angle,
    radon_sinogram,
    rasterize_weights,
)


def _delta_image(n: int = 65) -> np.ndarray:
    img = np.zeros((n, n), dtype=np.float64)
    img[n // 2, n // 2] = 1.0
    return img


def _lattice(columns: int, rows: int, *, jitter: float = 0.03, seed: int = 11):
    rng = np.random.default_rng(seed)
    pts = [(c + 0.5 * (r % 2), r * math.sqrt(3.0) / 2.0)
           for r in range(rows) for c in range(columns)]
    pts = np.asarray(pts, dtype=np.float64)
    pts += jitter * rng.standard_normal(pts.shape)
    return pts


def _ridge(points: np.ndarray, angle_deg: float, sigma: float = 2.0) -> np.ndarray:
    rad = math.radians(angle_deg)
    s = -points[:, 0] * math.sin(rad) + points[:, 1] * math.cos(rad)
    return np.exp(-(s / sigma) ** 2)


def test_oss_source_is_cited():
    assert OSS_SOURCE["license"] == "BSD-3-Clause"
    assert len(OSS_SOURCE["sha256"]) == 64


def test_centred_delta_projects_to_the_image_centre_for_every_angle():
    img = _delta_image()
    theta = np.arange(0.0, 180.0, 15.0)
    sino = radon_sinogram(img, theta)
    # the projection of a centred delta is a delta at the image centre: the
    # padding moves the centre to the middle of the padded square, and the peak
    # row must not move with the angle
    diagonal = math.sqrt(2.0) * img.shape[0]
    padded = img.shape[0] + int(math.ceil(diagonal - img.shape[0]))
    expected = img.shape[0] // 2 + (padded // 2 - img.shape[0] // 2)
    peak = np.argmax(sino, axis=0)
    assert np.all(np.abs(peak - expected) <= 1)
    # linear interpolation is NOT mass conserving for an impulse: the rotated
    # sum of a single pixel varies with the angle (recorded, not hidden)
    mass = sino.sum(axis=0)
    assert mass.min() > 0.9 and mass.max() < 1.5


def test_smooth_field_mass_is_conserved_by_the_rotation():
    """For the smooth fields this module is used on, the projection conserves mass."""
    n = 129
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.exp(-(((yy - n // 2) ** 2 + (xx - n // 2) ** 2) / (2 * 20.0 ** 2)))
    theta = np.arange(0.0, 180.0, 15.0)
    sino = radon_sinogram(img, theta)
    assert np.allclose(sino.sum(axis=0), img.sum(), rtol=2e-3)


def test_disc_projection_width_is_angle_independent():
    n, radius = 129, 40
    yy, xx = np.mgrid[0:n, 0:n]
    img = (((yy - n // 2) ** 2 + (xx - n // 2) ** 2) <= radius ** 2).astype(float)
    theta = np.arange(0.0, 180.0, 15.0)
    sino = radon_sinogram(img, theta)
    width = (sino > 0.5 * sino.max(axis=0)).sum(axis=0)
    assert width.min() > 0
    assert (width.max() - width.min()) <= 3          # pixels of the discrete disc
    # a line integral through the *centre* column is 2R long for every angle
    centre = sino.shape[0] // 2
    assert np.allclose(sino[centre], 2 * radius, rtol=0.02)


@pytest.mark.parametrize("angle_deg", [30.0, 45.0, 60.0, 70.0])
def test_ridge_inclination_convention(angle_deg):
    points = _lattice(24, 48)
    weight = _ridge(points, angle_deg)
    out = radon_projection_angle(points[:, 0], points[:, 1], weight,
                                 pixel_size=0.5, deposit_sigma=1.0,
                                 resolution_deg=1.0)
    assert out["angle_deg"] == pytest.approx(angle_deg, abs=2.0)
    # the sinogram column that wins is the ridge *normal* (ridge + 90 deg)
    assert out["sinogram_angle_deg"] == pytest.approx((angle_deg + 90.0) % 180.0, abs=2.0)


@pytest.mark.parametrize("angle_deg", [45.0, 60.0])
def test_reference_agrees_with_production_estimator_on_the_same_field(angle_deg):
    points = _lattice(24, 48)
    weight = _ridge(points, angle_deg)
    ours = radon_band_angle(points[:, 0], points[:, 1], weight, resolution_deg=0.5)
    ref = radon_projection_angle(points[:, 0], points[:, 1], weight,
                                 pixel_size=0.5, deposit_sigma=1.0,
                                 resolution_deg=0.5)
    assert abs(ours["folded_angle_deg"] - ref["angle_deg"]) < 3.0
    # every scale-free criterion of the same sinogram must see the same ridge;
    # raw variance is asserted only loosely (it is not scale free, see the
    # module docstring and the XVAL report)
    for name in ("energy", "peak", "entropy"):
        assert abs(ref["criteria_angles_deg"][name] - angle_deg) < 6.0, name


def test_undeposited_fine_raster_aliases_onto_the_lattice_directions():
    """Recorded failure mode: no deposition kernel + pixel_size << spacing."""
    points = _lattice(24, 48)
    weight = _ridge(points, 45.0)
    raster = rasterize_weights(points[:, 0], points[:, 1], weight,
                               pixel_size=0.125, deposit_sigma=0.0)
    # the raster is a set of isolated dots, not a field
    assert (raster["image"] > 0).mean() < 0.10
    out = radon_projection_angle(points[:, 0], points[:, 1], weight,
                                 pixel_size=0.125, deposit_sigma=0.0,
                                 resolution_deg=2.0)
    assert abs(out["angle_deg"] - 45.0) > 5.0