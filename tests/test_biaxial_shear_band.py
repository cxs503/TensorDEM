"""Tests for the biaxial shear-band measurement and cell.

The measurement operator is validated against synthetic bands of *known*
inclination, which is the only way to separate an estimator bias from a
material response: if the operator cannot recover a prescribed 60 deg band it
cannot be used to judge a DEM run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.biaxial_shear_band import (
    BandMeasurementError,
    fold_angle_deg,
    local_shear_strain,
    measure_band,
    radon_band_angle,
    theory_angles_deg,
)

torch.set_num_threads(1)


def synthetic_band(angle_deg: float, *, count: int = 4000, width: float = 0.9,
                   seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Particles on [0, 30]^2 carrying a Gaussian ridge of prescribed angle."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 30.0, count)
    y = rng.uniform(0.0, 30.0, count)
    rad = math.radians(angle_deg)
    s = -x * math.sin(rad) + y * math.cos(rad)
    weight = np.exp(-((s - s.mean()) / width) ** 2)
    return x, y, weight


@pytest.mark.parametrize("angle", [0.0, 15.0, 35.0, 45.0, 52.0, 60.0, 70.0, 85.0])
def test_radon_recovers_synthetic_band(angle):
    x, y, w = synthetic_band(angle)
    out = radon_band_angle(x, y, w, resolution_deg=0.5)
    assert abs(out["folded_angle_deg"] - angle) <= 1.0
    assert out["z_score"] > 5.0
    assert out["eta2"] > 0.6


def test_radon_has_no_angular_bias_from_boundary():
    """The projection must not snap to a box edge for a band near 45 deg."""
    errors = []
    for angle in (40.0, 50.0, 58.0, 62.0, 72.0):
        x, y, w = synthetic_band(angle, seed=11)
        errors.append(abs(radon_band_angle(x, y, w)["folded_angle_deg"] - angle))
    assert max(errors) <= 1.5


def test_radon_detects_conjugate_pair():
    x, y, w1 = synthetic_band(60.0)
    _, _, w2 = synthetic_band(120.0)
    out = radon_band_angle(x, y, np.maximum(w1, w2))
    assert out["conjugate"] is not None
    pair = sorted([out["folded_angle_deg"],
                   fold_angle_deg(out["conjugate"]["angle_deg"])])
    assert pair == pytest.approx([60.0, 60.0], abs=2.0)


def test_radon_rejects_degenerate_input():
    with pytest.raises(BandMeasurementError):
        radon_band_angle([0.0, 1.0], [0.0, 1.0], [1.0, 1.0])
    x, y, _ = synthetic_band(45.0, count=100)
    with pytest.raises(BandMeasurementError):
        radon_band_angle(x, y, np.zeros_like(x))


def test_fold_angle_maps_conjugates_together():
    assert fold_angle_deg(60.0) == pytest.approx(60.0)
    assert fold_angle_deg(120.0) == pytest.approx(60.0)
    assert fold_angle_deg(170.0) == pytest.approx(10.0)
    assert fold_angle_deg(-30.0) == pytest.approx(30.0)


def test_theory_angles_match_textbook_values():
    out = theory_angles_deg(30.0, 0.0)
    assert out["mohr_coulomb"] == pytest.approx(60.0)
    assert out["roscoe"] == pytest.approx(45.0)
    assert out["arthur"] == pytest.approx(52.5)
    out = theory_angles_deg(30.0, 10.0)
    assert out["roscoe"] == pytest.approx(50.0)
    assert out["arthur"] == pytest.approx(55.0)


def test_uniform_shear_is_removed_by_the_affine_fit():
    """A homogeneous deformation is *designed* to be removed from the field."""
    rng = np.random.default_rng(3)
    ref = rng.uniform(0.0, 10.0, (400, 2))
    grad = np.array([[0.0, 0.04], [0.0, 0.0]])
    now = ref + ref @ grad
    gamma = local_shear_strain(ref, now, k=8)
    assert np.max(np.abs(gamma)) < 1e-6


def test_local_shear_strain_sees_a_localised_shear_layer():
    """A tanh shear layer (finite width) survives the global affine removal."""
    rng = np.random.default_rng(5)
    ref = rng.uniform(0.0, 10.0, (800, 2))
    now = ref.copy()
    now[:, 1] += 0.03 * np.tanh((ref[:, 0] - 5.0) / 0.25)
    gamma = local_shear_strain(ref, now, k=8)
    layer = np.abs(ref[:, 0] - 5.0) < 0.5
    assert layer.sum() > 20
    assert gamma[layer].mean() > 5.0 * np.median(gamma[~layer])
    assert gamma[layer].mean() > 1e-3
    assert layer[int(np.argmax(gamma))]
