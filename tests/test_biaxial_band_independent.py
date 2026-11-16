"""Tests for the R8 independent band-orientation estimators.

The published estimator of CASE-BIAXIAL-SHEAR-BAND is validated elsewhere
(``tests/test_biaxial_shear_band.py``).  These tests pin the two independent
operators to *known* inclinations on the specimen geometry class, pin the
documented failure regime of the spectral operator (conjugate localisation), and
pin the error paths, so the R8 comparison cannot silently measure itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.biaxial_band_independent import (
    BandMeasurementError,
    band_model_fit,
    measure_band_independent,
    spectral_band_angle,
    synthetic_band_field,
    synthetic_lattice,
)

SPACING = 1.5


@pytest.fixture(scope="module")
def lattice() -> np.ndarray:
    return synthetic_lattice(columns=24, rows=48, spacing=SPACING, seed=3)


@pytest.mark.parametrize("angle_deg", [15.0, 30.0, 45.0, 60.0, 75.0])
def test_model_fit_recovers_single_band(lattice, angle_deg):
    field = synthetic_band_field(lattice, angle_deg=angle_deg, half_width=4.0,
                                 noise=0.05, seed=5)
    out = band_model_fit(lattice[:, 0], lattice[:, 1], field)
    assert abs(out["folded_angle_deg"] - angle_deg) <= 3.0
    assert out["r2"] > 0.8
    assert out["conjugate_ratio"] < 0.5           # one ridge, not an X


@pytest.mark.parametrize("angle_deg", [30.0, 52.0, 60.0])
def test_model_fit_survives_conjugate_pair(lattice, angle_deg):
    """X-shaped localisation: the conjugate term keeps the operator unbiased.

    This is the regime the published Radon focus loses (R8 envelope: MAE up to
    19.5 deg there), so the assertion is on the model fit, not on Radon.
    """
    field = synthetic_band_field(lattice, angle_deg=angle_deg, half_width=4.0,
                                 noise=0.05, conjugate_deg=180.0 - angle_deg,
                                 conjugate_fraction=1.0, seed=7)
    out = band_model_fit(lattice[:, 0], lattice[:, 1], field)
    assert abs(out["folded_angle_deg"] - angle_deg) <= 3.0
    assert out["conjugate_ratio"] > 0.7           # symmetric X recognised


def test_model_fit_is_translation_invariant(lattice):
    angle_deg = 52.0
    field = synthetic_band_field(lattice, angle_deg=angle_deg, half_width=4.0, seed=1)
    base = band_model_fit(lattice[:, 0], lattice[:, 1], field)["folded_angle_deg"]
    shifted = lattice + np.array([137.0, -241.0])
    moved = band_model_fit(shifted[:, 0], shifted[:, 1], field)["folded_angle_deg"]
    assert abs(base - moved) <= 1.0


def test_spectral_recovers_single_band(lattice):
    angle_deg = 45.0
    field = synthetic_band_field(lattice, angle_deg=angle_deg, half_width=4.0,
                                 noise=0.05, seed=2)
    out = spectral_band_angle(lattice[:, 0], lattice[:, 1], field)
    assert abs(out["folded_angle_deg"] - angle_deg) <= 8.0
    assert out["modes"] >= 6


def test_spectral_degrades_on_conjugate_pair(lattice):
    """Documented limitation: the spectral inertia is isotropic for an X."""
    angle_deg = 45.0
    single = synthetic_band_field(lattice, angle_deg=angle_deg, half_width=4.0,
                                  seed=2)
    double = synthetic_band_field(lattice, angle_deg=angle_deg, half_width=4.0,
                                  conjugate_deg=135.0, conjugate_fraction=1.0,
                                  seed=2)
    err_single = abs(spectral_band_angle(lattice[:, 0], lattice[:, 1],
                                        single)["folded_angle_deg"] - angle_deg)
    err_double = abs(spectral_band_angle(lattice[:, 0], lattice[:, 1],
                                        double)["folded_angle_deg"] - angle_deg)
    assert err_double > err_single


def test_measure_band_independent_recovers_displacement_band():
    """A slip field localised on a band must be measured at its inclination."""
    ref = synthetic_lattice(columns=20, rows=40, seed=11)
    angle_deg = 60.0
    centre = ref.mean(axis=0)
    normal = np.array([-math.sin(math.radians(angle_deg)),
                       math.cos(math.radians(angle_deg))])
    tangent = np.array([math.cos(math.radians(angle_deg)),
                        math.sin(math.radians(angle_deg))])
    u = (ref - centre) @ normal
    profile = np.tanh(u / 2.0)                    # smooth slip across the band
    now = ref + 0.6 * profile[:, None] * tangent[None, :]
    out = measure_band_independent(ref, now)
    assert abs(out["folded_angles_deg"]["radon"] - angle_deg) <= 4.0
    assert abs(out["folded_angles_deg"]["model"] - angle_deg) <= 4.0
    assert out["particles"] == ref.shape[0]


@pytest.mark.parametrize("bad", [
    (np.zeros(10), np.zeros(10), np.ones(10)),                 # too few samples
    (np.zeros(40), np.zeros(39), np.ones(40)),                 # length mismatch
])
def test_sample_validation(bad):
    with pytest.raises(BandMeasurementError):
        band_model_fit(*bad)


def test_non_finite_and_negative_weights_rejected():
    x = np.linspace(0.0, 10.0, 64)
    y = np.linspace(0.0, 20.0, 64)
    w = np.ones(64)
    w_bad = w.copy()
    w_bad[3] = np.nan
    with pytest.raises(BandMeasurementError):
        band_model_fit(x, y, w_bad)
    w_neg = w.copy()
    w_neg[3] = -1.0
    with pytest.raises(BandMeasurementError):
        spectral_band_angle(x, y, w_neg)


def test_zero_variance_field_rejected(lattice):
    with pytest.raises(BandMeasurementError):
        band_model_fit(lattice[:, 0], lattice[:, 1], np.ones(lattice.shape[0]))