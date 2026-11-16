"""Tests for the biaxial operator adjudication (``tensordem.biaxial_operator_audit``).

The point of the module is to decide *which* orientation operator should be
believed on a real late-window field, so the tests pin the properties the
adjudication rests on:

* a known-angle ridge injected on a lattice is recovered by **every** operator
  (the operators are correct on a clean field, so an error on a real field is
  attributable to the background, not to the operator math);
* the contamination mix behaves as a convex blend (endpoints exactly the two
  components), so an audit at ``c = 1`` measures the operator, not the mix;
* the planted audit is unbiased at ``c = 1`` and degrades as the real
  background is added back in (that degradation is the quantity the audit
  exists to measure);
* bias correction removes a known constant bias;
* the three classical laws evaluate to ``45 + phi/2``, ``45 + psi/2``,
  ``45 + (phi + psi)/4`` and the tolerance flags follow the case tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.biaxial_operator_audit import (
    OPERATORS,
    TOLERANCE,
    band_ridge_weight,
    bias_corrected_angle,
    law_test,
    lattice_spacing,
    mixed_field,
    operator_angles,
    planted_field_audit,
    signed_angle_diff_deg,
)
from tensordem.biaxial_shear_band import BandMeasurementError


def _lattice(columns: int = 24, rows: int = 48, spacing: float = 0.01,
             jitter: float = 5e-4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(columns), np.arange(rows))
    x = gx.ravel() * spacing + rng.normal(0.0, jitter, columns * rows)
    y = gy.ravel() * spacing + rng.normal(0.0, jitter, columns * rows)
    return np.column_stack([x, y])


@pytest.mark.parametrize("angle", [35.0, 45.0, 60.0, 70.0])
def test_every_operator_recovers_a_clean_planted_band(angle: float) -> None:
    pos = _lattice()
    ridge = band_ridge_weight(pos, angle_deg=angle,
                              half_width=2.5 * lattice_spacing(pos))
    out = operator_angles(pos, ridge)
    for name in OPERATORS:
        entry = out[name]
        assert "error" not in entry, (name, entry)
        assert abs(entry["angle_deg"] - angle) < 3.0, (name, angle, entry)


def test_mixed_field_endpoints_are_the_two_components() -> None:
    rng = np.random.default_rng(1)
    real = rng.normal(0.0, 2.5, 500)
    ridge = rng.normal(0.0, 0.3, 500)
    a = mixed_field(real, ridge, contamination=1.0)
    b = mixed_field(real, ridge, contamination=0.0)
    assert np.allclose(a, (ridge - ridge.mean()) / ridge.std())
    assert np.allclose(b, (real - real.mean()) / real.std())
    mid = mixed_field(real, ridge, contamination=0.5)
    assert np.allclose(mid, 0.5 * a + 0.5 * b)


def test_planted_audit_is_unbiased_on_a_clean_field_and_degrades_with_background() -> None:
    pos = _lattice()
    # a "real" field: a band at 55 deg plus strong noise (a stand-in for the
    # diffuse background of a specimen)
    rng = np.random.default_rng(2)
    ridge_real = band_ridge_weight(pos, angle_deg=55.0,
                                   half_width=2.5 * lattice_spacing(pos))
    gamma_real = 0.4 * ridge_real + rng.normal(0.0, 0.6, pos.shape[0])

    audit = planted_field_audit(pos, gamma_real, angles_deg=(40.0, 55.0, 70.0),
                               contaminations=(1.0, 0.5),
                               half_width=2.5 * lattice_spacing(pos))
    for name in OPERATORS:
        clean = audit["summary"][name]["1.00"]
        mixed = audit["summary"][name]["0.50"]
        assert clean["bias_abs_max_deg"] < 3.0, (name, clean)
        # the contaminated audit must be no better than the clean one: that is
        # what makes the c = 1 curve an operator property
        assert mixed["rmse_deg"] >= clean["rmse_deg"] - 1e-9, (name, clean, mixed)


def test_bias_correction_removes_a_known_constant_bias() -> None:
    audit = {"rows": [
        {"contamination": 1.0, "angle_true_deg": a, "radon_error": 3.0}
        for a in (40.0, 50.0, 60.0, 70.0)
    ], "summary": {"radon": {"1.00": {"n": 4}}}}
    out = bias_corrected_angle(57.0, audit, operator="radon", contamination=1.0)
    assert out["bias_deg"] == pytest.approx(3.0)
    assert out["corrected_deg"] == pytest.approx(54.0)


def test_signed_angle_difference_respects_the_conjugate_fold() -> None:
    assert signed_angle_diff_deg(35.0, 60.0) == pytest.approx(-25.0)
    assert signed_angle_diff_deg(65.0, 60.0) == pytest.approx(5.0)
    # 5 deg and 175 deg are the same inclination: the mistake is 10 deg, not 170
    assert signed_angle_diff_deg(5.0, 175.0) == pytest.approx(10.0)


def test_law_test_arithmetic_and_tolerance() -> None:
    out = law_test(57.0, phi_deg=35.46, psi_band_deg=30.4)
    assert out["predictions_deg"]["mohr_coulomb"] == pytest.approx(45.0 + 35.46 / 2)
    assert out["predictions_deg"]["roscoe"] == pytest.approx(45.0 + 30.4 / 2)
    assert out["predictions_deg"]["arthur"] == pytest.approx(
        45.0 + (35.46 + 30.4) / 4)
    for name in ("mohr_coulomb", "roscoe", "arthur"):
        predicted = out["predictions_deg"][name]
        assert out["relative_error"][name] == pytest.approx(
            abs(57.0 - predicted) / predicted)
    assert out["anchor_within_tolerance"] == (
        out["anchor_relative_error"] <= TOLERANCE)
    # a band angle 40 % away from every prediction must be outside tolerance
    bad = law_test(30.0, phi_deg=35.0, psi_band_deg=0.0)
    assert bad["laws_within_tolerance"] == []


def test_degenerate_inputs_raise() -> None:
    pos = _lattice(columns=8, rows=8)
    with pytest.raises(BandMeasurementError):
        mixed_field(np.zeros(10), np.ones(10), contamination=1.0)
    with pytest.raises(BandMeasurementError):
        mixed_field(np.arange(10.0), np.arange(10.0), contamination=1.5)
    with pytest.raises(BandMeasurementError):
        band_ridge_weight(pos, angle_deg=45.0, half_width=-1.0)
    with pytest.raises(BandMeasurementError):
        bias_corrected_angle(50.0, {"rows": [], "summary": {}}, operator="radon")
    with pytest.raises(BandMeasurementError):
        bias_corrected_angle(50.0, {"rows": [], "summary": {}}, operator="unknown")
    with pytest.raises(BandMeasurementError):
        law_test(math.nan, phi_deg=30.0, psi_band_deg=10.0)


def test_lattice_spacing_of_a_regular_grid() -> None:
    gx, gy = np.meshgrid(np.arange(10) * 0.02, np.arange(20) * 0.02)
    pos = np.column_stack([gx.ravel(), gy.ravel()])
    # area / count = (0.18 * 0.38) / 200 = 0.000342 -> spacing 0.0185, i.e. the
    # grid spacing to within the half-spacing edge correction
    assert lattice_spacing(pos) == pytest.approx(0.02, rel=0.1)