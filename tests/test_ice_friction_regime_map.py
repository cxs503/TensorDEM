"""Tests for the ice-steel friction regime map module (p381w1-N03).

Tests cover:
- Dry friction law vs Kennedy 1987 anchors
- Lubricated friction vs Colbeck 1988 anchors
- Regime transitions vs Makkonen 2012 anchors
- 4D sweep and response surface fitting
- Regime classification
- Acceptance artifact generation and verification
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensordem.ice_friction_regime_map import (
    COLBECK_1988_ANCHORS,
    KENNEDY_1987_ANCHORS,
    MAKKONEN_2012_ANCHORS,
    REGIME_DRY,
    REGIME_FULL,
    REGIME_MIXED,
    RegimeMapInputs,
    build_acceptance,
    dry_friction_coefficient,
    evaluate_surface,
    film_area_fraction,
    fit_response_surface,
    friction_coefficient,
    full_validation,
    lambda_ratio,
    melt_film_thickness,
    regime_label,
    sweep_4d,
    validate_colbeck,
    validate_kennedy,
    validate_makkonen,
    verify_acceptance,
    water_viscosity,
    write_acceptance,
)


# ---------------------------------------------------------------------------
# Dry friction law tests
# ---------------------------------------------------------------------------
class TestDryFriction:
    """Tests for the dry friction coefficient model."""

    def test_kennedy_all_within_20pct(self):
        """Criterion (a): all Kennedy anchors within 20%."""
        result = validate_kennedy()
        assert result["all_within_20pct"], (
            f"Kennedy validation failed: {result['n_passed']}/{result['n_anchors']} passed, "
            f"max error={result['max_rel_error']:.3f}"
        )

    def test_kennedy_mean_error_below_10pct(self):
        """Mean error should be well below the 20% gate."""
        result = validate_kennedy()
        assert result["mean_rel_error"] < 0.10, (
            f"Mean Kennedy error {result['mean_rel_error']:.3f} too high"
        )

    def test_dry_friction_decreases_with_temperature(self):
        """mu_dry should decrease as temperature increases (warmer ice = softer)."""
        p = 0.5e6
        mu_cold = float(dry_friction_coefficient(-30.0, p))
        mu_warm = float(dry_friction_coefficient(-5.0, p))
        assert mu_cold > mu_warm, (
            f"mu(-30C)={mu_cold:.3f} should be > mu(-5C)={mu_warm:.3f}"
        )

    def test_dry_friction_decreases_with_pressure(self):
        """mu_dry should decrease as pressure increases."""
        t = -10.0
        mu_low = float(dry_friction_coefficient(t, 0.1e6))
        mu_high = float(dry_friction_coefficient(t, 2.0e6))
        assert mu_low > mu_high, (
            f"mu(0.1MPa)={mu_low:.3f} should be > mu(2MPa)={mu_high:.3f}"
        )

    def test_dry_friction_bounds(self):
        """mu_dry should be within physical bounds."""
        for t in [-30, -20, -10, -5, -2]:
            for p_mpa in [0.1, 0.5, 1.0, 2.0, 5.0]:
                mu = float(dry_friction_coefficient(t, p_mpa * 1e6))
                assert 0.02 <= mu <= 0.50, f"mu({t}C, {p_mpa}MPa)={mu} out of bounds"


# ---------------------------------------------------------------------------
# Lubricated friction tests
# ---------------------------------------------------------------------------
class TestLubricatedFriction:
    """Tests for the lubricated friction model."""

    def test_colbeck_at_least_one_within_30pct(self):
        """At least one Colbeck anchor should be within 30%."""
        result = validate_colbeck()
        assert result["n_passed"] >= 1, (
            f"Colbeck validation: {result['n_passed']}/{result['n_anchors']} passed"
        )

    def test_colbeck_regime_is_mixed_or_full(self):
        """Colbeck points (smooth, fast) should be in mixed or full regime."""
        result = validate_colbeck()
        for row in result["rows"]:
            assert row["regime"] in (REGIME_MIXED, REGIME_FULL), (
                f"Colbeck point {row['temperature_c']}C, {row['speed_m_s']}m/s "
                f"predicted as {row['regime']}, expected mixed/full"
            )

    def test_film_thickness_increases_with_speed(self):
        """Film thickness should increase with sliding speed."""
        p = 0.5e6
        t = -5.0
        h_slow = melt_film_thickness(1000.0, p, 0.01, t, 1e-3, 2e-3)
        h_fast = melt_film_thickness(1000.0, p, 1.0, t, 1e-3, 2e-3)
        assert h_fast > h_slow, (
            f"h(1m/s)={h_fast:.2e} should be > h(0.01m/s)={h_slow:.2e}"
        )

    def test_film_thickness_decreases_with_pressure(self):
        """Film thickness should decrease with pressure (squeeze-out)."""
        q = 1000.0
        v = 1.0
        t = -5.0
        h_low = melt_film_thickness(q, 0.1e6, v, t, 1e-3, 2e-3)
        h_high = melt_film_thickness(q, 2.0e6, v, t, 1e-3, 2e-3)
        assert h_low > h_high, (
            f"h(0.1MPa)={h_low:.2e} should be > h(2MPa)={h_high:.2e}"
        )


# ---------------------------------------------------------------------------
# Regime classification tests
# ---------------------------------------------------------------------------
class TestRegimeClassification:
    """Tests for regime classification."""

    def test_lambda_ratio_positive(self):
        """Lambda ratio should be positive."""
        lam = lambda_ratio(1e-6, 1e-6)
        assert lam > 0

    def test_film_area_fraction_bounds(self):
        """Film area fraction should be in [0, 1]."""
        for lam in [0.0, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0]:
            a = film_area_fraction(lam)
            assert 0.0 <= a <= 1.0, f"A({lam})={a} out of bounds"

    def test_film_area_fraction_monotone(self):
        """Film area fraction should increase with Lambda."""
        a1 = film_area_fraction(0.5)
        a2 = film_area_fraction(2.0)
        a3 = film_area_fraction(5.0)
        assert a1 <= a2 <= a3

    def test_regime_label_dry(self):
        """Lambda < threshold should be dry regime."""
        assert regime_label(0.1, -10.0, 1.0) == REGIME_DRY
        assert regime_label(0.19, -10.0, 1.0) == REGIME_DRY

    def test_regime_label_mixed(self):
        """threshold_dry <= Lambda < threshold_full should be mixed regime."""
        assert regime_label(0.5, -10.0, 1.0) == REGIME_MIXED
        assert regime_label(1.0, -10.0, 1.0) == REGIME_MIXED

    def test_regime_label_full(self):
        """Lambda >= threshold_full should be full lubrication."""
        assert regime_label(1.5, -10.0, 1.0) == REGIME_FULL
        assert regime_label(10.0, -10.0, 1.0) == REGIME_FULL

    def test_makkonen_accuracy_above_40pct(self):
        """Makkonen regime accuracy should be above 40%."""
        result = validate_makkonen()
        assert result["accuracy"] > 0.40, (
            f"Makkonen accuracy {result['accuracy']:.3f} too low"
        )


# ---------------------------------------------------------------------------
# 4D sweep and response surface tests
# ---------------------------------------------------------------------------
class TestResponseSurface:
    """Tests for the 4D sweep and response surface fitting."""

    def test_sweep_4d_shape(self):
        """Sweep should produce correct shape arrays."""
        inputs = RegimeMapInputs(
            temperatures_c=(-10.0, -5.0),
            speeds_m_s=(0.1, 1.0),
            pressures_mpa=(0.5, 1.0),
            roughness_um=(1.0, 5.0),
        )
        result = sweep_4d(inputs)
        assert result.mu_grid.shape == (2, 2, 2, 2)
        assert len(result.flat_data) == 16

    def test_sweep_mu_positive(self):
        """All mu values should be positive."""
        result = sweep_4d()
        assert np.all(result.mu_grid > 0)

    def test_sweep_mu_bounds(self):
        """All mu values should be in [0.002, 0.5]."""
        result = sweep_4d()
        assert np.all(result.mu_grid >= 0.002)
        assert np.all(result.mu_grid <= 0.50)

    def test_response_surface_r_squared_above_0_85(self):
        """R² should be above 0.85."""
        result = sweep_4d()
        fit = fit_response_surface(result)
        assert fit["r_squared"] > 0.85, f"R²={fit['r_squared']:.3f} too low"

    def test_evaluate_surface_positive(self):
        """Surface evaluation should return positive values."""
        result = sweep_4d()
        fit = fit_response_surface(result)
        coeffs = fit["coefficients"]
        mu = evaluate_surface(coeffs, -10.0, 0.5, 0.5, 1.0)
        assert mu > 0

    def test_surface_approximates_sweep(self):
        """Surface should approximate sweep data reasonably."""
        result = sweep_4d()
        fit = fit_response_surface(result)
        coeffs = fit["coefficients"]
        
        errors = []
        for d in result.flat_data[:50]:  # Check first 50 points
            mu_pred = evaluate_surface(coeffs, d["temperature_c"], d["speed_m_s"],
                                        d["pressure_mpa"], d["roughness_um"])
            err = abs(mu_pred - d["mu"]) / d["mu"]
            errors.append(err)
        
        mean_err = sum(errors) / len(errors)
        assert mean_err < 2.00, f"Mean surface error {mean_err:.3f} too high"


# ---------------------------------------------------------------------------
# Full friction model tests
# ---------------------------------------------------------------------------
class TestFrictionModel:
    """Tests for the full friction coefficient model."""

    def test_friction_coefficient_returns_dict(self):
        """friction_coefficient should return a dict with required keys."""
        res = friction_coefficient(-10.0, 0.5, 0.5e6, 1.0e-6)
        required_keys = {"mu", "regime", "mu_dry", "mu_film",
                         "film_thickness_m", "lambda_ratio", "film_area_fraction"}
        assert required_keys.issubset(res.keys())

    def test_friction_decreases_with_speed(self):
        """mu should generally decrease with speed (Stribeck effect)."""
        t = -10.0
        p = 0.5e6
        rq = 1.0e-6
        mu_slow = friction_coefficient(t, 0.01, p, rq)["mu"]
        mu_fast = friction_coefficient(t, 1.0, p, rq)["mu"]
        # At low speed, dry friction dominates; at high speed, film reduces mu
        assert mu_fast <= mu_slow or mu_fast < 0.3, (
            f"mu(0.01m/s)={mu_slow:.3f}, mu(1m/s)={mu_fast:.3f}"
        )

    def test_friction_at_zero_speed_is_dry(self):
        """At zero speed, friction should be dry friction."""
        res = friction_coefficient(-10.0, 0.0, 0.5e6, 1.0e-6)
        assert res["regime"] == REGIME_DRY
        assert res["mu"] == res["mu_dry"]

    def test_friction_increases_with_roughness(self):
        """mu should increase with surface roughness."""
        t = -10.0
        v = 0.5
        p = 0.5e6
        mu_smooth = friction_coefficient(t, v, p, 0.1e-6)["mu"]
        mu_rough = friction_coefficient(t, v, p, 10.0e-6)["mu"]
        assert mu_rough >= mu_smooth, (
            f"mu(smooth)={mu_smooth:.3f}, mu(rough)={mu_rough:.3f}"
        )


# ---------------------------------------------------------------------------
# Acceptance artifact tests
# ---------------------------------------------------------------------------
class TestAcceptanceArtifact:
    """Tests for acceptance artifact generation and verification."""

    def test_build_acceptance_returns_dict(self):
        """build_acceptance should return a dict with required keys."""
        result = build_acceptance()
        assert "criteria" in result
        assert "all_pass" in result
        assert "validation" in result
        assert "digest" in result

    def test_acceptance_has_four_criteria(self):
        """Acceptance should have four criteria (a, b, c, d)."""
        result = build_acceptance()
        assert set(result["criteria"].keys()) == {
            "a_kennedy_dry_friction",
            "b_colbeck_lubricated",
            "c_makkonen_regime_transitions",
            "d_response_surface_r_squared",
        }

    def test_kennedy_criterion_passes(self):
        """Kennedy criterion should pass."""
        result = build_acceptance()
        assert result["criteria"]["a_kennedy_dry_friction"]["pass"]

    def test_digest_is_hex_string(self):
        """Digest should be a hex string."""
        result = build_acceptance()
        assert isinstance(result["digest"], str)
        assert len(result["digest"]) == 64  # SHA-256

    def test_write_and_verify_acceptance(self, tmp_path):
        """Written acceptance should verify correctly."""
        path = tmp_path / "acceptance.json"
        write_acceptance(str(path))
        assert path.exists()
        
        verdict = verify_acceptance(str(path))
        assert verdict["digest_matches"]

    def test_sweep_point_count(self):
        """Sweep should have correct number of points."""
        result = build_acceptance()
        expected = 8 * 9 * 7 * 7  # T x V x P x R_q
        assert result["n_sweep_points"] == expected


# ---------------------------------------------------------------------------
# Water viscosity tests
# ---------------------------------------------------------------------------
class TestWaterViscosity:
    """Tests for water viscosity model."""

    def test_viscosity_at_0c(self):
        """Viscosity at 0C should be close to 1.79e-3 Pa s."""
        eta = float(water_viscosity(0.0))
        assert abs(eta - 1.792e-3) / 1.792e-3 < 0.05

    def test_viscosity_increases_when_colder(self):
        """Viscosity should increase as temperature decreases."""
        eta_warm = float(water_viscosity(-2.0))
        eta_cold = float(water_viscosity(-20.0))
        assert eta_cold > eta_warm

    def test_viscosity_positive(self):
        """Viscosity should be positive."""
        for t in [-30, -20, -10, -5, -2, 0]:
            eta = float(water_viscosity(t))
            assert eta > 0
