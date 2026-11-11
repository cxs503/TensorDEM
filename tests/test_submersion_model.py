"""Tests for the submersion and momentum-flux correction model.

This test suite covers:
- Submersion resistance calculation
- Momentum flux calculation
- Correction application
- Concentration scaling
- Physical consistency checks
"""
from __future__ import annotations

import math
import pytest

from tensordem.broken_ice_resistance import STANDARD_CASE, CaseSpec
from tensordem.broken_ice_resistance.submersion_model import (
    SubmersionCorrection,
    apply_correction,
    apply_full_correction,
    compute_momentum_flux,
    compute_submersion_resistance,
    concentration_correction_factor,
)


# --- Basic calculations -----------------------------------------------------

def test_submersion_resistance_positive() -> None:
    """Submersion resistance must be positive for physical cases."""
    R_sub = compute_submersion_resistance(STANDARD_CASE)
    assert R_sub > 0


def test_momentum_flux_positive() -> None:
    """Momentum flux must be positive for physical cases."""
    R_mom = compute_momentum_flux(STANDARD_CASE)
    assert R_mom > 0


def test_submersion_scales_with_thickness_squared() -> None:
    """Submersion resistance scales with h^2."""
    spec1 = CaseSpec(thickness_m=1.0, speed_m_s=3.0, concentration=0.6)
    spec2 = CaseSpec(thickness_m=2.0, speed_m_s=3.0, concentration=0.6)
    
    R_sub1 = compute_submersion_resistance(spec1)
    R_sub2 = compute_submersion_resistance(spec2)
    
    # R_sub ~ h^2, so doubling h should quadruple R_sub
    ratio = R_sub2 / R_sub1
    assert abs(ratio - 4.0) < 0.01


def test_momentum_scales_with_speed_squared() -> None:
    """Momentum flux scales with V^2."""
    spec1 = CaseSpec(thickness_m=1.0, speed_m_s=2.0, concentration=0.6)
    spec2 = CaseSpec(thickness_m=1.0, speed_m_s=4.0, concentration=0.6)
    
    R_mom1 = compute_momentum_flux(spec1)
    R_mom2 = compute_momentum_flux(spec2)
    
    # R_mom ~ V^2, so doubling V should quadruple R_mom
    ratio = R_mom2 / R_mom1
    assert abs(ratio - 4.0) < 0.01


# --- Correction application -------------------------------------------------

def test_correction_increases_resistance() -> None:
    """Applying correction must increase the total resistance."""
    dem_N = 100e3  # 100 kN
    corr = apply_correction(STANDARD_CASE, dem_N)
    
    assert corr.hybrid_total_N > dem_N
    assert corr.total_correction_N > 0


def test_correction_components_sum() -> None:
    """Correction components must sum to total correction."""
    dem_N = 100e3
    corr = apply_correction(STANDARD_CASE, dem_N)
    
    expected_total = corr.submersion_N + corr.momentum_N
    assert abs(corr.total_correction_N - expected_total) < 1e-6


def test_hybrid_total_is_sum() -> None:
    """Hybrid total must be DEM + correction."""
    dem_N = 100e3
    corr = apply_correction(STANDARD_CASE, dem_N)
    
    expected_hybrid = dem_N + corr.total_correction_N
    assert abs(corr.hybrid_total_N - expected_hybrid) < 1e-6


def test_correction_fraction_in_range() -> None:
    """Correction fraction must be in [0, 1]."""
    dem_N = 100e3
    corr = apply_correction(STANDARD_CASE, dem_N)
    
    assert 0.0 <= corr.correction_fraction <= 1.0


# --- Concentration scaling --------------------------------------------------

def test_concentration_factor_unit_at_full_concentration() -> None:
    """Concentration factor must be 1.0 at C=1.0."""
    factor = concentration_correction_factor(1.0)
    assert factor == 1.0


def test_concentration_factor_zero_at_zero_concentration() -> None:
    """Concentration factor must be 0.0 at C=0.0."""
    factor = concentration_correction_factor(0.0)
    assert factor == 0.0


def test_concentration_factor_linear() -> None:
    """Concentration factor must be linear in C."""
    factor_05 = concentration_correction_factor(0.5)
    factor_10 = concentration_correction_factor(1.0)
    
    assert abs(factor_05 - 0.5 * factor_10) < 1e-6


def test_concentration_factor_rejects_invalid() -> None:
    """Concentration factor must reject values outside [0, 1]."""
    with pytest.raises(ValueError):
        concentration_correction_factor(-0.1)
    with pytest.raises(ValueError):
        concentration_correction_factor(1.1)


def test_full_correction_scales_with_concentration() -> None:
    """Full correction must scale with concentration."""
    dem_N = 100e3
    
    spec_full = CaseSpec(concentration=1.0, thickness_m=1.0, speed_m_s=3.0)
    spec_half = CaseSpec(concentration=0.5, thickness_m=1.0, speed_m_s=3.0)
    
    corr_full = apply_full_correction(spec_full, dem_N, apply_concentration=True)
    corr_half = apply_full_correction(spec_half, dem_N, apply_concentration=True)
    
    # Correction at C=0.5 should be half of correction at C=1.0
    # (DEM contact is the same, only correction scales)
    corr_diff_full = corr_full.total_correction_N
    corr_diff_half = corr_half.total_correction_N
    
    assert abs(corr_diff_half - 0.5 * corr_diff_full) < 1e-6


# --- Physical consistency ---------------------------------------------------

def test_zero_dem_gives_pure_correction() -> None:
    """Zero DEM contact should give pure correction."""
    corr = apply_correction(STANDARD_CASE, 0.0)
    
    assert corr.dem_contact_N == 0.0
    assert corr.hybrid_total_N == corr.total_correction_N
    assert corr.correction_fraction == 1.0


def test_large_dem_reduces_correction_fraction() -> None:
    """Large DEM contact should reduce correction fraction."""
    dem_small = 10e3
    dem_large = 1000e3
    
    corr_small = apply_correction(STANDARD_CASE, dem_small)
    corr_large = apply_correction(STANDARD_CASE, dem_large)
    
    assert corr_large.correction_fraction < corr_small.correction_fraction


def test_correction_monotonic_with_thickness() -> None:
    """Correction must increase with ice thickness."""
    spec_thin = CaseSpec(thickness_m=0.5, speed_m_s=3.0, concentration=0.6)
    spec_thick = CaseSpec(thickness_m=1.5, speed_m_s=3.0, concentration=0.6)
    
    dem_N = 100e3
    corr_thin = apply_correction(spec_thin, dem_N)
    corr_thick = apply_correction(spec_thick, dem_N)
    
    assert corr_thick.total_correction_N > corr_thin.total_correction_N


def test_correction_monotonic_with_speed() -> None:
    """Correction must increase with speed (due to momentum term)."""
    spec_slow = CaseSpec(thickness_m=1.0, speed_m_s=2.0, concentration=0.6)
    spec_fast = CaseSpec(thickness_m=1.0, speed_m_s=4.0, concentration=0.6)
    
    dem_N = 100e3
    corr_slow = apply_correction(spec_slow, dem_N)
    corr_fast = apply_correction(spec_fast, dem_N)
    
    assert corr_fast.total_correction_N > corr_slow.total_correction_N


# --- Data structure ---------------------------------------------------------

def test_correction_as_dict() -> None:
    """SubmersionCorrection.as_dict() must return all fields."""
    corr = apply_correction(STANDARD_CASE, 100e3)
    d = corr.as_dict()
    
    required_keys = [
        "submersion_N", "momentum_N", "total_correction_N",
        "dem_contact_N", "hybrid_total_N", "correction_fraction",
    ]
    for key in required_keys:
        assert key in d


def test_correction_is_frozen() -> None:
    """SubmersionCorrection must be immutable."""
    corr = apply_correction(STANDARD_CASE, 100e3)
    
    with pytest.raises(AttributeError):
        corr.submersion_N = 0.0  # type: ignore


# --- Edge cases -------------------------------------------------------------

def test_zero_thickness_raises() -> None:
    """Zero thickness must raise ValueError."""
    with pytest.raises(ValueError):
        CaseSpec(thickness_m=0.0)


def test_very_low_speed_gives_small_momentum() -> None:
    """Very low speed must give small momentum flux."""
    spec_low = CaseSpec(thickness_m=1.0, speed_m_s=0.1, concentration=0.6)
    spec_normal = CaseSpec(thickness_m=1.0, speed_m_s=3.0, concentration=0.6)
    
    R_mom_low = compute_momentum_flux(spec_low)
    R_mom_normal = compute_momentum_flux(spec_normal)
    
    # Momentum ~ V^2, so 0.1/3.0 should give (0.1/3.0)^2 = 0.001 ratio
    ratio = R_mom_low / R_mom_normal
    expected_ratio = (0.1 / 3.0) ** 2
    assert abs(ratio - expected_ratio) < 0.01


def test_very_small_dem_contact() -> None:
    """Very small DEM contact should work without numerical issues."""
    dem_N = 1e-6  # 1 micro-Newton
    corr = apply_correction(STANDARD_CASE, dem_N)
    
    assert math.isfinite(corr.hybrid_total_N)
    assert corr.hybrid_total_N > 0
