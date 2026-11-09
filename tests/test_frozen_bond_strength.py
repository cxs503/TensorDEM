"""Comprehensive tests for frozen bond strength model with mode transition.

Validates all machine-computable criteria:
(a) σ_bond(T) vs Sastry & Metzger 1986 data, ≤20% error (-1°C to -20°C)
(b) T_transition matches analytical estimate (≤2°C error)
(c) F_cohesive/F_adhesive ≥ 2 at mode switch
(d) σ_bond monotonically increases with decreasing T (Spearman ρ ≤ -0.95)
(e) Negative control: T > T_freeze → σ_bond = 0
"""
import pytest
import numpy as np
from scipy import stats
from scipy.optimize import brentq

from tensordem.frozen_bond_strength import (
    FrozenBondConfig,
    FrozenBondResult,
    sigma_bond_temperature,
    sigma_bond_time,
    sigma_bond_pressure,
    sigma_ice_temperature,
    failure_mode,
    transition_temperature,
    peak_load_jump_ratio,
    frozen_bond_strength,
    validate_against_sastry_metzger,
    SASTRY_METZGER_1986_DATA,
    T_FREEZE_FRESH,
    T_MIN_C,
    K_COHESIVE_DEFAULT,
)


# ---------------------------------------------------------------------------
# Criterion (a): Sastry & Metzger (1986) validation
# ---------------------------------------------------------------------------
class TestSastryMetzgerValidation:
    """Validate bond strength against Sastry & Metzger (1986) experimental data."""

    def test_all_points_within_20_percent(self):
        """All prediction errors must be ≤ 20%."""
        result = validate_against_sastry_metzger()
        assert result["max_relative_error"] <= 0.20, (
            f"Max error {result['max_relative_error']*100:.1f}% exceeds 20% threshold"
        )

    def test_mean_error_small(self):
        """Mean absolute error should be small."""
        result = validate_against_sastry_metzger()
        assert result["mean_relative_error"] <= 0.10, (
            f"MAE {result['mean_relative_error']*100:.1f}% exceeds 10%"
        )

    def test_per_point_errors(self):
        """Check each data point individually."""
        result = validate_against_sastry_metzger()
        for point in result["points"]:
            assert point["relative_error"] <= 0.20, (
                f"T={point['temperature_c']}°C: error {point['relative_error']*100:.1f}% > 20%"
            )

    def test_temperature_range_coverage(self):
        """Validation must cover -1°C to -20°C."""
        temps = [t for t, _ in SASTRY_METZGER_1986_DATA]
        assert min(temps) <= -20.0
        assert max(temps) >= -1.0


# ---------------------------------------------------------------------------
# Criterion (b): Transition temperature accuracy
# ---------------------------------------------------------------------------
class TestTransitionTemperature:
    """Validate transition temperature computation."""

    def test_transition_matches_analytical(self):
        """T_transition must match analytical estimate within 2°C."""
        cfg = FrozenBondConfig()
        t_trans = transition_temperature(cfg, 0.1, 100.0)
        
        # Analytical estimate via root finding
        def residual(T):
            sb = sigma_bond_temperature(T, cfg.t_freeze, cfg.sigma_0,
                                        cfg.beta, cfg.gamma, cfg.t_gamma)
            sb = sigma_bond_time(sb, 100.0, cfg.alpha, cfg.t_ref)
            sb *= (0.1 / cfg.p_ref) ** cfg.pressure_exponent
            si = sigma_ice_temperature(T, cfg.sigma_ice_0, cfg.beta_ice, cfg.t_freeze)
            return sb - si / cfg.k_cohesive
        
        t_analytical = brentq(residual, -30.0, -0.1)
        error = abs(t_trans - t_analytical)
        
        assert error <= 2.0, (
            f"T_transition error {error:.2f}°C exceeds 2°C threshold"
        )

    def test_transition_in_physical_range(self):
        """Transition must occur below freezing."""
        cfg = FrozenBondConfig()
        t_trans = transition_temperature(cfg, 0.1, 100.0)
        assert t_trans < cfg.t_freeze
        assert t_trans > T_MIN_C

    def test_transition_shifts_with_pressure(self):
        """Higher pressure should shift T_transition warmer."""
        cfg = FrozenBondConfig()
        t_low = transition_temperature(cfg, 0.01, 100.0)
        t_high = transition_temperature(cfg, 1.0, 100.0)
        assert t_high > t_low, (
            f"T_trans should increase with pressure: {t_low:.2f} vs {t_high:.2f}°C"
        )

    def test_transition_shifts_with_time(self):
        """Longer contact time should shift T_transition warmer."""
        cfg = FrozenBondConfig()
        t_short = transition_temperature(cfg, 0.1, 10.0)
        t_long = transition_temperature(cfg, 0.1, 1000.0)
        assert t_long > t_short, (
            f"T_trans should increase with time: {t_short:.2f} vs {t_long:.2f}°C"
        )


# ---------------------------------------------------------------------------
# Criterion (c): Peak load jump at mode transition
# ---------------------------------------------------------------------------
class TestPeakLoadJump:
    """Validate peak load discontinuity at mode transition."""

    def test_peak_load_ratio_ge_2(self):
        """F_cohesive/F_adhesive must be ≥ 2 at mode switch."""
        cfg = FrozenBondConfig()
        jump = peak_load_jump_ratio(cfg, 0.1, 100.0)
        
        assert jump["peak_load_ratio"] >= 2.0, (
            f"Peak load ratio {jump['peak_load_ratio']:.2f} < 2.0"
        )

    def test_failure_modes_correct(self):
        """Failure modes must be cohesive below and adhesive above transition."""
        cfg = FrozenBondConfig()
        jump = peak_load_jump_ratio(cfg, 0.1, 100.0)
        
        assert jump["failure_below"] == "cohesive"
        assert jump["failure_above"] == "adhesive"

    def test_peak_loads_positive(self):
        """Peak loads must be positive on both sides of transition."""
        cfg = FrozenBondConfig()
        jump = peak_load_jump_ratio(cfg, 0.1, 100.0)
        
        assert jump["peak_load_cohesive_mpa"] > 0
        assert jump["peak_load_adhesive_mpa"] > 0

    def test_k_cohesive_enhancement(self):
        """Cohesive peak load must include K_cohesive enhancement."""
        cfg = FrozenBondConfig()
        jump = peak_load_jump_ratio(cfg, 0.1, 100.0)
        
        # At t_below, ice strength × K_cohesive should give cohesive peak
        si = sigma_ice_temperature(jump["t_below_c"], cfg.sigma_ice_0,
                                   cfg.beta_ice, cfg.t_freeze)
        expected = si * cfg.k_cohesive
        assert jump["peak_load_cohesive_mpa"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Criterion (d): Monotonicity
# ---------------------------------------------------------------------------
class TestMonotonicity:
    """Validate σ_bond increases monotonically with decreasing T."""

    def test_spearman_rho_le_minus_0_95(self):
        """Spearman correlation must be ≤ -0.95."""
        cfg = FrozenBondConfig()
        temps = np.linspace(-20.0, -1.0, 20)
        sigmas = [sigma_bond_temperature(t, cfg.t_freeze, cfg.sigma_0,
                                         cfg.beta, cfg.gamma, cfg.t_gamma)
                  for t in temps]
        
        rho, pval = stats.spearmanr(temps, sigmas)
        
        assert rho <= -0.95, f"Spearman rho {rho:.4f} > -0.95"
        assert pval < 0.01, f"Correlation not significant: p={pval:.2e}"

    def test_strict_monotonicity(self):
        """Bond strength must strictly increase with decreasing T."""
        cfg = FrozenBondConfig()
        temps = np.linspace(-20.0, -1.0, 50)
        sigmas = [sigma_bond_temperature(t, cfg.t_freeze, cfg.sigma_0,
                                         cfg.beta, cfg.gamma, cfg.t_gamma)
                  for t in temps]
        
        # Check strict monotonicity
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i+1], (
                f"Not monotonic at T={temps[i]:.2f}°C: {sigmas[i]:.4f} vs {sigmas[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# Criterion (e): Negative control
# ---------------------------------------------------------------------------
class TestNegativeControl:
    """Validate σ_bond = 0 for T > T_freeze."""

    def test_zero_above_freezing(self):
        """Bond strength must be zero above freezing."""
        cfg = FrozenBondConfig()
        for t in [0.1, 0.5, 1.0, 5.0, 20.0]:
            s = sigma_bond_temperature(t, cfg.t_freeze, cfg.sigma_0,
                                       cfg.beta, cfg.gamma, cfg.t_gamma)
            assert s == 0.0, f"σ_bond({t}°C) = {s} ≠ 0"

    def test_ice_strength_zero_above_freezing(self):
        """Ice strength must also be zero above freezing."""
        cfg = FrozenBondConfig()
        for t in [0.1, 1.0, 5.0]:
            si = sigma_ice_temperature(t, cfg.sigma_ice_0, cfg.beta_ice, cfg.t_freeze)
            assert si == 0.0

    def test_result_no_bond_mode(self):
        """Failure mode must be 'no_bond' when T > T_freeze."""
        cfg = FrozenBondConfig()
        result = frozen_bond_strength(1.0, 0.1, 100.0, cfg)
        assert result.failure_mode == "no_bond"
        assert result.peak_load_mpa == 0.0
        assert not result.is_bonded


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------
class TestConfiguration:
    """Validate configuration parameters."""

    def test_default_config_valid(self):
        """Default configuration should be valid."""
        cfg = FrozenBondConfig()
        assert cfg.k_cohesive >= 2.0
        assert cfg.sigma_0 > 0
        assert cfg.t_freeze <= 0

    def test_k_cohesive_minimum(self):
        """K_cohesive must be ≥ 2.0."""
        with pytest.raises(ValueError, match="k_cohesive must be ≥ 2.0"):
            FrozenBondConfig(k_cohesive=1.5)

    def test_invalid_sigma_0(self):
        """sigma_0 must be positive."""
        with pytest.raises(ValueError):
            FrozenBondConfig(sigma_0=-1.0)

    def test_invalid_t_freeze(self):
        """t_freeze must be in valid range."""
        with pytest.raises(ValueError):
            FrozenBondConfig(t_freeze=1.0)
        with pytest.raises(ValueError):
            FrozenBondConfig(t_freeze=T_MIN_C - 1.0)


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_low_temperature(self):
        """Model should handle temperatures down to T_MIN_C."""
        cfg = FrozenBondConfig()
        s = sigma_bond_temperature(T_MIN_C, cfg.t_freeze, cfg.sigma_0,
                                   cfg.beta, cfg.gamma, cfg.t_gamma)
        assert s > 0.0

    def test_temperature_below_minimum_raises(self):
        """Temperature below T_MIN_C should raise."""
        cfg = FrozenBondConfig()
        with pytest.raises(ValueError):
            sigma_bond_temperature(T_MIN_C - 1.0, cfg.t_freeze, cfg.sigma_0,
                                   cfg.beta, cfg.gamma, cfg.t_gamma)

    def test_zero_contact_time_raises(self):
        """Zero contact time should raise."""
        with pytest.raises(ValueError):
            sigma_bond_time(0.5, 0.0)

    def test_negative_contact_time_raises(self):
        """Negative contact time should raise."""
        with pytest.raises(ValueError):
            sigma_bond_time(0.5, -1.0)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------
class TestIntegration:
    """End-to-end integration tests."""

    def test_full_evaluation_consistency(self):
        """Full evaluation should be internally consistent."""
        cfg = FrozenBondConfig()
        for t in [-20.0, -10.0, -5.0, -2.0]:
            res = frozen_bond_strength(t, 0.1, 100.0, cfg)
            
            # Recompute components
            sb_t = sigma_bond_temperature(t, cfg.t_freeze, cfg.sigma_0,
                                          cfg.beta, cfg.gamma, cfg.t_gamma)
            sb = sigma_bond_time(sb_t, 100.0, cfg.alpha, cfg.t_ref)
            sb *= (0.1 / cfg.p_ref) ** cfg.pressure_exponent
            si = sigma_ice_temperature(t, cfg.sigma_ice_0, cfg.beta_ice, cfg.t_freeze)
            
            assert res.sigma_bond_mpa == pytest.approx(sb, rel=1e-6)
            assert res.sigma_ice_mpa == pytest.approx(si, rel=1e-6)

    def test_failure_mode_consistency(self):
        """Failure mode should match strength comparison."""
        cfg = FrozenBondConfig()
        for t in [-20.0, -10.0, -5.0, -2.0, -1.0]:
            res = frozen_bond_strength(t, 0.1, 100.0, cfg)
            expected_mode = failure_mode(res.sigma_bond_mpa, res.sigma_ice_mpa,
                                        cfg.k_cohesive)
            assert res.failure_mode == expected_mode

    def test_peak_load_matches_mode(self):
        """Peak load should match failure mode."""
        cfg = FrozenBondConfig()
        for t in [-20.0, -10.0, -5.0, -2.0, -1.0]:
            res = frozen_bond_strength(t, 0.1, 100.0, cfg)
            
            if res.failure_mode == "cohesive":
                expected = res.sigma_ice_mpa * cfg.k_cohesive
            elif res.failure_mode == "adhesive":
                expected = res.sigma_bond_mpa
            else:
                expected = 0.0
            
            assert res.peak_load_mpa == pytest.approx(expected, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
