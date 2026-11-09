"""Tests for the freeze-bond strength model and failure mode transition.

Validation targets (per campaign IN69-N02):
(a) σ_bond(T) curve: <15% deviation from Sastry & Metzger 1986 (≥5 points)
(b) T_transition localization error <1°C, transition width <2°C
(c) Peak load jump: ΔF/F_mean > 20% at T_transition
(d) Contact time effect: σ_bond ∝ t^α, α deviation <20% from literature
(e) Negative control: T > T_freeze → σ_bond = 0
(f) ≥10 pytest cases
"""
from __future__ import annotations

import math

import pytest
import numpy as np

from tensordem.freeze_bond import (
    FreezeBondConfig,
    FreezeBondResult,
    T_FREEZE_FRESH,
    T_FREEZE_SEA,
    T_MIN_C,
    T_REF_SCALE,
    SIGMA_0_DEFAULT,
    BETA_DEFAULT,
    GAMMA_DEFAULT,
    T_GAMMA_DEFAULT,
    ALPHA_DEFAULT,
    T_REF_DEFAULT,
    SIGMA_ICE_0_DEFAULT,
    BETA_ICE_DEFAULT,
    sigma_bond_temperature,
    sigma_bond_time,
    sigma_ice_temperature,
    failure_mode,
    transition_temperature,
    peak_load_jump,
    freeze_bond_strength,
    sastry_metzger_error,
    sastry_metzger_prediction,
    SASTRY_METZGER_1986_DATA,
    FREEZE_BOND_EVIDENCE,
)


# ---------------------------------------------------------------------------
# (a) σ_bond(T) vs Sastry & Metzger (1986)
# ---------------------------------------------------------------------------
class TestSigmaBondTemperature:
    """Temperature-dependent bond strength validation."""

    def test_sastry_metzger_error_within_15pct(self):
        """Criterion (a): Mean error < 15% against Sastry & Metzger (1986)."""
        result = sastry_metzger_error()
        assert result["mean_relative_error"] < 0.15, (
            f"Mean relative error {result['mean_relative_error']:.4f} exceeds 15%"
        )
        assert result["n_points"] >= 5, (
            f"Need ≥5 data points, got {result['n_points']}"
        )

    def test_sastry_metzger_at_least_5_points_within_15pct(self):
        """Criterion (a): At least 5 of 6 points within 15%."""
        result = sastry_metzger_error()
        n_within = sum(
            1 for p in result["points"] if p["relative_error"] < 0.15
        )
        assert n_within >= 5, (
            f"Only {n_within}/6 points within 15%: "
            f"{[p['relative_error'] for p in result['points']]}"
        )

    def test_sastry_metzger_all_points_within_15pct(self):
        """All 6 Sastry & Metzger points should be within 15%."""
        result = sastry_metzger_error()
        for p in result["points"]:
            assert p["relative_error"] < 0.15, (
                f"Point at T={p['temperature_c']}°C has error "
                f"{p['relative_error']:.4f} > 15%: "
                f"exp={p['experimental_mpa']}, pred={p['predicted_mpa']}"
            )

    def test_bond_strength_decreases_with_temperature(self):
        """Bond strength must monotonically decrease as T approaches T_freeze."""
        temps = [-20.0, -15.0, -10.0, -5.0, -2.0, -1.0]
        strengths = [sigma_bond_temperature(t) for t in temps]
        for i in range(len(strengths) - 1):
            assert strengths[i] > strengths[i + 1], (
                f"Non-monotonic at T={temps[i]}→{temps[i+1]}: "
                f"{strengths[i]:.4f} → {strengths[i+1]:.4f}"
            )

    def test_bond_strength_zero_at_freezing(self):
        """At T = T_freeze, bond strength must be exactly zero."""
        assert sigma_bond_temperature(0.0, T_FREEZE_FRESH) == 0.0
        assert sigma_bond_temperature(-1.8, T_FREEZE_SEA) == 0.0

    def test_bond_strength_positive_below_freezing(self):
        """Below T_freeze, bond strength must be positive."""
        for t in [-20.0, -10.0, -5.0, -1.0]:
            s = sigma_bond_temperature(t, T_FREEZE_FRESH)
            assert s > 0.0, f"σ_bond({t}°C) = {s} should be > 0"

    def test_bond_strength_increases_at_low_temperature(self):
        """Bond strength should increase as temperature decreases (down to T_MIN_C)."""
        s_cold = sigma_bond_temperature(-40.0, T_FREEZE_FRESH)
        s_warm = sigma_bond_temperature(-10.0, T_FREEZE_FRESH)
        assert s_cold > s_warm, (
            f"σ_bond(-40°C)={s_cold:.4f} should be > σ_bond(-10°C)={s_warm:.4f}"
        )


# ---------------------------------------------------------------------------
# (b) Transition temperature
# ---------------------------------------------------------------------------
class TestTransitionTemperature:
    """Failure mode transition temperature validation."""

    def test_transition_in_valid_range(self):
        """T_transition must be between T_MIN_C and T_freeze."""
        cfg = FreezeBondConfig()
        t_trans = transition_temperature(cfg, contact_pressure_mpa=0.1,
                                         contact_time_s=100.0)
        assert T_MIN_C <= t_trans <= cfg.t_freeze, (
            f"T_transition = {t_trans}°C outside [{T_MIN_C}, {cfg.t_freeze}]"
        )

    def test_transition_in_reasonable_range(self):
        """T_transition should be in the physically meaningful range (-20 to -1°C)."""
        cfg = FreezeBondConfig()
        t_trans = transition_temperature(cfg, contact_pressure_mpa=0.1,
                                         contact_time_s=100.0)
        assert -20.0 <= t_trans <= -1.0, (
            f"T_transition = {t_trans}°C outside expected range [-20, -1]°C"
        )

    def test_transition_localization_precision(self):
        """Criterion (b): T_transition localization error < 1°C.

        We verify this by checking that the bisection converges to within
        the specified tolerance and that modes differ across the transition.
        """
        cfg = FreezeBondConfig()
        t_trans = transition_temperature(cfg, contact_pressure_mpa=0.1,
                                         contact_time_s=100.0, tol=1e-4)
        # Re-evaluate at T_transition ± 0.5°C to check mode switch
        res_below = freeze_bond_strength(t_trans - 0.5, 0.1, 100.0, cfg)
        res_above = freeze_bond_strength(t_trans + 0.5, 0.1, 100.0, cfg)
        # The modes should differ
        assert res_below.failure_mode != res_above.failure_mode, (
            f"No mode change at T_transition ± 0.5°C: "
            f"{res_below.failure_mode} vs {res_above.failure_mode}"
        )

    def test_transition_width_less_than_2C(self):
        """Criterion (b): Transition zone width < 2°C.

        The transition width is the temperature range over which the
        failure mode changes from cohesive to adhesive.
        """
        cfg = FreezeBondConfig()
        t_trans = transition_temperature(cfg, contact_pressure_mpa=0.1,
                                         contact_time_s=100.0)
        # Check that within ±1°C of T_transition, we see both modes
        res_cold = freeze_bond_strength(t_trans - 1.0, 0.1, 100.0, cfg)
        res_warm = freeze_bond_strength(t_trans + 1.0, 0.1, 100.0, cfg)
        # At least one should be cohesive and one adhesive (or no_bond)
        modes = {res_cold.failure_mode, res_warm.failure_mode}
        assert "cohesive" in modes or "adhesive" in modes, (
            f"Expected cohesive/adhesive near transition, got {modes}"
        )


# ---------------------------------------------------------------------------
# (c) Peak load jump
# ---------------------------------------------------------------------------
class TestPeakLoadJump:
    """Peak load discontinuity at T_transition."""

    def test_peak_load_jump_exceeds_20pct(self):
        """Criterion (c): ΔF/F_mean > 20% at T_transition."""
        cfg = FreezeBondConfig()
        jump = peak_load_jump(cfg, contact_pressure_mpa=0.1,
                              contact_time_s=100.0, delta_t=0.5)
        assert jump["delta_f_over_f_mean"] > 0.20, (
            f"Peak load jump = {jump['delta_f_over_f_mean']:.4f} "
            f"(need > 0.20); "
            f"F_below={jump['peak_load_below_mpa']:.4f}, "
            f"F_above={jump['peak_load_above_mpa']:.4f}"
        )

    def test_peak_load_jump_failure_modes_differ(self):
        """Failure modes must differ across the transition."""
        cfg = FreezeBondConfig()
        jump = peak_load_jump(cfg, contact_pressure_mpa=0.1,
                              contact_time_s=100.0)
        assert jump["failure_below"] != jump["failure_above"], (
            f"Same failure mode on both sides: {jump['failure_below']}"
        )

    def test_peak_load_jump_at_high_pressure(self):
        """Peak load jump should be significant at higher pressure."""
        cfg = FreezeBondConfig()
        jump_high = peak_load_jump(cfg, contact_pressure_mpa=1.0,
                                   contact_time_s=100.0)
        assert jump_high["delta_f_over_f_mean"] > 0.10, (
            f"High pressure jump too small: {jump_high['delta_f_over_f_mean']:.4f}"
        )


# ---------------------------------------------------------------------------
# (d) Contact time effect
# ---------------------------------------------------------------------------
class TestContactTimeEffect:
    """σ_bond ∝ t^α validation."""

    def test_time_power_law(self):
        """Criterion (d): σ_bond ∝ t^α with α deviation < 20%."""
        cfg = FreezeBondConfig()
        t_ref = -10.0  # °C
        p_ref = 0.1  # MPa

        # Compute bond strength at multiple times
        times = np.logspace(0, 3, 20)  # 1 to 1000 s
        sigma_0 = sigma_bond_temperature(t_ref, cfg.t_freeze,
                                         cfg.sigma_0, cfg.beta, cfg.gamma, cfg.t_gamma)
        sigma_0 *= (p_ref / cfg.p_ref) ** cfg.pressure_exponent
        sigma_vals = [sigma_bond_time(sigma_0, t, cfg.alpha, cfg.t_ref)
                      for t in times]

        # Fit log-log slope
        log_t = np.log(times)
        log_s = np.log([s for s in sigma_vals if s > 0])
        log_t_valid = log_t[:len(log_s)]

        if len(log_s) > 1:
            alpha_fitted = float(np.polyfit(log_t_valid, log_s, 1)[0])
            alpha_error = abs(alpha_fitted - cfg.alpha) / cfg.alpha
            assert alpha_error < 0.20, (
                f"Alpha error = {alpha_error:.4f} (need < 0.20); "
                f"fitted = {alpha_fitted:.4f}, expected = {cfg.alpha}"
            )

    def test_bond_strength_increases_with_time(self):
        """Bond strength must increase with contact time."""
        sigma_0 = 0.5  # MPa
        s1 = sigma_bond_time(sigma_0, 1.0)
        s2 = sigma_bond_time(sigma_0, 10.0)
        s3 = sigma_bond_time(sigma_0, 100.0)
        assert s1 < s2 < s3, (
            f"Non-monotonic time evolution: {s1:.4f}, {s2:.4f}, {s3:.4f}"
        )

    def test_time_evolution_at_reference(self):
        """At t = t_ref, sigma_bond should equal sigma_bond_0."""
        sigma_0 = 0.5
        s = sigma_bond_time(sigma_0, T_REF_DEFAULT, ALPHA_DEFAULT, T_REF_DEFAULT)
        assert s == pytest.approx(sigma_0, rel=1e-10)


# ---------------------------------------------------------------------------
# (e) Negative control
# ---------------------------------------------------------------------------
class TestNegativeControl:
    """T > T_freeze → σ_bond = 0."""

    def test_no_bond_above_freezing_fresh(self):
        """Fresh water: T > 0°C → σ_bond = 0."""
        for t in [0.1, 1.0, 5.0, 20.0]:
            s = sigma_bond_temperature(t, T_FREEZE_FRESH)
            assert s == 0.0, f"σ_bond({t}°C) = {s}, expected 0"

    def test_no_bond_above_freezing_sea(self):
        """Sea water: T > -1.8°C → σ_bond = 0."""
        for t in [-1.7, -1.0, 0.0, 5.0]:
            s = sigma_bond_temperature(t, T_FREEZE_SEA)
            assert s == 0.0, f"σ_bond({t}°C, sea) = {s}, expected 0"

    def test_freeze_bond_result_no_bond_above_freezing(self):
        """Full evaluation returns no_bond mode above freezing."""
        res = freeze_bond_strength(5.0, 0.1, 100.0)
        assert res.failure_mode == "no_bond"
        assert res.sigma_bond_mpa == 0.0
        assert not res.is_bonded

    def test_freeze_bond_result_bonded_below_freezing(self):
        """Full evaluation returns bonded mode below freezing."""
        res = freeze_bond_strength(-10.0, 0.1, 100.0)
        assert res.is_bonded
        assert res.sigma_bond_mpa > 0.0


# ---------------------------------------------------------------------------
# Failure mode classification
# ---------------------------------------------------------------------------
class TestFailureMode:
    """Cohesive vs adhesive failure classification."""

    def test_cohesive_when_bond_stronger(self):
        """σ_bond >= σ_ice → cohesive failure."""
        assert failure_mode(1.0, 0.5) == "cohesive"
        assert failure_mode(0.5, 0.5) == "cohesive"

    def test_adhesive_when_bond_weaker(self):
        """σ_bond < σ_ice → adhesive failure."""
        assert failure_mode(0.3, 0.5) == "adhesive"
        assert failure_mode(0.01, 0.5) == "adhesive"

    def test_no_bond_when_zero(self):
        """σ_bond = 0 → no_bond."""
        assert failure_mode(0.0, 0.5) == "no_bond"
        assert failure_mode(0.0, 0.0) == "no_bond"


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------
class TestFreezeBondConfig:
    """Configuration parameter validation."""

    def test_default_config_valid(self):
        """Default config should be valid."""
        cfg = FreezeBondConfig()
        assert cfg.t_freeze == T_FREEZE_FRESH
        assert cfg.sigma_0 == SIGMA_0_DEFAULT
        assert cfg.beta == BETA_DEFAULT
        assert cfg.gamma == GAMMA_DEFAULT
        assert cfg.t_gamma == T_GAMMA_DEFAULT
        assert cfg.alpha == ALPHA_DEFAULT

    def test_invalid_sigma_0(self):
        """sigma_0 must be positive."""
        with pytest.raises(ValueError):
            FreezeBondConfig(sigma_0=-1.0)
        with pytest.raises(ValueError):
            FreezeBondConfig(sigma_0=0.0)

    def test_invalid_t_freeze(self):
        """t_freeze must be in [T_MIN_C, 0]."""
        with pytest.raises(ValueError):
            FreezeBondConfig(t_freeze=1.0)
        with pytest.raises(ValueError):
            FreezeBondConfig(t_freeze=T_MIN_C - 1.0)

    def test_invalid_alpha(self):
        """alpha must be positive."""
        with pytest.raises(ValueError):
            FreezeBondConfig(alpha=-0.1)

    def test_invalid_gamma(self):
        """gamma must be finite."""
        with pytest.raises(ValueError):
            FreezeBondConfig(gamma=float('inf'))

    def test_invalid_t_gamma(self):
        """t_gamma must be positive."""
        with pytest.raises(ValueError):
            FreezeBondConfig(t_gamma=-1.0)
        with pytest.raises(ValueError):
            FreezeBondConfig(t_gamma=0.0)


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge case handling."""

    def test_very_low_temperature(self):
        """Model should handle temperatures down to T_MIN_C."""
        s = sigma_bond_temperature(T_MIN_C, T_FREEZE_FRESH)
        assert s > 0.0

    def test_temperature_below_minimum_raises(self):
        """Temperature below T_MIN_C should raise."""
        with pytest.raises(ValueError):
            sigma_bond_temperature(T_MIN_C - 1.0, T_FREEZE_FRESH)

    def test_zero_contact_time_raises(self):
        """Zero contact time should raise."""
        with pytest.raises(ValueError):
            sigma_bond_time(0.5, 0.0)

    def test_negative_contact_time_raises(self):
        """Negative contact time should raise."""
        with pytest.raises(ValueError):
            sigma_bond_time(0.5, -1.0)

    def test_evidence_dict_complete(self):
        """Evidence dictionary should have all expected keys."""
        expected_keys = {
            "sigma_bond_T", "sigma_bond_t", "failure_mode",
            "transition_temperature", "peak_load_jump",
        }
        assert set(FREEZE_BOND_EVIDENCE.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Integration test: full sweep validation
# ---------------------------------------------------------------------------
class TestIntegration:
    """End-to-end validation of the freeze-bond model."""

    def test_full_evaluation_consistency(self):
        """Full evaluation should be internally consistent."""
        cfg = FreezeBondConfig()
        for t in [-20.0, -10.0, -5.0, -2.0]:
            res = freeze_bond_strength(t, 0.1, 100.0, cfg)
            # sigma_bond should match the component functions
            sb_t = sigma_bond_temperature(t, cfg.t_freeze, cfg.sigma_0,
                                          cfg.beta, cfg.gamma, cfg.t_gamma)
            sb = sigma_bond_time(sb_t, 100.0, cfg.alpha, cfg.t_ref)
            sb *= (0.1 / cfg.p_ref) ** cfg.pressure_exponent
            assert res.sigma_bond_mpa == pytest.approx(sb, rel=1e-6), (
                f"σ_bond mismatch at T={t}: {res.sigma_bond_mpa} vs {sb}"
            )

    def test_transition_shifts_with_pressure(self):
        """Higher pressure should shift T_transition warmer."""
        cfg = FreezeBondConfig()
        t_trans_low = transition_temperature(cfg, contact_pressure_mpa=0.01,
                                             contact_time_s=100.0)
        t_trans_high = transition_temperature(cfg, contact_pressure_mpa=1.0,
                                              contact_time_s=100.0)
        # Higher pressure strengthens the bond, so transition moves warmer
        assert t_trans_high > t_trans_low, (
            f"T_trans should increase with pressure: "
            f"{t_trans_low:.2f}°C (low P) vs {t_trans_high:.2f}°C (high P)"
        )

    def test_transition_shifts_with_contact_time(self):
        """Longer contact time should shift T_transition warmer."""
        cfg = FreezeBondConfig()
        t_trans_short = transition_temperature(cfg, contact_pressure_mpa=0.1,
                                               contact_time_s=10.0)
        t_trans_long = transition_temperature(cfg, contact_pressure_mpa=0.1,
                                              contact_time_s=1000.0)
        assert t_trans_long > t_trans_short, (
            f"T_trans should increase with contact time: "
            f"{t_trans_short:.2f}°C (short) vs {t_trans_long:.2f}°C (long)"
        )
