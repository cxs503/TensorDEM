"""Tests for ridge sail resistance (Kujala 1994 / Riska 1997 model).

Machine acceptance criteria:
1. Breaking resistance > 0 for valid inputs
2. Pushing resistance > 0 for valid inputs
3. Total sail resistance = F_sb + F_sp
4. Breaking resistance scales with σ_f and h_s²
5. Pushing resistance scales with (1-n) and h_s²
6. Passive earth pressure coefficient K_p > 1 for φ > 0
7. Effective unit weight is negative for ice in seawater (buoyant)
8. SHA256 reproducibility for same input
"""
from __future__ import annotations

import hashlib
import json
import math
import pytest
import numpy as np

from tensordem.ridge_sail_resistance import (
    SailResistanceConfig,
    SailResistanceResult,
    compute_sail_resistance,
    kujala_breaking_resistance,
    riska_pushing_resistance,
    passive_earth_pressure_coefficient,
    effective_unit_weight,
)


class TestPassiveEarthPressureCoefficient:
    """Test K_p = tan²(45° + φ/2)."""

    def test_typical_phi(self):
        """For φ = 45°, K_p = tan²(67.5°) ≈ 5.83."""
        K_p = passive_earth_pressure_coefficient(45.0)
        expected = math.tan(math.radians(67.5)) ** 2
        assert abs(K_p - expected) < 1e-10

    def test_phi_zero(self):
        """For φ = 0°, K_p = tan²(45°) = 1."""
        K_p = passive_earth_pressure_coefficient(0.0)
        assert abs(K_p - 1.0) < 1e-10

    def test_phi_increases_Kp(self):
        """K_p increases with φ."""
        K1 = passive_earth_pressure_coefficient(30.0)
        K2 = passive_earth_pressure_coefficient(45.0)
        assert K2 > K1


class TestEffectiveUnitWeight:
    """Test γ' = (1-n)·(ρ_ice - ρ_water)·g."""

    def test_buoyant_ice(self):
        """Ice in seawater: γ' < 0 (buoyant)."""
        gamma = effective_unit_weight(0.35)
        assert gamma < 0, f"γ' = {gamma} should be negative for buoyant ice"

    def test_porosity_effect(self):
        """Higher porosity → smaller |γ'|."""
        g1 = effective_unit_weight(0.2)
        g2 = effective_unit_weight(0.5)
        assert abs(g2) < abs(g1), "Higher porosity should reduce |γ'|"

    def test_zero_porosity(self):
        """n → 0: γ' → (ρ_ice - ρ_water)·g."""
        gamma = effective_unit_weight(0.001)
        expected = (917.0 - 1025.0) * 9.81
        assert abs(gamma - expected) < abs(expected) * 0.01


class TestKujalaBreakingResistance:
    """Test F_sb = C_b · σ_f · h_s² · B / sin(α)."""

    def test_positive(self):
        """Breaking resistance must be positive."""
        F = kujala_breaking_resistance(h_s=2.0, B=20.0, alpha_deg=30.0, sigma_f=500e3)
        assert F > 0

    def test_scales_with_sigma_f(self):
        """F_sb ∝ σ_f."""
        F1 = kujala_breaking_resistance(h_s=2.0, B=20.0, alpha_deg=30.0, sigma_f=500e3)
        F2 = kujala_breaking_resistance(h_s=2.0, B=20.0, alpha_deg=30.0, sigma_f=1000e3)
        assert abs(F2 / F1 - 2.0) < 0.01

    def test_scales_with_h_s_squared(self):
        """F_sb ∝ h_s²."""
        F1 = kujala_breaking_resistance(h_s=1.0, B=20.0, alpha_deg=30.0, sigma_f=500e3)
        F2 = kujala_breaking_resistance(h_s=2.0, B=20.0, alpha_deg=30.0, sigma_f=500e3)
        assert abs(F2 / F1 - 4.0) < 0.01

    def test_scales_with_B(self):
        """F_sb ∝ B."""
        F1 = kujala_breaking_resistance(h_s=2.0, B=10.0, alpha_deg=30.0, sigma_f=500e3)
        F2 = kujala_breaking_resistance(h_s=2.0, B=20.0, alpha_deg=30.0, sigma_f=500e3)
        assert abs(F2 / F1 - 2.0) < 0.01

    def test_alpha_effect(self):
        """Smaller α → larger F_sb (sin(α) in denominator)."""
        F1 = kujala_breaking_resistance(h_s=2.0, B=20.0, alpha_deg=20.0, sigma_f=500e3)
        F2 = kujala_breaking_resistance(h_s=2.0, B=20.0, alpha_deg=40.0, sigma_f=500e3)
        assert F1 > F2


class TestRiskaPushingResistance:
    """Test F_sp = C_p · |γ'| · h_s² · B · K_p / 2."""

    def test_positive(self):
        """Pushing resistance must be positive."""
        F = riska_pushing_resistance(h_s=2.0, B=20.0, n=0.35, phi_deg=40.0)
        assert F > 0

    def test_scales_with_h_s_squared(self):
        """F_sp ∝ h_s²."""
        F1 = riska_pushing_resistance(h_s=1.0, B=20.0, n=0.35, phi_deg=40.0)
        F2 = riska_pushing_resistance(h_s=2.0, B=20.0, n=0.35, phi_deg=40.0)
        assert abs(F2 / F1 - 4.0) < 0.01

    def test_porosity_effect(self):
        """Higher porosity → lower F_sp (via |γ'|)."""
        F1 = riska_pushing_resistance(h_s=2.0, B=20.0, n=0.2, phi_deg=40.0)
        F2 = riska_pushing_resistance(h_s=2.0, B=20.0, n=0.5, phi_deg=40.0)
        assert F1 > F2

    def test_phi_effect(self):
        """Higher φ → higher F_sp (via K_p)."""
        F1 = riska_pushing_resistance(h_s=2.0, B=20.0, n=0.35, phi_deg=30.0)
        F2 = riska_pushing_resistance(h_s=2.0, B=20.0, n=0.35, phi_deg=45.0)
        assert F2 > F1


class TestSailResistanceConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Valid config should not raise."""
        cfg = SailResistanceConfig(h_s=2.0, B=20.0)
        assert cfg.h_s == 2.0
        assert cfg.B == 20.0

    def test_invalid_porosity(self):
        """Porosity outside (0, 1) should raise."""
        with pytest.raises(ValueError):
            SailResistanceConfig(h_s=2.0, B=20.0, n=0.0)
        with pytest.raises(ValueError):
            SailResistanceConfig(h_s=2.0, B=20.0, n=1.0)

    def test_invalid_alpha(self):
        """Alpha outside (0, 90) should raise."""
        with pytest.raises(ValueError):
            SailResistanceConfig(h_s=2.0, B=20.0, alpha=0.0)
        with pytest.raises(ValueError):
            SailResistanceConfig(h_s=2.0, B=20.0, alpha=90.0)


class TestComputeSailResistance:
    """Test the full sail resistance computation."""

    def test_result_components(self):
        """Result should contain all expected fields."""
        cfg = SailResistanceConfig(h_s=2.0, B=20.0)
        result = compute_sail_resistance(cfg)
        assert result.F_sb > 0
        assert result.F_sp > 0
        assert result.F_sail == result.F_sb + result.F_sp
        assert result.gamma_eff < 0  # buoyant
        assert result.K_p > 1

    def test_breaking_dominates(self):
        """For typical parameters, breaking resistance should dominate."""
        cfg = SailResistanceConfig(h_s=2.0, B=20.0, sigma_f=500e3)
        result = compute_sail_resistance(cfg)
        # Breaking is typically much larger than pushing for ice
        assert result.F_sb > result.F_sp

    def test_to_dict(self):
        """to_dict should return all fields."""
        cfg = SailResistanceConfig(h_s=2.0, B=20.0)
        result = compute_sail_resistance(cfg)
        d = result.to_dict()
        assert "F_sb_N" in d
        assert "F_sp_N" in d
        assert "F_sail_N" in d
        assert "gamma_eff_N_m3" in d
        assert "K_p" in d

    def test_sha256_reproducibility(self):
        """Same input → same SHA256."""
        cfg = SailResistanceConfig(h_s=2.0, B=20.0, sigma_f=500e3, n=0.35)
        result1 = compute_sail_resistance(cfg)
        result2 = compute_sail_resistance(cfg)

        params1 = json.dumps(result1.to_dict(), sort_keys=True)
        params2 = json.dumps(result2.to_dict(), sort_keys=True)
        assert hashlib.sha256(params1.encode()).hexdigest() == \
               hashlib.sha256(params2.encode()).hexdigest()


class TestSailResistanceScaling:
    """Test physical scaling behavior."""

    def test_larger_sail_more_resistance(self):
        """Larger sail → more resistance."""
        cfg1 = SailResistanceConfig(h_s=1.0, B=20.0)
        cfg2 = SailResistanceConfig(h_s=3.0, B=20.0)
        r1 = compute_sail_resistance(cfg1)
        r2 = compute_sail_resistance(cfg2)
        assert r2.F_sail > r1.F_sail

    def test_wider_ship_more_resistance(self):
        """Wider ship → more resistance."""
        cfg1 = SailResistanceConfig(h_s=2.0, B=10.0)
        cfg2 = SailResistanceConfig(h_s=2.0, B=30.0)
        r1 = compute_sail_resistance(cfg1)
        r2 = compute_sail_resistance(cfg2)
        assert r2.F_sail > r1.F_sail

    def test_stronger_ice_more_resistance(self):
        """Stronger ice → more resistance."""
        cfg1 = SailResistanceConfig(h_s=2.0, B=20.0, sigma_f=300e3)
        cfg2 = SailResistanceConfig(h_s=2.0, B=20.0, sigma_f=700e3)
        r1 = compute_sail_resistance(cfg1)
        r2 = compute_sail_resistance(cfg2)
        assert r2.F_sail > r1.F_sail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
