"""Tests for ridge pore fluid module (Terzaghi 1D consolidation coupling).

Machine acceptance criteria:
1. Effective stress = total stress - pore pressure (Terzaghi principle)
2. Excess pore pressure dissipates over time (consolidation)
3. Degree of consolidation U → 1 as T_v → ∞
4. Shear strength decreases with excess pore pressure
5. Hydraulic conductivity from Kozeny-Carman is positive and finite
6. Strength reduction factor ≤ 1
7. SHA256 reproducibility for same input
"""
from __future__ import annotations

import hashlib
import json
import math
import pytest
import numpy as np

from tensordem.ridge_pore_fluid import (
    PoreFluidState,
    TerzaghiConsolidation,
    KeelPoreFluidModel,
    PoreFluidConfig,
    compute_excess_pore_pressure,
    degree_of_consolidation,
    hydraulic_conductivity_kozeny_carman,
    effective_stress,
    shear_strength_mohr_coulomb,
)


class TestEffectiveStress:
    """Test σ' = σ - u (Terzaghi's principle)."""

    def test_basic(self):
        """σ' = σ - u."""
        assert effective_stress(100e3, 50e3) == 50e3

    def test_clamped_to_zero(self):
        """σ' ≥ 0 (cannot be negative)."""
        assert effective_stress(50e3, 100e3) == 0.0

    def test_zero_pore_pressure(self):
        """u = 0 → σ' = σ."""
        assert effective_stress(100e3, 0.0) == 100e3


class TestShearStrengthMohrCoulomb:
    """Test τ = c' + σ'·tan(φ')."""

    def test_basic(self):
        """Basic calculation."""
        tau = shear_strength_mohr_coulomb(100e3, cohesion=10e3, friction_angle_deg=30.0)
        expected = 10e3 + 100e3 * math.tan(math.radians(30.0))
        assert abs(tau - expected) < 1.0

    def test_zero_cohesion(self):
        """c' = 0 → τ = σ'·tan(φ')."""
        tau = shear_strength_mohr_coulomb(100e3, cohesion=0.0, friction_angle_deg=45.0)
        expected = 100e3 * math.tan(math.radians(45.0))
        assert abs(tau - expected) < 1.0

    def test_zero_friction(self):
        """φ' = 0 → τ = c'."""
        tau = shear_strength_mohr_coulomb(100e3, cohesion=10e3, friction_angle_deg=0.0)
        assert abs(tau - 10e3) < 1.0

    def test_zero_effective_stress(self):
        """σ' = 0 → τ = c'."""
        tau = shear_strength_mohr_coulomb(0.0, cohesion=10e3, friction_angle_deg=30.0)
        assert abs(tau - 10e3) < 1.0


class TestHydraulicConductivity:
    """Test Kozeny-Carman equation."""

    def test_positive(self):
        """k must be positive."""
        k = hydraulic_conductivity_kozeny_carman(porosity=0.35, d_mean=1.0)
        assert k > 0

    def test_finite(self):
        """k must be finite."""
        k = hydraulic_conductivity_kozeny_carman(porosity=0.35, d_mean=1.0)
        assert math.isfinite(k)

    def test_porosity_effect(self):
        """Higher porosity → higher k."""
        k1 = hydraulic_conductivity_kozeny_carman(porosity=0.2, d_mean=1.0)
        k2 = hydraulic_conductivity_kozeny_carman(porosity=0.5, d_mean=1.0)
        assert k2 > k1

    def test_particle_size_effect(self):
        """Larger particles → higher k (k ∝ d²)."""
        k1 = hydraulic_conductivity_kozeny_carman(porosity=0.35, d_mean=0.5)
        k2 = hydraulic_conductivity_kozeny_carman(porosity=0.35, d_mean=1.0)
        assert abs(k2 / k1 - 4.0) < 0.5  # approximately d² scaling

    def test_invalid_porosity(self):
        """Porosity outside (0, 1) should raise."""
        with pytest.raises(ValueError):
            hydraulic_conductivity_kozeny_carman(porosity=0.0, d_mean=1.0)
        with pytest.raises(ValueError):
            hydraulic_conductivity_kozeny_carman(porosity=1.0, d_mean=1.0)


class TestDegreeOfConsolidation:
    """Test U(T_v) from Terzaghi's series solution."""

    def test_zero_time(self):
        """T_v = 0 → U = 0."""
        assert degree_of_consolidation(0.0) == 0.0

    def test_large_time(self):
        """T_v → ∞ → U → 1."""
        assert degree_of_consolidation(100.0) == 1.0

    def test_monotonic(self):
        """U increases with T_v."""
        U1 = degree_of_consolidation(0.1)
        U2 = degree_of_consolidation(1.0)
        U3 = degree_of_consolidation(10.0)
        assert U1 < U2 < U3

    def test_typical_values(self):
        """Check known values."""
        # T_v = 0.196 → U ≈ 0.5
        U = degree_of_consolidation(0.196)
        assert abs(U - 0.5) < 0.05

        # T_v = 0.848 → U ≈ 0.9
        U = degree_of_consolidation(0.848)
        assert abs(U - 0.9) < 0.05


class TestExcessPorePressure:
    """Test analytical excess pore pressure solution."""

    def test_initial_condition(self):
        """t = 0 → Δu = q (undrained)."""
        delta_u = compute_excess_pore_pressure(
            surf_load=100e3, H=5.0, z=2.5, c_v=1e-6, t=0.0
        )
        assert delta_u == 100e3

    def test_dissipation(self):
        """Δu decreases with time (use parameters that show meaningful dissipation)."""
        # Use larger c_v and longer times to get meaningful T_v values
        # T_v = c_v * t / H_dr², with H_dr = 2.5
        # For c_v=1e-4, t=1000: T_v = 1e-4 * 1000 / 6.25 = 0.016
        # For c_v=1e-4, t=100000: T_v = 1e-4 * 100000 / 6.25 = 1.6
        u1 = compute_excess_pore_pressure(
            surf_load=100e3, H=5.0, z=2.5, c_v=1e-4, t=1000.0
        )
        u2 = compute_excess_pore_pressure(
            surf_load=100e3, H=5.0, z=2.5, c_v=1e-4, t=100000.0
        )
        assert u2 < u1, f"Expected dissipation: u1={u1:.1f}, u2={u2:.1f}"

    def test_long_time(self):
        """t → ∞ → Δu → 0."""
        delta_u = compute_excess_pore_pressure(
            surf_load=100e3, H=5.0, z=2.5, c_v=1e-4, t=1e8
        )
        assert delta_u < 1.0  # Essentially zero


class TestPoreFluidState:
    """Test per-layer pore fluid state."""

    def test_hydrostatic_pressure(self):
        """u_0 = ρ_w · g · depth."""
        state = PoreFluidState(depth=5.0, thickness=1.0, porosity=0.35)
        expected = 1025.0 * 9.81 * 5.0
        assert abs(state.hydrostatic_pressure - expected) < 1.0

    def test_total_pore_pressure(self):
        """u = u_0 + Δu."""
        state = PoreFluidState(
            depth=5.0, thickness=1.0, porosity=0.35,
            excess_pore_pressure=10e3
        )
        expected = state.hydrostatic_pressure + 10e3
        assert abs(state.total_pore_pressure - expected) < 1.0

    def test_update_effective_stress(self):
        """σ' = σ - u."""
        state = PoreFluidState(
            depth=5.0, thickness=1.0, porosity=0.35,
            excess_pore_pressure=10e3,
            total_stress=100e3,
        )
        state.update_effective_stress(cohesion=5e3, friction_angle_deg=35.0)
        expected_eff = max(0.0, 100e3 - state.total_pore_pressure)
        assert abs(state.effective_stress - expected_eff) < 1.0


class TestPoreFluidConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Valid config should not raise."""
        cfg = PoreFluidConfig(n_layers=10, keel_depth=5.0)
        assert cfg.n_layers == 10
        assert cfg.keel_depth == 5.0

    def test_layer_thickness(self):
        """Layer thickness = keel_depth / n_layers."""
        cfg = PoreFluidConfig(n_layers=10, keel_depth=5.0)
        assert abs(cfg.layer_thickness - 0.5) < 1e-10

    def test_computed_c_v(self):
        """c_v should be positive and finite."""
        cfg = PoreFluidConfig(n_layers=10, keel_depth=5.0)
        c_v = cfg.computed_c_v
        assert c_v > 0
        assert math.isfinite(c_v)

    def test_explicit_c_v(self):
        """Explicit c_v should be used."""
        cfg = PoreFluidConfig(n_layers=10, keel_depth=5.0, c_v=1e-5)
        assert cfg.computed_c_v == 1e-5


class TestKeelPoreFluidModel:
    """Test the full keel pore fluid model."""

    def test_initialization(self):
        """Model should initialize with hydrostatic conditions."""
        cfg = PoreFluidConfig(n_layers=5, keel_depth=5.0)
        model = KeelPoreFluidModel(config=cfg)
        assert len(model.layers) == 5
        assert model.time == 0.0
        # Initial excess pore pressure should be zero
        for layer in model.layers:
            assert layer.excess_pore_pressure == 0.0

    def test_apply_surface_load(self):
        """Surface load should create excess pore pressure."""
        cfg = PoreFluidConfig(n_layers=5, keel_depth=5.0)
        model = KeelPoreFluidModel(config=cfg)
        model.apply_surface_load(100e3)
        # Top layer should have highest excess pore pressure
        assert model.layers[0].excess_pore_pressure > 0
        # Bottom layer should have lowest
        assert model.layers[-1].excess_pore_pressure < model.layers[0].excess_pore_pressure

    def test_consolidation_dissipates(self):
        """Consolidation should reduce excess pore pressure."""
        cfg = PoreFluidConfig(n_layers=5, keel_depth=5.0, c_v=1e-4)
        model = KeelPoreFluidModel(config=cfg)
        model.apply_surface_load(100e3)
        initial_excess = model.layers[0].excess_pore_pressure

        model.consolidate(dt=3600.0)  # 1 hour
        assert model.layers[0].excess_pore_pressure < initial_excess

    def test_strength_reduction(self):
        """Strength reduction factor should be ≤ 1."""
        cfg = PoreFluidConfig(n_layers=5, keel_depth=5.0, c_v=1e-4)
        model = KeelPoreFluidModel(config=cfg)
        model.apply_surface_load(100e3)
        model.consolidate(dt=100.0)

        factors = model.get_strength_reduction_factor()
        assert all(f <= 1.0 for f in factors)

    def test_summary(self):
        """Summary should contain expected keys."""
        cfg = PoreFluidConfig(n_layers=5, keel_depth=5.0)
        model = KeelPoreFluidModel(config=cfg)
        model.apply_surface_load(100e3)
        summary = model.summary()
        assert "time_s" in summary
        assert "n_layers" in summary
        assert "max_excess_pore_pressure_Pa" in summary
        assert "mean_effective_stress_Pa" in summary

    def test_sha256_reproducibility(self):
        """Same input → same SHA256."""
        cfg = PoreFluidConfig(n_layers=5, keel_depth=5.0, c_v=1e-4)
        model1 = KeelPoreFluidModel(config=cfg)
        model1.apply_surface_load(100e3)
        model1.consolidate(dt=3600.0)

        model2 = KeelPoreFluidModel(config=cfg)
        model2.apply_surface_load(100e3)
        model2.consolidate(dt=3600.0)

        s1 = json.dumps(model1.summary(), sort_keys=True)
        s2 = json.dumps(model2.summary(), sort_keys=True)
        assert hashlib.sha256(s1.encode()).hexdigest() == \
               hashlib.sha256(s2.encode()).hexdigest()


class TestTerzaghiConsolidation:
    """Test the 1D consolidation solver."""

    def test_dissipation(self):
        """Excess pore pressure should dissipate over time."""
        solver = TerzaghiConsolidation(c_v=1e-6, H=5.0, n_nodes=20)
        initial = np.full(20, 100e3)
        history = solver.solve(initial, dt=100.0, n_steps=100)

        # Final excess should be less than initial
        assert np.mean(history[-1]) < np.mean(history[0])

    def test_boundary_conditions(self):
        """Drainage boundaries should have u = 0."""
        solver = TerzaghiConsolidation(c_v=1e-6, H=5.0, n_nodes=20, drainage="double")
        initial = np.full(20, 100e3)
        history = solver.solve(initial, dt=100.0, n_steps=10)

        # Boundaries should remain at 0
        assert history[-1, 0] == 0.0
        assert history[-1, -1] == 0.0

    def test_stability_check(self):
        """Unstable time step should raise."""
        solver = TerzaghiConsolidation(c_v=1e-4, H=5.0, n_nodes=20)
        initial = np.full(20, 100e3)
        # Very large dt should violate stability
        with pytest.raises(ValueError):
            solver.solve(initial, dt=1000.0, n_steps=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
