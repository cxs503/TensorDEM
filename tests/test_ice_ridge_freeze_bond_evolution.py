"""Tests for ice ridge freeze-bond strength time evolution model.

Covers:
- Freeze-bond formation criterion (temperature threshold)
- Exponential saturation growth law (initial slope, asymptotic limit)
- Bond strength evolution over time
- Keel resistance evolution (Lindqvist model)
- Integration with DEM bond parameters
- Edge cases (zero age, very large age, above-freezing temperature)
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from tensordem.ice_ridge_freeze_bond_evolution import (
    FREEZING_POINT_K,
    FreezeBondFormationCriterion,
    ExponentialSaturationGrowth,
    FreezeBondModel,
    LindqvistRidgeResistance,
    compute_bond_strength_evolution,
    compute_keel_resistance_evolution,
    run_freeze_bond_study,
)


class TestFreezeBondFormationCriterion:
    """Test freeze-bond formation based on temperature."""
    
    def test_below_freezing_forms_bond(self):
        """Temperature below freezing should form bond."""
        criterion = FreezeBondFormationCriterion()
        assert criterion.should_form_bond(263.15) is True  # -10°C
        assert criterion.should_form_bond(270.0) is True   # -3.15°C
    
    def test_above_freezing_no_bond(self):
        """Temperature above freezing should not form bond."""
        criterion = FreezeBondFormationCriterion()
        assert criterion.should_form_bond(280.0) is False  # +7°C
        assert criterion.should_form_bond(300.0) is False  # +27°C
    
    def test_at_freezing_no_bond(self):
        """Temperature at freezing point should not form bond (strict inequality)."""
        criterion = FreezeBondFormationCriterion()
        assert criterion.should_form_bond(FREEZING_POINT_K) is False
    
    def test_tensor_input(self):
        """Should handle tensor input."""
        criterion = FreezeBondFormationCriterion()
        temps = torch.tensor([260.0, 273.15, 280.0], dtype=torch.float64)
        result = criterion.should_form_bond(temps)
        expected = torch.tensor([True, False, False])
        assert torch.equal(result, expected)
    
    def test_custom_freezing_point(self):
        """Should accept custom freezing point."""
        criterion = FreezeBondFormationCriterion(freezing_temperature_K=270.0)
        assert criterion.should_form_bond(269.0) is True
        assert criterion.should_form_bond(271.0) is False


class TestExponentialSaturationGrowth:
    """Test exponential saturation growth law."""
    
    def test_zero_age_zero_strength(self):
        """At age=0, strength should be 0."""
        growth = ExponentialSaturationGrowth()
        assert growth.tensile_strength_at_age(0.0) == 0.0
        assert growth.shear_strength_at_age(0.0) == 0.0
    
    def test_large_age_asymptotic_strength(self):
        """At very large age, strength should approach asymptotic value."""
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            asymptotic_shear_strength_Pa=2.0e6,
            growth_time_s=1.0e5,
        )
        large_age = 1.0e7  # 100x growth time
        tensile = growth.tensile_strength_at_age(large_age)
        shear = growth.shear_strength_at_age(large_age)
        
        # Should be within 0.1% of asymptotic value
        assert abs(tensile - 1.0e6) / 1.0e6 < 1e-3
        assert abs(shear - 2.0e6) / 2.0e6 < 1e-3
    
    def test_initial_slope(self):
        """Initial slope should be sigma_inf / tau_growth."""
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=1.0e5,
        )
        small_age = 1.0e3  # 1% of growth time
        tensile = growth.tensile_strength_at_age(small_age)
        
        # Linear approximation: sigma ~ sigma_inf * t / tau
        expected = 1.0e6 * small_age / 1.0e5
        assert abs(tensile - expected) / expected < 0.01  # within 1%
    
    def test_strength_ratio(self):
        """Strength ratio should be 1 - exp(-t/tau)."""
        growth = ExponentialSaturationGrowth(growth_time_s=1.0e5)
        age = 5.0e4  # half of growth time
        ratio = growth.strength_ratio_at_age(age)
        expected = 1.0 - math.exp(-age / 1.0e5)
        assert abs(ratio - expected) < 1e-10
    
    def test_tensor_input(self):
        """Should handle tensor input."""
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=1.0e5,
        )
        ages = torch.tensor([0.0, 1.0e4, 1.0e5, 1.0e6], dtype=torch.float64)
        tensile = growth.tensile_strength_at_age(ages)
        
        assert tensile.shape == ages.shape
        assert tensile[0] == 0.0
        assert tensile[-1] > 0.99e6  # approaching asymptotic
    
    def test_negative_age_clamped(self):
        """Negative age should be clamped to 0."""
        growth = ExponentialSaturationGrowth()
        assert growth.tensile_strength_at_age(-100.0) == 0.0
    
    def test_parameter_validation(self):
        """Should reject invalid parameters."""
        with pytest.raises(ValueError):
            ExponentialSaturationGrowth(asymptotic_tensile_strength_Pa=-1.0)
        with pytest.raises(ValueError):
            ExponentialSaturationGrowth(growth_time_s=0.0)
        with pytest.raises(ValueError):
            ExponentialSaturationGrowth(asymptotic_shear_strength_Pa=float('inf'))


class TestFreezeBondModel:
    """Test complete freeze-bond model."""
    
    def test_below_freezing_returns_strength(self):
        """Below freezing should return non-zero strength."""
        formation = FreezeBondFormationCriterion()
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=1.0e5,
        )
        model = FreezeBondModel(formation, growth)
        
        tensile, shear = model.bond_strength_at_age(1.0e5, 263.15)
        assert tensile > 0.0
        assert shear > 0.0
    
    def test_above_freezing_returns_zero(self):
        """Above freezing should return zero strength."""
        formation = FreezeBondFormationCriterion()
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=1.0e5,
        )
        model = FreezeBondModel(formation, growth)
        
        tensile, shear = model.bond_strength_at_age(1.0e5, 280.0)
        assert tensile == 0.0
        assert shear == 0.0
    
    def test_tensor_temperature(self):
        """Should handle tensor temperature input."""
        formation = FreezeBondFormationCriterion()
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=1.0e5,
        )
        model = FreezeBondModel(formation, growth)
        
        ages = torch.tensor([1.0e4, 1.0e5], dtype=torch.float64)
        temps = torch.tensor([263.15, 280.0], dtype=torch.float64)
        
        tensile, shear = model.bond_strength_at_age(ages, temps)
        assert tensile[0] > 0.0  # below freezing
        assert tensile[1] == 0.0  # above freezing


class TestLindqvistRidgeResistance:
    """Test Lindqvist ridge resistance model."""
    
    def test_solid_fraction(self):
        """Solid fraction should be 1 - porosity."""
        ridge = LindqvistRidgeResistance(porosity=0.30)
        assert ridge.solid_fraction == 0.70
    
    def test_buoyancy_force(self):
        """Buoyancy force should be positive (downward for ice)."""
        ridge = LindqvistRidgeResistance(
            keel_depth_m=10.0,
            keel_width_m=20.0,
            porosity=0.30,
        )
        buoyancy = ridge.buoyancy_force_per_unit_length
        assert buoyancy > 0.0  # ice is buoyant, so net force is downward
    
    def test_freeze_bond_force(self):
        """Freeze-bond force should scale with bond strength."""
        ridge = LindqvistRidgeResistance(
            keel_depth_m=10.0,
            keel_width_m=20.0,
            porosity=0.30,
        )
        bond_strength = 1.0e6  # 1 MPa
        bond_area_fraction = 0.5
        
        force = ridge.freeze_bond_force(bond_strength, bond_area_fraction)
        assert force > 0.0
        
        # Should scale linearly with bond strength
        force2 = ridge.freeze_bond_force(2.0 * bond_strength, bond_area_fraction)
        assert abs(force2 - 2.0 * force) / force < 1e-10
    
    def test_total_resistance_components(self):
        """Total resistance should be sum of components."""
        ridge = LindqvistRidgeResistance()
        bond_strength = 1.0e6
        
        result = ridge.total_resistance(bond_strength)
        
        total = (result['rubble_friction_N_per_m'] +
                 result['freeze_bond_N_per_m'] +
                 result['buoyancy_N_per_m'])
        
        assert abs(result['total_N_per_m'] - total) < 1e-6
    
    def test_resistance_increases_with_bond_strength(self):
        """Resistance should increase with bond strength."""
        ridge = LindqvistRidgeResistance()
        
        result1 = ridge.total_resistance(0.5e6)
        result2 = ridge.total_resistance(1.0e6)
        
        assert result2['total_N_per_m'] > result1['total_N_per_m']
    
    def test_parameter_validation(self):
        """Should reject invalid parameters."""
        with pytest.raises(ValueError):
            LindqvistRidgeResistance(keel_depth_m=-10.0)
        with pytest.raises(ValueError):
            LindqvistRidgeResistance(porosity=1.5)
        with pytest.raises(ValueError):
            LindqvistRidgeResistance(ice_density_kg_m3=2000.0, water_density_kg_m3=1000.0)


class TestBondStrengthEvolution:
    """Test bond strength evolution computation."""
    
    def test_evolution_monotonic(self):
        """Bond strength should increase monotonically with age."""
        formation = FreezeBondFormationCriterion()
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=1.0e5,
        )
        model = FreezeBondModel(formation, growth)
        
        ages = np.linspace(0, 5e5, 50)
        result = compute_bond_strength_evolution(model, ages, 263.15)
        
        # Check monotonicity
        tensile = result['tensile_strength_Pa']
        assert np.all(np.diff(tensile) >= 0)
    
    def test_evolution_bounded(self):
        """Bond strength should be bounded by asymptotic value."""
        formation = FreezeBondFormationCriterion()
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=1.0e5,
        )
        model = FreezeBondModel(formation, growth)
        
        ages = np.linspace(0, 1e7, 100)
        result = compute_bond_strength_evolution(model, ages, 263.15)
        
        tensile = result['tensile_strength_Pa']
        assert np.all(tensile <= 1.0e6 * 1.001)  # within 0.1% of asymptotic


class TestKeelResistanceEvolution:
    """Test keel resistance evolution computation."""
    
    def test_resistance_increases_with_age(self):
        """Keel resistance should increase with age."""
        formation = FreezeBondFormationCriterion()
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=1.0e5,
        )
        bond_model = FreezeBondModel(formation, growth)
        ridge = LindqvistRidgeResistance()
        
        ages = np.linspace(0, 5e5, 50)
        result = compute_keel_resistance_evolution(ridge, bond_model, ages, 263.15)
        
        total = result['total_resistance_N_per_m']
        assert np.all(np.diff(total) >= 0)
    
    def test_resistance_components_sum(self):
        """Resistance components should sum to total."""
        formation = FreezeBondFormationCriterion()
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=1.0e5,
        )
        bond_model = FreezeBondModel(formation, growth)
        ridge = LindqvistRidgeResistance()
        
        ages = np.array([0.0, 1.0e5, 5.0e5])
        result = compute_keel_resistance_evolution(ridge, bond_model, ages, 263.15)
        
        for i in range(len(ages)):
            total = (result['rubble_friction_N_per_m'][i] +
                     result['freeze_bond_N_per_m'][i] +
                     result['buoyancy_N_per_m'][i])
            assert abs(result['total_resistance_N_per_m'][i] - total) < 1e-6


class TestFreezeBondStudy:
    """Test complete freeze-bond study."""
    
    def test_study_returns_expected_keys(self):
        """Study should return expected dictionary keys."""
        result = run_freeze_bond_study(max_age_hours=100.0, n_points=50)
        
        assert 'bond_evolution' in result
        assert 'resistance_evolution' in result
        assert 'parameters' in result
        assert 'summary' in result
    
    def test_study_summary_metrics(self):
        """Study summary should contain key metrics."""
        result = run_freeze_bond_study(max_age_hours=240.0, n_points=100)
        
        summary = result['summary']
        assert 'final_bond_strength_Pa' in summary
        assert 'final_resistance_N_per_m' in summary
        assert 'resistance_increase_percent' in summary
        
        # Resistance should increase
        assert summary['resistance_increase_percent'] > 0
    
    def test_study_physical_consistency(self):
        """Study results should be physically consistent."""
        result = run_freeze_bond_study(
            max_age_hours=240.0,
            asymptotic_tensile_Pa=1.0e6,
            growth_time_hours=100.0,
        )
        
        bond_evo = result['bond_evolution']
        resistance_evo = result['resistance_evolution']
        
        # Final bond strength should be close to asymptotic
        final_ratio = bond_evo['strength_ratio'][-1]
        assert final_ratio > 0.9  # >90% of asymptotic
        
        # Resistance should increase significantly
        initial = resistance_evo['total_resistance_N_per_m'][0]
        final = resistance_evo['total_resistance_N_per_m'][-1]
        assert final > 1.5 * initial  # at least 50% increase


class TestIntegrationWithDEM:
    """Test integration with DEM bond parameters."""
    
    def test_bond_strength_replaces_constant(self):
        """Time-dependent strength should replace constant DEM bond strength."""
        formation = FreezeBondFormationCriterion()
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=1.0e5,
        )
        model = FreezeBondModel(formation, growth)
        
        # At age=0, strength is 0 (unbonded)
        tensile_0, _ = model.bond_strength_at_age(0.0, 263.15)
        assert tensile_0 == 0.0
        
        # At large age, strength approaches asymptotic
        tensile_inf, _ = model.bond_strength_at_age(1.0e7, 263.15)
        assert tensile_inf > 0.99e6
        
        # This demonstrates the transition from unbonded to fully bonded
    
    def test_temperature_dependence(self):
        """Bond formation should depend on temperature."""
        formation = FreezeBondFormationCriterion()
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=1.0e5,
        )
        model = FreezeBondModel(formation, growth)
        
        age = 1.0e5
        
        # Below freezing: bond forms
        tensile_cold, _ = model.bond_strength_at_age(age, 263.15)
        assert tensile_cold > 0.0
        
        # Above freezing: no bond
        tensile_warm, _ = model.bond_strength_at_age(age, 280.0)
        assert tensile_warm == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
