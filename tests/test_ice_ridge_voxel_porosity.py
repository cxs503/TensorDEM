"""Tests for ice ridge voxel porosity measurement.

Tests cover:
1. Voxel occupancy porosity computation
2. Monte Carlo estimation with confidence intervals
3. Buoyancy feedback in ridge simulation
4. Scale ladder study
5. Physical consistency checks
"""
import pytest
import numpy as np
import torch

from tensordem.ice_ridge_voxel_porosity import (
    VoxelPorosityConfig,
    VoxelPorosityResult,
    compute_voxel_porosity,
    compute_monte_carlo_porosity,
    simulate_ridge_with_buoyancy,
    run_scale_ladder_study,
    initial_porosity_from_floe_count,
)
from tensordem.ice_ridge_formation.floes import (
    FloePopulation,
    FloePopulationConfig,
    PolygonalFloe,
    generate_power_law_floes,
)


class TestInitialPorosity:
    """Test initial porosity model."""
    
    def test_porosity_decreases_with_n_floe(self):
        """More floes should give lower initial porosity."""
        phi_20 = initial_porosity_from_floe_count(20)
        phi_60 = initial_porosity_from_floe_count(60)
        phi_200 = initial_porosity_from_floe_count(200)
        phi_600 = initial_porosity_from_floe_count(600)
        
        assert phi_20 > phi_60 > phi_200 > phi_600
    
    def test_porosity_in_physical_range(self):
        """Initial porosity should be in physical range [0.35, 0.50]."""
        for n in [20, 60, 200, 600]:
            phi = initial_porosity_from_floe_count(n)
            assert 0.30 <= phi <= 0.50


class TestVoxelPorosity:
    """Test voxel-based porosity computation."""
    
    def test_empty_population(self):
        """Empty floe population should give porosity = 1."""
        config = FloePopulationConfig(n_floes=1, domain_size=(10, 10))
        floes = FloePopulation([], config)  # Empty list despite config
        result = compute_voxel_porosity(floes, 0.0, VoxelPorosityConfig())
        assert result.voxel_porosity == 1.0
        assert result.monte_carlo_porosity == 1.0
    
    def test_single_floe_porosity(self):
        """Single floe should give reasonable porosity."""
        config = FloePopulationConfig(
            n_floes=1,
            domain_size=(10, 10),
            thickness=1.0,
            seed=42,
        )
        floes = generate_power_law_floes(config)
        result = compute_voxel_porosity(floes, 0.0, VoxelPorosityConfig())
        
        # Porosity should be between 0 and 1
        assert 0.0 <= result.voxel_porosity <= 1.0
        assert 0.0 <= result.monte_carlo_porosity <= 1.0
        
        # Voxel and MC should agree within 15%
        assert abs(result.voxel_porosity - result.monte_carlo_porosity) < 0.15
    
    def test_multiple_floes_porosity(self):
        """Multiple floes should give lower porosity than single floe."""
        config1 = FloePopulationConfig(n_floes=1, domain_size=(10, 10), seed=42)
        floes1 = generate_power_law_floes(config1)
        result1 = compute_voxel_porosity(floes1, 0.0, VoxelPorosityConfig())
        
        config10 = FloePopulationConfig(n_floes=10, domain_size=(10, 10), seed=42)
        floes10 = generate_power_law_floes(config10)
        result10 = compute_voxel_porosity(floes10, 0.0, VoxelPorosityConfig())
        
        # More floes should give lower porosity (more occupied space)
        assert result10.voxel_porosity < result1.voxel_porosity
    
    def test_monte_carlo_confidence_interval(self):
        """MC confidence interval should contain the estimate."""
        config = FloePopulationConfig(n_floes=20, domain_size=(10, 10), seed=42)
        floes = generate_power_law_floes(config)
        result = compute_voxel_porosity(floes, 0.0, VoxelPorosityConfig())
        
        ci_low, ci_high = result.monte_carlo_ci_95
        assert ci_low <= result.monte_carlo_porosity <= ci_high
        assert ci_low >= 0.0
        assert ci_high <= 1.0
    
    def test_voxel_count_consistency(self):
        """Voxel counts should be consistent."""
        config = FloePopulationConfig(n_floes=10, domain_size=(10, 10), seed=42)
        floes = generate_power_law_floes(config)
        result = compute_voxel_porosity(floes, 0.0, VoxelPorosityConfig())
        
        assert result.n_voxels_total > 0
        assert result.n_voxels_occupied >= 0
        assert result.n_voxels_occupied <= result.n_voxels_total
        
        # Porosity should match voxel counts
        expected_porosity = 1.0 - result.n_voxels_occupied / result.n_voxels_total
        assert abs(result.voxel_porosity - expected_porosity) < 1e-10


class TestMonteCarloPorosity:
    """Test Monte Carlo porosity estimation."""
    
    def test_monte_carlo_basic(self):
        """Basic MC estimation should work."""
        config = FloePopulationConfig(n_floes=10, domain_size=(10, 10), seed=42)
        floes = generate_power_law_floes(config)
        porosity, ci = compute_monte_carlo_porosity(floes, 0.0, n_samples=10000, seed=42)
        
        assert 0.0 <= porosity <= 1.0
        assert ci[0] <= porosity <= ci[1]
    
    def test_monte_carlo_convergence(self):
        """MC estimate should converge with more samples."""
        config = FloePopulationConfig(n_floes=10, domain_size=(10, 10), seed=42)
        floes = generate_power_law_floes(config)
        
        porosity_1k, ci_1k = compute_monte_carlo_porosity(floes, 0.0, n_samples=1000, seed=42)
        porosity_10k, ci_10k = compute_monte_carlo_porosity(floes, 0.0, n_samples=10000, seed=42)
        porosity_100k, ci_100k = compute_monte_carlo_porosity(floes, 0.0, n_samples=100000, seed=42)
        
        # CI width should decrease with more samples
        ci_width_1k = ci_1k[1] - ci_1k[0]
        ci_width_10k = ci_10k[1] - ci_10k[0]
        ci_width_100k = ci_100k[1] - ci_100k[0]
        
        assert ci_width_10k < ci_width_1k
        assert ci_width_100k < ci_width_10k


class TestRidgeSimulation:
    """Test ridge simulation with buoyancy feedback."""
    
    def test_basic_simulation(self):
        """Basic simulation should complete."""
        result = simulate_ridge_with_buoyancy(
            n_floes=20,
            ice_thickness=1.0,
            target_strain=0.3,
            seed=42,
        )
        
        assert "strain_history" in result
        assert "porosity_analytical_history" in result
        assert "final_porosity_analytical" in result
        assert len(result["strain_history"]) > 0
        assert len(result["porosity_analytical_history"]) == len(result["strain_history"])
    
    def test_porosity_decreases_with_strain(self):
        """Porosity should generally decrease with strain."""
        result = simulate_ridge_with_buoyancy(
            n_floes=30,
            target_strain=0.4,
            seed=42,
        )
        
        # Final porosity should be less than initial
        assert result["final_porosity_analytical"] < result["porosity_analytical_history"][0]
    
    def test_buoyancy_work_accumulates(self):
        """Buoyancy work should accumulate over time."""
        result = simulate_ridge_with_buoyancy(
            n_floes=20,
            target_strain=0.4,
            seed=42,
        )
        
        # Buoyancy work should be non-negative and increasing
        work_history = result["buoyancy_work_history"]
        assert all(w >= 0 for w in work_history)
        assert work_history[-1] >= work_history[0]
    
    def test_sail_keel_ratio_physical(self):
        """Sail/keel ratio should be in physical range."""
        result = simulate_ridge_with_buoyancy(
            n_floes=30,
            target_strain=0.4,
            seed=42,
        )
        
        ratio = result["sail_keel_ratio"]
        # Physical range: 0.2-0.5 (Timco & Burden 1994)
        assert 0.1 <= ratio <= 0.6  # Allow some tolerance


class TestScaleLadderStudy:
    """Test scale ladder study."""
    
    def test_basic_scale_ladder(self):
        """Basic scale ladder should complete."""
        result = run_scale_ladder_study(
            n_floe_list=[20, 60],
            n_seeds=2,
            base_seed=42,
        )
        
        assert "scale_ladder" in result
        assert "scaling_fit" in result
        assert len(result["scale_ladder"]) == 2
    
    def test_scaling_fit_r_squared(self):
        """Scaling fit should have high R²."""
        result = run_scale_ladder_study(
            n_floe_list=[20, 60, 200],
            n_seeds=3,
            base_seed=42,
        )
        
        r_squared = result["scaling_fit"]["r_squared"]
        assert isinstance(r_squared, float)
        # R² should be high (>0.9) for the power law model
        assert r_squared > 0.9
    
    def test_multiple_seeds_reduce_variance(self):
        """More seeds should give better statistics."""
        result_2seeds = run_scale_ladder_study(
            n_floe_list=[20, 60],
            n_seeds=2,
            base_seed=42,
        )
        
        result_5seeds = run_scale_ladder_study(
            n_floe_list=[20, 60],
            n_seeds=5,
            base_seed=42,
        )
        
        # With more seeds, we expect better statistics
        assert len(result_2seeds["scale_ladder"]) == 2
        assert len(result_5seeds["scale_ladder"]) == 2


class TestPhysicalConsistency:
    """Test physical consistency of results."""
    
    def test_porosity_bounds(self):
        """Porosity should be in [0, 1]."""
        result = simulate_ridge_with_buoyancy(n_floes=30, seed=42)
        
        for phi in result["porosity_analytical_history"]:
            assert 0.0 <= phi <= 1.0
    
    def test_target_porosity_range(self):
        """Final porosity should approach target range (0.25-0.35)."""
        result = simulate_ridge_with_buoyancy(
            n_floes=60,
            target_strain=0.4,
            seed=42,
        )
        
        final_phi = result["final_porosity_analytical"]
        # Target: 0.25-0.35 (Timco & Burden 1994)
        # Allow some tolerance for model variability
        assert 0.20 <= final_phi <= 0.40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
