"""Tests for DEM auto-calibration framework (p284w3-n05 / p360w1-n03 maturation).

Tests cover:
1. Forward problem: repose angle, shear box, compression simulations
2. Objective function computation
3. Bayesian optimization (GP surrogate, EI/UCB acquisition)
4. MCMC uncertainty quantification
5. Complete calibration pipeline
6. Material presets (glass beads, sand, gravel)
7. Acceptance criteria validation
8. Gelman-Rubin R-hat convergence diagnostic
9. Posterior predictive checks
"""
import math
import pytest
import numpy as np

from tensordem.auto_calibration import (
    ReposeBenchmark,
    ShearBoxBenchmark,
    CompressionBenchmark,
    ExperimentalData,
    CalibrationParameter,
    build_material,
    _simulate_repose,
    _simulate_shear_box,
    _simulate_compression,
    objective_function,
    GaussianProcessSurrogate,
    latin_hypercube_sample,
    bayesian_optimize,
    log_posterior,
    metropolis_hastings,
    run_calibration,
    gelman_rubin_rhat,
    posterior_predictive_check,
    GLASS_BEADS_EXP,
    GLASS_BEADS_PARAMS,
    SAND_EXP,
    SAND_PARAMS,
    GRAVEL_EXP,
    GRAVEL_PARAMS,
    MATERIAL_PRESETS,
    DEFAULT_PARAMETERS,
)
from tensordem.granular3d_core import Material3D as CoreMaterial3D


# Small benchmark configs for fast testing
SMALL_REPOSE = ReposeBenchmark(cylinder_radius=0.02, fill_height=0.02, seed=42)
SMALL_SHEAR = ShearBoxBenchmark(box_width=0.02, box_depth=0.02, fill_height=0.015, seed=42)
SMALL_COMP = CompressionBenchmark(specimen_radius=0.015, specimen_height=0.03, seed=42)


# ============================================================================
# Test 1-3: Forward problem - Repose angle simulation
# ============================================================================

class TestReposeSimulation:
    """Test angle-of-repose forward simulation."""

    def test_repose_angle_increases_with_friction(self):
        """Higher friction should produce steeper repose angles."""
        mat_low = CoreMaterial3D(mu=0.2, mu_rolling=0.02, restitution=0.3, kn=1e5)
        mat_high = CoreMaterial3D(mu=0.6, mu_rolling=0.08, restitution=0.3, kn=1e5)

        result_low = _simulate_repose(mat_low, SMALL_REPOSE)
        result_high = _simulate_repose(mat_high, SMALL_REPOSE)

        assert result_high["repose_angle_deg"] >= result_low["repose_angle_deg"], \
            f"High-friction angle ({result_high['repose_angle_deg']:.1f}) should >= " \
            f"low-friction angle ({result_low['repose_angle_deg']:.1f})"

    def test_repose_angle_increases_with_rolling_resistance(self):
        """Higher rolling resistance should produce steeper repose angles."""
        mat_low_r = CoreMaterial3D(mu=0.4, mu_rolling=0.0, restitution=0.3, kn=1e5)
        mat_high_r = CoreMaterial3D(mu=0.4, mu_rolling=0.1, restitution=0.3, kn=1e5)

        result_low = _simulate_repose(mat_low_r, SMALL_REPOSE)
        result_high = _simulate_repose(mat_high_r, SMALL_REPOSE)

        assert result_high["repose_angle_deg"] >= result_low["repose_angle_deg"], \
            "Rolling resistance should increase or maintain repose angle"

    def test_repose_angle_reasonable_range(self):
        """Repose angle should be in physically reasonable range (0-60 deg)."""
        mat = CoreMaterial3D(mu=0.4, mu_rolling=0.05, restitution=0.3, kn=1e5)
        result = _simulate_repose(mat, SMALL_REPOSE)

        angle = result["repose_angle_deg"]
        assert 0.0 <= angle <= 60.0, f"Repose angle {angle:.1f} outside reasonable range"
        assert result["n_particles"] > 0
        assert result["ke_final_J"] >= 0.0


# ============================================================================
# Test 4-5: Forward problem - Shear box simulation
# ============================================================================

class TestShearBoxSimulation:
    """Test direct shear box forward simulation."""

    def test_shear_phi_increases_with_friction(self):
        """Higher friction should produce higher internal friction angle."""
        mat_low = CoreMaterial3D(mu=0.2, mu_rolling=0.02, restitution=0.3, kn=1e5)
        mat_high = CoreMaterial3D(mu=0.6, mu_rolling=0.02, restitution=0.3, kn=1e5)

        result_low = _simulate_shear_box(mat_low, SMALL_SHEAR)
        result_high = _simulate_shear_box(mat_high, SMALL_SHEAR)

        assert result_high["phi_deg"] >= result_low["phi_deg"], \
            f"High-friction phi ({result_high['phi_deg']:.1f}) should >= " \
            f"low-friction phi ({result_low['phi_deg']:.1f})"

    def test_shear_box_produces_stress_history(self):
        """Shear box should produce non-empty stress history."""
        mat = CoreMaterial3D(mu=0.4, mu_rolling=0.05, restitution=0.3, kn=1e5)
        result = _simulate_shear_box(mat, SMALL_SHEAR)

        assert len(result["shear_stress_history"]) > 0
        assert len(result["displacement_history"]) > 0
        assert result["peak_shear_Pa"] >= 0.0
        assert result["n_particles"] > 0


# ============================================================================
# Test 6-7: Forward problem - Compression simulation
# ============================================================================

class TestCompressionSimulation:
    """Test uniaxial compression forward simulation."""

    def test_compression_produces_stress_strain_curve(self):
        """Compression should produce non-empty stress-strain curve."""
        mat = CoreMaterial3D(mu=0.4, mu_rolling=0.05, restitution=0.3, kn=1e5)
        result = _simulate_compression(mat, SMALL_COMP)

        assert len(result["stress_strain_curve"]) > 0
        assert result["n_particles"] > 0

    def test_compression_E_positive(self):
        """Elastic modulus should be non-negative."""
        mat = CoreMaterial3D(mu=0.4, mu_rolling=0.05, restitution=0.3, kn=1e5)
        result = _simulate_compression(mat, SMALL_COMP)

        assert result["E_Pa"] >= 0.0, f"E = {result['E_Pa']:.0f} Pa should be non-negative"
        assert 0.0 <= result["nu"] <= 0.5, f"nu = {result['nu']:.3f} should be in [0, 0.5]"


# ============================================================================
# Test 8-9: Objective function
# ============================================================================

class TestObjectiveFunction:
    """Test objective function computation."""

    def test_objective_zero_for_perfect_match(self):
        """Objective should be near zero when simulation matches experiment."""
        mat = CoreMaterial3D(mu=0.4, mu_rolling=0.05, restitution=0.3, kn=1e5)
        sim_result = _simulate_repose(mat, SMALL_REPOSE)

        exp_data = ExperimentalData(
            repose_angle_deg=sim_result["repose_angle_deg"],
            repose_angle_std=1.0,
        )

        params = {"mu": 0.4, "mu_rolling": 0.05, "restitution": 0.3, "kt_ratio": 0.5}
        result = objective_function(params, exp_data, repose_bench=SMALL_REPOSE)

        assert result["total_objective"] < 0.1, \
            f"Objective {result['total_objective']:.4f} should be near zero for perfect match"

    def test_objective_components_present(self):
        """Objective should have correct component keys."""
        exp_data = ExperimentalData(
            repose_angle_deg=30.0, repose_angle_std=2.0,
            shear_phi_deg=30.0, shear_phi_std=2.0,
        )
        params = {"mu": 0.5, "mu_rolling": 0.05, "restitution": 0.3, "kt_ratio": 0.5}
        result = objective_function(params, exp_data,
                                    repose_bench=SMALL_REPOSE,
                                    shear_bench=SMALL_SHEAR)

        assert "repose" in result["components"]
        assert "shear_phi" in result["components"]
        assert result["total_objective"] >= 0.0


# ============================================================================
# Test 10-13: Bayesian Optimization
# ============================================================================

class TestBayesianOptimization:
    """Test Bayesian optimization components."""

    def test_gp_surrogate_fit_and_predict(self):
        """GP should fit data and produce reasonable predictions."""
        gp = GaussianProcessSurrogate(length_scale=0.5, noise=1e-4)

        X = np.array([[0.2], [0.4], [0.6], [0.8]])
        y = np.array([1.0, 0.5, 0.3, 0.6])

        gp.fit(X, y)
        mu, var = gp.predict(np.array([[0.3], [0.5], [0.7]]))

        assert len(mu) == 3
        assert len(var) == 3
        assert all(v > 0 for v in var), "Variance should be positive"

    def test_expected_improvement_positive(self):
        """EI should be non-negative."""
        gp = GaussianProcessSurrogate(length_scale=0.5, noise=1e-4)
        X = np.array([[0.2], [0.5], [0.8]])
        y = np.array([1.0, 0.5, 0.8])
        gp.fit(X, y)

        X_test = np.array([[0.3], [0.6]])
        ei = gp.expected_improvement(X_test, y_best=0.5, xi=0.01)

        assert len(ei) == 2
        assert all(e >= 0 for e in ei), "EI should be non-negative"

    def test_latin_hypercube_sampling(self):
        """LHS should produce well-distributed samples."""
        samples = latin_hypercube_sample(20, 4, seed=42)

        assert samples.shape == (20, 4)
        assert np.all(samples >= 0) and np.all(samples <= 1)

        for d in range(4):
            assert np.min(samples[:, d]) < 0.15
            assert np.max(samples[:, d]) > 0.85

    def test_bayesian_optimization_converges(self):
        """BO should find a minimum for a simple quadratic function."""
        def quadratic(params: dict) -> float:
            x = params["x"]
            return (x - 0.5) ** 2

        params = [CalibrationParameter("x", 0.0, 1.0, prior_mean=0.5, prior_std=0.2)]

        result = bayesian_optimize(
            quadratic, params,
            n_initial=5, n_iterations=10,
            acquisition="EI", seed=42,
        )

        assert result["best_objective"] < 0.05, \
            f"BO should find near-optimal: obj={result['best_objective']:.4f}"
        assert abs(result["best_params"]["x"] - 0.5) < 0.15, \
            f"Best x={result['best_params']['x']:.3f} should be near 0.5"
        assert result["n_evaluations"] == 15


# ============================================================================
# Test 14-15: MCMC Uncertainty Quantification
# ============================================================================

class TestMCMC:
    """Test MCMC uncertainty quantification."""

    def test_mcmc_acceptance_rate_reasonable(self):
        """MCMC acceptance rate should be in reasonable range (0.05-0.95)."""
        exp_data = ExperimentalData(repose_angle_deg=25.0, repose_angle_std=3.0)
        params = [
            CalibrationParameter("mu", 0.1, 0.7, prior_mean=0.4, prior_std=0.15),
        ]

        result = metropolis_hastings(
            exp_data, params,
            n_samples=30, burn_in=10,
            step_size=0.1, seed=42,
            repose_bench=SMALL_REPOSE,
            max_steps=100,
        )

        assert 0.05 <= result["acceptance_rate"] <= 0.95, \
            f"Acceptance rate {result['acceptance_rate']:.3f} outside reasonable range"

    def test_mcmc_posterior_has_uncertainty(self):
        """Posterior should have non-zero uncertainty (CI width > 0)."""
        exp_data = ExperimentalData(repose_angle_deg=25.0, repose_angle_std=3.0)
        params = [
            CalibrationParameter("mu", 0.1, 0.7, prior_mean=0.4, prior_std=0.15),
        ]

        result = metropolis_hastings(
            exp_data, params,
            n_samples=30, burn_in=10,
            step_size=0.1, seed=42,
            repose_bench=SMALL_REPOSE,
            max_steps=100,
        )

        mu_summary = result["posterior_summary"]["mu"]
        ci_width = mu_summary["ci_95_high"] - mu_summary["ci_95_low"]
        assert ci_width > 0.0, "Posterior CI should have non-zero width"
        assert mu_summary["ci_95_low"] < mu_summary["ci_95_high"]


# ============================================================================
# Test 16-18: Complete calibration pipeline (FIXED: use small benchmarks)
# ============================================================================

class TestCalibrationPipeline:
    """Test complete auto-calibration pipeline."""

    def test_pipeline_runs_without_error(self):
        """Pipeline should complete without errors."""
        exp_data = ExperimentalData(
            repose_angle_deg=25.0,
            repose_angle_std=3.0,
        )
        params = [
            CalibrationParameter("mu", 0.1, 0.7, prior_mean=0.4, prior_std=0.15),
            CalibrationParameter("mu_rolling", 0.0, 0.1, prior_mean=0.05, prior_std=0.03),
        ]

        result = run_calibration(
            exp_data, params,
            n_bo_initial=3, n_bo_iterations=3,
            n_mcmc_samples=20, n_mcmc_burn_in=5,
            seed=42,
            radius=0.003,
            max_steps=80,
            repose_bench=SMALL_REPOSE,
            shear_bench=SMALL_SHEAR,
        )

        assert "best_params" in result
        assert "best_objective" in result
        assert "mcmc_posterior" in result
        assert "validation" in result
        assert result["elapsed_seconds"] > 0

    def test_pipeline_best_params_within_bounds(self):
        """Best parameters should be within specified bounds."""
        exp_data = ExperimentalData(repose_angle_deg=30.0, repose_angle_std=2.0)
        params = [
            CalibrationParameter("mu", 0.2, 0.6, prior_mean=0.4, prior_std=0.1),
            CalibrationParameter("mu_rolling", 0.01, 0.08, prior_mean=0.04, prior_std=0.02),
        ]

        result = run_calibration(
            exp_data, params,
            n_bo_initial=3, n_bo_iterations=3,
            n_mcmc_samples=15, n_mcmc_burn_in=5,
            seed=42,
            radius=0.003,
            max_steps=80,
            repose_bench=SMALL_REPOSE,
        )

        for p in params:
            val = result["best_params"][p.name]
            assert p.low <= val <= p.high, \
                f"Parameter {p.name} = {val} outside bounds [{p.low}, {p.high}]"

    def test_pipeline_mcmc_ci_contains_prior_mean(self):
        """With weak data, MCMC CI should contain the prior mean."""
        exp_data = ExperimentalData(
            repose_angle_deg=25.0,
            repose_angle_std=50.0,  # very uncertain
        )
        params = [
            CalibrationParameter("mu", 0.1, 0.7, prior_mean=0.4, prior_std=0.15),
        ]

        result = metropolis_hastings(
            exp_data, params,
            n_samples=30, burn_in=10,
            step_size=0.15, seed=42,
            repose_bench=SMALL_REPOSE,
            max_steps=100,
        )

        mu_summary = result["posterior_summary"]["mu"]
        assert mu_summary["ci_95_low"] <= 0.55, \
            f"CI low {mu_summary['ci_95_low']:.3f} should include prior region"
        assert mu_summary["ci_95_high"] >= 0.25, \
            f"CI high {mu_summary['ci_95_high']:.3f} should include prior region"


# ============================================================================
# Test 19-22: Material presets
# ============================================================================

class TestMaterialPresets:
    """Test material-specific calibration presets."""

    def test_presets_exist(self):
        """All three material presets should exist."""
        assert "glass_beads" in MATERIAL_PRESETS
        assert "sand" in MATERIAL_PRESETS
        assert "gravel" in MATERIAL_PRESETS

    def test_glass_beads_params(self):
        """Glass beads should have low friction parameters."""
        exp, params = MATERIAL_PRESETS["glass_beads"]
        mu_param = [p for p in params if p.name == "mu"][0]
        assert mu_param.low <= 0.25 <= mu_param.high
        assert mu_param.prior_mean is not None
        assert mu_param.prior_mean < 0.4

    def test_sand_params_higher_friction(self):
        """Sand should have higher friction than glass beads."""
        _, glass_params = MATERIAL_PRESETS["glass_beads"]
        _, sand_params = MATERIAL_PRESETS["sand"]

        glass_mu = [p for p in glass_params if p.name == "mu"][0]
        sand_mu = [p for p in sand_params if p.name == "mu"][0]

        assert sand_mu.prior_mean > glass_mu.prior_mean

    def test_gravel_params_highest_friction(self):
        """Gravel should have highest friction and rolling resistance."""
        _, sand_params = MATERIAL_PRESETS["sand"]
        _, gravel_params = MATERIAL_PRESETS["gravel"]

        sand_mu = [p for p in sand_params if p.name == "mu"][0]
        gravel_mu = [p for p in gravel_params if p.name == "mu"][0]

        assert gravel_mu.prior_mean >= sand_mu.prior_mean

    def test_experimental_data_has_repose_angles(self):
        """All presets should have repose angle data."""
        for name, (exp, params) in MATERIAL_PRESETS.items():
            assert exp.repose_angle_deg is not None
            assert 10.0 <= exp.repose_angle_deg <= 50.0


# ============================================================================
# Test 23-25: Build material and parameter utilities
# ============================================================================

class TestUtilities:
    """Test utility functions."""

    def test_build_material(self):
        """build_material should create valid Material3D."""
        params = {"mu": 0.4, "mu_rolling": 0.05, "restitution": 0.3, "kt_ratio": 0.5}
        mat = build_material(params, radius=0.005, density=2500.0, kn=1e5)

        assert mat.mu == 0.4
        assert mat.mu_rolling == 0.05
        assert mat.restitution == 0.3
        assert mat.kt_ratio == 0.5

    def test_default_parameters(self):
        """Default parameters should cover the standard calibration space."""
        assert len(DEFAULT_PARAMETERS) == 4
        names = [p.name for p in DEFAULT_PARAMETERS]
        assert "mu" in names
        assert "mu_rolling" in names
        assert "restitution" in names
        assert "kt_ratio" in names

    def test_calibration_parameter_bounds(self):
        """All default parameters should have valid bounds."""
        for p in DEFAULT_PARAMETERS:
            assert p.low < p.high
            assert p.low >= 0.0


# ============================================================================
# Test 26: Acceptance criteria - repose angle error < 5 deg
# ============================================================================

class TestAcceptanceCriteria:
    """Test that acceptance criteria can be met."""

    def test_repose_angle_error_within_5_degrees(self):
        """With proper calibration, repose angle error should be < 5 deg."""
        mat = CoreMaterial3D(mu=0.4, mu_rolling=0.05, restitution=0.3, kn=1e5)
        sim_result = _simulate_repose(mat, SMALL_REPOSE)

        exp_data = ExperimentalData(
            repose_angle_deg=sim_result["repose_angle_deg"],
            repose_angle_std=2.0,
        )

        params = {"mu": 0.4, "mu_rolling": 0.05, "restitution": 0.3, "kt_ratio": 0.5}
        result = objective_function(params, exp_data, repose_bench=SMALL_REPOSE)

        repose_component = result["components"].get("repose", 0.0)
        assert repose_component < 6.25, \
            f"Repose component {repose_component:.4f} exceeds threshold for 5 deg error"


# ============================================================================
# Test 27-28: Log-posterior computation
# ============================================================================

class TestLogPosterior:
    """Test log-posterior computation for MCMC."""

    def test_log_posterior_finite_for_valid_params(self):
        """Log-posterior should be finite for valid parameters."""
        exp_data = ExperimentalData(repose_angle_deg=25.0, repose_angle_std=3.0)
        params = [
            CalibrationParameter("mu", 0.1, 0.7, prior_mean=0.4, prior_std=0.15),
        ]
        test_params = {"mu": 0.4}

        lp = log_posterior(test_params, exp_data, params, repose_bench=SMALL_REPOSE, max_steps=100)
        assert math.isfinite(lp), f"Log-posterior should be finite, got {lp}"

    def test_log_posterior_decreases_with_mismatch(self):
        """Log-posterior should decrease as parameters deviate from optimum."""
        exp_data = ExperimentalData(repose_angle_deg=25.0, repose_angle_std=3.0)
        params = [
            CalibrationParameter("mu", 0.1, 0.7, prior_mean=0.4, prior_std=0.15),
        ]

        lp_good = log_posterior({"mu": 0.4}, exp_data, params, repose_bench=SMALL_REPOSE, max_steps=100)
        lp_bad = log_posterior({"mu": 0.1}, exp_data, params, repose_bench=SMALL_REPOSE, max_steps=100)

        assert lp_good > lp_bad, \
            f"Log-posterior at mu=0.4 ({lp_good:.2f}) should exceed mu=0.1 ({lp_bad:.2f})"


# ============================================================================
# Test 29: UCB acquisition function
# ============================================================================

class TestUCB:
    """Test UCB acquisition function."""

    def test_ucb_returns_array(self):
        """UCB should return an array of the correct shape."""
        gp = GaussianProcessSurrogate(length_scale=0.5, noise=1e-4)
        X = np.array([[0.2], [0.5], [0.8]])
        y = np.array([1.0, 0.5, 0.8])
        gp.fit(X, y)

        X_test = np.array([[0.3], [0.6], [0.7]])
        ucb_values = gp.ucb(X_test, kappa=2.0)

        assert len(ucb_values) == 3
        assert all(isinstance(v, (int, float, np.floating)) for v in ucb_values)


# ============================================================================
# Test 30-31: Gelman-Rubin R-hat convergence diagnostic
# ============================================================================

class TestGelmanRubin:
    """Test Gelman-Rubin R-hat convergence diagnostic."""

    def test_rhat_converged_chains_near_one(self):
        """R-hat should be near 1.0 for identical (converged) chains."""
        rng = np.random.default_rng(42)
        # Two chains from same distribution
        chain1 = [{"mu": rng.normal(0.5, 0.05)} for _ in range(100)]
        chain2 = [{"mu": rng.normal(0.5, 0.05)} for _ in range(100)]

        rhat = gelman_rubin_rhat([chain1, chain2], ["mu"])
        assert 0.9 < rhat["mu"] < 1.1, \
            f"R-hat for converged chains should be near 1.0, got {rhat['mu']:.3f}"

    def test_rhat_divergent_chains_above_threshold(self):
        """R-hat should be > 1.1 for chains with different means."""
        rng = np.random.default_rng(42)
        # Two chains from different distributions
        chain1 = [{"mu": rng.normal(0.3, 0.02)} for _ in range(100)]
        chain2 = [{"mu": rng.normal(0.7, 0.02)} for _ in range(100)]

        rhat = gelman_rubin_rhat([chain1, chain2], ["mu"])
        assert rhat["mu"] > 1.1, \
            f"R-hat for divergent chains should be > 1.1, got {rhat['mu']:.3f}"

    def test_rhat_single_chain_returns_nan(self):
        """R-hat with single chain should return NaN (undefined)."""
        chain1 = [{"mu": 0.5} for _ in range(50)]
        rhat = gelman_rubin_rhat([chain1], ["mu"])
        assert math.isnan(rhat["mu"]), "R-hat with 1 chain should be NaN"


# ============================================================================
# Test 32-33: Posterior predictive checks
# ============================================================================

class TestPosteriorPredictive:
    """Test posterior predictive check functionality."""

    def test_posterior_predictive_returns_coverage(self):
        """Posterior predictive check should return coverage info."""
        # Create a mock MCMC result
        samples = [{"mu": 0.4 + 0.01 * i} for i in range(50)]
        mcmc_result = {"samples": samples}

        exp_data = ExperimentalData(repose_angle_deg=25.0, repose_angle_std=3.0)
        params = [CalibrationParameter("mu", 0.1, 0.7, prior_mean=0.4, prior_std=0.15)]

        pp = posterior_predictive_check(
            mcmc_result, exp_data, params,
            n_predictive=10, max_steps=80,
            repose_bench=SMALL_REPOSE,
            radius=0.003,
        )

        assert "repose_angle" in pp
        assert "exp_value" in pp["repose_angle"]
        assert "pred_low_95" in pp["repose_angle"]
        assert "pred_high_95" in pp["repose_angle"]
        assert "covered_95" in pp["repose_angle"]

    def test_posterior_predictive_coverage_boolean(self):
        """Coverage flag should be boolean."""
        samples = [{"mu": 0.5} for _ in range(20)]
        mcmc_result = {"samples": samples}

        exp_data = ExperimentalData(repose_angle_deg=30.0, repose_angle_std=5.0)
        params = [CalibrationParameter("mu", 0.1, 0.7, prior_mean=0.4, prior_std=0.15)]

        pp = posterior_predictive_check(
            mcmc_result, exp_data, params,
            n_predictive=5, max_steps=80,
            repose_bench=SMALL_REPOSE,
            radius=0.003,
        )

        assert isinstance(pp["repose_angle"]["covered_95"], bool)
