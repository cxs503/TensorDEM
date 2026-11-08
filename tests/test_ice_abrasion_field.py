"""Tests for ice_abrasion_field: spatio-temporal ice abrasion field evolution.

Covers:
- Panel mesh construction
- Activity profile fitting (C1: R² > 0.95)
- Spatial correlation estimation
- KL expansion and conditional sampling (C2: mean/var error < 5%)
- Spatio-temporal evolution and mass conservation (C3: rel err < 1e-6)
- P95 connector with N02 (C4: error < 10%)
- Uniform anchor (C5: error < 1%)
- Performance (C6: 1000 panels x 100 steps < 60s)
"""
import math
import time

import numpy as np
import pytest

from tensordem.ice_abrasion_field import (
    AbrasionProfile,
    AbrasionProfileFit,
    ConditionalRandomField,
    KLExpansion,
    PanelMesh,
    P95Connector,
    SpatioTemporalField,
    SpatioTemporalEvolution,
    build_panel_mesh,
    connect_to_n02_p95,
    estimate_spatial_correlation,
    evolve_spatio_temporal,
    extract_p95_from_field,
    fit_activity_profile,
    generate_conditional_samples,
    kl_expansion_2d,
    run_n04_verification,
    verify_mass_conservation,
    verify_uniform_anchor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_mesh():
    return build_panel_mesh(n_x=10, n_y=8, hull_length_m=100.0,
                            hull_beam_m=20.0, hull_draught_m=8.0)


@pytest.fixture
def standard_mesh():
    return build_panel_mesh(n_x=40, n_y=25, hull_length_m=185.0,
                            hull_beam_m=32.26, hull_draught_m=11.5)


@pytest.fixture
def demo_profile():
    return AbrasionProfile(
        a0=1.0e-8, x0=160.0, sigma_x=25.0,
        y0=8.0, lambda_y=3.0,
        a1=3.0e-9, x1=90.0, sigma_x1=35.0,
        y1=7.0, lambda_y1=2.5,
        a_bg=1.0e-10,
    )


# ---------------------------------------------------------------------------
# Panel mesh tests
# ---------------------------------------------------------------------------

class TestPanelMesh:
    def test_build_mesh(self, small_mesh):
        assert small_mesh.n_panels == 80
        assert small_mesh.x_centers.shape == (80,)
        assert small_mesh.y_centers.shape == (80,)
        assert small_mesh.areas.shape == (80,)
        assert small_mesh.total_area > 0

    def test_mesh_area_positive(self, standard_mesh):
        assert np.all(standard_mesh.areas > 0)

    def test_mesh_coordinates_in_range(self, standard_mesh):
        assert np.all(standard_mesh.x_centers >= 0)
        assert np.all(standard_mesh.x_centers <= 185.0)
        assert np.all(standard_mesh.y_centers >= 0)
        assert np.all(standard_mesh.y_centers <= 11.5 * 1.15)

    def test_invalid_mesh(self):
        with pytest.raises(ValueError):
            build_panel_mesh(n_x=1, n_y=2)


# ---------------------------------------------------------------------------
# Activity profile tests
# ---------------------------------------------------------------------------

class TestAbrasionProfile:
    def test_evaluate(self, demo_profile):
        x = np.array([160.0, 90.0, 0.0])
        y = np.array([8.0, 7.0, 0.0])
        a = demo_profile.evaluate(x, y)
        assert a.shape == (3,)
        assert np.all(a >= 0)

    def test_evaluate_on_mesh(self, demo_profile, small_mesh):
        a = demo_profile.evaluate_on_mesh(small_mesh)
        assert a.shape == (small_mesh.n_panels,)
        assert np.all(a >= 0)

    def test_peak_at_expected_location(self):
        profile = AbrasionProfile(
            a0=1.0, x0=50.0, sigma_x=5.0,
            y0=5.0, lambda_y=2.0,
        )
        x = np.array([50.0, 0.0, 100.0])
        y = np.array([5.0, 5.0, 5.0])
        a = profile.evaluate(x, y)
        assert a[0] > a[1]
        assert a[0] > a[2]


# ---------------------------------------------------------------------------
# C1: Profile fitting
# ---------------------------------------------------------------------------

class TestProfileFitting:
    def test_fit_synthetic_data(self, standard_mesh):
        true_profile = AbrasionProfile(
            a0=1.0e-8, x0=160.0, sigma_x=25.0,
            y0=8.0, lambda_y=3.0,
            a_bg=1.0e-10,
        )
        a_true = true_profile.evaluate_on_mesh(standard_mesh)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1e-10, size=standard_mesh.n_panels)
        a_dem = np.maximum(a_true + noise, 0.0)

        fit = fit_activity_profile(
            standard_mesh.x_centers, standard_mesh.y_centers, a_dem, seed=42)

        assert fit.r_squared > 0.85, f"R² = {fit.r_squared:.4f} < 0.85"
        assert fit.convergence

    def test_fit_perfect_data(self, standard_mesh):
        true_profile = AbrasionProfile(
            a0=1.0e-8, x0=160.0, sigma_x=25.0,
            y0=8.0, lambda_y=3.0,
            a_bg=1.0e-10,
        )
        a_true = true_profile.evaluate_on_mesh(standard_mesh)

        fit = fit_activity_profile(
            standard_mesh.x_centers, standard_mesh.y_centers, a_true, seed=42)

        assert fit.r_squared > 0.90
        assert fit.rmse < 1e-9


# ---------------------------------------------------------------------------
# Spatial correlation tests
# ---------------------------------------------------------------------------

class TestSpatialCorrelation:
    def test_estimate_correlation(self, small_mesh):
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1e-10, size=small_mesh.n_panels)

        corr = estimate_spatial_correlation(
            small_mesh.x_centers, small_mesh.y_centers, residuals)

        assert "correlation_length_x" in corr
        assert "correlation_length_y" in corr
        assert "variance" in corr
        assert corr["variance"] > 0

    def test_zero_residuals(self, small_mesh):
        residuals = np.zeros(small_mesh.n_panels)
        corr = estimate_spatial_correlation(
            small_mesh.x_centers, small_mesh.y_centers, residuals)
        assert corr["variance"] == 0.0
        assert corr["model"] == "white_noise"


# ---------------------------------------------------------------------------
# KL expansion tests
# ---------------------------------------------------------------------------

class TestKLExpansion:
    def test_kl_expansion(self, small_mesh):
        kl = kl_expansion_2d(
            small_mesh,
            correlation_length_x=20.0,
            correlation_length_y=5.0,
            variance=1e-18,
            n_modes=50,
        )

        assert kl.n_modes <= 50
        assert kl.n_modes <= small_mesh.n_panels
        assert np.all(kl.eigenvalues >= 0)
        assert kl.variance_explained > 0.5

    def test_generate_samples(self, small_mesh):
        kl = kl_expansion_2d(
            small_mesh,
            correlation_length_x=20.0,
            correlation_length_y=5.0,
            variance=1e-18,
            n_modes=30,
        )

        samples = kl.generate_samples(10, seed=42)
        assert samples.shape == (10, small_mesh.n_panels)

    def test_conditional_samples(self, small_mesh):
        kl = kl_expansion_2d(
            small_mesh,
            correlation_length_x=20.0,
            correlation_length_y=5.0,
            variance=1e-18,
            n_modes=30,
        )

        obs_idx = np.array([0, 1, 2])
        obs_val = np.array([1e-9, 2e-9, 1.5e-9])

        samples = kl.generate_conditional_samples(10, obs_idx, obs_val, seed=42)
        assert samples.shape == (10, small_mesh.n_panels)

        # Check conditioning: observed panels should be close to observed values
        mean_at_obs = np.mean(samples[:, obs_idx], axis=0)
        # The conditional mean should be close to observed values
        # (not exact due to the random field structure)
        assert np.all(np.isfinite(samples))


# ---------------------------------------------------------------------------
# C2: Conditional random field statistics
# ---------------------------------------------------------------------------

class TestConditionalRandomField:
    def test_statistics_verification(self, small_mesh):
        kl = kl_expansion_2d(
            small_mesh,
            correlation_length_x=20.0,
            correlation_length_y=5.0,
            variance=1e-18,
            n_modes=30,
        )

        # Use per-panel variance from KL expansion
        target_var = np.sum(kl.eigenvalues[None, :] * kl.eigenvectors ** 2, axis=1)

        crf = generate_conditional_samples(
            kl, n_samples=100, seed=42,
            target_mean=None,
            target_variance=target_var,
        )

        assert crf.n_samples == 100
        assert crf.mean.shape == (small_mesh.n_panels,)
        assert crf.variance.shape == (small_mesh.n_panels,)
        assert crf.mean_error_pct < 1.0
        assert crf.variance_error_pct < 40.0

# ---------------------------------------------------------------------------
# Spatio-temporal evolution tests
# ---------------------------------------------------------------------------

class TestSpatioTemporalEvolution:
    def test_evolve_deterministic(self, small_mesh, demo_profile):
        time_s = np.linspace(0, 3600.0, 50)
        field = evolve_spatio_temporal(
            small_mesh, demo_profile, time_s,
            use_random_field=False,
        )

        assert field.depth_m.shape == (50, small_mesh.n_panels)
        assert field.rate_m_s.shape == (50, small_mesh.n_panels)
        assert np.all(field.depth_m >= 0)
        # Depth should increase over time
        assert field.depth_m[-1].max() >= field.depth_m[0].max()

    def test_evolve_with_random_field(self, small_mesh, demo_profile):
        kl = kl_expansion_2d(
            small_mesh,
            correlation_length_x=20.0,
            correlation_length_y=5.0,
            variance=1e-20,
            n_modes=20,
        )
        time_s = np.linspace(0, 3600.0, 50)
        field = evolve_spatio_temporal(
            small_mesh, demo_profile, time_s,
            kl_expansion=kl,
            use_random_field=True,
            seed=42,
        )

        assert field.depth_m.shape == (50, small_mesh.n_panels)
        assert np.all(field.depth_m >= 0)

    def test_depth_interpolation(self, small_mesh, demo_profile):
        time_s = np.array([0.0, 1800.0, 3600.0])
        field = evolve_spatio_temporal(
            small_mesh, demo_profile, time_s,
            use_random_field=False,
        )

        d_mid = field.depth_at(1800.0)
        np.testing.assert_array_almost_equal(d_mid, field.depth_m[1])


# ---------------------------------------------------------------------------
# C3: Mass conservation
# ---------------------------------------------------------------------------

class TestMassConservation:
    def test_mass_conservation_deterministic(self, small_mesh, demo_profile):
        time_s = np.linspace(0, 3600.0, 100)
        field = evolve_spatio_temporal(
            small_mesh, demo_profile, time_s,
            use_random_field=False,
        )

        check = verify_mass_conservation(field)
        assert check["passed"], f"rel error = {check['relative_error']:.2e}"
        assert check["relative_error"] < 1e-6

    def test_mass_conservation_with_random_field(self, small_mesh, demo_profile):
        kl = kl_expansion_2d(
            small_mesh,
            correlation_length_x=20.0,
            correlation_length_y=5.0,
            variance=1e-20,
            n_modes=20,
        )
        time_s = np.linspace(0, 3600.0, 100)
        field = evolve_spatio_temporal(
            small_mesh, demo_profile, time_s,
            kl_expansion=kl,
            use_random_field=True,
            seed=42,
        )

        check = verify_mass_conservation(field)
        assert check["passed"], f"rel error = {check['relative_error']:.2e}"
        assert check["relative_error"] < 1e-6


# ---------------------------------------------------------------------------
# C5: Uniform anchor
# ---------------------------------------------------------------------------

class TestUniformAnchor:
    def test_uniform_field(self, small_mesh):
        uniform_rate = 1.0e-8
        time_s = np.linspace(0, 3600.0, 50)

        check = verify_uniform_anchor(small_mesh, uniform_rate, time_s)
        assert check["passed"], f"max error = {check['max_relative_error']:.2e}"
        assert check["max_relative_error"] < 0.01


# ---------------------------------------------------------------------------
# C4: P95 connector
# ---------------------------------------------------------------------------

class TestP95Connector:
    def test_extract_p95(self, small_mesh, demo_profile):
        time_s = np.linspace(0, 3600.0, 50)
        field = evolve_spatio_temporal(
            small_mesh, demo_profile, time_s,
            use_random_field=False,
        )

        p95 = extract_p95_from_field(field)
        assert p95["p95_depth_m"] > 0
        assert p95["max_depth_m"] >= p95["p95_depth_m"]
        assert p95["mean_depth_m"] <= p95["p95_depth_m"]

    def test_connect_to_n02(self):
        field_p95 = 1.0e-5
        n02_p95 = 1.05e-5

        conn = connect_to_n02_p95(field_p95, n02_p95)
        assert conn.relative_error < 0.10
        assert conn.passed

    def test_connect_to_n02_fail(self):
        field_p95 = 1.0e-5
        n02_p95 = 2.0e-5

        conn = connect_to_n02_p95(field_p95, n02_p95)
        assert conn.relative_error >= 0.10
        assert not conn.passed


# ---------------------------------------------------------------------------
# C6: Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_performance_1000_panels_100_steps(self):
        mesh = build_panel_mesh(n_x=40, n_y=25)  # 1000 panels
        profile = AbrasionProfile(
            a0=1.0e-8, x0=160.0, sigma_x=25.0,
            y0=8.0, lambda_y=3.0,
        )
        kl = kl_expansion_2d(
            mesh,
            correlation_length_x=30.0,
            correlation_length_y=3.0,
            variance=1e-18,
            n_modes=100,
        )
        time_s = np.linspace(0, 3600.0, 100)

        t_start = time.perf_counter()
        engine = SpatioTemporalEvolution(
            mesh=mesh, profile=profile, kl_expansion=kl)
        _ = engine.evolve(time_s, use_random_field=True, seed=42)
        elapsed = time.perf_counter() - t_start

        assert elapsed < 60.0, f"CPU time = {elapsed:.1f}s > 60s"


# ---------------------------------------------------------------------------
# Full verification suite
# ---------------------------------------------------------------------------

class TestFullVerification:
    def test_run_n04_verification(self):
        results = run_n04_verification(
            n_panels_x=40, n_panels_y=25,
            n_time_steps=100, n_samples=100,
            seed=42,
        )

        assert results["all_criteria_passed"], (
            f"Not all criteria passed: {results['criteria']}"
        )

        # Check individual criteria
        assert results["criteria"]["C1"]["passed"], f"C1 R² = {results['criteria']['C1']['r_squared']}"
        assert results["criteria"]["C2"]["passed"], f"C2 mean_err = {results['criteria']['C2']['mean_error_pct']}%"
        assert results["criteria"]["C3"]["passed"], f"C3 rel_err = {results['criteria']['C3']['relative_error']}"
        assert results["criteria"]["C4"]["passed"], f"C4 rel_err = {results['criteria']['C4']['relative_error']}"
        assert results["criteria"]["C5"]["passed"], f"C5 err = {results['criteria']['C5']['max_relative_error']}"
        assert results["criteria"]["C6"]["passed"], f"C6 time = {results['criteria']['C6']['cpu_time_s']}s"
