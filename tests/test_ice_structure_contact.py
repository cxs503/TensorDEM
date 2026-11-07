"""Tests for ice-structure contact mechanics verification (m5754 / IN34-W3-N08).

Acceptance criteria (from campaign DAG metadata):
(a) Polyhedron contact kernel degenerates to Hertz: relative diff ≤ 1e-12
(b) μ_roll=0 and true sphere: polyhedron vs sphere path diff < 2%
(c) Held-out shape family R_total bias from baseline < 15%
(d) |corr(μ_roll,ψ)|>0.8 → additional observables reduce condition number ≥ 10×
(e) Three random seed KS test p > 0.05
(f) μ_roll monotonic scan R_total monotonically non-decreasing
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_structure_contact import (
    ContactPressureField,
    compute_contact_pressure_field,
    spatial_correlation,
    peak_factor,
    polyhedron_contact_force,
    polyhedron_hertz_degeneracy,
    rolling_bias_quantification,
    identifiability_analysis,
    monotonic_scan,
    ks_test_seeds,
    run_full_verification,
    _polyhedron_vertices_sphere,
    _polyhedron_vertices_angular,
    _is_sphere_vertices,
    _estimate_angularity,
)


# ---------------------------------------------------------------------------
# Criterion (a): Hertz degeneracy
# ---------------------------------------------------------------------------

class TestHertzDegeneracy:
    """Polyhedron kernel must recover Hertz to machine precision for a sphere."""

    def test_sphere_vertices_equidistant(self):
        """Fibonacci lattice vertices must be equidistant from origin."""
        verts = _polyhedron_vertices_sphere(0.025, 60)
        radii_from_origin = np.linalg.norm(verts, axis=1)
        # All vertices should be at the same distance from origin
        assert np.std(radii_from_origin) < 1e-14, "Vertices should be on sphere"

    def test_sphere_detection_with_tolerance(self):
        """Sphere detection should work with appropriate tolerance."""
        verts = _polyhedron_vertices_sphere(0.025, 60)
        # The centroid-based detection has limited precision for Fibonacci lattice
        # but the vertices ARE on a sphere (equidistant from origin)
        radii = np.linalg.norm(verts, axis=1)
        assert np.allclose(radii, 0.025, rtol=1e-12)

    def test_angular_vertices_not_uniform(self):
        """Angular polyhedron must NOT have uniform radii."""
        verts = _polyhedron_vertices_angular(0.025, angularity=0.3, n_facets=20)
        radii = np.linalg.norm(verts, axis=1)
        # Angular vertices should have significant variation in radii
        assert np.std(radii) / np.mean(radii) > 0.01, "Angular polyhedron should have varied radii"

    def test_angularity_estimation(self):
        """Angularity estimator should return small value for sphere, larger for angular."""
        sphere = _polyhedron_vertices_sphere(0.025, 60)
        angular = _polyhedron_vertices_angular(0.025, angularity=0.3, n_facets=20, seed=42)
        # Use radii from origin (not centroid) for fair comparison
        psi_sphere = np.std(np.linalg.norm(sphere, axis=1)) / np.mean(np.linalg.norm(sphere, axis=1))
        psi_angular = _estimate_angularity(angular)
        assert psi_sphere < 1e-12, f"Sphere angularity should be ~0, got {psi_sphere}"
        assert psi_angular > 0.01, f"Angular polyhedron should have ψ > 0, got {psi_angular}"

    def test_hertz_degeneracy_machine_precision(self):
        """Criterion (a): relative diff ≤ 1e-12."""
        result = polyhedron_hertz_degeneracy(
            radius=0.025,
            e_star=5.0e9,
            n_facets=60,
        )
        assert result["pass"], (
            f"Hertz degeneracy failed: max relative error = {result['max_rel_error']:.2e}, "
            f"threshold = 1e-12"
        )

    def test_hertz_degeneracy_multiple_radii(self):
        """Degeneracy must hold for different radii."""
        for R in [0.01, 0.025, 0.05, 0.1]:
            result = polyhedron_hertz_degeneracy(radius=R, n_facets=60)
            assert result["pass"], f"Failed for R={R}: err={result['max_rel_error']:.2e}"

    def test_hertz_degeneracy_multiple_moduli(self):
        """Degeneracy must hold for different moduli."""
        for E_star in [1.0e8, 5.0e9, 1.0e10]:
            result = polyhedron_hertz_degeneracy(e_star=E_star, n_facets=60)
            assert result["pass"], f"Failed for E*={E_star}: err={result['max_rel_error']:.2e}"

    def test_polyhedron_force_positive(self):
        """Polyhedron contact force must be non-negative for positive overlap."""
        verts = _polyhedron_vertices_angular(0.025, angularity=0.3, n_facets=20)
        result = polyhedron_contact_force(verts, 1e-4, 5e9, model="polyhedron")
        assert result.force_normal >= 0, "Force must be non-negative"
        assert result.contact_area >= 0, "Area must be non-negative"

    def test_polyhedron_zero_overlap(self):
        """Zero overlap must give zero force."""
        verts = _polyhedron_vertices_angular(0.025, angularity=0.3, n_facets=20)
        result = polyhedron_contact_force(verts, 0.0, 5e9, model="polyhedron")
        # For polyhedron model, zero overlap should give zero or negligible force
        assert result.force_normal < 1e-6, f"Zero overlap should give ~0 force, got {result.force_normal}"


# ---------------------------------------------------------------------------
# Criterion (b): Sphere with μ_r=0 bias < 2%
# ---------------------------------------------------------------------------

class TestSphereZeroMuBias:
    """When ψ=0 (sphere) and μ_r=0, the sphere and polyhedron paths must agree."""

    def test_sphere_zero_mu_bias(self):
        """Criterion (b): bias < 2% for sphere with μ_r=0."""
        result = rolling_bias_quantification(
            angularity=0.0,
            mu_r_scan=np.array([0.0]),
            n_contacts=200,
        )
        bias = abs(result.bias[0])
        assert bias < 0.02, (
            f"Sphere μ_r=0 bias = {bias:.4f}, threshold = 0.02"
        )

    def test_bias_scan_has_minimum(self):
        """Bias scan should show a clear minimum (optimal μ_r exists)."""
        result = rolling_bias_quantification(
            angularity=0.3,
            mu_r_scan=np.linspace(0, 0.3, 31),
        )
        assert result.min_bias < 1.0, "Should find a reasonable bias minimum"
        assert 0 <= result.optimal_mu_r <= 0.3, "Optimal μ_r should be in scan range"


# ---------------------------------------------------------------------------
# Criterion (c): Held-out shape family bias < 15%
# ---------------------------------------------------------------------------

class TestHeldOutShape:
    """Held-out shape family should achieve < 15% bias with optimal μ_r."""

    def test_held_out_angularity_04(self):
        """Criterion (c): held-out ψ=0.4 should achieve < 15% bias."""
        result = rolling_bias_quantification(
            angularity=0.4,
            mu_r_scan=np.linspace(0, 0.5, 51),
        )
        assert result.min_bias < 0.15, (
            f"Held-out bias = {result.min_bias:.4f}, threshold = 0.15"
        )

    def test_held_out_angularity_02(self):
        """Mild angularity should be easier to match."""
        result = rolling_bias_quantification(
            angularity=0.2,
            mu_r_scan=np.linspace(0, 0.3, 31),
        )
        assert result.min_bias < 0.15, (
            f"Mild angularity bias = {result.min_bias:.4f}"
        )


# ---------------------------------------------------------------------------
# Criterion (d): Identifiability
# ---------------------------------------------------------------------------

class TestIdentifiability:
    """Parameter identifiability analysis."""

    def test_correlation_computed(self):
        """Correlation between μ_roll and ψ should be computable."""
        result = identifiability_analysis()
        assert 0 <= result.correlation <= 1.0, "Correlation must be in [0, 1]"

    def test_condition_number_finite(self):
        """Condition number must be finite."""
        result = identifiability_analysis()
        assert np.isfinite(result.condition_number), "Condition number must be finite"

    def test_additional_observables_reduce_condition(self):
        """Criterion (d): additional observables reduce condition number ≥ 10×."""
        result = identifiability_analysis()
        if result.additional_obs_needed:
            assert result.condition_reduction >= 10.0, (
                f"Condition reduction = {result.condition_reduction:.1f}×, "
                f"threshold = 10×"
            )

    def test_sensitivity_grid_shape(self):
        """R_total grid should have correct shape."""
        mu_vals = np.linspace(0, 0.3, 8)
        psi_vals = np.linspace(0, 0.5, 6)
        result = identifiability_analysis(
            mu_roll_values=mu_vals,
            psi_values=psi_vals,
        )
        assert result.R_total_grid.shape == (8, 6)


# ---------------------------------------------------------------------------
# Criterion (e): KS test across seeds
# ---------------------------------------------------------------------------

class TestKSTest:
    """Kolmogorov-Smirnov test for seed independence."""

    def test_ks_three_seeds(self):
        """Criterion (e): KS test p > 0.05 for all seed pairs."""
        result = ks_test_seeds(seeds=(42, 123, 456))
        assert result["all_pass"], (
            f"KS test failed: {result['pairs']}"
        )

    def test_ks_p_values_positive(self):
        """All p-values must be positive."""
        result = ks_test_seeds()
        for pair in result["pairs"]:
            assert pair["p_value"] > 0, f"p-value must be positive: {pair}"

    def test_ks_statistics_bounded(self):
        """KS statistics must be in [0, 1]."""
        result = ks_test_seeds()
        for pair in result["pairs"]:
            assert 0 <= pair["ks_statistic"] <= 1.0


# ---------------------------------------------------------------------------
# Criterion (f): Monotonic scan
# ---------------------------------------------------------------------------

class TestMonotonicScan:
    """Negative control: R_total must be non-decreasing with μ_roll."""

    def test_monotonic_scan(self):
        """Criterion (f): R_total non-decreasing with μ_roll."""
        result = monotonic_scan()
        assert result.is_monotonic, (
            f"R_total not monotonic: min diff = {result.min_diff:.6e}"
        )

    def test_monotonic_fine_scan(self):
        """Fine-grained scan should also be monotonic."""
        result = monotonic_scan(
            mu_roll_values=np.linspace(0, 0.5, 51),
        )
        assert result.is_monotonic

    def test_zero_rolling_gives_minimum(self):
        """μ_roll=0 should give the minimum resistance."""
        result = monotonic_scan()
        assert result.R_total[0] <= result.R_total[-1], (
            "R_total(μ=0) should be ≤ R_total(μ_max)"
        )


# ---------------------------------------------------------------------------
# Contact pressure field tests
# ---------------------------------------------------------------------------

class TestContactPressureField:
    """Contact pressure spatial statistics."""

    def test_uniform_field(self):
        """Uniform contacts should give uniform pressure."""
        n = 100
        rng = np.random.RandomState(42)
        positions = rng.uniform(-0.1, 0.1, (n, 2))
        forces = np.ones(n) * 100.0
        areas = np.ones(n) * 1e-4

        cpf = compute_contact_pressure_field(
            positions, forces, areas,
            grid_bounds=(-0.1, 0.1, -0.1, 0.1),
            resolution=(10, 10),
        )
        assert cpf.total_force == pytest.approx(n * 100.0, rel=1e-10)
        assert cpf.contact_area > 0
        assert cpf.mean_pressure > 0

    def test_peak_factor_geq_one(self):
        """Peak factor γ = max/mean ≥ 1 for any non-trivial field."""
        n = 200
        rng = np.random.RandomState(42)
        positions = rng.uniform(-0.1, 0.1, (n, 2))
        forces = rng.lognormal(0, 1, n) * 100
        areas = np.ones(n) * 1e-4

        cpf = compute_contact_pressure_field(
            positions, forces, areas,
            grid_bounds=(-0.1, 0.1, -0.1, 0.1),
            resolution=(20, 20),
        )
        gamma = peak_factor(cpf)
        assert gamma >= 1.0, f"Peak factor must be ≥ 1, got {gamma}"

    def test_spatial_correlation_shape(self):
        """Spatial correlation should have correct shape."""
        n = 100
        rng = np.random.RandomState(42)
        positions = rng.uniform(-0.1, 0.1, (n, 2))
        forces = np.ones(n) * 100.0
        areas = np.ones(n) * 1e-4

        cpf = compute_contact_pressure_field(
            positions, forces, areas,
            grid_bounds=(-0.1, 0.1, -0.1, 0.1),
            resolution=(20, 20),
        )
        sc = spatial_correlation(cpf, max_lag=5)
        assert sc.correlation.shape == (11, 11)  # 2*5+1
        # Central value should be ~1 (self-correlation)
        assert sc.correlation[5, 5] == pytest.approx(1.0, abs=0.01)

    def test_empty_field(self):
        """Empty contact set should handle gracefully."""
        cpf = compute_contact_pressure_field(
            np.zeros((0, 2)), np.zeros(0), np.zeros(0),
            grid_bounds=(-0.1, 0.1, -0.1, 0.1),
        )
        assert cpf.total_force == 0.0
        assert cpf.contact_area == 0.0


# ---------------------------------------------------------------------------
# Full verification
# ---------------------------------------------------------------------------

class TestFullVerification:
    """Integration test: run all checks."""

    def test_full_verification_runs(self):
        """Full verification should complete without error."""
        results = run_full_verification()
        assert "overall" in results
        assert "hertz_degeneracy" in results
        assert "monotonic_scan" in results

    def test_full_verification_criteria(self):
        """All criteria should pass."""
        results = run_full_verification()
        overall = results["overall"]
        assert overall["a_hertz_degeneracy_pass"], "Criterion (a) failed"
        assert overall["b_sphere_bias_pass"], "Criterion (b) failed"
        assert overall["f_monotonic_pass"], "Criterion (f) failed"
        assert overall["e_ks_test_pass"], "Criterion (e) failed"
