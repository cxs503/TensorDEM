"""Tests for single-particle crushing verification module.

This test suite covers:
  - Weibull strength–size effect (C1: R²≥0.95, m偏差≤20%)
  - Mass conservation (C2: 1000次破碎误差≤0.1%)
  - Momentum conservation (C3: 范数误差≤0.1%)
  - Gaudin-Schuhmann破碎谱 (C4: α偏差≤15%)
  - Negative control (C5: 零载荷→零破碎)
  - Regression (C6: 现有套件全绿)

References
----------
- Weibull, W. (1951). A statistical distribution function of wide applicability.
- Gaudin, A.M. & Schuhmann, R. (1936). Empirical evaluation of the
  probability-size function in crushing.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from tensordem.single_particle_crushing import (
    WeibullStrengthModel,
    weibull_failure_probability,
    weibull_strength_size,
    fit_weibull_modulus,
    generate_weibull_strengths,
    FragmentReplacement,
    ConservationLedger,
    replace_parent_with_fragments,
    generate_fragment_velocities,
    GaudinSchuhmannDistribution,
    gaudin_cdf,
    gaudin_pdf,
    fit_gaudin_exponent,
    generate_gaudin_sizes,
)
from tensordem.single_particle_crushing.weibull_strength import weibull_r_squared
from tensordem.single_particle_crushing.gaudin_distribution import (
    gaudin_r_squared,
    gaudin_ks_statistic,
)
from tensordem.single_particle_crushing.fragment_replacement import (
    generate_fragment_masses,
    masses_to_radii,
)


# ============================================================================
# Weibull Strength–Size Effect Tests (C1)
# ============================================================================

class TestWeibullStrengthSizeEffect:
    """Tests for the Weibull strength–size effect model."""

    def test_weibull_strength_decreases_with_volume(self):
        """Larger particles should have lower characteristic strength."""
        model = WeibullStrengthModel(sigma_0=100e6, m=7.0, V_0=1e-9)
        sigma_small = weibull_strength_size(model, V=1e-9)  # reference
        sigma_large = weibull_strength_size(model, V=1e-6)  # 1000x larger
        assert sigma_large < sigma_small
        # Size effect: σ_f ∝ V^{-1/m}
        ratio = sigma_small / sigma_large
        expected_ratio = (1e-6 / 1e-9) ** (1.0 / 7.0)
        assert abs(ratio - expected_ratio) / expected_ratio < 1e-10

    def test_weibull_failure_probability_monotonic(self):
        """Failure probability should increase with stress."""
        model = WeibullStrengthModel(sigma_0=100e6, m=7.0, V_0=1e-9)
        V = 1e-9
        pf_low = weibull_failure_probability(model, sigma=50e6, V=V)
        pf_high = weibull_failure_probability(model, sigma=150e6, V=V)
        assert pf_high > pf_low
        assert 0 <= pf_low <= 1
        assert 0 <= pf_high <= 1

    def test_weibull_failure_probability_zero_stress(self):
        """Zero stress should give zero failure probability (C5 negative control)."""
        model = WeibullStrengthModel(sigma_0=100e6, m=7.0, V_0=1e-9)
        pf = weibull_failure_probability(model, sigma=0.0, V=1e-9)
        assert pf == 0.0

    def test_weibull_fit_synthetic_data(self):
        """MLE fit should recover the input Weibull modulus within 20% (C1)."""
        rng = np.random.default_rng(12345)
        true_m = 7.0
        true_sigma_0 = 100e6
        model = WeibullStrengthModel(sigma_0=true_sigma_0, m=true_m, V_0=1e-9)

        # Generate synthetic failure data
        n_samples = 100
        strengths = generate_weibull_strengths(model, n_samples, V=1e-9, rng=rng)

        # Fit
        m_fit, sigma_0_fit = fit_weibull_modulus(strengths, V_0=1e-9)

        # Check m within 20%
        m_error = abs(m_fit - true_m) / true_m
        assert m_error <= 0.20, f"Weibull modulus error {m_error:.1%} exceeds 20%"

        # Check R² ≥ 0.95
        r2 = weibull_r_squared(strengths, V_0=1e-9)
        assert r2 >= 0.95, f"R² = {r2:.4f} < 0.95"

    def test_weibull_fit_size_effect(self):
        """Weibull fit should capture size effect across multiple volumes."""
        rng = np.random.default_rng(54321)
        true_m = 8.0
        true_sigma_0 = 150e6
        V_0 = 1e-9
        model = WeibullStrengthModel(sigma_0=true_sigma_0, m=true_m, V_0=V_0)

        # Generate data at different volumes
        volumes = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7]
        n_per_volume = 30
        all_strengths = []
        all_volumes = []
        for V in volumes:
            strengths = generate_weibull_strengths(model, n_per_volume, V=V, rng=rng)
            all_strengths.extend(strengths.tolist())
            all_volumes.extend([V] * n_per_volume)

        # Fit with varying volumes
        m_fit, sigma_0_fit = fit_weibull_modulus(all_strengths, all_volumes, V_0)

        # Check m within 20%
        m_error = abs(m_fit - true_m) / true_m
        assert m_error <= 0.20, f"Weibull modulus error {m_error:.1%} exceeds 20%"

    def test_weibull_quartz_sand_reference(self):
        """Fit should be consistent with quartz sand reference (m=5-10)."""
        rng = np.random.default_rng(99999)
        # Quartz sand: m ≈ 7, σ_0 ≈ 200 MPa at 1mm reference
        model = WeibullStrengthModel(sigma_0=200e6, m=7.0, V_0=1e-9)
        strengths = generate_weibull_strengths(model, 200, V=1e-9, rng=rng)
        m_fit, sigma_0_fit = fit_weibull_modulus(strengths, V_0=1e-9)

        # m should be in [5, 10] range for quartz sand
        assert 5.0 <= m_fit <= 10.0, f"m={m_fit:.2f} outside quartz sand range [5,10]"


# ============================================================================
# Mass Conservation Tests (C2)
# ============================================================================

class TestMassConservation:
    """Tests for mass conservation in fragment replacement."""

    def test_single_event_mass_conservation(self):
        """Single fracture event should conserve mass exactly."""
        parent_mass = 1.0e-3  # 1 gram
        frag_masses = generate_fragment_masses(parent_mass, n_fragments=10)
        mass_sum = frag_masses.sum()
        error = abs(mass_sum - parent_mass)
        assert error < 1e-15, f"Mass error {error:.2e} kg exceeds machine precision"

    def test_1000_events_mass_conservation(self):
        """1000 fracture events should have cumulative mass error ≤ 0.1% (C2)."""
        rng = np.random.default_rng(42)
        ledger = ConservationLedger()

        for i in range(1000):
            parent_mass = rng.uniform(1e-4, 1e-2)
            n_frags = rng.integers(2, 20)
            replacement = replace_parent_with_fragments(
                parent_mass=parent_mass,
                parent_velocity=np.array([0.1, 0.0, 0.0]),
                parent_radius=0.01,
                density=2650.0,
                n_fragments=n_frags,
                rng=rng,
            )
            ledger.record(replacement)

        # C2: relative mass error ≤ 0.1%
        mean_error = ledger.mean_mass_error_relative
        max_error = ledger.max_mass_error_relative
        assert mean_error <= 0.001, f"Mean mass error {mean_error:.2%} exceeds 0.1%"
        assert max_error <= 0.001, f"Max mass error {max_error:.2%} exceeds 0.1%"

    def test_mass_conservation_gaudin_distribution(self):
        """Mass conservation with Gaudin-distributed fragments."""
        rng = np.random.default_rng(777)
        parent_mass = 5e-3
        frag_masses = generate_fragment_masses(
            parent_mass, n_fragments=15,
            size_distribution="gaudin", alpha=2.0, rng=rng
        )
        error = abs(frag_masses.sum() - parent_mass) / parent_mass
        assert error <= 0.001, f"Relative mass error {error:.2%} exceeds 0.1%"

    def test_mass_conservation_power_law_distribution(self):
        """Mass conservation with power-law distributed fragments."""
        rng = np.random.default_rng(888)
        parent_mass = 2e-3
        frag_masses = generate_fragment_masses(
            parent_mass, n_fragments=12,
            size_distribution="power_law", alpha=3.0, rng=rng
        )
        error = abs(frag_masses.sum() - parent_mass) / parent_mass
        assert error <= 0.001, f"Relative mass error {error:.2%} exceeds 0.1%"


# ============================================================================
# Momentum Conservation Tests (C3)
# ============================================================================

class TestMomentumConservation:
    """Tests for momentum conservation in fragment replacement."""

    def test_momentum_conservation_no_impulse(self):
        """Without fracture impulse, momentum should be exactly conserved."""
        parent_mass = 1e-3
        parent_vel = np.array([0.5, -0.2, 0.1])
        rng = np.random.default_rng(42)

        replacement = replace_parent_with_fragments(
            parent_mass=parent_mass,
            parent_velocity=parent_vel,
            parent_radius=0.01,
            density=2650.0,
            n_fragments=8,
            velocity_perturbation=0.05,
            rng=rng,
        )

        # C3: momentum error norm ≤ 0.1% of parent momentum
        p_parent = parent_mass * np.linalg.norm(parent_vel)
        p_error = replacement.momentum_error_norm
        relative_error = p_error / p_parent if p_parent > 0 else p_error
        assert relative_error <= 0.001, (
            f"Momentum error {relative_error:.2%} exceeds 0.1%"
        )

    def test_momentum_conservation_with_impulse(self):
        """With fracture impulse, momentum should include the impulse."""
        parent_mass = 1e-3
        parent_vel = np.array([0.3, 0.0, 0.0])
        fracture_impulse = np.array([0.0, 1e-4, 0.0])
        rng = np.random.default_rng(123)

        replacement = replace_parent_with_fragments(
            parent_mass=parent_mass,
            parent_velocity=parent_vel,
            parent_radius=0.01,
            density=2650.0,
            n_fragments=6,
            fracture_impulse=fracture_impulse,
            velocity_perturbation=0.02,
            rng=rng,
        )

        # Target momentum = m*v + Δp
        target_momentum = parent_mass * parent_vel + fracture_impulse
        final_momentum = np.sum(
            replacement.fragment_masses[:, np.newaxis] * replacement.fragment_velocities,
            axis=0
        )
        error = np.linalg.norm(final_momentum - target_momentum)
        target_norm = np.linalg.norm(target_momentum)
        relative_error = error / target_norm if target_norm > 0 else error
        assert relative_error <= 0.001, (
            f"Momentum error with impulse {relative_error:.2%} exceeds 0.1%"
        )

    def test_1000_events_momentum_conservation(self):
        """1000 fracture events should have cumulative momentum error ≤ 0.1% (C3)."""
        rng = np.random.default_rng(42)
        ledger = ConservationLedger()

        for i in range(1000):
            parent_mass = rng.uniform(1e-4, 1e-2)
            parent_vel = rng.standard_normal(3) * 0.3
            n_frags = rng.integers(2, 15)
            replacement = replace_parent_with_fragments(
                parent_mass=parent_mass,
                parent_velocity=parent_vel,
                parent_radius=0.01,
                density=2650.0,
                n_fragments=n_frags,
                velocity_perturbation=0.05,
                rng=rng,
            )
            ledger.record(replacement)

        # C3: relative momentum error ≤ 0.1%
        mean_error = ledger.mean_momentum_error_relative
        max_error = ledger.max_momentum_error_relative
        assert mean_error <= 0.001, f"Mean momentum error {mean_error:.2%} exceeds 0.1%"
        assert max_error <= 0.001, f"Max momentum error {max_error:.2%} exceeds 0.1%"


# ============================================================================
# Gaudin-Schuhmann Distribution Tests (C4)
# ============================================================================

class TestGaudinDistribution:
    """Tests for the Gaudin-Schuhmann破碎粒度分布."""

    def test_gaudin_cdf_boundary_values(self):
        """CDF should be 0 at d=0 and 1 at d=d_max."""
        dist = GaudinSchuhmannDistribution(alpha=2.0, d_max=0.01)
        assert gaudin_cdf(0.0, dist) == 0.0
        assert abs(gaudin_cdf(0.01, dist) - 1.0) < 1e-10
        assert gaudin_cdf(-0.001, dist) == 0.0
        assert gaudin_cdf(0.02, dist) == 1.0

    def test_gaudin_cdf_monotonic(self):
        """CDF should be monotonically increasing."""
        dist = GaudinSchuhmannDistribution(alpha=2.5, d_max=0.01)
        d_values = np.linspace(0, 0.01, 100)
        cdf_values = gaudin_cdf(d_values, dist)
        assert np.all(np.diff(cdf_values) >= 0)

    def test_gaudin_pdf_normalization(self):
        """PDF should integrate to 1 over [0, d_max]."""
        dist = GaudinSchuhmannDistribution(alpha=2.0, d_max=0.01)
        d_values = np.linspace(0, 0.01, 10000)
        pdf_values = gaudin_pdf(d_values, dist)
        integral = np.trapezoid(pdf_values, d_values)
        assert abs(integral - 1.0) < 0.01, f"PDF integral = {integral:.4f}, expected 1.0"

    def test_gaudin_fit_synthetic_data(self):
        """MLE fit should recover the input α within 15% (C4)."""
        rng = np.random.default_rng(42)
        true_alpha = 2.5
        d_max = 0.01
        dist = GaudinSchuhmannDistribution(alpha=true_alpha, d_max=d_max)

        # Generate synthetic data
        n_samples = 500
        sizes = generate_gaudin_sizes(dist, n_samples, rng=rng)

        # Fit
        alpha_fit, alpha_stderr = fit_gaudin_exponent(sizes, d_max=d_max)

        # C4: α偏差 ≤ 15%
        alpha_error = abs(alpha_fit - true_alpha) / true_alpha
        assert alpha_error <= 0.15, f"α error {alpha_error:.1%} exceeds 15%"

    def test_gaudin_fit_r_squared(self):
        """Gaudin fit should have R² ≥ 0.95 for clean synthetic data."""
        rng = np.random.default_rng(42)
        dist = GaudinSchuhmannDistribution(alpha=2.0, d_max=0.005)
        sizes = generate_gaudin_sizes(dist, 300, rng=rng)
        r2 = gaudin_r_squared(sizes, d_max=0.005)
        assert r2 >= 0.95, f"R² = {r2:.4f} < 0.95"

    def test_gaudin_limestone_reference(self):
        """Fit should be consistent with limestone reference (α ≈ 1.5-3.0)."""
        rng = np.random.default_rng(555)
        # Limestone crushing: α ≈ 2.0
        dist = GaudinSchuhmannDistribution(alpha=2.0, d_max=0.008)
        sizes = generate_gaudin_sizes(dist, 400, rng=rng)
        alpha_fit, _ = fit_gaudin_exponent(sizes, d_max=0.008)
        assert 1.5 <= alpha_fit <= 3.0, f"α={alpha_fit:.2f} outside limestone range [1.5,3.0]"


# ============================================================================
# Negative Control Tests (C5)
# ============================================================================

class TestNegativeControl:
    """Negative control: zero load → zero fragmentation."""

    def test_zero_stress_zero_failure(self):
        """Zero applied stress should give zero failure probability."""
        model = WeibullStrengthModel(sigma_0=100e6, m=7.0, V_0=1e-9)
        pf = weibull_failure_probability(model, sigma=0.0, V=1e-9)
        assert pf == 0.0

    def test_zero_stress_no_breakage(self):
        """At zero stress, no particles should break (fracture count = 0)."""
        model = WeibullStrengthModel(sigma_0=100e6, m=7.0, V_0=1e-9)
        n_particles = 1000
        # At zero stress, all particles should survive
        n_broken = 0
        for _ in range(n_particles):
            pf = weibull_failure_probability(model, sigma=0.0, V=1e-9)
            # pf = 0 means no breakage
            if pf > 0:
                n_broken += 1
        assert n_broken == 0, f"Expected 0 broken at zero stress, got {n_broken}"


# ============================================================================
# Conservation Ledger Tests
# ============================================================================

class TestConservationLedger:
    """Tests for the conservation ledger tracking."""

    def test_ledger_records_events(self):
        """Ledger should correctly count and accumulate events."""
        ledger = ConservationLedger()
        rng = np.random.default_rng(42)

        for _ in range(10):
            replacement = replace_parent_with_fragments(
                parent_mass=1e-3,
                parent_velocity=np.array([0.1, 0.0, 0.0]),
                parent_radius=0.01,
                density=2650.0,
                n_fragments=5,
                rng=rng,
            )
            ledger.record(replacement)

        assert ledger.n_events == 10
        summary = ledger.summary()
        assert summary["n_events"] == 10

    def test_ledger_summary_structure(self):
        """Ledger summary should contain all required fields."""
        ledger = ConservationLedger()
        replacement = replace_parent_with_fragments(
            parent_mass=1e-3,
            parent_velocity=np.array([0.1, 0.0, 0.0]),
            parent_radius=0.01,
            density=2650.0,
            n_fragments=5,
        )
        ledger.record(replacement)
        summary = ledger.summary()

        required_keys = [
            "n_events", "total_parent_mass_kg", "total_mass_error_kg",
            "mean_mass_error_relative", "max_mass_error_relative",
            "total_momentum_error_kg_m_s", "mean_momentum_error_relative",
            "max_momentum_error_relative",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_single_fragment_equals_parent(self):
        """Single fragment should have same mass as parent."""
        parent_mass = 1e-3
        frag_masses = generate_fragment_masses(parent_mass, n_fragments=1)
        assert len(frag_masses) == 1
        assert abs(frag_masses[0] - parent_mass) < 1e-15

    def test_many_fragments_mass_conservation(self):
        """Mass conservation with many fragments (N=100)."""
        rng = np.random.default_rng(42)
        parent_mass = 1e-2
        frag_masses = generate_fragment_masses(parent_mass, n_fragments=100, rng=rng)
        error = abs(frag_masses.sum() - parent_mass) / parent_mass
        assert error < 1e-10, f"Mass error {error:.2e} for 100 fragments"

    def test_masses_to_radii_consistency(self):
        """Radii should be consistent with masses and density."""
        masses = np.array([1e-4, 2e-4, 3e-4])
        density = 2650.0
        radii = masses_to_radii(masses, density)

        # Check: m = ρ * (4/3) * π * r³
        for m, r in zip(masses, radii):
            m_check = density * (4.0 / 3.0) * math.pi * r ** 3
            assert abs(m_check - m) / m < 1e-10

    def test_weibull_invalid_parameters(self):
        """Invalid parameters should raise ValueError."""
        with pytest.raises(ValueError):
            WeibullStrengthModel(sigma_0=-1, m=7.0, V_0=1e-9)
        with pytest.raises(ValueError):
            WeibullStrengthModel(sigma_0=100e6, m=-1, V_0=1e-9)
        with pytest.raises(ValueError):
            WeibullStrengthModel(sigma_0=100e6, m=7.0, V_0=-1e-9)

    def test_gaudin_invalid_parameters(self):
        """Invalid parameters should raise ValueError."""
        with pytest.raises(ValueError):
            GaudinSchuhmannDistribution(alpha=-1, d_max=0.01)
        with pytest.raises(ValueError):
            GaudinSchuhmannDistribution(alpha=2.0, d_max=-0.01)
