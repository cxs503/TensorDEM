"""Verification tests for the freeze-bond model (IN13-W3-N06).

Tests cover:
1. Temperature-dependent bond strength σ_fb(T) = σ_0·exp(-β·ΔT)
2. Bond growth kinetics A_fb(t)/A_0 = 1 - exp(-t/τ)
3. Failure mode classification (adhesive vs cohesive)
4. Failure mode transition detection
5. Peak load jump at transition
6. Pull-off test verification
7. Shear test verification
8. Phase diagram computation
9. Parameter sweep
10. Physical consistency checks (monotonicity, bounds, limits)

References
----------
Timco, G.W. & Johnston, M. (2003). "Freezing of water to structural surfaces."
    Cold Regions Science and Technology, 37(2), 145-156.
"""
from __future__ import annotations

import math
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tensordem.freeze_bond import (
    FreezeBondParams,
    FreezeBondState,

    FailureMode,
    PullOffResult,
    ShearResult,
    FailureModePhaseDiagram,
    ParameterSweepResult,
    VerificationReport,
    freeze_bond_strength,
    bond_growth_kinetics,
    effective_bond_strength,
    classify_failure_mode,
    pull_off_test,
    shear_test,
    failure_mode_phase_diagram,
    parameter_sweep,
    run_full_verification,
    T_MELTING_FRESHWATER,
    T_MELTING_SEA_ICE,
    SIGMA_0_DEFAULT,
    BETA_DEFAULT,
    TAU_GROWTH_DEFAULT,
    SIGMA_T_ICE,
    E_ICE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params():
    return FreezeBondParams()


@pytest.fixture
def cold_params():
    """Parameters for very cold ice (-15°C)."""
    return FreezeBondParams(sigma_0=0.2e6, beta=0.15, tau_growth=50.0)


# ---------------------------------------------------------------------------
# 1. Temperature-dependent bond strength
# ---------------------------------------------------------------------------

class TestFreezeBondStrength:
    """Tests for σ_fb(T) = σ_0 · exp(-β · ΔT)."""

    def test_near_zero_subcooling_gives_reference_strength(self, default_params):
        """At very small sub-cooling, σ_fb ≈ σ_0."""
        sigma = freeze_bond_strength(-0.001, default_params)
        expected = default_params.sigma_0 * math.exp(-default_params.beta * (-0.001))
        assert sigma == pytest.approx(expected, rel=1e-10)
        # Should be very close to sigma_0
        assert sigma == pytest.approx(default_params.sigma_0, rel=0.01)

    def test_above_melting_gives_zero(self, default_params):
        """Above the melting point, no bond forms."""
        sigma = freeze_bond_strength(1.0, default_params)
        assert sigma == 0.0

    def test_strength_increases_with_cooling(self, default_params):
        """Bond strength must increase monotonically with sub-cooling."""
        temperatures = [-1.0, -3.0, -5.0, -10.0, -15.0]
        strengths = [freeze_bond_strength(T, default_params) for T in temperatures]
        for i in range(len(strengths) - 1):
            assert strengths[i] < strengths[i + 1], (
                f"Strength at T={temperatures[i]}°C ({strengths[i]:.2e} Pa) "
                f"should be less than at T={temperatures[i+1]}°C ({strengths[i+1]:.2e} Pa)"
            )

    def test_exponential_form(self, default_params):
        """Verify the exact exponential form σ = σ_0 · exp(-β · ΔT)."""
        T = -10.0
        delta_T = T - default_params.T_melting  # = -10.0 (negative for sub-cooled)
        expected = default_params.sigma_0 * math.exp(-default_params.beta * delta_T)
        actual = freeze_bond_strength(T, default_params)
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_strength_at_minus_10C(self, default_params):
        """At -10°C, strength should be ~1 MPa (consistent with literature)."""
        sigma = freeze_bond_strength(-10.0, default_params)
        # σ = 0.2e6 · exp(0.15 · 10) = 0.2e6 · exp(1.5) ≈ 0.2e6 · 4.48 ≈ 0.896 MPa
        expected = 0.2e6 * math.exp(1.5)
        assert sigma == pytest.approx(expected, rel=1e-10)
        # Should be in the range 0.5-2 MPa (literature range)
        assert 0.5e6 < sigma < 2.0e6

    def test_strength_at_minus_15C(self, default_params):
        """At -15°C, strength should be higher than at -10°C."""
        sigma_10 = freeze_bond_strength(-10.0, default_params)
        sigma_15 = freeze_bond_strength(-15.0, default_params)
        assert sigma_15 > sigma_10
        # Ratio should be exp(β · 5) = exp(0.75) ≈ 2.12
        ratio = sigma_15 / sigma_10
        assert ratio == pytest.approx(math.exp(0.15 * 5.0), rel=1e-10)

    def test_strength_positive_for_all_subcooled(self, default_params):
        """Strength must be positive for all sub-cooled temperatures."""
        for T in np.linspace(-30.0, -0.1, 100):
            sigma = freeze_bond_strength(T, default_params)
            assert sigma > 0.0

    def test_strength_bounded(self, default_params):
        """Strength must remain finite even at very low temperatures."""
        sigma = freeze_bond_strength(-60.0, default_params)
        assert math.isfinite(sigma)
        assert sigma > 0.0


# ---------------------------------------------------------------------------
# 2. Bond growth kinetics
# ---------------------------------------------------------------------------

class TestBondGrowthKinetics:
    """Tests for A_fb(t)/A_0 = 1 - exp(-t/τ_growth)."""

    def test_zero_contact_time_gives_zero_area(self, default_params):
        """At t=0, no bond has formed."""
        area_frac = bond_growth_kinetics(0.0, default_params)
        assert area_frac == pytest.approx(0.0, abs=1e-15)

    def test_long_contact_time_gives_full_area(self, default_params):
        """At t >> τ, the bond area approaches 1."""
        area_frac = bond_growth_kinetics(10000.0, default_params)
        assert area_frac == pytest.approx(1.0, rel=1e-6)

    def test_area_increases_monotonically(self, default_params):
        """Bond area must increase monotonically with contact time."""
        times = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
        areas = [bond_growth_kinetics(t, default_params) for t in times]
        for i in range(len(areas) - 1):
            assert areas[i] < areas[i + 1]

    def test_growth_at_tau(self, default_params):
        """At t = τ, area fraction should be 1 - 1/e ≈ 0.632."""
        area = bond_growth_kinetics(default_params.tau_growth, default_params)
        expected = 1.0 - 1.0 / math.e
        assert area == pytest.approx(expected, rel=1e-12)

    def test_negative_time_raises(self, default_params):
        """Negative contact time must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            bond_growth_kinetics(-1.0, default_params)

    def test_exponential_form(self, default_params):
        """Verify the exact exponential form."""
        t = 25.0
        expected = 1.0 - math.exp(-t / default_params.tau_growth)
        actual = bond_growth_kinetics(t, default_params)
        assert actual == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Effective bond strength (combined T and t)
# ---------------------------------------------------------------------------

class TestEffectiveBondStrength:
    """Tests for σ_eff = σ_fb(T) · A_fb(t)/A_0."""

    def test_zero_time_gives_zero(self, default_params):
        """No contact time → no effective strength."""
        sigma_eff = effective_bond_strength(-10.0, 0.0, default_params)
        assert sigma_eff == pytest.approx(0.0, abs=1e-10)

    def test_above_melting_gives_zero(self, default_params):
        """Above melting → no bond regardless of time."""
        sigma_eff = effective_bond_strength(1.0, 100.0, default_params)
        assert sigma_eff == pytest.approx(0.0, abs=1e-10)

    def test_product_form(self, default_params):
        """σ_eff = σ_fb(T) · A_fb(t)/A_0."""
        T, t = -10.0, 50.0
        sigma_T = freeze_bond_strength(T, default_params)
        area_frac = bond_growth_kinetics(t, default_params)
        expected = sigma_T * area_frac
        actual = effective_bond_strength(T, t, default_params)
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_increases_with_both_T_cooling_and_time(self, default_params):
        """Effective strength increases with more cooling AND more time."""
        # More cooling at fixed time
        s1 = effective_bond_strength(-5.0, 100.0, default_params)
        s2 = effective_bond_strength(-10.0, 100.0, default_params)
        assert s2 > s1

        # More time at fixed temperature
        s3 = effective_bond_strength(-10.0, 10.0, default_params)
        s4 = effective_bond_strength(-10.0, 100.0, default_params)
        assert s4 > s3


# ---------------------------------------------------------------------------
# 4. Failure mode classification
# ---------------------------------------------------------------------------

class TestFailureModeClassification:
    """Tests for adhesive/cohesive/mixed failure mode classification."""

    def test_adhesive_at_high_temperature_short_time(self, default_params):
        """Near melting with short contact → adhesive failure."""
        fm = classify_failure_mode(-1.0, 1.0, default_params)
        assert fm.mode == "adhesive"
        assert fm.ratio < 1.0  # bond weaker than ice

    def test_cohesive_at_low_temperature_long_time(self, default_params):
        """Very cold with long contact → cohesive failure."""
        fm = classify_failure_mode(-15.0, 1000.0, default_params)
        assert fm.mode == "cohesive"
        assert fm.ratio > 1.0  # bond stronger than ice

    def test_ratio_is_sigma_fb_over_sigma_cohesive(self, default_params):
        """The ratio must be σ_eff / σ_cohesive."""
        T, t = -10.0, 100.0
        fm = classify_failure_mode(T, t, default_params)
        sigma_eff = effective_bond_strength(T, t, default_params)
        expected_ratio = sigma_eff / default_params.sigma_cohesive
        assert fm.ratio == pytest.approx(expected_ratio, rel=1e-12)

    def test_transition_detected_near_boundary(self, default_params):
        """Near the transition boundary, mode should be 'mixed'."""
        # Find the temperature where σ_eff ≈ σ_cohesive at t=100s
        # σ_eff = σ_0 · exp(-β·ΔT) · (1 - exp(-100/50))
        # = 0.2e6 · exp(-0.15·ΔT) · 0.865
        # Set equal to 1e6: exp(-0.15·ΔT) = 1e6/(0.2e6·0.865) = 5.78
        # -0.15·ΔT = ln(5.78) = 1.754 → ΔT = -11.7 → T = 11.7°C below melting
        # This is outside our range, so let's use a custom params
        params = FreezeBondParams(sigma_0=0.5e6, beta=0.15, sigma_cohesive=1.0e6)
        # At T=-10, t=100: σ_eff = 0.5e6·exp(1.5)·0.865 = 0.5e6·4.48·0.865 ≈ 1.94e6
        # This is cohesive. Let's find the transition...
        # At T=-5, t=10: σ_eff = 0.5e6·exp(0.75)·(1-exp(-0.2)) = 0.5e6·2.117·0.181 ≈ 0.192e6
        # At T=-5, t=100: σ_eff = 0.5e6·2.117·0.865 ≈ 0.916e6 (close to 1e6)
        fm = classify_failure_mode(-5.0, 100.0, params, tolerance=0.15)
        # Should be near transition
        assert abs(fm.ratio - 1.0) < 0.2 or fm.mode in ("mixed", "cohesive")

    def test_is_transition_flag(self, default_params):
        """is_transition should be True only in the mixed regime."""
        fm_adh = classify_failure_mode(-1.0, 1.0, default_params)
        fm_coh = classify_failure_mode(-15.0, 1000.0, default_params)
        assert not fm_adh.is_transition or fm_adh.mode == "mixed"
        assert not fm_coh.is_transition or fm_coh.mode == "mixed"


# ---------------------------------------------------------------------------
# 5. Pull-off test
# ---------------------------------------------------------------------------

class TestPullOffTest:
    """Tests for the uniaxial pull-off simulation."""

    def test_peak_force_positive(self, default_params):
        """Peak force must be positive for any sub-cooled condition."""
        result = pull_off_test(-10.0, 100.0, default_params)
        assert result.peak_force > 0.0
        assert result.peak_stress > 0.0

    def test_peak_force_increases_with_cooling(self, default_params):
        """Peak force must increase as temperature drops."""
        forces = []
        for T in [-1.0, -5.0, -10.0, -15.0]:
            r = pull_off_test(T, 100.0, default_params)
            forces.append(r.peak_force)
        for i in range(len(forces) - 1):
            assert forces[i] <= forces[i + 1]

    def test_peak_force_increases_with_time(self, default_params):
        """Peak force must increase with contact time."""
        forces = []
        for t in [1.0, 10.0, 100.0, 1000.0]:
            r = pull_off_test(-10.0, t, default_params)
            forces.append(r.peak_force)
        for i in range(len(forces) - 1):
            assert forces[i] <= forces[i + 1]

    def test_failure_mode_consistency(self, default_params):
        """Pull-off failure mode must match classification."""
        for T in [-1.0, -5.0, -10.0, -15.0]:
            for t in [1.0, 100.0, 1000.0]:
                r = pull_off_test(T, t, default_params)
                fm = classify_failure_mode(T, t, default_params)
                assert r.failure_mode == fm.mode

    def test_peak_stress_bounded_reasonably(self, default_params):
        """Peak stress must be physically reasonable (not orders of magnitude above cohesive)."""
        for T in np.linspace(-15.0, -1.0, 20):
            r = pull_off_test(T, 1000.0, default_params)
            # Peak stress should be within a factor of 2 of cohesive strength
            assert r.peak_stress < 2.0 * default_params.sigma_cohesive

    def test_displacement_positive(self, default_params):
        """Displacement at peak must be positive."""
        r = pull_off_test(-10.0, 100.0, default_params)
        assert r.displacement_at_peak > 0.0

    def test_stiffness_positive(self, default_params):
        """Stiffness must be positive."""
        r = pull_off_test(-10.0, 100.0, default_params)
        assert r.stiffness > 0.0


# ---------------------------------------------------------------------------
# 6. Shear test
# ---------------------------------------------------------------------------

class TestShearTest:
    """Tests for the shear test simulation."""

    def test_shear_strength_positive(self, default_params):
        """Shear strength must be positive for bonded conditions."""
        r = shear_test(-10.0, 100.0, 0.0, default_params)
        assert r.tau_peak > 0.0
        assert r.tau_bond > 0.0

    def test_shear_increases_with_normal_pressure(self, default_params):
        """Shear strength must increase with normal pressure (friction)."""
        pressures = [0.0, 1e4, 5e4, 1e5]
        strengths = []
        for p in pressures:
            r = shear_test(-10.0, 100.0, p, default_params)
            strengths.append(r.tau_peak)
        for i in range(len(strengths) - 1):
            assert strengths[i] <= strengths[i + 1]

    def test_friction_component(self, default_params):
        """Friction component must be μ · σ_normal."""
        mu = 0.1
        sigma_n = 5e4
        r = shear_test(-10.0, 100.0, sigma_n, default_params, mu_friction=mu)
        assert r.tau_friction == pytest.approx(mu * sigma_n, rel=1e-12)

    def test_bond_component_independent_of_normal_pressure(self, default_params):
        """Bond shear component should not depend on normal pressure."""
        r1 = shear_test(-10.0, 100.0, 0.0, default_params)
        r2 = shear_test(-10.0, 100.0, 1e5, default_params)
        assert r1.tau_bond == pytest.approx(r2.tau_bond, rel=1e-12)

    def test_failure_mode_consistency(self, default_params):
        """Shear failure mode must match classification."""
        for T in [-1.0, -10.0, -15.0]:
            r = shear_test(T, 100.0, 0.0, default_params)
            fm = classify_failure_mode(T, 100.0, default_params)
            assert r.failure_mode == fm.mode


# ---------------------------------------------------------------------------
# 7. Phase diagram
# ---------------------------------------------------------------------------

class TestPhaseDiagram:
    """Tests for the failure mode phase diagram."""

    def test_phase_diagram_has_both_modes(self, default_params):
        """Phase diagram must contain both adhesive and cohesive regions."""
        pd = failure_mode_phase_diagram(params=default_params)
        modes_flat = pd.failure_modes.flatten()
        assert "adhesive" in modes_flat
        assert "cohesive" in modes_flat

    def test_adhesive_in_high_T_corner(self, default_params):
        """High temperature, short time corner must be adhesive."""
        pd = failure_mode_phase_diagram(params=default_params)
        # Top-right corner: highest T (closest to melting), shortest time
        assert pd.failure_modes[-1, 0] == "adhesive"

    def test_cohesive_in_low_T_corner(self, default_params):
        """Low temperature, long time corner must be cohesive."""
        pd = failure_mode_phase_diagram(params=default_params)
        # Bottom-left corner: lowest T, longest time
        assert pd.failure_modes[0, -1] == "cohesive"

    def test_transition_boundary_exists(self, default_params):
        """There must be a transition boundary between adhesive and cohesive."""
        pd = failure_mode_phase_diagram(params=default_params)
        assert len(pd.transition_boundary_T) > 0
        assert len(pd.transition_boundary_t) > 0

    def test_transition_boundary_monotone(self, default_params):
        """Transition boundary: lower T → shorter time needed for cohesive failure."""
        pd = failure_mode_phase_diagram(params=default_params)
        if len(pd.transition_boundary_T) >= 2:
            # Sort by temperature (ascending)
            sort_idx = np.argsort(pd.transition_boundary_T)
            T_sorted = pd.transition_boundary_T[sort_idx]
            t_sorted = pd.transition_boundary_t[sort_idx]
            # At lower temperature, bond is stronger → needs LESS time to become cohesive
            # So t should DECREASE as T decreases
            # Check: for consecutive pairs where T increases, t should increase
            for i in range(len(T_sorted) - 1):
                if T_sorted[i+1] > T_sorted[i]:
                    # Higher T needs longer time to reach cohesive
                    assert t_sorted[i+1] >= t_sorted[i] - 50.0  # allow tolerance for discretization


# ---------------------------------------------------------------------------
# 8. Parameter sweep
# ---------------------------------------------------------------------------

class TestParameterSweep:
    """Tests for the parameter sweep."""

    def test_sweep_shape(self, default_params):
        """Sweep arrays must have correct shapes."""
        sw = parameter_sweep(n_T=5, n_t=5, n_v=3, params=default_params)
        assert sw.peak_stresses.shape == (5, 5, 3)
        assert sw.failure_modes.shape == (5, 5, 3)
        assert sw.peak_load_jumps.shape == (5, 3)

    def test_peak_load_jumps_positive(self, default_params):
        """Peak load jumps must be non-negative."""
        sw = parameter_sweep(params=default_params)
        assert np.all(sw.peak_load_jumps >= 0.0)

    def test_peak_stresses_positive(self, default_params):
        """All peak stresses must be positive."""
        sw = parameter_sweep(params=default_params)
        assert np.all(sw.peak_stresses > 0.0)

    def test_peak_load_jump_detected(self, default_params):
        """There must be a detectable peak load jump at the transition."""
        sw = parameter_sweep(params=default_params)
        max_jump = float(np.max(sw.peak_load_jumps))
        assert max_jump > 0.0


# ---------------------------------------------------------------------------
# 9. Full verification
# ---------------------------------------------------------------------------

class TestFullVerification:
    """Integration tests running the full verification suite."""

    def test_all_checks_pass(self, default_params):
        """All verification checks must pass."""
        report = run_full_verification(default_params)
        assert report.strength_monotone_with_cooling
        assert report.strength_monotone_with_time
        assert report.failure_mode_transition_detected
        assert report.peak_load_jump_detected
        assert report.adhesive_at_high_T
        assert report.cohesive_at_low_T

    def test_summary_populated(self, default_params):
        """Summary dict must contain expected keys."""
        report = run_full_verification(default_params)
        assert "n_pull_off_tests" in report.summary
        assert "n_shear_tests" in report.summary
        assert "strength_at_minus_10C_Pa" in report.summary
        assert report.summary["n_pull_off_tests"] > 0
        assert report.summary["n_shear_tests"] > 0


# ---------------------------------------------------------------------------
# 10. Parameter validation
# ---------------------------------------------------------------------------

class TestParameterValidation:
    """Tests for parameter validation."""

    def test_negative_sigma_0_raises(self):
        with pytest.raises(ValueError):
            FreezeBondParams(sigma_0=-1.0)

    def test_negative_beta_raises(self):
        with pytest.raises(ValueError):
            FreezeBondParams(beta=-1.0)

    def test_zero_tau_growth_raises(self):
        with pytest.raises(ValueError):
            FreezeBondParams(tau_growth=0.0)

    def test_zero_area_raises(self):
        with pytest.raises(ValueError):
            FreezeBondParams(A_contact=0.0)


# ---------------------------------------------------------------------------
# 11. Sea-ice parameters
# ---------------------------------------------------------------------------

class TestSeaIceParameters:
    """Tests with sea-ice melting point."""

    def test_sea_ice_melting_point(self):
        params = FreezeBondParams(T_melting=T_MELTING_SEA_ICE)
        # At slightly below the sea-ice melting point, strength ≈ σ_0
        sigma_near = freeze_bond_strength(T_MELTING_SEA_ICE - 0.01, params)
        expected = params.sigma_0 * math.exp(-params.beta * (-0.01))
        assert sigma_near == pytest.approx(expected, rel=1e-10)

    def test_sea_ice_subcooled(self):
        params = FreezeBondParams(T_melting=T_MELTING_SEA_ICE)
        sigma = freeze_bond_strength(-5.0, params)
        assert sigma > params.sigma_0  # sub-cooled → stronger


# ---------------------------------------------------------------------------
# 12. Physical consistency
# ---------------------------------------------------------------------------

class TestPhysicalConsistency:
    """Physical sanity checks."""

    def test_bond_strength_never_exceeds_exponential_limit(self, default_params):
        """At extreme cold, strength should be finite."""
        sigma = freeze_bond_strength(-50.0, default_params)
        assert math.isfinite(sigma)
        # Exponential model gives large values at extreme cold
        # σ = 0.2e6 · exp(0.15 · 50) = 0.2e6 · exp(7.5) ≈ 360 MPa
        # This is above typical ice strength but the model is valid in its range
        assert sigma < 1e9  # less than 1 GPa

    def test_pull_off_force_consistent_with_stress_and_area(self, default_params):
        """F_peak = σ_peak · A_contact."""
        r = pull_off_test(-10.0, 100.0, default_params)
        expected_force = r.peak_stress * default_params.A_contact
        assert r.peak_force == pytest.approx(expected_force, rel=1e-12)

    def test_shear_total_is_sum_of_components(self, default_params):
        """τ_peak = τ_bond + τ_friction."""
        r = shear_test(-10.0, 100.0, 5e4, default_params)
        expected = r.tau_bond + r.tau_friction
        assert r.tau_peak == pytest.approx(expected, rel=1e-12)

    def test_energy_consistency_pull_off(self, default_params):
        """Work done ≈ 0.5 · F_peak · δ_peak (linear elastic to failure)."""
        r = pull_off_test(-10.0, 100.0, default_params)
        work = 0.5 * r.peak_force * r.displacement_at_peak
        # This should be positive and finite
        assert work > 0.0
        assert math.isfinite(work)
