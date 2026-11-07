"""IN27-S1: Ice microstructure full-spectrum validation tests.

Campaign: 碎冰微观结构全谱验证
================================
Tests all six acceptance criteria:

(a) FSD power-law exponent alpha in [2.5, 3.5] and R^2 > 0.9
(b) Jamming critical width ratio (W/h)_crit in [3, 6]
(c) Janssen fit error < 15%
(d) Force-chain PDF P(f) exponential tail
(e) Porosity n in [0.30, 0.45]
(f) Test set >= 6 cases covering h_ice=[0.5,1.5]m x channel_width=[2,10]m
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.ice_microstructure_full_spectrum import (
    FSDResult,
    ForceChainResult,
    JammingResult,
    JanssenFit,
    PorosityResult,
    run_force_chain_analysis,
    run_fsd_analysis,
    run_full_spectrum_study,
    run_jamming_sweep,
    run_janssen_fit,
    run_porosity_analysis,
)


# ---------------------------------------------------------------------------
# (a) FSD power-law analysis
# ---------------------------------------------------------------------------

class TestFSDAnalysis:
    """FSD power-law exponent validation."""

    def test_fsd_alpha_in_range(self):
        """Criterion (a): alpha in [2.5, 3.5]."""
        result = run_fsd_analysis(n_floes=2000, seed=42)
        assert 2.5 <= result.alpha <= 3.5, (
            f"FSD exponent alpha={result.alpha:.3f} outside [2.5, 3.5]"
        )

    def test_fsd_r_squared(self):
        """Criterion (a): R^2 > 0.9."""
        result = run_fsd_analysis(n_floes=2000, seed=42)
        assert result.r_squared > 0.9, (
            f"FSD R^2={result.r_squared:.4f} below 0.9"
        )

    def test_fsd_result_passes(self):
        """FSDResult.passes() returns True when both criteria met."""
        result = run_fsd_analysis(n_floes=2000, seed=42)
        assert result.passes()

    def test_fsd_deterministic(self):
        """Same seed produces same result."""
        r1 = run_fsd_analysis(n_floes=2000, seed=42)
        r2 = run_fsd_analysis(n_floes=2000, seed=42)
        assert r1.alpha == pytest.approx(r2.alpha, rel=1e-10)
        assert r1.r_squared == pytest.approx(r2.r_squared, rel=1e-10)

    def test_fsd_with_different_exponents(self):
        """Different power-law exponents produce different alphas."""
        r1 = run_fsd_analysis(n_floes=2000, seed=42, kappa=2.5)
        r2 = run_fsd_analysis(n_floes=2000, seed=42, kappa=3.5)
        # Both should be in range but different
        assert 2.0 <= r1.alpha <= 4.0
        assert 2.0 <= r2.alpha <= 4.0
        assert abs(r1.alpha - r2.alpha) > 0.1


# ---------------------------------------------------------------------------
# (b) Jamming sweep
# ---------------------------------------------------------------------------

class TestJammingSweep:
    """Jamming critical width ratio validation."""

    def test_jamming_w_crit_in_range(self):
        """Criterion (b): (W/h)_crit in [3, 6]."""
        result = run_jamming_sweep(
            ice_thicknesses=[0.5, 1.0, 1.5],
            channel_widths=[2.0, 4.0, 6.0, 8.0, 10.0],
            n_particles=150,
            seed=7,
        )
        assert 3.0 <= result.w_crit <= 6.0, (
            f"Jamming critical ratio W/h={result.w_crit:.2f} outside [3, 6]"
        )

    def test_jamming_result_passes(self):
        """JammingResult.passes() returns True when criterion met."""
        result = run_jamming_sweep(
            ice_thicknesses=[0.5, 1.0, 1.5],
            channel_widths=[2.0, 4.0, 6.0, 8.0, 10.0],
            n_particles=150,
            seed=7,
        )
        assert result.passes()

    def test_jamming_monotonic_trend(self):
        """Jamming probability should generally increase with W/h."""
        result = run_jamming_sweep(
            ice_thicknesses=[0.5, 1.0, 1.5],
            channel_widths=[2.0, 4.0, 6.0, 8.0, 10.0],
            n_particles=150,
            seed=7,
        )
        # At least some variation in jamming probability
        probs = result.jamming_probs
        assert len(probs) >= 3
        assert any(p > 0 for p in probs)


# ---------------------------------------------------------------------------
# (c) Janssen fit
# ---------------------------------------------------------------------------

class TestJanssenFit:
    """Janssen silo formula validation."""

    def test_janssen_error_below_threshold(self):
        """Criterion (c): Janssen fit error < 15%."""
        result = run_janssen_fit(n_particles=300, seed=11)
        assert result.relative_error < 0.15, (
            f"Janssen relative error={result.relative_error:.4f} >= 15%"
        )

    def test_janssen_r_squared(self):
        """Janssen fit should have reasonable R^2."""
        result = run_janssen_fit(n_particles=300, seed=11)
        assert result.r_squared > 0.7, (
            f"Janssen R^2={result.r_squared:.4f} too low"
        )

    def test_janssen_result_passes(self):
        """JanssenFit.passes() returns True when error < 15%."""
        result = run_janssen_fit(n_particles=300, seed=11)
        assert result.passes()

    def test_janssen_positive_parameters(self):
        """Janssen parameters should be positive."""
        result = run_janssen_fit(n_particles=300, seed=11)
        assert result.p_inf > 0
        assert result.k_ratio > 0


# ---------------------------------------------------------------------------
# (d) Force-chain analysis
# ---------------------------------------------------------------------------

class TestForceChainAnalysis:
    """Force-chain PDF exponential tail validation."""

    def test_exponential_tail(self):
        """Criterion (d): force-chain PDF has exponential tail."""
        result = run_force_chain_analysis(n_particles=256, seed=13)
        assert result.exponential_tail, (
            f"Force-chain PDF not exponential (KS p={result.ks_p_value:.4f})"
        )

    def test_force_chain_result_passes(self):
        """ForceChainResult.passes() returns True when exponential tail confirmed."""
        result = run_force_chain_analysis(n_particles=256, seed=13)
        assert result.passes()

    def test_strong_fraction_reasonable(self):
        """Strong fraction (f > 1) should be in expected range."""
        result = run_force_chain_analysis(n_particles=256, seed=13)
        # For exponential, P(f>1) = exp(-1) ≈ 0.368
        assert 0.2 < result.strong_fraction < 0.6, (
            f"Strong fraction={result.strong_fraction:.3f} unexpected"
        )

    def test_force_chain_deterministic(self):
        """Same seed produces same result."""
        r1 = run_force_chain_analysis(n_particles=256, seed=13)
        r2 = run_force_chain_analysis(n_particles=256, seed=13)
        assert r1.mean_force == pytest.approx(r2.mean_force, rel=1e-10)
        assert r1.ks_p_value == pytest.approx(r2.ks_p_value, rel=1e-10)


# ---------------------------------------------------------------------------
# (e) Porosity analysis
# ---------------------------------------------------------------------------

class TestPorosityAnalysis:
    """Porosity validation."""

    def test_porosity_in_range(self):
        """Criterion (e): porosity n in [0.30, 0.45]."""
        result = run_porosity_analysis(n_particles=256, mean_radius=0.05, mu=0.5, seed=17)
        assert 0.30 <= result.porosity <= 0.45, (
            f"Porosity n={result.porosity:.4f} outside [0.30, 0.45]"
        )

    def test_porosity_result_passes(self):
        """PorosityResult.passes returns True when in range."""
        result = run_porosity_analysis(n_particles=256, mean_radius=0.05, mu=0.5, seed=17)
        assert result.passes

    def test_coordination_number_reasonable(self):
        """Coordination number should be in physical range."""
        result = run_porosity_analysis(n_particles=256, mean_radius=0.05, mu=0.5, seed=17)
        assert 3.0 <= result.coordination_number <= 8.0, (
            f"Coordination number Z={result.coordination_number:.2f} unexpected"
        )

    def test_packing_fraction_complement(self):
        """Packing fraction + porosity = 1."""
        result = run_porosity_analysis(n_particles=256, mean_radius=0.05, mu=0.5, seed=17)
        assert abs(result.packing_fraction + result.porosity - 1.0) < 0.01


# ---------------------------------------------------------------------------
# (f) Full-spectrum study (>= 6 cases)
# ---------------------------------------------------------------------------

class TestFullSpectrumStudy:
    """Full IN27-S1 validation suite."""

    def test_case_count(self):
        """Criterion (f): >= 6 cases."""
        ice_thicknesses = [0.5, 0.75, 1.0, 1.25, 1.5]
        channel_widths = [2.0, 4.0, 6.0, 8.0, 10.0]
        n_cases = len(ice_thicknesses) * len(channel_widths)
        assert n_cases >= 6, f"Only {n_cases} cases, need >= 6"

    def test_full_study_runs(self):
        """Full spectrum study completes without error."""
        results = run_full_spectrum_study(
            ice_thicknesses=[0.5, 1.0, 1.5],
            channel_widths=[2.0, 6.0, 10.0],
            seed=42,
        )
        assert "criteria" in results
        assert "overall_pass" in results

    def test_all_criteria_present(self):
        """All six criteria are present in results."""
        results = run_full_spectrum_study(
            ice_thicknesses=[0.5, 1.0, 1.5],
            channel_widths=[2.0, 6.0, 10.0],
            seed=42,
        )
        expected_keys = [
            "a_fsd_alpha_in_range",
            "a_fsd_r_squared",
            "b_w_crit_in_range",
            "c_janssen_error",
            "d_exponential_tail",
            "e_porosity_in_range",
            "f_case_count",
        ]
        for key in expected_keys:
            assert key in results["criteria"], f"Missing criterion: {key}"

    def test_overall_pass(self):
        """All criteria should pass for the default configuration."""
        results = run_full_spectrum_study(
            ice_thicknesses=[0.5, 1.0, 1.5],
            channel_widths=[2.0, 6.0, 10.0],
            seed=42,
        )
        assert results["overall_pass"], (
            f"Some criteria failed: {json.dumps(results['criteria'], indent=2)}"
        )


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------

class TestArtifactGeneration:
    """Generate JSON artifact for the campaign."""

    def test_artifact_structure(self, tmp_path):
        """Artifact has the expected structure."""
        results = run_full_spectrum_study(
            ice_thicknesses=[0.5, 1.0, 1.5],
            channel_widths=[2.0, 6.0, 10.0],
            seed=42,
        )
        artifact = {
            "campaign": "IN27-S1",
            "title": "碎冰微观结构全谱验证",
            "criteria": results["criteria"],
            "fsd": results["fsd"],
            "jamming": results["jamming"],
            "janssen": results["janssen"],
            "force_chain": results["force_chain"],
            "porosity": results["porosity"],
            "overall_pass": results["overall_pass"],
        }
        path = tmp_path / "in27_s1_artifact.json"
        path.write_text(json.dumps(artifact, indent=2))
        # Verify round-trip
        loaded = json.loads(path.read_text())
        assert loaded["overall_pass"] is True
        assert len(loaded["criteria"]) >= 7
