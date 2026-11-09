"""Multi-year (MY) ice parameterization and Lindqvist MY extension tests.

Tests cover:
1. IceAgeProperties: FY/MY property ranges and defaults
2. my_correction_factor: k_MY(σ_c, h_i) behavior
3. MixedIceField: bimodal FSD sampling, effective strength
4. lindqvist_my_components: FY/MY/mixed resistance decomposition
5. lindqvist_my_sweep: parameter sweep completeness
6. Negative controls and edge cases
7. Provenance ledger hash stability
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.level_ice import formula as F
from tensordem.level_ice.multiyear_ice import (
    FY_DEFAULT,
    FY_RANGES,
    MY_DEFAULT,
    MY_RANGES,
    IceAgeProperties,
    MixedIceField,
    lindqvist_my_components,
    lindqvist_my_sweep,
    my_correction_factor,
    my_ledger_sha256,
)


# ---------------------------------------------------------------------------
# IceAgeProperties
# ---------------------------------------------------------------------------

class TestIceAgeProperties:
    def test_fy_defaults_in_range(self):
        fy = IceAgeProperties.fy()
        assert fy.label == "FY"
        lo, hi = FY_RANGES["compressive_strength_pa"]
        assert lo <= fy.compressive_strength_pa <= hi
        lo, hi = FY_RANGES["flexural_strength_pa"]
        assert lo <= fy.flexural_strength_pa <= hi
        lo, hi = FY_RANGES["elastic_modulus_pa"]
        assert lo <= fy.elastic_modulus_pa <= hi

    def test_my_defaults_in_range(self):
        my = IceAgeProperties.my()
        assert my.label == "MY"
        lo, hi = MY_RANGES["compressive_strength_pa"]
        assert lo <= my.compressive_strength_pa <= hi
        lo, hi = MY_RANGES["flexural_strength_pa"]
        assert lo <= my.flexural_strength_pa <= hi
        lo, hi = MY_RANGES["elastic_modulus_pa"]
        assert lo <= my.elastic_modulus_pa <= hi

    def test_my_stronger_than_fy(self):
        fy = IceAgeProperties.fy()
        my = IceAgeProperties.my()
        assert my.compressive_strength_pa > fy.compressive_strength_pa
        assert my.flexural_strength_pa > fy.flexural_strength_pa
        assert my.elastic_modulus_pa > fy.elastic_modulus_pa
        assert my.fracture_toughness_pa_m05 > fy.fracture_toughness_pa_m05

    def test_custom_properties(self):
        fy = IceAgeProperties.fy(sigma_c=2.0e6, sigma_f=0.6e6, E=5.5e9, K_IC=0.25e6)
        assert fy.compressive_strength_pa == 2.0e6
        assert fy.flexural_strength_pa == 0.6e6
        assert fy.elastic_modulus_pa == 5.5e9
        assert fy.fracture_toughness_pa_m05 == 0.25e6

    def test_to_ice_state(self):
        my = IceAgeProperties.my()
        ice = my.to_ice_state(thickness_m=1.5)
        assert ice.thickness_m == 1.5
        assert ice.compressive_strength_pa == my.compressive_strength_pa
        assert ice.flexural_strength_pa == my.flexural_strength_pa
        assert ice.elastic_modulus_pa == my.elastic_modulus_pa

    def test_as_dict(self):
        fy = IceAgeProperties.fy()
        d = fy.as_dict()
        assert d["label"] == "FY"
        assert "compressive_strength_pa" in d
        assert "fracture_toughness_pa_m05" in d


# ---------------------------------------------------------------------------
# my_correction_factor
# ---------------------------------------------------------------------------

class TestMyCorrectionFactor:
    def test_fy_baseline_gives_k_near_1(self):
        k = my_correction_factor(FY_DEFAULT["compressive_strength_pa"], 1.0)
        assert k == pytest.approx(1.0, abs=0.01)

    def test_my_ice_gives_k_greater_than_1(self):
        k = my_correction_factor(MY_DEFAULT["compressive_strength_pa"], 1.0)
        assert k > 1.0
        # MY sigma_c ~ 5x FY, alpha=0.6 → k ~ 5^0.6 ≈ 2.6
        expected = (MY_DEFAULT["compressive_strength_pa"] / FY_DEFAULT["compressive_strength_pa"]) ** 0.6
        assert k == pytest.approx(expected, rel=0.01)

    def test_k_increases_with_strength(self):
        k1 = my_correction_factor(2.0e6, 1.0)
        k2 = my_correction_factor(8.0e6, 1.0)
        assert k2 > k1

    def test_k_increases_with_thickness(self):
        k1 = my_correction_factor(7.5e6, 0.5)
        k2 = my_correction_factor(7.5e6, 2.0)
        assert k2 > k1

    def test_k_never_below_1(self):
        k = my_correction_factor(0.5e6, 0.3)  # weaker than FY baseline
        assert k >= 1.0

    def test_dimensional_consistency(self):
        """k_MY should be dimensionless."""
        k = my_correction_factor(7.5e6, 1.5)
        assert isinstance(k, float)
        assert math.isfinite(k)
        assert k > 0

    def test_custom_parameters(self):
        k = my_correction_factor(7.5e6, 1.0, alpha=0.5, beta=0.3, h_ref=1.5)
        assert k > 1.0


# ---------------------------------------------------------------------------
# MixedIceField
# ---------------------------------------------------------------------------

class TestMixedIceField:
    def test_pure_fy(self):
        field = MixedIceField(
            my_fraction=0.0, thickness_m=1.0,
            fy_properties=IceAgeProperties.fy(),
            my_properties=IceAgeProperties.my(),
        )
        assert field.fy_fraction == 1.0
        assert field.my_fraction == 0.0
        eff = field.effective_strength()
        assert eff["compressive_strength_pa"] == pytest.approx(FY_DEFAULT["compressive_strength_pa"])

    def test_pure_my(self):
        field = MixedIceField(
            my_fraction=1.0, thickness_m=1.0,
            fy_properties=IceAgeProperties.fy(),
            my_properties=IceAgeProperties.my(),
        )
        eff = field.effective_strength()
        assert eff["compressive_strength_pa"] == pytest.approx(MY_DEFAULT["compressive_strength_pa"])

    def test_mixed_50_50(self):
        field = MixedIceField(
            my_fraction=0.5, thickness_m=1.0,
            fy_properties=IceAgeProperties.fy(),
            my_properties=IceAgeProperties.my(),
        )
        eff = field.effective_strength()
        expected_sigma_c = 0.5 * FY_DEFAULT["compressive_strength_pa"] + 0.5 * MY_DEFAULT["compressive_strength_pa"]
        assert eff["compressive_strength_pa"] == pytest.approx(expected_sigma_c)

    def test_fsd_sampling_returns_correct_count(self):
        field = MixedIceField(
            my_fraction=0.3, thickness_m=1.0,
            fy_properties=IceAgeProperties.fy(),
            my_properties=IceAgeProperties.my(),
        )
        sizes = field.sample_fsd(100, seed=42)
        assert len(sizes) == 100

    def test_fsd_bimodal_characteristic_sizes(self):
        """MY floes should be larger than FY floes on average.

        Note: power-law with exponent -2 is heavy-tailed; the mean is dominated
        by rare large values. We check that MY floes are consistently larger
        by comparing the ratio of means/medians.
        """
        fy_field = MixedIceField(
            my_fraction=0.0, thickness_m=1.0,
            fy_properties=IceAgeProperties.fy(),
            my_properties=IceAgeProperties.my(),
            fy_char_floe_size_m=10.0,
            my_char_floe_size_m=40.0,
        )
        my_field = MixedIceField(
            my_fraction=1.0, thickness_m=1.0,
            fy_properties=IceAgeProperties.fy(),
            my_properties=IceAgeProperties.my(),
            fy_char_floe_size_m=10.0,
            my_char_floe_size_m=40.0,
        )
        fy_sizes = fy_field.sample_fsd(500, seed=123)
        my_sizes = my_field.sample_fsd(500, seed=123)

        # MY floes should be larger (ratio ~4x for characteristic sizes)
        assert np.mean(my_sizes) > np.mean(fy_sizes)
        # The ratio of means should be close to the ratio of characteristic sizes
        # (within power-law spread, expect ~3-5x for 4x char size ratio)
        ratio = np.mean(my_sizes) / np.mean(fy_sizes)
        assert 2.0 < ratio < 6.0, f"MY/FY mean ratio {ratio:.2f} outside expected range"

        # Also check max values (MY should have larger maximum floes)
        assert np.max(my_sizes) > np.max(fy_sizes)

    def test_to_ice_state(self):
        field = MixedIceField(
            my_fraction=0.5, thickness_m=1.5,
            fy_properties=IceAgeProperties.fy(),
            my_properties=IceAgeProperties.my(),
        )
        ice = field.to_ice_state()
        assert ice.thickness_m == 1.5
        eff = field.effective_strength()
        assert ice.compressive_strength_pa == pytest.approx(eff["compressive_strength_pa"])

    def test_invalid_my_fraction_rejected(self):
        with pytest.raises(ValueError):
            MixedIceField(
                my_fraction=1.5, thickness_m=1.0,
                fy_properties=IceAgeProperties.fy(),
                my_properties=IceAgeProperties.my(),
            )

    def test_as_dict(self):
        field = MixedIceField(
            my_fraction=0.25, thickness_m=1.0,
            fy_properties=IceAgeProperties.fy(),
            my_properties=IceAgeProperties.my(),
        )
        d = field.as_dict()
        assert d["my_fraction"] == 0.25
        assert "effective_strength" in d


# ---------------------------------------------------------------------------
# lindqvist_my_components
# ---------------------------------------------------------------------------

class TestLindqvistMyComponents:
    def test_fy_reduces_to_standard(self):
        """Pure FY should give same total as standard Lindqvist (within k_MY=1)."""
        ice = F.IceState(thickness_m=1.0)
        hull = F.ICEBREAKER
        speed = 2.0

        standard = F.lindqvist_components(ice, hull, speed)
        my_result = lindqvist_my_components(ice, hull, speed, ice_age="FY")

        # FY: k_MY should be ~1, so corrected terms ≈ original
        assert my_result["k_MY"] == pytest.approx(1.0, abs=0.01)
        assert my_result["k_effective"] == pytest.approx(1.0, abs=0.01)
        assert my_result["crushing_N"] == pytest.approx(standard["crushing_N"])
        # Total should be very close (bending/submersion uncorrected for FY)
        assert my_result["total_N"] == pytest.approx(standard["total_N"], rel=0.01)

    def test_my_increases_resistance(self):
        """MY ice should produce higher resistance than FY."""
        fy_ice = F.IceState(
            thickness_m=1.0,
            compressive_strength_pa=FY_DEFAULT["compressive_strength_pa"],
            flexural_strength_pa=FY_DEFAULT["flexural_strength_pa"],
        )
        my_ice = F.IceState(
            thickness_m=1.0,
            compressive_strength_pa=MY_DEFAULT["compressive_strength_pa"],
            flexural_strength_pa=MY_DEFAULT["flexural_strength_pa"],
        )
        hull = F.ICEBREAKER
        speed = 3.0

        fy_result = lindqvist_my_components(fy_ice, hull, speed, ice_age="FY")
        my_result = lindqvist_my_components(my_ice, hull, speed, ice_age="MY")

        assert my_result["total_N"] > fy_result["total_N"]
        assert my_result["k_MY"] > fy_result["k_MY"]

    def test_mixed_interpolates(self):
        """Mixed field resistance should interpolate between FY and MY."""
        # Use appropriate IceState for each case
        fy_ice = F.IceState(
            thickness_m=1.0,
            compressive_strength_pa=FY_DEFAULT["compressive_strength_pa"],
            flexural_strength_pa=FY_DEFAULT["flexural_strength_pa"],
        )
        my_ice = F.IceState(
            thickness_m=1.0,
            compressive_strength_pa=MY_DEFAULT["compressive_strength_pa"],
            flexural_strength_pa=MY_DEFAULT["flexural_strength_pa"],
        )
        # For mixed, use the mixed field's effective properties
        mixed_field = MixedIceField(
            my_fraction=0.5, thickness_m=1.0,
            fy_properties=IceAgeProperties.fy(),
            my_properties=IceAgeProperties.my(),
        )
        mixed_ice = mixed_field.to_ice_state()

        hull = F.ICEBREAKER
        speed = 3.0

        fy_res = lindqvist_my_components(fy_ice, hull, speed, ice_age="FY", my_fraction=0.0)
        my_res = lindqvist_my_components(my_ice, hull, speed, ice_age="MY", my_fraction=1.0)
        mixed_res = lindqvist_my_components(mixed_ice, hull, speed, ice_age="mixed", my_fraction=0.5)

        assert fy_res["total_N"] < mixed_res["total_N"] < my_res["total_N"]

    def test_component_split_sums_to_total(self):
        result = lindqvist_my_components(
            F.IceState(thickness_m=1.5), F.ICEBREAKER, 4.0,
            ice_age="mixed", my_fraction=0.5,
        )
        parts = (result["crushing_N"] + result["bending_corrected_N"]
                 + result["submersion_corrected_N"] + result["added_drag_N"])
        assert parts == pytest.approx(result["total_N"], rel=1e-12)

    def test_fractions_sum_less_than_1(self):
        result = lindqvist_my_components(
            F.IceState(thickness_m=1.0), F.TRANSPORT, 2.0,
            ice_age="MY",
        )
        frac_sum = (result["crushing_fraction"] + result["bending_fraction"]
                    + result["submersion_fraction"] + result["drag_fraction"])
        assert frac_sum == pytest.approx(1.0, rel=1e-12)

    def test_k_effective_for_mixed(self):
        """k_effective should interpolate between 1 and k_MY."""
        fy_ice = F.IceState(
            thickness_m=1.0,
            compressive_strength_pa=FY_DEFAULT["compressive_strength_pa"],
            flexural_strength_pa=FY_DEFAULT["flexural_strength_pa"],
        )
        my_ice = F.IceState(
            thickness_m=1.0,
            compressive_strength_pa=MY_DEFAULT["compressive_strength_pa"],
            flexural_strength_pa=MY_DEFAULT["flexural_strength_pa"],
        )
        hull = F.ICEBREAKER
        speed = 2.0

        fy_res = lindqvist_my_components(fy_ice, hull, speed, ice_age="FY")
        my_res = lindqvist_my_components(my_ice, hull, speed, ice_age="MY")
        mixed_res = lindqvist_my_components(fy_ice, hull, speed, ice_age="mixed", my_fraction=0.5)

        assert fy_res["k_effective"] == pytest.approx(1.0, abs=0.01)
        assert my_res["k_effective"] > 1.0
        # For mixed: k_eff = 1 + 0.5 * (k_MY - 1), where k_MY is computed from mixed defaults
        expected_k_my = my_correction_factor(
            0.5 * FY_DEFAULT["compressive_strength_pa"] + 0.5 * MY_DEFAULT["compressive_strength_pa"],
            1.0
        )
        expected_k_eff = 1.0 + 0.5 * (expected_k_my - 1.0)
        assert mixed_res["k_effective"] == pytest.approx(expected_k_eff, rel=0.01)

    def test_thickness_scaling(self):
        """Thicker ice should have higher k_MY."""
        k1 = my_correction_factor(MY_DEFAULT["compressive_strength_pa"], 0.5)
        k2 = my_correction_factor(MY_DEFAULT["compressive_strength_pa"], 2.0)
        assert k2 > k1

    def test_speed_affects_drag_only(self):
        """Speed should only affect the drag term, not k_MY."""
        ice = F.IceState(
            thickness_m=1.0,
            compressive_strength_pa=MY_DEFAULT["compressive_strength_pa"],
            flexural_strength_pa=MY_DEFAULT["flexural_strength_pa"],
        )
        hull = F.ICEBREAKER

        r1 = lindqvist_my_components(ice, hull, 1.0, ice_age="MY")
        r2 = lindqvist_my_components(ice, hull, 5.0, ice_age="MY")

        assert r1["k_MY"] == pytest.approx(r2["k_MY"])
        assert r1["crushing_N"] == pytest.approx(r2["crushing_N"])
        assert r2["added_drag_N"] > r1["added_drag_N"]

    def test_invalid_inputs_rejected(self):
        ice = F.IceState(thickness_m=1.0)
        hull = F.ICEBREAKER
        with pytest.raises(ValueError):
            lindqvist_my_components(ice, hull, -1.0, ice_age="FY")
        with pytest.raises(ValueError):
            lindqvist_my_components(ice, hull, 2.0, ice_age="mixed", my_fraction=1.5)


# ---------------------------------------------------------------------------
# lindqvist_my_sweep
# ---------------------------------------------------------------------------

class TestLindqvistMySweep:
    def test_sweep_completeness(self):
        results = lindqvist_my_sweep(
            F.ICEBREAKER,
            my_fractions=(0.0, 0.5, 1.0),
            thicknesses_m=(0.5, 1.0, 2.0),
            speeds_kn=(2.0, 6.0, 10.0),
        )
        expected = 3 * 3 * 3  # 27 combinations
        assert len(results) == expected

    def test_sweep_monotonic_in_my_fraction(self):
        """At fixed thickness and speed, resistance should increase with MY fraction."""
        results = lindqvist_my_sweep(
            F.ICEBREAKER,
            my_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
            thicknesses_m=(1.0,),
            speeds_kn=(4.0,),
        )
        totals = [r["total_N"] for r in results]
        for i in range(len(totals) - 1):
            assert totals[i] <= totals[i + 1], (
                f"Resistance should increase with MY fraction: {totals[i]} > {totals[i+1]}"
            )

    def test_sweep_monotonic_in_thickness(self):
        """At fixed MY fraction and speed, resistance should increase with thickness."""
        results = lindqvist_my_sweep(
            F.ICEBREAKER,
            my_fractions=(0.5,),
            thicknesses_m=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
            speeds_kn=(4.0,),
        )
        totals = [r["total_N"] for r in results]
        for i in range(len(totals) - 1):
            assert totals[i] <= totals[i + 1], (
                f"Resistance should increase with thickness: {totals[i]} > {totals[i+1]}"
            )

    def test_sweep_monotonic_in_speed(self):
        """At fixed MY fraction and thickness, resistance should increase with speed."""
        results = lindqvist_my_sweep(
            F.ICEBREAKER,
            my_fractions=(0.5,),
            thicknesses_m=(1.0,),
            speeds_kn=(2.0, 4.0, 6.0, 8.0, 10.0),
        )
        totals = [r["total_N"] for r in results]
        for i in range(len(totals) - 1):
            assert totals[i] <= totals[i + 1], (
                f"Resistance should increase with speed: {totals[i]} > {totals[i+1]}"
            )

    def test_sweep_params_present(self):
        results = lindqvist_my_sweep(
            F.ICEBREAKER,
            my_fractions=(0.0, 1.0),
            thicknesses_m=(1.0,),
            speeds_kn=(4.0,),
        )
        for r in results:
            assert "sweep_params" in r
            assert "my_fraction" in r["sweep_params"]
            assert "thickness_m" in r["sweep_params"]
            assert "speed_kn" in r["sweep_params"]
            assert "speed_m_s" in r["sweep_params"]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestMyProvenance:
    def test_ledger_hash_is_stable(self):
        h1 = my_ledger_sha256()
        h2 = my_ledger_sha256()
        assert h1 == h2
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)

    def test_ledger_hash_is_hex_string(self):
        digest = my_ledger_sha256()
        assert isinstance(digest, str)
        int(digest, 16)  # valid hex


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

class TestNegativeControls:
    def test_zero_thickness_rejected(self):
        with pytest.raises(ValueError):
            my_correction_factor(7.5e6, 0.0)

    def test_zero_strength_rejected(self):
        with pytest.raises(ValueError):
            my_correction_factor(0.0, 1.0)

    def test_negative_speed_rejected(self):
        ice = F.IceState(thickness_m=1.0)
        hull = F.ICEBREAKER
        with pytest.raises(ValueError):
            lindqvist_my_components(ice, hull, -1.0, ice_age="FY")

    def test_negative_my_fraction_rejected(self):
        ice = F.IceState(thickness_m=1.0)
        hull = F.ICEBREAKER
        with pytest.raises(ValueError):
            lindqvist_my_components(ice, hull, 2.0, ice_age="mixed", my_fraction=-0.1)
