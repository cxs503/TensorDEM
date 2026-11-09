"""Tests for ice-ship collision energy conservation verification surface.

Covers all six acceptance criteria:
(a) Energy conservation residual ≤ 2%
(b) Analytical anchor: three-point bending plastic hinge, numerical vs closed-form ≤ 1%
(c) Negative control: V = 0 → d = 0
(d) Double hull: E_outer + E_inner = E_total (≤ 2%), inner hull intact declaration
(e) Minorsky empirical: relative error ≤ 30%
(f) Parameter sweep: V = 2/4/6/8 m/s, d monotonically increasing
"""
from __future__ import annotations

import math
import pytest

from tensordem.ice_collision_energy import (
    PlasticHingeConfig,
    DoubleHullConfig,
    CollisionScenario,
    CollisionResult,
    compute_plastic_moment,
    compute_plastic_section_modulus,
    compute_plastic_hinge_energy,
    compute_friction_energy,
    solve_penetration_depth,
    compute_double_hull_absorption,
    compute_damage_geometry,
    compute_minorsky_damage_volume,
    run_collision_scenario,
    run_parameter_sweep,
    negative_control,
    analytical_anchor_three_point_bending,
    energy_conservation_check,
)


# ---------------------------------------------------------------------------
# (a) Energy conservation residual ≤ 2%
# ---------------------------------------------------------------------------
class TestEnergyConservation:
    """Criterion (a): E_k = E_plastic + E_friction + E_residual, residual ≤ 2%."""

    @pytest.mark.parametrize("velocity", [2.0, 4.0, 6.0, 8.0])
    def test_residual_within_2_percent(self, velocity: float) -> None:
        cfg = PlasticHingeConfig(
            width=2.0, thickness=0.02, sigma_yield=3.55e8, hinge_length=0.5)
        scenario = CollisionScenario(
            mass=1.0e5,
            velocity=velocity,
            friction_coeff=0.1,
            normal_force=5.0e6,
            hull_config=cfg,
        )
        result = run_collision_scenario(scenario)
        check = energy_conservation_check(result)
        assert check["passed"], (
            f"Energy conservation failed at V={velocity} m/s: "
            f"residual_fraction={check['residual_fraction']:.6e} > 0.02"
        )

    def test_energy_balance_exact(self) -> None:
        """Verify E_k ≈ E_plastic + E_friction within tolerance."""
        cfg = PlasticHingeConfig()
        scenario = CollisionScenario(
            mass=5.0e4, velocity=3.0, friction_coeff=0.1,
            normal_force=3.0e6, hull_config=cfg)
        result = run_collision_scenario(scenario)
        E_total = result.plastic_energy + result.friction_energy
        assert abs(result.kinetic_energy - E_total) / result.kinetic_energy <= 0.02


# ---------------------------------------------------------------------------
# (b) Analytical anchor: three-point bending plastic hinge ≤ 1%
# ---------------------------------------------------------------------------
class TestAnalyticalAnchor:
    """Criterion (b): rectangular beam three-point bending, numerical vs closed-form ≤ 1%."""

    def test_three_point_bending_error_within_1_percent(self) -> None:
        anchor = analytical_anchor_three_point_bending(
            load=1.0e6,
            span=2.0,
            width=0.5,
            thickness=0.02,
            sigma_yield=3.55e8,
        )
        assert anchor["relative_error"] <= 0.01, (
            f"Analytical anchor error {anchor['relative_error']:.6e} > 1%"
        )

    def test_closed_form_equals_numerical(self) -> None:
        """For perfectly plastic material, closed form and numerical must match."""
        anchor = analytical_anchor_three_point_bending(
            load=5.0e5, span=1.0, width=0.3, thickness=0.015,
            sigma_yield=2.35e8)
        rel = anchor["relative_error"]
        assert rel < 1.0e-10, (
            f"Closed-form and numerical differ by {rel:.6e} "
            f"(E_pl_closed={anchor['E_pl_closed_form']:.6f}, "
            f"E_pl_numerical={anchor['E_pl_numerical']:.6f})"
        )

    def test_plastic_moment_correct(self) -> None:
        """M_p = σ_yield · b · h² / 4."""
        Mp = compute_plastic_moment(3.55e8, 1.0, 0.02)
        expected = 3.55e8 * 1.0 * 0.02 ** 2 / 4.0
        assert abs(Mp - expected) < 1.0, f"Mp={Mp}, expected={expected}"

    def test_plastic_section_modulus(self) -> None:
        """Z_p = b · h² / 4."""
        zp = compute_plastic_section_modulus(0.5, 0.01)
        expected = 0.5 * 0.01 ** 2 / 4.0
        assert abs(zp - expected) < 1.0e-15


# ---------------------------------------------------------------------------
# (c) Negative control: V = 0 → d = 0
# ---------------------------------------------------------------------------
class TestNegativeControl:
    """Criterion (c): V = 0 → d = 0, else FAIL."""

    def test_zero_velocity_zero_penetration(self) -> None:
        result = negative_control()
        assert result.penetration_depth == 0.0, (
            f"Negative control FAILED: V=0 but d={result.penetration_depth:.6e} m"
        )
        assert result.kinetic_energy == 0.0
        assert result.plastic_energy == 0.0
        assert result.friction_energy == 0.0

    def test_zero_velocity_scenario(self) -> None:
        scenario = CollisionScenario(
            mass=1.0e6, velocity=0.0, friction_coeff=0.1,
            normal_force=1.0e7, hull_config=PlasticHingeConfig())
        result = run_collision_scenario(scenario)
        assert result.penetration_depth == 0.0


# ---------------------------------------------------------------------------
# (d) Double hull: E_outer + E_inner = E_total ≤ 2%, inner hull intact
# ---------------------------------------------------------------------------
class TestDoubleHull:
    """Criterion (d): double-hull energy distribution."""

    def test_energy_sum_within_2_percent(self) -> None:
        outer_cfg = PlasticHingeConfig(width=2.0, thickness=0.02, sigma_yield=3.55e8, hinge_length=0.5)
        inner_cfg = PlasticHingeConfig(width=2.0, thickness=0.015, sigma_yield=3.55e8, hinge_length=0.5)
        dh_cfg = DoubleHullConfig(outer_config=outer_cfg, inner_config=inner_cfg, gap=0.8)

        scenario = CollisionScenario(
            mass=1.0e5, velocity=4.0, friction_coeff=0.1,
            normal_force=5.0e6, hull_config=outer_cfg, double_hull=dh_cfg)
        result = run_collision_scenario(scenario)

        assert result.inner_hull_intact is not None, "Double hull analysis not performed"
        if result.inner_hull_intact:
            assert result.inner_energy == 0.0, "Inner energy should be 0 when intact"
            assert result.outer_energy is not None
            # Total absorbed = outer energy
            total = result.outer_energy
            assert abs(result.kinetic_energy - total) / result.kinetic_energy <= 0.02

    def test_inner_hull_intact_declaration(self) -> None:
        """Low energy impact: inner hull should be declared intact."""
        outer_cfg = PlasticHingeConfig(width=2.0, thickness=0.02, sigma_yield=3.55e8, hinge_length=0.5)
        inner_cfg = PlasticHingeConfig(width=2.0, thickness=0.015, sigma_yield=3.55e8, hinge_length=0.5)
        dh_cfg = DoubleHullConfig(outer_config=outer_cfg, inner_config=inner_cfg, gap=0.8)

        scenario = CollisionScenario(
            mass=1.0e4, velocity=1.0, friction_coeff=0.1,
            normal_force=1.0e6, hull_config=outer_cfg, double_hull=dh_cfg)
        result = run_collision_scenario(scenario)

        assert result.inner_hull_intact is True, (
            f"Inner hull should be intact at low energy, got {result.inner_hull_intact}"
        )

    def test_double_hull_energy_balance(self) -> None:
        """E_outer + E_inner should equal total absorbed energy."""
        outer_cfg = PlasticHingeConfig(width=2.0, thickness=0.02, sigma_yield=3.55e8, hinge_length=0.5)
        inner_cfg = PlasticHingeConfig(width=2.0, thickness=0.015, sigma_yield=3.55e8, hinge_length=0.5)
        dh_cfg = DoubleHullConfig(outer_config=outer_cfg, inner_config=inner_cfg, gap=0.8)

        dh_result = compute_double_hull_absorption(
            kinetic_energy=8.0e5,
            outer_config=outer_cfg,
            inner_config=inner_cfg,
            friction_coeff=0.1,
            normal_force=5.0e6,
        )
        total = dh_result["outer_energy"] + dh_result["inner_energy"]
        assert abs(total - dh_result["total_absorbed"]) / dh_result["total_absorbed"] <= 0.02


# ---------------------------------------------------------------------------
# (e) Minorsky empirical: relative error ≤ 30%
# ---------------------------------------------------------------------------
class TestMinorskyEmpirical:
    """Criterion (e): Minorsky formula cross-validation, relative error ≤ 30%."""

    def test_minorsky_volume_formula(self) -> None:
        """V_damage = C · M · V² / σ_yield."""
        vol = compute_minorsky_damage_volume(1.0e5, 4.0, 3.55e8, C=0.5)
        expected = 0.5 * 1.0e5 * 4.0 ** 2 / 3.55e8
        assert abs(vol - expected) < 1.0e-15

    def test_minorsky_relative_error_within_30_percent(self) -> None:
        """Energy-equivalent damage volume vs Minorsky should be within 30%.

        The energy-equivalent volume V_eq = E_absorbed / σ_yield should match
        V_minorsky = C·M·V²/σ_yield with C=0.5, since E_absorbed ≈ E_k = ½MV².
        """
        cfg = PlasticHingeConfig(width=2.0, thickness=0.02, sigma_yield=3.55e8, hinge_length=0.5)
        scenario = CollisionScenario(
            mass=1.0e5, velocity=4.0, friction_coeff=0.1,
            normal_force=5.0e6, hull_config=cfg, contact_width=2.0)
        result = run_collision_scenario(scenario)

        assert result.minorsky_volume is not None, "Minorsky volume not computed"
        assert result.minorsky_relative_error is not None, "Minorsky error not computed"
        assert result.minorsky_relative_error <= 0.30, (
            f"Minorsky relative error {result.minorsky_relative_error:.6e} > 30%"
        )

    def test_minorsky_zero_velocity(self) -> None:
        """Minorsky volume should be 0 at V=0."""
        vol = compute_minorsky_damage_volume(1.0e5, 0.0, 3.55e8)
        assert vol == 0.0


# ---------------------------------------------------------------------------
# (f) Parameter sweep: V = 2/4/6/8 m/s, d monotonically increasing
# ---------------------------------------------------------------------------
class TestParameterSweep:
    """Criterion (f): penetration depth monotonically increasing with velocity."""

    def test_monotonic_penetration_depth(self) -> None:
        velocities = [2.0, 4.0, 6.0, 8.0]
        results = run_parameter_sweep(
            velocities=velocities,
            mass=1.0e5,
            hull_config=PlasticHingeConfig(
                width=2.0, thickness=0.02, sigma_yield=3.55e8, hinge_length=0.5),
            friction_coeff=0.1,
            normal_force=5.0e6,
            contact_width=2.0,
        )

        depths = [r.penetration_depth for r in results]
        for i in range(1, len(depths)):
            assert depths[i] > depths[i - 1], (
                f"Penetration depth not monotonic: d(V={velocities[i-1]})={depths[i-1]:.6f} "
                f">= d(V={velocities[i]})={depths[i]:.6f}"
            )

    def test_sweep_energy_conservation(self) -> None:
        """All sweep points must satisfy energy conservation."""
        velocities = [2.0, 4.0, 6.0, 8.0]
        results = run_parameter_sweep(
            velocities=velocities,
            mass=1.0e5,
            hull_config=PlasticHingeConfig(),
            friction_coeff=0.1,
            normal_force=5.0e6,
        )
        for v, r in zip(velocities, results):
            check = energy_conservation_check(r)
            assert check["passed"], f"Energy conservation failed at V={v}: {check}"


# ---------------------------------------------------------------------------
# Additional unit tests
# ---------------------------------------------------------------------------
class TestPlasticHingeConfig:
    def test_plastic_section_modulus_property(self) -> None:
        cfg = PlasticHingeConfig(width=1.0, thickness=0.02)
        expected = 1.0 * 0.02 ** 2 / 4.0
        assert abs(cfg.plastic_section_modulus - expected) < 1.0e-15

    def test_plastic_moment_property(self) -> None:
        cfg = PlasticHingeConfig(width=1.0, thickness=0.02, sigma_yield=3.55e8)
        expected = 3.55e8 * 1.0 * 0.02 ** 2 / 4.0
        assert abs(cfg.plastic_moment - expected) < 1.0

    def test_invalid_thickness(self) -> None:
        with pytest.raises(ValueError):
            PlasticHingeConfig(thickness=-0.01)

    def test_invalid_sigma_yield(self) -> None:
        with pytest.raises(ValueError):
            PlasticHingeConfig(sigma_yield=0)


class TestCollisionScenario:
    def test_invalid_velocity(self) -> None:
        with pytest.raises(ValueError):
            CollisionScenario(velocity=-1.0)

    def test_valid_zero_velocity(self) -> None:
        s = CollisionScenario(velocity=0.0)
        assert s.velocity == 0.0


class TestDamageGeometry:
    def test_zero_penetration(self) -> None:
        geom = compute_damage_geometry(0.0, 2.0, 0.02)
        assert geom["L_damage"] == 0.0
        assert geom["W_damage"] == 2.0
        assert geom["d_damage"] == 0.0
        assert geom["V_damage"] == 0.0

    def test_nonzero_penetration(self) -> None:
        geom = compute_damage_geometry(0.1, 2.0, 0.02)
        assert geom["L_damage"] > 0
        assert geom["W_damage"] == 2.0
        assert geom["d_damage"] == 0.1
        assert geom["V_damage"] > 0


class TestFrictionEnergy:
    def test_zero_penetration(self) -> None:
        E = compute_friction_energy(0.1, 5.0e6, 0.0)
        assert E == 0.0

    def test_linear_with_depth(self) -> None:
        E1 = compute_friction_energy(0.1, 5.0e6, 0.01)
        E2 = compute_friction_energy(0.1, 5.0e6, 0.02)
        assert abs(E2 - 2 * E1) < 1.0


class TestPlasticHingeEnergy:
    def test_zero_penetration(self) -> None:
        E, theta = compute_plastic_hinge_energy(1.0e5, 0.0, 0.5)
        assert E == 0.0
        assert theta == 0.0

    def test_increasing_with_depth(self) -> None:
        E1, _ = compute_plastic_hinge_energy(1.0e5, 0.01, 0.5)
        E2, _ = compute_plastic_hinge_energy(1.0e5, 0.02, 0.5)
        assert E2 > E1
