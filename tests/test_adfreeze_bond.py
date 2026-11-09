"""Tests for adfreeze bond model: cyclic degradation, S-N curve, energy balance.

Verification matrix (P82W4-N01):
  (a) Single-bond S-N: COV ≤ 15% at τ/τ_adh ∈ {0.5, 0.7, 0.9}
  (b) Parameter identification: β, N₀ error ≤ 10%
  (c) Detachment criterion: D_crit=0.95 prediction vs discrete events ≤ 5%
  (d) Negative control: τ/τ_adh < 0.3 → no degradation at 1e6 cycles
  (e) Energy balance: closure ratio ∈ [0.97, 1.03]
"""
from __future__ import annotations

import json
import math
import os
import sys
import numpy as np
import pytest

from tensordem.adfreeze_bond import (
    AdfreezeBondModel,
    AdfreezeBondParams,
    AdfreezeBondState,
    CyclicShearDriver,
    compute_sn_curve,
    identify_degradation_params,
    energy_balance_check,
    run_single_bond_verification,
    run_interface_verification,
    run_waterline_verification,
)


# ---------------------------------------------------------------------------
# Basic constitutive tests
# ---------------------------------------------------------------------------

class TestAdfreezeBondConstitutive:
    """Test the basic elastic-brittle constitutive law."""

    def test_params_derived_quantities(self):
        p = AdfreezeBondParams()
        assert p.F_adh == pytest.approx(p.tau_adh * p.A_bond)
        assert p.delta_y == pytest.approx(p.F_adh / p.k_adh)
        assert p.delta_crit == pytest.approx(p.delta_crit_factor * p.delta_y)
        assert p.G_adh_per_bond == pytest.approx(p.G_adh * p.A_bond)

    def test_elastic_force_undamaged(self):
        """Undamaged bond: F = k_adh * delta."""
        model = AdfreezeBondModel()
        state = AdfreezeBondState()
        delta = 1e-5  # 10 μm
        F = model.compute_force(state, delta)
        assert F == pytest.approx(model.params.k_adh * delta)

    def test_stiffness_degrades_with_damage(self):
        """Effective stiffness decreases with damage."""
        model = AdfreezeBondModel()
        p = model.params
        state = AdfreezeBondState(damage=0.5)
        k_eff = model.effective_stiffness(state)
        # k_eff = k_adh * ((1-0.5) + eta*0.5) = k_adh * (0.5 + 0.005)
        expected = p.k_adh * (0.5 + p.eta * 0.5)
        assert k_eff == pytest.approx(expected)

    def test_broken_bond_friction_only(self):
        """Broken bond returns friction-only force."""
        model = AdfreezeBondModel()
        state = AdfreezeBondState(broken=True, damage=1.0)
        delta = 1e-5
        F = model.compute_force(state, delta)
        # After full damage, stiffness is eta*k_adh, friction applies
        k_res = model.params.eta * model.params.k_adh
        F_expected = -model.params.mu_friction * k_res * abs(delta)
        assert F == pytest.approx(F_expected)

    def test_force_zero_at_zero_displacement(self):
        """Zero displacement gives zero force."""
        model = AdfreezeBondModel()
        state = AdfreezeBondState()
        assert model.compute_force(state, 0.0) == 0.0

    def test_force_linear_in_displacement(self):
        """Force is linear in displacement for undamaged bond."""
        model = AdfreezeBondModel()
        state = AdfreezeBondState()
        F1 = model.compute_force(state, 1e-5)
        F2 = model.compute_force(state, 2e-5)
        assert F2 == pytest.approx(2 * F1)


# ---------------------------------------------------------------------------
# Degradation law tests
# ---------------------------------------------------------------------------

class TestDegradationLaw:
    """Test the D(N) = 1 - exp(-(N/N₀)^β) law."""

    def test_damage_monotone(self):
        """Damage increases monotonically with cycles."""
        model = AdfreezeBondModel()
        r = 0.7
        D_prev = 0.0
        for N in [1, 10, 100, 500, 1000]:
            D = model.damage_at_cycle(N, r)
            assert D >= D_prev
            D_prev = D

    def test_damage_bounded(self):
        """Damage is bounded in [0, 1]."""
        model = AdfreezeBondModel()
        for r in [0.4, 0.7, 0.9, 1.0]:
            for N in [1, 100, 10000, 1000000]:
                D = model.damage_at_cycle(N, r)
                assert 0.0 <= D <= 1.0

    def test_damage_zero_at_N0(self):
        """D(0) = 0."""
        model = AdfreezeBondModel()
        assert model.damage_at_cycle(0, 0.7) == 0.0

    def test_fatigue_limit(self):
        """Below fatigue limit, damage is always zero."""
        model = AdfreezeBondModel()
        for r in [0.0, 0.1, 0.2, 0.29]:
            assert model.damage_at_cycle(1000000, r) == 0.0
            assert model.cycles_to_failure(r) is None

    def test_cycles_to_failure_decreases_with_stress(self):
        """N_f decreases as stress ratio increases."""
        model = AdfreezeBondModel()
        N_f_prev = float('inf')
        for r in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            N_f = model.cycles_to_failure(r)
            assert N_f is not None
            assert N_f < N_f_prev
            N_f_prev = N_f

    def test_N0_at_stress_above_fatigue(self):
        """N₀ is finite above fatigue limit."""
        model = AdfreezeBondModel()
        N0 = model.N0_at_stress(0.7)
        assert N0 > 0
        assert N0 < float('inf')

    def test_N0_at_stress_below_fatigue(self):
        """N₀ is infinite below fatigue limit."""
        model = AdfreezeBondModel()
        N0 = model.N0_at_stress(0.2)
        assert N0 == float('inf')

    def test_damage_at_Nf_equals_Dcrit(self):
        """D(N_f) should equal D_crit by construction."""
        model = AdfreezeBondModel()
        for r in [0.5, 0.7, 0.9]:
            N_f = model.cycles_to_failure(r)
            D_at_Nf = model.damage_at_cycle(N_f, r)
            assert D_at_Nf == pytest.approx(model.params.D_crit, abs=0.02)


# ---------------------------------------------------------------------------
# Criterion (a): S-N curve COV ≤ 15%
# ---------------------------------------------------------------------------

class TestCriterionA_SN_Curve:
    """S-N curve: COV ≤ 15% at each stress ratio."""

    def test_sn_curve_cov(self):
        sn = compute_sn_curve(
            stress_ratios=[0.5, 0.7, 0.9],
            n_repeats=5,
        )
        for r, cov in sn["cov"].items():
            assert cov <= 0.15, f"COV at r={r} is {cov:.3f} > 0.15"

    def test_sn_curve_analytical_consistency(self):
        """Mean N_f should be close to analytical (no scatter) value."""
        sn = compute_sn_curve(
            stress_ratios=[0.5, 0.7, 0.9],
            n_repeats=5,
        )
        for r in [0.5, 0.7, 0.9]:
            analytical = sn["analytical"][r]
            # Mean of scattered values should be within 10% of analytical
            samples = [p.N_f for p in sn["points"] if p.stress_ratio == r]
            mean_Nf = np.mean(samples)
            rel_error = abs(mean_Nf - analytical) / analytical
            assert rel_error <= 0.10, (
                f"Mean N_f at r={r}: {mean_Nf:.0f} vs analytical {analytical:.0f}, "
                f"error={rel_error:.3f}"
            )


# ---------------------------------------------------------------------------
# Criterion (b): Parameter identification error ≤ 10%
# ---------------------------------------------------------------------------

class TestCriterionB_ParamIdentification:
    """Identify β and N₀ from synthetic hysteresis data."""

    def test_beta_identification(self):
        result = identify_degradation_params(
            stress_ratio=0.7,
            n_cycles=500,
            true_beta=1.5,
            true_N_ref=1000.0,
            noise_level=0.02,
        )
        assert result["beta_error"] <= 0.10, (
            f"β error: {result['beta_error']:.3f} > 0.10 "
            f"(true={result['true_beta']:.2f}, "
            f"identified={result['beta_identified']:.2f})"
        )

    def test_N0_identification(self):
        result = identify_degradation_params(
            stress_ratio=0.7,
            n_cycles=500,
            true_beta=1.5,
            true_N_ref=1000.0,
            noise_level=0.02,
        )
        assert result["N0_error"] <= 0.10, (
            f"N₀ error: {result['N0_error']:.3f} > 0.10 "
            f"(true={result['true_N0']:.1f}, "
            f"identified={result['N0_identified']:.1f})"
        )

    def test_identification_different_beta(self):
        """Test with a different true β value."""
        result = identify_degradation_params(
            stress_ratio=0.8,
            n_cycles=300,
            true_beta=2.0,
            true_N_ref=500.0,
            noise_level=0.01,
        )
        assert result["beta_error"] <= 0.15, (
            f"β error: {result['beta_error']:.3f} > 0.15"
        )
        assert result["N0_error"] <= 0.15, (
            f"N₀ error: {result['N0_error']:.3f} > 0.15"
        )


# ---------------------------------------------------------------------------
# Criterion (c): Detachment criterion self-consistency
# ---------------------------------------------------------------------------

class TestCriterionC_Detachment:
    """D_crit=0.95 prediction matches discrete failure events."""

    def test_interface_single_stress(self):
        result = run_interface_verification(
            n_bonds=20,
            stress_ratio=0.7,
            n_cycles_max=10000,
        )
        assert result["criterion_c_pass"], (
            f"Detachment prediction error: {result['criterion_c_error']:.3f} > 0.05 "
            f"(predicted={result['N_f_predicted']}, "
            f"actual_mean={result['N_f_actual_mean']:.0f})"
        )

    def test_interface_multiple_stress(self):
        """Test at multiple stress ratios."""
        for r in [0.6, 0.8]:
            result = run_interface_verification(
                n_bonds=15,
                stress_ratio=r,
                n_cycles_max=20000,
            )
            assert result["criterion_c_error"] <= 0.05, (
                f"At r={r}: error={result['criterion_c_error']:.3f}"
            )


# ---------------------------------------------------------------------------
# Criterion (d): Negative control - fatigue limit
# ---------------------------------------------------------------------------

class TestCriterionD_FatigueLimit:
    """τ/τ_adh < 0.3 → no degradation at 1e6 cycles."""

    def test_no_damage_below_fatigue(self):
        model = AdfreezeBondModel()
        D = model.damage_at_cycle(1_000_000, 0.2)
        assert D == 0.0

    def test_no_failure_below_fatigue(self):
        model = AdfreezeBondModel()
        N_f = model.cycles_to_failure(0.2)
        assert N_f is None

    def test_single_bond_negative_control(self):
        result = run_single_bond_verification()
        nc = result["negative_control"]
        assert nc["criterion_d_pass"], (
            f"Negative control failed: N_f={nc['N_f']}, "
            f"D_after_1e6={nc['D_after_1e6_cycles']}"
        )

    def test_cyclic_driver_no_failure(self):
        """Driver should not report failure below fatigue limit."""
        driver = CyclicShearDriver()
        result = driver.run(stress_ratio=0.2, n_cycles_max=1000)
        assert result.n_cycles_to_failure is None

    def test_advance_cycle_no_damage_below_fatigue(self):
        """advance_cycle should not increase damage below fatigue limit."""
        model = AdfreezeBondModel()
        state = AdfreezeBondState()
        p = model.params
        delta_amp = 0.2 * p.F_adh / p.k_adh
        for _ in range(10000):
            dD = model.advance_cycle(state, 0.2, delta_amp)
            assert dD == 0.0
            assert state.damage == 0.0
            assert not state.broken


# ---------------------------------------------------------------------------
# Criterion (e): Energy balance
# ---------------------------------------------------------------------------

class TestCriterionE_EnergyBalance:
    """Energy closure ratio ∈ [0.97, 1.03].
    
    For the cycle-wise damage model, the energy balance is:
    - Mechanical energy released by stiffness degradation = fracture energy
    - E_released = ½ (k_initial - k_final) δ₀²
    - E_fracture = G_adh * A_bond * D_final
    
    The closure ratio = E_fracture / E_released should be ≈ 1.
    
    Note: This is a modeling choice - we prescribe G_adh as the energy
    per unit damage, and the mechanical energy release follows from the
    stiffness degradation. The model is thermodynamically consistent if
    G_adh is calibrated to match the physical fracture process.
    """

    def test_energy_balance_moderate_stress(self):
        result = energy_balance_check(stress_ratio=0.7, n_cycles=50)
        assert result["in_balance_band"], (
            f"Energy closure ratio: {result['closure_ratio']:.4f} "
            f"not in [0.97, 1.03]"
        )

    def test_energy_balance_low_stress(self):
        result = energy_balance_check(stress_ratio=0.5, n_cycles=100)
        assert result["in_balance_band"], (
            f"Energy closure ratio: {result['closure_ratio']:.4f} "
            f"not in [0.97, 1.03]"
        )

    def test_energy_components_nonnegative(self):
        result = energy_balance_check(stress_ratio=0.7, n_cycles=50)
        assert result["energy_fracture_J"] >= 0
        assert result["energy_friction_J"] >= 0
        assert result["energy_elastic_J"] >= 0

    def test_fracture_energy_matches_damage(self):
        """Fracture energy should equal G_adh * A_bond * D_final."""
        p = AdfreezeBondParams()
        model = AdfreezeBondModel(p)
        state = AdfreezeBondState()
        r = 0.7
        delta_amp = r * p.F_adh / p.k_adh
        
        for _ in range(100):
            model.advance_cycle(state, r, delta_amp)
        
        expected_fracture = p.G_adh_per_bond * state.damage
        assert state.energy_fracture == pytest.approx(expected_fracture, rel=1e-10)


# ---------------------------------------------------------------------------
# Multi-scale verification
# ---------------------------------------------------------------------------

class TestMultiScaleVerification:
    """Run all three verification levels."""

    def test_single_bond(self):
        result = run_single_bond_verification()
        assert result["criterion_a_pass"]
        assert result["negative_control"]["criterion_d_pass"]

    def test_interface(self):
        result = run_interface_verification()
        assert result["criterion_c_pass"]

    def test_waterline(self):
        result = run_waterline_verification()
        assert len(result["patches"]) == 5
        # All patches should show progressive failure
        for patch in result["patches"]:
            if patch["stress_ratio"] > 0.3:
                assert patch["first_failure"] is not None, (
                    f"Patch at r={patch['stress_ratio']} should fail"
                )


# ---------------------------------------------------------------------------
# Full verification report
# ---------------------------------------------------------------------------

class TestFullVerificationReport:
    """Generate and validate the full verification report."""

    def test_full_report(self, tmp_path):
        """Run all verifications and save report JSON."""
        np.random.seed(42)
        
        report = {
            "mission": "P82W4-N01",
            "title": "Adfreeze bond cyclic degradation verification",
            "criteria": {},
        }
        
        # (a) S-N curve
        sn = run_single_bond_verification()
        report["criteria"]["a_sn_curve"] = {
            "pass": sn["criterion_a_pass"],
            "mean_cov": sn["sn_curve"]["mean_cov"],
            "cov_per_ratio": sn["sn_curve"]["cov"],
            "threshold": 0.15,
        }
        
        # (b) Parameter identification
        ident = identify_degradation_params()
        report["criteria"]["b_param_identification"] = {
            "pass": ident["beta_error"] <= 0.10 and ident["N0_error"] <= 0.10,
            "beta_error": ident["beta_error"],
            "N0_error": ident["N0_error"],
            "threshold": 0.10,
        }
        
        # (c) Detachment criterion
        det = run_interface_verification()
        report["criteria"]["c_detachment"] = {
            "pass": det["criterion_c_pass"],
            "error": det["criterion_c_error"],
            "N_f_predicted": det["N_f_predicted"],
            "N_f_actual_mean": det["N_f_actual_mean"],
            "threshold": 0.05,
        }
        
        # (d) Negative control
        report["criteria"]["d_fatigue_limit"] = {
            "pass": sn["negative_control"]["criterion_d_pass"],
            "N_f_at_0.2": sn["negative_control"]["N_f"],
            "D_at_1e6": sn["negative_control"]["D_after_1e6_cycles"],
        }
        
        # (e) Energy balance
        eb = energy_balance_check(stress_ratio=0.7, n_cycles=50)
        report["criteria"]["e_energy_balance"] = {
            "pass": eb["in_balance_band"],
            "closure_ratio": eb["closure_ratio"],
            "energy_fracture_J": eb["energy_fracture_J"],
            "threshold": [0.97, 1.03],
        }
        
        # Overall pass
        all_pass = all(
            report["criteria"][k]["pass"]
            for k in report["criteria"]
        )
        report["overall_pass"] = all_pass
        
        # Save
        report_path = tmp_path / "verification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        assert all_pass, (
            f"Some criteria failed: "
            + ", ".join(
                k for k, v in report["criteria"].items()
                if not v["pass"]
            )
        )
