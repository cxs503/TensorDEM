"""Particle Impact Damper (PID) — test suite.

Tests cover:
- Single-particle energy ledger accuracy and conservation
- Analytic energy loss formulas
- Multi-particle model physics (filling ratio, Γ, frequency, restitution)
- Damping ratio identification (energy method + half-power bandwidth)
- Parameter sweep and surface export
- Acceptance criteria from campaign IN32-D02
"""
from __future__ import annotations

import math
import unittest

from tensordem.pid import (
    CollisionEvent,
    SingleParticleEnergyLedger,
    MultiParticlePIDModel,
    DampingRatioResult,
    compute_zeta_energy,
    compute_zeta_half_power,
    identify_damping_ratio_from_ledger,
    identify_damping_ratio_from_model,
    SweepConfig,
    run_sweep,
    ZetaSurface,
    single_particle_energy_loss_per_collision,
    single_particle_zeta_analytic,
    FeasibilityResult,
    assess_feasibility,
    _filling_ratio_efficiency,
    _gamma_saturation,
    _frequency_tuning,
    _restitution_factor,
)


# ===================================================================
# Acceptance criterion (a): Single-particle energy ledger residual < 1%
# ===================================================================

class SingleParticleEnergyLedgerTests(unittest.TestCase):
    """Single-particle energy ledger accuracy and conservation."""

    def test_energy_balance_residual_below_1pct(self):
        """Criterion (a): energy balance residual < 1%."""
        ledger = SingleParticleEnergyLedger(
            mass=0.001, e_target=0.5, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, dt=1e-4, max_steps=200000,
        )
        ledger.run()
        self.assertLess(
            ledger.energy_balance_residual, 0.01,
            f"Energy balance residual {ledger.energy_balance_residual:.2e} exceeds 1%",
        )

    def test_energy_balance_residual_near_machine_epsilon(self):
        """Residual should be at machine precision level (ledger is exact)."""
        ledger = SingleParticleEnergyLedger(
            mass=0.001, e_target=0.7, cavity_length=0.1,
            amplitude=0.015, frequency=8.0, dt=1e-4, max_steps=150000,
        )
        ledger.run()
        self.assertLess(
            ledger.energy_balance_residual, 1e-10,
            f"Residual {ledger.energy_balance_residual:.2e} exceeds machine precision",
        )

    def test_collisions_recorded(self):
        """Ledger must record collision events."""
        ledger = SingleParticleEnergyLedger(
            mass=0.001, e_target=0.5, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, dt=1e-4, max_steps=200000,
        )
        ledger.run()
        self.assertGreater(ledger.n_collisions, 0, "No collisions recorded")
        self.assertGreater(len(ledger.collisions), 0)

    def test_collision_event_fields(self):
        """Each collision event must have valid fields."""
        ledger = SingleParticleEnergyLedger(
            mass=0.001, e_target=0.5, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, dt=1e-4, max_steps=200000,
        )
        ledger.run()
        evt = ledger.collisions[0]
        self.assertGreater(evt.v_in, 0)
        self.assertGreaterEqual(evt.v_out, 0)
        self.assertGreaterEqual(evt.e_measured, 0)
        self.assertGreaterEqual(evt.delta_E_J, 0)
        self.assertGreaterEqual(evt.time_s, 0)

    def test_measured_restitution_matches_target(self):
        """Measured restitution should match target within numerical tolerance."""
        ledger = SingleParticleEnergyLedger(
            mass=0.001, e_target=0.6, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, dt=1e-4, max_steps=200000,
        )
        ledger.run()
        # Average measured e across collisions
        e_measured_avg = sum(c.e_measured for c in ledger.collisions) / len(ledger.collisions)
        self.assertAlmostEqual(
            e_measured_avg, 0.6, delta=0.05,
            msg=f"Average measured e={e_measured_avg:.4f} differs from target 0.6",
        )

# ===================================================================
# Acceptance criterion (b): Dual-method ζ_eq cross-validation < 5%
# ===================================================================

class DampingRatioIdentificationTests(unittest.TestCase):
    """Equivalent damping ratio identification via two methods."""

    def test_energy_method_formula(self):
        """ζ_eq = ΔE_cycle / (4π E_max) must be correctly computed."""
        delta_E = 0.1
        E_max = 1.0
        zeta = compute_zeta_energy(delta_E, E_max)
        expected = 0.1 / (4.0 * math.pi * 1.0)
        self.assertAlmostEqual(zeta, expected, places=12)

    def test_energy_method_zero_E_max(self):
        """ζ should be 0 when E_max is zero."""
        self.assertEqual(compute_zeta_energy(0.1, 0.0), 0.0)

    def test_half_power_bandwidth_formula(self):
        """ζ_eq = Δf / (2 f_n) must be correctly computed."""
        # Construct a frequency response with clear -3dB crossings
        # -3dB threshold = 1/sqrt(2) ≈ 0.707
        freq_response = {
            4.0: 0.5,
            4.4: 0.65,
            4.5: 0.75,   # above threshold
            5.0: 1.0,    # peak
            5.5: 0.75,   # above threshold
            5.6: 0.65,
            6.0: 0.5,
        }
        f_n = 5.0
        zeta = compute_zeta_half_power(freq_response, f_n)
        # f_low between 4.4 and 4.5, f_high between 5.5 and 5.6
        # Δf ≈ 1.0, ζ ≈ 0.1
        self.assertGreater(zeta, 0.05, "ζ should be positive")
        self.assertLess(zeta, 0.15, "ζ should be reasonable")

    def test_dual_method_consistency(self):
        """Criterion (b): energy and bandwidth methods agree within 5%."""
        model = MultiParticlePIDModel(
            n_particles=20, mass=0.01, primary_mass=1.0,
            primary_stiffness=1000.0, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, e=0.5,
        )
        result = identify_damping_ratio_from_model(model, 0.3, 2.0, 1.0, 0.5)
        # For single-frequency case, both methods should give the same result
        self.assertAlmostEqual(
            result.zeta_energy, result.zeta_bandwidth,
            places=10,
            msg="Energy and bandwidth methods should agree for single-frequency case",
        )
        self.assertLess(
            result.zeta_discrepancy_pct, 5.0,
            f"Discrepancy {result.zeta_discrepancy_pct:.1f}% exceeds 5%",
        )


# ===================================================================
# Acceptance criterion (c): Optimal filling ratio matches literature
# ===================================================================

class FillingRatioDependenceTests(unittest.TestCase):
    """ζ_eq(φ) must rise then fall with optimum near φ ≈ 0.2–0.4."""

    def test_optimal_filling_ratio_in_literature_range(self):
        """Criterion (c): optimal φ in [0.2, 0.4]."""
        model = MultiParticlePIDModel(
            n_particles=20, mass=0.01, primary_mass=1.0,
            primary_stiffness=1000.0, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, e=0.5,
        )
        zetas = {}
        for phi in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            zetas[phi] = model.compute_zeta(phi, 2.0, 1.0, 0.5)

        optimal_phi = max(zetas, key=zetas.get)
        self.assertGreaterEqual(
            optimal_phi, 0.2,
            f"Optimal φ={optimal_phi} below literature range [0.2, 0.4]",
        )
        self.assertLessEqual(
            optimal_phi, 0.4,
            f"Optimal φ={optimal_phi} above literature range [0.2, 0.4]",
        )

    def test_zeta_rises_then_falls(self):
        """ζ(φ) must be non-monotonic: rise then fall."""
        model = MultiParticlePIDModel(
            n_particles=20, mass=0.01, primary_mass=1.0,
            primary_stiffness=1000.0, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, e=0.5,
        )
        zetas = [model.compute_zeta(phi, 2.0, 1.0, 0.5)
                 for phi in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        # Must increase then decrease
        self.assertLess(zetas[0], zetas[2])  # ζ(0.1) < ζ(0.3)
        self.assertGreater(zetas[2], zetas[5])  # ζ(0.3) > ζ(0.6)

    def test_filling_ratio_efficiency_function(self):
        """_filling_ratio_efficiency must peak at phi_opt."""
        self.assertAlmostEqual(_filling_ratio_efficiency(0.0), 0.0)
        self.assertAlmostEqual(_filling_ratio_efficiency(0.3, phi_opt=0.3), 1.0)
        self.assertGreater(_filling_ratio_efficiency(0.3), _filling_ratio_efficiency(0.6))


# ===================================================================
# Acceptance criterion (d): N=1, e=1 → ζ_eq < 0.1%
# ===================================================================

class PerfectRestitutionTests(unittest.TestCase):
    """With e=1 (perfectly elastic), damping should be negligible."""

    def test_perfect_restitution_near_zero_damping(self):
        """Criterion (d): N=1, e=1 → ζ < 0.001."""
        model = MultiParticlePIDModel(
            n_particles=1, mass=0.01, primary_mass=1.0,
            primary_stiffness=1000.0, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, e=1.0,
        )
        zeta = model.compute_zeta(0.1, 2.0, 1.0, 1.0)
        self.assertLess(
            zeta, 0.001,
            f"ζ={zeta:.4e} exceeds 0.1% for e=1",
        )

    def test_restitution_factor_zero_at_e1(self):
        """_restitution_factor(1.0) must be exactly 0."""
        self.assertEqual(_restitution_factor(1.0), 0.0)

    def test_restitution_factor_positive_below_e1(self):
        """_restitution_factor(e) > 0 for e < 1."""
        for e in [0.3, 0.5, 0.6, 0.9]:
            self.assertGreater(
                _restitution_factor(e), 0.0,
                f"_restitution_factor({e}) should be positive",
            )


# ===================================================================
# Analytic energy loss tests
# ===================================================================

class AnalyticEnergyLossTests(unittest.TestCase):
    """Closed-form energy loss formulas."""

    def test_energy_loss_formula(self):
        """ΔE = ½m·v²·(1 - e²) must be correct."""
        dE = single_particle_energy_loss_per_collision(0.5, 2.0, 0.01)
        expected = 0.5 * 0.01 * 4.0 * (1 - 0.25)
        self.assertAlmostEqual(dE, expected, places=15)

    def test_energy_loss_zero_at_perfect_restitution(self):
        """ΔE = 0 when e = 1."""
        dE = single_particle_energy_loss_per_collision(1.0, 2.0, 0.01)
        self.assertEqual(dE, 0.0)

    def test_energy_loss_quadratic_in_velocity(self):
        """ΔE ∝ v²."""
        dE1 = single_particle_energy_loss_per_collision(0.5, 1.0, 0.01)
        dE2 = single_particle_energy_loss_per_collision(0.5, 2.0, 0.01)
        self.assertAlmostEqual(dE2 / dE1, 4.0, places=10)

    def test_analytic_zeta_formula(self):
        """single_particle_zeta_analytic must use correct formula."""
        zeta = single_particle_zeta_analytic(0.5, 2, 1.0, 0.01, 1.0)
        delta_E = 0.5 * 0.01 * 1.0 ** 2 * (1 - 0.25)
        expected = 2 * delta_E / (4.0 * math.pi * 1.0)
        self.assertAlmostEqual(zeta, expected, places=12)


# ===================================================================
# Parameter sweep and surface tests
# ===================================================================

class ParameterSweepTests(unittest.TestCase):
    """Dimensionless parameter sweep and surface export."""

    def test_sweep_runs(self):
        """Sweep must complete without errors."""
        config = SweepConfig(
            gamma_values=[1.0, 2.0],
            freq_ratio_values=[0.8, 1.0, 1.2],
            filling_ratio_values=[0.2, 0.3, 0.4],
            restitution_values=[0.5, 0.7],
            n_particles=10,
        )
        results = run_sweep(config)
        self.assertGreater(len(results), 0)

    def test_surface_construction(self):
        """ZetaSurface must be constructible from sweep results."""
        config = SweepConfig(
            gamma_values=[1.0, 2.0],
            freq_ratio_values=[0.8, 1.0, 1.2],
            filling_ratio_values=[0.2, 0.3, 0.4],
            restitution_values=[0.5, 0.7],
            n_particles=10,
        )
        results = run_sweep(config)
        surface = ZetaSurface.from_results(results, config)
        self.assertEqual(len(surface.gamma_axis), 2)
        self.assertEqual(len(surface.freq_ratio_axis), 3)
        self.assertEqual(len(surface.filling_ratio_axis), 3)
        self.assertEqual(len(surface.restitution_axis), 2)

    def test_surface_serialization(self):
        """ZetaSurface.to_dict must produce valid dict."""
        config = SweepConfig(
            gamma_values=[1.0, 2.0],
            freq_ratio_values=[1.0],
            filling_ratio_values=[0.3],
            restitution_values=[0.5],
            n_particles=10,
        )
        results = run_sweep(config)
        surface = ZetaSurface.from_results(results, config)
        d = surface.to_dict()
        self.assertIn("gamma_axis", d)
        self.assertIn("zeta_values", d)
        self.assertIn("metadata", d)

    def test_gamma_saturation(self):
        """ζ should increase with Γ but saturate."""
        model = MultiParticlePIDModel(
            n_particles=20, mass=0.01, primary_mass=1.0,
            primary_stiffness=1000.0, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, e=0.5,
        )
        z_low = model.compute_zeta(0.3, 0.5, 1.0, 0.5)
        z_high = model.compute_zeta(0.3, 4.0, 1.0, 0.5)
        self.assertGreater(z_high, z_low, "ζ should increase with Γ")

    def test_frequency_tuning_peaks_at_resonance(self):
        """ζ should peak at f/f_n = 1."""
        model = MultiParticlePIDModel(
            n_particles=20, mass=0.01, primary_mass=1.0,
            primary_stiffness=1000.0, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, e=0.5,
        )
        zetas = {fr: model.compute_zeta(0.3, 2.0, fr, 0.5)
                 for fr in [0.6, 0.8, 1.0, 1.2, 1.4]}
        max_fr = max(zetas, key=zetas.get)
        self.assertEqual(max_fr, 1.0, "Peak should be at f/f_n = 1.0")

    def test_restitution_dependence(self):
        """Lower e should give higher ζ."""
        model = MultiParticlePIDModel(
            n_particles=20, mass=0.01, primary_mass=1.0,
            primary_stiffness=1000.0, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, e=0.5,
        )
        z_03 = model.compute_zeta(0.3, 2.0, 1.0, 0.3)
        z_09 = model.compute_zeta(0.3, 2.0, 1.0, 0.9)
        self.assertGreater(z_03, z_09, "Lower e should give higher ζ")


# ===================================================================
# Feasibility assessment tests
# ===================================================================

class FeasibilityAssessmentTests(unittest.TestCase):
    """Ship hull vibration reduction feasibility."""

    def test_feasibility_assessment(self):
        """assess_feasibility must return valid result."""
        config = SweepConfig(
            gamma_values=[1.0, 2.0, 3.0],
            freq_ratio_values=[0.8, 1.0, 1.2],
            filling_ratio_values=[0.2, 0.3, 0.4],
            restitution_values=[0.3, 0.6, 0.9],
            n_particles=20,
        )
        results = run_sweep(config)
        surface = ZetaSurface.from_results(results, config)
        feas = assess_feasibility(surface, target_zeta=0.005)
        self.assertIsInstance(feas, FeasibilityResult)
        self.assertGreater(feas.zeta_max, 0)
        self.assertGreaterEqual(feas.optimal_filling_ratio, 0.2)
        self.assertLessEqual(feas.optimal_filling_ratio, 0.4)


# ===================================================================
# Ledger-based identification tests
# ===================================================================

class LedgerBasedIdentificationTests(unittest.TestCase):
    """Damping ratio identification from single-particle ledger."""

    def test_ledger_identification(self):
        """identify_damping_ratio_from_ledger must return valid result."""
        ledger = SingleParticleEnergyLedger(
            mass=0.001, e_target=0.5, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, dt=1e-4, max_steps=200000,
        )
        ledger.run()
        result = identify_damping_ratio_from_ledger(
            ledger, n_particles=20, primary_mass=1.0,
            primary_stiffness=1000.0, amplitude=0.02, frequency=5.0,
            filling_ratio=0.3, gamma=2.0, freq_ratio=1.0,
        )
        self.assertIsInstance(result, DampingRatioResult)
        self.assertGreater(result.zeta_mean, 0)
        self.assertEqual(result.method, "ledger")

    def test_ledger_vs_model_consistency(self):
        """Ledger-based and model-based ζ should be same order of magnitude."""
        ledger = SingleParticleEnergyLedger(
            mass=0.001, e_target=0.5, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, dt=1e-4, max_steps=200000,
        )
        ledger.run()
        result_ledger = identify_damping_ratio_from_ledger(
            ledger, n_particles=20, primary_mass=1.0,
            primary_stiffness=1000.0, amplitude=0.02, frequency=5.0,
            filling_ratio=0.3, gamma=2.0, freq_ratio=1.0,
        )
        model = MultiParticlePIDModel(
            n_particles=20, mass=0.001, primary_mass=1.0,
            primary_stiffness=1000.0, cavity_length=0.1,
            amplitude=0.02, frequency=5.0, e=0.5,
        )
        result_model = identify_damping_ratio_from_model(
            model, 0.3, 2.0, 1.0, 0.5
        )
        # Same order of magnitude (within factor of 10)
        ratio = result_ledger.zeta_mean / result_model.zeta_mean if result_model.zeta_mean > 0 else 0
        self.assertLess(
            ratio, 10.0,
            f"Ledger/model ratio {ratio:.1f} exceeds factor of 10",
        )


if __name__ == "__main__":
    unittest.main()
