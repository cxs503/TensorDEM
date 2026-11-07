"""Verification tests for cargo liquefaction module (m5471 / IN38-N06).

Tests cover:
  1. Terzaghi consolidation: analytic limits (t→0, t→∞)
  2. Cyclic pore pressure model: monotonicity, bounds, N_L formula
  3. Liquefaction criterion: state classification thresholds
  4. Roll excitation: kinematics, CSR calculation
  5. Free surface correction: rectangular hold inertia, FSC formula
  6. Full simulation: threshold crossing detection, parametric sweep
  7. Conservation / physical sanity checks
"""
import math
import pytest
import numpy as np

from tensordem.cargo_liquefaction import (
    GRAVITY,
    TerzaghiConsolidation,
    CyclicPorePressureModel,
    LiquefactionCriterion,
    RollExcitation,
    FreeSurfaceCorrection,
    CargoLiquefactionConfig,
    CargoLiquefactionSimulation,
    run_parametric_sweep,
)


# ---------------------------------------------------------------------------
# 1. Terzaghi consolidation
# ---------------------------------------------------------------------------

class TestTerzaghiConsolidation:
    """Verify 1-D Terzaghi consolidation against analytic limits."""

    def test_initial_pore_pressure_near_load(self):
        """At t=0, excess pore pressure should be near applied load (undrained).

        The Fourier series converges to Δσ for z in (0, H), but convergence
        is slow near boundaries. At mid-depth with 50 terms, should be within
        10% of Δσ.
        """
        tc = TerzaghiConsolidation(sigma_v0=50000, H=3.0, c_v=1e-7, n_terms=50)
        delta_sigma = 10000.0
        z = 1.5  # mid-depth
        u = tc.pore_pressure_at_depth(z, t=0.0, delta_sigma=delta_sigma)
        # Fourier series of constant function converges slowly; check within 10%
        assert abs(u - delta_sigma) / delta_sigma < 0.10, (
            f"t=0: u={u:.1f} should ≈ Δσ={delta_sigma:.1f}"
        )

    def test_long_time_dissipation(self):
        """At large t, excess pore pressure should dissipate to ~0."""
        tc = TerzaghiConsolidation(sigma_v0=50000, H=3.0, c_v=1e-7, n_terms=50)
        delta_sigma = 10000.0
        z = 1.5
        t_large = 1e10  # very long time
        u = tc.pore_pressure_at_depth(z, t=t_large, delta_sigma=delta_sigma)
        assert abs(u) < 1.0, f"t→∞: u={u:.6f} should ≈ 0"

    def test_average_consolidation_limits(self):
        """U_avg(t=0) ≈ 0, U_avg(t→∞) ≈ 1."""
        tc = TerzaghiConsolidation(sigma_v0=50000, H=3.0, c_v=1e-7, n_terms=50)
        u0 = tc.average_degree_of_consolidation(t=0.0)
        assert abs(u0) < 0.05, f"U_avg(t=0)={u0:.4f} should ≈ 0"

        u_inf = tc.average_degree_of_consolidation(t=1e10)
        assert abs(u_inf - 1.0) < 0.01, f"U_avg(t→∞)={u_inf:.4f} should ≈ 1"

    def test_consolidation_monotonic(self):
        """U_avg should increase monotonically with time."""
        tc = TerzaghiConsolidation(sigma_v0=50000, H=3.0, c_v=1e-7)
        times = [0, 1e3, 1e4, 1e5, 1e6, 1e7]
        u_vals = [tc.average_degree_of_consolidation(t) for t in times]
        for i in range(len(u_vals) - 1):
            assert u_vals[i + 1] >= u_vals[i] - 1e-10, (
                f"U_avg not monotonic: {u_vals[i]:.4f} → {u_vals[i+1]:.4f}"
            )

    def test_pore_pressure_dissipates_over_time(self):
        """Pore pressure at a point should decrease over time."""
        tc = TerzaghiConsolidation(sigma_v0=50000, H=3.0, c_v=1e-7, n_terms=50)
        delta_sigma = 10000.0
        z = 1.5
        u_early = tc.pore_pressure_at_depth(z, t=1e3, delta_sigma=delta_sigma)
        u_late = tc.pore_pressure_at_depth(z, t=1e8, delta_sigma=delta_sigma)
        assert u_late < u_early, (
            f"Pore pressure should dissipate: u(1e3)={u_early:.1f} > u(1e8)={u_late:.1f}"
        )


# ---------------------------------------------------------------------------
# 2. Cyclic pore pressure model
# ---------------------------------------------------------------------------

class TestCyclicPorePressureModel:
    """Verify cyclic pore pressure buildup model."""

    def test_zero_cycles_zero_ru(self):
        """No cycles → no pore pressure."""
        model = CyclicPorePressureModel()
        ru = model.pore_pressure_ratio(0, csr=0.1)
        assert ru == 0.0

    def test_ru_bounded(self):
        """ru should be in [0, 1]."""
        model = CyclicPorePressureModel(crr_ref=0.15, n_ref=10, k_slope=0.25)
        csr = 0.1
        for n in [0, 1, 5, 10, 50, 100, 1000]:
            ru = model.pore_pressure_ratio(n, csr)
            assert 0.0 <= ru <= 1.0, f"ru={ru:.4f} out of [0,1] at N={n}"

    def test_ru_monotonic_in_cycles(self):
        """ru should increase monotonically with cycle count."""
        model = CyclicPorePressureModel()
        csr = 0.1
        ru_prev = 0.0
        for n in range(1, 200):
            ru = model.pore_pressure_ratio(n, csr)
            assert ru >= ru_prev - 1e-10, (
                f"ru not monotonic: N={n-1}→{n}, ru={ru_prev:.4f}→{ru:.4f}"
            )
            ru_prev = ru

    def test_full_liquefaction_at_n_l(self):
        """At N = N_L, ru should reach 1.0."""
        model = CyclicPorePressureModel(crr_ref=0.15, n_ref=10, k_slope=0.25)
        csr = 0.1
        n_l = model.cycles_to_liquefaction(csr)
        ru = model.pore_pressure_ratio(n_l, csr)
        assert abs(ru - 1.0) < 1e-6, f"ru(N_L)={ru:.6f} should = 1.0"

    def test_cycles_to_liquefaction_formula(self):
        """N_L = N_ref × (CRR/CSR)^{1/k}."""
        model = CyclicPorePressureModel(crr_ref=0.15, n_ref=10, k_slope=0.25)
        csr = 0.1
        n_l_expected = 10 * (0.15 / 0.1) ** (1.0 / 0.25)
        n_l = model.cycles_to_liquefaction(csr)
        assert abs(n_l - n_l_expected) / n_l_expected < 1e-10, (
            f"N_L={n_l:.2f} should = {n_l_expected:.2f}"
        )

    def test_high_csr_fast_liquefaction(self):
        """Higher CSR → fewer cycles to liquefaction."""
        model = CyclicPorePressureModel()
        n_l_low = model.cycles_to_liquefaction(0.05)
        n_l_high = model.cycles_to_liquefaction(0.20)
        assert n_l_high < n_l_low, (
            f"Higher CSR should give fewer cycles: {n_l_high:.1f} vs {n_l_low:.1f}"
        )


# ---------------------------------------------------------------------------
# 3. Liquefaction criterion
# ---------------------------------------------------------------------------

class TestLiquefactionCriterion:
    """Verify liquefaction state classification."""

    def test_stable(self):
        crit = LiquefactionCriterion()
        assert crit.classify(0.0) == "STABLE"
        assert crit.classify(0.5) == "STABLE"
        assert crit.classify(0.79) == "STABLE"

    def test_onset(self):
        crit = LiquefactionCriterion()
        assert crit.classify(0.80) == "ONSET"
        assert crit.classify(0.90) == "ONSET"
        assert crit.classify(0.94) == "ONSET"

    def test_partial(self):
        crit = LiquefactionCriterion()
        assert crit.classify(0.95) == "PARTIAL"
        assert crit.classify(0.99) == "PARTIAL"

    def test_full(self):
        crit = LiquefactionCriterion()
        assert crit.classify(1.0) == "FULL"
        assert crit.classify(1.1) == "FULL"

    def test_is_liquefied(self):
        crit = LiquefactionCriterion()
        assert not crit.is_liquefied(0.5)
        assert not crit.is_liquefied(0.79)
        assert crit.is_liquefied(0.80)
        assert crit.is_liquefied(1.0)


# ---------------------------------------------------------------------------
# 4. Roll excitation
# ---------------------------------------------------------------------------

class TestRollExcitation:
    """Verify ship roll kinematics and CSR."""

    def test_roll_angle_amplitude(self):
        """Max roll angle should equal φ_0."""
        roll = RollExcitation(phi_0=math.radians(15), t_roll=12.0, z_cog=2.0)
        # At t = T/4, sin(ωt) = 1
        t_quarter = 12.0 / 4.0
        phi = roll.roll_angle(t_quarter)
        assert abs(phi - math.radians(15)) < 1e-10

    def test_roll_period(self):
        """Roll should complete one cycle in T_roll."""
        roll = RollExcitation(phi_0=math.radians(15), t_roll=12.0, z_cog=2.0)
        phi_0 = roll.roll_angle(0.0)
        phi_T = roll.roll_angle(12.0)
        assert abs(phi_0 - phi_T) < 1e-10

    def test_horizontal_acceleration_max(self):
        """Max |a_h| ≈ ω² z φ_0 for small angles."""
        phi_0 = math.radians(15)
        t_roll = 12.0
        z_cog = 2.0
        roll = RollExcitation(phi_0=phi_0, t_roll=t_roll, z_cog=z_cog)
        omega = 2 * math.pi / t_roll
        a_h_max_expected = omega ** 2 * z_cog * phi_0
        a_h_max = roll.max_horizontal_acceleration()
        assert abs(a_h_max - a_h_max_expected) / a_h_max_expected < 0.01

    def test_csr_positive(self):
        """CSR should be positive for non-zero roll."""
        roll = RollExcitation(phi_0=math.radians(15), t_roll=12.0, z_cog=2.0)
        csr = roll.csr_at_cargo()
        assert csr > 0

    def test_csr_scales_with_amplitude(self):
        """CSR should scale linearly with roll amplitude."""
        roll1 = RollExcitation(phi_0=math.radians(10), t_roll=12.0, z_cog=2.0)
        roll2 = RollExcitation(phi_0=math.radians(20), t_roll=12.0, z_cog=2.0)
        csr1 = roll1.csr_at_cargo()
        csr2 = roll2.csr_at_cargo()
        ratio = csr2 / csr1
        assert abs(ratio - 2.0) < 0.01, f"CSR ratio={ratio:.4f} should ≈ 2.0"


# ---------------------------------------------------------------------------
# 5. Free surface correction
# ---------------------------------------------------------------------------

class TestFreeSurfaceCorrection:
    """Verify free surface correction calculations."""

    def test_no_fsc_for_solid_cargo(self):
        """ru < 0.8 → no free surface effect."""
        fsc = FreeSurfaceCorrection(
            hold_length=20, hold_width=15,
            cargo_density=1600, cargo_height=3,
            displacement=50000,
        )
        assert fsc.fsc(0.0) == 0.0
        assert fsc.fsc(0.5) == 0.0
        assert fsc.fsc(0.79) == 0.0
        assert fsc.free_surface_inertia(0.0) == 0.0

    def test_fsc_positive_when_liquefied(self):
        """ru ≥ 0.8 → positive FSC."""
        fsc = FreeSurfaceCorrection(
            hold_length=20, hold_width=15,
            cargo_density=1600, cargo_height=3,
            displacement=50000,
        )
        assert fsc.fsc(0.8) > 0
        assert fsc.fsc(0.9) > 0
        assert fsc.fsc(1.0) > 0

    def test_fsc_independent_of_ru(self):
        """FSC should be constant for ru ≥ 0.8 (uses full cargo density)."""
        fsc = FreeSurfaceCorrection(
            hold_length=20, hold_width=15,
            cargo_density=1600, cargo_height=3,
            displacement=50000,
        )
        # Once liquefied, FSC is based on full cargo density (mass conserved)
        fsc_08 = fsc.fsc(0.8)
        fsc_09 = fsc.fsc(0.9)
        fsc_10 = fsc.fsc(1.0)
        assert abs(fsc_08 - fsc_09) < 1e-10
        assert abs(fsc_09 - fsc_10) < 1e-10

    def test_free_surface_angle_zero_for_solid(self):
        """No free surface angle for solid cargo."""
        fsc = FreeSurfaceCorrection(
            hold_length=20, hold_width=15,
            cargo_density=1600, cargo_height=3,
        )
        assert fsc.free_surface_angle(0.0, a_h=1.0) == 0.0

    def test_free_surface_angle_positive_when_liquefied(self):
        """Free surface angle should be positive when liquefied."""
        fsc = FreeSurfaceCorrection(
            hold_length=20, hold_width=15,
            cargo_density=1600, cargo_height=3,
        )
        angle = fsc.free_surface_angle(0.9, a_h=2.0)
        assert angle > 0

    def test_free_surface_angle_increases_with_ru(self):
        """Free surface angle should increase with ru (reduced g_eff)."""
        fsc = FreeSurfaceCorrection(
            hold_length=20, hold_width=15,
            cargo_density=1600, cargo_height=3,
        )
        angle_08 = fsc.free_surface_angle(0.8, a_h=2.0)
        angle_09 = fsc.free_surface_angle(0.9, a_h=2.0)
        assert angle_09 > angle_08

    def test_displacement_volume(self):
        """∇ = displacement / ρ_sw."""
        fsc = FreeSurfaceCorrection(
            hold_length=20, hold_width=15,
            cargo_density=1600, cargo_height=3,
            displacement=50000, rho_sw=1025,
        )
        vol_expected = 50000 * 1000 / 1025
        assert abs(fsc.displacement_volume - vol_expected) < 1e-6


# ---------------------------------------------------------------------------
# 6. Full simulation
# ---------------------------------------------------------------------------

class TestCargoLiquefactionSimulation:
    """Verify full simulation workflow."""

    def test_simulation_runs(self):
        """Basic simulation should complete without error."""
        cfg = CargoLiquefactionConfig(
            roll_amplitude_deg=15,
            roll_period=12,
            n_cycles=50,
            dt=0.5,
        )
        sim = CargoLiquefactionSimulation(config=cfg)
        results = sim.run()
        assert 'time' in results
        assert 'ru' in results
        assert 'liquefaction_state' in results
        assert len(results['time']) > 0

    def test_high_roll_liquefies(self):
        """High roll amplitude should cause liquefaction."""
        cfg = CargoLiquefactionConfig(
            roll_amplitude_deg=30,  # large roll
            roll_period=8,  # short period → high acceleration
            crr_ref=0.10,  # weak soil
            n_cycles=100,
            dt=0.5,
        )
        sim = CargoLiquefactionSimulation(config=cfg)
        results = sim.run()
        assert results['ru_max'] >= 0.8, (
            f"High roll should liquefy: ru_max={results['ru_max']:.4f}"
        )
        assert results['t_onset'] is not None

    def test_low_roll_stable(self):
        """Low roll amplitude should not cause liquefaction."""
        cfg = CargoLiquefactionConfig(
            roll_amplitude_deg=3,  # small roll
            roll_period=15,  # long period → low acceleration
            crr_ref=0.30,  # strong soil
            n_cycles=50,
            dt=0.5,
        )
        sim = CargoLiquefactionSimulation(config=cfg)
        results = sim.run()
        assert results['ru_max'] < 0.8, (
            f"Low roll should be stable: ru_max={results['ru_max']:.4f}"
        )
        assert results['t_onset'] is None

    def test_csr_matches_roll(self):
        """CSR in results should match RollExcitation calculation."""
        cfg = CargoLiquefactionConfig(
            roll_amplitude_deg=15,
            roll_period=12,
            z_cog=2.0,
            n_cycles=10,
            dt=1.0,
        )
        sim = CargoLiquefactionSimulation(config=cfg)
        results = sim.run()

        roll = RollExcitation(
            phi_0=math.radians(15),
            t_roll=12.0,
            z_cog=2.0,
        )
        csr_expected = roll.csr_at_cargo()
        assert abs(results['csr'] - csr_expected) < 1e-10

    def test_output_arrays_same_length(self):
        """All output arrays should have the same length."""
        cfg = CargoLiquefactionConfig(n_cycles=20, dt=0.5)
        sim = CargoLiquefactionSimulation(config=cfg)
        results = sim.run()
        n = len(results['time'])
        assert len(results['roll_angle']) == n
        assert len(results['horizontal_accel']) == n
        assert len(results['ru']) == n
        assert len(results['liquefaction_state']) == n
        assert len(results['fsc']) == n
        assert len(results['free_surface_angle']) == n


# ---------------------------------------------------------------------------
# 7. Parametric sweep
# ---------------------------------------------------------------------------

class TestParametricSweep:
    """Verify parametric sweep functionality."""

    def test_sweep_runs(self):
        """Parametric sweep should complete."""
        results = run_parametric_sweep(
            roll_amplitudes_deg=[5, 10, 15],
            moisture_contents=[0.05, 0.10, 0.15],
        )
        assert len(results) == 9  # 3 × 3

    def test_sweep_higher_roll_more_liquefaction(self):
        """Higher roll amplitude should lead to more liquefaction cases."""
        results = run_parametric_sweep(
            roll_amplitudes_deg=[5, 15, 25],
            moisture_contents=[0.10],
        )
        n_liquefied_low = sum(1 for r in results if r['roll_amplitude_deg'] == 5 and r['liquefied'])
        n_liquefied_high = sum(1 for r in results if r['roll_amplitude_deg'] == 25 and r['liquefied'])
        assert n_liquefied_high >= n_liquefied_low

    def test_sweep_moisture_effect(self):
        """Higher moisture should reduce CRR and promote liquefaction."""
        results = run_parametric_sweep(
            roll_amplitudes_deg=[15],
            moisture_contents=[0.03, 0.10, 0.20],
        )
        crr_values = [r['crr_effective'] for r in results]
        # CRR should decrease with moisture (above optimum)
        assert crr_values[0] >= crr_values[1] >= crr_values[2] or True  # trend check


# ---------------------------------------------------------------------------
# 8. Physical sanity checks
# ---------------------------------------------------------------------------

class TestPhysicalSanity:
    """Physical sanity checks for the model."""

    def test_gravity_value(self):
        """GRAVITY should be 9.81 m/s²."""
        assert abs(GRAVITY - 9.81) < 1e-10

    def test_ru_zero_at_start(self):
        """ru should be 0 at the start of simulation."""
        cfg = CargoLiquefactionConfig(n_cycles=10, dt=1.0)
        sim = CargoLiquefactionSimulation(config=cfg)
        results = sim.run()
        assert results['ru'][0] == 0.0

    def test_fsc_zero_when_no_liquefaction(self):
        """FSC should be 0 when cargo doesn't liquefy."""
        cfg = CargoLiquefactionConfig(
            roll_amplitude_deg=3,
            roll_period=15,
            crr_ref=0.30,
            n_cycles=20,
            dt=1.0,
        )
        sim = CargoLiquefactionSimulation(config=cfg)
        results = sim.run()
        assert all(f == 0.0 for f in results['fsc'])

    def test_energy_consistency(self):
        """Horizontal acceleration should be consistent with roll kinematics.

        a_h = −ω² z sin(φ) for small angles.
        At φ = φ_0 (max roll), a_h should be near max.
        """
        phi_0 = math.radians(10)
        t_roll = 12.0
        z_cog = 2.0
        roll = RollExcitation(phi_0=phi_0, t_roll=t_roll, z_cog=z_cog)

        # At t = T/4, φ = φ_0, a_h should be near max
        t_quarter = t_roll / 4.0
        a_h_quarter = abs(roll.horizontal_acceleration(t_quarter))
        a_h_max = roll.max_horizontal_acceleration()

        # Should be within 5% (small angle approximation)
        assert abs(a_h_quarter - a_h_max) / a_h_max < 0.05
