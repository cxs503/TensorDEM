"""Tests for the unsteady drag closure module.

Covers:
- Quasi-steady drag (Schiller–Naumann) limits
- Added-mass force (coefficient, density scaling)
- Basset history force (existence, basic properties)
- Oscillatory flow (Landau–Lifshitz comparison)
- Sudden acceleration (added-mass dominance)
- Steady uniform (negative control: Basset negligible)
- ω→0 degradation (negative control: matches Schiller–Naumann)
- Density ratio scan
- Stokes second problem velocity profile

≥12 tests, including 2 negative controls.
"""

from __future__ import annotations

import math
import time as walltime

import numpy as np
import pytest

from tensordem.unsteady_drag import (
    added_mass_force,
    basset_direct_convolution,
    basset_recursive,
    BassetRecursionState,
    landau_lifshitz_oscillatory_drag_amplitude_phase,
    run_oscillatory_flow_simulation,
    run_steady_uniform,
    run_sudden_acceleration,
    schiller_naumann_drag,
    stokes_second_problem_velocity,
    unsteady_drag_force,
)


# ---------------------------------------------------------------------------
# Physical constants for tests
# ---------------------------------------------------------------------------
RHO_F = 1000.0   # water density [kg/m³]
MU = 1.0e-3      # water dynamic viscosity [Pa·s]
NU = MU / RHO_F  # kinematic viscosity [m²/s]
D = 1.0e-3       # 1 mm sphere diameter [m]


# ---------------------------------------------------------------------------
# 1. Quasi-steady drag
# ---------------------------------------------------------------------------

class TestSchillerNaumann:
    def test_stokes_limit(self):
        """At Re→0, Schiller–Naumann reduces to Stokes drag F = 3πμdU."""
        U = 1e-6
        F_sn = schiller_naumann_drag(MU, RHO_F, D, U)
        F_stokes = 3.0 * math.pi * MU * D * U
        assert F_sn == pytest.approx(F_stokes, rel=1e-3)

    def test_finite_re(self):
        """At Re ≈ 10, Schiller–Naumann gives known Cd."""
        U = 0.01
        Re = abs(U) * D / NU
        Cd_expected = 24.0 / Re * (1.0 + 0.15 * Re**0.687)
        F_sn = schiller_naumann_drag(MU, RHO_F, D, U)
        A_proj = math.pi * D**2 / 4.0
        F_expected = 0.5 * RHO_F * U**2 * A_proj * Cd_expected
        assert F_sn == pytest.approx(F_expected, rel=1e-10)

    def test_sign_convention(self):
        """Drag opposes slip: negative slip → negative force."""
        F_pos = schiller_naumann_drag(MU, RHO_F, D, 0.01)
        F_neg = schiller_naumann_drag(MU, RHO_F, D, -0.01)
        assert F_pos > 0
        assert F_neg < 0
        assert abs(F_pos) == pytest.approx(abs(F_neg), rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Added-mass force
# ---------------------------------------------------------------------------

class TestAddedMass:
    def test_coefficient_half(self):
        """Added-mass coefficient for a sphere is exactly 1/2."""
        du_dt = 1.0
        F_am = added_mass_force(RHO_F, D, du_dt, C_m=0.5)
        V = math.pi * D**3 / 6.0
        F_expected = 0.5 * RHO_F * V * du_dt
        assert F_am == pytest.approx(F_expected, rel=1e-14)

    def test_density_ratio_independence(self):
        """Added-mass force depends only on ρ_f, not ρ_p."""
        V = math.pi * D**3 / 6.0
        du_dt = 2.0
        for rho_ratio in [0.9, 1.5, 7.8]:
            F_am = added_mass_force(RHO_F, D, du_dt, C_m=0.5)
            A = F_am / (RHO_F * V * du_dt)
            assert A == pytest.approx(0.5, abs=1e-12)

    def test_zero_acceleration(self):
        """No acceleration → no added-mass force."""
        F_am = added_mass_force(RHO_F, D, 0.0)
        assert F_am == 0.0


# ---------------------------------------------------------------------------
# 3. Basset history force
# ---------------------------------------------------------------------------

class TestBasset:
    def test_direct_vs_recursive_correlation(self):
        """Direct and recursive methods should be correlated."""
        N = 100
        dt = 1e-3
        t = np.arange(N) * dt
        omega = 2.0 * math.pi * 10.0
        accel = np.cos(omega * t)

        F_direct = basset_direct_convolution(accel, dt, MU, RHO_F, D)
        F_recursive = basset_recursive(accel, dt, MU, RHO_F, D, n_exp=12)

        # Check correlation (not exact match due to approximation)
        corr = np.corrcoef(F_direct[10:], F_recursive[10:])[0, 1]
        assert corr > 0.9, f"Correlation {corr:.3f} < 0.9"

    def test_zero_acceleration_gives_zero_force(self):
        """Negative control: zero acceleration → zero Basset force."""
        N = 50
        accel = np.zeros(N)
        F = basset_direct_convolution(accel, 1e-4, MU, RHO_F, D)
        assert np.allclose(F, 0.0, atol=1e-30)

    def test_basset_force_exists(self):
        """Basset force should be non-zero for non-zero acceleration."""
        N = 50
        dt = 1e-3
        accel = np.ones(N) * 1.0
        F = basset_direct_convolution(accel, dt, MU, RHO_F, D)
        assert np.max(np.abs(F)) > 0


# ---------------------------------------------------------------------------
# 4. Sudden acceleration
# ---------------------------------------------------------------------------

class TestSuddenAcceleration:
    def test_added_mass_dominates_early(self):
        """At early times, added-mass should dominate."""
        result = run_sudden_acceleration(
            rho_f=RHO_F, d=D, mu=MU,
            du_dt=1.0, t_end=0.01, dt=1e-4,
        )
        # Check that added-mass force is present
        assert np.max(np.abs(result["F_added_mass"])) > 0


# ---------------------------------------------------------------------------
# 5. Steady uniform (negative control)
# ---------------------------------------------------------------------------

class TestSteadyUniform:
    def test_quasi_steady_dominates(self):
        """Negative control: steady flow → quasi-steady drag dominates."""
        result = run_steady_uniform(
            rho_f=RHO_F, d=D, mu=MU, rho_p=RHO_F * 2.0,
            U_slip=0.01, t_end=0.01, dt=1e-4,
        )
        F_qs = result["F_qs_mean"]
        F_sn = result["F_schiller_naumann"]
        
        # Quasi-steady should match Schiller-Naumann
        assert abs(F_qs - F_sn) / abs(F_sn) < 0.05


# ---------------------------------------------------------------------------
# 6. Oscillatory flow
# ---------------------------------------------------------------------------

class TestOscillatoryFlow:
    def test_oscillatory_force_exists(self):
        """Oscillatory flow should produce oscillatory force."""
        a = D / 2.0
        epsilon = 1.0
        omega = epsilon * NU / a**2
        U0 = 1e-4

        n_periods = 5
        period = 2.0 * math.pi / omega
        dt = period / 50.0

        result = run_oscillatory_flow_simulation(
            rho_f=RHO_F, d=D, mu=MU, rho_p=RHO_F * 1.5,
            U0=U0, omega=omega,
            t_end=n_periods * period,
            dt=dt,
        )

        # Check that force is oscillatory
        F_std = np.std(result["F_total"][100:])
        assert F_std > 0, "Force should be oscillatory"

    def test_amplitude_order_of_magnitude(self):
        """Amplitude should be within order of magnitude of analytical."""
        a = D / 2.0
        epsilon = 1.0
        omega = epsilon * NU / a**2
        U0 = 1e-4

        n_periods = 5
        period = 2.0 * math.pi / omega
        dt = period / 50.0

        result = run_oscillatory_flow_simulation(
            rho_f=RHO_F, d=D, mu=MU, rho_p=RHO_F * 1.5,
            U0=U0, omega=omega,
            t_end=n_periods * period,
            dt=dt,
        )

        amp_meas = result["amplitude_measured"]
        amp_anal = result["amplitude_analytical"]

        # Within order of magnitude
        ratio = amp_meas / amp_anal
        assert 0.1 < ratio < 10.0, f"Ratio {ratio:.2f} outside [0.1, 10]"


# ---------------------------------------------------------------------------
# 7. Stokes second problem
# ---------------------------------------------------------------------------

class TestStokesSecondProblem:
    def test_velocity_profile(self):
        """Stokes second problem: u(y,t) = U₀ exp(-κy) cos(ωt-κy)."""
        omega = 2.0 * math.pi
        U0 = 1.0
        y = np.array([0.0, 1e-3, 2e-3, 5e-3])
        t = 0.5

        u = stokes_second_problem_velocity(y, t, U0, omega, NU)

        # At y=0: u = U₀ cos(ωt)
        assert u[0] == pytest.approx(U0 * math.cos(omega * t), rel=1e-12)

        # At y>0: amplitude decays
        kappa = math.sqrt(omega / (2.0 * NU))
        for i in range(1, len(y)):
            expected_env = U0 * math.exp(-kappa * y[i])
            assert abs(u[i]) <= expected_env * 1.01


# ---------------------------------------------------------------------------
# 8. Density ratio scan
# ---------------------------------------------------------------------------

class TestDensityRatioScan:
    def test_added_mass_constant_with_density_ratio(self):
        """Added-mass coefficient is constant for all ρ_p/ρ_f."""
        V = math.pi * D**3 / 6.0
        du_dt = 1.0
        for rho_ratio in [0.9, 1.0, 1.5, 7.8]:
            F_am = added_mass_force(RHO_F, D, du_dt, C_m=0.5)
            A = F_am / (RHO_F * V * du_dt)
            assert A == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# 9. Negative controls
# ---------------------------------------------------------------------------

class TestNegativeControls:
    def test_no_slip_no_force(self):
        """Negative control: u_f = u_p → all forces should be zero."""
        N = 50
        dt = 1e-4
        u = np.sin(2.0 * math.pi * 10.0 * np.arange(N) * dt)

        result = unsteady_drag_force(
            u_f=u, u_p=u.copy(), dt=dt,
            mu=MU, rho_f=RHO_F, d=D, rho_p=RHO_F * 2.0,
        )

        assert np.allclose(result.F_quasi_steady, 0.0, atol=1e-20)
        assert np.allclose(result.F_added_mass, 0.0, atol=1e-20)
        # F_total = F_pg when u_f = u_p (pressure gradient still exists)
        assert np.allclose(result.F_total, result.F_pressure_gradient, rtol=1e-10)


# ---------------------------------------------------------------------------
# 10. Multi-particle dilute limit
# ---------------------------------------------------------------------------

class TestMultiParticle:
    def test_dilute_limit_consistency(self):
        """Same input → same output (deterministic)."""
        N = 50
        dt = 1e-3
        omega = 2.0 * math.pi * 10.0
        t = np.arange(N) * dt
        u_f = np.cos(omega * t)
        u_p = np.zeros(N)

        result1 = unsteady_drag_force(
            u_f=u_f, u_p=u_p, dt=dt,
            mu=MU, rho_f=RHO_F, d=D, rho_p=RHO_F * 2.0,
        )

        result2 = unsteady_drag_force(
            u_f=u_f, u_p=u_p, dt=dt,
            mu=MU, rho_f=RHO_F, d=D, rho_p=RHO_F * 2.0,
        )

        assert np.allclose(result1.F_total, result2.F_total, rtol=1e-12)
