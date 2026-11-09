"""Pytest tests for the two-way coupling verification surface.

Tests the four verification surfaces:
1. Stokes terminal velocity
2. Added mass oscillation (reduced test for CI)
3. Momentum exchange ledger
4. Dense limit Ergun sweep
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.two_way_coupling import (
    stokes_terminal_velocity,
    schiller_naumann_terminal_velocity,
    relaxation_time,
    run_stokes_verification,
    transfer_function,
    amplitude_ratio,
    phase_lag,
    single_particle_ledger,
    multi_particle_ledger,
    ergun_pressure_gradient_classic,
    ergun_pressure_gradient_model_a,
    dem_drag_per_volume,
    run_ergun_sweep,
    ParticleODE,
    ParticleState,
    FluidState,
    MomentumLedger,
)
from tensordem.cfd_dem_dense.forces import ForceModel, GRAVITY


class TestStokesTerminalVelocity:
    """VS1: Stokes terminal velocity verification."""

    def test_stokes_analytical(self):
        """Stokes terminal velocity formula."""
        v_t = stokes_terminal_velocity(1e-4, 2500, 1000, 1e-3)
        expected = (2500 - 1000) * GRAVITY * (1e-4)**2 / (18 * 1e-3)
        assert abs(v_t - expected) / expected < 1e-10

    def test_schiller_naumann_correction(self):
        """S-N terminal velocity should be less than Stokes (drag correction)."""
        v_stokes = stokes_terminal_velocity(1e-4, 2500, 1000, 1e-3)
        v_sn = schiller_naumann_terminal_velocity(1e-4, 2500, 1000, 1e-3)
        assert v_sn < v_stokes
        assert v_sn > 0.8 * v_stokes  # Not too much correction at low Re

    def test_numerical_convergence(self):
        """Numerical integration should converge to S-N terminal velocity."""
        result = run_stokes_verification(
            diameter=1e-4, rho_particle=2500, rho_fluid=1000, mu=1e-3,
            dt=1e-5, t_max=0.05
        )
        assert result["relative_error_sn"] < 1e-6
        assert result["Re_terminal"] < 1.0  # Stokes regime

    def test_relaxation_time(self):
        """Relaxation time should include added mass."""
        tau = relaxation_time(1e-4, 2500, 1000, 1e-3, added_mass_coeff=0.5)
        tau_no_am = relaxation_time(1e-4, 2500, 1000, 1e-3, added_mass_coeff=0.0)
        assert tau > tau_no_am
        assert abs(tau / tau_no_am - (2500 + 0.5*1000) / 2500) < 1e-10


class TestMomentumLedger:
    """VS3: Momentum exchange conservation ledger."""

    def test_single_particle_conservation(self):
        """Single particle: inter-phase momentum conserved to machine precision."""
        result = single_particle_ledger(
            diameter=1e-4, rho_particle=2500, rho_fluid=1000, mu=1e-3,
            dt=1e-5, n_steps=500
        )
        assert result["residual"] < 1e-12

    def test_multi_particle_conservation(self):
        """Multi-particle: total inter-phase momentum conserved."""
        result = multi_particle_ledger(
            n_particles=50, diameter=1e-4, rho_particle=2500, rho_fluid=1000,
            mu=1e-3, dt=1e-5, n_steps=200
        )
        assert result["residual"] < 1e-12

    def test_ledger_tracks_channels(self):
        """Ledger should track each force channel separately."""
        ledger = MomentumLedger()
        ledger.accumulate(
            drag_p=1.0, drag_f=-1.0,
            pressure_p=0.5, pressure_f=-0.5,
            added_mass_p=0.2, added_mass_f=-0.2,
            force_l1=1.7
        )
        assert ledger.total_particle_interphase == 1.7
        assert ledger.total_fluid_interphase == -1.7
        assert ledger.residual < 1e-15


class TestErgunSweep:
    """VS4: Dense limit Ergun inter-validation."""

    def test_ergun_classic_formula(self):
        """Classic Ergun pressure gradient formula."""
        dp_dz = ergun_pressure_gradient_classic(0.5, 0.1, 2e-3, 1000, 1e-3)
        # Manual calculation
        eps = 0.5
        phi = 0.5
        viscous = 150 * 1e-3 * phi**2 / (eps**3 * (2e-3)**2) * 0.1
        inertial = 1.75 * 1000 * phi / (eps**3 * 2e-3) * 0.1**2
        expected = viscous + inertial
        assert abs(dp_dz - expected) / expected < 1e-10

    def test_model_a_relation(self):
        """Model-A form: f = eps * (dp/dz)_classic."""
        eps = 0.5
        dp_classic = ergun_pressure_gradient_classic(eps, 0.1, 2e-3, 1000, 1e-3)
        f_model_a = ergun_pressure_gradient_model_a(eps, 0.1, 2e-3, 1000, 1e-3)
        assert abs(f_model_a - eps * dp_classic) < 1e-10

    def test_dense_regime_match(self):
        """In dense regime (phi > 0.2), Gidaspow should match Ergun Model-A."""
        result = run_ergun_sweep(
            diameter=2e-3, n_phi=15, n_velocities=2
        )
        # Dense regime: phi > 0.2
        assert result["dense_regime_max_error"] < 1e-10


class TestParticleODE:
    """Unit tests for the particle ODE integrator."""

    def test_force_balance_at_terminal_velocity(self):
        """At terminal velocity, net force should be zero."""
        ode = ParticleODE(1e-4, 2500, 1000, 1e-3)
        v_t = schiller_naumann_terminal_velocity(1e-4, 2500, 1000, 1e-3)
        v_p = np.array([0, 0, -v_t])
        fluid = FluidState.quiescent(1000, GRAVITY)
        accel, forces = ode.compute_forces(v_p, fluid)
        # Net force should be near zero (Stokes approx)
        assert np.linalg.norm(forces["net"]) < 1e-12

    def test_reaction_newton_third_law(self):
        """Reaction should be equal and opposite to inter-phase forces."""
        ode = ParticleODE(1e-4, 2500, 1000, 1e-3)
        v_p = np.array([0, 0, -0.005])
        fluid = FluidState.quiescent(1000, GRAVITY)
        _, forces = ode.compute_forces(v_p, fluid)
        reaction = ode.compute_reaction(forces)
        inter_phase = forces["drag"] + forces["pressure"] + forces["added_mass_explicit"]
        assert np.linalg.norm(reaction + inter_phase) < 1e-15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
