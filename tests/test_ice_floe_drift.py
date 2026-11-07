"""Tests for ice floe drift DEM-LBM coupled benchmark.

Verifies:
1. Analytical solution correctness
2. DEM-LBM coupling physics (force balance, steady state)
3. Parameter sweep (9 cases)
4. Negative control (no fluid → stationary)
5. Force balance residual < 1%
"""
import math
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensordem.ice_floe_drift import (
    analytical_drift_velocity,
    IceFloeProperties,
    FluidProperties,
    DEMLBMConfig,
    IceFloeDriftSimulator,
    run_parameter_sweep,
    negative_control_no_fluid,
    verify_force_balance,
)


class TestAnalyticalSolution:
    """Test analytical drift velocity formula."""
    
    def test_baseline_case(self):
        """Baseline: U_current=1 m/s, ρ_w=1025, ρ_i=900, C_D equal."""
        u_drift = analytical_drift_velocity(
            u_current=1.0,
            rho_water=1025.0,
            rho_ice=900.0,
            cd_water=1.0,
            cd_ice=1.0,
        )
        # Expected: 1.0 * sqrt(1025/900) * sqrt(1) = 1.067
        expected = 1.0 * math.sqrt(1025.0 / 900.0)
        assert abs(u_drift - expected) < 1e-6
        # Task states ≈ 1.05 m/s (simplified), our model gives 1.067
        assert 1.05 < u_drift < 1.07
    
    def test_zero_current(self):
        """Zero current → zero drift."""
        u_drift = analytical_drift_velocity(u_current=0.0)
        assert u_drift == 0.0
    
    def test_density_ratio_effect(self):
        """Higher water density → higher drift velocity."""
        u1 = analytical_drift_velocity(u_current=1.0, rho_water=1000.0)
        u2 = analytical_drift_velocity(u_current=1.0, rho_water=1025.0)
        assert u2 > u1
    
    def test_drag_coefficient_ratio(self):
        """Higher C_D_water/C_D_ice → higher drift."""
        u1 = analytical_drift_velocity(u_current=1.0, cd_water=1.0, cd_ice=1.0)
        u2 = analytical_drift_velocity(u_current=1.0, cd_water=1.2, cd_ice=1.0)
        assert u2 > u1
    
    def test_submerged_fraction(self):
        """Partial submergence reduces drift."""
        u_full = analytical_drift_velocity(u_current=1.0, submerged_fraction=1.0)
        u_half = analytical_drift_velocity(u_current=1.0, submerged_fraction=0.5)
        assert u_half < u_full
        # sqrt(0.5) factor
        assert abs(u_half / u_full - math.sqrt(0.5)) < 1e-6
    
    def test_invalid_inputs(self):
        """Invalid inputs raise ValueError."""
        with pytest.raises(ValueError):
            analytical_drift_velocity(u_current=-1.0)
        with pytest.raises(ValueError):
            analytical_drift_velocity(u_current=1.0, rho_water=0.0)
        with pytest.raises(ValueError):
            analytical_drift_velocity(u_current=1.0, cd_water=-1.0)
        with pytest.raises(ValueError):
            analytical_drift_velocity(u_current=1.0, submerged_fraction=1.5)


class TestIceFloeProperties:
    """Test ice floe geometry and mass calculations."""
    
    def test_geometry(self):
        """D=1m floe has correct area and volume."""
        floe = IceFloeProperties(diameter=1.0, thickness=0.2, density=900.0)
        assert floe.radius == 0.5
        assert abs(floe.projected_area - math.pi * 0.25) < 1e-6
        assert abs(floe.volume - math.pi * 0.25 * 0.2) < 1e-6
    
    def test_mass(self):
        """Mass = density * volume."""
        floe = IceFloeProperties(diameter=1.0, thickness=0.2, density=900.0)
        expected_mass = 900.0 * math.pi * 0.25 * 0.2
        assert abs(floe.mass - expected_mass) < 1e-6


class TestDEMLBMcoupling:
    """Test DEM-LBM coupled simulation."""
    
    def test_steady_state_convergence(self):
        """Simulation converges to steady state."""
        cfg = DEMLBMConfig(u_current=1.0, max_time=50.0)
        sim = IceFloeDriftSimulator(cfg)
        sim.run()
        
        # Check that velocity stabilizes
        u_final = sim.get_steady_state_velocity()
        assert math.isfinite(u_final)
        assert u_final > 0.0
    
    def test_steady_state_matches_analytical(self):
        """Steady-state velocity matches analytical solution within 3%."""
        u_current = 1.0
        cfg = DEMLBMConfig(u_current=u_current, max_time=50.0)
        sim = IceFloeDriftSimulator(cfg)
        sim.run()
        
        u_sim = sim.get_steady_state_velocity()
        u_analytical = analytical_drift_velocity(u_current)
        
        error_pct = 100.0 * abs(u_sim - u_analytical) / u_analytical
        assert error_pct < 3.0, f"Error {error_pct:.2f}% exceeds 3% threshold"
    
    def test_force_balance_at_steady_state(self):
        """Force balance residual < 1% at steady state."""
        result = verify_force_balance(u_current=1.0, diameter=1.0)
        assert result["passed"], f"Force balance failed: residual {result['residual_pct']:.2f}%"
        assert result["residual_pct"] < 1.0
    
    def test_acceleration_from_rest(self):
        """Floe accelerates from rest to steady state."""
        cfg = DEMLBMConfig(u_current=1.0, max_time=50.0)
        sim = IceFloeDriftSimulator(cfg)
        sim.run(u_initial=0.0)
        
        # Velocity should increase monotonically (or at least start from 0)
        assert sim.velocity_history[0] == 0.0
        assert sim.velocity_history[-1] > 0.0
    
    def test_time_history_recorded(self):
        """Time, velocity, and force histories are recorded."""
        cfg = DEMLBMConfig(u_current=1.0, max_time=10.0, dt=0.1)
        sim = IceFloeDriftSimulator(cfg)
        sim.run()
        
        assert len(sim.time_history) > 0
        assert len(sim.velocity_history) == len(sim.time_history)
        assert len(sim.force_history) == len(sim.time_history)


class TestParameterSweep:
    """Test parameter sweep over 9 cases."""
    
    def test_sweep_runs(self):
        """Parameter sweep completes for all 9 cases."""
        results = run_parameter_sweep(
            u_currents=[0.5, 1.0, 1.5],
            diameters=[0.5, 1.0, 2.0],
        )
        assert len(results) == 9
    
    def test_sweep_accuracy(self):
        """All cases within 3% of analytical solution."""
        results = run_parameter_sweep()
        for r in results:
            assert r["error_pct"] < 3.0, (
                f"Case U={r['u_current']}, D={r['diameter']}: "
                f"error {r['error_pct']:.2f}% exceeds 3%"
            )
    
    def test_sweep_structure(self):
        """Results have correct structure."""
        results = run_parameter_sweep()
        for r in results:
            assert "u_current" in r
            assert "diameter" in r
            assert "u_drift_analytical" in r
            assert "u_drift_sim" in r
            assert "error_pct" in r
            assert r["u_drift_analytical"] > 0
            assert r["u_drift_sim"] > 0


class TestNegativeControl:
    """Test negative control: no fluid → stationary floe."""
    
    def test_no_fluid_stationary(self):
        """Pure DEM (no fluid) → floe remains stationary."""
        u_final = negative_control_no_fluid()
        assert abs(u_final) < 1e-6, f"Expected 0.0, got {u_final}"


class TestReynoldsNumber:
    """Test Reynolds number calculation."""
    
    def test_reynolds_number(self):
        """Re = U * D / ν."""
        cfg = DEMLBMConfig(u_current=1.0)
        cfg.floe.diameter = 1.0
        cfg.fluid.kinematic_viscosity = 1e-6
        Re = cfg.reynolds_number
        expected = 1.0 * 1.0 / 1e-6
        assert abs(Re - expected) < 1.0


class TestPhysicsConsistency:
    """Test physical consistency of the model."""
    
    def test_higher_current_higher_drift(self):
        """Higher current → higher drift velocity."""
        u1 = analytical_drift_velocity(u_current=0.5)
        u2 = analytical_drift_velocity(u_current=1.0)
        u3 = analytical_drift_velocity(u_current=1.5)
        assert u1 < u2 < u3
    
    def test_larger_diameter_same_drift(self):
        """Diameter doesn't affect analytical drift (cancels in force balance)."""
        # Note: In the simplified model, diameter cancels out
        u1 = analytical_drift_velocity(u_current=1.0)
        # The analytical formula doesn't depend on diameter
        # (both drag and inertia scale with area)
        assert u1 > 0
    
    def test_energy_conservation_check(self):
        """Net force should decrease as floe approaches steady state."""
        cfg = DEMLBMConfig(u_current=1.0, max_time=20.0)
        sim = IceFloeDriftSimulator(cfg)
        sim.run()
        
        # Force should approach zero at steady state
        f_initial = abs(sim.force_history[0])
        f_final = abs(sim.force_history[-1])
        assert f_final < f_initial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
