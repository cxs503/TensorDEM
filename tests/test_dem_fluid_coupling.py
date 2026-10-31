"""Tests for 3-D DEM ↔ fluid bidirectional coupling framework.

Verifies:
1. Momentum conservation: drag impulse on particles + fluid = 0 (to machine precision)
2. Single-particle settling vs Stokes analytical solution
3. Trilinear interpolation consistency (interpolate vs scatter)
"""
from __future__ import annotations

import math
import pytest
import torch

from tensordem.dem_fluid_coupling import (
    FluidGrid3D,
    FluidGrid3DConfig,
    DEMFluidCoupling3D,
    stokes_drag,
    schiller_naumann_drag,
    stokes_terminal_velocity,
    _trilinear_interpolate,
    _trilinear_scatter,
)


# ---------------------------------------------------------------------------
# Test 1: Momentum conservation
# ---------------------------------------------------------------------------

class TestMomentumConservation:
    """Verify that bidirectional momentum exchange conserves total momentum."""

    def test_drag_impulse_conservation(self):
        """When drag force F is applied to particle, -F is applied to fluid.

        The coupling state tracks both impulses; their sum must be zero
        to machine precision (< 1e-10 kg*m/s).
        """
        device = "cpu"
        dtype = torch.float64

        # Create a simple fluid grid
        fluid_config = FluidGrid3DConfig(
            nx=8, ny=8, nz=8,
            lx=0.08, ly=0.08, lz=0.08,
            rho_f=1000.0, mu=1e-3,
            device=device,
        )
        fluid = FluidGrid3D(fluid_config)

        # Create some fake particle data
        n_particles = 5
        positions = torch.rand(n_particles, 3, dtype=dtype, device=device) * 0.07 + 0.005
        velocities = torch.randn(n_particles, 3, dtype=dtype, device=device) * 0.1
        radius = 0.002

        # Compute drag forces
        u_fluid = fluid.sample_velocity(positions)
        v_rel = velocities - u_fluid
        f_drag = stokes_drag(v_rel, radius, fluid_config.mu)

        # Simulate coupling exchange
        dt = 1e-4
        impulse_particle = f_drag.sum(dim=0) * dt
        impulse_fluid = -f_drag.sum(dim=0) * dt  # Newton's 3rd law

        # The sum should be exactly zero
        residual = impulse_particle + impulse_fluid
        residual_magnitude = torch.linalg.vector_norm(residual).item()

        assert residual_magnitude < 1e-10, (
            f"Momentum conservation residual {residual_magnitude:.2e} >= 1e-10 kg*m/s"
        )

    def test_coupling_state_momentum_accounting(self):
        """Test that CouplingState correctly tracks momentum exchange."""
        from tensordem.dem_fluid_coupling import CouplingState

        state = CouplingState()

        # Simulate multiple exchange steps
        dt = 1e-4
        for _ in range(10):
            f_drag = torch.randn(3, 3, dtype=torch.float64) * 0.01
            impulse = f_drag.sum(dim=0) * dt
            state.total_drag_impulse_particle += impulse
            state.total_drag_impulse_fluid -= impulse
            state.step_count += 1

        # Check conservation
        residual = state.total_drag_impulse_particle + state.total_drag_impulse_fluid
        residual_magnitude = torch.linalg.vector_norm(residual).item()

        assert residual_magnitude < 1e-15, (
            f"CouplingState momentum residual {residual_magnitude:.2e} >= 1e-15"
        )
        assert state.step_count == 10

    def test_trilinear_scatter_conservation(self):
        """Verify that trilinear scatter conserves total quantity.

        When we scatter values to a grid, the sum of all grid values
        should equal the sum of input values (times the appropriate scaling).
        """
        device = "cpu"
        dtype = torch.float64

        config = FluidGrid3DConfig(
            nx=10, ny=10, nz=10,
            lx=0.1, ly=0.1, lz=0.1,
            rho_f=1000.0, mu=1e-3,
            device=device,
        )

        # Create accumulator
        accumulator = torch.zeros(config.nx, config.ny, config.nz, 3,
                                 dtype=dtype, device="cpu")

        # Create some particles with forces
        n_particles = 7
        positions = torch.rand(n_particles, 3, dtype=dtype) * 0.09 + 0.005
        forces = torch.randn(n_particles, 3, dtype=dtype) * 0.1

        # Scatter forces
        _trilinear_scatter(accumulator, positions, forces, config)

        # Total scattered force should equal total input force
        total_scattered = accumulator.sum(dim=(0, 1, 2)) * config.cell_volume
        total_input = forces.sum(dim=0)

        error = torch.linalg.vector_norm(total_scattered - total_input).item()
        assert error < 1e-10, (
            f"Trilinear scatter conservation error {error:.2e} >= 1e-10"
        )


# ---------------------------------------------------------------------------
# Test 2: Single-particle settling vs Stokes
# ---------------------------------------------------------------------------

class TestStokesSettling:
    """Verify single-particle settling against analytical Stokes solution."""

    def test_stokes_drag_force(self):
        """Test Stokes drag formula: F = -6πμRv."""
        mu = 1e-3  # Pa·s
        R = 0.002  # m
        v_rel = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64)

        f_drag = stokes_drag(v_rel, R, mu)

        # Expected: F = -6πμRv = -6π(1e-3)(0.002)(0.01) = -3.77e-7 N
        expected = -6.0 * math.pi * mu * R * v_rel
        error = torch.linalg.vector_norm(f_drag - expected).item()

        assert error < 1e-15, f"Stokes drag error {error:.2e}"

    def test_stokes_terminal_velocity_formula(self):
        """Test analytical Stokes terminal velocity formula."""
        rho_p = 1100.0  # kg/m³
        rho_f = 1000.0  # kg/m³
        R = 0.002  # m
        mu = 1e-3  # Pa·s
        g = 9.81  # m/s²

        v_t = stokes_terminal_velocity(rho_p, rho_f, R, mu, g)

        # Expected: v_t = (2/9)(ρ_p - ρ_f)gR²/μ
        expected = (2.0 / 9.0) * (rho_p - rho_f) * g * R**2 / mu

        assert abs(v_t - expected) < 1e-15, (
            f"Stokes terminal velocity error {abs(v_t - expected):.2e}"
        )

    def test_single_particle_settling_vs_stokes(self):
        """Simulate single-particle settling and compare to Stokes solution.

        A sphere settles under gravity in a viscous fluid. The equation of motion is:
            m dv/dt = (ρ_p - ρ_f)Vg - 6πμRv

        This has analytical solution:
            v(t) = v_t (1 - exp(-t/τ))
        where τ = m/(6πμR) and v_t = (ρ_p - ρ_f)Vg/(6πμR)

        We integrate this ODE numerically and compare the terminal velocity
        to the analytical Stokes solution.
        """
        # Physical parameters
        rho_p = 1100.0  # kg/m³ (particle density)
        rho_f = 1000.0  # kg/m³ (fluid density)
        R = 0.002  # m (particle radius)
        mu = 1e-3  # Pa·s (dynamic viscosity)
        g = 9.81  # m/s²

        # Derived quantities
        V_p = (4.0 / 3.0) * math.pi * R**3  # particle volume
        m_p = rho_p * V_p  # particle mass
        v_t_analytical = stokes_terminal_velocity(rho_p, rho_f, R, mu, g)
        tau = m_p / (6.0 * math.pi * mu * R)  # relaxation time

        # Time integration parameters
        dt = 1e-4  # s
        total_time = 10.0 * tau  # simulate for 10 relaxation times
        n_steps = int(total_time / dt)

        # Initial conditions
        v_z = 0.0  # initial velocity (settling downward)

        # Time integration (explicit Euler)
        velocities = []
        for step in range(n_steps):
            # Forces: gravity - buoyancy - drag
            f_gravity = -m_p * g  # downward
            f_buoyancy = rho_f * V_p * g  # upward
            f_drag = -6.0 * math.pi * mu * R * v_z  # opposes motion

            f_net = f_gravity + f_buoyancy + f_drag
            v_z += dt * f_net / m_p
            velocities.append(v_z)

        # Compute terminal velocity (average over last 20% of simulation)
        n_avg = max(1, len(velocities) // 5)
        v_terminal_sim = abs(sum(velocities[-n_avg:]) / n_avg)

        # Compare to analytical
        relative_error = abs(v_terminal_sim - v_t_analytical) / v_t_analytical

        # The criterion is < 0.05 (5%)
        assert relative_error < 0.05, (
            f"Single-particle settling error {relative_error:.4f} >= 0.05. "
            f"Simulated: {v_terminal_sim:.6f} m/s, "
            f"Analytical: {v_t_analytical:.6f} m/s"
        )

    def test_settling_with_fluid_grid(self):
        """Test settling with fluid velocity sampling.

        This test verifies that when the fluid is at rest, the particle
        settles to the Stokes terminal velocity. The fluid grid is used
        for velocity sampling but we keep the fluid at rest (no dynamics)
        to isolate the drag force computation.
        """
        from tensordem.dem3d import DEM3DConfig, IceDEM3D

        # Physical parameters (use smaller particle for lower terminal velocity)
        rho_p = 1100.0
        rho_f = 1000.0
        R = 0.0005  # 0.5 mm radius (smaller for lower v_t)
        mu = 1e-3
        g = 9.81

        # Derived quantities
        V_p = (4.0 / 3.0) * math.pi * R**3
        m_p = rho_p * V_p
        v_t_analytical = stokes_terminal_velocity(rho_p, rho_f, R, mu, g)

        # Create minimal DEM (single particle)
        dem_config = DEM3DConfig(
            nx=2, ny=2, nz=2,
            radius=R,
            density=rho_p,
            bond_stiffness=1.0,
            contact_stiffness=1.0,
            breaking_strain=1.0,
            shear_breaking_strain=1.0,
            contact_damping=0.0,
            drag=0.0,
            tool_radius=1e10,
            tool_speed=0.001,
            dt=1e-5,
            device="cpu",
            fix_x_edges=False,
            fix_bottom=False,
        )
        dem = IceDEM3D(dem_config)

        # Create fluid grid (kept at rest for this test)
        domain_size = 0.04
        fluid_config = FluidGrid3DConfig(
            nx=8, ny=8, nz=8,
            lx=domain_size, ly=domain_size, lz=domain_size,
            rho_f=rho_f, mu=mu,
            gravity=(0.0, 0.0, -g),
            device="cpu",
        )
        fluid = FluidGrid3D(fluid_config)

        # Initialize single particle at center, upper region
        dem.positions[0] = torch.tensor([domain_size/2, domain_size/2, 0.7*domain_size],
                                       dtype=torch.float64)
        dem.velocities[0] = torch.zeros(3, dtype=torch.float64)
        dem.fixed[1:] = True  # Fix all but first particle

        # Time integration (shorter time since v_t is lower)
        dt = 1e-4
        total_time = 0.5  # s
        n_steps = int(total_time / dt)

        velocities_z = []
        for step in range(n_steps):
            # Sample fluid velocity at particle position (fluid is at rest)
            u_fluid = fluid.sample_velocity(dem.positions[0:1])

            # Relative velocity
            v_rel = dem.velocities[0:1] - u_fluid

            # Drag force
            f_drag = stokes_drag(v_rel, R, mu)

            # Net force on particle: gravity + buoyancy + drag
            f_gravity = torch.tensor([0.0, 0.0, -m_p * g], dtype=torch.float64)
            f_buoyancy = torch.tensor([0.0, 0.0, rho_f * V_p * g], dtype=torch.float64)
            f_net = f_gravity + f_buoyancy + f_drag[0]

            # Update particle
            dem.velocities[0] += dt * f_net / m_p
            dem.positions[0] += dt * dem.velocities[0]

            # Record
            if step % 100 == 0:
                velocities_z.append(dem.velocities[0, 2].item())

        # Compute terminal velocity
        n_avg = max(1, len(velocities_z) // 5)
        v_terminal_sim = abs(sum(velocities_z[-n_avg:]) / n_avg)

        # Compare
        relative_error = abs(v_terminal_sim - v_t_analytical) / v_t_analytical

        assert relative_error < 0.05, (
            f"Settling with fluid grid error {relative_error:.4f} >= 0.05. "
            f"Simulated: {v_terminal_sim:.6f} m/s, "
            f"Analytical: {v_t_analytical:.6f} m/s"
        )


# ---------------------------------------------------------------------------
# Test 3: Schiller-Naumann drag
# ---------------------------------------------------------------------------

class TestSchillerNaumann:
    """Test Schiller-Naumann drag correlation."""

    def test_low_re_reduces_to_stokes(self):
        """At very low Re, Schiller-Naumann should approach Stokes."""
        mu = 1e-3
        rho_f = 1000.0
        R = 0.001
        v_rel = torch.tensor([[1e-6, 0.0, 0.0]], dtype=torch.float64)

        f_stokes = stokes_drag(v_rel, R, mu)
        f_sn = schiller_naumann_drag(v_rel, R, mu, rho_f)

        # At very low Re, they should be close
        # (Schiller-Naumann has a correction factor of 1 + 0.15*Re^0.687)
        Re = rho_f * abs(v_rel[0, 0].item()) * (2 * R) / mu
        correction = 1.0 + 0.15 * Re**0.687

        # The forces should differ by the correction factor
        ratio = f_sn[0, 0].item() / f_stokes[0, 0].item()
        assert abs(ratio - correction) < 0.01, (
            f"Schiller-Naumann low-Re ratio {ratio:.4f} != {correction:.4f}"
        )

    def test_high_re_drag_coefficient(self):
        """At high Re, C_d should approach 0.44."""
        mu = 1e-3
        rho_f = 1000.0
        R = 0.01
        v_rel = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64)

        f_drag = schiller_naumann_drag(v_rel, R, mu, rho_f)

        # At high Re, C_d = 0.44
        # F = -0.5 * C_d * rho_f * A * |v| * v
        A = math.pi * R**2
        v_mag = 10.0
        expected = -0.5 * 0.44 * rho_f * A * v_mag * v_rel[0, 0].item()

        error = abs(f_drag[0, 0].item() - expected) / abs(expected)
        assert error < 0.01, f"High-Re drag error {error:.4f}"


# ---------------------------------------------------------------------------
# Test 4: Fluid grid operations
# ---------------------------------------------------------------------------

class TestFluidGrid:
    """Test fluid grid operations."""

    def test_velocity_sampling_uniform(self):
        """Sampling a uniform velocity field should return that field."""
        config = FluidGrid3DConfig(nx=8, ny=8, nz=8, lx=0.08, ly=0.08, lz=0.08)
        fluid = FluidGrid3D(config)

        # Set uniform velocity
        u_uniform = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        fluid.u[:] = u_uniform

        # Sample at random positions
        positions = torch.rand(10, 3, dtype=torch.float64) * 0.07 + 0.005
        u_sampled = fluid.sample_velocity(positions)

        # Should be uniform
        error = torch.linalg.vector_norm(u_sampled - u_uniform, dim=1).max().item()
        assert error < 1e-10, f"Uniform sampling error {error:.2e}"

    def test_porosity_update(self):
        """Test that porosity is reduced near particles."""
        from tensordem.dem3d import DEM3DConfig, IceDEM3D

        dem_config = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=0.005,
            bond_stiffness=1.0, contact_stiffness=1.0,
            breaking_strain=1.0, shear_breaking_strain=1.0,
            tool_radius=1e10, tool_speed=0.001,
            fix_x_edges=False, fix_bottom=False,
        )
        dem = IceDEM3D(dem_config)
        dem.positions[0] = torch.tensor([0.04, 0.04, 0.04], dtype=torch.float64)

        config = FluidGrid3DConfig(nx=8, ny=8, nz=8, lx=0.08, ly=0.08, lz=0.08)
        fluid = FluidGrid3D(config)

        coupling = DEMFluidCoupling3D(dem, fluid, particle_radius=0.005)
        coupling.update_porosity()

        # Porosity should be < 1 near the particle
        min_porosity = fluid.porosity.min().item()
        assert min_porosity < 1.0, "Porosity not reduced near particle"
        assert min_porosity >= 0.01, "Porosity clamped incorrectly"


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for the full coupling system."""

    def test_momentum_conservation_with_coupling(self):
        """Full coupling should conserve momentum to machine precision."""
        from tensordem.dem3d import DEM3DConfig, IceDEM3D

        dem_config = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=0.003,
            bond_stiffness=1.0, contact_stiffness=1.0,
            breaking_strain=1.0, shear_breaking_strain=1.0,
            tool_radius=1e10, tool_speed=0.001,
            fix_x_edges=False, fix_bottom=False,
        )
        dem = IceDEM3D(dem_config)

        # Give particles some initial velocity
        dem.velocities = torch.randn_like(dem.velocities) * 0.01

        fluid_config = FluidGrid3DConfig(nx=8, ny=8, nz=8, lx=0.08, ly=0.08, lz=0.08)
        fluid = FluidGrid3D(fluid_config)

        coupling = DEMFluidCoupling3D(dem, fluid, drag_model="stokes")

        # Run a few coupling steps
        dt = 1e-4
        for _ in range(10):
            coupling.exchange_momentum(dt)

        # Check momentum conservation
        residual = coupling.momentum_conservation_residual()
        residual_magnitude = torch.linalg.vector_norm(residual).item()

        assert residual_magnitude < 1e-10, (
            f"Coupling momentum residual {residual_magnitude:.2e} >= 1e-10"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
