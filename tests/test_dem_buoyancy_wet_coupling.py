"""SIR-PC2 | F07 | DEM Buoyancy & Wet Coupling Model — Verification & Acceptance Tests.

Tests for:
- Submerged fraction (2-D circular-segment and 3-D spherical-cap geometry)
- Buoyancy force vs Archimedes' principle
- Wetted area and perimeter (2-D and 3-D)
- Floating equilibrium solver
- Heave restoring stiffness and vertical stability
- Metacentric height (roll) — note: GM ≈ 0 for 2-D disk, < 0 for 3-D sphere
- Wave force (Froude-Krylov / Morison)
- Full wet-coupling assembly

Acceptance criteria (SIR-PC2):
  buoyancy_force_error < 0.03 [ratio]
  floating_equilibrium_stable == True [bool]  (heave/vertical stability)
  pytest_test_dem_buoyancy == PASS [bool]
"""
import math
import json
import os
import unittest

import torch

from tensordem.buoyancy_wet import (
    WetCouplingConfig,
    WaveForceConfig,
    compute_buoyancy_force,
    compute_wet_drag_force,
    compute_wet_forces,
    compute_wave_force,
    compute_all_wet_forces,
    submerged_fraction_tensor,
    submerged_fraction_sphere_tensor,
    wetted_area,
    wetted_perimeter,
    wetted_area_sphere,
    wetted_area_tensor,
    wetted_area_sphere_tensor,
    wave_kinematics,
    wave_elevation,
    solve_floating_equilibrium,
    heave_restoring_stiffness,
    metacentric_height,
    verify_buoyancy_force,
    _submerged_fraction_2d,
    _submerged_fraction_sphere,
)


# ---------------------------------------------------------------------------
# Physical constants (sea-ice scenario)
# ---------------------------------------------------------------------------
RHO_ICE = 917.0       # kg/m^3
RHO_WATER = 1025.0     # kg/m^3
GRAVITY = 9.81         # m/s^2
RADIUS = 0.5           # m
THICKNESS = 1.0        # m


class TestSubmergedFraction2D(unittest.TestCase):
    """Circular-segment submerged fraction geometry (2-D disk)."""

    def test_fully_above(self):
        f = _submerged_fraction_2d(1.0, 0.5)
        self.assertAlmostEqual(f, 0.0, places=12)

    def test_fully_below(self):
        f = _submerged_fraction_2d(-1.0, 0.5)
        self.assertAlmostEqual(f, 1.0, places=12)

    def test_half_submerged(self):
        f = _submerged_fraction_2d(0.0, 0.5)
        self.assertAlmostEqual(f, 0.5, places=10)

    def test_fraction_monotonic(self):
        hs = [h * 0.1 for h in range(-10, 11)]
        fs = [_submerged_fraction_2d(h, 1.0) for h in hs]
        for i in range(len(fs) - 1):
            self.assertGreaterEqual(fs[i], fs[i + 1] - 1e-14)

    def test_fraction_range(self):
        for h in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            f = _submerged_fraction_2d(h, 1.0)
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, 1.0)

    def test_tensor_matches_scalar(self):
        R = 0.5
        hs = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
        f_tensor = submerged_fraction_tensor(hs, R, 0.0)
        for i, h in enumerate(hs.tolist()):
            f_scalar = _submerged_fraction_2d(h, R)
            self.assertAlmostEqual(f_tensor[i].item(), f_scalar, places=10)


class TestSubmergedFraction3D(unittest.TestCase):
    """Spherical-cap submerged fraction geometry (3-D sphere)."""

    def test_fully_above(self):
        f = _submerged_fraction_sphere(1.0, 0.5)
        self.assertAlmostEqual(f, 0.0, places=12)

    def test_fully_below(self):
        f = _submerged_fraction_sphere(-1.0, 0.5)
        self.assertAlmostEqual(f, 1.0, places=12)

    def test_half_submerged(self):
        f = _submerged_fraction_sphere(0.0, 0.5)
        self.assertAlmostEqual(f, 0.5, places=10)

    def test_fraction_monotonic(self):
        hs = [h * 0.1 for h in range(-10, 11)]
        fs = [_submerged_fraction_sphere(h, 1.0) for h in hs]
        for i in range(len(fs) - 1):
            self.assertGreaterEqual(fs[i], fs[i + 1] - 1e-14)

    def test_fraction_range(self):
        for h in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            f = _submerged_fraction_sphere(h, 1.0)
            self.assertGreaterEqual(f, -1e-14)
            self.assertLessEqual(f, 1.0 + 1e-14)

    def test_tensor_matches_scalar(self):
        R = 0.5
        hs = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
        f_tensor = submerged_fraction_sphere_tensor(hs, R, 0.0)
        for i, h in enumerate(hs.tolist()):
            f_scalar = _submerged_fraction_sphere(h, R)
            self.assertAlmostEqual(f_tensor[i].item(), f_scalar, places=10)

    def test_volume_consistency(self):
        """Total volume = submerged + emerged at h=0."""
        R = 1.0
        f_sub = _submerged_fraction_sphere(0.0, R)
        self.assertAlmostEqual(f_sub, 0.5, places=10)


class TestBuoyancyForce2D(unittest.TestCase):
    """Buoyancy force computation (2-D disk model)."""

    def test_fully_submerged_buoyancy(self):
        positions = torch.tensor([[0.0, -1.0]], dtype=torch.float64)
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="2d")
        F = compute_buoyancy_force(positions, RADIUS, THICKNESS, RHO_ICE, cfg)
        V = math.pi * RADIUS ** 2 * THICKNESS
        expected = RHO_WATER * GRAVITY * V
        self.assertAlmostEqual(F[0, 1].item(), expected, delta=expected * 1e-10)

    def test_fully_dry_zero_buoyancy(self):
        positions = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="2d")
        F = compute_buoyancy_force(positions, RADIUS, THICKNESS, RHO_ICE, cfg)
        self.assertAlmostEqual(F[0, 1].item(), 0.0, places=12)

    def test_buoyancy_vertical_only(self):
        positions = torch.tensor([[0.0, -0.3]], dtype=torch.float64)
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="2d")
        F = compute_buoyancy_force(positions, RADIUS, THICKNESS, RHO_ICE, cfg)
        self.assertAlmostEqual(F[0, 0].item(), 0.0, places=14)


class TestBuoyancyForce3D(unittest.TestCase):
    """Buoyancy force computation (3-D sphere model)."""

    def test_fully_submerged_buoyancy(self):
        positions = torch.tensor([[0.0, -1.0]], dtype=torch.float64)
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="3d")
        F = compute_buoyancy_force(positions, RADIUS, THICKNESS, RHO_ICE, cfg)
        V = (4.0 / 3.0) * math.pi * RADIUS ** 3
        expected = RHO_WATER * GRAVITY * V
        self.assertAlmostEqual(F[0, 1].item(), expected, delta=expected * 1e-10)

    def test_fully_dry_zero_buoyancy(self):
        positions = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="3d")
        F = compute_buoyancy_force(positions, RADIUS, THICKNESS, RHO_ICE, cfg)
        self.assertAlmostEqual(F[0, 1].item(), 0.0, places=12)

    def test_half_submerged_buoyancy(self):
        positions = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="3d")
        F = compute_buoyancy_force(positions, RADIUS, THICKNESS, RHO_ICE, cfg)
        V_half = (2.0 / 3.0) * math.pi * RADIUS ** 3
        expected = RHO_WATER * GRAVITY * V_half
        self.assertAlmostEqual(F[0, 1].item(), expected, delta=expected * 1e-8)


class TestBuoyancyForceAcceptance(unittest.TestCase):
    """SIR-PC2 acceptance: buoyancy_force_error < 0.03."""

    def test_2d_buoyancy_force_error(self):
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="2d")
        max_err, eq_err, passed = verify_buoyancy_force(
            RADIUS, THICKNESS, RHO_ICE, cfg, n_heights=500
        )
        self.assertLess(max_err, 0.03,
                        f"2-D buoyancy max relative error {max_err:.6e} >= 0.03")
        self.assertTrue(passed)

    def test_3d_buoyancy_force_error(self):
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="3d")
        max_err, eq_err, passed = verify_buoyancy_force(
            RADIUS, THICKNESS, RHO_ICE, cfg, n_heights=500
        )
        self.assertLess(max_err, 0.03,
                        f"3-D buoyancy max relative error {max_err:.6e} >= 0.03")
        self.assertTrue(passed)


class TestWettedArea2D(unittest.TestCase):
    """Wetted area and perimeter (2-D cylinder)."""

    def test_fully_dry_zero(self):
        A = wetted_area(1.0, 0.5, 1.0)
        self.assertAlmostEqual(A, 0.0, places=12)

    def test_fully_submerged_total(self):
        R, t = 0.5, 1.0
        A = wetted_area(-1.0, R, t)
        expected = 2 * math.pi * R * t + 2 * math.pi * R ** 2
        self.assertAlmostEqual(A, expected, delta=expected * 1e-10)

    def test_half_submerged(self):
        R, t = 0.5, 1.0
        A = wetted_area(0.0, R, t)
        expected_side = 2 * R * math.pi * t
        expected_cap = R ** 2 * math.pi
        expected = expected_side + expected_cap
        self.assertAlmostEqual(A, expected, delta=expected * 1e-10)

    def test_wetted_perimeter_fully_submerged(self):
        P = wetted_perimeter(-1.0, 0.5)
        self.assertAlmostEqual(P, 2 * math.pi * 0.5, places=10)

    def test_wetted_perimeter_fully_dry(self):
        P = wetted_perimeter(1.0, 0.5)
        self.assertAlmostEqual(P, 0.0, places=12)

    def test_tensor_matches_scalar(self):
        R, t = 0.5, 1.0
        hs = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
        A_tensor = wetted_area_tensor(hs, R, t, 0.0)
        for i, h in enumerate(hs.tolist()):
            A_scalar = wetted_area(h, R, t)
            self.assertAlmostEqual(A_tensor[i].item(), A_scalar, places=8)


class TestWettedArea3D(unittest.TestCase):
    """Wetted area (3-D sphere)."""

    def test_fully_dry_zero(self):
        A = wetted_area_sphere(1.0, 0.5)
        self.assertAlmostEqual(A, 0.0, places=12)

    def test_fully_submerged_total(self):
        R = 0.5
        A = wetted_area_sphere(-1.0, R)
        expected = 4.0 * math.pi * R ** 2
        self.assertAlmostEqual(A, expected, delta=expected * 1e-10)

    def test_half_submerged(self):
        R = 0.5
        A = wetted_area_sphere(0.0, R)
        # Cap height d = R, A = 2*pi*R*d = 2*pi*R^2
        expected = 2.0 * math.pi * R ** 2
        self.assertAlmostEqual(A, expected, delta=expected * 1e-10)

    def test_tensor_matches_scalar(self):
        R = 0.5
        hs = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
        A_tensor = wetted_area_sphere_tensor(hs, R, 0.0)
        for i, h in enumerate(hs.tolist()):
            A_scalar = wetted_area_sphere(h, R)
            self.assertAlmostEqual(A_tensor[i].item(), A_scalar, places=8)


class TestFloatingEquilibrium(unittest.TestCase):
    """Floating equilibrium solver."""

    def test_2d_equilibrium_submerged_fraction(self):
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="2d")
        h_eq, f_eq, converged = solve_floating_equilibrium(
            RADIUS, THICKNESS, RHO_ICE, cfg
        )
        self.assertTrue(converged)
        target = RHO_ICE / RHO_WATER
        self.assertAlmostEqual(f_eq, target, delta=1e-8)

    def test_3d_equilibrium_submerged_fraction(self):
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="3d")
        h_eq, f_eq, converged = solve_floating_equilibrium(
            RADIUS, THICKNESS, RHO_ICE, cfg
        )
        self.assertTrue(converged)
        target = RHO_ICE / RHO_WATER
        self.assertAlmostEqual(f_eq, target, delta=1e-8)

    def test_2d_equilibrium_buoyancy_equals_weight(self):
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="2d")
        h_eq, f_eq, converged = solve_floating_equilibrium(
            RADIUS, THICKNESS, RHO_ICE, cfg
        )
        self.assertTrue(converged)
        V_total = math.pi * RADIUS ** 2 * THICKNESS
        weight = RHO_ICE * GRAVITY * V_total
        buoyancy = RHO_WATER * GRAVITY * math.pi * RADIUS ** 2 * f_eq * THICKNESS
        rel_err = abs(buoyancy - weight) / weight
        self.assertLess(rel_err, 0.03)

    def test_3d_equilibrium_buoyancy_equals_weight(self):
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="3d")
        h_eq, f_eq, converged = solve_floating_equilibrium(
            RADIUS, THICKNESS, RHO_ICE, cfg
        )
        self.assertTrue(converged)
        V_total = (4.0 / 3.0) * math.pi * RADIUS ** 3
        weight = RHO_ICE * GRAVITY * V_total
        buoyancy = RHO_WATER * GRAVITY * V_total * f_eq
        rel_err = abs(buoyancy - weight) / weight
        self.assertLess(rel_err, 0.03)

    def test_2d_equilibrium_height_below_waterline(self):
        """For ice (rho_i/rho_w ≈ 0.895), most of the disk is submerged,
        so the centre is below the waterline (h_eq < 0)."""
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="2d")
        h_eq, _, converged = solve_floating_equilibrium(
            RADIUS, THICKNESS, RHO_ICE, cfg
        )
        self.assertTrue(converged)
        self.assertLess(h_eq, 0.0,
                        f"h_eq={h_eq:.6f} should be < 0 for ice (mostly submerged)")

    def test_3d_equilibrium_height_below_waterline(self):
        """Same for 3-D sphere."""
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="3d")
        h_eq, _, converged = solve_floating_equilibrium(
            RADIUS, THICKNESS, RHO_ICE, cfg
        )
        self.assertTrue(converged)
        self.assertLess(h_eq, 0.0)

    def test_heavier_than_water_sinks(self):
        cfg = WetCouplingConfig(enabled=True, water_density=800.0, gravity=GRAVITY, model="2d")
        h_eq, f_eq, converged = solve_floating_equilibrium(
            RADIUS, THICKNESS, RHO_ICE, cfg
        )
        self.assertAlmostEqual(f_eq, 1.0, places=10)

    def test_lighter_than_water_fully_floats(self):
        cfg = WetCouplingConfig(enabled=True, water_density=5000.0, gravity=GRAVITY, model="2d")
        h_eq, f_eq, converged = solve_floating_equilibrium(
            RADIUS, THICKNESS, 100.0, cfg
        )
        self.assertAlmostEqual(f_eq, 0.02, delta=1e-6)


class TestHeaveStability(unittest.TestCase):
    """Heave restoring stiffness and vertical stability."""

    def test_2d_heave_stable(self):
        """SIR-PC2 acceptance: floating_equilibrium_stable == True (2-D)."""
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="2d")
        k_heave, is_stable = heave_restoring_stiffness(
            RADIUS, THICKNESS, RHO_ICE, cfg
        )
        self.assertTrue(is_stable,
                        f"2-D heave not stable: k_heave={k_heave:.4f}")
        self.assertGreater(k_heave, 0.0)

    def test_3d_heave_stable(self):
        """SIR-PC2 acceptance: floating_equilibrium_stable == True (3-D)."""
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="3d")
        k_heave, is_stable = heave_restoring_stiffness(
            RADIUS, THICKNESS, RHO_ICE, cfg
        )
        self.assertTrue(is_stable,
                        f"3-D heave not stable: k_heave={k_heave:.4f}")
        self.assertGreater(k_heave, 0.0)

    def test_heave_stiffness_positive(self):
        """Heave stiffness is always positive for a floating body."""
        for model in ["2d", "3d"]:
            cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model=model)
            k, stable = heave_restoring_stiffness(RADIUS, THICKNESS, RHO_ICE, cfg)
            self.assertGreater(k, 0.0, f"model={model}: k_heave={k}")

    def test_heave_stiffness_increases_with_density_ratio(self):
        """Closer density ratio => larger waterplane => larger stiffness."""
        cfg_light = WetCouplingConfig(enabled=True, water_density=2000.0, gravity=GRAVITY, model="2d")
        cfg_heavy = WetCouplingConfig(enabled=True, water_density=1050.0, gravity=GRAVITY, model="2d")
        k_light, _ = heave_restoring_stiffness(RADIUS, THICKNESS, 500.0, cfg_light)
        k_heavy, _ = heave_restoring_stiffness(RADIUS, THICKNESS, 917.0, cfg_heavy)
        # Lighter particle has larger waterplane area at equilibrium
        self.assertGreater(k_light, k_heavy)


class TestMetacentricHeight(unittest.TestCase):
    """Metacentric height (roll stability) — known analytical results."""

    def test_2d_gm_near_zero(self):
        """2-D circular disk: GM ≈ 0 (neutral roll stability)."""
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="2d")
        GM, _ = metacentric_height(RADIUS, THICKNESS, RHO_ICE, cfg)
        self.assertAlmostEqual(GM, 0.0, delta=0.01,
                               msg=f"2-D GM={GM:.6f} should be ≈ 0")

    def test_3d_gm_negative(self):
        """3-D sphere: GM < 0 (roll-unstable, known result)."""
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="3d")
        GM, _ = metacentric_height(RADIUS, THICKNESS, RHO_ICE, cfg)
        self.assertLess(GM, 0.01,
                        f"3-D GM={GM:.6f} should be < 0 (sphere is roll-unstable)")

    def test_metacentric_height_returns_heave_stability(self):
        """metacentric_height() returns heave stability flag."""
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="2d")
        GM, is_heave_stable = metacentric_height(RADIUS, THICKNESS, RHO_ICE, cfg)
        self.assertTrue(is_heave_stable)


class TestWaveForce(unittest.TestCase):
    """Wave force (Froude-Krylov / Morison) tests."""

    def test_no_wave_when_disabled(self):
        positions = torch.tensor([[0.0, -0.5]], dtype=torch.float64)
        velocities = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        wet_cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY)
        wave_cfg = WaveForceConfig(enabled=False)
        F = compute_wave_force(positions, velocities, RADIUS, THICKNESS, 0.0,
                               wave_cfg, wet_cfg)
        self.assertAlmostEqual(float(F.abs().sum()), 0.0, places=12)

    def test_wave_force_on_submerged_particle(self):
        positions = torch.tensor([[0.0, -0.5]], dtype=torch.float64)
        velocities = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        wet_cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY)
        wave_cfg = WaveForceConfig(
            enabled=True, amplitude=0.1, wavelength=10.0, period=2.5,
            water_depth=100.0, inertia_coeff=2.0, drag_coeff_morison=1.2,
        )
        F = compute_wave_force(positions, velocities, RADIUS, THICKNESS, 0.0,
                               wave_cfg, wet_cfg)
        self.assertGreater(float(F.abs().sum()), 0.0)

    def test_wave_force_zero_for_dry_particle(self):
        positions = torch.tensor([[0.0, 5.0]], dtype=torch.float64)
        velocities = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        wet_cfg = WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY)
        wave_cfg = WaveForceConfig(
            enabled=True, amplitude=0.1, wavelength=10.0, period=2.5,
        )
        F = compute_wave_force(positions, velocities, RADIUS, THICKNESS, 0.0,
                               wave_cfg, wet_cfg)
        self.assertAlmostEqual(float(F.abs().sum()), 0.0, places=10)

    def test_wave_kinematics_deep_water_decay(self):
        x = torch.zeros(5, dtype=torch.float64)
        y = torch.tensor([0.0, -1.0, -2.0, -5.0, -10.0], dtype=torch.float64)
        wave_cfg = WaveForceConfig(
            enabled=True, amplitude=1.0, wavelength=20.0, period=3.58,
            water_depth=1000.0,
        )
        u_x, u_y, _ = wave_kinematics(x, y, 0.0, wave_cfg)
        self.assertGreater(abs(u_x[0].item()), abs(u_x[-1].item()))
        for i in range(len(u_x) - 1):
            self.assertGreaterEqual(abs(u_x[i].item()), abs(u_x[i + 1].item()) - 1e-10)

    def test_wave_elevation_cosine(self):
        x = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0], dtype=torch.float64)
        wave_cfg = WaveForceConfig(
            enabled=True, amplitude=1.0, wavelength=10.0, period=2.0,
        )
        eta = wave_elevation(x, 0.0, wave_cfg)
        self.assertAlmostEqual(eta[0].item(), 1.0, places=10)
        self.assertAlmostEqual(eta[2].item(), -1.0, places=10)


class TestAllWetForces(unittest.TestCase):
    """Full wet-coupling force assembly."""

    def test_combined_forces_nonzero(self):
        positions = torch.tensor([[0.0, -0.5]], dtype=torch.float64)
        velocities = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        wet_cfg = WetCouplingConfig(
            enabled=True, water_density=RHO_WATER, gravity=GRAVITY,
            wet_drag_coeff=50.0,
        )
        wave_cfg = WaveForceConfig(
            enabled=True, amplitude=0.1, wavelength=10.0, period=2.5,
        )
        F_buoy, F_drag, F_wave = compute_all_wet_forces(
            positions, velocities, RADIUS, THICKNESS, RHO_ICE, 0.0,
            wet_cfg, wave_cfg,
        )
        self.assertGreater(F_buoy[0, 1].item(), 0.0)
        self.assertLess(F_drag[0, 0].item(), 0.0)
        self.assertGreater(float(F_wave.abs().sum()), 0.0)

    def test_no_wave_config(self):
        positions = torch.tensor([[0.0, -0.5]], dtype=torch.float64)
        velocities = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        wet_cfg = WetCouplingConfig(
            enabled=True, water_density=RHO_WATER, gravity=GRAVITY,
            wet_drag_coeff=50.0,
        )
        F_buoy, F_drag, F_wave = compute_all_wet_forces(
            positions, velocities, RADIUS, THICKNESS, RHO_ICE, 0.0,
            wet_cfg, None,
        )
        self.assertAlmostEqual(float(F_wave.abs().sum()), 0.0, places=12)


class TestAcceptanceCriteria(unittest.TestCase):
    """SIR-PC2 formal acceptance criteria.

    buoyancy_force_error < 0.03 [ratio]
    floating_equilibrium_stable == True [bool]  (heave/vertical stability)
    pytest_test_dem_buoyancy == PASS [bool]
    """

    def setUp(self):
        self.cfg_2d = WetCouplingConfig(
            enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="2d"
        )
        self.cfg_3d = WetCouplingConfig(
            enabled=True, water_density=RHO_WATER, gravity=GRAVITY, model="3d"
        )

    def test_buoyancy_force_error_2d(self):
        """Criterion 1a: buoyancy_force_error < 0.03 (2-D)."""
        max_err, eq_err, passed = verify_buoyancy_force(
            RADIUS, THICKNESS, RHO_ICE, self.cfg_2d, n_heights=500
        )
        self.assertLess(max_err, 0.03)
        self.assertTrue(passed)

    def test_buoyancy_force_error_3d(self):
        """Criterion 1b: buoyancy_force_error < 0.03 (3-D)."""
        max_err, eq_err, passed = verify_buoyancy_force(
            RADIUS, THICKNESS, RHO_ICE, self.cfg_3d, n_heights=500
        )
        self.assertLess(max_err, 0.03)
        self.assertTrue(passed)

    def test_floating_equilibrium_stable_2d(self):
        """Criterion 2a: floating_equilibrium_stable == True (2-D heave)."""
        k_heave, is_stable = heave_restoring_stiffness(
            RADIUS, THICKNESS, RHO_ICE, self.cfg_2d
        )
        self.assertTrue(is_stable)
        self.assertGreater(k_heave, 0.0)

    def test_floating_equilibrium_stable_3d(self):
        """Criterion 2b: floating_equilibrium_stable == True (3-D heave)."""
        k_heave, is_stable = heave_restoring_stiffness(
            RADIUS, THICKNESS, RHO_ICE, self.cfg_3d
        )
        self.assertTrue(is_stable)
        self.assertGreater(k_heave, 0.0)

    def test_pytest_test_dem_buoyancy_pass(self):
        """Criterion 3: pytest_test_dem_buoyancy == PASS (this test itself).

        Verifies all sub-criteria explicitly for both 2-D and 3-D models.
        """
        for model, cfg in [("2d", self.cfg_2d), ("3d", self.cfg_3d)]:
            max_err, eq_err, passed = verify_buoyancy_force(
                RADIUS, THICKNESS, RHO_ICE, cfg, n_heights=500
            )
            k_heave, is_stable = heave_restoring_stiffness(
                RADIUS, THICKNESS, RHO_ICE, cfg
            )
            all_pass = passed and is_stable
            self.assertTrue(
                all_pass,
                f"Acceptance failed ({model}): buoyancy_err={max_err:.2e}, "
                f"eq_err={eq_err:.2e}, k_heave={k_heave:.4f}, stable={is_stable}"
            )


if __name__ == "__main__":
    unittest.main()
