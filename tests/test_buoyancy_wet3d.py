"""Verification of the 3-D spherical-particle buoyancy kernel.

Every assertion is scored against an *independent* route, not against the kernel
itself:

* geometry  -> 20 001-point Simpson quadrature of the circular cross-section,
                an independent algebraic factorisation, and the waterplane
                derivative identity ``-dV/dh = pi (R^2 - h^2)``;
* equilibrium -> the polynomial root of ``u^3 - 3u + 2 - 4r = 0`` (numpy) and the
                Archimedes force balance ``F_b = rho_p g V``;
* dynamics -> a 1-DOF integration whose steady state and oscillation period are
                compared with the analytic draft and the analytic frequency
                ``omega = sqrt(rho_w g A_wp / m)``;
* cross-dimension misuse -> the 2-D disk kernel of the SI-D module, quantified
                against the 3-D sphere (this is the trap the module documents).
"""
import math
import pathlib
import sys
import unittest

import numpy as np
import torch

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # keep the local tree importable
    sys.path.insert(0, str(_SRC))

from tensordem.buoyancy_wet import WetCouplingConfig, _submerged_fraction_2d  # noqa: E402
from tensordem.buoyancy_wet3d import (  # noqa: E402
    compute_buoyancy_force_3d,
    compute_wet_drag_force_3d,
    compute_wet_forces_3d,
    equilibrium_centre_height,
    equilibrium_draft,
    hydrostatic_stiffness_sphere,
    submerged_fraction_sphere,
    submerged_fraction_sphere_factored,
    submerged_fraction_sphere_tensor,
    submerged_volume_sphere,
    waterplane_area_sphere,
)

R = 0.05
RHO_W = 1025.0
RHO_ICE = 917.0
G = 9.81


def simpson_volume(h, R=R, n=20000):
    """Independent submerged volume.

    In the centre-relative coordinate ``q = z - h`` the free surface sits at
    ``q = -h``, so the submerged cap is ``q in [-R, -h]``.
    """
    z = np.linspace(-R, -h, n + 1)
    f = np.pi * (R ** 2 - z ** 2)
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float(np.sum(w * f) * (z[1] - z[0]) / 3.0)


class GeometryKernelTests(unittest.TestCase):
    def test_endpoints_and_half(self):
        self.assertAlmostEqual(submerged_fraction_sphere(R, R), 0.0, places=15)
        self.assertAlmostEqual(submerged_fraction_sphere(-R, R), 1.0, places=15)
        self.assertAlmostEqual(submerged_fraction_sphere(0.0, R), 0.5, places=15)
        self.assertAlmostEqual(submerged_volume_sphere(R, R), 0.0, places=15)
        self.assertAlmostEqual(
            submerged_volume_sphere(-R, R), 4.0 / 3.0 * math.pi * R ** 3, places=15
        )

    def test_outside_range_is_clamped(self):
        self.assertEqual(submerged_fraction_sphere(3.0 * R, R), 0.0)
        self.assertEqual(submerged_fraction_sphere(-3.0 * R, R), 1.0)

    def test_two_algebraic_factorisations_agree(self):
        for frac in np.linspace(-1.0, 1.0, 2001):
            h = float(frac) * R
            self.assertAlmostEqual(
                submerged_fraction_sphere(h, R),
                submerged_fraction_sphere_factored(h, R),
                places=12,
                msg=f"factorisations differ at h/R={frac}",
            )

    def test_matches_simpson_quadrature(self):
        for frac in np.linspace(-0.999, 0.999, 41):
            h = float(frac) * R
            v = submerged_volume_sphere(h, R)
            vq = simpson_volume(h)
            self.assertLess(abs(v - vq) / max(vq, 1e-30), 1e-8,
                            msg=f"quadrature mismatch at h/R={frac}")

    def test_derivative_equals_waterplane_area(self):
        dh = 1e-9 * R
        for frac in np.linspace(-0.95, 0.95, 39):
            h = float(frac) * R
            dVdh = -(submerged_volume_sphere(h + dh, R)
                     - submerged_volume_sphere(h - dh, R)) / (2 * dh)
            self.assertAlmostEqual(dVdh / waterplane_area_sphere(h, R), 1.0,
                                   delta=1e-6, msg=f"at h/R={frac}")

    def test_monotone_decreasing(self):
        hs = np.linspace(-R, R, 5001)
        f = np.array([submerged_fraction_sphere(float(h), R) for h in hs])
        self.assertTrue(np.all(np.diff(f) < 0))

    def test_tensor_route_matches_scalar(self):
        hs = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64) * R
        got = submerged_fraction_sphere_tensor(hs, R)
        want = torch.tensor([submerged_fraction_sphere(float(h), R) for h in hs],
                            dtype=torch.float64)
        self.assertLess(float(torch.max(torch.abs(got - want))), 1e-14)
        self.assertTrue(bool(torch.all(got >= 0.0)) and bool(torch.all(got <= 1.0)))

    def test_water_level_offset(self):
        y = torch.tensor([0.0, 0.0], dtype=torch.float64)
        a = submerged_fraction_sphere_tensor(y, R, water_level=0.0)
        b = submerged_fraction_sphere_tensor(y + 0.5 * R, R, water_level=0.5 * R)
        self.assertLess(float(torch.max(torch.abs(a - b))), 1e-15)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            submerged_volume_sphere(0.0, 0.0)
        with self.assertRaises(ValueError):
            submerged_volume_sphere(float("nan"), R)


class ArchimedesEquilibriumTests(unittest.TestCase):
    def test_fraction_residual_and_polynomial_root(self):
        for rho_p in (330.0, 500.0, 700.0, 850.0, RHO_ICE, 1000.0):
            r = rho_p / RHO_W
            h_eq = equilibrium_centre_height(R, rho_p, RHO_W)
            self.assertAlmostEqual(submerged_fraction_sphere(h_eq, R), r, places=12)
            roots = np.roots([1.0, 0.0, -3.0, 2.0 - 4.0 * r])
            real = [float(np.real(x)) for x in roots
                    if abs(float(np.imag(x))) < 1e-9 and abs(float(np.real(x))) <= 1.0]
            self.assertTrue(real, msg=f"no root in [-1,1] for r={r}")
            u_poly = min(real, key=lambda u: abs(u * R - h_eq))
            self.assertLess(abs(u_poly * R - h_eq) / R, 1e-8)

    def test_archimedes_force_balance(self):
        for rho_p in (500.0, 917.0):
            h_eq = equilibrium_centre_height(R, rho_p, RHO_W)
            f_b = RHO_W * G * submerged_volume_sphere(h_eq, R)
            weight = rho_p * G * 4.0 / 3.0 * math.pi * R ** 3
            self.assertLess(abs(f_b - weight) / weight, 1e-12)

    def test_ice_equilibrium_is_physically_sane(self):
        h_eq = equilibrium_centre_height(R, RHO_ICE, RHO_W)
        self.assertLess(h_eq, 0.0)                      # centre below the surface
        self.assertAlmostEqual(submerged_fraction_sphere(h_eq, R), RHO_ICE / RHO_W,
                               places=12)
        self.assertAlmostEqual(equilibrium_draft(R, RHO_ICE, RHO_W), R - h_eq, places=15)
        self.assertGreater(equilibrium_draft(R, RHO_ICE, RHO_W), R)  # > half immersed

    def test_denser_than_water_cannot_float(self):
        with self.assertRaises(ValueError):
            equilibrium_centre_height(R, RHO_W + 1.0, RHO_W)

    def test_equilibrium_matches_tensor_force(self):
        """Force balance must also hold through the public tensor path."""
        h_eq = equilibrium_centre_height(R, RHO_ICE, RHO_W)
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_W, gravity=G)
        pos = torch.tensor([[0.0, h_eq, 0.0]], dtype=torch.float64)
        f_b = compute_buoyancy_force_3d(pos, R, cfg)
        weight = RHO_ICE * G * 4.0 / 3.0 * math.pi * R ** 3
        self.assertLess(abs(float(f_b[0, 1]) - weight) / weight, 1e-12)
        self.assertEqual(float(f_b[0, 0]), 0.0)
        self.assertEqual(float(f_b[0, 2]), 0.0)


class ForceTests(unittest.TestCase):
    def test_disabled_gives_zero(self):
        pos = torch.zeros((3, 3), dtype=torch.float64)
        f = compute_buoyancy_force_3d(pos, R, WetCouplingConfig())
        self.assertLess(float(torch.max(torch.abs(f))), 1e-30)

    def test_fully_submerged_magnitude(self):
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_W, gravity=G,
                               water_level=0.0)
        pos = torch.tensor([[0.0, -3.0 * R, 0.0]], dtype=torch.float64)
        f = compute_buoyancy_force_3d(pos, R, cfg)
        want = RHO_W * G * 4.0 / 3.0 * math.pi * R ** 3
        self.assertLess(abs(float(f[0, 1]) - want) / want, 1e-14)

    def test_fully_dry_gives_zero(self):
        cfg = WetCouplingConfig(enabled=True, water_density=RHO_W, gravity=G)
        pos = torch.tensor([[0.0, 3.0 * R, 0.0]], dtype=torch.float64)
        f = compute_buoyancy_force_3d(pos, R, cfg)
        self.assertEqual(float(f[0, 1]), 0.0)

    def test_wet_drag_opposes_velocity_and_scales_with_submergence(self):
        cfg = WetCouplingConfig(enabled=True, wet_drag_coeff=2.0)
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 5.0 * R, 0.0]], dtype=torch.float64)
        vel = torch.tensor([[0.3, -0.4, 0.0], [0.3, -0.4, 0.0]], dtype=torch.float64)
        f = compute_wet_drag_force_3d(pos, vel, R, cfg)
        self.assertLess(float(f[0, 0]), 0.0)
        self.assertGreater(float(f[0, 1]), 0.0)
        self.assertEqual(float(f[1, 0]), 0.0)  # dry particle: no wet drag
        # half-submerged particle at centre height 0 -> f=0.5
        self.assertAlmostEqual(float(f[0, 0]), -2.0 * 0.5 * 0.3, places=14)

    def test_shape_validation(self):
        cfg = WetCouplingConfig(enabled=True)
        with self.assertRaises(ValueError):
            compute_buoyancy_force_3d(torch.zeros((4, 2)), R, cfg)
        with self.assertRaises(ValueError):
            compute_wet_drag_force_3d(torch.zeros((4, 3)), torch.zeros((4, 2)), R, cfg)
        b, d = compute_wet_forces_3d(torch.zeros((2, 3)), torch.zeros((2, 3)), R, cfg)
        self.assertEqual(tuple(b.shape), (2, 3))
        self.assertEqual(tuple(d.shape), (2, 3))


class VerticalDynamicsTests(unittest.TestCase):
    """One-dimensional floating-body dynamics against the analytic draft/period."""

    def _relax(self, rho_p=RHO_ICE, drag=2.0, dt=2e-4, t_end=4.0):
        volume = 4.0 / 3.0 * math.pi * R ** 3
        m = rho_p * volume
        k = RHO_W * G * volume

        def accel(h, v):
            f_sub = submerged_fraction_sphere(h, R)
            return (k * f_sub - drag * f_sub * v - m * G) / m

        h, v = -R, 0.0
        for _ in range(int(round(t_end / dt))):
            v += accel(h, v) * dt
            h += v * dt
        return h, m, k

    def test_relaxes_to_analytic_draft(self):
        h, m, _ = self._relax()
        h_eq = equilibrium_centre_height(R, RHO_ICE, RHO_W)
        self.assertLess(abs(h - h_eq) / R, 3e-3)   # < 0.3 % of a radius

    def test_oscillation_period_matches_analytic(self):
        volume = 4.0 / 3.0 * math.pi * R ** 3
        m = RHO_ICE * volume
        k = RHO_W * G * volume
        h_eq = equilibrium_centre_height(R, RHO_ICE, RHO_W)
        omega = math.sqrt(hydrostatic_stiffness_sphere(h_eq, R, RHO_W, G) / m)
        dt = 2e-5
        h, v = h_eq + 0.01 * R, 0.0
        prev, crossings = h, []
        for i in range(int(round(2.5 * 2 * math.pi / omega / dt))):
            v += (k * submerged_fraction_sphere(h, R) - m * G) / m * dt
            h += v * dt
            if prev > h_eq >= h:
                crossings.append((i + 1) * dt)
            prev = h
        self.assertGreaterEqual(len(crossings), 2)
        period = crossings[-1] - crossings[-2]
        self.assertLess(abs(period - 2 * math.pi / omega) / (2 * math.pi / omega), 0.03)


class CrossDimensionTrapTests(unittest.TestCase):
    """The 2-D disk kernel must not be reused for spheres — quantify the error.

    The error is strongly asymmetric: it is large for shallowly immersed
    particles (small submerged fraction) and small for deeply immersed ones,
    because both kernels are normalised to their own total measure.
    """

    def test_shallow_immersion_error_exceeds_gate(self):
        """h = +R/2 (quarter-deep cap): 2-D area fraction is 25.1 % too high."""
        h = 0.5 * R
        f2 = _submerged_fraction_2d(h, R)
        f3 = submerged_fraction_sphere(h, R)
        rel = abs(f2 - f3) / f3
        self.assertGreater(rel, 0.15, msg=f"reuse error only {rel:.3f}")
        self.assertAlmostEqual(rel, 0.2512, places=3)

    def test_deep_immersion_error_is_small(self):
        """h = -R/2: the same kernels agree to ~4.7 % on the submerged fraction."""
        h = -0.5 * R
        rel = abs(_submerged_fraction_2d(h, R) - submerged_fraction_sphere(h, R)) \
            / submerged_fraction_sphere(h, R)
        self.assertLess(rel, 0.06)
        self.assertGreater(rel, 0.04)

    def test_known_half_submerged_values(self):
        self.assertAlmostEqual(_submerged_fraction_2d(0.5 * R, R), 0.1955011, places=6)
        self.assertAlmostEqual(submerged_fraction_sphere(0.5 * R, R), 0.15625, places=12)

    def test_equilibrium_draft_error_from_reusing_2d_kernel(self):
        """Engineering consequence: the ice draft would be ~4.7 % too deep."""
        def f2d(h):
            return _submerged_fraction_2d(h, R)

        r = RHO_ICE / RHO_W
        lo, hi = -R, R
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f2d(mid) > r:
                lo = mid
            else:
                hi = mid
        h2d = 0.5 * (lo + hi)
        h3d = equilibrium_centre_height(R, RHO_ICE, RHO_W)
        draft_err = abs((R - h2d) - (R - h3d)) / (R - h3d)
        self.assertLess(draft_err, 0.15)      # below the platform gate
        self.assertGreater(draft_err, 0.03)   # but above a 3 % tolerance
        self.assertAlmostEqual(draft_err, 0.0491, places=4)


if __name__ == "__main__":
    unittest.main()