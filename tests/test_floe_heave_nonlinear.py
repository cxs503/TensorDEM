"""Finite-amplitude heave verification for the 2-D floe buoyancy model.

The linear heave anchors of ``docs/ice_floe_physics_validation.md`` were closed
earlier, but that document records the nonlinear period as *not verified*.  This
module checks the reference supplied by
:mod:`tensordem.floe_heave_nonlinear` (closed-form disk geometry + energy
quadrature) and then checks the shipped ``RigidHeaveFloe`` /
``compute_buoyancy_force`` path against it.

Every anchor is paired with a control that must fail, so that none of them can
silently degrade into a tautology:

* the linear period must *not* reproduce the measured period at finite
  amplitude (otherwise the nonlinear reference would be redundant);
* a saturated ``I_A`` (the form that ignores linear growth below full
  submergence) must fail the derivative test;
* a stubbed-out buoyancy force must produce no oscillation at all.
"""
from __future__ import annotations

import math
import unittest
from unittest import mock

import torch

torch.set_num_threads(1)

from tensordem.buoyancy_wet import WetCouplingConfig, submerged_fraction_tensor
from tensordem.floe_buoyancy import (
    RigidHeaveFloe,
    discrete_buoyancy_profile,
    discrete_stiffness,
    hex_lattice_floe,
    mean_period,
    upward_crossing_periods,
)
from tensordem.floe_heave_nonlinear import (
    HeaveOscillator,
    exact_heave_period,
    gauss_legendre,
    mean_segment_area,
    measure_heave_period,
    segment_area,
    segment_area_antiderivative,
)

RADIUS = 0.10
THICKNESS = 1.0
DENSITY_ICE = 917.0


def _cfg(**kw) -> WetCouplingConfig:
    base = dict(enabled=True, water_density=1025.0, gravity=9.81,
                wet_drag_coeff=0.0, model="2d")
    base.update(kw)
    return WetCouplingConfig(**base)


def _oscillator(n_particles: int = 19) -> HeaveOscillator:
    offsets = hex_lattice_floe(RADIUS, n_particles)
    return HeaveOscillator.from_config(offsets[:, 1].tolist(), RADIUS,
                                       THICKNESS, DENSITY_ICE, _cfg())


class GeometryTests(unittest.TestCase):
    """The closed-form disk geometry, independent of the production formula."""

    def test_segment_area_endpoints_and_half(self):
        self.assertAlmostEqual(segment_area(-RADIUS, RADIUS), math.pi * RADIUS ** 2, places=15)
        self.assertAlmostEqual(segment_area(RADIUS, RADIUS), 0.0, places=15)
        self.assertAlmostEqual(segment_area(0.0, RADIUS), 0.5 * math.pi * RADIUS ** 2, places=15)
        # below full submergence the area saturates at the full disk
        for h in (-1.2 * RADIUS, -3.0 * RADIUS, -50.0 * RADIUS):
            self.assertAlmostEqual(segment_area(h, RADIUS), math.pi * RADIUS ** 2, places=15)

    def test_matches_production_submerged_fraction(self):
        for i in range(41):
            h = (-1.5 + 3.0 * i / 40.0) * RADIUS
            frac = float(submerged_fraction_tensor(torch.tensor([h], dtype=torch.float64),
                                                   RADIUS, 0.0)[0])
            self.assertAlmostEqual(segment_area(h, RADIUS) / (math.pi * RADIUS ** 2), frac,
                                   places=13)

    def test_antiderivative_derivative_matches_area(self):
        """dI_A/dh = A(h) over the whole range, including below full submergence."""
        step = 1e-6 * RADIUS
        worst = 0.0
        for i in range(61):
            h = (-1.6 + 3.2 * i / 60.0) * RADIUS
            fd = (segment_area_antiderivative(h + step, RADIUS)
                  - segment_area_antiderivative(h - step, RADIUS)) / (2.0 * step)
            scale = max(abs(segment_area(h, RADIUS)), 1e-3 * math.pi * RADIUS ** 2)
            worst = max(worst, abs(fd - segment_area(h, RADIUS)) / scale)
        self.assertLess(worst, 1e-7)

    def test_antiderivative_endpoints(self):
        self.assertAlmostEqual(segment_area_antiderivative(-RADIUS, RADIUS), 0.0, places=15)
        self.assertAlmostEqual(segment_area_antiderivative(RADIUS, RADIUS),
                               math.pi * RADIUS ** 3, places=15)
        # above the disk the sweep is constant, below it grows linearly at pi R^2
        self.assertAlmostEqual(segment_area_antiderivative(2.0 * RADIUS, RADIUS),
                               math.pi * RADIUS ** 3, places=15)
        self.assertAlmostEqual(segment_area_antiderivative(-3.0 * RADIUS, RADIUS),
                               math.pi * RADIUS ** 2 * (-2.0 * RADIUS), places=15)

    def test_saturated_antiderivative_control_fails(self):
        """Negative control: the saturated form must violate dI_A/dh = A(h)."""

        def saturated(h: float) -> float:
            return segment_area_antiderivative(min(max(h, -RADIUS), RADIUS), RADIUS)

        h = -2.0 * RADIUS
        step = 1e-6 * RADIUS
        fd = (saturated(h + step) - saturated(h - step)) / (2.0 * step)
        rel = abs(fd - segment_area(h, RADIUS)) / segment_area(h, RADIUS)
        self.assertGreater(rel, 0.5, "saturated antiderivative should be caught")

    def test_mean_area_short_span_matches_exact(self):
        for h in (-0.4 * RADIUS, -0.05 * RADIUS, 0.0, 0.4 * RADIUS):
            for d in (1e-9 * RADIUS, 1e-5 * RADIUS, 1e-3 * RADIUS):
                fast = mean_segment_area(h - d, h, RADIUS)
                exact = (segment_area_antiderivative(h, RADIUS)
                         - segment_area_antiderivative(h - d, RADIUS)) / d
                self.assertAlmostEqual(fast, exact, delta=1e-7 * max(exact, math.pi * RADIUS ** 2))

    def test_gauss_legendre_integrates_polynomials(self):
        nodes, weights = gauss_legendre(8)
        for k in (0, 1, 5, 15):  # exact up to degree 2n-1 = 15
            approx = sum(w * x ** k for x, w in zip(nodes, weights))
            exact = 0.0 if k % 2 else 2.0 / (k + 1)
            self.assertAlmostEqual(approx, exact, places=13)


class ForceTests(unittest.TestCase):
    """The analytic force/potential must agree with the shipped kernel."""

    def setUp(self):
        self.osc = _oscillator()
        self.offsets = hex_lattice_floe(RADIUS, 19)
        self.cfg = _cfg()
        self.y_eq = self.osc.equilibrium()

    def test_aggregate_buoyancy_matches_shipped_kernel(self):
        for y in (self.y_eq - 0.3 * RADIUS, self.y_eq, self.y_eq + 0.25 * RADIUS,
                  self.y_eq - 2.0 * RADIUS):
            production = discrete_buoyancy_profile(self.offsets, RADIUS, THICKNESS,
                                                   DENSITY_ICE, self.cfg, y)
            analytic = self.osc.buoyancy(y)
            self.assertLess(abs(analytic - production) / production, 1e-12)

    def test_equilibrium_zeroes_the_force(self):
        residual = self.osc.force(self.y_eq) / (self.osc.mass * self.osc.gravity)
        self.assertLess(abs(residual), 1e-12)

    def test_equilibrium_matches_discrete_profile(self):
        production = discrete_buoyancy_profile(self.offsets, RADIUS, THICKNESS,
                                               DENSITY_ICE, self.cfg, self.y_eq)
        self.assertLess(abs(production / self.osc.weight - 1.0), 1e-10)

    def test_stiffness_matches_production_closed_form(self):
        production = discrete_stiffness(self.offsets, RADIUS, THICKNESS, DENSITY_ICE,
                                       self.cfg, self.y_eq)
        analytic = self.osc.curve_coefficients(self.y_eq)[0]
        self.assertLess(abs(analytic - production) / production, 1e-12)

    def test_curve_coefficients_match_finite_differences(self):
        """k1, k2, k3 recovered from central differences of the net force."""
        k1, k2, k3 = self.osc.curve_coefficients(self.y_eq)
        d = 1e-4 * RADIUS

        def f(x: float) -> float:
            return self.osc.force(self.y_eq + x)

        # F(x) = -k1 x - k2 x^2 - k3 x^3
        fd1 = -(f(d) - f(-d)) / (2.0 * d)
        fd2 = -(f(d) - 2.0 * f(0.0) + f(-d)) / (2.0 * d * d)
        fd3 = -(f(2 * d) - 2 * f(d) + 2 * f(-d) - f(-2 * d)) / (12.0 * d ** 3)
        self.assertLess(abs(fd1 / k1 - 1.0), 1e-8)
        self.assertLess(abs(fd2 / k2 - 1.0), 1e-5)
        self.assertLess(abs(fd3 / k3 - 1.0), 1e-4)

    def test_well_depth_matches_potential_difference(self):
        for x in (-0.3 * RADIUS, -0.05 * RADIUS, 0.0, 0.05 * RADIUS):
            span = 0.2 * RADIUS
            stably = self.osc.well_depth(x, self.y_eq, x + span)
            direct = (self.osc.relative_potential(x + span, self.y_eq)
                      - self.osc.relative_potential(x, self.y_eq))
            self.assertAlmostEqual(stably, direct, delta=1e-6 * abs(direct))


class QuadratureTests(unittest.TestCase):
    """The nonlinear reference itself: limits, convergence and monotonicity."""

    def setUp(self):
        self.osc = _oscillator()
        self.y_eq = self.osc.equilibrium()
        self.t0 = self.osc.linear_period(self.y_eq)

    def test_linear_limit_reproduces_small_amplitude_period(self):
        ref = exact_heave_period(self.osc, 1e-4 * RADIUS, self.y_eq)
        self.assertLess(abs(ref.period / self.t0 - 1.0), 1e-6)

    def test_series_matches_quadrature_at_small_amplitude(self):
        ref = exact_heave_period(self.osc, 0.01 * RADIUS, self.y_eq)
        self.assertLess(abs(ref.series_period / ref.period - 1.0), 1e-6)

    def test_quadrature_converges_with_node_count(self):
        coarse = exact_heave_period(self.osc, 0.3 * RADIUS, self.y_eq, n_nodes=60).period
        fine = exact_heave_period(self.osc, 0.3 * RADIUS, self.y_eq, n_nodes=300).period
        self.assertLess(abs(fine / coarse - 1.0), 1e-9)

    def test_period_grows_with_amplitude(self):
        periods = [exact_heave_period(self.osc, a * RADIUS, self.y_eq).period
                   for a in (0.05, 0.1, 0.2, 0.3, 0.5)]
        self.assertEqual(periods, sorted(periods))
        # and the growth is large enough to be measurable, not numerical noise
        self.assertGreater(periods[-1] / periods[0] - 1.0, 1e-2)

    def test_well_is_asymmetric(self):
        ref = exact_heave_period(self.osc, 0.3 * RADIUS, self.y_eq)
        self.assertNotAlmostEqual(abs(ref.x_lo) / ref.x_hi, 1.0, places=2)

    def test_energy_above_the_well_floor_is_positive(self):
        ref = exact_heave_period(self.osc, 0.2 * RADIUS, self.y_eq)
        gap = self.osc.well_depth(0.0, self.y_eq, ref.x_hi)
        self.assertAlmostEqual(gap, ref.energy, delta=1e-9 * ref.energy)


class SolverTests(unittest.TestCase):
    """The shipped integrator against the exact nonlinear reference."""

    def setUp(self):
        self.osc = _oscillator()
        self.cfg = _cfg()
        self.y_eq = self.osc.equilibrium()

    def test_solver_matches_exact_finite_amplitude_period(self):
        amplitude = 0.3 * RADIUS
        ref = exact_heave_period(self.osc, amplitude, self.y_eq)
        measured = measure_heave_period(self.osc, self.cfg, amplitude, 2e-3,
                                        n_cycles=4, skip_cycles=1, y_eq=self.y_eq)
        self.assertLess(abs(measured.period / ref.period - 1.0), 1e-4)
        # cycle-to-cycle spread: the measured oscillation is really periodic
        self.assertLess(max(measured.periods) - min(measured.periods), 1e-6 * ref.period)

    def test_linear_reference_is_rejected_at_finite_amplitude(self):
        """Anti-tautology: linear theory must *not* explain the measurement."""
        amplitude = 0.3 * RADIUS
        ref = exact_heave_period(self.osc, amplitude, self.y_eq)
        measured = measure_heave_period(self.osc, self.cfg, amplitude, 2e-3,
                                        n_cycles=4, skip_cycles=1, y_eq=self.y_eq)
        linear_error = abs(measured.period / ref.linear_period - 1.0)
        nonlinear_error = abs(measured.period / ref.period - 1.0)
        self.assertGreater(linear_error, 20.0 * nonlinear_error)
        self.assertGreater(linear_error, 1e-3)

    def test_integrator_error_is_second_order_in_dt(self):
        amplitude = 0.2 * RADIUS
        ref = exact_heave_period(self.osc, amplitude, self.y_eq)
        err = []
        for dt in (4e-3, 2e-3):
            m = measure_heave_period(self.osc, self.cfg, amplitude, dt,
                                     n_cycles=4, skip_cycles=1, y_eq=self.y_eq)
            err.append(abs(m.period / ref.period - 1.0))
        ratio = err[0] / err[1]
        self.assertGreater(ratio, 3.0)
        self.assertLess(ratio, 6.0)

    def test_stubbed_buoyancy_produces_no_oscillation(self):
        """Negative control: with buoyancy forced to zero the anchor cannot pass."""
        offsets = hex_lattice_floe(RADIUS, 19)
        floe = RigidHeaveFloe(offsets, RADIUS, THICKNESS, DENSITY_ICE, self.cfg,
                              2e-3, y0=self.y_eq + 0.2 * RADIUS)
        with mock.patch("tensordem.floe_buoyancy.compute_buoyancy_force",
                        side_effect=lambda positions, *a, **k: torch.zeros_like(positions)):
            traj = floe.run(600)
        periods = upward_crossing_periods(traj.t, traj.y, self.y_eq)
        self.assertEqual(periods, [])
        with self.assertRaises(ValueError):
            mean_period(periods)


if __name__ == "__main__":
    unittest.main()