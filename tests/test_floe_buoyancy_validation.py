"""Validation tests for the buoyancy / wet-drag floe model (``buoyancy_wet``).

Anchors (all analytic, no fitted constants):
  * submerged-fraction geometry vs an independent per-column quadrature;
  * static Archimedes draft f_sub = rho_i / rho_w, including the trap that the
    "thin-slab" linearisation h = -(1 - rho_i/rho_w) R is wrong by >50% for a disk;
  * heave natural period T = 2 pi sqrt(m/K), K = rho_w g t chord(h_eq);
  * wet-drag damping ratio zeta = C f*/(2 m omega) via log decrement;
  * energy drift of the undamped integrator;
  * discrete lattice floe: aggregate buoyancy balance and the closed-form
    waterplane stiffness.

Tolerances are the acceptance gates, not best-case numbers.
"""
from __future__ import annotations

import math
import unittest

import torch

torch.set_num_threads(1)

from tensordem.buoyancy_wet import WetCouplingConfig, submerged_fraction_tensor
from tensordem.floe_buoyancy import (
    RigidHeaveFloe,
    analytic_heave_period,
    body_envelope,
    discrete_buoyancy_profile,
    discrete_stiffness,
    equilibrium_center_height,
    hex_lattice_floe,
    linear_heave_stiffness,
    log_decrement_damping_ratio,
    mean_period,
    slab_lattice,
    upward_crossing_periods,
    waterplane_chord,
)

RHO_ICE = 917.0
RHO_WATER = 1025.0
GRAVITY = 9.81
RADIUS = 0.5
THICKNESS = 1.0


def _cfg(wet_drag: float = 0.0) -> WetCouplingConfig:
    return WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY,
                             water_level=0.0, wet_drag_coeff=wet_drag)


def _reference_submerged_fraction(h: float, radius: float, n: int = 200001) -> float:
    """Column-chord quadrature, independent of the engine's acos formula."""
    dx = 2.0 * radius / n
    x = torch.linspace(-radius + 0.5 * dx, radius - 0.5 * dx, n, dtype=torch.float64)
    s = torch.sqrt(torch.clamp(radius * radius - x * x, min=0.0))
    sub = torch.clamp(torch.minimum(h + s, torch.zeros_like(s)) - (h - s), min=0.0)
    return float(sub.sum()) * dx / (math.pi * radius * radius)


def _disk_period() -> tuple[float, float, float]:
    h_eq = equilibrium_center_height(RHO_ICE, RHO_WATER, RADIUS)
    k = linear_heave_stiffness(RHO_WATER, GRAVITY, THICKNESS, RADIUS, h_eq)
    m = RHO_ICE * math.pi * RADIUS ** 2 * THICKNESS
    return h_eq, k, analytic_heave_period(m, k)


class SubmergedFractionGeometryTests(unittest.TestCase):
    def test_matches_column_quadrature(self):
        worst = 0.0
        for k in range(25):
            h = (-1.2 + 2.4 * k / 24.0) * RADIUS
            engine = float(submerged_fraction_tensor(torch.tensor([h]), RADIUS, 0.0)[0])
            worst = max(worst, abs(engine - _reference_submerged_fraction(h, RADIUS)))
        self.assertLess(worst, 1e-6)

    def test_monotone_and_bounded(self):
        heights = [-RADIUS * 1.5 + 0.05 * RADIUS * k for k in range(60)]
        fracs = [float(submerged_fraction_tensor(torch.tensor([h]), RADIUS, 0.0)[0])
                 for h in heights]
        self.assertAlmostEqual(fracs[0], 1.0, places=12)
        self.assertAlmostEqual(fracs[-1], 0.0, places=12)
        for a, b in zip(fracs, fracs[1:]):
            self.assertGreaterEqual(a, b - 1e-12)


class ArchimedesStaticsTests(unittest.TestCase):
    def test_exact_draft_matches_density_ratio(self):
        h_eq = equilibrium_center_height(RHO_ICE, RHO_WATER, RADIUS)
        f = float(submerged_fraction_tensor(torch.tensor([h_eq]), RADIUS, 0.0)[0])
        self.assertLess(abs(f - RHO_ICE / RHO_WATER) / (RHO_ICE / RHO_WATER), 1e-6)

    def test_thin_slab_linearisation_is_not_usable(self):
        """Documents the trap: the naive draft formula is off by tens of percent."""
        ratio = RHO_ICE / RHO_WATER
        h_eq = equilibrium_center_height(RHO_ICE, RHO_WATER, RADIUS)
        naive = -(1.0 - ratio) * RADIUS
        self.assertGreater(abs(naive - h_eq) / abs(h_eq), 0.5)

    def test_chord_shrinks_as_draft_deepens(self):
        shallow = waterplane_chord(RADIUS, -0.05 * RADIUS)
        deep = waterplane_chord(RADIUS, -0.9 * RADIUS)
        self.assertLess(deep, shallow)
        self.assertAlmostEqual(waterplane_chord(RADIUS, RADIUS), 0.0, places=12)


class HeaveDynamicsTests(unittest.TestCase):
    def test_period_matches_analytic(self):
        h_eq, _k, period = _disk_period()
        dt = period / 500.0
        floe = RigidHeaveFloe(torch.zeros((1, 2), dtype=torch.float64), RADIUS,
                              THICKNESS, RHO_ICE, _cfg(), dt, y0=h_eq + 0.01 * RADIUS)
        floe.run(int(4.0 * period / dt), record_every=max(int(0.01 * period / dt), 1))
        measured = mean_period(upward_crossing_periods(floe.trajectory.t,
                                                      floe.trajectory.y, h_eq))
        self.assertLess(abs(measured - period) / period, 0.005)

    def test_wet_drag_damping_ratio(self):
        h_eq, k, period = _disk_period()
        m = RHO_ICE * math.pi * RADIUS ** 2 * THICKNESS
        omega = 2.0 * math.pi / period
        zeta = 0.05
        c_wet = 2.0 * m * omega * zeta / (RHO_ICE / RHO_WATER)
        dt = period / 250.0
        floe = RigidHeaveFloe(torch.zeros((1, 2), dtype=torch.float64), RADIUS,
                              THICKNESS, RHO_ICE, _cfg(c_wet), dt, y0=h_eq + 0.02 * RADIUS)
        floe.run(int(14.0 * period / dt), record_every=max(int(0.02 * period / dt), 1))
        traj = floe.trajectory
        measured = log_decrement_damping_ratio(traj.t, traj.y, h_eq, max_cycles=8)
        self.assertLess(abs(measured - zeta) / zeta, 0.05)

    def test_undamped_energy_is_conserved(self):
        h_eq, k, period = _disk_period()
        amp = 0.02 * RADIUS
        dt = period / 500.0
        floe = RigidHeaveFloe(torch.zeros((1, 2), dtype=torch.float64), RADIUS,
                              THICKNESS, RHO_ICE, _cfg(), dt, y0=h_eq + amp)
        floe.run(int(6.0 * period / dt), record_every=max(int(0.02 * period / dt), 1),
                 record_energy=True)
        energy = floe.trajectory.energy
        scale = 0.5 * k * amp ** 2
        self.assertLess((max(energy) - min(energy)) / scale, 1e-3)

    def test_disabled_coupling_is_inert(self):
        cfg = WetCouplingConfig(enabled=False)
        floe = RigidHeaveFloe(torch.zeros((1, 2), dtype=torch.float64), RADIUS,
                              THICKNESS, RHO_ICE, cfg, 1e-3, y0=0.0)
        self.assertEqual(floe.buoyancy_force(0.0), 0.0)
        self.assertEqual(floe.wet_drag_force(0.0, 1.0), 0.0)


class LatticeFloeTests(unittest.TestCase):
    def test_hex_floe_is_tangent_and_area_consistent(self):
        n = 19
        a = RADIUS / math.sqrt(n)
        offsets = hex_lattice_floe(a, n)
        self.assertEqual(tuple(offsets.shape), (n, 2))
        dist = torch.cdist(offsets, offsets)
        dist.fill_diagonal_(float("inf"))
        self.assertGreaterEqual(float(dist.min()), 2.0 * a - 1e-9)

    def test_aggregate_draft_balances_weight(self):
        n = 19
        a = RADIUS / math.sqrt(n)
        offsets = hex_lattice_floe(a, n)
        mass = RHO_ICE * math.pi * a ** 2 * THICKNESS * n
        lo, hi = -float(offsets.abs().max()) - a, float(offsets.abs().max()) + a
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if discrete_buoyancy_profile(offsets, a, THICKNESS, RHO_ICE, _cfg(), mid) \
                    > mass * GRAVITY:
                lo = mid
            else:
                hi = mid
        y_eq = 0.5 * (lo + hi)
        f_bar = float(submerged_fraction_tensor(offsets[:, 1] + y_eq, a, 0.0).mean())
        self.assertLess(abs(f_bar - RHO_ICE / RHO_WATER) / (RHO_ICE / RHO_WATER), 1e-4)

    def test_discrete_stiffness_matches_numerical_derivative(self):
        a = 0.05
        offsets = slab_lattice(0.6, 0.4, a)
        y0 = -0.05
        k_closed = discrete_stiffness(offsets, a, THICKNESS, RHO_ICE, _cfg(), y0)
        eps = 1e-7
        f_plus = discrete_buoyancy_profile(offsets, a, THICKNESS, RHO_ICE, _cfg(), y0 + eps)
        f_minus = discrete_buoyancy_profile(offsets, a, THICKNESS, RHO_ICE, _cfg(), y0 - eps)
        k_num = -(f_plus - f_minus) / (2.0 * eps)
        self.assertGreater(k_closed, 0.0)
        self.assertLess(abs(k_num - k_closed) / k_closed, 1e-3)

    def test_slab_band_mean_matches_area_identity(self):
        """Mean stiffness over the partial-submergence band = rho_w g t A/H_band."""
        a = 0.0625
        offsets = slab_lattice(1.0, 2.0, a)
        n = int(offsets.shape[0])
        area_solid = n * math.pi * a ** 2
        top = float(offsets[:, 1].max()) + a
        bottom = float(offsets[:, 1].min()) - a
        band = top - bottom              # partial-submergence band height
        samples = 121
        # y = -top is fully submerged, y = -bottom is fully dry; sample across.
        ks = [discrete_stiffness(offsets, a, THICKNESS, RHO_ICE, _cfg(),
                                -top + band * k / (samples - 1)) for k in range(samples)]
        k_mean = sum(ks) / len(ks)
        k_identity = RHO_WATER * GRAVITY * THICKNESS * area_solid / band
        self.assertLess(abs(k_mean - k_identity) / k_identity, 0.03)

    def test_slab_envelope_is_closed(self):
        a = 0.05
        offsets = slab_lattice(0.6, 0.4, a)
        width, height = body_envelope(offsets, a)
        self.assertGreater(width, 0.5)
        self.assertGreater(height, 0.3)


if __name__ == "__main__":
    unittest.main()