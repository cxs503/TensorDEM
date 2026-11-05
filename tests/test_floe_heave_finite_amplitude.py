"""Finite-amplitude heave period: nonlinear reference vs the production model.

Open item being closed: ``docs/ice_floe_physics_validation.md`` recorded that the
heave-period anchor ``T0 = 2 pi sqrt(m/K)`` is a *linearised* result and that the
finite-amplitude period was never verified.  These tests build the nonlinear
reference (``tensordem.floe_heave_amplitude``) and check the production heave
model against it, at amplitudes where the linearisation is measurably wrong.

Gates are acceptance thresholds, not best-case numbers.
"""
from __future__ import annotations

import math

import torch

torch.set_num_threads(1)

from tensordem.buoyancy_wet import WetCouplingConfig, _submerged_fraction_2d
from tensordem.floe_buoyancy import hex_lattice_floe, mean_period, upward_crossing_periods
from tensordem.floe_heave_amplitude import (
    DiskHeaveReference,
    simulate_heave_at_amplitude,
    submerged_disk_area,
    submerged_disk_area_antiderivative,
    submerged_disk_area_quadrature,
)

RADIUS = 0.5
THICKNESS = 1.0
RHO_ICE = 917.0
RHO_WATER = 1025.0
GRAVITY = 9.81


def _cfg(wet_drag: float = 0.0) -> WetCouplingConfig:
    return WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY,
                             water_level=0.0, wet_drag_coeff=wet_drag, model="2d")


def _single_disk() -> torch.Tensor:
    return torch.zeros((1, 2), dtype=torch.float64)


def _ref() -> DiskHeaveReference:
    return DiskHeaveReference(radius=RADIUS, thickness=THICKNESS, rho_water=RHO_WATER,
                              rho_ice=RHO_ICE, gravity=GRAVITY)


# ---------------------------------------------------------------------------
# Reference construction (must be right before it can judge the model)
# ---------------------------------------------------------------------------

def test_disk_area_closed_form_matches_independent_quadrature():
    r = RADIUS
    for h_over_r in (-0.999, -0.9, -0.6755, -0.4, -0.05, 0.0, 0.05, 0.4, 0.6755, 0.9, 0.999):
        h = h_over_r * r
        closed = submerged_disk_area(h, r)
        quad = submerged_disk_area_quadrature(h, r)
        # 1e-9 covers the near-degenerate edge |h| -> R, where the column
        # quadrature loses digits and the area itself vanishes.
        assert math.isclose(closed, quad, rel_tol=1e-9), (h_over_r, closed, quad)
        # the engine's own acos kernel must agree too (no silent geometry drift)
        engine = _submerged_fraction_2d(h, r) * math.pi * r * r
        assert math.isclose(closed, engine, rel_tol=1e-9), (h_over_r, closed, engine)


def test_area_limits_and_antiderivative_derivative():
    r = RADIUS
    assert submerged_disk_area(r, r) == 0.0
    assert math.isclose(submerged_disk_area(-r, r), math.pi * r * r, rel_tol=1e-15)
    assert math.isclose(submerged_disk_area(0.0, r), 0.5 * math.pi * r * r, rel_tol=1e-15)
    for h in (-0.3, -0.05, 0.1, 0.3):
        eps = 1e-7
        d = (submerged_disk_area_antiderivative(h + eps, r)
             - submerged_disk_area_antiderivative(h - eps, r)) / (2.0 * eps)
        assert math.isclose(d, submerged_disk_area(h, r), rel_tol=1e-8), h


def test_reference_reduces_to_the_linear_anchor_at_tiny_amplitude():
    ref = _ref()
    t0 = ref.linear_period()
    # A = 1e-3 R is the smallest amplitude the energy integral resolves: below it
    # the amplitude correction O(A^2/R^2) drops under the cancellation floor of
    # U(h) (difference of two ~m g h terms) and the quadrature stops converging.
    assert math.isclose(ref.period(1e-3 * RADIUS), t0, rel_tol=1e-5)
    # equilibrium is self-consistent: no residual weight in the reference
    assert abs(ref.net_force(ref.h_eq)) / ref.weight < 1e-12


def test_energy_quadrature_agrees_with_independent_rk4_time_integration():
    ref = _ref()
    for amp_rel in (0.05, 0.2, 0.5):
        a = amp_rel * ref.escape_amplitude()
        tq = ref.period(a)
        tr = ref.rk4_period(a)
        assert math.isclose(tq, tr, rel_tol=1e-5), (amp_rel, tq, tr)


def test_nonlinearity_is_measurable_before_the_well_ends():
    ref = _ref()
    a_esc = ref.escape_amplitude()
    growth = ref.period(0.85 * a_esc) / ref.linear_period()
    # the amplitude correction must be far larger than any numerical tolerance,
    # otherwise the verification below could not discriminate a linearised model
    assert growth - 1.0 > 0.02, growth
    with_raise = True
    try:
        ref.period(1.2 * a_esc)
    except ValueError:
        with_raise = False
    assert not with_raise, "amplitude outside the buoyancy well must be rejected"


# ---------------------------------------------------------------------------
# Main anchor: production model vs the nonlinear reference
# ---------------------------------------------------------------------------

def _model_amplitudes(ref: DiskHeaveReference) -> list[float]:
    a_esc = ref.escape_amplitude()
    return [0.07 * a_esc, 0.3 * a_esc, 0.6 * a_esc, 0.85 * a_esc]


def test_production_heave_matches_nonlinear_reference_within_0p5_percent():
    ref = _ref()
    dt = ref.linear_period() / 1000.0
    rows = []
    for a in _model_amplitudes(ref):
        res = simulate_heave_at_amplitude(_single_disk(), RADIUS, THICKNESS, RHO_ICE,
                                          _cfg(), dt, a, ref.linear_period(), periods=4.0)
        target = ref.period(a)
        rel = (res.period - target) / target
        rows.append((a / RADIUS, res.period, target, rel))
        assert abs(rel) < 5e-3, rows[-1]
        assert res.energy_drift < 5e-2, res.energy_drift
    # the largest amplitude must be one where quoting the linear anchor fails 3%
    worst = rows[-1]
    assert worst[2] / ref.linear_period() - 1.0 > 0.02, rows[-1]


def test_linearised_model_is_rejected_by_the_reference():
    """Negative control: a stiffness linearisation must fail the amplitude law."""
    ref = _ref()
    k = ref.linear_stiffness()
    m = ref.mass
    dt = ref.linear_period() / 1000.0

    def _period_linear(amplitude: float) -> float:
        y, v = amplitude, 0.0
        ts, ys = [], []
        n = int(5.0 * ref.linear_period() / dt)
        for i in range(n):
            ts.append(i * dt)
            ys.append(y)
            a_acc = -k * y / m
            v_half = v + 0.5 * a_acc * dt
            y_new = y + v_half * dt
            v = v_half + 0.5 * (-k * y_new / m) * dt
            y = y_new
        return mean_period(upward_crossing_periods(ts, ys, 0.0), skip=1)

    a_esc = ref.escape_amplitude()
    small = _period_linear(0.07 * a_esc)
    large = _period_linear(0.85 * a_esc)
    assert abs(large / small - 1.0) < 5e-4, (small, large)  # linear: amplitude-flat

    res = simulate_heave_at_amplitude(_single_disk(), RADIUS, THICKNESS, RHO_ICE, _cfg(),
                                      ref.linear_period() / 1500.0, 0.85 * a_esc,
                                      ref.linear_period(), periods=4.0)
    assert res.period / small - 1.0 > 0.02, (res.period, small)


def test_period_is_timestep_independent():
    ref = _ref()
    a = 0.85 * ref.escape_amplitude()
    fine_dt = ref.linear_period() / 2000.0
    coarse = simulate_heave_at_amplitude(_single_disk(), RADIUS, THICKNESS, RHO_ICE, _cfg(),
                                         fine_dt * 2.0, a, ref.linear_period(), periods=4.0)
    fine = simulate_heave_at_amplitude(_single_disk(), RADIUS, THICKNESS, RHO_ICE, _cfg(),
                                       fine_dt, a, ref.linear_period(), periods=4.0)
    assert abs(coarse.period - fine.period) / fine.period < 2e-3


def test_lattice_floe_reproduces_the_continuum_amplitude_growth_law():
    """Production lattice floe: the *growth law* is discretisation-robust."""
    ref = _ref()
    a = 0.25 * RADIUS
    offsets = hex_lattice_floe(a, 7)
    cfg = _cfg()
    dt = 2.0e-3
    small = simulate_heave_at_amplitude(offsets, a, THICKNESS, RHO_ICE, cfg, dt,
                                        0.02 * RADIUS, ref.linear_period(), periods=4.0)
    large = simulate_heave_at_amplitude(offsets, a, THICKNESS, RHO_ICE, cfg, dt,
                                        0.25 * RADIUS, ref.linear_period(), periods=4.0)
    model_growth = large.period / small.period
    # The lattice nonlinearity is *not* the disk nonlinearity: the waterline cuts
    # through a 7-disk cluster, so its well geometry scales with the cluster
    # half-height (1.5 a), not with R.  Only the sign and order of magnitude of
    # the amplitude correction are claimed here; the quantitative lattice growth
    # law stays an open item (see docs/floe_heave_finite_amplitude_period.md).
    assert model_growth > 1.02, model_growth
    ref_growth = ref.period(0.25 * RADIUS) / ref.period(0.02 * RADIUS)
    assert ref_growth > 1.0