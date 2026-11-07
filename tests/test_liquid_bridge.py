"""Verification of the liquid-bridge kernels (closed form + Young–Laplace reference).

The exact reference is checked by its *own* invariants before it is used to
judge the closed form:

* the axial force is a first integral of the axisymmetric Young–Laplace
  problem, so the neck- and contact-line evaluations must agree,
* the shooting residuals (contact angle, volume) must be small,
* the closed form must reproduce its known analytic limits (``h -> 0`` with
  ``V^(1/3) >> h`` gives ``2 pi gamma R cos(theta)``; ``theta -> 90 deg``
  gives zero).

Criterion C3 of the campaign (closed form within 5 % of the reference at three
points) is *not* claimed here: the measured deviation is 15-30 % in the
converged window and much worse for slender bridges, and the campaign reports
that honestly.
"""
from __future__ import annotations

import math

import pytest

from tensordem.liquid_bridge import (
    delenne_liquid_bridge_force,
    delenne_rupture_distance,
    solve_axisymmetric_bridge,
)

R = 1.0e-3       # m
GAMMA = 0.072    # N/m (water)
F0 = 2.0 * math.pi * GAMMA * R


def test_closed_form_monotone_in_gap_and_volume():
    base = delenne_liquid_bridge_force(R, GAMMA, 1e-12, 0.0)
    assert base == pytest.approx(F0, rel=1e-12)
    assert delenne_liquid_bridge_force(R, GAMMA, 1e-12, 1e-4) < base
    assert delenne_liquid_bridge_force(R, GAMMA, 1e-12, 2e-4) < \
        delenne_liquid_bridge_force(R, GAMMA, 1e-12, 1e-4)
    assert delenne_liquid_bridge_force(R, GAMMA, 4e-12, 0.0) == \
        pytest.approx(F0, rel=1e-12)


def test_closed_form_scales_with_cosine_of_contact_angle():
    for theta in (0.0, 30.0, 60.0):
        f = delenne_liquid_bridge_force(R, GAMMA, 1e-12, 0.0, theta)
        assert f == pytest.approx(F0 * math.cos(math.radians(theta)), rel=1e-12)
    # non-wetting: no pendular bridge
    assert delenne_liquid_bridge_force(R, GAMMA, 1e-12, 0.0, 120.0) == 0.0


def test_closed_form_large_volume_limit():
    """V^(1/3) >> h: the gap correction vanishes (Derjaguin limit)."""
    f = delenne_liquid_bridge_force(R, GAMMA, 1e-9, 1e-7)
    assert f == pytest.approx(F0, rel=1e-3)


def test_closed_form_validation_errors():
    with pytest.raises(ValueError):
        delenne_liquid_bridge_force(-1.0, GAMMA, 1e-12, 0.0)
    with pytest.raises(ValueError):
        delenne_liquid_bridge_force(R, GAMMA, -1e-12, 0.0)
    with pytest.raises(ValueError):
        delenne_liquid_bridge_force(R, GAMMA, 1e-12, -1e-6)


def test_rupture_distance_estimate():
    d = delenne_rupture_distance(1e-12, 0.0)
    assert d == pytest.approx(0.1 * 1e-4, rel=1e-12)
    assert delenne_rupture_distance(1e-12, 60.0) > d


@pytest.mark.parametrize("gap_over_r", [1e-4, 1e-3, 2e-3])
def test_exact_reference_is_a_solution_of_the_free_boundary_problem(gap_over_r):
    """Shooting residuals + first-integral (force constancy) verification."""
    sol = solve_axisymmetric_bridge(R, GAMMA, 0.02 * R ** 3, gap_over_r * R, 0.0)
    assert sol.converged
    assert sol.residual < 1e-6
    assert sol.extra["angle_solved_deg"] == pytest.approx(0.0, abs=0.02)
    assert sol.extra["volume_solved"] == pytest.approx(sol.volume, rel=1e-6)
    # axial force is a first integral: neck vs contact-line evaluation
    f_contact = sol.extra["force_at_contact_line"]
    assert f_contact == pytest.approx(sol.force, rel=2e-2)
    assert sol.force > 0.0
    assert sol.neck_radius > 0.0
    assert sol.contact_radius > 0.0


def test_exact_reference_approaches_derjaguin_limit():
    """h -> 0 and V^(1/3) >> h: F -> O(2 pi gamma R cos theta)."""
    sol = solve_axisymmetric_bridge(R, GAMMA, 0.02 * R ** 3, 1e-4 * R, 0.0)
    assert sol.converged
    assert 0.6 < sol.force / F0 < 1.05


def test_exact_reference_decreases_with_gap():
    f_near = solve_axisymmetric_bridge(R, GAMMA, 0.02 * R ** 3, 1e-4 * R).force
    f_far = solve_axisymmetric_bridge(R, GAMMA, 0.02 * R ** 3, 1e-2 * R).force
    assert f_far < f_near


def test_exact_reference_decreases_with_contact_angle():
    f0 = solve_axisymmetric_bridge(R, GAMMA, 0.02 * R ** 3, 1e-3 * R, 0.0).force
    f60 = solve_axisymmetric_bridge(R, GAMMA, 0.02 * R ** 3, 1e-3 * R, 60.0).force
    assert f60 < f0
    # cos-scaling is only approximate in the exact solution
    assert 0.3 < f60 / f0 < 0.9


def test_closed_form_overestimates_the_exact_bridge():
    """Measured (honest) model bias: the closed form is 15-30 % high at theta=0."""
    sol = solve_axisymmetric_bridge(R, GAMMA, 0.02 * R ** 3, 1e-3 * R, 0.0)
    assert sol.converged
    assert sol.force < sol.force_delenne
    assert 0.6 < sol.force / sol.force_delenne < 0.95
    assert sol.relative_difference > 0.05  # criterion C3 is therefore NOT met