"""Unit coverage for :mod:`tensordem.buoyancy_verification`.

The free-float verification campaign (``run_campaign``) is exercised elsewhere
only through its aggregate verdicts, and ten of the module's eighteen public
names -- the analytic references that the verdicts are measured against -- were
never referenced by a test.  This file pins those references:

* :func:`chord_width` / :func:`submerged_fraction_quadrature` -- the
  independent geometric route (waterplane chord, substituted Simpson
  quadrature): limits, complementarity ``f(h) + f(-h) = 1``, panel
  independence, and the documented ``|h| >= R`` cut-off;
* :func:`solve_equilibrium_height` -- Archimedes inversion (``f_sub`` equals
  ``rho_ice / rho_water``) plus its saturation clamps;
* :class:`FreeFloatParameters` -- the linearised oscillator it derives from the
  stiffness identity (``omega_d / omega_n`` relation);
* :func:`vertical_acceleration` / :func:`integrate_free_float` -- the force
  routine it wraps: force balance at the draft, restoring sign, the
  ``apply_gravity`` flag, and damped decay of the RK4 trajectory;
* :func:`analytic_displacement` / :func:`analytic_envelope` -- the linearised
  reference the simulated motion is scored against.

Two negative controls then inject a *wrong* reference into the gates and
require them to turn red (``geometry_check`` / ``stiffness_check`` with a
corrupted reference must report ``passed = False``), so a silently broken
reference cannot be mistaken for a passing verification.
"""
from __future__ import annotations

import math

import pytest

from tensordem import buoyancy_verification as bv

RADIUS = 0.5


def test_chord_width_geometry() -> None:
    assert bv.chord_width(0.0, RADIUS) == pytest.approx(2.0 * RADIUS)
    assert bv.chord_width(0.3, RADIUS) == pytest.approx(0.8, rel=1e-15)
    # symmetric in the centre height
    for h in (-0.4, -0.2, 0.0, 0.1, 0.45):
        assert bv.chord_width(-h, RADIUS) == pytest.approx(bv.chord_width(h, RADIUS), rel=1e-15)
    # no waterplane once the disk clears the free surface
    assert bv.chord_width(RADIUS, RADIUS) == 0.0
    assert bv.chord_width(2.0 * RADIUS, RADIUS) == 0.0
    assert bv.chord_width(-1.5 * RADIUS, RADIUS) == 0.0


def test_submerged_fraction_quadrature_limits_and_complementarity() -> None:
    assert bv.submerged_fraction_quadrature(0.0, RADIUS) == pytest.approx(0.5, abs=1e-12)
    assert bv.submerged_fraction_quadrature(RADIUS, RADIUS) == 0.0
    assert bv.submerged_fraction_quadrature(-RADIUS, RADIUS) == 1.0
    assert bv.submerged_fraction_quadrature(2.0 * RADIUS, RADIUS) == 0.0
    assert bv.submerged_fraction_quadrature(-2.0 * RADIUS, RADIUS) == 1.0

    for h in (-0.45, -0.25, 0.0, 0.2, 0.4):
        dry = bv.submerged_fraction_quadrature(h, RADIUS)
        assert 0.0 <= dry <= 1.0
        assert dry + bv.submerged_fraction_quadrature(-h, RADIUS) == pytest.approx(1.0, abs=1e-12)

    # the sin-substituted integrand is smooth: 10x more panels changes nothing
    for h in (-0.3, 0.1):
        coarse = bv.submerged_fraction_quadrature(h, RADIUS, panels=200)
        fine = bv.submerged_fraction_quadrature(h, RADIUS, panels=2000)
        assert coarse == pytest.approx(fine, abs=1e-9)


def test_equilibrium_height_inverts_archimedes() -> None:
    for rho_ice in (850.0, 917.0, 1000.0):
        h_eq = bv.solve_equilibrium_height(RADIUS, rho_ice, 1025.0)
        assert -RADIUS < h_eq < RADIUS
        assert bv.submerged_fraction_quadrature(h_eq, RADIUS) == pytest.approx(
            rho_ice / 1025.0, abs=1e-9)

    # denser ice floats deeper
    light = bv.solve_equilibrium_height(RADIUS, 850.0, 1025.0)
    heavy = bv.solve_equilibrium_height(RADIUS, 1000.0, 1025.0)
    assert heavy < light


def test_equilibrium_height_saturates_outside_the_physical_range() -> None:
    """Negative case: non-floating densities must clamp, not return a fake root."""
    assert bv.solve_equilibrium_height(RADIUS, 1100.0, 1025.0) == -RADIUS
    assert bv.solve_equilibrium_height(RADIUS, 0.0, 1025.0) == RADIUS


def test_free_float_fixture_linearised_oscillator() -> None:
    params = bv.FreeFloatParameters()
    assert params.mass == pytest.approx(917.0 * math.pi * RADIUS ** 2 * 1.0, rel=1e-15)
    assert params.weight == pytest.approx(params.mass * params.gravity, rel=1e-15)

    h_eq = params.equilibrium_height()
    assert params.stiffness() == pytest.approx(
        -bv.buoyancy_stiffness_analytic(h_eq, RADIUS, params.rho_water,
                                        params.gravity, params.thickness), rel=1e-15)
    assert params.stiffness() > 0.0
    assert params.omega_n() == pytest.approx(math.sqrt(params.stiffness() / params.mass), rel=1e-15)
    assert 0.0 < params.damping_ratio() < 1.0
    assert params.omega_d() == pytest.approx(
        params.omega_n() * math.sqrt(1.0 - params.damping_ratio() ** 2), rel=1e-12)
    assert params.omega_d() < params.omega_n()


def test_stiffness_identity_sign_and_cutoff() -> None:
    # inside the waterplane the gradient is restoring (negative dF_b/dh)
    assert bv.buoyancy_stiffness_analytic(0.0, RADIUS, 1025.0, 9.81, 1.0) == pytest.approx(
        -1025.0 * 9.81 * 1.0 * 2.0 * RADIUS, rel=1e-15)
    assert bv.buoyancy_stiffness_analytic(0.0, RADIUS, 1025.0, 9.81, 1.0) < 0.0
    # outside it the force is constant, so the gradient vanishes
    assert bv.buoyancy_stiffness_analytic(1.5 * RADIUS, RADIUS, 1025.0, 9.81, 1.0) == 0.0
    assert bv.buoyancy_stiffness_analytic(-1.5 * RADIUS, RADIUS, 1025.0, 9.81, 1.0) == 0.0


def test_vertical_acceleration_balance_and_restoring_sign() -> None:
    params = bv.FreeFloatParameters()
    h_eq = params.equilibrium_height()

    # released at the draft the net force vanishes
    assert bv.vertical_acceleration(h_eq, 0.0, params) == pytest.approx(0.0, abs=1e-9)
    # displaced upward -> accelerated downward, displaced downward -> upward
    up = bv.vertical_acceleration(h_eq + 0.02, 0.0, params)
    down = bv.vertical_acceleration(h_eq - 0.02, 0.0, params)
    assert up < 0.0 < down
    assert abs(up) == pytest.approx(abs(down), rel=0.1)


def test_vertical_acceleration_respects_the_gravity_flag() -> None:
    """Negative case: dropping the weight term must flip the balance."""
    params = bv.FreeFloatParameters()
    h_eq = params.equilibrium_height()
    without_gravity = bv.vertical_acceleration(h_eq, 0.0, params, apply_gravity=False)
    with_gravity = bv.vertical_acceleration(h_eq, 0.0, params, apply_gravity=True)
    assert without_gravity > 0.0
    assert without_gravity == pytest.approx(params.gravity, rel=0.02)
    assert with_gravity == pytest.approx(0.0, abs=1e-9)


def test_analytic_linear_response_is_consistent() -> None:
    params = bv.FreeFloatParameters()
    assert bv.analytic_displacement(0.0, params) == pytest.approx(params.amplitude, rel=1e-15)
    # linear in the release amplitude (sign flips with it)
    assert bv.analytic_displacement(0.3, params, amplitude=-params.amplitude) == pytest.approx(
        -bv.analytic_displacement(0.3, params), rel=1e-15)
    # at half a damped period the response sits on the negative envelope
    t_half = math.pi / params.omega_d()
    assert bv.analytic_displacement(t_half, params) == pytest.approx(
        -bv.analytic_envelope(t_half, params), rel=1e-12)
    # the envelope decays monotonically
    envelopes = [bv.analytic_envelope(t, params) for t in (0.0, 1.0, 2.0)]
    assert envelopes[0] > envelopes[1] > envelopes[2] > 0.0
    # sequence input matches the scalar evaluation
    assert bv.analytic_displacement([0.0, 1.0], params) == pytest.approx(
        [bv.analytic_displacement(0.0, params), bv.analytic_displacement(1.0, params)])


def test_free_float_integration_is_damped() -> None:
    params = bv.FreeFloatParameters(duration=4.0, dt=5e-2)
    h_eq = params.equilibrium_height()
    history = bv.integrate_free_float(params, h_eq + params.amplitude, sample_every=1)

    times, heights, velocities = history["time"], history["height"], history["velocity"]
    assert len(times) == len(heights) == len(velocities) == 81
    assert times[0] == 0.0 and times[-1] == pytest.approx(4.0)
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    assert heights[0] == pytest.approx(h_eq + params.amplitude)
    assert velocities[0] == 0.0

    deviation = [abs(y - h_eq) for y in heights]
    assert max(deviation) <= 1.5 * params.amplitude
    first_half = max(deviation[: len(deviation) // 2])
    second_half = max(deviation[len(deviation) // 2:])
    assert second_half < 0.8 * first_half
    assert second_half == pytest.approx(bv.analytic_envelope(2.0, params), rel=0.3)


def test_geometry_gate_detects_a_corrupted_quadrature(monkeypatch) -> None:
    """Negative control: the gate must fail when the reference is wrong."""
    assert bv.geometry_check()["passed"] is True
    monkeypatch.setattr(bv, "submerged_fraction_quadrature",
                        lambda h, radius, panels=2000: 0.5)
    result = bv.geometry_check()
    assert result["passed"] is False
    assert result["max_abs_error"] > result["tolerance"] > 0.0


def test_stiffness_gate_detects_a_sign_error(monkeypatch) -> None:
    """Negative control: a sign-flipped stiffness reference must fail the gate."""
    params = bv.FreeFloatParameters()
    assert bv.stiffness_check(params)["passed"] is True

    original = bv.buoyancy_stiffness_analytic
    monkeypatch.setattr(
        bv, "buoyancy_stiffness_analytic",
        lambda h, radius, rho_water, gravity, thickness: abs(
            original(h, radius, rho_water, gravity, thickness)),
    )
    result = bv.stiffness_check(params)
    assert result["passed"] is False
    assert max(row["rel_error"] for row in result["rows"]) > 0.5