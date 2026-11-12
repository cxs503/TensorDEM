"""m16340 -- stability and accuracy of the viscous rolling-resistance branch.

The honest FAIL in ``docs/clump_rolling_resistance_calibration.md`` §3.4 recorded
that the ``"viscous"`` branch overflows at its own default
``rolling_omega_ref = 1e-3`` rad/s with ``dt = 2e-4`` s, and that the 5.74 deg
"tilt threshold" it produced was a numerical artefact.  These tests pin the
replacement behaviour:

* the run is **refused with an actionable message** instead of overflowing
  (``ContactConfig.rolling_stability_guard``);
* the analytic stability window ``lambda*dt < 4`` (divergence) and
  ``lambda*dt <= 2`` (monotone) is verified numerically, with
  ``lambda = mu_r |F_n| r / (omega_ref I)``;
* inside the window the free spin-down follows ``exp(-lambda t)`` to first order
  and the incline steady state matches ``omega* = omega_ref tan(theta)/mu_r``
  exactly for the declared (explicit) law;
* the optional gain-limited (implicit) branch is stable at any step and inflates
  the steady rolling rate by the documented ``1 + lambda*dt/2``;
* dissipated rolling work equals the kinetic-energy drop to round-off.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.clump import Clump, SphereMember
from tensordem.clump_dynamics import ClumpBody, ContactConfig, PlaneContact
from tensordem.clump_rolling import viscous_damping_gain, viscous_stability_dt

RADIUS = 0.05
DENSITY = 900.0
GRAVITY = 9.81
MU_R = 0.9


def _clump(radius: float = RADIUS, density: float = DENSITY) -> Clump:
    return Clump([SphereMember((0.0, 0.0, 0.0), radius)], density=density)


def _inertia(clump: Clump) -> float:
    return float(clump.inertia_about_com[0, 0])


def _rate(clump: Clump, mu_r: float, omega_ref: float, radius: float = RADIUS) -> float:
    """Analytic ``lambda = M_lim / (omega_ref I)`` [1/s] for a sphere on a plane."""
    limit = mu_r * clump.mass * GRAVITY * radius
    return limit / (omega_ref * _inertia(clump))


def _body(clump: Clump, radius: float = RADIUS, **kwargs) -> ClumpBody:
    penetration = clump.mass * GRAVITY / 1.0e5
    return ClumpBody(clump, position=(0.0, 0.0, radius - penetration),
                     angular_velocity=(1.0, 0.0, 0.0), gravity=(0.0, 0.0, -GRAVITY),
                     **kwargs)


def _config(*, omega_ref: float, implicit: bool = False, guard: bool = True,
            mu_r: float = MU_R, friction: float = 0.0) -> ContactConfig:
    return ContactConfig(normal_stiffness=1.0e5, restitution=0.5, friction=friction,
                         rolling_friction=mu_r, rolling_model="viscous",
                         rolling_omega_ref=omega_ref,
                         rolling_viscous_implicit=implicit,
                         rolling_stability_guard=guard)


def _spin_down(*, omega_ref: float, dt: float, steps: int, implicit: bool = False,
               guard: bool = True, radius: float = RADIUS,
               density: float = DENSITY) -> tuple[ClumpBody, PlaneContact]:
    clump = _clump(radius, density)
    plane = PlaneContact((0.0, 0.0, 1.0),
                         config=_config(omega_ref=omega_ref, implicit=implicit, guard=guard))
    body = _body(clump, radius=radius)
    body.run(steps, dt, (plane,))
    return body, plane


# --------------------------------------------------------------------------
# 1. the documented failure is now refused, not silent
# --------------------------------------------------------------------------
def test_guard_refuses_the_documented_unstable_default() -> None:
    clump = _clump()
    lam = _rate(clump, MU_R, 1.0e-3)
    assert lam * 2.0e-4 > 4.0                       # the configuration is divergent
    with pytest.raises(ValueError) as excinfo:
        _spin_down(omega_ref=1.0e-3, dt=2.0e-4, steps=5)
    message = str(excinfo.value)
    assert "stability window" in message
    assert "lambda*dt" in message
    assert "rolling_omega_ref" in message           # tells the user what to change


def test_guard_off_reproduces_the_recorded_blow_up() -> None:
    """Honest FAIL preserved: without the guard the run still overflows."""
    with pytest.raises((ValueError, OverflowError, FloatingPointError)):
        _spin_down(omega_ref=1.0e-3, dt=2.0e-4, steps=400, guard=False)


def test_guard_lets_resolved_configurations_through() -> None:
    clump = _clump()
    lam = _rate(clump, MU_R, 0.1)
    assert lam * 2.0e-4 < 2.0
    body, plane = _spin_down(omega_ref=0.1, dt=2.0e-4, steps=200)
    assert math.isfinite(float(np.linalg.norm(body.angular_velocity)))
    assert abs(float(body.angular_velocity[0])) < 1.0
    assert plane.moment_work() > 0.0


# --------------------------------------------------------------------------
# 2. the analytic stability window, verified numerically
# --------------------------------------------------------------------------
def test_stability_window_scales_as_predicted() -> None:
    growth: list[float] = []
    for radius, density in ((RADIUS, DENSITY), (0.02, DENSITY), (RADIUS, 2100.0)):
        clump = _clump(radius, density)
        lam = _rate(clump, MU_R, 0.1, radius)
        window = viscous_stability_dt(_inertia(clump), clump.mass * GRAVITY,
                                      radius, MU_R, 0.1)
        assert window["lambda"] == pytest.approx(lam, rel=1e-12)
        assert window["dt_monotone"] == pytest.approx(2.0 / lam, rel=1e-12)
        assert window["dt_divergence"] == pytest.approx(4.0 / lam, rel=1e-12)
        assert window["dt_safe"] < window["dt_monotone"]
        # just inside / outside the divergence bound
        inside, _ = _spin_down(omega_ref=0.1, dt=3.8 / lam, steps=400, guard=False,
                               radius=radius, density=density)
        assert abs(float(inside.angular_velocity[0])) < 1.0   # bounded below 4
        # ... and grows above the bound; the growth is geometry independent at
        # fixed lambda*dt, which is what "lambda*dt is the control parameter" means
        try:
            outside, _ = _spin_down(omega_ref=0.1, dt=4.4 / lam, steps=400, guard=False,
                                    radius=radius, density=density)
            amplitude = abs(float(outside.angular_velocity[0]))
        except (ValueError, OverflowError, FloatingPointError):
            amplitude = math.inf
        assert amplitude > 10.0
        growth.append(amplitude)
    assert max(growth) / min(growth) < 1.01


def test_damping_gain_is_never_stiffer_than_the_declared_law() -> None:
    inertia = _inertia(_clump())
    gamma = 5.0
    for dt in (0.0, 1e-6, 1e-4, 1e-2, 1.0):
        gain = viscous_damping_gain(gamma, dt, inertia)
        assert gamma * 0.0 <= gain <= gamma * (1.0 + 1e-15)
        assert abs(1.0 - 0.5 * gain * dt / inertia) <= 1.0      # contraction
    assert viscous_damping_gain(gamma, 1e-9, inertia) == pytest.approx(gamma, rel=1e-5)
    assert viscous_damping_gain(gamma, 1e-4, None) == gamma     # legacy signature
    assert viscous_damping_gain(0.0, 1e-4, inertia) == 0.0


# --------------------------------------------------------------------------
# 3. free spin-down against the exact solution
# --------------------------------------------------------------------------
@pytest.mark.parametrize("implicit", [False, True])
def test_free_decay_is_first_order_accurate(implicit: bool) -> None:
    clump = _clump()
    omega_ref = 0.1
    lam = _rate(clump, MU_R, omega_ref)
    for lambda_dt, tolerance in ((0.01, 0.005), (0.05, 0.03)):
        dt = lambda_dt / lam
        steps = max(1, int(round(math.log(1.0 / 0.35) / lambda_dt)))
        body, _ = _spin_down(omega_ref=omega_ref, dt=dt, steps=steps, implicit=implicit)
        measured = float(body.angular_velocity[0])
        exact = math.exp(-lam * steps * dt)
        assert measured == pytest.approx(exact, rel=tolerance)


def test_implicit_branch_stays_bounded_at_the_default_stiffness() -> None:
    body, plane = _spin_down(omega_ref=1.0e-3, dt=2.0e-4, steps=200, implicit=True)
    omega = float(body.angular_velocity[0])
    assert math.isfinite(omega)
    assert abs(omega) <= 1.0 + 1e-12          # no growth, no sign explosion
    ke = 0.5 * _inertia(_clump()) * omega * omega
    assert plane.moment_work() == pytest.approx(0.5 * _inertia(_clump()) - ke, rel=1e-9)


# --------------------------------------------------------------------------
# 4. steady state: the declared law is preserved, the opt-in branch is not
# --------------------------------------------------------------------------
def _incline(theta_deg: float, *, mu_r: float, omega_ref: float, dt: float,
             steps: int, implicit: bool) -> dict[str, float]:
    clump = _clump()
    theta = math.radians(theta_deg)
    normal = np.array([math.sin(theta), 0.0, math.cos(theta)])
    plane = PlaneContact(normal, config=_config(omega_ref=omega_ref, implicit=implicit,
                                                mu_r=mu_r, friction=0.6))
    penetration = clump.mass * GRAVITY * math.cos(theta) / 1.0e5
    body = ClumpBody(clump, position=normal * (RADIUS - penetration),
                     gravity=(0.0, 0.0, -GRAVITY))
    body.run(steps, dt, (plane,))
    omega = float(body.angular_velocity[1])
    downslope = np.array([math.cos(theta), 0.0, -math.sin(theta)])
    return {"omega": omega, "analytic": omega_ref * math.tan(theta) / mu_r,
            "speed": float(body.velocity @ downslope),
            "lambda_dt": _rate(clump, mu_r, omega_ref) * dt}


def test_incline_steady_state_matches_the_closed_form() -> None:
    """``omega* = omega_ref tan(theta)/mu_r`` is exact for the declared law."""
    result = _incline(10.0, mu_r=0.3, omega_ref=0.1, dt=2.0e-4, steps=4000, implicit=False)
    assert result["lambda_dt"] < 2.0
    assert result["omega"] == pytest.approx(result["analytic"], rel=1e-3)
    # rolling without slip: the contact-point speed vanishes, so v = omega r
    assert result["speed"] == pytest.approx(result["omega"] * RADIUS, rel=5e-3)


def test_implicit_branch_inflates_the_steady_state_as_documented() -> None:
    result = _incline(10.0, mu_r=0.3, omega_ref=0.1, dt=2.0e-4, steps=4000, implicit=True)
    inflation = 1.0 + result["lambda_dt"] / 2.0
    assert result["omega"] / result["analytic"] == pytest.approx(inflation, rel=0.02)


# --------------------------------------------------------------------------
# 5. energy bookkeeping
# --------------------------------------------------------------------------
def test_moment_work_equals_the_kinetic_energy_drop() -> None:
    clump = _clump()
    inertia = _inertia(clump)
    for lambda_dt in (0.01, 0.1, 0.5):
        lam = _rate(clump, MU_R, 0.1)
        dt = lambda_dt / lam
        steps = max(1, int(round(math.log(1.0 / 0.35) / lambda_dt)))
        body, plane = _spin_down(omega_ref=0.1, dt=dt, steps=steps)
        omega = float(body.angular_velocity[0])
        drop = 0.5 * inertia * (1.0 - omega * omega)
        assert plane.moment_work() == pytest.approx(drop, rel=1e-9)
        assert plane.moment_power_max() <= 1e-12


def test_zero_rolling_friction_is_untouched() -> None:
    clump = _clump()
    plane = PlaneContact((0.0, 0.0, 1.0),
                         config=ContactConfig(normal_stiffness=1.0e5, restitution=0.5,
                                              friction=0.3, rolling_friction=0.0))
    body = _body(clump)
    body.run(100, 2.0e-4, (plane,))
    assert plane.moment_work() == 0.0
    assert math.isfinite(float(np.linalg.norm(body.angular_velocity)))