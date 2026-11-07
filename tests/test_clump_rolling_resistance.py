"""Clump rolling / drilling resistance: contact law, statics and tilt calibration.

Pinned here:

* **Legacy behaviour is untouched** when ``rolling_friction``/``drilling_friction``
  stay at zero (bit-identical trajectory to a configuration built without the new
  fields), so existing clump results cannot move.
* **The moment law** saturates at ``mu_r |F_n| r`` (rolling) and ``mu_t |F_n| r``
  (drilling), opposes the relative rotation, is pointwise dissipative for ``cdt``
  / ``viscous``, and stores at most ``M_lim**2 / (2 k_r)`` per member for ``epsd``.
* **Statics**: for a single-point support the first-motion angle is exactly
  ``arctan(mu_r)`` -- identical to the verified sphere law of
  :mod:`tensordem.rolling_resistance` -- while an elongated clump is limited by
  Coulomb sliding (``arctan(mu_s)``) because its contact couple ``a/h`` exceeds the
  rolling term.  Both are reproduced by the tilt-table harness to a few percent.
* **Calibration**: ``invert_rolling_friction`` round-trips, and the identifiability
  gate refuses shapes whose support geometry hides the rolling coefficient.
* **Energy**: the elastic-plastic branch can return stored energy but never creates
  it, and the drilling spin-down follows its closed form.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.clump import Clump, SphereMember, double_sphere_clump
from tensordem.clump_dynamics import (ClumpBody, ClumpContact, ContactConfig,
                                      PlaneContact, quaternion_from_axis_angle)
from tensordem.clump_rolling import (calibration_regime, com_height,
                                     critical_tilt_angle_deg, invert_rolling_friction,
                                     resistance_moment, rolling_component,
                                     rolling_friction_identifiable, support_offset,
                                     tilt_test)
from tensordem.rolling_resistance import critical_incline_angle_deg

RADIUS = 0.05
DENSITY = 900.0


def _sphere() -> Clump:
    return Clump([SphereMember((0.0, 0.0, 0.0), RADIUS)], density=DENSITY)


# --------------------------------------------------------------------------
# 1. no regression when the new coefficients are zero
# --------------------------------------------------------------------------
def test_zero_rolling_friction_keeps_legacy_trajectory() -> None:
    clump = double_sphere_clump(RADIUS, density=DENSITY)
    angle = math.radians(10.0)
    normal = np.array([math.sin(angle), 0.0, math.cos(angle)])
    results = []
    for kwargs in ({}, {"rolling_friction": 0.0, "drilling_friction": 0.0}):
        config = ContactConfig(normal_stiffness=1.0e5, restitution=0.5, friction=0.5,
                               **kwargs)
        plane = PlaneContact(normal, config=config)
        penetration = clump.mass * 9.81 * math.cos(angle) / (2.0 * 1.0e5)
        body = ClumpBody(clump, position=normal * (RADIUS - penetration),
                         gravity=(0.0, 0.0, -9.81))
        body.run(200, 2.0e-4, (plane,))
        results.append((body.position.copy(), body.velocity.copy(), plane.moment_work()))
    assert np.array_equal(results[0][0], results[1][0])
    assert np.array_equal(results[0][1], results[1][1])
    assert results[1][2] == 0.0


# --------------------------------------------------------------------------
# 2. moment law: magnitude, direction, dissipation
# --------------------------------------------------------------------------
def test_saturated_moment_magnitude_and_direction() -> None:
    normal = np.array([0.0, 0.0, 1.0])
    omega = np.array([0.0, 3.0, 0.0])
    moment, state = resistance_moment(normal, omega, 4.0, RADIUS, mu_r=0.25, model="cdt")
    assert np.isclose(np.linalg.norm(moment), 0.25 * 4.0 * RADIUS)
    assert moment[1] < 0.0                                  # opposes the rotation
    assert np.allclose(state, 0.0)

    spin, _ = resistance_moment(normal, np.array([0.0, 0.0, 2.0]), 4.0, RADIUS,
                                mu_t=0.05, model="cdt")
    assert np.isclose(np.linalg.norm(spin), 0.05 * 4.0 * RADIUS)
    assert spin[2] < 0.0

    zero, _ = resistance_moment(normal, np.zeros(3), 4.0, RADIUS, mu_r=0.25, model="cdt")
    assert np.allclose(zero, 0.0)
    none, _ = resistance_moment(normal, omega, 0.0, RADIUS, mu_r=0.25, model="cdt")
    assert np.allclose(none, 0.0)


def test_rolling_and_drilling_components_are_orthogonal() -> None:
    normal = np.array([0.0, 0.0, 1.0])
    roll, spin = rolling_component(np.array([1.0, 2.0, 3.0]), normal)
    assert np.isclose(roll @ spin, 0.0)
    assert np.isclose(spin[2], 3.0)


@pytest.mark.parametrize("model", ["cdt", "viscous"])
def test_moment_is_pointwise_dissipative(model: str) -> None:
    normal = np.array([0.3, -0.2, 0.9])
    normal = normal / np.linalg.norm(normal)
    rng = np.random.default_rng(7)
    for _ in range(50):
        omega = rng.normal(size=3)
        moment, _ = resistance_moment(normal, omega, 7.0, RADIUS, mu_r=0.3,
                                      model=model, omega_ref=0.1)
        assert float(moment @ omega) <= 1e-15


def test_epsd_stores_bounded_moment() -> None:
    normal = np.array([0.0, 0.0, 1.0])
    omega = np.array([0.0, 0.5, 0.0])
    force, radius, mu_r = 6.0, RADIUS, 0.2
    k_r = 2.25 * mu_r * 1.0e5 * radius ** 2
    limit = mu_r * force * radius
    state = np.zeros(3)
    for _ in range(2000):
        _, state = resistance_moment(normal, omega, force, radius, mu_r=mu_r,
                                     model="epsd", moment_state=state, dt=2.0e-4,
                                     normal_stiffness=1.0e5)
        assert np.linalg.norm(state) <= limit * (1.0 + 1e-12)
    stored = float(state @ state) / (2.0 * k_r)
    assert stored <= limit ** 2 / (2.0 * k_r) * (1.0 + 1e-12)
    assert np.linalg.norm(state) == pytest.approx(limit, rel=1e-9)


# --------------------------------------------------------------------------
# 3. statics: analytic thresholds and their cross-check
# --------------------------------------------------------------------------
def test_single_point_support_reduces_to_sphere_law() -> None:
    for mu_r in (0.05, 0.2, 0.35):
        theta = critical_tilt_angle_deg(mu_r, RADIUS, RADIUS, downhill_offset=0.0)
        assert theta == pytest.approx(math.degrees(math.atan(mu_r)), rel=0, abs=1e-12)
        assert theta == pytest.approx(critical_incline_angle_deg(mu_r), rel=0, abs=1e-12)


def test_couple_and_sliding_branches() -> None:
    # elongated clump: a/h = 1 dominates the rolling term, sliding binds at mu_s
    theta = critical_tilt_angle_deg(0.2, RADIUS, RADIUS, mu_s=0.9, downhill_offset=RADIUS)
    assert theta == pytest.approx(math.degrees(math.atan(0.9)), abs=1e-12)
    # with a high sliding coefficient the couple + rolling branch binds
    theta = critical_tilt_angle_deg(0.2, RADIUS, RADIUS, mu_s=3.0, downhill_offset=RADIUS)
    assert theta == pytest.approx(math.degrees(math.atan(1.2)), abs=1e-12)


def test_geometry_helpers_read_the_clump() -> None:
    dumbbell = double_sphere_clump(RADIUS, density=DENSITY)
    assert com_height(dumbbell) == pytest.approx(RADIUS)
    assert support_offset(dumbbell) == pytest.approx(RADIUS)
    sphere = _sphere()
    assert com_height(sphere) == pytest.approx(RADIUS)
    assert support_offset(sphere) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 4. calibration inversion and identifiability
# --------------------------------------------------------------------------
def test_calibration_round_trip() -> None:
    for mu_r in (0.05, 0.1, 0.25):
        theta = critical_tilt_angle_deg(mu_r, RADIUS, RADIUS, mu_s=0.9, downhill_offset=0.0)
        assert invert_rolling_friction(theta, RADIUS, RADIUS) == pytest.approx(mu_r, rel=1e-12)


def test_identifiability_gate() -> None:
    # sphere: rolling is identifiable whenever mu_r < mu_s
    assert rolling_friction_identifiable(RADIUS, RADIUS, mu_s=0.9, downhill_offset=0.0)
    assert calibration_regime(11.3, 0.2, RADIUS, RADIUS, mu_s=0.9) == "rolling+couple"
    # elongated clump with a = h and mu_s = 0.9: sliding binds, no information on mu_r
    assert calibration_regime(42.0, 0.2, RADIUS, RADIUS, mu_s=0.9,
                              downhill_offset=RADIUS) == "sliding"
    # a measured angle explained by the support geometry alone clamps to zero
    assert invert_rolling_friction(41.99, RADIUS, RADIUS, downhill_offset=RADIUS) == 0.0


# --------------------------------------------------------------------------
# 5. tilt-table harness against the statics
# --------------------------------------------------------------------------
def test_tilt_test_sphere_matches_arctan_mu_r() -> None:
    sphere = _sphere()
    mu_r = 0.2
    result = tilt_test(sphere, 11.31, mu_r=mu_r, mu_s=0.9, duration=0.30, dt=2.0e-4)
    assert not result.moved
    below = tilt_test(sphere, 9.0, mu_r=mu_r, mu_s=0.9, duration=0.30, dt=2.0e-4)
    assert not below.moved
    above = tilt_test(sphere, 14.0, mu_r=mu_r, mu_s=0.9, duration=0.30, dt=2.0e-4)
    assert above.moved
    assert above.rotation_deg > 1.0                          # it actually rolls
    assert above.dissipated_moment_work > 0.0


def test_tilt_test_clump_is_sliding_limited() -> None:
    dumbbell = double_sphere_clump(RADIUS, density=DENSITY)
    mu_r, mu_s = 0.2, 0.9
    reference = math.degrees(math.atan(mu_s))
    below = tilt_test(dumbbell, 0.95 * reference, mu_r=mu_r, mu_s=mu_s,
                      duration=0.30, dt=2.0e-4)
    above = tilt_test(dumbbell, 1.05 * reference, mu_r=mu_r, mu_s=mu_s,
                      duration=0.30, dt=2.0e-4)
    assert not below.moved
    assert above.moved
    # sliding: translation without the rolling relation |dx| = r |dphi|
    ratio = (math.radians(above.rotation_deg) * RADIUS
             / abs(above.downhill_displacement))
    assert ratio < 0.3


# --------------------------------------------------------------------------
# 6. energy audit and drilling spin-down
# --------------------------------------------------------------------------
def test_no_energy_creation_below_threshold() -> None:
    sphere = _sphere()
    angle = math.radians(9.0)
    config = ContactConfig(normal_stiffness=1.0e5, restitution=0.2, friction=0.9,
                           rolling_friction=0.2, rolling_model="epsd")
    normal = np.array([math.sin(angle), 0.0, math.cos(angle)])
    plane = PlaneContact(normal, config=config)
    penetration = sphere.mass * 9.81 * math.cos(angle) / 1.0e5
    body = ClumpBody(sphere, position=normal * (RADIUS - penetration),
                     orientation=quaternion_from_axis_angle((0, 1, 0), angle),
                     gravity=(0.0, 0.0, -9.81))
    history = body.run(1500, 2.0e-4, (plane,), record_every=25)
    energy = np.asarray(history["total_energy"], dtype=np.float64)
    assert np.max(energy - energy[0]) <= 1.0e-9
    assert plane.moment_work() > 0.0
    assert isinstance(plane.rolling_moment_state(), np.ndarray)


def test_drilling_spin_down_closed_form() -> None:
    sphere = _sphere()
    mu_t, omega0, dt = 0.05, 2.0, 5.0e-5
    config = ContactConfig(normal_stiffness=1.0e5, restitution=0.2, friction=0.9,
                           drilling_friction=mu_t)
    plane = PlaneContact((0.0, 0.0, 1.0), config=config)
    penetration = sphere.mass * 9.81 / 1.0e5
    body = ClumpBody(sphere, position=(0.0, 0.0, RADIUS - penetration),
                     angular_velocity=(0.0, 0.0, omega0), gravity=(0.0, 0.0, -9.81))
    steps = 800
    body.run(steps, dt, (plane,))
    inertia = float(sphere.inertia_about_com[2, 2])
    alpha = -mu_t * sphere.mass * 9.81 * RADIUS / inertia
    expected = omega0 + alpha * steps * dt
    assert float(body.angular_velocity[2]) == pytest.approx(expected, rel=1e-9)
    assert np.linalg.norm(body.position[:2]) < 1.0e-9        # pure spin, no drift


def test_reset_clears_resistance_state() -> None:
    sphere = _sphere()
    config = ContactConfig(normal_stiffness=1.0e5, friction=0.9, rolling_friction=0.2)
    plane = PlaneContact((0.0, 0.0, 1.0), config=config)
    body = ClumpBody(sphere, position=(0.0, 0.0, RADIUS - 1.0e-5),
                     angular_velocity=(0.5, 0.0, 0.0), gravity=(0.0, 0.0, -9.81))
    body.run(50, 2.0e-4, (plane,))
    assert np.any(plane.rolling_moment_state() != 0.0)
    plane.reset()
    assert plane.rolling_moment_state().shape == (0, 3)
    assert plane.moment_work() == 0.0
    assert isinstance(plane, ClumpContact)