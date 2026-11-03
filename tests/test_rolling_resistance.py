"""Verification tests for the rolling-resistance contact laws.

Three levels are checked:

1. **Law-level invariants** (torque direction, saturation, spin separation,
   plastic memory, energy sign) -- cheap, analytic.
2. **Analytical inclined-plane references** (``theta_c = arctan(mu_r)`` and
   ``a = g(sin t - mu_r cos t)/(1 + I/mR^2)``) matched by an independent
   rigid-body integration that only uses the contact laws of
   :mod:`tensordem.rolling_resistance`.
3. **Energy semantics**: the viscous branch must be strictly dissipative and
   the elastic-plastic branch conservative within its elastic range.
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.rolling_resistance import (
    RollingResistanceConfig,
    SphereInclineConfig,
    critical_incline_angle_deg,
    fit_acceleration,
    incline_rolling_acceleration,
    incline_sliding_acceleration,
    pure_rolling_feasible,
    rolling_component,
    rolling_resistance_torque,
    simulate_sphere_on_incline,
)

DT = 1.0e-4
K_N = 2.0e4


def _harness(**kwargs) -> SphereInclineConfig:
    return SphereInclineConfig(k_n=K_N, dt=DT, **kwargs)


def _k_r(mu_r: float, radius: float = 0.01, k_n: float = K_N) -> float:
    """Rolling stiffness scaled with the normal stiffness (a modelling choice)."""
    return 2.25 * mu_r * k_n * radius ** 2


class ConfigValidationTests(unittest.TestCase):
    def test_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            RollingResistanceConfig(model="wobble", mu_r=0.1)

    def test_rejects_negative_mu(self):
        with self.assertRaises(ValueError):
            RollingResistanceConfig(model="cdt", mu_r=-0.1)

    def test_epsd_requires_stiffness(self):
        with self.assertRaises(ValueError):
            RollingResistanceConfig(model="epsd", mu_r=0.1)

    def test_epsd_rejects_nonpositive_stiffness(self):
        with self.assertRaises(ValueError):
            RollingResistanceConfig(model="epsd", mu_r=0.1, k_r=0.0)

    def test_zero_mu_disables_model(self):
        config = RollingResistanceConfig(model="cdt", mu_r=0.0)
        self.assertFalse(config.enabled)
        omega = torch.zeros(2, 3, dtype=torch.float64)
        omega[:, 2] = -3.0
        f_n = torch.ones(2, dtype=torch.float64)
        r_eff = torch.full((2,), 0.01, dtype=torch.float64)
        torque, state = rolling_resistance_torque(omega, f_n, r_eff, config)
        self.assertTrue(torch.equal(torque, torch.zeros_like(omega)))
        self.assertTrue(torch.equal(state, torch.zeros_like(omega)))


class ConstantDirectionalTorqueTests(unittest.TestCase):
    """Model 1 of Ai et al. (2011): saturated torque opposing rolling."""

    def setUp(self):
        self.config = RollingResistanceConfig(model="cdt", mu_r=0.3)
        self.f_n = torch.tensor([2.0, 5.0], dtype=torch.float64)
        self.r_eff = torch.tensor([0.01, 0.02], dtype=torch.float64)

    def test_magnitude_equals_rolling_limit(self):
        omega = torch.zeros(2, 3, dtype=torch.float64)
        omega[:, 2] = -4.0
        torque, _ = rolling_resistance_torque(omega, self.f_n, self.r_eff, self.config)
        expected = 0.3 * self.f_n * self.r_eff
        self.assertTrue(torch.allclose(torque.norm(dim=1), expected, rtol=1e-14))

    def test_torque_opposes_rotation(self):
        omega = torch.tensor([[0.0, 0.0, -4.0], [0.0, 1.5, 0.0]], dtype=torch.float64)
        torque, _ = rolling_resistance_torque(omega, self.f_n, self.r_eff, self.config)
        # Collinear and antiparallel: torque . omega == -|torque| |omega|
        dot = (torque * omega).sum(dim=1)
        self.assertTrue(torch.all(dot < 0.0))
        self.assertTrue(
            torch.allclose(dot, -torque.norm(dim=1) * omega.norm(dim=1), rtol=1e-12)
        )

    def test_spin_about_normal_is_not_resisted(self):
        normal = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        omega = torch.zeros(2, 3, dtype=torch.float64)
        omega[:, 1] = -6.0  # pure drilling spin about the contact normal
        torque, _ = rolling_resistance_torque(
            omega, self.f_n, self.r_eff, self.config, normal=normal
        )
        self.assertTrue(torch.allclose(torque, torch.zeros_like(omega), atol=0.0))

    def test_rolling_component_is_projected(self):
        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        omega = torch.tensor([[1.0, 7.0, 2.0]], dtype=torch.float64)
        roll = rolling_component(omega, normal)
        self.assertTrue(torch.allclose(roll, torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float64)))

    def test_zero_velocity_regularisation(self):
        omega = torch.zeros(2, 3, dtype=torch.float64)
        torque, _ = rolling_resistance_torque(omega, self.f_n, self.r_eff, self.config)
        self.assertTrue(torch.allclose(torque, torch.zeros_like(omega)))

    def test_separated_contact_carries_no_torque(self):
        omega = torch.zeros(2, 3, dtype=torch.float64)
        omega[:, 2] = -4.0
        f_n = torch.tensor([2.0, 0.0], dtype=torch.float64)
        torque, _ = rolling_resistance_torque(omega, f_n, self.r_eff, self.config)
        self.assertGreater(float(torque[0].norm()), 0.0)
        self.assertTrue(torch.allclose(torque[1], torch.zeros(3, dtype=torch.float64)))

    def test_rejects_negative_normal_force(self):
        omega = torch.zeros(1, 3, dtype=torch.float64)
        with self.assertRaises(ValueError):
            rolling_resistance_torque(
                omega, torch.tensor([-1.0], dtype=torch.float64), self.r_eff[:1], self.config
            )


class ViscousModelTests(unittest.TestCase):
    def test_torque_is_linear_in_omega_and_saturates_at_omega_ref(self):
        config = RollingResistanceConfig(model="viscous", mu_r=0.2, omega_ref=2.0)
        f_n = torch.tensor([3.0], dtype=torch.float64)
        r_eff = torch.tensor([0.01], dtype=torch.float64)
        limit = 0.2 * 3.0 * 0.01
        for speed in (0.5, 2.0, 9.0):
            omega = torch.tensor([[0.0, 0.0, -speed]], dtype=torch.float64)
            torque, _ = rolling_resistance_torque(omega, f_n, r_eff, config)
            expected = min(limit * speed / 2.0, float("inf"))
            self.assertAlmostEqual(float(torque.norm()), expected, places=14)
        omega = torch.zeros(1, 3, dtype=torch.float64)
        torque, _ = rolling_resistance_torque(omega, f_n, r_eff, config)
        self.assertAlmostEqual(float(torque.norm()), 0.0, places=15)

    def test_power_always_dissipative(self):
        config = RollingResistanceConfig(model="viscous", mu_r=0.25)
        generator = torch.Generator().manual_seed(20261101)
        omega = torch.randn(64, 3, generator=generator, dtype=torch.float64)
        f_n = torch.rand(64, generator=generator, dtype=torch.float64) + 1e-3
        r_eff = torch.rand(64, generator=generator, dtype=torch.float64) * 0.02 + 1e-3
        torque, _ = rolling_resistance_torque(omega, f_n, r_eff, config)
        power = (torque * omega).sum(dim=1)
        self.assertTrue(torch.all(power <= 1e-15))


class ElasticPlasticSpringTests(unittest.TestCase):
    """Model 3 of Ai et al. (2011): EPSD rolling spring with plastic clamp."""

    def setUp(self):
        self.mu_r = 0.25
        self.k_r = _k_r(self.mu_r)
        self.config = RollingResistanceConfig(model="epsd", mu_r=self.mu_r, k_r=self.k_r)
        self.f_n = torch.tensor([1.5], dtype=torch.float64)
        self.r_eff = torch.tensor([0.01], dtype=torch.float64)
        self.limit = self.mu_r * 1.5 * 0.01

    def _drive(self, speeds, dt=DT, state=None):
        """Integrate the law over a prescribed spin history.

        Work is accumulated with the trapezoidal rule,
        ``dW = 0.5 (M_n + M_{n+1}) . dtheta``, which is exact for the linear
        rolling spring: with the plain right-endpoint rule the bookkeeping
        carries an O(dtheta^2) defect per step that would be mistaken for
        non-conservation.
        """
        state = torch.zeros(1, 3, dtype=torch.float64) if state is None else state
        work = 0.0
        for speed in speeds:
            omega = torch.tensor([[0.0, 0.0, float(speed)]], dtype=torch.float64)
            previous = state
            torque, state = rolling_resistance_torque(
                omega, self.f_n, self.r_eff, self.config, torque_state=state, dt=dt
            )
            work += 0.5 * float(((previous + torque) * omega).sum()) * dt
        return state, work

    def test_elastic_slope_is_rolling_stiffness(self):
        # Ten steps at omega = -0.01 rad/s accumulate dtheta = 0.01 * 10 * dt.
        state, _ = self._drive([-0.01] * 10)  # small rotation, far below the limit
        expected = self.k_r * 0.01 * 10 * DT
        self.assertLess(abs(float(state.norm())), self.limit)
        self.assertAlmostEqual(float(state.norm()), expected, places=16)
        # The EPSD branch reports the elastic spring torque, i.e. the new state.
        omega = torch.tensor([[0.0, 0.0, -0.01]], dtype=torch.float64)
        torque, new_state = rolling_resistance_torque(
            omega, self.f_n, self.r_eff, self.config, torque_state=state, dt=DT
        )
        self.assertTrue(torch.equal(torque, new_state))

    def test_saturates_at_limit(self):
        state, _ = self._drive([-50.0] * 5)
        self.assertAlmostEqual(float(state.norm()), self.limit, places=14)

    def test_plastic_memory_after_unloading(self):
        state, _ = self._drive([-50.0] * 5)
        held, _ = self._drive([0.0] * 20, state=state)
        self.assertAlmostEqual(float(held.norm()), self.limit, places=14)

    def test_elastic_cycle_is_conservative(self):
        outbound = [-0.02] * 25
        inbound = [0.02] * 25
        state, work = self._drive(outbound + inbound)
        self.assertAlmostEqual(float(state.norm()), 0.0, places=13)
        self.assertAlmostEqual(work, 0.0, places=13)

    def test_saturated_rotation_dissipates(self):
        _, work = self._drive([-5.0] * 40)
        self.assertLess(work, 0.0)

    def test_state_resets_on_separation(self):
        state, _ = self._drive([-50.0] * 5)
        torque, reset = rolling_resistance_torque(
            torch.tensor([[-0.0, 0.0, -1.0]], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            self.r_eff,
            self.config,
            torque_state=state,
            dt=DT,
        )
        self.assertAlmostEqual(float(torque.norm()), 0.0, places=15)
        self.assertAlmostEqual(float(reset.norm()), 0.0, places=15)

    def test_requires_time_step(self):
        state = torch.zeros(1, 3, dtype=torch.float64)
        with self.assertRaises(ValueError):
            rolling_resistance_torque(
                state, self.f_n, self.r_eff, self.config, torque_state=state, dt=None
            )


class AnalyticalReferenceTests(unittest.TestCase):
    def test_incline_acceleration_vanishes_at_critical_angle(self):
        for mu_r in (0.05, 0.2, 0.5):
            theta_c = critical_incline_angle_deg(mu_r)
            self.assertAlmostEqual(
                incline_rolling_acceleration(theta_c, mu_r), 0.0, places=12
            )

    def test_infinite_inertia_limit(self):
        # Without rotational inertia the sphere slides: a = g(sin t - mu_r cos t).
        theta = 20.0
        mu_r = 0.1
        a = incline_rolling_acceleration(
            theta, mu_r, inertia_factor=1e-15
        )
        expected = 9.81 * (math.sin(math.radians(theta)) - mu_r * math.cos(math.radians(theta)))
        self.assertAlmostEqual(a, expected, places=12)

    def test_pure_rolling_feasibility_transition(self):
        # mu = 0.9, mu_r = 0.2: threshold tan(t) = (0.9*1.4 - 0.2)/0.4 = 2.65.
        limit_deg = math.degrees(math.atan((0.9 * 1.4 - 0.2) / 0.4))
        self.assertTrue(pure_rolling_feasible(limit_deg - 1.0, 0.2, 0.9))
        self.assertFalse(pure_rolling_feasible(limit_deg + 1.0, 0.2, 0.9))

    def test_sliding_reference_at_steep_incline(self):
        # mu = 0.9 and mu_r = 0.2: pure rolling is feasible up to 69.3 deg
        # (tan t = (0.9*1.4 - 0.2)/0.4 = 2.65), so 60 deg rolls and 75 deg slips.
        self.assertTrue(pure_rolling_feasible(60.0, 0.2, 0.9))
        self.assertFalse(pure_rolling_feasible(75.0, 0.2, 0.9))
        a = incline_sliding_acceleration(75.0, 0.9)
        self.assertAlmostEqual(
            a, 9.81 * (math.sin(math.radians(75.0)) - 0.9 * math.cos(math.radians(75.0))),
            places=12,
        )

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            pure_rolling_feasible(10.0, -0.1, 0.9)
        with self.assertRaises(ValueError):
            incline_sliding_acceleration(10.0, float("nan"))

    def test_critical_angle_monotone_in_mu(self):
        values = [critical_incline_angle_deg(m) for m in (0.0, 0.1, 0.5, 1.0)]
        self.assertEqual(values[0], 0.0)
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))


class InclineSimulationTests(unittest.TestCase):
    """Independent rigid-body check of the contact law against the analytics."""

    MU_R = 0.05
    DURATION = 0.06
    SETTLE = 0.02

    def _config(self, model="epsd"):
        return RollingResistanceConfig(
            model=model, mu_r=self.MU_R, k_r=_k_r(self.MU_R)
        )

    def test_supercritical_acceleration_matches_analytic(self):
        theta_c = critical_incline_angle_deg(self.MU_R)
        angles = [1.2 * theta_c, 1.5 * theta_c, 10.0, 15.0]
        result = simulate_sphere_on_incline(
            angles,
            self._config(),
            config=_harness(),
            duration=self.DURATION,
            settle_time=self.SETTLE,
        )
        accel, r_squared = fit_acceleration(result)
        self.assertTrue(torch.all(r_squared > 0.99))
        for index, theta in enumerate(angles):
            expected = incline_rolling_acceleration(theta, self.MU_R)
            self.assertGreater(expected, 0.0)
            relative_error = abs(float(accel[index]) - expected) / expected
            self.assertLess(
                relative_error,
                0.03,
                msg=f"theta={theta} deg: simulated {float(accel[index]):.5f} vs "
                f"analytic {expected:.5f} m/s^2",
            )

    def test_pure_rolling_without_slip(self):
        result = simulate_sphere_on_incline(
            [12.0],
            self._config(),
            config=_harness(),
            duration=self.DURATION,
            settle_time=self.SETTLE,
        )
        radius = _harness().radius
        ratio = float(result["omega"][-1, 0] * radius / result["v"][-1, 0])
        # Rolling without slip: the contact-point velocity vanishes, omega R = -v.
        self.assertAlmostEqual(ratio, -1.0, delta=0.02)

    def test_subcritical_angle_stays_at_rest(self):
        theta_c = critical_incline_angle_deg(self.MU_R)
        result = simulate_sphere_on_incline(
            [0.8 * theta_c],
            self._config(),
            config=_harness(),
            duration=self.DURATION,
            settle_time=self.SETTLE,
        )
        radius = _harness().radius
        drift = abs(float(result["x"][-1, 0])) / radius
        self.assertLess(drift, 1e-3, msg=f"creep drift {drift:.3e} of the radius")

    def test_constant_directional_torque_gives_same_sliding_regime(self):
        theta = 15.0
        result = simulate_sphere_on_incline(
            [theta],
            self._config(model="cdt"),
            config=_harness(),
            duration=self.DURATION,
            settle_time=self.SETTLE,
        )
        accel, _ = fit_acceleration(result)
        expected = incline_rolling_acceleration(theta, self.MU_R)
        self.assertLess(abs(float(accel[0]) - expected) / expected, 0.03)

    def test_zero_rolling_resistance_accelerates_faster(self):
        theta = 15.0
        frictionless = RollingResistanceConfig(model="epsd", mu_r=0.0, k_r=1.0)
        baseline = simulate_sphere_on_incline(
            [theta], frictionless, config=_harness(),
            duration=self.DURATION, settle_time=self.SETTLE,
        )
        with_rr = simulate_sphere_on_incline(
            [theta], self._config(), config=_harness(),
            duration=self.DURATION, settle_time=self.SETTLE,
        )
        a_base, _ = fit_acceleration(baseline)
        a_rr, _ = fit_acceleration(with_rr)
        expected_base = 9.81 * math.sin(math.radians(theta)) / (1.0 + 0.4)
        self.assertLess(abs(float(a_base[0]) - expected_base) / expected_base, 0.03)
        self.assertLess(float(a_rr[0]), float(a_base[0]))

    def test_viscous_branch_is_dissipative_in_time_domain(self):
        config = RollingResistanceConfig(model="viscous", mu_r=0.2)
        result = simulate_sphere_on_incline(
            [10.0], config, config=_harness(),
            duration=self.DURATION, settle_time=0.0,
        )
        power = (result["torque"] * result["omega"]).sum(dim=1)
        self.assertTrue(torch.all(power <= 1e-18))


if __name__ == "__main__":
    unittest.main()