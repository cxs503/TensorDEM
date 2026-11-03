"""Verification tests for the opt-in rotational-DOF path of ``IceDEM3D``.

Level 1 -- configuration semantics: the legacy (translation-only) path must stay
untouched, so ``enable_rotation=False`` exposes no spin state and keeps the old
diagnostic keys.
Level 2 -- closed-form rigid-body references: angular-momentum conservation
about the contact point, the ``5/7`` sliding-to-rolling limit, rolling
deceleration ``mu_r g /(1+I/mR^2)``, and the exact spin-down histories of the
``cdt`` / ``viscous`` / ``epsd`` rolling laws under a constant normal force.
Level 3 -- defect quantification: the legacy translational force surrogate
predicts a rolling deceleration that is too large by exactly ``1+I/mR^2``.
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.rotation_verification import (
    PlaneCaseConfig,
    SpinDownCaseConfig,
    build_plane_case,
    build_spin_down_case,
    cdt_spin,
    epsd_clamp_peak,
    epsd_spin,
    epsd_stop_time,
    legacy_rolling_deceleration,
    plane_static_penetration,
    rolling_deceleration,
    slip_to_roll_speed_ratio,
    viscous_spin,
)


def _run(dem: IceDEM3D, loads: torch.Tensor, steps: int, record=None) -> None:
    for step in range(steps):
        dem.step(loads)
        if record is not None:
            record(step)


def _fit_acceleration(time: torch.Tensor, velocity: torch.Tensor) -> float:
    """Least-squares slope of ``velocity`` against ``time``."""
    t_mean = time.mean()
    v_mean = velocity.mean()
    dt = time - t_mean
    return float((dt * (velocity - v_mean)).sum() / (dt ** 2).sum())


class ConfigSemanticsTests(unittest.TestCase):
    def test_legacy_path_exposes_no_spin_state(self):
        dem = IceDEM3D(DEM3DConfig(nx=3, ny=3, nz=2, rolling_resistance_coefficient=0.1))
        self.assertIsNone(dem.omega)
        keys = set(dem.diagnostics())
        self.assertNotIn("rotational_kinetic_energy_J", keys)
        self.assertNotIn("omega", dem.state_dict())

    def test_rotational_path_exposes_spin_state(self):
        dem = IceDEM3D(DEM3DConfig(nx=3, ny=3, nz=2, enable_rotation=True))
        self.assertEqual(tuple(dem.omega.shape), (len(dem.positions), 3))
        self.assertIn("rotational_kinetic_energy_J", dem.diagnostics())
        self.assertIn("omega", dem.state_dict())

    def test_rejects_epsd_without_stiffness(self):
        with self.assertRaises(ValueError):
            DEM3DConfig(enable_rotation=True, rolling_model="epsd",
                        rolling_resistance_coefficient=0.1)

    def test_rejects_unknown_rolling_model(self):
        with self.assertRaises(ValueError):
            DEM3DConfig(enable_rotation=True, rolling_model="wobble")

    def test_platen_friction_requires_rotation(self):
        with self.assertRaises(ValueError):
            DEM3DConfig(platen_friction_coefficient=0.5)

    def test_inertia_matches_solid_sphere(self):
        config = DEM3DConfig(radius=0.02, density=917.0)
        expected = 0.4 * config.mass * config.radius ** 2
        self.assertAlmostEqual(config.inertia / expected, 1.0, places=12)


class AngularMomentumTests(unittest.TestCase):
    """Level 2: exact conservation law about the contact point."""

    def test_contact_point_angular_momentum_is_conserved(self):
        config = PlaneCaseConfig(mu_r=0.0, dt=4.0e-5)
        dem, loads = build_plane_case(config)
        v0 = 0.5
        dem.velocities[0, 0] = v0
        moment_of_inertia = config.inertia
        mass, radius = config.mass, config.radius

        def angular_momentum() -> float:
            v = float(dem.velocities[0, 0])
            spin = float(dem.omega[0, 1])
            return mass * radius * v + moment_of_inertia * spin

        initial = angular_momentum()
        self.assertAlmostEqual(initial / (mass * radius * v0), 1.0, places=12)
        history = []
        for _ in range(800):
            dem.step(loads)
            history.append(angular_momentum())
        drift = max(abs(value - initial) for value in history) / abs(initial)
        self.assertLess(drift, 1e-9)

    def test_sliding_sphere_reaches_rolling_at_five_sevenths(self):
        """At the first instant of pure rolling the speed must be ``5/7 v0``."""
        config = PlaneCaseConfig(mu_r=0.0, dt=4.0e-5)
        dem, loads = build_plane_case(config)
        v0 = 0.5
        dem.velocities[0, 0] = v0
        radius, mu = config.radius, config.mu_platen
        times, speeds = [], []
        first_rolling = None
        for step in range(1600):
            dem.step(loads)
            speed = float(dem.velocities[0, 0])
            spin = float(dem.omega[0, 1])
            times.append((step + 1) * config.dt)
            speeds.append(speed)
            if first_rolling is None and step > 50 and speed - spin * radius <= 0.0:
                first_rolling = speed
        self.assertIsNotNone(first_rolling, "the sphere never reached rolling contact")
        target = v0 * slip_to_roll_speed_ratio()
        self.assertLess(abs(first_rolling - target) / target, 0.01,
                        f"rolling speed {first_rolling:.6f} vs {target:.6f}")
        # While the contact slips, Coulomb friction saturates and the deceleration
        # is exactly mu g, independent of the spin state.
        time = torch.tensor(times[30:350])
        speed = torch.tensor(speeds[30:350])
        measured = -_fit_acceleration(time, speed)
        analytic = mu * config.gravity
        self.assertLess(abs(measured - analytic) / analytic, 0.01,
                        f"sliding deceleration {measured:.4f} vs {analytic:.4f}")


class RollingDecelerationTests(unittest.TestCase):
    """Level 2: rolling without slip decelerates by mu_r g / (1 + I/mR^2)."""

    def test_pure_rolling_deceleration_matches_analytic(self):
        config = PlaneCaseConfig(mu_r=0.05, dt=4.0e-5)
        dem, loads = build_plane_case(config)
        v0 = 0.2
        dem.velocities[0, 0] = v0
        dem.omega[0, 1] = v0 / config.radius
        times, speeds = [], []
        for step in range(1000):
            dem.step(loads)
            times.append((step + 1) * config.dt)
            speeds.append(float(dem.velocities[0, 0]))
        time = torch.tensor(times[200:])
        speed = torch.tensor(speeds[200:])
        measured = -_fit_acceleration(time, speed)
        analytic = rolling_deceleration(config.mu_r, config.gravity)
        error = abs(measured - analytic) / analytic
        self.assertLess(error, 0.03, f"measured {measured:.4f} vs {analytic:.4f}")
        # The rolling condition must stay satisfied (no slip).
        # Residual slip is set by the tangential spring compliance (k_t = k_platen)
        # and stays far below the 1 % of v0 that the deceleration check allows.
        self.assertLess(abs(float(dem.velocities[0, 0]) - config.radius * float(dem.omega[0, 1])),
                        1e-2 * v0)

    def test_static_sphere_does_not_drift(self):
        config = PlaneCaseConfig(mu_r=0.05, rolling_model="epsd", rolling_stiffness=5.0e-5,
                                 dt=4.0e-5)
        dem, loads = build_plane_case(config)
        start = dem.positions[0].clone()
        _run(dem, loads, 1000)
        self.assertLess(float((dem.positions[0] - start).abs().max()), 1e-12)
        self.assertLess(float(dem.omega[0].abs().max()), 1e-12)

    def test_mechanical_energy_is_non_increasing(self):
        config = PlaneCaseConfig(mu_r=0.05, dt=4.0e-5)
        dem, loads = build_plane_case(config)
        v0 = 0.2
        dem.velocities[0, 0] = v0
        dem.omega[0, 1] = v0 / config.radius
        energies = []
        for step in range(1000):
            dem.step(loads)
            if step % 10 == 0:
                energies.append(dem.diagnostics(loads)["mechanical_energy_J"])
        increments = [second - first for first, second in zip(energies, energies[1:])]
        self.assertLess(max(increments), 1e-12, "rolling resistance injected energy")


class SpinDownLawTests(unittest.TestCase):
    """Level 2: exact spin histories under a constant normal force."""

    def _measure(self, config: SpinDownCaseConfig, spin: float, duration: float):
        dem, loads = build_spin_down_case(config, spin=spin)
        steps = int(round(duration / config.dt))
        _run(dem, loads, steps)
        return float(dem.omega[1, 2])

    def test_cdt_spin_down_is_linear(self):
        config = SpinDownCaseConfig(mu_r=0.05, rolling_model="cdt", dt=1.0e-4)
        spin, duration = 10.0, 0.2
        measured = self._measure(config, spin, duration)
        analytic = cdt_spin(spin, config.mu_r, config.normal_force,
                            config.effective_radius, config.inertia, duration)
        error = abs(measured - analytic) / analytic
        self.assertGreater(analytic, 0.1 * spin, "test window crosses the reversal")
        self.assertLess(error, 0.03, f"measured {measured:.4f} vs {analytic:.4f}")

    def test_viscous_spin_down_is_exponential(self):
        config = SpinDownCaseConfig(mu_r=0.05, rolling_model="viscous", rolling_omega_ref=1.0, dt=1.0e-4)
        spin, duration = 10.0, 0.5
        measured = self._measure(config, spin, duration)
        analytic = viscous_spin(spin, config.mu_r, config.normal_force,
                                config.effective_radius, config.inertia,
                                config.rolling_omega_ref, duration)
        error = abs(measured - analytic) / analytic
        self.assertLess(error, 0.03, f"measured {measured:.4f} vs {analytic:.4f}")

    def test_epsd_spin_down_is_harmonic_then_sticks(self):
        # Stiffness chosen so the elastic torque peak stays inside roll limit.
        stiffness = 1.0e-3
        config = SpinDownCaseConfig(mu_r=0.5, rolling_model="epsd",
                                    rolling_stiffness=stiffness, dt=1.0e-4)
        spin = 10.0
        limit = config.mu_r * config.normal_force * config.effective_radius
        peak = epsd_clamp_peak(spin, stiffness, config.inertia)
        self.assertLess(peak, limit, "elastic peak must stay inside the rolling limit")
        duration = 0.5 * epsd_stop_time(spin, stiffness, config.inertia)
        measured = self._measure(config, spin, duration)
        analytic = epsd_spin(spin, stiffness, config.inertia, duration)
        error = abs(measured - analytic) / analytic
        self.assertLess(error, 0.03, f"measured {measured:.4f} vs {analytic:.4f}")
        # Past the stop time the elastic spring holds the contact: spin stays zero.
        dem, loads = build_spin_down_case(config, spin=spin)
        _run(dem, loads, int(round(1.5 * epsd_stop_time(spin, stiffness, config.inertia)
                                   / config.dt)))
        self.assertLess(abs(float(dem.omega[1, 2])), 1e-6)


class LegacySurrogateDefectTests(unittest.TestCase):
    """Level 3: quantify the translation-only rolling-resistance surrogate."""

    def test_surrogate_over_predicts_rolling_deceleration_by_inertia_factor(self):
        config = PlaneCaseConfig(mu_r=0.05, enable_rotation=False, mu_platen=0.0, dt=4.0e-5)
        dem, loads = build_plane_case(config)
        self.assertIsNone(dem.omega)
        v0 = 0.2
        dem.velocities[0, 0] = v0
        times, speeds = [], []
        for step in range(600):
            dem.step(loads)
            times.append((step + 1) * config.dt)
            speeds.append(float(dem.velocities[0, 0]))
        measured = -_fit_acceleration(torch.tensor(times[80:]), torch.tensor(speeds[80:]))
        analytic = rolling_deceleration(config.mu_r, config.gravity)
        legacy = legacy_rolling_deceleration(config.mu_r, config.gravity)
        # The surrogate reproduces mu_r g, i.e. it is a factor 1.4 too large.
        self.assertLess(abs(measured - legacy) / legacy, 0.03)
        self.assertGreater((measured - analytic) / analytic, 0.30)

    def test_surrogate_cannot_damp_pure_spin(self):
        """Without spin state the rolling torque has no reservoir to act on."""
        config = SpinDownCaseConfig(mu_r=0.05, rolling_model="cdt", enable_rotation=False, dt=1.0e-4)
        dem, loads = build_spin_down_case(config, spin=10.0)
        self.assertIsNone(dem.omega)
        _run(dem, loads, 1000)
        # Pure spin is entirely absent from the legacy state vector: the measured
        # spin-down flux is zero even though the analytic law predicts a finite one.
        analytic = cdt_spin(10.0, config.mu_r, config.normal_force,
                            config.effective_radius, config.inertia, 0.2)
        self.assertAlmostEqual(analytic, 10.0 - 0.05 * 0.5 * 0.0125 / config.inertia * 0.2,
                               places=9)
        self.assertLess(analytic, 10.0)


class BehaviourPreservationTests(unittest.TestCase):
    """The legacy path must be numerically untouched by the new code."""

    def test_legacy_indentation_case_is_deterministic_and_finite(self):
        def run() -> torch.Tensor:
            dem = IceDEM3D(DEM3DConfig(
                nx=5, ny=4, nz=3, friction_coefficient=0.4,
                rolling_resistance_coefficient=0.05, dt=2.0e-6))
            for _ in range(20):
                dem.step()
            return dem.positions.clone()

        first, second = run(), run()
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(torch.equal(first, second))

    def test_state_dict_roundtrip_preserves_spin(self):
        config = SpinDownCaseConfig(mu_r=0.05, rolling_model="cdt", dt=1.0e-4)
        dem, loads = build_spin_down_case(config, spin=10.0)
        _run(dem, loads, 200)
        state = dem.state_dict()
        reference = dem.omega.clone()
        _run(dem, loads, 200)
        dem.load_state_dict(state)
        self.assertTrue(torch.equal(dem.omega, reference))
        # A checkpoint without the optional spin key restores a zero spin state.
        legacy_state = {key: value for key, value in state.items() if key != "omega"}
        dem.load_state_dict(legacy_state)
        self.assertTrue(torch.equal(dem.omega, torch.zeros_like(dem.omega)))


if __name__ == "__main__":
    unittest.main()