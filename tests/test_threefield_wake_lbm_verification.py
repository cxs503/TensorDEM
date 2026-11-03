"""Analytic verification of the D3Q19 + Smagorinsky propeller-race kernel."""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.threefield.wake_lbm_verification import (
    HARNESS_SCHEMA,
    PeriodicShearWave,
    analytic_smagorinsky_tau,
    conservation_check,
    equilibrium_moment_check,
    galilean_invariance_check,
    log_linear_decay_rate,
    measure_shear_decay,
    sgs_damped_decay_check,
    smagorinsky_constant_consistency,
    smagorinsky_tau_check,
    streaming_reference_check,
    zero_thrust_uniform_check,
)


def _single_thread() -> None:
    torch.set_num_threads(1)


class EquilibriumMomentTests(unittest.TestCase):
    def test_equilibrium_reproduces_density_velocity_and_stress(self):
        _single_thread()
        result = equilibrium_moment_check()
        self.assertLess(result["worst_relative_error"], 1.0e-13)
        self.assertLess(result["weight_sum_minus_one"], 1.0e-15)
        self.assertLess(result["cxy_moment_error"], 1.0e-15)


class StreamingStencilTests(unittest.TestCase):
    def test_stream_matches_independent_scatter_reference(self):
        _single_thread()
        result = streaming_reference_check()
        self.assertEqual(result["worst_relative_error"], 0.0)


class BoundedStepTests(unittest.TestCase):
    def test_uniform_flow_is_a_fixed_point_of_the_bounded_step(self):
        _single_thread()
        result = zero_thrust_uniform_check(steps=20)
        self.assertLess(result["worst_relative_error"], 1.0e-12)

    def test_periodic_mass_and_momentum_are_conserved(self):
        _single_thread()
        result = conservation_check(steps=50)
        self.assertLess(result["worst_relative_error"], 1.0e-13)


class ViscosityTests(unittest.TestCase):
    def test_shear_wave_decay_reproduces_nu_k_squared(self):
        _single_thread()
        result = measure_shear_decay(nx=32, tau0=0.7)
        self.assertLess(result["relative_error"], 3.0e-3)
        self.assertAlmostEqual(
            result["viscosity_measured"], result["viscosity_predicted"], delta=2.0e-4
        )

    def test_complex_mode_amplitude_tracks_advected_wave(self):
        _single_thread()
        fixture = PeriodicShearWave(nx=32, tau0=0.7, amplitude=1.0e-3, bulk_velocity=0.04)
        fixture.step(80)
        # A pure sine rotated by a phase drift still has the initial amplitude
        # up to the viscous decay; a sine-only projection would report ~cos(phase).
        decay = math.exp(-fixture.viscosity * fixture.wavenumber**2 * fixture.steps)
        self.assertAlmostEqual(fixture.mode_amplitude() / fixture.amplitude0, decay, delta=5.0e-3)

    def test_galilean_invariance_of_the_decay_rate(self):
        _single_thread()
        result = galilean_invariance_check(nx=32, tau0=0.7)
        self.assertLess(result["relative_change"], 1.0e-2)


class SmagorinskyTests(unittest.TestCase):
    def test_tau_eff_matches_the_closed_form_cell_by_cell(self):
        _single_thread()
        result = smagorinsky_tau_check()
        self.assertLess(result["worst_relative_error"], 1.0e-12)
        self.assertAlmostEqual(
            result["max_eddy_viscosity"], result["analytic_eddy_viscosity_at_peak"], places=12
        )

    def test_analytic_tau_reduces_to_tau0_without_strain(self):
        self.assertEqual(analytic_smagorinsky_tau(0.5006, 0.16, 0.0), 0.5006)
        with self.assertRaises(ValueError):
            analytic_smagorinsky_tau(0.5, 0.16, 1.0)

    def test_eddy_viscosity_convention_is_documented_and_enforced(self):
        """The Voigt single-count stress norm halves nu_t relative to ``Cs^2 |S|``."""
        result = smagorinsky_constant_consistency(0.5006, 0.16, 1.0e-3)
        self.assertAlmostEqual(result["eddy_viscosity_ratio"], 1.0 / math.sqrt(2.0), places=3)
        self.assertAlmostEqual(result["effective_cs_ratio"], 2.0 ** -0.25, places=3)
        self.assertAlmostEqual(result["effective_cs"] / 0.16, 2.0 ** -0.25, places=3)

    def test_sgs_damped_decay_matches_the_closed_form_ode(self):
        _single_thread()
        result = sgs_damped_decay_check(steps=120)
        self.assertLess(result["worst_relative_error"], 0.02)
        self.assertGreater(result["rate_enhancement"], 1.0)


class DecayRateHelperTests(unittest.TestCase):
    def test_log_linear_fit_recovers_a_prescribed_rate(self):
        rate = 2.0e-3
        trace = [(t, 1.0e-3 * math.exp(-rate * t)) for t in range(0, 1000, 10)]
        self.assertAlmostEqual(log_linear_decay_rate(trace) / rate, 1.0, places=8)

    def test_schema_is_versioned(self):
        self.assertTrue(HARNESS_SCHEMA.startswith("tensordem.wake-lbm-verification/"))


if __name__ == "__main__":
    unittest.main()