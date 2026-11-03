"""Tests for the Rayleigh-wave step estimate and the measured dt threshold.

Three independent anchors are used, in the spirit of the repository's other
verification pages:

* ``c_R / c_s`` against the literature values of the Rayleigh secular equation
  (0.874032, 0.919402, 0.955313 at ``nu = 0, 0.25, 0.5``);
* the lattice constants against the closed form ``C11 = 13k/3a``,
  ``C12 = C44 = 7k/3a`` of :func:`tensordem.lattice_elasticity.analytic_box_constants`;
* ``omega_max`` against a power iteration on the same matrix, and the predicted
  ``dt_crit = 2/omega_max`` against a bisection on the *real* engine.
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.rayleigh_timestep import (
    RAYLEIGH_RATIO_LITERATURE,
    cubic_isotropic_moduli,
    divergence_threshold,
    energy_amplification,
    energy_amplification_closed_form,
    exact_bond_omega_max,
    lattice_isotropic_moduli,
    rayleigh_dt_comparison,
    rayleigh_speed_ratio,
    rayleigh_speed_ratio_fit,
    rayleigh_timestep_limit,
    rayleigh_wave_speed,
)
from tensordem.stability import stability_report

RADIUS = 0.025
DENSITY = 917.0
BOND_STIFFNESS = 2000.0
SPACING = 2.0 * RADIUS


def elastic_config(**overrides):
    """Tool-free, damping-free bonded box: a closed linear elastic system."""
    params = dict(
        nx=3, ny=3, nz=3, radius=RADIUS, density=DENSITY,
        bond_stiffness=BOND_STIFFNESS, contact_damping=0.0, drag=0.0,
        tool_radius=1e-6, tool_gap=10.0, tool_speed=0.5, contact_model="linear",
    )
    params.update(overrides)
    return DEM3DConfig(**params)


def elastic_model(**overrides) -> IceDEM3D:
    return IceDEM3D(elastic_config(**overrides))


class RayleighSecularEquationTest(unittest.TestCase):
    def test_matches_literature_ratios(self):
        for nu, reference in RAYLEIGH_RATIO_LITERATURE.items():
            with self.subTest(nu=nu):
                value = rayleigh_speed_ratio(nu)
                self.assertLess(abs(value - reference) / reference, 1e-5)

    def test_ratio_is_monotone_in_poisson(self):
        values = [rayleigh_speed_ratio(nu) for nu in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))

    def test_ratio_below_shear_and_above_the_surface_wave_branch(self):
        for nu in (0.0, 0.25, 0.49):
            with self.subTest(nu=nu):
                ratio = rayleigh_speed_ratio(nu)
                self.assertGreater(ratio, 0.8)
                self.assertLess(ratio, 1.0)

    def test_engineering_fit_reproduces_secular_equation_within_half_a_percent(self):
        worst = 0.0
        for step in range(0, 46):
            nu = 0.45 * step / 45.0
            exact = rayleigh_speed_ratio(nu)
            worst = max(worst, abs(rayleigh_speed_ratio_fit(nu) - exact) / exact)
        self.assertLess(worst, 0.005)
        self.assertGreater(worst, 1e-4)  # it is a fit, not an identity

    def test_rejects_poisson_outside_the_physical_range(self):
        for bad in (-0.1, 0.51, float("nan"), float("inf")):
            with self.subTest(nu=bad):
                with self.assertRaises(ValueError):
                    rayleigh_speed_ratio(bad)


class LatticeModuliTest(unittest.TestCase):
    def test_closed_form_constants(self):
        moduli = lattice_isotropic_moduli(BOND_STIFFNESS, SPACING)
        unit = BOND_STIFFNESS / SPACING
        self.assertAlmostEqual(moduli["C11"], 13.0 / 3.0 * unit, places=9)
        self.assertAlmostEqual(moduli["C12"], 7.0 / 3.0 * unit, places=9)
        self.assertAlmostEqual(moduli["C44"], 7.0 / 3.0 * unit, places=9)

    def test_isotropic_averages_bracket_the_hill_value(self):
        moduli = lattice_isotropic_moduli(BOND_STIFFNESS, SPACING)
        unit = BOND_STIFFNESS / SPACING
        self.assertAlmostEqual(moduli["K_Pa"], 3.0 * unit, places=9)
        self.assertAlmostEqual(moduli["G_voigt_Pa"], 9.0 / 5.0 * unit, places=9)
        self.assertAlmostEqual(moduli["G_reuss_Pa"], 35.0 / 23.0 * unit, places=9)
        self.assertLessEqual(moduli["G_reuss_Pa"], moduli["G_vrh_Pa"] + 1e-9)
        self.assertLessEqual(moduli["G_vrh_Pa"], moduli["G_voigt_Pa"] + 1e-9)
        self.assertAlmostEqual(moduli["poisson_vrh"], 0.2663132, places=6)

    def test_cauchy_relation_and_zener_ratio(self):
        moduli = cubic_isotropic_moduli(13.0 / 3.0, 7.0 / 3.0, 7.0 / 3.0)
        self.assertAlmostEqual(moduli["C44_Pa"] if "C44_Pa" in moduli else 7.0 / 3.0,
                               7.0 / 3.0, places=12)
        self.assertAlmostEqual(moduli["zener_anisotropy"], 7.0 / 3.0, places=12)
        self.assertAlmostEqual(moduli["poisson_c11_c12"], 7.0 / 20.0, places=12)

    def test_wave_speeds_are_ordered(self):
        speeds = rayleigh_wave_speed(bond_stiffness=BOND_STIFFNESS,
                                     mass=DENSITY * 4.0 / 3.0 * math.pi * RADIUS**3,
                                     radius=RADIUS)
        self.assertLess(speeds["c_R_m_per_s"], speeds["c_s_m_per_s"])
        self.assertLess(speeds["c_s_m_per_s"], speeds["c_p_m_per_s"])
        self.assertAlmostEqual(speeds["c_R_m_per_s"] / speeds["c_s_m_per_s"],
                               rayleigh_speed_ratio(speeds["poisson_vrh"]), places=12)

    def test_rayleigh_step_scales_as_inverse_sqrt_stiffness(self):
        mass = DENSITY * 4.0 / 3.0 * math.pi * RADIUS**3
        soft = rayleigh_timestep_limit(bond_stiffness=BOND_STIFFNESS, mass=mass, radius=RADIUS)
        stiff = rayleigh_timestep_limit(bond_stiffness=4.0 * BOND_STIFFNESS, mass=mass, radius=RADIUS)
        ratio = soft["dt_rayleigh_two_radii_s"] / stiff["dt_rayleigh_two_radii_s"]
        self.assertAlmostEqual(ratio, 2.0, places=9)
        self.assertAlmostEqual(stiff["dt_rayleigh_pi_radius_s"] / stiff["dt_rayleigh_two_radii_s"],
                               0.5 * math.pi, places=12)


class ExactSpectrumTest(unittest.TestCase):
    def test_eigh_matches_power_iteration(self):
        model = elastic_model()
        exact = exact_bond_omega_max(model)
        matrix = exact["stiffness_matrix"]
        vector = torch.rand(matrix.shape[0], dtype=torch.float64)
        for _ in range(400):
            vector = matrix @ vector
            vector = vector / vector.norm()
        lam = float(vector @ (matrix @ vector))
        lam_eigh = float(torch.linalg.eigvalsh(matrix)[-1])
        self.assertLess(abs(lam - lam_eigh) / lam_eigh, 1e-10)
        self.assertAlmostEqual(exact["omega_max_rad_s"],
                               math.sqrt(lam_eigh / float(model.config.mass)), places=9)

    def test_gershgorin_bound_contains_the_exact_frequency(self):
        model = elastic_model()
        exact = exact_bond_omega_max(model)
        report = stability_report(model)
        self.assertGreater(report["omega_upper_rad_s"], exact["omega_max_rad_s"])
        looseness = report["omega_upper_rad_s"] / exact["omega_max_rad_s"]
        # Central-force bonds along different directions partly cancel, so the
        # row-sum bound is not tight for the 26-neighbour lattice.
        self.assertGreater(looseness, 1.5)
        self.assertLess(looseness, 3.0)

    def test_engine_guard_still_refuses_steps_above_recommended(self):
        config = elastic_config()
        with self.assertRaises(ValueError):
            IceDEM3D(DEM3DConfig(**{**config.__dict__, "dt": 1.5 * config.recommended_dt}))

    def test_comparison_reports_the_guard_as_the_most_conservative(self):
        comparison = rayleigh_dt_comparison(elastic_model())
        self.assertGreater(comparison["dt_rayleigh_over_dt_crit"], 1.0)
        self.assertLess(comparison["engine_recommended_over_dt_crit"], 0.1)
        self.assertGreater(comparison["dt_rayleigh_over_dt_crit"],
                           comparison["engine_recommended_over_dt_crit"])


class EngineThresholdTest(unittest.TestCase):
    def test_peak_energy_follows_the_closed_form(self):
        closed = energy_amplification_closed_form
        self.assertAlmostEqual(closed(0.5), 2.0, places=12)
        self.assertAlmostEqual(closed(0.9), 10.0, places=12)
        self.assertAlmostEqual(closed(0.25), 4.0 / 3.0, places=12)
        for ratio in (0.1, 0.25, 0.5, 0.8):
            with self.subTest(ratio=ratio):
                measured = energy_amplification(
                    elastic_model, dt_over_crit=ratio, n_steps=200
                )["peak_over_initial"]
                self.assertLess(abs(measured / closed(ratio) - 1.0), 0.01)

    def test_half_the_critical_step_is_bounded_and_stable(self):
        trial = energy_amplification(elastic_model, dt_over_crit=0.5, n_steps=200)
        self.assertFalse(bool(trial["diverged"]))
        self.assertLess(float(trial["peak_over_initial"]), 2.5)

    def test_marginal_step_grows_algebraically_but_is_flagged_stable(self):
        trial = energy_amplification(elastic_model, dt_over_crit=1.0, n_steps=200)
        self.assertFalse(bool(trial["diverged"]))
        # spectrally stable is not enough: the artificial energy is enormous
        self.assertGreater(float(trial["peak_over_initial"]), 100.0)

    def test_step_above_the_limit_diverges(self):
        trial = energy_amplification(elastic_model, dt_over_crit=1.05, n_steps=200)
        self.assertTrue(bool(trial["diverged"]))

    def test_measured_threshold_matches_two_over_omega_max(self):
        measurement = divergence_threshold(elastic_model, n_steps=200, iterations=6)
        self.assertLess(float(measurement["abs_relative_error"]), 0.01)
        self.assertAlmostEqual(float(measurement["measured_threshold_ratio"]), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()