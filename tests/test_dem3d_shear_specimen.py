"""Tests for the DEM3D mode-II (direct-shear) specimen protocol."""
import math
import unittest

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.shear_specimen import (
    apparent_shear_modulus_from_history,
    affine_shear_invariant,
    affine_shear_probe,
    bond_shear_invariant,
    critical_engineering_shear_strain,
    direct_shear_specimen_area,
    direct_shear_specimen_height,
    lattice_spacing,
    run_direct_shear,
    shear_failure_onset,
    shear_modulus_closed_form,
    uniaxial_failure_onset,
    uniaxial_shear_criterion_strain,
)


def _affine_config(**overrides):
    values = {
        "nx": 3, "ny": 3, "nz": 3, "drag": 0.0, "fix_x_edges": False,
        "tool_gap": 20.0, "breaking_strain": 1.0e3, "shear_breaking_strain": 0.03,
    }
    values.update(overrides)
    return DEM3DConfig(**values)


def _shear_config(**overrides):
    values = {
        "nx": 5, "ny": 3, "nz": 3, "drag": 10.0, "tool_gap": 20.0, "tool_speed": 1.0,
        "breaking_strain": 1.0e3, "shear_breaking_strain": 0.03,
    }
    values.update(overrides)
    return DEM3DConfig(**values)


class ShearSpecimenClosedFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_closed_form_invariant_matches_leading_order_and_solver(self):
        self.assertAlmostEqual(affine_shear_invariant(0.0), 0.0, places=18)
        # Leading order is gamma/sqrt(3); the quartic term is a sub-percent effect.
        for gamma in (1.0e-3, 1.0e-2, 5.0e-2):
            expected = affine_shear_invariant(gamma)
            self.assertLess(abs(expected - gamma / math.sqrt(3.0)), 0.02 * expected)
        probe = affine_shear_probe(_affine_config())
        self.assertLess(probe["max_relative_error"], 1.0e-12)
        for row in probe["rows"]:
            self.assertLess(row["particle_spread"], 1.0e-14)

    def test_plane_isotropy_and_rotation_invariance(self):
        probe = affine_shear_probe(_affine_config())
        self.assertLess(probe["isotropy_absolute_spread"], 1.0e-12)
        self.assertLess(probe["rotation_invariant"], 1.0e-12)

    def test_shear_modulus_closed_form(self):
        config = _affine_config()
        a = lattice_spacing(config)
        self.assertAlmostEqual(a, 2.0 * config.radius, places=18)
        self.assertAlmostEqual(shear_modulus_closed_form(config),
                               7.0 * config.bond_stiffness / (3.0 * a), places=9)

    def test_threshold_mapping_closed_forms(self):
        for s in (0.003, 0.01, 0.03, 0.05):
            gamma_c = critical_engineering_shear_strain(s)
            # gamma_c is sqrt(3)*s to first order, and slightly below it.
            self.assertLess(gamma_c, math.sqrt(3.0) * s)
            self.assertGreater(gamma_c, 0.99 * math.sqrt(3.0) * s)
            self.assertAlmostEqual(affine_shear_invariant(gamma_c), s, places=12)
            eps = uniaxial_shear_criterion_strain(s)
            # The quadratic term pulls the uniaxial onset slightly below 1.5 s.
            self.assertGreater(eps, 1.4 * s)
            self.assertLess(eps, 1.5 * s)


class ShearSpecimenOnsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_affine_shear_onset_matches_closed_form(self):
        for threshold in (0.01, 0.03):
            result = shear_failure_onset(_affine_config(shear_breaking_strain=threshold))
            self.assertLess(result["relative_error"], 1.0e-9, result)
            self.assertEqual(result["failure_mode_codes"], [2])
            self.assertAlmostEqual(result["local_bond_invariant_at_onset"], threshold,
                                   places=12)

    def test_uniaxial_onset_matches_closed_form(self):
        result = uniaxial_failure_onset(_affine_config())
        self.assertLess(result["relative_error"], 1.0e-9, result)
        self.assertEqual(result["failure_mode_codes"], [2])
        self.assertLess(result["onset_strain"], 1.5 * 0.03)


class ShearSpecimenDirectShearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_geometry_helpers(self):
        config = _shear_config()
        self.assertAlmostEqual(direct_shear_specimen_height(config),
                               (config.nz - 1) * 2.0 * config.radius, places=18)
        self.assertAlmostEqual(direct_shear_specimen_area(config),
                               config.nx * config.ny * (2.0 * config.radius) ** 2, places=18)

    def test_direct_shear_response_is_deterministic_and_monotone(self):
        first = run_direct_shear(_shear_config(), velocity=0.1, gamma_max=0.01)
        second = run_direct_shear(_shear_config(), velocity=0.1, gamma_max=0.01)
        self.assertEqual(first["protocol"]["steps"], second["protocol"]["steps"])
        self.assertEqual([row["tau_Pa"] for row in first["history"]],
                         [row["tau_Pa"] for row in second["history"]])
        taus = [row["tau_Pa"] for row in first["history"]]
        self.assertTrue(all(tau >= 0.0 for tau in taus))
        self.assertGreater(taus[-1], taus[len(taus) // 3])
        self.assertEqual(first["history"][-1]["broken_bonds"], 0)

    def test_direct_shear_rate_convergence_is_monotone(self):
        """The ramp is rate dependent; the deviation must shrink with the rate.

        The quasi-static limit is *not* reached at these rates (documented in
        ``docs/dem3d_shear_specimen.md``), so this test asserts convergence
        ordering rather than a specific tolerance.
        """
        taus = {}
        for velocity in (0.4, 0.2, 0.1):
            result = run_direct_shear(_shear_config(), velocity=velocity,
                                      gamma_max=0.02)
            taus[velocity] = result["history"][-1]["tau_Pa"]
        dev_fast = abs(taus[0.4] - taus[0.2]) / taus[0.2]
        dev_slow = abs(taus[0.2] - taus[0.1]) / taus[0.1]
        self.assertLess(dev_slow, dev_fast)
        self.assertLess(dev_slow, 0.25)

    def test_apparent_modulus_is_positive_and_below_affine_constant(self):
        result = run_direct_shear(_shear_config(), velocity=0.1, gamma_max=0.02)
        fit = apparent_shear_modulus_from_history(result["history"], 0.012, 0.019)
        self.assertGreater(fit["apparent_shear_modulus_Pa"], 0.0)
        # The clamped bottom plus free lateral faces make the driven block softer
        # than the affine constant; the deficit is the documented specimen effect.
        self.assertLess(fit["ratio_to_C44"] if "ratio_to_C44" in fit else
                        fit["apparent_shear_modulus_Pa"] / result["C44_Pa"], 1.0)
        self.assertLess(fit["relative_residual"], 0.1)

    def test_bond_shear_invariant_matches_threshold_at_affine_onset(self):
        config = _affine_config()
        sim = IceDEM3D(config)
        gamma = critical_engineering_shear_strain(config.shear_breaking_strain)
        sim.positions = sim.initial_positions.clone()
        sim.positions[:, 0] += gamma * sim.initial_positions[:, 2]
        invariant = bond_shear_invariant(sim)
        self.assertAlmostEqual(float(invariant.max()), config.shear_breaking_strain,
                              places=12)

    def test_shear_to_failure_stays_on_mode_two(self):
        config = _shear_config(gamma_max=None) if False else _shear_config()
        result = run_direct_shear(config, velocity=0.2, gamma_max=0.06)
        first = result["first_failure"]
        self.assertIsNotNone(first)
        self.assertEqual(first["failure_mode_codes"], [2])
        self.assertLess(abs(first["max_bond_invariant"] / config.shear_breaking_strain - 1.0),
                        0.02)
        self.assertEqual(set(result["failure_mode_counts"].keys()), {2})


if __name__ == "__main__":
    unittest.main()