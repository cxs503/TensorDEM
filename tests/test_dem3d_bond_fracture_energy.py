"""Tests for the closed-form bond fracture-energy law and its audit of the solver."""
import math
import unittest

import torch

from tensordem.bond_fracture_energy import (
    AXIS_PLANE_FACTOR,
    analytic_fracture_energy_J_per_m2,
    audit_config,
    bond_failure_energy_J,
    crossing_offset_sum,
    fracture_energy_from_config,
    ice_fracture_energy_J_per_m2,
    required_bond_stiffness,
    required_breaking_strain,
)
from tensordem.dem3d import DEM3DConfig
from scripts.benchmark_dem3d_bond_fracture_energy import (
    controlled_tensile_test,
    physical_band,
    resolution_series,
    size_series,
)


class BondFractureEnergyLawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_crossing_offset_sum_is_axis_independent(self):
        for axis in (0, 1, 2):
            self.assertEqual(crossing_offset_sum(axis), 21.0)
        self.assertEqual(AXIS_PLANE_FACTOR, 10.5)

    def test_closed_form_matches_default_parameter_set(self):
        # 10.5 * 2000 N/m * 0.015^2 = 4.725 J/m^2
        self.assertAlmostEqual(analytic_fracture_energy_J_per_m2(2000.0, 0.015), 4.725, places=12)

    def test_single_bond_energy_scales_with_rest_length_squared(self):
        short = bond_failure_energy_J(2000.0, 0.01, 0.05)
        long = bond_failure_energy_J(2000.0, 0.01, 0.05 * math.sqrt(2.0))
        self.assertAlmostEqual(long / short, 2.0, places=12)

    def test_calibration_targets_round_trip(self):
        target = 1.5
        strain = required_breaking_strain(2000.0, target)
        self.assertAlmostEqual(analytic_fracture_energy_J_per_m2(2000.0, strain), target, places=12)
        stiffness = required_bond_stiffness(strain, target)
        self.assertAlmostEqual(analytic_fracture_energy_J_per_m2(stiffness, strain), target, places=12)

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            crossing_offset_sum(3)
        with self.assertRaises(ValueError):
            bond_failure_energy_J(2000.0, 0.0, 0.05)
        with self.assertRaises(ValueError):
            required_breaking_strain(2000.0, -1.0)
        with self.assertRaises(ValueError):
            ice_fracture_energy_J_per_m2(0.1, 0.0)

    def test_ice_fracture_energy_reference_value(self):
        # K = 0.1 MPa*sqrt(m), E = 7.5 GPa, nu = 0.3 -> 1.2133 J/m^2
        value = ice_fracture_energy_J_per_m2(0.1, 7.5)
        self.assertAlmostEqual(value, 1e10 * (1.0 - 0.09) / 7.5e9, places=10)
        low, high = physical_band()
        self.assertLess(low, high)
        self.assertAlmostEqual(low, ice_fracture_energy_J_per_m2(0.05, 10.0), places=12)

    def test_finite_specimen_converges_to_closed_form(self):
        rows = size_series((5, 9, 13))
        errors = [row["footprint_relative_error"] for row in rows]
        self.assertEqual(errors, sorted(errors, reverse=True))
        self.assertGreater(errors[-1], 0.0)
        for row in rows:
            self.assertAlmostEqual(
                row["analytic_J_per_m2"], analytic_fracture_energy_J_per_m2(2000.0, 0.015), places=12
            )

    def test_crack_plane_orientation_and_resolution_do_not_change_fracture_energy(self):
        values = []
        for axis in (0, 1, 2):
            measured = fracture_energy_from_config(DEM3DConfig(nx=7, ny=7, nz=7), axis=axis)
            values.append(measured["measured_fracture_energy_J_per_m2"])
        self.assertAlmostEqual(max(values) - min(values), 0.0, places=12)

        radii = [row["measured_J_per_m2"] for row in resolution_series((0.01, 0.02, 0.05), size=7)]
        self.assertAlmostEqual(max(radii) - min(radii), 0.0, places=12)

    def test_config_audit_reports_physical_comparison(self):
        audit = audit_config(DEM3DConfig())
        payload = audit.as_dict()
        self.assertAlmostEqual(payload["analytic_fracture_energy_J_per_m2"], 4.725, places=12)
        self.assertGreater(payload["ratio_to_band_midpoint"], 0.0)
        self.assertAlmostEqual(
            payload["required_breaking_strain_for_band_midpoint"],
            required_breaking_strain(2000.0, 0.5 * sum(payload["physical_band_J_per_m2"])),
            places=12,
        )


class ControlledTensileAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_release_ledger_matches_recorded_energy_and_analytic_failure_point(self):
        result = controlled_tensile_test(size=3, breaking_strain=0.01, increment=5e-4)
        self.assertGreater(result["broken_bond_count"], 0)
        self.assertLess(result["broken_bond_count"], result["bond_count"])
        self.assertLess(result["ledger_relative_discrepancy"], 1e-12)
        self.assertLess(result["max_relative_energy_recompute_error"], 1e-12)
        self.assertTrue(result["overshoot_within_one_increment"])
        self.assertGreaterEqual(result["min_detected_strain"], 0.01 - 1e-12)
        self.assertTrue(result["broken_bonds_are_stretched_ones"])
        # The recorded estimate is the elastic energy of the spring at detection,
        # so it can only differ from the analytic failure energy by the strain
        # overshoot of one prescribed increment, i.e. ~2 * increment / strain.
        self.assertLess(result["max_relative_energy_error_vs_analytic_failure"], 0.03)

    def test_no_bond_breaks_below_the_breaking_strain(self):
        strained = controlled_tensile_test(size=3, breaking_strain=0.02, increment=1e-3,
                                           final_strain=0.015)
        self.assertEqual(strained["broken_bond_count"], 0)
        self.assertEqual(strained["recorded_release_energy_J"], 0.0)
        at_threshold = controlled_tensile_test(size=3, breaking_strain=0.02, increment=1e-3,
                                               final_strain=0.025)
        self.assertGreater(at_threshold["broken_bond_count"], 0)


if __name__ == "__main__":
    unittest.main()