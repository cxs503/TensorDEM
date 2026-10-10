"""Regression tests for 3-D fracture objectivity and release-energy bookkeeping."""
import math
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.validate_dem3d_fracture import (
    check_mixed_mode_failure_and_ledger,
    check_rigid_rotation_objectivity,
    check_single_bond_release_energy,
    run_validation,
)


class DEM3DFractureObjectivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rigid_rotation_preserves_force_covariance_and_does_not_break_bonds(self):
        result = check_rigid_rotation_objectivity()
        self.assertTrue(result["passed"], result)
        self.assertLess(result["relative_force_rotation_error"], 1.0e-8)
        self.assertEqual(result["broken_bonds"], 0)

    def test_single_bond_release_matches_stored_elastic_energy(self):
        result = check_single_bond_release_energy()
        self.assertTrue(result["passed"], result)
        self.assertGreater(result["expected_release_J"], 0.0)
        self.assertAlmostEqual(result["observed_release_J"], result["expected_release_J"], places=14)

    def test_mixed_mode_failure_is_counted_and_energy_ledger_matches(self):
        result = check_mixed_mode_failure_and_ledger()
        self.assertTrue(result["passed"], result)
        self.assertGreater(result["mixed_mode_bonds"], 0)
        self.assertGreaterEqual(result["sum_per_bond_release_J"], 0.0)
        self.assertTrue(math.isfinite(result["ledger_error_J"]))

    def test_validation_report_is_written_and_machine_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_validation(Path(tmp))
            self.assertEqual(report["verdict"], "PASS", report)
            self.assertEqual(report["check_count"], 3)
            self.assertTrue((Path(tmp) / "dem3d_fracture_validation.json").is_file())


if __name__ == "__main__":
    unittest.main()
