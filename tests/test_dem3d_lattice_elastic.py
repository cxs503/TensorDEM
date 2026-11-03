"""Tests for the 3-D lattice elastic-constant audit module.

These checks are *verification* checks: they compare closed-form lattice sums
with numbers measured from the solver's own bond network, and they pin the
non-affine (relaxed) response so a future change to the bond topology, force law
or stiffness convention cannot pass silently.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.dem3d import DEM3DConfig, IceDEM3D  # noqa: E402
from tensordem import dem3d_lattice_elastic as lattice  # noqa: E402

RADIUS = 0.025
STIFFNESS = 2000.0


def _config(size: int, radius: float = RADIUS, stiffness: float = STIFFNESS) -> DEM3DConfig:
    return DEM3DConfig(nx=size, ny=size, nz=size, radius=radius, bond_stiffness=stiffness)


class OffsetSetTests(unittest.TestCase):
    def test_half_space_offsets_are_unique_and_complete(self) -> None:
        offsets = lattice.bond_offset_set()
        self.assertEqual(len(offsets), 13)
        self.assertEqual(len(set(offsets)), 13)
        for offset in offsets:
            negative = tuple(-value for value in offset)
            self.assertNotIn(negative, offsets)
            self.assertIn(negative, {tuple(-v for v in other) for other in offsets})

    def test_interior_coordination_matches_solver(self) -> None:
        solver = IceDEM3D(_config(5))
        core = lattice._core_mask(solver.initial_positions, lattice.spacing(RADIUS))
        index = int(torch.nonzero(core, as_tuple=False)[len(torch.nonzero(core, as_tuple=False)) // 2])
        incident = (solver.pairs[:, 0] == index) | (solver.pairs[:, 1] == index)
        self.assertEqual(int(incident.sum().item()), 26)
        other = torch.where(solver.pairs[incident, 0] == index,
                            solver.pairs[incident, 1], solver.pairs[incident, 0])
        direction = ((solver.initial_positions[other] - solver.initial_positions[index])
                     / lattice.spacing(RADIUS)).round()
        measured = {tuple(int(v) for v in row) for row in direction.tolist()}
        expected = set()
        for offset in lattice.bond_offset_set():
            expected.add(offset)
            expected.add(tuple(-value for value in offset))
        self.assertEqual(measured, expected)


class AnalyticConstantTests(unittest.TestCase):
    def test_closed_form_coefficients(self) -> None:
        result = lattice.analytic_constants(STIFFNESS, RADIUS)
        spacing = lattice.spacing(RADIUS)
        for key, expected in (("C11_Pa", 13.0 * STIFFNESS / (3.0 * spacing)),
                              ("C12_Pa", 7.0 * STIFFNESS / (3.0 * spacing)),
                              ("C44_Pa", 7.0 * STIFFNESS / (3.0 * spacing)),
                              ("young_100_Pa", 2.7 * STIFFNESS / spacing)):
            self.assertAlmostEqual(result[key] / expected, 1.0, places=12)
        self.assertAlmostEqual(result["poisson_100"], 0.35, places=12)
        self.assertEqual(result["coordination_number"], 26)

    def test_cubic_symmetry_and_cauchy_relation(self) -> None:
        result = lattice.analytic_constants(STIFFNESS, RADIUS)
        for key, deviation in result["cubic_symmetry_relative_deviation"].items():
            self.assertLess(deviation, 1e-12, msg=key)
        self.assertLess(result["cauchy_relation_relative_deviation"], 1e-12)
        self.assertAlmostEqual(result["zener_anisotropy_C44_over_half_C11_minus_C12"], 7.0 / 3.0, places=12)

    def test_dimensionless_modulus_is_scale_invariant(self) -> None:
        base = lattice.analytic_constants(STIFFNESS, RADIUS)
        for radius, stiffness in ((0.01, 2000.0), (0.05, 2000.0), (0.025, 5e5)):
            scaled = lattice.analytic_constants(stiffness, radius)
            self.assertAlmostEqual(scaled["E_over_k_over_a"], base["E_over_k_over_a"], places=12)
            self.assertAlmostEqual(scaled["poisson_100"], base["poisson_100"], places=12)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            lattice.analytic_constants(0.0, RADIUS)
        with self.assertRaises(ValueError):
            lattice.analytic_constants(STIFFNESS, 0.0)


class AffinePatchTests(unittest.TestCase):
    def test_affine_constants_reproduce_lattice_sum(self) -> None:
        patch = lattice.affine_patch(_config(6))
        self.assertLess(patch["max_relative_error"], 1e-9)
        self.assertGreater(patch["core_particles"], 0)
        for key in ("C11", "C12", "C44"):
            self.assertLess(patch["relative_error"][key], 1e-9, msg=key)

    def test_affine_response_is_linear_in_strain(self) -> None:
        small = lattice.affine_patch(_config(5), strain=1e-6)
        larger = lattice.affine_patch(_config(5), strain=1e-5)
        self.assertAlmostEqual(larger["measured"]["C11_Pa"] / small["measured"]["C11_Pa"], 1.0, places=5)

    def test_solver_energy_matches_manual_bond_energy(self) -> None:
        solver = IceDEM3D(_config(5))
        displacement = lattice._apply_affine(solver, "xx", 1e-6)
        diagnostics = solver.diagnostics()
        extension = ((displacement[solver.pairs[:, 1]] - displacement[solver.pairs[:, 0]])
                     * solver.initial_normals).sum(dim=1)
        manual = 0.5 * STIFFNESS * extension.square().sum()
        self.assertAlmostEqual(float(diagnostics["bond_elastic_energy_J"]) / float(manual), 1.0, places=5)


class HessianConsistencyTests(unittest.TestCase):
    def test_hessian_matches_solver_force_law(self) -> None:
        check = lattice.hessian_cross_check(_config(5))
        self.assertTrue(check["pass"], msg=str(check))
        self.assertLess(check["energy_relative_difference"], 1e-4)
        self.assertLess(check["force_relative_difference"], 1e-3)

    def test_hessian_check_is_deterministic(self) -> None:
        first = lattice.hessian_cross_check(_config(4), seed=7)
        second = lattice.hessian_cross_check(_config(4), seed=7)
        self.assertEqual(first["quadratic_form_energy_J"], second["quadratic_form_energy_J"])


class RelaxedPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.axial = lattice.relaxed_patch(_config(4), mode="xx", boundary="x_faces")
        cls.shear = lattice.relaxed_patch(_config(4), mode="shear", boundary="yz_faces")
        cls.analytic = lattice.analytic_constants(STIFFNESS, RADIUS)

    def test_free_dof_equilibrium(self) -> None:
        self.assertLess(self.axial["free_dof_residual_N"], 1e-12)
        self.assertLess(self.shear["free_dof_residual_N"], 1e-12)
        self.assertGreater(self.axial["free_dof_count"], 0)

    def test_relaxed_moduli_obey_variational_bound(self) -> None:
        # Freeing internal degrees of freedom at fixed prescribed strain can only
        # lower the strain energy, so the relaxed constant must not exceed the
        # affine constant evaluated on the same specimen with the same estimator.
        self.assertLessEqual(self.axial["ratio_to_affine_same_specimen"], 1.0 + 1e-12)
        self.assertLessEqual(self.shear["ratio_to_affine_same_specimen"], 1.0 + 1e-12)
        # ... and the affine same-specimen value must itself reproduce the
        # laterally constrained analytic value (C11 for tension).
        self.assertAlmostEqual(self.axial["affine_same_specimen_Pa"]
                               / self.analytic["C11_Pa"], 1.0, places=9)

    def test_tension_relaxation_approaches_traction_free_limit(self) -> None:
        # A tension test with free lateral faces relaxes towards E100, so the
        # small specimen sits between E100 and C11, above the trait-free limit.
        ratio = self.axial["relaxed_constant_Pa"] / self.analytic["young_100_Pa"]
        self.assertGreater(ratio, 1.0)
        self.assertLess(ratio, self.analytic["C11_Pa"] / self.analytic["young_100_Pa"])

    def test_shear_softening_is_material(self) -> None:
        ratio = self.shear["relaxed_constant_Pa"] / self.analytic["C44_Pa"]
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 0.9, msg="central-force lattice should soften strongly in shear")

    def test_relaxed_poisson_ratio_is_positive(self) -> None:
        self.assertGreater(self.axial["poisson_ratio_relaxed"], 0.0)
        self.assertLess(self.axial["poisson_ratio_relaxed"], self.analytic["poisson_100"])

    def test_relaxation_rejects_oversized_specimens(self) -> None:
        with self.assertRaises(ValueError):
            lattice.relaxed_patch(DEM3DConfig(nx=16, ny=16, nz=16), mode="xx")


class CalibrationTests(unittest.TestCase):
    def test_stiffness_calibration_recovers_target_modulus(self) -> None:
        calibration = lattice.stiffness_calibration(9.0e9, RADIUS, target_poisson=0.3)
        self.assertLess(calibration["young_relative_error"], 1e-12)
        self.assertAlmostEqual(calibration["required_bond_stiffness_N_per_m"],
                               9.0e9 * lattice.spacing(RADIUS) / 2.7, places=3)
        self.assertAlmostEqual(calibration["achieved_poisson_100"], 0.35, places=12)
        self.assertTrue(calibration["poisson_is_stiffness_independent"])
        # A scalar stiffness cannot match an isotropic shear modulus.
        self.assertGreater(calibration["shear_relative_error"], 1.0)
        self.assertAlmostEqual(calibration["zener_anisotropy"], 7.0 / 3.0, places=12)


if __name__ == "__main__":
    unittest.main()