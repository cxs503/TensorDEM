"""Tests for the DEM3D bonded-lattice affine (Cauchy-Born) elasticity module.

The reference values are analytic:

* the closed-form cubic constants of the 13-neighbour central-force lattice,
  ``C11 = 13/3 k/a`` and ``C12 = C44 = 7/3 k/a`` (see
  ``tensordem.lattice_elasticity.analytic_box_constants``);
* the Cauchy relation ``C12 = C44`` for a central-force network without
  pre-stress;
* the shear mechanism ``C44 = 0`` of a nearest-neighbour-only simple-cubic
  lattice.

Every test therefore has an independent reference, not a self-comparison.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.benchmark_dem3d_lattice_elasticity import (
    PROTOCOL,
    SPECIMENS,
    run_benchmark,
)
from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.lattice_elasticity import (
    VOIGT_PAIRS,
    affine_stiffness_from_bonds,
    affine_stiffness_from_solver,
    affine_stiffness_tensor,
    analytic_box_constants,
    compliance_tensor,
    cubic_constants,
    directional_young_modulus,
    finite_lattice_bonds,
    ideal_box_bonds,
    isotropic_moduli,
    max_phonon_frequency,
    solver_virial_stress,
    voigt_reuss_hill,
    voigt_strain_basis,
    zener_ratio,
)


def _sim(nx: int = 6, ny: int = 5, nz: int = 4, **overrides) -> IceDEM3D:
    values = {
        "nx": nx, "ny": ny, "nz": nz,
        "drag": 0.0, "contact_damping": 0.0, "fix_x_edges": False,
        "tool_radius": 0.01, "tool_gap": 0.0, "shear_breaking_strain": 0.5,
    }
    values.update(overrides)
    return IceDEM3D(DEM3DConfig(**values))


class AnalyticConstantsTests(unittest.TestCase):
    k = 2000.0
    a = 0.05

    def setUp(self):
        self.normals, self.lengths = ideal_box_bonds(self.a)

    def test_bond_classes_match_the_solver_neighbourhood(self):
        cfg = DEM3DConfig(nx=4, ny=4, nz=4)
        sim = IceDEM3D(cfg)
        bonds, _ = finite_lattice_bonds(4, 4, 4, 2.0 * cfg.radius)
        self.assertEqual(len(bonds), len(sim.pairs))
        # a fully periodic cell would have 13 classes; the finite box has fewer
        # bonds per particle because of the surfaces
        self.assertEqual(len(self.normals), 13)
        fractions = sorted(
            round(float(length) / self.a, 6) for length in self.lengths)
        self.assertEqual(fractions, [1.0] * 3 + [round(math.sqrt(2), 6)] * 6
                         + [round(math.sqrt(3), 6)] * 4)

    def test_voigt_sum_reproduces_the_closed_form(self):
        stiffness = affine_stiffness_tensor(
            self.normals, self.lengths, self.k, self.a ** 3)
        constants = cubic_constants(stiffness)
        analytic = analytic_box_constants(self.k, self.a)
        unit = self.k / self.a
        self.assertAlmostEqual(analytic["C11"] / unit, 13.0 / 3.0, places=12)
        self.assertAlmostEqual(analytic["C12"] / unit, 7.0 / 3.0, places=12)
        self.assertAlmostEqual(analytic["C44"] / unit, 7.0 / 3.0, places=12)
        for name in ("C11", "C12", "C44"):
            self.assertLess(abs(constants[name] - analytic[name]) / analytic[name], 1e-12)

    def test_cubic_symmetry_and_matrix_symmetry(self):
        stiffness = affine_stiffness_tensor(
            self.normals, self.lengths, self.k, self.a ** 3)
        constants = cubic_constants(stiffness)
        self.assertLess(constants["off_pattern_residual"], 1e-14)
        self.assertLess(constants["symmetry_residual"], 1e-14)
        # the three normal diagonals and the three shear diagonals are equal
        self.assertAlmostEqual(float(stiffness[0, 0]), float(stiffness[2, 2]), places=6)
        self.assertAlmostEqual(float(stiffness[3, 3]), float(stiffness[5, 5]), places=6)

    def test_cauchy_relation_holds_exactly(self):
        stiffness = affine_stiffness_tensor(
            self.normals, self.lengths, self.k, self.a ** 3)
        constants = cubic_constants(stiffness)
        self.assertLess(abs(constants["C12"] - constants["C44"]) / constants["C44"], 1e-14)

    def test_axis_only_lattice_has_zero_shear_stiffness(self):
        axis = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float64)
        lengths = torch.full((3,), self.a, dtype=torch.float64)
        constants = cubic_constants(
            affine_stiffness_tensor(axis, lengths, self.k, self.a ** 3))
        unit = self.k / self.a
        self.assertAlmostEqual(constants["C11"] / unit, 1.0, places=12)
        self.assertAlmostEqual(constants["C12"] / unit, 0.0, places=12)
        self.assertLess(abs(constants["C44"] / unit), 1e-14)

    def test_positive_definite_and_moduli(self):
        stiffness = affine_stiffness_tensor(
            self.normals, self.lengths, self.k, self.a ** 3)
        eigenvalues = torch.linalg.eigvalsh(stiffness)
        self.assertGreater(float(eigenvalues.min()), 0.0)
        constants = cubic_constants(stiffness)
        unit = self.k / self.a
        # closed form for the two Young moduli along [100] and [111]
        self.assertAlmostEqual(
            directional_young_modulus(stiffness, (1, 0, 0)) / unit, 2.7, places=10)
        self.assertAlmostEqual(
            directional_young_modulus(stiffness, (1, 1, 1)) / unit,
            5.558823529411765, places=9)
        # anisotropic network: body diagonal is much stiffer than the cube axis
        self.assertGreater(
            directional_young_modulus(stiffness, (1, 1, 1)),
            directional_young_modulus(stiffness, (1, 0, 0)))
        bounds = voigt_reuss_hill(constants["C11"], constants["C12"], constants["C44"])
        self.assertLessEqual(bounds["shear_reuss"], bounds["shear_hill"])
        self.assertLessEqual(bounds["shear_hill"], bounds["shear_voigt"])
        isotropic = isotropic_moduli(constants["C11"], constants["C12"], constants["C44"])
        self.assertAlmostEqual(isotropic["nu_100"], 0.35, places=10)
        self.assertAlmostEqual(
            zener_ratio(constants["C11"], constants["C12"], constants["C44"]),
            7.0 / 3.0, places=10)

    def test_compliance_tensor_matches_the_cubic_formula(self):
        stiffness = affine_stiffness_tensor(
            self.normals, self.lengths, self.k, self.a ** 3)
        compliance = torch.linalg.inv(stiffness)
        tensor = compliance_tensor(compliance)
        # S1111 = (C11 + C12) / ((C11 - C12)(C11 + 2 C12))
        constants = cubic_constants(stiffness)
        expected = (constants["C11"] + constants["C12"]) / (
            (constants["C11"] - constants["C12"]) * (constants["C11"] + 2 * constants["C12"]))
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), expected, places=12)
        # S1212 = 1 / (4 C44)
        self.assertAlmostEqual(
            float(tensor[0, 1, 0, 1]), 1.0 / (4.0 * constants["C44"]), places=12)
        # symmetry of the expanded tensor
        for i, j, k_, l in ((0, 1, 1, 2), (0, 0, 1, 1), (1, 2, 0, 2), (0, 1, 2, 2)):
            self.assertAlmostEqual(float(tensor[i, j, k_, l]), float(tensor[k_, l, i, j]),
                                   places=14)

    def test_voigt_strain_basis_is_symmetric_with_unit_engineering_shear(self):
        basis = voigt_strain_basis()
        self.assertEqual(tuple(basis.shape), (6, 3, 3))
        for index in range(3):
            self.assertAlmostEqual(float(basis[index].sum()), 1.0, places=15)
        for index in range(3, 6):
            self.assertAlmostEqual(float(basis[index].sum()), 1.0, places=15)  # 2 * 0.5

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            affine_stiffness_tensor(self.normals, self.lengths, self.k, 0.0)
        with self.assertRaises(ValueError):
            affine_stiffness_tensor(self.normals, self.lengths[:-1], self.k, self.a ** 3)
        with self.assertRaises(ValueError):
            analytic_box_constants(0.0, self.a)
        with self.assertRaises(ValueError):
            ideal_box_bonds(0.0)
        with self.assertRaises(ValueError):
            directional_young_modulus(
                affine_stiffness_tensor(self.normals, self.lengths, self.k, self.a ** 3),
                (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            affine_stiffness_from_bonds(torch.zeros((2, 3), dtype=torch.float64), 1.0, 1.0)


class SolverCrossCheckTests(unittest.TestCase):
    """The affine tensor must describe the bond network the solver really builds."""

    def test_affine_sum_matches_the_solver_force_virial(self):
        sim = _sim()
        stiffness = affine_stiffness_from_solver(sim)
        scale = 1e-6
        virial = torch.zeros((6, 6), dtype=torch.float64)
        for index, (i, j) in enumerate(VOIGT_PAIRS):
            strain = torch.zeros((3, 3), dtype=torch.float64)
            if i == j:
                strain[i, i] = scale
            else:
                strain[i, j] = strain[j, i] = 0.5 * scale
            virial[:, index] = solver_virial_stress(sim, strain) / scale
        relative = float((virial - stiffness).abs().max() / stiffness.abs().max())
        # residual is the O(strain) geometric term of the finite affine strain
        self.assertLess(relative, 1e-5)
        self.assertLess(float((virial - virial.T).abs().max() / virial.abs().max()), 1e-6)

    def test_virial_is_restored_after_the_measurement(self):
        sim = _sim()
        positions = sim.positions.clone()
        velocities = sim.velocities.clone()
        alive = sim.alive.clone()
        tool_start = sim.tool_start.clone()
        strain = torch.zeros((3, 3), dtype=torch.float64)
        strain[0, 0] = 1e-6
        solver_virial_stress(sim, strain)
        self.assertTrue(torch.equal(sim.positions, positions))
        self.assertTrue(torch.equal(sim.velocities, velocities))
        self.assertTrue(torch.equal(sim.alive, alive))
        self.assertTrue(torch.equal(sim.tool_start, tool_start))

    def test_touching_indenter_does_not_contaminate_the_virial(self):
        """Regression: an indenter resting on the specimen must be moved away."""
        sim = _sim(nx=9, ny=5, nz=3)
        distances = torch.linalg.vector_norm(sim.positions - sim.tool_position, dim=1)
        self.assertLessEqual(
            float(distances.min()), sim.config.radius + sim.config.tool_radius + 1e-12)
        stiffness = affine_stiffness_from_solver(sim)
        strain = torch.zeros((3, 3), dtype=torch.float64)
        strain[2, 2] = 1e-6
        stress = solver_virial_stress(sim, strain) / 1e-6
        self.assertLess(float((stress - stiffness[:, 2]).abs().max())
                        / float(stiffness.abs().max()), 1e-5)

    def test_finite_specimen_is_softer_than_the_infinite_lattice(self):
        sim = _sim(nx=6, ny=5, nz=4)
        finite = cubic_constants(affine_stiffness_from_solver(sim))
        analytic = analytic_box_constants(sim.config.bond_stiffness, 2.0 * sim.config.radius)
        self.assertLess(finite["C11"], analytic["C11"])
        self.assertLess(finite["C33"] if "C33" in finite else finite["C11"], analytic["C11"])
        # surfaces remove at most ~60% of the axial stiffness for this small prism
        self.assertLess(finite["C11"] / analytic["C11"], 0.9)
        self.assertGreater(finite["C11"] / analytic["C11"], 0.3)

    def test_platens_are_rejected(self):
        sim = _sim(top_platen_enabled=True)
        with self.assertRaises(ValueError):
            solver_virial_stress(sim, torch.zeros((3, 3), dtype=torch.float64))


class PhononBoundTests(unittest.TestCase):
    def test_omega_max_and_critical_step(self):
        cfg = DEM3DConfig()
        normals, lengths = ideal_box_bonds(2.0 * cfg.radius)
        omega = max_phonon_frequency(normals, lengths, cfg.bond_stiffness, cfg.mass,
                                     9, 2.0 * cfg.radius)
        self.assertGreater(omega, 0.0)
        dt_crit = 2.0 / omega
        # the conservative recommended step must sit inside the elastic bound
        self.assertLess(cfg.recommended_dt, dt_crit)
        # and the bound must shrink when the stiffer contact spring is used
        omega_contact = max_phonon_frequency(normals, lengths, cfg.contact_stiffness,
                                            cfg.mass, 9, 2.0 * cfg.radius)
        self.assertGreater(omega_contact, omega)

    def test_invalid_phonon_inputs(self):
        normals, lengths = ideal_box_bonds(0.05)
        with self.assertRaises(ValueError):
            max_phonon_frequency(normals, lengths, 2000.0, 0.0)
        with self.assertRaises(ValueError):
            max_phonon_frequency(normals, lengths, 2000.0, 0.06, 2)


class BenchmarkRunnerTests(unittest.TestCase):
    def test_benchmark_passes_and_writes_audit_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "lattice"
            doc = Path(tmp) / "lattice.md"
            report = run_benchmark(output, doc_path=doc)
            self.assertEqual(report["protocol"], PROTOCOL)
            self.assertEqual(report["verdict"], "PASS", report["checks"])
            self.assertEqual(len(report["specimens"]), len(SPECIMENS))
            for name in ("lattice_elasticity_report.json", "lattice_checks.csv",
                         "surface_deficiency.csv", "directional_modulus.csv",
                         "phonon_dispersion.csv"):
                self.assertTrue((output / name).is_file(), name)
            payload = json.loads((output / "lattice_elasticity_report.json").read_text())
            self.assertEqual(payload["verdict"], "PASS")
            self.assertTrue(doc.is_file())
            text = doc.read_text()
            self.assertIn(PROTOCOL, text)
            self.assertIn("13/3", text)

    def test_documented_numbers_come_from_the_report(self):
        """The markdown must be rendered from the report, not hand-written."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "lattice"
            doc = Path(tmp) / "lattice.md"
            report = run_benchmark(output, doc_path=doc)
            text = doc.read_text()
            for specimen in report["specimens"]:
                grid = "x".join(str(v) for v in specimen["grid"])
                self.assertIn(grid, text)
                self.assertIn(f"{specimen['diagonal_over_infinite']['C11']:.4f}", text)
            self.assertIn(f"{report['phonon']['omega_max_rad_per_s']:.6g}", text)


if __name__ == "__main__":
    unittest.main()