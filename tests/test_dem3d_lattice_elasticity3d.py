"""Verification of the DEM3D lattice elastic constants against closed-form values.

Reference values are hand-derived from the solver's own bond topology (six face
neighbours plus twelve edge diagonals, thirteen bonds owned per node) using
``C_ijkl = (1/a^3) * sum k*l^2*n_i*n_j*n_k*n_l``:

* ``C11 = 13k/(3a)`` -- face bond (1), four in-plane diagonals of each of the three
  planes (3 * 4 * 2ka^2 / 4 = 6ka^2 total) and the body diagonals (8/3 ka^2 ... see
  the audit document for the term-by-term table),
* ``C12 = C44 = 7k/(3a)`` -- the Cauchy relation that a pure central-force network
  must satisfy exactly,
* ``E = 2.7 k/a`` and ``nu = 0.35`` for uniaxial stress.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.lattice_elasticity3d import (
    HALF_SPACE_OFFSETS,
    bond_stiffness_for_target_modulus,
    derived_moduli,
    lattice_bonds,
    lattice_positions,
    lattice_spacing,
    periodic_elastic_constants,
    relaxed_uniaxial_modulus,
)

STIFFNESS = 2000.0
RADIUS = 0.025
SPACING = 2.0 * RADIUS


class BondTopologyTests(unittest.TestCase):
    def test_thirteen_half_space_neighbours(self):
        self.assertEqual(len(HALF_SPACE_OFFSETS), 13)
        self.assertEqual(len(set(HALF_SPACE_OFFSETS)), 13)
        # Six face + twelve edge diagonals own 13 of the 26 neighbours of a node.
        self.assertEqual(sum(1 for offset in HALF_SPACE_OFFSETS
                             if sum(abs(value) for value in offset) == 1), 3)
        self.assertEqual(sum(1 for offset in HALF_SPACE_OFFSETS
                             if sum(abs(value) for value in offset) == 2), 6)
        self.assertEqual(sum(1 for offset in HALF_SPACE_OFFSETS
                             if sum(abs(value) for value in offset) == 3), 4)

    def test_lattice_spacing_is_two_radii(self):
        self.assertEqual(lattice_spacing(RADIUS), SPACING)
        with self.assertRaises(ValueError):
            lattice_spacing(0.0)

    def test_bond_topology_matches_solver(self):
        nx, ny, nz = 5, 4, 3
        engine = IceDEM3D(DEM3DConfig(nx=nx, ny=ny, nz=nz, radius=RADIUS))
        positions = engine.positions.cpu().numpy()
        pairs = lattice_bonds(nx, ny, nz)
        self.assertEqual(len(pairs), len(engine.pairs))
        mine = lattice_positions(nx, ny, nz, SPACING)

        def signature(node_pairs, coords):
            vectors = (coords[node_pairs[:, 1]] - coords[node_pairs[:, 0]]) / SPACING
            first = vectors[np.arange(len(vectors)), np.argmax(vectors != 0, axis=1)]
            vectors = np.where(first[:, None] < 0, -vectors, vectors)
            return sorted(tuple(np.round(vector, 6)) for vector in vectors)

        # Node numbering differs, so compare the bond multiset as lattice vectors.
        self.assertEqual(signature(pairs, mine),
                         signature(np.asarray(engine.pairs.cpu()), positions))


class ClosedFormConstantTests(unittest.TestCase):
    def setUp(self):
        self.constants = periodic_elastic_constants(STIFFNESS, SPACING)
        self.unit = STIFFNESS / SPACING

    def test_c11_equals_thirteen_thirds_k_over_a(self):
        self.assertAlmostEqual(self.constants["C11_Pa"] / self.unit, 13.0 / 3.0, places=12)

    def test_c12_and_c44_equal_seven_thirds_k_over_a(self):
        self.assertAlmostEqual(self.constants["C12_Pa"] / self.unit, 7.0 / 3.0, places=12)
        self.assertAlmostEqual(self.constants["C44_Pa"] / self.unit, 7.0 / 3.0, places=12)

    def test_cauchy_relation_holds_for_central_force_network(self):
        self.assertEqual(self.constants["C12_Pa"], self.constants["C44_Pa"])

    def test_cubic_symmetry(self):
        for name in ("C22_Pa", "C33_Pa"):
            self.assertAlmostEqual(self.constants[name], self.constants["C11_Pa"], places=6)
        for name in ("C55_Pa", "C66_Pa", "C13_Pa"):
            self.assertAlmostEqual(self.constants[name], self.constants["C44_Pa"], places=6)

    def test_constants_scale_linearly_with_stiffness(self):
        doubled = periodic_elastic_constants(2.0 * STIFFNESS, SPACING)
        self.assertAlmostEqual(doubled["C11_Pa"] / self.constants["C11_Pa"], 2.0, places=12)

    def test_constants_scale_inversely_with_spacing(self):
        # Constants scale as k/a, so halving the spacing doubles them at fixed k.
        finer = periodic_elastic_constants(STIFFNESS, SPACING / 2.0)
        self.assertAlmostEqual(finer["C11_Pa"] / self.constants["C11_Pa"], 2.0, places=12)

    def test_invalid_arguments_rejected(self):
        for stiffness, spacing in ((0.0, SPACING), (-1.0, SPACING),
                                   (STIFFNESS, 0.0), (math.nan, SPACING)):
            with self.assertRaises(ValueError):
                periodic_elastic_constants(stiffness, spacing)


class DerivedModuliTests(unittest.TestCase):
    def test_uniaxial_moduli_hand_values(self):
        unit = STIFFNESS / SPACING
        moduli = derived_moduli(13.0 / 3.0 * unit, 7.0 / 3.0 * unit, 7.0 / 3.0 * unit)
        self.assertAlmostEqual(moduli["E_Pa"] / unit, 2.7, places=12)
        self.assertAlmostEqual(moduli["nu"], 0.35, places=12)
        self.assertAlmostEqual(moduli["K_Pa"] / unit, 3.0, places=12)
        self.assertAlmostEqual(moduli["G_voigt_Pa"] / unit, 7.0 / 3.0, places=12)
        self.assertAlmostEqual(moduli["zener_anisotropy"], 7.0 / 3.0, places=12)
        self.assertAlmostEqual(moduli["cauchy_residual_Pa"], 0.0, places=9)

    def test_reuss_shear_below_voigt(self):
        unit = STIFFNESS / SPACING
        moduli = derived_moduli(13.0 / 3.0 * unit, 7.0 / 3.0 * unit, 7.0 / 3.0 * unit)
        self.assertLess(moduli["G_reuss_Pa"], moduli["G_voigt_Pa"])

    def test_required_stiffness_round_trip(self):
        unit = STIFFNESS / SPACING
        required = bond_stiffness_for_target_modulus(9.0e9, SPACING, 2.7)
        self.assertAlmostEqual(9.0e9 / (required / SPACING * 2.7), 1.0, places=12)
        with self.assertRaises(ValueError):
            bond_stiffness_for_target_modulus(-1.0, SPACING, 2.7)


class FiniteSampleMeasurementTests(unittest.TestCase):
    """The measured finite-sample modulus must lie above the analytic value.

    A loaded sample whose end faces cannot contract freely is stiffer than the
    infinite crystal, so the measurement brackets the analytic answer from above and
    must approach it as the sample grows. This is the boundary bias the audit
    quantifies; it is a property of the benchmark geometry, not a solver defect.
    """

    @classmethod
    def setUpClass(cls):
        cfg = DEM3DConfig()
        cls.analytic = derived_moduli(
            *(periodic_elastic_constants(cfg.bond_stiffness, lattice_spacing(cfg.radius))[key]
              for key in ("C11_Pa", "C12_Pa", "C44_Pa")))["E_Pa"]
        cls.cfg = cfg
        cls.spacing = lattice_spacing(cfg.radius)

    def measure(self, nx, ny, nz, control):
        positions = lattice_positions(nx, ny, nz, self.spacing)
        pairs = lattice_bonds(nx, ny, nz)
        return relaxed_uniaxial_modulus(positions, pairs, self.cfg.bond_stiffness,
                                        control=control)

    def test_measured_modulus_brackets_analytic_from_above(self):
        for control in ("displacement", "load"):
            case = self.measure(9, 5, 3, control)
            self.assertGreater(case["E_Pa"], self.analytic)
            self.assertLess(case["E_Pa"], 1.75 * self.analytic)

    def test_boundary_bias_shrinks_with_sample_size(self):
        for control in ("displacement", "load"):
            small = self.measure(9, 5, 3, control)["E_Pa"] / self.analytic - 1.0
            large = self.measure(21, 11, 9, control)["E_Pa"] / self.analytic - 1.0
            self.assertGreater(small, large)
            self.assertLess(large, 0.20)

    def test_measured_poisson_ratio_approaches_analytic(self):
        case = self.measure(21, 11, 9, "load")
        self.assertLess(abs(case["poisson_ratio_measured"] - 0.35), 0.02)

    def test_control_protocols_agree_within_a_few_percent(self):
        displacement = self.measure(13, 7, 5, "displacement")["E_Pa"]
        load = self.measure(13, 7, 5, "load")["E_Pa"]
        self.assertLess(abs(load / displacement - 1.0), 0.10)

    def test_invalid_control_rejected(self):
        positions = lattice_positions(5, 5, 5, self.spacing)
        pairs = lattice_bonds(5, 5, 5)
        with self.assertRaises(ValueError):
            relaxed_uniaxial_modulus(positions, pairs, self.cfg.bond_stiffness,
                                     control="traction")


if __name__ == "__main__":
    unittest.main()