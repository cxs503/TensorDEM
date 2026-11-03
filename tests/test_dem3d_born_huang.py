"""Verification tests for the periodic k = 0 Born-Huang internal-strain module.

The checks fall in three groups:

1. **Topology identity** -- the offset set rebuilt in ``born_huang`` must equal the
   one ``dem3d_lattice_elastic`` uses, and the solver's own bond list.
2. **Closed-form anchors** -- the solver lattice must reproduce the analytic
   affine constants and, because it is a one-atom-basis central-force lattice,
   must have *zero* internal strain to machine precision.
3. **Negative controls** -- a two-atom-basis lattice with inequivalent springs
   must show a non-zero internal strain that matches its closed form to machine
   precision. Without this the "zero internal strain" result above could be
   produced by an engine that can never report anything else.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem import born_huang as bh  # noqa: E402

RADIUS = 0.025
STIFFNESS = 2000.0
SPACING = 2.0 * RADIUS


class TopologyTests(unittest.TestCase):
    def test_half_space_offsets_match_merged_module(self) -> None:
        from tensordem import dem3d_lattice_elastic as merged

        mine = {tuple(int(v) for v in row) for row in bh.solver_half_space_offsets()}
        theirs = {tuple(int(v) for v in row) for row in merged.bond_offset_set()}
        self.assertEqual(mine, theirs)
        self.assertEqual(len(mine), 13)

    def test_bond_count_matches_coordination_number(self) -> None:
        network = bh.solver_supercell_bonds(SPACING, (3, 3, 3), STIFFNESS)
        self.assertAlmostEqual(network["bonds_per_cell"], 13.0, places=12)
        self.assertEqual(network["coordination_number"], 26)
        # Every bond vector must be one of the 26 shell vectors.
        lengths = np.linalg.norm(network["vectors"], axis=1) / SPACING
        allowed = {1.0, math.sqrt(2.0), math.sqrt(3.0)}
        for value in np.unique(np.round(lengths, 10)):
            self.assertTrue(any(abs(value - a) < 1e-9 for a in allowed), value)


class SolverLatticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = bh.solver_supercell_bonds(SPACING, (3, 3, 3), STIFFNESS)
        cls.result = bh.bonded_lattice_analysis(cls.network)

    def test_affine_constants_match_closed_form(self) -> None:
        k, a = STIFFNESS, SPACING
        # The biaxial probe has two unit strain components, so the reported
        # quantity is 2 * (C11 + C12), not C11 + C12.
        expected = {"xx": 13.0 * k / (3.0 * a), "biaxial": 40.0 * k / (3.0 * a),
                    "shear": 7.0 * k / (3.0 * a)}
        for mode, value in expected.items():
            measured = self.result["modes"][mode]["affine_constant_Pa"]
            self.assertLess(abs(measured / value - 1.0), 1e-12, mode)

    def test_affine_state_is_force_balanced(self) -> None:
        """The inversion symmetry of the 26-neighbour shell cancels f exactly."""
        for mode, data in self.result["modes"].items():
            self.assertLess(data["affine_force_relative"], 1e-12, mode)
            self.assertGreater(data["affine_force_scale_N"], 0.0, mode)

    def test_internal_strain_vanishes(self) -> None:
        for mode, data in self.result["modes"].items():
            self.assertLess(data["internal_strain_max_m"], 1e-15, mode)
            self.assertLess(abs(data["ratio_relaxed_to_affine"] - 1.0), 1e-12, mode)

    def test_cauchy_relation(self) -> None:
        modes = self.result["modes"]
        c11 = modes["xx"]["affine_constant_Pa"]
        c12 = 0.5 * modes["biaxial"]["affine_constant_Pa"] - c11
        c44 = modes["shear"]["affine_constant_Pa"]
        self.assertLess(abs(c12 - c44) / c44, 1e-12)
        self.assertGreater(c11, 0.0)


class DiatomicControlTests(unittest.TestCase):
    """Two-atom basis with inequivalent springs: the engine must relax."""

    def test_closed_form_agreement(self) -> None:
        for k1, k2 in ((2.0, 1.0), (1.0, 1.0), (5.0, 0.2), (1.0, 4.0)):
            network = bh.diatomic_chain_bonds(1.0, k1, k2, cells=3)
            data = bh.bonded_lattice_analysis(network, modes=("xx",))["modes"]["xx"]
            closed = bh.diatomic_chain_closed_form(1.0, k1, k2)
            self.assertLess(abs(data["affine_constant_Pa"] / closed["C11_affine_Pa"] - 1.0), 1e-12)
            self.assertLess(abs(data["relaxed_constant_Pa"] / closed["C11_relaxed_Pa"] - 1.0), 1e-12)
            measured = data["internal_strain_per_unit_strain"][0, 0]
            self.assertLess(abs(measured - closed["internal_strain_per_unit_strain"]), 1e-12)

    def test_relaxation_is_a_strict_lower_bound(self) -> None:
        network = bh.diatomic_chain_bonds(1.0, 5.0, 0.2, cells=3)
        data = bh.bonded_lattice_analysis(network, modes=("xx",))["modes"]["xx"]
        self.assertLess(data["relaxed_constant_Pa"], data["affine_constant_Pa"])
        self.assertGreater(data["internal_strain_max_m"], 1e-3)

    def test_equal_springs_have_no_internal_strain(self) -> None:
        network = bh.diatomic_chain_bonds(1.0, 2.0, 2.0, cells=3)
        data = bh.bonded_lattice_analysis(network, modes=("xx",))["modes"]["xx"]
        self.assertLess(data["internal_strain_max_m"], 1e-15)
        self.assertLess(abs(data["ratio_relaxed_to_affine"] - 1.0), 1e-12)

    def test_three_dimensional_two_basis_control(self) -> None:
        """Same axial answer once transverse same-species springs are added."""
        network = bh.diatomic_chain_bonds(1.0, 2.0, 1.0, cells=3,
                                          transverse_stiffness=0.7)
        modes = bh.bonded_lattice_analysis(network, modes=("xx", "yy"))["modes"]
        closed = bh.diatomic_chain_closed_form(1.0, 2.0, 1.0)
        self.assertLess(abs(modes["xx"]["affine_constant_Pa"] / closed["C11_affine_Pa"] - 1.0), 1e-12)
        self.assertLess(abs(modes["xx"]["relaxed_constant_Pa"] / closed["C11_relaxed_Pa"] - 1.0), 1e-12)
        # Transverse springs are inert under eps_xx and affine under eps_yy.
        self.assertLess(abs(modes["yy"]["affine_constant_Pa"] - 0.7), 1e-12)
        self.assertLess(abs(modes["yy"]["ratio_relaxed_to_affine"] - 1.0), 1e-12)


class FreeSurfaceReproductionTests(unittest.TestCase):
    """Reproduce the merged free-surface audit at the sizes it is well posed."""

    def test_documented_sizes_are_reproduced(self) -> None:
        documented = {4: 0.7430, 8: 0.5732}
        for size, ratio in documented.items():
            data = bh.free_surface_specimen(size, mode="shear")
            self.assertLess(abs(data["ratio_relaxed_to_affine"] / ratio - 1.0), 5e-4, size)
            self.assertLess(data["free_dof_residual_N"], 1e-9)

    def test_relaxation_is_not_a_surface_skin(self) -> None:
        """The non-affine field reaches the core, so it is not local relaxation."""
        data = bh.free_surface_specimen(8, mode="shear")
        quotient = data["core_rms_displacement_m"] / data["surface_rms_displacement_m"]
        self.assertGreater(quotient, 0.5)


if __name__ == "__main__":
    unittest.main()