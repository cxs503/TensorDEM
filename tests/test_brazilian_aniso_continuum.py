"""Verification of the anisotropic-continuum reference used to attribute the
Brazilian-disc centre-field miss.

The finite-element branch of `scripts/brazilian_aniso_continuum.py` is only
allowed to act as a *reference* for the bonded-sphere specimen because it is
validated here, in a limit where the answer is known:

1. a uniform-strain patch test (exact, machine precision);
2. the isotropic limit reproduces the Hondros point-load solution
   (``sigma_zz/sigma_xx = -3``, ``sigma_xx(0,0) = 2 P/(pi D t)``) as the contact
   arc is narrowed, at a mesh-converged resolution;
3. the plane-stress reduction of the cubic tensor agrees with the textbook
   isotropic plane-stress matrix, and with the *discrete* bond network's own
   virial stress under the same homogeneous strain -- so the continuum tensor and
   the DEM bond law are the same constitutive law, not two models that merely
   look similar;
4. the anisotropic branch lands near the DEM's measured centre field and, by a
   wide margin, away from Hondros -- the attribution claim itself.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from brazilian_aniso_continuum import (  # noqa: E402
    gauss_stresses,
    isotropic_constants,
    plane_stress_matrix,
    quarter_disc_mesh,
    run_case,
)
from tensordem.lattice_elasticity3d import (  # noqa: E402
    HALF_SPACE_OFFSETS,
    periodic_elastic_constants,
)

RADIUS = 0.15
THICKNESS = 0.06
DIAMETER = 2.0 * RADIUS
YOUNG = 270000.0
POISSON = 0.35
BOND_STIFFNESS = 2000.0
SPACING = 0.02
DEM_ARC_RATIO = 0.0668693921916482          # measured on the nx=nz=15 specimen
HONDROS_RATIO = -3.0
HONDROS_TENSION = 2.0 / math.pi             # sigma_xx(0,0) / (P / (D t))


def _isotropic_case(rings: int, patch_ratio: float) -> dict:
    return run_case(radius=RADIUS, thickness=THICKNESS, rings=rings, sectors=rings,
                    patch_ratio=patch_ratio, total_load=1.0,
                    constitutive="isotropic-plane-stress",
                    constants=isotropic_constants(YOUNG, POISSON))


def _cubic_case(rings: int, patch_ratio: float, constitutive: str = "cubic-plane-stress") -> dict:
    constants = periodic_elastic_constants(BOND_STIFFNESS, SPACING)
    return run_case(radius=RADIUS, thickness=THICKNESS, rings=rings, sectors=rings,
                    patch_ratio=patch_ratio, total_load=1.0,
                    constitutive=constitutive,
                    constants=(constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]))


def _bond_network_stress(strain: np.ndarray) -> np.ndarray:
    """Affine (Born) virial stress of the periodic 26-neighbour bond network.

    ``sigma_ij = sum_bonds k L^2 (n . eps . n) n_i n_j / a^3`` over one bond per
    pair, computed independently of :func:`periodic_elastic_constants` (which is
    written as a four-index contraction).
    """
    offsets = np.asarray(HALF_SPACE_OFFSETS, dtype=float)
    lengths = np.linalg.norm(offsets, axis=1)
    normals = offsets / lengths[:, None]
    extension = np.einsum("bi,ij,bj->b", normals, strain, normals)
    weights = BOND_STIFFNESS * (lengths * SPACING) ** 2 * extension / SPACING**3
    return np.einsum("b,bi,bj->ij", weights, normals, normals)


class PatchTest(unittest.TestCase):
    """A linear displacement field must be reproduced exactly."""

    def test_uniform_strain_is_exact(self) -> None:
        nodes, elements, _ = quarter_disc_mesh(RADIUS, 12, 12)
        d_matrix = plane_stress_matrix(*isotropic_constants(YOUNG, POISSON))
        strain = np.array([1.0e-4, -2.0e-4, 3.0e-4])       # eps_xx, eps_zz, gamma_xz
        gradient = np.array([[strain[0], 0.5 * strain[2]],
                             [0.5 * strain[2], strain[1]]])
        field = gauss_stresses(nodes, elements, nodes @ gradient, d_matrix)
        exact = d_matrix @ strain
        for component, value in (("sigma_xx", exact[0]), ("sigma_zz", exact[1]),
                                 ("sigma_xz", exact[2])):
            measured = float(np.mean(field[component]))
            self.assertLess(abs(measured / value - 1.0), 1e-12, component)


class PlaneStressReductionTests(unittest.TestCase):
    def test_isotropic_limit_matches_textbook_matrix(self) -> None:
        d = plane_stress_matrix(*isotropic_constants(YOUNG, POISSON))
        factor = YOUNG / (1.0 - POISSON**2)
        textbook = factor * np.array([[1.0, POISSON, 0.0],
                                      [POISSON, 1.0, 0.0],
                                      [0.0, 0.0, 0.5 * (1.0 - POISSON)]])
        np.testing.assert_allclose(d, textbook, rtol=1e-12, atol=0.0)

    def test_cubic_reduction_matches_discrete_bond_network(self) -> None:
        """``sigma_yy = 0`` and the in-plane stress match the bond network's own virial."""
        constants = periodic_elastic_constants(BOND_STIFFNESS, SPACING)
        c11, c12, c44 = constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]
        d_matrix = plane_stress_matrix(c11, c12, c44)
        eps_xx, eps_zz, gamma = 1.0e-4, -2.5e-4, 1.2e-4
        eps_yy = -(c12 / c11) * (eps_xx + eps_zz)          # plane-stress condition
        strain = np.array([[eps_xx, 0.0, 0.5 * gamma],
                           [0.0, eps_yy, 0.0],
                           [0.5 * gamma, 0.0, eps_zz]])
        stress = _bond_network_stress(strain)
        self.assertLess(abs(stress[1, 1]) / max(abs(stress[0, 0]), 1e-30), 1e-12)
        np.testing.assert_allclose(stress[0, 0],
                                   d_matrix[0, 0] * eps_xx + d_matrix[0, 1] * eps_zz,
                                   rtol=1e-12)
        np.testing.assert_allclose(stress[2, 2],
                                   d_matrix[1, 0] * eps_xx + d_matrix[1, 1] * eps_zz,
                                   rtol=1e-12)
        np.testing.assert_allclose(stress[0, 2], c44 * gamma, rtol=1e-12)


class IsotropicHondrosValidation(unittest.TestCase):
    """The reference is only usable if the isotropic branch reproduces Hondros."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.narrow = _isotropic_case(60, 0.002)
        cls.specimen_arc = _isotropic_case(60, DEM_ARC_RATIO)
        cls.coarse = _isotropic_case(20, DEM_ARC_RATIO)

    def test_point_load_principal_ratio(self) -> None:
        ratio = self.narrow["centre_principal_ratio"]
        self.assertLess(abs(ratio / HONDROS_RATIO - 1.0), 0.05,
                        f"centre ratio {ratio:+.4f} vs Hondros -3")

    def test_point_load_centre_tension(self) -> None:
        value = self.narrow["centre_sigma_xx_over_P_on_Dt"]
        self.assertLess(abs(value / HONDROS_TENSION - 1.0), 0.07,
                        f"centre sigma_xx/(P/Dt) = {value:.4f} vs 2/pi = {HONDROS_TENSION:.4f}")

    def test_finite_contact_arc_does_not_move_the_centre_field(self) -> None:
        """0.2 % vs 6.7 % arcs differ by < 1 %: the arc is not the cause of the miss."""
        wide = self.specimen_arc["centre_principal_ratio"]
        narrow = self.narrow["centre_principal_ratio"]
        self.assertLess(abs(wide - narrow) / abs(HONDROS_RATIO), 0.01)

    def test_mesh_convergence(self) -> None:
        fine = _isotropic_case(160, DEM_ARC_RATIO)["centre_principal_ratio"]
        value = self.specimen_arc["centre_principal_ratio"]
        self.assertLess(abs(fine - value) / abs(HONDROS_RATIO), 0.01,
                        f"rings 60 -> 160 changed the ratio by {abs(fine - value) / 3:.4f}")
        self.assertLess(abs(self.coarse["centre_principal_ratio"] - value) / 3.0, 0.02)

    def test_load_equilibrium(self) -> None:
        self.assertLess(self.specimen_arc["support_to_load_rel_error"], 1e-9)
        self.assertLess(self.specimen_arc["global_balance_rel"], 1e-11)


class AnisotropyAttribution(unittest.TestCase):
    """The attribution claim: the miss is cubic anisotropy, not discretisation."""

    #: Relaxed bonded-sphere values from docs/dem3d_brazilian_hondros_closure.md
    #: (nx = nz = 15) and the nx = 21 / 27 refinement rungs of
    #: artifacts/dem3d_brazilian_aniso/attribution.json.
    DEM_RATIOS = {15: -2.0149, 21: -2.0721, 27: -2.0058}
    DEM_RESOLVED = -2.0300

    @classmethod
    def setUpClass(cls) -> None:
        cls.cubic = _cubic_case(60, DEM_ARC_RATIO)
        cls.cubic_strain = _cubic_case(60, DEM_ARC_RATIO, "cubic-plane-strain")
        cls.isotropic = _isotropic_case(60, DEM_ARC_RATIO)

    def test_cubic_continuum_reproduces_the_lattice_centre_field(self) -> None:
        """Cubic FEM and the three DEM resolutions agree to within 3 %."""
        ratio = self.cubic["centre_principal_ratio"]
        for nx, value in self.DEM_RATIOS.items():
            self.assertLess(abs(ratio / value - 1.0), 0.03,
                            f"nx={nx}: continuum {ratio:+.4f} vs DEM {value:+.4f}")

    def test_cubic_continuum_misses_hondros_beyond_the_manifest_tolerance(self) -> None:
        ratio = self.cubic["centre_principal_ratio"]
        self.assertLess(ratio, -1.85)          # nowhere near the isotropic value
        self.assertGreater(abs(ratio / HONDROS_RATIO - 1.0), 0.15,
                           "the anisotropic continuum would sit inside the 15 % band")

    def test_anisotropy_explains_the_miss_better_than_the_isotropic_limit(self) -> None:
        cubic_gap = abs(self.cubic["centre_principal_ratio"] - self.DEM_RESOLVED)
        isotropic_gap = abs(self.isotropic["centre_principal_ratio"] - self.DEM_RESOLVED)
        self.assertLess(cubic_gap, 0.02)
        self.assertGreater(isotropic_gap / cubic_gap, 25.0)

    def test_plane_strain_bracket(self) -> None:
        """The 3-layer body sits between plane stress and plane strain."""
        plane_stress = self.cubic["centre_principal_ratio"]
        plane_strain = self.cubic_strain["centre_principal_ratio"]
        self.assertLess(plane_strain, -1.8)
        # Thicker constraint raises the ratio towards -2: the measured DEM value
        # (-2.03) sits on the plane-stress side, consistent with its free y faces
        # (the relaxed specimen reports centre sigma_yy = -0.42 Pa, i.e. ~0).
        self.assertGreater(plane_strain, plane_stress)
        for value in (plane_stress, plane_strain):
            self.assertLess(abs(value / HONDROS_RATIO - 1.0), 0.45)

    def test_born_anisotropy_is_what_is_missing(self) -> None:
        constants = periodic_elastic_constants(BOND_STIFFNESS, SPACING)
        zener = 2.0 * constants["C44_Pa"] / (constants["C11_Pa"] - constants["C12_Pa"])
        self.assertGreater(zener, 2.0)
        self.assertLess(abs(constants["C12_Pa"] - constants["C44_Pa"]), 1e-6)
        self.assertLess(zener, 2.5)


if __name__ == "__main__":
    unittest.main()