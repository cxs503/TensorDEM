"""Independent verification of bond_stiffness_for_target_modulus against the Born homogenisation pipeline.

Why this test exists
--------------------
``bond_stiffness_for_target_modulus(target_E, spacing, modulus_factor)`` inverts
the relation ``E = modulus_factor * k / a`` to recover ``k``.  The existing test
``test_required_stiffness_round_trip`` passes ``modulus_factor=2.7`` directly,
which is the correct value for the 26-neighbour simple-cubic lattice, but it
does not *derive* that factor from the topology.  If the lattice were ever
changed (e.g. adding body diagonals), the hardcoded factor would become wrong
and the round-trip would still pass because the same wrong factor is used in
both directions.

This test derives ``modulus_factor`` independently from
``periodic_elastic_constants`` + ``derived_moduli`` and then checks that the
round-trip ``k -> E -> k`` is exact to machine precision.

All quantities are SI.
"""
from __future__ import annotations

import math
import unittest

from tensordem.lattice_elasticity3d import (
    bond_stiffness_for_target_modulus,
    derived_moduli,
    lattice_spacing,
    periodic_elastic_constants,
)


class ModulusFactorDerivationTests(unittest.TestCase):
    """Derive the modulus factor from the bond topology and verify the round-trip."""

    def test_factor_derived_from_constants_matches_hand_value(self):
        """modulus_factor = E / (k/a) must equal 2.7 for the 26-neighbour lattice."""
        k = 2000.0
        a = 0.05  # 2 * 0.025
        constants = periodic_elastic_constants(k, a)
        moduli = derived_moduli(
            constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]
        )
        derived_factor = moduli["E_Pa"] / (k / a)
        self.assertAlmostEqual(derived_factor, 2.7, places=12)
        # Also verify the hand-derived expression:
        # C11 = 13k/(3a), C12 = 7k/(3a)
        # E = C11 - 2*C12^2/(C11+C12) = 13k/(3a) - 2*(7k/(3a))^2 / (20k/(3a))
        #   = k/a * (13/3 - 2*49/9 / (20/3)) = k/a * (13/3 - 98/60) = k/a * 2.7
        hand = 13.0 / 3.0 - 2.0 * (7.0 / 3.0) ** 2 / (13.0 / 3.0 + 7.0 / 3.0)
        self.assertAlmostEqual(derived_factor, hand, places=12)

    def test_round_trip_exact_for_derived_factor(self):
        """k -> E -> k must be exact when modulus_factor is derived from constants."""
        k_orig = 2000.0
        a = 0.05
        constants = periodic_elastic_constants(k_orig, a)
        moduli = derived_moduli(
            constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]
        )
        factor = moduli["E_Pa"] / (k_orig / a)
        k_recovered = bond_stiffness_for_target_modulus(moduli["E_Pa"], a, factor)
        self.assertAlmostEqual(k_recovered, k_orig, places=12)

    def test_round_trip_exact_for_hardcoded_factor(self):
        """The existing hardcoded factor 2.7 also gives an exact round-trip."""
        k_orig = 2000.0
        a = 0.05
        E = 2.7 * k_orig / a
        k_recovered = bond_stiffness_for_target_modulus(E, a, 2.7)
        self.assertAlmostEqual(k_recovered, k_orig, places=12)

    def test_round_trip_across_stiffness_range(self):
        """Round-trip must hold for a wide range of stiffness values."""
        a = 0.05
        for k in (100.0, 500.0, 2000.0, 10000.0, 1e6):
            constants = periodic_elastic_constants(k, a)
            moduli = derived_moduli(
                constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]
            )
            factor = moduli["E_Pa"] / (k / a)
            k_recovered = bond_stiffness_for_target_modulus(moduli["E_Pa"], a, factor)
            self.assertAlmostEqual(k_recovered, k, places=8,
                                   msg=f"k={k}: recovered {k_recovered}")

    def test_round_trip_across_spacing_range(self):
        """Round-trip must hold for a wide range of lattice spacings."""
        k = 2000.0
        for a in (0.01, 0.025, 0.05, 0.1, 0.5):
            constants = periodic_elastic_constants(k, a)
            moduli = derived_moduli(
                constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]
            )
            factor = moduli["E_Pa"] / (k / a)
            E = moduli["E_Pa"]
            k_recovered = bond_stiffness_for_target_modulus(E, a, factor)
            self.assertAlmostEqual(k_recovered, k, places=8,
                                   msg=f"a={a}: recovered {k_recovered}")

    def test_factor_is_resolution_independent(self):
        """modulus_factor must be independent of spacing (it cancels in E = factor * k/a)."""
        k = 2000.0
        factors = []
        for a in (0.01, 0.025, 0.05, 0.1, 0.2):
            constants = periodic_elastic_constants(k, a)
            moduli = derived_moduli(
                constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]
            )
            factors.append(moduli["E_Pa"] / (k / a))
        self.assertAlmostEqual(max(factors) - min(factors), 0.0, places=12)

    def test_invalid_arguments_rejected(self):
        for args, name in (
            ((-1.0, 0.05, 2.7), "target_E"),
            ((0.0, 0.05, 2.7), "target_E"),
            ((1.0, 0.0, 2.7), "spacing"),
            ((1.0, 0.05, 0.0), "modulus_factor"),
            ((1.0, -0.05, 2.7), "spacing"),
            ((1.0, 0.05, -1.0), "modulus_factor"),
            ((float("nan"), 0.05, 2.7), "target_E"),
            ((1.0, float("nan"), 2.7), "spacing"),
            ((1.0, 0.05, float("nan")), "modulus_factor"),
        ):
            with self.assertRaises(ValueError, msg=f"{name}={args}"):
                bond_stiffness_for_target_modulus(*args)


class DerivedModuliConsistencyTests(unittest.TestCase):
    """Cross-check derived moduli against the elastic constants."""

    def test_poisson_from_cauchy_matches_formula(self):
        """nu = C12 / (C11 + C12) must equal isotropic_poisson_from_cauchy."""
        k = 2000.0
        a = 0.05
        constants = periodic_elastic_constants(k, a)
        moduli = derived_moduli(
            constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]
        )
        self.assertAlmostEqual(
            moduli["nu"], moduli["isotropic_poisson_from_cauchy"], places=15
        )

    def test_cauchy_residual_is_zero(self):
        """Central-force network: C12 - C44 = 0 exactly."""
        k = 2000.0
        a = 0.05
        constants = periodic_elastic_constants(k, a)
        moduli = derived_moduli(
            constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]
        )
        self.assertAlmostEqual(moduli["cauchy_residual_Pa"], 0.0, places=12)

    def test_bulk_modulus_formula(self):
        """K = (C11 + 2*C12) / 3 must hold."""
        k = 2000.0
        a = 0.05
        constants = periodic_elastic_constants(k, a)
        moduli = derived_moduli(
            constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]
        )
        expected = (constants["C11_Pa"] + 2.0 * constants["C12_Pa"]) / 3.0
        self.assertAlmostEqual(moduli["K_Pa"], expected, places=12)

    def test_zener_anisotropy_formula(self):
        """Zener = 2*C44 / (C11 - C12) must hold."""
        k = 2000.0
        a = 0.05
        constants = periodic_elastic_constants(k, a)
        moduli = derived_moduli(
            constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]
        )
        expected = 2.0 * constants["C44_Pa"] / (constants["C11_Pa"] - constants["C12_Pa"])
        self.assertAlmostEqual(moduli["zener_anisotropy"], expected, places=12)

    def test_reuss_shear_below_voigt(self):
        """Reuss bound must be below Voigt bound for any anisotropic crystal."""
        k = 2000.0
        a = 0.05
        constants = periodic_elastic_constants(k, a)
        moduli = derived_moduli(
            constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"]
        )
        self.assertLess(moduli["G_reuss_Pa"], moduli["G_voigt_Pa"])

    def test_invalid_stiffness_rejected(self):
        with self.assertRaises(ValueError):
            derived_moduli(-1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            derived_moduli(1.0, -2.0, 1.0)  # C11 + C12 < 0


if __name__ == "__main__":
    unittest.main()
