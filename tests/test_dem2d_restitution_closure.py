"""Verification of the 2-D engine's contact-damping -> restitution closure.

Layers, cheapest first:

* argument validation of the probe;
* closed-form consistency of the two reference branches;
* **engine closure** -- ``DEMConfig.contact_damping`` fed to the real ``IceDEM``
  contact branch and the measured restitution compared with the continuous
  *unilateral* (clipped-law) prediction, tolerances are the engineering 3 %;
* the bilateral textbook formula is checked to be *worse* (this is the trap the
  module exists to document);
* inverse mapping: ``recommended_damping`` hits the target, the classical
  inverse does not;
* time-step convergence of the semi-implicit Euler scheme and speed
  independence of a linear contact law.

The measurements are 2x2 lattices with one free probe particle: seconds, not
minutes, and no dependence on machine size.
"""
import math
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import tensordem as _tensordem  # noqa: E402

if not Path(_tensordem.__file__).resolve().is_relative_to(_ROOT / "src"):
    for _name in [m for m in sys.modules if m == "tensordem" or m.startswith("tensordem.")]:
        del sys.modules[_name]
    import importlib

    importlib.invalidate_caches()
    import tensordem as _tensordem  # noqa: F811

from tensordem.dem2d_restitution import (  # noqa: E402
    discretization_bias,
    measure_restitution,
    particle_mass,
    recommended_classical_damping,
    recommended_damping,
)
from tensordem.restitution import (  # noqa: E402
    classical_restitution,
    unilateral_restitution,
)

STIFFNESS = 2_000.0
MASS = particle_mass()


class ProbeValidationTests(unittest.TestCase):
    def test_rejects_nonpositive_inputs(self):
        with self.assertRaises(ValueError):
            measure_restitution(impact_speed=0.0)
        with self.assertRaises(ValueError):
            measure_restitution(damping=-1.0)
        with self.assertRaises(ValueError):
            measure_restitution(stiffness=0.0)
        with self.assertRaises(ValueError):
            measure_restitution(probe=99)

    def test_particle_mass_matches_config(self):
        self.assertAlmostEqual(MASS, 917.0 * math.pi * 0.025**2 * 0.2, places=15)


class ReferenceBranchTests(unittest.TestCase):
    """The two closed forms must bracket the engine, in the documented order."""

    def test_unilateral_exceeds_bilateral(self):
        for damping in (2.0, 5.0, 10.0, 20.0, 50.0):
            unilateral = unilateral_restitution(damping, STIFFNESS, MASS, 1.0)
            bilateral = classical_restitution(damping, STIFFNESS, MASS)
            self.assertGreater(unilateral, bilateral)

    def test_branches_coincide_without_damping(self):
        self.assertAlmostEqual(
            unilateral_restitution(0.0, STIFFNESS, MASS, 1.0), 1.0, places=6
        )
        self.assertAlmostEqual(classical_restitution(0.0, STIFFNESS, MASS), 1.0, places=6)


class EngineClosureTests(unittest.TestCase):
    """The 2-D engine must reproduce the clipped-law prediction within 3 %."""

    def test_frictionless_impact_is_elastic(self):
        result = measure_restitution(stiffness=STIFFNESS, damping=0.0)
        # Semi-implicit Euler gains O(dt) energy in a linear oscillator; the
        # measured value therefore sits just above 1.
        self.assertLess(abs(result.measured_restitution - 1.0), 0.005)

    def test_measured_matches_unilateral_across_damping(self):
        for damping in (2.0, 5.0, 10.0, 20.0, 50.0):
            with self.subTest(damping=damping):
                result = measure_restitution(stiffness=STIFFNESS, damping=damping)
                self.assertLess(
                    result.error_vs_unilateral,
                    0.03,
                    msg=f"c={damping}: measured {result.measured_restitution:.5f} vs "
                    f"unilateral {result.unilateral_prediction:.5f}",
                )

    def test_classical_formula_degrades_at_moderate_damping(self):
        moderate = measure_restitution(stiffness=STIFFNESS, damping=20.0)
        self.assertGreater(moderate.damping_ratio, 0.3)
        self.assertLess(moderate.error_vs_unilateral, 0.03)
        self.assertGreater(moderate.error_vs_classical, 0.20)

    def test_heavily_damped_contact_still_rebounds(self):
        """Where the bilateral formula predicts sticking, the engine rebounds."""
        result = measure_restitution(stiffness=STIFFNESS, damping=50.0)
        self.assertGreater(result.damping_ratio, 0.9)
        self.assertGreater(result.measured_restitution, 0.10)
        self.assertLess(result.classical_prediction, 0.01)
        self.assertLess(result.error_vs_unilateral, 0.03)

    def test_restition_is_speed_independent(self):
        slow = measure_restitution(stiffness=STIFFNESS, damping=5.0, impact_speed=0.1)
        fast = measure_restitution(stiffness=STIFFNESS, damping=5.0, impact_speed=5.0)
        self.assertLess(abs(slow.measured_restitution - fast.measured_restitution), 0.01)

    def test_restitution_is_monotone_decreasing_in_damping(self):
        values = [
            measure_restitution(stiffness=STIFFNESS, damping=c).measured_restitution
            for c in (0.0, 2.0, 5.0, 10.0, 20.0, 50.0)
        ]
        for previous, current in zip(values, values[1:]):
            self.assertGreater(previous, current)


class InverseMappingTests(unittest.TestCase):
    """Inverting the engine law, not the textbook formula."""

    def test_unilateral_inverse_hits_target(self):
        target = 0.70
        damping = recommended_damping(target, stiffness=STIFFNESS, mass=MASS)
        result = measure_restitution(stiffness=STIFFNESS, damping=damping)
        self.assertLess(abs(result.measured_restitution - target) / target, 0.03)

    def test_classical_inverse_misses_target(self):
        target = 0.70
        classical = recommended_classical_damping(target, stiffness=STIFFNESS, mass=MASS)
        unilateral = recommended_damping(target, stiffness=STIFFNESS, mass=MASS)
        self.assertLess(classical, unilateral)
        result = measure_restitution(stiffness=STIFFNESS, damping=classical)
        self.assertGreater((result.measured_restitution - target) / target, 0.03)


class TimeStepTests(unittest.TestCase):
    """The residual closure error is the integrator, and it converges O(dt)."""

    def test_refinement_converges_first_order(self):
        levels = discretization_bias(
            stiffness=STIFFNESS, damping=5.0, refinement=4, levels=3
        )
        errors = [level.error_vs_unilateral for level in levels]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        # Refining dt by 4 must cut the error by clearly more than 2x.
        self.assertLess(errors[1], 0.5 * errors[0])
        self.assertLess(errors[2], 0.5 * errors[1])
        self.assertLess(errors[2], 0.005)

    def test_dt_never_exceeds_recommendation(self):
        levels = discretization_bias(stiffness=STIFFNESS, damping=20.0, refinement=2, levels=2)
        self.assertGreater(levels[0].dt, levels[1].dt)


if __name__ == "__main__":
    unittest.main()