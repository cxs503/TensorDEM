"""Tests for the engine-faithful restitution mapping (``tensordem.restitution``).

Two layers:

* reference-solver properties (pure numerics, milliseconds);
* engine closure -- the damping produced by ``damping_for_restitution`` is fed
  back into ``IceDEM3D`` and the measured restitution is compared with the
  target.  These runs are small (8 particles, tens of contact steps) but they
  exercise the real contact branches, so the tolerances are the engineering
  ones (3%), not floating-point ones.
"""
import math
import sys
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

# This worktree is not always the one the editable install points at, so a
# previously imported ``tensordem`` package can shadow the local sources when
# pytest collects several modules in one session.  Drop such stale modules.
import tensordem as _tensordem  # noqa: E402

if not Path(_tensordem.__file__).resolve().is_relative_to(_ROOT / "src"):
    for _name in [m for m in sys.modules if m == "tensordem" or m.startswith("tensordem.")]:
        del sys.modules[_name]
    import importlib

    importlib.invalidate_caches()
    import tensordem as _tensordem  # noqa: F811

from tensordem.restitution import (
    classical_damping,
    classical_restitution,
    damping_for_restitution,
    effective_mass,
    unilateral_contact_detail,
    unilateral_restitution,
)

STIFFNESS = 5000.0
MASS = 917.0 * (4.0 / 3.0) * math.pi * 0.025 ** 3
M_PAIR = MASS / 2.0
M_WALL = MASS


class EffectiveMassTests(unittest.TestCase):
    def test_sphere_pair_halves_the_mass(self):
        self.assertAlmostEqual(effective_mass(MASS, MASS), MASS / 2.0, places=15)

    def test_wall_keeps_particle_mass(self):
        self.assertAlmostEqual(effective_mass(MASS), MASS, places=15)
        self.assertAlmostEqual(effective_mass(MASS, math.inf), MASS, places=15)

    def test_invalid_mass_rejected(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                effective_mass(bad)


class UnilateralReferenceTests(unittest.TestCase):
    def test_elastic_limit(self):
        """No dashpot => perfectly elastic rebound."""
        self.assertAlmostEqual(unilateral_restitution(0.0, STIFFNESS, M_PAIR), 1.0, places=9)

    def test_monotonic_in_damping(self):
        values = [unilateral_restitution(c, STIFFNESS, M_PAIR)
                  for c in (0.0, 1.0, 5.0, 10.0, 20.0, 40.0)]
        for upper, lower in zip(values, values[1:]):
            self.assertGreater(upper, lower)

    def test_resolution_convergence(self):
        """Doubling the sample count must not move e beyond ~1e-5 relative."""
        coarse = unilateral_restitution(10.0, STIFFNESS, M_PAIR, samples_per_period=2000)
        fine = unilateral_restitution(10.0, STIFFNESS, M_PAIR, samples_per_period=4000)
        self.assertLess(abs(fine - coarse) / fine, 1e-5)

    def test_detachment_is_reported_for_heavy_damping(self):
        """The clipped dashpot opens the contact at delta > 0 when strongly damped."""
        heavy = unilateral_contact_detail(40.0, STIFFNESS, M_PAIR)
        self.assertTrue(heavy.detached)
        self.assertGreater(heavy.peak_overlap, 0.0)

    def test_velocity_independence(self):
        """The linear clipped law gives the same e at any impact speed."""
        slow = unilateral_restitution(10.0, STIFFNESS, M_PAIR, impact_speed=0.05)
        fast = unilateral_restitution(10.0, STIFFNESS, M_PAIR, impact_speed=5.0)
        self.assertLess(abs(slow - fast) / slow, 1e-9)

    def test_unilateral_exceeds_bilateral(self):
        """Dropping the tensile dashpot branch can only retain more energy."""
        for c in (2.0, 5.0, 10.0, 20.0):
            self.assertGreater(
                unilateral_restitution(c, STIFFNESS, M_PAIR),
                classical_restitution(c, STIFFNESS, M_PAIR),
            )

    def test_small_damping_agrees_with_bilateral(self):
        """The two branches must merge in the weakly damped limit."""
        c = 1.0
        cl = classical_restitution(c, STIFFNESS, M_PAIR)
        uni = unilateral_restitution(c, STIFFNESS, M_PAIR)
        self.assertLess(abs(uni - cl) / cl, 0.02)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            unilateral_restitution(-1.0, STIFFNESS, M_PAIR)
        with self.assertRaises(ValueError):
            unilateral_restitution(1.0, 0.0, M_PAIR)
        with self.assertRaises(ValueError):
            unilateral_restitution(1.0, STIFFNESS, -0.1)
        with self.assertRaises(ValueError):
            unilateral_restitution(1.0, STIFFNESS, M_PAIR, impact_speed=0.0)


class DampingInversionTests(unittest.TestCase):
    def test_round_trip(self):
        """Inverting the reference law recovers the requested restitution."""
        for target in (0.9, 0.7, 0.5, 0.3):
            for m_eff in (M_PAIR, M_WALL):
                c = damping_for_restitution(target, STIFFNESS, m_eff)
                back = unilateral_restitution(c, STIFFNESS, m_eff)
                self.assertLess(abs(back - target) / target, 1e-5,
                                msg=f"target {target}, m_eff {m_eff}")

    def test_inversion_differs_from_bilateral_mapping(self):
        """The clipped law needs *more* damping to reach the same restitution.

        Dropping the tensile dashpot branch removes dissipation, so the
        bilateral formula always under-prescribes the damping coefficient.
        """
        for m_eff in (M_PAIR, M_WALL):
            c_clip = damping_for_restitution(0.3, STIFFNESS, m_eff)
            c_classical = classical_damping(0.3, STIFFNESS, m_eff)
            self.assertGreater(c_clip, c_classical)
            self.assertGreater(c_clip / c_classical, 1.2)

    def test_invalid_target_rejected(self):
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                damping_for_restitution(bad, STIFFNESS, M_PAIR)


class EngineClosureTests(unittest.TestCase):
    """Feed the inverted damping into IceDEM3D and measure the rebound."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from verify_dem3d_restitution import measure_pair, measure_platen
        cls.measure_pair = staticmethod(measure_pair)
        cls.measure_platen = staticmethod(measure_platen)

    def _check(self, runner, m_eff, target, tol=0.03):
        c = damping_for_restitution(target, STIFFNESS, m_eff)
        measured = runner(c)["restitution"]
        self.assertLess(abs(measured - target) / target, tol,
                        msg=f"c={c:.4g} target={target} measured={measured:.5f}")

    def test_pair_impact_matches_target(self):
        self._check(self.measure_pair, M_PAIR, 0.7)
        self._check(self.measure_pair, M_PAIR, 0.5)

    def test_platen_impact_matches_target(self):
        self._check(self.measure_platen, M_WALL, 0.5)

    def test_engine_matches_reference_law(self):
        """Engine vs the reference solution of the same clipped law."""
        for damping in (5.0, 20.0):
            measured = self.measure_pair(damping)["restitution"]
            reference = unilateral_restitution(damping, STIFFNESS, M_PAIR)
            self.assertLess(abs(measured - reference) / reference, 0.03,
                            msg=f"damping {damping}: {measured:.5f} vs {reference:.5f}")


if __name__ == "__main__":
    unittest.main()