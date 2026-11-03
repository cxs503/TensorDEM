"""Pin the pair-contact damping defect localised by
``scripts/audit_dem3d_pair_damping_gap.py``.

The two red tests in ``tests/test_restitution.py::EngineClosureTests`` are left
red on purpose: this module records *why* they are red so the debt cannot be
mistaken for a numeric accident and so a fix is signalled by an
``unexpected success`` rather than by silence.

Measured facts (see ``docs/dem3d_pair_damping_contract_gap.md``):

* the platen branch executes the documented clipped law ``F = max(0, k d - c v)``
  to within 0.5 %;
* the particle-particle branch ignores ``DEM3DConfig.contact_damping``
  completely (``e = 1.0000`` at ``c_n = 40`` N s/m against a documented 0.0663);
* its ``restitution_coefficient`` path uses the *bilateral* zeta mapping, so it
  under-damps the clipped law by up to +31.6 % in ``e``.

Everything here is one isolated impact, so the module runs in seconds.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

STIFFNESS = 5000.0
RADIUS = 0.025
DENSITY = 917.0
PARTICLE_MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3
M_EFF_PAIR = PARTICLE_MASS / 2.0
M_EFF_WALL = PARTICLE_MASS


def _refs():
    import verify_dem3d_restitution as ref
    from tensordem.restitution import unilateral_restitution

    return ref, unilateral_restitution


class PlatenBranchControl(unittest.TestCase):
    """Control: the platen branch does execute the documented law."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        ref, unilateral_restitution = _refs()
        cls.ref = ref
        # bind as a staticmethod: a plain function assigned to a TestCase class
        # attribute would be turned into a bound method
        cls.unilateral = staticmethod(unilateral_restitution)

    def test_platen_restitution_matches_documented_clipped_law(self):
        for c_n in (5.0, 20.0):
            measured = self.ref.measure_platen(c_n)["restitution"]
            expected = self.unilateral(c_n, STIFFNESS, M_EFF_WALL)
            rel = abs(measured - expected) / expected
            self.assertLess(rel, 0.03, f"c_n={c_n}: {measured:.5f} vs {expected:.5f}")


class PairBranchDeviations(unittest.TestCase):
    """Debt pinned as expected failures: a fix shows up as an unexpected success."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        ref, unilateral_restitution = _refs()
        cls.ref = ref
        # bind as a staticmethod: a plain function assigned to a TestCase class
        # attribute would be turned into a bound method
        cls.unilateral = staticmethod(unilateral_restitution)

    @unittest.expectedFailure
    def test_pair_contact_damping_reaches_the_engine(self):
        """``contact_damping`` must damp a sphere-sphere contact (it does not)."""
        measured = self.ref.measure_pair(40.0)["restitution"]
        expected = self.unilateral(40.0, STIFFNESS, M_EFF_PAIR)
        self.assertLess(
            abs(measured - expected) / expected,
            0.03,
            f"c_n=40: engine {measured:.5f} vs documented law {expected:.5f}",
        )

    @unittest.expectedFailure
    def test_pair_restitution_coefficient_uses_the_clipped_inverse(self):
        """``restitution_coefficient`` must invert the *clipped* law, not bilateral."""
        from tensordem.restitution import damping_for_restitution

        target = 0.3
        c_n = damping_for_restitution(target, STIFFNESS, M_EFF_PAIR)
        sim_e = self._pair_with_restitution_coefficient(target, c_n)
        self.assertLess(
            abs(sim_e - target) / target,
            0.03,
            f"target {target}: measured {sim_e:.5f}",
        )

    def _pair_with_restitution_coefficient(self, target: float, c_n: float) -> float:
        import audit_dem3d_pair_damping_gap as probe

        return probe.measure_pair_with(0.0, target)["restitution"]


if __name__ == "__main__":
    unittest.main()