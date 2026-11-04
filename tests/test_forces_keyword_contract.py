"""Keyword-signature contract between ``IceDEM3D.forces`` and its subclasses.

Motivation (m3155).  The trunk test suite carried six red tests whose single
root cause was a *subclass signature drift*: ``IceDEM3D.forces`` gained the
keyword-only ``update_history`` flag (frozen contact-history evaluation, added
with ``contact_invariants``), but the override in
``tensordem.dem3d_wedge.MovingWedgeBowIceDEM3D.forces`` was never widened.  Any
caller that pinned the history -- including the three-field hull-ice coupling
shim ``tensordem.threefield.hull_ice.BowContactDEM`` -- therefore died with
``TypeError: forces() got an unexpected keyword argument 'update_history'``.

A behavioural test on the wedge solver alone does not catch that class of
defect, because the caller that trips it lives in another subsystem.  These
tests check the *contract* instead: every ``forces`` override on the public
solver surface must accept every keyword the base class accepts, and the
frozen-history call must actually execute.
"""
from __future__ import annotations

import inspect
import unittest

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.dem3d_wedge import MovingWedgeBowIceDEM3D

torch.set_num_threads(1)

#: ``IceDEM3D.forces`` subclasses that are part of the public solver surface.
FORCES_OVERRIDES = (MovingWedgeBowIceDEM3D,)


def _keyword_only(func) -> set[str]:
    params = inspect.signature(func).parameters
    return {name for name, p in params.items() if p.kind is p.KEYWORD_ONLY}


class ForcesKeywordContractTests(unittest.TestCase):
    def test_overrides_accept_every_base_keyword(self):
        base = _keyword_only(IceDEM3D.forces)
        self.assertIn("update_history", base)
        self.assertIn("update_fracture", base)
        for cls in FORCES_OVERRIDES:
            with self.subTest(cls=cls.__name__):
                missing = base - _keyword_only(cls.forces)
                self.assertEqual(
                    missing, set(),
                    msg=f"{cls.__name__}.forces() does not accept {sorted(missing)}",
                )

    def test_every_base_keyword_is_callable(self):
        base = _keyword_only(IceDEM3D.forces)
        for cls in FORCES_OVERRIDES:
            with self.subTest(cls=cls.__name__):
                sim = cls(DEM3DConfig(particle_radius=0.1), bow_speed=0.05, initial_gap=0.05)
                force, reaction = sim.forces(**{k: False for k in base})
                self.assertEqual(force.shape[-1], 3)
                self.assertEqual(tuple(reaction.shape), (3,))
                repeated, _ = sim.forces(**{k: False for k in base})
                self.assertTrue(torch.equal(force, repeated))


if __name__ == "__main__":
    unittest.main()
