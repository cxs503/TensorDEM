"""Contract guard for ``forces()`` overrides in the 3-D solver hierarchy.

Motivating regression
---------------------
``IceDEM3D.forces`` gained a keyword-only ``update_history`` parameter (the fix
for the "monitoring a friction simulation changes its trajectory" defect
documented in :mod:`tensordem.contact_invariants`).  Two subclasses override
``forces`` with the *old* fixed signature::

    def forces(self, external_forces=None, *, update_fracture=True): ...

Because ``IceDEM3D.diagnostics`` probes with
``forces(..., update_fracture=False, update_history=False)``, every probe on
those subclasses raised ``TypeError: ... got an unexpected keyword argument
'update_history'``.  Concretely, on `main`:

* ``tests/test_threefield.py`` (2 tests) and
  ``tests/test_dem3d_moving_wedge_bow.py`` (3 tests) and
  ``tests/test_dem3d_wedge_resistance_sensitivity.py`` (1 test) were red -- the
  entire moving-wedge bow-resistance capability and the three-field hull-ice
  pipeline could not complete a single pass;
* ``RudderIceLoadDEM`` failed *silently*: ``step()`` never probes, so no test
  noticed that ``diagnostics()`` -- the only public reaction/pressure report --
  was unreachable.

The static test below walks the whole class graph, so a *future* subclass that
pins the old signature fails immediately instead of silently disabling
monitoring.  The runtime tests pin the base-class side-effect contract that
``update_history``/``update_fracture`` exist to protect.
"""
from __future__ import annotations

import inspect
import unittest
from unittest import mock

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.dem3d_wedge import MovingWedgeBowIceDEM3D
from tensordem.rudder_ice_load import RudderIceLoadDEM

# Imported for their registration side effect: every module that defines an
# IceDEM3D subclass must be imported before the class graph is walked.
import tensordem.ridge_dem  # noqa: F401
import tensordem.threefield.hull_ice  # noqa: F401

BASE_KEYWORDS = ("update_fracture", "update_history")


def _small_config(**overrides) -> DEM3DConfig:
    """Cheap lattice: 18 particles, no edge supports, sub-critical stiffness."""
    base = dict(
        nx=3, ny=2, nz=3, radius=0.02, tool_speed=0.1,
        bond_stiffness=1000.0, contact_stiffness=3000.0,
        breaking_strain=0.5, shear_breaking_strain=0.5,
        drag=0.1, contact_damping=0.5,
        fix_x_edges=False, fix_bottom=False,
    )
    base.update(overrides)
    return DEM3DConfig(**base)


def _class_graph(root: type) -> list[type]:
    seen: list[type] = []
    stack = [root]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.append(current)
        stack.extend(current.__subclasses__())
    return seen


def _wedge(overlap: bool = True) -> MovingWedgeBowIceDEM3D:
    sim = MovingWedgeBowIceDEM3D(_small_config(), bow_speed=0.1, wedge_angle_deg=45.0)
    if overlap:
        # Park the whole lattice on the bow tip so the bow reaction is nonzero.
        sim.positions[:, 0] = sim.bow_tip_x
        sim.positions[:, 2] = sim.bow_tip_z
        sim.velocities.zero_()
    return sim


class ForcesOverrideSignatureTests(unittest.TestCase):
    """Static guard: an override must remain callable with the base keywords."""

    def test_every_forces_override_accepts_the_base_keywords(self):
        graph = _class_graph(IceDEM3D)
        # Guard against a vacuous pass if the graph ever stops importing.
        self.assertIn(MovingWedgeBowIceDEM3D, graph)
        self.assertIn(RudderIceLoadDEM, graph)

        overrides = [c for c in graph if "forces" in c.__dict__]
        self.assertGreaterEqual(len(overrides), 3)  # base + 2 subclasses
        for cls in overrides:
            with self.subTest(override=cls.__name__):
                sig = inspect.signature(cls.forces)
                params = sig.parameters
                for keyword in BASE_KEYWORDS:
                    self.assertIn(
                        keyword, params,
                        f"{cls.__name__}.forces() does not accept {keyword!r}; "
                        "IceDEM3D.diagnostics() probes with it and will raise TypeError",
                    )
                    self.assertEqual(
                        params[keyword].kind, inspect.Parameter.KEYWORD_ONLY,
                        f"{cls.__name__}.forces() must take {keyword!r} keyword-only",
                    )
                # The diagnostic probe call shape must bind on the override.
                sig.bind(object(), None, update_fracture=False, update_history=False)

    def test_diagnostics_is_reachable_on_every_forces_override(self):
        factories = {
            "MovingWedgeBowIceDEM3D": lambda: _wedge(),
            "RudderIceLoadDEM": lambda: RudderIceLoadDEM(_small_config()),
        }
        for name, factory in factories.items():
            with self.subTest(model=name):
                sim = factory()
                sim.step()
                diag = sim.diagnostics()  # raised TypeError before the fix
                self.assertIsInstance(diag, dict)
                self.assertTrue(any(key.endswith("_reaction_x_N") for key in diag))


class ProbeSideEffectContractTests(unittest.TestCase):
    """Runtime guard: a probe must not mutate solver state."""

    def test_probe_does_not_commit_the_stored_reaction(self):
        sim = _wedge()
        sim.forces()  # owning call commits the stored reaction
        committed = sim.last_reaction.clone()
        self.assertGreater(float(committed.abs().sum()), 0.0, "test needs a live bow reaction")

        # Advance the bow without touching positions so the next probe measures a
        # strictly larger overlap: the probe value must differ from the committed
        # one, otherwise this test could pass vacuously.
        sim.time += 0.5
        diag = sim.diagnostics()
        probed = torch.tensor(
            [diag["bow_reaction_x_N"], diag["bow_reaction_y_N"], diag["bow_reaction_z_N"]],
            dtype=committed.dtype,
        )
        self.assertFalse(torch.allclose(probed, committed),
                         "probe reaction should differ from the committed one")
        self.assertTrue(torch.equal(sim.last_reaction, committed),
                        "diagnostics() committed a reaction it was only asked to measure")

    def test_probe_does_not_alias_the_returned_reaction(self):
        sim = _wedge()
        _, reaction = sim.forces()
        self.assertIsNot(reaction, sim.last_reaction,
                         "forces() must copy into the preallocated last_reaction, not rebind it")
        snapshot = sim.last_reaction.clone()
        reaction.zero_()  # a caller scribbling on its result must not move solver state
        self.assertTrue(torch.equal(sim.last_reaction, snapshot))

    def test_committed_reaction_equals_the_returned_reaction(self):
        sim = _wedge()
        _, reaction = sim.forces()
        self.assertTrue(torch.equal(sim.last_reaction, reaction))
        self.assertGreater(float(reaction.abs().sum()), 0.0)

    def test_update_history_is_forwarded_to_the_base_solver(self):
        sim = _wedge()
        seen: dict[str, object] = {}
        original = IceDEM3D.forces

        def spy(self, external_forces=None, *, update_fracture=True, update_history=True):
            seen["update_history"] = update_history
            return original(self, external_forces, update_fracture=update_fracture,
                            update_history=update_history)

        with mock.patch.object(IceDEM3D, "forces", spy):
            sim.forces(update_history=False)
            self.assertIs(seen["update_history"], False)
            sim.diagnostics()
            self.assertIs(seen["update_history"], False)

    def test_probe_is_bitwise_idempotent(self):
        sim = _wedge()
        first, _ = sim.forces(update_fracture=False, update_history=False)
        second, _ = sim.forces(update_fracture=False, update_history=False)
        self.assertTrue(torch.equal(first, second),
                        "repeated probe changed the force state (history advanced)")


if __name__ == "__main__":
    unittest.main()