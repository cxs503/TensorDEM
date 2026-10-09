"""Regression tests for per-bond fracture-event bookkeeping and restart safety."""

import copy
import unittest

import torch

from tensordem import DEMConfig, IceDEM


class FractureEventTests(unittest.TestCase):
    def make_sim(self, *, tensile=1.0e-4, shear=10.0):
        return IceDEM(DEMConfig(
            nx=4,
            ny=3,
            fix_edges=False,
            drag=0.0,
            breaking_strain=tensile,
            shear_breaking_strain=shear,
            tool_gap=10.0,
            tool_speed=0.01,
        ))

    def test_tensile_failure_records_mode_time_location_and_energy(self):
        sim = self.make_sim(tensile=1.0e-4, shear=10.0)
        sim.positions[5, 0] += 1.0e-3
        sim.forces()

        events = sim.fracture_events()
        self.assertTrue(events)
        self.assertTrue(all(event["mode"] == "tensile" for event in events))
        self.assertTrue(all(event["time_s"] == 0.0 for event in events))
        self.assertTrue(all(event["released_spring_energy_J"] >= 0.0 for event in events))
        self.assertTrue(all(len(event["midpoint_m"]) == 2 for event in events))
        self.assertAlmostEqual(
            sum(event["released_spring_energy_J"] for event in events),
            sim.accounting["fracture_release_J"],
            places=14,
        )

    def test_shear_failure_is_reported_separately_from_tension(self):
        sim = self.make_sim(tensile=10.0, shear=1.0e-3)
        # Affine simple shear: x' = x + gamma*y. The local objective strain
        # should trigger the shear criterion without axial tensile failure.
        sim.positions[:, 0] += 0.02 * sim.positions[:, 1]
        sim.forces()
        events = sim.fracture_events()
        self.assertTrue(events)
        self.assertTrue(all(event["mode"] == "shear" for event in events))

    def test_restart_round_trip_preserves_fracture_ledger(self):
        sim = self.make_sim(tensile=1.0e-4, shear=10.0)
        sim.positions[5, 0] += 1.0e-3
        sim.forces()
        state = sim.snapshot()
        restored = IceDEM.from_snapshot(state)

        self.assertEqual(restored.fracture_events(), sim.fracture_events())
        self.assertEqual(restored.broken_bonds, sim.broken_bonds)
        self.assertEqual(
            restored.accounting["fracture_release_J"],
            sim.accounting["fracture_release_J"],
        )

    def test_restart_rejects_inconsistent_failure_ledger(self):
        sim = self.make_sim(tensile=1.0e-4, shear=10.0)
        sim.positions[5, 0] += 1.0e-3
        sim.forces()
        state = copy.deepcopy(sim.snapshot())
        broken = next(i for i, mode in enumerate(state["failure_mode"]) if mode > 0)
        state["failure_mode"][broken] = 0
        with self.assertRaises(ValueError):
            IceDEM.from_snapshot(state)

    def test_repeated_force_evaluation_does_not_duplicate_failure_energy(self):
        sim = self.make_sim(tensile=1.0e-4, shear=10.0)
        sim.positions[5, 0] += 1.0e-3
        sim.forces()
        first_energy = sim.accounting["fracture_release_J"]
        first_events = sim.fracture_events()
        sim.forces()
        self.assertEqual(sim.accounting["fracture_release_J"], first_energy)
        self.assertEqual(sim.fracture_events(), first_events)


if __name__ == "__main__":
    unittest.main()
