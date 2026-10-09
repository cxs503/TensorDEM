import unittest

import torch

from tensordem import DEMConfig, IceDEM


class LoadInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_external_load_is_additive_and_step_uses_it(self):
        cfg = DEMConfig(nx=3, ny=2, drag=0, fix_edges=False)
        sim = IceDEM(cfg)
        baseline, _ = sim.forces()
        loads = torch.zeros_like(sim.positions)
        loads[2, 1] = 12.5
        loaded, _ = sim.forces(external_forces=loads)
        torch.testing.assert_close(loaded - baseline, loads)

        sim.step(external_forces=loads)
        self.assertGreater(float(sim.velocities[2, 1]), 0.0)
        self.assertTrue(bool(torch.isfinite(sim.positions).all()))

    def test_external_load_shape_and_finiteness_are_validated(self):
        sim = IceDEM(DEMConfig(nx=3, ny=2, fix_edges=False))
        with self.assertRaises(ValueError):
            sim.forces(torch.zeros((len(sim.positions), 3), dtype=torch.float64))
        invalid = torch.zeros_like(sim.positions)
        invalid[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            sim.step(invalid)
        with self.assertRaises(TypeError):
            sim.forces([[0.0, 0.0]] * len(sim.positions))

    def test_fixed_edge_support_reaction_is_reported(self):
        sim = IceDEM(DEMConfig(nx=5, ny=3, drag=0))
        loads = torch.zeros_like(sim.positions)
        loads[sim.fixed, 0] = 3.0
        force, _ = sim.forces(external_forces=loads)
        expected = -force[sim.fixed].sum(0)
        self.assertTrue(bool(torch.isfinite(expected).all()))
        sim.step(external_forces=loads)
        torch.testing.assert_close(
            sim.positions[sim.fixed], sim.initial_positions[sim.fixed], atol=0, rtol=0
        )
        diag = sim.diagnostics()
        self.assertIn("boundary_reaction_x", diag)
        self.assertIn("boundary_reaction_y", diag)


if __name__ == "__main__":
    unittest.main()
