import unittest
import torch

from tensordem import DEMConfig, IceDEM


class IceDEMHullIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_prescribed_polyline_produces_balanced_horizontal_reaction(self):
        cfg = DEMConfig(nx=5, ny=3, tool_speed=0.2, fix_edges=False)
        profile = torch.tensor([[0.0, -0.1], [0.0, 0.2]], dtype=torch.float64)
        sim = IceDEM(
            cfg,
            hull_profile=profile,
            hull_start=(0.01, 0.0),
            hull_velocity=(0.2, 0.0),
        )
        force, reaction = sim.forces()
        self.assertGreater(float(reaction[0]), 0.0)
        torch.testing.assert_close(force.sum(dim=0) - reaction * -1, force.sum(dim=0) + (-reaction), rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(sim.tool_position[0]), 0.01, places=12)

    def test_hull_moves_at_prescribed_horizontal_velocity(self):
        cfg = DEMConfig(nx=5, ny=3, tool_speed=0.2, fix_edges=False)
        profile = torch.tensor([[0.0, -0.1], [0.0, 0.2]], dtype=torch.float64)
        sim = IceDEM(
            cfg,
            hull_profile=profile,
            hull_start=(-0.3, 0.0),
            hull_velocity=(0.2, 0.0),
        )
        x0 = float(sim.tool_position[0])
        sim.step()
        self.assertAlmostEqual(float(sim.tool_position[0]), x0 + 0.2 * sim.dt, places=12)

    def test_invalid_hull_profile_rejected(self):
        cfg = DEMConfig(nx=5, ny=3)
        with self.assertRaisesRegex(ValueError, "zero-length"):
            IceDEM(
                cfg,
                hull_profile=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            )


if __name__ == "__main__":
    unittest.main()
