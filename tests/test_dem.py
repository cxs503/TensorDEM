import csv
import math
from pathlib import Path
import tempfile
import unittest

import torch

from tensordem import DEMConfig, IceDEM
from tensordem.cli import main


class DEMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, **kwargs):
        return IceDEM(DEMConfig(nx=3, ny=2, drag=0, fix_edges=False, **kwargs))

    def test_equilibrium_and_action_reaction(self):
        sim = self.model()
        force, reaction = sim.forces()
        torch.testing.assert_close(force, torch.zeros_like(force), atol=1e-10, rtol=0)
        torch.testing.assert_close(reaction, torch.zeros_like(reaction))
        sim.positions[1, 0] += 0.0001
        force, _ = sim.forces()
        torch.testing.assert_close(force.sum(0), torch.zeros(2, dtype=torch.float64))

    def test_rigid_rotation_preserves_bonds(self):
        sim = self.model()
        angle = 0.6
        rotation = torch.tensor([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ], dtype=torch.float64)
        sim.positions = sim.positions @ rotation.T + torch.tensor([4.0, -4.0])
        force, _ = sim.forces()
        self.assertEqual(sim.broken_bonds, 0)
        torch.testing.assert_close(force, torch.zeros_like(force), atol=1e-10, rtol=0)

    def test_tension_breaks_irreversibly_but_compression_does_not(self):
        sim = self.model()
        sim.positions *= 0.9
        sim.forces()
        self.assertEqual(sim.broken_bonds, 0)
        sim.positions = sim.initial_positions * 1.02
        force, _ = sim.forces()
        self.assertEqual(sim.broken_bonds, int(sim.bonded.sum().item()))
        torch.testing.assert_close(force, torch.zeros_like(force))
        sim.positions = sim.initial_positions * 0.9
        force, _ = sim.forces()
        self.assertFalse(bool(sim.alive.any()))
        self.assertGreater(float(force.abs().sum()), 0)
        torch.testing.assert_close(force.sum(0), torch.zeros(2, dtype=torch.float64))

    def test_objective_shear_breaks_bonds_but_rigid_rotation_does_not(self):
        rotated = self.model(shear_breaking_strain=0.02, breaking_strain=0.5)
        angle = 0.6
        rotation = torch.tensor([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ], dtype=torch.float64)
        rotated.positions = rotated.initial_positions @ rotation.T + torch.tensor([2.0, -3.0])
        rotated.forces()
        self.assertEqual(rotated.broken_bonds, 0)

        sheared = self.model(shear_breaking_strain=0.02, breaking_strain=0.5)
        sheared.positions = sheared.initial_positions.clone()
        sheared.positions[:, 0] += 0.08 * sheared.positions[:, 1]
        sheared.forces()
        self.assertGreater(sheared.broken_bonds, 0)
        # Damage is irreversible even if the original configuration is restored.
        sheared.positions = sheared.initial_positions.clone()
        sheared.forces()
        self.assertGreater(sheared.broken_bonds, 0)

    def test_contact_repels_without_attraction(self):
        sim = self.model()
        sim.alive[:] = False
        sim.positions[1, 0] -= 0.005
        force, _ = sim.forces()
        self.assertLess(float(force[0, 0]), 0)
        self.assertGreater(float(force[1, 0]), 0)
        sim.velocities[:, 0] = sim.positions[:, 0] * 1e6
        force, _ = sim.forces()
        torch.testing.assert_close(force, torch.zeros_like(force))

    def test_tool_reaction_and_fixed_edges(self):
        sim = IceDEM(DEMConfig(nx=5, ny=3, drag=0))
        sim.time = 0.02
        force, reaction = sim.forces()
        self.assertGreater(float(reaction[1]), 0)
        torch.testing.assert_close(force.sum(0), -reaction)
        for _ in range(100):
            sim.step()
        torch.testing.assert_close(
            sim.positions[sim.fixed], sim.initial_positions[sim.fixed], atol=0, rtol=0
        )
        self.assertFalse(bool(sim.velocities[sim.fixed].any()))

    def test_default_demo_fractures_and_stays_finite(self):
        sim = IceDEM()
        previous = 0
        for _ in range(2000):
            sim.step()
            self.assertGreaterEqual(sim.broken_bonds, previous)
            previous = sim.broken_bonds
        self.assertGreater(sim.broken_bonds, 0)
        self.assertTrue(bool(torch.isfinite(sim.positions).all()))

    def test_unforced_motion_conserves_momentum(self):
        sim = self.model()
        sim.positions += torch.tensor([0.0, -10.0])
        sim.velocities[:] = torch.tensor([0.1, 0.05])
        initial_momentum = sim.velocities.sum(0).clone()
        for _ in range(50):
            sim.step()
        torch.testing.assert_close(sim.velocities.sum(0), initial_momentum)

    def test_input_validation(self):
        for kwargs in (
            {"nx": 1}, {"ny": 2.5}, {"radius": 0}, {"drag": -1},
            {"dt": math.nan}, {"breaking_strain": math.inf}, {"tool_speed": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DEMConfig(**kwargs)
        with self.assertRaises(ValueError):
            IceDEM(DEMConfig(dt=1.0))
        with self.assertRaises(ValueError):
            IceDEM(DEMConfig(device="meta"))

    def test_coincident_particles_have_finite_repulsion(self):
        sim = self.model()
        sim.alive[:] = False
        sim.positions[1] = sim.positions[0]
        force, _ = sim.forces()
        self.assertTrue(bool(torch.isfinite(force).all()))
        self.assertGreater(float(force.abs().sum()), 0)

    def test_cli_exports_initial_and_final_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            main(["--nx", "5", "--ny", "3", "--steps", "7",
                  "--save-every", "5", "--output", directory])
            output = Path(directory)
            data = torch.load(output / "trajectory.pt", weights_only=True)
            self.assertEqual(data["positions"].shape, (3, 15, 2))
            self.assertEqual(data["times"][0].item(), 0)
            self.assertAlmostEqual(data["times"][-1].item(), 7 * data["dt"])
            with (output / "history.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(float(rows[-1]["time"]), data["times"][-1].item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_cpu(self):
        cpu = self.model()
        gpu = self.model(device="cuda")
        for _ in range(10):
            cpu.step()
            gpu.step()
        torch.testing.assert_close(cpu.positions, gpu.positions.cpu())


if __name__ == "__main__":
    unittest.main()
