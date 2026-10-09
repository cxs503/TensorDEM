import math
import unittest
import torch
from tensordem.dem3d import DEM3DConfig, IceDEM3D


class DEM3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, **kwargs):
        return IceDEM3D(DEM3DConfig(nx=3, ny=3, nz=2, drag=0.0,
                                    fix_x_edges=False, **kwargs))

    def test_initial_equilibrium_and_force_balance(self):
        sim = self.model()
        force, reaction = sim.forces()
        self.assertTrue(torch.isfinite(force).all())
        torch.testing.assert_close(force.sum(0), -reaction, atol=1e-10, rtol=0)
        self.assertLess(float(force.abs().max()), 1e-8)

    def test_rigid_rotation_does_not_break_bonds(self):
        sim = self.model()
        a, b = 0.41, -0.27
        rx = torch.tensor([[1., 0., 0.], [0., math.cos(a), -math.sin(a)],
                           [0., math.sin(a), math.cos(a)]], dtype=torch.float64)
        ry = torch.tensor([[math.cos(b), 0., math.sin(b)], [0., 1., 0.],
                           [-math.sin(b), 0., math.cos(b)]], dtype=torch.float64)
        sim.positions = sim.initial_positions @ (rx @ ry).T + torch.tensor([2., -3., 4.])
        force, _ = sim.forces()
        self.assertEqual(sim.broken_bonds, 0)
        self.assertLess(float(force.abs().max()), 1e-7)

    def test_tension_failure_is_irreversible(self):
        sim = self.model(breaking_strain=0.01, shear_breaking_strain=0.5)
        sim.positions = sim.initial_positions * 1.03
        sim.forces()
        self.assertGreater(sim.broken_bonds, 0)
        broken = sim.broken_bonds
        sim.positions = sim.initial_positions.clone()
        sim.forces()
        self.assertEqual(sim.broken_bonds, broken)

    def test_external_load_shape_and_finiteness(self):
        sim = self.model()
        with self.assertRaises(ValueError):
            sim.forces(torch.zeros((len(sim.positions), 2)))
        loads = torch.zeros_like(sim.positions)
        loads[0, 0] = 1.0
        force, _ = sim.forces(loads)
        self.assertTrue(torch.isfinite(force).all())
        with self.assertRaises(ValueError):
            loads[0, 0] = float("nan")
            sim.forces(loads)

    def test_step_fixed_boundary_and_finite_state(self):
        sim = IceDEM3D(DEM3DConfig(nx=4, ny=3, nz=2))
        fixed0 = sim.positions[sim.fixed].clone()
        for _ in range(20):
            sim.step()
        torch.testing.assert_close(sim.positions[sim.fixed], fixed0, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(sim.positions).all())
        self.assertEqual(sim.step_count, 20)

    def test_restart_snapshot_and_validation(self):
        sim = self.model()
        snap = sim.state_dict()
        self.assertEqual(snap["dimension"], 3)
        self.assertEqual(tuple(snap["positions"].shape), (18, 3))
        with self.assertRaises(ValueError):
            DEM3DConfig(nx=1)
        with self.assertRaises(ValueError):
            DEM3DConfig(dt=math.nan)


    def test_checkpoint_round_trip_continues_identically(self):
        first = self.model()
        for _ in range(5):
            first.step()
        snapshot = first.state_dict()
        for _ in range(5):
            first.step()

        resumed = self.model()
        resumed.load_state_dict(snapshot)
        for _ in range(5):
            resumed.step()
        torch.testing.assert_close(resumed.positions, first.positions, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(resumed.velocities, first.velocities, atol=1e-12, rtol=1e-12)
        self.assertEqual(resumed.step_count, first.step_count)
        self.assertEqual(resumed.time, first.time)
        self.assertTrue(torch.equal(resumed.alive, first.alive))

    def test_checkpoint_rejects_incompatible_or_corrupt_state(self):
        sim = self.model()
        snapshot = sim.state_dict()
        broken = dict(snapshot)
        broken["positions"] = torch.zeros((1, 3))
        with self.assertRaises(ValueError):
            sim.load_state_dict(broken)
        incompatible = IceDEM3D(DEM3DConfig(nx=4, ny=3, nz=2, drag=0.0,
                                             fix_x_edges=False))
        with self.assertRaises(ValueError):
            incompatible.load_state_dict(snapshot)


if __name__ == "__main__":
    unittest.main()
