import unittest
import torch
from tensordem.hull_contact import polyline_contact_forces


class PolylineHullContactTests(unittest.TestCase):
    def test_known_force_and_action_reaction(self):
        p = torch.tensor([[0.0, 0.04], [0.5, 0.20]], dtype=torch.float64)
        v = torch.zeros_like(p)
        hull = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        force, reaction = polyline_contact_forces(
            p, v, hull, torch.zeros(2, dtype=torch.float64),
            particle_radius=0.05, hull_radius=0.0, stiffness=1000.0)
        torch.testing.assert_close(force[0], torch.tensor([0.0, 10.0], dtype=torch.float64))
        torch.testing.assert_close(force[1], torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(force.sum(dim=0) + reaction, torch.zeros(2, dtype=torch.float64))

    def test_endpoint_contact(self):
        p = torch.tensor([[1.03, 0.0]], dtype=torch.float64)
        force, _ = polyline_contact_forces(
            p, torch.zeros_like(p),
            torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
            particle_radius=0.05, hull_radius=0.0, stiffness=100.0)
        self.assertGreater(float(force[0, 0]), 0.0)
        self.assertAlmostEqual(float(force[0, 1]), 0.0, places=12)

    def test_damping_does_not_create_attraction(self):
        p = torch.tensor([[0.0, 0.04]], dtype=torch.float64)
        v = torch.tensor([[0.0, 10.0]], dtype=torch.float64)
        force, _ = polyline_contact_forces(
            p, v, torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64), particle_radius=0.05,
            hull_radius=0.0, stiffness=100.0, damping=100.0)
        torch.testing.assert_close(force, torch.zeros_like(force))

    def test_rejects_degenerate_segments_and_bad_shapes(self):
        p = torch.zeros((1, 2), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "zero-length"):
            polyline_contact_forces(
                p, p, torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
                torch.zeros(2, dtype=torch.float64), particle_radius=0.05,
                hull_radius=0.0, stiffness=100.0)
        with self.assertRaisesRegex(ValueError, "same shape"):
            polyline_contact_forces(
                p, torch.zeros((2, 2), dtype=torch.float64),
                torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
                torch.zeros(2, dtype=torch.float64), particle_radius=0.05,
                hull_radius=0.0, stiffness=100.0)


if __name__ == "__main__":
    unittest.main()
