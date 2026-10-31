"""Tests for 3-D triangle-mesh hull contact (STL/OBJ).

Acceptance criteria (SIR-PC2 / m611):
  - sphere_contact_force_error < 0.1  (ratio vs analytical penalty solution)
  - contact_detection_complexity <= O(N)
  - pytest_test_hull_mesh_contact == PASS
"""
from __future__ import annotations

import math
import os
import struct
import tempfile
import time
import unittest

import torch

from tensordem.hull_mesh_contact import (
    load_obj,
    load_stl,
    make_flat_mesh,
    mesh_contact_forces,
)


class TestFlatPlaneAnalytical(unittest.TestCase):
    """Sphere pressing into a flat triangulated plane – analytical penalty solution."""

    def test_sphere_on_flat_plane_force_magnitude(self):
        """A sphere of radius R at height h < R above a z=0 plane should
        experience F = k * (R - h) in the +z direction (penalty model)."""
        R = 0.05
        h = 0.03  # overlap = R - h = 0.02
        k = 10000.0
        verts, faces = make_flat_mesh(size=2.0, normal="z", offset=0.0, nx=4, ny=4)
        pos = torch.tensor([[0.0, 0.0, h]], dtype=torch.float64)
        vel = torch.zeros(1, 3, dtype=torch.float64)
        forces, reaction = mesh_contact_forces(
            pos, vel, verts, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=R, stiffness=k,
        )
        expected_fz = k * (R - h)  # 10000 * 0.02 = 200 N
        fz = float(forces[0, 2].item())
        error_ratio = abs(fz - expected_fz) / expected_fz
        self.assertLess(error_ratio, 0.1,
                        f"Force error ratio {error_ratio:.4f} >= 0.1 "
                        f"(got {fz:.4f}, expected {expected_fz:.4f})")
        # Action-reaction
        torch.testing.assert_close(forces.sum(dim=0) + reaction,
                                   torch.zeros(3, dtype=torch.float64),
                                   atol=1e-10, rtol=0)

    def test_no_contact_when_separated(self):
        """Sphere above the plane with no overlap → zero force."""
        R = 0.05
        h = 0.06  # no overlap
        verts, faces = make_flat_mesh(size=2.0, normal="z", offset=0.0, nx=2, ny=2)
        pos = torch.tensor([[0.0, 0.0, h]], dtype=torch.float64)
        vel = torch.zeros(1, 3, dtype=torch.float64)
        forces, _ = mesh_contact_forces(
            pos, vel, verts, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=R, stiffness=10000.0,
        )
        self.assertAlmostEqual(float(forces.abs().sum().item()), 0.0, places=12)

    def test_force_direction_is_normal_to_plane(self):
        """Contact force on a flat plane should be purely normal (no tangential)."""
        R = 0.05
        h = 0.04
        verts, faces = make_flat_mesh(size=2.0, normal="z", offset=0.0, nx=4, ny=4)
        pos = torch.tensor([[0.3, -0.2, h]], dtype=torch.float64)
        vel = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float64)
        forces, _ = mesh_contact_forces(
            pos, vel, verts, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=R, stiffness=10000.0, damping=0.0,
        )
        # Only z-component should be nonzero
        self.assertAlmostEqual(float(forces[0, 0].item()), 0.0, places=10)
        self.assertAlmostEqual(float(forces[0, 1].item()), 0.0, places=10)
        self.assertGreater(float(forces[0, 2].item()), 0.0)


class TestDamping(unittest.TestCase):
    def test_damping_reduces_force_on_approach(self):
        """Approaching sphere should have larger force due to damping."""
        R = 0.05
        h = 0.04
        verts, faces = make_flat_mesh(size=2.0, normal="z", offset=0.0, nx=2, ny=2)
        pos = torch.tensor([[0.0, 0.0, h]], dtype=torch.float64)
        vel_down = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        vel_zero = torch.zeros(1, 3, dtype=torch.float64)
        f_damp, _ = mesh_contact_forces(
            pos, vel_down, verts, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=R, stiffness=10000.0, damping=100.0,
        )
        f_nodamp, _ = mesh_contact_forces(
            pos, vel_zero, verts, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=R, stiffness=10000.0, damping=0.0,
        )
        self.assertGreater(float(f_damp[0, 2].item()), float(f_nodamp[0, 2].item()))

    def test_damping_no_attraction_on_separation(self):
        """Separating sphere with damping should not create attractive force."""
        R = 0.05
        h = 0.04
        verts, faces = make_flat_mesh(size=2.0, normal="z", offset=0.0, nx=2, ny=2)
        pos = torch.tensor([[0.0, 0.0, h]], dtype=torch.float64)
        vel_up = torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float64)
        forces, _ = mesh_contact_forces(
            pos, vel_up, verts, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=R, stiffness=10000.0, damping=100.0,
        )
        # Force should still be repulsive (positive z) or zero, never attractive
        self.assertGreaterEqual(float(forces[0, 2].item()), 0.0)


class TestEdgeAndVertexContact(unittest.TestCase):
    def test_sphere_near_triangle_edge(self):
        """Sphere close to an edge but outside the triangle should still detect contact."""
        R = 0.05
        # Single triangle in the xz-plane
        verts = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 1.0],
        ], dtype=torch.float64)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        # Sphere near the edge (0,0,0)-(1,0,0), slightly below y=0
        pos = torch.tensor([[0.5, -0.03, 0.0]], dtype=torch.float64)
        vel = torch.zeros(1, 3, dtype=torch.float64)
        forces, _ = mesh_contact_forces(
            pos, vel, verts, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=R, stiffness=10000.0,
        )
        # Should detect contact (overlap = R - 0.03 = 0.02)
        # Force should be in -y direction (pushing sphere away from triangle)
        self.assertLess(float(forces[0, 1].item()), 0.0)

    def test_sphere_near_triangle_vertex(self):
        """Sphere near a vertex should detect contact."""
        R = 0.05
        verts = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 1.0],
        ], dtype=torch.float64)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        # Sphere near vertex (0,0,0)
        pos = torch.tensor([[-0.03, 0.0, 0.0]], dtype=torch.float64)
        vel = torch.zeros(1, 3, dtype=torch.float64)
        forces, _ = mesh_contact_forces(
            pos, vel, verts, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=R, stiffness=10000.0,
        )
        # Force should be in -x direction (pushing sphere away from vertex)
        self.assertLess(float(forces[0, 0].item()), 0.0)


class TestActionReaction(unittest.TestCase):
    def test_multiple_spheres_action_reaction(self):
        """Total force on spheres + reaction on hull = 0 (Newton's 3rd law)."""
        R = 0.05
        verts, faces = make_flat_mesh(size=2.0, normal="z", offset=0.0, nx=4, ny=4)
        pos = torch.tensor([
            [0.0, 0.0, 0.03],
            [0.3, 0.2, 0.04],
            [-0.2, 0.1, 0.02],
        ], dtype=torch.float64)
        vel = torch.zeros(3, 3, dtype=torch.float64)
        forces, reaction = mesh_contact_forces(
            pos, vel, verts, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=R, stiffness=10000.0,
        )
        torch.testing.assert_close(
            forces.sum(dim=0) + reaction,
            torch.zeros(3, dtype=torch.float64),
            atol=1e-10, rtol=0,
        )


class TestSTLAndOBJLoading(unittest.TestCase):
    def test_load_binary_stl(self):
        """Write a minimal binary STL and read it back."""
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            # Header (80 bytes)
            f.write(b"\x00" * 80)
            # 1 triangle
            f.write(struct.pack("<I", 1))
            # Normal
            f.write(struct.pack("<fff", 0.0, 0.0, 1.0))
            # 3 vertices
            f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
            f.write(struct.pack("<fff", 1.0, 0.0, 0.0))
            f.write(struct.pack("<fff", 0.5, 1.0, 0.0))
            # Attribute byte count
            f.write(struct.pack("<H", 0))
            f.flush()
            verts, faces = load_stl(f.name)
        os.unlink(f.name)
        self.assertEqual(verts.shape, (3, 3))
        self.assertEqual(faces.shape, (1, 3))

    def test_load_ascii_stl(self):
        """Write a minimal ASCII STL and read it back."""
        stl_text = """solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0.5 1 0
    endloop
  endfacet
endsolid test
"""
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False, mode="w") as f:
            f.write(stl_text)
            f.flush()
            verts, faces = load_stl(f.name)
        os.unlink(f.name)
        self.assertEqual(verts.shape, (3, 3))
        self.assertEqual(faces.shape, (1, 3))

    def test_load_obj(self):
        """Write a minimal OBJ and read it back."""
        obj_text = """# test
v 0 0 0
v 1 0 0
v 0.5 1 0
v 0.5 0.5 1
f 1 2 3
f 1 2 4
"""
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w") as f:
            f.write(obj_text)
            f.flush()
            verts, faces = load_obj(f.name)
        os.unlink(f.name)
        self.assertEqual(verts.shape, (4, 3))
        self.assertEqual(faces.shape, (2, 3))


class TestComplexity(unittest.TestCase):
    def test_linear_scaling(self):
        """Contact detection should scale linearly with number of spheres.

        We measure wall-clock time for N and 2N spheres; the ratio should
        be < 3 (allowing for constant overhead), confirming O(N) behaviour.
        """
        verts, faces = make_flat_mesh(size=10.0, normal="z", offset=0.0, nx=10, ny=10)
        torch.manual_seed(42)

        def run(n_spheres):
            pos = torch.rand(n_spheres, 3, dtype=torch.float64)
            pos[:, 0] = (pos[:, 0] - 0.5) * 8.0
            pos[:, 1] = (pos[:, 1] - 0.5) * 8.0
            pos[:, 2] = 0.03  # slight overlap
            vel = torch.zeros_like(pos)
            t0 = time.perf_counter()
            mesh_contact_forces(
                pos, vel, verts, faces,
                torch.zeros(3, dtype=torch.float64),
                particle_radius=0.05, stiffness=10000.0,
            )
            return time.perf_counter() - t0

        # Warm up
        run(100)

        n1 = 500
        n2 = 2000
        t1 = run(n1)
        t2 = run(n2)
        ratio = t2 / max(t1, 1e-9)
        # For O(N), 4x more spheres should take ~4x time. Allow factor 2 for noise.
        expected_ratio = n2 / n1  # 4x
        self.assertLess(ratio, expected_ratio * 2.0,
                        f"Scaling ratio {ratio:.2f} suggests worse than O(N)")


class TestForceErrorCriterion(unittest.TestCase):
    """Explicit acceptance criterion: sphere_contact_force_error < 0.1."""

    def test_force_error_below_threshold(self):
        """Compare mesh contact force against analytical penalty solution
        for multiple configurations."""
        R = 0.05
        k = 10000.0
        errors = []
        for h in [0.01, 0.02, 0.03, 0.04, 0.049]:
            verts, faces = make_flat_mesh(size=2.0, normal="z", offset=0.0, nx=8, ny=8)
            pos = torch.tensor([[0.0, 0.0, h]], dtype=torch.float64)
            vel = torch.zeros(1, 3, dtype=torch.float64)
            forces, _ = mesh_contact_forces(
                pos, vel, verts, faces,
                torch.zeros(3, dtype=torch.float64),
                particle_radius=R, stiffness=k,
            )
            expected = k * (R - h)
            got = float(forces[0, 2].item())
            err = abs(got - expected) / max(expected, 1e-15)
            errors.append(err)
        max_err = max(errors)
        self.assertLess(max_err, 0.1,
                        f"Max force error {max_err:.6f} >= 0.1")


class TestHullMeshContactPytest(unittest.TestCase):
    """Integration test matching the acceptance criterion name."""

    def test_pytest_test_hull_mesh_contact(self):
        """End-to-end: sphere on a bowl-shaped mesh (curved hull)."""
        # Create a simple bowl shape (hemisphere)
        n_theta = 16
        n_phi = 8
        radius_hull = 0.5
        verts_list = []
        # Bottom vertex
        verts_list.append([0.0, 0.0, -radius_hull])
        for j in range(1, n_phi + 1):
            phi = math.pi / 2 * j / n_phi  # 0 to pi/2 (bottom hemisphere)
            for i in range(n_theta):
                theta = 2 * math.pi * i / n_theta
                x = radius_hull * math.sin(phi) * math.cos(theta)
                y = radius_hull * math.sin(phi) * math.sin(theta)
                z = -radius_hull * math.cos(phi)
                verts_list.append([x, y, z])
        verts = torch.tensor(verts_list, dtype=torch.float64)
        faces_list = []
        # Bottom cap triangles
        for i in range(n_theta):
            i_next = (i + 1) % n_theta
            faces_list.append([0, 1 + i, 1 + i_next])
        # Ring triangles
        for j in range(n_phi - 1):
            for i in range(n_theta):
                i_next = (i + 1) % n_theta
                a = 1 + j * n_theta + i
                b = 1 + j * n_theta + i_next
                c = 1 + (j + 1) * n_theta + i
                d = 1 + (j + 1) * n_theta + i_next
                faces_list.append([a, b, d])
                faces_list.append([a, d, c])
        faces = torch.tensor(faces_list, dtype=torch.int64)

        # Place a sphere touching the bottom of the bowl
        R = 0.05
        sphere_pos = torch.tensor([[0.0, 0.0, -radius_hull + R * 0.6]], dtype=torch.float64)
        vel = torch.zeros(1, 3, dtype=torch.float64)
        forces, reaction = mesh_contact_forces(
            sphere_pos, vel, verts, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=R, stiffness=10000.0,
        )
        # Force should be upward (pushing sphere away from hull)
        self.assertGreater(float(forces[0, 2].item()), 0.0)
        # Action-reaction
        torch.testing.assert_close(
            forces.sum(dim=0) + reaction,
            torch.zeros(3, dtype=torch.float64),
            atol=1e-10, rtol=0,
        )


class TestInputValidation(unittest.TestCase):
    def test_bad_shapes(self):
        with self.assertRaises(ValueError):
            mesh_contact_forces(
                torch.zeros(1, 2, dtype=torch.float64),  # wrong dim
                torch.zeros(1, 2, dtype=torch.float64),
                torch.zeros(3, 3, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.int64),
                torch.zeros(3, dtype=torch.float64),
                particle_radius=0.05, stiffness=1000.0,
            )

    def test_bad_stiffness(self):
        verts, faces = make_flat_mesh()
        with self.assertRaises(ValueError):
            mesh_contact_forces(
                torch.zeros(1, 3, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
                verts, faces,
                torch.zeros(3, dtype=torch.float64),
                particle_radius=0.05, stiffness=-1.0,
            )

    def test_empty_input(self):
        verts, faces = make_flat_mesh()
        forces, reaction = mesh_contact_forces(
            torch.zeros(0, 3, dtype=torch.float64),
            torch.zeros(0, 3, dtype=torch.float64),
            verts, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=0.05, stiffness=1000.0,
        )
        self.assertEqual(forces.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
