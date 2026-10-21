"""Tests for fracture generation from FEM failure faces."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from tensordem.fracture_gen import (
    FailureFace,
    FractureField,
    FractureGenConfig,
    FractureLedger,
    build_bond_topology,
    generate_fracture_field,
    generate_particle_lattice,
    identify_bonds_crossing_face,
    load_failure_faces,
)


class TestFailureFaceLoading(unittest.TestCase):
    """Test loading failure faces from JSON."""

    def test_load_valid_json(self):
        """Test loading a valid failure_faces.json."""
        data = {
            "faces": [
                {
                    "id": 0,
                    "area": 0.01,
                    "normal": [0.0, 0.0, 1.0],
                    "centroid": [0.0, 0.0, 0.5],
                    "G": 100.0,
                    "released_energy": 1.0,
                }
            ],
            "units": "SI",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            faces = load_failure_faces(f.name)

        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].id, 0)
        self.assertEqual(faces[0].area, 0.01)
        self.assertTrue(
            torch.allclose(faces[0].normal, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        )
        self.assertTrue(
            torch.allclose(faces[0].centroid, torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64))
        )
        self.assertEqual(faces[0].G, 100.0)
        self.assertEqual(faces[0].released_energy, 1.0)

        Path(f.name).unlink()

    def test_reject_non_si_units(self):
        """Test that non-SI units are rejected."""
        data = {"faces": [], "units": "CGS"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            with self.assertRaises(ValueError) as ctx:
                load_failure_faces(f.name)
            self.assertIn("Only SI units supported", str(ctx.exception))

            Path(f.name).unlink()


class TestParticleLatticeGeneration(unittest.TestCase):
    """Test particle lattice generation."""

    def test_simple_cubic_lattice(self):
        """Test that a simple cubic lattice is generated correctly."""
        domain = (0.0, 0.4, 0.0, 0.4, 0.0, 0.4)
        radius = 0.1
        positions = generate_particle_lattice(domain, radius)

        # Should have 2x2x2 = 8 particles
        self.assertEqual(len(positions), 8)

        # Check that particles are at expected positions
        # First particle should be at (0.1, 0.1, 0.1)
        expected_first = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        self.assertTrue(torch.allclose(positions[0], expected_first))

    def test_analytical_particle_count(self):
        """Test that particle count matches analytical expectation."""
        domain = (0.0, 0.6, 0.0, 0.6, 0.0, 0.6)
        radius = 0.1
        positions = generate_particle_lattice(domain, radius)

        # Spacing = 0.2, domain = 0.6, so 3 particles per direction
        # Total = 3x3x3 = 27
        self.assertEqual(len(positions), 27)


class TestBondTopology(unittest.TestCase):
    """Test bond topology construction."""

    def test_bonds_connect_nearest_neighbors(self):
        """Test that bonds connect nearest neighbors."""
        domain = (0.0, 0.4, 0.0, 0.4, 0.0, 0.4)
        radius = 0.1
        positions = generate_particle_lattice(domain, radius)
        bond_indices, rest_lengths = build_bond_topology(positions, radius)

        # All bonds should have length = 2*radius
        expected_length = 2.0 * radius
        self.assertTrue(
            torch.allclose(
                rest_lengths,
                torch.full_like(rest_lengths, expected_length),
                atol=1e-10,
            )
        )

    def test_bond_count_simple_cubic(self):
        """Test bond count for simple cubic lattice."""
        domain = (0.0, 0.4, 0.0, 0.4, 0.0, 0.4)
        radius = 0.1
        positions = generate_particle_lattice(domain, radius)
        bond_indices, _ = build_bond_topology(positions, radius)

        # 2x2x2 lattice: 12 edges in a cube
        self.assertEqual(len(bond_indices), 12)


class TestBondBreakage(unittest.TestCase):
    """Test bond breakage at failure faces."""

    def test_bonds_crossing_plane_are_broken(self):
        """Test that bonds crossing a failure plane are identified."""
        # Create a simple 2x2x2 lattice
        domain = (0.0, 0.4, 0.0, 0.4, 0.0, 0.4)
        radius = 0.1
        positions = generate_particle_lattice(domain, radius)
        bond_indices, _ = build_bond_topology(positions, radius)

        # Create a failure face at z=0.2 (middle of the domain)
        face = FailureFace(
            id=0,
            area=0.04,
            normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            centroid=torch.tensor([0.2, 0.2, 0.2], dtype=torch.float64),
            G=100.0,
            released_energy=1.0,
        )

        crosses = identify_bonds_crossing_face(bond_indices, positions, face)

        # Bonds crossing z=0.2 plane should be identified
        # In a 2x2x2 lattice, there are 4 vertical bonds crossing the middle
        self.assertEqual(crosses.sum().item(), 4)

    def test_bonds_parallel_to_face_not_broken(self):
        """Test that bonds parallel to a failure face are not broken."""
        domain = (0.0, 0.4, 0.0, 0.4, 0.0, 0.4)
        radius = 0.1
        positions = generate_particle_lattice(domain, radius)
        bond_indices, _ = build_bond_topology(positions, radius)

        # Create a failure face at x=0.2 (between particle layers at x=0.1 and x=0.3)
        face = FailureFace(
            id=0,
            area=0.04,
            normal=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            centroid=torch.tensor([0.2, 0.2, 0.2], dtype=torch.float64),
            G=100.0,
            released_energy=1.0,
        )

        crosses = identify_bonds_crossing_face(bond_indices, positions, face)

        # Bonds in x-direction cross this plane (from x=0.1 to x=0.3)
        # But we expect 4 bonds to cross (2x2 grid in y-z plane)
        self.assertEqual(crosses.sum().item(), 4)


class TestFractureFieldGeneration(unittest.TestCase):
    """Test complete fracture field generation."""

    def _make_field_and_faces(self):
        faces = [
            FailureFace(
                id=0,
                area=0.04,
                normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
                centroid=torch.tensor([0.3, 0.3, 0.3], dtype=torch.float64),
                G=100.0,
                released_energy=1.0,
            )
        ]
        config = FractureGenConfig(
            density=917.0,
            particle_radius=0.1,
            domain_bounds=(0.0, 0.6, 0.0, 0.6, 0.0, 0.6),
        )
        field = generate_fracture_field(faces, config)
        return field, faces, config

    def test_mass_conservation(self):
        """Test that total mass is conserved (criterion c1)."""
        field, faces, config = self._make_field_and_faces()

        # 3x3x3 = 27 particles
        expected_mass = (
            27 * 917.0 * (4.0 / 3.0) * math.pi * 0.1**3
        )

        mass_error = abs(field.total_mass - expected_mass)
        self.assertLess(
            mass_error,
            1e-12,
            f"Mass error {mass_error} exceeds tolerance 1e-12",
        )

    def test_energy_accounting(self):
        """Test energy accounting (criterion c2)."""
        faces = [
            FailureFace(
                id=0,
                area=0.04,
                normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
                centroid=torch.tensor([0.3, 0.3, 0.3], dtype=torch.float64),
                G=100.0,
                released_energy=1.0,
            ),
            FailureFace(
                id=1,
                area=0.04,
                normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
                centroid=torch.tensor([0.3, 0.3, 0.5], dtype=torch.float64),
                G=100.0,
                released_energy=2.0,
            ),
        ]
        config = FractureGenConfig(
            density=917.0,
            particle_radius=0.1,
            domain_bounds=(0.0, 0.6, 0.0, 0.6, 0.0, 0.6),
        )
        field = generate_fracture_field(faces, config)

        ledger = FractureLedger(field, faces)
        accounting = ledger.energy_accounting()

        self.assertAlmostEqual(
            accounting["total_released_energy"], 3.0, places=12
        )
        self.assertGreater(accounting["num_broken_bonds"], 0)
        expected_unit = 3.0 / accounting["num_broken_bonds"]
        self.assertAlmostEqual(
            accounting["unit_bond_energy"], expected_unit, places=12
        )

    def test_bonds_broken_at_faces(self):
        """Test that bonds at failure faces are broken (criterion c3)."""
        field, faces, config = self._make_field_and_faces()

        ledger = FractureLedger(field, faces)
        check = ledger.check_bonds_at_faces()

        self.assertTrue(
            check["all_face_bonds_broken"],
            "Not all bonds crossing failure faces are broken",
        )

    def test_intact_region_bonds_preserved(self):
        """Test that bonds in intact regions are preserved (criterion c3)."""
        field, faces, config = self._make_field_and_faces()

        self.assertGreater(
            field.num_intact_bonds,
            0,
            "All bonds are broken, but some should be intact",
        )

    def test_analytical_enumeration(self):
        """Test analytical enumeration (criterion c4)."""
        domain = (0.0, 0.6, 0.0, 0.6, 0.0, 0.6)
        radius = 0.1
        density = 917.0

        faces = []

        config = FractureGenConfig(
            density=density,
            particle_radius=radius,
            domain_bounds=domain,
        )
        field = generate_fracture_field(faces, config)

        expected_particles = 27
        expected_mass = (
            expected_particles * density * (4.0 / 3.0) * math.pi * radius**3
        )

        self.assertEqual(
            field.num_particles,
            expected_particles,
            f"Expected {expected_particles} particles, got {field.num_particles}",
        )

        mass_error = abs(field.total_mass - expected_mass)
        self.assertLess(
            mass_error,
            1e-12,
            f"Mass error {mass_error} exceeds tolerance 1e-12",
        )

        self.assertEqual(
            field.num_broken_bonds,
            0,
            "No failure faces, so no bonds should be broken",
        )


class TestFractureLedger(unittest.TestCase):
    """Test fracture ledger accounting."""

    def _make_field(self):
        domain = (0.0, 0.4, 0.0, 0.4, 0.0, 0.4)
        radius = 0.1
        density = 917.0

        positions = generate_particle_lattice(domain, radius)
        velocities = torch.zeros_like(positions)
        bond_indices, rest_lengths = build_bond_topology(positions, radius)
        alive = torch.ones(len(bond_indices), dtype=torch.bool)

        field = FractureField(
            positions=positions,
            velocities=velocities,
            bond_indices=bond_indices,
            rest_lengths=rest_lengths,
            alive=alive,
            particle_radius=radius,
            density=density,
            bond_stiffness=2000.0,
        )
        return field

    def test_mass_conservation_error(self):
        """Test mass conservation error computation."""
        field = self._make_field()
        ledger = FractureLedger(field, [])

        expected_mass = (
            8 * 917.0 * (4.0 / 3.0) * math.pi * 0.1**3
        )

        error = ledger.mass_conservation_error(expected_mass)
        self.assertLess(error, 1e-12)

    def test_energy_accounting_no_broken_bonds(self):
        """Test energy accounting when no bonds are broken."""
        field = self._make_field()

        ledger = FractureLedger(field, [])
        accounting = ledger.energy_accounting()

        self.assertEqual(accounting["total_released_energy"], 0.0)
        self.assertEqual(accounting["num_broken_bonds"], 0)
        self.assertEqual(accounting["unit_bond_energy"], 0.0)


class TestIntegrationWithIceDEM3D(unittest.TestCase):
    """Test integration with existing IceDEM3D engine."""

    def test_field_can_be_converted_to_initial_state(self):
        """Test that FractureField can be converted to IceDEM3D initial state."""
        faces = [
            FailureFace(
                id=0,
                area=0.04,
                normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
                centroid=torch.tensor([0.3, 0.3, 0.3], dtype=torch.float64),
                G=100.0,
                released_energy=1.0,
            )
        ]

        config = FractureGenConfig(
            density=917.0,
            particle_radius=0.1,
            domain_bounds=(0.0, 0.6, 0.0, 0.6, 0.0, 0.6),
        )
        field = generate_fracture_field(faces, config)

        initial_state = field.to_icedem3d_initial_state()

        for key in (
            "positions",
            "velocities",
            "bond_indices",
            "rest_lengths",
            "alive",
        ):
            self.assertIn(key, initial_state)

        self.assertEqual(initial_state["positions"].shape[1], 3)
        self.assertEqual(initial_state["velocities"].shape[1], 3)
        self.assertEqual(initial_state["bond_indices"].shape[1], 2)
        self.assertEqual(
            len(initial_state["rest_lengths"]),
            len(initial_state["bond_indices"]),
        )
        self.assertEqual(
            len(initial_state["alive"]),
            len(initial_state["bond_indices"]),
        )


if __name__ == "__main__":
    unittest.main()
