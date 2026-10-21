"""Unit tests for dem_export module."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from tensordem.dem_export import (
    DemExport,
    Fragment,
    DEM_EXPORT_SCHEMA,
    load_json,
    load_to_tensors,
    save_json,
    serialize_fragments,
)


class FragmentTests(unittest.TestCase):
    """Tests for the Fragment dataclass."""

    def test_valid_2d(self):
        f = Fragment(id=0, centroid_m=[1.0, 2.0], mass_kg=1.5,
                     radius_m=0.1, velocity_m_s=[0.5, -0.3])
        self.assertEqual(f.id, 0)
        self.assertEqual(len(f.centroid_m), 2)

    def test_valid_3d(self):
        f = Fragment(id=1, centroid_m=[1.0, 2.0, 3.0], mass_kg=2.0,
                     radius_m=0.2, velocity_m_s=[0.1, 0.2, 0.3])
        self.assertEqual(len(f.centroid_m), 3)

    def test_invalid_mass(self):
        with self.assertRaises(ValueError):
            Fragment(id=0, centroid_m=[0.0, 0.0], mass_kg=-1.0,
                     radius_m=0.1, velocity_m_s=[0.0, 0.0])

    def test_invalid_radius(self):
        with self.assertRaises(ValueError):
            Fragment(id=0, centroid_m=[0.0, 0.0], mass_kg=1.0,
                     radius_m=0.0, velocity_m_s=[0.0, 0.0])

    def test_invalid_dimension(self):
        with self.assertRaises(ValueError):
            Fragment(id=0, centroid_m=[0.0], mass_kg=1.0,
                     radius_m=0.1, velocity_m_s=[0.0])

    def test_mismatched_dimensions(self):
        with self.assertRaises(ValueError):
            Fragment(id=0, centroid_m=[0.0, 0.0], mass_kg=1.0,
                     radius_m=0.1, velocity_m_s=[0.0, 0.0, 0.0])


class DemExportTests(unittest.TestCase):
    """Tests for the DemExport class."""

    def test_schema_constant(self):
        export = DemExport()
        self.assertEqual(export.schema, "tensordem.dem-export/1")

    def test_to_dict(self):
        f = Fragment(id=0, centroid_m=[1.0, 2.0], mass_kg=1.5,
                     radius_m=0.1, velocity_m_s=[0.5, -0.3])
        export = DemExport(fragments=[f])
        d = export.to_dict()
        self.assertEqual(d["schema"], "tensordem.dem-export/1")
        self.assertEqual(len(d["fragments"]), 1)
        self.assertEqual(d["fragments"][0]["id"], 0)

    def test_to_json_roundtrip(self):
        fragments = [
            Fragment(id=i, centroid_m=[float(i), float(i + 1)], mass_kg=1.0 + i,
                     radius_m=0.1 * (i + 1), velocity_m_s=[0.1 * i, 0.2 * i])
            for i in range(3)
        ]
        export = DemExport(fragments=fragments)
        text = export.to_json()
        loaded = DemExport.from_json(text)
        self.assertEqual(loaded.schema, export.schema)
        self.assertEqual(len(loaded.fragments), len(export.fragments))
        for a, b in zip(export.fragments, loaded.fragments):
            self.assertEqual(a.id, b.id)
            self.assertAlmostEqual(a.mass_kg, b.mass_kg)

    def test_from_dict_invalid_schema(self):
        with self.assertRaises(ValueError):
            DemExport.from_dict({"schema": "unknown", "fragments": []})


class SerializationTests(unittest.TestCase):
    """Tests for serialize_fragments and load_to_tensors."""

    def test_serialize_empty(self):
        export = serialize_fragments([], [], [], [])
        self.assertEqual(len(export.fragments), 0)

    def test_serialize_single(self):
        export = serialize_fragments(
            centroids=[[1.0, 2.0]],
            masses=[1.5],
            radii=[0.1],
            velocities=[[0.5, 0.0]],
        )
        self.assertEqual(len(export.fragments), 1)
        f = export.fragments[0]
        self.assertEqual(f.id, 0)
        self.assertAlmostEqual(f.mass_kg, 1.5)

    def test_serialize_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            serialize_fragments(
                centroids=[[1.0, 2.0]],
                masses=[1.5, 2.0],
                radii=[0.1],
                velocities=[[0.5, 0.0]],
            )

    def test_load_to_tensors_2d(self):
        fragments = [
            Fragment(id=0, centroid_m=[1.0, 2.0], mass_kg=1.0,
                     radius_m=0.1, velocity_m_s=[0.0, 0.0]),
            Fragment(id=1, centroid_m=[3.0, 4.0], mass_kg=2.0,
                     radius_m=0.2, velocity_m_s=[1.0, 0.0]),
        ]
        export = DemExport(fragments=fragments)
        tensors = load_to_tensors(export)
        self.assertEqual(tensors["positions"].shape, (2, 2))
        self.assertEqual(tensors["velocities"].shape, (2, 2))
        self.assertEqual(tensors["masses"].shape, (2,))
        self.assertEqual(tensors["radii"].shape, (2,))
        torch.testing.assert_close(
            tensors["positions"], torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        )
        torch.testing.assert_close(
            tensors["masses"], torch.tensor([1.0, 2.0], dtype=torch.float64)
        )

    def test_load_to_tensors_3d(self):
        fragments = [
            Fragment(id=0, centroid_m=[1.0, 2.0, 3.0], mass_kg=1.0,
                     radius_m=0.1, velocity_m_s=[0.0, 0.0, 0.0]),
        ]
        export = DemExport(fragments=fragments)
        tensors = load_to_tensors(export)
        self.assertEqual(tensors["positions"].shape, (1, 3))
        self.assertEqual(tensors["velocities"].shape, (1, 3))

    def test_load_to_tensors_empty(self):
        export = DemExport(fragments=[])
        tensors = load_to_tensors(export)
        self.assertEqual(tensors["positions"].shape, (0, 2))


class FileIOTests(unittest.TestCase):
    """Tests for save_json and load_json."""

    def test_save_and_load(self):
        fragments = [
            Fragment(id=0, centroid_m=[0.0, 0.0], mass_kg=1.0,
                     radius_m=0.1, velocity_m_s=[0.0, 0.0]),
        ]
        export = DemExport(fragments=fragments)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_export.json"
            save_json(export, path)
            loaded = load_json(path)
            self.assertEqual(len(loaded.fragments), 1)
            self.assertEqual(loaded.fragments[0].id, 0)

    def test_saved_json_validates_against_schema(self):
        fragments = [
            Fragment(id=i, centroid_m=[float(i), 0.0], mass_kg=1.0,
                     radius_m=0.1, velocity_m_s=[0.0, 0.0])
            for i in range(5)
        ]
        export = DemExport(fragments=fragments)
        text = export.to_json()
        data = json.loads(text)
        # Basic schema checks
        self.assertEqual(data["schema"], "tensordem.dem-export/1")
        self.assertIsInstance(data["fragments"], list)
        self.assertEqual(len(data["fragments"]), 5)
        for f in data["fragments"]:
            self.assertIn("id", f)
            self.assertIn("centroid_m", f)
            self.assertIn("mass_kg", f)
            self.assertIn("radius_m", f)
            self.assertIn("velocity_m_s", f)


if __name__ == "__main__":
    unittest.main()
