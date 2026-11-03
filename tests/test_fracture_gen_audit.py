"""Independent verification audit tests for the SI-2/DEM fracture generator.

These tests replace the self-referential acceptance criteria of
``prereg/fracture_gen_prereg.json`` (c1 compared the DEM mass with its own
formula, c2 multiplied a defined ratio back) with falsifiable lattice
analytics. See ``docs/dem3d_fracture_gen_audit.md`` and
``benchmarks/fracture_gen_audit.py``.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

# This checkout is exercised in place: the ambient `tensordem` install may point
# at another worktree, so import the local sources first and drop any already
# imported copy that came from outside this repository.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
for _name in [n for n in sys.modules if n == "tensordem" or n.startswith("tensordem.")]:
    _file = getattr(sys.modules[_name], "__file__", "") or ""
    if not _file.startswith(str(SRC)):
        del sys.modules[_name]

import torch

from tensordem.fracture_gen import (
    FailureFace,
    FractureGenConfig,
    FractureLedger,
    build_bond_topology,
    generate_fracture_field,
    generate_particle_lattice,
)


def _face(area: float, centroid, G: float = 500.0, normal=(0.0, 0.0, 1.0)) -> FailureFace:
    return FailureFace(
        id=0,
        area=area,
        normal=torch.tensor(normal, dtype=torch.float64),
        centroid=torch.tensor(centroid, dtype=torch.float64),
        G=G,
        released_energy=G * area,
    )


def _brute_force_bonds(positions: torch.Tensor, spacing: float) -> list[list[int]]:
    pairs = []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if float(torch.linalg.norm(positions[j] - positions[i])) <= spacing * 1.01:
                pairs.append([i, j])
    return pairs


class LatticeEnumerationAudit(unittest.TestCase):
    """c4: closed-form enumeration of a simple cubic lattice."""

    def test_bond_count_matches_3n2n1(self):
        for n in (2, 3, 4):
            side = 0.2 * n
            positions = generate_particle_lattice((0.0, side, 0.0, side, 0.0, side), 0.1)
            bonds, rest = build_bond_topology(positions, 0.1)
            self.assertEqual(len(positions), n**3)
            self.assertEqual(len(bonds), 3 * n * n * (n - 1))
            self.assertEqual(len(rest), len(bonds))
            self.assertTrue(torch.allclose(rest, torch.full_like(rest, 0.2)))

    def test_vectorised_topology_matches_brute_force(self):
        positions = generate_particle_lattice((0.0, 0.4, 0.0, 0.4, 0.0, 0.4), 0.1)
        bonds, _ = build_bond_topology(positions, 0.1)
        reference = _brute_force_bonds(positions, 0.2)
        self.assertEqual(bonds.tolist(), reference)


class MassMappingAudit(unittest.TestCase):
    """c1: DEM mass must be compared with the FEM continuum mass rho*V."""

    def setUp(self):
        self.radius = 0.1
        self.side = 0.6
        self.density = 917.0
        self.bounds = (0.0, self.side, 0.0, self.side, 0.0, self.side)

    def _field(self, mapping: str):
        cfg = FractureGenConfig(
            density=self.density,
            particle_radius=self.radius,
            domain_bounds=self.bounds,
            mass_mapping=mapping,
        )
        return generate_fracture_field([], cfg)

    def test_equal_volume_spheres_deficit_is_pi_over_6(self):
        field = self._field("equal_volume_spheres")
        fem_mass = self.density * self.side**3
        rel = abs(field.total_mass - fem_mass) / fem_mass
        self.assertAlmostEqual(rel, 1.0 - math.pi / 6.0, places=12)
        # The frozen mapping loses 47.6% of the continuum mass: far above the
        # 15% acceptance band, so criterion c1 fails against rho*V.
        self.assertGreater(rel, 0.15)

    def test_cell_volume_mapping_reproduces_continuum_mass(self):
        field = self._field("cell_volume")
        fem_mass = self.density * self.side**3
        rel = abs(field.total_mass - fem_mass) / fem_mass
        self.assertLess(rel, 1e-12)
        self.assertEqual(field.volume_fraction, 1.0)
        self.assertAlmostEqual(field.particle_mass, self.density * (2 * self.radius) ** 3)

    def test_lattice_quantisation_is_reported(self):
        # L = 0.5 m is not a multiple of 2r = 0.2 m: the lattice covers only
        # 0.4 m per axis, so even the cell_volume mapping cannot conserve mass.
        cfg = FractureGenConfig(
            density=self.density,
            particle_radius=self.radius,
            domain_bounds=(0.0, 0.5, 0.0, 0.5, 0.0, 0.5),
            mass_mapping="cell_volume",
        )
        field = generate_fracture_field([], cfg)
        acc = FractureLedger(field, []).mass_accounting(domain_volume=0.5**3)
        self.assertEqual(acc["num_particles"], 8)
        self.assertAlmostEqual(acc["lattice_volume_covered"], 0.4**3)
        # relative_error is signed (DEM - reference)/reference; quantisation
        # only ever loses volume here, so compare magnitudes.
        self.assertAlmostEqual(abs(acc["relative_error"]), abs(0.4**3 - 0.5**3) / 0.5**3)

    def test_mass_accounting_reports_mapping_name(self):
        acc = FractureLedger(self._field("cell_volume"), []).mass_accounting()
        self.assertEqual(acc["mass_mapping"], "cell_volume")
        for key in ("fem_continuum_mass", "dem_mass", "relative_error"):
            self.assertIn(key, acc)


class FaceExtentAudit(unittest.TestCase):
    """c2/c3: bond count and fracture energy per bond vs A/(2r)^2 and G(2r)^2."""

    def setUp(self):
        self.radius = 0.05
        self.n_axis = 10
        self.side = self.n_axis * 2 * self.radius  # 1.0 m
        self.G = 500.0
        self.spacing_sq = (2 * self.radius) ** 2
        self.bounds = (0.0, self.side, 0.0, self.side, 0.0, self.side)

    def _run(self, area: float, extent: str):
        face = _face(area, (self.side / 2,) * 3, G=self.G)
        cfg = FractureGenConfig(
            density=917.0,
            particle_radius=self.radius,
            domain_bounds=self.bounds,
            face_extent=extent,
            mass_mapping="cell_volume",
        )
        field = generate_fracture_field([face], cfg)
        return face, field, FractureLedger(field, [face]).energy_accounting()

    def test_plane_extent_breaks_full_cross_section(self):
        # An infinite plane ignores the declared face area: it breaks the full
        # lattice cross-section. That is exact only when A equals it.
        _, field, acc = self._run(self.side**2, "plane")
        self.assertEqual(field.num_broken_bonds, 100)
        self.assertLess(acc["unit_bond_energy_relative_error"], 1e-12)

    def test_plane_extent_overbreaks_small_faces(self):
        # A 0.04 m^2 face on a 1.0 m cross-section is over-counted 25x, so the
        # implied energy per bond is 96% too low: the shipped c2 check cannot
        # detect this because it is an algebraic identity.
        _, field, acc = self._run(0.04, "plane")
        self.assertEqual(field.num_broken_bonds, 100)
        self.assertAlmostEqual(acc["analytic_bond_count"], 4.0)
        self.assertGreater(acc["unit_bond_energy_relative_error"], 0.9)

    def test_disc_extent_tracks_declared_area(self):
        # Disc extent is area consistent: exactly 4 bonds for A = 4*(2r)^2.
        _, field, acc = self._run(4 * self.spacing_sq, "disc")
        self.assertEqual(field.num_broken_bonds, 4)
        self.assertLess(acc["unit_bond_energy_relative_error"], 1e-12)

    def test_disc_never_breaks_more_than_plane(self):
        for area in (0.04, 0.25, 1.0):
            _, plane_field, _ = self._run(area, "plane")
            _, disc_field, _ = self._run(area, "disc")
            self.assertLessEqual(disc_field.num_broken_bonds, plane_field.num_broken_bonds)

    def test_area_only_contract_is_under_determined(self):
        # The contract carries only area+centroid, so the bond count cannot be
        # resolved better than O(1) per face: a circular disc under-covers the
        # corners of a square face of the same area (88 vs 100 bonds, 12%).
        _, field, acc = self._run(self.side**2, "disc")
        self.assertEqual(field.num_broken_bonds, 88)
        self.assertAlmostEqual(
            acc["bond_count_relative_error"], abs(88 - 100) / 100, places=12
        )

    def test_energy_accounting_exposes_falsifiable_keys(self):
        face, field, acc = self._run(0.25, "disc")
        for key in (
            "analytic_bond_count",
            "analytic_unit_bond_energy",
            "unit_bond_energy_relative_error",
            "input_consistency_error",
            "bond_count_relative_error",
        ):
            self.assertIn(key, acc)
        self.assertAlmostEqual(acc["analytic_unit_bond_energy"], self.G * self.spacing_sq)
        self.assertAlmostEqual(acc["input_consistency_error"], 0.0, places=12)
        self.assertEqual(acc["face_extent"], "disc")

    def test_legacy_fixture_input_inconsistency_is_flagged(self):
        # benchmarks/fracture_gen_ledger.py ships G=100, area=0.04 but
        # released_energy=1.0, i.e. 75% below G*area = 4.0 J.
        face = FailureFace(
            id=0,
            area=0.04,
            normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            centroid=torch.tensor([0.3, 0.3, 0.3], dtype=torch.float64),
            G=100.0,
            released_energy=1.0,
        )
        cfg = FractureGenConfig(
            density=917.0, particle_radius=0.1, domain_bounds=(0.0, 0.6, 0.0, 0.6, 0.0, 0.6)
        )
        acc = FractureLedger(generate_fracture_field([face], cfg), [face]).energy_accounting()
        self.assertAlmostEqual(acc["input_consistency_error"], 3.0, places=12)

    def test_in_plane_translation_keeps_bond_count(self):
        counts = set()
        for shift in (0.0, 0.05, 0.1):
            face = _face(0.36, (0.3 + shift, 0.3, 0.3), G=100.0)
            cfg = FractureGenConfig(
                density=917.0,
                particle_radius=0.1,
                domain_bounds=(0.0, 0.6, 0.0, 0.6, 0.0, 0.6),
                face_extent="plane",
            )
            counts.add(generate_fracture_field([face], cfg).num_broken_bonds)
        self.assertEqual(counts, {9})


class ConfigValidationAudit(unittest.TestCase):
    def test_unknown_mass_mapping_rejected(self):
        with self.assertRaises(ValueError):
            FractureGenConfig(mass_mapping="lumped")
        with self.assertRaises(ValueError):
            FractureGenConfig(face_extent="box")

    def test_degenerate_normal_rejected(self):
        from tensordem.fracture_gen import identify_bonds_crossing_face

        face = _face(0.04, (0.3, 0.3, 0.3), normal=(0.0, 0.0, 0.0))
        positions = generate_particle_lattice((0.0, 0.4, 0.0, 0.4, 0.0, 0.4), 0.1)
        bonds, _ = build_bond_topology(positions, 0.1)
        with self.assertRaises(ValueError):
            identify_bonds_crossing_face(bonds, positions, face, face_extent="disc")


if __name__ == "__main__":
    unittest.main()