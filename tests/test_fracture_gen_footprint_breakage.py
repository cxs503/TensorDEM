"""Complementary analytic verification of the restored fracture-field criteria.

``tests/test_fracture_gen_mass_footprint.py`` pins the *frozen* SI-2 contract
(mass deficit ``pi/6 - 1``, infinite-plane over-break, layer-coincident plane
degeneracy) on the reference ``n = 8`` lattice.  This module pins the
*replacement* contract that trunk actually runs, and does so against anchors
that are derived independently of the implementation:

1. ``mass_radius`` must be the radius of the sphere that carries
   ``particle_mass`` at the field density: ``4/3 pi r_m^3 rho == m_p``.  For
   ``mass_mode="cell"`` this forces ``r_m / r = 2 (3/4pi)^{1/3} = 1.2407``.
2. The default mapping must be volume consistent: ``M_DEM / (rho V) = 1`` to
   machine precision, versus ``pi/6`` for the frozen sphere mapping.
3. ``mass_mapping`` aliases must be exact synonyms of ``mass_mode``.
4. ``on_plane_tolerance`` may only change the *degenerate* case: a plane that
   passes between two layers must sever the same bonds with and without the
   tolerance (it must not silently start cutting coplanar bonds).
5. ``footprint_aware`` is a restriction, never an extension: for every radius
   the aware-bond set is a subset of the frozen set and is empty outside the
   footprint disc, including for an inclined face.
6. Energy accounting is decoupled from the breakage criterion: the released
   energy per broken bond is the face total divided by the number of bonds the
   chosen criterion actually severed.

The lattice used here is ``n = 4`` (r = 0.125 m, s = 0.25 m, domain (0,1)^3,
64 particles, 3*16*3 = 144 bonds) so the module stays cheap; the resolution
study belongs to the ``n = 8`` audit module.

Run: python -m pytest tests/test_fracture_gen_footprint_breakage.py -q
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tensordem.fracture_gen import (
    MASS_RADIUS_FACTOR,
    FailureFace,
    FractureGenConfig,
    FractureLedger,
    face_footprint_radius,
    generate_fracture_field,
    identify_bonds_crossing_face,
    normalise_mass_mapping,
)

RADIUS = 0.125
SPACING = 2.0 * RADIUS
DOMAIN = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)  # n = 4 layers per axis
N_LAYERS = 4
N_PARTICLES = N_LAYERS**3
N_BONDS = 3 * N_LAYERS**2 * (N_LAYERS - 1)  # 144
DENSITY = 917.0
CONTINUUM_VOLUME = (N_LAYERS * SPACING) ** 3  # 1.0 m^3, no quantisation error
PLANE_AREA = (N_LAYERS * SPACING) ** 2  # 1.0 m^2


def face(area=PLANE_AREA, z=0.5, extent_radius=None, normal=(0.0, 0.0, 1.0),
         centroid=None):
    if centroid is None:
        centroid = (0.5, 0.5, z)
    return FailureFace(
        id=0,
        area=area,
        normal=torch.tensor(normal, dtype=torch.float64),
        centroid=torch.tensor(centroid, dtype=torch.float64),
        G=100.0,
        released_energy=1.0,
        extent_radius=extent_radius,
    )


def field_for(faces, **config_kwargs):
    config = FractureGenConfig(
        density=DENSITY,
        particle_radius=RADIUS,
        domain_bounds=DOMAIN,
        **config_kwargs,
    )
    return generate_fracture_field(faces, config)


def broken_midpoints(field):
    i_idx, j_idx = field.bond_indices[~field.alive].T
    if len(i_idx) == 0:
        return torch.empty((0, 3), dtype=torch.float64)
    return 0.5 * (field.positions[i_idx] + field.positions[j_idx])


class TestMassRadiusAnchor(unittest.TestCase):
    def test_mass_radius_carries_particle_mass(self):
        """Independent anchor: 4/3 pi r_m^3 rho must reproduce particle_mass."""
        for mode in ("cell", "sphere"):
            f = field_for([face()], mass_mapping=mode)
            r_m = f.mass_radius
            volume = (4.0 / 3.0) * math.pi * r_m**3
            self.assertAlmostEqual(volume * DENSITY / f.particle_mass, 1.0, places=12,
                                   msg=f"mode={mode}")

    def test_cell_mass_radius_is_volume_equivalent_to_the_cell(self):
        f = field_for([face()], mass_mapping="cell")
        self.assertAlmostEqual(f.mass_radius / RADIUS, MASS_RADIUS_FACTOR, places=12)
        # closed form of the same number: r_eq = (3 h^3 / 4 pi)^{1/3}
        r_eq = (3.0 * SPACING**3 / (4.0 * math.pi)) ** (1.0 / 3.0)
        self.assertAlmostEqual(f.mass_radius / r_eq, 1.0, places=12)
        # contact radius stays r so neighbouring spheres still overlap-free
        self.assertLess(f.particle_radius, f.mass_radius)

    def test_sphere_mass_radius_is_the_contact_radius(self):
        f = field_for([face()], mass_mapping="sphere")
        self.assertEqual(f.mass_radius, RADIUS)


class TestDefaultMappingContract(unittest.TestCase):
    def test_trunk_default_is_volume_consistent(self):
        """Default field represents rho * V exactly (the frozen mapping does not)."""
        default = FractureLedger(field_for([face()]), [face()]).mass_accounting(
            CONTINUUM_VOLUME
        )
        frozen = FractureLedger(
            field_for([face()], mass_mapping="equal_volume_sphere"), [face()]
        ).mass_accounting(CONTINUUM_VOLUME)
        self.assertAlmostEqual(default["relative_error"], 0.0, places=12)
        self.assertTrue(default["within_15pct"])
        self.assertAlmostEqual(frozen["relative_error"], math.pi / 6 - 1, places=12)
        self.assertFalse(frozen["within_15pct"])
        self.assertAlmostEqual(default["dem_mass"] / frozen["dem_mass"],
                               1.0 / (math.pi / 6), places=12)

    def test_aliases_are_exact_synonyms(self):
        pairs = [
            ("lattice_cell", "cell"),
            ("lattice-cell", "cell"),
            ("volume_cell", "cell"),
            ("equal_volume_sphere", "sphere"),
            ("equal-volume-sphere", "sphere"),
            ("frozen", "sphere"),
        ]
        for alias, canonical in pairs:
            self.assertEqual(normalise_mass_mapping(alias), canonical)
        a = field_for([face()], mass_mapping="lattice_cell")
        b = field_for([face()], mass_mode="cell")
        self.assertEqual(a.total_mass, b.total_mass)
        self.assertEqual(a.mass_mode, "cell")

    def test_alias_conflict_resolved_in_favour_of_mass_mapping(self):
        f = field_for([face()], mass_mode="sphere", mass_mapping="lattice_cell")
        self.assertEqual(f.mass_mode, "cell")


class TestFootprintRadiusAnchor(unittest.TestCase):
    def test_equal_area_disc_radius(self):
        self.assertAlmostEqual(face_footprint_radius(face(area=1.0)), math.sqrt(1.0 / math.pi))
        self.assertAlmostEqual(face_footprint_radius(face(area=0.5, extent_radius=0.9)), 0.9)

    def test_invalid_footprints_rejected(self):
        with self.assertRaises(ValueError):
            face_footprint_radius(face(area=0.04, extent_radius=0.0))
        with self.assertRaises(ValueError):
            face_footprint_radius(face(area=0.04, extent_radius=-1.0))
        with self.assertRaises(ValueError):
            face_footprint_radius(face(area=0.0))
        with self.assertRaises(ValueError):
            face_footprint_radius(face(area=math.nan))


class TestOnPlaneToleranceScope(unittest.TestCase):
    def test_tolerance_does_not_change_a_non_coincident_plane(self):
        """z = 0.5 lies between layers: tolerance must not add or remove bonds."""
        strict = field_for([face(z=0.5)])
        tolerant = field_for([face(z=0.5)], on_plane_tolerance=1e-12)
        self.assertEqual(strict.num_broken_bonds, tolerant.num_broken_bonds)
        self.assertEqual(strict.num_broken_bonds, N_LAYERS**2)

    def test_tolerance_recovers_coincident_layer(self):
        strict = field_for([face(z=0.375)])  # 0.375 is a lattice layer
        tolerant = field_for([face(z=0.375)], on_plane_tolerance=1e-12)
        self.assertEqual(strict.num_broken_bonds, 0)
        self.assertEqual(tolerant.num_broken_bonds, 2 * N_LAYERS**2)

    def test_negative_tolerance_rejected_at_config_and_function_level(self):
        with self.assertRaises(ValueError):
            FractureGenConfig(density=DENSITY, particle_radius=RADIUS,
                              domain_bounds=DOMAIN, on_plane_tolerance=-1e-9)
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        bonds = torch.tensor([[0, 1]], dtype=torch.long)
        with self.assertRaises(ValueError):
            identify_bonds_crossing_face(bonds, positions, face(z=0.5),
                                         on_plane_tolerance=-1.0)


class TestFootprintIsARestriction(unittest.TestCase):
    def test_aware_is_subset_of_frozen(self):
        for radius in (0.125, 0.25, 0.375, 0.75):
            area = math.pi * radius**2
            frozen = field_for([face(area=area, extent_radius=radius)])
            aware = field_for([face(area=area, extent_radius=radius)],
                              footprint_aware=True)
            self.assertLessEqual(aware.num_broken_bonds, frozen.num_broken_bonds)
            self.assertGreaterEqual(aware.num_intact_bonds, frozen.num_intact_bonds)

    def test_aware_breaks_nothing_outside_the_disc(self):
        radius = 0.375
        f = field_for([face(area=math.pi * radius**2, extent_radius=radius)],
                      footprint_aware=True)
        mid = broken_midpoints(f)
        self.assertGreater(len(mid), 0)
        drift = torch.linalg.norm(mid[:, :2] - torch.tensor([0.5, 0.5],
                                   dtype=torch.float64), dim=1)
        self.assertTrue(bool((drift <= radius + 1e-12).all()))

    def test_aware_covers_whole_plane_when_footprint_exceeds_it(self):
        """A footprint larger than the cross-section degenerates to the frozen cut."""
        big = 10.0
        frozen = field_for([face(area=math.pi * big**2, extent_radius=big)])
        aware = field_for([face(area=math.pi * big**2, extent_radius=big)],
                          footprint_aware=True)
        self.assertEqual(aware.num_broken_bonds, frozen.num_broken_bonds)
        self.assertEqual(aware.num_broken_bonds, N_LAYERS**2)

    def test_inclined_face_footprint_is_measured_in_plane(self):
        """A 45-degree face with R < half-diagonal must not cut the far corner.

        The centroid is deliberately off the lattice-invariant position
        (0.5, 0.5, 0.5): a 45-degree plane through that point passes exactly
        through a particle layer everywhere, so only the degenerate on-plane
        case would be exercised.
        """
        radius = 0.25
        centroid = (0.5, 0.5, 0.5625)
        inclined = face(area=math.pi * radius**2, extent_radius=radius,
                        normal=(1.0, 0.0, 1.0), centroid=centroid)
        frozen = field_for([inclined])
        f = field_for([inclined], footprint_aware=True)
        mid = broken_midpoints(f)
        self.assertGreater(len(mid), 0)
        self.assertLess(f.num_broken_bonds, frozen.num_broken_bonds)
        unit = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        unit = unit / torch.linalg.norm(unit)
        offset = mid - torch.tensor(centroid, dtype=torch.float64)
        offset = offset - torch.sum(offset * unit, dim=1, keepdim=True) * unit
        drift = torch.linalg.norm(offset, dim=1)
        self.assertTrue(bool((drift <= radius + 1e-12).all()))


class TestEnergyDecoupledFromBreakage(unittest.TestCase):
    def test_unit_bond_energy_follows_the_severed_count(self):
        faces = [face(area=math.pi * 0.25**2, extent_radius=0.25)]
        frozen = FractureLedger(field_for(faces), faces).energy_accounting()
        aware = FractureLedger(
            field_for(faces, footprint_aware=True), faces
        ).energy_accounting()
        self.assertEqual(frozen["total_released_energy"], 1.0)
        self.assertEqual(aware["total_released_energy"], 1.0)
        for accounting in (frozen, aware):
            self.assertEqual(accounting["num_broken_bonds"] > 0, True)
            self.assertAlmostEqual(
                accounting["unit_bond_energy"] * accounting["num_broken_bonds"],
                1.0,
                places=12,
            )
        # the footprint criterion spends the same energy on fewer bonds
        self.assertGreater(aware["unit_bond_energy"], frozen["unit_bond_energy"])

    def test_ledger_rederivation_is_consistent_with_the_field(self):
        """check_bonds_at_faces must reuse the field's own criteria."""
        faces = [face(area=math.pi * 0.25**2, extent_radius=0.25)]
        for kwargs in ({}, {"footprint_aware": True}, {"on_plane_tolerance": 1e-12}):
            f = field_for(faces, **kwargs)
            audit = FractureLedger(f, faces).check_bonds_at_faces()
            self.assertTrue(audit["all_face_bonds_broken"], msg=str(kwargs))
            self.assertEqual(audit["bonds_crossing_faces"], f.num_broken_bonds)


if __name__ == "__main__":
    unittest.main()