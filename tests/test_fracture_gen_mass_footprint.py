"""Validation-debt audit tests for FEM-failure-face -> DEM ice-field generation.

Three defects of the frozen SI-2/DEM contract are pinned against independent
analytic baselines:

1. Mass mapping: the frozen equal-volume-sphere mapping yields
   ``density * N * (pi/6) s^3`` for a simple-cubic lattice of spacing ``s``,
   i.e. exactly ``pi/6 - 1 = -47.64%`` of the continuum solid mass. The frozen
   acceptance criterion only compared the DEM mass with the same sphere formula
   and therefore could not see it. ``mass_mapping="lattice_cell"`` removes it.
2. Face footprint: the frozen plane-intersection criterion cuts every bond
   crossing the *infinite* plane, so a small face over-breaks by the ratio of
   plane area to footprint area. ``footprint_aware=True`` limits breakage to the
   footprint disc; the surviving count matches ``pi R^2 / s^2`` and the exact
   lattice enumeration.
3. Layer-coincident plane: with the strict sign test, a face plane passing
   through a particle layer severs nothing (0 % of the analytic count).
   ``on_plane_tolerance > 0`` restores the expected ``2 n^2`` bonds.

Default-mapping note (supersession): this module originally assumed the
legacy equal-volume-sphere mapping was the *default*, with the corrected
``lattice_cell`` mapping opt-in.  The later merged audit
(``docs/dem3d_fracture_mass_mapping_audit.md``) made the volume-consistent
mapping the default, because a -47.64 % mass deficit is a first-order physics
bug, not a preference.  The legacy-deficit assertions below are therefore
selected explicitly with ``mass_mapping="equal_volume_sphere"`` (they are *not*
weakened: the deficit is still pinned to 1e-12), and
``test_default_mapping_is_volume_consistent`` pins the default so the decision
cannot silently regress again.

Reference lattice used below: radius r = 0.125 m, spacing s = 0.25 m, domain
(0, 2)^3 -> n = 8 layers per axis, N = 512 particles, 1344 nearest-neighbour
bonds (3 n^2 (n - 1)). All coordinates are exactly representable in binary, so
plane alignment is not decided by floating-point noise.
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
    build_bond_topology,
    face_footprint_radius,
    generate_fracture_field,
    generate_particle_lattice,
    identify_bonds_crossing_face,
)

RADIUS = 0.125
SPACING = 2.0 * RADIUS
DOMAIN = (0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
N_LAYERS = 8
N_PARTICLES = N_LAYERS**3
N_BONDS = 3 * N_LAYERS**2 * (N_LAYERS - 1)
DENSITY = 917.0
CONTINUUM_VOLUME = (N_LAYERS * SPACING) ** 3  # matches the lattice exactly
PLANE_AREA = (N_LAYERS * SPACING) ** 2


def face(area=PLANE_AREA, z=0.5, extent_radius=None):
    return FailureFace(
        id=0,
        area=area,
        normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        centroid=torch.tensor([1.0, 1.0, z], dtype=torch.float64),
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


def enumerated_footprint_count(radius, z=0.5):
    """Exact count of lattice columns inside a disc of ``radius`` (m)."""
    coords = torch.arange(
        RADIUS, DOMAIN[1], SPACING, dtype=torch.float64
    )
    count = 0
    for x in coords:
        for y in coords:
            if math.hypot(float(x) - 1.0, float(y) - 1.0) <= radius:
                count += 1
    return count


class TestMassMappingAudit(unittest.TestCase):
    def test_lattice_and_bond_counts_match_analytic(self):
        f = field_for([face()])
        self.assertEqual(f.num_particles, N_PARTICLES)
        self.assertEqual(f.bond_indices.shape[0], N_BONDS)

    def test_frozen_mapping_deficit_is_exactly_pi_over_six(self):
        f = field_for([face()], mass_mapping="equal_volume_sphere")
        ledger = FractureLedger(f, [face()])
        accounting = ledger.mass_accounting(CONTINUUM_VOLUME)
        self.assertAlmostEqual(accounting["relative_error"], math.pi / 6 - 1, places=12)
        self.assertAlmostEqual(accounting["relative_error"], -0.4764, places=4)
        # beyond the platform's 15 % acceptance bar -> must be reported as FAIL
        self.assertFalse(accounting["within_15pct"])

    def test_frozen_mapping_tolerance_against_its_own_formula_is_vacuous(self):
        """The prereg criterion 1 cannot detect the deficit it is named after."""
        f = field_for([face()], mass_mapping="equal_volume_sphere")
        ledger = FractureLedger(f, [face()])
        sphere_mass = N_PARTICLES * DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3
        self.assertAlmostEqual(ledger.mass_conservation_error(sphere_mass), 0.0, places=9)

    def test_default_mapping_is_volume_consistent(self):
        """Trunk default must stay the corrected mapping (anti-regression pin).

        The frozen contract compared the DEM mass with the sphere formula used
        to build it; that is why the -47.64 % deficit survived review.  If a
        future edit reverts the default, this test fails instead of the deficit
        silently returning.
        """
        self.assertEqual(FractureGenConfig().mass_mode, "cell")
        default = field_for([face()])
        accounting = FractureLedger(default, [face()]).mass_accounting(
            CONTINUUM_VOLUME
        )
        self.assertTrue(accounting["within_15pct"])
        self.assertAlmostEqual(accounting["relative_error"], 0.0, places=12)
        self.assertAlmostEqual(default.mass_radius, RADIUS * MASS_RADIUS_FACTOR, places=12)
        # both declared vocabularies are accepted and land on the same mode
        self.assertEqual(
            field_for([face()], mass_mapping="lattice_cell").mass_mode, "cell"
        )
        self.assertEqual(
            field_for([face()], mass_mapping="equal_volume_sphere").mass_mode, "sphere"
        )

    def test_lattice_cell_mapping_hits_continuum_mass(self):
        f = field_for([face()], mass_mapping="lattice_cell")
        ledger = FractureLedger(f, [face()])
        accounting = ledger.mass_accounting(CONTINUUM_VOLUME)
        self.assertTrue(accounting["within_15pct"])
        self.assertAlmostEqual(accounting["relative_error"], 0.0, places=12)
        # mass radius, not geometric radius, carries the correction
        self.assertAlmostEqual(f.mass_radius / RADIUS, MASS_RADIUS_FACTOR, places=12)
        self.assertAlmostEqual(f.particle_mass, DENSITY * SPACING**3, places=12)

    def test_mass_mapping_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            field_for([face()], mass_mapping="unit_solid_volume")

    def test_mass_accounting_rejects_non_positive_volume(self):
        ledger = FractureLedger(field_for([face()]), [face()])
        with self.assertRaises(ValueError):
            ledger.mass_accounting(0.0)


class TestFootprintAwareBreakage(unittest.TestCase):
    def test_full_plane_crossing_count_equals_area_over_spacing_squared(self):
        f = field_for([face()])
        expected = PLANE_AREA / SPACING**2
        self.assertEqual(expected, N_LAYERS**2)
        self.assertEqual(f.num_broken_bonds, int(expected))

    def test_frozen_criterion_over_breaks_small_face(self):
        radius = 0.5
        area = math.pi * radius**2
        frozen = field_for([face(area=area, extent_radius=radius)])
        aware = field_for(
            [face(area=area, extent_radius=radius)], footprint_aware=True
        )
        # infinite-plane cut severs the whole cross-section regardless of area
        self.assertEqual(frozen.num_broken_bonds, N_LAYERS**2)
        exact = enumerated_footprint_count(radius)
        self.assertEqual(aware.num_broken_bonds, exact)
        self.assertGreater(
            frozen.num_broken_bonds / aware.num_broken_bonds, 4.0,
            "expected a large over-break factor for a small finite face",
        )
        # intact bonds must be restored (no mass of the field silently severed)
        self.assertGreater(aware.num_intact_bonds, frozen.num_intact_bonds)

    def test_footprint_count_follows_pi_r_squared_over_s_squared(self):
        for radius in (0.5, 0.75, 1.0):
            area = math.pi * radius**2
            f = field_for([face(area=area, extent_radius=radius)], footprint_aware=True)
            analytic = area / SPACING**2
            rel = (f.num_broken_bonds - analytic) / analytic
            self.assertLess(
                abs(rel), 0.15,
                f"radius {radius}: {f.num_broken_bonds} vs analytic {analytic:.2f}",
            )

    def test_footprint_radius_derived_from_area_when_extent_absent(self):
        self.assertAlmostEqual(face_footprint_radius(face(area=0.04)), math.sqrt(0.04 / math.pi))
        self.assertEqual(face_footprint_radius(face(area=0.04, extent_radius=0.3)), 0.3)
        with self.assertRaises(ValueError):
            face_footprint_radius(face(area=0.04, extent_radius=-1.0))

    def test_offset_face_footprint_breaks_only_local_bonds(self):
        """A face near the domain edge must not cut the far side of the plane."""
        off = FailureFace(
            id=1,
            area=math.pi * 0.5**2,
            normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            centroid=torch.tensor([0.375, 0.375, 0.5], dtype=torch.float64),
            G=100.0,
            released_energy=1.0,
            extent_radius=0.5,
        )
        f = field_for([off], footprint_aware=True)
        i_idx, j_idx = f.bond_indices[~f.alive].T
        broken = f.positions[torch.stack([i_idx, j_idx], dim=1)]
        plane_pts = broken.mean(dim=1)
        drift = torch.linalg.norm(plane_pts[:, :2] - off.centroid[:2].to(plane_pts.dtype), dim=1)
        self.assertTrue(bool((drift <= 0.5 + 1e-12).all()))


class TestLayerCoincidentPlane(unittest.TestCase):
    def test_strict_criterion_severs_nothing_on_lattice_layer(self):
        """0.625 m is an exact lattice coordinate: the cut is invisible."""
        f = field_for([face(z=0.625)], mass_mapping="lattice_cell")
        self.assertEqual(f.num_broken_bonds, 0)
        self.assertEqual(f.num_intact_bonds, N_BONDS)

    def test_on_plane_tolerance_restores_two_bonds_per_column(self):
        f = field_for([face(z=0.625)], on_plane_tolerance=1e-12)
        self.assertEqual(f.num_broken_bonds, 2 * N_LAYERS**2)
        self.assertEqual(f.num_broken_bonds, 128)

    def test_on_plane_tolerance_keeps_coplanar_bonds(self):
        """Bonds lying inside the plane (both endpoints on it) stay intact."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.5], [0.25, 0.0, 0.5], [0.0, 0.0, 0.75]],
            dtype=torch.float64,
        )
        bonds = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
        cut = identify_bonds_crossing_face(bonds, positions, face(z=0.5), on_plane_tolerance=1e-12)
        self.assertEqual(cut.tolist(), [False, True])

    def test_negative_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            field_for([face()], on_plane_tolerance=-1e-9)


class TestTopologyAndCriterionWiring(unittest.TestCase):
    """Regression pins for the parts of the contract that are easy to break
    while keeping the audit numbers green."""

    def test_vectorised_topology_matches_brute_force(self):
        """The blocked distance-matrix search must equal the pairwise loop.

        ``build_bond_topology`` was an O(N^2) Python loop (6.8 s for 512
        particles); it is now a row-blocked ``cdist`` search.  Bond *order* is
        part of the contract (regression artefacts index into it), so the
        brute-force reference is compared element-wise, not as a set.
        """
        positions = generate_particle_lattice((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), RADIUS)
        self.assertEqual(len(positions), 4**3)
        spacing = 2.0 * RADIUS
        pairs = [
            [i, j]
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
            if float(torch.linalg.norm(positions[j] - positions[i])) <= spacing * 1.01
        ]
        expected = torch.tensor(pairs, dtype=torch.long)
        actual, rest = build_bond_topology(positions, RADIUS)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(rest.shape[0], expected.shape[0])

        # the reference lattice of this module stays at the analytic count
        big, _ = build_bond_topology(
            generate_particle_lattice(DOMAIN, RADIUS), RADIUS
        )
        self.assertEqual(big.shape[0], N_BONDS)

    def test_degenerate_lattices_return_empty_topology(self):
        for n in (0, 1):
            positions = torch.zeros((n, 3), dtype=torch.float64)
            bonds, rest = build_bond_topology(positions, RADIUS)
            self.assertEqual(bonds.shape, (0, 2))
            self.assertEqual(rest.shape, (0,))

    def test_ledger_reidentifies_with_the_configured_criterion(self):
        """``check_bonds_at_faces`` must not re-cut with a stricter criterion.

        With ``footprint_aware=True`` the field breaks a subset of the bonds the
        infinite-plane criterion would break; if the ledger re-identified with
        the frozen criterion it would report a mismatch and silently invalidate
        every footprint-aware audit.
        """
        radius = 0.5
        faces = [face(area=math.pi * radius**2, extent_radius=radius)]
        aware = field_for(faces, footprint_aware=True)
        check = FractureLedger(aware, faces).check_bonds_at_faces()
        self.assertTrue(check["all_face_bonds_broken"])
        self.assertEqual(check["bonds_crossing_faces"], aware.num_broken_bonds)
        self.assertLess(check["bonds_crossing_faces"], N_LAYERS**2)


class TestAuditReproducibility(unittest.TestCase):
    def test_audit_script_numbers_match_module(self):
        """The published audit table is reproducible from the module API."""
        script = Path(__file__).resolve().parent.parent / "benchmarks" / "fracture_gen_mass_footprint_audit.py"
        self.assertTrue(script.exists(), "audit script must stay in the repo")
        source = script.read_text()
        self.assertIn("def run_audit", source)
        namespace: dict[str, object] = {"__file__": str(script), "__name__": "audit_probe"}
        exec(compile(source, str(script), "exec"), namespace)
        audit = namespace["run_audit"](write_artifacts=False)
        self.assertEqual(audit["lattice"]["n_particles"], N_PARTICLES)
        self.assertEqual(audit["lattice"]["n_bonds"], N_BONDS)
        self.assertEqual(
            audit["plane_cut"]["broken_bonds_full_plane"], N_LAYERS**2
        )
        self.assertAlmostEqual(
            audit["mass"]["frozen_relative_error"], math.pi / 6 - 1, places=12
        )
        self.assertTrue(audit["mass"]["lattice_cell_within_15pct"])


if __name__ == "__main__":
    unittest.main()