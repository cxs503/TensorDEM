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

Reference lattice used below: radius r = 0.125 m, spacing s = 0.25 m, domain
(0, 2)^3 -> n = 8 layers per axis, N = 512 particles, 1344 nearest-neighbour
bonds (3 n^2 (n - 1)). All coordinates are exactly representable in binary, so
plane alignment is not decided by floating-point noise.

Mapping selection: the three defects above are properties of the *frozen* SI-2
contract, which trunk keeps available as ``mass_mapping="sphere"`` /
``"equal_volume_sphere"``. Trunk's default is the corrected volume-consistent
``mass_mode="cell"`` mapping (``docs/dem3d_fracture_mass_mapping_audit.md``),
so the two tests that assert the frozen π/6 - 1 deficit request that mapping
explicitly; every asserted number is unchanged. The complementary contract --
"the default mapping is volume consistent" -- is pinned in
``tests/test_fracture_gen_footprint_breakage.py``.
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
        # ``mass_mode="cell"`` is the repaired default on main; the frozen
        # equal-volume-sphere baseline is selected by name. The frozen mapping
        # is opt-in: trunk's default is the corrected volume-consistent cell
        # mapping (asserted separately below).
        f = field_for([face()], mass_mapping="sphere")
        ledger = FractureLedger(f, [face()])
        accounting = ledger.mass_accounting(CONTINUUM_VOLUME)
        self.assertAlmostEqual(accounting["relative_error"], math.pi / 6 - 1, places=12)
        self.assertAlmostEqual(accounting["relative_error"], -0.4764, places=4)
        # beyond the platform's 15 % acceptance bar -> must be reported as FAIL
        self.assertFalse(accounting["within_15pct"])

    def test_frozen_mapping_tolerance_against_its_own_formula_is_vacuous(self):
        """The prereg criterion 1 cannot detect the deficit it is named after."""
        f = field_for([face()], mass_mapping="sphere")
        ledger = FractureLedger(f, [face()])
        sphere_mass = N_PARTICLES * DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3
        self.assertAlmostEqual(ledger.mass_conservation_error(sphere_mass), 0.0, places=9)

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