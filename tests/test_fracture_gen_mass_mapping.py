"""Analytic validation of the FEM->DEM fracture-field mass/volume mapping.

Every assertion here is checked against a closed-form geometric reference, not
against the generator's own output:

* simple-cubic packing fraction of equal spheres  V_sphere/V_cell = pi/6,
* volume-equivalent radius   (4/3) pi r_eq^3 = h^3   =>   r_eq = 0.6203505 h,
* lattice occupancy   N(W) = floor((W - r)/h) + 1,
* nearest-neighbour bond count of a free simple-cubic lattice
  Nx Ny (Nz-1) + Nx (Ny-1) Nz + (Nx-1) Ny Nz.
"""
from __future__ import annotations

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
    generate_fracture_field,
    generate_particle_lattice,
)
from tensordem.fracture_mass_mapping import (
    SPHERE_MAPPING_DEFICIT,
    analytic_bond_count,
    domain_volume,
    equal_volume_radius,
    expected_lattice_counts,
    lattice_axis_count,
    lattice_cell_volume,
    lattice_spacing,
    mass_mapping_report,
    particle_cell_volume,
    particle_mass,
    particle_sphere_volume,
    resolution_sweep,
    sphere_to_cell_volume_ratio,
    volume_quantisation_error,
)

RHO = 917.0
R = 0.1
DOMAIN = (0.0, 0.6, 0.0, 0.6, 0.0, 0.6)


def _field(mass_mode: str = "cell", domain=DOMAIN, radius: float = R,
           faces: list[FailureFace] | None = None) -> FractureField:
    config = FractureGenConfig(
        density=RHO,
        particle_radius=radius,
        domain_bounds=domain,
        mass_mode=mass_mode,
    )
    return generate_fracture_field(faces or [], config)


class SimpleCubicGeometryTests(unittest.TestCase):
    def test_packing_fraction_is_pi_over_six(self):
        """Inscribed sphere / cube of side 2r is exactly pi/6."""
        r = 0.37
        self.assertAlmostEqual(
            particle_sphere_volume(r) / particle_cell_volume(r), math.pi / 6.0, places=14
        )
        self.assertAlmostEqual(sphere_to_cell_volume_ratio(), math.pi / 6.0, places=15)
        self.assertAlmostEqual(SPHERE_MAPPING_DEFICIT, 1.0 - math.pi / 6.0, places=15)
        # documented headline number
        self.assertAlmostEqual(100.0 * SPHERE_MAPPING_DEFICIT, 47.6401, places=4)

    def test_lattice_spacing_is_two_radii(self):
        self.assertEqual(lattice_spacing(0.125), 0.25)
        self.assertAlmostEqual(particle_cell_volume(0.125), 0.25 ** 3, places=15)

    def test_equal_volume_radius_matches_cell_volume(self):
        """The volume-equivalent sphere radius reproduces the cell volume."""
        for radius in (0.05, 0.1, 0.25, 1.0):
            h = lattice_spacing(radius)
            r_eq = equal_volume_radius(h)
            self.assertAlmostEqual(
                (4.0 / 3.0) * math.pi * r_eq ** 3 / particle_cell_volume(radius),
                1.0,
                places=12,
            )
            # 0.62035 h, i.e. 1.2407 r
            self.assertAlmostEqual(r_eq / h, (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0), places=12)

    def test_axis_count_matches_generator(self):
        """N(W) = floor((W - r)/h) + 1 reproduces generate_particle_lattice."""
        for radius, widths in (
            (0.1, (0.6, 0.65, 0.7, 0.55, 0.2, 0.1)),
            (0.05, (0.5, 0.55, 0.8, 0.6)),
            (0.25, (1.0, 1.1, 1.25)),
        ):
            for width in widths:
                positions = generate_particle_lattice((0.0, width, 0.0, width, 0.0, width), radius)
                realised = len(torch.unique(positions[:, 0]))
                self.assertEqual(
                    lattice_axis_count(width, radius),
                    realised,
                    f"radius={radius} width={width}",
                )

    def test_too_small_domain_is_rejected(self):
        """A domain thinner than one radius cannot hold a particle row."""
        with self.assertRaisesRegex(ValueError, "at least one particle radius"):
            generate_particle_lattice((0.0, 0.05, 0.0, 0.6, 0.0, 0.6), 0.1)
        self.assertEqual(lattice_axis_count(0.05, 0.1), 0)

    def test_bond_count_matches_analytic_simple_cubic(self):
        """Nearest-neighbour bonds equal the closed-form simple-cubic count."""
        for domain, radius in (
            ((0.0, 0.6, 0.0, 0.6, 0.0, 0.6), 0.1),
            ((0.0, 0.4, 0.0, 0.6, 0.0, 0.8), 0.1),
            ((0.0, 0.3, 0.0, 0.3, 0.0, 0.3), 0.05),
        ):
            field = _field("cell", domain=domain, radius=radius)
            counts = expected_lattice_counts(domain, radius)
            self.assertEqual(field.num_particles, counts[0] * counts[1] * counts[2])
            self.assertEqual(len(field.bond_indices), analytic_bond_count(counts))


class MassMappingTests(unittest.TestCase):
    def test_legacy_sphere_mapping_reproduces_pi_over_six_deficit(self):
        """The pre-audit mapping reproduces exactly V_sphere/V_cell of the ice."""
        field = _field("sphere")
        fem_mass = RHO * domain_volume(DOMAIN)
        self.assertAlmostEqual(field.total_mass / fem_mass, math.pi / 6.0, places=12)
        self.assertAlmostEqual(
            field.total_mass / fem_mass - 1.0, -SPHERE_MAPPING_DEFICIT, places=12
        )

    def test_cell_mapping_conserves_mass_on_integer_multiple_domain(self):
        """mass_mode='cell' gives M_DEM = rho * V_domain to machine precision."""
        field = _field("cell")
        fem_mass = RHO * domain_volume(DOMAIN)
        self.assertAlmostEqual(field.represented_volume, domain_volume(DOMAIN), places=15)
        self.assertAlmostEqual(field.total_mass / fem_mass, 1.0, places=13)

    def test_mass_mapping_report_gates(self):
        """1 % gate: cell mode passes, legacy sphere mode fails by 47.6 %."""
        cell = mass_mapping_report(RHO, R, DOMAIN, mass_mode="cell", tolerance=0.01)
        self.assertTrue(cell["within_tolerance"])
        self.assertLess(abs(cell["relative_mass_error"]), 1e-12)
        self.assertLess(abs(cell["lattice_quantisation_relative_error"]), 1e-12)
        self.assertLess(abs(cell["material_mapping_relative_error"]), 1e-12)
        self.assertEqual(cell["num_particles"], 27)
        self.assertEqual(cell["reference_source"], "density*domain_volume")

        legacy = mass_mapping_report(RHO, R, DOMAIN, mass_mode="sphere", tolerance=0.01)
        self.assertFalse(legacy["within_tolerance"])
        self.assertAlmostEqual(legacy["relative_mass_error"], math.pi / 6.0 - 1.0, places=12)
        self.assertAlmostEqual(
            legacy["material_mapping_relative_error"], math.pi / 6.0 - 1.0, places=12
        )
        # the two mappings differ by exactly one factor of pi/6
        self.assertAlmostEqual(
            cell["dem_total_mass_kg"] / legacy["dem_total_mass_kg"], 6.0 / math.pi, places=12
        )

    def test_quantisation_error_is_isolated_from_material_error(self):
        """Non-multiple domain: error splits into quantisation x material terms."""
        domain = (0.0, 0.65, 0.0, 0.65, 0.0, 0.65)  # W/h = 3.25
        counts = expected_lattice_counts(domain, R)
        self.assertEqual(counts, (3, 3, 3))
        analytic_q = (lattice_cell_volume(domain, R) - domain_volume(domain)) / domain_volume(domain)
        self.assertAlmostEqual(analytic_q, (0.6 ** 3 - 0.65 ** 3) / 0.65 ** 3, places=14)
        self.assertAlmostEqual(volume_quantisation_error(domain, R), analytic_q, places=14)
        # realised lattice volume from the generator matches the analytic term
        field = _field("cell", domain=domain)
        realised = field.represented_volume / domain_volume(domain) - 1.0
        self.assertAlmostEqual(realised, analytic_q, places=13)
        report = mass_mapping_report(RHO, R, domain, mass_mode="cell", tolerance=0.01)
        self.assertAlmostEqual(
            report["lattice_quantisation_relative_error"], analytic_q, places=14
        )
        self.assertAlmostEqual(report["relative_mass_error"], analytic_q, places=13)
        # quantisation alone exceeds a 1 % gate, so this case must fail loudly
        self.assertFalse(report["within_tolerance"])

    def test_fem_mass_reference_is_honoured(self):
        """A supplied FEM mass overrides density*volume and is reported as such."""
        field = _field("cell")
        fem_mass = 1.02 * RHO * domain_volume(DOMAIN)  # +2 % FEM-side mass
        ledger = FractureLedger(field, [])
        report = ledger.mass_mapping_report(fem_mass=fem_mass, tolerance=0.01)
        self.assertEqual(report["reference_source"], "fem_mass")
        # M_DEM / (1.02 M_domain) - 1 = 1/1.02 - 1 = -0.0196078
        self.assertAlmostEqual(report["relative_mass_error"], 1.0 / 1.02 - 1.0, places=12)
        self.assertFalse(report["within_tolerance"])
        loose = ledger.mass_mapping_report(fem_mass=fem_mass, tolerance=0.05)
        self.assertTrue(loose["within_tolerance"])

    def test_ledger_report_infers_domain_from_particles(self):
        """Without domain bounds the ledger expands the centre bbox by r."""
        field = _field("cell")
        ledger = FractureLedger(field, [])
        inferred = ledger.mass_mapping_report(tolerance=0.01)
        # centres span 0.4 m; adding one radius per side recovers the 0.6 m
        # lattice extent, so the inferred reference volume is exactly V_domain
        self.assertAlmostEqual(inferred["domain_volume_m3"], 0.6 ** 3, places=12)
        self.assertEqual(inferred["num_particles_source"], "caller")
        self.assertAlmostEqual(inferred["relative_mass_error"], 0.0, places=12)
        self.assertTrue(inferred["within_tolerance"])
        # the same inference path exposes the legacy deficit unchanged
        legacy = FractureLedger(_field("sphere"), []).mass_mapping_report(tolerance=0.01)
        self.assertAlmostEqual(legacy["relative_mass_error"], math.pi / 6.0 - 1.0, places=12)
        self.assertFalse(legacy["within_tolerance"])

    def test_invalid_mass_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "mass_mode"):
            FractureGenConfig(density=RHO, particle_radius=R, domain_bounds=DOMAIN, mass_mode="bogus")
        with self.assertRaisesRegex(ValueError, "mass_mode"):
            particle_mass(RHO, R, "bogus")
        with self.assertRaisesRegex(ValueError, "mass_mode"):
            FractureField(
                positions=torch.zeros(1, 3, dtype=torch.float64),
                velocities=torch.zeros(1, 3, dtype=torch.float64),
                bond_indices=torch.empty((0, 2), dtype=torch.long),
                rest_lengths=torch.empty(0, dtype=torch.float64),
                alive=torch.empty(0, dtype=torch.bool),
                particle_radius=R,
                density=RHO,
                bond_stiffness=2000.0,
                mass_mode="bogus",
            )

    def test_resolution_sweep_isolates_material_term(self):
        """Independent of resolution: cell maps exactly, sphere loses pi/6."""
        sweep = resolution_sweep(RHO, DOMAIN, particles_per_axis=(2, 3, 4))
        self.assertEqual(len(sweep["cases"]), 3)
        for row in sweep["cases"]:
            self.assertLess(abs(row["cell_mode_relative_error"]), 1e-12)
            self.assertTrue(row["cell_mode_within_tolerance"])
            self.assertAlmostEqual(
                row["sphere_mode_relative_error"], math.pi / 6.0 - 1.0, places=12
            )
            self.assertFalse(row["sphere_mode_within_tolerance"])
            self.assertLess(abs(row["lattice_quantisation_relative_error"]), 1e-12)
            self.assertEqual(row["num_particles"], row["particles_per_axis"] ** 3)


class FractureFieldStillFeedsIceDEM3DTests(unittest.TestCase):
    def test_mass_mode_propagates_through_config(self):
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
        cell = _field("cell", faces=faces)
        legacy = _field("sphere", faces=faces)
        self.assertEqual(cell.mass_mode, "cell")
        self.assertEqual(legacy.mass_mode, "sphere")
        # fracture topology is unaffected by the mass mapping
        self.assertEqual(cell.num_broken_bonds, legacy.num_broken_bonds)
        self.assertGreater(cell.num_broken_bonds, 0)
        self.assertAlmostEqual(
            cell.total_mass / legacy.total_mass, 6.0 / math.pi, places=12
        )
        # initial-state contract for IceDEM3D is unchanged
        state = cell.to_icedem3d_initial_state()
        self.assertEqual(
            sorted(state),
            ["alive", "bond_indices", "positions", "rest_lengths", "velocities"],
        )


class AuditScriptTests(unittest.TestCase):
    """The audit artifact generator must reproduce the analytic anchors."""

    def test_audit_artifacts_match_analytic_anchors(self):
        import json

        from scripts.audit_fracture_mass_mapping import main as audit_main

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                audit_main(
                    ["--output", tmp, "--particles-per-axis", "2", "3", "--quantisation-points", "5"]
                ),
                0,
            )
            out = Path(tmp)
            report = json.loads(
                (out / "fracture_mass_mapping_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["protocol"], "tensordem-fracture-mass-mapping-audit-v1"
            )
            anchors = report["analytic_anchors"]
            self.assertAlmostEqual(
                anchors["simple_cubic_packing_fraction_pi_over_6"], math.pi / 6.0, places=15
            )
            self.assertAlmostEqual(
                anchors["legacy_sphere_mapping_deficit"], 1.0 - math.pi / 6.0, places=15
            )
            # frozen SI-2 reference case: cell within 1 %, legacy at -47.64 %
            replay = report["frozen_si2_reference_case"]
            self.assertLess(abs(replay["relative_error"]["cell"]), 1e-12)
            self.assertAlmostEqual(
                replay["relative_error"]["sphere"], math.pi / 6.0 - 1.0, places=12
            )
            # mass mapping must not perturb the fracture topology
            self.assertEqual(replay["cell"]["num_bonds"], replay["sphere"]["num_bonds"])
            self.assertEqual(
                replay["cell"]["num_bonds"],
                analytics_bond_count_for_domain(),
            )
            # every swept resolution keeps the material term at zero
            for row in report["resolution_sweep"]:
                self.assertLess(abs(row["cell_mode_relative_error"]), 1e-12)
                self.assertAlmostEqual(
                    row["sphere_mode_relative_error"], math.pi / 6.0 - 1.0, places=12
                )
            # quantisation term is a pure function of the domain extent
            for row in report["quantisation_sweep"]:
                expected = (row["covered_extent_m"] / row["extent_m"]) ** 3 - 1.0
                self.assertAlmostEqual(
                    row["quantisation_relative_error"], expected, places=12
                )
            self.assertEqual(len(report["quantisation_sweep"]), 5)
            self.assertEqual(
                report["ship_scale_analytic"]["lattice_counts"], [50, 50, 4]
            )
            self.assertTrue((out / "fracture_mass_mapping_sweep.csv").is_file())
            self.assertTrue((out / report["figure"]).is_file())


def analytics_bond_count_for_domain() -> int:
    return analytic_bond_count(expected_lattice_counts(DOMAIN, R))


if __name__ == "__main__":
    unittest.main()