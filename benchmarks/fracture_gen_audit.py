"""Independent verification audit of the SI-2/DEM fracture-field generator.

The Increment 2.5 acceptance criteria frozen in
``prereg/fracture_gen_prereg.json`` cannot fail as written:

* c1 "mass conservation" compares the DEM particle sum with a hand-computed
  product of the *same* sphere formula (``N * rho * 4/3 pi r^3``), never with
  the FEM continuum mass ``rho * V``.
* c2 "energy accounting" defines ``unit_bond_energy = sum(released)/N_broken``
  and then multiplies it back, which is an algebraic identity.

This script replaces both with falsifiable lattice analytics:

* particle/mass enumeration against closed forms for a simple cubic lattice,
* DEM mass against the FEM continuum mass ``rho * V`` under each mass mapping,
* number of bonds crossing a failure face against ``A / (2r)^2``,
* fracture energy per bond against ``G * (2r)^2``,
* input self-consistency ``released_energy`` vs ``G * area``.

Outputs (machine readable, zero hand-copied numbers in the report):
``benchmarks/fracture_gen_audit_report.json``

Usage::

    python benchmarks/fracture_gen_audit.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tensordem.fracture_gen import (
    FailureFace,
    FractureGenConfig,
    FractureLedger,
    build_bond_topology,
    generate_fracture_field,
    generate_particle_lattice,
)

REPORT_PATH = Path(__file__).resolve().parent / "fracture_gen_audit_report.json"


def lattice_bond_count_audit() -> dict:
    """A1: bond count of an n^3 simple cubic lattice vs 3 n^2 (n-1)."""
    rows = []
    for n in (2, 3, 4, 5):
        side = 2.0 * 0.1 * n
        bounds = (0.0, side, 0.0, side, 0.0, side)
        positions = generate_particle_lattice(bounds, 0.1)
        bonds, _ = build_bond_topology(positions, 0.1)
        analytic = 3 * n * n * (n - 1)
        rows.append(
            {
                "n_per_axis": n,
                "num_particles": int(len(positions)),
                "num_bonds": int(len(bonds)),
                "analytic_bonds": analytic,
                "particle_count_exact": len(positions) == n**3,
                "bond_count_exact": len(bonds) == analytic,
            }
        )
    return {
        "criterion": "c4_lattice_enumeration",
        "rows": rows,
        "all_exact": all(r["bond_count_exact"] and r["particle_count_exact"] for r in rows),
    }


def mass_mapping_audit() -> dict:
    """A2: DEM mass vs FEM continuum mass rho*V under both mass mappings."""
    radius = 0.1
    side = 0.6
    density = 917.0
    bounds = (0.0, side, 0.0, side, 0.0, side)
    volume = side**3
    fem_mass = density * volume
    rows = []
    for mapping in ("equal_volume_spheres", "cell_volume"):
        cfg = FractureGenConfig(
            density=density,
            particle_radius=radius,
            domain_bounds=bounds,
            mass_mapping=mapping,
            face_extent="plane",
        )
        field = generate_fracture_field([], cfg)
        ledger = FractureLedger(field, [])
        acc = ledger.mass_accounting(domain_volume=volume)
        rows.append(
            {
                "mass_mapping": mapping,
                "volume_fraction": acc["volume_fraction"],
                "num_particles": acc["num_particles"],
                "dem_mass_kg": acc["dem_mass"],
                "fem_continuum_mass_kg": fem_mass,
                "relative_error": acc["relative_error"],
            }
        )
    # lattice quantisation: coverage error of the discretised volume
    quant = []
    for ratio in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        L = ratio * 2.0 * radius
        n = len(generate_particle_lattice((0, L, 0, L, 0, L), radius))
        n_axis = round(n ** (1.0 / 3.0))
        covered = n_axis * 2.0 * radius
        quant.append(
            {
                "domain_length_m": L,
                "particles_per_axis": n_axis,
                "covered_length_m": covered,
                "cell_mass_relative_error": abs(covered**3 - L**3) / L**3,
            }
        )
    return {
        "criterion": "c1_mass_conservation",
        "domain_volume_m3": volume,
        "fem_continuum_mass_kg": fem_mass,
        "rows": rows,
        "equal_sphere_deficit_fraction": 1.0 - math.pi / 6.0,
        "quantisation_sweep": quant,
    }


def energy_audit() -> dict:
    """A3: bonds crossing a face and fracture energy per bond vs lattice analytics."""
    radius = 0.05
    n_axis = 10
    side = n_axis * 2.0 * radius  # 1.0 m
    density = 917.0
    G = 500.0  # J/m^2
    bounds = (0.0, side, 0.0, side, 0.0, side)
    spacing_sq = (2.0 * radius) ** 2
    full_cross_section = side**2  # 1.0 m^2
    centroid = torch.tensor([side / 2, side / 2, side / 2], dtype=torch.float64)

    rows = []
    for area in (0.01, 0.04, 0.09, 0.25, 0.49, 1.0):
        face = FailureFace(
            id=0,
            area=area,
            normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            centroid=centroid.clone(),
            G=G,
            released_energy=G * area,
        )
        analytic_bonds = area / spacing_sq
        analytic_unit_energy = G * spacing_sq
        entry = {
            "face_area_m2": area,
            "analytic_bonds": analytic_bonds,
            "analytic_unit_bond_energy_J": analytic_unit_energy,
        }
        for extent in ("plane", "disc"):
            cfg = FractureGenConfig(
                density=density,
                particle_radius=radius,
                domain_bounds=bounds,
                face_extent=extent,
                mass_mapping="cell_volume",
            )
            field = generate_fracture_field([face], cfg)
            acc = FractureLedger(field, [face]).energy_accounting()
            entry[extent] = {
                "num_broken_bonds": acc["num_broken_bonds"],
                "bond_count_relative_error": acc["bond_count_relative_error"],
                "unit_bond_energy_J": acc["unit_bond_energy"],
                "unit_bond_energy_relative_error": acc["unit_bond_energy_relative_error"],
            }
        entry["plane_overbreaks_factor"] = (
            entry["plane"]["num_broken_bonds"] / entry["disc"]["num_broken_bonds"]
            if entry["disc"]["num_broken_bonds"]
            else float("inf")
        )
        entry["full_cross_section_m2"] = full_cross_section
        rows.append(entry)

    disc_errors = [r["disc"]["unit_bond_energy_relative_error"] for r in rows]
    valid = [e for e in disc_errors if not math.isnan(e)]
    return {
        "criterion": "c2_energy_accounting",
        "lattice": {
            "particle_radius_m": radius,
            "particles_per_axis": n_axis,
            "num_particles": n_axis**3,
            "domain_side_m": side,
            "bond_spacing_sq_m2": spacing_sq,
        },
        "rows": rows,
        "disc_max_unit_energy_relative_error": max(valid) if valid else float("nan"),
        "disc_max_bond_count_relative_error": max(
            (r["disc"]["bond_count_relative_error"] for r in rows
             if not math.isnan(r["disc"]["bond_count_relative_error"])),
            default=float("nan"),
        ),
    }


def legacy_fixture_audit() -> dict:
    """A4: input self-consistency of the shipped benchmark fixture (G=100, A=0.04)."""
    face = FailureFace(
        id=0,
        area=0.04,
        normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        centroid=torch.tensor([0.3, 0.3, 0.3], dtype=torch.float64),
        G=100.0,
        released_energy=1.0,
    )
    cfg = FractureGenConfig(
        density=917.0,
        particle_radius=0.1,
        domain_bounds=(0.0, 0.6, 0.0, 0.6, 0.0, 0.6),
        face_extent="plane",
    )
    field = generate_fracture_field([face], cfg)
    ledger = FractureLedger(field, [face])
    acc = ledger.energy_accounting()
    expected_released = face.G * face.area
    return {
        "criterion": "input_consistency",
        "declared_released_energy_J": face.released_energy,
        "G_times_area_J": expected_released,
        "input_consistency_error_J": acc["input_consistency_error"],
        "relative_inconsistency": abs(face.released_energy - expected_released) / expected_released,
        "num_broken_bonds_plane": acc["num_broken_bonds"],
        "analytic_bonds_for_declared_area": acc["analytic_bond_count"],
        "implied_unit_bond_energy_J": acc["unit_bond_energy"],
        "analytic_unit_bond_energy_J": acc["analytic_unit_bond_energy"],
        "unit_bond_energy_relative_error": acc["unit_bond_energy_relative_error"],
    }


def invariance_audit() -> dict:
    """A5: rigid-body translation must not change the broken-bond count."""
    radius = 0.1
    side = 0.6
    G = 100.0
    area = 0.36
    base = FailureFace(
        id=0,
        area=area,
        normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        centroid=torch.tensor([0.3, 0.3, 0.3], dtype=torch.float64),
        G=G,
        released_energy=G * area,
    )
    cfg = FractureGenConfig(
        density=917.0,
        particle_radius=radius,
        domain_bounds=(0.0, side, 0.0, side, 0.0, side),
        face_extent="plane",
    )
    counts = []
    for shift in (0.0, 0.05, 0.1):
        face = FailureFace(
            id=0,
            area=area,
            normal=base.normal,
            centroid=base.centroid + torch.tensor([shift, 0.0, 0.0], dtype=torch.float64),
            G=G,
            released_energy=G * area,
        )
        field = generate_fracture_field([face], cfg)
        counts.append({"in_plane_shift_m": shift, "num_broken_bonds": field.num_broken_bonds})
    values = {c["num_broken_bonds"] for c in counts}
    return {
        "criterion": "in_plane_translation_invariance",
        "rows": counts,
        "invariant": len(values) == 1,
        "analytic_bonds": area / (2.0 * radius) ** 2,
    }


def main() -> int:
    torch.set_default_dtype(torch.float64)
    report = {
        "module": "tensordem.fracture_gen",
        "increment": "SI-2/DEM 2.5 (FEM failure faces -> DEM bonded-particle field)",
        "a1_lattice_enumeration": lattice_bond_count_audit(),
        "a2_mass_mapping": mass_mapping_audit(),
        "a3_energy_and_faces": energy_audit(),
        "a4_legacy_fixture": legacy_fixture_audit(),
        "a5_invariance": invariance_audit(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nAudit report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())