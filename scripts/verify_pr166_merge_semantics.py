"""Semantic verification of the PR #166 (agent/m2942) three-way merge.

The merge of ``main`` into ``agent/m2942`` conflicted in exactly one file,
``src/tensordem/fracture_gen.py``, because both sides introduced the same two
symbols (``MASS_RADIUS_FACTOR`` and ``face_footprint_radius``) plus the
``FailureFace.extent_radius`` field.  The resolution kept one canonical
definition of each.  This script proves that the union is *semantically* what
both sides meant, against independent analytic references -- it is not a
"imports fine" smoke test:

P1 -- constant identity.  m2942 wrote ``(6/pi)^(1/3)``, trunk wrote
      ``2 (3/(4 pi))^(1/3)``.  They must be bit-comparable: max |diff| == 0.
P2 -- footprint radius.  ``face_footprint_radius`` must return
      ``sqrt(A/pi)`` when ``extent_radius`` is absent and the explicit value
      when present; the equal-area identity must hold to machine precision.
P3 -- mass mapping.  Volume-consistent ``mass_mode="cell"`` must give
      ``m / (rho h^3) == 1`` exactly; the frozen ``"sphere"`` mapping must give
      ``pi/6 - 1 = -47.64 %``.  These are the numbers the two red-then-green
      tests of ``tests/test_fracture_gen_mass_footprint.py`` assert.
P4 -- severed-bond count vs footprint radius.  For a face disc of radius R on a
      simple-cubic lattice of spacing h, the expected number of nearest-neighbour
      bonds crossing the disc is ``pi R^2 / h^2`` in the continuum limit and
      ``4 n^2``-style exactly by lattice enumeration.  Sweeping R shows the
      implemented criterion tracks the analytic curve instead of the
      area-independent plane criterion.

Every number written to ``artifacts/pr166/`` comes from this script; nothing is
transcribed by hand.

Run:  python3 scripts/verify_pr166_merge_semantics.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tensordem.fracture_gen import (  # noqa: E402
    MASS_RADIUS_FACTOR,
    FailureFace,
    FractureGenConfig,
    build_bond_topology,
    face_footprint_radius,
    generate_fracture_field,
    generate_particle_lattice,
    identify_bonds_crossing_face,
)

OUT = ROOT / "artifacts" / "pr166"

# Reference lattice used by the frozen verification contracts:
# r = 0.125 m, h = 2r = 0.25 m, domain (0,2)^3 -> 8 layers/axis, N = 512.
RADIUS = 0.125
SPACING = 2.0 * RADIUS
DOMAIN = (0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
DENSITY = 917.0


def p1_constant_identity() -> dict:
    m2942_form = (6.0 / math.pi) ** (1.0 / 3.0)
    trunk_form = 2.0 * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    return {
        "trunk_form_2*(3/4pi)^(1/3)": trunk_form,
        "m2942_form_(6/pi)^(1/3)": m2942_form,
        "module_value": MASS_RADIUS_FACTOR,
        "abs_diff_trunk_vs_m2942": abs(trunk_form - m2942_form),
        "abs_diff_module_vs_m2942": abs(MASS_RADIUS_FACTOR - m2942_form),
        "bitwise_equal": MASS_RADIUS_FACTOR == m2942_form == trunk_form,
    }


def _face(area: float, extent_radius: float | None = None) -> FailureFace:
    return FailureFace(
        id=0,
        area=area,
        normal=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        centroid=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        G=1.0,
        released_energy=1.0,
        extent_radius=extent_radius,
    )


def p2_footprint_radius() -> dict:
    rows = []
    for area in (math.pi * 0.0625, math.pi * 0.25, math.pi * 0.5625, 1.0):
        r = face_footprint_radius(_face(area))
        rows.append(
            {
                "area": area,
                "radius": r,
                "analytic_sqrt_area_over_pi": math.sqrt(area / math.pi),
                "abs_error": abs(r - math.sqrt(area / math.pi)),
            }
        )
    explicit = face_footprint_radius(_face(math.pi * 0.0625, extent_radius=0.37))
    return {
        "equal_area_rows": rows,
        "max_abs_error": max(r["abs_error"] for r in rows),
        "explicit_extent_radius": 0.37,
        "explicit_returned": explicit,
        "explicit_abs_error": abs(explicit - 0.37),
    }


def p3_mass_mapping() -> dict:
    config = FractureGenConfig(
        density=DENSITY, particle_radius=RADIUS, domain_bounds=DOMAIN
    )
    field = generate_fracture_field([], config)
    cell_volume = SPACING**3
    cell_mass = field.particle_mass
    sphere_field = generate_fracture_field(
        [], FractureGenConfig(
            density=DENSITY,
            particle_radius=RADIUS,
            domain_bounds=DOMAIN,
            mass_mode="sphere",
        )
    )
    sphere_mass = sphere_field.particle_mass
    continuum_mass = DENSITY * cell_volume
    return {
        "num_particles": int(field.num_particles),
        "num_bonds": int(field.bond_indices.shape[0]),
        "cell_volume_h3": cell_volume,
        "continuum_mass_density_times_h3": continuum_mass,
        "cell_mode_mass": cell_mass,
        "cell_mode_relative_error_vs_continuum": abs(cell_mass - continuum_mass) / continuum_mass,
        "sphere_mode_mass": sphere_mass,
        "sphere_mode_relative_deficit": (sphere_mass - continuum_mass) / continuum_mass,
        "analytic_pi_over_6_minus_1": math.pi / 6.0 - 1.0,
        "mass_radius": field.mass_radius,
        "analytic_mass_radius_factor": MASS_RADIUS_FACTOR,
        "mass_radius_relative_error": abs(field.mass_radius - MASS_RADIUS_FACTOR * RADIUS)
        / (MASS_RADIUS_FACTOR * RADIUS),
    }


def _exact_enumeration(radius: float) -> int:
    """Independently enumerate the nearest-neighbour bonds a disc of ``radius``
    around the face centroid (1, 1, 1) must sever, without calling the module.

    The face plane is ``x = 1`` with the lattice on ``x = 0.125 + 0.25 k``, so
    exactly one ``x``-bond per (y, z) column straddles the plane, and its
    crossing point sits at ``(1, y, z)``.  A column is severed iff its in-plane
    offset from the centroid is inside the disc.  With 8 x 8 columns this is a
    16-point radial test.
    """
    offsets = [RADIUS + SPACING * k - 1.0 for k in range(8)]
    return sum(
        1
        for dy in offsets
        for dz in offsets
        if math.sqrt(dy * dy + dz * dz) <= radius
    )


def p4_bond_count_sweep() -> dict:
    """``identify_bonds_crossing_face`` returns a boolean *mask* over bonds.

    Reference hierarchy: the exact lattice enumeration above is the discrete
    truth (the implemented criterion must match it exactly); ``pi R^2 / h^2`` is
    the continuum limit it must approach as ``R / h`` grows.
    """
    positions = generate_particle_lattice(DOMAIN, RADIUS)
    bond_indices, _rest = build_bond_topology(positions, RADIUS)
    rows = []
    for radius in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
        face = _face(math.pi * radius**2, extent_radius=radius)
        disc = identify_bonds_crossing_face(
            bond_indices, positions, face, footprint_aware=True
        )
        plane = identify_bonds_crossing_face(
            bond_indices, positions, face, footprint_aware=False
        )
        exact = _exact_enumeration(radius)
        continuum = math.pi * radius**2 / SPACING**2
        disc_count = int(disc.sum().item())
        rows.append(
            {
                "footprint_radius": radius,
                "bond_count_disc": disc_count,
                "bond_count_plane": int(plane.sum().item()),
                "exact_lattice_enumeration": exact,
                "matches_enumeration": disc_count == exact,
                "analytic_pi_r2_over_h2": continuum,
                "relative_error_vs_enumeration": (
                    abs(disc_count - exact) / exact if exact else 0.0
                ),
                "relative_error_vs_continuum": abs(disc_count - continuum) / continuum,
                "r_over_h": radius / SPACING,
            }
        )
    return {
        "rows": rows,
        "num_particles": int(positions.shape[0]),
        "num_bonds": int(bond_indices.shape[0]),
        "all_match_enumeration": all(r["matches_enumeration"] for r in rows),
        "max_error_vs_enumeration": max(r["relative_error_vs_enumeration"] for r in rows),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "P1_constant_identity": p1_constant_identity(),
        "P2_footprint_radius": p2_footprint_radius(),
        "P3_mass_mapping": p3_mass_mapping(),
        "P4_bond_count_sweep": p4_bond_count_sweep(),
    }
    (OUT / "verify_pr166.json").write_text(json.dumps(report, indent=2, sort_keys=False))
    for key in ("P1_constant_identity", "P2_footprint_radius", "P3_mass_mapping"):
        print(key, json.dumps(report[key], indent=2))
    print("P4 bond-count sweep (disc):")
    for row in report["P4_bond_count_sweep"]["rows"]:
        print(
            f"  R={row['footprint_radius']:.2f}  disc={row['bond_count_disc']:3d} "
            f"exact={row['exact_lattice_enumeration']:3d} "
            f"plane={row['bond_count_plane']:3d} "
            f"piR2/h2={row['analytic_pi_r2_over_h2']:7.3f} "
            f"err_vs_exact={row['relative_error_vs_enumeration']*100:5.2f}% "
            f"err_vs_continuum={row['relative_error_vs_continuum']*100:6.2f}%"
        )
    print(f"wrote {OUT/'verify_pr166.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())