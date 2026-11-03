"""Independent verification of the restored SI-2/DEM fracture_gen audit API.

Re-derives every anchor that the merged audit documents publish, from closed
forms coded here (not from the module under test), and writes
``artifacts/fracture_gen_restore/verification.{json,png}``.

Anchors
-------
M1  inscribed-sphere mapping: DEM mass / continuum mass = pi/6 (error pi/6 - 1)
M2  lattice-cell mapping:     DEM mass = rho * (2r)^3 per particle, error 0
M3  mass radius factor:       r_mass / r = (6/pi)^(1/3) = 1.240701
G1  plane cut between layers: n^2 = A/s^2 severed bonds
G2  plane cut on a layer:     0 bonds strict, 2 n^2 bonds with on_plane_tolerance
G3  footprint disc:           exact lattice enumeration of pi R^2 / s^2
E1  energy per broken bond:   G * s^2 (falsifiable, replaces the algebraic identity)
E2  input consistency:        |G A - released_energy|
T1  topology:                 vectorised build equals brute-force scan exactly
T2  topology cost:            measured time for 10^3 particles

Usage: python scripts/verify_fracture_gen_audit_api.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensordem.fracture_gen import (  # noqa: E402
    MASS_RADIUS_FACTOR,
    FailureFace,
    FractureGenConfig,
    FractureLedger,
    build_bond_topology,
    face_footprint_radius,
    generate_fracture_field,
    generate_particle_lattice,
)

RADIUS = 0.125
SPACING = 2.0 * RADIUS
DOMAIN = (0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
N_LAYERS = 8
DENSITY = 917.0
CONTINUUM_VOLUME = (N_LAYERS * SPACING) ** 3
PLANE_AREA = (N_LAYERS * SPACING) ** 2
OUT = ROOT / "artifacts" / "fracture_gen_restore"


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


def field(faces, **kw):
    cfg = FractureGenConfig(
        density=DENSITY, particle_radius=RADIUS, domain_bounds=DOMAIN, **kw
    )
    return generate_fracture_field(faces, cfg)


def enumerate_columns(radius, z=0.5):
    """Independent count of lattice columns inside a disc of ``radius``."""
    coords = torch.arange(RADIUS, DOMAIN[1], SPACING, dtype=torch.float64)
    total = 0
    for x in coords:
        for y in coords:
            if math.hypot(float(x) - 1.0, float(y) - 1.0) <= radius:
                total += 1
    return total


def rel(measured, analytic):
    return (measured - analytic) / analytic


def brute_force_bonds(positions, spacing):
    pairs = []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if float(torch.linalg.norm(positions[j] - positions[i])) <= spacing * 1.01:
                pairs.append([i, j])
    return pairs


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # --- M1/M2/M3 mass mapping ------------------------------------------------
    frozen = field([face()], mass_mapping="equal_volume_sphere")
    cell = field([face()], mass_mapping="lattice_cell")
    frozen_acct = FractureLedger(frozen, [face()]).mass_accounting(CONTINUUM_VOLUME)
    cell_acct = FractureLedger(cell, [face()]).mass_accounting(CONTINUUM_VOLUME)
    rows.append(
        {
            "anchor": "M1 inset-sphere mass",
            "analytic": math.pi / 6.0 - 1.0,
            "measured": frozen_acct["relative_error"],
            "relative_error": abs(
                rel(frozen_acct["relative_error"], math.pi / 6.0 - 1.0)
            ),
            "note": "DEM/continuum - 1 = pi/6 - 1",
        }
    )
    rows.append(
        {
            "anchor": "M2 lattice-cell mass",
            "analytic": 0.0,
            "measured": cell_acct["relative_error"],
            "relative_error": abs(cell_acct["relative_error"]),
            "note": "mass = rho h^3 per particle",
        }
    )
    analytic_factor = (6.0 / math.pi) ** (1.0 / 3.0)
    rows.append(
        {
            "anchor": "M3 mass radius factor",
            "analytic": analytic_factor,
            "measured": cell.mass_radius / cell.particle_radius,
            "relative_error": abs(rel(cell.mass_radius / cell.particle_radius, analytic_factor)),
            "note": "r_mass/r = (6/pi)^(1/3)",
        }
    )

    # --- G1/G2 plane cuts -----------------------------------------------------
    g1 = field([face()])
    analytic_bonds = PLANE_AREA / SPACING**2
    rows.append(
        {
            "anchor": "G1 plane cut between layers",
            "analytic": analytic_bonds,
            "measured": float(g1.num_broken_bonds),
            "relative_error": abs(rel(g1.num_broken_bonds, analytic_bonds)),
            "note": "A / s^2",
        }
    )
    strict = field([face(z=RADIUS + 5 * SPACING / 2)], on_plane_tolerance=0.0)
    tolerant = field([face(z=RADIUS + 5 * SPACING / 2)], on_plane_tolerance=1e-12)
    on_layer_z = float(g1.positions[g1.positions[:, 2] == 0.625][0, 2]) if False else 0.625
    strict = field([face(z=on_layer_z)], on_plane_tolerance=0.0)
    tolerant = field([face(z=on_layer_z)], on_plane_tolerance=1e-12)
    rows.append(
        {
            "anchor": "G2 plane on a layer, strict",
            "analytic": 2.0 * N_LAYERS**2,
            "measured": float(strict.num_broken_bonds),
            "relative_error": abs(rel(strict.num_broken_bonds, 2.0 * N_LAYERS**2)),
            "note": "strict sign test is blind to a coplanar cut",
            "expected_defect": True,
        }
    )
    rows.append(
        {
            "anchor": "G2 plane on a layer, on_plane_tolerance=1e-12",
            "analytic": 2.0 * N_LAYERS**2,
            "measured": float(tolerant.num_broken_bonds),
            "relative_error": abs(rel(tolerant.num_broken_bonds, 2.0 * N_LAYERS**2)),
            "note": "2 n^2 bonds restored",
        }
    )

    # --- G3 footprint discs ---------------------------------------------------
    footprint = []
    for r in (0.25, 0.5, 0.75, 1.0, 1.25):
        area = math.pi * r**2
        exact = enumerate_columns(r)
        aware = field([face(area=area, extent_radius=r)], footprint_aware=True)
        frozen_plane = field([face(area=area, extent_radius=r)])
        footprint.append(
            {
                "radius_m": r,
                "area_m2": area,
                "analytic_pi_r2_over_s2": area / SPACING**2,
                "exact_enumeration": exact,
                "footprint_aware": int(aware.num_broken_bonds),
                "frozen_infinite_plane": int(frozen_plane.num_broken_bonds),
                "error_vs_analytic": rel(aware.num_broken_bonds, area / SPACING**2),
                "error_vs_exact": rel(aware.num_broken_bonds, exact),
                "overbreak_factor_frozen": frozen_plane.num_broken_bonds
                / max(1, aware.num_broken_bonds),
            }
        )
    rows.append(
        {
            "anchor": "G3 footprint disc, worst |error| vs pi R^2/s^2 for R<=1 m",
            "analytic": 1.0,
            "measured": 1.0
            + max(abs(f["error_vs_analytic"]) for f in footprint if f["radius_m"] <= 1.0),
            "relative_error": max(
                abs(f["error_vs_analytic"]) for f in footprint if f["radius_m"] <= 1.0
            ),
            "note": "pi R^2/s^2 is a lattice mean field; exact enumeration is the hard baseline",
            "expected_defect": True,
        }
    )
    rows.append(
        {
            "anchor": "G3 footprint disc, max |error| vs exact enumeration",
            "analytic": 1.0,
            "measured": 1.0
            + max(abs(f["error_vs_exact"]) for f in footprint),
            "relative_error": max(abs(f["error_vs_exact"]) for f in footprint),
            "note": "footprint-aware cut reproduces the discrete column count",
        }
    )

    # --- E1/E2 energy accounting ---------------------------------------------
    e_face = FailureFace(
        id=0,
        area=PLANE_AREA,
        normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        centroid=torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64),
        G=100.0,
        released_energy=100.0 * PLANE_AREA,
    )
    e_field = field([e_face])
    energy = FractureLedger(e_field, [e_face]).energy_accounting()
    analytic_unit = e_face.G * SPACING**2
    rows.append(
        {
            "anchor": "E1 energy per broken bond",
            "analytic": analytic_unit,
            "measured": energy["unit_bond_energy"],
            "relative_error": abs(rel(energy["unit_bond_energy"], analytic_unit)),
            "note": "G s^2 with A = n^2 s^2",
        }
    )
    inconsistent = FailureFace(
        id=1,
        area=0.04,
        normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        centroid=torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64),
        G=100.0,
        released_energy=1.0,
    )
    flag = FractureLedger(field([inconsistent]), [inconsistent]).energy_accounting()
    rows.append(
        {
            "anchor": "E2 input consistency |G A - released|",
            "analytic": 3.0,
            "measured": flag["input_consistency_error"],
            "relative_error": abs(rel(flag["input_consistency_error"], 3.0)),
            "note": "legacy fixture G*area=4 J vs released 1 J",
        }
    )

    # --- T1/T2 topology -------------------------------------------------------
    small = generate_particle_lattice((0.0, 0.4, 0.0, 0.4, 0.0, 0.4), 0.1)
    bonds, _ = build_bond_topology(small, 0.1)
    reference = brute_force_bonds(small, 0.2)
    identical = bonds.tolist() == reference
    large = generate_particle_lattice((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), 0.05)
    start = time.perf_counter()
    large_bonds, _ = build_bond_topology(large, 0.05)
    vectorised_seconds = time.perf_counter() - start
    analytic_large = 3 * 10**2 * 9
    rows.append(
        {
            "anchor": "T1 topology equals brute force (n=4)",
            "analytic": float(len(reference)),
            "measured": float(len(bonds)),
            "relative_error": 0.0 if identical else 1.0,
            "note": "pair list identical, not just the count" if identical else "MISMATCH",
        }
    )
    rows.append(
        {
            "anchor": "T2 bond count, n=10 (3n^2(n-1))",
            "analytic": float(analytic_large),
            "measured": float(len(large_bonds)),
            "relative_error": abs(rel(len(large_bonds), analytic_large)),
            "note": f"vectorised build {vectorised_seconds:.3f} s for {len(large)} particles",
        }
    )

    worst = max(r["relative_error"] for r in rows if not r.get("expected_defect"))
    payload = {
        "protocol": "tensordem-fracture-gen-audit-verification-v1",
        "lattice": {
            "particle_radius_m": RADIUS,
            "spacing_m": SPACING,
            "n_layers": N_LAYERS,
            "n_particles": int(frozen.num_particles),
            "n_bonds": int(frozen.bond_indices.shape[0]),
            "n_bonds_analytic": 3 * N_LAYERS**2 * (N_LAYERS - 1),
        },
        "mass": {
            "continuum_mass_kg": CONTINUUM_VOLUME * DENSITY,
            "frozen_mass_kg": frozen.total_mass,
            "frozen_relative_error": frozen_acct["relative_error"],
            "cell_mass_kg": cell.total_mass,
            "cell_relative_error": cell_acct["relative_error"],
            "mass_radius_factor": cell.mass_radius / cell.particle_radius,
            "mass_radius_factor_analytic": MASS_RADIUS_FACTOR,
            "footprint_radius_from_area_m": face_footprint_radius(face(area=0.04)),
        },
        "footprint": footprint,
        "energy": energy,
        "topology": {
            "vectorised_seconds_n10": vectorised_seconds,
            "pairs_identical_to_brute_force": identical,
        },
        "anchors": rows,
        "worst_relative_error": worst,
        "acceptance_3pct": worst < 0.03,
    }
    (OUT / "verification.json").write_text(json.dumps(payload, indent=2, default=float))

    _plot(payload, footprint)
    _print_table(rows, worst)
    return payload


def _plot(payload, footprint) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    names = ["inscribed\nsphere", "lattice\ncell"]
    errors = [
        100.0 * payload["mass"]["frozen_relative_error"],
        100.0 * payload["mass"]["cell_relative_error"],
    ]
    ax.bar(names, errors, color=["tab:red", "tab:green"])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(-15.0, color="gray", ls="--", lw=1, label="-15 % bar")
    ax.set_ylabel("DEM mass error vs rho V  [%]")
    ax.set_title("mass mapping (n=8 lattice)")
    ax.legend(loc="lower right", fontsize=8)
    for i, v in enumerate(errors):
        ax.annotate(f"{v:.2f} %", (i, v), ha="center", va="bottom" if v > 0 else "top")
    ax.set_ylim(-60, 10)

    ax = axes[1]
    radii = [f["radius_m"] for f in footprint]
    ax.plot(radii, [f["analytic_pi_r2_over_s2"] for f in footprint], "k-", label="pi R^2/s^2")
    ax.plot(radii, [f["exact_enumeration"] for f in footprint], "ks", ms=6, label="exact lattice")
    ax.plot(radii, [f["footprint_aware"] for f in footprint], "o-", color="tab:green",
            label="footprint-aware cut")
    ax.plot(radii, [f["frozen_infinite_plane"] for f in footprint], "^-", color="tab:red",
            label="frozen infinite plane")
    ax.set_xlabel("face footprint radius R [m]")
    ax.set_ylabel("severed bonds")
    ax.set_title("finite face over-breakage")
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = [r["anchor"].split(",")[0].replace(" ", "\n", 1) for r in payload["anchors"]]
    vals = [max(r["relative_error"], 1e-17) for r in payload["anchors"]]
    ax.barh(range(len(vals)), vals, color=["tab:green" if v < 0.03 else "tab:red" for v in vals])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xscale("log")
    ax.axvline(0.03, color="k", ls="--", lw=1, label="3 % bar")
    ax.set_xlabel("|relative error| vs closed form")
    ax.set_title("independent anchors")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "verification.png", dpi=140)
    plt.close(fig)


def _print_table(rows, worst) -> None:
    print(f"{'anchor':52s} {'analytic':>12s} {'measured':>12s} {'rel.err':>10s}")
    for r in rows:
        print(
            f"{r['anchor'][:52]:52s} {r['analytic']:12.6g} {r['measured']:12.6g} "
            f"{r['relative_error']:10.3e}"
        )
    print(f"\nworst |relative error| = {worst:.3e}  (<3 %: {worst < 0.03})")


if __name__ == "__main__":
    main()