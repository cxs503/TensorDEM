#!/usr/bin/env python
"""Audit the FEM->DEM fracture-field mass/volume mapping against analytic geometry.

Two closed-form biases are quantified:

1. **Material mapping.**  A particle on the simple-cubic lattice of spacing
   ``h = 2r`` owns a cell of volume ``h^3``; an inscribed sphere of radius ``r``
   holds only ``pi/6 = 52.36 %`` of it, so the legacy ``mass_mode="sphere"``
   under-counts the mass of the represented ice volume by 47.64 %.
2. **Lattice quantisation.**  Particles are placed at ``r + k h < W`` per axis,
   so unless ``W`` is an integer multiple of ``h`` the lattice covers ``N h``
   instead of ``W``; the relative volume error ``(N h - W)/W`` is reported
   separately so it is never confused with a material-mapping residual.

Usage
-----
    python scripts/audit_fracture_mass_mapping.py --output artifacts/fracture-mass-mapping

Writes ``fracture_mass_mapping_audit.json``, ``..._sweep.csv`` and
``fracture_mass_mapping_audit.png`` (two-panel figure) into ``--output``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.fracture_mass_mapping import (  # noqa: E402
    SPHERE_MAPPING_DEFICIT,
    domain_volume,
    expected_lattice_counts,
    mass_mapping_report,
    resolution_sweep,
    sphere_to_cell_volume_ratio,
    volume_quantisation_error,
)

DENSITY = 917.0  # sea-ice density, kg/m^3
DOMAIN = (0.0, 0.6, 0.0, 0.6, 0.0, 0.6)  # frozen SI-2 fracture-gen smoke domain
RADIUS = 0.1  # frozen SI-2 particle radius, m
TOLERANCE = 0.01  # 1 % mass-mapping gate


def _sweep_rows(particles_per_axis=(2, 3, 4, 5, 6, 8, 10)) -> list[dict]:
    sweep = resolution_sweep(
        DENSITY, DOMAIN, particles_per_axis=particles_per_axis, tolerance=TOLERANCE
    )
    return sweep["cases"]


def _quantisation_rows(widths=None, radius=RADIUS) -> list[dict]:
    if widths is None:
        widths = [0.60 + 0.005 * i for i in range(41)]  # 0.600 ... 0.800 m
    rows = []
    for width in widths:
        bounds = (0.0, width, 0.0, width, 0.0, width)
        counts = expected_lattice_counts(bounds, radius)
        rows.append(
            {
                "extent_m": width,
                "particles_per_axis": counts[0],
                "covered_extent_m": counts[0] * 2.0 * radius,
                "quantisation_relative_error": volume_quantisation_error(bounds, radius),
            }
        )
    return rows


def _ship_scale_analytical() -> dict:
    """Analytic row for a ship-ice sheet (no lattice built, geometry only).

    A 10 m x 10 m x 0.8 m level-ice sheet is far too large for the O(N^2) bond
    builder of the current generator, but the *mass mapping* is pure geometry,
    so the error is available in closed form at the intended radius.
    """
    sheet = (0.0, 10.0, 0.0, 10.0, 0.0, 0.8)
    radius = 0.1
    cell = mass_mapping_report(DENSITY, radius, sheet, "cell", tolerance=TOLERANCE)
    sphere = mass_mapping_report(DENSITY, radius, sheet, "sphere", tolerance=TOLERANCE)
    return {
        "purpose": "ship-scale level-ice sheet mass mapping (analytic only)",
        "domain_bounds_m": list(sheet),
        "ice_volume_m3": domain_volume(sheet),
        "particle_radius_m": radius,
        "lattice_counts": cell["lattice_counts"],
        "num_particles": cell["num_particles"],
        "cell_mode": {k: cell[k] for k in (
            "dem_total_mass_kg", "reference_mass_kg", "relative_mass_error",
            "lattice_quantisation_relative_error", "within_tolerance")},
        "sphere_mode": {k: sphere[k] for k in (
            "dem_total_mass_kg", "reference_mass_kg", "relative_mass_error",
            "lattice_quantisation_relative_error", "within_tolerance")},
        "mass_ratio_sphere_over_cell": sphere["dem_total_mass_kg"] / cell["dem_total_mass_kg"],
    }


def _reference_replay() -> dict:
    """Replay the frozen SI-2 smoke case in both mappings."""
    from tensordem.fracture_gen import FractureGenConfig, generate_fracture_field

    out = {}
    for mode in ("cell", "sphere"):
        field = generate_fracture_field(
            [],
            FractureGenConfig(
                density=DENSITY,
                particle_radius=RADIUS,
                domain_bounds=DOMAIN,
                mass_mode=mode,
            ),
        )
        out[mode] = {
            "num_particles": field.num_particles,
            "num_bonds": len(field.bond_indices),
            "particle_mass_kg": field.particle_mass,
            "total_mass_kg": field.total_mass,
            "represented_volume_m3": field.represented_volume,
        }
    ice_mass = DENSITY * domain_volume(DOMAIN)
    out["reference"] = {
        "ice_volume_m3": domain_volume(DOMAIN),
        "ice_mass_kg": ice_mass,
    }
    out["relative_error"] = {
        mode: out[mode]["total_mass_kg"] / ice_mass - 1.0 for mode in ("cell", "sphere")
    }
    return out


def _plot(sweep_rows, quant_rows, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    n = [row["particles_per_axis"] for row in sweep_rows]
    cell_err = [100.0 * row["cell_mode_relative_error"] for row in sweep_rows]
    sphere_err = [100.0 * row["sphere_mode_relative_error"] for row in sweep_rows]

    ax = axes[0]
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axhspan(-100.0 * TOLERANCE, 100.0 * TOLERANCE, color="0.85", zorder=0,
               label=f"±{100 * TOLERANCE:.0f} % gate")
    ax.plot(n, sphere_err, "o-", color="C3",
            label=r"legacy sphere: $-\,(1-\pi/6)=-47.64\,\%$")
    ax.plot(n, cell_err, "s-", color="C0", label="cell (volume consistent): 0 %")
    ax.axhline(-100.0 * SPHERE_MAPPING_DEFICIT, color="C3", ls=":", lw=0.9)
    ax.set_xlabel("particles per axis  (r = W / 2n)")
    ax.set_ylabel("relative mass error vs $\\rho V_{domain}$  [%]")
    ax.set_title("(a) material mapping term (quantisation = 0)")
    ax.set_ylim(-55.0, 6.0)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = [row["extent_m"] for row in quant_rows]
    y = [100.0 * row["quantisation_relative_error"] for row in quant_rows]
    ax.plot(x, y, "-", color="C2", lw=1.2)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axhspan(-100.0 * TOLERANCE, 100.0 * TOLERANCE, color="0.85", zorder=0,
               label=f"±{100 * TOLERANCE:.0f} % gate")
    ax.set_xlabel("domain extent W along each axis  [m]   (h = 0.2 m)")
    ax.set_ylabel("lattice coverage error  $(Nh - W)/W$  [%]")
    ax.set_title("(b) lattice quantisation term (cell mapping)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "TensorDEM FEM→DEM fracture field: mass/volume mapping audit "
        f"(simple-cubic packing fraction π/6 = {sphere_to_cell_volume_ratio():.5f})",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="artifacts/fracture-mass-mapping")
    ap.add_argument(
        "--particles-per-axis",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 8, 10],
        help="lattice resolutions swept for the material-mapping panel",
    )
    ap.add_argument(
        "--quantisation-points",
        type=int,
        default=41,
        help="number of domain extents swept in the quantisation panel",
    )
    args = ap.parse_args(argv)
    if any(n < 1 for n in args.particles_per_axis):
        ap.error("--particles-per-axis entries must be >= 1")
    if args.quantisation_points < 2:
        ap.error("--quantisation-points must be >= 2")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    sweep_rows = _sweep_rows(tuple(args.particles_per_axis))
    quant_rows = _quantisation_rows(
        [0.60 + 0.20 * i / (args.quantisation_points - 1)
         for i in range(args.quantisation_points)]
    )
    replay = _reference_replay()
    sheet = _ship_scale_analytical()

    report = {
        "protocol": "tensordem-fracture-mass-mapping-audit-v1",
        "analytic_anchors": {
            "simple_cubic_packing_fraction_pi_over_6": sphere_to_cell_volume_ratio(),
            "legacy_sphere_mapping_deficit": SPHERE_MAPPING_DEFICIT,
            "legacy_sphere_mapping_deficit_pct": 100.0 * SPHERE_MAPPING_DEFICIT,
            "equal_volume_radius_over_spacing": equal_volume_ratio(),
        },
        "tolerance_relative": TOLERANCE,
        "frozen_si2_reference_case": replay,
        "resolution_sweep": sweep_rows,
        "quantisation_sweep": quant_rows,
        "ship_scale_analytic": sheet,
        "figure": "fracture_mass_mapping_audit.png",
    }

    (out / "fracture_mass_mapping_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (out / "fracture_mass_mapping_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)
    _plot(sweep_rows, quant_rows, out / report["figure"])

    print(f"simple-cubic packing fraction pi/6 = {sphere_to_cell_volume_ratio():.6f}")
    print(f"legacy sphere mapping mass error   = {-100 * SPHERE_MAPPING_DEFICIT:.2f} %")
    print(
        "cell mapping mass error (frozen case) = "
        f"{100 * replay['relative_error']['cell']:.3e} %"
    )
    print(
        "ship-scale 10x10x0.8 m sheet, cell mode error = "
        f"{100 * sheet['cell_mode']['relative_mass_error']:.3e} %, "
        f"sphere mode = {100 * sheet['sphere_mode']['relative_mass_error']:.2f} %"
    )
    print(f"wrote {out}")
    return 0


def equal_volume_ratio() -> float:
    from tensordem.fracture_mass_mapping import equal_volume_radius

    return equal_volume_radius(1.0)


if __name__ == "__main__":
    raise SystemExit(main())