#!/usr/bin/env python3
"""Audit script: lateral periodic closure of the DEM3D direct-shear specimen.

Reproduces every number in ``docs/dem3d_shear_periodic_closure.md`` and
``docs/dem3d-shear-periodic-closure/``: the periodic series (exactness of
``C44 = 7k/(3a)``), the three boundary-condition variants (mechanism diagnostic)
and the height-independence series.  Runtime is ~10 s single thread; the free-surface
reference curve is read from the committed cross-section study instead of recomputed.

    python scripts/audit_dem3d_shear_periodic_closure.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tensordem.dem3d import DEM3DConfig
from tensordem.shear_periodic import (
    affine_residual,
    height_series,
    periodic_series,
    static_shear_periodic,
)
from tensordem.shear_specimen import lattice_spacing, shear_modulus_closed_form

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "dem3d-shear-periodic-closure"
REF = ROOT / "docs" / "dem3d-shear-cross-section-convergence" / "series.csv"
FIGURES = OUT / "figures"

SIZES = [3, 4, 5, 7, 9, 11, 13, 15, 17, 21, 25, 29, 33, 41, 49, 61]
VARIANTS = [
    ("periodic_x+periodic_y", True, True),
    ("periodic_x+free_y", True, False),
    ("free_x+free_y", False, False),
]
NZS = [3, 4, 5, 6, 7, 9]
GATE = 0.03


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    cfg = DEM3DConfig()
    a = lattice_spacing(cfg)
    c44 = shear_modulus_closed_form(cfg)

    # 1. periodic series (both lateral directions closed)
    periodic = periodic_series(SIZES, nz=3, periodic_x=True, periodic_y=True)
    rows = periodic["rows"]
    _write_csv(
        OUT / "periodic_series.csv",
        rows,
        ["n", "nz", "particle_count", "bond_count", "area_ratio",
         "G_nominal_over_C44", "G_effective_over_C44", "deficit",
         "reaction_N", "reaction_affine_prediction_N", "reaction_over_affine_prediction",
         "affine_residual_over_bond_force", "free_residual_over_bond_force",
         "max_free_displacement_over_gamma_h"],
    )

    # 2. boundary-condition variants at a fixed, deliberately small size
    variant_rows = []
    for label, px, py in VARIANTS:
        for n in (5, 9, 15, 25):
            got = static_shear_periodic(n, n, 3, periodic_x=px, periodic_y=py)
            diag = affine_residual(n, n, 3, periodic_x=px, periodic_y=py)
            variant_rows.append({
                "variant": label,
                "periodic_x": px,
                "periodic_y": py,
                "n": n,
                "particle_count": got["particle_count"],
                "bond_count": got["bond_count"],
                "area_ratio": got["area_ratio"],
                "G_nominal_over_C44": got["G_nominal_over_C44"],
                "G_effective_over_C44": got["G_effective_over_C44"],
                "deficit": got["deficit"],
                "reaction_over_affine_prediction": got["reaction_over_affine_prediction"],
                "affine_residual_over_bond_force": diag["affine_residual_over_bond_force"],
                "free_residual_over_bond_force": got["free_residual_over_bond_force"],
            })
    _write_csv(
        OUT / "bc_variants.csv",
        variant_rows,
        ["variant", "n", "periodic_x", "periodic_y", "particle_count", "bond_count",
         "area_ratio", "G_nominal_over_C44", "G_effective_over_C44", "deficit",
         "reaction_over_affine_prediction", "affine_residual_over_bond_force",
         "free_residual_over_bond_force"],
    )

    # 3. height independence under periodic closure
    heights = height_series(NZS, n=9, periodic_x=True, periodic_y=True)
    _write_csv(
        OUT / "height_series.csv",
        heights["rows"],
        ["n", "nz", "height_over_spacing", "particle_count", "G_effective_over_C44",
         "deficit", "reaction_over_affine_prediction", "free_residual_over_bond_force"],
    )

    # 4. free-surface reference curve (committed) for the comparison figure
    ref_n, ref_deficit = [], []
    with REF.open() as handle:
        for row in csv.DictReader(handle):
            ref_n.append(int(row["n"]))
            ref_deficit.append(float(row["deficit"]))

    # ---- field map: equilibrium field minus the affine field -----------------
    n_map = 25
    free_field = static_shear_periodic(
        n_map, n_map, 3, periodic_x=False, periodic_y=False, gamma=1e-3,
        return_displacement=True)
    periodic_field = static_shear_periodic(
        n_map, n_map, 3, periodic_x=True, periodic_y=True, gamma=1e-3,
        return_displacement=True)
    height = float(
        free_field["positions_m"][:, 2].max() - free_field["positions_m"][:, 2].min())
    field_summary = {
        "n": n_map,
        "free_max_abs_deviation_over_gamma_h": float(
            np.abs(free_field["displacement_m"][:, 0]
                   - 1e-3 * free_field["positions_m"][:, 2]).max() / (1e-3 * height)),
        "periodic_max_abs_deviation_over_gamma_h": float(
            np.abs(periodic_field["displacement_m"][:, 0]
                   - 1e-3 * periodic_field["positions_m"][:, 2]).max() / (1e-3 * height)),
    }

    # ---- summary ---------------------------------------------------------
    max_def = max(abs(r["deficit"]) for r in rows)
    max_aff = max(r["affine_residual_over_bond_force"] for r in rows)
    freefree5 = next(r for r in variant_rows if r["variant"] == "free_x+free_y" and r["n"] == 5)
    px5 = next(r for r in variant_rows if r["variant"] == "periodic_x+free_y" and r["n"] == 5)
    worst_height = max(abs(r["deficit"]) for r in heights["rows"])

    lines = [
        "# DEM3D direct-shear: lateral periodic closure — generated summary",
        "",
        f"`a = {a:.4f} m`, `k = {cfg.bond_stiffness:.1f} N/m`, "
        f"`C44 = 7k/(3a) = {c44:.4f} Pa`, `gamma = 1e-3`, `nz = 3` unless stated.",
        "",
        "## 1. Periodic cross-section series (`n x n`, both lateral faces periodic)",
        "",
        "| n | particles | bonds | A_eff/A_nom | G_eff/C44 | deficit | affine residual / k gamma a |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | {r['particle_count']} | {r['bond_count']} | "
            f"{r['area_ratio']:.6f} | {r['G_effective_over_C44']:.12f} | "
            f"{r['deficit']:+.3e} | {r['affine_residual_over_bond_force']:.1e} |")
    lines += [
        "",
        f"- max |deficit| over the series = **{max_def:.3e}** "
        f"(gate {GATE:.0%}: {'met' if max_def < GATE else 'NOT met'})",
        f"- max affine residual before solving = **{max_aff:.2e}** `k gamma a`",
        "",
        "## 2. Boundary-condition variants (mechanism diagnostic)",
        "",
        "| variant | n | A_eff/A_nom | G_nominal/C44 | G_eff/C44 | deficit | affine residual |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in variant_rows:
        lines.append(
            f"| {r['variant']} | {r['n']} | {r['area_ratio']:.6f} | "
            f"{r['G_nominal_over_C44']:.6f} | {r['G_effective_over_C44']:.12f} | "
            f"{r['deficit']:+.3e} | {r['affine_residual_over_bond_force']:.1e} |")
    lines += [
        "",
        f"- one pair of free lateral faces (`n = 5`): deficit = **{px5['deficit']:+.3e}** "
        f"(affine residual {px5['affine_residual_over_bond_force']:.1e} `k gamma a`)",
        f"- two pairs of free lateral faces (`n = 5`): deficit = **{freefree5['deficit']:.6f}** "
        f"(affine residual {freefree5['affine_residual_over_bond_force']:.1e} `k gamma a`)",
        "",
        "## 3. Height independence under periodic closure (`n = 9`)",
        "",
        "| nz | h/a | particles | G_eff/C44 | deficit |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in heights["rows"]:
        lines.append(
            f"| {r['nz']} | {r['height_over_spacing']:.0f} | {r['particle_count']} | "
            f"{r['G_effective_over_C44']:.12f} | {r['deficit']:+.3e} |")
    lines += [
        "",
        f"- max |deficit| over the height series = **{worst_height:.3e}**",
        "",
        "## 3b. Equilibrium field vs the affine field (`n = 25`, `nz = 3`)",
        "",
        f"- free lateral faces: max |u_x - gamma z| / (gamma h) = "
        f"**{field_summary['free_max_abs_deviation_over_gamma_h']:.4f}**",
        f"- periodic lateral faces: max |u_x - gamma z| / (gamma h) = "
        f"**{field_summary['periodic_max_abs_deviation_over_gamma_h']:.1e}**",
        "",
        "## 4. Comparison with the free-surface cross-section study",
        "",
        "| n | free-surface deficit (committed) | periodic deficit (this run) |",
        "| --- | --- | --- |",
    ]
    periodic_by_n = {r["n"]: r["deficit"] for r in rows}
    for n, d in zip(ref_n, ref_deficit):
        if n in periodic_by_n:
            lines.append(f"| {n} | {d:.6f} | {periodic_by_n[n]:+.3e} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")

    # ---- figures ---------------------------------------------------------
    figure = _figure(ref_n, ref_deficit, rows, variant_rows, heights["rows"], FIGURES)
    field_figure = _field_figure(free_field, periodic_field, a, height, FIGURES)

    # ---- machine-readable report ----------------------------------------
    report = {
        "study": "dem3d-shear-periodic-closure",
        "config": {
            "lattice_spacing_m": a,
            "bond_stiffness_N_per_m": cfg.bond_stiffness,
            "C44_Pa": c44,
            "gamma": 1e-3,
            "nz": 3,
            "gate": GATE,
        },
        "periodic_series": rows,
        "bc_variants": variant_rows,
        "height_series": heights["rows"],
        "reference_free_surface_series": {
            "source": str(REF.relative_to(ROOT)),
            "n": ref_n,
            "deficit": ref_deficit,
        },
        "headline": {
            "max_abs_deficit_periodic": max_def,
            "max_affine_residual_periodic": max_aff,
            "max_abs_deficit_height_series": worst_height,
            "deficit_one_free_pair_n5": px5["deficit"],
            "deficit_two_free_pairs_n5": freefree5["deficit"],
            "gate_met": bool(max_def < GATE),
        },
        "field_map": field_summary,
        "figures": [figure, field_figure],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report["headline"], **field_summary}, indent=2))
    return 0


def _figure(ref_n, ref_deficit, rows, variant_rows, height_rows, figures: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    ax.semilogx(ref_n, ref_deficit, "o-", color="#c0392b",
                label="free lateral faces (cross-section study)")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("cross-section size n")
    ax.set_ylabel("deficit  1 - G_eff/C44")
    ax.set_title("free surfaces vs affine closure")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["free_x+free_y", "periodic_x+free_y", "periodic_x+periodic_y"]
    colors = {"free_x+free_y": "#c0392b", "periodic_x+free_y": "#e67e22",
              "periodic_x+periodic_y": "#1f77b4"}
    for label in labels:
        pts = [r for r in variant_rows if r["variant"] == label]
        ax.semilogx([p["n"] for p in pts],
                    [max(abs(p["deficit"]), 1e-17) for p in pts],
                    "o-", color=colors[label], label=label)
    ax.set_xlabel("cross-section size n")
    ax.set_ylabel("|deficit|  (floor 1e-17)")
    ax.set_title("periodic closure removes the deficit")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot([r["nz"] for r in height_rows],
            [r["G_effective_over_C44"] for r in height_rows], "s-", color="#27ae60")
    ax.set_ylim(0.99, 1.001)
    ax.set_xlabel("layers nz  (h/a = nz - 1)")
    ax.set_ylabel("G_eff / C44")
    ax.set_title("height independence (n = 9, periodic)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = figures / "periodic_closure.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {
        "name": path.name,
        "path": str(path.relative_to(figures.parents[2])),
        "caption": "Left: free-surface deficit of the committed cross-section study "
                   "(0.687 at n=5 to 0.980 at n=61) against the exact affine closure "
                   "(deficit 0). Middle: the same size sweep under three boundary "
                   "variants -- the deficit vanishes to round-off as soon as the "
                   "lateral faces are periodic in both directions. Right: G_eff/C44 = 1 "
                   "for nz = 3..9 under periodic closure, i.e. no h/a bias.",
        "bytes": path.stat().st_size,
    }


def _field_figure(free_field, periodic_field, spacing, height, figures: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, field, title in zip(
            axes,
            (free_field, periodic_field),
            ("free lateral faces", "periodic lateral faces")):
        positions = field["positions_m"]
        z = positions[:, 2]
        deviation = (field["displacement_m"][:, 0] - 1e-3 * z) / (1e-3 * height)
        mid = np.isclose(z, z.min() + spacing)
        n = int(round(max(positions[:, 0].max(), positions[:, 1].max()) / spacing)) + 1
        grid = np.full((n, n), np.nan)
        grid[(positions[mid, 1] / spacing).astype(int),
             (positions[mid, 0] / spacing).astype(int)] = deviation[mid]
        image = ax.imshow(grid.T * 1e2, origin="lower", cmap="RdBu_r",
                          vmin=-8.0, vmax=8.0)
        ax.set_title(f"{title}\nmax |dev| = {np.nanmax(np.abs(grid)) * 1e2:.1e} % gamma h",
                     fontsize=10)
        ax.set_xlabel("x / a")
    axes[0].set_ylabel("y / a")
    fig.colorbar(image, ax=axes, shrink=0.85, label="(u_x - gamma z)/(gamma h)  [%]")
    path = figures / "field_map.png"
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return {
        "name": path.name,
        "path": str(path.relative_to(figures.parents[2])),
        "caption": "Mid-layer equilibrium field minus the affine field, "
                   "(u_x - gamma z)/(gamma h) in percent of the driven displacement, "
                   "n = 25, nz = 3. Left: free lateral faces -- the boundary layer "
                   "occupies the edge zone and is what the cross-section study fits. "
                   "Right: lateral periodic closure -- the deviation is at round-off, "
                   "i.e. the affine field is the exact solution and there is nothing to "
                   "extrapolate.",
        "bytes": path.stat().st_size,
    }


if __name__ == "__main__":
    raise SystemExit(main())