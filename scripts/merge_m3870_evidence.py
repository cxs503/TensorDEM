"""Reproduce the evidence behind the PR #168 (branch ``agent/m2910``) cure.

The cure is a three-way merge of ``main`` into ``agent/m2910``; this script
re-derives, from the *merged* module, every number quoted in
``docs/m3870_pr168_merge_note.md`` and writes the figure.  Nothing is hand
copied: run it and compare with the note.

    python scripts/merge_m3870_evidence.py [--output runs/m3870_merge]

Outputs
-------
``evidence.json``   -- every number in the note, plus the git SHAs of the
                       merge parents recorded by the merge commit.
``merge_evidence.png`` -- 2 panels: (a) DEM/continuum mass residual per mass
                       mapping against the 3 % and 15 % bars, (b) severed bond
                       count per ``face_extent`` criterion against the
                       lattice analytic ``A / h^2``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.fracture_gen import (  # noqa: E402
    MASS_MODE_TO_MAPPING,
    FractureGenConfig,
    FractureLedger,
    FailureFace,
    generate_fracture_field,
)

DENSITY = 917.0
RADIUS = 0.05
N_AXIS = 10
SIDE = N_AXIS * 2 * RADIUS  # 1.0 m, h = 2 r = 0.1 m
BOUNDS = (0.0, SIDE, 0.0, SIDE, 0.0, SIDE)
G = 500.0
SPACING_SQ = (2 * RADIUS) ** 2


def face(area: float, centroid: tuple[float, float, float], G_: float = G) -> FailureFace:
    return FailureFace(
        id=0,
        area=area,
        normal=torch.tensor([0.0, 0.0, 1.0]),
        centroid=torch.tensor(centroid),
        G=G_,
        released_energy=G_ * area,
    )


def mass_rows() -> list[dict]:
    """Material-mapping residual of every accepted mapping spelling."""
    rows = []
    for mapping in ("cell_volume", "lattice_cell", "equal_volume_spheres", "sphere"):
        cfg = FractureGenConfig(
            density=DENSITY,
            particle_radius=RADIUS,
            domain_bounds=BOUNDS,
            mass_mapping=mapping,
        )
        field = generate_fracture_field([], cfg)
        acc = FractureLedger(field, []).mass_accounting(SIDE**3)
        rows.append(
            {
                "requested_spelling": mapping,
                "reported_mass_mapping": field.mass_mapping,
                "mass_mode": field.mass_mode,
                "num_particles": acc["num_particles"],
                "dem_mass_kg": acc["dem_mass"],
                "continuum_mass_kg": acc["fem_continuum_mass"],
                "relative_error": acc["relative_error"],
                "relative_error_magnitude": acc["relative_error_magnitude"],
                "absolute_relative_error": acc["absolute_relative_error"],
                "within_3pct": acc["absolute_relative_error"] <= 0.03,
                "within_15pct": acc["within_15pct"],
            }
        )
    # both magnitude spellings must be the same number, not two definitions
    for row in rows:
        assert math.isclose(
            row["relative_error_magnitude"], row["absolute_relative_error"], abs_tol=0.0
        ), row
    return rows


def bond_rows() -> list[dict]:
    """Severed-bond count per face-extent criterion vs the lattice analytic."""
    rows = []
    area = 0.16  # 0.4 m x 0.4 m face -> pi R^2 / h^2 ~ 50 bonds
    analytic = area / SPACING_SQ
    for extent in ("plane", "disc"):
        cfg = FractureGenConfig(
            density=DENSITY,
            particle_radius=RADIUS,
            domain_bounds=BOUNDS,
            face_extent=extent,
            mass_mapping="cell_volume",
        )
        f = face(area, (SIDE / 2, SIDE / 2, SIDE / 2))
        field = generate_fracture_field([f], cfg)
        acc = FractureLedger(field, [f]).energy_accounting()
        rows.append(
            {
                "face_extent": extent,
                "broken_bonds": int(field.num_broken_bonds),
                "analytic_bond_count": analytic,
                "bond_count_relative_error": acc["bond_count_relative_error"],
                "unit_bond_energy": acc["unit_bond_energy"],
                "analytic_unit_bond_energy": acc["analytic_unit_bond_energy"],
                "unit_bond_energy_relative_error": acc["unit_bond_energy_relative_error"],
            }
        )
    return rows


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def shas() -> dict:
    """Merge topology: the cure is the merge commit HEAD.

    ``parents[0]`` is the pre-cure tip of ``agent/m2910`` (the old SHA) and
    ``parents[1]`` is the trunk tip that was merged in; the merge base is the
    commit the branch forked from.
    """
    out: dict = {}
    try:
        out["head"] = git("rev-parse", "HEAD")
        out["parents"] = git("rev-list", "--parents", "-n", "1", "HEAD").split()[1:]
        out["subject"] = git("log", "-1", "--pretty=%s", "HEAD")
        if len(out["parents"]) == 2:
            out["old_branch_sha"], out["merged_trunk_sha"] = out["parents"]
            out["merge_base"] = git("merge-base", *out["parents"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        out["note"] = "git metadata unavailable"
    return out


def histogram_panel(ax, rows):
    labels = [row["reported_mass_mapping"] for row in rows]
    errors = [100.0 * row["absolute_relative_error"] for row in rows]
    colors = ["#1f77b4" if row["reported_mass_mapping"] == "cell_volume" else "#d62728"
              for row in rows]
    bars = ax.bar(range(len(rows)), errors, color=colors, width=0.6)
    ax.axhline(15.0, color="k", ls="--", lw=1.2, label="15 % material-mapping bar")
    ax.axhline(3.0, color="#ff7f0e", ls=":", lw=1.2, label="3 % (research target)")
    for bar, row in zip(bars, rows):
        ax.annotate(
            f"{100.0 * row['absolute_relative_error']:.2f} %",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(
        [f"{row}\n(asked: {row2})" for row, row2 in zip(labels, [r["requested_spelling"] for r in rows])],
        fontsize=7,
    )
    ax.set_ylabel("|M_DEM - rho V| / (rho V)  [%]")
    ax.set_title("(a) mass mapping residual, 1000-particle lattice (1 m^3)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def bond_panel(ax, rows):
    idx = range(len(rows))
    broken = [row["broken_bonds"] for row in rows]
    analytic = [row["analytic_bond_count"] for row in rows]
    ax.bar([i - 0.2 for i in idx], broken, width=0.4, label="DEM severed bonds",
           color="#2ca02c")
    ax.bar([i + 0.2 for i in idx], analytic, width=0.4,
           label="analytic A/h^2", color="#9467bd")
    for i, row in zip(idx, rows):
        ax.annotate(
            f"{100.0 * row['bond_count_relative_error']:+.1f} %",
            (i, max(row["broken_bonds"], row["analytic_bond_count"])),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(list(idx))
    ax.set_xticklabels([f"face_extent={row['face_extent']}" for row in rows])
    ax.set_ylabel("severed bonds on a 1 m^3 lattice (h = 0.1 m)")
    ax.set_title("(b) frozen plane criterion vs footprint disc, A = 0.16 m^2")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="runs/m3870_merge")
    args = ap.parse_args()
    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    masses = mass_rows()
    bonds = bond_rows()
    evidence = {
        "module": "tensordem.fracture_gen (post-merge)",
        "lattice": {
            "density_kg_m3": DENSITY,
            "particle_radius_m": RADIUS,
            "spacing_m": 2 * RADIUS,
            "n_axis": N_AXIS,
            "particles": N_AXIS**3,
            "domain_m3": SIDE**3,
        },
        "mass_mapping": masses,
        "bond_severing": bonds,
        "canonical_reporting_names": MASS_MODE_TO_MAPPING,
        "git": shas(),
    }
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    histogram_panel(axes[0], masses)
    bond_panel(axes[1], bonds)
    fig.tight_layout()
    fig.savefig(out / "merge_evidence.png", dpi=150)
    print(json.dumps(evidence, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())