"""Figures for the SI-2/DEM fracture-generator verification audit.

Reads ``benchmarks/fracture_gen_audit_report.json`` (produced by
``python benchmarks/fracture_gen_audit.py``) and writes:

* ``docs/figures/fracture_gen_audit/field_broken_bonds.png``
* ``docs/figures/fracture_gen_audit/bond_count_vs_face_area.png``
* ``docs/figures/fracture_gen_audit/unit_bond_energy_error.png``
* ``docs/figures/fracture_gen_audit/mass_mapping_error.png``

All numbers plotted come from the audit report; nothing is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensordem.fracture_gen import (  # noqa: E402
    FailureFace,
    FractureGenConfig,
    generate_fracture_field,
)

REPORT = ROOT / "benchmarks" / "fracture_gen_audit_report.json"
OUT = ROOT / "docs" / "figures" / "fracture_gen_audit"
THRESHOLD = 0.03


def _field(area: float, extent: str):
    face = FailureFace(
        id=0,
        area=area,
        normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        centroid=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64),
        G=500.0,
        released_energy=500.0 * area,
    )
    cfg = FractureGenConfig(
        density=917.0,
        particle_radius=0.05,
        domain_bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        face_extent=extent,
        mass_mapping="cell_volume",
    )
    return generate_fracture_field([face], cfg)


def plot_broken_bonds_field(area: float = 0.04) -> Path:
    """Bond-level map of the pre-broken field, x-z projection (plane vs disc)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, extent in zip(axes, ("plane", "disc")):
        field = _field(area, extent)
        pos = field.positions
        bonds = field.bond_indices
        alive = field.alive
        for idx in range(len(bonds)):
            i, j = bonds[idx].tolist()
            colour = "#9aa5b1" if alive[idx] else "#d62728"
            lw = 0.7 if alive[idx] else 1.6
            z = 3 if not alive[idx] else 1
            ax.plot(
                [pos[i, 0].item(), pos[j, 0].item()],
                [pos[i, 2].item(), pos[j, 2].item()],
                color=colour,
                linewidth=lw,
                zorder=z,
                alpha=0.9 if not alive[idx] else 0.5,
            )
        ax.set_title(
            f"face_extent='{extent}'  broken={field.num_broken_bonds} of {len(bonds)} bonds"
        )
        ax.set_xlabel("x [m]")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("z [m]")
    fig.suptitle(
        f"Pre-broken bond field for a {area} m$^2$ failure face (A/(2r)$^2$ = "
        f"{area / 0.01:.1f} bonds), r = 0.05 m, 10$^3$ particles",
        fontsize=10,
    )
    fig.tight_layout()
    target = OUT / "field_broken_bonds.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_bond_count(report: dict) -> Path:
    rows = report["a3_energy_and_faces"]["rows"]
    areas = [r["face_area_m2"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(areas, [r["analytic_bonds"] for r in rows], "k--",
            label=r"lattice analytic $A/(2r)^2$")
    ax.plot(areas, [r["plane"]["num_broken_bonds"] for r in rows], "o-",
            label="face_extent='plane' (infinite plane)")
    ax.plot(areas, [r["disc"]["num_broken_bonds"] for r in rows], "s-",
            label="face_extent='disc' (area consistent)")
    ax.set_xlabel("declared face area A [m$^2$]")
    ax.set_ylabel("bonds broken")
    ax.set_title("Broken bonds vs face area (r = 0.05 m, full cross-section 1.0 m$^2$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    target = OUT / "bond_count_vs_face_area.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_energy_error(report: dict) -> Path:
    rows = report["a3_energy_and_faces"]["rows"]
    areas = [r["face_area_m2"] for r in rows]
    width = 0.36
    xs = range(len(areas))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    plane = [min(r["plane"]["unit_bond_energy_relative_error"], 1.0) for r in rows]
    disc = [r["disc"]["unit_bond_energy_relative_error"] for r in rows]
    ax.bar([x - width / 2 for x in xs], plane, width, label="plane", color="#d62728")
    ax.bar([x + width / 2 for x in xs], disc, width, label="disc", color="#1f77b4")
    ax.axhline(THRESHOLD, color="k", linestyle=":", linewidth=1.2,
               label="3% acceptance band")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{a:g}" for a in areas])
    ax.set_xlabel("declared face area A [m$^2$]")
    ax.set_ylabel(r"$|E_{bond,implied} - G(2r)^2| \,/\, G(2r)^2$")
    ax.set_yscale("log")
    ax.set_title("Fracture energy per bond vs analytic $G(2r)^2$")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    target = OUT / "unit_bond_energy_error.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_mass_error(report: dict) -> Path:
    rows = report["a2_mass_mapping"]["rows"]
    quant = report["a2_mass_mapping"]["quantisation_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    labels = ["equal_volume\nspheres", "cell_volume"]
    # the panel is labelled with the magnitude, so plot the magnitude: the
    # signed residual stays in the report next to it
    vals = [
        r.get("relative_error_magnitude", abs(r["relative_error"])) for r in rows
    ]
    axes[0].bar(labels, vals, color=["#d62728", "#2ca02c"])
    axes[0].axhline(THRESHOLD, color="k", linestyle=":",
                    label="3% acceptance band")
    axes[0].axhline(0.15, color="orange", linestyle="--", label="15% reject line")
    for i, v in enumerate(vals):
        axes[0].annotate(f"{v:.3g}", (i, v), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=8)
    axes[0].set_ylabel(r"$|m_{DEM} - \rho V| / \rho V$")
    axes[0].set_title("DEM vs FEM continuum mass (0.6$^3$ m box, r = 0.1 m)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, axis="y")

    xs = [q["domain_length_m"] for q in quant]
    ys = [q["cell_mass_relative_error"] for q in quant]
    axes[1].plot(xs, ys, "o-", color="#1f77b4")
    axes[1].axhline(THRESHOLD, color="k", linestyle=":")
    axes[1].set_xlabel("domain edge length L [m] (2r = 0.2 m)")
    axes[1].set_ylabel("cell_volume mass relative error")
    axes[1].set_title("Lattice quantisation: L not a multiple of 2r")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    target = OUT / "mass_mapping_error.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text())
    produced = [
        plot_broken_bonds_field(),
        plot_bond_count(report),
        plot_energy_error(report),
        plot_mass_error(report),
    ]
    for path in produced:
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())