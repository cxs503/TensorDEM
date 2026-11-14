#!/usr/bin/env python3
"""Generate figures for CASE-DEM-OEDOMETER-STIFF benchmark."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ART = REPO / "artifacts" / "case_dem_oedometer_stiff"
FIG = ART / "figures"


def figure_stress_strain() -> str:
    """Stress-strain curve from the two load stages."""
    dem_path = ART / "dem_result.json"
    dem = json.loads(dem_path.read_text())
    stages = dem["stages"]

    stress = [s["stress"] for s in stages]
    height = [s["height"] for s in stages]
    h0 = height[0]
    strain = [(h0 - h) / h0 for h in height]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(strain, [s / 1e3 for s in stress], "o-", color="C0", ms=8, lw=2)
    
    # Linear fit
    p = np.polyfit(strain, stress, 1)
    strain_fit = np.linspace(0, strain[-1] * 1.1, 100)
    ax.plot(strain_fit, np.polyval(p, strain_fit) / 1e3, "--", color="C3", lw=1.5,
            label=f"E_oed = {p[0]/1e6:.3f} MPa")

    ax.set_xlabel(r"Vertical strain $\varepsilon_v$")
    ax.set_ylabel("Vertical stress $\sigma_v$ (kPa)")
    ax.set_title("Oedometer stress-strain curve")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Annotate points
    for i, (s, e) in enumerate(zip(strain, stress)):
        ax.annotate(f"σ={s/1e3:.1f} kPa\nn={stages[i]['porosity']:.3f}\nZ={stages[i]['coordination']:.1f}",
                    (e, s/1e3), textcoords="offset points", xytext=(8, 8), fontsize=7)

    fig.tight_layout()
    path = FIG / "stress_strain.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_comparison() -> str:
    """Bar chart comparing DEM vs micromechanical E_oed."""
    dem_path = ART / "dem_result.json"
    ref_path = ART / "reference.json"
    dem = json.loads(dem_path.read_text())
    ref = json.loads(ref_path.read_text())

    E_dem = dem["E_oed"]
    E_mm = ref["micromechanical"]["E_oed"]
    E_sc = ref["simple_cubic_limit"]["E_oed"]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    labels = ["DEM\n(this work)", "Micromechanical\n(Chang et al.)", "Simple Cubic\n(kn/d)"]
    values = [E_dem / 1e6, E_mm / 1e6, E_sc / 1e6]
    colors = ["C0", "C1", "C3"]

    bars = ax.bar(labels, values, color=colors, edgecolor="k", lw=0.5)
    ax.set_ylabel("E_oed (MPa)")
    ax.set_title("Constrained modulus: DEM vs reference")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Add error annotation
    error_pct = (E_dem - E_mm) / E_mm * 100
    ax.annotate(f"Error: {error_pct:.1f}%", xy=(0.5, 0.85), xycoords="axes fraction",
                fontsize=10, ha="center", color="C3" if abs(error_pct) > 10 else "C2")

    fig.tight_layout()
    path = FIG / "comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_packing() -> str:
    """Schematic of the DEM packing."""
    dem_path = ART / "dem_result.json"
    dem = json.loads(dem_path.read_text())
    cfg = dem["config"]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    # Draw box
    w = cfg["box_width"]
    h = dem["stages"][1]["height"]  # final height
    rect = plt.Rectangle((0, 0), w, h, fill=False, edgecolor="k", lw=2)
    ax.add_patch(rect)

    # Draw particles (schematic)
    n_row = cfg["n_row"]
    n_layer = cfg["n_layer"]
    r = cfg["radius"]
    np.random.seed(cfg["seed"])

    # Generate random positions within the box
    for _ in range(cfg["particle_count"]):
        x = np.random.uniform(r, w - r)
        y = np.random.uniform(r, h - r)
        circle = plt.Circle((x, y), r, fill=True, facecolor="C0", alpha=0.6, edgecolor="k", lw=0.5)
        ax.add_patch(circle)

    ax.set_xlim(-r, w + r)
    ax.set_ylim(-r, h + r)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"DEM oedometer cell: {cfg['particle_count']} particles, "
                 f"n={dem['porosity']:.3f}, Z={dem['coordination']:.1f}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = FIG / "packing.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    f1 = figure_stress_strain()
    print(f"  {f1}")
    f2 = figure_comparison()
    print(f"  {f2}")
    f3 = figure_packing()
    print(f"  {f3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
