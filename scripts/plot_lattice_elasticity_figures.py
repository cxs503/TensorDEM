#!/usr/bin/env python3
"""Regenerate the 4 missing figures for docs/dem3d_lattice_elasticity.md.

Reads data from docs/lattice-elasticity/lattice_elasticity_report.json
and writes PNGs to docs/lattice-elasticity/.

Usage:
    python scripts/plot_lattice_elasticity_figures.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_PATH = "docs/lattice-elasticity/lattice_elasticity_report.json"
OUT_DIR = "docs/lattice-elasticity"


def load_report():
    with open(REPORT_PATH) as f:
        return json.load(f)


def fig1_stiffness_matrices(r):
    """Figure 1: Bond-class rest-length bar chart."""
    bond_classes = r["infinite"]["bond_classes"]
    offsets = [bc["offset"] for bc in bond_classes]
    rest_lengths = [bc["rest_length_over_spacing"] for bc in bond_classes]
    kinds = [bc["kind"] for bc in bond_classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f77b4" if k == "axis" else "#ff7f0e" if k == "face-diagonal" else "#2ca02c"
              for k in kinds]
    labels = [f"({dx},{dy},{dz})\n{k}" for (dx, dy, dz), k in zip(offsets, kinds)]

    ax.barh(range(len(rest_lengths)), rest_lengths, color=colors, edgecolor="white")
    ax.set_yticks(range(len(rest_lengths)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Rest length / lattice spacing (a)")
    ax.set_title("DEM3D 13-neighbour lattice: bond classes and rest lengths")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lattice_stiffness_matrices.png"), dpi=150)
    plt.close()
    print("  -> lattice_stiffness_matrices.png")


def fig2_directional_young_modulus(r):
    """Figure 2: Polar plot of directional Young's modulus in the (100) plane."""
    directional = r["directional"]["100"]
    angles_deg = [d["angle_deg"] for d in directional]
    E_vals = [d["E_Pa"] for d in directional]

    # Mirror to full 0-360 for symmetry (45-degree symmetry in (100) plane)
    angles_full = []
    E_full = []
    for base in [0, 45, 90, 135, 180, 225, 270, 315]:
        angles_full.extend([a + base for a in angles_deg])
        E_full.extend(E_vals)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    theta = np.deg2rad(angles_full)
    ax.plot(theta, E_full, "b-", linewidth=2, label="E(θ) [Pa]")
    ax.fill(theta, E_full, alpha=0.15, color="blue")

    E_min, E_max = min(E_vals), max(E_vals)
    ax.set_title(f"Directional Young's Modulus E(θ) in (100) plane\n"
                 f"E_min={E_min:.0f} Pa, E_max={E_max:.0f} Pa, "
                 f"anisotropy={E_max/E_min:.3f}", fontsize=11)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "directional_young_modulus.png"), dpi=150)
    plt.close()
    print("  -> directional_young_modulus.png")


def fig3_surface_deficiency(r):
    """Figure 3: Surface deficiency — C11/C11_inf vs specimen thickness (nz)."""
    curve = r["surface_deficiency_curve"]
    nz_vals = [c["nz"] for c in curve]
    c11_ratios = [c["C11_over_C11_infinite"] for c in curve]
    c33_ratios = [c["C33_over_C11_infinite"] for c in curve]
    zener = [c["zener_like"] for c in curve]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(nz_vals, c11_ratios, "bo-", label="C11/C11_inf", linewidth=2, markersize=6)
    ax1.plot(nz_vals, c33_ratios, "rs-", label="C33/C11_inf", linewidth=2, markersize=6)
    ax1.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="infinite limit")
    ax1.set_xlabel("Specimen thickness (nz layers)")
    ax1.set_ylabel("Stiffness ratio")
    ax1.set_title("Surface deficiency: finite-size stiffness reduction")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(nz_vals, zener, "g^-", linewidth=2, markersize=6)
    ax2.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="isotropic (Zener=1)")
    ax2.set_xlabel("Specimen thickness (nz layers)")
    ax2.set_ylabel("Zener anisotropy ratio")
    ax2.set_title("Anisotropy vs specimen thickness")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "surface_deficiency.png"), dpi=150)
    plt.close()
    print("  -> surface_deficiency.png")


def fig4_phonon_dispersion(r):
    """Figure 4: Phonon dispersion curves along [100]."""
    dispersion = r["dispersion"]
    q_vals = [d["q_over_pi_over_a"] for d in dispersion]
    w1 = [d["omega1"] for d in dispersion]
    w2 = [d["omega2"] for d in dispersion]
    w3 = [d["omega3"] for d in dispersion]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(q_vals, w1, "b-", linewidth=1.5, label="ω₁ (transverse, degenerate)")
    ax.plot(q_vals, w2, "b--", linewidth=1.2, alpha=0.6)
    ax.plot(q_vals, w3, "r-", linewidth=1.5, label="ω₂ (longitudinal)")

    ax.set_xlabel("q / (π/a)")
    ax.set_ylabel("ω [rad/s]")
    ax.set_title("Phonon dispersion along [100] direction\n"
                 "13-neighbour simple-cubic lattice, k=2000 N/m")
    ax.axvline(1.0, color="k", linestyle=":", alpha=0.4, label="BZ boundary (q=π/a)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phonon_dispersion.png"), dpi=150)
    plt.close()
    print("  -> phonon_dispersion.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    r = load_report()
    print(f"Loaded report: {r['verdict']} ({r['pass_count']}/{r['case_count']} checks)")
    print("Generating figures...")
    fig1_stiffness_matrices(r)
    fig2_directional_young_modulus(r)
    fig3_surface_deficiency(r)
    fig4_phonon_dispersion(r)
    print("Done. All 4 figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
