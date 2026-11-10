"""Generate verification figures for campaign p224w2_n06."""
from __future__ import annotations

import math
import json
import base64
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.thermal.lattice_conductivity import (
    lattice_porosity,
    lattice_coordination,
    lattice_constant,
    lattice_effective_conductivity,
)
from tensordem.thermal.packed_bed import (
    zehner_schluender_conductivity,
    wiener_bounds,
)
from tensordem.terramechanics.bekker import BekkerModel
from tensordem.terramechanics.janosi import JanosiModel
from tensordem.terramechanics.coupling import track_drawbar_pull


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_lattice_conductivity_fig() -> dict:
    """Figure 1: Lattice effective conductivity vs contact radius."""
    r = 0.005
    ks = 2.2
    kf = 0.026

    a_c_over_R = np.linspace(0.005, 0.15, 100)
    fig, ax = plt.subplots(figsize=(8, 5))

    for lt, color, ls in [("SC", "blue", "-"), ("BCC", "green", "--"), ("FCC", "red", "-.")]:
        k_vals = [
            lattice_effective_conductivity(lt, r, a * r, ks, kf)
            for a in a_c_over_R
        ]
        ax.plot(a_c_over_R, k_vals, color=color, linestyle=ls, linewidth=2,
                label=f"{lt} (Z={lattice_coordination(lt)}, φ={lattice_porosity(lt):.3f})")

    # ZS reference at phi=0.40
    k_zs = zehner_schluender_conductivity(0.40, ks, kf)
    ax.axhline(k_zs, color="gray", linestyle=":", linewidth=1.5,
               label=f"ZS (φ=0.40): {k_zs:.4g}")

    ax.set_xlabel("Contact radius / Sphere radius (a_c / R)", fontsize=12)
    ax.set_ylabel("Effective thermal conductivity (W/m/K)", fontsize=12)
    ax.set_title("Lattice Effective Thermal Conductivity vs Contact Radius", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    png_b64 = fig_to_base64(fig)
    plt.close(fig)
    return {
        "name": "lattice_conductivity_vs_contact",
        "caption": "Effective thermal conductivity of SC/BCC/FCC sphere lattices as a function of normalized contact radius a_c/R. ZS correlation at φ=0.40 shown for reference.",
        "png_base64": png_b64,
    }


def generate_bekker_fig() -> dict:
    """Figure 2: Bekker pressure-sinkage curves for different soils."""
    z = np.linspace(0.001, 0.15, 200)
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, model, color in [
        ("Clay", BekkerModel.clay(), "blue"),
        ("Sandy loam", BekkerModel.sandy_loam(), "green"),
        ("Dry sand", BekkerModel.dry_sand(), "red"),
    ]:
        p = [model.pressure(zi) for zi in z]
        ax.plot(z * 100, p, color=color, linewidth=2,
                label=f"{name} (k_c={model.k_c:.0f}, k_φ={model.k_phi:.0f}, n={model.n})")

    ax.set_xlabel("Sinkage (cm)", fontsize=12)
    ax.set_ylabel("Pressure (Pa)", fontsize=12)
    ax.set_title("Bekker Pressure–Sinkage Relations", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_b64 = fig_to_base64(fig)
    plt.close(fig)
    return {
        "name": "bekker_pressure_sinkage",
        "caption": "Bekker pressure-sinkage curves for clay, sandy loam, and dry sand. Parameters from Wong (2001).",
        "png_base64": png_b64,
    }


def generate_janosi_fig() -> dict:
    """Figure 3: Janosi shear-traction curves."""
    j = np.linspace(0, 0.15, 200)
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, model, color in [
        ("Clay", JanosiModel.clay(), "blue"),
        ("Sandy loam", JanosiModel.sandy_loam(), "green"),
        ("Dry sand", JanosiModel.dry_sand(), "red"),
    ]:
        p = 50000.0  # 50 kPa
        curve = model.shear_curve(p, j_max=0.15, n_points=200)
        ax.plot(curve["j"] * 100, curve["tau"] / 1000, color=color, linewidth=2,
                label=f"{name} (c={model.cohesion:.0f} Pa, φ={model.friction_angle_deg}°, K={model.K} m)")

    ax.set_xlabel("Shear displacement (cm)", fontsize=12)
    ax.set_ylabel("Shear stress (kPa)", fontsize=12)
    ax.set_title("Janosi Shear–Traction Curves (p = 50 kPa)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_b64 = fig_to_base64(fig)
    plt.close(fig)
    return {
        "name": "janosi_shear_traction",
        "caption": "Janosi shear-traction curves for clay, sandy loam, and dry sand at 50 kPa normal pressure.",
        "png_base64": png_b64,
    }


def generate_slip_dp_fig() -> dict:
    """Figure 4: Drawbar pull vs slip ratio."""
    s = np.linspace(0.01, 0.99, 100)
    fig, ax = plt.subplots(figsize=(8, 5))

    weight = 50000.0
    b = 0.3
    L = 2.0
    kc = 0.0
    kph = 1e6
    n = 1.0
    K = 0.03

    for name, c, phi_deg, color in [
        ("Firm soil (c=5 kPa, φ=30°)", 5000.0, 30.0, "blue"),
        ("Soft soil (c=2 kPa, φ=20°)", 2000.0, 20.0, "green"),
        ("Cohesive soil (c=10 kPa, φ=15°)", 10000.0, 15.0, "red"),
    ]:
        phi_rad = math.radians(phi_deg)
        dp = np.array([track_drawbar_pull(weight, b, L, si, kc, kph, n, c, phi_rad, K) for si in s])
        ax.plot(s * 100, dp / 1000, color=color, linewidth=2, label=name)

    ax.set_xlabel("Slip ratio (%)", fontsize=12)
    ax.set_ylabel("Drawbar pull (kN)", fontsize=12)
    ax.set_title("Drawbar Pull vs Slip Ratio (50 kN vehicle)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_b64 = fig_to_base64(fig)
    plt.close(fig)
    return {
        "name": "drawbar_pull_vs_slip",
        "caption": "Drawbar pull as a function of slip ratio for a 50 kN tracked vehicle on different soil types.",
        "png_base64": png_b64,
    }


def main():
    figures = [
        generate_lattice_conductivity_fig(),
        generate_bekker_fig(),
        generate_janosi_fig(),
        generate_slip_dp_fig(),
    ]

    # Save figures as PNG files
    import os
    outdir = "/root/tp-agent/work/m9915/artifacts/p224w2_n06"
    os.makedirs(outdir, exist_ok=True)

    for fig_data in figures:
        png_path = os.path.join(outdir, f"{fig_data['name']}.png")
        with open(png_path, "wb") as f:
            f.write(base64.b64decode(fig_data["png_base64"]))
        print(f"Saved: {png_path}")

    # Save figure metadata
    meta_path = os.path.join(outdir, "figures.json")
    with open(meta_path, "w") as f:
        json.dump(figures, f, indent=2)
    print(f"Saved: {meta_path}")

    return figures


if __name__ == "__main__":
    main()
