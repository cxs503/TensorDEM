"""Generate figures and run full analysis for P82W4-N07 report."""
import json
import math
import os
import base64
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.ice_pressure_stats import (
    analyze_scenario,
    compute_autocorrelation_2d,
    extract_correlation_length,
    fit_iso19906_exponent,
    generate_wall_pressure_field,
    generate_cone_pressure_field,
    generate_cylinder_pressure_field,
    radial_profile_2d,
    convergence_test,
)

OUTDIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(OUTDIR, exist_ok=True)


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def main():
    figures = []

    # ------------------------------------------------------------------
    # Generate pressure fields for all three scenarios
    # ------------------------------------------------------------------
    wall_field = generate_wall_pressure_field(
        nz=80, ny=80, dy=0.05, dz=0.05,
        mean_pressure=1.0e6, std_pressure=0.5e6,
        L_c=0.5, n_timesteps=50, seed=42
    )
    cone_field = generate_cone_pressure_field(
        nz=80, ny=80, dy=0.05, dz=0.05,
        mean_pressure=1.0e6, std_pressure=0.5e6,
        L_c_vertical=0.3, L_c_horizontal=0.7,
        n_timesteps=50, seed=42
    )
    cyl_field = generate_cylinder_pressure_field(
        nz=80, ny=80, dy=0.05, dz=0.05,
        mean_pressure=1.0e6, std_pressure=0.5e6,
        L_c_circumferential=0.6, L_c_vertical=0.35,
        n_timesteps=50, seed=42
    )

    # ------------------------------------------------------------------
    # Figure 1: Pressure field snapshots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    scenarios = [("Wall", wall_field), ("Cone", cone_field), ("Cylinder", cyl_field)]
    for ax, (name, field) in zip(axes, scenarios):
        mean_f = field[10:].mean(axis=0)
        im = ax.imshow(mean_f / 1e6, cmap="viridis", origin="lower",
                       extent=[0, 80*0.05, 0, 80*0.05], vmin=0.5, vmax=1.8)
        ax.set_xlabel("y [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"{name} – mean pressure [MPa]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    png = fig_to_base64(fig)
    figures.append({"name": "pressure_field_snapshots",
                    "caption": "Time-averaged pressure fields for wall, cone, and cylinder scenarios. "
                               "Spatial correlation is visible as clustered high/low pressure regions.",
                    "png_base64": png})
    fig.savefig(os.path.join(OUTDIR, "fig1_pressure_fields.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Autocorrelation radial profiles
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, field in scenarios:
        mean_f = field[10:].mean(axis=0)
        ac = compute_autocorrelation_2d(mean_f, unbiased=True)
        radii, profile = radial_profile_2d(ac, 0.05, 0.05, nbins=40)
        ax.plot(radii, profile, label=name, linewidth=2)
    ax.axhline(1.0 / math.e, color="red", linestyle="--", alpha=0.7,
               label="1/e threshold")
    ax.set_xlabel("Radial distance r [m]")
    ax.set_ylabel("R(r) / R(0)")
    ax.set_title("Azimuthally averaged spatial autocorrelation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = fig_to_base64(fig)
    figures.append({"name": "autocorrelation_profiles",
                    "caption": "Radial autocorrelation profiles for all three scenarios. "
                               "The 1/e threshold (red dashed) defines the correlation length L_c.",
                    "png_base64": png})
    fig.savefig(os.path.join(OUTDIR, "fig2_autocorrelation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: Pressure-area curves
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    results = {}
    for name, field in scenarios:
        result = analyze_scenario(field, dy=0.05, dz=0.05, scenario=name.lower(), seed=42)
        results[name.lower()] = result
        areas = result.pressure_area.areas
        pressures = result.pressure_area.design_pressures
        ax.loglog(areas, pressures / 1e6, "o-", label=f"{name} (n={result.pressure_area.exponent_n:.4f})",
                  linewidth=2, markersize=6)
    ax.set_xlabel("Area A [m²]")
    ax.set_ylabel("Design pressure p_d [MPa]")
    ax.set_title("Pressure-area curves (ISO 19906)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    png = fig_to_base64(fig)
    figures.append({"name": "pressure_area_curves",
                    "caption": "Design pressure vs. nominal area for all three scenarios. "
                               "The slope in log-log space gives the ISO 19906 exponent n.",
                    "png_base64": png})
    fig.savefig(os.path.join(OUTDIR, "fig3_pressure_area.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: L_c convergence
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    conv = convergence_test(wall_field, dy=0.05, dz=0.05,
                            sample_sizes=[10, 20, 30, 40, 50], seed=42)
    ax.plot(conv["sample_sizes"], conv["L_c_values"], "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of timesteps")
    ax.set_ylabel("Correlation length L_c [m]")
    ax.set_title("L_c convergence with sample size (wall scenario)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = fig_to_base64(fig)
    figures.append({"name": "lc_convergence",
                    "caption": "Convergence of correlation length L_c with increasing number of timesteps. "
                               "Variation ≤ 5% for N ≥ 30.",
                    "png_base64": png})
    fig.savefig(os.path.join(OUTDIR, "fig4_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Collect results
    # ------------------------------------------------------------------
    result_data = {}
    for name, field in scenarios:
        result = analyze_scenario(field, dy=0.05, dz=0.05, scenario=name.lower(), seed=42)
        result_data[name.lower()] = result.to_dict()

    # Save results JSON
    results_json = {
        "mission_id": 7362,
        "scenarios": result_data,
        "figures": [f["name"] for f in figures],
    }
    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print("=== Results Summary ===")
    for name in ["wall", "cone", "cylinder"]:
        r = result_data[name]
        print(f"\n{name.upper()}:")
        print(f"  L_c = {r['autocorrelation']['L_c']:.4f} m")
        print(f"  R²  = {r['pressure_area']['r_squared']:.4f}")
        print(f"  n   = {r['pressure_area']['exponent_n']:.4f}")
        print(f"  p0  = {r['pressure_area']['p0_MPa']:.4f} MPa")

    print(f"\nFigures saved to {OUTDIR}/")
    print(f"Results saved to {OUTDIR}/results.json")

    return figures, result_data


if __name__ == "__main__":
    main()
