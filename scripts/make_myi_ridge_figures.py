#!/usr/bin/env python
"""Figures for the MYI ridge keel interaction campaign."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensordem.myi_ridge import geometry, panel, scaling  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "myi_ridge"
FIGURES = ROOT / "docs" / "myi-ridge-keel-interaction" / "figures"


def load() -> dict:
    return json.loads((ARTIFACTS / "myi_ridge_results.json").read_text())


def fig_geometry(results: dict) -> Path:
    ensemble = results["geometry"]["ensemble"]
    stats = geometry.MyiRidgeStatistics()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    edges = np.asarray(ensemble["keel_depth_histogram"]["edges"])
    counts = np.asarray(ensemble["keel_depth_histogram"]["counts"])
    axes[0].bar(0.5 * (edges[1:] + edges[:-1]), counts, width=np.diff(edges) * 0.9,
                color="#3b6ea5", label="sampled keel depths")
    axes[0].axvline(ensemble["keel_depth_mean_target"], color="k", ls="--",
                    label=f"target mean {ensemble['keel_depth_mean_target']:.1f} m")
    axes[0].axvline(ensemble["keel_depth_mean_sampled"], color="#c44e52", ls=":",
                    label=f"sampled mean {ensemble['keel_depth_mean_sampled']:.2f} m")
    axes[0].set_xlabel("keel depth h_keel [m]")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=7)
    axes[0].set_title("(a) keel-depth distribution")

    sections = results["geometry"]["sections"]
    target = [s["porosity_target"] for s in sections]
    measured = [s["porosity_measured"] for s in sections]
    depths = [s["keel_depth"] for s in sections]
    scatter = axes[1].scatter(target, measured, c=depths, cmap="viridis", s=45)
    axes[1].plot([0.18, 0.42], [0.18, 0.42], "k--", lw=0.8)
    axes[1].fill_between([0.18, 0.42], [0.16, 0.40], [0.20, 0.44], color="grey", alpha=0.2,
                         label="+/- 0.02 acceptance band")
    axes[1].set_xlabel("target porosity n")
    axes[1].set_ylabel("measured porosity")
    axes[1].set_title("(b) porosity reconstruction")
    axes[1].legend(fontsize=7)
    fig.colorbar(scatter, ax=axes[1], label="h_keel [m]")

    section = geometry.build_keel_section(20.0, 0.30, stats, seed=20261108)
    for centre, radius, is_block in zip(section.centers, section.radii, section.is_block):
        axes[2].add_patch(plt.Circle((centre[0], centre[1]), radius,
                                     color="#2b5d8a" if is_block else "#9ecae1",
                                     ec="k" if is_block else "none", lw=0.2))
    half = section.keel_width / 2.0
    axes[2].plot([-half, 0.0, half], [0.0, -section.keel_depth, 0.0], "k-", lw=1.0)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("depth below waterline [m]")
    axes[2].set_title("(c) reconstructed keel section (h=20 m, n=0.30)")
    fig.tight_layout()
    path = FIGURES / "geometry_statistics.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_force_traces(results: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    for record in results["cases"]:
        advance = np.asarray(record["force_trace_advance"])
        force = np.asarray(record["force_trace_force"]) / 1e6
        if abs(record["h_keel"] - results["base"]["h_keel"]) < 1e-9 and \
                abs(record["sigma_ice"] - results["base"]["sigma_ice"]) < 1e-6:
            axes[0].plot(advance, force, lw=1.3, label=record["label"])
        axes[1].plot(advance, force, lw=0.9, label=record["label"])
    for axis, title in ((axes[0], "(a) base ice strength, h = %.0f m" % results["base"]["h_keel"]),
                        (axes[1], "(b) all sweep cases")):
        axis.set_xlabel("hull advance [m]")
        axis.set_ylabel("plate force per unit width [MN/m]")
        axis.set_title(title)
        axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = FIGURES / "dem_force_traces.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_scaling(results: dict) -> Path:
    cases = results["cases"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    specs = (
        ("sigma_ice", {"h_keel": 20.0, "hull_velocity": 2.0}, "sigma_ice [Pa]"),
        ("h_keel", {"sigma_ice": 7.5e6, "hull_velocity": 2.0}, "h_keel [m]"),
        ("hull_velocity", {"sigma_ice": 7.5e6, "h_keel": 20.0}, "V [m/s]"),
    )
    for (key, fixed, label), axis in zip(specs, axes):
        selected = [c for c in cases if all(abs(c[k] - v) < 1e-9 for k, v in fixed.items())]
        selected.sort(key=lambda c: c[key])
        x = np.asarray([c[key] for c in selected])
        peak = np.asarray([c["F_max_peak"] for c in selected]) / 1e6
        steady = np.asarray([c["F_max_steady"] for c in selected]) / 1e6
        axis.plot(x, peak, "o-", color="#c44e52", label="F_max (peak)")
        axis.plot(x, steady, "s--", color="#3b6ea5", label="F_max (steady)")
        axis.set_xscale("log")
        axis.set_yscale("log")
        if key in ("sigma_ice", "h_keel") and x.size >= 3:
            fit = scaling.fit_power_law(x, np.asarray([c["F_max_peak"] for c in selected]))
            axis.set_title(f"{label.split()[0]}: exponent {fit['exponent']:.2f}"
                           f" +/- {fit['exponent_stderr']:.2f}")
        else:
            axis.set_title(f"{label.split()[0]} dependence")
        axis.set_xlabel(label)
        axis.set_ylabel("F_max = p h_keel L_c [MN]")
        axis.legend(fontsize=7)
    fig.tight_layout()
    path = FIGURES / "load_scaling.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_panel(results: dict) -> Path:
    pressure = float(results["design_pressure"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    thicknesses = np.linspace(0.015, 0.090, 20)
    for spacing, colour in ((2.4, "#c44e52"), (3.0, "#3b6ea5"), (3.6, "#55a868")):
        stress = [panel.solve_panel(spacing, spacing, thickness, pressure,
                                    elements_x=10, elements_y=10).stress_max / 1e6
                  for thickness in thicknesses]
        axes[0].plot(thicknesses * 1000.0, stress, color=colour,
                     label=f"frame spacing {spacing:.1f} m")
    allowable = panel.STEEL_EH36["yield_strength"] / 1.15 / 1e6
    axes[0].axhline(allowable, color="k", ls="--",
                    label=f"allowable {allowable:.0f} MPa (EH36/1.15)")
    axes[0].set_xlabel("plate thickness [mm]")
    axes[0].set_ylabel("peak bending stress [MPa]")
    axes[0].set_title(f"(a) ice-belt panel at p = {pressure/1e6:.2f} MPa")
    axes[0].legend(fontsize=7)

    young, nu = panel.STEEL_EH36["young_modulus"], panel.STEEL_EH36["poisson_ratio"]
    q, side, thickness = 1.0e4, 1.0, 0.030
    d0 = panel.plate_rigidity(young, thickness, nu)
    meshes = list(range(4, 17, 2))
    ratios = [panel.solve_panel(side, side, thickness, q, elements_x=m, elements_y=m,
                                boundary="clamped").deflection_centre
              / (0.00126 * q * side ** 4 / d0) for m in meshes]
    axes[1].plot(meshes, ratios, "o-", label="clamped, w_centre / Timoshenko")
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("elements per side")
    axes[1].set_ylabel("FEM / analytic")
    axes[1].set_title("(b) plate-panel FEM verification")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = FIGURES / "structural_check.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = load()
    paths = [fig_geometry(results), fig_force_traces(results), fig_scaling(results),
             fig_panel(results)]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()