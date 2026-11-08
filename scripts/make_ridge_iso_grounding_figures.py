"""Figures for the ISO 19906 / DEM ridge-grounding node (IN155-N11).

Every figure is generated from the committed case JSONs and the merged ledger;
no number is typed by hand.  Run after ``scripts/run_ridge_iso_grounding.py``.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.ridge_dem import RidgeKeel, RidgeMaterial  # noqa: E402
from tensordem.ridge_geotech import MohrCoulombSoil, PipeSection  # noqa: E402
from tensordem.ridge_grounding import GroundingScenario, run_grounding  # noqa: E402

CASES = os.path.join(ROOT, "artifacts", "ridge_iso_grounding", "cases")
OUT = os.path.join(ROOT, "artifacts", "ridge_iso_grounding", "figures")


def load(name: str) -> dict:
    with open(os.path.join(CASES, f"{name}.json")) as handle:
        return json.load(handle)


def ledger() -> dict:
    with open(os.path.join(ROOT, "artifacts", "ridge_iso_grounding",
                           "ledger.json")) as handle:
        return json.load(handle)


def figure_load_displacement() -> str:
    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    specs = [("cover_d1_compliant", "d/D = 1, undrained", "C0"),
             ("cover_d2_compliant", "d/D = 2, undrained", "C1"),
             ("cover_d1_compliant_drained", "d/D = 1, drained (phi=30)", "C2")]
    for name, label, colour in specs:
        record = load(name)
        load_curve = np.asarray(record["loads"]["pipe_load"]) / 1e3
        displacement = np.asarray(record["loads"]["pipe_displacement"])
        if displacement.max() <= 0.0:
            continue
        axes.plot(displacement, load_curve, colour, lw=1.0, alpha=0.55)
        axes.plot(displacement, np.asarray(record["metrics"]["peak_load_per_length"]
                                           * np.ones_like(displacement)) / 1e3,
                  colour, ls=":", lw=1.0)
        axes.annotate(label, xy=(displacement[np.argmax(load_curve)],
                                 load_curve.max()), xytext=(4, 6),
                      textcoords="offset points", color=colour, fontsize=8)
        iso = record["iso"]["vertical_total_n_per_m"] / 1e3
        axes.axhline(iso, color=colour, ls="--", lw=0.8, alpha=0.8)
    axes.set_xlabel("pipe vertical displacement w [m]")
    axes.set_ylabel("vertical line load V [kN/m]")
    axes.set_title("DEM load-displacement (solid), peak (dotted) and ISO 19906 form (dashed)")
    axes.grid(alpha=0.3)
    path = os.path.join(OUT, "load_displacement.png")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_cover_envelope() -> str:
    data = ledger()
    rows = data["cover_envelope"]["rows"]
    ratios = [row["cover_over_d"] for row in rows]
    dem = [row["dem_capacity_n_per_m"] / 1e3 for row in rows]
    iso = [row["iso_vertical_n_per_m"] / 1e3 for row in rows]
    demand = [row["dem_demand_n_per_m"] / 1e3 for row in rows]
    disp = [row["vertical_displacement_m"] for row in rows]
    figure, axes = plt.subplots(figsize=(7.2, 4.8))
    axes.plot(ratios, iso, "k--", marker="s", label="ISO 19906 form V_ISO")
    axes.plot(ratios, dem, "C0-o", label="DEM capacity (compliant pipe)")
    axes.plot(ratios, demand, "C3-^", label="DEM ice-side demand (rigid pipe)")
    for ratio, value, displacement in zip(ratios, dem, disp):
        if value == 0.0:
            axes.annotate("no keel-pipe\ncontact", xy=(ratio, 5), ha="center",
                          fontsize=8, color="grey")
    axes.set_xlabel("cover depth d / D")
    axes.set_ylabel("vertical line load [kN/m]")
    axes.set_yscale("symlog", linthresh=20.0)
    axes.grid(alpha=0.3)
    twin = axes.twinx()
    twin.plot(ratios, [d / data["criteria"]["c_cover_instability"]["vertical_limit_m"]
                       for d in disp], "C2:", marker="o", lw=1.0,
              label="w / (D/10)")
    twin.set_ylabel("settlement over the instability limit (w/(D/10))")
    twin.set_yscale("symlog", linthresh=0.1)
    lines, labels = axes.get_legend_handles_labels()
    extra, extra_labels = twin.get_legend_handles_labels()
    axes.legend(lines + extra, labels + extra_labels, fontsize=8, loc="upper left")
    axes.set_title("Cover-depth envelope: capacity, demand and settlement")
    path = os.path.join(OUT, "cover_envelope.png")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_grid_convergence() -> str:
    data = ledger()
    rows = data["grid"]["rows"]
    particles = [row["particles"] for row in rows]
    capacity = [row["capacity"] / 1e3 for row in rows]
    figure, axes = plt.subplots(figsize=(6.0, 4.2))
    axes.plot(particles, capacity, "C0-o")
    for count, value in zip(particles, capacity):
        axes.annotate(f"{value:.1f} kN/m", xy=(count, value), xytext=(6, -12),
                      textcoords="offset points", fontsize=8)
    change = data["grid"]["capacity_change"]
    axes.set_title(f"Grid independence: capacity change {100 * change:+.2f} % "
                   f"for {data['grid']['particle_ratio']:.2f}x particles")
    axes.set_xlabel("particle count")
    axes.set_ylabel("DEM capacity [kN/m]")
    axes.grid(alpha=0.3)
    path = os.path.join(OUT, "grid_convergence.png")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_energy_closure() -> str:
    data = ledger()
    rows = data["energy"]["rows"]
    dt = [row["dt"] for row in rows]
    closure = [100.0 * row["closure_error"] for row in rows]
    figure, axes = plt.subplots(figsize=(6.0, 4.2))
    axes.loglog(dt, closure, "C3-o", label="whole record")
    axes.axhline(5.0, color="k", ls="--", lw=0.8, label="campaign tolerance 5 %")
    for factor, record_name in ((1.0, "energy_dt1"), (0.5, "energy_dt0.5"),
                                (0.25, "energy_dt0.25")):
        record = load(record_name)
        window = record["energy"]["windows"]
        axes.plot(record["dt"], 100.0 * window["closure_error"], "C0:s",
                  label="loading window (2nd half)" if factor == 1.0 else None)
        axes.annotate(f"{100 * window['closure_error']:.0f} %",
                      xy=(record["dt"], 100 * window["closure_error"]),
                      xytext=(6, 4), textcoords="offset points", fontsize=8)
    axes.set_xlabel("time step dt [s]")
    axes.set_ylabel("energy closure error [%]")
    axes.set_title("Energy-ledger closure vs time step (honest FAIL at 5 %)")
    axes.grid(alpha=0.3, which="both")
    axes.legend(fontsize=8)
    path = os.path.join(OUT, "energy_closure.png")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_lateral() -> str:
    data = ledger()
    rows = data["lateral"]["rows"]
    labels = [f"d/D={row['cover_over_d']:g}\n{row['soil']}" for row in rows]
    peak = [row["dem_lateral_peak_n_per_m"] / 1e3 for row in rows]
    sustained = [row["dem_lateral_sustained_n_per_m"] / 1e3 for row in rows]
    iso = [row["iso_lateral_n_per_m"] / 1e3 for row in rows]
    index = np.arange(len(rows))
    figure, axes = plt.subplots(figsize=(7.4, 4.4))
    axes.bar(index - 0.25, peak, width=0.25, label="DEM peak (impulse)", color="C3")
    axes.bar(index, sustained, width=0.25, label="DEM sustained (2nd half)", color="C0")
    axes.bar(index + 0.25, iso, width=0.25, label="ISO 19906 H (N_p s_u D)", color="0.6")
    axes.set_xticks(index)
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_ylabel("horizontal line load [kN/m]")
    axes.set_title("Lateral loading: rubble impact vs soil break-out capacity")
    axes.grid(alpha=0.3, axis="y")
    axes.legend(fontsize=8)
    path = os.path.join(OUT, "lateral_loads.png")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_field() -> str:
    """Early-loading field: particle force magnitude, mudline, pipe and subgrade."""
    soil = MohrCoulombSoil(friction_angle_deg=0.0, cohesion=30.0e3,
                           saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                           poisson_ratio=0.3, water_unit_weight=10.06e3,
                           undrained=True)
    section = PipeSection(outer_diameter=0.914, wall_thickness=0.0159,
                          young_modulus=207.0e9, yield_strength=448.0e6)
    material = RidgeMaterial(particle_radius=0.5, drag=2.0e4,
                             freeze_bond_fraction=1.0,
                             bond_tensile_strength=100.0e3,
                             bond_shear_strength=200.0e3)
    scenario = GroundingScenario(
        keel=RidgeKeel(12.0, 24.0, 0.10), material=material, soil=soil,
        section=section, cover_depth=section.outer_diameter, tip_clearance=0.10,
        stroke=0.10 + section.outer_diameter + 1.1, platen_velocity=0.3,
        pipe_response="compliant", sample_every=10)
    result = run_grounding(scenario, max_steps=6000)
    snapshot = result.final_snapshot
    positions = np.asarray(snapshot["positions"])
    magnitude = np.asarray(snapshot["force_magnitude"]) / 1e3
    figure, axes = plt.subplots(figsize=(7.6, 4.8))
    scatter = axes.scatter(positions[:, 0], positions[:, 1], c=magnitude,
                           s=18, cmap="viridis")
    axis = np.linspace(-0.75, 0.75, 100)
    centre = -scenario.pipe_centre_height * 0
    axes.plot(axis, np.sqrt(np.clip(0.5 * section.outer_diameter ** 2
                                    - axis ** 2, 0.0)) - (scenario.cover_depth
                                                          + 0.5 * section.outer_diameter),
              "w-", lw=1.0)
    axes.plot(axis, -np.sqrt(np.clip(0.5 * section.outer_diameter ** 2
                                     - axis ** 2, 0.0)) - (scenario.cover_depth
                                                           + 0.5 * section.outer_diameter),
              "w-", lw=1.0)
    axes.axhline(0.0, color="C0", ls="--", lw=1.0, label="mudline")
    axes.set_xlim(-8.0, 8.0)
    axes.set_ylim(-(scenario.cover_depth + 1.0), 6.0)
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_title(f"Keel particles and contact force magnitude at "
                   f"t = {snapshot['time']:.2f} s ({snapshot['broken_bonds']} bonds broken)")
    figure.colorbar(scatter, ax=axes, label="|F| on particle [kN]")
    axes.legend(fontsize=8, loc="upper right")
    axes.grid(alpha=0.2)
    path = os.path.join(OUT, "field_early_loading.png")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(6)
    paths = [figure_load_displacement(), figure_cover_envelope(),
             figure_grid_convergence(), figure_energy_closure(),
             figure_lateral(), figure_field()]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()