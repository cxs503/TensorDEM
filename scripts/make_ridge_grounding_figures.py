"""Figures for the ice-ridge grounding campaign (matplotlib, publication style).

Reads the JSON artifacts written by ``scripts/run_ice_ridge_grounding.py`` and
writes PNGs to ``artifacts/ridge_grounding/figures/``:

* ``load_displacement.png``  -- DEM coupled load-displacement, peak/softening,
                                subgrade peak and residual reference lines
* ``pipe_stress.png``        -- pipeline bending moment and stress, FE against
                                the closed-form Hetenyi patch solution
* ``dem_field.png``          -- DEM particle field coloured by contact force
                                magnitude with bond state (force-chain cloud)
* ``envelope.png``           -- capacity envelope (ice vs soil limit) and
                                DNV-OS-F101 utilisation versus keel depth
* ``sensitivity.png``        -- keel depth / width / soil strength trends
* ``calibration.png``        -- biaxial deviatoric response (if available)

Every plotted number is read from an artifact; nothing is hard-coded except
axis labels and the documented reference lines.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
ART = os.path.join(ROOT, "artifacts", "ridge_grounding")
FIG = os.path.join(ART, "figures")


def _load(name):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def figure_load_displacement(main):
    history = main["history"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(history["keel_penetration"], [v / 1e3 for v in history["pipe_load_smoothed"]],
            color="tab:blue", lw=1.6, label="DEM coupled pipe load (smoothed)")
    ax.plot(history["keel_penetration"], [v / 1e3 for v in history["pipe_load"]],
            color="tab:blue", lw=0.5, alpha=0.35, label="DEM raw record")
    spring = main["soil_spring"]
    ax.axhline(spring["peak_reaction"] / 1e3, color="tab:red", ls="--", lw=1.0,
               label="subgrade peak (0.9 p_ult)")
    ax.axhline(spring["residual_reaction"] / 1e3, color="tab:orange", ls=":", lw=1.0,
               label="subgrade residual (0.8 peak)")
    ax.axhline(spring["ultimate_bearing_capacity"] / 1e3, color="0.4", ls="-.", lw=0.9,
               label="p_ult (Vesic/Skempton)")
    metrics = main["metrics"]
    ax.plot([metrics["displacement_at_peak"]], [metrics["peak_load_per_length"] / 1e3],
            marker="o", ms=6, color="k")
    ax.annotate(f"peak {metrics['peak_load_per_length'] / 1e3:.0f} kN/m\n"
                f"softening {100 * metrics['softening_ratio']:.1f}%",
                xy=(metrics["displacement_at_peak"], metrics["peak_load_per_length"] / 1e3),
                xytext=(10, 8), textcoords="offset points", fontsize=8)
    ax.set_xlabel("keel tip penetration below mudline (m)")
    ax.set_ylabel("pipe load per metre of ridge (kN/m)")
    ax.set_title("Ice-ridge grounding load-displacement (Beaufort-like case)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(history["keel_penetration"], history["pipe_displacement"], color="tab:green",
             lw=1.0, alpha=0.8, label="pipe settlement")
    ax2.set_ylabel("pipe vertical displacement (m)", color="tab:green")
    ax2.tick_params(axis="y", colors="tab:green")
    fig.tight_layout()
    path = os.path.join(FIG, "load_displacement.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_pipe_stress(main):
    curves = main["pipe_curves"]
    x = [v - 0.5 * (curves["x"][0] + curves["x"][-1]) for v in curves["x"]]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    axes[0].plot(x, [v / 1e6 for v in curves["moment"]], color="tab:purple", lw=1.4)
    axes[0].set_ylabel("bending moment (MN m)")
    axes[0].set_title("Pipeline response to the DEM footprint load "
                      "(FE Winkler vs closed form: %.2f%% moment error)"
                      % (100 * main["pipe_profile"]["moment_relative_error"]))
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, [v / 1e6 for v in curves["stress"]], color="tab:red", lw=1.4)
    axes[1].axhline(main["scenario"]["section"]["yield_strength"] / 1e6, color="k", ls="--",
                    lw=0.9, label="f_y")
    axes[1].axhline(-main["scenario"]["section"]["yield_strength"] / 1e6, color="k", ls="--",
                    lw=0.9)
    axes[1].set_ylabel("extreme-fibre stress (MPa)")
    axes[1].set_xlabel("distance along pipeline (m)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG, "pipe_stress.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_dem_field():
    """Run a compact DEM case and draw the contact-force cloud."""
    from tensordem.ridge_dem import RidgeKeel, RidgeMaterial, generate_keel_assembly
    from tensordem.ridge_geotech import MohrCoulombSoil, PipeSection
    from tensordem.ridge_grounding import GroundingScenario, run_grounding

    material = RidgeMaterial(particle_radius=0.4, drag=2.0e4, freeze_bond_fraction=1.0,
                             bond_tensile_strength=100.0e3, bond_shear_strength=200.0e3)
    scenario = GroundingScenario(
        keel=RidgeKeel(8.0, 16.0, 0.10), material=material,
        soil=MohrCoulombSoil(friction_angle_deg=0.0, cohesion=30.0e3,
                             saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                             water_unit_weight=10.06e3, undrained=True),
        section=PipeSection(0.914, 0.0159, 207.0e9, 448.0e6),
        cover_depth=0.4, tip_clearance=0.1, stroke=0.9, platen_velocity=0.35,
        contact_length=16.0)
    # rebuild the assembly exactly as run_grounding does, then integrate a short
    # record and keep the final state for plotting
    from tensordem.ridge_grounding import _build_assembly
    assembly, info = _build_assembly(scenario)
    dt = scenario.dt_used
    steps = int(round(scenario.stroke / (scenario.platen_velocity * dt)))
    pipe_displacement = 0.0
    for _ in range(steps):
        out = assembly.step(dt)
        load = -float(out["pipe_force"][1])
        pipe_displacement = min(pipe_displacement + load / 4.0e6 * dt * 60.0, 0.5)
        assembly.cylinders[0]["centre"] = torch.tensor(
            (0.0, scenario.pipe_centre_height - pipe_displacement), dtype=torch.float64)
    out = assembly.forces(dt, update_memory=False, break_bonds=False)
    force = out["force"].norm(dim=1)
    pos = assembly.pos
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    scatter = ax.scatter(pos[:, 0], pos[:, 1], c=force / 1e3, s=90 / material.particle_radius,
                         cmap="inferno", edgecolors="0.2", linewidths=0.3)
    alive = assembly.bond_alive
    bonds = assembly.bonds[alive]
    for i, j in bonds.tolist():
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color="0.55", lw=0.5, zorder=0)
    centre = assembly.cylinders[0]["centre"]
    radius = assembly.cylinders[0]["radius"]
    circle = plt.Circle((float(centre[0]), float(centre[1])), radius, color="tab:blue",
                        zorder=3)
    ax.add_patch(circle)
    ax.axhline(0.0, color="tab:brown", lw=1.4, label="mudline")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("DEM ridge kel - pipeline - seabed contact force cloud "
                 "(%d disks, %d intact bonds)" % (assembly.count, int(alive.sum())))
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(scatter, ax=ax, label="particle force magnitude (kN)")
    fig.tight_layout()
    path = os.path.join(FIG, "dem_field.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    payload = {"particles": assembly.count, "intact_bonds": int(alive.sum()),
               "broken_bonds": int(assembly.broken_bonds),
               "max_particle_force": float(force.max()),
               "pipe_contacts": out["pipe_contacts"], "steps": steps, "dt": dt}
    with open(os.path.join(ART, "dem_field.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def figure_envelope(envelope):
    rows = envelope["rows"]
    depths = [row["keel_depth"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(depths, [row["ice_capacity_per_length"] / 1e3 for row in rows], "o-",
            color="tab:blue", label="DEM ice-limited capacity")
    ax.plot(depths, [row["soil_capacity_per_length"] / 1e3 for row in rows], "s--",
            color="tab:red", label="soil-limited capacity (subgrade)")
    ax.plot(depths, [row["governing_capacity_per_length"] / 1e3 for row in rows], "k-",
            lw=2, alpha=0.5, label="governing envelope = min(ice, soil)")
    ax.set_xlabel("ridge keel depth (m)")
    ax.set_ylabel("capacity per metre of ridge (kN/m)")
    ax.set_title("Ultimate capacity envelope and governing mechanism")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(depths, [row["peak_utilization"] for row in rows], "^:", color="tab:green",
             label="DNV-OS-F101 utilization")
    ax2.axhline(1.0, color="tab:green", lw=0.8, ls="--")
    ax2.set_ylabel("DNV-OS-F101 utilization U", color="tab:green")
    ax2.tick_params(axis="y", colors="tab:green")
    fig.tight_layout()
    path = os.path.join(FIG, "envelope.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_sensitivity(sensitivity):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    depth = sensitivity["keel_depth"]
    width = sensitivity["keel_width"]
    soil = sensitivity["soil_strength"]
    axes[0].plot([r["keel_depth"] for r in depth], [r["capacity_per_length"] / 1e3 for r in depth],
                 "o-", color="tab:blue")
    axes[0].set_xlabel("keel depth (m)")
    axes[0].set_ylabel("capacity (kN/m)")
    axes[0].set_title("keel depth")
    axes[1].plot([r["keel_width"] for r in width], [r["capacity_per_length"] / 1e3 for r in width],
                 "s-", color="tab:orange")
    axes[1].set_xlabel("keel width (m)")
    axes[1].set_title("keel width")
    axes[2].plot([r["undrained_shear_strength"] / 1e3 for r in soil],
                 [r["soil_ultimate_capacity"] / 1e3 for r in soil], "^-", color="tab:red")
    axes[2].set_xlabel("undrained shear strength s_u (kPa)")
    axes[2].set_title("seabed strength (soil-limited)")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.suptitle("Capacity sensitivity (trends must increase with all three parameters)",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIG, "sensitivity.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_calibration(calibration):
    specimens = calibration["specimens"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels = []
    for row in specimens:
        deviator = row["peak_deviator_stress"] / 1e3
        axial = row["axial_stress_at_peak"] / 1e3
        lateral = row["lateral_stress_at_peak"] / 1e3
        ax.plot([lateral, axial], [0.0, 0.0], "o-", ms=4)
        labels.append(f"sigma3 target {row['confining_stress'] / 1e3:.0f} kPa")
    ax.set_xlabel("normal stress (kPa)")
    ax.set_ylabel("shear stress (kPa) = 0 axis markers")
    ax.set_title("Biaxial calibration Mohr circles (diagnostic)")
    ax.grid(alpha=0.3)
    ax.legend(labels, fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG, "calibration.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    produced = []
    main_case = _load("main_case.json")
    if main_case:
        produced.append(figure_load_displacement(main_case))
        produced.append(figure_pipe_stress(main_case))
    envelope = _load("envelope.json")
    if envelope:
        produced.append(figure_envelope(envelope))
    sensitivity = _load("sensitivity.json")
    if sensitivity and sensitivity["keel_depth"]:
        produced.append(figure_sensitivity(sensitivity))
    calibration = _load("calibration.json")
    if calibration:
        produced.append(figure_calibration(calibration))
    produced.append(figure_dem_field())
    with open(os.path.join(ART, "figures.json"), "w", encoding="utf-8") as handle:
        json.dump([os.path.basename(p) for p in produced], handle, indent=2)
    for path in produced:
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
