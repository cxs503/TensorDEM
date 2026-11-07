"""Figures for the IN22-N02 channel 6-DOF capsize study.

Reads ``artifacts/channel_6dof.json`` and writes the paper figures into
``artifacts/channel_6dof/``.  Every number plotted comes from the artifact, so
the figures cannot drift from the machine-readable results.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(HERE, "..", "artifacts")
FIGURE_DIR = os.path.join(ARTIFACT_DIR, "channel_6dof")


def _load() -> Dict[str, object]:
    with open(os.path.join(ARTIFACT_DIR, "channel_6dof.json"), encoding="utf-8") as handle:
        return json.load(handle)


def fig_righting_moment(artifact: Dict[str, object]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colours = {0.3: "tab:blue", 0.5: "tab:red"}
    for key, analysis in artifact["capsize_analysis"].items():
        curve = analysis["curve"]
        tilt = np.array(curve["tilt_deg"])
        moment = np.array(curve["righting_moment_nm"])
        colour = colours.get(round(float(key), 3), "tab:gray")
        axes[0].plot(tilt, moment / 1e3, color=colour,
                     label=f"t = {float(key):.1f} m")
        axes[0].axvline(analysis["capsize_angle_moment_deg"], color=colour,
                        linestyle="--", linewidth=1.0)
        axes[0].axvline(analysis["capsize_angle_hessian_deg"], color=colour,
                        linestyle=":", linewidth=1.6)
        axes[1].plot(tilt, np.array(curve["stiffness_nm_per_rad"]) / 1e6,
                     color=colour, label=f"t = {float(key):.1f} m")
        axes[1].axhline(0.0, color="k", linewidth=0.8)
        axes[1].axvline(analysis["capsize_angle_hessian_deg"], color=colour,
                        linestyle=":")
    axes[0].set_xlabel("roll angle φ [deg]")
    axes[0].set_ylabel("righting moment $M_r$ [kN m]")
    axes[0].set_title("(a) righting-moment curve and its fold $\\theta_s$\n"
                      "dashed = force route, dotted = Hessian route")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("roll angle φ [deg]")
    axes[1].set_ylabel("$d^2U/d\\phi^2$ [MN m / rad]")
    axes[1].set_title("(b) attitude Hessian: first zero = second bifurcation")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig_righting_moment.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def fig_energy_landscape(artifact: Dict[str, object]) -> str:
    colours = {0.3: "tab:blue", 0.5: "tab:red"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, analysis in artifact["capsize_analysis"].items():
        curve = analysis["curve"]
        tilt = np.array(curve["tilt_deg"])
        energy = np.array(curve["energy_j"])
        energy = energy - energy.min()
        colour = colours.get(round(float(key), 3), "tab:gray")
        ax.plot(tilt, energy / 1e3, color=colour, label=f"t = {float(key):.1f} m")
        theta = analysis["capsize_angle_hessian_deg"]
        ax.axvline(theta, color=colour, linestyle=":", linewidth=1.2)
        ax.plot([theta], [np.interp(theta, tilt, energy) / 1e3], "o",
                color=colour, markersize=4)
    ax.set_xlabel("roll angle φ [deg]")
    ax.set_ylabel("heave-relaxed potential $U_c(\\phi) - U_c(0)$ [kJ]")
    ax.set_title("attitude potential: flat well, saddle at $\\theta_s$, "
                 "edge-on secondary well")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig_energy_landscape.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _series_run(artifact: Dict[str, object], group: str,
                 target: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """Find the run carrying the time series (any shard file may hold it)."""
    target = target or {"concentration": 0.8, "thickness": 0.5, "seed": 11}
    for name in sorted(os.listdir(ARTIFACT_DIR)):
        if not (name.startswith("channel_6dof_shard_") and name.endswith(".json")):
            continue
        with open(os.path.join(ARTIFACT_DIR, name), encoding="utf-8") as handle:
            payload = json.load(handle)
        for run in payload["runs"]:
            if (run["group"] == group and run["case"] == target
                    and abs(run["dt"] - 0.008) < 1e-12):
                return run
    return {}


def fig_width_series(artifact: Dict[str, object]) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    styles = {"tilt_locked": ("tab:gray", "-"), "six_dof": ("tab:blue", "-"),
              "six_dof_shoulder": ("tab:red", "-")}
    plotted = False
    for group, (colour, style) in styles.items():
        run = _series_run(artifact, group)
        if not run:
            continue
        times = np.array(run["time_series"])
        width = np.array(run["cleared_width_series"])
        ax.plot(times, width, color=colour, linestyle=style, label=group)
        plotted = True
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cleared channel width [m]")
    ax.set_title("cleared width, 8/10 coverage / t = 0.5 m / seed 11")
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig_width_series.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def fig_delta_width(artifact: Dict[str, object]) -> str:
    grouped = artifact["criteria"]["d_width_table"]["grouped"]
    labels = [f"{row['concentration']:.1f}/{row['thickness']:.1f}"
              for row in grouped]
    x = np.arange(len(grouped))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    for offset, key, colour, name in (
            (-0.5 * width, "delta_width_rel_six_dof", "tab:blue", "6-DOF (vertical wall)"),
            (0.5 * width, "delta_width_rel_six_dof_shoulder", "tab:red",
             "6-DOF + shoulder submersion")):
        means = [row[key]["mean"] * 100.0 for row in grouped]
        stds = [row[key]["std"] * 100.0 for row in grouped]
        axes[0].bar(x + offset, means, width, yerr=stds, capsize=3,
                    color=colour, label=name)
    axes[0].axhline(0.0, color="k", linewidth=0.8)
    axes[0].set_ylabel("ΔW / W_tilt-locked [%]")
    axes[0].set_title("cleared-width change relative to the tilt-locked H1 baseline")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    caps_lock = [row["capsize_fraction_six_dof"] for row in grouped]
    axes[1].plot(x, [value["mean"] * 100.0 for value in caps_lock], "o-",
                 color="tab:blue", label="capsized fraction, vertical wall")
    axes[1].plot(x, [row["capsize_fraction_six_dof_shoulder"]["mean"] * 100.0
                     for row in grouped], "s-", color="tab:red",
                 label="capsized fraction, shoulder submersion")
    axes[1].set_ylabel("capsized floes [%]")
    axes[1].set_xlabel("coverage / thickness  (3 seeds, mean ± s.d.)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig_delta_width.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def fig_tilt_and_resistance(artifact: Dict[str, object]) -> str:
    rows = artifact["criteria"]["d_width_table"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    groups = (("tilt_locked", "tab:gray"), ("six_dof", "tab:blue"),
              ("six_dof_shoulder", "tab:red"))
    for group, colour in groups:
        tilt = [row[f"max_tilt_deg_{group}"] for row in rows]
        resistance = [row[f"resistance_{group}"] for row in rows]
        coverage = [row["concentration"] for row in rows]
        thickness = [row["thickness"] for row in rows]
        axes[0].scatter(coverage, tilt, color=colour, label=group,
                        marker="o" if group == "tilt_locked" else "s")
        axes[1].scatter(coverage, np.array(resistance) / 1e3, color=colour,
                        label=group)
    axes[0].set_xlabel("coverage")
    axes[0].set_ylabel("max floe tilt [deg]")
    axes[0].set_title("(a) peak attitude excursion per transit")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("coverage")
    axes[1].set_ylabel("mean hull resistance [kN]")
    axes[1].set_title("(b) ice resistance")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig_tilt_resistance.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main() -> int:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    artifact = _load()
    paths = [fig_righting_moment(artifact), fig_energy_landscape(artifact),
             fig_width_series(artifact), fig_delta_width(artifact),
             fig_tilt_and_resistance(artifact)]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())