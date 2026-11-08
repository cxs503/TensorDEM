#!/usr/bin/env python3
"""Figures for the level-ice resistance DEM campaign (reads the JSON artifact).

Usage::

    python scripts/plot_level_ice_campaign.py \
        artifacts/level_ice/level_ice_campaign.json artifacts/level_ice/figures
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def fig_speed(artifact: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=False)
    for ax, hull in zip(axes, ("icebreaker", "transport")):
        block = artifact["sweeps"][hull]
        v = np.array(block["speeds_m_s"], dtype=float)
        dem = np.array(block["dem_speed_n_per_m"]) / 1e3
        lind = np.array(block["lindqvist_per_width_n_per_m"]) / 1e3
        riska = np.array(block["riska_per_width_n_per_m"]) / 1e3
        ax.plot(v, dem, "o-", color="#b2182b", lw=2, label="DEM section (measured)")
        ax.plot(v, lind, "s--", color="#2166ac", lw=1.6,
                label="Lindqvist 1989 / B")
        ax.plot(v, riska, "^:", color="#4daf4a", lw=1.4, label="Riska-type / B")
        ax.set_yscale("log")
        ax.set_xlabel("ship speed v [m/s]")
        ax.set_ylabel("resistance per unit breadth [kN/m]")
        ax.set_title(f"{hull}: R(v), h = 1.0 m"
                     f"   (Pearson={block['pearson_dem_vs_lindqvist']:.3f})")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Level-ice resistance: bonded-particle section vs engineering "
                 "anchors", fontsize=11)
    fig.tight_layout()
    path = out / "fig1_speed_sweep.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_thickness(artifact: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, hull in zip(axes, ("icebreaker", "transport")):
        block = artifact["sweeps"][hull]
        h = np.array(block["thickness_m"], dtype=float)
        dem = np.array(block["dem_thickness_n_per_m"]) / 1e3
        lind = np.array(block["lindqvist_thickness_per_width_n_per_m"]) / 1e3
        ax.loglog(h, dem, "o-", color="#b2182b", lw=2,
                  label=f"DEM  n={block['dem_thickness_exponent']:.2f}")
        ax.loglog(h, lind, "s--", color="#2166ac", lw=1.6,
                  label=f"Lindqvist/B  n={block['lindqvist_thickness_exponent']:.2f}")
        ax.set_xlabel("ice thickness h [m]")
        ax.set_ylabel("resistance per unit breadth [kN/m]")
        ax.set_title(f"{hull}: R(h) at v = 3 m/s")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig2_thickness_sweep.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_components(artifact: dict, out: Path) -> Path:
    labels: list[str] = []
    dem_crush: list[float] = []
    dem_other: list[float] = []
    ref_crush: list[float] = []
    ref_other: list[float] = []
    for hull in ("icebreaker", "transport"):
        for target, data in artifact["sweeps"][hull]["acceptance"].items():
            labels.append(target.replace("_", " "))
            dem_crush.append(data["dem_crushing_fraction"])
            dem_other.append(1.0 - data["dem_crushing_fraction"])
            ref_crush.append(data["lindqvist_crushing_fraction"])
            ref_other.append(1.0 - data["lindqvist_crushing_fraction"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    width = 0.36
    ax.bar(x - width / 2, dem_crush, width, label="DEM crushing fraction",
           color="#b2182b")
    ax.bar(x + width / 2, ref_crush, width, label="Lindqvist crushing fraction",
           color="#2166ac")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("crushing share of the resistance [-]")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax.set_title("Component split: DEM section vs Lindqvist three-component anchor")
    fig.tight_layout()
    path = out / "fig3_component_split.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_numerics(artifact: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    gci = artifact["mesh_convergence"]
    levels = np.array(gci["levels"], dtype=float)
    values = np.array(gci["values_n_per_m"]) / 1e6
    axes[0].plot(levels, values, "o-", color="#762a83")
    for lev, val in zip(levels, values):
        axes[0].annotate(f"{val:.2f}", (lev, val), textcoords="offset points",
                         xytext=(4, 4), fontsize=8)
    axes[0].set_xlabel("layers through the thickness n_z")
    axes[0].set_ylabel("R [MN/m]")
    axes[0].set_title(f"mesh: GCI = {gci['gci_pct']:.0f}% (p={gci['observed_order']:.2f})")
    axes[0].grid(alpha=0.3)

    mass = artifact["mass_scaling"]
    ms = np.array(mass["mass_scales"], dtype=float)
    mv = np.array(mass["values_n_per_m"]) / 1e6
    axes[1].semilogx(ms, mv, "s-", color="#1b7837")
    axes[1].set_xlabel("mass-scaling factor")
    axes[1].set_ylabel("R [MN/m]")
    axes[1].set_title(f"quasi-static limit: spread {100*mass['relative_spread']:.1f}%")
    axes[1].grid(alpha=0.3, which="both")

    dim = artifact["dimensional_check"]
    names = [c["name"] for c in dim["cases"]]
    ratios = [c["ratio"] for c in dim["cases"]]
    axes[2].bar(range(len(names)), ratios, color="#f4a582")
    axes[2].axhline(1.0, color="k", lw=1)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("M_break / (sigma_f h^2/6 L) [-]")
    axes[2].set_title(f"dimensional check: sigma exp {dim['sigma_exponent']:.3f}, "
                      f"h exp {dim['h_exponent']:.3f}")
    axes[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out / "fig4_numerics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_field(artifact: dict, out: Path) -> Path | None:
    snapshot = None
    for hull in artifact["sweeps"]:
        for case in artifact["sweeps"][hull]["speed_curve"]:
            if "snapshot" in case:
                snapshot = case["snapshot"]
                break
        if snapshot:
            break
    if snapshot is None:
        return None
    x = np.array(snapshot["x"])
    z = np.array(snapshot["z"])
    u = np.array(snapshot["u"])
    active = np.array(snapshot["active"]).astype(bool)
    alive = np.array(snapshot["alive"]).astype(bool)
    break_x = np.array(snapshot["break_x"])
    nx, nz = snapshot["nx"], snapshot["nz"]
    uz = np.where(active, u[:, 1], np.nan).reshape(nz, nx)
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 5.4), sharex=False)
    im = axes[0].pcolormesh(np.array(x).reshape(nz, nx), np.array(z).reshape(nz, nx),
                            uz * 1e3, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=axes[0], label="vertical displacement u_z [mm]")
    axes[0].set_ylabel("z [m]")
    axes[0].set_title("DEM section: vertical displacement field at the end of the "
                      "h=1.0 m, v=3 m/s run")
    axes[0].plot(snapshot["hull_x"], snapshot["z_top"], "r<", ms=8,
                 label="hull face top")
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].plot(x, z, ",", color="#cccccc", ms=2)
    if break_x.size:
        axes[1].hist(break_x, bins=60, color="#b2182b",
                     label=f"broken bonds ({int((~alive).sum())})")
    if snapshot.get("break_x"):
        pass
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("z [m]")
    axes[1].set_title("fracture locations along the sheet")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = out / "fig5_section_field.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv: list[str]) -> int:
    artifact_path = Path(argv[1]) if len(argv) > 1 else Path(
        "artifacts/level_ice/level_ice_campaign.json")
    out = Path(argv[2]) if len(argv) > 2 else Path("artifacts/level_ice/figures")
    out.mkdir(parents=True, exist_ok=True)
    artifact = _load(artifact_path)
    produced = [fig_speed(artifact, out), fig_thickness(artifact, out),
                fig_components(artifact, out), fig_numerics(artifact, out)]
    field = fig_field(artifact, out)
    if field is not None:
        produced.append(field)
    for path in produced:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))