#!/usr/bin/env python3
"""Generate figures for the m7359 Uikku 3D mass-conservation report."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.level_ice.sheet import SheetConfig
from tensordem.level_ice.sheet3d import LevelIceSheet3D, Sheet3DConfig


def generate_wb_scan_figure(output_path: str):
    """Figure 1: Resistance vs W/B ratio showing monotonic decrease."""
    # Use a representative case (Test 103: h=0.77m, v=0.2m/s)
    cfg = SheetConfig(
        thickness_m=0.77,
        particle_size_m=0.77 / 4,
        length_m=8.0,
        flexural_strength_pa=724e3,
        compressive_strength_pa=1748e3,
        shear_strength_pa=2 * 724e3,
        elastic_modulus_pa=929e6,
        buoyancy=True,
        bow_angle_deg=30.0,
        hull_speed_m_s=0.2,
        hull_friction=0.05,
        crush_cap_pa=1748e3,
        mass_scale=200.0,
    )

    wb_ratios = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    section_resistances = []
    escape_factors = []

    for wb in wb_ratios:
        cfg3d = Sheet3DConfig(
            sheet_config=cfg,
            channel_width_m=wb * 26.0,
            beam_m=26.0,
            enable_reservoir=True,
        )
        sim = LevelIceSheet3D(cfg3d)
        result = sim.run(distance_m=1.0, record_stride=50)
        r = result["steady"]["mean_total_n_per_m"]
        section_resistances.append(r)
        # The escape factor is applied inside step()
        ef = 0.3 / wb if wb > 0.3 else 1.0
        escape_factors.append(ef)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Section resistance vs W/B
    ax1.plot(wb_ratios, section_resistances, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("W/B ratio", fontsize=12)
    ax1.set_ylabel("Section resistance [N/m]", fontsize=12)
    ax1.set_title("(a) Section line load vs channel width", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=section_resistances[0], color="r", linestyle="--", alpha=0.5, label="2D limit")
    ax1.legend()

    # Right: Escape factor vs W/B
    ax2.plot(wb_ratios, escape_factors, "rs-", linewidth=2, markersize=8)
    ax2.set_xlabel("W/B ratio", fontsize=12)
    ax2.set_ylabel("Escape factor $f_{esc}$", fontsize=12)
    ax2.set_title("(b) Lateral escape reduction factor", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="2D limit (no escape)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_mass_conservation_figure(output_path: str):
    """Figure 2: Mass conservation time series."""
    cfg = SheetConfig(
        thickness_m=1.0,
        particle_size_m=0.25,
        length_m=10.0,
        flexural_strength_pa=0.5e6,
        compressive_strength_pa=1.5e6,
        shear_strength_pa=1.0e6,
        buoyancy=True,
        bow_angle_deg=30.0,
        hull_speed_m_s=1.0,
        hull_friction=0.05,
        crush_cap_pa=1.5e6,
        mass_scale=200.0,
    )

    # Run with reservoir
    cfg3d = Sheet3DConfig(sheet_config=cfg, channel_width_m=20.0, beam_m=20.0, enable_reservoir=True)
    sim = LevelIceSheet3D(cfg3d)
    result = sim.run(distance_m=2.0, record_stride=20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Mass components over time
    ledger = result["mass_ledger"]
    history = sim.ledger.history
    if history["time_s"]:
        t = np.array(history["time_s"])
        intact = np.array(history["intact_kg"])
        broken = np.array(history["broken_kg"])
        reservoir = np.array(history["reservoir_kg"])
        total = np.array(history["total_kg"])

        ax1.fill_between(t, 0, intact, alpha=0.5, label="Intact")
        ax1.fill_between(t, intact, intact + broken, alpha=0.5, label="Broken (active)")
        ax1.fill_between(t, intact + broken, total, alpha=0.5, label="Reservoir")
        ax1.plot(t, total, "k-", linewidth=2, label="Total")
        ax1.set_xlabel("Time [s]", fontsize=12)
        ax1.set_ylabel("Mass [kg]", fontsize=12)
        ax1.set_title("(a) Mass partition over time", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

    # Right: Relative conservation error
    if history["rel_error"]:
        rel_err = np.array(history["rel_error"])
        ax2.semilogy(t, np.maximum(rel_err, 1e-18), "g-", linewidth=2)
        ax2.axhline(y=1e-6, color="r", linestyle="--", label="Threshold (1e-6)")
        ax2.set_xlabel("Time [s]", fontsize=12)
        ax2.set_ylabel("Relative mass error", fontsize=12)
        ax2.set_title("(b) Mass conservation error", fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=1e-18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_negative_control_figure(output_path: str):
    """Figure 3: Negative control - reservoir disabled vs enabled."""
    cfg = SheetConfig(
        thickness_m=1.0,
        particle_size_m=0.25,
        length_m=10.0,
        flexural_strength_pa=0.5e6,
        compressive_strength_pa=1.5e6,
        shear_strength_pa=1.0e6,
        buoyancy=True,
        bow_angle_deg=30.0,
        hull_speed_m_s=1.0,
        hull_friction=0.05,
        crush_cap_pa=1.5e6,
        mass_scale=200.0,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # With reservoir
    cfg_yes = Sheet3DConfig(sheet_config=cfg, channel_width_m=20.0, beam_m=20.0, enable_reservoir=True)
    sim_yes = LevelIceSheet3D(cfg_yes)
    r_yes = sim_yes.run(distance_m=2.0, record_stride=20)

    # Without reservoir
    cfg_no = Sheet3DConfig(sheet_config=cfg, channel_width_m=20.0, beam_m=20.0, enable_reservoir=False)
    sim_no = LevelIceSheet3D(cfg_no)
    r_no = sim_no.run(distance_m=2.0, record_stride=20)

    # Left: Resistance history
    t_yes = r_yes["time_s"]
    t_no = r_no["time_s"]
    ax1.plot(t_yes, r_yes["total_n_per_m"] / 1e3, "b-", linewidth=2, label="With reservoir")
    ax1.plot(t_no, r_no["total_n_per_m"] / 1e3, "r-", linewidth=2, label="No reservoir (2D)")
    ax1.set_xlabel("Time [s]", fontsize=12)
    ax1.set_ylabel("Resistance [kN/m]", fontsize=12)
    ax1.set_title("(a) Resistance history: reservoir effect", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Mass conservation comparison
    h_yes = sim_yes.ledger.history
    h_no = sim_no.ledger.history
    if h_yes["time_s"]:
        t = np.array(h_yes["time_s"])
        total_yes = np.array(h_yes["total_kg"])
        ax2.plot(t, total_yes / total_yes[0], "b-", linewidth=2, label="With reservoir")
    if h_no["time_s"]:
        t = np.array(h_no["time_s"])
        total_no = np.array(h_no["total_kg"])
        ax2.plot(t, total_no / total_no[0], "r-", linewidth=2, label="No reservoir (2D)")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Mass ratio $M(t)/M_0$", fontsize=12)
    ax2.set_title("(b) Mass conservation: reservoir vs deletion", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    out_dir = Path("artifacts/m7359")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    generate_wb_scan_figure(str(out_dir / "fig1_wb_scan.png"))
    generate_mass_conservation_figure(str(out_dir / "fig2_mass_conservation.png"))
    generate_negative_control_figure(str(out_dir / "fig3_negative_control.png"))
    print("Done.")
