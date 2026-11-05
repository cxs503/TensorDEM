#!/usr/bin/env python3
"""Generate missing figures for dem3d_scale_benchmark.md from existing results.

Reads results-dem3d-scale/scaling_report.json and produces:
- artifacts/dem3d_scale_benchmark.png (wall-clock scaling plot)
- artifacts/dem3d_energy_drift.png (energy drift plot)

This is a post-processing script; it does not run any DEM simulation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    report_path = Path(__file__).resolve().parents[1] / "results-dem3d-scale" / "scaling_report.json"
    with open(report_path) as f:
        report = json.load(f)

    cases = report["cases"]
    labels = [c["case_id"].replace("scale_", "") for c in cases]
    n_particles = [c["actual_n"] for c in cases]
    ms_per_step = [c["wallclock_per_step_ms"] for c in cases]
    init_s = [c["init_wallclock_s"] for c in cases]
    drift = [c["energy_drift_per_step_J"] for c in cases]

    # Figure 1: Wall-clock scaling
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, ms_per_step, color=["#2196F3", "#4CAF50", "#FF9800"], width=0.5)
    # Ideal linear scaling line
    ideal_100k = ms_per_step[0] * (n_particles[2] / n_particles[0])
    ax.axhline(ideal_100k, color="red", linestyle="--", alpha=0.7,
               label=f"Ideal linear ({ideal_100k:.0f} ms)")
    ax.set_ylabel("Wall-clock per step [ms]")
    ax.set_title("DEM3D Scale Benchmark: Wall-Clock vs Particle Count")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(ms_per_step):
        ax.text(i, v + 200, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    out1 = Path(__file__).resolve().parents[1] / "artifacts" / "dem3d_scale_benchmark.png"
    out1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out1, dpi=150)
    print(f"wrote {out1}")

    # Figure 2: Energy drift
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.bar(labels, drift, color=["#2196F3", "#4CAF50", "#FF9800"], width=0.5)
    ax2.set_ylabel("Energy drift per step [J]")
    ax2.set_title("DEM3D Scale Benchmark: Energy Conservation")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(drift):
        ax2.text(i, v, f"{v:.2e}", ha="center", fontsize=9, va="bottom")
    fig2.tight_layout()
    out2 = Path(__file__).resolve().parents[1] / "artifacts" / "dem3d_energy_drift.png"
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
