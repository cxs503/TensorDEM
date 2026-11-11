#!/usr/bin/env python
"""Generate visualization figures for GAP-DEM-GPU-100K benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "gap_dem_gpu_100k" / "results.json"
FIGURES_DIR = ROOT / "artifacts" / "gap_dem_gpu_100k"


def generate_scaling_plot() -> str:
    """Generate a scaling plot (time per step vs particle count)."""
    data = json.loads(RESULTS.read_text())

    scales = []
    times = []
    particles = []
    for r in data["results"]:
        if r["status"] == "ok":
            scales.append(r["scale_label"])
            particles.append(r["n_particles"])
            times.append(r["time_per_step_s"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Time per step vs particles
    ax1.bar(scales, times, color=["#4CAF50", "#FF9800", "#F44336"], edgecolor="white", linewidth=1.5)
    ax1.axhline(y=2.0, color="red", linestyle="--", linewidth=2, label="Threshold (2.0 s)")
    ax1.set_xlabel("Scale")
    ax1.set_ylabel("Time per step (s)")
    ax1.set_title("DEM3D Wall-Clock Time per Step")
    ax1.legend()
    for i, (s, t) in enumerate(zip(scales, times)):
        ax1.text(i, t + 0.1, f"{t:.2f}s", ha="center", fontweight="bold")

    # Right: Throughput (particles·steps/s)
    throughputs = [r["throughput_particles_steps_per_s"] for r in data["results"] if r["status"] == "ok"]
    ax2.bar(scales, throughputs, color=["#2196F3", "#00BCD4", "#009688"], edgecolor="white", linewidth=1.5)
    ax2.set_xlabel("Scale")
    ax2.set_ylabel("Throughput (N·steps/s)")
    ax2.set_title("DEM3D Particle Throughput")
    for i, (s, tp) in enumerate(zip(scales, throughputs)):
        ax2.text(i, tp + 500, f"{tp:,.0f}", ha="center", fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "scaling_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def generate_acceptance_plot() -> str:
    """Generate an acceptance status plot."""
    data = json.loads(RESULTS.read_text())

    fig, ax = plt.subplots(figsize=(6, 4))

    scales = []
    times = []
    colors = []
    for r in data["results"]:
        if r["status"] == "ok":
            scales.append(r["scale_label"])
            times.append(r["time_per_step_s"])
            if r.get("acceptance") is True:
                colors.append("#4CAF50")  # green
            elif r.get("acceptance") is False:
                colors.append("#F44336")  # red
            else:
                colors.append("#FF9800")  # orange (no acceptance field)

    bars = ax.bar(scales, times, color=colors, edgecolor="white", linewidth=1.5)
    ax.axhline(y=2.0, color="red", linestyle="--", linewidth=2, label="Threshold (2.0 s)")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Time per step (s)")
    ax.set_title("GAP-DEM-GPU-100K Acceptance Status (CPU Baseline)")
    ax.legend()

    # Add status labels
    for i, (s, t, r) in enumerate(zip(scales, times, [rr for rr in data["results"] if rr["status"] == "ok"])):
        status = "✅ PASS" if r.get("acceptance") else ("❌ FAIL" if r.get("acceptance") is False else "⚠️ N/A")
        ax.text(i, t + 0.1, status, ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    out_path = FIGURES_DIR / "acceptance_status.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    p1 = generate_scaling_plot()
    print(f"Scaling plot: {p1}")

    p2 = generate_acceptance_plot()
    print(f"Acceptance plot: {p2}")


if __name__ == "__main__":
    main()
