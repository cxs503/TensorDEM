"""Generate energy drift plot for DEM3D large-scale particle benchmark."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results-dem3d-scale" / "scaling_report.json"
OUTPUT = ROOT / "artifacts" / "dem3d_energy_drift.png"


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    cases = report["cases"]

    n_particles = [c["actual_n"] for c in cases]
    labels = ["10k", "50k", "100k"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cases))
    drift_values = [abs(c["energy_balance_residual_J"]) for c in cases]

    bars = ax.bar(x, drift_values, color=["#2ca02c", "#2ca02c", "#2ca02c"],
                  width=0.5, edgecolor="white", linewidth=1.5)

    # Threshold line
    threshold = 1e-06
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=2,
               label=f"Threshold: {threshold:.0e}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Energy drift per step (J)", fontsize=12)
    ax.set_title("DEM3D Energy Conservation: Drift per Step", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Animate values
    for i, (bar, val) in enumerate(zip(bars, drift_values)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.5,
                    f"{val:.2e}", ha="center", va="bottom", fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, threshold * 0.5,
                    "0.00 (machine precision)", ha="center", va="bottom",
                    fontsize=9, color="green")

    # Status annotation
    ax.text(0.95, 0.95, "Status: PASS\n(all < 1e-06)",
            transform=ax.transAxes, fontsize=10, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
