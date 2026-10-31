"""Generate scaling plot for DEM3D large-scale particle benchmark."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results-dem3d-scale" / "scaling_report.json"
OUTPUT = ROOT / "artifacts" / "dem3d_scale_benchmark.png"


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    cases = report["cases"]

    n_particles = [c["actual_n"] for c in cases]
    wallclock_per_step = [c["wallclock_per_step_ms"] for c in cases]

    # Ideal linear scaling line (through 10k point)
    n_ideal = np.array([n_particles[0], n_particles[-1]])
    t_ideal = wallclock_per_step[0] * n_ideal / n_particles[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(n_particles, wallclock_per_step, "o-", color="#1f77b4",
              linewidth=2, markersize=10, label="Measured")
    ax.loglog(n_ideal, t_ideal, "--", color="#ff7f0e", linewidth=2,
              label="Ideal O(N) scaling")

    # Annotate points
    labels = ["10k", "50k", "100k"]
    for i, (n, t, label) in enumerate(zip(n_particles, wallclock_per_step, labels)):
        ax.annotate(f"{label}\n{t:.0f} ms/step", (n, t),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=9, color="#1f77b4")

    ax.set_xlabel("Particle count N", fontsize=12)
    ax.set_ylabel("Wall-clock per step (ms)", fontsize=12)
    ax.set_title("DEM3D Large-Scale Particle Benchmark: Wall-Clock Scaling", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    # Add ratio annotation
    ratio = report["wallclock_100k_ratio_vs_ideal"]
    ax.text(0.95, 0.05, f"100k ratio vs ideal: {ratio:.2f}×\n(threshold: < 2.0×)",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
