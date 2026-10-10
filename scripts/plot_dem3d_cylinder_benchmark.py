"""Create publication-friendly plots from cylinder benchmark CSV outputs.

Usage:
    python scripts/plot_dem3d_cylinder_benchmark.py results-cylinder
Requires the optional viz dependency (matplotlib).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no data rows in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory created by the cylinder benchmark driver")
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="png")
    args = parser.parse_args()
    curve_path = args.results / "load_displacement.csv"
    summary_path = args.results / "benchmark_summary.json"
    if not curve_path.is_file() or not summary_path.is_file():
        parser.error(f"{args.results} must contain load_displacement.csv and benchmark_summary.json")
    rows = read_rows(curve_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    displacement_mm = [float(row["compression_displacement_m"]) * 1e3 for row in rows]
    top_load = [float(row["top_load_N"]) for row in rows]
    bottom_load = [float(row["bottom_load_N"]) for row in rows]
    broken = [int(float(row["broken_bonds"])) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(displacement_mm, top_load, label="Top platen")
    axes[0].plot(displacement_mm, bottom_load, linestyle="--", label="Bottom platen")
    axes[0].set_ylabel("Reaction magnitude (N)")
    axes[0].set_title("Cylinder diametral compression — solver benchmark")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].step(displacement_mm, broken, where="post")
    axes[1].set_xlabel("Imposed platen closure (mm)")
    axes[1].set_ylabel("Cumulative broken bonds")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        f"N={summary['configuration']['nx']}×{summary['configuration']['ny']}×"
        f"{summary['configuration']['nz']}; samples={summary['samples']}",
        fontsize=9,
    )

    outputs = []
    for ext in (("png", "pdf") if args.format == "both" else (args.format,)):
        target = args.results / f"cylinder_compression.{ext}"
        fig.savefig(target, dpi=180 if ext == "png" else None)
        outputs.append(target.name)
    plt.close(fig)
    print(json.dumps({"plots": outputs, "samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
