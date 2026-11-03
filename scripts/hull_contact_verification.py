"""Verify the polyline hull-contact kernel against closed-form geometry.

Writes a machine-readable report, a CSV of the discretization scan, and two
figures.  This is a geometry verification of an already-shipped kernel; it does
not calibrate the physical resistance of a ship in ice.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tensordem.hull_contact_verification import (
    LEVELS,
    FRACTIONS,
    bow_polyline,
    build_report,
    exact_circular_hull_force,
    polyline_contact_forces,
    write_report,
    write_summary_csv,
)


def parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=parse_int_list, default=list(LEVELS),
                        help="bow discretization levels (segments)")
    parser.add_argument("--fractions", type=parse_floats, default=list(FRACTIONS),
                        help="penetration depths as a fraction of the particle radius")
    parser.add_argument("--radius", type=float, default=0.5, help="bow radius (m)")
    parser.add_argument("--particle-radius", type=float, default=0.05, help="disk radius (m)")
    parser.add_argument("--stiffness", type=float, default=1000.0, help="penalty stiffness (N/m)")
    parser.add_argument("--output", type=Path, default=Path("docs/hull-contact-verification"),
                        help="output directory")
    return parser


def figure_convergence(report, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["error_table"]
    fractions = sorted({row["penetration_fraction"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for fraction in fractions:
        series = [r for r in rows if r["penetration_fraction"] == fraction and r["dropout_fraction"] == 0]
        if len(series) < 2:
            continue
        segs = [r["segments"] for r in series]
        axes[0].loglog(segs, [r["max_relative_magnitude_error"] for r in series], "o-",
                       label=f"penetration = {fraction:g} $R_p$")
        axes[1].loglog(segs, [r["max_direction_error_deg"] for r in series], "s-",
                       label=f"penetration = {fraction:g} $R_p$")
    guide = [r for r in rows if r["dropout_fraction"] == 0]
    if guide:
        top = max(r["segments"] for r in guide)
        axes[0].loglog([32, top], [0.4 * (32 / 32.0) ** -2.0, 0.4 * (top / 32.0) ** -2.0], "k--", lw=1,
                       label=r"slope $M^{-2}$")
        axes[1].loglog([32, top], [180.0 / 32, 180.0 / top], "k--", lw=1, label=r"slope $M^{-1}$")
    axes[0].set_xlabel("bow segments $M$")
    axes[0].set_ylabel("max $|\\delta F| / F$")
    axes[0].set_title("force magnitude: second order")
    axes[1].set_xlabel("bow segments $M$")
    axes[1].set_ylabel("max direction error (deg)")
    axes[1].set_title("force direction: first order")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_geometry(path: Path, *, radius=0.5, particle_radius=0.05, stiffness=1000.0,
                    segments=16, fraction=0.02) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    vertices = bow_polyline(radius, segments)
    overlap = fraction * particle_radius
    distance = particle_radius + radius - overlap
    angles = torch.arange(segments, dtype=torch.float64) * (2.0 * math.pi / segments) + math.pi / segments
    positions = torch.stack([distance * torch.cos(angles), distance * torch.sin(angles)], dim=1)
    kernel, _ = polyline_contact_forces(
        positions, torch.zeros_like(positions), vertices, torch.zeros(2, dtype=torch.float64),
        particle_radius=particle_radius, hull_radius=0.0, stiffness=stiffness)
    exact, _ = exact_circular_hull_force(
        positions, (0.0, 0.0), radius, particle_radius=particle_radius, stiffness=stiffness)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    theta = torch.linspace(0, 2 * math.pi, 400, dtype=torch.float64)
    ax.plot(radius * torch.cos(theta), radius * torch.sin(theta), "k--", lw=1, label="exact circular hull")
    closed = torch.cat([vertices, vertices[:1]], dim=0)
    ax.plot(closed[:, 0], closed[:, 1], "-", color="tab:gray", lw=1.2,
            label=f"inscribed polyline, {segments} segments")
    scale = 4.0 * particle_radius / max(float(exact.norm(dim=1).max()), 1e-30)
    for i in range(positions.shape[0]):
        p = positions[i]
        ax.plot([p[0]], [p[1]], ".", ms=4, color="tab:blue")
        ax.arrow(float(p[0]), float(p[1]),
                 float(exact[i, 0]) * scale, float(exact[i, 1]) * scale,
                 head_width=0.008, color="tab:blue", alpha=0.8, length_includes_head=True)
        norm = float(kernel[i].norm())
        if norm > 0:
            ax.arrow(float(p[0]), float(p[1]),
                     float(kernel[i, 0]) * scale, float(kernel[i, 1]) * scale,
                     head_width=0.008, color="tab:red", alpha=0.9, length_includes_head=True)
    ax.set_aspect("equal")
    ax.set_title(f"contact at {fraction:g}$R_p$ penetration: red = polyline, blue = exact\n"
                 "zero-length red arrows = spurious contact loss", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = build_report(levels=args.levels, fractions=args.fractions, radius=args.radius,
                          particle_radius=args.particle_radius, stiffness=args.stiffness)
    write_report(report, args.output / "report.json")
    write_summary_csv(report, args.output / "discretization_scan.csv")
    figure_convergence(report, args.output / "convergence.png")
    figure_geometry(args.output / "bow_contact_geometry.png", radius=args.radius,
                    particle_radius=args.particle_radius, stiffness=args.stiffness)
    summary = {
        "report": str(args.output / "report.json"),
        "rows": len(report["error_table"]),
        "distance_max_abs_error_m": report["distance_check"]["max_abs_distance_error_m"],
        "vertex_relative_error": report["vertex_check"]["relative_error"],
        "degenerate_reversal_dev": report["invariance"]["degenerate_reversal_max_abs_dev"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
