"""Generate deterministic 3-D DEM benchmark cloud maps, history plots, and GIFs.

Input is the sampled trajectory written by scripts/benchmark_dem3d.py. Matplotlib
is an optional visualization dependency; this tool never changes the simulation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "tensordem-dem3d-visualization-trajectory-v1":
        raise ValueError(f"unsupported visualization trajectory: {path}")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("visualization trajectory has no frames")
    n = payload["initial_positions"].shape[0]
    if payload["initial_positions"].shape != (n, 3):
        raise ValueError("initial_positions must have shape (N, 3)")
    if payload["pairs"].ndim != 2 or payload["pairs"].shape[1] != 2:
        raise ValueError("pairs must have shape (B, 2)")
    for frame in frames:
        if frame["positions"].shape != (n, 3) or frame["alive"].shape != payload["pairs"].shape[:1]:
            raise ValueError("frame positions/alive dimensions do not match topology")
        if not torch.isfinite(frame["positions"]).all():
            raise ValueError("trajectory contains non-finite positions")
    return payload


def _damage_fraction(alive: torch.Tensor, pairs: torch.Tensor, n: int) -> torch.Tensor:
    damage = torch.zeros(n, dtype=torch.float64)
    degree = torch.zeros(n, dtype=torch.float64)
    if pairs.numel():
        for endpoint in (pairs[:, 0], pairs[:, 1]):
            degree.index_add_(0, endpoint.long(), torch.ones(len(endpoint), dtype=torch.float64))
            broken = (~alive).to(torch.float64)
            damage.index_add_(0, endpoint.long(), broken)
    return torch.where(degree > 0, damage / degree.clamp_min(1), torch.zeros_like(damage))


def _axes_equal(ax: Any, points: torch.Tensor) -> None:
    low = points.min(dim=0).values.numpy()
    high = points.max(dim=0).values.numpy()
    center = (low + high) / 2
    span = max(float((high - low).max()), 1e-9)
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
    ax.set_box_aspect((1, 1, 1))


def _plot_frame(plt: Any, payload: dict[str, Any], frame: dict[str, Any], path: Path,
                field: str = "damage") -> None:
    positions = frame["positions"].to(torch.float64)
    initial = payload["initial_positions"].to(torch.float64)
    pairs = payload["pairs"].long()
    damage = _damage_fraction(frame["alive"].bool(), pairs, len(positions))
    if field == "displacement":
        values = torch.linalg.vector_norm(positions - initial, dim=1)
        label = "Displacement magnitude (m)"
    else:
        values = damage
        label = "Broken-bond fraction per particle (-)"
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        positions[:, 0].numpy(), positions[:, 1].numpy(), positions[:, 2].numpy(),
        c=values.numpy(), cmap="inferno" if field == "damage" else "viridis",
        vmin=0.0 if field == "damage" else None,
        vmax=1.0 if field == "damage" else None,
        s=28, depthshade=False,
    )
    fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08, label=label)
    tool = frame.get("tool_position")
    if isinstance(tool, torch.Tensor) and torch.isfinite(tool).all() and float(torch.linalg.vector_norm(tool)) < 1e3:
        ax.scatter([float(tool[0])], [float(tool[1])], [float(tool[2])],
                   marker="s", s=100, c="tab:blue", label="prescribed indenter")
        ax.legend(loc="upper right")
    _axes_equal(ax, positions)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"DEM3D {field} | step={frame['step']} | t={frame['time_s']:.6g} s")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_history(plt: Any, case_dir: Path, output: Path) -> bool:
    candidates = [
        ("history_3d.csv", ("reaction_z", "peak_abs_reaction_z_N"), "Reaction z (N)"),
        ("ucs_history.csv", ("mean_compressive_load_N", "engineering_stress_Pa"), "Compression response"),
        ("platen_compression_history.csv", ("top_platen_reaction_z_N", "bottom_platen_reaction_z_N"), "Platen reaction (N)"),
    ]
    for filename, keys, ylabel in candidates:
        path = case_dir / filename
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        xkey = "time" if "time" in rows[0] else ("time_s" if "time_s" in rows[0] else "step")
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        plotted = False
        for key in keys:
            if key in rows[0]:
                xs = [float(row[xkey]) for row in rows]
                ys = [float(row[key]) for row in rows]
                if all(math.isfinite(x) for x in xs + ys):
                    ax.plot(xs, ys, label=key)
                    plotted = True
        if plotted:
            ax.set_xlabel("Time (s)" if xkey in ("time", "time_s") else "Step")
            ax.set_ylabel(ylabel)
            ax.set_title(f"DEM3D history — {case_dir.name}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.savefig(output / "force_history.png", dpi=160)
            plt.close(fig)
            return True
        plt.close(fig)
    return False


def visualize_case(case_dir: Path, output: Path, *, make_gif: bool = True) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectory_path = case_dir / "visualization_trajectory_3d.pt"
    report: dict[str, Any] = {
        "case": case_dir.name,
        "trajectory_available": trajectory_path.is_file(),
        "outputs": [],
        "limitations": [
            "Damage color is incident broken-bond fraction per particle, not a continuum stress field.",
            "Only saved sample times are animated; interpolation between frames is not performed.",
            "Visualization does not imply calibration or experimental validation.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    if trajectory_path.is_file():
        payload = _load(trajectory_path)
        frames = payload["frames"]
        for name, index, field in (
            ("damage_initial.png", 0, "damage"),
            ("damage_final.png", len(frames) - 1, "damage"),
            ("displacement_final.png", len(frames) - 1, "displacement"),
        ):
            _plot_frame(plt, payload, frames[index], output / name, field)
            report["outputs"].append(name)
        if make_gif and len(frames) >= 2:
            from matplotlib.animation import FuncAnimation, PillowWriter
            initial = payload["initial_positions"].to(torch.float64)
            all_positions = torch.cat([f["positions"].to(torch.float64) for f in frames], dim=0)
            fig = plt.figure(figsize=(8, 6), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")
            first = frames[0]
            damage = _damage_fraction(first["alive"].bool(), payload["pairs"].long(), len(first["positions"]))
            scat = ax.scatter(first["positions"][:, 0].numpy(), first["positions"][:, 1].numpy(),
                              first["positions"][:, 2].numpy(), c=damage.numpy(), cmap="inferno",
                              vmin=0, vmax=1, s=28, depthshade=False)
            fig.colorbar(scat, ax=ax, shrink=0.72, pad=0.08, label="Broken-bond fraction per particle (-)")
            _axes_equal(ax, all_positions)
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
            title = ax.set_title("")
            def update(index: int):
                frame = frames[index]
                pos = frame["positions"]
                alive = frame["alive"].bool()
                values = _damage_fraction(alive, payload["pairs"].long(), len(pos))
                scat._offsets3d = (pos[:, 0].numpy(), pos[:, 1].numpy(), pos[:, 2].numpy())
                scat.set_array(values.numpy())
                title.set_text(f"DEM3D damage | step={frame['step']} | t={frame['time_s']:.6g} s")
                return scat, title
            animation = FuncAnimation(fig, update, frames=len(frames), interval=250, blit=False)
            animation.save(output / "damage_evolution.gif", writer=PillowWriter(fps=4))
            plt.close(fig)
            report["outputs"].append("damage_evolution.gif")
        report["frame_count"] = len(frames)
        report["particle_count"] = int(payload["initial_positions"].shape[0])
    if _plot_history(plt, case_dir, output):
        report["outputs"].append("force_history.png")
    (output / "visualization_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results-dem3d-benchmark-suite"))
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-visualizations"))
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f"input directory does not exist: {args.input}")
    cases = sorted(path for path in args.input.iterdir() if path.is_dir())
    if not cases:
        parser.error(f"no case directories found in {args.input}")
    reports = [visualize_case(case, args.output / case.name, make_gif=not args.no_gif) for case in cases]
    summary = {
        "protocol": "tensordem-dem3d-visualization-v1",
        "case_count": len(reports),
        "cases_with_clouds": sum(any(name.startswith(("damage_", "displacement_")) for name in row["outputs"]) for row in reports),
        "cases_with_gif": sum("damage_evolution.gif" in row["outputs"] for row in reports),
        "cases_with_history_plot": sum("force_history.png" in row["outputs"] for row in reports),
        "cases": reports,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["cases_with_clouds"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
