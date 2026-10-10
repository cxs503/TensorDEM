"""Assemble paper-ready comparison curves from archived TensorDEM benchmark outputs.

This script is post-processing only. It never runs or modifies the DEM solver and
never fabricates missing reference data. Figures are emitted only when a matching
CSV with finite numeric columns is present.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_csv(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.is_file():
        return direct
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    return matches[0] if matches else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _numeric(rows: list[dict[str, str]], key: str) -> list[float] | None:
    if not rows or key not in rows[0]:
        return None
    try:
        values = [float(row[key]) for row in rows]
    except (TypeError, ValueError, KeyError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def _emit_curve(
    root: Path, output: Path, manifest: list[dict[str, Any]], *,
    filename: str, figure_name: str, xkey: str, ykey: str,
    xlabel: str, ylabel: str, title: str, transform_y: Callable[[float], float] | None = None,
) -> None:
    path = _find_csv(root, filename)
    if path is None:
        manifest.append({"figure": figure_name, "status": "SKIPPED", "reason": f"input CSV not found: {filename}"})
        return
    rows = _read_csv(path)
    xs, ys = _numeric(rows, xkey), _numeric(rows, ykey)
    if xs is None or ys is None or len(xs) != len(ys) or not xs:
        manifest.append({"figure": figure_name, "status": "SKIPPED", "input": str(path), "reason": f"required finite numeric columns missing: {xkey}, {ykey}"})
        return
    if transform_y:
        ys = [transform_y(value) for value in ys]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.plot(xs, ys, linewidth=1.8, label=path.parent.name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend()
    target = output / figure_name
    fig.savefig(target, dpi=220)
    plt.close(fig)
    manifest.append({"figure": figure_name, "status": "GENERATED", "input": str(path), "x_column": xkey, "y_column": ykey, "points": len(xs)})


def _emit_resolution(root: Path, output: Path, manifest: list[dict[str, Any]]) -> None:
    filename = "bending_resolution_summary.csv"
    path = _find_csv(root, filename)
    name = "bending_resolution_sensitivity.png"
    if path is None:
        manifest.append({"figure": name, "status": "SKIPPED", "reason": f"input CSV not found: {filename}"})
        return
    rows = _read_csv(path)
    xkey = "resolution_factor"
    xs = _numeric(rows, xkey)
    candidates = [
        ("peak_support_reaction_abs_N", "Peak support reaction (N)"),
        ("peak_midspan_deflection_abs_m", "Peak midspan deflection (m)"),
        ("broken_bonds", "Broken bonds (-)"),
        ("final_broken_bonds", "Final broken bonds (-)"),
    ]
    series = [(key, label, _numeric(rows, key)) for key, label in candidates]
    series = [(key, label, values) for key, label, values in series if values is not None]
    if xs is None or not series:
        manifest.append({"figure": name, "status": "SKIPPED", "input": str(path), "reason": "resolution factor or recognized observable columns are missing"})
        return
    fig, axes = plt.subplots(len(series), 1, figsize=(8.5, 3.6 * len(series)), constrained_layout=True, squeeze=False)
    for ax, (key, label, ys) in zip(axes[:, 0], series):
        ax.plot(xs, ys, marker="o", linewidth=1.7)
        ax.set_xlabel("Particle-resolution factor (-)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.28)
    fig.suptitle("Three-point-bending particle-resolution sensitivity")
    target = output / name
    fig.savefig(target, dpi=220)
    plt.close(fig)
    manifest.append({"figure": name, "status": "GENERATED", "input": str(path), "series": [key for key, _, _ in series], "points": len(xs), "note": "Sensitivity only; stiffness is not recalibrated with resolution."})


def _emit_ucs_timestep(root: Path, output: Path, manifest: list[dict[str, Any]]) -> None:
    filename = "ucs_timestep_convergence.csv"
    path = _find_csv(root, filename)
    name = "ucs_timestep_sensitivity.png"
    if path is None:
        manifest.append({"figure": name, "status": "SKIPPED", "reason": f"input CSV not found: {filename}"})
        return
    rows = _read_csv(path)
    xkey = "dt_factor"
    xs = _numeric(rows, xkey)
    candidates = [
        ("peak_stress_Pa", "Peak engineering stress (Pa)"),
        ("peak_engineering_stress_Pa", "Peak engineering stress (Pa)"),
        ("peak_load_N", "Peak load (N)"),
        ("peak_stress_relative_delta_vs_fine", "Peak-stress relative difference vs finest (-)"),
    ]
    series = [(key, label, _numeric(rows, key)) for key, label in candidates]
    series = [(key, label, values) for key, label, values in series if values is not None]
    if xs is None or not series:
        manifest.append({"figure": name, "status": "SKIPPED", "input": str(path), "reason": "dt_factor or recognized peak-observable columns are missing"})
        return
    fig, axes = plt.subplots(len(series), 1, figsize=(8.5, 3.6 * len(series)), constrained_layout=True, squeeze=False)
    for ax, (key, label, ys) in zip(axes[:, 0], series):
        ax.plot(xs, ys, marker="o", linewidth=1.7)
        ax.set_xlabel("Time-step factor relative to recommended step (-)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.28)
    fig.suptitle("UCS time-step sensitivity")
    target = output / name
    fig.savefig(target, dpi=220)
    plt.close(fig)
    manifest.append({"figure": name, "status": "GENERATED", "input": str(path), "series": [key for key, _, _ in series], "points": len(xs), "note": "The finest tested step is a within-campaign reference, not an exact solution."})


def _emit_formula_comparison(root: Path, output: Path, manifest: list[dict[str, Any]]) -> None:
    filename = "mt_uikku_formula_baselines_cases.csv"
    path = _find_csv(root, filename)
    name = "formula_reference_comparison.png"
    if path is None:
        manifest.append({"figure": name, "status": "SKIPPED", "reason": f"input CSV not found: {filename}"})
        return
    rows = _read_csv(path)
    ref_key = "experimental_reference_kN"
    reference = _numeric(rows, ref_key)
    model_keys = [
        ("lindqvist_prediction_kN", "Lindqvist"),
        ("riska_prediction_kN", "Riska"),
        ("jeong_prediction_kN", "Jeong"),
        ("keinonen_prediction_kN", "Keinonen"),
    ]
    models = [(key, label, _numeric(rows, key)) for key, label in model_keys]
    models = [(key, label, values) for key, label, values in models if values is not None]
    if reference is None or not models:
        manifest.append({"figure": name, "status": "SKIPPED", "input": str(path), "reason": "reference values or published formula prediction columns are missing"})
        return
    labels = [row.get("test_id", str(i + 1)) for i, row in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(labels, reference, marker="o", linewidth=2.2, label="Experimental reference")
    for key, label, values in models:
        ax.plot(labels, values, marker="s", linewidth=1.4, label=label)
    ax.set_xlabel("Reference case")
    ax.set_ylabel("Ice resistance (kN)")
    ax.set_title("Published empirical formulae versus reference data")
    ax.grid(True, alpha=0.28)
    ax.legend()
    ax.tick_params(axis="x", rotation=25)
    target = output / name
    fig.savefig(target, dpi=220)
    plt.close(fig)
    manifest.append({"figure": name, "status": "GENERATED", "input": str(path), "series": ["experimental_reference_kN", *[key for key, _, _ in models]], "points": len(reference), "note": "This compares published empirical formulae with reference data, not TensorDEM with experiment."})


def build_paper_figures(input_root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    _emit_curve(input_root, output, manifest,
        filename="ucs_history.csv", figure_name="ucs_stress_strain.png",
        xkey="axial_engineering_strain", ykey="engineering_stress_Pa",
        xlabel="Axial engineering strain (-)", ylabel="Engineering stress (Pa)",
        title="TensorDEM DEM3D platen-compression response")
    _emit_curve(input_root, output, manifest,
        filename="three_point_bending_history.csv", figure_name="three_point_bending_load_deflection.png",
        xkey="midspan_deflection_m", ykey="support_reaction_z_N",
        xlabel="Midspan deflection (m)", ylabel="Support reaction z (N)",
        title="Three-point-bending numerical load-transfer fixture",
        transform_y=abs)
    _emit_curve(input_root, output, manifest,
        filename="moving_wedge_bow_history.csv", figure_name="moving_wedge_resistance_time.png",
        xkey="time_s", ykey="ice_resistance_N",
        xlabel="Time (s)", ylabel="Ice resistance (N)",
        title="Prescribed moving-wedge resistance history")
    _emit_curve(input_root, output, manifest,
        filename="moving_wedge_bow_history.csv", figure_name="moving_wedge_resistance_vs_travel.png",
        xkey="bow_tip_x_m", ykey="ice_resistance_N",
        xlabel="Bow-tip x coordinate (m)", ylabel="Ice resistance (N)",
        title="Moving-wedge resistance versus bow position")
    _emit_resolution(input_root, output, manifest)
    _emit_ucs_timestep(input_root, output, manifest)
    _emit_formula_comparison(input_root, output, manifest)
    result = {
        "protocol": "tensordem-dem3d-paper-figures-v1",
        "input_root": str(input_root.resolve()),
        "output_dir": str(output.resolve()),
        "generated_count": sum(item["status"] == "GENERATED" for item in manifest),
        "skipped_count": sum(item["status"] == "SKIPPED" for item in manifest),
        "figures": manifest,
        "interpretation": (
            "Plots are post-processing products of existing CSV files. Missing inputs are reported, not synthesized. "
            "A numerical regression or sensitivity plot is not experimental validation."
        ),
    }
    (output / "paper_figure_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "paper_figures_index.md").open("w", encoding="utf-8") as stream:
        stream.write("# Paper figure index\n\n")
        stream.write(f"Generated: {result['generated_count']}; skipped: {result['skipped_count']}.\n\n")
        for item in manifest:
            stream.write(f"## {item['figure']} — {item['status']}\n\n")
            stream.write(f"{item.get('input', item.get('reason', ''))}\n\n")
            if item.get("note"):
                stream.write(f"Interpretation: {item['note']}\n\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("."),
                        help="Root containing benchmark output folders (default: current directory).")
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-paper-figures"))
    args = parser.parse_args()
    if not args.input_root.is_dir():
        parser.error(f"input root does not exist: {args.input_root}")
    report = build_paper_figures(args.input_root, args.output)
    print(json.dumps({
        "protocol": report["protocol"],
        "generated_count": report["generated_count"],
        "skipped_count": report["skipped_count"],
        "output_dir": report["output_dir"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
