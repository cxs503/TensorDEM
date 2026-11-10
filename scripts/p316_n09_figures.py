"""Figures for the P316-W1-N09 closure report (read-only over results.json).

Produces four PNGs in ``artifacts/p316_n09/figures``:

1. ``closure_history.png``   base-case resistance history with the three stacked
   components and the closure residual on a twin axis;
2. ``component_shares_vs_thickness.png``  crushing/bending/submersion shares vs
   ice thickness, with the Lindqvist shares as open markers;
3. ``dem_vs_lindqvist.png``  component-by-component DEM vs Lindqvist bars at the
   base case, plus the reduction-convention sensitivity;
4. ``numerical_axes.png``    particle refinement and mass-scaling axes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p316_n09"
FIG = ART / "figures"
MECHANISMS = ("crushing", "bending", "submersion")
COLORS = {"crushing": "#1f4e79", "bending": "#c55a11", "submersion": "#2e75b6"}


def _load() -> tuple[dict, list]:
    results = json.loads((ART / "results.json").read_text())
    series = json.loads((ART / "series_base.json").read_text())
    return results, series


def figure_history(series: list[dict], path: Path) -> None:
    time = [row["time_s"] * 1e3 for row in series]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    positive = {name: [max(row[f"r_{name}_N"], 0.0) for row in series] for name in MECHANISMS}
    negative = {name: [min(row[f"r_{name}_N"], 0.0) for row in series] for name in MECHANISMS}
    bottom = [0.0] * len(series)
    for name in MECHANISMS:
        ax.fill_between(time, bottom, [b + v for b, v in zip(bottom, positive[name])],
                        color=COLORS[name], alpha=0.85, label=f"R_{name}")
        bottom = [b + v for b, v in zip(bottom, positive[name])]
    for name in MECHANISMS:
        ax.fill_between(time, [0.0 if not any(negative[name]) else v for v in negative[name]],
                        0.0, color=COLORS[name], alpha=0.35)
    ax.plot(time, [row["r_total_N"] for row in series], color="k", lw=1.0, label="R_total")
    ax.set_xlabel("simulated time [ms]")
    ax.set_ylabel("hull reaction along heading [N]")
    ax.set_title("Contact-force decomposition, base case (stacked components)")
    residual = max(abs(row["relative_residual"]) for row in series)
    ax.text(0.02, 0.98, f"max |residual|/|R| = {residual:.1e}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_shares(results: dict, path: Path) -> None:
    sweep = results["thickness_sweep"]
    keys = sorted(sweep, key=float)
    thickness = [float(key) for key in keys]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for name in MECHANISMS:
        ax.plot(thickness, [sweep[key]["steady"]["shares"][name] for key in keys],
                "o-", color=COLORS[name], label=f"DEM {name}")
        ax.plot(thickness, [sweep[key]["steady"]["shares"][name] for key in keys],
                "o", mfc="none", mec=COLORS[name])
    ax.set_xlabel("ice thickness h [m]")
    ax.set_ylabel("share of the heading resistance [-]")
    ax.set_title("Mechanism shares vs ice thickness (DEM, flow-through window)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_comparison(results: dict, path: Path) -> None:
    base = results["base"]
    rows = base["lindqvist"]["rows"]
    labels = [row["mechanism"] for row in rows]
    dem = [row["dem_N_m"] for row in rows]
    lind = [row["lindqvist_N_m"] for row in rows]
    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=160)
    axes[0].bar([i - 0.2 for i in x], dem, width=0.4, color="#1f4e79", label="DEM")
    axes[0].bar([i + 0.2 for i in x], lind, width=0.4, color="#a6a6a6", label="Lindqvist")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("component per unit width [N/m]")
    axes[0].set_title(f"components (total error {base['lindqvist']['total_signed_error_pct']:+.1f}%)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, axis="y")
    conventions = [
        ("L = model length", base["lindqvist"]["total_signed_error_pct"]),
        ("L = plate length", base["lindqvist_characteristic_length"]["total_signed_error_pct"]),
        ("crushing as line load", base["lindqvist_line_load"]["error_pct"]),
    ]
    axes[1].barh([c[0] for c in conventions], [c[1] for c in conventions], color="#c55a11")
    axes[1].axvline(0.0, color="k", lw=0.8)
    axes[1].axvline(-15.0, color="r", ls="--", lw=0.8)
    axes[1].axvline(15.0, color="r", ls="--", lw=0.8)
    axes[1].set_xlabel("signed deviation from Lindqvist [%]")
    axes[1].set_title("reduction-convention sensitivity")
    axes[1].grid(alpha=0.3, axis="x")
    fig.suptitle("DEM vs Lindqvist on the same geometry", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_axes(results: dict, path: Path) -> None:
    coarse = results["refinement"]["coarse"]["steady_per_width"]["r_total_N_m"]
    fine = results["refinement"]["fine"]["steady_per_width"]["r_total_N_m"]
    mass = results["mass_scale_axis"]
    mass_keys = sorted(mass, key=float)
    mass_values = [mass[key]["steady_per_width"]["r_total_N_m"] for key in mass_keys]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=160)
    axes[0].bar(["coarse r", "fine r/2"], [coarse, fine], color=["#1f4e79", "#2e75b6"])
    change = results["criteria"]["d_grid_refinement"]["relative_change_pct"]
    axes[0].set_ylabel("R per unit width [N/m]")
    axes[0].set_title(f"particle refinement: change {change:.1f}%")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].plot(mass_keys, mass_values, "o-", color="#c55a11")
    axes[1].set_xlabel("mass scale [-]")
    axes[1].set_ylabel("R per unit width [N/m]")
    axes[1].set_title(f"mass-scaling axis: spread {results['criteria']['mass_scale_axis']['spread_pct']:.1f}%")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    results, series = _load()
    figure_history(series, FIG / "closure_history.png")
    figure_shares(results, FIG / "component_shares_vs_thickness.png")
    figure_comparison(results, FIG / "dem_vs_lindqvist.png")
    figure_axes(results, FIG / "numerical_axes.png")
    print(f"figures written to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())