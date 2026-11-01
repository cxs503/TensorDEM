"""Figures for the ice-hull-propeller three-field coupling study.

Reads ``artifacts/threefield/*.json`` and writes PNGs to
``artifacts/threefield/figures/``. Every number in the figures is read from the
artifacts (nothing is typed in by hand).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensordem.threefield.fragmentation import (  # noqa: E402
    cdf_agreement_r2,
    empirical_cdf,
    weibull_cdf,
)
from tensordem.threefield.thermal import (  # noqa: E402
    ThermalProperties,
    analytic_diameter_ransom_marshall,
    integrate_melting,
)

ARTIFACTS = ROOT / "artifacts" / "threefield"
FIGURES = ARTIFACTS / "figures"


def _load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def figure_resistance(report: dict) -> Path:
    rows = report["criteria"]["a_resistance_vs_formulas"]["detail"]["rows"]
    thickness = [row["thickness_m"] for row in rows]
    dem = [row["dem_resistance_per_width_N_m"] for row in rows]
    lindqvist = [row["formula_lindqvist_per_width_N_m"] for row in rows]
    riska = [row["formula_riska_per_width_N_m"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(thickness, dem, "o-", label="DEM bow contact (unit width)")
    axes[0].plot(thickness, lindqvist, "s--", label="Lindqvist baseline")
    axes[0].plot(thickness, riska, "^--", label="Riska baseline")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("ice thickness h [m]")
    axes[0].set_ylabel("ice resistance per unit bow width [N/m]")
    axes[0].set_title("(a) DEM vs engineering baselines")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    errors = [row["raw_error_pct_riska"] for row in rows]
    axes[1].bar([str(h) for h in thickness], errors, color="#b5651d")
    axes[1].axhline(15.0, color="crimson", linestyle="--", label="15 % gate")
    axes[1].set_xlabel("ice thickness h [m]")
    axes[1].set_ylabel("relative error vs Riska [%]")
    axes[1].set_title("(b) uncalibrated resistance error")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    figure.suptitle(
        "Fig. 1  DEM hull-ice resistance vs level-ice baselines "
        f"(worst raw error {report['criteria']['a_resistance_vs_formulas']['worst_raw_error_pct']:.1f} %)"
    )
    path = FIGURES / "fig1_resistance.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_wake(wake: dict) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    stations = [key for key in wake["profiles"] if key.isdigit()]
    stations.sort(key=int)
    colours = plt.cm.viridis([i / max(len(stations) - 1, 1) for i in range(len(stations))])
    for colour, station in zip(colours, stations):
        profile = wake["profiles"][station]
        radius = profile["radius"]
        velocity = profile["mean_axial_velocity"]
        axes[0].plot(radius, velocity, "o-", color=colour, label=f"x = {station} lu")
        audit = wake["profile_audit"][station]
        sigma = audit["fitted_sigma_lattice"]
        amplitude = audit["fitted_amplitude"]
        u_in = wake["config"]["inflow_velocity"]
        model = [u_in + amplitude * math.exp(-(r**2) / (2 * sigma**2)) for r in radius]
        axes[0].plot(radius, model, "--", color=colour, alpha=0.7)
        axes[1].plot(radius, profile["axial_rms_velocity"], "o-", color=colour, label=f"x = {station} lu")
    axes[0].set_xlabel("radius r [lattice units]")
    axes[0].set_ylabel("mean axial velocity [lattice]")
    axes[0].set_title("(a) race profile, LBM (solid) vs Gaussian model (dashed)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("radius r [lattice units]")
    axes[1].set_ylabel("axial rms velocity [lattice]")
    axes[1].set_title("(b) resolved turbulence intensity")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    distances = [float(station) - float(wake["config"]["disk_diameter_lattice"]) for station in stations]
    sigmas = [wake["profile_audit"][station]["fitted_sigma_lattice"] for station in stations]
    axes[2].plot(distances, sigmas, "o-", label=r"resolved $\sigma(x)$")
    beta = wake["spreading"]["beta_literature"]
    floor = wake["spreading"]["sigma_floor"]
    axes[2].plot(
        distances,
        [floor + beta * d for d in distances],
        "s--",
        label=rf"empirical $\beta={beta}$",
    )
    axes[2].set_xlabel("distance from actuator disk [lattice units]")
    axes[2].set_ylabel(r"Gaussian width $\sigma$ [lattice]")
    axes[2].set_title(r"(c) width growth vs empirical spreading law")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    worst = report_worst(wake)
    figure.suptitle(
        f"Fig. 2  D3Q19 + Smagorinsky propeller race (shape error {worst:.1f} % in the race core)"
    )
    path = FIGURES / "fig2_wake.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def report_worst(wake: dict) -> float:
    values = [
        audit["shape_error_pct"]
        for audit in wake["profile_audit"].values()
        if isinstance(audit, dict)
    ]
    return max(values) if values else float("nan")


def figure_fragments_and_melting(report: dict) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    lengths = report["coupled_pass"]["fragment_lengths_m"]
    pairs = empirical_cdf(lengths)
    axes[0].plot([p[0] for p in pairs], [p[1] for p in pairs], "o", ms=3, label="DEM census (simulated)")
    fit = report["criteria"]["c_fragment_size_weibull"]["detail"]["fit"]
    xs = [value for value, _ in pairs]
    axes[0].plot(
        xs,
        [weibull_cdf(x, fit["shape"], fit["scale"]) for x in xs],
        "-",
        label=f"Weibull fit k={fit['shape']:.2f}, R²={fit['r_squared']:.2f}",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("floe length L [m]")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("(a) fragment size distribution")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    props = ThermalProperties()
    detail = report["criteria"]["d_melting_vs_ranz_marshall"]["detail"]
    d0 = detail["initial_diameter_m"]
    duration = detail["residence_time_s"]
    history = integrate_melting(
        d0, props, relative_velocity_m_s=detail["relative_velocity_m_s"],
        duration_s=duration, dt_s=duration / 240.0, include_sensible=False,
    )
    times = [row["time_s"] for row in history]
    axes[1].plot(times, [row["diameter_m"] for row in history], "-", label="numerical (RK2)")
    axes[1].plot(
        times,
        [analytic_diameter_ransom_marshall(d0, t, detail["relative_velocity_m_s"], props,
                                           include_sensible=False) for t in times],
        "--",
        label="Ranz-Marshall closed form",
    )
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("fragment diameter [m]")
    axes[1].set_title(
        f"(b) melting: diameter error {detail['diameter_error_pct']:.2f} %"
    )
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    ledger = report["criteria"]["e_mass_conservation"]["detail"]
    labels = ["initial", "surviving", "melted", "balance"]
    values = [
        ledger["initial_mass_kg"],
        ledger["surviving_mass_kg"],
        ledger["melted_mass_kg"],
        ledger["balance_kg"],
    ]
    axes[2].bar(labels, values, color=["#444444", "#2a7ab9", "#c0504d", "#76b041"])
    axes[2].set_ylabel("ice mass [kg]")
    axes[2].set_title(
        f"(c) mass ledger, relative error {ledger['relative_error']:.1e}"
    )
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].grid(alpha=0.3, axis="y")
    figure.suptitle("Fig. 3  Fragment statistics, thermal erosion and the mass ledger")
    path = FIGURES / "fig3_fragments_thermal.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_coupling_summary(report: dict) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    names = list(report["criteria"].keys())
    passed = [1.0 if report["criteria"][name]["passed"] else 0.0 for name in names]
    colours = ["#2e7d32" if value else "#c62828" for value in passed]
    axes[0].barh([name.split("_")[0] for name in names], passed, color=colours)
    axes[0].set_xlim(0, 1.2)
    axes[0].set_xlabel("gate status (1 = pass)")
    axes[0].set_title("(a) acceptance gates")
    for index, name in enumerate(names):
        value = report["criteria"][name]
        text = ""
        for key in (
            "worst_calibrated_error_pct",
            "worst_shape_error_pct",
            "fit_r_squared",
            "melt_fraction_error_pct",
            "relative_error",
        ):
            if key in value:
                text = f"{key.split('_')[-2]}={value[key]:.2e}" if value[key] < 1e-3 else f"{value[key]:.2f}"
                break
        axes[0].text(1.02, index, text, va="center", fontsize=7)
    follower = report["follower_ship"]
    axes[1].bar(
        ["melted\nfraction", "floe size\nratio", "resistance relief\n(floe refinement)"],
        [follower["melted_fraction"], follower["floe_size_ratio"], follower["resistance_relief_from_floe_refinement"]],
        color=["#c0504d", "#2a7ab9", "#76b041"],
    )
    axes[1].set_ylabel("dimensionless")
    axes[1].set_title("(b) downstream impact on a following ship")
    axes[1].grid(alpha=0.3, axis="y")
    figure.suptitle(f"Fig. 4  Coupled three-field verdict: {report['verdict']}")
    path = FIGURES / "fig4_coupling_summary.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    report = _load("threefield_report.json")
    wake = _load("wake_lbm.json")
    paths = [
        figure_resistance(report),
        figure_wake(wake),
        figure_fragments_and_melting(report),
        figure_coupling_summary(report),
    ]
    print(json.dumps({"figures": [str(path.relative_to(ROOT)) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())