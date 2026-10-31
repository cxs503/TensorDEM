"""Calibrate the hull-ice friction temperature dependence and integrate it.

Pipeline
--------
1. DEM sliding-friction experiment at five temperatures, ten repeats each.
2. Least-squares identification of ``mu(T) = mu0 + alpha |T - T_melt|^beta``.
3. Reduced MD ice/iron interface shear at the same temperatures with the single
   free parameter identified at the reference temperature only.
4. Cross-scale comparison of the two friction coefficients.
5. Level-ice resistance with the fitted ``mu(T)`` and the -5 degC / -20 degC
   comparison.
6. Machine-checkable acceptance criteria, JSON/CSV artefacts and figures.

Usage::

    python scripts/calibrate_hull_ice_friction.py --output artifacts/hull_ice_friction
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.hull_ice_friction import (  # noqa: E402
    MU_LITERATURE_HIGH,
    MU_LITERATURE_LOW,
    MU_REFERENCE_TEMPERATURE_C,
    DEMFrictionConfig,
    dem_temperature_sweep,
    fit_mu_temperature,
    friction_temperature_model,
)
from tensordem.ice_resistance import (  # noqa: E402
    VesselIceGeometry,
    resistance_temperature_comparison,
    resistance_temperature_sweep,
)
from tensordem.md_interface_friction import (  # noqa: E402
    MDFrictionConfig,
    calibrate_epsilon_iron_water,
    md_temperature_sweep,
)

TEMPERATURES = [-5.0, -10.0, -15.0, -20.0, -25.0]
REFERENCE_TEMPERATURE = MU_REFERENCE_TEMPERATURE_C


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def run(output: Path, repeats: int, md_repeats: int, fast: bool, reuse_dem: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    timings = {}
    torch.set_num_threads(1)

    # --- 1. DEM sweep ------------------------------------------------------
    started = time.time()
    if reuse_dem is not None:
        previous = json.loads(Path(reuse_dem).read_text(encoding="utf-8"))
        dem = previous["dem_sweep"]
        repeats = previous["dem_repeats_per_temperature"]
        dem_config = DEMFrictionConfig(**previous["dem_configuration"])
        timings["dem_sweep_s"] = previous["timings"].get("dem_sweep_s", 0.0)
        timings["dem_sweep_reused_from"] = str(reuse_dem)
    else:
        dem_config = DEMFrictionConfig(
            steps=1200 if fast else 2000, n_x=7 if fast else 9, n_y=4
        )
        dem = dem_temperature_sweep(
            TEMPERATURES, repeats=repeats, template=dem_config
        )
        timings["dem_sweep_s"] = time.time() - started

    dem_points = [(record["temperature_C"], record["mu_mean"]) for record in dem]
    fit = fit_mu_temperature([point[0] for point in dem_points], [point[1] for point in dem_points])
    mu_parameters = {"mu0": fit["mu0"], "alpha": fit["alpha"], "beta": fit["beta"]}

    # --- 3. MD calibration + sweep -----------------------------------------
    started = time.time()
    md_config = MDFrictionConfig(steps=800 if fast else 2000)
    calibration = calibrate_epsilon_iron_water(
        mu_parameters["mu0"] + mu_parameters["alpha"] * abs(REFERENCE_TEMPERATURE) ** mu_parameters["beta"],
        REFERENCE_TEMPERATURE,
        grid=(0.03, 0.13, 0.29) if fast else (0.03, 0.08, 0.13, 0.18, 0.23, 0.29, 0.35),
        template=md_config,
        repeats=md_repeats,
    )
    md_template = MDFrictionConfig(
        **{
            **md_config.__dict__,
            "lj_epsilon_iron_water_kJ": calibration["lj_epsilon_iron_water_kJ"],
        }
    )
    md = md_temperature_sweep(TEMPERATURES, repeats=md_repeats, template=md_template)
    timings["md_sweep_s"] = time.time() - started

    # --- 4. cross-scale comparison ----------------------------------------
    cross_scale = []
    for dem_record, md_record in zip(dem, md):
        if abs(dem_record["temperature_C"] - md_record["temperature_C"]) > 1e-9:
            raise RuntimeError("temperature grids do not line up")
        reference = md_record["mu_mean"]
        error = abs(dem_record["mu_mean"] - reference) / reference
        cross_scale.append(
            {
                "temperature_C": dem_record["temperature_C"],
                "mu_dem": dem_record["mu_mean"],
                "mu_dem_std": dem_record["mu_std"],
                "mu_md": reference,
                "mu_md_std": md_record["mu_std"],
                "absolute_difference": dem_record["mu_mean"] - reference,
                "relative_difference_to_md": error,
                "within_30_percent": error < 0.30,
            }
        )
    max_cross_scale_error = max(record["relative_difference_to_md"] for record in cross_scale)

    # --- 5. resistance -----------------------------------------------------
    geometry = VesselIceGeometry()
    resistance = resistance_temperature_sweep(
        TEMPERATURES, geometry=geometry, mu_parameters=mu_parameters
    )
    comparison = resistance_temperature_comparison(
        -5.0, -20.0, geometry=geometry, mu_parameters=mu_parameters
    )

    # --- 6. criteria -------------------------------------------------------
    reference_mu = float(
        friction_temperature_model(REFERENCE_TEMPERATURE, **mu_parameters)
    )
    band_centre = 0.5 * (MU_LITERATURE_LOW + MU_LITERATURE_HIGH)
    monotone = all(
        colder["mu_mean"] > warmer["mu_mean"]
        for warmer, colder in zip(dem[:-1], dem[1:])
    )

    def criterion(name: str, value, passed: bool, **extra) -> dict:
        return {"criterion": name, "value": value, "passed": bool(passed), **extra}

    criteria = [
        criterion(
            "a_mu_monotone_decreasing_with_temperature",
            {"mu_by_temperature": {record["temperature_C"]: record["mu_mean"] for record in dem}},
            monotone,
            note="d(mu)/dT < 0: friction increases as the ice cools",
        ),
        criterion(
            "b_reference_mu_versus_literature",
            {
                "mu_at_reference": reference_mu,
                "literature_band": [MU_LITERATURE_LOW, MU_LITERATURE_HIGH],
                "relative_error_to_band_centre": abs(reference_mu - band_centre) / band_centre,
                "within_literature_band": MU_LITERATURE_LOW <= reference_mu <= MU_LITERATURE_HIGH,
            },
            MU_LITERATURE_LOW <= reference_mu <= MU_LITERATURE_HIGH
            or abs(reference_mu - band_centre) / band_centre < 0.20,
            note="Bowden & Tabor (1954), ice on steel at about -10 degC",
        ),
        criterion(
            "c_cross_scale_agreement",
            {
                "max_relative_difference": max_cross_scale_error,
                "per_temperature": cross_scale,
            },
            max_cross_scale_error < 0.30,
            note="|mu_DEM - mu_MD| / mu_MD at matched temperatures",
        ),
        criterion(
            "d_temperature_effect_on_resistance",
            {
                "warm_temperature_C": -5.0,
                "cold_temperature_C": -20.0,
                "warm_resistance_N": comparison["warm"]["total_resistance_N"],
                "cold_resistance_N": comparison["cold"]["total_resistance_N"],
                "relative_increase": comparison["delta_relative_to_warm"],
            },
            comparison["delta_relative_to_warm"] > 0.10,
            note="resistance of the colder state must exceed the warmer state by >10 %",
        ),
    ]
    passed = all(item["passed"] for item in criteria)
    criteria_block = {"all_passed": passed, "criteria": criteria}

    summary = {
        "protocol": "tensordem-hull-ice-friction-calibration-v1",
        "temperatures_C": TEMPERATURES,
        "dem_repeats_per_temperature": repeats,
        "md_repeats_per_temperature": md_repeats,
        "dem_configuration": {
            **{key: value for key, value in dem_config.__dict__.items()},
        },
        "mu_temperature_fit": fit,
        "md_calibration": calibration,
        "dem_sweep": dem,
        "md_sweep": md,
        "cross_scale": cross_scale,
        "resistance": resistance,
        "resistance_comparison": comparison,
        "geometry": {key: value for key, value in geometry.__dict__.items()},
        "criteria": criteria_block,
        "timings": timings,
        "artefacts": {},
    }

    # --- write artefacts ---------------------------------------------------
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    with (output / "dem_friction.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "temperature_C",
                "mu_mean",
                "mu_std",
                "mu_adhesion_mean",
                "mu_ploughing_mean",
                "normal_load_N_mean",
                "interface_mu",
                "repeats",
            ]
        )
        for record in dem:
            writer.writerow(
                [
                    record["temperature_C"],
                    record["mu_mean"],
                    record["mu_std"],
                    record["mu_adhesion_mean"],
                    record["mu_ploughing_mean"],
                    record["normal_load_N_mean"],
                    record["interface_mu"],
                    record["repeats"],
                ]
            )
    with (output / "md_friction.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["temperature_C", "mu_mean", "mu_std", "kinetic_temperature_K_mean"])
        for record in md:
            writer.writerow(
                [
                    record["temperature_C"],
                    record["mu_mean"],
                    record["mu_std"],
                    record["kinetic_temperature_K_mean"],
                ]
            )
    with (output / "cross_scale.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "temperature_C",
                "mu_dem",
                "mu_md",
                "relative_difference_to_md",
                "within_30_percent",
            ]
        )
        for record in cross_scale:
            writer.writerow(
                [
                    record["temperature_C"],
                    record["mu_dem"],
                    record["mu_md"],
                    record["relative_difference_to_md"],
                    record["within_30_percent"],
                ]
            )
    with (output / "resistance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "temperature_C",
                "friction_coefficient",
                "crushing_N",
                "breaking_N",
                "submersion_N",
                "friction_N",
                "total_N",
                "friction_share",
            ]
        )
        for record in resistance:
            writer.writerow(
                [
                    record["temperature_C"],
                    record["friction_coefficient"],
                    record["crushing_resistance_N"],
                    record["breaking_resistance_N"],
                    record["submersion_resistance_N"],
                    record["friction_resistance_N"],
                    record["total_resistance_N"],
                    record["friction_share"],
                ]
            )

    figures = make_figures(output, dem, md, fit, cross_scale, resistance, comparison)
    summary["artefacts"] = {
        "summary_json": str(output / "summary.json"),
        "dem_csv": str(output / "dem_friction.csv"),
        "md_csv": str(output / "md_friction.csv"),
        "cross_scale_csv": str(output / "cross_scale.csv"),
        "resistance_csv": str(output / "resistance.csv"),
        "figures": [str(path) for path in figures],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    return summary


def make_figures(output, dem, md, fit, cross_scale, resistance, comparison) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    temperatures = [record["temperature_C"] for record in dem]
    band_low = [MU_LITERATURE_LOW] * len(temperatures)
    band_high = [MU_LITERATURE_HIGH] * len(temperatures)

    # figure 1: mu(T) DEM + fit + MD + literature band
    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    axes.errorbar(
        temperatures,
        [record["mu_mean"] for record in dem],
        yerr=[record["mu_std"] for record in dem],
        fmt="o",
        capsize=3,
        label="DEM sliding experiment (10 repeats)",
        color="tab:blue",
    )
    grid = [t / 10.0 for t in range(-250, -19)]
    axes.plot(
        grid,
        [friction_temperature_model(value, **{k: fit[k] for k in ("mu0", "alpha", "beta")}) for value in grid],
        "-",
        color="tab:blue",
        lw=1.2,
        label=(
            f"fit $\\mu=\\mu_0+\\alpha|T|^{{\\beta}}$: "
            f"$\\mu_0$={fit['mu0']:.4f}, $\\alpha$={fit['alpha']:.4f}, $\\beta$={fit['beta']:.3f}"
        ),
    )
    axes.errorbar(
        temperatures,
        [record["mu_mean"] for record in md],
        yerr=[record["mu_std"] for record in md],
        fmt="s",
        capsize=3,
        label="MD ice/iron interface shear",
        color="tab:red",
    )
    axes.fill_between(
        temperatures, band_low, band_high, color="0.75", alpha=0.5,
        label="Bowden & Tabor (1954) band",
    )
    axes.set_xlabel("ice temperature [degC]")
    axes.set_ylabel("friction coefficient $\\mu$ [-]")
    axes.set_title("Hull-ice friction versus temperature: DEM, MD, literature band")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7, loc="upper right")
    path = output / "fig1_mu_vs_temperature.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    # figure 2: cross-scale 1:1 plot
    figure, axes = plt.subplots(figsize=(4.8, 4.6))
    limits = [
        0.0,
        max(
            max(record["mu_dem"] for record in cross_scale),
            max(record["mu_md"] for record in cross_scale),
        )
        * 1.35,
    ]
    axes.plot(limits, limits, "k--", lw=1.0, label="1:1")
    axes.plot(limits, [value * 1.30 for value in limits], "k:", lw=0.8, label="+30 %")
    axes.plot(limits, [value * 0.70 for value in limits], "k:", lw=0.8, label="-30 %")
    axes.scatter(
        [record["mu_md"] for record in cross_scale],
        [record["mu_dem"] for record in cross_scale],
        c=[record["temperature_C"] for record in cross_scale],
        cmap="viridis",
        s=42,
        zorder=3,
    )
    for record in cross_scale:
        axes.annotate(
            f"{record['temperature_C']:.0f} C",
            (record["mu_md"], record["mu_dem"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    axes.set_xlabel("$\\mu_{MD}$ [-]")
    axes.set_ylabel("$\\mu_{DEM}$ [-]")
    axes.set_title("Cross-scale friction comparison")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7)
    path = output / "fig2_cross_scale.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    # figure 3: resistance components versus temperature
    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    temperatures_r = [record["temperature_C"] for record in resistance]
    axes.stackplot(
        temperatures_r,
        [
            [record["crushing_resistance_N"] / 1e3 for record in resistance],
            [record["breaking_resistance_N"] / 1e3 for record in resistance],
            [record["submersion_resistance_N"] / 1e3 for record in resistance],
            [record["friction_resistance_N"] / 1e3 for record in resistance],
        ],
        labels=["crushing", "breaking", "submersion", "hull-ice friction"],
        colors=["tab:gray", "tab:olive", "tab:cyan", "tab:orange"],
        alpha=0.85,
    )
    axes.plot(
        temperatures_r,
        [record["total_resistance_N"] / 1e3 for record in resistance],
        "k-",
        lw=1.4,
        label="total",
    )
    axes.set_xlabel("ice temperature [degC]")
    axes.set_ylabel("level-ice resistance [kN]")
    axes.set_title(
        "Level-ice resistance with temperature-dependent hull friction "
        f"($\\Delta R$ = {comparison['delta_relative_to_warm'] * 100:.1f} % for -5 -> -20 degC)"
    )
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7, loc="center left")
    path = output / "fig3_resistance_vs_temperature.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    # figure 4: DEM friction decomposition
    figure, axes = plt.subplots(figsize=(6.4, 4.2))
    axes.plot(
        temperatures,
        [record["mu_adhesion_mean"] for record in dem],
        "o-",
        label="adhesion (tangential) contribution",
        color="tab:blue",
    )
    axes.plot(
        temperatures,
        [record["mu_ploughing_mean"] for record in dem],
        "s-",
        label="ploughing (normal projection) contribution",
        color="tab:red",
    )
    axes.plot(
        temperatures,
        [record["interface_mu"] for record in dem],
        "k--",
        lw=1.0,
        label="imposed ice/steel contact law $\\mu_c(T)$",
    )
    axes.set_xlabel("ice temperature [degC]")
    axes.set_ylabel("friction coefficient contribution [-]")
    axes.set_title("DEM friction decomposition at the ice/steel interface")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7)
    path = output / "fig4_dem_decomposition.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "hull_ice_friction"
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--md-repeats", type=int, default=2)
    parser.add_argument("--fast", action="store_true", help="short runs for smoke tests")
    parser.add_argument(
        "--reuse-dem",
        type=Path,
        default=None,
        help="reuse the DEM sweep of an existing summary.json (recompute only MD/resistance)",
    )
    arguments = parser.parse_args()
    summary = run(
        arguments.output,
        arguments.repeats,
        arguments.md_repeats,
        arguments.fast,
        arguments.reuse_dem,
    )
    print(
        json.dumps(
            {
                "criteria": summary["criteria"],
                "mu_fit": {
                    key: summary["mu_temperature_fit"][key]
                    for key in ("mu0", "alpha", "beta", "r_squared", "max_abs_residual")
                },
                "artefacts": summary["artefacts"],
                "timings": summary["timings"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()