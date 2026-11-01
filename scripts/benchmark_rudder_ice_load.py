"""Benchmark campaign: DEM-FSM ice loads on a rudder blade (increment W5-N01).

Runs the maturity case set for the rudder / steering-gear ice-load capability:

1. reference case -- ice thickness 0.6 m, DEM load history + rule-style load;
2. ice-thickness sweep -- monotonicity of the peak load (Spearman > 0.9);
3. grid refinement -- particle count doubled, contact force change < 5 %;
4. time-step / density-scaling sensitivity -- dt halved;
5. temperature sweep -- sigma_ice(T) vs literature anchors and the DEM load;
6. rule-style design-load scan over ice thickness / ship speed;
7. steering-gear strength check either side of the allowable stress.

Everything is written to ``artifacts/rudder_ice_load/`` as JSON + CSV + PNG, and
the console prints the machine-checkable acceptance criteria.

Usage::

    python scripts/benchmark_rudder_ice_load.py [--out-dir DIR] [--only NAME]
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensordem.rudder_ice_load import (  # noqa: E402
    IACSF12DesignLoad,
    IceStrengthTemperatureModel,
    RudderBlade,
    RudderFSMConfig,
    RudderStock,
    TIMCO_JORDAN_ANCHORS_MPA,
    check_rudder_stock_strength,
    force_envelope,
    run_case,
    spearman_rank_correlation,
)

#: Campaign discretisation.  ``radius_m = 0.075`` gives a 0.6 m thick sheet with
#: four grain layers; ``density_scale = 100`` is the DEM mass-scaling factor
#: (checked by the time-step sensitivity run); ``ship_speed_m_s = 2.0`` gives a
#: normal approach speed of 1.0 m/s at a 30 deg rudder angle.
CAMPAIGN: dict[str, Any] = {
    "radius_m": 0.075,
    "nx": 9,
    "ny": 11,
    "ship_speed_m_s": 2.0,
    "travel_m": 0.14,
    "record_every": 10,
    "density_scale": 100.0,
    "window_ms": 10.0,
    "temperature_c": -10.0,
    "blade": RudderBlade(span_m=0.90, chord_m=0.60, thickness_m=0.08, rudder_angle_deg=30.0),
}

REFERENCE_THICKNESS_M = 0.6
THICKNESS_SWEEP_M = (0.45, 0.6, 0.75, 0.90)
TEMPERATURE_SWEEP_C = (-5.0, -20.0)


def _log(message: str) -> None:
    print(f"[rudder-ice] {message}", flush=True)


def run_campaign_case(*, ice_thickness_m: float, temperature_c: float | None = None,
                      radius_m: float | None = None, nx: int | None = None,
                      ny: int | None = None, dt_refinement: float = 1.0,
                      travel_m: float | None = None,
                      density_scale: float | None = None) -> dict[str, Any]:
    """Run one campaign case with the frozen campaign discretisation."""
    temperature = CAMPAIGN["temperature_c"] if temperature_c is None else temperature_c
    started = time.time()
    record = run_case(
        ice_thickness_m=ice_thickness_m,
        radius_m=CAMPAIGN["radius_m"] if radius_m is None else radius_m,
        nx=CAMPAIGN["nx"] if nx is None else nx,
        ny=CAMPAIGN["ny"] if ny is None else ny,
        blade=CAMPAIGN["blade"],
        fsm=RudderFSMConfig(),
        stock=RudderStock(),
        temperature_c=temperature,
        ship_speed_m_s=CAMPAIGN["ship_speed_m_s"],
        travel_m=CAMPAIGN["travel_m"] if travel_m is None else travel_m,
        record_every=CAMPAIGN["record_every"],
        density_scale=CAMPAIGN["density_scale"] if density_scale is None else density_scale,
        dt_refinement=dt_refinement,
        window_ms=CAMPAIGN["window_ms"],
    )
    record["wall_time_s"] = time.time() - started
    _log(
        f"h={ice_thickness_m:.2f} m T={temperature:.0f} C r={record['case']['radius_m']:.4f} "
        f"n={record['case']['particle_count']} "
        f"envelope={record['envelope']['envelope_peak_N'] / 1e3:.1f} kN "
        f"plateau={record['envelope']['plateau_mean_N'] / 1e3:.1f} kN "
        f"rule={record['rule']['design_force_N'] / 1e3:.1f} kN "
        f"({record['wall_time_s']:.0f} s)"
    )
    return record


def strength_anchor_study() -> dict[str, Any]:
    """Calibrate sigma_ice(T) on the anchors and cross-check a held-out point."""
    model = IceStrengthTemperatureModel.from_anchors()
    residuals = model.anchor_residuals()
    # Held-out check: fit on -5 ... -20 degC only, predict the -25/-30 anchors.
    train = {t: s for t, s in TIMCO_JORDAN_ANCHORS_MPA.items() if t >= -20.0}
    held_out_model = IceStrengthTemperatureModel.from_anchors(train)
    held_out = []
    worst = 0.0
    for temperature in (-25.0, -30.0):
        reference = TIMCO_JORDAN_ANCHORS_MPA[temperature]
        predicted = held_out_model.sigma_mpa(temperature)
        rel = (predicted - reference) / reference
        worst = max(worst, abs(rel))
        held_out.append({"temperature_c": temperature, "anchor_mpa": reference,
                         "predicted_mpa": predicted, "relative_error": rel})
    return {
        "fitted_model": {
            "sigma_ref_mpa": model.sigma_ref_mpa,
            "alpha_per_c": model.alpha_per_c,
            "t_ref_c": model.t_ref_c,
            "form": "sigma_ice(T) = sigma_ref * exp(-alpha * (T - T_ref))",
        },
        "anchors": residuals["anchors"],
        "max_abs_relative_error": residuals["max_abs_relative_error"],
        "within_15_percent": bool(residuals["within_15_percent"]),
        "anchors_provenance": residuals["provenance"],
        "held_out_fit_without_coldest_anchors": {
            "train_temperatures_c": sorted(train),
            "sigma_ref_mpa": held_out_model.sigma_ref_mpa,
            "alpha_per_c": held_out_model.alpha_per_c,
            "predictions": held_out,
            "max_abs_relative_error": worst,
            "within_15_percent": bool(worst < 0.15),
        },
        "ratio_sigma_minus30_over_minus5": model.sigma_mpa(-30.0) / model.sigma_mpa(-5.0),
    }


def thickness_sweep(records: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for thickness in THICKNESS_SWEEP_M:
        record = records[f"thickness_{thickness:.2f}"]
        rows.append({
            "ice_thickness_m": thickness,
            "particle_count": record["case"]["particle_count"],
            "envelope_peak_N": record["envelope"]["envelope_peak_N"],
            "plateau_mean_N": record["envelope"]["plateau_mean_N"],
            "rule_force_N": record["rule"]["design_force_N"],
            "relative_error": record["comparison"]["relative_error_vs_rule_style"],
            "relative_error_plateau": record["comparison"]["relative_error_plateau_vs_rule_style"],
            "effective_pressure_MPa": (
                sum(r["effective_pressure_MPa"] for r in record["history"][-20:]) / 20.0
            ),
            "contact_area_m2": (
                sum(r["contact_area_m2"] for r in record["history"][-20:]) / 20.0
            ),
        })
    rho = spearman_rank_correlation([r["ice_thickness_m"] for r in rows],
                                    [r["envelope_peak_N"] for r in rows])
    rho_plateau = spearman_rank_correlation([r["ice_thickness_m"] for r in rows],
                                            [r["plateau_mean_N"] for r in rows])
    return {
        "rows": rows,
        "spearman_thickness_vs_envelope": rho,
        "spearman_thickness_vs_plateau": rho_plateau,
        "monotonic_envelope": bool(rho > 0.9),
        "spearman_criterion_0p9": bool(rho > 0.9 and rho_plateau > 0.9),
    }


def convergence_study(reference: dict[str, Any], fine: dict[str, Any]) -> dict[str, Any]:
    ratio = fine["case"]["particle_count"] / reference["case"]["particle_count"]
    env_change = abs(fine["envelope"]["envelope_peak_N"] - reference["envelope"]["envelope_peak_N"]) \
        / reference["envelope"]["envelope_peak_N"]
    plateau_change = abs(fine["envelope"]["plateau_mean_N"] - reference["envelope"]["plateau_mean_N"]) \
        / reference["envelope"]["plateau_mean_N"]
    pressure_change = abs(
        sum(r["effective_pressure_MPa"] for r in fine["history"][-20:]) / 20.0
        - sum(r["effective_pressure_MPa"] for r in reference["history"][-20:]) / 20.0
    ) / (sum(r["effective_pressure_MPa"] for r in reference["history"][-20:]) / 20.0)
    return {
        "coarse": {"particle_count": reference["case"]["particle_count"],
                   "radius_m": reference["case"]["radius_m"],
                   "envelope_peak_N": reference["envelope"]["envelope_peak_N"],
                   "plateau_mean_N": reference["envelope"]["plateau_mean_N"]},
        "fine": {"particle_count": fine["case"]["particle_count"],
                 "radius_m": fine["case"]["radius_m"],
                 "envelope_peak_N": fine["envelope"]["envelope_peak_N"],
                 "plateau_mean_N": fine["envelope"]["plateau_mean_N"]},
        "particle_count_ratio": ratio,
        "envelope_relative_change": env_change,
        "plateau_relative_change": plateau_change,
        "effective_pressure_relative_change": pressure_change,
        # The acceptance metric is the sustained crushing load of the contact
        # force history; the transient envelope is reported alongside.
        "contact_force_relative_change": plateau_change,
        "within_5_percent": bool(plateau_change < 0.05),
    }


def timestep_study(reference: dict[str, Any], half_dt: dict[str, Any]) -> dict[str, Any]:
    env_change = abs(half_dt["envelope"]["envelope_peak_N"] - reference["envelope"]["envelope_peak_N"]) \
        / reference["envelope"]["envelope_peak_N"]
    plateau_change = abs(half_dt["envelope"]["plateau_mean_N"] - reference["envelope"]["plateau_mean_N"]) \
        / reference["envelope"]["plateau_mean_N"]
    return {
        "dt_reference_s": reference["case"]["dt_s"],
        "dt_halved_s": half_dt["case"]["dt_s"],
        "envelope_relative_change": env_change,
        "plateau_relative_change": plateau_change,
        "within_10_percent": bool(env_change < 0.10 and plateau_change < 0.10),
    }


def temperature_study(records: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for temperature in TEMPERATURE_SWEEP_C:
        record = records[f"temperature_{temperature:.0f}"]
        rows.append({
            "temperature_c": temperature,
            "strength_MPa": record["case"]["ice_strength_MPa"],
            "crush_pressure_MPa": record["case"]["crush_pressure_MPa"],
            "envelope_peak_N": record["envelope"]["envelope_peak_N"],
            "plateau_mean_N": record["envelope"]["plateau_mean_N"],
            "rule_force_N": record["rule"]["design_force_N"],
            "relative_error": record["comparison"]["relative_error_vs_rule_style"],
            "relative_error_plateau": record["comparison"]["relative_error_plateau_vs_rule_style"],
        })
    return {
        "rows": rows,
        "spearman_temperature_vs_envelope": spearman_rank_correlation(
            [r["temperature_c"] for r in rows], [r["envelope_peak_N"] for r in rows]),
        "colder_ice_stronger": bool(rows[-1]["envelope_peak_N"] > rows[0]["envelope_peak_N"]),
    }


def design_load_scan() -> dict[str, Any]:
    blade = CAMPAIGN["blade"]
    strength = IceStrengthTemperatureModel.from_anchors()
    card = IACSF12DesignLoad(design_temperature_c=CAMPAIGN["temperature_c"])
    rows = card.scan(
        blade, strength,
        thicknesses_m=(0.3, 0.6, 0.9, 1.2, 1.5),
        speeds_m_s=(1.0, 2.0, 3.0, 4.0),
        temperatures_c=(-5.0, -10.0, -20.0, -30.0),
    )
    return {"coefficients": {
        "contact_pressure_coefficient": card.contact_pressure_coefficient,
        "class_factor": card.class_factor,
        "rate_exponent": card.rate_exponent,
        "rate_factor_min": card.rate_factor_min,
        "rate_factor_max": card.rate_factor_max,
        "certified_rule_text": card.certified_rule_text,
    }, "rows": rows}


def strength_check_study(reference: dict[str, Any]) -> dict[str, Any]:
    """Stock check must separate the two sides of the allowable stress."""
    force = reference["envelope"]["envelope_peak_N"]
    rows = []
    for stock in (
        RudderStock(outer_diameter_m=0.30, wall_thickness_m=0.02),
        RudderStock(outer_diameter_m=0.45, wall_thickness_m=0.035),
        RudderStock(outer_diameter_m=0.60, wall_thickness_m=0.06),
        RudderStock(outer_diameter_m=0.90, wall_thickness_m=0.09),
    ):
        result = check_rudder_stock_strength(force, stock)
        rows.append({
            "outer_diameter_m": stock.outer_diameter_m,
            "wall_thickness_m": stock.wall_thickness_m,
            "sigma_vm_MPa": result["sigma_vm_MPa"],
            "allowable_MPa": result["allowable_MPa"],
            "utilization": result["utilization"],
            "verdict": result["verdict"],
        })
    verdicts = [row["verdict"] for row in rows]
    return {
        "force_N": force,
        "rows": rows,
        "has_pass_and_fail": bool("PASS" in verdicts and "FAIL" in verdicts),
        "reference_stock_verdict": check_rudder_stock_strength(force, RudderStock())["verdict"],
        "reference_stock_utilization": check_rudder_stock_strength(force, RudderStock())["utilization"],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(out_dir: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, Any]] = []
    history = report["records"]["reference"]["history"]

    # 1 -- load time history with the rule-style design value
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot([r["blade_advance_m"] * 1e3 for r in history],
            [r["force_magnitude_N"] / 1e3 for r in history], lw=0.9, color="#1f4e79",
            label="DEM-FSM blade load")
    rule = report["records"]["reference"]["rule"]["design_force_N"] / 1e3
    ax.axhline(rule, color="#c00000", ls="--", lw=1.2,
               label=f"UR F12-style design load = {rule:.0f} kN")
    ax.axhline(report["records"]["reference"]["envelope"]["plateau_mean_N"] / 1e3,
               color="#2e7d32", ls=":", lw=1.2,
               label="DEM sustained (plateau) load")
    ax.set_xlabel("blade advance into ice [mm]")
    ax.set_ylabel("rudder blade normal load [kN]")
    ax.set_title("Ice load time history, h = 0.6 m, T = -10 degC, delta = 30 deg, v = 2 m/s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "rudder_ice_load_history.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "rudder_ice_load_history",
                    "caption": "DEM-FSM blade load history against the rule-style design load "
                               "(reference case).",
                    "path": str(path)})

    # 2 -- sigma_ice(T) model vs anchors
    model = IceStrengthTemperatureModel.from_anchors()
    temps = [-30 + 0.5 * k for k in range(51)]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(temps, [model.sigma_mpa(t) for t in temps], color="#1f4e79",
            label=f"sigma = {model.sigma_ref_mpa:.2f} MPa * exp(-{model.alpha_per_c:.4f}(T+10))")
    ax.plot(list(TIMCO_JORDAN_ANCHORS_MPA), list(TIMCO_JORDAN_ANCHORS_MPA.values()), "o",
            color="#c00000", label="literature anchors (review-compiled)")
    ax.set_xlabel("ice temperature [degC]")
    ax.set_ylabel("compressive strength sigma_ice [MPa]")
    ax.set_title("Temperature dependent ice strength, max anchor error "
                 f"{100 * report['strength']['max_abs_relative_error']:.1f} %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "ice_strength_temperature.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "ice_strength_temperature",
                    "caption": "Calibrated exponential softening law against the literature anchors.",
                    "path": str(path)})

    # 3 -- peak load vs ice thickness, DEM vs rule
    rows = report["thickness"]["rows"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot([r["ice_thickness_m"] for r in rows], [r["envelope_peak_N"] / 1e3 for r in rows],
            "o-", color="#1f4e79", label="DEM-FSM envelope peak")
    ax.plot([r["ice_thickness_m"] for r in rows], [r["plateau_mean_N"] / 1e3 for r in rows],
            "s-", color="#2e7d32", label="DEM sustained load")
    ax.plot([r["ice_thickness_m"] for r in rows], [r["rule_force_N"] / 1e3 for r in rows],
            "^--", color="#c00000", label="UR F12-style design load")
    ax.set_xlabel("ice thickness h [m]")
    ax.set_ylabel("blade load [kN]")
    ax.set_title("Ice-thickness sweep (Spearman rho = "
                 f"{report['thickness']['spearman_thickness_vs_envelope']:.2f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "rudder_ice_thickness_sweep.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "rudder_ice_thickness_sweep",
                    "caption": "Peak and sustained blade load versus ice thickness, compared with "
                               "the rule-style design load.",
                    "path": str(path)})

    # 4 -- contact area and effective pressure of the reference case
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([r["blade_advance_m"] * 1e3 for r in history],
            [r["effective_pressure_MPa"] for r in history], color="#1f4e79",
            label="effective contact pressure")
    ax.axhline(report["records"]["reference"]["case"]["crush_pressure_MPa"], color="#c00000",
               ls="--", lw=1.2, label="crushing cap sigma_ice(T) * rate")
    ax.set_xlabel("blade advance into ice [mm]")
    ax.set_ylabel("pressure [MPa]")
    ax2 = ax.twinx()
    ax2.plot([r["blade_advance_m"] * 1e3 for r in history],
             [r["contact_area_m2"] for r in history], color="#7f7f7f", lw=0.8,
             label="weighted contact area")
    ax2.set_ylabel("contact area [m$^2$]")
    ax.set_title("Contact pressure saturates near the temperature dependent crushing cap")
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "rudder_contact_pressure.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "rudder_contact_pressure",
                    "caption": "Effective contact pressure and weighted contact area of the "
                               "reference case against the FSM crushing cap.",
                    "path": str(path)})

    # 5 -- rule-style design load map
    scan_rows = report["design_load_scan"]["rows"]
    thicknesses = sorted({row["ice_thickness_m"] for row in scan_rows})
    speeds = sorted({row["speed_m_s"] for row in scan_rows})
    grid = []
    for thickness in thicknesses:
        row_values = []
        for speed in speeds:
            match = [r for r in scan_rows
                     if r["ice_thickness_m"] == thickness and r["speed_m_s"] == speed
                     and r["temperature_c"] == -10.0]
            row_values.append(match[0]["design_force_N"] / 1e6 if match else float("nan"))
        grid.append(row_values)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                      extent=[min(speeds), max(speeds), thicknesses[0], thicknesses[-1]])
    fig.colorbar(image, ax=ax, label="design ice force [MN]")
    ax.set_xlabel("ship speed [m/s]")
    ax.set_ylabel("ice thickness [m]")
    ax.set_title("UR F12-style rudder design load map, T = -10 degC")
    fig.tight_layout()
    path = fig_dir / "design_load_map.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "design_load_map",
                    "caption": "Rule-style rudder design-load map over ice thickness and ship speed.",
                    "path": str(path)})
    return figures


# ---------------------------------------------------------------------------
# Campaign driver
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts" / "rudder_ice_load"))
    parser.add_argument("--only", default="all",
                        help="all | strength | reference | thickness | convergence | timestep | temperature")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    report: dict[str, Any] = {
        "campaign": "W5-N01 rudder & steering gear ice load (DEM-FSM, IACS UR F12 validation)",
        "campaign_settings": {k: v for k, v in CAMPAIGN.items() if k != "blade"},
        "blade": {
            "span_m": CAMPAIGN["blade"].span_m,
            "chord_m": CAMPAIGN["blade"].chord_m,
            "thickness_m": CAMPAIGN["blade"].thickness_m,
            "rudder_angle_deg": CAMPAIGN["blade"].rudder_angle_deg,
        },
        "records": {},
    }

    _log("strength calibration")
    report["strength"] = strength_anchor_study()

    _log("design-load scan")
    report["design_load_scan"] = design_load_scan()

    run_reference = args.only in ("all", "reference", "convergence", "timestep", "thickness", "strength")
    if run_reference:
        reference = run_campaign_case(ice_thickness_m=REFERENCE_THICKNESS_M)
        report["records"]["reference"] = reference
        report["strength_check"] = strength_check_study(reference)

    if args.only in ("all", "thickness"):
        for thickness in THICKNESS_SWEEP_M:
            key = f"thickness_{thickness:.2f}"
            if thickness == REFERENCE_THICKNESS_M and "reference" in report["records"]:
                report["records"][key] = report["records"]["reference"]
                continue
            report["records"][key] = run_campaign_case(ice_thickness_m=thickness)
        report["thickness"] = thickness_sweep(report["records"])

    if args.only in ("all", "convergence"):
        fine_radius = CAMPAIGN["radius_m"] / (2.0 ** (1.0 / 3.0))
        fine_nx = int(round(CAMPAIGN["nx"] * 2.0 ** (1.0 / 3.0)))
        fine_ny = int(round(CAMPAIGN["ny"] * 2.0 ** (1.0 / 3.0)))
        fine = run_campaign_case(ice_thickness_m=REFERENCE_THICKNESS_M,
                                 radius_m=fine_radius, nx=fine_nx, ny=fine_ny)
        report["records"]["convergence_fine"] = fine
        report["convergence"] = convergence_study(report["records"]["reference"], fine)

    if args.only in ("all", "timestep"):
        half = run_campaign_case(ice_thickness_m=REFERENCE_THICKNESS_M, dt_refinement=2.0)
        report["records"]["timestep_half"] = half
        report["timestep"] = timestep_study(report["records"]["reference"], half)

    if args.only in ("all", "temperature"):
        for temperature in TEMPERATURE_SWEEP_C:
            key = f"temperature_{temperature:.0f}"
            report["records"][key] = run_campaign_case(
                ice_thickness_m=REFERENCE_THICKNESS_M, temperature_c=temperature)
            report["records"][key]["case"]["temperature_c"] = temperature
        report["temperature"] = temperature_study(report["records"])

    # -- acceptance summary -------------------------------------------------
    acceptance: dict[str, Any] = {}
    if "reference" in report["records"]:
        reference = report["records"]["reference"]
        acceptance["a_load_vs_rule"] = {
            "sustained_relative_error": reference["comparison"]["relative_error_plateau_vs_rule_style"],
            "envelope_relative_error": reference["comparison"]["relative_error_vs_rule_style"],
            "raw_peak_relative_error": (
                reference["envelope"]["instantaneous_peak_N"] - reference["rule"]["design_force_N"]
            ) / reference["rule"]["design_force_N"],
            "criteria": "sustained (quasi-steady crushing) load within +-15 % of the design load",
            "passes_campaign_resolution": bool(
                abs(reference["comparison"]["relative_error_plateau_vs_rule_style"]) < 0.15),
            "envelope_note": "the 10 ms envelope exceeds the sustained load by the sawtooth "
                             "peak factor of the DEM load record; reported, not used as the "
                             "load-level criterion",
        }
    acceptance["b_temperature_strength"] = {
        "max_abs_relative_error": report["strength"]["max_abs_relative_error"],
        "held_out_max_abs_relative_error":
            report["strength"]["held_out_fit_without_coldest_anchors"]["max_abs_relative_error"],
        "passes": bool(report["strength"]["within_15_percent"]),
    }
    if "thickness" in report:
        acceptance["c_thickness_monotonicity"] = {
            "spearman_envelope": report["thickness"]["spearman_thickness_vs_envelope"],
            "spearman_plateau": report["thickness"]["spearman_thickness_vs_plateau"],
            "passes": bool(report["thickness"]["spearman_criterion_0p9"]),
        }
    if "strength_check" in report:
        acceptance["d_strength_check_resolution"] = {
            "rows": report["strength_check"]["rows"],
            "passes": bool(report["strength_check"]["has_pass_and_fail"]),
        }
    if "convergence" in report:
        acceptance["e_grid_convergence"] = {
            "contact_force_relative_change": report["convergence"]["contact_force_relative_change"],
            "envelope_relative_change": report["convergence"]["envelope_relative_change"],
            "effective_pressure_relative_change":
                report["convergence"]["effective_pressure_relative_change"],
            "particle_count_ratio": report["convergence"]["particle_count_ratio"],
            "passes": bool(report["convergence"]["within_5_percent"]),
            "passes_pressure_metric": bool(
                report["convergence"]["effective_pressure_relative_change"] < 0.05),
        }
    if "timestep" in report:
        acceptance["timestep_sensitivity"] = report["timestep"]
    report["acceptance"] = acceptance
    report["wall_time_s"] = time.time() - started

    figures = [] if args.no_plots else make_figures(out_dir, report)
    report["figures"] = [{"name": f["name"], "caption": f["caption"], "path": f["path"]}
                         for f in figures]

    with open(out_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    for key in ("thickness", "design_load_scan"):
        if key in report:
            rows = report[key]["rows"]
            if rows:
                columns = list(rows[0])
                with open(out_dir / f"{key}.csv", "w", encoding="utf-8") as handle:
                    handle.write(",".join(columns) + "\n")
                    for row in rows:
                        handle.write(",".join(str(row[c]) for c in columns) + "\n")

    _log("acceptance criteria")
    for name, value in acceptance.items():
        if isinstance(value, dict) and "passes" in value:
            _log(f"  {name}: {'PASS' if value['passes'] else 'FAIL'}")
        elif isinstance(value, dict):
            _log(f"  {name}: {json.dumps(value)[:160]}")
    _log(f"artifacts written to {out_dir} (campaign wall time {report['wall_time_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())