"""Freeze-bond age evolution of an ice-ridge keel: full artifact runner.

Produces everything the campaign node (IN54W1-N05) claims, programmatically:

``artifacts/ridge_freeze_bond_age/model.json``
    freeze-bond tensile/shear strength at the campaign ages, the temperature /
    salinity maps, the literature-anchor band and the ``eta_b`` calibration;
``artifacts/ridge_freeze_bond_age/cooling.json``
    1-D transient cooling of the bond region (latent lens included), its time
    scales and the convergence / energy-ledger checks;
``artifacts/ridge_freeze_bond_age/age_sweep.json``
    DEM keel resistance at ``t = 0, 6, 24, 72, 168 h``, the agent factor
    ``k_age(t)`` and the per-seed force records;
``artifacts/ridge_freeze_bond_age/ledger.json``
    the machine-checkable acceptance ledger of the campaign node;
``artifacts/ridge_freeze_bond_age/figures/*.png``
    four figures, each with its SHA-256 recorded in the ledger.

No number in the campaign report is copied by hand: the report is rendered from
these artifacts.

Usage::

    python scripts/run_ridge_freeze_bond_age.py            # full deck (~6 min)
    python scripts/run_ridge_freeze_bond_age.py --quick    # smoke test (~20 s)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.ridge_dem import RidgeKeel, RidgeMaterial  # noqa: E402
from tensordem.ridge_freeze_bond_age import (  # noqa: E402
    LITERATURE_ANCHOR, FreezeBondAgeModel, aged_material,
    compare_growth_time_scales, conduction_time, freezing_point_celsius,
    ice_tensile_strength, keel_resistance, lens_cooling_transient,
    lindqvist_age_correction, run_keel_age_sweep)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "artifacts", "ridge_freeze_bond_age")
#: the smoke deck must never overwrite the committed full-deck artifacts
OUT_QUICK = os.path.join(ROOT, "artifacts", "ridge_freeze_bond_age_quick")

# --------------------------------------------------------------------- decks
AGES = (0.0, 6.0, 24.0, 72.0, 168.0)
#: 0.5 m is the particle radius of the DEM keel deck: the ice mass that feeds
#: one freeze bond and has to be cooled before the bond reaches full strength.
DEM_MATERIAL = RidgeMaterial(particle_radius=0.5, freeze_bond_fraction=0.65,
                             bond_tensile_strength=100.0e3,
                             bond_shear_strength=200.0e3)
DEM_KEEL = RidgeKeel(keel_depth=8.0, keel_width=12.0, porosity=0.25)
#: pore-water state of the deck, repeated here only for the growth-time
#: cross-check (kept in one place so the two numbers cannot drift apart)
DEM_TEMPERATURE_C = -10.0
DEM_SALINITY_PPT = 5.0


def compact_sweep(sweep: dict, limit: int = 600) -> dict:
    """Down-sample the recorded force histories before persisting them.

    The sweep keeps every DEM step of every seed (22 000 steps x 10 runs), which
    is ~12 MB of JSON for a curve that is plotted at 160 dpi.  The histories are
    therefore strided to at most ``limit`` samples, and the stride is recorded so
    the artifact stays honest about what was dropped.  Peak/residual/work and
    every scalar diagnostic are computed from the full history and are untouched.
    """
    for record in sweep["records"]:
        for entry in record.get("history", []):
            force = entry.get("force") or []
            displacement = entry.get("displacement") or []
            stride = max(1, math.ceil(len(force) / float(limit)))
            entry["force"] = force[::stride]
            entry["displacement"] = displacement[::stride]
            entry["stride"] = stride
            entry["samples_full"] = len(force)
            entry["downsampled"] = True
    return sweep


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(name: str, payload: dict) -> str:
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    return path


# ------------------------------------------------------------------ model
def build_model(thermal_length: float) -> FreezeBondAgeModel:
    return FreezeBondAgeModel.calibrated(thermal_length=thermal_length)


def model_block(model: FreezeBondAgeModel) -> dict:
    ages = list(AGES) + [0.5, 1.0, 3.0, 48.0, 336.0]
    curve = [{"age_hours": age, "sigma_b_pa": model.sigma_at(age),
              "tau_b_pa": model.tau_at(age),
              "bond_fraction": model.bond_fraction_at(age)} for age in ages]
    band = []
    for sigma_ref in LITERATURE_ANCHOR["band_pa"]:
        variant = FreezeBondAgeModel.calibrated(sigma_inf_target_pa=sigma_ref,
                                               thermal_length=model.thermal_length)
        band.append({
            "sigma_inf_target_pa": sigma_ref,
            "area_fraction": variant.area_fraction,
            "sigma_inf_pa": variant.sigma_inf_pa,
            "sigma_at_ages_pa": {str(age): variant.sigma_at(age) for age in AGES},
        })
    # temperature / salinity map of the asymptotic strength
    temperature_map = {}
    for temperature in (-2.5, -5.0, -10.0, -15.0, -20.0):
        temperature_map[str(temperature)] = {
            str(salinity): ice_tensile_strength(temperature, salinity)
            * model.area_fraction
            for salinity in (0.0, 5.0, 10.0, 20.0, 35.0)
        }
    return {
        "anchor": LITERATURE_ANCHOR,
        "model": model.as_dict(),
        "curve": curve,
        "band": band,
        "sigma_inf_temperature_salinity_map_pa": temperature_map,
        "anchor_error_relative": abs(model.sigma_inf_pa
                                     - float(LITERATURE_ANCHOR["sigma_b_inf_pa"]))
        / float(LITERATURE_ANCHOR["sigma_b_inf_pa"]),
        "fresh_water_gain_relative": (ice_tensile_strength(model.temperature_c, 0.0)
                                      / ice_tensile_strength(model.temperature_c,
                                                             model.salinity_ppt) - 1.0),
    }


def cooling_block(model: FreezeBondAgeModel) -> dict:
    solution = model.cooling_solution
    convergence = []
    for nodes in (21, 41, 81):
        run = lens_cooling_transient(model.thermal_length, ambient_c=model.temperature_c,
                                     initial_c=model.freezing_point_c,
                                     lens_thickness=model.lens_thickness,
                                     salinity_ppt=model.salinity_ppt, nodes=nodes)
        convergence.append({"nodes": nodes, "tau_half_hours": run["tau_half_hours"],
                            "t95_hours": run["t95_hours"],
                            "steps": run["steps"],
                            "energy_closure_relative_error":
                                run["energy_closure_relative_error"]})
    latent_free = lens_cooling_transient(
        model.thermal_length, ambient_c=model.temperature_c,
        initial_c=model.freezing_point_c, lens_thickness=0.0,
        salinity_ppt=model.salinity_ppt)
    scales = []
    for half in (0.25, 0.5, 1.0):
        run = lens_cooling_transient(half, ambient_c=model.temperature_c,
                                     initial_c=model.freezing_point_c,
                                     lens_thickness=model.lens_thickness,
                                     salinity_ppt=model.salinity_ppt)
        scales.append({"thermal_length_m": half,
                       "tau_half_hours": run["tau_half_hours"],
                       "conduction_time_hours": run["conduction_time_hours"],
                       "ratio": run["tau_half_hours"] / run["conduction_time_hours"]})
    exp_curve = [{"age_hours": age,
                  "model": (1.0 - math.exp(-age / model.tau_f_hours))
                  if model.tau_f_hours else 0.0}
                 for age in (0.0, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0,
                             120.0, 168.0, 240.0, 336.0)]
    numeric = [{"age_hours": float(t) / 3600.0, "strength_proxy":
                strength / max(solution["strength_proxy"][-1], 1e-12)}
               for t, strength in zip(solution["time_s"].tolist(),
                                      solution["strength_proxy"])
               if float(t) <= 336.0 * 3600.0][::max(
                   len(solution["time_s"]) // 400, 1)]
    return {
        "solution": {key: value for key, value in solution.items()
                     if key not in ("time_s", "bond_temperature_c",
                                    "frozen_fraction", "strength_proxy")},
        "temperature_curve": {
            "age_hours": [float(t) / 3600.0 for t in solution["time_s"].tolist()][::max(
                len(solution["time_s"]) // 400, 1)],
            "bond_temperature_c": [float(t) for t in solution["bond_temperature_c"].tolist()][::max(
                len(solution["time_s"]) // 400, 1)],
            "frozen_fraction": [float(t) for t in solution["frozen_fraction"].tolist()][::max(
                len(solution["time_s"]) // 400, 1)],
        },
        "numeric_strength_proxy": numeric,
        "exponential_model": exp_curve,
        "convergence": convergence,
        "latent_free": {"tau_half_hours": latent_free["tau_half_hours"],
                        "conduction_time_hours": latent_free["conduction_time_hours"],
                        "relative_deviation":
                            latent_free["tau_half_relative_deviation"],
                        "energy_closure_relative_error":
                            latent_free["energy_closure_relative_error"]},
        "thermal_length_sweep": scales,
        "conduction_time_hours": conduction_time(model.thermal_length) / 3600.0,
        "freezing_point_c": freezing_point_celsius(model.salinity_ppt),
    }


# ------------------------------------------------------------------ ledger
def acceptance_ledger(model_block: dict, cooling: dict, sweep: dict,
                      pytest_block: dict | None = None) -> dict:
    pytest_block = pytest_block or {"skipped": True}
    model = model_block["model"]
    curve = {record["age_hours"]: record for record in model_block["curve"]}
    ages_sorted = sorted(curve)
    sigmas = [curve[age]["sigma_b_pa"] for age in ages_sorted]
    monotone_sigma = all(b > a - 1e-9 for a, b in zip(sigmas, sigmas[1:]))
    asymptote = model["sigma_inf_pa"]
    limit_ok = (abs(sigmas[-1] - asymptote) / asymptote < 0.02
                if asymptote else False)
    zero_ok = curve[0.0]["sigma_b_pa"] == 0.0

    anchor_error = model_block["anchor_error_relative"]
    band = model_block["band"]
    band_covers = all(record["sigma_inf_pa"] >= min(LITERATURE_ANCHOR["band_pa"])
                      and record["sigma_inf_pa"] <= max(LITERATURE_ANCHOR["band_pa"])
                      for record in band)

    tau_f = model["tau_f_hours"]
    tau_cond = cooling["conduction_time_hours"]
    ratio = tau_f / tau_cond
    latent_free_dev = cooling["latent_free"]["relative_deviation"]
    convergence_spread = (max(record["tau_half_hours"] for record in cooling["convergence"])
                          - min(record["tau_half_hours"] for record in cooling["convergence"])) \
        / cooling["convergence"][1]["tau_half_hours"]

    records = sweep["records"]
    peaks = [record["peak_force_n_per_m"] for record in records]
    spreads = [record["peak_force_seed_spread_n_per_m"] for record in records]
    monotone_r = all(b > a for a, b in zip(peaks, peaks[1:]))
    monotone_excess = [peaks[index + 1] - peaks[index]
                       - 0.5 * (spreads[index + 1] + spreads[index])
                       for index in range(len(peaks) - 1)]
    monotone_r_beyond_spread = all(value > 0.0 for value in monotone_excess)
    unbonded_ok = (records[0]["sigma_b_pa"] == 0.0
                   and records[0]["bond_fraction"] == 0.0)
    mature_peaks = [record["peak_force_n_per_m"] for record in records
                    if record["age_hours"] >= 24.0]
    mature_monotone = all(b > a for a, b in zip(mature_peaks, mature_peaks[1:]))

    fresh_gain = model_block["fresh_water_gain_relative"]

    criteria = [
        {
            "id": "C1_sigma_limits_and_monotonicity",
            "statement": "sigma_b(t) -> 0 as t -> 0, -> sigma_inf as t -> inf, monotone increasing",
            "value": {"sigma_b(0)": curve[0.0]["sigma_b_pa"],
                      "sigma_b(168h)/sigma_inf": sigmas[-1] / asymptote,
                      "monotone": monotone_sigma},
            "pass": bool(zero_ok and monotone_sigma and limit_ok),
        },
        {
            "id": "C2_anchor_error",
            "statement": "sigma_inf vs Ettema & Schaefer (1986) T=-10C, S=5ppt within 20 %",
            "value": {"sigma_inf_pa": asymptote,
                      "anchor_pa": LITERATURE_ANCHOR["sigma_b_inf_pa"],
                      "relative_error": anchor_error,
                      "band_covered": band_covers,
                      "note": "eta_b is calibrated on the anchor, so the anchor "
                              "residual is zero by construction; the honest "
                              "content is the band (20-200 kPa) and the T/S trend"},
            "pass": bool(anchor_error < 0.20),
        },
        {
            "id": "C3_tau_f_vs_conduction",
            "statement": ("tau_f within a factor 3 of the closed-form conduction "
                          "time rho c h^2/(4k) (order-of-magnitude consistency), "
                          "and the 1-D solution is node-convergent within 5 %"),
            "value": {"tau_f_hours": tau_f, "conduction_time_hours": tau_cond,
                      "ratio": ratio,
                      "sensible_only_relative_deviation": latent_free_dev,
                      "convergence_spread_relative": convergence_spread},
            "pass": bool(1.0 / 3.0 < ratio < 3.0 and convergence_spread < 0.05),
            "caveat": ("the conduction time scale itself is reproduced to "
                       f"{100 * latent_free_dev:.1f} % by the sensible-heat "
                       "solution; the full model with the latent lens is "
                       f"{ratio:.2f}x the comparator, i.e. same order of "
                       "magnitude but outside the 50 % band. That excess is the "
                       "Stefan delay of the lens and is reported, not tuned "
                       "away (see docs/ridge_freeze_bond_age_evolution.md)."),
        },
        {
            "id": "C4_R_keel_monotone_and_unbonded_zero",
            "statement": ("campaign premise: R_keel(t) monotone increasing over "
                          "all ages, with R_keel(0) = the unbonded keel "
                          "(reported as FAIL if the young-age dip is real)"),
            "value": {"peak_force_n_per_m": peaks,
                      "peak_force_seed_spread_n_per_m": spreads,
                      "step_excess_over_spread_n_per_m": monotone_excess,
                      "age_factor": [record["age_factor"] for record in records],
                      "monotone": monotone_r,
                      "monotone_beyond_seed_spread": monotone_r_beyond_spread,
                      "t0_unbonded": unbonded_ok},
            "pass": bool(monotone_r and unbonded_ok),
        },
        {
            "id": "C4b_mature_aging_raises_resistance",
            "statement": ("above the young-age transition (t >= 24 h) keel "
                          "resistance grows monotonically with age and "
                          "k_age(168 h) > 1.4"),
            "value": {
                "mature_ages_hours": [record["age_hours"] for record in records
                                      if record["age_hours"] >= 24.0],
                "mature_peaks_n_per_m": [record["peak_force_n_per_m"]
                                         for record in records
                                         if record["age_hours"] >= 24.0],
                "k_age_168h": records[-1]["age_factor"],
            },
            "pass": bool(mature_monotone and records[-1]["age_factor"] > 1.4),
        },
        {
            "id": "C5_salinity_dependence",
            "statement": "sigma_inf increases as S -> 0 (fresh water)",
            "value": {"relative_gain_fresh_water": fresh_gain,
                      "map_pa": model_block["sigma_inf_temperature_salinity_map_pa"]},
            "pass": bool(fresh_gain > 0.0),
        },
        {
            "id": "C6_pytest_chain",
            "statement": "full chain under pytest (monotonicity, t=0 degeneration, S dependence)",
            "value": pytest_block,
            "pass": (None if pytest_block.get("returncode") is None
                     else pytest_block["returncode"] == 0),
        },
    ]
    return {
        "node": "IN54W1-N05",
        "mission_id": 7096,
        "repo": "TensorDEM",
        "criteria": criteria,
        "all_pass": all(item["pass"] is True for item in criteria),
        "all_pass_ignoring_unresolved": all(bool(item["pass"]) for item in criteria
                                           if item["pass"] is not None),
        "unresolved": [item["id"] for item in criteria if item["pass"] is None],
    }


# ------------------------------------------------------------------ figures
def make_figures(model_block: dict, cooling: dict, sweep: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = os.path.join(OUT, "figures")
    os.makedirs(folder, exist_ok=True)
    figures: list[dict] = []

    # --- Figure 1: sigma_b(t), tau_b(t) with the literature band -----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = [0.5 * step for step in range(1, 673)]
    axes[0].plot(grid, [model_block["model"]["sigma_inf_pa"] * 0.0 + 0.0 for _ in grid],
                 alpha=0.0)
    sigma = [record["sigma_b_pa"] / 1e3 for record in model_block["curve"]
             if record["age_hours"] in AGES]
    ages = [record["age_hours"] for record in model_block["curve"]
            if record["age_hours"] in AGES]
    tau = [record["tau_b_pa"] / 1e3 for record in model_block["curve"]
           if record["age_hours"] in AGES]
    model = model_block["model"]
    dense = [0.5 * step for step in range(1, 673)]
    axes[0].plot(dense, [1000.0 * model["sigma_inf_pa"] / 1e3
                         * (1.0 - math.exp(-age / model["tau_f_hours"]))
                         for age in dense], "-", color="tab:blue", lw=1.6,
                 label=r"$\sigma_b(t)=\sigma_{b\infty}(1-e^{-t/\tau_f})$")
    axes[0].plot(dense, [1000.0 * model["tau_inf_pa"] / 1e3
                         * (1.0 - math.exp(-age / model["tau_f_hours"]))
                         for age in dense], "--", color="tab:red", lw=1.2,
                 label=r"$\tau_b(t)=\kappa\,\sigma_b(t)$")
    axes[0].plot(ages, sigma, "o", color="tab:blue", label="campaign ages")
    axes[0].plot(ages, tau, "s", color="tab:red", alpha=0.6)
    axes[0].axhline(float(LITERATURE_ANCHOR["band_pa"][0]) / 1e3, color="0.5",
                    lw=0.8, ls=":")
    axes[0].axhline(float(LITERATURE_ANCHOR["band_pa"][1]) / 1e3, color="0.5",
                    lw=0.8, ls=":", label="literature band (20-200 kPa)")
    axes[0].set_xlabel("age t [h]")
    axes[0].set_ylabel("bond strength [kPa]")
    axes[0].set_title(f"freeze bond growth, T={model['temperature_c']} C, "
                      f"S={model['salinity_ppt']} ppt")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    # right panel: band sensitivity of eta_b
    for record in model_block["band"]:
        axes[1].plot(dense, [record["sigma_inf_pa"] / 1e3
                             * (1.0 - math.exp(-age / model["tau_f_hours"]))
                             for age in dense], lw=1.2,
                     label=(r"$\sigma_{b\infty}$=%.0f kPa ($\eta_b$=%.3f)"
                            % (record["sigma_inf_pa"] / 1e3, record["area_fraction"])))
    axes[1].set_xlabel("age t [h]")
    axes[1].set_ylabel(r"$\sigma_b(t)$ [kPa]")
    axes[1].set_title("calibration band of the frozen contact-area fraction")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(folder, "fig1_freeze_bond_strength.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "fig1_freeze_bond_strength.png", "path": path,
                    "caption": "Freeze-bond tensile and shear strength versus "
                               "age for the calibrated model (T=-10 C, S=5 ppt) "
                               "with the campaign ages marked, and the 20-200 kPa "
                               "calibration band."})

    # --- Figure 2: cooling solution vs the exponential model ---------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    temperature = cooling["temperature_curve"]
    axes[0].plot(temperature["age_hours"], temperature["bond_temperature_c"],
                 "-", color="tab:green", lw=1.5, label="bond-plane temperature (1-D)")
    axes[0].axhline(cooling["freezing_point_c"], color="0.5", ls=":", lw=0.8,
                    label=r"$T_m(S)$")
    axes[0].axvline(cooling["solution"]["lens_freeze_time_hours"], color="tab:orange",
                    ls="--", lw=1.0, label="lens frozen")
    axes[0].set_xlim(0, 120)
    axes[0].set_xlabel("age t [h]")
    axes[0].set_ylabel(r"$T_{bond}$ [$^\circ$C]")
    axes[0].set_title("conductive cooling of the ice mass feeding the bond")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    numeric = cooling["numeric_strength_proxy"]
    axes[1].plot([record["age_hours"] for record in numeric],
                 [record["strength_proxy"] for record in numeric],
                 "-", color="tab:purple", lw=1.6,
                 label="1-D model (frozen area x brine strength)")
    axes[1].plot([record["age_hours"] for record in cooling["exponential_model"]],
                 [record["model"] for record in cooling["exponential_model"]],
                 "--", color="tab:blue", lw=1.4, label="exponential model, "
                 r"$\tau_f$ from the 50 % time")
    axes[1].set_xlim(0, 168)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("age t [h]")
    axes[1].set_ylabel(r"$\sigma_b(t)/\sigma_{b\infty}$")
    axes[1].set_title(r"$\tau_f$=%.1f h vs $\rho c h^2/(4k)$=%.1f h"
                      % (cooling["solution"]["tau_half_hours"],
                         cooling["conduction_time_hours"]))
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(folder, "fig2_bond_cooling_timescale.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "fig2_bond_cooling_timescale.png", "path": path,
                    "caption": "Left: bond-plane temperature of the 1-D "
                               "conduction solution. Right: the physically "
                               "modelled strength growth against the prescribed "
                               "exponential form and the analytic conduction "
                               "time scale."})

    # --- Figure 3: R_keel(t) and k_age(t) ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    records = sweep["records"]
    ages = [record["age_hours"] for record in records]
    peaks = [record["peak_force_n_per_m"] / 1e3 for record in records]
    spreads = [record["peak_force_seed_spread_n_per_m"] / 1e3 for record in records]
    axes[0].errorbar(ages, peaks, yerr=spreads, fmt="o-", color="tab:blue",
                     capsize=3, label="peak press reaction (mean of seeds)")
    axes[0].plot(ages, [record["residual_force_n_per_m"] / 1e3 for record in records],
                 "s--", color="tab:gray", label="residual (post-failure)")
    axes[0].set_xlabel("age t [h]")
    axes[0].set_ylabel(r"$R_{keel}$ [kN/m]")
    axes[0].set_title("DEM keel resistance vs age (confined press)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].plot(ages, [record["age_factor"] for record in records], "o-",
                 color="tab:red", label=r"$k_{age}(t)=R(t)/R(0)$")
    sig = [record["sigma_b_pa"] / 1e3 for record in records]
    axes[1].set_xlabel("age t [h]")
    axes[1].set_ylabel(r"$k_{age}$ [-]")
    axes[1].set_title(r"age factor and $\sigma_b$ (right axis: kPa)")
    axes2 = axes[1].twinx()
    axes2.plot(ages, sig, ":", color="tab:purple", label=r"$\sigma_b(t)$ [kPa]")
    axes2.set_ylabel(r"$\sigma_b$ [kPa]")
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = axes2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, fontsize=7, loc="lower right")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(folder, "fig3_keel_resistance_age.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "fig3_keel_resistance_age.png", "path": path,
                    "caption": "Left: DEM keel resistance (peak and residual) at "
                               "the campaign ages. Right: the age factor "
                               "k_age(t) that multiplies the Lindqvist ridge "
                               "keel term, together with the bond strength."})

    # --- Figure 4: temperature / salinity map ------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    temperature_axis = [-2.5, -5.0, -10.0, -15.0, -20.0]
    salinities = (0.0, 5.0, 10.0, 20.0, 35.0)
    for salinity in salinities:
        ax.plot(temperature_axis,
                [model_block["sigma_inf_temperature_salinity_map_pa"][str(temperature)][str(salinity)] / 1e3
                 for temperature in temperature_axis],
                "o-", label=f"S={salinity} ppt")
    ax.set_xlabel(r"contact temperature $T_c$ [$^\circ$C]")
    ax.set_ylabel(r"$\sigma_{b\infty}$ [kPa]")
    ax.set_title(r"asymptotic freeze-bond strength ($\eta_b$=%.3f)"
                 % model_block["model"]["area_fraction"])
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(folder, "fig4_strength_map.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "fig4_strength_map.png", "path": path,
                    "caption": "Asymptotic freeze-bond strength over contact "
                               "temperature and pore-water salinity."})
    for figure in figures:
        figure["sha256"] = _sha256(figure["path"])
    return figures


def run_pytest_chain(skip: bool) -> dict:
    """Execute the module's pytest chain and return its machine verdict."""
    command = [sys.executable, "-m", "pytest",
               "tests/test_ridge_freeze_bond_age.py", "-q"]
    label = "python -m pytest tests/test_ridge_freeze_bond_age.py -q"
    if skip:
        return {"skipped": True, "command": label, "returncode": None}
    import subprocess
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    return {
        "skipped": False,
        "command": label,
        "returncode": completed.returncode,
        "summary_line": tail[-1] if tail else "",
        "test_file": "tests/test_ridge_freeze_bond_age.py",
    }


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="tiny keel deck and one seed (smoke test)")
    parser.add_argument("--seeds", type=int, default=2,
                        help="DEM realisations per age")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true",
                        help="do not run the pytest chain (C6 then stays unresolved)")
    parser.add_argument("--skip-mode-check", action="store_true",
                        help="skip the contact-mode sensitivity probe")
    parser.add_argument("--recompact", action="store_true",
                        help="re-write an existing age_sweep.json with "
                             "down-sampled histories and rebuild ledger/figures")
    parser.add_argument("--reuse-sweep", action="store_true",
                        help="reuse artifacts/ridge_freeze_bond_age/age_sweep.json "
                             "instead of re-running the DEM (ledger, figures and "
                             "the pytest chain are recomputed)")
    args = parser.parse_args(argv)
    global OUT
    if args.quick:
        OUT = OUT_QUICK
    os.makedirs(OUT, exist_ok=True)
    started = time.time()

    thermal_length = DEM_MATERIAL.particle_radius
    model = build_model(thermal_length)
    model_payload = model_block(model)
    model_payload["deck"] = {"thermal_length_m": thermal_length,
                             "ages_hours": list(AGES)}
    # cross-check against the declared Arrhenius growth time of the sibling
    # module tensordem.ridge_freeze_bond_aging (same law, prescribed tau_g)
    model_payload["growth_time_comparison"] = compare_growth_time_scales(
        temperature_c=DEM_TEMPERATURE_C, salinity_ppt=DEM_SALINITY_PPT,
        lens_thickness=FreezeBondAgeModel().lens_thickness)
    _write_json("model.json", model_payload)
    print(f"[model] sigma_inf={model.sigma_inf_pa/1e3:.2f} kPa "
          f"eta_b={model.area_fraction:.4f} tau_f={model.tau_f_hours:.2f} h",
          flush=True)

    cooling = cooling_block(model)
    _write_json("cooling.json", cooling)
    print(f"[cooling] tau_f={cooling['solution']['tau_half_hours']:.2f} h "
          f"rho c h^2/(4k)={cooling['conduction_time_hours']:.2f} h "
          f"sensible-only dev={100 * cooling['latent_free']['relative_deviation']:.1f} %",
          flush=True)

    seeds = tuple(20261031 + 37 * index for index in range(1 if args.quick else args.seeds))
    if args.recompact:
        with open(os.path.join(OUT, "age_sweep.json")) as handle:
            sweep = json.load(handle)
        for record in sweep["records"]:
            for entry in record.get("history", []):
                entry.pop("stride", None)
                entry.pop("samples_full", None)
        sweep = compact_sweep(sweep)
        _write_json("age_sweep.json", sweep)
        print("[dem] recompacted the persisted sweep (--recompact)", flush=True)
    elif args.reuse_sweep:
        with open(os.path.join(OUT, "age_sweep.json")) as handle:
            sweep = json.load(handle)
        print("[dem] reusing the persisted sweep (--reuse-sweep)", flush=True)
    elif args.quick:
        keel, material = RidgeKeel(4.0, 6.0, 0.25), RidgeMaterial(
            particle_radius=0.75, freeze_bond_fraction=0.65,
            bond_tensile_strength=100.0e3, bond_shear_strength=200.0e3)
        ages = (0.0, 168.0)
        sweep = run_keel_age_sweep(ages, keel=keel, material=material, model=model,
                                   seeds=seeds, press_velocity=0.1, stroke=0.03)
    else:
        sweep = run_keel_age_sweep(AGES, keel=DEM_KEEL, material=DEM_MATERIAL,
                                   model=model, seeds=seeds)
    for record in sweep["records"]:
        print(f"[dem] t={record['age_hours']:6.1f} h sigma={record['sigma_b_pa']/1e3:7.2f} kPa "
              f"R_peak={record['peak_force_n_per_m']/1e3:8.2f} kN/m "
              f"R_res={record['residual_force_n_per_m']/1e3:8.2f} kN/m "
              f"k_age={record['age_factor']:.3f}", flush=True)
    sweep["lindqvist_coupling"] = lindqvist_age_correction(sweep["records"])
    _write_json("age_sweep.json", compact_sweep(sweep))  # before the optional probe
    if not args.skip_mode_check and not (args.reuse_sweep
                                        and "contact_mode_sensitivity" in sweep):
        # contact-mode sensitivity: a frozen lens sits *on top of* the ice-ice
        # contact ("parallel"); the historical "exclusive" mode switches the
        # contact law off for a bonded pair.  Both are measured so the report
        # can state which convention produced the headline curve.
        mode_keel = RidgeKeel(4.0, 6.0, 0.25) if args.quick else DEM_KEEL
        mode_material = (RidgeMaterial(particle_radius=0.75,
                                       freeze_bond_fraction=0.65,
                                       bond_tensile_strength=100.0e3,
                                       bond_shear_strength=200.0e3)
                         if args.quick else DEM_MATERIAL)
        oldest = max(record["age_hours"] for record in sweep["records"])
        probes = {}
        for mode in ("parallel", "exclusive"):
            probe = keel_resistance(
                mode_keel, aged_material(mode_material, model, oldest),
                press_velocity=0.1 if args.quick else 0.02,
                stroke=0.03 if args.quick else 0.10,
                bond_contact_mode=mode)
            probes[mode] = {
                "peak_n_per_m": probe["peak_n_per_m"],
                "residual_n_per_m": probe["residual_n_per_m"],
                "broken_bonds": probe["broken_bonds"],
                "bonds": probe["bonds"],
            }
            print(f"[mode] {mode:9s} peak={probe['peak_n_per_m']/1e3:8.2f} kN/m "
                  f"res={probe['residual_n_per_m']/1e3:8.2f} kN/m", flush=True)
        sweep["contact_mode_sensitivity"] = {
            "age_hours": oldest,
            "probes": probes,
            "headline_mode": sweep["apparatus"]["bond_contact_mode"],
            "note": ("mode comparison at the oldest age; 'parallel' is the "
                     "physical convention (the frozen lens adds tensile/shear "
                     "capacity to the contact), 'exclusive' is the historical "
                     "ridge_dem convention kept for regression"),
        }
    _write_json("age_sweep.json", compact_sweep(sweep))

    pytest_block = run_pytest_chain(args.skip_pytest)
    ledger = acceptance_ledger(model_payload, cooling, sweep, pytest_block)
    figures = [] if args.no_figures else make_figures(model_payload, cooling, sweep)
    ledger["figures"] = [{"name": figure["name"], "sha256": figure["sha256"],
                          "path": os.path.relpath(figure["path"],
                                                  os.path.join(OUT, "..", ".."))}
                         for figure in figures]
    ledger["runtime_seconds"] = time.time() - started
    ledger["artifact_sha256"] = {
        name: _sha256(os.path.join(OUT, name))
        for name in ("model.json", "cooling.json", "age_sweep.json")}
    _write_json("ledger.json", ledger)
    print(f"[ledger] all_pass={ledger['all_pass']} "
          f"all_pass_ignoring_unresolved={ledger['all_pass_ignoring_unresolved']} "
          f"unresolved={ledger['unresolved']}", flush=True)
    return 0 if ledger["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())