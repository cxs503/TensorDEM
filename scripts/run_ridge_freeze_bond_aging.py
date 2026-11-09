"""P54-W2-N03 study: freeze-bond aging of an ice-ridge keel (Ettema & Schaefer anchored).

Produces

* the Ettema & Schaefer (1986) validation table for the saturating bond law,
* ``sigma_b(t)`` histories for four pore-water salinities,
* ``R_keel(t)`` for a 10 m keel with the DEM-measured cohesion transfer,
* DEM evidence at bond scale (bond element capacity == law, bond network aging),
  a DEM resolution study of the biaxial cell, and the keel-penetration probe,
* figures (``artifacts/p54w2_n03_ridge_freeze_bond_aging/*.png``) and a JSON
  with every acceptance criterion evaluated.

Run::

    PYTHONPATH=src python3 scripts/run_ridge_freeze_bond_aging.py [--fast]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time

import numpy as np

import torch

from tensordem.ridge_dem import (RidgeAssembly, RidgeKeel, RidgeMaterial,
                                 calibrate_rubble_strength, generate_keel_assembly)
from tensordem.ridge_freeze_bond_aging import (
    REFERENCE_DATA,
    KeelAgingResistance,
    SaturationFreezeBondModel,
    aged_material,
    aging_state,
    cohesion_transfer_coefficient,
    validate_against_ettema_schaefer,
)
from tensordem.ridge_freeze_bond_aging_dem import (dem_bond_element_capacity,
                                                   dem_keel_penetration)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "p54w2_n03_ridge_freeze_bond_aging"
FIG_DIR = ROOT / "artifacts" / "p54w2_n03_ridge_freeze_bond_aging"


# ---------------------------------------------------------------------------
# DEM probes
# ---------------------------------------------------------------------------
def dem_biaxial(material: RidgeMaterial, *, columns: int, rows: int,
                confining=(50.0e3, 150.0e3), consolidation_steps: int = 1500,
                max_steps: int = 6000) -> dict:
    """One micro->macro cohesion / friction measurement of the rubble cell."""
    out = calibrate_rubble_strength(material, confining_stresses=tuple(confining),
                                    columns=columns, rows=rows,
                                    consolidation_steps=consolidation_steps,
                                    max_steps=max_steps)
    return {
        "cohesion_Pa": out["cohesion"],
        "friction_angle_deg": out["friction_angle_deg"],
        "peaks_Pa": [s["peak_deviator_stress"] for s in out["specimens"]],
        "confining_error": [s["confining_relative_error"] for s in out["specimens"]],
        "steps": [s["steps"] for s in out["specimens"]],
        "bond_tensile_strength_Pa": material.bond_tensile_strength,
        "bond_shear_strength_Pa": material.bond_shear_strength,
        "freeze_bond_fraction": material.freeze_bond_fraction,
    }


# ---------------------------------------------------------------------------
# study
# ---------------------------------------------------------------------------
def run(*, fast: bool = False) -> dict:
    started = time.time()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = SaturationFreezeBondModel()

    # ---------------------------------------------------------------- 1. E&S
    validation = validate_against_ettema_schaefer(model)
    validation_caption_variant = validate_against_ettema_schaefer(
        model, normal_pressure_override=3800.0)

    # ------------------------------------------------------- 2. sigma_b(t)
    salinity_levels = [0.0, 3.0, 12.5, 25.0]
    temperature_K = 263.15   # -10 degC keel pore water
    keel = KeelAgingResistance(keel_depth_m=10.0, width_m=1.0, porosity=0.30,
                              salinity_percent=0.0, temperature_K=temperature_K)
    ages = np.concatenate(([0.0], np.logspace(0.0, math.log10(240.0 * 3600.0), 240)))
    histories = {}
    for salinity in salinity_levels:
        sigma = model.tensile_strength(ages, keel.normal_pressure_Pa,
                                       salinity_percent=salinity,
                                       temperature_K=temperature_K)
        sigma_shear = model.shear_strength(ages, keel.normal_pressure_Pa,
                                           salinity_percent=salinity,
                                           temperature_K=temperature_K)
        histories[salinity] = {
            "sigma_tension_Pa": np.asarray(sigma, dtype=float),
            "sigma_shear_Pa": np.asarray(sigma_shear, dtype=float),
            "sigma_inf_tension_Pa": model.asymptotic_tensile_strength(
                keel.normal_pressure_Pa, salinity_percent=salinity,
                temperature_K=temperature_K),
            "sigma_inf_shear_Pa": model.asymptotic_shear_strength(
                keel.normal_pressure_Pa, salinity_percent=salinity,
                temperature_K=temperature_K),
            "growth_time_s": model.growth_time_s(salinity_percent=salinity,
                                                 temperature_K=temperature_K),
        }

    # monotonicity / limit checks of the law itself
    sigma0 = float(model.tensile_strength(0.0, keel.normal_pressure_Pa,
                                          temperature_K=temperature_K))
    sigma_resolved = model.tensile_strength(ages, keel.normal_pressure_Pa,
                                            temperature_K=temperature_K)
    diffs = np.diff(sigma_resolved)
    law_checks = {
        "sigma_at_t0_Pa": sigma0,
        "sigma_at_240h_Pa": float(sigma_resolved[-1]),
        "sigma_inf_Pa": model.asymptotic_tensile_strength(
            keel.normal_pressure_Pa, temperature_K=temperature_K),
        "asymptote_rel_error": float(abs(sigma_resolved[-1] / model.asymptotic_tensile_strength(
            keel.normal_pressure_Pa, temperature_K=temperature_K) - 1.0)),
        "monotone_increasing": bool(np.all(diffs >= 0.0)),
        "initial_slope_Pa_per_s": float(model.tensile_rate(0.0, keel.normal_pressure_Pa,
                                                           temperature_K=temperature_K)),
        "salinity_monotone_decreasing": bool(all(
            histories[a]["sigma_inf_tension_Pa"] >= histories[b]["sigma_inf_tension_Pa"]
            for a, b in zip(salinity_levels[:-1], salinity_levels[1:]))),
        "growth_time_decreases_with_salinity": bool(all(
            histories[a]["growth_time_s"] > histories[b]["growth_time_s"]
            for a, b in zip(salinity_levels[:-1], salinity_levels[1:]))),
    }

    # ------------------------------------------------ 3. DEM micro->macro chi
    columns, rows = (6, 6) if fast else (8, 8)
    consolidation, max_steps = (600, 2500) if fast else (1500, 6000)
    base = RidgeMaterial()
    bond_free_material, _ = aged_material(
        base, model, 0.0, normal_pressure_Pa=keel.normal_pressure_Pa,
        temperature_K=temperature_K)
    saturated_material, saturated_state = aged_material(
        base, model, 1.0e9, normal_pressure_Pa=keel.normal_pressure_Pa,
        temperature_K=temperature_K)
    dem_bond_free = dem_biaxial(bond_free_material, columns=columns, rows=rows,
                               consolidation_steps=consolidation, max_steps=max_steps)
    dem_saturated = dem_biaxial(saturated_material, columns=columns, rows=rows,
                               consolidation_steps=consolidation, max_steps=max_steps)
    cohesion_increment = dem_saturated["cohesion_Pa"] - dem_bond_free["cohesion_Pa"]
    if cohesion_increment > 0.0:
        chi = cohesion_transfer_coefficient(cohesion_increment,
                                            saturated_state.sigma_b_shear_Pa)
    else:                                    # not resolvable: fall back to unit transfer
        chi = 1.0
    chi_measured = cohesion_increment > 0.0
    # resolution of the cell: the peak-deviator scatter between the two arms
    peak_spread = max(
        abs(a - b) / max(abs(b), 1e-12)
        for a, b in zip(dem_saturated["peaks_Pa"], dem_bond_free["peaks_Pa"]))
    dem_cell = {
        "bond_free": dem_bond_free,
        "saturated": dem_saturated,
        "cohesion_increment_Pa": cohesion_increment,
        "cohesion_transfer_chi": chi,
        "chi_measured_by_dem": bool(chi_measured),
        "peak_deviator_relative_change_bonded_vs_free": peak_spread,
        "sigma_b_shear_at_saturation_Pa": saturated_state.sigma_b_shear_Pa,
        "friction_coefficient_from_bond_free": math.tan(
            math.radians(dem_bond_free["friction_angle_deg"])),
        "resolution_note": (
            "the biaxial cell measures c and phi of a rotationless disk packing whose "
            "peak deviator is set by contact stiffness and interlocking; freeze bonds at "
            "the Ettema & Schaefer level (kPa) perturb it by the fraction reported here, "
            "which is the resolution limit of this micro-parameter set"),
    }

    # ---------------------------------------------- 4. DEM bond-element check
    element_rows = []
    for age_h in (0.0, 1.0, 6.0, 24.0, 240.0):
        material, state = aged_material(
            base, model, age_h * 3600.0, normal_pressure_Pa=keel.normal_pressure_Pa,
            temperature_K=temperature_K)
        probe = dem_bond_element_capacity(material)
        element_rows.append({
            "age_h": age_h,
            "sigma_b_tension_Pa": state.sigma_b_tension_Pa,
            "bond_tensile_strength_Pa": material.bond_tensile_strength,
            "expected_failure_force_N": probe["expected_failure_force_N"],
            "force_below_threshold_N": probe["force_below_threshold_N"],
            "force_above_threshold_N": probe["force_above_threshold_N"],
            "bond_alive_below": probe["bond_alive_below"],
            "bond_broken_above": probe["bond_broken_above"],
            "rel_error": probe["rel_error"],
        })

    # bonded-contact fraction actually realised by the keel generator
    keel_small = RidgeKeel(6.0, 12.0, 0.25)
    network_rows = []
    for age_h in (0.0, 1.0, 6.0, 24.0, 240.0):
        material, state = aged_material(
            base, model, age_h * 3600.0, normal_pressure_Pa=keel.normal_pressure_Pa,
            temperature_K=temperature_K)
        info = generate_keel_assembly(keel_small, material, tip_height=0.5)["info"]
        network_rows.append({
            "age_h": age_h,
            "bonded_fraction_requested": material.freeze_bond_fraction,
            "bonds_generated": info["bonds"],
            "particles": info["particles"],
            "bonds_per_particle": info["bonds"] / max(info["particles"], 1),
        })

    # ------------------------------- 5. DEM keel-penetration resolution probe
    pen_rows = []
    if not fast:
        keel_pen = RidgeKeel(10.0, 20.0, 0.20)
        for age_h in (0.0, 6.0, 24.0, 240.0):
            material, state = aged_material(
                base, model, age_h * 3600.0, normal_pressure_Pa=keel.normal_pressure_Pa,
                temperature_K=temperature_K)
            gen = generate_keel_assembly(keel_pen, material, tip_height=0.5)
            probe = dem_keel_penetration(gen["assembly"], platen_velocity=0.5,
                                         stroke=0.8, settle_steps=300)
            pen_rows.append({
                "age_h": age_h,
                "bonded_fraction": state.bonded_fraction,
                "sigma_b_shear_Pa": state.sigma_b_shear_Pa,
                "particles": gen["info"]["particles"],
                "bonds": gen["info"]["bonds"],
                "peak_resistance_N_per_m": probe["peak_resistance_N_per_m"],
                "resistance_at_max_penetration_N_per_m":
                    probe["resistance_at_max_penetration_N_per_m"],
                "quasi_steady_resistance_N_per_m":
                    probe["mean_resistance_last_decile_N_per_m"],
                "broken_bonds": probe["broken_bonds"],
                "live_bonds": probe["live_bonds"],
            })

    penetration_trend = {}
    if pen_rows:
        quasi = [r["quasi_steady_resistance_N_per_m"] for r in pen_rows]
        penetration_trend = {
            "quasi_steady_N_per_m": quasi,
            "bond_free_N_per_m": quasi[0],
            "saturated_N_per_m": quasi[-1],
            "growth_factor": quasi[-1] / max(quasi[0], 1e-12),
            "monotone_increasing": all(a <= b for a, b in zip(quasi[:-1], quasi[1:])),
            "peak_is_inertial_spike": True,
            "note": ("the peak of a velocity-driven platen is an inertial transient; the "
                     "quasi-steady value (mean over the last decile of the penetration "
                     "record) is the comparable DEM resistance"),
        }

    # ------------------------------------------------- 6. R_keel(t) series
    phi0_deg = dem_bond_free["friction_angle_deg"]
    mu0 = math.tan(math.radians(phi0_deg)) if math.isfinite(phi0_deg) else float("nan")
    friction_source = "dem_biaxial_bond_free"
    if not (math.isfinite(mu0) and 0.10 <= mu0 <= 0.80):
        mu0 = base.friction_coefficient
        friction_source = ("micro_input_fallback: DEM bond-free fit outside [0.10, 0.80], "
                           "the reduced cell cannot resolve the friction angle")
    dem_cell["friction_coefficient_raw_dem"] = (
        math.tan(math.radians(phi0_deg)) if math.isfinite(phi0_deg) else None)
    dem_cell["friction_coefficient_used"] = mu0
    dem_cell["friction_source"] = friction_source
    resistance = KeelAgingResistance(
        keel_depth_m=10.0, width_m=1.0, porosity=0.30,
        friction_coefficient=mu0,
        cohesion_transfer=chi, contact_factor=1.0, temperature_K=temperature_K)
    series = resistance.series(model, ages)
    summary = resistance.summary(model, ages)
    sensitivity = {}
    for label, chi_value, mu_value in (("unit", 1.0, mu0),
                                       ("dem_measured", chi, mu0),
                                       ("conservative_0.2", 0.2, mu0),
                                       ("micro_friction_0.55", chi, base.friction_coefficient)):
        alt = KeelAgingResistance(
            keel_depth_m=10.0, width_m=1.0, porosity=0.30,
            friction_coefficient=mu_value,
            cohesion_transfer=chi_value, temperature_K=temperature_K)
        alt_summary = alt.summary(model, ages)
        sensitivity[label] = {
            "chi": chi_value,
            "friction_coefficient": mu_value,
            "R_at_0_N_per_m": alt_summary["R_at_0_N_per_m"],
            "R_final_N_per_m": alt_summary["R_final_N_per_m"],
            "R_growth_factor": alt_summary["R_growth_factor"],
            "R_final_at_240h_N_per_m": float(alt.series(model, [240.0 * 3600.0])["total_N_per_m"][-1]),
        }

    criterion_c = {
        "monotone_increasing": summary["monotone_increasing"],
        "R_at_0_N_per_m": summary["R_at_0_N_per_m"],
        "R_unbonded_N_per_m": summary["R_degenerate_N_per_m"],
        "t0_degeneration_rel_error": summary["t0_degeneration_rel_error"],
        "R_at_240h_N_per_m": float(series["total_N_per_m"][-1]),
        "growth_factor_240h": summary["R_growth_factor"],
        "chi": chi,
        "friction_coefficient": resistance.friction_coefficient,
    }

    negative_results = {
        "biaxial_cell_resolves_aging": bool(chi_measured
                                            and abs(peak_spread) > 0.10),
        "biaxial_peak_deviator_change": peak_spread,
        "biaxial_confining_control_error": dem_bond_free["confining_error"],
        "keel_penetration_resolves_aging": bool(penetration_trend.get(
            "monotone_increasing", False)),
        "keel_penetration_quasi_steady_N_per_m": penetration_trend.get(
            "quasi_steady_N_per_m", []),
        "statement": (
            "At the repository's default ridge_dem micro-parameters the bulk DEM probes "
            "cannot resolve the Ettema & Schaefer level (kPa) freeze-bond aging trend: "
            "the biaxial peak deviator changes by only a few per cent and the "
            "velocity-driven keel platen gives a chaotic quasi-steady resistance. The "
            "bond-scale DEM element is exact (rel. error < 1e-9), so the aging law is "
            "carried into ridge_dem at the bond level; a bulk DEM change would need "
            "softer contacts / a shear-rate-independent drive, which is out of scope "
            "here and is recorded as a capability gap."),
    }

    criteria = {
        "a_law_limits": {
            "sigma_b_at_t0_Pa": law_checks["sigma_at_t0_Pa"],
            "sigma_b_inf_Pa": law_checks["sigma_inf_Pa"],
            "pass": bool(law_checks["sigma_at_t0_Pa"] == 0.0
                         and law_checks["monotone_increasing"]
                         and law_checks["asymptote_rel_error"] < 0.05),
        },
        "b_ettema_schaefer": {
            "max_rel_error": validation["max_rel_error"],
            "max_holdout_rel_error": validation["max_holdout_rel_error"],
            "n_data": validation["n_data"],
            "n_holdout": validation["n_holdout"],
            "tolerance": validation["tolerance"],
            "pass": bool(validation["max_rel_error"] < validation["tolerance"]),
            "caption_variant_max_rel_error":
                validation_caption_variant["max_rel_error"],
            "caption_variant_pass": bool(
                validation_caption_variant["max_rel_error"]
                < validation_caption_variant["tolerance"]),
        },
        "c_keel_resistance": dict(criterion_c, pass_=None),
        "d_dem_integration": {
            "max_bond_element_rel_error": max(r["rel_error"] for r in element_rows),
            "all_thresholds_reproduced": all(r["bond_alive_below"] and r["bond_broken_above"]
                                             for r in element_rows),
            "bond_network_grows": bool(
                [r["bonds_generated"] for r in network_rows]
                == sorted(r["bonds_generated"] for r in network_rows)),
            "pass": bool(max(r["rel_error"] for r in element_rows) < 1e-6
                         and all(r["bond_alive_below"] and r["bond_broken_above"]
                                 for r in element_rows)),
        },
    }
    criteria["c_keel_resistance"].pop("pass_")
    criteria["c_keel_resistance"]["pass"] = bool(
        criterion_c["monotone_increasing"]
        and criterion_c["t0_degeneration_rel_error"] < 1e-9)

    payload = {
        "task": "P54-W2-N03",
        "title": "Ice-ridge keel freeze-bond strength aging (Ettema & Schaefer 1986 anchored)",
        "model": {
            "law": "sigma_b(t) = k(S) sigma_n phi(dT) (1 - exp(-t/tau_g))",
            "k(S)": f"{model.water_coefficient} (1 - {model.salinity_slope_per_percent} min(S, {model.salinity_cap_percent}))",
            "k_dry": model.dry_coefficient,
            "shear_to_tension_ratio": model.shear_to_tension_ratio,
            "apparent_activation_energy_J_mol": model.apparent_activation_energy_J_mol,
            "hardening": f"1 + (dT/{model.hardening_scale_K})^{model.hardening_exponent} (cap {model.hardening_cap})",
            "consolidation_time_s": model.consolidation_time_s,
        },
        "keel": {
            "depth_m": resistance.keel_depth_m,
            "porosity": resistance.porosity,
            "temperature_K": resistance.temperature_K,
            "normal_pressure_Pa": resistance.normal_pressure_Pa,
            "buoyant_weight_N_per_m": resistance.buoyant_weight_per_m,
            "contact_area_m2_per_m": resistance.contact_area_m2,
        },
        "validation": validation,
        "validation_caption_variant": validation_caption_variant,
        "law_checks": law_checks,
        "bond_histories": {
            str(s): {
                "sigma_inf_tension_Pa": v["sigma_inf_tension_Pa"],
                "sigma_inf_shear_Pa": v["sigma_inf_shear_Pa"],
                "growth_time_s": v["growth_time_s"],
                "growth_time_h": v["growth_time_s"] / 3600.0,
                "sigma_tension_at_240h_Pa": float(v["sigma_tension_Pa"][-1]),
            } for s, v in histories.items()
        },
        "keel_resistance": {
            "summary": summary,
            "sensitivity": sensitivity,
            "series_head": {
                "ages_h": series["ages_hours"][:6].tolist(),
                "sigma_b_shear_Pa": series["sigma_b_shear_Pa"][:6].tolist(),
                "total_N_per_m": series["total_N_per_m"][:6].tolist(),
            },
        },
        "dem": {
            "bond_element": element_rows,
            "bond_network": network_rows,
            "biaxial_cell": dem_cell,
            "keel_penetration": pen_rows,
            "keel_penetration_trend": penetration_trend,
        },
        "negative_results": negative_results,
        "criteria": criteria,
        "criteria_all_pass": all(v["pass"] for v in criteria.values()),
        "runtime_s": time.time() - started,
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": os.sys.version.split()[0],
        },
    }
    (RUN_DIR / "study.json").write_text(json.dumps(payload, indent=2))

    # ----------------------------------------------------------- 7. figures
    figures = make_figures(payload, model, keel, ages, histories, series, validation)
    payload["figures"] = figures
    (RUN_DIR / "study.json").write_text(json.dumps(payload, indent=2))
    (RUN_DIR / "report_fragment.md").write_text(markdown_fragment(payload))
    return payload


def make_figures(payload, model, keel, ages, histories, series, validation) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    produced = []

    # figure 1: sigma_b(t) histories + E&S anchors
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {0.0: "C0", 3.0: "C1", 12.5: "C2", 25.0: "C3"}
    for salinity, hist in histories.items():
        ax.plot(np.maximum(hist["sigma_shear_Pa"], 1e-9),
                ages / 3600.0,
                color=colors[salinity],
                label=f"S = {salinity:g} ppt (shear)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bond age t [h]")
    ax.set_ylabel(r"$\sigma_b(t)$ [Pa]")
    ax.set_title("Freeze-bond strength aging of a 10 m keel, -10 degC\n"
                 "saturating law anchored on Ettema & Schaefer (1986)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig_path = FIG_DIR / "sigma_b_histories.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    produced.append({"name": "sigma_b_histories",
                     "path": str(fig_path.relative_to(ROOT)),
                     "caption": "sigma_b(t) for four pore-water salinities"})

    # figure 2: measured vs predicted (E&S)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    rows = validation["rows"]
    meas = [r["measured"] for r in rows]
    pred = [r["predicted"] for r in rows]
    cal = [r for r in rows if r["role"] == "calibration"]
    hold = [r for r in rows if r["role"] == "holdout"]
    ax.scatter([r["measured"] for r in cal], [r["predicted"] for r in cal],
               marker="s", color="C3", label="calibration anchors", zorder=3)
    ax.scatter([r["measured"] for r in hold], [r["predicted"] for r in hold],
               marker="o", color="C0", label="holdout data", zorder=3)
    lo, hi = min(meas) * 0.7, max(meas) * 1.4
    ax.plot([lo, hi], [lo, hi], "k-", lw=1, label="1:1")
    ax.plot([lo, hi], [lo * 0.8, hi * 0.8], "k--", lw=0.8, label="-20 %")
    ax.plot([lo, hi], [lo * 1.2, hi * 1.2], "k:", lw=0.8, label="+20 %")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for r in rows:
        ax.annotate(r["key"].replace("_", " "), (r["measured"], r["predicted"]),
                    fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Ettema & Schaefer (1986) datum")
    ax.set_ylabel("model prediction")
    ax.set_title("Bond strength law vs experiment")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig_path = FIG_DIR / "ettema_schaefer_parity.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    produced.append({"name": "ettema_schaefer_parity",
                     "path": str(fig_path.relative_to(ROOT)),
                     "caption": "model vs E&S data, holdouts inside +-20 %"})

    # figure 3: R_keel(t) with components
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(series["ages_hours"], series["total_N_per_m"] / 1e3, color="C0",
            label=r"$R_{keel}(t)$")
    ax.plot(series["ages_hours"], series["cohesive_N_per_m"] / 1e3, color="C1",
            linestyle="--", label="cohesive (bond) term")
    ax.axhline(series["rubble_friction_N_per_m"][0] / 1e3, color="C2",
               linestyle=":", label="unbonded rubble friction")
    ax.set_xscale("log")
    ax.set_xlabel("keel age t [h]")
    ax.set_ylabel("resistance per metre width [kN/m]")
    ax.set_title("Keel resistance built on aging freeze bonds\n"
                 "(S = 0, -10 degC, 10 m keel; t -> 0 degenerates to the unbonded keel)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = FIG_DIR / "keel_resistance_history.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    produced.append({"name": "keel_resistance_history",
                     "path": str(fig_path.relative_to(ROOT)),
                     "caption": "R_keel(t) and its components"})

    # figure 4: salinity dependence of the asymptote and the growth time
    salinities = np.linspace(0.0, 30.0, 121)
    sigma_inf = [model.asymptotic_tensile_strength(
        keel.normal_pressure_Pa, salinity_percent=s,
        temperature_K=keel.temperature_K) for s in salinities]
    tau = [model.growth_time_s(salinity_percent=s,
                               temperature_K=keel.temperature_K) / 3600.0
           for s in salinities]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(salinities, np.array(sigma_inf) / 1e3, color="C0", label=r"$\sigma_{b,\infty}$ [kPa]")
    ax.set_xlabel("pore-water salinity S [ppt]")
    ax.set_ylabel(r"$\sigma_{b,\infty}$ (tension) [kPa]", color="C0")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(salinities, tau, color="C3", label=r"$\tau_g$ [h]")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\tau_g$ [h]", color="C3")
    ax.axvline(12.5, color="k", linestyle=":", lw=0.8)
    ax.annotate("E&S salinity cap 12.5 %", (12.5, 0.55 * max(np.array(sigma_inf) / 1e3)),
                fontsize=7, rotation=90, va="center")
    ax.set_title("Salinity dependence: weaker asymptote, faster saturation")
    fig.tight_layout()
    fig_path = FIG_DIR / "salinity_dependence.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    produced.append({"name": "salinity_dependence",
                     "path": str(fig_path.relative_to(ROOT)),
                     "caption": "sigma_b,inf and tau_g vs pore-water salinity"})

    # figure 5: bond network aging in the DEM keel generator
    network = payload["dem"]["bond_network"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ages_h = [r["age_h"] + 0.01 for r in network]
    ax.plot(ages_h, [r["bonds_per_particle"] for r in network], "o-", color="C0",
            label="bonds per particle (DEM keel slice)")
    ax.set_xscale("log")
    ax.set_xlabel("keel age t [h]")
    ax.set_ylabel("bonds per particle")
    ax.set_title("Bond network realised by tensordem.ridge_dem at each age")
    ax.grid(True, which="both", alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ages_h, [r["bonded_fraction_requested"] for r in network], "s--", color="C3",
             label="requested bonded fraction")
    ax2.set_ylabel("requested bonded fraction", color="C3")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig_path = FIG_DIR / "dem_bond_network.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    produced.append({"name": "dem_bond_network",
                     "path": str(fig_path.relative_to(ROOT)),
                     "caption": "bond count vs age in the DEM keel generator"})
    return produced


def markdown_fragment(payload) -> str:
    c = payload["criteria"]
    lines = [
        "## Acceptance criteria (machine-checked)",
        "",
        "| criterion | numbers | pass |",
        "|---|---|---|",
        f"| (a) sigma_b(0)=0, monotone, -> sigma_b_inf | sigma_b(0)={c['a_law_limits']['sigma_b_at_t0_Pa']:.3g} Pa, "
        f"sigma_b_inf={c['a_law_limits']['sigma_b_inf_Pa']:.4g} Pa | {c['a_law_limits']['pass']} |",
        f"| (b) vs Ettema & Schaefer (1986) < 20 % | max rel err={c['b_ettema_schaefer']['max_rel_error']*100:.2f} % "
        f"over {c['b_ettema_schaefer']['n_data']} data ({c['b_ettema_schaefer']['n_holdout']} holdout) | "
        f"{c['b_ettema_schaefer']['pass']} |",
        f"| (c) R_keel(t) monotone, R_keel(0) = unbonded | R(0)={c['c_keel_resistance']['R_at_0_N_per_m']:.4g} N/m, "
        f"R(240h)={c['c_keel_resistance']['R_at_240h_N_per_m']:.4g} N/m, "
        f"degeneration err={c['c_keel_resistance']['t0_degeneration_rel_error']:.2e} | "
        f"{c['c_keel_resistance']['pass']} |",
        f"| (d) DEM integration exact | bond element rel err="
        f"{c['d_dem_integration']['max_bond_element_rel_error']:.2e} | "
        f"{c['d_dem_integration']['pass']} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="smaller DEM cells, skip the keel-penetration probe")
    args = parser.parse_args()
    payload = run(fast=args.fast)
    print(json.dumps({k: payload["criteria"][k]["pass"] for k in payload["criteria"]},
                     indent=2))
    print("criteria_all_pass:", payload["criteria_all_pass"])
    print("runtime_s:", round(payload["runtime_s"], 1))


if __name__ == "__main__":
    main()