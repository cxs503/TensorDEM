#!/usr/bin/env python
"""Figures + machine-readable result summary for the ice-steel interface strength node.

Everything is computed here (no hand-typed numbers): the script re-runs the kernel
experiments, reads the cluster sweep aggregate, and writes

* ``artifacts/figures/*.png`` -- the report figures,
* ``artifacts/ice_contact_strength.json`` -- the judgement numbers with the paths of
  the artifacts they came from.

Usage
-----
    python scripts/run_ice_contact_strength_report.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensordem.ice_contact_strength import (  # noqa: E402
    ADHESION_LITERATURE_BAND_MPA,
    AdhesionModel,
    ConstantFriction,
    InterfaceStrength,
    MultiplicativeFrictionLaw,
    adhesion_band_check,
    contact_traction_capacity,
    cross_model_error,
    direct_shear_case,
    inclined_plane_case_onset,
    inclined_plane_onset_angle,
    interface_shear_strength,
    pull_off_case,
    slide_resistance,
)
from tensordem.ice_steel_friction import (  # noqa: E402
    CALIBRATION_ANCHORS,
    VALIDATION_ANCHORS,
    fit_ice_steel_friction,
)

SWEEP = ROOT / "artifacts" / "ice_contact_strength_sweep" / "aggregate.json"
FIG_DIR = ROOT / "artifacts" / "figures"
OUT = ROOT / "artifacts" / "ice_contact_strength.json"

TEMPERATURES = (-2.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0)
PRESSURES = tuple(np.logspace(math.log10(0.02), math.log10(5.0), 25))
SPEEDS = tuple(np.logspace(-4.0, 0.0, 25))


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def figure_friction(law: MultiplicativeFrictionLaw) -> pathlib.Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for temperature in (-2.0, -10.0, -20.0, -30.0):
        mu = [law.mu(0.1, temperature, p) for p in PRESSURES]
        axes[0].semilogx(PRESSURES, mu, marker="", label=f"T={temperature:g} °C")
    axes[0].set_xlabel("contact pressure p [MPa]")
    axes[0].set_ylabel("mu [-]")
    axes[0].set_title("pressure weakening  f(p)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", alpha=0.3)

    for temperature in (-2.0, -10.0, -20.0, -30.0):
        mu = [law.mu(v, temperature, 0.2) for v in SPEEDS]
        axes[1].semilogx(SPEEDS, mu, label=f"T={temperature:g} °C")
    axes[1].set_xlabel("sliding speed v [m/s]")
    axes[1].set_title("frictional-melting weakening  g(v)")
    axes[1].grid(True, which="both", alpha=0.3)

    temperatures = np.linspace(-30.0, -1.0, 200)
    axes[2].plot(temperatures, [law.mu(0.1, float(t), 0.2) for t in temperatures],
                 label="fitted law")
    cal = [(row.temperature_c, row.mu) for row in CALIBRATION_ANCHORS
           if abs(row.v_m_per_s - 0.01) < 1e-12 and abs(row.pressure_mpa - 0.2) < 1e-12]
    val = [(row.temperature_c, row.mu) for row in VALIDATION_ANCHORS]
    if cal:
        axes[2].plot([c[0] for c in cal], [c[1] for c in cal], "o", label="calibration rows")
    axes[2].plot([v[0] for v in val], [v[1] for v in val], "s", mfc="none",
                 label="holdout rows")
    axes[2].set_xlabel("temperature [°C]")
    axes[2].set_title("dryness  h(T) + anchors")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("multiplicative ice-steel friction law  mu = mu_0 f(p) g(v) h(T)", fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / "friction_law_factors.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_adhesion(adhesion: AdhesionModel) -> pathlib.Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    temperatures = np.linspace(-30.0, -0.1, 300)
    axes[0].plot(temperatures, [adhesion.tau_ad(float(t), 0.1) / 1e6 for t in temperatures],
                 label="tau_ad(T), p=0.1 MPa")
    low, high = ADHESION_LITERATURE_BAND_MPA
    axes[0].axhspan(low, high, color="tab:gray", alpha=0.18,
                    label=f"reported band {low}-{high} MPa")
    axes[0].set_xlabel("temperature [°C]")
    axes[0].set_ylabel("adfreeze shear strength [MPa]")
    axes[0].set_title("temperature dependence (band-consistent, not fitted)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    times = np.logspace(-1.0, 3.0, 200)
    for temperature in (-2.0, -10.0, -20.0):
        axes[1].semilogx(times, [adhesion.tau_ad(temperature, 0.1, float(t)) / 1e6
                                 for t in times], label=f"T={temperature:g} °C")
    axes[1].set_xlabel("contact time t_c [s]")
    axes[1].set_ylabel("tau_ad [MPa]")
    axes[1].set_title(f"freeze-bond development (t_sat={adhesion.t_sat_s:g} s)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "adhesion_law.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_interface(strength: InterfaceStrength) -> pathlib.Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    temperature = -10.0
    friction_term, adhesion_term, tau = [], [], []
    for p in PRESSURES:
        value = interface_shear_strength(strength, pressure_mpa=float(p),
                                         temperature_c=temperature, v_m_per_s=0.1,
                                         contact_time_s=60.0)
        friction_term.append(value["friction_term_pa"] / 1e6)
        adhesion_term.append(value["adhesion_term_pa"] / 1e6)
        tau.append(value["tau_shear_pa"] / 1e6)
    axes[0].semilogx(PRESSURES, friction_term, label="mu p  (friction)")
    axes[0].semilogx(PRESSURES, adhesion_term, label="tau_ad  (adhesion)")
    axes[0].semilogx(PRESSURES, tau, "k--", label="tau_shear = min(sum, tau_coh)")
    axes[0].set_xlabel("contact pressure p [MPa]")
    axes[0].set_ylabel("traction [MPa]")
    axes[0].set_title(f"interface shear strength at T={temperature:g} °C")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", alpha=0.3)

    grid_t = np.linspace(-30.0, -1.0, 120)
    grid_p = np.logspace(math.log10(0.02), math.log10(5.0), 120)
    share = np.zeros((grid_p.size, grid_t.size))
    for j, temperature in enumerate(grid_t):
        for i, p in enumerate(grid_p):
            value = interface_shear_strength(strength, pressure_mpa=float(p),
                                             temperature_c=float(temperature),
                                             v_m_per_s=0.1, contact_time_s=60.0)
            share[i, j] = value["adhesion_share"]
    image = axes[1].pcolormesh(grid_t, grid_p, share, shading="auto", cmap="viridis")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("temperature [°C]")
    axes[1].set_ylabel("contact pressure [MPa]")
    axes[1].set_title("adhesion share of the interface strength")
    fig.colorbar(image, ax=axes[1], label="tau_ad / tau_shear")
    fig.tight_layout()
    path = FIG_DIR / "interface_strength_map.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def kernel_experiments() -> dict:
    const = InterfaceStrength(friction=ConstantFriction(0.10))
    inclined = []
    for temperature in (-5.0, -10.0, -20.0):
        result = inclined_plane_case_onset(mass_kg=86.0, area_m2=1.0e-4,
                                           temperature_c=temperature, strength=const,
                                           contact_time_s=60.0)
        inclined.append({"temperature_c": temperature, **{k: result[k] for k in
                        ("theta_measured_deg", "theta_reference_deg", "rel_error",
                         "passes_gate")},
                         "mu_law": result["reference"]["mu_law"],
                         "adhesion_force_n": result["reference"]["adhesion_force_n"]})
    shear = []
    for normal in (1000.0, 5000.0, 20000.0):
        result = direct_shear_case(area_m2=0.04, normal_force_n=normal,
                                   temperature_c=-10.0, strength=const,
                                   contact_time_s=60.0)
        shear.append({"normal_force_n": normal,
                      "peak_force_n": result["peak_force_n"],
                      "residual_force_n": result["residual_force_n"],
                      "predicted_peak_force_n": result["predicted_peak_force_n"],
                      "predicted_residual_force_n": result["predicted_residual_force_n"],
                      "peak_rel_error": result["peak_rel_error"],
                      "residual_rel_error": result["residual_rel_error"],
                      "peak_over_residual": result["peak_over_residual"]})
    pull = {}
    for temperature in (-5.0, -10.0, -20.0):
        pull[f"{temperature:g}"] = pull_off_case(area_m2=0.01, temperature_c=temperature,
                                                strength=const, contact_time_s=60.0)
    pull_time = {f"{t:g}": pull_off_case(area_m2=0.01, temperature_c=-10.0,
                                        strength=const, contact_time_s=t)
                 for t in (1.0, 10.0, 60.0)}
    return {"inclined_plane": inclined, "direct_shear": shear,
            "pull_off_temperature": pull, "pull_off_contact_time": pull_time}


def figure_kernel(experiments: dict) -> pathlib.Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    inclined = experiments["inclined_plane"]
    measured = [row["theta_measured_deg"] for row in inclined]
    reference = [row["theta_reference_deg"] for row in inclined]
    axes[0].plot(reference, measured, "o", label="DEM onset")
    limits = [min(reference + measured) * 0.95, max(reference + measured) * 1.05]
    axes[0].plot(limits, limits, "k--", lw=0.8, label="1:1")
    axes[0].fill_between(limits, [v * 0.97 for v in limits], [v * 1.03 for v in limits],
                         color="tab:green", alpha=0.15, label="±3 % gate")
    for row in inclined:
        axes[0].annotate(f"{row['temperature_c']:g} °C",
                         (row["theta_reference_deg"], row["theta_measured_deg"]),
                         textcoords="offset points", xytext=(4, -8), fontsize=7)
    axes[0].set_xlabel("closed-form onset angle [deg]")
    axes[0].set_ylabel("DEM onset angle [deg]")
    axes[0].set_title("inclined plane: stick-slip onset")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    shear = experiments["direct_shear"]
    normals = [row["normal_force_n"] for row in shear]
    axes[1].loglog(normals, [row["peak_force_n"] for row in shear], "o-",
                   label="measured peak")
    axes[1].loglog(normals, [row["predicted_peak_force_n"] for row in shear], "x--",
                   label="min(mu p + tau_ad, tau_coh) A")
    axes[1].loglog(normals, [row["residual_force_n"] for row in shear], "s-",
                   label="measured residual")
    axes[1].loglog(normals, [row["predicted_residual_force_n"] for row in shear], "d--",
                   label="mu p A")
    axes[1].set_xlabel("normal load N [N]")
    axes[1].set_ylabel("shear force [N]")
    axes[1].set_title("direct shear: peak vs residual")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.3)

    times = sorted(float(k) for k in experiments["pull_off_contact_time"])
    values = [experiments["pull_off_contact_time"][f"{t:g}"]["peak_traction_pa"] / 1e6
              for t in times]
    predicted = [experiments["pull_off_contact_time"][f"{t:g}"]["predicted_traction_pa"] / 1e6
                 for t in times]
    axes[2].semilogx(times, values, "o", label="measured pull-off peak")
    axes[2].semilogx(times, predicted, "x--", label="sigma_ad(T, p, t_c)")
    axes[2].set_xlabel("contact time [s]")
    axes[2].set_ylabel("tensile strength [MPa]")
    axes[2].set_title("mode-I pull-off vs contact time")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "kernel_verification.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_resistance(strength: InterfaceStrength) -> tuple[pathlib.Path, dict]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    rows = []
    for temperature in (-2.0, -10.0, -20.0, -30.0):
        speeds = [1.0e-3, 1.0e-2, 0.1, 0.3, 1.0]
        totals = [slide_resistance(v, temperature, 0.2, strength,
                                   contact_time_s=60.0, sticking_fraction=0.05)
                  for v in speeds]
        axes[0].semilogx(speeds, [row["total_resistance_n"] / 1e3 for row in totals],
                         label=f"T={temperature:g} °C")
        for row in totals:
            rows.append(row)
    axes[0].set_xlabel("ship speed [m/s]")
    axes[0].set_ylabel("reference ice resistance [kN]")
    axes[0].set_title("resistance with mu(p, v, T) + 5 % sticking area")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", alpha=0.3)

    mu_scales = np.linspace(0.5, 1.5, 41)
    base = slide_resistance(0.1, -10.0, 0.2, strength, contact_time_s=60.0,
                            sticking_fraction=0.0)
    crushing = base["crushing_resistance_n"]
    normal = base["normal_load_n"]
    mu0 = base["mu"]
    axes[1].plot(mu_scales, [(crushing + mu0 * s * normal) / 1e3 for s in mu_scales],
                 label="friction-only sensitivity")
    full = [slide_resistance(0.1, -10.0, 0.2, strength, contact_time_s=60.0,
                             sticking_fraction=f)["total_resistance_n"] / 1e3
            for f in np.linspace(0.0, 0.2, 9)]
    axes[1].plot(np.linspace(0.5, 1.5, 9), full, "s--",
                 label="interface law, sticking fraction 0 -> 0.2")
    axes[1].set_xlabel("mu multiplier / sticking-fraction axis")
    axes[1].set_ylabel("reference resistance [kN]")
    axes[1].set_title("sensitivity channels of the interface law")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "resistance_impact.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, {"rows": rows, "mu0": mu0, "normal_load_n": normal,
                  "crushing_resistance_n": crushing}


def inject_markdown(payload: dict) -> pathlib.Path:
    """Write the results tables between the docs markers (no hand-typed numbers)."""
    doc = ROOT / "docs" / "ICE_CONTACT_STRENGTH_LAW.md"
    if not doc.exists():
        return doc
    law = payload["friction_law"]
    lines: list[str] = []
    lines.append("### 3.1 Friction law `mu = mu_0 f(p) g(v) h(T)` (weighted LSQ on the "
                 "compiled anchors)")
    lines.append("")
    lines.append("| `mu_0` (at 0.01 m/s, -10 degC, 0.2 MPa) | `n_p` per decade of p | "
                 "`n_v` per decade of v | `lam` per degC |")
    lines.append("|---|---|---|---|")
    lines.append(f"| {law['mu_ref']:.5f} | {law['n_p']:.5f} | {law['n_v']:.5f} | "
                 f"{law['lam_per_c']:.5f} |")
    lines.append("")
    cross = payload["cross_model"]
    lines.append(f"Form sensitivity vs the additive law on the same anchors: **max "
                 f"{100 * cross['max_rel_diff']:.1f} %**, RMS "
                 f"{100 * cross['rms_rel_diff']:.1f} %; holdout max error "
                 f"**{100 * payload['holdout_max_rel_error']:.1f} %** (gate 15 %).")
    lines.append("")
    lines.append("### 3.2 Adhesion `tau_ad(T)` vs the reported band (MPa, p = 0.1 MPa)")
    lines.append("")
    lines.append("| T [degC] | " + " | ".join(f"{row['temperature_c']:g}"
                                              for row in payload["adhesion_band_check"]["rows"]) + " |")
    lines.append("|---" * (len(payload["adhesion_band_check"]["rows"]) + 1) + "|")
    lines.append("| tau_ad | " + " | ".join(f"{row['tau_ad_mpa']:.3f}"
                                             for row in payload["adhesion_band_check"]["rows"]) + " |")
    lines.append("")
    lines.append(f"Band {payload['adhesion_band_check']['band_mpa']} MPa, all rows inside: "
                 f"**{payload['adhesion_band_check']['all_inside_band']}**, monotone in "
                 f"cooling: **{payload['adhesion_band_check']['monotone_in_cooling']}**, "
                 f"melting limit `tau_ad(0) = "
                 f"{payload['adhesion_band_check']['melting_limit_mpa']:.3f} MPa`.")
    lines.append("")
    lines.append("### 3.3 DEM kernel vs closed form (constant `mu = 0.10`, fully "
                 "developed bond)")
    lines.append("")
    lines.append("| experiment | condition | measured | closed form | rel. error | gate |")
    lines.append("|---|---|---|---|---|---|")
    for row in payload["kernel_experiments"]["inclined_plane"]:
        lines.append(f"| inclined plane | T = {row['temperature_c']:g} degC | "
                     f"{row['theta_measured_deg']:.4f} deg | {row['theta_reference_deg']:.4f} deg | "
                     f"{row['rel_error']:.2e} | 3 % |")
    for row in payload["kernel_experiments"]["direct_shear"]:
        lines.append(f"| direct shear peak | N = {row['normal_force_n']:g} N | "
                     f"{row['peak_force_n']:.1f} N | {row['predicted_peak_force_n']:.1f} N | "
                     f"{row['peak_rel_error']:.2e} | 3 % |")
        lines.append(f"| direct shear residual | N = {row['normal_force_n']:g} N | "
                     f"{row['residual_force_n']:.1f} N | {row['predicted_residual_force_n']:.1f} N | "
                     f"{row['residual_rel_error']:.2e} | 5 % |")
    for key, row in payload["kernel_experiments"]["pull_off_temperature"].items():
        lines.append(f"| pull-off (mode I) | T = {key} degC | "
                     f"{row['peak_traction_pa'] / 1e6:.4f} MPa | "
                     f"{row['predicted_traction_pa'] / 1e6:.4f} MPa | {row['rel_error']:.2e} | 3 % |")
    for key, row in payload["kernel_experiments"]["pull_off_contact_time"].items():
        lines.append(f"| pull-off (mode I) | t_c = {key} s | "
                     f"{row['peak_traction_pa'] / 1e6:.4f} MPa | "
                     f"{row['predicted_traction_pa'] / 1e6:.4f} MPa | {row['rel_error']:.2e} | 3 % |")
    lines.append("")
    shear = payload["kernel_experiments"]["direct_shear"]
    lines.append("Peak/residual ratio of the interface (a constant-`mu` model has no "
                 "second state): " + ", ".join(f"N={row['normal_force_n']:g} N -> "
                                               f"{row['peak_over_residual']:.1f}x"
                                               for row in shear) + ".")
    lines.append("")
    cap = payload["cohesive_cap"]
    lines.append(f"Cohesive cap `tau_coh(-10 degC) = {cap['tau_coh_pa_at_minus10'] / 1e6:.2f} MPa`; "
                 f"the `min()` binds only above **{cap['cap_binding_pressure_mpa_at_minus10']:.0f} MPa** "
                 f"of contact pressure (inactive in the 0.02-5 MPa operational range).")
    sweep = payload.get("sweep_summary")
    if sweep:
        lines.append("")
        lines.append("### 3.4 Cluster sweep (T x p x v x t_c)")
        lines.append("")
        lines.append(f"{sweep['n_rows']} rows. Friction term and adhesion term cross at:")
        lines.append("")
        lines.append("| T [degC] | " + " | ".join(f"{row['temperature_c']:g}"
                                                  for row in sweep["crossover_friction_equals_adhesion"]) + " |")
        lines.append("|---" * (len(sweep["crossover_friction_equals_adhesion"]) + 1) + "|")
        lines.append("| p_crossover [MPa] | " + " | ".join(
            f"{row['p_crossover_mpa']:.1f}"
            for row in sweep["crossover_friction_equals_adhesion"]) + " |")
        lines.append("")
        lines.append("Mean adhesion share per temperature (over p, v, t_c): " + ", ".join(
            f"{row['temperature_c']:g} degC: {row['adhesion_share_mean']:.2f}"
            for row in sweep["per_temperature"]) + ".")
    block = "<!-- RESULTS:BEGIN -->\n" + "\n".join(lines) + "\n<!-- RESULTS:END -->"
    text = doc.read_text()
    begin = text.index("<!-- RESULTS:BEGIN -->")
    end = text.index("<!-- RESULTS:END -->") + len("<!-- RESULTS:END -->")
    doc.write_text(text[:begin] + block + text[end:])
    return doc


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    law = MultiplicativeFrictionLaw.fit()
    additive = fit_ice_steel_friction()
    strength = InterfaceStrength.calibrated()
    adhesion = AdhesionModel()

    sweep = json.loads(SWEEP.read_text()) if SWEEP.exists() else None
    # the kernel experiments are the expensive part: run them once and reuse the dict
    # for both the figure and the judgement numbers
    experiments = kernel_experiments()
    figures = {
        "friction": figure_friction(law),
        "adhesion": figure_adhesion(adhesion),
        "interface": figure_interface(strength),
        "kernel": figure_kernel(experiments),
    }
    resistance_figure, resistance_rows = figure_resistance(strength)
    figures["resistance"] = resistance_figure

    band = adhesion_band_check()
    cross = cross_model_error(law, additive, CALIBRATION_ANCHORS)
    holdout = []
    for row in VALIDATION_ANCHORS:
        value = law.mu(row.v_m_per_s, row.temperature_c, row.pressure_mpa)
        holdout.append({"label": row.label, "mu_measured": row.mu, "mu_law": value,
                        "rel_error": abs(value - row.mu) / row.mu, "source": row.source})

    # cap-binding pressure at -10 degC (where min() switches to the ice-side cap)
    low, high = 0.1, 500.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        value = interface_shear_strength(strength, pressure_mpa=mid, temperature_c=-10.0,
                                         v_m_per_s=0.1, contact_time_s=60.0)
        if value["cap_binds"]:
            high = mid
        else:
            low = mid
    cap_pressure = 0.5 * (low + high)

    payload = {
        "node": "p223w3-n09-dem-ice-friction",
        "repo": "TensorDEM",
        "module": "src/tensordem/ice_contact_strength.py",
        "tests": "tests/test_ice_contact_strength.py",
        "sweep_script": "scripts/run_ice_contact_strength_sweep.py",
        "figure_script": "scripts/run_ice_contact_strength_report.py",
        "friction_law": law.as_dict(),
        "friction_law_additive_reference": additive.as_dict(),
        "cross_model": {"max_rel_diff": cross["max_rel_diff"],
                        "rms_rel_diff": cross["rms_rel_diff"]},
        "holdout_rows": holdout,
        "holdout_max_rel_error": max(row["rel_error"] for row in holdout),
        "adhesion_model": adhesion.as_dict(),
        "adhesion_band_check": band,
        "cohesive_cap": {"fraction_of_sigma_c": strength.cohesive_fraction,
                         "tau_coh_pa_at_minus10": strength.tau_coh_pa(-10.0),
                         "cap_binding_pressure_mpa_at_minus10": cap_pressure},
        "kernel_experiments": experiments,
        "resistance": {
            "mu0": resistance_rows["mu0"],
            "normal_load_n": resistance_rows["normal_load_n"],
            "rows": resistance_rows["rows"],
        },
        "sweep_aggregate": str(SWEEP.relative_to(ROOT)) if sweep else None,
        "sweep_summary": ({"n_rows": sweep["n_rows"],
                           "crossover_friction_equals_adhesion":
                               sweep["crossover_friction_equals_adhesion"],
                           "per_temperature": sweep["per_temperature"]}
                          if sweep else None),
        "figures": {key: str(value.relative_to(ROOT)) for key, value in figures.items()},
        "gates": {
            "implementation_gate": 0.03,
            "residual_gate": 0.05,
            "holdout_gate": 0.15,
        },
    }
    OUT.write_text(json.dumps(_jsonable(payload), indent=1))
    doc_path = inject_markdown(_jsonable(payload))
    print(json.dumps({"artifact": str(OUT), "doc": str(doc_path),
                      "figures": payload["figures"],
                      "holdout_max_rel_error": payload["holdout_max_rel_error"],
                      "cap_binding_pressure_mpa": cap_pressure,
                      "inclined_errors": [row["rel_error"] for row in
                                          experiments["inclined_plane"]],
                      "shear_peak_errors": [row["peak_rel_error"] for row in
                                            experiments["direct_shear"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())