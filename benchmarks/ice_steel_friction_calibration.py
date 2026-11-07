"""Calibrate, validate and audit the ice-steel friction law ``mu(v, T, p)``.

Run from the repository root::

    PYTHONPATH=src python3 benchmarks/ice_steel_friction_calibration.py

Outputs
-------
``artifacts/ice_steel_friction.json``
    Full machine-readable record: anchor set with provenance, both fitted laws,
    holdout / leave-one-source-out / k-fold validation tables, the deliberate
    extrapolation probes, the +-20 % sensitivity study, the constant-mu negative
    control and the DEM realisation probes.  Every block carries an ``evidence``
    field (REAL / MOCK / STUB / WITHHELD semantics from the platform audit).
``docs/ice-steel-friction/*.png``
    Six figures: coefficient surface slices with the compiled anchors, parity
    plot, sensitivity, negative control and DEM probe.
``runs/ice_steel_friction/summary.txt``
    One-screen digest of the numbers quoted in the report.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.ice_steel_friction import (          # noqa: E402
    CALIBRATION_ANCHORS,
    EXTRAPOLATION_PROBES,
    MU_CONSTANT_BASELINE,
    VALIDATION_ANCHORS,
    VALIDATION_GATE,
    IceResistanceCase,
    IceSteelFrictionModel,
    calibrated_envelope,
    dem_friction_probe,
    fit_ice_steel_friction,
    holdout_error_table,
    ice_resistance,
    kfold_validation,
    leave_one_source_out,
    negative_control_table,
    operating_envelope,
    sensitivity_table,
)

ARTIFACT = ROOT / "artifacts" / "ice_steel_friction.json"
FIGURE_DIR = ROOT / "docs" / "ice-steel-friction"
RUN_DIR = ROOT / "runs" / "ice_steel_friction"

DEM_PROBES = (
    dict(v=0.01, temperature_c=-10.0, pressure_mpa=0.5),
    dict(v=0.1, temperature_c=-10.0, pressure_mpa=0.5),
    dict(v=1.0, temperature_c=-20.0, pressure_mpa=1.0),
)


def _aic(model: IceSteelFrictionModel, anchors) -> float:
    """AIC in coefficient units on the calibration residuals."""
    residuals = np.array([model.mu(a.v_m_per_s, a.temperature_c, a.pressure_mpa) - a.mu
                          for a in anchors])
    n = len(residuals)
    rss = float(np.sum(residuals ** 2))
    k = len(model.coefficients)
    return float(n * math.log(rss / n) + 2 * k)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> dict:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)

    cal = list(CALIBRATION_ANCHORS)
    models = {form: fit_ice_steel_friction(cal, form=form)
              for form in ("loglinear", "loglinear+thermal")}
    recommended = models["loglinear+thermal"]

    record: dict = {
        "case": {
            "name": "ice-steel friction coefficient mu(v, T, p)",
            "physical_background": (
                "The tangential traction that a steel hull carries against ice is "
                "Coulomb-saturated at mu*N. Ice friction is speed-weakened by "
                "frictional melting, temperature-dependent through the amount of "
                "liquid water at the interface and pressure-weakened by pressure "
                "melting and real-contact-area growth; the ice-resistance chain in "
                "this repository used a constant coefficient instead."),
            "model_form": ("mu = mu_ref + a*log10(v/v_ref) + b*(T-T_ref) "
                           "+ c*(T-T_ref)^2 + d*log10(p/p_ref)"),
            "reference_condition": {"v_m_per_s": recommended.v_ref,
                                    "temperature_c": recommended.t_ref_c,
                                    "pressure_mpa": recommended.p_ref_mpa},
            "units": {"v": "m/s", "T": "degC (below melting)", "p": "MPa",
                      "mu": "-"},
        },
        "evidence": {
            "law_implementation": "REAL - closed-form weighted least squares, no "
                                  "iterative solver, machine-checked monotonicity",
            "anchor_set": "LIT-RANGE - compiled representative values transcribed "
                          "from the cited literature, not digitised measurements",
            "independent_experimental_validation": "WITHHELD - no certified "
                                                   "experimental data set available "
                                                   "in the worktree; the <=10 % gate is "
                                                   "measured against the compiled "
                                                   "independent source rows only",
            "dem_realisation": "REAL - existing Cundall-Strack platen contact, "
                               "coefficient recovered from the measured slip "
                               "acceleration",
            "resistance_case": "RULE-STYLE reference case (documented coefficients), "
                               "not a class-society resistance prediction",
        },
        "provenance": recommended.provenance,
        "anchors": {
            "calibration": [row.as_dict() for row in cal],
            "validation": [row.as_dict() for row in VALIDATION_ANCHORS],
            "extrapolation": [row.as_dict() for row in EXTRAPOLATION_PROBES],
            "n_calibration": len(cal),
            "n_validation": len(VALIDATION_ANCHORS),
            "sources": sorted({row.source for row in cal}),
            "envelope": calibrated_envelope(cal),
        },
        "fits": {},
        "validation": {},
        "sensitivity": {},
        "negative_control": {},
        "dem_probes": [],
    }

    for form, model in models.items():
        record["fits"][form] = {
            "coefficients": {
                "mu_ref": model.mu_ref, "a_log_v": model.a_log_v,
                "b_temperature": model.b_temperature,
                "c_temperature2": model.c_temperature2,
                "d_log_p": model.d_log_p,
            },
            "n_terms": len(model.coefficients),
            "fit_rms_rel": model.fit_rms_rel,
            "fit_max_rel": model.fit_max_rel,
            "aic": _aic(model, cal),
            "mu_at_reference": model.mu_at_reference,
            "predicted_mu_at_classic_design_point": {
                "v_1_m_per_s_T_minus5_p_1_MPa": float(model.mu(1.0, -5.0, 1.0)),
            },
        }
        record["validation"][form] = {
            "independent_holdout": holdout_error_table(model, VALIDATION_ANCHORS),
            "leave_one_source_out": leave_one_source_out(cal, form=form),
            "kfold": kfold_validation(cal, k=5, form=form),
            "extrapolation_probes": holdout_error_table(model, EXTRAPOLATION_PROBES),
        }

    # coverage check on the extrapolation probes
    record["validation"]["extrapolation_coverage"] = [
        {"label": row.label, "coverage": recommended.coverage(
            row.v_m_per_s, row.temperature_c, row.pressure_mpa)}
        for row in EXTRAPOLATION_PROBES
    ]

    envelope = operating_envelope()
    record["sensitivity"] = sensitivity_table(recommended, envelope=envelope)
    record["negative_control"] = negative_control_table(recommended, envelope=envelope)

    for spec in DEM_PROBES:
        record["dem_probes"].append(dem_friction_probe(recommended, **spec))

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speeds = np.logspace(-4, 0.5, 120)
    figure_paths = []

    def anchor_xy(rows, key):
        return ([row[key] for row in rows], [row["mu"] for row in rows])

    # fig 1: coefficient vs sliding speed
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for temperature, style in ((-2.0, "-"), (-10.0, "--"), (-20.0, "-."), (-30.0, ":")):
        ax.plot(speeds, [recommended.mu(v, temperature, 0.2) for v in speeds],
                style, label=f"law, T={temperature:.0f} degC")
    xs, ys = anchor_xy([row.as_dict() for row in cal if row.source.startswith("Timco")], "v_m_per_s")
    ax.plot(xs, ys, "o", color="k", label="compiled anchors (T=-10 degC)")
    ax.set_xscale("log")
    ax.set_xlabel("sliding speed v [m/s]")
    ax.set_ylabel("friction coefficient mu [-]")
    ax.set_title("Ice-steel friction: speed weakening at 0.2 MPa")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = FIGURE_DIR / "mu_vs_speed.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figure_paths.append(path)

    # fig 2: coefficient vs temperature
    temps = np.linspace(-30.0, -1.0, 120)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for speed, style in ((0.001, ":"), (0.01, "-"), (0.1, "--"), (1.0, "-.")):
        ax.plot(temps, [recommended.mu(speed, t, 0.2) for t in temps], style,
                label=f"law, v={speed:g} m/s")
    xs, ys = anchor_xy([row.as_dict() for row in cal if row.source.startswith("Barnes")],
                       "temperature_c")
    ax.plot(xs, ys, "s", color="k", label="compiled anchors (v=0.01 m/s)")
    ax.set_xlabel("ice temperature T [degC]")
    ax.set_ylabel("friction coefficient mu [-]")
    ax.set_title("Ice-steel friction: colder ice rubs harder (p = 0.2 MPa)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    path = FIGURE_DIR / "mu_vs_temperature.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figure_paths.append(path)

    # fig 3: coefficient vs contact pressure
    pressures = np.logspace(-1.5, 0.6, 120)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(pressures, [recommended.mu(0.01, -10.0, p) for p in pressures], "-",
            label="law, v=0.01 m/s, T=-10 degC")
    xs, ys = anchor_xy([row.as_dict() for row in cal if row.source.startswith("Oksanen")],
                       "pressure_mpa")
    ax.plot(xs, ys, "d", color="k", label="compiled anchors")
    ax.set_xscale("log")
    ax.set_xlabel("contact pressure p [MPa]")
    ax.set_ylabel("friction coefficient mu [-]")
    ax.set_title("Ice-steel friction: pressure weakening")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    path = FIGURE_DIR / "mu_vs_pressure.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figure_paths.append(path)

    # fig 4: parity plot
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    groups = (("calibration", cal, "o", "calibration anchors"),
              ("validation", list(VALIDATION_ANCHORS), "s", "independent validation"),
              ("extrapolation", list(EXTRAPOLATION_PROBES), "^", "deliberate extrapolation"))
    for _, rows, marker, label in groups:
        xs = [row.mu for row in rows]
        ys = [recommended.mu(row.v_m_per_s, row.temperature_c, row.pressure_mpa)
              for row in rows]
        ax.plot(xs, ys, marker, label=label, alpha=0.85)
    lims = [0.02, 0.17]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    for band, alpha in ((0.10, 0.12),):
        ax.fill_between(lims, [x * (1 - band) for x in lims],
                        [x * (1 + band) for x in lims], color="C0", alpha=alpha,
                        label=f"+-{band:.0%} band")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("compiled coefficient mu [-]")
    ax.set_ylabel("law prediction mu [-]")
    ax.set_title("Parity: mu(v, T, p) against the reference set")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    path = FIGURE_DIR / "parity.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figure_paths.append(path)

    # fig 5: sensitivity + negative control
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    scales = np.linspace(0.8, 1.2, 21)
    case = IceResistanceCase()
    for spec, style in (((0.01, -10.0, 0.5), "-"), ((0.5, -15.0, 1.0), "--"),
                        ((1.0, -5.0, 0.3), "-.")):
        totals = [ice_resistance(*spec, recommended, case=case, mu_scale=scale)
                  ["total_resistance_n"] / 1e3 for scale in scales]
        axes[0].plot(scales, totals, style,
                     label=f"v={spec[0]:g} m/s, T={spec[1]:g} degC, p={spec[2]:g} MPa")
    axes[0].set_xlabel("mu scale factor [-]"); axes[0].set_ylabel("total resistance [kN]")
    axes[0].set_title("+-20 % sensitivity: monotone, sub-linear")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=7)

    rows = record["negative_control"]["rows"]
    inside = [row for row in rows if row["inside_calibrated_envelope"]]
    axes[1].plot([row["mu_model"] for row in inside],
                 [row["friction_resistance_rel_error"] * 100 for row in inside],
                 "o", ms=4)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("calibrated mu [-]")
    axes[1].set_ylabel("constant mu=0.1 friction error [%]")
    axes[1].set_title("Negative control: cost of the hard-coded mu")
    axes[1].grid(True, alpha=0.3)
    path = FIGURE_DIR / "sensitivity_negative_control.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figure_paths.append(path)

    # fig 6: DEM realisation
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    targets = [probe["mu_target"] for probe in record["dem_probes"]]
    effective = [probe["mu_effective"] for probe in record["dem_probes"]]
    ax.plot(targets, effective, "o", label="DEM probe (slip branch)")
    lims = [0.0, max(targets) * 1.15]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.fill_between(lims, [x * 0.97 for x in lims], [x * 1.03 for x in lims],
                    color="C0", alpha=0.15, label="+-3 % gate")
    ax.set_xlabel("prescribed / calibrated mu [-]")
    ax.set_ylabel("effective mu measured in DEM [-]")
    ax.set_title("DEM realisation of the calibrated coefficient")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    path = FIGURE_DIR / "dem_probe.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figure_paths.append(path)

    record["figures"] = [str(path.relative_to(ROOT)) for path in figure_paths]

    # ---------------- acceptance vector ----------------
    holdout = record["validation"]["loglinear+thermal"]["independent_holdout"]
    holdout4 = record["validation"]["loglinear"]["independent_holdout"]
    kfold_best = record["validation"]["loglinear+thermal"]["kfold"]
    sens = record["sensitivity"]
    control = record["negative_control"]
    record["acceptance"] = {
        "a_least_squares_calibration": {
            "pass": True,
            "fit_rms_rel": recommended.fit_rms_rel,
            "fit_max_rel": recommended.fit_max_rel,
            "n_anchors": recommended.n_anchors,
        },
        "b_independent_validation_le_10pct": {
            "pass": bool(holdout["passes_gate"]),
            "form": "loglinear+thermal",
            "max_rel_error": holdout["max_rel_error"],
            "rms_rel_error": holdout["rms_rel_error"],
            "n_within_gate": holdout["n_within_gate"],
            "n": holdout["n"],
            "campaign_form_loglinear_max_rel_error": holdout4["max_rel_error"],
            "campaign_form_loglinear_pass": bool(holdout4["passes_gate"]),
            "kfold_max_rel_error": kfold_best["pooled_max_rel_error"],
            "gate": VALIDATION_GATE,
        },
        "c_sensitivity_pm20pct_monotone": {
            "pass": bool(sens["all_monotone_increase"]),
            "n_points": sens["n_points"],
            "relative_change_range": [sens["relative_change_min"],
                                      sens["relative_change_max"]],
            "bounded_by_scale_change": bool(sens["all_bounded_by_scale_change"]),
            "mean_friction_share": sens["mean_friction_share"],
        },
        "d_negative_control_constant_mu": {
            "pass": bool(control["max_abs_mu_rel_error"] > 0.10),
            "constant_mu": MU_CONSTANT_BASELINE,
            "in_envelope": control["in_envelope"],
            "n_out_of_envelope": control["n_out_of_envelope"],
            "n_floor_bound": control["n_floor_bound"],
        },
        "e_dem_realisation_le_3pct": {
            "pass": bool(all(probe["passes_dem_gate"] for probe in record["dem_probes"])),
            "max_slope_rel_error": max(probe["slope_rel_error"]
                                       for probe in record["dem_probes"]),
            "max_mu_rel_error": max(probe["mu_rel_error"]
                                    for probe in record["dem_probes"]),
        },
        "f_limitations_disclosed": {
            "pass": True,
            "extrapolation_probe_max_error": record["validation"][
                "loglinear+thermal"]["extrapolation_probes"]["max_rel_error"],
            "anchor_set_evidence": "LIT-RANGE compiled representative values",
            "independent_experimental_validation": "WITHHELD",
            "power_density_gap": (
                "no calibration row combines v > 0.1 m/s with p > 1 MPa; the "
                "coverage flag, not the per-variable box, is the validity test"),
            "floe_scale_dem_transit": "not run; contact-level DEM evidence only",
        },
    }

    ARTIFACT.write_text(json.dumps(_jsonable(record), indent=2, allow_nan=False))

    # ---------------- digest ----------------
    summary_lines = [
        "# ice-steel friction mu(v,T,p) - calibration digest",
        f"anchors: {len(cal)} calibration / {len(VALIDATION_ANCHORS)} validation / "
        f"{len(EXTRAPOLATION_PROBES)} extrapolation",
        "",
        "law (loglinear+thermal):",
        f"  mu_ref={recommended.mu_ref:.6f} a_log_v={recommended.a_log_v:.6f} "
        f"b_T={recommended.b_temperature:.6f} c_T2={recommended.c_temperature2:.3e} "
        f"d_log_p={recommended.d_log_p:.6f}",
        f"  fit rms={recommended.fit_rms_rel:.2%} max={recommended.fit_max_rel:.2%}",
        f"law (loglinear, campaign form):",
        f"  mu_ref={models['loglinear'].mu_ref:.6f} "
        f"a_log_v={models['loglinear'].a_log_v:.6f} "
        f"b_T={models['loglinear'].b_temperature:.6f} "
        f"d_log_p={models['loglinear'].d_log_p:.6f}",
        f"  fit rms={models['loglinear'].fit_rms_rel:.2%} "
        f"max={models['loglinear'].fit_max_rel:.2%}",
        "",
        f"validation (independent rows): thermal max={holdout['max_rel_error']:.2%} "
        f"({holdout['n_within_gate']}/{holdout['n']} <= 10 %) | "
        f"loglinear max={holdout4['max_rel_error']:.2%}",
        f"k-fold(5): thermal max={kfold_best['pooled_max_rel_error']:.2%} | "
        f"loglinear max={record['validation']['loglinear']['kfold']['pooled_max_rel_error']:.2%}",
        f"extrapolation probes: max={record['validation']['loglinear+thermal']['extrapolation_probes']['max_rel_error']:.2%}",
        "",
        f"sensitivity: monotone={sens['all_monotone_increase']} "
        f"dR/R in [{sens['relative_change_min']:.2%}, {sens['relative_change_max']:.2%}] "
        f"for mu +-20 %, mean friction share={sens['mean_friction_share']:.1%}",
        f"negative control mu=0.1: in-envelope mu error max="
        f"{control['in_envelope']['max_abs_mu_rel_error']:.1%} "
        f"resistance error max={control['in_envelope']['max_abs_resistance_rel_error']:.1%} "
        f"rms={control['in_envelope']['rms_resistance_rel_error']:.1%}; "
        f"{control['n_out_of_envelope']} envelope-exceeding points, "
        f"{control['n_floor_bound']} floor-bound",
        "",
        "DEM probes:",
    ]
    for probe in record["dem_probes"]:
        summary_lines.append(
            f"  {probe['source']}: mu_target={probe['mu_target']:.4f} "
            f"mu_effective={probe['mu_effective']:.4f} "
            f"slope err={probe['slope_rel_error']:.2%} pass={probe['passes_dem_gate']}")
    summary_lines.append("")
    summary_lines.append("acceptance:")
    for key, value in record["acceptance"].items():
        summary_lines.append(f"  {key}: pass={value.get('pass')}")
    summary = "\n".join(summary_lines) + "\n"
    (RUN_DIR / "summary.txt").write_text(summary)
    print(summary)
    print(f"artifact: {ARTIFACT}")
    print(f"figures: {len(figure_paths)} in {FIGURE_DIR}")
    return record


if __name__ == "__main__":
    main()