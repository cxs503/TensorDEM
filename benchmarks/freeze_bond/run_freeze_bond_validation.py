"""Benchmark / validation driver for the freeze-bond mode-transition node.

Node P46W3-N07 — freeze-bond strength and the adhesive -> cohesive failure-mode
transition.  Produces, under ``benchmarks/freeze_bond/``:

* ``freeze_bond_report.json``   — every acceptance number, programmatically
  injected (no hand-typed values),
* ``figures/*.png``             — constitutive law + mode map, F-u curves,
  discrete-vs-continuum and kink diagnostics,
* ``curves.csv``                — the evaluated model curves (audit trail),
* ``anchor_comparison.csv``     — anchor-vs-model point table.

Run:  python benchmarks/freeze_bond/run_freeze_bond_validation.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:  # allow running from a bare checkout
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.freeze_bond.mode_transition import (  # noqa: E402
    ACCEPTANCE,
    BondInterfaceParams,
    bond_strength_temperature,
    failure_mode_map,
    fiber_bundle_pull_off,
    ice_tensile_strength,
    load_anchor_set,
    peak_force,
    peak_slope_kink,
    pull_off_curve,
    run_validation_report,
    transition_temperature,
)

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"


def write_curves_csv(report: dict, params: BondInterfaceParams) -> Path:
    path = OUT / "curves.csv"
    t_c = report["parameters"]["tau_growth_s"] and load_anchor_set()["test_conditions"]["contact_time_s"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["temperature_C", "mode", "u_um", "F_N", "F_over_peak"])
        for temp in (-2.0, -5.0, -8.0, -11.0, -15.0, -20.0):
            cd = pull_off_curve(temp, t_c, params)
            for u, f in zip(cd.u_m, cd.force_n):
                w.writerow([temp, cd.mode, f"{u * 1e6:.6f}", f"{f:.6f}", f"{f / cd.peak_force_n:.6f}"])
    return path


def write_anchor_csv(report: dict) -> Path:
    path = OUT / "anchor_comparison.csv"
    rows = report["peak_validation"]["rows"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "T_C",
                "mode_obs",
                "mode_pred",
                "mode_agree",
                "F_peak_obs_N",
                "F_peak_pred_N",
                "rel_error",
                "in_transition_band",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["T_c"],
                    r["mode_obs"],
                    r["mode_pred"],
                    r["mode_agree"],
                    r["F_peak_obs_n"],
                    f"{r['F_peak_pred_n']:.4f}",
                    f"{r['rel_error']:.5f}",
                    r["in_transition_band"],
                ]
            )
    return path


def fig_constitutive(params: BondInterfaceParams, anchor: dict, t_c: float) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    temps = np.linspace(-0.05, -25.0, 400)
    sb = np.array([bond_strength_temperature(t, params) for t in temps]) / 1e6
    si = np.array([ice_tensile_strength(t, params) for t in temps]) / 1e6
    ax1.plot(temps, sb, "C3-", lw=2.2, label=r"$\sigma_{bond}=\sigma_0e^{-\beta(T-T_m)}$")
    ax1.plot(temps, si, "C0-", lw=2.2, label=r"$\sigma_{ice}=\sigma_{ice,ref}(\Delta T/\Delta T_{ref})^n$")
    tr = transition_temperature(params, contact_time_s=t_c)
    ax1.axvline(tr, color="k", ls=":", lw=1.2)
    ax1.annotate(
        f"$T_{{trans}}$={tr:.2f} °C", xy=(tr, 1.55), xytext=(tr - 6.5, 1.75),
        arrowprops=dict(arrowstyle="->", lw=1.0), fontsize=10,
    )
    obs = [(r["T_c"], r["F_peak_N"] / (params.area_m2 * 1e6)) for r in anchor["peak_tests"]]
    ax1.plot([o[0] for o in obs], [o[1] for o in obs], "ks", ms=6, label="anchor (peak tests, MPa)")
    ax1.set_xlabel("temperature [°C]")
    ax1.set_ylabel("strength [MPa]")
    ax1.set_title("(a) bond / ice strength and the mode transition")
    ax1.set_xlim(-25, 0)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    temps_map = np.linspace(0.0, -25.0, 251)
    times = np.logspace(0, 3.2, 121)
    mp = failure_mode_map(temps_map, times, params)
    ax2.pcolormesh(temps_map, times, mp["mode_codes"], cmap="coolwarm", shading="auto", vmin=0, vmax=2)
    t_trans = [transition_temperature(params, contact_time_s=tt) for tt in times]
    ax2.plot(t_trans, times, "k--", lw=1.5, label=r"$T_{trans}(t_c)$")
    ax2.set_yscale("log")
    ax2.set_xlabel("temperature [°C]")
    ax2.set_ylabel("contact time [s]")
    ax2.set_title("(b) failure-mode map (blue=no bond, white=adhesive, red=cohesive)")
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path = FIG / "fig1_constitutive_and_mode_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_curves(params: BondInterfaceParams, anchor: dict, t_c: float) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, temp in zip(axes, (-5.0, -15.0)):
        cd = pull_off_curve(temp, t_c, params)
        ax.plot(cd.u_m * 1e6, cd.force_n, "C0-", lw=2.2, label=f"model ({cd.mode}, $G_f$={cd.G_f_J_m2:.1f} J/m²)")
        anchor_curve = next(c for c in anchor["curves"] if abs(c["T_c"] - temp) < 1e-6)
        ax.plot(anchor_curve["u_um"], anchor_curve["F_N"], "ks--", ms=5, lw=1.0,
                label=f"anchor T={temp:.0f} °C ({anchor_curve['mode']})")
        pred_r2 = next(
            c for c in run_validation_report(params, t_c)["checks"]["c_force_displacement"]["curves"]
            if abs(c["T_c"] - temp) < 1e-6
        )
        ax.annotate(f"$R^2$={pred_r2['r2']:.4f}", xy=(0.55, 0.85), xycoords="axes fraction", fontsize=10)
        ax.set_xlabel("displacement u [µm]")
        ax.set_ylabel("force F [N]")
        ax.set_title(f"(F-u) T = {temp:.0f} °C")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "fig2_force_displacement.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_discrete(params: BondInterfaceParams, t_c: float) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    report = run_validation_report(params, t_c)
    bundles = report["checks"]["discrete_bundle"]["cases"]
    for temp, colour in ((-5.0, "C3"), (-15.0, "C0")):
        bd = fiber_bundle_pull_off(temp, t_c, params)
        cd = pull_off_curve(temp, t_c, params)
        ax1.plot(bd.u_m * 1e6, bd.force_n / bd.peak_force_n, colour + "-", lw=2.0,
                 label=f"discrete bundle T={temp:.0f} °C ({bd.mode})")
        ax1.plot(cd.u_m * 1e6, cd.force_n / cd.peak_force_n, "k--", lw=1.2, label=f"continuum T={temp:.0f} °C")
    ax1.set_xlabel("displacement u [µm]")
    ax1.set_ylabel("F / $F_{peak}$ [-]")
    ax1.set_title("(c) 4096-fibre discrete pull-off vs closed form")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    ax1.set_xlim(0, 55)

    kink = peak_slope_kink(params, t_c)
    tr = kink["transition_temperature_c"]
    temps = np.linspace(tr - 12, tr + 6, 120)
    fp = np.array([peak_force(t, t_c, params)["peak_force_n"] for t in temps])
    modes = [peak_force(t, t_c, params)["mode"] for t in temps]
    colours = ["C0" if m == "cohesive" else "C3" for m in modes]
    ax2.scatter(temps, fp, c=colours, s=10)
    ax2.axvline(tr, color="k", ls=":", lw=1.2)
    ax2.annotate(
        f"$T_{{trans}}$={tr:.2f} °C", xy=(tr, fp.min()), xytext=(tr - 9, fp.min() * 0.75),
        arrowprops=dict(arrowstyle="->", lw=1.0), fontsize=9,
    )
    ax2.set_xlabel("temperature [°C]")
    ax2.set_ylabel("peak force $F_p$ [N]")
    ax2.set_title(
        f"(d) kink, not jump: slope ratio = {kink['slope_ratio']:.2f}"
    )
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = FIG / "fig3_discrete_and_kink.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    params = BondInterfaceParams()
    anchor = load_anchor_set()
    t_c = anchor["test_conditions"]["contact_time_s"]
    report = run_validation_report(params, t_c)
    report["artifacts"] = {
        "module": "src/tensordem/freeze_bond/mode_transition.py",
        "anchor_set": "benchmarks/freeze_bond/freeze_bond_anchor_set.json",
        "tests": "tests/test_freeze_bond_mode_transition.py",
        "report_md": "docs/freeze_bond_mode_transition_report.md",
    }
    report["acceptance_thresholds"] = ACCEPTANCE
    report["figures"] = [
        str(fig_constitutive(params, anchor, t_c).relative_to(ROOT)),
        str(fig_curves(params, anchor, t_c).relative_to(ROOT)),
        str(fig_discrete(params, t_c).relative_to(ROOT)),
    ]
    report["artifacts"]["curves_csv"] = str(write_curves_csv(report, params).relative_to(ROOT))
    report["artifacts"]["anchor_csv"] = str(write_anchor_csv(report).relative_to(ROOT))
    path = OUT / "freeze_bond_report.json"
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    s = report["checks"]
    print(json.dumps(
        {
            "all_checks_passed": report["all_checks_passed"],
            "beta_input": s["a_temperature_law"]["beta_input_per_C"],
            "beta_fitted": s["a_temperature_law"]["beta_fitted_per_C"],
            "T_trans_model_C": s["b_mode_transition"]["T_trans_model_c"],
            "T_trans_anchor_C": s["b_mode_transition"]["T_trans_anchor_c"],
            "T_trans_abs_error_C": s["b_mode_transition"]["abs_error_c"],
            "min_curve_r2": s["c_force_displacement"]["min_r2"],
            "max_curve_peak_rel_error": s["c_force_displacement"]["max_peak_rel_error"],
            "max_curve_Gf_rel_error": s["c_force_displacement"]["max_fracture_energy_rel_error"],
            "max_peak_rel_error": s["peak_force_vs_temperature"]["max_rel_error"] if "max_rel_error" in s["peak_force_vs_temperature"] else report["peak_validation"]["max_rel_error"],
            "mode_agreement": report["peak_validation"]["mode_agreement"],
            "ledger_max_rel_error": s["energy_ledger"]["max_energy_rel_error"],
            "bundle_min_r2": s["discrete_bundle"]["min_r2"],
            "bundle_energy_rel_error": s["discrete_bundle"]["max_energy_rel_error"],
            "slope_ratio_kink": s["kink_signature"]["slope_ratio"],
            "jump_ratio_k1": s["kink_signature"]["jump_k1"]["jump_ratio"],
            "jump_ratio_k2": s["kink_signature"]["jump_k2"]["jump_ratio"],
            "report": str(path.relative_to(ROOT)),
        },
        indent=2,
    ))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())