#!/usr/bin/env python3
"""Figures for the thermal-conduction + granular-collapse verification run.

Reads ``runs/dem_thermal_granular/{thermal.json,summary.json,case_*.json}`` and
writes PNGs into ``artifacts/dem_thermal_granular/``.  Every plotted number comes
from those JSON files (no hand-copied values).
"""
from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = "runs/dem_thermal_granular"
OUT = "artifacts/dem_thermal_granular"


def _load(name: str):
    with open(os.path.join(RUNS, name)) as handle:
        return json.load(handle)


def fig_chain_steady(thermal: dict) -> str:
    steady = thermal["chain_steady"]
    x = np.asarray(steady["profile_x_over_l"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot(x, steady["profile_analytic_k"], "k-", lw=2.0, label="closed form (series)")
    ax.plot(x, steady["profile_numeric_k"], "r--", lw=1.2, label="network solve")
    ax.set_xlabel("x / L  (duplicated abscissa at the contacts)")
    ax.set_ylabel("temperature (K)")
    ax.set_title("(a) sphere chain: steady profile, N=%d, m=%d"
                 % (steady["n_spheres"], steady["cells_per_sphere"]))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(x, steady["profile_analytic_k"], "k-", lw=2.0, label="closed form")
    ax.plot(x, steady["profile_numeric_k"], "r--", lw=1.2, label="network solve")
    ax.set_xlim(0.0, 2.0 / steady["n_spheres"])
    ax.set_ylim(steady["t_hot_k"] - 12.0, steady["t_hot_k"] + 1.0)
    ax.set_xlabel("x / L")
    ax.set_ylabel("temperature (K)")
    ax.set_title("(b) first two contacts: $qR_c$ jumps")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("Thermal contact chain: max |T - T$_{analytic}$| = %.2e K (%.1e of the span)"
                 % (steady["max_abs_error_k"], steady["max_rel_error"]), fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "fig1_chain_steady.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_chain_transient(thermal: dict) -> str:
    tr = thermal["chain_transient"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot(tr["time_s"], tr["exact_hot_flux_w"], "k-", lw=2.0,
            label="matrix exponential (exact)")
    ax.plot(tr["time_s"], tr["hot_flux_w"], "r--", lw=1.2, label="implicit Euler")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("hot-end heat flow (W)")
    ax.set_title("(a) transient of the same RC network")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    steps = [c["steps_per_tau"] for c in tr["convergence"]]
    err = [c["max_rel_error"] for c in tr["convergence"]]
    ax.loglog(steps, err, "o-", label="measured")
    ref = np.asarray(steps, dtype=float)
    ax.loglog(ref, err[0] * (ref / ref[0]) ** -1.0, "k:", label="first order")
    ax.set_xlabel("steps per relaxation time (dt$^{-1}$)")
    ax.set_ylabel("max relative error vs exact")
    ax.set_title("(b) time-discretisation error, order = %.3f"
                 % float(np.mean(tr["convolution_orders"])))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(OUT, "fig2_chain_transient.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_bond_gate(thermal: dict) -> str:
    gate = thermal["ice_bond_gate"]
    curve = gate["strength_curve"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(curve["t_k"], np.asarray(curve["sigma_pa"]) / 1e6, "k-", lw=2.0,
            label=r"$\sigma(T)=\sigma_{ref}\exp[\frac{Q}{nR}(\frac{1}{T}-\frac{1}{T_{ref}})]$")
    ax.axhline(gate["applied_stress_pa"] / 1e6, color="C3", ls="--",
               label="applied bond stress %.2f MPa" % (gate["applied_stress_pa"] / 1e6))
    ax.axvline(gate["t_bond_critical_k"], color="C0", ls=":",
               label="critical $T_{bond}$ = %.2f K" % gate["t_bond_critical_k"])
    ax.plot([gate["t_bond_critical_k"]], [gate["strength_critical_pa"] / 1e6], "o", color="C0")
    ax.set_xlabel("bond temperature (K)")
    ax.set_ylabel("tensile strength (MPa)")
    ax.set_title("ice-bond thermal-mechanical gate\nanalytic failure %.1f s vs simulated %.1f s (%.2f %%)"
                 % (gate["analytic_failure_time_s"], gate["simulated_failure_time_s"],
                    100.0 * gate["failure_time_rel_error"]), fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "fig3_bond_gate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_runout_scaling(summary: dict) -> str:
    cases = [c for c in summary["cases"] if c["task"]["kind"] == "aspect"]
    cases.sort(key=lambda c: c["task"]["aspect"])
    a = np.asarray([c["task"]["aspect"] for c in cases], dtype=float)
    refs = [
        ("0.5d", "o", "0.5 d (any grain)", 0.5),
        ("2.0d", "s", "2 d (default)", 2.0),
        ("3.0d", "^", "3 d", 3.0),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for key, marker, label, _ in refs:
        y = np.asarray([c["runout_thresholds_m"][key] for c in cases]) - \
            np.asarray([c["initial_length_m"] for c in cases])
        y = np.maximum(y / np.asarray([c["initial_length_m"] for c in cases]), 1e-6)
        ax.loglog(a, y, marker + "-", label="toe at " + label)
    fit = summary["fit_low_aspect"]
    grid = np.linspace(min(a), 3.0, 50)
    ax.loglog(grid, fit["prefactor"] * grid ** fit["exponent"], "k--", lw=1.0,
              label="fit a<=3: p = %.3f" % fit["exponent"])
    ax.loglog(grid, grid, "k:", lw=1.0, label="Lajeunesse 2005 (p = 1)")
    ax.loglog(grid, 0.7 * grid ** (2.0 / 3.0), "k-.", lw=1.0, label="Lube 2005 2D (p = 2/3)")
    ax.set_xlabel("initial aspect ratio a = H / L0")
    ax.set_ylabel(r"$\Delta L / L_0$")
    ax.set_title("(a) 2D collapse runout vs aspect ratio")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    for key, marker, label, _ in refs:
        y = np.asarray([c["runout_thresholds_m"][key] for c in cases]) - \
            np.asarray([c["initial_length_m"] for c in cases])
        y = np.maximum(y / np.asarray([c["initial_length_m"] for c in cases]), 1e-6)
        ax.semilogx(a, y / a ** 1.0, marker + "-", label="toe at " + label)
    ax.axhline(1.0, color="k", ls=":", label="pure linear (p = 1)")
    ax.set_xlabel("initial aspect ratio a = H / L0")
    ax.set_ylabel(r"$(\Delta L / L_0) / a$")
    ax.set_title("(b) collapse onto $a^1$: flat line would mean p = 1")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(OUT, "fig4_runout_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_front_history() -> str:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for name in sorted(os.listdir(RUNS)):
        if not (name.startswith("case_aspect_") and name.endswith(".json")):
            continue
        with open(os.path.join(RUNS, name)) as handle:
            payload = json.load(handle)
        hist = np.asarray(payload["front_history"], dtype=float)
        l0 = payload["result"]["initial_length_m"]
        tau = payload["result"]["free_fall_time_s"]
        ax.plot(hist[:, 0] / tau, hist[:, 1] / l0,
                label="a = %g" % payload["result"]["aspect"])
    ax.set_xlabel(r"$t / \sqrt{2H/g}$")
    ax.set_ylabel("deposit toe / $L_0$")
    ax.set_title("front kinematics of the released column")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "fig5_front_kinematics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_energy_ledger(summary: dict) -> str:
    ledger = None
    for case in summary["cases"]:
        if case["task"]["kind"] == "ledger":
            ledger = case["energy_ledger"]
    if ledger is None:
        raise SystemExit("no ledger case in the summary")
    labels = ["E_pot released", "E_kin final", "E_elastic", "E_diss measured"]
    values = [ledger["potential_initial_j"] - ledger["potential_j"],
              ledger["kinetic_j"],
              ledger["elastic_normal_j"] + ledger["elastic_tangential_j"],
              ledger["dissipated_j"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    bars = ax.bar(labels, values, color=["C0", "C1", "C2", "C3"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, "%.2f J" % value,
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("energy (J)")
    ax.set_title("(a) ledger terms, closure error %.2f %%"
                 % (100.0 * ledger["closure_error"]))
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    channels = ["damping", "friction"]
    values2 = [ledger["dissipated_damping_j"], ledger["dissipated_friction_j"]]
    total = sum(values2)
    ax.bar(channels, values2, color=["C4", "C5"])
    for i, value in enumerate(values2):
        ax.text(i, value, "%.1f %%\n%.2f J" % (100.0 * value / total, value),
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("dissipated energy (J)")
    ax.set_title("(b) channel split (forward-measured)")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Figure 6: potential-energy / dissipation ledger of the a = 2 collapse",
                 fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "fig6_energy_ledger.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_inertial_scan(summary: dict) -> str:
    scan = summary["inertial_number_scan"]
    g = np.asarray([s["gravity"] for s in scan])
    i_peak = np.asarray([s["inertial_number_peak"] for s in scan])
    runout = np.asarray([s["relative_runout"] for s in scan])
    mu = np.asarray([s["mu_flow_mean"] for s in scan])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.semilogx(g, i_peak, "o-", label=r"$I_{peak}$")
    ax.semilogx(g, [s["inertial_number_mean"] for s in scan], "s--",
                label=r"$I_{mean}$")
    ax.set_xlabel("gravity (m/s$^2$)")
    ax.set_ylabel("inertial number I")
    ax.set_title("(a) inertial-number sweep at a = 2")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    ax.plot(i_peak, runout, "o-", label=r"$\Delta L/L_0$ vs $I_{peak}$")
    ax2 = ax.twinx()
    ax2.plot(i_peak, mu, "s--", color="C1", label=r"$\bar\mu(I)$")
    ax.set_xlabel(r"$I_{peak}$")
    ax.set_ylabel(r"$\Delta L / L_0$")
    ax2.set_ylabel(r"flow-averaged $\mu$")
    ax.set_title("(b) runout and friction vs inertial number")
    ax.grid(alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=7)
    fig.tight_layout()
    path = os.path.join(OUT, "fig7_inertial_scan.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    thermal = _load("thermal.json")
    summary = _load("summary.json")
    made = [fig_chain_steady(thermal), fig_chain_transient(thermal),
            fig_bond_gate(thermal), fig_runout_scaling(summary),
            fig_front_history(), fig_energy_ledger(summary),
            fig_inertial_scan(summary)]
    for path in made:
        print(path, os.path.getsize(path), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())