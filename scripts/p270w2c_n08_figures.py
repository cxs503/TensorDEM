#!/usr/bin/env python
"""Figures for P270W2C-N08: response surface, identifiability, Pareto, holdout, disc field.

Reads ``artifacts/p270w2c_n08/report.json`` (written by
``scripts/p270w2c_n08_feasibility.py --stage analyze``) and writes the PNGs below
``artifacts/p270w2c_n08/figures/``.  Kept separate from the report renderer so a
figure can be regenerated without touching the markdown.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "p270w2c_n08"

PARAM_LABELS = {
    "k_b": r"$k_b$",
    "eps_b": r"$\varepsilon_b$",
    "mu_s": r"$\mu_s$",
    "mu_d": r"$\mu_d$",
    "k_c": r"$k_c$",
    "k_p": r"$k_p$",
}
PARAM_COLUMNS = {
    "k_b": "bond_stiffness",
    "eps_b": "breaking_strain",
    "mu_s": "static_friction",
    "mu_d": "dynamic_friction",
    "k_c": "contact_stiffness",
    "k_p": "tangential_contact_stiffness",
}


def figure_response_surface(report: dict, path: Path) -> None:
    surface = report["response_surface"]
    names = list(surface)
    theta = np.array([list(surface[n]["theta"].values()) for n in names], dtype=float)
    keys = ("ucs_MPa", "strength_ratio", "weibull_m")
    labels = (r"UCS [MPa]", r"$\sigma_c/\sigma_t$ [-]", r"Weibull $m$ [-]")
    fig, axes = plt.subplots(3, 6, figsize=(16.0, 7.8))
    for row, (key, label) in enumerate(zip(keys, labels)):
        values = np.array([surface[n][key] for n in names], dtype=float)
        for col, parameter in enumerate(PARAM_LABELS):
            ax = axes[row, col]
            x = theta[:, col]
            mask = np.isfinite(values) & (values > 0)
            ax.scatter(x[mask], values[mask], s=12, alpha=0.75, color="#1f77b4")
            if row == 0:
                ax.set_title(PARAM_LABELS[parameter], fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if key in ("ucs_MPa", "strength_ratio"):
                band = (report["targets"]["ucs_band_MPa"] if key == "ucs_MPa"
                        else report["targets"]["strength_ratio_band"])
                ax.axhspan(band[0], band[1], color="#2ca02c", alpha=0.15)
            if key == "weibull_m":
                ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
    fig.suptitle("Response surface $\\theta \\to$ (UCS, $\\sigma_c/\\sigma_t$, $m$); "
                 "shaded band = acceptance target", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_jacobian(report: dict, path: Path) -> None:
    diagnostics = report["AC2_identifiability"]["diagnostics"]
    jacobian = np.array(report["AC2_identifiability"]["jacobian"]["jacobian"], dtype=float)
    correlation = np.array(diagnostics["correlation"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4),
                             gridspec_kw={"width_ratios": [1.4, 1.0, 1.0]})
    ax = axes[0]
    limit = float(np.nanmax(np.abs(jacobian))) if np.any(np.isfinite(jacobian)) else 1.0
    image = ax.imshow(jacobian, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit)
    ax.set_xticks(range(jacobian.shape[1]))
    ax.set_xticklabels(list(PARAM_LABELS), rotation=35, fontsize=9)
    ax.set_yticks(range(jacobian.shape[0]))
    ax.set_yticklabels(["UCS", r"$\sigma_t$", "m"], fontsize=9)
    for i in range(jacobian.shape[0]):
        for j in range(jacobian.shape[1]):
            ax.text(j, i, f"{jacobian[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(r"$\partial \log_{10} y / \partial u$", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046)

    ax = axes[1]
    image = ax.imshow(correlation, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(correlation.shape[0]))
    ax.set_xticklabels(list(PARAM_LABELS), rotation=35, fontsize=8)
    ax.set_yticks(range(correlation.shape[0]))
    ax.set_yticklabels(list(PARAM_LABELS), fontsize=8)
    for i in range(correlation.shape[0]):
        for j in range(correlation.shape[1]):
            value = correlation[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.6 else "black")
    ax.set_title("parameter correlation matrix", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046)

    ax = axes[2]
    pairs = diagnostics["collinearity"]["pairs"][:8]
    labels = [f"{p['pair'][0]}-{p['pair'][1]}" for p in pairs]
    values = [p["collinearity"] if math.isfinite(p["collinearity"]) else 1e6 for p in pairs]
    ax.barh(range(len(values)), np.log10(np.maximum(values, 1.0)), color="#d62728")
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(r"$\log_{10}\gamma$ (Brun collinearity)", fontsize=8)
    ax.axvline(2.0, color="k", ls="--", lw=0.8)
    ax.set_title(f"cond(J) = {diagnostics['cond']:.2e}, rank = {diagnostics['rank']}",
                 fontsize=10)
    fig.suptitle("Identifiability audit of $\\theta=(k_b,\\varepsilon_b,\\mu_s,\\mu_d,k_c,k_p)$ "
                 "from three observables", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_pareto(report: dict, path: Path) -> None:
    population = np.array(report["AC3_pareto"]["population_objectives"], dtype=float)
    front = report["AC3_pareto"]["front"]
    front_obj = (np.array([row["objectives"] for row in front], dtype=float)
                 if front else np.zeros((0, 3)))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    pairs = ((0, 1), (0, 2), (1, 2))
    names = ("UCS violation", r"$\sigma_c/\sigma_t$ violation", "m violation")
    for ax, (a, b) in zip(axes, pairs):
        ax.scatter(population[:, a], population[:, b], s=10, alpha=0.35,
                   color="#7f7f7f", label="NSGA-II population")
        if front_obj.size:
            ax.scatter(front_obj[:, a], front_obj[:, b], s=24, color="#d62728",
                       label="Pareto front")
        ax.set_xlabel(names[a], fontsize=9)
        ax.set_ylabel(names[b], fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("NSGA-II on the response-surface surrogate: normalised target "
                 "violations (0 = target met)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_size_effect(report: dict, path: Path) -> None:
    holdout = report["AC4_holdout"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    family = report.get("reference_family", {})
    thicknesses = sorted(float(t) for t in family)
    if thicknesses:
        means = [family[f"{t:.3f}"]["mean_Pa"] / 1e6 for t in thicknesses]
        stds = [family[f"{t:.3f}"]["std_Pa"] / 1e6 for t in thicknesses]
        ax.errorbar(thicknesses, means, yerr=stds, fmt="o", color="#1f77b4",
                    label="DEM family mean $\\pm$ 1 sd")
    m = float(holdout["reference"].get("m", float("nan")))
    sigma_ref = float(holdout["reference"].get("sigma_ref_Pa", float("nan")))
    if math.isfinite(m) and m > 0 and math.isfinite(sigma_ref):
        grid = np.linspace(0.04, 0.9, 200)
        ax.plot(grid, sigma_ref * grid ** (-1.0 / m) / 1e6, "-", color="#d62728",
                label=f"power-law fit, m = {m:.2f}")
    for t in (0.05, 0.8):
        ax.axvline(t, color="k", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("specimen thickness $t$ [m]", fontsize=9)
    ax.set_ylabel("tensile strength $\\sigma_t$ [MPa]", fontsize=9)
    ax.set_title("Weibull size effect (fitted on $t$ = 0.1/0.2/0.4, dotted = held out)",
                 fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    rows = holdout["reference"]["rows"]
    if rows:
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [row["measured_Pa"] / 1e6 for row in rows], width=0.4,
               label="DEM (held out)", color="#1f77b4")
        ax.bar(x + 0.2, [row["predicted_Pa"] / 1e6 for row in rows], width=0.4,
               label="power-law prediction", color="#ff7f0e")
        for i, row in enumerate(rows):
            top = max(row["measured_Pa"], row["predicted_Pa"]) / 1e6
            ax.text(i, top * 1.03, f"{row['relative_error'] * 100:+.1f} %",
                    ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"t = {row['thickness_m']:.2f} m" for row in rows], fontsize=9)
        ax.set_ylabel("$\\sigma_t$ [MPa]", fontsize=9)
        ax.set_title("AC4 held-out extrapolation error", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_brazilian(report: dict, path: Path) -> None:
    ac5 = report["AC5_brazilian_field"]
    rows = ac5["rows"]
    target = ac5["target"]
    tolerance = ac5["tolerance"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row["n_lattice"], row["n_layers"]), []).append(row)
    for (n_lattice, n_layers), members in sorted(groups.items()):
        members = sorted(members, key=lambda r: r["region_ratio"])
        ax.plot([m["region_ratio"] for m in members],
                [m["ratio_at_peak"] for m in members], "o-",
                label=f"{n_lattice}$\\times${n_layers} lattice")
    ax.axhline(target, color="k", ls="--", label="Hondros (line load) = -3.0")
    ax.axhspan(target * (1 + tolerance), target * (1 - tolerance), color="#2ca02c",
               alpha=0.15, label="AC5 tolerance band")
    ax.set_xlabel("stress-averaging radius / D", fontsize=9)
    ax.set_ylabel("$\\sigma_{load}/\\sigma_{transverse}$ at disc centre", fontsize=9)
    ax.set_title("Brazilian disc: central principal-stress ratio", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    for (n_lattice, n_layers), members in sorted(groups.items()):
        member = max(members, key=lambda r: r["region_ratio"])
        history = member.get("ratio_history") or []
        finite = [(i, v) for i, v in enumerate(history) if v is not None and math.isfinite(v)]
        if finite:
            ax.plot([i for i, _ in finite], [v for _, v in finite], "o-",
                    label=f"{n_lattice}$\\times${n_layers}, r/D = {member['region_ratio']:.2f}")
        transverse = member.get("sigma_transverse_history_Pa") or []
        load = member.get("sigma_load_history_Pa") or []
        if transverse and load:
            ax.plot(np.arange(len(transverse)),
                    np.array(transverse) / np.maximum(np.abs(load), 1e-30), "s--",
                    alpha=0.5, label=f"{n_lattice}$\\times${n_layers} $\\sigma_x/|\\sigma_z|$")
    ax.axhline(target, color="k", ls="--")
    ax.set_xlabel("read-out index", fontsize=9)
    ax.set_ylabel("ratio", fontsize=9)
    ax.set_title("ratio history (peak = last read-out)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_ucs(report: dict, path: Path) -> None:
    trace = report.get("reference_trace", {})
    band = report["targets"]["ucs_band_MPa"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ax = axes[0]
    if trace.get("strain"):
        ax.plot(np.array(trace["strain"]) * 100.0,
                np.array(trace["stress_Pa"]) / 1e6, "o-", color="#1f77b4")
    ax.axhspan(band[0], band[1], color="#2ca02c", alpha=0.12, label="literature band")
    ax.set_xlabel("axial strain [%]", fontsize=9)
    ax.set_ylabel("nominal stress [MPa]", fontsize=9)
    ax.set_title("Uniaxial compression at the reference $\\theta$", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    negative = report["AC6_negative_control"]["ucs_MPa"]
    if negative:
        values = np.clip(negative, 0.0, 40.0)
        ax.hist(values, bins=40, color="#7f7f7f")
    ax.axvspan(band[0], band[1], color="#2ca02c", alpha=0.25, label="literature band")
    ax.set_xlabel("UCS [MPa] (clipped at 40)", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title(f"AC6 negative control: {report['AC6_negative_control']['n_samples']} random $\\theta$",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    report = json.loads((ARTIFACTS / "report.json").read_text())
    figures = ARTIFACTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure_response_surface(report, figures / "fig_response_surface.png")
    figure_jacobian(report, figures / "fig_jacobian.png")
    figure_pareto(report, figures / "fig_pareto.png")
    figure_size_effect(report, figures / "fig_size_effect.png")
    figure_brazilian(report, figures / "fig_brazilian.png")
    figure_ucs(report, figures / "fig_ucs.png")
    print("figures written to", figures)


if __name__ == "__main__":
    main()