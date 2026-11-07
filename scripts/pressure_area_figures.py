#!/usr/bin/env python3
"""Render the figures of the pressure-area campaign from its JSON artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("artifacts/pressure_area_index")


def _load():
    summary = json.loads((OUT / "summary.json").read_text())
    scans = {}
    for path in sorted(OUT.glob("scan_*.json")):
        label = path.stem.replace("scan_", "")
        scans[label] = json.loads(path.read_text())
    growth = json.loads((OUT / "contact_growth.json").read_text())
    return summary, scans, growth


def fig_pressure_area(summary, scans):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140)
    ax = axes[0]
    for label, style in (("primary", dict(color="C0", marker="o")),
                         ("control_uniform", dict(color="C3", marker="s"))):
        data = scans[label]
        fit = data["fit"]
        areas = np.array(data["areas_m2"])
        means = [data["seed_mean_pressure_Pa"][f"{a:g}"] for a in areas]
        sds = [data["seed_stdev_pressure_Pa"][f"{a:g}"] for a in areas]
        ax.errorbar(areas, means, yerr=sds, capsize=3, lw=1.2, ms=4,
                    label=(f"{'correlated' if label == 'primary' else 'uniform'} "
                           f"field  ex={fit['exponent']:.3f}"), **style)
        slope, intercept = np.polyfit(np.log(areas), np.log(means), 1)
        xs = np.logspace(np.log10(areas.min()), np.log10(areas.max()), 50)
        ax.plot(xs, np.exp(intercept) * xs ** slope, ls="--", lw=0.9,
                color=style["color"])
    # reference slope -1/2
    xs = np.array([0.5, 10.0])
    p0 = scans["primary"]["seed_mean_pressure_Pa"]["0.5"]
    ax.plot(xs, p0 * (xs / xs[0]) ** -0.5, color="k", lw=0.8, ls=":",
            label="reference A^-0.5")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nominal contact area A [m$^2$]")
    ax.set_ylabel("elastic-limit global pressure $p_G$ [Pa]")
    ax.set_title("Pressure-area relation from DEM indentation")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    labels, exs, los, his = [], [], [], []
    for label in ("primary", "primary_wide", "control_uniform", "control_wide",
                  "l_sensitivity_0.05", "l_sensitivity_0.2",
                  "m_sensitivity_cov0.36", "m_sensitivity_cov0.26"):
        if label not in scans:
            continue
        fit = scans[label]["fit"]
        labels.append(label.replace("_", " "))
        exs.append(fit["exponent"]); los.append(fit["ci_low"]); his.append(fit["ci_high"])
    y = np.arange(len(labels))
    ax.errorbar(exs, y, xerr=[np.array(exs) - np.array(los),
                              np.array(his) - np.array(exs)],
                fmt="o", color="C0", capsize=3, ms=4)
    ax.axvline(0.5, color="k", ls=":", lw=0.8, label="field value ex = 0.5")
    ax.axvline(0.0, color="C3", ls=":", lw=0.8, label="simultaneous-failure limit ex = 0")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("fitted exponent ex  (bootstrap 95 % CI)")
    ax.set_title("Exponent family: statistics and correlation length")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    path = OUT / "fig_pressure_area.png"
    fig.savefig(path); plt.close(fig)
    return path


def fig_contact_growth(growth):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=140)
    ax = axes[0]
    for series in growth[:4]:
        disp = np.array(series["displacement_m"]) * 1e3
        area = np.array(series["contact_area_m"])
        ax.plot(disp, area, lw=1.2,
                label=f"A = {series['area_m2']:g} m$^2$")
        if len(disp) > 3:
            ax.plot(disp, area[0] * (disp / disp[0]) ** 0.5, ls=":", lw=0.7,
                    color="k")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("punch penetration $u$ [mm]")
    ax.set_ylabel("true contact area $A_c$ [m per m depth]")
    ax.set_title("Contact area grows with penetration ($A_c \\propto u^{1/2}$ dotted)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for series in growth[:4]:
        disp = np.array(series["displacement_m"]) * 1e3
        force = np.array(series["force_N_per_m"]) / 1e6
        ax.plot(disp, force, lw=1.2, label=f"A = {series['area_m2']:g} m$^2$")
    ax.set_xlabel("punch penetration $u$ [mm]")
    ax.set_ylabel("punch force per unit depth [MN/m]")
    ax.set_title("Force-penetration response (elastic branch to first failure)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig_contact_growth.png"
    fig.savefig(path); plt.close(fig)
    return path


def fig_statistics(scans):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=140)
    ax = axes[0]
    for label, marker in (("primary", "o"), ("control_uniform", "s")):
        data = scans[label]
        areas = np.array(data["areas_m2"])
        ns = [np.mean([r["non_simultaneity"] for r in data["runs"]
                       if r["area_m2"] == a]) for a in areas]
        ax.plot(areas, ns, marker=marker, ms=4, lw=1.1,
                label=("correlated field" if label == "primary" else "uniform field"))
    ax.set_xscale("log")
    ax.set_xlabel("nominal contact area A [m$^2$]")
    ax.set_ylabel("non-simultaneity index")
    ax.set_title("Failure non-simultaneity: correlated vs uniform strength")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    data = scans["primary"]
    events = []
    for run in data["runs"]:
        events.extend(run["failure_events"])
    if events:
        ax.hist(events, bins=range(0, max(events) + 2), color="C0", alpha=0.8)
    ax.set_xlabel("bonds removed in a single failure event")
    ax.set_ylabel("count")
    ax.set_title("Per-event failure size (primary scan)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig_statistics.png"
    fig.savefig(path); plt.close(fig)
    return path


def main() -> int:
    summary, scans, growth = _load()
    made = [fig_pressure_area(summary, scans), fig_contact_growth(growth),
            fig_statistics(scans)]
    for path in made:
        print(path, path.stat().st_size // 1024, "KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())