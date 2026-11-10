#!/usr/bin/env python
"""Figures for the material-class repose-angle calibration (p399w2-n07).

Reads ``artifacts/repose_material_calibration/calibration.json`` (written by
``scripts/calibrate_repose_materials.py``) and writes PNGs to
``docs/repose_material_calibration/``:

* ``fig1_calibration_curves.png``   theta_r(mu) per material + targets + mu*
* ``fig2_inverse_curves.png``       theta_target -> mu*(theta) inversion
* ``fig3_deposit_sections.png``     x-z sections of the calibrated deposits
* ``fig4_controls.png``             negative control + mu_r sensitivity
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOURS = {"glass_bead": "#1f77b4", "sand": "#ff7f0e", "gravel": "#2ca02c"}
LABELS = {"glass_bead": "glass beads", "sand": "sand", "gravel": "gravel"}


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def fig_calibration(payload: dict, out: str) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for key, entry in payload["materials"].items():
        nodes = entry["scan_nodes"]
        mu = [n["mu"] for n in nodes]
        theta = [n["theta_r_deg"] for n in nodes]
        err = [n["theta_r_std_deg"] for n in nodes]
        ax.errorbar(mu, theta, yerr=err, marker="o", ms=4, lw=1.6, capsize=2,
                    color=COLOURS[key], label=f"{LABELS[key]} (DEM)")
        target = entry["material"]["target_theta_deg"]
        ax.axhline(target, color=COLOURS[key], ls=":", lw=1.0, alpha=0.8)
        cal = entry.get("calibration", {})
        if cal.get("mu_star") is not None and math.isfinite(cal.get("mu_star", float("nan"))):
            ax.plot([cal["mu_star"]], [cal["theta_sim_deg"]], marker="*", ms=13,
                    color=COLOURS[key], mec="k", mew=0.5,
                    label=f"{LABELS[key]}: mu*={cal['mu_star']:.3f}")
        ax.annotate(f"target {target:.0f}$^\\circ$", xy=(0.21, target + 0.8),
                    color=COLOURS[key], fontsize=8)
    ax.set_xlabel(r"sliding friction coefficient $\mu$ [-]")
    ax.set_ylabel(r"angle of repose $\theta_r$ [deg]")
    ax.set_title("DEM repose-angle scan and calibrated $\\mu$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    path = os.path.join(out, "fig1_calibration_curves.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_inverse(payload: dict, out: str) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for key, entry in payload["materials"].items():
        table = entry["table"]
        mu = table["mu"]
        theta = table["monotone_theta_deg"]
        ax.plot(mu, theta, marker="s", ms=3.5, lw=1.4, color=COLOURS[key],
                label=f"{LABELS[key]}: $\\mu(\\theta)$")
        target = entry["material"]["target_theta_deg"]
        ax.axhline(target, color=COLOURS[key], ls=":", lw=1.0, alpha=0.8)
    ax.set_xlabel(r"$\mu$ [-]")
    ax.set_ylabel(r"$\theta_r$ [deg] (monotone table)")
    ax.set_title("Inversion basis $\\mu \\rightarrow \\theta_r$ and target levels")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    path = os.path.join(out, "fig2_inverse_curves.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_deposits(payload: dict, runs_dir: str, out: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9), sharey=False)
    found = False
    for ax, (key, entry) in zip(axes, payload["materials"].items()):
        pattern = os.path.join(runs_dir, f"{key}__verification__*.json")
        import glob
        files = [f for f in sorted(glob.glob(pattern)) if "positions_xyz" in _load(f)]
        if not files:
            ax.set_title(f"{LABELS[key]}: no snapshot")
            ax.axis("off")
            continue
        record = _load(files[0])
        pos = record["positions_xyz"]
        xs = [p[0] * 1e3 for p in pos]
        zs = [p[2] * 1e3 for p in pos]
        sc = ax.scatter(xs, zs, c=zs, s=3.0, cmap="viridis", linewidths=0)
        cal = entry.get("calibration", {})
        ax.set_title(f"{LABELS[key]}, "
                     f"$\\mu$={record['job']['mu']:.3f}\n"
                     f"$\\theta_r$={cal.get('theta_sim_deg', float('nan')):.1f}$^\\circ$",
                     fontsize=9)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="z [mm]")
        found = True
    if not found:
        plt.close(fig)
        return ""
    fig.suptitle("Calibrated deposits: vertical x-z section (colour = height)")
    path = os.path.join(out, "fig3_deposit_sections.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_controls(payload: dict, out: str) -> str:
    keys = list(payload["materials"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    neg = [payload["materials"][k]["negative_control"]["theta_r_deg"] for k in keys]
    cal = [payload["materials"][k].get("calibration", {}).get("theta_sim_deg", float("nan"))
           for k in keys]
    x = range(len(keys))
    ax1.bar([i - 0.2 for i in x], cal, width=0.4, color="tab:blue", label="calibrated pile")
    ax1.bar([i + 0.2 for i in x], neg, width=0.4, color="tab:red",
            label=r"negative control ($\mu=\mu_r=0$)")
    ax1.axhline(5.0, color="k", ls="--", lw=1.0, label="5$^\\circ$ control limit")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([LABELS[k] for k in keys])
    ax1.set_ylabel(r"$\theta_r$ [deg]")
    ax1.set_title("Negative control: no friction $\\Rightarrow$ no slope")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, axis="y")

    width = 0.8 / max(1, len(keys))
    for i, key in enumerate(keys):
        entry = payload["materials"][key]
        rows = entry["sensitivity"]
        base_mur = entry["material"]["mu_rolling"]
        factors = [r["mu_rolling"] / base_mur for r in rows]
        base_theta = next((r["theta_r_deg"] for r in rows
                           if abs(r["mu_rolling"] / base_mur - 1.0) < 1e-9),
                          rows[0]["theta_r_deg"])
        delta = [r["theta_r_deg"] - base_theta for r in rows]
        ax2.bar([f + (i - len(keys) / 2 + 0.5) * width for f in range(len(factors))],
                delta, width=width, color=COLOURS[key], label=LABELS[key])
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["0.5 $\\mu_r$", "1.0 $\\mu_r$", "1.5 $\\mu_r$"])
    ax2.set_ylabel(r"$\Delta\theta_r$ [deg]")
    ax2.set_title("Rolling-resistance sensitivity at the calibrated $\\mu$")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, axis="y")
    path = os.path.join(out, "fig4_controls.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact",
                        default="artifacts/repose_material_calibration/calibration.json")
    parser.add_argument("--runs", default="artifacts/repose_material_calibration/runs")
    parser.add_argument("--out", default="docs/repose_material_calibration")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    payload = _load(args.artifact)
    paths = [fig_calibration(payload, args.out), fig_inverse(payload, args.out),
             fig_deposits(payload, args.runs, args.out), fig_controls(payload, args.out)]
    for path in paths:
        if path:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())