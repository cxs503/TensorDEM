"""Figures for the DEM auto-calibration report (reads the artifact JSON only).

    PYTHONPATH=src python3 scripts/plot_dem_calibration.py \
        --artifacts-dir artifacts/dem_auto_calibration --out-dir artifacts/dem_auto_calibration

Every number plotted comes from the JSON records written by
``scripts/calibrate_dem_materials.py``; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MATERIALS = ("glass_beads", "sand", "gravel")


def _load(directory: str) -> list[dict]:
    records = []
    for name in MATERIALS:
        path = os.path.join(directory, f"{name}.json")
        with open(path, "r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def fig_response_surface(records: list[dict], out: str) -> str:
    """Repose angle and objective against the sliding friction, per material."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for record in records:
        table = record["search"]["table"]
        mu = [row["params"]["mu"] for row in table]
        theta = [row["responses"]["theta_r_deg"] for row in table]
        obj = [row["objective"] for row in table]
        axes[0].scatter(mu, theta, s=22, alpha=0.8, label=record["material"])
        axes[1].scatter(mu, obj, s=22, alpha=0.8, label=record["material"])
        truth = record["truth"]
        axes[0].axhline(record["targets"]["theta_r_deg"], ls=":", lw=1,
                        color=axes[0].collections[-1].get_facecolor()[0])
        axes[0].axvline(truth["mu"], ls="--", lw=1,
                        color=axes[0].collections[-1].get_facecolor()[0])
    axes[0].set_xlabel(r"grain-grain friction $\mu_{pp}$ [-]")
    axes[0].set_ylabel(r"repose angle $\theta_r$ [deg]")
    axes[0].set_title("forward response (dotted: twin target, dashed: true $\\mu$)")
    axes[1].set_xlabel(r"grain-grain friction $\mu_{pp}$ [-]")
    axes[1].set_ylabel(r"objective $J$ [-]")
    axes[1].set_yscale("log")
    axes[1].axhline(0.05, color="k", ls="--", lw=1, label="acceptance 0.05")
    axes[1].set_title("calibration objective")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_convergence(records: list[dict], out: str) -> str:
    """Best objective so far against the number of forward evaluations."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for record in records:
        table = record["search"]["table"]
        y = np.array([row["objective"] for row in table])
        ax.plot(np.arange(1, y.size + 1), np.minimum.accumulate(y), marker="o",
                ms=3, label=f"{record['material']} (n={y.size})")
    ax.axhline(0.05, color="k", ls="--", lw=1, label="acceptance 0.05")
    ax.set_xlabel("forward DEM evaluations")
    ax.set_ylabel("best objective so far $J$ [-]")
    ax.set_yscale("log")
    ax.set_title("BayesOpt convergence (GP + expected improvement)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_posterior(records: list[dict], out: str) -> str:
    """95 % credible intervals against the twin-experiment truth."""
    names = ("mu", "restitution", "stiffness_ratio")
    labels = {
        "mu": r"$\mu_{pp}$",
        "restitution": "$e$",
        "stiffness_ratio": "$k_n/k_s$",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for j, name in enumerate(names):
        ax = axes[j]
        for k, record in enumerate(records):
            summary = record["posterior"]["summary"][name]
            lo, hi, mid = summary["lo95"], summary["hi95"], summary["median"]
            ax.plot([k, k], [lo, hi], color="C0", lw=6, alpha=0.5, solid_capstyle="round")
            ax.plot(k, mid, "o", color="C0")
            ax.plot(k, record["truth"][name], "*", color="C3", ms=12)
        ax.set_xticks(range(len(records)))
        ax.set_xticklabels([r["material"] for r in records], rotation=20, fontsize=8)
        ax.set_ylabel(labels[name])
        ax.set_title(f"{labels[name]}: 95 % CI (bar) vs truth (star)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_generalisation(records: list[dict], out: str) -> str:
    """Predicted vs target repose angle on the independent validation geometry."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    lo, hi = 0.0, 0.0
    for record in records:
        gen = record["generalisation"]
        ax.plot(gen["theta_target_deg"], gen["theta_predicted_deg"], "o", ms=8,
                label=record["material"])
        lo = min(lo, gen["theta_target_deg"])
        hi = max(hi, gen["theta_target_deg"], gen["theta_predicted_deg"])
    xs = np.linspace(0.0, hi * 1.1, 10)
    ax.plot(xs, xs, "k--", lw=1, label="1:1")
    ax.fill_between(xs, 0.9 * xs, 1.1 * xs, color="k", alpha=0.08, label="+/-10 %")
    ax.set_xlabel("target repose angle, validation geometry [deg]")
    ax.set_ylabel("prediction with calibrated parameters [deg]")
    ax.set_title("generalisation (no refitting)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", default="artifacts/dem_auto_calibration")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or args.artifacts_dir
    os.makedirs(out_dir, exist_ok=True)
    records = _load(args.artifacts_dir)
    for record in records:
        table = record["search"]["table"]
        for row in table:
            if "responses" not in row:
                row["responses"] = None
        # responses are cached in the record only when the search ran with them;
        # rebuild from truth-independent data if absent
        for row in table:
            if row.get("responses") is None:
                row["responses"] = {"theta_r_deg": float("nan")}
    paths = [
        fig_response_surface(records, os.path.join(out_dir, "fig1_response_surface.png")),
        fig_convergence(records, os.path.join(out_dir, "fig2_convergence.png")),
        fig_posterior(records, os.path.join(out_dir, "fig3_posterior.png")),
        fig_generalisation(records, os.path.join(out_dir, "fig4_generalisation.png")),
    ]
    for path in paths:
        print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())