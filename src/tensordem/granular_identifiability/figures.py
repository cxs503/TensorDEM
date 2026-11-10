"""Figures for the granular identifiability study (matplotlib, Agg backend)."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .targets import TARGET_NAMES  # noqa: E402

__all__ = ["make_figures"]

_LABELS = {
    "grain_friction_ratio": r"grain friction ratio $\tau/\sigma$ (-)",
    "packing_fraction": r"settled packing fraction $\phi$ (-)",
    "restitution": r"impact restitution $e$ (-)",
    "impact_duration_s": r"impact duration $\tau_c$ (s)",
}
_PARAM_LABELS = {
    "friction": r"$\mu$",
    "restitution": r"$e$",
    "stiffness": r"$k_n$",
}


def _save(fig, out_dir: str, name: str, caption: str) -> dict:
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": name, "path": path, "caption": caption}


def make_figures(report: dict, out_dir: str) -> list[dict]:
    """Render the study figures and return their descriptors."""
    os.makedirs(out_dir, exist_ok=True)
    figures: list[dict] = []
    params = ["friction", "restitution", "stiffness"]

    # 1. observables vs parameter perturbation ----------------------------
    raw = report["propagation"]["raw"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, target in zip(axes.ravel(), TARGET_NAMES):
        for name in params:
            row = raw[name]
            y = np.asarray([obs[TARGET_NAMES.index(target)] for obs in row["y"]])
            frac = np.asarray([(p - row["values"][0]) / row["values"][0] for p in row["values"]])
            base = y[0]
            ax.plot(100 * frac, 100 * (y - base) / abs(base), "o-", label=_PARAM_LABELS[name])
        ax.set_title(_LABELS[target], fontsize=10)
        ax.set_xlabel("parameter perturbation (%)")
        ax.set_ylabel("observable change (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Forward observables of test_granular_01: +-10 % parameter scan", fontsize=11)
    figures.append(
        _save(
            fig,
            out_dir,
            "fig1_observable_sensitivity.png",
            "Each observable against a +-10 % perturbation of the three contact "
            "parameters; the straightness of the traces is the linear-response "
            "(uncertainty propagation) criterion.",
        )
    )

    # 2. discretisation ladder -------------------------------------------
    changes = report["discretisation"]["changes"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.18
    x = np.arange(len(TARGET_NAMES))
    for k, (key, label) in enumerate([("dt_vs_dt_half", "|dt -> dt/2|"), ("max_relative_change", "max over ladder")]):
        vals = [
            100.0 * (changes[t][key] if changes[t][key] is not None else np.nan)
            for t in TARGET_NAMES
        ]
        ax.bar(x + (k - 0.5) * width, vals, width, label=label)
    ax.axhline(1.0, color="crimson", ls="--", lw=1, label="1 % criterion")
    ax.set_xticks(x)
    ax.set_xticklabels([_LABELS[t] for t in TARGET_NAMES], fontsize=8, rotation=12)
    ax.set_ylabel("relative change of the observable (%)")
    ax.set_title("Discretisation independence of the observables")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    figures.append(
        _save(
            fig,
            out_dir,
            "fig2_discretisation_ladder.png",
            "Relative change of each observable between successive time-step "
            "refinements (dt, dt/2, dt/4) against the 1 % acceptance line.",
        )
    )

    # 3. FIM spectrum + correlation --------------------------------------
    fim = report["fim"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    eig = np.asarray(fim["eigenvalues"], dtype=float)
    ax1.semilogy(np.arange(1, eig.size + 1), eig, "o-")
    ax1.set_xlabel("eigen-index")
    ax1.set_ylabel("FIM eigenvalue (unit-cube coordinates)")
    ax1.set_title(
        f"Fisher information spectrum\ncond = {fim['condition_number']:.2f} "
        f"(bulk-only: {fim['bulk_only']['condition_number']:.1f})",
        fontsize=10,
    )
    ax1.grid(alpha=0.3, which="both")
    corr = np.asarray(fim["correlation"], dtype=float)
    im = ax2.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax2.set_xticks(range(len(params)))
    ax2.set_yticks(range(len(params)))
    ax2.set_xticklabels([_PARAM_LABELS[p] for p in params])
    ax2.set_yticklabels([_PARAM_LABELS[p] for p in params])
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax2.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax2.set_title("parameter correlation from the FIM", fontsize=10)
    fig.colorbar(im, ax=ax2, fraction=0.046)
    figures.append(
        _save(
            fig,
            out_dir,
            "fig3_fisher_information.png",
            "Fisher-information eigenvalue spectrum and the implied parameter "
            "correlation matrix at the reference point; the bulk-only control "
            "shows the information content of the flow observables alone.",
        )
    )

    # 4. inversion recovery ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    runs = report["inversion"]["runs"]
    labels = [run["start"]["name"] for run in runs]
    x = np.arange(len(params))
    width = 0.8 / max(1, len(runs))
    for k, run in enumerate(runs):
        errs = [100.0 * run["recovery"][p]["rel_error"] for p in params]
        ax.bar(x + (k - (len(runs) - 1) / 2) * width, errs, width,
               label=f"start {labels[k]} (chi2={run['chi2']:.2f})")
    ax.axhline(5.0, color="crimson", ls="--", lw=1, label="5 % criterion")
    ax.set_xticks(x)
    ax.set_xticklabels([_PARAM_LABELS[p] for p in params])
    ax.set_ylabel("|recovered - true| / true (%)")
    ax.set_title("Parameter recovery from a noise-perturbed synthetic observation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    figures.append(
        _save(
            fig,
            out_dir,
            "fig4_inversion_recovery.png",
            "Relative recovery error of the three contact parameters from two "
            "starting points, against the 5 % acceptance line.",
        )
    )

    # 5. settled bed state (the "field" plot of the case) ----------------
    prof = report["nominal"]["profile_summary"]
    x_c = np.asarray(prof["x"], dtype=float)
    h = np.asarray(prof["height"], dtype=float)
    phi = np.asarray([v if v is not None else np.nan for v in prof["porosity"]], dtype=float)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax1.fill_between(x_c, 0.0, h, color="tab:brown", alpha=0.35)
    ax1.plot(x_c, h, "o-", color="tab:brown", ms=3)
    ax1.set_ylabel("free-surface height (m)")
    ax1.set_title("Settled bed of test_granular_01 (vertical profiles)", fontsize=10)
    ax1.grid(alpha=0.3)
    ax2.plot(x_c, 1.0 - phi, "s-", color="tab:blue", ms=3)
    ax2.set_ylabel(r"solid fraction $1-\epsilon$ in bin")
    ax2.set_xlabel("x (m)")
    ax2.grid(alpha=0.3)
    figures.append(
        _save(
            fig,
            out_dir,
            "fig5_settled_bed_profiles.png",
            "Settled-bed free-surface height and local solid fraction of the "
            "forward case at the reference parameters (the packing-fraction "
            "observable is the integral of the second panel).",
        )
    )
    return figures