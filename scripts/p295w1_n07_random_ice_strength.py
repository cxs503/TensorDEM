#!/usr/bin/env python
"""Figures for P295W1-N07 -- random ice-plate strength field and force ensemble.

Every number plotted is read from ``runs/P295W1_N07/study.json`` (produced by
:func:`tensordem.random_ice_strength.study.run_study`); nothing is typed here.
Run:

    PYTHONPATH=src python scripts/p295w1_n07_random_ice_strength.py [--rerun]

``--rerun`` regenerates the study JSON first.  Four figures are written into
``artifacts/p295w1_n07/``:

``fig1_qq_weibull.png``     Weibull QQ plot + lognormal negative control
``fig2_correlation.png``    realisation correlation vs kernel for L = 5/10/20 m
``fig3_ensemble.png``       breaking-force ensemble histogram + running mean
``fig4_size_effect.png``    F_5%/B vs width, 4 sweep variants + both models
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402

STUDY = Path("runs/P295W1_N07/study.json")
OUT = Path("artifacts/p295w1_n07")


def _load(rerun: bool) -> dict:
    if rerun or not STUDY.exists():
        from tensordem.random_ice_strength import run_study

        run_study()
    return json.loads(STUDY.read_text())


def fig_qq(study: dict) -> Path:
    a = study["stage_a_marginal"]
    marg = a["marginal"]
    mle = a["mle"]
    from scipy.special import gammaincinv

    x = np.asarray(a["qq_sample"], dtype=float)
    k, s0 = marg["modulus"], marg["scale_pa"]
    # theoretical Weibull quantiles for the plotting positions of the stored,
    # decimated sample (median-rank positions recomputed on the full sample size)
    n_full = int(mle["n"])
    idx = np.arange(1, x.size + 1)
    p = (idx * (n_full / x.size) - 0.3) / (n_full + 0.4)
    theory = s0 * (-np.log1p(-p)) ** (1.0 / k)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes[0]
    ax.plot(theory / 1e6, x / 1e6, "o", ms=3.0, alpha=0.75,
            label=f"sampled strengths (n={n_full})")
    lim = [0.0, max(theory.max(), x.max()) / 1e6 * 1.03]
    ax.plot(lim, lim, "k--", lw=1.0, label="exact Weibull (k, sigma_0)")
    ax.set_xlabel(r"theoretical Weibull quantile  $F^{-1}(p)$  [MPa]")
    ax.set_ylabel("ordered sample  [MPa]")
    ax.set_title(f"(a) Weibull QQ plot:  $R^2$ = {a['qq_r2']:.5f}\n"
                 f"$k_{{MLE}}$ = {mle['k_hat']:.3f} (input {k:.2f}), "
                 f"$\\sigma_0$ = {mle['scale_hat']/1e6:.4f} MPa "
                 f"(input {s0/1e6:.4f})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    nc = a["negative_control"]
    ax.bar(["Weibull sample\n(accept)", "lognormal control\n(reject)"],
           [a["qq_r2"], nc["qq_r2"]], color=["#2b7a3d", "#9c2b2b"], width=0.55)
    ax.axhline(0.95, color="k", ls="--", lw=1.0, label="acceptance $R^2 \\geq 0.95$")
    for i, v in enumerate([a["qq_r2"], nc["qq_r2"]]):
        ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0.85, 1.005)
    ax.set_ylabel("QQ-plot $R^2$")
    ax.set_title(f"(b) negative control $\\sigma_{{\\ln}}$={nc['sigma_ln']:.1f}: "
                 f"$R^2$={nc['qq_r2']:.4f}, KS p={nc['ks_p_value']:.1e}")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("P295W1-N07 (a) Weibull marginal of beam-strip flexural strength",
                 fontsize=11)
    fig.tight_layout()
    path = OUT / "fig1_qq_weibull.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_correlation(study: dict) -> Path:
    b = study["stage_b_field"]
    rows = sorted(b["lengths"], key=lambda r: r["length_m"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes[0]
    colours = ["#1f77b4", "#d62728", "#2ca02c"]
    for colour, row in zip(colours, rows):
        lags = np.asarray(row["lags_m"])
        ax.plot(lags, row["rho_target"], "-", color=colour, lw=2.2,
                label=f"kernel $e^{{-(d/L)^2}}$, L={row['length_m']:.0f} m")
        ax.plot(lags, row["rho_empirical"], "o", color=colour, ms=2.6, alpha=0.7,
                label=f"realisations: L2 err {row['l2_relative']*100:.1f}%")
    ax.axhline(math.exp(-1.0), color="k", ls=":", lw=1.0)
    ax.set_xlabel("separation $d$ [m]")
    ax.set_ylabel("correlation  $\\rho(d)$")
    ax.set_title(f"(a) Gaussian field correlation, "
                 f"N={b['n_realisations']} realisations, {b['grid_points']} grid points")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([4.0, 22.0], [4.0, 22.0], "k--", lw=1.0, label="measured = prescribed")
    for colour, row in zip(colours, rows):
        ax.plot([row["length_m"]], [row["measured_length_1e_m"]], "o", ms=9,
                color=colour, label=f"L={row['length_m']:.0f} m: "
                                    f"{row['measured_length_1e_m']:.2f} m")
        ax.plot([row["length_m"]], [row["rank_measured_length_1e_m"]], "s", ms=7,
                mfc="none", color=colour)
    ax.set_xlabel("prescribed correlation length $L$ [m]")
    ax.set_ylabel("measured $1/e$ length [m]")
    ax.set_title("(b) measured 1/e crossing vs prescribed $L$\n"
                 f"worst relative error {b['worst_length_relative_error']*100:.2f}% "
                 f"(filled: Gaussian field, open: rank/copula)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.suptitle("P295W1-N07 (b) spatial correlation of the strength field", fontsize=11)
    fig.tight_layout()
    path = OUT / "fig2_correlation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_ensemble(study: dict) -> Path:
    c = study["stage_c_ensemble"]
    f = np.asarray(c["force_samples"], dtype=float) / 1e3          # kN
    stats = c["statistics"]
    per_width = np.asarray(c["force_samples"], dtype=float) / c["width_m"] / 1e3  # kN/m
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes[0]
    ax.hist(per_width, bins=12, color="#4c72b0", edgecolor="k", alpha=0.8, density=True)
    ax.axvline(c["q05_bootstrap"]["estimate"] / c["width_m"] / 1e3, color="#9c2b2b",
               lw=2.0, label=f"$F_{{5\\%}}/B$ = {c['q05_bootstrap']['estimate']/c['width_m']/1e3:.2f} kN/m")
    ax.axvline(stats["mean"] / c["width_m"] / 1e3, color="k", lw=2.0,
               label=f"$\\mu_F/B$ = {stats['mean']/c['width_m']/1e3:.2f} kN/m")
    ax.axvline(c["deterministic_per_width"] / 1e3, color="#2ca02c", ls="--", lw=2.0,
               label=f"deterministic = {c['deterministic_per_width']/1e3:.2f} kN/m")
    ax.set_xlabel("breaking force per unit width $F/B$ [kN/m]")
    ax.set_ylabel("probability density")
    ax.set_title(f"(a) N={c['n_ensemble']} realisations, B={c['width_m']:.0f} m, "
                 f"L={c['correlation_length_m']:.0f} m\n"
                 f"$\\sigma_F/\\mu_F$ = {stats['cov']:.3f}, "
                 f"$F_{{5\\%}}/\\mu_F$ = {stats['q05_over_mean']:.3f}")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    running = np.asarray(c["convergence"]["running_mean"]) / c["width_m"] / 1e3
    k = np.arange(1, running.size + 1)
    ax.plot(k, running, "o-", ms=3.5, color="#4c72b0", label="running mean $\\mu_F(k)/B$")
    ax.fill_between(k, running - stats["se"] / c["width_m"] / 1e3,
                    running + stats["se"] / c["width_m"] / 1e3, color="#4c72b0",
                    alpha=0.18, label="final $\\pm 1$ s.e.")
    ax.axhline(c["deterministic_per_width"] / 1e3, color="#2ca02c", ls="--", lw=1.6,
               label="deterministic model")
    rob = c["seed_robustness"]
    ax.set_xlabel("realisation index $k$")
    ax.set_ylabel("$\\mu_F(k)/B$ [kN/m]")
    ax.set_title(f"(b) mean convergence: s.e./mean = {stats['se_relative']*100:.2f}% "
                 f"(limit 5%),\ndrift = {c['convergence']['max_relative_drift']*100:.2f}%; "
                 f"CV inside band for {rob['n_cov_inside_band']}/{rob['n_seeds']} seeds")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.suptitle("P295W1-N07 (c) breaking-force ensemble statistics (weakest-link model)",
                 fontsize=11)
    fig.tight_layout()
    path = OUT / "fig3_ensemble.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_size_effect(study: dict) -> Path:
    d = study["stage_d_size_effect"]
    widths = np.asarray(d["widths_m"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    ax = axes[0]
    styles = {"nested_n50": ("o-", "#1f77b4", "nested realisations, N=50"),
              "nested_n2000": ("s-", "#d62728", "nested realisations, N=2000"),
              "independent_n50": ("^:", "#2ca02c", "independent seeds, N=50"),
              "independent_n2000": ("v:", "#9467bd", "independent seeds, N=2000")}
    for tag, (style, colour, label) in styles.items():
        curve = np.asarray(d["q05_over_width_curve"][tag], dtype=float) / 1e3
        ax.plot(widths, curve, style, color=colour, ms=5, lw=1.8, label=label)
    bundle = np.asarray(d["bundle_contrast"]["q05_over_width_curve"], dtype=float) / 1e3
    ax.plot(widths, bundle, "--", color="0.45", lw=1.8,
            label="bundle model (contrast): rises, opposite sign")
    ax.set_xlabel("contact-line width $B$ [m]")
    ax.set_ylabel("$F_{5\\%}/B$ [kN/m]")
    ax.set_title("(a) size effect of the normalised 5 % breaking force\n"
                 f"weakest-link: exponent n = {d['power_law_q05']['nested_n2000']['exponent']:.3f}"
                 f" vs Weibull anchor 1/k = {d['weibull_1_over_k']:.3f}")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    logw = np.log(widths)
    for tag, (style, colour, label) in styles.items():
        curve = np.asarray(d["q05_over_width_curve"][tag], dtype=float)
        ax.plot(logw, np.log(curve / curve[0]), style, color=colour, ms=4, lw=1.4,
                label=label)
    fit = d["power_law_q05"]["nested_n2000"]
    ax.plot(logw, -fit["exponent"] * (logw - logw[0]), "k--", lw=1.2,
            label=f"power law, n={fit['exponent']:.3f} ($R^2$={fit['r2']:.4f})")
    ax.plot(logw, -(1.0 / d["weibull_1_over_k"]) * (logw - logw[0]), "k:", lw=1.2,
            label=f"Weibull weakest link $1/k$={d['weibull_1_over_k']:.3f}")
    ax.set_xlabel("$\\ln(B/B_0)$")
    ax.set_ylabel("$\\ln[F_{5\\%}(B)/F_{5\\%}(B_0)]$")
    ax.set_title("(b) log-log size effect; the scatter of the independent-seed\n"
                 "N=50 curve is why pointwise monotonicity needs nesting")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.suptitle("P295W1-N07 (c) width (size) effect 10 -> 100 m", fontsize=11)
    fig.tight_layout()
    path = OUT / "fig4_size_effect.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="regenerate study.json first")
    args = parser.parse_args()
    study = _load(args.rerun)
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_qq, fig_correlation, fig_ensemble, fig_size_effect):
        print("wrote", fn(study))
    print("checks:", {k: v["pass"] for k, v in study["checks"].items()})
    print("all_pass:", study["all_pass"])
    return 0 if study["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())