"""Figures for the ice-abrasion extreme-statistic certification.

Reads ``artifacts/extreme_stats/certification.json`` (produced by
``scripts/extreme_stats_certification.py``) and the stored depth fields, and
writes the paper figures plus ``figure_numbers.json`` with every number that
appears on a figure (no hand-copied values in the report).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tensordem.extreme_statistics as es

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "extreme_stats"
FIG = OUT / "figs"


def load_field(cell, rep=0, case=None):
    name = case or ("c%g" % cell)
    data = np.load(OUT / "fields" / ("field_%s_r%02d.npz" % (name, rep)))
    return data["depth"].reshape(tuple(int(v) for v in data["shape"]))


def fig_profile_intervals() -> dict:
    """Profile-likelihood interval of a GEV tail quantile (asymmetric)."""
    rng = np.random.default_rng(4242)
    parent = es.GEVParams(1.0, 1.0, 0.25)
    sample = es.gev_ppf(parent, rng.random(400))
    ci = es.gev_profile_ci(sample, 0.95)
    dci = es.delta_method_ci(sample, 0.95)
    truth = parent.quantile(0.95)
    q_grid = np.linspace(ci["ci_lo"] * 0.9, ci["ci_hi"] * 1.1, 41)
    dev = []
    for q in q_grid:
        ll = es.gev_loglik(es.GEVParams(q - es.gev_fit(sample).sigma
                                        * es._quantile_term(es.gev_fit(sample).xi, 0.95),
                                        es.gev_fit(sample).sigma, es.gev_fit(sample).xi), sample)
        dev.append(max(2.0 * (es.gev_loglik(es.gev_fit(sample), sample) - ll), 0.0))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    xs = np.sort(sample)
    axes[0].plot(xs, np.linspace(0, 1, xs.size), lw=1.2, color="0.2",
                 label="empirical CDF (n=400)")
    grid = np.linspace(xs.min(), xs.max(), 400)
    axes[0].plot(grid, parent.cdf(grid), "--", color="tab:red", lw=1.2,
                 label="parent GEV (xi=0.25)")
    axes[0].axvline(truth, color="tab:red", lw=0.8, ls=":")
    axes[0].axvspan(ci["ci_lo"], ci["ci_hi"], color="tab:blue", alpha=0.12,
                    label="profile CI of P95")
    axes[0].axvline(ci["estimate"], color="tab:blue", lw=1.0, label="P95 MLE")
    axes[0].set_xlabel("depth (arb. units)")
    axes[0].set_ylabel("F(x)")
    axes[0].set_title("GEV tail quantile: estimate and profile interval")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].plot(q_grid, dev, lw=1.4, color="tab:blue", label="profile deviance")
    axes[1].axhline(es._chi2_cut(0.95), color="tab:red", ls="--", lw=1.0,
                    label="chi2(1) 95 % cut")
    axes[1].axvline(ci["estimate"], color="0.2", lw=0.8)
    axes[1].axvline(ci["ci_lo"], color="tab:blue", ls=":", lw=0.9)
    axes[1].axvline(ci["ci_hi"], color="tab:blue", ls=":", lw=0.9)
    axes[1].set_xlabel("trial P95 value")
    axes[1].set_ylabel("2 (l_max - l_profile)")
    axes[1].set_title("Asymmetric interval from the likelihood ratio")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_profile_likelihood_ci.png")
    plt.close(fig)
    return {"parent_p95": truth, "profile": ci, "delta": dci,
            "profile_covers_truth": bool(ci["ci_lo"] <= truth <= ci["ci_hi"])}


def fig_coverage(cert: dict) -> dict:
    keys = [("coverage_gumbel", "Gumbel (n=200)"),
            ("coverage_frechet", "Frechet xi=0.25 (n=400)"),
            ("coverage_weibull", "Weibull xi=-0.25 (n=400)")]
    methods = ["profile", "delta", "order"]
    labels = {"profile": "profile likelihood", "delta": "delta method",
              "order": "order statistic"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    width = 0.25
    x = np.arange(len(keys))
    out = {}
    for i, m in enumerate(methods):
        vals = [cert["synthetic"][k]["methods"][m]["coverage"] for k, _ in keys]
        ax.bar(x + (i - 1) * width, vals, width, label=labels[m])
        out[m] = vals
    ax.axhline(cert["synthetic"]["coverage_gumbel"]["level"], color="k", ls="--",
               lw=1.0, label="nominal 95 %")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in keys], fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("empirical coverage")
    ax.set_title("Coverage of 95 % intervals vs Monte Carlo (profile / delta / order)")
    for k, _ in keys:
        ax.annotate("n_rep=%d" % cert["synthetic"][k]["n_rep"], (0, 0.53), fontsize=7,
                    ha="left", alpha=0.0)
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_interval_coverage.png")
    plt.close(fig)
    return {"coverage_by_method": out}


def fig_theta(cert: dict) -> dict:
    rows = cert["synthetic"]["theta_recovery"]["rows"]
    runs = cert["synthetic"]["theta_recovery"]["run_lengths"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    out = {}
    for r in rows:
        ax.plot(runs, r["theta_runs_vs_runlength"], "o-", ms=4, lw=1.2,
                label="a=%.1f (theta_true=%.2f)" % (r["a"], r["theta_true"]))
        ax.axhline(r["theta_true"], color="0.6", lw=0.7, ls=":")
        out["a=%.1f" % r["a"]] = {"theta_true": r["theta_true"],
                                  "theta_plateau": r["theta_plateau"],
                                  "runs": r["theta_runs_vs_runlength"],
                                  "block_estimator": r["block_estimator"]}
    ax.set_xlabel("declustering run length r")
    ax.set_ylabel("extremal index theta")
    ax.set_title("Extremal-index recovery (max-AR(1) process) and block-length stability")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_extremal_index_stability.png")
    plt.close(fig)
    return {"theta": out, "run_lengths": runs}


def fig_resolution(cert: dict) -> dict:
    fld = cert["field"]
    per = fld["certificate"]["per_level"]
    levels = fld["certificate"]["levels_m"]
    xs, ys, yerr = [], [], []
    for h in levels:
        lvl = per["%.4g" % h]
        xs.append(h)
        ys.append(lvl["p95_mean_m"])
        yerr.append(0.5 * (lvl["p95_ci_normal_m"][1] - lvl["p95_ci_normal_m"][0]))
    model = fld["resolution_model"]
    iso_rows = [r for r in model["rows"] if r["case"] == "iso"]
    cont = model["continuum_limit"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=150)
    ax = axes[0]
    ax.errorbar(xs, ys, yerr=yerr, fmt="o", ms=6, color="tab:blue", capsize=3,
                label="DEM field P95 (mean of replicates, 95 % CI)")
    ax.plot([r["dx_m"] for r in iso_rows], [r["model_p95_m"] for r in iso_rows],
            "s--", ms=5, color="tab:red",
            label="two-mechanism model (1 fitted hit rate)")
    ax.plot([r["dx_m"] for r in cont], [r["model_p95_m"] for r in cont], ":",
            color="tab:red", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("cell size h (m)")
    ax.set_ylabel("P95 wear depth (m/year)")
    ax.set_title("P95 is a cell-size-of-record quantity (log-log)")
    ax.annotate("GCI(0.25 vs 1.0 m) = %.1f %%\nsampling CI = +-%.2f %%"
                % (fld["gci_w10n06"]["single_realisation"]["gci_fine_pct"],
                   per["%.4g" % levels[0]]["p95_ci_rel_halfwidth_pct"]),
                (xs[0], ys[-1]), fontsize=8, ha="left", va="bottom")
    ax.legend(fontsize=7)
    ax = axes[1]
    names = [r["case"].replace("aniso_", "") for r in model["rows"]]
    meas = [r["measured_p95_m"] for r in model["rows"]]
    mod = [r["model_p95_m"] for r in model["rows"]]
    idx = np.arange(len(names))
    ax.bar(idx - 0.2, meas, 0.4, label="measured")
    ax.bar(idx + 0.2, mod, 0.4, label="model")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("P95 wear depth (m/year)")
    ax.set_title("Anisotropic refinement reproduces the x/z split")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_resolution_law.png")
    plt.close(fig)
    return {"levels_m": levels, "measured_p95_m": ys, "ci_halfwidth_m": yerr,
            "model_iso_m": [r["model_p95_m"] for r in iso_rows],
            "model_rows": model["rows"], "continuum": cont,
            "gci_pct": fld["gci_w10n06"]["single_realisation"]["gci_fine_pct"]}


def fig_field_maps() -> dict:
    field = load_field(0.25)
    nz, nx = field.shape
    coarse = field.reshape(nz // 3 * 1, -1) if False else None
    # coarsen x by 4 (185 cells) keeping z resolution for the comparison panel
    coarse = field[:, : nx // 4 * 4].reshape(nz, nx // 4, 4).mean(axis=2)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), dpi=150, sharex=False)
    x = np.linspace(0, 185, nx)
    im0 = axes[0].imshow(field * 1e6, aspect="auto", cmap="inferno",
                         extent=[0, 185, nz, 0])
    axes[0].set_title("depth field, dx = dz = 0.25 m (P95 = %.1f um/yr)"
                      % (np.quantile(field, 0.95) * 1e6))
    axes[0].set_ylabel("band row (waterline band)")
    fig.colorbar(im0, ax=axes[0], label="depth (um/yr)")
    im1 = axes[1].imshow(coarse * 1e6, aspect="auto", cmap="inferno",
                         extent=[0, 185, nz, 0])
    axes[1].set_title("same field, along-hull cell 1.0 m (P95 = %.1f um/yr)"
                      % (np.quantile(coarse, 0.95) * 1e6))
    axes[1].set_xlabel("x along hull (m)")
    axes[1].set_ylabel("band row (waterline band)")
    fig.colorbar(im1, ax=axes[1], label="depth (um/yr)")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_depth_field_resolution.png")
    plt.close(fig)
    return {"p95_fine_um": float(np.quantile(field, 0.95) * 1e6),
            "p95_coarse_x_um": float(np.quantile(coarse, 0.95) * 1e6),
            "shape": [nz, nx],
            "max_um": float(field.max() * 1e6)}


def fig_budget(cert: dict) -> dict:
    fld = cert["field"]
    per = fld["certificate"]["per_level"]
    finest = "%.4g" % fld["certificate"]["levels_m"][0]
    half = per[finest]["p95_ci_rel_halfwidth_pct"]
    verdict = fld["verdict"]
    model_misfit = max(abs(r["model_rel_deviation_pct"])
                       for r in fld["resolution_model"]["rows"])
    items = [("sampling CI\n(24 replicates)", half, "tab:green"),
             ("finest-pair\nresolution", abs(verdict["delta_finest_pair_pct"]),
              "tab:orange"),
             ("W10-N06 GCI\n(0.25 vs 1.0 m)",
              abs(fld["gci_w10n06"]["single_realisation"]["gci_fine_pct"]), "tab:red"),
             ("model misfit\n(max, all grids)", model_misfit, "tab:blue")]
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=150)
    xs = np.arange(len(items))
    bars = ax.bar(xs, [v for _, v, _ in items], color=[c for _, _, c in items])
    ax.set_yscale("log")
    ax.set_ylabel("relative difference (%)")
    for b, (name, v, _c) in zip(bars, items):
        ax.annotate("%.2f %%" % v, (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _, _ in items], fontsize=8)
    ax.set_title("Variance budget of P95: sampling error <> resolution measure effect")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_variance_budget.png")
    plt.close(fig)
    return {"sampling_halfwidth_pct": half,
            "finest_pair_pct": abs(verdict["delta_finest_pair_pct"]),
            "gci_pct": abs(fld["gci_w10n06"]["single_realisation"]["gci_fine_pct"]),
            "model_misfit_pct": model_misfit,
            "ratio_gci_to_sampling": abs(fld["gci_w10n06"]["single_realisation"]["gci_fine_pct"]) / half}


def main() -> dict:
    cert = json.loads((OUT / "certification.json").read_text())
    numbers = {"fig1": fig_profile_intervals(), "fig2": fig_coverage(cert),
               "fig3": fig_theta(cert), "fig4": fig_resolution(cert),
               "fig5": fig_field_maps(), "fig6": fig_budget(cert)}
    (OUT / "figure_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers["fig6"], indent=1))
    return numbers


if __name__ == "__main__":
    main()
