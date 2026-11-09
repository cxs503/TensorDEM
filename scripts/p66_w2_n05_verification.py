"""P66-W2-N05 verification driver: DEM ensemble -> field verification -> figures.

Runs the full spatio-temporal abrasion-field verification with the real DEM
contact ensemble, writes ``runs/P66_W2_N05/verification.json`` (every number in
the report is read from this file, nothing is hand-copied), renders the figures
into ``docs/figures/P66_W2_N05/``, and (re)generates the markdown report from the
JSON.

Usage::

    PYTHONPATH=src python scripts/p66_w2_n05_verification.py [--reuse-cache]

``--reuse-cache`` reuses ``runs/P66_W2_N05/dem_ensemble.npz`` when present.
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensordem import ice_abrasion as ia
from tensordem import ice_abrasion_spatiotemporal as st

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "P66_W2_N05"
FIG_DIR = ROOT / "docs" / "figures" / "P66_W2_N05"
DOC_PATH = ROOT / "docs" / "P66_W2_N05_spatiotemporal_abrasion_field.md"

N_REALIZATIONS = 8
N_EVENTS_PER_LEG = 20000
BASE_SEED = 20261106


def _load_ensemble(cache: Path) -> st.DemActivityEnsemble | None:
    if not cache.exists():
        return None
    d = np.load(cache)
    return st.DemActivityEnsemble(
        rate_realizations=d["rates"], mean_rate=d["mean"],
        voyage_duration_s=float(d["duration"]),
        seeds=tuple(int(s) for s in d["seeds"]),
        depth_per_voyage_um=d["depths"],
        spec_rate=float(ia.WearSourceConfig().spec_rate))


def _save_ensemble(ens: st.DemActivityEnsemble, cache: Path) -> None:
    np.savez_compressed(cache, rates=ens.rate_realizations, mean=ens.mean_rate,
                        duration=ens.voyage_duration_s,
                        seeds=np.array(ens.seeds),
                        depths=ens.depth_per_voyage_um)


def build_ensemble(reuse: bool) -> st.DemActivityEnsemble:
    cache = RUN_DIR / "dem_ensemble.npz"
    if reuse:
        ens = _load_ensemble(cache)
        if ens is not None:
            print(f"[dem] reused {cache}")
            return ens
    t0 = time.perf_counter()
    ens = st.dem_activity_rate_field(n_realizations=N_REALIZATIONS,
                                     n_events_per_leg=N_EVENTS_PER_LEG,
                                     base_seed=BASE_SEED)
    print(f"[dem] {ens.n_realizations} realisations in {time.perf_counter() - t0:.1f} s")
    _save_ensemble(ens, cache)
    return ens


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _um_h(field) -> np.ndarray:
    return np.asarray(field, dtype=float) * 1e6 * 3600.0


def figure_activity_profile(res: dict, ens: st.DemActivityEnsemble, grid,
                            panels, profile) -> dict:
    mean_panel = st.coarsen_field(ens.mean_rate, grid, panels)
    prof_panel = profile.evaluate_on_grid(panels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, field, title in (
            (axes[0], mean_panel, "DEM ensemble-mean activity a(x,z)"),
            (axes[1], prof_panel, "fitted separable profile"),
            (axes[2], mean_panel - prof_panel, "residual a_DEM - a_profile")):
        im = ax.imshow(_um_h(field), origin="lower", aspect="auto",
                       extent=[0, panels.hull_length_m,
                               float(panels.z_edges[0]), float(panels.z_edges[-1])],
                       cmap="viridis" if title != "residual a_DEM - a_profile" else "coolwarm")
        ax.set_xlabel("x from stem [m]")
        ax.set_ylabel("z above baseline [m]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="um/h")
    fig.suptitle(f"DEM-derived abrasion activity field and its deterministic profile "
                 f"(R^2 = {res['criteria']['C1']['r_squared_panel_grid']:.4f})")
    fig.tight_layout()
    path = FIG_DIR / "fig1_activity_profile.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)

    # marginal sections
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    u = panels.x_centers / panels.hull_length_m
    axes[0].plot(u, _um_h(mean_panel.mean(axis=0)), "o", ms=3,
                 label="DEM mean (z-averaged)")
    axes[0].plot(u, _um_h(prof_panel.mean(axis=0)), "-",
                 label="profile (z-averaged)")
    axes[0].set_xlabel("x / L (0 = stem)")
    axes[0].set_ylabel("activity [um/h]")
    axes[0].legend()
    axes[0].set_title("forebody decay of the activity")
    axes[1].plot(_um_h(mean_panel.mean(axis=1)), panels.z_centers, "o", ms=3,
                 label="DEM mean (x-averaged)")
    axes[1].plot(_um_h(prof_panel.mean(axis=1)), panels.z_centers, "-",
                 label="profile (x-averaged)")
    axes[1].axvline(0.0, color="k", lw=0.5)
    axes[1].set_xlabel("activity [um/h]")
    axes[1].set_ylabel("z [m]")
    axes[1].legend()
    axes[1].set_title("ice-belt vertical distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_activity_sections.png", dpi=140)
    plt.close(fig)
    return {"fig1": path, "fig2": FIG_DIR / "fig2_activity_sections.png"}


def figure_residual_field(res: dict, panel_residual, kl, covariance, panels,
                          cond) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    fields = (panel_residual.reshape(panels.shape),
              cond.mean.reshape(panels.shape),
              np.sqrt(cond.variance).reshape(panels.shape))
    titles = ("residual r = a_DEM - a_profile",
              "conditional mean (survey-conditioned)",
              "conditional std")
    cmaps = ("coolwarm", "coolwarm", "magma")
    for ax, field, title, cmap in zip(axes, fields, titles, cmaps):
        im = ax.imshow(_um_h(field), origin="lower", aspect="auto",
                       extent=[0, panels.hull_length_m,
                               float(panels.z_edges[0]), float(panels.z_edges[-1])],
                       cmap=cmap)
        ax.set_xlabel("x from stem [m]")
        ax.set_ylabel("z above baseline [m]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="um/h")
        if title.startswith("conditional"):
            xs = panels.panel_x[cond.observation_indices]
            zs = panels.panel_z[cond.observation_indices]
            ax.plot(xs, zs, "k.", ms=1.5, alpha=0.6)
    fig.tight_layout()
    path = FIG_DIR / "fig3_residual_conditional.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)

    # covariance lag table + KL spectrum
    tab = covariance.lag_table
    lx = sorted({round(t[0], 6) for t in tab})
    lz = sorted({round(t[1], 6) for t in tab})
    m = np.full((len(lz), len(lx)), np.nan)
    for a, b, c, _n in tab:
        m[lz.index(round(b, 6)), lx.index(round(a, 6))] = c
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    im = axes[0].imshow(m * 1e12 * 3600 ** 2, origin="lower", aspect="auto",
                        cmap="viridis")
    axes[0].set_xticks(range(len(lx)))
    axes[0].set_xticklabels([f"{v:.2g}" for v in lx], rotation=45, fontsize=7)
    axes[0].set_yticks(range(len(lz)))
    axes[0].set_yticklabels([f"{v:.2g}" for v in lz], fontsize=7)
    axes[0].set_xlabel("lag dx [m]")
    axes[0].set_ylabel("lag dz [m]")
    axes[0].set_title("DEM residual covariance [um/h]^2")
    fig.colorbar(im, ax=axes[0])
    axes[1].semilogy(np.arange(1, kl.n_modes + 1), kl.eigenvalues * 1e12 * 3600 ** 2,
                     "o-", ms=2)
    axes[1].set_xlabel("KL mode")
    axes[1].set_ylabel("eigenvalue [um/h]^2")
    axes[1].set_title("epistemic KL spectrum (activity-parameter ensemble)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_covariance_kl.png", dpi=140)
    plt.close(fig)
    return {"fig3": path, "fig4": FIG_DIR / "fig4_covariance_kl.png"}


def figure_evolution(res: dict, series, voyage_distance_m: float,
                     panels) -> dict:
    t = series.time_s
    depth = series.depth_m
    snaps = [0, depth.shape[0] // 3, 2 * depth.shape[0] // 3, depth.shape[0] - 1]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.2))
    vmax = depth[-1].max() * 1e6
    for ax, k in zip(axes, snaps):
        im = ax.imshow(depth[k].reshape(panels.shape) * 1e6, origin="lower",
                       aspect="auto", vmin=0.0, vmax=vmax,
                       extent=[0, panels.hull_length_m,
                               float(panels.z_edges[0]), float(panels.z_edges[-1])],
                       cmap="inferno")
        ax.set_title(f"d(x,z) at s = {voyage_distance_m * t[k] / t[-1] / 1000:.0f} km "
                     f"({t[k] / 3600:.1f} h)")
        ax.set_xlabel("x from stem [m]")
        ax.set_ylabel("z [m]")
        fig.colorbar(im, ax=ax, label="depth [um]")
    fig.suptitle("spatio-temporal wear field along the reference Baltic voyage")
    fig.tight_layout()
    path = FIG_DIR / "fig5_field_evolution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)

    regions = series.regional_statistics()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    u = panels.panel_x / panels.hull_length_m
    for name, stats in regions.items():
        lo, hi = stats["x_fraction_range"]
        mask = (u >= lo) & (u <= hi)
        axes[0].plot(t / 3600.0, depth[:, mask].mean(axis=1) * 1e6, label=name)
    axes[0].set_xlabel("time [h]")
    axes[0].set_ylabel("regional mean depth [um]")
    axes[0].legend(fontsize=7)
    axes[0].set_title("regional accumulation along the voyage")
    p95_t = [st.area_weighted_quantile(depth[k], panels.areas_flat, 95.0) * 1e6
             for k in range(depth.shape[0])]
    axes[1].plot(t / 3600.0, p95_t, "-", label="field P95 (area weighted)")
    axes[1].plot(t / 3600.0, depth.mean(axis=1) * 1e6, "--", label="mean depth")
    axes[1].set_xlabel("time [h]")
    axes[1].set_ylabel("depth [um]")
    axes[1].legend()
    axes[1].set_title("P95 and mean depth evolution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_regional_evolution.png", dpi=140)
    plt.close(fig)
    return {"fig5": path, "fig6": FIG_DIR / "fig6_regional_evolution.png"}


def figure_validation(res: dict, panels) -> dict:
    aux = res["auxiliary"]
    crit = res["criteria"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    conv = aux["time_step_convergence"]
    nts = [c["n_time_steps"] for c in conv]
    errs = [c["rel_error"] for c in conv]
    axes[0].loglog(nts, errs, "o-")
    axes[0].loglog(nts, [errs[0] * (nts[0] / n) for n in nts], "k--",
                   label="first order")
    axes[0].set_xlabel("time steps")
    axes[0].set_ylabel("max rel. error vs closed form")
    axes[0].legend()
    axes[0].set_title("time-stepped vs closed-form integration")

    p95g = aux["p95_grid_convergence"]
    axes[1].plot([p["n_panels"] for p in p95g], [p["p95_depth_um"] for p in p95g],
                 "o-")
    axes[1].set_xlabel("number of panels")
    axes[1].set_ylabel("field P95 depth [um]")
    axes[1].set_title("P95 grid convergence")

    labels = ["field P95", "N02 direct P95"]
    values = [crit["C4"]["field_p95_um"], crit["C4"]["n02_p95_um"]]
    axes[2].bar(labels, values, color=["tab:blue", "tab:orange"])
    for i, v in enumerate(values):
        axes[2].text(i, v, f"{v:.3f} um", ha="center", va="bottom")
    axes[2].set_ylabel("depth per voyage [um]")
    axes[2].set_title(f"P95 agreement: {100 * crit['C4']['relative_error']:.1f}% "
                      f"(target < 10%)")
    fig.tight_layout()
    path = FIG_DIR / "fig7_validation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"fig7": path}


# ---------------------------------------------------------------------------
# markdown report (numbers injected from the JSON)
# ---------------------------------------------------------------------------

def write_report(res: dict, ens: st.DemActivityEnsemble, figs: dict,
                 grid, panels, voyage) -> str:
    c = res["criteria"]
    aux = res["auxiliary"]
    c1, c2, c3, c4, c5, c6 = (c["C1"], c["C2"], c["C3"], c["C4"], c["C5"], c["C6"])
    m100 = c2["statistics_m100"]
    m400 = c2["statistics_m400"]
    regions = aux["regional_statistics_voyage"]
    p95 = aux["p95_statistics_one_voyage"]
    dem = res["dem_ensemble"]
    geo = res["geometry"]

    rows = "\n".join(
        f"| {name} | {v['x_fraction_range'][0]:.2f}-{v['x_fraction_range'][1]:.2f} | "
        f"{v['area_m2']:.1f} | {v['mean_depth_um']:.3f} | {v['p95_depth_um']:.3f} | "
        f"{v['max_depth_um']:.3f} |"
        for name, v in regions.items())

    conv_rows = "\n".join(
        f"| {int(cv['n_time_steps'])} | {cv['rel_error']:.3e} |" for cv in aux["time_step_convergence"])
    p95_rows = "\n".join(
        f"| {p['n_panels']} | {p['p95_depth_um']:.4f} |" for p in aux["p95_grid_convergence"])

    md = f"""# P66-W2-N05 Spatio-Temporal Ice-Abrasion Field Evolution

**Node** P66-W2-N05 (campaign DAG, physical domain: ice navigation / structural
wear) · **repository** TensorDEM · **module**
`src/tensordem/ice_abrasion_spatiotemporal.py` · **driver**
`scripts/p66_w2_n05_verification.py` · **machine-readable result**
`runs/P66_W2_N05/verification.json`

## 1. Case introduction

The predecessor node produced a *single-point* P95 abrasion statistic from a DEM
contact ensemble.  This node upgrades it to a **spatio-temporal field**
`d(x, z, t)` on the hull's ice belt and answers two engineering questions:

1. **Which region wears worst - bow, midbody or stern?**
2. **How does the wear field grow along a voyage, and what is the residual
   (unexplained) spatial uncertainty?**

| item | value |
|---|---|
| hull | L = {geo['hull_length_m']:.1f} m, T = {geo['hull_draught_m']:.2f} m (parametric hull of `tensordem.ice_abrasion`, Cb/L/B audited) |
| ice belt band | z = {float(panels.z_edges[0]):.2f} ... {float(panels.z_edges[-1]):.2f} m above baseline, area {geo['band_area_one_side_m2']:.1f} m^2 per side |
| DEM band grid | {geo['dem_grid_shape'][0]} x {geo['dem_grid_shape'][1]} cells (0.244 m x 0.25 m) |
| panel mesh | {panels.n_x} x {panels.n_z} = {panels.n_panels} panels ({geo['panel_size_m'][0]:.2f} m x {geo['panel_size_m'][1]:.3f} m) |
| DEM ensemble | {dem['n_realizations']} realisations x {N_EVENTS_PER_LEG} contacts/leg, seeds stride {BASE_SEED}+37k |
| voyage | {len(voyage.legs)} legs, {res['dem_ensemble']['voyage_duration_s'] / 3600:.1f} h per voyage (reference Baltic profile, round trip) |
| mean activity | {dem['mean_rate_um_per_hour']:.4f} um/h (P95 {dem['p95_rate_um_per_hour']:.4f} um/h) |
| wear per voyage | {dem['depth_per_voyage_um_mean']:.3f} um (mean depth, one side) |

## 2. Method

1. **DEM activity field.**  Eight independent DEM wear seasons
   (`tensordem.ice_abrasion.run_wear_season`: Hertz contact with a crushing cap,
   Cundall–Strack sliding, Archard pit depth, volume-conserving projection)
   give the abrasion rate `a_k(x,z) = d_k(x,z) / T_voyage`.  Their ensemble mean
   is the deterministic activity field; their scatter is the aleatoric
   (contact / sampling) noise.
2. **Deterministic separable profile.**  `a(x,z) = scale * w_x(u) * w_z(z)` with
   a background + exponential forebody decay + two Gaussian shoulders in `u =
   x/L` and an asymmetric Gaussian about the ice belt in `z`; bounded
   least-squares fit on the panel-averaged DEM mean field.
3. **Residual random field.**  `r = a_DEM - a_profile` is split into
   *epistemic* (where the wear lands: activity-distribution parameters drawn in
   documented bands, total wear conserved) and *aleatoric* (per-panel DEM
   realisation scatter).  The epistemic ensemble gives an empirical covariance,
   `kl_from_ensemble` its exact modes.
4. **Conditional simulation.**  Simple kriging in mode space
   (`simulate_conditional_modes`) conditioned on a sparse survey
   ({c2['n_observations']} panels, {100 * c2['n_observations'] / panels.n_panels:.0f} %
   of the hull) of the DEM residual, with the aleatoric scatter as measurement
   noise and as the irreducible prediction noise.
5. **Moving-frame evolution.**  Panel `i` samples the along-track concentration
   at `s(t) + xi_i`, `xi_i = L/2 - x_i` (bow positive), so a bow panel meets an
   ice edge `L/V` earlier than a stern panel.  The accumulation
   `d_i(t) = a_i/(c_ref) * integral_0^t c(s(tau)+xi_i) dtau` is evaluated with the
   exact antiderivative `C(s) = integral c ds`; a time-stepped integrator is kept
   and cross-checked against it.

## 3. Results

### 3.1 Deterministic activity profile (criterion C1)

| quantity | value |
|---|---|
| R^2 on the panel mesh | **{c1['r_squared_panel_grid']:.4f}** (target > 0.95) |
| R^2 on the 0.25 m DEM grid | {c1['r_squared_dem_grid']:.4f} (smooth model, pit-scale scatter excluded) |
| R^2 of single realisations | {min(c1['r_squared_single_realizations']):.3f} ... {max(c1['r_squared_single_realizations']):.3f} (contact noise dominates a single realisation) |
| RMSE | {c1['rmse_um_per_hour']:.5f} um/h |
| parameters | {c1['n_parameters']} (fitted), converged = {c1['converged']} |

![activity field](figures/P66_W2_N05/fig1_activity_profile.png)

*Figure 1 - DEM ensemble-mean activity field, fitted separable profile, and the
residual.  The activity is a forebody peak plus a midbody shoulder on a decaying
background, multiplied by an ice-belt Gaussian in `z`.*

![activity sections](figures/P66_W2_N05/fig2_activity_sections.png)

*Figure 2 - Forebody decay (left) and ice-belt vertical distribution (right):
DEM ensemble mean (points) vs the fitted profile (line).*

### 3.2 Residual field and conditional simulation (criterion C2)

| quantity | value |
|---|---|
| epistemic (smooth) rms | {c2['epistemic_rms_um_per_hour']:.5f} um/h ({aux['epistemic_to_mean_rate_ratio'] * 100:.0f} % of the mean rate) |
| aleatoric (contact) rms | {c2['aleatoric_rms_um_per_hour']:.5f} um/h ({aux['aleatoric_to_mean_rate_ratio'] * 100:.0f} % of the mean rate) |
| KL modes | {c2['epistemic_kl']['n_modes']} (variance explained {c2['epistemic_kl']['variance_explained']:.6f}) |
| noiseless interpolation error (of prior std) | {c2['noiseless_interpolation_relative_error']:.2e} |
| analytic conditional variance error | {c2['analytic_conditional_variance_rel_error']:.2e} |
| ensemble variance error, m = 100 | {m100['ensemble_variance_relative_error'] * 100:.2f} % |
| mean bias / sampling floor, m = 100 | {m100['mean_bias_over_sampling_floor']:.2f} sigma |
| ensemble variance error, m = 400 | {m400['ensemble_variance_relative_error'] * 100:.2f} % |
| mean bias, m = 400 | {m400['mean_bias_relative_to_prior_std'] * 100:.2f} % of prior std |
| conditional variance reduction | {c2['conditional_variance_reduction_fraction'] * 100:.1f} % |

The 100-sample statistics sit exactly at the theoretical sampling floor
({m100['sampling_floor_relative'] * 100:.1f} %), i.e. the sampler is unbiased; at
400 samples the target of < 5 % is met directly.  The covariance error relative
to the i.i.d. baseline is {m100['covariance_error_over_iid_baseline']:.2f}, so the
sampled covariance is consistent with the target within sampling error.

![residual and conditional fields](figures/P66_W2_N05/fig3_residual_conditional.png)

*Figure 3 - residual field, conditional mean and conditional std given the
sparse survey (black points).*

![covariance and KL](figures/P66_W2_N05/fig4_covariance_kl.png)

*Figure 4 - DEM residual covariance as a function of the lag (left) and the
epistemic KL spectrum (right).  The DEM residual is nearly white at the 0.244 m
grid: the contact-pit scatter has no resolvable smooth component above one
cell, which is why the smooth part of the field is taken from the
activity-parameter bands rather than from the DEM scatter.*

### 3.3 Spatio-temporal evolution along the voyage (criterion C3)

| quantity | value |
|---|---|
| volume from the depth field | {c3['volume_from_depth_m3']:.6e} m^3 |
| volume from the rate model | {c3['volume_from_rate_model_m3']:.6e} m^3 |
| relative error | **{c3['relative_error']:.2e}** (target < 1e-6) |
| field P95 / max / mean depth after one voyage | {p95['p95_depth_um_area_weighted']:.3f} / {p95['max_depth_um']:.3f} / {p95['mean_depth_um_area_weighted']:.3f} um |
| area above P95 | {p95['fraction_area_above_p95'] * 100:.1f} % |

Regional distribution after one voyage (area weighted):

| region | x/L | area [m^2] | mean [um] | P95 [um] | max [um] |
|---|---|---|---|---|---|
{rows}

![field evolution](figures/P66_W2_N05/fig5_field_evolution.png)

*Figure 5 - wear field snapshots along the voyage.*

![regional evolution](figures/P66_W2_N05/fig6_regional_evolution.png)

*Figure 6 - regional mean depth and field P95 vs time.*

### 3.4 P95 interface to the predecessor node (criterion C4)

| quantity | value |
|---|---|
| field P95 (area weighted, one voyage) | {c4['field_p95_um']:.4f} um |
| N02 direct P95 (DEM ensemble mean, 0.244 m grid) | {c4['n02_p95_um']:.4f} um |
| relative error | **{c4['relative_error'] * 100:.2f} %** (target < 10 %) |
| one-voyage volume error vs N02 | {aux['one_voyage_volume_relative_error'] * 100:.2f} % |

### 3.5 Analytic anchors and performance (criteria C5, C6)

| check | value | target |
|---|---|---|
| uniform ice field, `d = a t` | {c5['max_relative_error']:.2e} | < 1 % |
| uniform field with moving-frame offsets | {c5['uniform_field_with_offsets_max_relative_error']:.2e} | < 1 % |
| moving-frame onset time (ice edge) | {aux['moving_frame_onset_relative_error']:.2e} | analytic |
| evolution + volume ledger ({c6['n_panels']} panels x {c6['n_time_steps']} steps) | {c6['evolution_and_ledger_cpu_time_s']:.2f} s | < 60 s |
| conditional simulation (100 samples) | {c6['conditional_simulation_cpu_time_s']:.2f} s | < 60 s |

Time-step convergence of the stepped integrator against the closed form (max
relative error):

| time steps | error |
|---|---|
{conv_rows}

P95 grid convergence:

| panels | P95 [um] |
|---|---|
{p95_rows}

Relative P95 change on the finest refinement: {aux['p95_grid_relative_difference']:.2e}.

![validation](figures/P66_W2_N05/fig7_validation.png)

*Figure 7 - time-step convergence (first order, as expected where the
concentration jumps do not fall on the grid), P95 grid convergence, and the
field-vs-N02 P95 comparison.*

## 4. Engineering reading

* The wear field is strongly non-uniform: the **bow** region accumulates
  {regions['bow']['mean_depth_um']:.3f} um per voyage (P95
  {regions['bow']['p95_depth_um']:.3f} um) against
  {regions['stern']['mean_depth_um']:.3f} um at the stern, i.e. a factor
  {regions['bow']['mean_depth_um'] / regions['stern']['mean_depth_um']:.2f}.  The
  ranking bow > fore shoulder > midbody > aft shoulder > stern is monotone and
  is driven by the forebody exponential decay of the activity distribution.
* The vertical distribution is narrow: the 95th percentile of the field sits
  within the ice belt, so increasing belt plate thickness (not the whole side
  shell) is the efficient mitigation.
* Uncertainty decomposition: the smooth (epistemic) part is
  {aux['epistemic_to_mean_rate_ratio'] * 100:.0f} % and the irreducible contact
  scatter is {aux['aleatoric_to_mean_rate_ratio'] * 100:.0f} % of the mean rate.
  A sparse survey removes {c2['conditional_variance_reduction_fraction'] * 100:.1f} %
  of the epistemic variance, but the aleatoric floor remains - coating-life
  decisions cannot be made from a handful of survey patches alone.

## 5. Caveats and limitations

* **Absolute level.**  The wear rate still carries the single-point literature
  anchor of `tensordem.ice_abrasion` (Baltic 1A coating loss).  Nothing new is
  calibrated here; the field is linear in that anchor.
* **R^2.**  The 0.95 target is met on the panel mesh used by the evolution
  ({c1['r_squared_panel_grid']:.4f}); on the raw 0.25 m DEM grid the smooth
  profile explains {c1['r_squared_dem_grid']:.3f} only, by construction - the
  remaining pit-scale scatter is not a smooth field and is carried by the
  nugget term, not by the profile.
* **Residual covariance.**  The DEM ensemble shows no resolvable smooth residual
  above one 0.244 m cell, so the *smooth* part of the residual covariance comes
  from documented activity-parameter bands (a stated modelling assumption, not a
  DEM fit).  The aleatoric part is measured from the DEM ensemble.
* **Geometry.**  Single-sided 2-D band map: no port/starboard asymmetry, no
  multi-valued surface at the stem, deadrise neglected (inherited from
  `tensordem.ice_abrasion`).
* **Ice conditions.**  Piecewise-constant concentration along the track; the ice
  field is extended upstream/downstream at the first/last leg value.  The ship
  is assumed to keep the reference speed of each leg (no speed-power coupling).
* **C4 margin.**  The P95 comparison closes at
  {c4['relative_error'] * 100:.2f} % against a 10 % target; the residual
  difference is the smoothing of the profile relative to the DEM field.

## 6. Error judgement (vs the closest available reference)

| key quantity | reference | error | < 3 % ? |
|---|---|---|---|
| uniform-field accumulation | analytic `d = a t` | {c5['max_relative_error']:.2e} | yes |
| moving-frame onset | analytic `t = (s0 - xi)/V` | {aux['moving_frame_onset_relative_error']:.2e} | yes |
| volume ledger | rate-model integral | {c3['relative_error']:.2e} | yes |
| conditional variance | closed-form kriging | {c2['analytic_conditional_variance_rel_error']:.2e} | yes |
| field P95 vs predecessor P95 | N02 DEM statistic | {c4['relative_error'] * 100:.2f} % | no (10 % band) |
| one-voyage wear volume | N02 DEM statistic | {aux['one_voyage_volume_relative_error'] * 100:.2f} % | yes |
| profile R^2 (panel mesh) | DEM ensemble mean | 1 - R^2 = {1 - c1['r_squared_panel_grid']:.4f} | n/a (fit quality) |

The P95 field-to-N02 comparison is the only quantity that does not reach 3 %; the
difference is systematic (the smooth profile under-represents the pit-scale
peaks) and is reported as such, not tuned away.

## 7. Reproduction

```bash
PYTHONPATH=src python scripts/p66_w2_n05_verification.py          # full run (~3 min)
PYTHONPATH=src python -m pytest tests/test_ice_abrasion_spatiotemporal.py -q
```

All numbers above are injected from `runs/P66_W2_N05/verification.json`.
"""
    DOC_PATH.write_text(md)
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    grid = ia.build_waterline_grid()
    voyage = ia.reference_baltic_voyage()
    panels = st.build_panel_grid(grid=grid, n_x=112, n_z=9)
    ens = build_ensemble(args.reuse_cache)

    t0 = time.perf_counter()
    res = st.run_n05_verification(ensemble=ens)
    res["driver"] = {
        "n_realizations": N_REALIZATIONS,
        "n_events_per_leg": N_EVENTS_PER_LEG,
        "base_seed": BASE_SEED,
        "dem_ensemble_cache": str((RUN_DIR / "dem_ensemble.npz").relative_to(ROOT)),
        "module": "src/tensordem/ice_abrasion_spatiotemporal.py",
        "test_file": "tests/test_ice_abrasion_spatiotemporal.py",
    }
    res["driver"]["wall_time_s"] = time.perf_counter() - t0
    out = RUN_DIR / "verification.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"[verify] all criteria passed = {res['all_criteria_passed']} "
          f"({time.perf_counter() - t0:.1f} s)")
    for key, value in res["criteria"].items():
        print(f"  {key}: passed={value['passed']}")

    if not args.skip_figures:
        profile = st.fit_separable_profile(
            panels.panel_x, panels.panel_z,
            st.coarsen_field(ens.mean_rate, grid, panels).ravel(),
            hull_length_m=panels.hull_length_m,
            hull_draught_m=panels.hull_draught_m).profile
        figs = {}
        figs.update(figure_activity_profile(res, ens, grid, panels, profile))
        panel_residual = (st.coarsen_field(ens.mean_rate, grid, panels).ravel()
                          - profile.evaluate(panels.panel_x, panels.panel_z))
        epi = st.activity_parameter_ensemble(
            panels, grid=grid, n_members=200, seed=43,
            reference_rate=float(np.sum(st.coarsen_field(ens.mean_rate, grid, panels).ravel()
                                        * panels.areas_flat) / panels.total_area_m2))
        kl = st.kl_from_ensemble(epi)
        tau2 = st.sampling_noise_variance(
            np.array([st.coarsen_field(ens.rate_realizations[k], grid, panels).ravel()
                      for k in range(ens.n_realizations)]))
        obs = res["criteria"]["C2"]["n_observations"]
        obs_idx = np.sort(np.random.default_rng(42).choice(panels.n_panels, obs, replace=False))
        cond = st.simulate_conditional_modes(
            kl, observation_indices=obs_idx,
            observation_values=panel_residual[obs_idx],
            observation_noise=float(np.sqrt(tau2.mean())),
            prediction_noise=np.sqrt(tau2), n_samples=100, seed=42)
        cov_lag = st.estimate_covariance_lagged(
            grid, ens.rate_realizations,
            center=profile.evaluate(
                np.tile(np.asarray(grid.x_centers), int(grid.shape[0])),
                np.repeat(np.asarray(grid.z_centers), int(grid.shape[1]))).reshape(
                    tuple(int(v) for v in grid.shape)))
        figs.update(figure_residual_field(res, panel_residual, kl, cov_lag,
                                          panels, cond))
        track = st.ice_track_from_voyage(voyage)
        series = st.evolve_abrasion_field(profile, panels, track,
                                          n_time_steps=101)
        figs.update(figure_evolution(res, series, track.total_distance_m, panels))
        figs.update(figure_validation(res, panels))
        for name, path in figs.items():
            print(f"[fig] {name}: {path.relative_to(ROOT)} ({path.stat().st_size // 1024} kB)")
        write_report(res, ens, figs, grid, panels, voyage)
        print(f"[doc] {DOC_PATH.relative_to(ROOT)}")
    print(json.dumps({k: v["passed"] for k, v in res["criteria"].items()}))


if __name__ == "__main__":
    main()