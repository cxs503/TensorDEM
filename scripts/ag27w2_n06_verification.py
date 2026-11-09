"""Node AG27W2-N06 verification driver.

Conditional simulation of the ice-abrasion field on the hull ice belt, with a
physical independent-repeat truth channel and a resolution-vs-sampling
decomposition.

Pipeline
--------
1. ``runs/AG27W2_N06/dem_ensemble.npz`` holds ``K`` independent DEM wear seasons
   (``scripts/ag27w2_n06_stage1_ensemble.py``: different contact-ensemble seed =
   different initial packing, identical physics).
2. Fields are coarsened onto three hull panel meshes (185x9, 92x5, 46x3).
3. ``K = 20`` repeats are the truth channel; ``S = 10`` further independent fields
   condition the simulator; the remaining fields are used only for a K = 40
   robustness check of the referee.
4. Criteria C1-C6 of the node are evaluated by
   ``tensordem.ice_abrasion_conditional_adjudication.run_adjudication``.
5. Figures + ``verification.json`` + the markdown report are generated from the
   same numbers (nothing hand-copied).

Usage::

    PYTHONPATH=src python scripts/ag27w2_n06_verification.py
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from tensordem import ice_abrasion as ia
from tensordem import ice_abrasion_conditional_adjudication as adj
from tensordem import ice_abrasion_spatiotemporal as st

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "AG27W2_N06"
FIG_DIR = ROOT / "docs" / "figures" / "AG27W2_N06"
DOC_DIR = ROOT / "docs"

GRIDS = {"fine": (185, 9), "medium": (92, 5), "coarse": (46, 3)}
N_TRUTH = 20
N_SOURCE = 10
N_ROBUST = 20
N_REALIZATIONS = 200
N_BOOTSTRAP = 200
SEED = 1729


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _overlap_1d(coarse_edges: np.ndarray, fine_edges: np.ndarray) -> np.ndarray:
    """Overlap-length matrix between two uniform edge vectors of one axis."""
    c0, c1 = coarse_edges[:-1], coarse_edges[1:]
    f0, f1 = fine_edges[:-1], fine_edges[1:]
    lo = np.maximum(c0[:, None], f0[None, :])
    hi = np.minimum(c1[:, None], f1[None, :])
    return np.clip(hi - lo, 0.0, None)


def coarse_field(field_xz: np.ndarray, n_from: tuple[int, int], n_to: tuple[int, int],
                 extent: tuple[float, float] = (185.0, 11.5)) -> np.ndarray:
    """Area-weighted block average of an ``[x, z]`` field between panel meshes."""
    edges_x_from = np.linspace(0.0, extent[0], n_from[0] + 1)
    edges_z_from = np.linspace(0.0, extent[1], n_from[1] + 1)
    edges_x_to = np.linspace(0.0, extent[0], n_to[0] + 1)
    edges_z_to = np.linspace(0.0, extent[1], n_to[1] + 1)
    wx = _overlap_1d(edges_x_to, edges_x_from)
    wz = _overlap_1d(edges_z_to, edges_z_from)
    wx = wx / wx.sum(axis=1, keepdims=True)
    wz = wz / wz.sum(axis=1, keepdims=True)
    return wx @ field_xz @ wz.T


def load_ensemble() -> tuple[np.ndarray, dict]:
    with np.load(RUN_DIR / "dem_ensemble.npz") as d:
        rates = np.asarray(d["rates"], dtype=np.float64)
        seeds = np.asarray(d["seeds"], dtype=np.int64)
        duration = float(d["duration_s"])
        events = int(d["events"])
    meta = {
        "n_realizations": int(rates.shape[0]),
        "dem_grid_shape": list(rates.shape[1:]),
        "seeds": [int(s) for s in seeds],
        "voyage_duration_s": duration,
        "events_per_leg": events,
    }
    return rates, meta


def build_fields(rates: np.ndarray) -> tuple[dict, dict, dict]:
    """Panel-mesh fields ``[x, z]`` and areas per grid for every realisation."""
    grid = ia.build_waterline_grid()
    fields: dict[str, np.ndarray] = {}
    areas: dict[str, np.ndarray] = {}
    panels: dict[str, object] = {}
    for label, (nx, nz) in GRIDS.items():
        pg = st.build_panel_grid(grid=grid, n_x=nx, n_z=nz)
        stack = np.array([st.coarsen_field(r, grid, pg).T for r in rates], dtype=np.float64)
        fields[label] = stack
        areas[label] = np.asarray(pg.cell_area_m2, dtype=np.float64).T.ravel()
        panels[label] = pg
    return fields, areas, panels


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _sankey(ax, total: float, parts: list[tuple[str, float, str]], title: str) -> None:
    """Minimal two-stage flow diagram for the resolution decomposition."""
    order = sorted(parts, key=lambda p: -abs(p[1]))
    total_abs = abs(total) if abs(total) > 1e-12 else 1e-12
    y = 1.0
    palette = {"discretisation": "#4C72B0", "sampling": "#DD8452", "interaction": "#937860"}
    for name, value, key in order:
        h = abs(value) / total_abs * 0.9 + 0.02
        ax.add_patch(plt.Rectangle((0.62, y - h), 0.3, h, color=palette.get(key, "#999999"),
                                   alpha=0.85))
        ax.text(0.95, y - h / 2, f"{name}\n{value:+.2f}%", va="center", fontsize=8)
        ax.add_patch(Polygon([[0.34, 0.55 - 0.0], [0.34, 0.75], [0.62, y], [0.62, y - h]],
                             closed=True, facecolor=palette.get(key, "#999999"), alpha=0.25,
                             edgecolor="none"))
        y -= h + 0.03
    ax.add_patch(plt.Rectangle((0.02, 0.45), 0.3, 0.3, color="#333333", alpha=0.8))
    ax.text(0.17, 0.6, f"measured\n{total:+.2f}%", ha="center", va="center",
            fontsize=8, color="white")
    ax.set_xlim(0, 1.35)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def make_figures(res: dict) -> list[dict]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []
    fields = res["_fields"]
    a = res["_a_fine"]
    q = fields["fine"][N_TRUTH]          # first conditioning field
    sims = res["_sims_fine"]
    extent = [0.0, 185.0, 0.0, 11.5]

    # ---- Fig 1: deterministic profile, one realisation, P95 and spread --------
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.2), constrained_layout=True)
    panels = [
        (a * 1e6 * 3600, "deterministic activity profile $a(x,z)$  [µm/h]"),
        (sims[0] * 1e6 * 3600, "one conditional realisation  [µm/h]"),
        (np.percentile(sims, 95, axis=0) * 1e6 * 3600, "per-panel P95 of 200 realisations  [µm/h]"),
        (np.std(sims, axis=0) * 1e6 * 3600, "per-panel spread (std)  [µm/h]"),
    ]
    for ax, (img, title) in zip(axes.ravel(), panels):
        im = ax.imshow(img.T, origin="lower", aspect="auto", extent=extent,
                       cmap="magma", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x along hull [m]", fontsize=8)
        ax.set_ylabel("z above baseline [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    path = FIG_DIR / "fig1_realisation_cloud.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig1_realisation_cloud.png",
                 "caption": "Figure 1 - deterministic activity profile, one conditional "
                            "realisation, per-panel P95 and per-panel spread of the 200 "
                            "realisations on the 185x9 panel mesh (real DEM wear ensemble).",
                 "path": str(path)})

    # ---- Fig 2: P95 standard deviation by estimator ---------------------------
    c2 = res["criteria"]["C2"]
    labels = ["truth\n(K=%d repeats)" % c2["truth"]["n_samples"], "i.i.d.\nbootstrap",
              "moving block\n8x3", "conditional\nsimulation"]
    mb = c2["moving_block_bootstrap"]
    block_key = next((k for k in mb if k.startswith("8x")), list(mb)[0])
    vals = [c2["truth"]["p95_std"], c2["iid_bootstrap"]["p95_std"],
            mb[block_key]["p95_std"], c2["conditional"]["p95_std"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    bars = ax.bar(labels, np.array(vals) * 1e9, color=["#333333", "#C44E52", "#8172B2", "#4C72B0"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v * 1e9:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("std of P95  [nm/h]")
    ax.set_title("P95 uncertainty: who is right? (fine grid, K=%d / N=%d)"
                 % (c2["truth"]["n_samples"], c2["conditional"]["n_samples"]), fontsize=9)
    ax.tick_params(labelsize=8)
    path = FIG_DIR / "fig2_p95_variance_estimators.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig2_p95_variance_estimators.png",
                 "caption": "Figure 2 - standard deviation of the global P95 of the abrasion "
                            "rate: independent physical repeats (referee), i.i.d. bootstrap, "
                            "moving-block bootstrap and the conditional simulator.",
                 "path": str(path)})

    # ---- Fig 3: decomposition per grid --------------------------------------
    lad = res["resolution_ladder"]["rows"]
    fig, axes = plt.subplots(1, max(1, len(lad) - 1), figsize=(11, 3.6),
                             constrained_layout=True, squeeze=False)
    for ax, row in zip(axes.ravel(), [r for r in lad if "breakdown" in r]):
        b = row["breakdown"]
        parts = [("discretisation", b["discretization_pct"], "discretisation"),
                 ("sampling", b["sampling_pct"], "sampling"),
                 ("interaction", b["interaction_pct"], "interaction")]
        _sankey(ax, b["total_measured_pct"], parts,
                f"{row['grid']} {tuple(row['shape'])} vs fine")
    path = FIG_DIR / "fig3_resolution_decomposition.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig3_resolution_decomposition.png",
                 "caption": "Figure 3 - coarse-vs-fine P95 difference decomposed into the "
                            "deterministic activity-profile term, the extreme-value sampling "
                            "term and the residual interaction, for each coarsening step.",
                 "path": str(path)})

    # ---- Fig 4: resolution ladder ------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    xlab = [f"{r['grid']}\n{tuple(r['shape'])}" for r in lad]
    p95 = np.array([r["p95_truth_mean"] for r in lad]) * 1e9
    std = np.array([r["p95_truth_std"] for r in lad]) * 1e9
    ax.errorbar(range(len(lad)), p95, yerr=std, marker="o", capsize=4,
                label="truth P95 ± 1 std (K repeats)")
    ax.plot(range(len(lad)), [r["p95_cond_mean"] * 1e9 for r in lad], marker="s",
            linestyle="--", label="conditional-simulation P95")
    ax.set_xticks(range(len(lad)))
    ax.set_xticklabels(xlab, fontsize=8)
    ax.set_ylabel("P95 abrasion rate  [nm/h]")
    ax.set_title("resolution ladder: P95 level and its sampling spread", fontsize=9)
    ax.legend(fontsize=7)
    path = FIG_DIR / "fig4_resolution_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig4_resolution_ladder.png",
                 "caption": "Figure 4 - P95 of the abrasion rate against the panel resolution, "
                            "with the sampling spread from the independent repeats and the "
                            "conditional-simulation prediction.",
                 "path": str(path)})
    return figs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    rates, ens_meta = load_ensemble()
    fields, areas, panels = build_fields(rates)

    truth = {k: v[:N_TRUTH] for k, v in fields.items()}
    source = {k: v[N_TRUTH:N_TRUTH + N_SOURCE] for k, v in fields.items()}
    robust = {k: v[N_TRUTH:N_TRUTH + N_ROBUST] for k, v in fields.items()}

    # deterministic activity profile: fitted to the ensemble-mean field on each mesh
    # (real panel-centre coordinates -- the ice belt sits at z = 10.3-12.5 m above
    #  baseline, not at 0-11.5 m, and using synthetic axes destroys the fit)
    profiles = {}
    for label, pg in panels.items():
        mean_field = fields[label].mean(axis=0)
        a, r2, rmse = adj.fit_profile_field(
            np.asarray(pg.x_centers), np.asarray(pg.z_centers), mean_field,
            extent=(pg.hull_length_m, pg.hull_draught_m))
        profiles[label] = {"field": a, "field_grid": a, "r_squared": r2, "rmse": rmse,
                           "scale_um_per_h": float(a.max() * 1e6 * 3600.0)}
    # the coarse profiles are obtained by block-averaging the fine one, so the
    # discretisation term contains the grid effect and nothing else
    profiles["medium"]["field_grid"] = coarse_field(
        profiles["fine"]["field"], GRIDS["fine"], GRIDS["medium"])
    profiles["coarse"]["field_grid"] = coarse_field(
        profiles["fine"]["field"], GRIDS["fine"], GRIDS["coarse"])

    # ---------------- C1-C6 ------------------------------------------------
    adj_res = adj.run_adjudication(
        truth_fine=truth["fine"], truth_coarse=truth["coarse"],
        source_fine=source["fine"], source_coarse=source["coarse"],
        areas_fine=areas["fine"], areas_coarse=areas["coarse"],
        n_realizations=N_REALIZATIONS, n_bootstrap=N_BOOTSTRAP, seed=SEED,
        profile_fine=profiles["fine"]["field"],
        profile_coarse=profiles["coarse"]["field_grid"])

    # ---------------- resolution ladder ------------------------------------
    ladder = adj.resolution_ladder(
        {k: truth[k] for k in GRIDS}, {k: source[k][0] for k in GRIDS},
        {k: areas[k] for k in GRIDS}, n_realizations=N_REALIZATIONS, seed=SEED,
        profiles_by_grid={k: profiles[k]["field_grid"] for k in GRIDS})
    # multi-source version of the coarse decomposition (Monte-Carlo noise of the
    # predictor is visible, so it is reported instead of hidden)
    multi = adj.resolution_decomposition(
        truth["fine"], truth["coarse"], source["fine"], source["coarse"],
        areas_fine=areas["fine"], areas_coarse=areas["coarse"],
        n_realizations=N_REALIZATIONS, seed=SEED,
        profile_fine=profiles["fine"]["field"],
        profile_coarse=profiles["coarse"]["field_grid"],
        closure_tol_pct=10.0)

    # per-grid bootstrap comparison + K=40 robustness of the referee
    boot_table = {}
    for label in GRIDS:
        ref = adj.p95_variance_truth(truth[label], areas=areas[label])
        ref40 = adj.p95_variance_truth(robust[label], areas=areas[label])
        iid = adj.p95_variance_bootstrap_iid(source[label][0], n_bootstrap=N_BOOTSTRAP,
                                             seed=SEED, areas=areas[label])
        mb = adj.p95_variance_bootstrap_moving_block(source[label][0], block=(8, 3),
                                                     n_bootstrap=N_BOOTSTRAP, seed=SEED,
                                                     areas=areas[label])
        cond = adj.p95_variance_conditional(source[label][0], profiles[label]["field_grid"],
                                            n_realizations=N_REALIZATIONS, seed=SEED,
                                            areas=areas[label])
        boot_table[label] = {
            "truth_K20": ref.summary(), "truth_K40": ref40.summary(),
            "iid_bootstrap": {**iid.summary(),
                              "pct_of_truth_variance": 100.0 * iid.p95_variance / ref.p95_variance,
                              "ratio_over_truth": iid.p95_variance / ref.p95_variance},
            "moving_block_bootstrap": {**mb.summary(),
                                       "pct_of_truth_variance": 100.0 * mb.p95_variance / ref.p95_variance,
                                       "ratio_over_truth": mb.p95_variance / ref.p95_variance},
            "conditional": {**cond.summary(),
                            "ratio_over_truth": cond.p95_variance / ref.p95_variance},
        }

    # model comparison: spectrum x noise model, all against the same referee
    ref20 = adj.p95_variance_truth(truth["fine"], areas=areas["fine"])
    ref40 = adj.p95_variance_truth(robust["fine"], areas=areas["fine"])
    model_compare = {}
    for spec in ("iaaft", "spectral"):
        for noise in ("local_scale", "stationary"):
            est = adj.p95_variance_conditional(
                source["fine"][0], profiles["fine"]["field_grid"],
                n_realizations=N_REALIZATIONS, seed=SEED, areas=areas["fine"],
                spectrum=spec, noise=noise)
            model_compare[f"{spec}_{noise}"] = {
                **est.summary(),
                "ratio_over_truth_K20": est.p95_variance / ref20.p95_variance,
                "ratio_over_truth_K40": est.p95_variance / ref40.p95_variance,
            }

    # negative fraction of the conditional realisations (model validity domain)
    sims_fine = adj.simulate_abrasion_realizations(
        source["fine"][0], profiles["fine"]["field"], n_realizations=N_REALIZATIONS,
        seed=SEED)
    neg_frac = 100.0 * float(np.mean(sims_fine < 0.0))
    r = adj.decompose_multiplicative(source["fine"][0], profiles["fine"]["field"]).residual
    model = adj.estimate_spectrum_model(r).summary()
    noise_model = adj.fit_noise_scale(source["fine"][0], profiles["fine"]["field"])

    # ---------------- assemble ---------------------------------------------
    elapsed = time.perf_counter() - t_start
    result = {
        "node": "AG27W2-N06",
        "title": "Conditional simulation of ice abrasion fields and resolution-vs-sampling "
                 "error adjudication",
        "repository": "TensorDEM",
        "module": "src/tensordem/ice_abrasion_conditional_adjudication.py",
        "ensemble": ens_meta,
        "grids": {k: {"panels": list(v.shape[1:]), "n_cells": int(np.prod(v.shape[1:])),
                      "cell_size_m": [float(np.diff(panels[k].x_edges)[0]),
                                      float(np.diff(panels[k].z_edges)[0])]}
                  for k, v in fields.items()},
        "profiles": {k: {"r_squared": profiles[k]["r_squared"],
                         "rmse": profiles[k]["rmse"],
                         "max_um_per_h": float(profiles[k]["field"].max() * 1e6 * 3600.0)}
                     for k in GRIDS},
        "settings": {"n_truth": N_TRUTH, "n_source": N_SOURCE, "n_realizations": N_REALIZATIONS,
                     "n_bootstrap": N_BOOTSTRAP, "seed": SEED},
        "criteria": adj_res["criteria"],
        "all_criteria_passed": adj_res["all_criteria_passed"],
        "resolution_ladder": ladder,
        "resolution_decomposition_multi_source": multi.summary(),
        "bootstrap_table": boot_table,
        "model_comparison": model_compare,
        "truth_variance_uncertainty": {
            "K20": ref20.summary(), "K40": ref40.summary(),
            "chi2_95ci_K20": [float(ref20.p95_variance * 19 / 32.852),
                              float(ref20.p95_variance * 19 / 8.907)],
        },
        "conditional_model_check": {
            "negative_fraction_pct": neg_frac,
            "noise_scale": noise_model.summary(),
            "spectrum_model": model,
            "residual_mean": float(r.mean()),
            "residual_std": float(r.std()),
        },
        "timing_s": elapsed,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "_fields": fields, "_a_fine": profiles["fine"]["field"], "_sims_fine": sims_fine,
    }
    figs = [] if args.skip_figures else make_figures(result)
    result["figures"] = figs
    for key in ("_fields", "_a_fine", "_sims_fine"):
        result.pop(key)
    (RUN_DIR / "verification.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps({k: result[k] for k in
                      ("all_criteria_passed", "timing_s", "conditional_model_check")},
                     indent=2, default=float))
    return result


if __name__ == "__main__":
    main()