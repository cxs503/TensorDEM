"""Merge the sharded column-collapse runs into one artifact + figures.

Reads ``runs/p165w2_n06/column_*.json`` and writes

* ``artifacts/dem/p165w2_n06/column_collapse.json``  (all numbers the report cites)
* ``artifacts/dem/p165w2_n06/figures/fig6_column_*.png``

Everything is computed from the run JSONs -- no number is typed by hand.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs", "p165w2_n06")
OUT = os.path.join(REPO, "artifacts", "dem", "p165w2_n06")
FIG = os.path.join(OUT, "figures")


def load() -> List[Dict[str, object]]:
    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS, "column_*.json"))):
        with open(path) as fh:
            row = json.load(fh)
        row["source"] = os.path.relpath(path, REPO)
        rows.append(row)
    return rows


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rows = load()
    if not rows:
        raise SystemExit(f"no runs found in {RUNS}")
    mu_ref = 0.45
    n_series = sorted([r for r in rows
                       if abs(r["mu"] - mu_ref) < 1e-9 and abs(r["dt_scale"] - 1.0) < 1e-9
                       and r["label"] != "n2k_seed7"],
                      key=lambda r: r["n_actual"])
    a = np.array([r["repose_angle_deg"] for r in n_series])
    spread = float((a.max() - a.min()) / max(a.mean(), 1e-30)) if len(a) else float("nan")
    settled = [r for r in n_series if r["plateau_reached"]]
    a_set = np.array([r["repose_angle_deg"] for r in settled]) if settled else a
    spread_settled = float((a_set.max() - a_set.min()) / max(a_set.mean(), 1e-30)) \
        if len(a_set) else float("nan")
    friction = sorted([r for r in rows if abs(r["dt_scale"] - 1.0) < 1e-9
                       and r["n_target"] == 2000 and r["label"] != "n2k_seed7"],
                      key=lambda r: r["mu"])
    dt_series = sorted([r for r in rows if r["n_target"] == 2000
                        and abs(r["mu"] - mu_ref) < 1e-9
                        and r["label"].startswith("n2k_dt")],
                       key=lambda r: r["dt_scale"])
    dt_all = [r for r in rows if r["label"] == "n2k_mu0.45"] + dt_series
    dt_all.sort(key=lambda r: r["dt_scale"])
    seeds = [r for r in rows if abs(r["mu"] - mu_ref) < 1e-9 and r["n_target"] == 2000]
    seed_spread = float(max(r["repose_angle_deg"] for r in seeds)
                        - min(r["repose_angle_deg"] for r in seeds)) if len(seeds) > 1 \
        else float("nan")
    steps_per_s = [r["total_steps"] / max(r["wall_clock_s"], 1e-9) for r in rows]
    summary = {
        "n_series_range": [int(n_series[0]["n_actual"]), int(n_series[-1]["n_actual"])]
        if n_series else None,
        "n_series_angles_deg": {int(r["n_actual"]): r["repose_angle_deg"] for r in n_series},
        "n_series_spread_rel": spread,
        "n_series_spread_rel_settled_only": spread_settled,
        "reaches_32k": any(r["n_actual"] >= 30000 and r["plateau_reached"] for r in rows),
        "friction_scan": [{"mu": r["mu"], "repose_angle_deg": r["repose_angle_deg"],
                           "label": r["label"]} for r in friction],
        "dt_convergence": [{"dt": r["dt"], "dt_scale": r["dt_scale"],
                            "repose_angle_deg": r["repose_angle_deg"],
                            "plateau_reached": r["plateau_reached"]} for r in dt_all],
        "dt_max_angle_change_deg": (
            float(max(r["repose_angle_deg"] for r in dt_all)
                  - min(r["repose_angle_deg"] for r in dt_all)) if len(dt_all) > 1
            else float("nan")),
        "seed_spread_deg": seed_spread,
        "all_plateau_reached": all(r["plateau_reached"] for r in rows),
        "throughput_steps_per_s": {r["label"]: r["total_steps"]
                                   / max(r["wall_clock_s"], 1e-9) for r in rows},
        "throughput_steps_per_s_median": float(np.median(steps_per_s)),
    }
    payload = {"campaign_node": "P165-W2-N06", "axis_mu_reference": mu_ref,
               "runs": rows, "summary": summary}
    path = os.path.join(OUT, "column_collapse.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote", path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig 6: repose angle vs N (with the settled flag) and vs mu
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    if n_series:
        n = [r["n_actual"] for r in n_series]
        th = [r["repose_angle_deg"] for r in n_series]
        ok = [r["plateau_reached"] for r in n_series]
        axes[0].plot([x for x, o in zip(n, ok) if o], [y for y, o in zip(th, ok) if o],
                     "o-", label="settled (plateau)")
        bad = [(x, y) for x, y, o in zip(n, th, ok) if not o]
        if bad:
            axes[0].plot([b[0] for b in bad], [b[1] for b in bad], "x--",
                         label="not settled (step budget)")
        axes[0].axhspan(np.mean(th) * 0.95, np.mean(th) * 1.05, color="g", alpha=0.12,
                        label="+-5% band")
        axes[0].set_xlabel("particle number N")
        axes[0].set_ylabel("repose angle (deg)")
        axes[0].set_xscale("log")
        axes[0].set_title(f"particle-number convergence (spread {spread:.2%})")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3)
    if friction:
        axes[1].plot([r["mu"] for r in friction],
                     [r["repose_angle_deg"] for r in friction], "s-")
        axes[1].set_xlabel("inter-particle friction mu")
        axes[1].set_ylabel("repose angle (deg)")
        axes[1].set_title(f"friction scan (N={friction[0]['n_actual']})")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p6 = os.path.join(FIG, "fig6_column_repose.png")
    fig.savefig(p6, dpi=160)
    plt.close(fig)

    # fig 7: one heap per N (side view) for the settled runs
    if n_series:
        k = len(n_series)
        fig, axes = plt.subplots(1, k, figsize=(3.0 * k, 3.0), squeeze=False)
        for ax, r in zip(axes[0], n_series):
            cloud_path = os.path.join(RUNS, f"cloud_{r['label']}.npz")
            if not os.path.exists(cloud_path):
                ax.axis("off")
                continue
            pos = np.load(cloud_path)["pos"]
            rad = float(np.load(cloud_path)["radius"])
            ax.scatter(pos[:, 0] * 1e3, pos[:, 2] * 1e3, s=1.5, c=pos[:, 1] * 1e3,
                       cmap="viridis", linewidths=0)
            ax.set_title(f"N={r['n_actual']}  θ={r['repose_angle_deg']:.1f}°",
                         fontsize=8)
            ax.set_xlabel("x (mm)", fontsize=7)
            ax.set_aspect("equal", adjustable="datalim")
            ax.tick_params(labelsize=6)
        axes[0][0].set_ylabel("z (mm)", fontsize=7)
        fig.tight_layout()
        p7 = os.path.join(FIG, "fig7_column_clouds.png")
        fig.savefig(p7, dpi=150)
        plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()