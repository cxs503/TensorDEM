"""Figures for the repose-angle metrology fix (H1 C2 / anchor H1-BROKEN-ICE-DEM).

Fig 1 is generated from synthetic piles with a **known** slope, so it is a
validation of the estimator itself: true angle -> recovered angle for the naive
whole-profile fit (the pre-fix behaviour) and for the BIC flank fit.  Fig 2
plots the DEM repose runs from ``artifacts/h1_broken_ice_dem/repose_shard_*.json``
once the cluster shards are fetched.

    PYTHONPATH=src python benchmarks/h1_broken_ice_dem/make_figures_round3.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.polyhedron_floe3d.pile_surface import (  # noqa: E402
    bin_free_surface,
    fit_flank,
)

OUT = os.path.join(ROOT, "artifacts", "h1_broken_ice_dem")
BAND = (25.0, 40.0)


def synthetic(angle_deg: float, n_bodies: int, apron_share: float,
             radius: float = 0.30, body_r: float = 0.05, seed: int = 7):
    rng = np.random.default_rng(seed)
    slope = math.tan(math.radians(angle_deg))
    n_apron = int(round(n_bodies * apron_share))
    xy, top = [], []
    for _ in range(n_bodies - n_apron):
        r = radius * math.sqrt(rng.random())
        th = rng.uniform(0.0, 2.0 * math.pi)
        xy.append((r * math.cos(th), r * math.sin(th)))
        top.append((radius - r) * slope + body_r)
    for _ in range(n_apron):
        r = 1.6 * radius * math.sqrt(rng.random())
        th = rng.uniform(0.0, 2.0 * math.pi)
        xy.append((r * math.cos(th), r * math.sin(th)))
        top.append(body_r)
    xy = np.asarray(xy)
    return bin_free_surface(np.hypot(xy[:, 0], xy[:, 1]), np.asarray(top))


def naive_angle(profile) -> float:
    slope, _ = np.polyfit(np.asarray(profile["radius"]), np.asarray(profile["height"]), 1)
    return math.degrees(math.atan(max(-slope, 0.0)))


def fig1(truth=np.arange(15.0, 45.1, 2.5)):
    naive_clean, naive_apron, flank_clean, flank_apron = [], [], [], []
    for a in truth:
        for share, naive, flank in ((0.0, naive_clean, flank_clean),
                                    (0.4, naive_apron, flank_apron)):
            prof = synthetic(float(a), 400, share)
            naive.append(naive_angle(prof))
            flank.append(fit_flank(prof["radius"], prof["height"],
                                   prof["count"], min_drop=0.05)["angle_deg"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhspan(*BAND, color="0.85", label="ice-rubble literature band (25-40 deg)")
    ax.plot([10, 46], [10, 46], "k--", lw=1, label="1:1")
    ax.plot(truth, naive_clean, "s-", ms=3.5, label="naive whole-profile fit, no apron")
    ax.plot(truth, naive_apron, "s--", ms=3.5, label="naive whole-profile fit, +apron")
    ax.plot(truth, flank_clean, "o-", ms=3.5, label="BIC flank fit, no apron")
    ax.plot(truth, flank_apron, "o--", ms=3.5, label="BIC flank fit, +apron")
    ax.set_xlabel("true cone angle [deg]")
    ax.set_ylabel("recovered angle [deg]")
    ax.set_title("Repose-angle estimator vs synthetic ground truth (n=400 bodies)")
    ax.set_xlim(13, 46)
    ax.set_ylim(10, 46)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    path = os.path.join(OUT, "fig4_estimator_ground_truth.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    worst_naive = max(abs(n - a) for a, n in zip(truth, naive_apron))
    worst_flank = max(abs(f - a) for a, f in zip(truth, flank_apron)
                      if math.isfinite(f))
    return {"figure": path, "worst_error_naive_apron_deg": worst_naive,
            "worst_error_flank_apron_deg": worst_flank,
            "truth_deg": [float(t) for t in truth],
            "naive_clean": naive_clean, "naive_apron": naive_apron,
            "flank_clean": flank_clean, "flank_apron": flank_apron}


def fig2():
    shards = sorted(glob.glob(os.path.join(OUT, "repose_shard_*.json")))
    if not shards:
        return None
    rows = []
    for path in shards:
        with open(path) as fh:
            data = json.load(fh)
        case = data.get("cases", {}).get("repose")
        if not case:
            continue
        for run in case["runs"]:
            rows.append(run)
    if not rows:
        return None
    rows.sort(key=lambda r: r["friction"])
    mu = [r["friction"] for r in rows]
    angle = [r.get("repose_angle_deg", float("nan")) for r in rows]
    fit = [r.get("repose_flank", {}) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhspan(*BAND, color="0.85", label="literature band 25-40 deg")
    ax.plot(mu, angle, "o-", color="C3", label="DEM flank angle (n=45, dt=2e-4, kn=5e4)")
    for m, a, f in zip(mu, angle, fit):
        if f.get("toe_radius_m") is not None and math.isfinite(a):
            ax.annotate("flank bins %d, apron share %.2f"
                        % (f.get("n_flank_bins", 0), f.get("apron_body_share", float("nan"))),
                        (m, a), fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("ice-ice friction mu [-]")
    ax.set_ylabel("angle of repose [deg]")
    ax.set_title("H1 C2 rubble repose: flank angle vs friction")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(OUT, "fig5_repose_flank_vs_mu.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"figure": path, "runs": rows}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    out = {"fig1": fig1()}
    f2 = fig2()
    out["fig2"] = f2
    print(json.dumps({k: (v if k == "fig1" else {"figure": v["figure"]} if v else None)
                      for k, v in out.items()}, indent=1, default=float))
    with open(os.path.join(OUT, "round3_figures.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())