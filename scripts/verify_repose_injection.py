"""Verification campaign for the trickle-injection repose driver (TensorDEM).

Produces ``artifacts/repose_injection/verification.json`` (all numbers quoted by
``docs/repose_injection_verification.md``), a CSV of the case matrix, and two
figures.  Nothing is hard-coded downstream: the document reads the JSON.

Usage
-----
    PYTHONPATH=src python scripts/verify_repose_injection.py --workers 4
    PYTHONPATH=src python scripts/verify_repose_injection.py --quick
    PYTHONPATH=src python scripts/verify_repose_injection.py --figures-only
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from typing import Any

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.repose_injection import (  # noqa: E402
    InjectionCase,
    ReposeInjectionError,
    kinematic_bound_margin,
    simulate_injection,
    slide_threshold,
    surface_flank_angle,
)

OUT_DIR = os.path.join("artifacts", "repose_injection")
FIG_DIR = os.path.join(OUT_DIR, "figures")


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------

BASE = dict(n_particles=140, injection_rate=120.0, settle_time=0.35, mu=0.5,
            mu_rolling=0.20, radius=6.0e-3)


def case_matrix(quick: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, purpose: str, **over):
        cfg = dict(BASE)
        cfg.update(over)
        rows.append({"name": name, "purpose": purpose, "case": cfg})

    add("baseline", "reference heap (mu=0.50, mu_r=0.20, monodisperse)")
    add("rate_slow", "injection-rate independence: 60 /s", injection_rate=60.0)
    add("rate_fast", "injection-rate independence: 240 /s", injection_rate=240.0)
    if quick:
        add("mu_r_free", "rolling negative control (mu_r=0)", mu_rolling=0.0)
        return rows
    add("mu_r_free", "rolling negative control (mu_r=0)", mu_rolling=0.0)
    add("mu_r_low", "rolling sweep (mu_r=0.10)", mu_rolling=0.10)
    add("mu_r_high", "rolling sweep (mu_r=0.30)", mu_rolling=0.30)
    add("poly_25", "polydispersity effect (+-25 % radius)", polydispersity=0.25)
    add("legacy_floor", "legacy floor convention (spin term dropped)", floor_spin_term=False)
    add("mu_low", "sliding-friction sweep (mu=0.35)", mu=0.35, mu_rolling=0.10)
    return rows


def _run_one(item: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(item["case"])
    t0 = time.time()
    out: dict[str, Any] = {"name": item["name"], "purpose": item["purpose"],
                           "config": cfg}
    try:
        result = simulate_injection(InjectionCase(**cfg))
    except (ReposeInjectionError, RuntimeError, ValueError) as exc:
        out.update({"status": "INVALID", "error": f"{type(exc).__name__}: {exc}",
                    "wall_seconds": time.time() - t0})
        return out
    keep = ("n_particles", "dt", "steps", "simulated_time", "settled", "max_speed",
            "kinetic_energy_J", "apex_height", "footprint_radius", "theta_r_deg",
            "r2", "theta_flank_deg", "flank_r2", "flank_sector_std_deg",
            "flank_points", "injection_time", "mean_radius", "radius_min",
            "radius_max", "n_bins", "angle_error")
    out.update({k: result.get(k) for k in keep})
    out["status"] = "VALID" if result.get("theta_r_deg") is not None else "INVALID"
    out["wall_seconds"] = time.time() - t0
    if result.get("theta_r_deg") is not None:
        out["kinematic_bound"] = kinematic_bound_margin(result["theta_r_deg"],
                                                        cfg.get("mu", BASE["mu"]))
    return out


# ---------------------------------------------------------------------------
# Single-grain anchors
# ---------------------------------------------------------------------------

def run_anchors() -> dict[str, Any]:
    anchors: dict[str, Any] = {}
    anchors["A1_sliding_onset"] = slide_threshold(0.5, allow_rotation=False)
    anchors["A2_rolling_onset_mur020"] = slide_threshold(0.5, mu_rolling=0.20,
                                                         allow_rotation=True)
    # Ideal rolling (mu_r = 0): any tilt above zero must roll, so the bracket is
    # taken around a small angle instead of around atan(mu).
    anchors["A2b_ideal_rolling_mur0"] = slide_threshold(
        0.5, mu_rolling=0.0, allow_rotation=True, bracket=(0.25, 1.0))
    anchors["A3_legacy_floor_mur0"] = slide_threshold(
        0.5, mu_rolling=0.0, allow_rotation=True, floor_spin_term=False)
    anchors["A3b_legacy_floor_mur020"] = slide_threshold(
        0.5, mu_rolling=0.20, allow_rotation=True, floor_spin_term=False)
    for key, value in anchors.items():
        value["case"] = key
    return anchors


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(payload: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    made: list[str] = []
    rows = [r for r in payload["cases"] if r["status"] == "VALID"]

    # 1. mu_r sweep -> theta (with arctan(mu_r) reference)
    sweep = [r for r in rows if r["name"].startswith(("baseline", "mu_r_"))]
    if len(sweep) >= 2:
        sweep.sort(key=lambda r: r["config"]["mu_rolling"])
        mur = [r["config"]["mu_rolling"] for r in sweep]
        th = [r["theta_r_deg"] for r in sweep]
        thf = [r["theta_flank_deg"] or float("nan") for r in sweep]
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        ax.plot(mur, th, "o-", label="radial-bin estimator")
        ax.plot(mur, thf, "s--", label="free-surface fit")
        ax.plot(mur, [math.degrees(math.atan(m)) for m in mur], "k:",
                label=r"$\arctan(\mu_r)$")
        ax.set_xlabel(r"rolling coefficient $\mu_r$ [-]")
        ax.set_ylabel(r"repose angle $\theta_r$ [deg]")
        ax.set_title("Trickle-injection repose vs rolling resistance")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        path = os.path.join(FIG_DIR, "repose_vs_mu_rolling.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(path)

    # 2. rate independence
    rates = [r for r in rows if r["name"].startswith("rate_")]
    if rates:
        base = next((r for r in rows if r["name"] == "baseline"), rates[0])
        pts = sorted([base] + rates, key=lambda r: r["config"]["injection_rate"])
        x = [p["config"]["injection_rate"] for p in pts]
        y = [p["theta_r_deg"] for p in pts]
        yf = [p["theta_flank_deg"] or float("nan") for p in pts]
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(x, y, "o-", label="radial-bin estimator")
        ax.plot(x, yf, "s--", label="free-surface fit")
        ax.set_xlabel("injection rate [grains/s]")
        ax.set_ylabel(r"$\theta_r$ [deg]")
        ax.set_title("Rate control: angle vs release rate")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        path = os.path.join(FIG_DIR, "repose_vs_rate.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(path)

    # 3. heap cross-section of the baseline run (regenerated on demand)
    return made


def heap_profile_figure(payload: dict[str, Any]) -> str | None:
    """Re-run the baseline heap and plot z vs r with the flank fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = next((r for r in payload["cases"]
                 if r["name"] == "baseline" and r["status"] == "VALID"), None)
    if base is None:
        return None
    cfg = dict(base["config"])
    result = simulate_injection(InjectionCase(**cfg), return_particles=True)
    pos = result["positions"]
    rad = result["radii"]
    rel = pos[:, :2] - pos[:, :2].mean(dim=0)
    radial = torch.linalg.vector_norm(rel, dim=1).numpy()
    z = pos[:, 2].numpy()
    flank = surface_flank_angle(pos, rad)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(radial * 1e3, z * 1e3, s=6, alpha=0.6, label="grains (drop, top view)")
    ax.scatter(radial * 1e3, -z * 1e3, s=6, alpha=0.25, label="mirror")
    rr = [0.0, flank["r_toe"] * 1e3]
    zz = [(flank["intercept"]) * 1e3,
          (flank["intercept"] + flank["slope"] * flank["r_toe"]) * 1e3]
    ax.plot(rr, zz, "r-", lw=1.6,
            label=fr"flank fit $\theta$={flank['theta_deg']:.2f}$^\circ$")
    ax.set_xlabel("radial distance [mm]")
    ax.set_ylabel("height [mm]")
    ax.set_title(fr"trickle-injection heap, N={result['n_particles']}, "
                 fr"$\mu_r$={cfg['mu_rolling']}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "heap_profile.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--heap-figure", action="store_true",
                    help="also re-run the baseline heap for the profile figure")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "verification.json")

    if args.figures_only:
        payload = json.load(open(json_path))
        payload["figures"] = make_figures(payload)
        if args.heap_figure:
            heap = heap_profile_figure(payload)
            if heap:
                payload["figures"].append(heap)
        json.dump(payload, open(json_path, "w"), indent=2)
        print(json.dumps(payload["figures"], indent=2))
        return 0

    t0 = time.time()
    anchors = run_anchors()
    matrix = case_matrix(args.quick)
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            cases = pool.map(_run_one, matrix)
    else:
        cases = [_run_one(item) for item in matrix]

    payload = {
        "protocol": "tensordem-trickle-injection-repose-v1",
        "generated_by": "scripts/verify_repose_injection.py",
        "wall_seconds": time.time() - t0,
        "anchors": anchors,
        "cases": cases,
        "quick": bool(args.quick),
    }
    payload["figures"] = make_figures(payload)
    json.dump(payload, open(json_path, "w"), indent=2)
    with open(os.path.join(OUT_DIR, "cases.csv"), "w") as fh:
        fh.write("case,status,mu,mu_rolling,poly,rate,n,theta_r_deg,theta_flank_deg,"
                 "settled,max_speed,wall_s\n")
        for r in cases:
            c = r["config"]
            fh.write(",".join(str(v) for v in (
                r["name"], r["status"], c.get("mu"), c.get("mu_rolling"),
                c.get("polydispersity", 0.0), c.get("injection_rate"),
                r.get("n_particles"), r.get("theta_r_deg"),
                r.get("theta_flank_deg"), r.get("settled"), r.get("max_speed"),
                round(r.get("wall_seconds", 0.0), 1))) + "\n")
    print(json.dumps({"json": json_path, "cases": len(cases),
                      "valid": sum(1 for r in cases if r["status"] == "VALID"),
                      "wall_seconds": payload["wall_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())