"""CASE-DEM-PLANE-SHEAR-MUIDI round 2 (m16427): re-adjudication with the
steady-state bed metric.

Round 1 (`docs/CASE_DEM_PLANE_SHEAR_MUIDI.md`) reported

    phi(I) = 0.5218 for all six inertial numbers  ->  "device does not
    reproduce shear densification", phi gate FAIL.

That phi was `n*V_p/(W*D*plate_height)` with `plate_height` captured **before**
the shear phase, i.e. the settled height, which is identical for every shear
rate by construction (same seed, no shear during settle).  The round-1 phi
verdict therefore measured nothing about the material.

This script re-adjudicates the same runs with the device's steady-state
bookkeeping (`phi_bed_steady`, tail average of the servo-updated platen
height) and keeps the old metric in the figure as the documented defect.

Inputs : runs/sweep_baseline.json (+ optional runs/sweep_stiff.json)
Outputs: runs/analysis_r2.json, figures/{phi_of_I_r2,error_vs_I_r2}.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(name: str):
    path = RUNS / name
    return json.loads(path.read_text()) if path.exists() else None


def _phi_ref(I: float, ref: dict) -> float:
    return (ref["phi_c"] + (ref["phi_j"] - ref["phi_c"])
            / (ref["I_0"] / I + 1.0))


def _direction(rows, key: str) -> tuple[float, float]:
    """Least-squares slope of phi vs log10(I) (per decade) and its sign test."""
    x = np.log10([r["I_measured"] for r in rows])
    y = np.array([r[key] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    n = len(x)
    se = float(np.sqrt(np.sum(resid ** 2) / max(n - 2, 1))
               / np.sqrt(np.sum((x - x.mean()) ** 2)))
    return float(slope), se


def main() -> None:
    base = _load("sweep_baseline.json")
    if base is None:
        raise SystemExit("runs/sweep_baseline.json missing; run run_sweep.py first")
    stiff = _load("sweep_stiff.json")
    ref = base["reference"]
    rows = base["rows"]

    out: dict = {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 2,
        "mission": "m16427",
        "arm": base.get("arm"),
        "reference": ref,
        "ref_constants_in_repo": {
            "mu_s_mu_2_I_0": "tensordem.brash_ice_rheology.mu_i.LITERATURE_JOP "
                             "= (0.36, 0.84, 0.30) -- anchored",
            "phi_c_phi_j": "no in-repo anchor found (grep brash_ice_rheology.mu_i / "
                           "granular_column.scaling); hand-copied -> "
                           "PROVENANCE-UNVERIFIED",
        },
    }

    # ---- metric defect evidence -------------------------------------------
    phi_settled = np.array([r["phi_bed_settled"] for r in rows], dtype=float)
    phi_steady = np.array([r["phi_bed_steady"] for r in rows], dtype=float)
    dilat = np.array([r["dilatancy"] for r in rows], dtype=float)
    I = np.array([r["I_measured"] for r in rows], dtype=float)
    out["metric_defect"] = {
        "phi_settled_spread": float(phi_settled.max() - phi_settled.min()),
        "phi_settled_unique_values": int(len(np.unique(np.round(phi_settled, 12)))),
        "plate_height_settled_unique_values": int(
            len({round(r["plate_height_settled"], 15) for r in rows})),
        "phi_steady_spread": float(phi_steady.max() - phi_steady.min()),
        "dilatancy_range": [float(dilat.min()), float(dilat.max())],
        "verdict": ("round-1 phi metric is degenerate: the settled platen height "
                    "is identical at every shear rate, so phi_bed_settled cannot "
                    "carry an I-dependence"),
    }

    # ---- error tables ------------------------------------------------------
    points = []
    for r in rows:
        mu_err = 100.0 * (r["mu_measured"] / r["ref_mu"] - 1.0)
        phi_s_err = 100.0 * (r["phi_bed_steady"] / r["ref_phi"] - 1.0)
        phi_set_err = 100.0 * (r["phi_bed_settled"] / r["ref_phi"] - 1.0)
        points.append({
            "I": float(r["I_measured"]),
            "mu": float(r["mu_measured"]), "ref_mu": float(r["ref_mu"]),
            "mu_err_pct": float(mu_err),
            "phi_steady": float(r["phi_bed_steady"]),
            "phi_settled": float(r["phi_bed_settled"]),
            "dilatancy": float(r["dilatancy"]),
            "ref_phi": float(r["ref_phi"]),
            "phi_steady_err_pct": float(phi_s_err),
            "phi_settled_err_pct": float(phi_set_err),
            "K_contacts": float(r["coordination"]),
            "wall_clock_s": float(r["wall_clock_s"]),
        })
    out["points"] = points

    mu_err = np.array([p["mu_err_pct"] for p in points])
    ps_err = np.array([p["phi_steady_err_pct"] for p in points])
    out["gates"] = {
        "mu_within_10pct": bool(np.all(np.abs(mu_err) < 10.0)),
        "mu_worst_err_pct": float(np.max(np.abs(mu_err))),
        "mu_failing_points": [float(p["I"]) for p in points
                              if abs(p["mu_err_pct"]) >= 10.0],
        "phi_steady_within_10pct": bool(np.all(np.abs(ps_err) < 10.0)),
        "phi_steady_worst_err_pct": float(np.max(np.abs(ps_err))),
        "phi_steady_failing_points": [float(p["I"]) for p in points
                                      if abs(p["phi_steady_err_pct"]) >= 10.0],
    }

    # ---- direction diagnostic (does measured phi(I) rise or fall?) ---------
    slope_steady, se_steady = _direction(rows, "phi_bed_steady")
    slope_settled, se_settled = _direction(rows, "phi_bed_settled")
    ref_slope = (math.log10(ref["phi_j"]) - math.log10(ref["phi_c"]))
    out["direction"] = {
        "dphi_steady_per_decade": slope_steady,
        "dphi_steady_per_decade_se": se_steady,
        "dphi_settled_per_decade": slope_settled,
        "dphi_settled_per_decade_se": se_settled,
        "ref_implied_slope_per_decade_high_I": ref_slope,
        "measured_sign": ("rising" if slope_steady > 0 else "falling"),
        "ref_sign": "rising (phi_j=0.64 > phi_c=0.58)",
        "consistent_with_ref_direction": bool(slope_steady > 0),
        "consistent_with_dilatancy": bool(slope_steady < 0),
    }

    # ---- optional stiffness arm -------------------------------------------
    if stiff is not None:
        srow = stiff["rows"][0]
        bmatch = min(rows, key=lambda r: abs(r["I_measured"] - srow["I_measured"]))
        out["stiffness_arm"] = {
            "arm": stiff.get("arm"),
            "kn": stiff["config"]["kn"],
            "I": float(srow["I_measured"]),
            "phi_steady": float(srow["phi_bed_steady"]),
            "mu": float(srow["mu_measured"]),
            "baseline_I": float(bmatch["I_measured"]),
            "baseline_phi_steady": float(bmatch["phi_bed_steady"]),
            "baseline_mu": float(bmatch["mu_measured"]),
            "phi_shift_pct": float(100.0 * (srow["phi_bed_steady"]
                                            / bmatch["phi_bed_steady"] - 1.0)),
            "mu_shift_pct": float(100.0 * (srow["mu_measured"] / bmatch["mu_measured"] - 1.0)),
        }

    # ---- figures -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=150)
    i_grid = np.logspace(-3.3, 0.5, 200)
    ax.semilogx(i_grid, [_phi_ref(i, ref) for i in i_grid], "-", color="0.35",
                lw=2, label=r"ref $\phi_c$=0.58$\to\phi_j$=0.64 (provenance unverified)")
    ax.semilogx(I, phi_settled, "x", color="tab:red", ms=9, mew=2,
                label="round-1 metric: settled height (frozen, degenerate)")
    ax.semilogx(I, phi_steady, "s-", color="tab:green", ms=7,
                label="round-2 metric: steady-state bed")
    for p in points:
        ax.annotate(f"{p['phi_steady_err_pct']:+.1f}%",
                    (p["I"], p["phi_steady"]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7)
    ax.set_xlabel("inertial number $I$")
    ax.set_ylabel(r"solid fraction $\phi$")
    ax.set_title("Plane shear: which $\\phi$ was actually measured?")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "phi_of_I_r2.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(10, color="r", ls=":", lw=1)
    ax.axhline(-10, color="r", ls=":", lw=1)
    ax.semilogx(I, mu_err, "o-", color="tab:blue", label=r"$\mu$ error")
    ax.semilogx(I, ps_err, "s-", color="tab:green",
                label=r"$\phi$ error (steady-state metric)")
    ax.set_xlabel("inertial number $I$")
    ax.set_ylabel("relative error vs reference [%]")
    ax.set_title("Round 2 error summary (band = $\\pm$10 % gate)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "error_vs_I_r2.png")
    plt.close(fig)

    (RUNS / "analysis_r2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("metric_defect", "gates", "direction")},
                     indent=1))


if __name__ == "__main__":
    main()