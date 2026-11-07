"""m6296: analysis + figures for the Beverloo silo-discharge verification.

Reads the per-case JSON written by ``scripts/m6296_silo_beverloo_sweep.py``
(``runs/m6296_silo/<case>.json``) and produces

* ``runs/m6296_silo/summary.json``  -- every number used in the report,
* ``docs/figures/m6296_silo_*.png`` -- comparison figures,
* a markdown table on stdout (pasted into the doc, zero hand-copied digits).

Usage
-----
    PYTHONPATH=src python scripts/m6296_beverloo_analysis.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.silo_discharge import beverloo_fit, beverloo_prediction  # noqa: E402

RUNS = "runs/m6296_silo"
FIGS = "docs/figures"

# literature (Beverloo et al. 1961; glass beads / sand, smooth silo)
C_LIT, K_LIT = 0.58, 1.4
C_LIT_RANGE = (0.55, 0.60)
K_LIT_RANGE = (1.2, 1.6)


def load() -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        name = os.path.basename(path)[:-5]
        if name in ("summary", "snapshot"):
            continue
        with open(path, encoding="utf8") as fh:
            out[name] = json.load(fh)
    return out


def main() -> int:
    runs = load()
    need = {"D3", "D4", "D5"}
    if not need.issubset(runs):
        print(f"missing cases: {need - set(runs)} (have {sorted(runs)})")
        return 1

    series = sorted(need, key=lambda n: runs[n]["orifice_diameter"])
    D = np.array([runs[n]["orifice_diameter"] for n in series])
    W = np.array([runs[n]["flow_rate_kg_s"] for n in series])
    S = np.array([runs[n]["flow_rate_std_kg_s"] for n in series])
    d = runs[series[0]]["particle_diameter"]
    rho_b = float(np.mean([runs[n]["bulk_density_kg_m3"] for n in series]))
    rho_b_spread = float(np.ptp([runs[n]["bulk_density_kg_m3"] for n in series]))
    g = runs[series[0]]["gravity"]

    fit = beverloo_fit(W, D, d, rho_b, g, k_range=(1.0, 2.0), n_k=401)

    # pure power-law slope of W vs D (no offset): expected ~2.5 once D >> d
    slope_D = float(np.polyfit(np.log(D), np.log(W), 1)[0])
    # slope of W vs the offset diameter at the fitted k
    Deff = D - fit["k"] * d
    slope_Deff = float(np.polyfit(np.log(Deff), np.log(W), 1)[0])

    # literature-coefficient prediction (no fitting at all)
    W_lit = np.array([beverloo_prediction(C_LIT, K_LIT, Di, d, rho_b, g) for Di in D])
    rel_lit = (W - W_lit) / W_lit
    # same with the *measured* bulk density of each run
    W_lit_run = np.array([beverloo_prediction(C_LIT, K_LIT, runs[n]["orifice_diameter"], d,
                                              runs[n]["bulk_density_kg_m3"], g)
                          for n in series])
    rel_lit_run = (W - W_lit_run) / W_lit_run

    summary: dict = {
        "fit": {"C": fit["C"], "k": fit["k"], "max_rel_err": fit["max_rel_err"],
                "ssr": fit["ssr"]},
        "literature": {"C": C_LIT, "k": K_LIT,
                       "C_range": list(C_LIT_RANGE), "k_range": list(K_LIT_RANGE)},
        "C_in_literature_range": bool(C_LIT_RANGE[0] <= fit["C"] <= C_LIT_RANGE[1]),
        "k_in_literature_range": bool(K_LIT_RANGE[0] <= fit["k"] <= K_LIT_RANGE[1]),
        "bulk_density_kg_m3": rho_b,
        "bulk_density_spread_kg_m3": rho_b_spread,
        "particle_diameter_m": d,
        "exponent_slope_vs_D": slope_D,
        "exponent_slope_vs_Deff": slope_Deff,
        "exponent_target": 2.5,
        "cases": {n: {"D_over_d": runs[n]["D_over_d"],
                      "W_kg_s": runs[n]["flow_rate_kg_s"],
                      "W_std": runs[n]["flow_rate_std_kg_s"],
                      "W_windows": runs[n]["flow_rate_windows_kg_s"],
                      "n_initial": runs[n]["n_initial"],
                      "n_discharged": runs[n]["n_discharged"],
                      "packing_fraction": runs[n]["packing_fraction"],
                      "wall_seconds": runs[n]["wall_seconds"],
                      "fill_height_m": runs[n]["fill_height"],
                      "H_over_D": runs[n]["fill_height"] / runs[n]["orifice_diameter"],
                      "D_silo_over_D": 2.0 * runs[n]["silo_radius"] / runs[n]["orifice_diameter"],
                      "W_lit": float(wl)}
                 for n, wl in zip(series, W_lit)},
        "rel_err_vs_literature": dict(zip(series, rel_lit.round(6).tolist())),
        "rel_err_vs_literature_per_run_rho_b": dict(zip(series, rel_lit_run.round(6).tolist())),
        "fitted_max_rel_err": fit["max_rel_err"],
    }

    # ---- consistency checks that need sibling cases -----------------------
    checks = {}
    if "grav" in runs:
        g2 = runs["grav"]["gravity"]
        W2 = runs["grav"]["flow_rate_kg_s"]
        W1 = runs["D4"]["flow_rate_kg_s"]
        expected = math.sqrt(g2 / runs["D4"]["gravity"])
        checks["gravity"] = {"g": g2, "W": W2, "W_ref": W1,
                             "ratio": W2 / W1, "expected_ratio": expected,
                             "rel_err": (W2 / W1 - expected) / expected}
    if "head" in runs:
        W2 = runs["head"]["flow_rate_kg_s"]
        W1 = runs["D4"]["flow_rate_kg_s"]
        checks["head"] = {"fill_height": runs["head"]["fill_height"],
                          "fill_height_ref": runs["D4"]["fill_height"],
                          "W": W2, "W_ref": W1,
                          "rel_change": (W2 - W1) / W1}
    summary["checks"] = checks

    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, "summary.json"), "w", encoding="utf8") as fh:
        json.dump(summary, fh, indent=2)

    # ---- markdown table ---------------------------------------------------
    print("\n| case | D/d | H/D | n0 | discharged | rho_b (kg/m3) | W (kg/s) | "
          "W_lit (kg/s) | rel.err vs lit. |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in series:
        c = summary["cases"][n]
        print(f"| {n} | {c['D_over_d']:.2f} | {c['H_over_D']:.2f} | {c['n_initial']} | "
              f"{c['n_discharged']} | {runs[n]['bulk_density_kg_m3']:.0f} | "
              f"{c['W_kg_s']:.4f} +- {c['W_std']:.4f} | {c['W_lit']:.4f} | "
              f"{100 * summary['rel_err_vs_literature'][n]:+.1f}% |")
    print(f"\nfitted C = {fit['C']:.4f}  k = {fit['k']:.4f}  "
          f"(literature C = {C_LIT}, k = {K_LIT})")
    print(f"slope log W vs log D      = {slope_D:.3f} (target 2.5)")
    print(f"slope log W vs log(D-k d) = {slope_Deff:.3f} (target 2.5)")
    for name, c in checks.items():
        print(f"{name}: {c}")

    # ---- figures ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"matplotlib unavailable ({exc}); figures skipped")
        return 0

    os.makedirs(FIGS, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    # fig 1: W vs D/d with the fitted and the literature Beverloo curves
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    Dd = np.linspace(D.min() * 0.93, D.max() * 1.06, 200)
    ax.errorbar(D / d, W, yerr=S, fmt="o", ms=6, capsize=3, color="C0",
                label="DEM (this work)")
    ax.plot(Dd / d, [beverloo_prediction(fit["C"], fit["k"], x, d, rho_b, g) for x in Dd],
            "-", color="C3", lw=1.4,
            label=f"Beverloo fit  C={fit['C']:.3f}, k={fit['k']:.2f}")
    ax.plot(Dd / d, [beverloo_prediction(C_LIT, K_LIT, x, d, rho_b, g) for x in Dd],
            "--", color="k", lw=1.1, label=f"literature  C={C_LIT}, k={K_LIT}")
    ax.set_xlabel("orifice ratio D/d")
    ax.set_ylabel("mass flow rate W (kg/s)")
    ax.set_title("Beverloo silo discharge: DEM vs literature law")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "m6296_silo_flow_vs_orifice.png"))
    plt.close(fig)

    # fig 2: log-log collapse on (D - k d)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.errorbar(Deff / d, W, yerr=S, fmt="o", ms=6, capsize=3, color="C0")
    ref = W[0] * (Deff / Deff[0]) ** 2.5
    ax.plot(Deff / d, ref, "--", color="k", lw=1.1, label="slope 5/2 reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("effective orifice ratio (D - k d)/d   [k fitted]")
    ax.set_ylabel("W (kg/s)")
    ax.set_title(f"orifice-law exponent: fitted slope {slope_Deff:.2f} vs 2.50")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "m6296_silo_loglog_exponent.png"))
    plt.close(fig)

    # fig 3: per-window flow series (stationarity / start-up transient)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, n in enumerate(sorted(runs)):
        if n in ("snapshot",):
            continue
        w = np.array(runs[n]["flow_rate_windows_kg_s"])
        if w.size == 0:
            continue
        label = f"{n} (D/d={runs[n]['D_over_d']:.0f}"
        if n == "grav":
            label += ", g/4"
        if n == "head":
            label += ", 1.75H"
        label += ")"
        ax.plot(np.arange(1, w.size + 1), w / max(runs[n]["flow_rate_kg_s"], 1e-12),
                "o-", ms=4, lw=1.0, label=label)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("flow-rate window index")
    ax.set_ylabel("W_window / W_mean")
    ax.set_title("per-window scatter (start-up transient and noise)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "m6296_silo_windows.png"))
    plt.close(fig)

    # fig 4: snapshot field (if the snapshot case was run)
    snap_path = os.path.join(RUNS, "snapshot.json")
    if os.path.exists(snap_path):
        with open(snap_path, encoding="utf8") as fh:
            snap = json.load(fh)
        pos = np.asarray(snap["positions"])
        spd = np.asarray(snap["speed"])
        fig, ax = plt.subplots(figsize=(5.6, 3.9))
        sc = ax.scatter(pos[:, 0] * 1e3, pos[:, 2] * 1e3, c=spd, s=6, cmap="viridis")
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_aspect("equal")
        ax.set_title(f"DEM silo snapshot, t = {snap['t']:.2f} s "
                     f"(colour: speed, max {spd.max():.2f} m/s)")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("|v| (m/s)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "m6296_silo_field_snapshot.png"))
        plt.close(fig)
        print("wrote snapshot field figure")

    print("wrote", FIGS, "figures and", os.path.join(RUNS, "summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())