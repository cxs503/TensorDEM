#!/usr/bin/env python3
"""AUDIT-RUNTEST-6 (m14498): figures for the run-evidence report.

All curves are read from the JSON artifacts produced by the real runs; nothing is
re-typed by hand.

fig1_resistance_three_way.png  -- DEM vs Lindqvist vs measured over the 12
                                  full-scale operating points, plus the +-15%
                                  band the acceptance criterion claims.
fig2_sobol_estimator.png       -- (a) S1 vs ST per parameter (must satisfy
                                  ST >= S1), (b) sum(S1) across seeds/sample
                                  counts against the <= 1 bound.
fig3_convergence_order.png     -- log-log residual vs dt for the ledger channels
                                  with fitted slopes.

Run: ``PYTHONPATH=src python3 scripts/audit_runtest6_figures.py``
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIG = RUNS / "figures"


def fig1() -> Path:
    fs = json.loads((RUNS / "fullscale" / "results.json").read_text())
    cmp_ = fs["three_way_comparison"]
    ids = [c["id"] for c in cmp_]
    dem = np.array([c["R_DEM_kN"] for c in cmp_])
    lin = np.array([c["R_Lindqvist_kN"] for c in cmp_])
    mea = np.array([c["R_measured_kN"] for c in cmp_])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    x = np.arange(len(ids))
    ax.plot(x, lin, "s--", color="#888", label="Lindqvist (analytic)")
    ax.plot(x, dem, "o-", color="#c0392b", label="model as shipped ('DEM')")
    ax.plot(x, mea, "^", color="#2c3e50", label="full-scale measured")
    ax.set_xticks(x, ids, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("resistance R [kN]")
    ax.set_title("brash-ice straight-ahead resistance, 12 operating points")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    eL = np.array([c["err_DEM_vs_Lindqvist"] for c in cmp_]) * 100
    eM = np.array([c["err_DEM_vs_measured"] for c in cmp_]) * 100
    w = 0.4
    ax.bar(x - w / 2, eL, w, color="#7f8c8d", label="model vs Lindqvist")
    ax.bar(x + w / 2, eM, w, color="#e67e22", label="model vs measured")
    ax.axhline(-15, color="#c0392b", ls="--", lw=1,
               label="criterion (a) ±15% (checked on point 1 only)")
    ax.axhline(-20, color="#2980b9", ls=":", lw=1, label="criterion (b) mean ±20%")
    ax.set_xticks(x, ids, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("relative error [%]")
    ax.set_title(f"11/12 points miss the Lindqvist 15% band; mean {np.abs(eL).mean():.1f}%")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG / "fig1_resistance_three_way.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig2() -> Path:
    sb = json.loads((RUNS / "sobol_probe" / "results.json").read_text())
    prod = sb["production_n500"]
    names = list(prod["S1"])
    s1 = [prod["S1"][n] for n in names]
    st = [prod["ST_raw"][n] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, s1, 0.4, label="S1 (first order)", color="#2c7fb8")
    ax.bar(x + 0.2, st, 0.4, label="ST raw (total order)", color="#e6550d")
    ax.set_xticks(x, names)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Sobol index (n=500, seed=42)")
    ax.set_title("ST < S1 for 3 of 4 parameters (impossible for independent inputs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    scan = sb["estimator_convergence"]
    labels = [f"n={r['n_samples']}\nseed={r['seed']}" for r in scan]
    s1s = [r["S1_sum"] for r in scan]
    sts = [r["raw_ST_sum"] for r in scan]
    xi = np.arange(len(scan))
    ax.plot(xi, s1s, "o-", color="#2c7fb8", label="ΣS1 (must be ≤ 1)")
    ax.plot(xi, sts, "s-", color="#e6550d", label="ΣST raw (must be 1 ± 0.01)")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.axhspan(0.99, 1.01, color="#999", alpha=0.2, label="criterion (c) band")
    ax.set_xticks(xi, labels, fontsize=7)
    ax.set_title("estimator not converged: ΣS1 spans 0.67–1.73 across seeds")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig2_sobol_estimator.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig3() -> Path:
    cv = json.loads((RUNS / "convergence" / "results.json").read_text())
    hertz = cv["sweeps"]["hertz_lossless"]
    dt = np.array(hertz["dt"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    for key, style, color in (
        ("residual_explicit", "o-", "#e6550d"),
        ("residual_staggered_power", "s--", "#7f8c8d"),
        ("residual_average_velocity", "^:", "#2c7fb8"),
        ("residual_state", "d-", "#31a354"),
    ):
        y = np.array(hertz[key])
        ax.loglog(dt, np.maximum(y, 1e-17), style, color=color, label=key.replace("residual_", ""))
    ax.set_xlabel("dt [s]")
    ax.set_ylabel("residual / E0")
    ax.set_title("ledger closure residuals vs dt (Hertz, lossless)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    fits = hertz["fits_library"]
    keys = [k for k in fits if fits[k]]
    slopes = [fits[k]["slope"] for k in keys]
    r2 = [fits[k]["r2"] for k in keys]
    ax.barh(range(len(keys)), slopes, color="#2c7fb8")
    ax.set_yticks(range(len(keys)), [k.replace("residual_", "") for k in keys], fontsize=8)
    for i, (s, r) in enumerate(zip(slopes, r2)):
        ax.text(s + 0.05, i, f"{s:+.3f} (r²={r:.4f})", va="center", fontsize=7)
    ax.axvline(1.0, color="#c0392b", ls="--", lw=1, label="documented O(dt) = 1")
    ax.set_xlabel("fitted slope")
    ax.set_title("measured convergence order per convention")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIG / "fig3_convergence_order.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig4() -> Path:
    """Steady broken-ice transit: resistance time history + energy ledger split."""
    case = json.loads((RUNS / "ice_transit_cases" / "standard.json").read_text())
    series = case["series"]
    t = [r["time_s"] for r in series]
    r = [r["resistance_kN"] for r in series]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.plot(t, r, lw=0.6, color="#2c3e50", alpha=0.6, label="instantaneous")
    import numpy as np
    w = np.array(case["steady_window_s"])
    ax.axvspan(w[0], w[1], color="#f0b27a", alpha=0.25, label="declared steady window")
    ax.axhline(case["mean_kN"], color="#c0392b", ls="--", lw=1,
               label=f"mean {case['mean_kN']:.1f} kN")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("resistance [kN]")
    ax.set_title(f"steady brash-ice transit, 63 floes, 10 ship lengths "
                 f"(drift {case['stationarity_drift']:+.3f} vs gate 0.15)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labs = ["thrust\nwork", "numerical\ninjection", "removed", "dissipation",
            "Δstored", "residual"]
    diss = sum(case["dissipation_kJ"].values())
    stored = sum(case["energy_stored_change_kJ"].values())
    vals = [case["thrust_work_kJ"], case["numerical_injection_kJ"], case["removed_kJ"],
            diss, stored, case["ledger_residual_kJ"]]
    colors = ["#2c3e50", "#c0392b", "#7f8c8d", "#e67e22", "#2980b9", "#8e44ad"]
    ax.bar(range(len(vals)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v/1e3:.1f} MJ", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(vals)), labs, fontsize=7)
    ax.set_ylabel("[kJ]")
    ax.set_title(f"energy ledger: residual {100*case['ledger_residual_rel']:.2f}% of thrust work "
                 f"(gate 10%)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG / "fig4_ice_transit_ledger.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    if (RUNS / "fullscale" / "results.json").exists():
        made.append(fig1())
    if (RUNS / "sobol_probe" / "results.json").exists():
        made.append(fig2())
    if (RUNS / "convergence" / "results.json").exists():
        made.append(fig3())
    if (RUNS / "ice_transit_cases" / "standard.json").exists():
        made.append(fig4())
    for p in made:
        print(p, p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())