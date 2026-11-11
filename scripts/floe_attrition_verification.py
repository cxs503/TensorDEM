"""Run criteria C1-C7 of node p472w2-N05 and write the artifacts + figures.

Outputs (all under ``artifacts/floe_attrition/``):

* ``criteria.json``  -- every measured number, threshold, pass flag and provenance
* ``figures/*.png``  -- the six figures of the report

Every number quoted in the report/markdown is injected from ``criteria.json``;
nothing is hand copied.

Usage::

    PYTHONPATH=src python scripts/floe_attrition_verification.py
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from tensordem.floe_attrition import (PRODUCTION, ArchardWear, InterlockingFriction,  # noqa: E402
                                      RoundedCube, RoundedPolygon, criteria,
                                      build_design_matrix, integrate_polygon_attrition,
                                      load_sphere_baseline, rounded_polygon_vertices)
from tensordem.floe_attrition.verification import (attrition_closed_form,  # noqa: E402
                                                   _fit_roundness_decay)

OUT = os.path.join(ROOT, "artifacts", "floe_attrition")
FIG = os.path.join(OUT, "figures")


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def figure_geometry(path: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for n in (3, 4, 5, 6):
        base = RoundedPolygon(n, 1.0, 0.0)
        fracs = np.linspace(0.0, 1.0, 60)
        axes[0].plot(fracs, [base.with_fillet(f * base.max_fillet_radius).wadell_roundness
                             for f in fracs], label=f"n = {n}")
    axes[0].set_xlabel("wear state  r / r_max")
    axes[0].set_ylabel("Wadell roundness  $R = 4\\pi A/P^2$")
    axes[0].set_title("(a) 2D roundness family")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)

    base = RoundedPolygon(4, 0.6, 0.0)
    for frac, colour in zip((0.02, 0.35, 0.70, 1.0),
                            ("#1f4e79", "#2e75b6", "#9dc3e6", "#f2b134")):
        v = rounded_polygon_vertices(base.with_fillet(frac * base.max_fillet_radius),
                                     arc_samples=24)
        v = np.vstack([v, v[0]])
        axes[1].plot(v[:, 0], v[:, 1], color=colour, lw=1.6,
                     label=f"R = {base.with_fillet(frac * base.max_fillet_radius).wadell_roundness:.3f}")
    axes[1].set_aspect("equal")
    axes[1].set_title("(b) floe shapes along the wear path (square, a = 0.6 m)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.3)

    cube = RoundedCube(1.0, 0.0)
    r = np.linspace(0.0, cube.max_radius, 60)
    psi = [cube.with_radius(x).sphericity for x in r]
    r3 = [cube.with_radius(x).wadell_roundness for x in r]
    axes[2].plot(r / cube.max_radius, psi, label="sphericity $\\Psi$")
    axes[2].plot(r / cube.max_radius, r3, "--", label="3D Wadell roundness $r/R_{ins}$")
    axes[2].set_xlabel("wear state  r / (a/2)")
    axes[2].set_ylabel("shape measure")
    axes[2].set_title("(c) 3D rounded-cube family (Steiner)")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_archard(path: str, c2: dict) -> str:
    poly = RoundedPolygon(PRODUCTION["n_sides"], PRODUCTION["edge_length"],
                          PRODUCTION["initial_fillet_ratio"] * PRODUCTION["edge_length"])
    wear = ArchardWear(PRODUCTION["wear_coefficient"], PRODUCTION["hardness_pa"])
    hist = integrate_polygon_attrition(poly, wear,
                                       pressure=PRODUCTION["pressure_pa"],
                                       slip_per_step=PRODUCTION["slip_per_transit"],
                                       n_steps=PRODUCTION["n_transits"])
    a = hist.arrays()
    s = a["sliding"]
    closed = attrition_closed_form(poly, wear, PRODUCTION["pressure_pa"], s)
    classic = wear.worn_volume(PRODUCTION["pressure_pa"] * poly.contact_patch_width, s)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(s, a["worn_geometric"], label="this module (exact geometry inversion)",
                 color="#1f4e79", lw=2.0)
    axes[0].plot(s, closed, "--", label="Archard closed form (patch feedback)",
                 color="#f2b134", lw=1.6)
    axes[0].plot(s, classic, ":", label="classical Archard (constant patch)",
                 color="#c00000", lw=1.6)
    axes[0].set_xlabel("sliding distance  s [m]")
    axes[0].set_ylabel("worn area per unit depth  [m$^2$]")
    axes[0].set_title("(a) corner wear vs sliding distance")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.3)

    err = np.abs(a["worn_geometric"][1:] - closed[1:]) / np.maximum(closed[1:], 1e-30)
    err_classic = np.abs(a["worn_geometric"] - classic) / np.maximum(classic, 1e-30)
    axes[1].loglog(s[1:], np.maximum(err, 1e-18), color="#1f4e79",
                   label="vs patch-feedback closed form")
    axes[1].axhline(c2["threshold"], color="#c00000", ls="--", label="10 % acceptance")
    axes[1].set_xlabel("sliding distance  s [m]")
    axes[1].set_ylabel("relative error  [-]")
    axes[1].set_title("(b) numerical error of the wear integration")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, float(err_classic[-1])


def figure_decay_law(path: str, c3: dict) -> str:
    poly = RoundedPolygon(PRODUCTION["n_sides"], PRODUCTION["edge_length"],
                          PRODUCTION["initial_fillet_ratio"] * PRODUCTION["edge_length"])
    wear = ArchardWear(PRODUCTION["wear_coefficient"], PRODUCTION["hardness_pa"])
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(c3["rows"])))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for colour, row in zip(colours, c3["rows"]):
        hist = integrate_polygon_attrition(poly, wear, pressure=row["pressure"],
                                           slip_per_step=PRODUCTION["slip_per_transit"],
                                           n_steps=4000)
        a = hist.arrays()
        axes[0].plot(a["sliding"], a["roundness"], color=colour, lw=1.8,
                     label=f"p = {row['pressure']/1e5:.1f}e5 Pa")
        fit = _fit_roundness_decay(a["sliding"], a["roundness"])
        sel = a["sliding"] <= fit["window_end_sliding"]
        model = 1.0 - (1.0 - a["roundness"][0]) * np.exp(
            -(a["sliding"][sel] / fit["tau"]) ** fit["beta"])
        axes[0].plot(a["sliding"][sel], model, "k:", lw=1.0)
    axes[0].set_xlabel("sliding distance  s [m]")
    axes[0].set_ylabel("Wadell roundness  R")
    axes[0].set_title("(a) roundness evolution (dots: fit window end)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(alpha=0.3)

    p = np.array([r["pressure"] for r in c3["rows"]])
    tau = np.array([r["tau_sliding_m"] for r in c3["rows"]])
    axes[1].loglog(p, tau, "o", color="#1f4e79", label="fitted relaxation length $\\tau$")
    ref = tau[0] * (p / p[0]) ** PRODUCTION["tau_pressure_exponent"]
    axes[1].loglog(p, ref, "--", color="#c00000",
                   label=f"pre-registered slope {PRODUCTION['tau_pressure_exponent']:+.0f}")
    axes[1].set_xlabel("corner contact pressure  p [Pa]")
    axes[1].set_ylabel("$\\tau$  [m of sliding]")
    axes[1].set_title(f"(b) measured exponent {c3['tau_pressure_exponent']:+.4f} "
                      f"(deviation {100*c3['tau_exponent_deviation']:.2f} %)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_design(path: str, c5: dict) -> str:
    cells = build_design_matrix(PRODUCTION["roundness_levels"],
                                PRODUCTION["concentrations"],
                                replicates=PRODUCTION["replicates"],
                                edge_length=PRODUCTION["edge_length"])
    anova = c5["anova"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    colours = plt.cm.plasma(np.linspace(0.1, 0.8, len(PRODUCTION["concentrations"])))
    for colour, c in zip(colours, PRODUCTION["concentrations"]):
        sub = [cell for cell in cells if abs(cell.concentration - c) < 1e-12]
        means = [np.mean([s.mu_effective for s in sub
                          if abs(s.roundness_level - lvl) < 1e-12])
                 for lvl in PRODUCTION["roundness_levels"]]
        sds = [np.std([s.mu_effective for s in sub
                       if abs(s.roundness_level - lvl) < 1e-12], ddof=1)
               for lvl in PRODUCTION["roundness_levels"]]
        axes[0].errorbar(PRODUCTION["roundness_levels"], means, yerr=sds, marker="o",
                         capsize=3, color=colour, label=f"C = {c:.2f}")
    axes[0].set_xlabel("Wadell roundness level  R")
    axes[0].set_ylabel("$\\mu_{eff}$  [-]")
    axes[0].set_title("(a) interaction plot (replicate mean $\\pm$ s.d.)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)

    rows = anova["table"]
    labels = [r["source"] for r in rows if r["source"] != "residual"]
    eta = [100.0 * r["eta_squared"] for r in rows if r["source"] != "residual"]
    axes[1].bar(labels, eta, color=["#1f4e79", "#f2b134", "#c00000"])
    axes[1].set_ylabel("share of total variance  $\\eta^2$ [%]")
    axes[1].set_title("(b) ANOVA effect sizes (3x3, m = %d)" % anova["replicates"])
    for i, (value, row) in enumerate(zip(eta, rows)):
        axes[1].text(i, value, f"{value:.1f} %\np = {row['p']:.1e}", ha="center",
                     va="bottom", fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_resistance(path: str, c6: dict) -> str:
    curves = c6["curves"]
    n = np.asarray(curves["transits"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    axes[0].plot(n, np.asarray(curves["resistance"]) / 1e3, color="#1f4e79", lw=2.0,
                 label="rounding + consolidation")
    axes[0].plot(n, np.asarray(curves["resistance_rounding_only"]) / 1e3, "--",
                 color="#2e75b6", label="rounding only")
    axes[0].plot(n, np.asarray(curves["resistance_consolidation_only"]) / 1e3, ":",
                 color="#c00000", label="consolidation only")
    ax2 = axes[0].twinx()
    ax2.plot(n, np.asarray(curves["contact_energy"]) / 1e6, color="#f2b134", lw=1.2,
             alpha=0.8)
    ax2.set_ylabel("cumulative contact energy  [MJ/m]")
    axes[0].set_xlabel("transits  N")
    axes[0].set_ylabel("ice resistance  [kN/m of waterline]")
    axes[0].set_title("(a) multi-pass resistance decay (scenario C = %.2f)"
                      % c6["scenario"]["concentration"])
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].grid(alpha=0.3)

    shares = c6["shares"]
    vals = [shares["share_rounding_pct"], shares["share_consolidation_pct"]]
    axes[1].bar(["rounding\n(attrition)", "consolidation\n(freeze bonds)"], vals,
                color=["#2e75b6", "#c00000"])
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_ylabel("share of the net resistance change  [%]")
    axes[1].set_title("(b) shares close to %.6f %% (residual %.1e)"
                      % (shares["share_sum_pct"], c6["closure_residual"]))
    for i, value in enumerate(vals):
        axes[1].text(i, value, f"{value:+.1f} %", ha="center",
                     va="bottom" if value >= 0 else "top", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_sphere_limit(path: str, c4: dict) -> str:
    baseline = load_sphere_baseline()
    model = InterlockingFriction()
    conc = np.linspace(baseline.bracket[0], baseline.bracket[1], 60)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for frac in (0.0, 0.5, 1.0):
        base = RoundedPolygon(4, 0.6, 0.0)
        idx = base.with_fillet(frac * base.max_fillet_radius).interlocking_index
        axes[0].plot(conc, [model.mu(idx, c, baseline) for c in conc],
                     label=f"I = {idx:.3f}")
    axes[0].plot(conc, [baseline.mu(c) for c in conc], "k--", lw=1.4,
                 label="m9375 sphere baseline")
    axes[0].set_xlabel("ice concentration  C")
    axes[0].set_ylabel("$\\mu_{eff}$  [-]")
    axes[0].set_title("(a) closure at three angularities")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.3)

    rows = c4["rows"]
    c = [r["concentration"] for r in rows]
    width = 0.26
    for k, (key, label, colour) in enumerate((
            ("mu_sphere_baseline", "sphere baseline", "#1f4e79"),
            ("mu_near_circle", "near circle (R = 1 - 1e-6)", "#2e75b6"),
            ("mu_after_full_attrition", "after full attrition", "#f2b134"))):
        axes[1].bar(np.asarray(c) + (k - 1) * width, [r[key] for r in rows], width,
                    color=colour, label=label)
    axes[1].set_xlabel("ice concentration  C")
    axes[1].set_ylabel("$\\mu_{eff}$  [-]")
    axes[1].set_title("(b) C4: the R -> 1 limit returns the baseline "
                      "(max rel. error %.1e)" % c4["worst_relative_error"])
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    t0 = time.time()
    payload = criteria()
    payload["provenance"] = {
        "node": "p472w2-N05",
        "git_revision": git_revision(),
        "generated_by": "scripts/floe_attrition_verification.py",
        "generated_at_unix": time.time(),
        "wall_seconds": time.time() - t0,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sphere_baseline_source": load_sphere_baseline().source,
    }
    figures = {
        "fig1_geometry": figure_geometry(os.path.join(FIG, "fig1_geometry.png")),
        "fig3_decay_law": figure_decay_law(os.path.join(FIG, "fig3_decay_law.png"),
                                           payload["outputs"]["C3"]),
        "fig4_design": figure_design(os.path.join(FIG, "fig4_design.png"),
                                     payload["outputs"]["C5"]),
        "fig5_resistance": figure_resistance(os.path.join(FIG, "fig5_resistance.png"),
                                             payload["outputs"]["C6"]),
        "fig6_sphere_limit": figure_sphere_limit(os.path.join(FIG, "fig6_sphere_limit.png"),
                                                 payload["outputs"]["C4"]),
    }
    fig2, classic_under = figure_archard(os.path.join(FIG, "fig2_archard.png"),
                                         payload["outputs"]["C2"])
    figures["fig2_archard"] = fig2
    payload["outputs"]["C2"]["classical_formula_underestimate_pct_figure"] = classic_under
    payload["figures"] = figures
    payload["provenance"]["wall_seconds"] = time.time() - t0

    path = os.path.join(OUT, "criteria.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=float)

    for name, block in payload["outputs"].items():
        print(f"{name:4s} {'PASS' if block['passed'] else 'FAIL'}")
    print("all_passed:", payload["all_passed"])
    print("wrote", path)
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())