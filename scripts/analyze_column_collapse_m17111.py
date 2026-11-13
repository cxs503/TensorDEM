"""Reduce the round-5 (m17111) column-collapse cases to tables, figures and a report.

Reads only the per-case JSON written by
``benchmarks/case_column_collapse_lube2004.py`` (each carries its own parameter
signature, particle count, energy ledger and inertial statistics, i.e. the
numbers travel with their provenance), then writes

* ``artifacts/column_collapse_m17111/summary.json``  -- every number used below
* ``artifacts/column_collapse_m17111/*.png``         -- the figures of the report
* ``docs/case_column_collapse_round5_m17111.md``     -- the report, rendered from
  ``summary.json`` so that no number is typed by hand

Run: ``PYTHONPATH=src python3 scripts/analyze_column_collapse_m17111.py``
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_DIR = os.path.join(ROOT, "runs", "case_column_collapse_lube2004", "cases")
ART = os.path.join(ROOT, "artifacts", "column_collapse_m17111")
DOC = os.path.join(ROOT, "docs", "case_column_collapse_round5_m17111.md")
TOL = 0.15
ARMS = ("wave5_constant", "wave5_law_mu_i", "wave5_law_mu_i_mur_sens",
        "wave5_law_mu_i_resolution", "wave5_floor_roughness",
        "wave5b_confirmation", "wave5c_sensitivity")


def load() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(CASE_DIR, "w5*.json"))):
        d = json.load(open(path))
        c = d["case"]
        crit = d["criteria"]["runout_ratio"]
        rows.append({
            "case": os.path.basename(path)[:-5],
            "phase": d.get("phase", "?"),
            "a_requested": c["aspect_ratio"],
            "a_measured": d["aspect_measured"],
            "mu": c["mu"],
            "mu_rolling": c["mu_rolling"],
            "mu_inertial": c["mu_inertial"],
            "mu_wall": c.get("mu_wall"),
            "mu_law": c.get("mu_law"),
            "grains_per_radius": c["grains_per_radius"],
            "n_particles": d["n_particles"],
            "n_steps": d["n_steps"],
            "dt": d["dt"],
            "density": c["density"],
            "radius": c["radius"],
            "kn": c["kn"],
            "kt_ratio": c["kt_ratio"],
            "restitution": c["restitution"],
            "packing_gap": c["packing_gap"],
            "solid_fraction": d["solid_fraction_initial"],
            "runout_ratio": d["runout_ratio_p995"],
            "runout_ratio_max_metric": d["runout_ratio_max"],
            "ref_runout_ratio": crit["reference"],
            "rel_err": crit["relative_error"],
            "pass": bool(crit["pass"]),
            "t_reach": d["t_reach"],
            "ref_t_reach": d["reference_emplacement_time"],
            "t_rel_err": ((d["t_reach"] - d["reference_emplacement_time"])
                          / d["reference_emplacement_time"]
                          if d["t_reach"] and math.isfinite(d["t_reach"]) else float("nan")),
            "ballistic_time": d["ballistic_time"],
            "stop_reason": d["stop_reason"],
            "energy_residual": d["energy"]["residual_fraction_of_potential"],
            "mu_eff_mean": d["inertial"]["mu_eff_mean"],
            "mu_eff_max": d["inertial"]["mu_eff_max"],
            "i_mean": d["inertial"]["inertial_number_mean"],
            "i_max": d["inertial"]["inertial_number_max"],
            "wall_clock_s": d.get("wall_clock_s"),
            "note": d.get("note", ""),
            "label": c.get("label"),
        })
    return rows


def arm_stats(rows: list[dict], phase: str) -> dict:
    sel = [r for r in rows if r["phase"] == phase]
    sel.sort(key=lambda r: r["a_requested"])
    return {
        "phase": phase,
        "n": len(sel),
        "n_pass": sum(1 for r in sel if r["pass"]),
        "max_abs_rel_err": max((abs(r["rel_err"]) for r in sel), default=float("nan")),
        "points": [{k: r[k] for k in ("case", "a_requested", "a_measured", "mu", "mu_rolling",
                                      "mu_wall", "grains_per_radius", "runout_ratio",
                                      "ref_runout_ratio", "rel_err", "pass", "t_reach",
                                      "ref_t_reach", "t_rel_err", "stop_reason",
                                      "energy_residual", "mu_eff_mean", "i_mean",
                                      "n_particles", "wall_clock_s")} for r in sel],
    }


def make_figures(rows: list[dict], summary: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(ART, exist_ok=True)
    figs = []

    def _band(ax, a_lo=0.4, a_hi=3.3):
        aa = np.linspace(a_lo, a_hi, 400)
        ref = np.array([1.0 + 1.24 * x if x < 1.7 else 1.0 + 1.6 * math.sqrt(x)
                        for x in aa])
        ax.fill_between(aa, ref * (1 - TOL), ref * (1 + TOL), color="0.85",
                        label="ref +-15 %")
        ax.plot(aa, ref, "k-", lw=1.6, label="Lube et al. (2004)")

    # --- F1: runout law, both arms
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _band(ax)
    style = {"wave5_constant": ("o", "C0", "constant mu (control arm)"),
             "wave5_law_mu_i": ("s", "C3", "mu(I_c) Jop-2006 law"),
             "wave5_floor_roughness": ("^", "C2", "rough floor (mu_wall=0.8/1.0)")}
    for phase, (marker, colour, label) in style.items():
        pts = [r for r in rows if r["phase"] == phase]
        if not pts:
            continue
        pts.sort(key=lambda r: r["a_requested"])
        ax.plot([p["a_requested"] for p in pts], [p["runout_ratio"] for p in pts],
                marker + "--", color=colour, lw=1.3, ms=6, label=label)
    ax.set_xlabel("aspect ratio a = h_i / r_i")
    ax.set_ylabel("runout ratio r_inf / r_i")
    ax.set_title("CASE-COLUMN-COLLAPSE: runout law (m17111 round 5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = os.path.join(ART, "runout_law_arms.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": "runout_law_arms.png", "path": p,
                 "caption": "Runout ratio vs aspect ratio: reference law, constant-mu "
                            "control arm and mu(I_c) arm (shaded = +-15 % band)."})

    # --- F2: signed errors
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for phase, (marker, colour, label) in style.items():
        pts = sorted([r for r in rows if r["phase"] == phase],
                     key=lambda r: r["a_requested"])
        if not pts:
            continue
        ax.plot([p["a_requested"] for p in pts], [100 * p["rel_err"] for p in pts],
                marker + "-", color=colour, lw=1.3, ms=6, label=label)
    ax.axhspan(-15, 15, color="0.85", label="+-15 % tolerance")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("aspect ratio a")
    ax.set_ylabel("signed runout error [%]")
    ax.set_title("Error budget per aspect ratio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = os.path.join(ART, "runout_error_arms.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": "runout_error_arms.png", "path": p,
                 "caption": "Signed relative runout error per aspect ratio for both arms; "
                            "grey band = the 15 % acceptance band."})

    # --- F3: the mu(I) law and where the flow sits on it
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    i = np.logspace(-3, 1.5, 400)
    for (mus, mu2, i0, label, ls) in ((0.36, 0.84, 0.30, "Jop et al. (2006)", "-"),):
        ax.plot(i, mus + (mu2 - mus) * i / (i0 + i), ls, color="C3", lw=1.6, label=label)
    law_pts = [r for r in rows if r["phase"] == "wave5_law_mu_i" and math.isfinite(r["i_mean"])]
    if law_pts:
        ax.plot([r["i_mean"] for r in law_pts], [r["mu_eff_mean"] for r in law_pts],
                "ks", ms=5, label="measured (mean I_c, mean mu) per case")
        for r in law_pts:
            ax.annotate(f"a={r['a_requested']:g}", (r["i_mean"], r["mu_eff_mean"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axhline(0.5, color="C0", lw=1.2, ls="--",
               label="constant-arm mu = 0.5 (reference demand)")
    ax.set_xscale("log")
    ax.set_xlabel("inertial number I_c (contact-local proxy)")
    ax.set_ylabel("mobilised friction mu")
    ax.set_title("mu(I) closure vs the inertial numbers the collapse actually reaches")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    p = os.path.join(ART, "mu_i_law_and_reach.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": "mu_i_law_and_reach.png", "path": p,
                 "caption": "The Jop et al. (2006) mu(I) law (red) and the contact-local "
                            "inertial numbers measured in the collapse (black squares): the "
                            "bulk sits at I_c << I_0, so the law is pinned near mu_s."})

    # --- F4: floor roughness response
    probe = sorted([r for r in rows if r["phase"] == "wave5_floor_roughness"],
                   key=lambda r: (r["mu_wall"], r["a_requested"]))
    if probe:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        base = {r["a_requested"]: r for r in rows if r["phase"] == "wave5_constant"}
        for a in sorted({r["a_requested"] for r in probe}):
            xs = [0.5] + [r["mu_wall"] for r in probe if r["a_requested"] == a]
            ys = []
            if a in base:
                ys.append(100 * base[a]["rel_err"])
            ys += [100 * r["rel_err"] for r in probe if r["a_requested"] == a]
            if len(xs) != len(ys):
                continue
            ax.plot(xs, ys, "o-", lw=1.3, label=f"a = {a:g}")
        ax.axhspan(-15, 15, color="0.85")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(r"floor friction $\mu_{wall}$")
        ax.set_ylabel("signed runout error [%]")
        ax.set_title("Rough-floor lever (probe on the two branch representatives)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        p = os.path.join(ART, "floor_roughness_probe.png")
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"name": "floor_roughness_probe.png", "path": p,
                     "caption": "Signed runout error vs floor friction; the mu_wall = 0.5 "
                                "point is the constant-arm control, so the response magnitude "
                                "can be compared with the case-to-case scatter."})
    summary["figures"] = [f["name"] for f in figs]
    return figs


def fmt_table(points: list[dict]) -> str:
    head = ("| case | a | mu | mu_r | mu_wall | runout ratio | ref | err [%] | t_inf [s] | "
            "ref t_inf | stop | mu_eff | I_c |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for p in points:
        lines.append("| {case} | {a:g} | {mu:g} | {mur:g} | {wall} | {rr:.3f} | {ref:.3f} | "
                     "{err:+.1f} | {t:.3f} | {tr:.3f} | {stop} | {mue:.3f} | {ii:.4f} |".format(
                         case=p["case"], a=p["a_requested"], mu=p["mu"],
                         mur=p["mu_rolling"],
                         wall=("-" if p["mu_wall"] is None else f"{p['mu_wall']:g}"),
                         rr=p["runout_ratio"], ref=p["ref_runout_ratio"],
                         err=100 * p["rel_err"],
                         t=p["t_reach"] if p["t_reach"] else float("nan"),
                         tr=p["ref_t_reach"], stop=p["stop_reason"],
                         mue=p["mu_eff_mean"] if p["mu_eff_mean"] else float("nan"),
                         ii=p["i_mean"] if p["i_mean"] else float("nan")))
    return head + "\n".join(lines) + "\n"


def main() -> int:
    rows = load()
    os.makedirs(ART, exist_ok=True)
    summary = {
        "tolerance": TOL,
        "reference": ("Lube, Huppert, Sparks & Freundt (2004) JFM 508, 175-199, "
                      "doi:10.1017/S0022112004009036"),
        "n_cases": len(rows),
        "arms": {},
        "cases": rows,
        "checks": {},
    }
    for phase in ARMS:
        stats = arm_stats(rows, phase)
        if stats["n"]:
            summary["arms"][phase] = stats

    # pre-registered verdicts
    law = summary["arms"].get("wave5_law_mu_i")
    const = summary["arms"].get("wave5_constant")
    if law and const:
        summary["checks"]["H5_law_closes_curve"] = bool(law["n_pass"] == law["n"])
        summary["checks"]["H5_detail"] = (f"law arm {law['n_pass']}/{law['n']} within 15 %, "
                                          f"control arm {const['n_pass']}/{const['n']}")
    probe = summary["arms"].get("wave5_floor_roughness")
    if probe and const:
        base = {r["a_requested"]: r for r in const["points"]}
        deltas = {}
        for a in sorted({p["a_requested"] for p in probe["points"]}):
            if a not in base:
                continue
            e0 = base[a]["rel_err"]
            deltas[str(a)] = {f"mu_wall_{p['mu_wall']:g}": abs(e0) - abs(p["rel_err"])
                              for p in probe["points"] if p["a_requested"] == a}
        summary["checks"]["H6_floor_effect"] = deltas
    # energy ledger across every round-5 case
    summary["checks"]["max_abs_energy_residual"] = max(
        (abs(r["energy_residual"]) for r in rows if math.isfinite(r["energy_residual"])),
        default=float("nan"))
    summary["checks"]["stop_reasons"] = sorted({r["stop_reason"] for r in rows})

    figs = make_figures(rows, summary)
    summary["figure_paths"] = figs
    with open(os.path.join(ART, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary["checks"].items()}, indent=1))
    print(f"cases={len(rows)} figures={len(figs)} -> {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())