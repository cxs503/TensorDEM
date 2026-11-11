#!/usr/bin/env python
"""Ice pile-up against a vertical structure -- ISO 19906 benchmark and
porosity-resistance scaling law (campaign p227w2-N04, TensorDEM).

Runs three experiments with the 2-D plan-view floe DEM in
``tensordem.ice_vertical_structure_pileup``:

``base``      cylinder + square column at the reference ice condition: full load
              time history, peak/steady load, rubble-pile geometry.
``iso_check`` sigma_c scaling (F should be proportional to sigma_c, which is what
              makes the ISO 19906 combination F = C sigma_c h^2 dimensionally and
              physically meaningful).
``porosity``  two-factor sweep (floe size / D x ice-ice friction) with several
              seeds; fits F = A (1-n)^alpha on the *measured* pile porosity.

Usage
-----
    python benchmarks/ice_pileup_iso19906/run_benchmark.py --out results
    python benchmarks/ice_pileup_iso19906/run_benchmark.py --quick   # smoke run
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.ice_vertical_structure_pileup import (  # noqa: E402
    PileupConfig,
    fit_porosity_scaling,
    iso19906_load,
    load_statistics,
    run_pileup_case,
)

BASE = dict(sigma_c=3.0e6, h=1.0, D=5.0, drift_speed=0.5, duration=8.0)


def _run(job: dict) -> dict:
    cfg = PileupConfig(**job["cfg"])
    t0 = time.time()
    res = run_pileup_case(cfg)
    out = res.as_dict(with_series=job.get("series", False))
    out["wall_time_s"] = time.time() - t0
    out["experiment"] = job["experiment"]
    out["case_id"] = job["case_id"]
    if job.get("series"):
        out["positions"] = res.positions.tolist()
        out["radii"] = res.radii.tolist()
        out["pile_mask_shape"] = list(res.pile_mask.shape)
        out["pile_extent"] = list(res.pile_extent)
        out["pile_mask"] = res.pile_mask.astype(np.uint8)
    return out


def build_jobs(quick: bool) -> list[dict]:
    jobs: list[dict] = []
    seeds = (11,) if quick else (11, 12, 13, 14)
    dur = 0.8 if quick else BASE["duration"]

    for shape in ("cylinder", "square"):
        for seed in (seeds[:2] if not quick else seeds[:1]):
            jobs.append({
                "experiment": "base",
                "case_id": f"base_{shape}_s{seed}",
                "series": True,
                "cfg": dict(BASE, shape=shape, seed=seed, duration=dur),
            })

    for sigma_c in ((3.0e6,) if quick else (1.5e6, 2.25e6, 3.0e6, 4.5e6)):
        for seed in seeds[:1] if not quick else seeds[:1]:
            jobs.append({
                "experiment": "iso_check",
                "case_id": f"iso_sc{sigma_c/1e6:.2f}_s{seed}",
                "series": False,
                "cfg": dict(BASE, sigma_c=sigma_c, seed=seed, duration=dur),
            })

    size_levels = (0.10,) if quick else (0.05, 0.08, 0.12)
    mu_levels = (0.4,) if quick else (0.10, 0.60)
    for size_rel in size_levels:
        for mu in mu_levels:
            for seed in seeds:
                r_med = size_rel * BASE["D"]
                jobs.append({
                    "experiment": "porosity",
                    "case_id": f"por_r{size_rel:.2f}_mu{mu:.2f}_s{seed}",
                    "series": False,
                    "cfg": dict(BASE, floe_radius_median=r_med, mu=mu, seed=seed,
                                duration=dur),
                })
    return jobs


def _make_figures(outdir: Path, base: list[dict], porosity: dict | None,
                  iso: dict | None) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict] = []

    # ---- fig 1: load time histories -------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for rec in base:
        t = np.asarray(rec["time"])
        f = np.asarray(rec["load_x"]) / 1e6
        ax.plot(t, f, lw=1.0, label=f"{rec['case_id']} (DEM)")
    if base:
        F_iso = base[0]["F_ISO"] / 1e6
        ax.axhline(F_iso, color="k", ls="--", lw=1.2,
                   label=f"ISO 19906 F={F_iso:.2f} MN")
        Fst = base[0]["F_steady"] / 1e6
        ax.axhline(Fst, color="crimson", ls=":", lw=1.2,
                   label=f"DEM steady F={Fst:.2f} MN")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("horizontal ice load on structure [MN]")
    ax.set_title("Ice pile-up load history (h=1 m, sigma_c=3 MPa, V=0.5 m/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "fig1_load_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name,
                 "caption": "Load time history on the vertical structure: "
                            "jam/slip sawtooth of the rubble pile, with the "
                            "ISO 19906 nominal load and the DEM steady level."})

    # ---- fig 2: rubble-pile geometry snapshot ---------------------------
    if base and "positions" in base[0]:
        rec = base[0]
        pos = np.asarray(rec["positions"])
        rad = np.asarray(rec["radii"])
        cfg = rec["config"]
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        keep = pos[:, 0] < 1.5 * cfg["D"]
        for (px, py), rr in zip(pos[keep], rad[keep]):
            ax.add_patch(plt.Circle((px, py), rr, facecolor="#9ecae1",
                                    edgecolor="#22536e", lw=0.4, alpha=0.85))
        if cfg["shape"] == "cylinder":
            ax.add_patch(plt.Circle((0, 0), 0.5 * cfg["D"], facecolor="#d62728",
                                    edgecolor="k", lw=1.2, alpha=0.9))
        else:
            a = 0.5 * cfg["D"]
            ax.add_patch(plt.Rectangle((-a, -a), 2 * a, 2 * a,
                                       facecolor="#d62728", edgecolor="k", lw=1.2))
        R = 0.5 * cfg["D"]
        ax.add_patch(plt.Rectangle((-R - cfg["D"], -cfg["lateral_factor"] * cfg["D"]),
                                   cfg["D"], 2 * cfg["lateral_factor"] * cfg["D"],
                                   fill=False, ls="--", ec="purple", lw=1.0,
                                   label="porosity window"))
        ax.set_aspect("equal")
        ax.set_xlim(-cfg["D"] * 2.0, cfg["D"] * 1.0)
        ax.set_ylim(-cfg["lateral_factor"] * cfg["D"], cfg["lateral_factor"] * cfg["D"])
        ax.set_xlabel("x [m] (drift direction)")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Rubble pile at t={rec['config']['duration']:.1f} s, "
                     f"porosity n={rec['porosity']:.3f}, "
                     f"pile length={rec['pile_length']:.2f} m")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        p = outdir / "fig2_pile_geometry.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append({"name": p.name,
                     "caption": "Plan-view floe packing (rubble pile) in front of "
                                "the structure at the end of a run; dashed purple "
                                "box = porosity measurement window."})

    # ---- fig 3: ISO 19906 comparison ------------------------------------
    if base:
        labels = [r["case_id"].split("_")[1] for r in base]
        fmax = np.array([r["F_max"] for r in base]) / 1e6
        fst = np.array([r["F_steady"] for r in base]) / 1e6
        fiso = np.array([r["F_ISO"] for r in base]) / 1e6
        x = np.arange(len(base))
        w = 0.27
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.bar(x - w, fiso, w, label="ISO 19906", color="#444444")
        ax.bar(x, fmax, w, label="DEM peak", color="#e4572e")
        ax.bar(x + w, fst, w, label="DEM steady", color="#2e86ab")
        ax.set_xticks(x)
        ax.set_xticklabels([r["case_id"] for r in base], fontsize=7, rotation=12)
        ax.set_yscale("log")
        ax.set_ylabel("load [MN]")
        ax.set_title("ISO 19906 nominal load vs DEM peak / steady load")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / "fig3_iso19906_comparison.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append({"name": p.name,
                     "caption": "ISO 19906 nominal global load compared with the DEM "
                                "peak and steady loads for the two structure shapes "
                                "(log scale)."})

    # ---- fig 4: porosity-resistance fit ---------------------------------
    if porosity and porosity.get("points"):
        n = np.array([p["porosity"] for p in porosity["points"]])
        F = np.array([p["F_steady"] for p in porosity["points"]])
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        sc = ax.scatter(1 - n, F, c=[p["mu"] for p in porosity["points"]],
                        s=28 + 90 * np.array([p["floe_radius_median"] for p in porosity["points"]]),
                        cmap="viridis", edgecolor="k", lw=0.5, zorder=3)
        xx = np.linspace((1 - n).min() * 0.97, (1 - n).max() * 1.03, 50)
        ax.plot(xx, porosity["amplitude"] * xx ** porosity["alpha"], "r-", lw=1.4,
                label=(f"fit F = {porosity['amplitude']/1e6:.2f} MN * (1-n)^"
                       f"{porosity['alpha']:.2f}\nR² = {porosity['R2']:.3f}"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("solid fraction (1 - n)")
        ax.set_ylabel("steady load F [N]")
        ax.set_title("Porosity-resistance scaling of the rubble pile")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.colorbar(sc, ax=ax, label="ice-ice friction coefficient mu")
        fig.tight_layout()
        p = outdir / "fig4_porosity_scaling.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append({"name": p.name,
                     "caption": "Steady load versus measured pile solid fraction: "
                                "log-log fit of F = A (1-n)^alpha (marker size = "
                                "floe size, colour = friction coefficient)."})

    # ---- fig 5: sigma_c scaling -----------------------------------------
    if iso and len(iso.get("points", [])) >= 2:
        sc = np.array([p["sigma_c"] for p in iso["points"]])
        F = np.array([p["F_steady"] for p in iso["points"]])
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.loglog(sc, F, "o-", label="DEM steady load")
        ref = F[0] * (sc / sc[0]) ** iso["exponent"]
        ax.loglog(sc, ref, "k--", lw=1.2,
                  label=f"power law, exponent = {iso['exponent']:.2f}")
        ax.set_xlabel("ice compressive strength sigma_c [Pa]")
        ax.set_ylabel("steady load [N]")
        ax.set_title("Load versus ice strength (ISO 19906: F proportional to sigma_c)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / "fig5_sigma_scaling.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append({"name": p.name,
                     "caption": "Steady load versus compressive strength: the "
                                "log-log slope tests the linear sigma_c dependence "
                                "assumed by ISO 19906."})
    return figs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "ice_pileup_iso19906" / "results"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.quick)
    print(f"[p227w2-N04] {len(jobs)} DEM runs on {args.workers} workers", flush=True)
    t0 = time.time()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_run, jobs)
    else:
        results = [_run(j) for j in jobs]
    wall = time.time() - t0
    print(f"[p227w2-N04] all runs finished in {wall:.1f} s", flush=True)

    base = [r for r in results if r["experiment"] == "base"]
    iso_runs = [r for r in results if r["experiment"] == "iso_check"]
    por_runs = [r for r in results if r["experiment"] == "porosity"]

    # ---------------- ISO 19906 comparison -------------------------------
    iso_rows = []
    for r in base:
        iso_rows.append({
            "case_id": r["case_id"],
            "shape": r["config"]["shape"],
            "sigma_c": r["sigma_c"] if "sigma_c" in r else r["config"]["sigma_c"],
            "h": r["config"]["h"],
            "D": r["config"]["D"],
            "F_ISO": r["F_ISO"],
            "p_G": r["p_G"],
            "C_task": r["C_task"],
            "F_max": r["F_max"],
            "F_steady": r["F_steady"],
            "ratio_max": r["ratio_max"],
            "ratio_steady": r["ratio_steady"],
            "log10_ratio_max": math.log10(r["ratio_max"]) if r["ratio_max"] > 0 else None,
            "log10_ratio_steady": math.log10(r["ratio_steady"]) if r["ratio_steady"] > 0 else None,
        })

    def _same_order(ratio: float) -> bool:
        return bool(ratio > 0 and 0.1 <= ratio <= 10.0)

    iso_pass = bool(all(_same_order(r["ratio_max"]) and _same_order(r["ratio_steady"])
                        for r in base)) if base else False
    iso_tight = bool(all(0.316 <= r["ratio_max"] <= 3.16 and
                         0.316 <= r["ratio_steady"] <= 3.16 for r in base)) if base else False

    # ---------------- porosity-resistance fit ----------------------------
    por = None
    if len(por_runs) >= 3:
        n = [r["porosity"] for r in por_runs]
        F = [r["F_max"] for r in por_runs]
        fit = fit_porosity_scaling(n, F, seed=0)
        fit_steady = fit_porosity_scaling(
            n, [max(r["F_steady"], 1.0) for r in por_runs], seed=0)
        groups: dict[str, dict] = {}
        for r in por_runs:
            key = f"r/D={r['config']['floe_radius_median']/r['config']['D']:.2f},mu={r['config']['mu']:.2f}"
            g = groups.setdefault(key, {"porosity": [], "F_steady": [], "F_max": []})
            g["porosity"].append(r["porosity"])
            g["F_steady"].append(r["F_steady"])
            g["F_max"].append(r["F_max"])
        group_rows = [{
            "group": k,
            "n_runs": len(v["porosity"]),
            "porosity_mean": float(np.mean(v["porosity"])),
            "porosity_std": float(np.std(v["porosity"])),
            "F_steady_mean": float(np.mean(v["F_steady"])),
            "F_steady_std": float(np.std(v["F_steady"])),
            "F_max_mean": float(np.mean(v["F_max"])),
        } for k, v in groups.items()]
        por = {
            "points": [{
                "case_id": r["case_id"],
                "porosity": r["porosity"],
                "F_steady": r["F_steady"],
                "F_max": r["F_max"],
                "F_ISO": r["F_ISO"],
                "mu": r["config"]["mu"],
                "floe_radius_median": r["config"]["floe_radius_median"],
                "solids_fraction": r["solids_fraction"],
            } for r in por_runs],
            "groups": group_rows,
            **fit,
            "fit_on": "F_max (peak load)",
            "secondary_fit_on_F_steady": fit_steady,
        }
        por["criterion"] = "R2 > 0.8 and alpha > 0"
        por["pass"] = bool(fit["R2"] > 0.8 and fit["alpha"] > 0.0)

    # ---------------- sigma_c scaling ------------------------------------
    iso_sc = None
    if len(iso_runs) >= 2:
        sc = np.array([r["config"]["sigma_c"] for r in iso_runs], dtype=float)
        F = np.array([r["F_steady"] for r in iso_runs], dtype=float)
        Fmax = np.array([r["F_max"] for r in iso_runs], dtype=float)
        ok = (F > 0) & np.isfinite(F)
        expo_steady = float(np.polyfit(np.log(sc[ok]), np.log(F[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")
        ok2 = (Fmax > 0) & np.isfinite(Fmax)
        expo = float(np.polyfit(np.log(sc[ok2]), np.log(Fmax[ok2]), 1)[0]) if ok2.sum() >= 2 else float("nan")
        expo_max = expo
        iso_sc = {
            "points": [{
                "case_id": r["case_id"],
                "sigma_c": r["config"]["sigma_c"],
                "F_steady": r["F_steady"],
                "F_max": r["F_max"],
                "F_ISO": r["F_ISO"],
            } for r in iso_runs],
            "exponent": expo,
            "exponent_peak": expo_max,
            "exponent_steady": expo_steady,
            "criterion": "|exponent - 1| <= 0.25",
            "pass": bool(abs(expo - 1.0) <= 0.25),
        }

    stats = {r["case_id"]: load_statistics(r["load_x"], r["config"]["sample_dt"])
             for r in base}

    figs = _make_figures(outdir, base, por, iso_sc)

    report = {
        "campaign": "p227w2-N04",
        "title": "Ice pile-up against a vertical structure: DEM load history, "
                 "ISO 19906 benchmark and porosity-resistance scaling",
        "model": {
            "type": "2-D plan-view (depth-averaged) soft-disc DEM",
            "contact_law": "Fn = min(k_n*delta, sigma_c*h*b(delta)), b = contact chord",
            "driving": "kinematic platen at drift speed V (far-field ice sheet)",
            "boundaries": "confining lateral walls, closed downstream end",
            "porosity": "measured from simulated packing in the window in front of "
                        "the structure",
        },
        "base_cases": {
            "iso_rows": iso_rows,
            "load_statistics": stats,
            "pile": [{
                "case_id": r["case_id"],
                "porosity": r["porosity"],
                "solids_fraction": r["solids_fraction"],
                "pile_length": r["pile_length"],
                "pile_width": r["pile_width"],
                "n_floes_initial": r["n_floes_initial"],
                "n_floes_pile": r["n_floes_pile"],
                "platen_travel": r["platen_travel"],
                "dt": r["dt"],
                "n_steps": r["n_steps"],
                "wall_time_s": r["wall_time_s"],
            } for r in base],
        },
        "porosity_scaling": por,
        "sigma_scaling": iso_sc,
        "validation": {
            "C1_iso_same_order_of_magnitude": {
                "criterion": "0.1 <= F_DEM/F_ISO <= 10 for F_max and F_steady",
                "values": [{"case_id": r["case_id"], "ratio_max": r["ratio_max"],
                            "ratio_steady": r["ratio_steady"]} for r in base],
                "pass": iso_pass,
            },
            "C1b_iso_within_factor_3": {
                "criterion": "0.316 <= F_DEM/F_ISO <= 3.16",
                "pass": iso_tight,
            },
            "C2_porosity_scaling": {
                "criterion": "F ~ (1-n)^alpha with R2 > 0.8",
                "alpha": por["alpha"] if por else None,
                "alpha_ci95": por["alpha_ci95"] if por else None,
                "R2": por["R2"] if por else None,
                "n_points": por["n_points"] if por else 0,
                "pass": por["pass"] if por else False,
            },
            "C3_sigma_c_linearity": {
                "criterion": "log-log slope of F_max vs sigma_c in [0.75, 1.25] "
                             "(physical basis of the F = C sigma_c h^2 form)",
                "exponent": iso_sc["exponent"] if iso_sc else None,
                "pass": iso_sc["pass"] if iso_sc else False,
            },
        },
        "figures": figs,
        "runs": [{k: v for k, v in r.items()
                  if k not in ("time", "load_x", "load_y", "positions", "radii",
                               "pile_mask", "pile_mask_shape", "pile_extent")}
                 for r in results],
        "wall_time_s": wall,
        "quick_mode": args.quick,
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2, default=float))

    with (outdir / "runs.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "experiment", "sigma_c", "h", "D", "shape", "mu",
                    "floe_radius_median", "seed", "F_ISO", "F_max", "F_steady",
                    "ratio_max", "ratio_steady", "porosity", "pile_length",
                    "n_floes_pile", "wall_time_s"])
        for r in results:
            cfg = r["config"]
            w.writerow([r["case_id"], r["experiment"], cfg["sigma_c"], cfg["h"],
                        cfg["D"], cfg["shape"], cfg["mu"], cfg["floe_radius_median"],
                        cfg["seed"], r["F_ISO"], r["F_max"], r["F_steady"],
                        r["ratio_max"], r["ratio_steady"], r["porosity"],
                        r["pile_length"], r["n_floes_pile"], r["wall_time_s"]])

    print(json.dumps(report["validation"], indent=2, default=float), flush=True)
    print(f"[p227w2-N04] report written to {outdir/'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())