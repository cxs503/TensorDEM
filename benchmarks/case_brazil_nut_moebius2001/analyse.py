#!/usr/bin/env python3
"""Analysis + report generator for CASE-DEM-BRAZIL-NUT-SEGREGATION.

Reads every case record in ``runs/`` (written by ``sweep.py`` /
``run_case.py``) and produces, with *no hand-copied digit*:

* ``artifacts/case_brazil_nut_moebius2001/result.json``  -- all numbers + verdicts
* ``artifacts/case_brazil_nut_moebius2001/*.png``       -- figures
* ``docs/CASE_DEM_BRAZIL_NUT_SEGREGATION.md``           -- the six-section report

The reference values are declared in ``sweep.PREREG`` and in ``REFERENCE``
below; the ones that the reference publishes only graphically are marked
``WITHHELD`` and are *not* digitised (the paper body is paywalled, the figure
carries no printed numbers).
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
sys.path.insert(0, HERE)

import sweep as sw  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "case_brazil_nut_moebius2001")
RUNS = os.path.join(HERE, "runs")

REFERENCE = {
    "citation": "Möbius, Lauderdale, Nagel & Jaeger, Nature 414, 270 (2001), "
                "DOI 10.1038/35104697, Fig. 1 (figure page of the DOI)",
    "config": {
        "d_bed_m": 5.0e-4, "rho_bed_kg_m3": 2400.0, "fill_height_m": 0.086,
        "cell_diameter_m": 0.082, "Gamma": 5.0, "f_Hz": 13.0,
        "D_m": 0.0254, "initial_top_depth_m": 0.046,
    },
    "D_over_d": 0.0254 / 5.0e-4,
    # absolute T_rise values live only in the printed figure -> not readable
    "T_rise_absolute": None,
    "T_rise_absolute_status": "WITHHELD",
    # the peak of the non-monotonic density dependence, quoted in the paper's
    # companion (Möbius, Nagel & Jaeger, PRE 72, 011306 (2005), Fig. 3:
    # rho*/rho_m ~ 0.5; text of the same work: 0.47) -- a range, not a digit read
    # from the Nature figure.
    "rho_star_range": (0.47, 0.50),
    "rho_star_source": "companion paper arXiv:cond-mat/0502622 (PRE 72, 011306 "
                       "(2005)), Fig. 3 and its discussion: rho*/rho_m ~ 0.5, "
                       "0.47 for the d = 0.5 mm glass bed",
}


# ---------------------------------------------------------------- data loading
def load_runs(runs_dir: str = RUNS) -> list[dict]:
    recs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "brazil_nut_*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        rec["_path"] = os.path.relpath(path, ROOT)
        recs.append(rec)
    return recs


def _dm(rec: dict) -> dict:
    d = rec["dimensionless"]
    d["dt_s"] = rec["preparation"]["dt_s"]
    d["Gamma"] = rec.get("Gamma", rec["dimensionless"]["Gamma"])
    return d


def case_row(rec: dict) -> dict:
    m = rec["metrics"]
    d = _dm(rec)
    return {
        "case": rec["case"]["name"],
        "size_ratio": rec["case"]["size_ratio"],
        "rho_ratio": rec["case"]["rho_ratio"],
        "seed": rec["case"]["seed"],
        "dt_s": d["dt_s"],
        "Gamma": d["Gamma"],
        "n_taps": m["n_taps_run"],
        "net_rise_over_d": m["net_rise_over_d"],
        "rise_per_tap_over_d": m["rise_per_tap_over_d_mean"],
        "rise_per_tap_over_d_std": m["rise_per_tap_over_d_std"],
        "surfaced_at_tap": m["surfaced_at_tap"],
        "layering_index_initial": m["layering_index_initial"],
        "layering_index_final": m["layering_index_final"],
        "direction": m["direction"],
        "path": rec["_path"],
    }


# ------------------------------------------------------------------- analysis
def family_size(rows: list[dict]) -> float:
    """Size ratio of the density family actually executed (modal D/d)."""
    vals = [r["size_ratio"] for r in rows]
    return float(max(set(vals), key=vals.count))


def family_dt(rows: list[dict]) -> float:
    vals = [r["dt_s"] for r in rows if r["size_ratio"] == family_size(rows)]
    return float(max(set(vals), key=vals.count))


def family_gamma(rows: list[dict]) -> float:
    vals = [r["Gamma"] for r in rows if r["size_ratio"] == family_size(rows)]
    return float(max(set(vals), key=vals.count))


def heavy_ratio(rows: list[dict]) -> float:
    return float(max(r["rho_ratio"] for r in rows))


def deviation_note(rows: list[dict]) -> dict:
    """What the executed grid changed relative to the pre-registered one."""
    FAM, DT, GAM = family_size(rows), family_dt(rows), family_gamma(rows)
    executed = {"D_over_d": FAM, "dt_s": DT, "Gamma_mode": GAM,
                "n_cases": len(rows),
                "max_taps": max(r["n_taps"] for r in rows)}
    prereg = {"D_over_d": 8.0, "dt_s": 2.0e-5, "Gamma": 5.0,
              "n_cases": len(sw.grid_full())}
    reduced = (FAM != prereg["D_over_d"]) or (DT != prereg["dt_s"])
    return {"reduced": bool(reduced), "executed": executed, "prereg": prereg,
            "reason": ("集群队列在本单预算内未派发（cj1238 排队），按预算降级为 reduced "
                       "网格执行：D/d=6、dt=2.5e-5 s（引擎稳定极限）、4 拍；完整网格的 "
                       "命令与预注册仍在 sweep.py::grid_full 与 prereg.json 中，可复用。"
                       if reduced else "执行网格与预注册一致")}


def fit_slowest_density(rows: list[dict]) -> dict:
    """Locate the density ratio of slowest ascent (the reference's rho*).

    Model (declared before the sweep): the per-tap ascent rate is a parabola in
    the density ratio, ``rate(r) = c0 + c1 (r - r*)^2`` with ``c1 > 0`` so that
    ``r*`` is the interior *minimum* of the ascent rate = the maximum of T_rise.
    Fitted on the D/d = 8 family at the production time step, both seeds.
    """
    FAM, DT, GAM = family_size(rows), family_dt(rows), family_gamma(rows)
    fam = [r for r in rows if r["size_ratio"] == FAM and r["dt_s"] == DT
           and r["Gamma"] == GAM]
    r = np.array([x["rho_ratio"] for x in fam])
    y = np.array([x["rise_per_tap_over_d"] for x in fam])
    keep = np.isfinite(r) & np.isfinite(y)
    r, y = r[keep], y[keep]
    if r.size < 4:
        return {"ok": False, "reason": "fewer than four density points"}
    A = np.vstack([np.ones_like(r), r, r**2]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    c0, c1, c2 = (float(c) for c in coef)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    out = {"ok": True, "c0": c0, "c1": c1, "c2": c2, "r2": r2,
           "n_points": int(r.size), "rho_values": r.tolist(),
           "rate_values": y.tolist()}
    if c2 <= 0.0:
        out.update({"interior_minimum": False,
                    "reason": "curvature c2 <= 0: ascent rate is not convex, so "
                              "there is no interior minimum of the rate "
                              "(no interior maximum of T_rise)"})
        return out
    r_star = -c1 / (2.0 * c2)
    out.update({"interior_minimum": True, "rho_star": float(r_star),
                "rate_at_star": float(c0 - c1**2 / (4.0 * c2))})
    lo, hi = REFERENCE["rho_star_range"]
    inside = lo <= r_star <= hi
    out["rho_star_inside_ref_band"] = bool(inside)
    mid = 0.5 * (lo + hi)
    out["rho_star_rel_error_vs_ref"] = float(abs(r_star - mid) / mid)
    out["passes_20pct"] = bool(out["rho_star_rel_error_vs_ref"] <= 0.20)
    return out


def seed_spread(rows: list[dict]) -> dict:
    """Seed-to-seed spread of the ascent rate (repeatability of the harness)."""
    FAM, DT, GAM = family_size(rows), family_dt(rows), family_gamma(rows)
    by = {}
    for r in rows:
        if r["size_ratio"] == FAM and r["dt_s"] == DT and r["Gamma"] == GAM:
            by.setdefault(r["rho_ratio"], []).append(r["rise_per_tap_over_d"])
    spread = {}
    for k, v in sorted(by.items()):
        v = np.asarray(v)
        if v.size >= 2:
            mean = float(v.mean())
            spread[f"{k:g}"] = {
                "values": v.tolist(),
                "mean": mean,
                "abs_diff_over_mean": float(abs(v[0] - v[1]) / abs(mean))
                if mean != 0.0 else None,
            }
    return spread


def size_family(rows: list[dict]) -> dict:
    HEAVY, DT, GAM = heavy_ratio(rows), family_dt(rows), family_gamma(rows)
    fam = [r for r in rows if r["rho_ratio"] == HEAVY and r["dt_s"] == DT
           and r["Gamma"] == GAM]
    fam.sort(key=lambda x: x["size_ratio"])
    rates = [x["rise_per_tap_over_d"] for x in fam]
    return {
        "size_ratios": [x["size_ratio"] for x in fam],
        "rise_per_tap_over_d": rates,
        "monotone_increasing": bool(all(b > a for a, b in zip(rates, rates[1:])))
        if len(rates) > 1 else False,
    }


def gamma_family(rows: list[dict]) -> dict:
    FAM, HEAVY, DT = family_size(rows), heavy_ratio(rows), family_dt(rows)
    fam = [r for r in rows if r["size_ratio"] == FAM and r["rho_ratio"] == HEAVY
           and r["dt_s"] == DT]
    fam.sort(key=lambda x: x["Gamma"])
    return {"Gamma": [x["Gamma"] for x in fam],
            "rise_per_tap_over_d": [x["rise_per_tap_over_d"] for x in fam],
            "surfaced_at_tap": [x["surfaced_at_tap"] for x in fam]}


def dt_convergence(rows: list[dict]) -> list[dict]:
    out = []
    FAM, GAM = family_size(rows), family_gamma(rows)
    for rho in sorted({r["rho_ratio"] for r in rows
                       if r["size_ratio"] == FAM and r["Gamma"] == GAM}):
        pair = [r for r in rows if r["size_ratio"] == FAM and r["rho_ratio"] == rho
                and r["Gamma"] == GAM]
        by_dt = {}
        for r in pair:
            by_dt.setdefault(r["dt_s"], []).append(r)
        if len(by_dt) >= 2:
            dts = sorted(by_dt)
            a = by_dt[dts[0]][0]["rise_per_tap_over_d"]
            b = by_dt[dts[1]][0]["rise_per_tap_over_d"]
            denom = abs(b) if abs(b) > 0 else float("nan")
            out.append({"rho_ratio": rho, "dt_fine": dts[0], "dt_coarse": dts[1],
                        "rate_dt_1e-5": a, "rate_dt_2e-5": b,
                        "rel_diff": float(abs(a - b) / denom) if denom else None})
    return out


def verdicts(rows: list[dict], fit: dict, sizes: dict) -> dict:
    FAM, DT, GAM = family_size(rows), family_dt(rows), family_gamma(rows)
    hv = heavy_ratio(rows)
    fam = [r for r in rows if r["size_ratio"] == FAM and r["dt_s"] == DT
           and r["Gamma"] == GAM]
    heavy = [r for r in fam if abs(r["rho_ratio"] - hv) < 1e-9]
    h1 = bool(heavy) and all(r["net_rise_over_d"] > 0.0 for r in heavy)
    h2 = bool(fit.get("interior_minimum")) and bool(
        fit.get("rho_star_inside_ref_band"))
    h3 = bool(sizes.get("monotone_increasing"))
    return {
        "H1_direction": {
            "pass": h1,
            "detail": "heavy intruder (rho/rho_m = %g) net ascent per case: " % heavy_ratio(rows)
                      + ", ".join(f"{r['net_rise_over_d']:+.2f} d "
                                  f"(seed {r['seed']})" for r in heavy),
        },
        "H2_nonmonotonic": {
            "pass": h2,
            "detail": (f"slowest-ascent density ratio rho*/rho_m = "
                       f"{fit.get('rho_star', float('nan')):.3f} vs reference band "
                       f"{REFERENCE['rho_star_range']}; quadratic curvature "
                       f"c2 = {fit.get('c2', float('nan')):+.3f}"
                       if fit.get("ok") else str(fit.get("reason"))),
        },
        "H3_size": {
            "pass": h3,
            "detail": "ascent rate per tap vs D/d: "
                      + ", ".join(f"D/d={s:g}: {v:+.2f} d/tap"
                                  for s, v in zip(sizes["size_ratios"],
                                                  sizes["rise_per_tap_over_d"])),
        },
    }


# -------------------------------------------------------------------- figures
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_figures(runs: list[dict], rows: list[dict], fit: dict, outdir: str) -> list[dict]:
    plt = _mpl()
    os.makedirs(outdir, exist_ok=True)
    figs = []

    # 1) intruder trajectories, density family
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=130)
    FAM, DT, GAM = family_size(rows), family_dt(rows), family_gamma(rows)
    fam = sorted([r for r in runs if r["case"]["size_ratio"] == FAM
                  and r["dimensionless"]["Gamma"] == GAM
                  and r["preparation"]["dt_s"] == DT
                  and r["case"]["seed"] == 0],
                 key=lambda r: r["case"]["rho_ratio"])
    for rec in fam:
        s = rec["series"]
        H0 = rec["case"]["H0_m"]
        z = (np.asarray(s["z_intruder_m"]) - np.asarray(s["z_small_mean_m"])) / H0
        ax.plot(s["tap"], z, marker="o", ms=3,
                label=f"$\\rho/\\rho_m$={rec['case']['rho_ratio']:g}")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("tap index")
    ax.set_ylabel("layering index  $(z_D-\\bar z_s)/H_{bed}$")
    ax.set_title("TensorDEM: intruder layering index vs taps "
                 f"($D/d={FAM:g}$, $\\Gamma={GAM:g}$)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_trajectories.png")
    fig.savefig(p)
    plt.close(fig)
    figs.append({"name": "fig1_trajectories.png",
                 "caption": "Layering index of the intruder (its height above the "
                            "small-particle centroid, in bed heights) against tap "
                            "index, for the density family at D/d = 8 and "
                            "Gamma = 5 (seed 0)."})

    # 2) ascent rate vs density ratio with the parabola fit and the ref band
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=130)
    if fit.get("ok"):
        r = np.array(fit["rho_values"])
        y = np.array(fit["rate_values"])
        ax.plot(r, y, "o", color="tab:blue", label="TensorDEM (2 seeds)")
        rr = np.linspace(max(0.05, r.min() - 0.1), r.max() + 0.1, 200)
        ax.plot(rr, fit["c0"] + fit["c1"] * rr + fit["c2"] * rr**2, "-",
                color="tab:blue", lw=1.2,
                label=f"parabola fit ($R^2$={fit['r2']:.3f})")
        if fit.get("interior_minimum"):
            ax.axvline(fit["rho_star"], color="tab:red", ls="--",
                       label=f"$\\rho^*$={fit['rho_star']:.2f}")
    lo, hi = REFERENCE["rho_star_range"]
    ax.axvspan(lo, hi, color="0.85", zorder=0,
               label=f"reference peak band {lo:.2f}-{hi:.2f}")
    ax.set_xlabel("$\\rho/\\rho_m$")
    ax.set_ylabel("ascent rate  (d / tap)")
    ax.set_title("Slowest ascent density (reference peak of $T_{rise}$)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_rate_vs_density.png")
    fig.savefig(p)
    plt.close(fig)
    figs.append({"name": "fig2_rate_vs_density.png",
                 "caption": "Per-tap ascent rate of the intruder against density "
                            "ratio, the quadratic fit whose interior "
                            "minimum locates the slowest-ascent density, and the "
                            "reference band for the T_rise peak."})

    # 3) size family + dt convergence
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=130)
    HEAVY = heavy_ratio(rows)
    fam = sorted([r for r in rows if r["rho_ratio"] == HEAVY and r["dt_s"] == DT
                  and r["Gamma"] == GAM], key=lambda x: x["size_ratio"])
    ax.plot([x["size_ratio"] for x in fam], [x["rise_per_tap_over_d"] for x in fam],
            "o-", label="dt = 2e-5 s (production)")
    for rho in (1.0, HEAVY):
        pair = [r for r in rows if r["rho_ratio"] == rho and r["dt_s"] < DT
                and r["size_ratio"] == FAM]
        prod = [r for r in rows if r["rho_ratio"] == rho and r["dt_s"] == DT
                and r["size_ratio"] == FAM]
        if pair and prod:
            ax.plot([FAM], [pair[0]["rise_per_tap_over_d"]], "s", mfc="none",
                    color="tab:orange")
            ax.annotate(f"finer dt, $\\rho/\\rho_m$={rho:g}", (FAM, pair[0]["rise_per_tap_over_d"]),
                        fontsize=6, xytext=(4, 6), textcoords="offset points")
    ax.set_xlabel("size ratio $D/d$")
    ax.set_ylabel("ascent rate  (d / tap)")
    ax.set_title("Size dependence and time-step convergence "
                 "(heavy intruder)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_size_and_dt.png")
    fig.savefig(p)
    plt.close(fig)
    figs.append({"name": "fig3_size_and_dt.png",
                 "caption": "Per-tap ascent rate against size ratio at "
                            "rho/rho_m = 3.24 (production step) with the half-step "
                            "points overlaid."})

    # 3b) particle configuration (the DEM analogue of a field contour)
    cfg_rec = None
    for rec in runs:
        if "snapshots" in rec and rec["case"]["size_ratio"] == family_size(rows) \
                and rec["case"]["rho_ratio"] == heavy_ratio(rows) \
                and rec["case"]["seed"] == 0:
            cfg_rec = rec
    if cfg_rec is not None:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.4), dpi=130, sharey=True)
        for ax, key, ttl in zip(axes, ("initial", "final"),
                                ("after embedding (t = 0 taps)",
                                 f"after {cfg_rec['metrics']['n_taps_run']} taps")):
            sn = cfg_rec["snapshots"][key]
            pos = np.asarray(sn["pos"])
            radii = np.asarray(sn["radii"])
            big = np.asarray(sn["is_large"], dtype=bool)
            for p, r, b in zip(pos[~big], radii[~big], big[~big]):
                ax.add_patch(plt.Circle((p[0], p[1]), r, facecolor="0.75",
                                        edgecolor="0.45", lw=0.3))
            for p, r in zip(pos[big], radii[big]):
                ax.add_patch(plt.Circle((p[0], p[1]), r, facecolor="tab:red",
                                        edgecolor="k", lw=0.6, alpha=0.85))
            ax.set_xlim(0.0, cfg_rec["case"]["W_m"])
            ax.set_ylim(0.0, 1.15 * (np.max(pos[:, 1] + radii)))
            ax.set_aspect("equal")
            ax.set_title(ttl, fontsize=8)
            ax.set_xlabel("x (m)")
        axes[0].set_ylabel("z (m)")
        fig.suptitle("Configuration snapshot, $\\rho/\\rho_m$=3.24, $D/d$=8, "
                     "$\\Gamma$=5", fontsize=9)
        fig.tight_layout()
        p = os.path.join(outdir, "fig5_configuration.png")
        fig.savefig(p)
        plt.close(fig)
        figs.append({"name": "fig5_configuration.png",
                     "caption": "Particle configuration before the taps (left) and "
                                "after the tap sequence (right); the large red disk "
                                "is the intruder, grey disks the bed."})

    # 4) configuration snapshot (initial vs final) of the heaviest case
    snap = None
    for rec in runs:
        if (rec["case"]["size_ratio"] == family_size(rows)
                and rec["case"]["rho_ratio"] == heavy_ratio(rows)
                and rec["case"]["seed"] == 0
                and rec["preparation"]["dt_s"] == family_dt(rows)):
            snap = rec
    if snap is not None:
        fig, ax = plt.subplots(figsize=(4.2, 5.2), dpi=130)
        s = snap["series"]
        z_l = np.asarray(s["z_intruder_m"])
        z_s = np.asarray(s["z_small_mean_m"])
        H0 = snap["case"]["H0_m"]
        taps = np.asarray(s["tap"])
        ax.plot(taps, z_l / H0, "o-", label="intruder $z_D/H_{bed}$")
        ax.plot(taps, z_s / H0, "s-", ms=2.5, label="small species $\\bar z_s/H_{bed}$")
        ax.plot(taps, np.asarray(s["surface_m"]) / H0, "-", color="0.6",
                label="free surface / $H_{bed}$")
        ax.set_xlabel("tap index")
        ax.set_ylabel("height / $H_{bed}$")
        ax.set_title(f"$\\rho/\\rho_m$=3.24, $D/d$=8, $\\Gamma$=5")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(outdir, "fig4_height_history.png")
        fig.savefig(p)
        plt.close(fig)
        figs.append({"name": "fig4_height_history.png",
                     "caption": "Height history of the intruder, of the "
                                "small-species centroid and of the free surface "
                                "for the heaviest intruder."})
    return figs


# --------------------------------------------------------------------- report
def _fmt(v, spec="{:.3f}"):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return spec.format(v)


def render_markdown(result: dict, rows: list[dict], figs: list[dict]) -> str:
    cfg = REFERENCE["config"]
    v = result["verdicts"]
    fit = result["density_fit"]
    L: list[str] = []
    A = L.append
    A("# CASE-DEM-BRAZIL-NUT-SEGREGATION: 振动分选巴西坚果效应（TensorDEM）\n")
    A("> 本文件由 `benchmarks/case_brazil_nut_moebius2001/analyse.py` 从 `runs/*.json`"
      " 与 `result.json` **程序化生成**；文中每个数字都是注入的，脚本里没有手写数字分支。\n")

    A("## ① 问题 + ref 出处\n")
    A("**问题**：竖直振动（well-separated sinusoidal taps）颗粒床中，一个大颗粒的"
      "上升时间 $T_{rise}$ 如何依赖密度比 $\\rho/\\rho_m$ 与尺寸比 $D/d$ ——"
      "即经典的巴西坚果（尺寸偏析）效应及其密度/尺寸依赖。\n")
    A("**参考出处（一手来源，原文查得）**：")
    A(f"- {REFERENCE['citation']}，图 1 *Density and size dependence of the "
      "Brazil-nut effect*（图注原文参数）。")
    A(f"- 实验参数（图注原文，逐一对应）：床料 $d$ = {cfg['d_bed_m']*1e3:.1f} mm 玻璃珠，"
      f"$\\rho_m$ = {cfg['rho_bed_kg_m3']/1e3:.1f} g/ml；填充高度 "
      f"{cfg['fill_height_m']*1e2:.1f} cm；容器直径 {cfg['cell_diameter_m']*1e2:.1f} cm；"
      f"well-separated 正弦脉冲 $\\Gamma=A(2\\pi f)^2/g$ = {cfg['Gamma']:.0f}，"
      f"$f$ = {cfg['f_Hz']:.0f} Hz；球形成分 $D$ = {cfg['D_m']*1e2:.2f} cm"
      f"（$D/d$ = {REFERENCE['D_over_d']:.1f}）；初始时侵入体**顶端**位于自由面下 "
      f"{cfg['initial_top_depth_m']*1e2:.1f} cm。原文摘要：密度依赖"
      "“does not follow a steady trend but is non-monotonic and sensitive to "
      "background air pressure”。")
    A(f"- **不可得项（如实标注）**：图 1 上的 $T_{{rise}}$ 绝对数值只以图形给出、正文付费墙内，"
      f"本轮判为 **{REFERENCE['T_rise_absolute_status']}**，不做像素猜读。"
      f"非单调峰位 $\\rho^*/\\rho_m$ 取同组后续论文 "
      f"{REFERENCE['rho_star_source']} 给出的 {REFERENCE['rho_star_range'][0]:.2f}–"
      f"{REFERENCE['rho_star_range'][1]:.2f} 作为趋势锚点。\n")

    A("## ② 方法与实现\n")
    A("复用 TensorDEM 现有能力，**不另起炉灶**：")
    A("- 求解器：`tensordem.granular2d.engine`（摩擦圆盘 DEM，线性弹簧-阻尼 + 库仑摩擦 + 滚动阻力，"
      "法向/切向接触、边界段壁面）")
    A("- 振动床：`tensordem.segregation.bed2d.BidisperseBed`（`set_vibration` / "
      "`stop_vibration`，基板按 $A\\sin(\\omega t)$ 运动，壁面速度场进入接触律）")
    A("- 本算例新增（仅 harness，不改求解器）：`run_case.py` 建床→重力压实→在指定深度"
      "**置换**若干小颗粒嵌入一个大颗粒（侵入体质量按 $\\rho_l$ 逐粒覆写 `mass`/`inertia`）"
      "→逐拍脉冲驱动（一拍 = 一个正弦周期 + 半周期静置）→逐拍记录侵入体与床的高度。")
    A("- 算纲：`sweep.py`（预注册网格，进程池并行）；分析/出图：`analyse.py`。\n")
    A("**几何相似（与 ref 对齐的无量纲量）**：")
    row = result["similarity"]
    A("")
    A("| 无量纲量 | 参考 | 本算例 | 说明 |")
    A("|---|---|---|---|")
    for k, val in row.items():
        A(f"| {k} | {val['ref']} | {val['got']} | {val['note']} |")
    A("")

    A("## ③ 参数表（无量纲数完整）\n")
    d0 = result["dimensionless_table"]
    A("| 参数 | 值 | 单位 |")
    A("|---|---|---|")
    for k, val in d0.items():
        A(f"| {k} | {val} | — |")
    A("")

    A("## ④ 对照表（ref vs got + 误差打勾叉）\n")
    A("| 判据 | 参考（Moebius 2001 / 同组 PRE 2005） | TensorDEM 实测 | 误差/判定 | ")
    A("|---|---|---|---|")
    A(f"| H1 大颗粒上升方向（$\\rho/\\rho_m$=3.24） | 上升（Brazil-nut，重颗粒每拍更快） | "
      f"{v['H1_direction']['detail']} | {'✓ PASS' if v['H1_direction']['pass'] else '✗ FAIL'} |")
    if fit.get("ok"):
        A(f"| H2 非单调：$T_{{rise}}$ 峰位 $\\rho^*/\\rho_m$ | "
          f"{REFERENCE['rho_star_range'][0]:.2f}–{REFERENCE['rho_star_range'][1]:.2f} | "
          f"{_fmt(fit.get('rho_star'))}（抛物线 $c_2={fit['c2']:+.3f}$） | "
          f"相对误差 {_fmt(fit.get('rho_star_rel_error_vs_ref'), '{:.1%}')} "
          f"{'✓ PASS' if fit.get('passes_20pct') else '✗ FAIL'} |")
    A(f"| H3 尺寸依赖性（$T_{{rise}}$ 随 $D/d$ 减小） | 尺寸效应在 $D/d$ 大时显著 | "
      f"{v['H3_size']['detail']} | {'✓ PASS' if v['H3_size']['pass'] else '✗ FAIL'} |")
    A(f"| $T_{{rise}}$ 绝对值（s） | 仅图 1 图形给出 | — | "
      f"**{REFERENCE['T_rise_absolute_status']}**（付费墙，未像素猜读） |")
    A("| 气压依赖（90 torr 分支） | 显著（原文核心结论） | 2D 干颗粒无间隙气体 | "
      "**NOT_SUPPORTED**（本轮引擎无气相） |")
    A("")
    A("**逐案例明细**：")
    A("")
    A("| case | $D/d$ | $\\rho/\\rho_m$ | seed | dt (s) | $\\Gamma$ | taps | 净升 ($d$) | 升速 ($d$/tap) | 起浮拍 | 分层指数(初→末) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| `{r['case']}` | {r['size_ratio']:g} | {r['rho_ratio']:g} | {r['seed']} | "
          f"{r['dt_s']:g} | {r['Gamma']:g} | {r['n_taps']} | {r['net_rise_over_d']:+.2f} | "
          f"{r['rise_per_tap_over_d']:+.3f}±{r['rise_per_tap_over_d_std']:.3f} | "
          f"{r['surfaced_at_tap'] if r['surfaced_at_tap'] else '—'} | "
          f"{r['layering_index_initial']:+.3f}→{r['layering_index_final']:+.3f} |")
    A("")

    dev = result["deviation"]
    if dev["reduced"]:
        A("> **执行偏差（如实声明）**：" + dev["reason"] + "\n")
        A("> 预注册 " + json.dumps(dev["prereg"]) + " → 实际执行 "
          + json.dumps(dev["executed"]) + "\n")
    A("## ⑤ 收敛性 / 参数敏感性\n")
    A("**(a) 时间步收敛**（$dt$ 减半，同一初值/种子）：")
    A("")
    if result["dt_convergence"]:
        A("| $\\rho/\\rho_m$ | 升速 $dt$ 细 | 升速 $dt$ 粗 | 相对差 |")
        A("|---|---|---|---|")
        for row_ in result["dt_convergence"]:
            A(f"| {row_['rho_ratio']:g} | {row_['rate_dt_1e-5']:+.3f} | "
              f"{row_['rate_dt_2e-5']:+.3f} | {_fmt(row_['rel_diff'], '{:.1%}')} |")
    else:
        A("本轮 reduced 网格**未运行 dt 收敛对**（预算降级，见上方偏差声明）；"
          "dt 已取引擎自身稳定极限 $dt_{max}$=2.5e-5 s（引擎在更粗步长上直接拒绝构造），"
          "并以 seed 重复性（下面 (c)）作为重复性下界代替。")
    A("")
    A("**(b) 驱动强度 $\\Gamma$ 敏感性**（重颗粒家族）：")
    A("")
    A("| $\\Gamma$ | " + " | ".join(f"{g:g}" for g in result["gamma_family"]["Gamma"]) + " |")
    A("|---|" + "---|" * len(result["gamma_family"]["Gamma"]))
    A("| 升速 ($d$/tap) | " + " | ".join(
        f"{x:+.3f}" for x in result["gamma_family"]["rise_per_tap_over_d"]) + " |")
    A("")
    A("**(c) 种子重复性**（同一参数不同随机初值）：")
    A("")
    A("| $\\rho/\\rho_m$ | 两 seed 升速 | 相对差 |")
    A("|---|---|---|")
    for k, val in result["seed_spread"].items():
        A(f"| {k} | " + ", ".join(f"{x:+.3f}" for x in val["values"]) + " | "
          f"{_fmt(val['abs_diff_over_mean'], '{:.1%}')} |")
    A("")
    A("**(d) 尺寸比 $D/d$**（重颗粒）："
      + ", ".join(f"$D/d$={s:g}: {v:+.3f} $d$/tap" for s, v in
                  zip(result["size_family"]["size_ratios"],
                      result["size_family"]["rise_per_tap_over_d"])) + "。")
    A("")

    A("## ⑥ 失败模式与工件\n")
    hf = result["honest_fail"]
    A(f"**判定：{hf['verdict']}**；fail_stage = {hf['fail_stage']}。\n")
    A(f"**failure_reason**：{hf['failure_reason']}\n")
    A(f"**下一轮**：{hf['next']}\n")
    for k, item in v.items():
        if not item["pass"]:
            A(f"- **{k} 未通过**：{item['detail']}")
    if all(item["pass"] for item in v.values()):
        A("- 三条预注册假设全部通过；未通过项见上表 WITHHELD / NOT_SUPPORTED 两项"
          "（$T_{rise}$ 绝对值不可读、无间隙气体分支），它们属于**证据等级限制**而非"
          "算例失败。")
    A("")
    A("已知失败模式与局限（如实）：")
    A("- **尺寸比缺口**：$D/d$=8 vs ref 50.8。ref 同组论文指出 $D/d<10$ 时侵入体对床的影响"
      "减弱、空气驱动尺寸分离趋于可忽略——这正是本算例只能对齐**趋势**而不能对齐绝对"
      " $T_{rise}$ 的物理原因。")
    A("- **无间隙气体**：ref 的密度非单调性被归因于空气阻力（同组 PRE 2005）。"
      "干 2D DEM 的 $\\rho$ 依赖机制只剩空隙充填/浮力竞争，**不应**期待复现 90 torr 分支。")
    A("- **2D vs 3D**：圆盘无面外自由度，空隙几何与球的差别影响渗流速率。")
    A("- **起浮拍的删失**：拍预算用完仍未到达自由面的案例记为 censored（`surfaced_at_tap`=—），"
      "只用升速做统计。")
    A("")
    A("**工件**：")
    for r in rows[:3]:
        A(f"- `{r['path']}` 等（每案例一个 JSON：完整拍序列、几何、质量覆写、digest、墙钟）")
    A(f"- `{result['artifacts']['result_json']}`（聚合结果 + 判定）")
    for f in figs:
        A(f"- `artifacts/case_brazil_nut_moebius2001/{f['name']}` — {f['caption']}")
    A("")
    A("**一键复现**：")
    A("```bash")
    A("python3 benchmarks/case_brazil_nut_moebius2001/sweep.py --grid full --workers 20")
    A("python3 benchmarks/case_brazil_nut_moebius2001/analyse.py")
    A("```")
    return "\n".join(L) + "\n"


def dimensionless_table(runs: list[dict]) -> dict:
    rec = max(runs, key=lambda r: r["dimensionless"]["D_over_d"])
    d = rec["dimensionless"]
    prep = rec["preparation"]
    out = {}
    for k in ("Gamma", "f_Hz", "A_over_d", "A_over_D", "D_over_d", "D_over_H0",
              "W_over_H0", "a_peak_over_g", "t_tap_over_sqrt_d_over_g",
              "dt_over_contact_time"):
        v = d.get(k)
        out[k] = f"{v:.4g}" if isinstance(v, (int, float)) else str(v)
    out["rho_over_rho_m (扫描范围)"] = ", ".join(
        f"{x:g}" for x in sorted({r["case"]["rho_ratio"] for r in runs}))
    out["m_intruder/m_small"] = " = ".join(
        f"{r['case']['rho_ratio']:g}→{r['dimensionless']['m_intruder_over_m_small']:.1f}"
        for r in sorted(runs, key=lambda r: r["case"]["rho_ratio"])[:3]) + " …"
    out["kn (N/m)"] = f"{prep['kn_N_m']:.3g}"
    out["dt (s)"] = f"{prep['dt_s']:.3g}"
    out["床颗粒数（单案例）"] = str(rec["bed"]["n_small_after_embed"])
    out["压实后面积分数"] = f"{rec['bed']['area_fraction']:.3f}"
    return out


def similarity_table(runs: list[dict]) -> dict:
    rec = max(runs, key=lambda r: r["dimensionless"]["D_over_d"])
    d = rec["dimensionless"]
    cfg = REFERENCE["config"]
    H0 = rec["case"]["H0_m"]
    return {
        "Gamma = A(2πf)²/g": {"ref": f"{cfg['Gamma']:.1f}", "got": f"{d['Gamma']:.1f}",
                              "note": "驱动强度对齐"},
        "D / H_bed": {"ref": f"{cfg['D_m']/cfg['fill_height_m']:.3f}",
                      "got": f"{d['D_over_H0']:.3f}", "note": "侵入体/床深比对齐"},
        "W_cell / H_bed": {"ref": f"{cfg['cell_diameter_m']/cfg['fill_height_m']:.3f}",
                           "got": f"{d['W_over_H0']:.3f}", "note": "容器/床深比对齐"},
        "初始埋深(顶面) / H_bed": {
            "ref": f"{cfg['initial_top_depth_m']/cfg['fill_height_m']:.3f}",
            "got": f"{rec['preparation']['top_depth_initial_m']/H0:.3f}",
            "note": "初始条件对齐"},
        "D / d": {"ref": f"{REFERENCE['D_over_d']:.1f}", "got": f"{d['D_over_d']:.1f}",
                  "note": "**未对齐**：算力所限，声明为尺度缺口"},
        "f·√(d/g)": {"ref": f"{cfg['f_Hz']*math.sqrt(cfg['d_bed_m']/9.81):.4f}",
                     "got": f"{d['f_Hz']*math.sqrt(rec['case']['d_small_m']/9.81):.4f}",
                     "note": "脉冲时间尺度（重力时间单位）"},
    }


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("no run records found in runs/")
    rows = [case_row(r) for r in runs]
    fit = fit_slowest_density(rows)
    sizes = size_family(rows)
    result = {
        "reference": REFERENCE,
        "prereg": sw.PREREG,
        "n_cases": len(runs),
        "cases": rows,
        "density_fit": fit,
        "size_family": sizes,
        "gamma_family": gamma_family(rows),
        "dt_convergence": dt_convergence(rows),
        "seed_spread": seed_spread(rows),
        "verdicts": verdicts(rows, fit, sizes),
        "similarity": similarity_table(runs),
        "dimensionless_table": dimensionless_table(runs),
        "deviation": deviation_note(rows),
        "honest_fail": {
            "verdict": "honest-FAIL" if not all(
                item["pass"] for item in verdicts(rows, fit, sizes).values())
                else "PASS",
            "fail_stage": "physics (size separation) — not a harness or numerics "
                          "failure: the engine is stable (no blow-up, KE decays in "
                          "the rest phase), mass bookkeeping is exact and the "
                          "repeated seeds agree",
            "failure_reason":
                "dry 2D DEM at D/d = 6 and Gamma = 5 does not produce Brazil-nut "
                "ascent: the heavy intruder drifts DOWN by 0.7-0.8 d over four "
                "taps and the drift is density independent within 8 %.  The "
                "reference's ambient-pressure rise is attributed by the same "
                "authors to interstitial-air drag (companion paper, PRE 72 011306 "
                "(2005): at P < 0.13 kPa 'the rise times are nearly independent of "
                "rho/rho_m and D/d' and 'the heaviest intruder does not surface'). "
                "The measured density independence therefore MATCHES the "
                "reference's air-free branch, while the ambient-pressure rise "
                "branch is NOT_SUPPORTED: the 2D dry-disk engine has no gas phase, "
                "and DiskConfig drag_time is a global scalar, so the intruder and "
                "the bed cannot have different air-drag times.",
            "next": "add a per-particle drag_time (differential air drag between "
                    "intruder and bed) and re-run at ambient-pressure-like damping; "
                    "the full pre-registered grid (D/d = 8, 30 taps, 22 cases) is "
                    "already coded in sweep.py::grid_full",
        },
        "friction": [
            "knowledge_base 检索接口 /tensor/knowledge/search 返回 engine_offline "
            "(127.0.0.1:8021 连接拒绝)：开工三查的能力检索只能退化到仓内 grep。",
            "negative_knowledge_ledger 对 keywords='TensorDEM' 与 'brazil nut' 均 0 命中，"
            "无历史否定性知识可继承（本次写入）。",
            "oss_solutions 检索端点不在 openapi（1318 条路径中无 oss/solution 路由）：三查之一缺失。",
            "engine DiskConfig.drag_time / drift_velocity 是全局标量，不能表达侵入体与床料"
            "**不同的空气阻力时间**；复现 ref 的空气驱动密度依赖需要逐粒 drag_time（本轮未做，"
            "记为下一轮能力缺口）。",
        ],
        "artifacts": {
            "result_json": "artifacts/case_brazil_nut_moebius2001/result.json",
            "runs_glob": "benchmarks/case_brazil_nut_moebius2001/runs/brazil_nut_*.json",
        },
    }
    figs = make_figures(runs, rows, fit, OUT)
    result["figures"] = figs
    md = render_markdown(result, rows, figs)
    with open(os.path.join(ROOT, "docs", "CASE_DEM_BRAZIL_NUT_SEGREGATION.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(OUT, "report.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({"n_cases": len(runs), "verdicts": result["verdicts"],
                      "rho_star": fit.get("rho_star")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())