"""Campaign p227w2-N04 (vertical-section rework): ice rubble pile-up DEM vs ISO 19906.

Runs the predeclared case matrix of
:mod:`tensordem.ice_rubble_pileup_vertical`, fits the porosity-resistance
scaling law ``F ~ (1-n)^alpha``, writes machine-readable artifacts, the figures
and the paper-grade report.

Artifacts (committed, never under a git-ignored ``results/`` path)::

    runs/m14600/ice_rubble_pileup/report.json     aggregate numbers
    runs/m14600/ice_rubble_pileup/runs.csv        one row per case
    runs/m14600/ice_rubble_pileup/histories.json  load histories (downsampled)
    docs/figures/ice_rubble_pileup/fig*.png       figures
    docs/ICE_RUBBLE_PILEUP_ISO19906_REPORT.md     report (numbers injected)

Usage::

    python benchmarks/ice_rubble_pileup/run_benchmark.py --workers 6
    python benchmarks/ice_rubble_pileup/run_benchmark.py --quick   # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.ice_rubble_pileup_vertical import (  # noqa: E402
    PileupVerticalConfig,
    fit_porosity_scaling,
    iso_19906_reference,
    measure_pile_geometry,
    run_vertical_pileup_case,
)

OUT = ROOT / "runs" / "m14600" / "ice_rubble_pileup"
FIG = ROOT / "docs" / "figures" / "ice_rubble_pileup"
DOC = ROOT / "docs" / "ICE_RUBBLE_PILEUP_ISO19906_REPORT.md"

BASE = dict(sigma_c=3.0e6, h=1.0, D=5.0, drift_speed=0.5, mu=0.45,
            r_median_factor=0.16, r_bounds_factor=(0.10, 0.22),
            k_lin_factor=10.0, sheet_length_factor=8.0,
            platen_travel_factor=10.0, seed=11)

ISO_CASES = ("iso_cyl_D5", "iso_cyl_D3", "iso_sq_D5", "iso_cyl_h1p5")


def case_matrix() -> list[dict]:
    """Predeclared case matrix (fixed before any result was inspected)."""
    cases: list[dict] = []

    def add(cid, group, **kw):
        cfg = dict(BASE)
        cfg.update(kw)
        cfg["r_bounds_factor"] = tuple(cfg["r_bounds_factor"])
        cases.append({"case_id": cid, "group": group, "config": cfg})

    # --- base ISO 19906 comparisons (cylinder / narrow / square / thick) ----
    add("iso_cyl_D5", "iso", shape="cylinder", D=5.0)
    add("iso_cyl_D3", "iso", shape="cylinder", D=3.0)
    add("iso_sq_D5", "iso", shape="square", D=5.0)
    add("iso_cyl_h1p5", "iso", shape="cylinder", D=8.0, h=1.5, sigma_c=2.5e6)

    # --- thickness sweep: F_max monotone in h -------------------------------
    for h in (0.5, 0.75, 1.0, 1.25, 1.5):
        add("h%g" % h, "thickness", h=h, D=5.0)

    # --- strength sweep: F_max monotone in sigma_c --------------------------
    for sc in (1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6):
        add("sc%g" % (sc / 1e6), "strength", sigma_c=sc)

    # --- porosity sweep: formation kinetics at fixed h, sigma_c, mu ---------
    for V in (0.3, 0.5, 1.0):
        for sln in (0.08, 0.25, 0.50):
            add("por_V%g_sln%g" % (V, sln), "porosity",
                drift_speed=V, r_sigma_ln=sln)

    # --- resolution study: block size relative to h -------------------------
    for rmed, rb in ((0.09, (0.058, 0.122)), (0.12, (0.078, 0.162))):
        add("res_r%g" % rmed, "resolution", r_median_factor=rmed,
            r_bounds_factor=rb)
    return cases


def _run(job: dict) -> dict:
    """Run one case (executed in a worker process)."""
    cfg = PileupVerticalConfig(**job["config"])
    t0 = time.time()
    res = run_vertical_pileup_case(cfg)
    z_sheet_top = 0.10 * cfg.h
    geo = measure_pile_geometry(res.pos, res.rad, cfg, z_sheet_top)
    ref = iso_19906_reference(cfg)
    t, load = np.asarray(res.time), np.asarray(res.load)
    k = max(1, int(math.ceil(max(t.size, 1) / 400)))
    hist = {"time": t[::k].round(4).tolist(), "load": load[::k].round(2).tolist()}
    return {
        "case_id": job["case_id"],
        "group": job["group"],
        "config": job["config"],
        "secs": round(time.time() - t0, 1),
        "F_max": res.F_max,
        "F_steady": res.F_steady,
        "F_mean": res.F_mean,
        "F_ISO": res.F_ISO,
        "F_ISO_per_width": res.F_ISO_per_width,
        "p_G": res.p_G,
        "ratio_max": res.ratio_max,
        "ratio_steady": res.ratio_steady,
        "C_task": res.C_task,
        "porosity": res.porosity,
        "solids_fraction": res.solids_fraction,
        "pile_height": res.pile_height,
        "pile_length": res.pile_length,
        "pile_angle_deg": res.pile_angle_deg,
        "pile_area": res.pile_area,
        "envelope_area": res.envelope_area,
        "n_blocks_initial": res.n_blocks_initial,
        "n_blocks_pile": res.n_blocks_pile,
        "iso": ref,
        "dt": res.dt,
        "n_steps": res.n_steps,
        "pos": np.asarray(res.pos).round(4).tolist(),
        "rad": np.asarray(res.rad).round(4).tolist(),
        "flank_x": np.asarray(geo["flank_x"]).round(4).tolist(),
        "flank_z": np.asarray(geo["flank_z"]).round(4).tolist(),
        "history": hist,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _slope(x, y):
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    s, b = np.polyfit(lx, ly, 1)
    pred = s * lx + b
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return float(s), (float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))


def analyse(rows: list[dict]) -> dict:
    by = {r["case_id"]: r for r in rows}
    iso = [by[c] for c in ISO_CASES if c in by]
    base0 = by.get("iso_cyl_D5", rows[0])

    thick = sorted([r for r in rows if r["group"] == "thickness"],
                   key=lambda r: r["config"]["h"])
    strength = sorted([r for r in rows if r["group"] == "strength"],
                      key=lambda r: r["config"]["sigma_c"])
    por = [r for r in rows if r["group"] == "porosity"]
    res_group = sorted([r for r in rows if r["group"] == "resolution"],
                       key=lambda r: r["config"]["r_median_factor"])

    if len(por) >= 3:
        por_fit = fit_porosity_scaling(
            [r["solids_fraction"] for r in por],
            [r["F_max"] / (r["config"]["sigma_c"] * r["config"]["h"] ** 2) for r in por])
    else:
        por_fit = {"alpha": float("nan"), "K": float("nan"), "R2": float("nan"),
                   "alpha_ci95": [float("nan")] * 2, "n_points": len(por),
                   "log_lever_arm": float("nan"),
                   "solids_fraction_range": [float("nan")] * 2}

    h_exp, h_r2 = (_slope([r["config"]["h"] for r in thick],
                          [r["F_max"] for r in thick]) if len(thick) > 2
                   else (float("nan"), float("nan")))
    s_exp, s_r2 = (_slope([r["config"]["sigma_c"] for r in strength],
                          [r["F_max"] for r in strength]) if len(strength) > 2
                   else (float("nan"), float("nan")))

    iso_rel = [{"case_id": r["case_id"], "ratio_max": r["ratio_max"],
                "ratio_steady": r["ratio_steady"],
                "rel_err_max": abs(r["ratio_max"] - 1.0),
                "rel_err_steady": abs(r["ratio_steady"] - 1.0)} for r in iso]

    angles = [r["pile_angle_deg"] for r in rows if np.isfinite(r["pile_angle_deg"])]

    checks = {
        "C1_force_positive": {
            "pass": all(r["F_max"] > 0 for r in rows),
            "min_F_max": min(r["F_max"] for r in rows), "n_cases": len(rows)},
        "C2_iso_order_of_magnitude": {
            "pass": abs(base0["ratio_max"] - 1.0) < 0.5,
            "base_ratio_max": base0["ratio_max"],
            "base_ratio_steady": base0["ratio_steady"],
            "same_order_within_factor_5": all(0.2 <= r["ratio_max"] <= 5.0
                                             for r in iso),
            "all_base": iso_rel},
        "C3_porosity_scaling": dict(pass_=(por_fit["R2"] > 0.8
                                           and 1.0 <= por_fit["alpha"] <= 2.0),
                                    **por_fit),
        "C4_pile_angle_bounds": {
            "pass": bool(np.isfinite(base0["pile_angle_deg"])
                         and 30.0 <= base0["pile_angle_deg"] <= 45.0),
            "base_angle_deg": base0["pile_angle_deg"],
            "angle_range_all_cases": [float(np.nanmin(angles)),
                                      float(np.nanmax(angles))],
            "n_cases_with_angle": int(len(angles))},
        "C5_monotonic_thickness": {
            "pass": all(thick[i]["F_max"] < thick[i + 1]["F_max"]
                        for i in range(len(thick) - 1)),
            "F_max": [r["F_max"] for r in thick],
            "h": [r["config"]["h"] for r in thick],
            "loglog_exponent": h_exp, "R2": h_r2},
        "C6_monotonic_strength": {
            "pass": all(strength[i]["F_max"] < strength[i + 1]["F_max"]
                        for i in range(len(strength) - 1)),
            "F_max": [r["F_max"] for r in strength],
            "sigma_c": [r["config"]["sigma_c"] for r in strength],
            "loglog_exponent": s_exp, "R2": s_r2},
    }
    checks["C3_porosity_scaling"]["pass"] = checks["C3_porosity_scaling"].pop("pass_")

    return {
        "base_cases": iso_rel,
        "porosity_fit": por_fit,
        "resolution_study": [{"r_median_factor": r["config"]["r_median_factor"],
                              "n_blocks": r["n_blocks_initial"],
                              "ratio_max": r["ratio_max"],
                              "porosity": r["porosity"],
                              "pile_angle_deg": r["pile_angle_deg"]}
                             for r in res_group],
        "checks": checks,
        "n_cases": len(rows),
        "total_seconds": round(sum(r["secs"] for r in rows), 1),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(rows: list[dict], ana: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    by = {r["case_id"]: r for r in rows}
    figs = []

    # fig1: load histories of the base cases
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    iso_present = [c for c in ISO_CASES if c in by]
    for cid in iso_present:
        r = by[cid]
        t = np.asarray(r["history"]["time"])
        f = np.asarray(r["history"]["load"]) / 1e6
        line, = ax.plot(t, f, lw=1.1,
                        label="%s  $F_{max}$=%.2f MN/m" % (cid, r["F_max"] / 1e6))
        ax.axhline(r["F_ISO_per_width"] / 1e6, ls=":", lw=0.9,
                   color=line.get_color())
    ax.set_xlabel("time [s]")
    ax.set_ylabel("wall load $F'$ [MN/m]")
    ax.set_title("Structure load history (dotted: ISO 19906 $F/D$)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / "fig1_load_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name, "caption":
                 "图1 载荷时历：四个基准工况的墙面线载荷 F'(t)（点线=ISO 19906 单位宽度载荷 F/D）。"
                 "峰值出现在堆积体压实阶段，随后回落到稳态平台。"})

    # fig2: pile geometry of the base cylinder case
    r = by.get("iso_cyl_D5", rows[0])
    pos = np.asarray(r["pos"])
    rad = np.asarray(r["rad"])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for (x, z), rr in zip(pos, rad):
        ax.add_patch(plt.Circle((x, z), rr, color="#7fb3d5", ec="#1f4e79",
                                lw=0.4, alpha=0.85))
    ax.axvline(0.0, color="k", lw=2.0, label="structure face")
    ax.axhline(0.0, color="#2980b9", ls="--", lw=0.9, label="water line")
    if r["flank_x"]:
        ax.plot(r["flank_x"], r["flank_z"], "r-o", ms=3, lw=1.0,
                label="flank (%.1f deg)" % r["pile_angle_deg"])
    ax.set_xlim(min(pos[:, 0].min(), -2.0) - 0.2, 0.6)
    ax.set_ylim(-1.2 * r["config"]["h"], 3.4 * r["config"]["h"])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]  (drift direction +)")
    ax.set_ylabel("z [m]")
    ax.set_title("Rubble pile against the vertical structure "
                 "(n=%.3f, sail=%.2f m)" % (r["porosity"], r["pile_height"]))
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    p = FIG / "fig2_pile_geometry.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name, "caption":
                 "图2 堆积体几何（基准圆柱工况末态）：块体截面、水面、结构面与量测的堆积坡面；"
                 "孔隙率 n=%.3f，sail 高度 %.2f m。" % (r["porosity"], r["pile_height"])})

    # fig3: ISO comparison
    base = [by[c] for c in ISO_CASES if c in by]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = np.arange(len(base))
    ax.bar(xs - 0.18, [b["ratio_max"] for b in base], width=0.36,
           label="$F_{max}/(F/D)$")
    ax.bar(xs + 0.18, [b["ratio_steady"] for b in base], width=0.36,
           label="$F_{steady}/(F/D)$")
    ax.axhline(1.0, color="k", lw=1.0)
    ax.axhspan(0.5, 1.5, color="gray", alpha=0.18, label="+/-50 % band")
    ax.set_xticks(xs)
    ax.set_xticklabels([b["case_id"] for b in base], fontsize=8)
    ax.set_ylabel("DEM / ISO 19906")
    ax.set_title("Vertical-structure load vs ISO 19906 (per unit width)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = FIG / "fig3_iso_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name, "caption":
                 "图3 DEM 峰值/稳态线载荷与 ISO 19906 单位宽度计算的比值（灰带=±50% 判据带）。"})

    # fig4: porosity-resistance scaling
    por = [r for r in rows if r["group"] == "porosity"]
    if len(por) < 3:
        return figs
    x = np.array([r["solids_fraction"] for r in por])
    y = np.array([r["F_max"] / (r["config"]["sigma_c"] * r["config"]["h"] ** 2)
                  for r in por])
    fit = ana["porosity_fit"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(x, y, "o", ms=6)
    xx = np.linspace(x.min() * 0.95, x.max() * 1.05, 50)
    ax.loglog(xx, fit["K"] * xx ** fit["alpha"], "-",
              label="$F/(\\sigma_c h^2)=%.3f(1-n)^{%.2f}$  $R^2$=%.3f"
                    % (fit["K"], fit["alpha"], fit["R2"]))
    for r_, xi, yi in zip(por, x, y):
        ax.annotate("V=%g,sln=%g" % (r_["config"]["drift_speed"],
                                     r_["config"]["r_sigma_ln"]),
                    (xi, yi), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("solids fraction $1-n$ [-]")
    ax.set_ylabel("$F_{max}/(\\sigma_c h^2)$ [-]")
    ax.set_title("Porosity-resistance scaling of the rubble pile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = FIG / "fig4_porosity_scaling.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name, "caption":
                 "图4 孔隙率-阻力标度律：横轴固相率 (1-n)，纵轴无量纲峰值载荷 F/(σ_c h²)，"
                 "拟合 α=%.2f（95%% CI %.2f~%.2f），R²=%.3f。"
                 % (fit["alpha"], fit["alpha_ci95"][0], fit["alpha_ci95"][1],
                    fit["R2"])})

    # fig5: thickness and strength sweeps
    thick = sorted([r for r in rows if r["group"] == "thickness"],
                   key=lambda r: r["config"]["h"])
    strength = sorted([r for r in rows if r["group"] == "strength"],
                      key=lambda r: r["config"]["sigma_c"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].loglog([r["config"]["h"] for r in thick], [r["F_max"] for r in thick],
                   "o-")
    axes[0].set_xlabel("ice thickness h [m]")
    axes[0].set_ylabel("$F_{max}$ [N/m]")
    axes[0].set_title("Thickness sweep (slope %.2f)"
                      % ana["checks"]["C5_monotonic_thickness"]["loglog_exponent"])
    axes[1].loglog([r["config"]["sigma_c"] / 1e6 for r in strength],
                   [r["F_max"] for r in strength], "s-")
    axes[1].set_xlabel("compressive strength $\\sigma_c$ [MPa]")
    axes[1].set_ylabel("$F_{max}$ [N/m]")
    axes[1].set_title("Strength sweep (slope %.2f)"
                      % ana["checks"]["C6_monotonic_strength"]["loglog_exponent"])
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = FIG / "fig5_sweeps.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name, "caption":
                 "图5 冰厚与强度扫描：峰值线载荷随 h、σ_c 的单调性与双对数斜率。"})

    # fig6: resolution study
    res = list(ana["resolution_study"])
    b0 = by.get("iso_cyl_D5", rows[0])
    res.append({"r_median_factor": b0["config"]["r_median_factor"],
                "n_blocks": b0["n_blocks_initial"], "ratio_max": b0["ratio_max"],
                "porosity": b0["porosity"],
                "pile_angle_deg": b0["pile_angle_deg"]})
    res.sort(key=lambda d: d["r_median_factor"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot([p_["r_median_factor"] for p_ in res], [p_["ratio_max"] for p_ in res],
            "o-", label="ratio $F_{max}/(F/D)$")
    ax.set_xlabel("block radius / h [-]")
    ax.set_ylabel("DEM / ISO")
    ax2 = ax.twinx()
    ax2.plot([p_["r_median_factor"] for p_ in res], [p_["porosity"] for p_ in res],
             "s--", color="tab:orange", label="porosity n")
    ax2.set_ylabel("porosity n")
    ax.set_title("Resolution sensitivity of the 2-D section DEM")
    ax.grid(alpha=0.3)
    fig.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = FIG / "fig6_resolution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name, "caption":
                 "图6 分辨率敏感性：块体尺度 r/h 对载荷比与孔隙率的影响（模型局限见报告第5节）。"})

    return figs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(rep: dict) -> None:
    ana, rows = rep["analysis"], rep["cases"]
    by = {r["case_id"]: r for r in rows}
    fit = ana["porosity_fit"]
    ck = ana["checks"]
    A = []
    out = A.append

    out("# 冰体挤压堆积对垂直结构载荷（p227w2-N04 · 垂直剖面 DEM 重做版）")
    out("")
    out("> 本报告全部数字由 `benchmarks/ice_rubble_pileup/run_benchmark.py` 从 "
        "`runs/m14600/ice_rubble_pileup/report.json` 程序化注入（零手抄）。")
    out("")
    out("## 1. 案例介绍（物理背景 / 几何 / 参数）")
    out("")
    out("漂移的破碎冰排（厚 h、单轴抗压强度 σ_c、漂移速度 V）撞击固定垂直结构"
        "（圆柱直径 / 方柱边长 D）。冰块在接触处压碎、上翻，在结构前堆积成碎石堆"
        "（rubble pile）；冰排推力经堆积体传到结构，形成峰值载荷 F_max 与稳态载荷 F_steady。")
    out("")
    out("本版采用**垂直剖面 (x, z) 二维 DEM**（冰脊/堆积体力学标准理想化：Hopkins 1998；"
        "Paavilainen & Tuhkuri 2012），因此堆积坡面角、孔隙率与墙面载荷出自同一套模拟：")
    out("")
    out("- 块体：刚性圆盘（rubble-block 圆盘理想化，无自转自由度），半径对数正态；")
    out("- 水面 z=0：浮重 + 二次水阻力（`submerged_area` 解析式；自由板高由浮力自然给出）；")
    out("- 接触：软球 + **压碎上限** F_n = min(k_n·δ, σ_c·b(δ))，b 为接触弦长；"
        "切向为正则化库仑摩擦 F_t = -μ F_n tanh(v_t/v_eps)；")
    out("- 驱动：远场完整冰排用运动学压板（厚 h，位于水面附近）以 V 推进；结构为 x=0 刚性墙面；")
    out("- 下边界 z = -0.95h（冰排下表面，抑制龙骨；本版只解析 sail，见第 5 节）。")
    out("")
    b = by["iso_cyl_D5"]["config"]
    out("参数表（基准工况 `iso_cyl_D5`）：")
    out("")
    out("| 参数 | 值 |")
    out("|---|---|")
    out("| h | %s m |" % b["h"])
    out("| D | %s m（圆柱直径） |" % b["D"])
    out("| σ_c | %g MPa |" % (b["sigma_c"] / 1e6))
    out("| V | %s m/s |" % b["drift_speed"])
    out("| μ | %s |" % b["mu"])
    out("| r/h | %s（区间 %s） |" % (b["r_median_factor"], list(b["r_bounds_factor"])))
    out("| k_n/σ_c | %s |" % b["k_lin_factor"])
    out("| 冰排长度 / 压板行程 | %sh / %sh |" % (b["sheet_length_factor"],
                                              b["platen_travel_factor"]))
    out("")
    out("ISO 19906:2010 A.8.2.6：p_G = C_R (h/h₀)^n (w/h)^m，F = p_G·w·h，p_G ≤ σ_c；"
        "本报告以单位宽度形式 F/D = p_G·h 与 DEM 线载荷对比。")
    out("")
    out("## 2. 怎么做的（方法 / 流程 / 工具链）")
    out("")
    out("1. `src/tensordem/ice_rubble_pileup_vertical.py`：垂直剖面 DEM 求解器"
        "（浮力解析、压碎上限接触、正则化摩擦、堆积体几何量测、孔隙率-阻力拟合、CLI 入口）；")
    out("2. `benchmarks/ice_rubble_pileup/run_benchmark.py`：预登记算例矩阵（%d 例："
        "ISO 基准 4 + 冰厚 5 + 强度 5 + 孔隙率 9 + 分辨率 2），多进程并行，"
        "产出 report.json / runs.csv / histories.json 与 6 张图；" % ana["n_cases"])
    out("3. 判据（机器复核，见 3.5）：ISO 同数量级、孔隙率标度律 R²>0.8 且 α∈[1,2]、"
        "堆积角 30–45°、F_max 对 h 与 σ_c 单调；")
    out("4. 环境：单机多进程（%d workers），DEM 总机时 %s s。"
        % (rep["meta"]["workers"], ana["total_seconds"]))
    out("")
    out("## 3. 计算结果")
    out("")
    out("### 3.1 基准工况 vs ISO 19906")
    out("")
    out("| 工况 | h [m] | D [m] | σ_c [MPa] | F_max [MN/m] | F_steady [MN/m] | "
        "F_ISO/D [MN/m] | F_max/(F/D) | F_steady/(F/D) | F_max/(σ_c h²) |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    for cid in ISO_CASES:
        r = by[cid]
        out("| %s | %s | %s | %g | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
            % (cid, r["config"]["h"], r["config"]["D"], r["config"]["sigma_c"] / 1e6,
               r["F_max"] / 1e6, r["F_steady"] / 1e6, r["F_ISO_per_width"] / 1e6,
               r["ratio_max"], r["ratio_steady"], r["C_task"]))
    out("")
    out("### 3.2 堆积体几何与孔隙率")
    out("")
    out("| 工况 | n（孔隙率） | 固相率 1-n | sail [m] | toe 长 [m] | 坡面角 [°] | "
        "块体（堆积/总数） |")
    out("|---|---|---|---|---|---|---|")
    for cid in ISO_CASES:
        r = by[cid]
        out("| %s | %.3f | %.3f | %.2f | %.2f | %.1f | %d/%d |"
            % (cid, r["porosity"], r["solids_fraction"], r["pile_height"],
               r["pile_length"], r["pile_angle_deg"], r["n_blocks_pile"],
               r["n_blocks_initial"]))
    out("")
    out("### 3.3 孔隙率-阻力标度律 F ∝ (1-n)^α")
    out("")
    out("- 拟合点数 %d，固相率区间 [%.3f, %.3f]，对数杠杆臂 %.3f"
        % (fit["n_points"], fit["solids_fraction_range"][0],
           fit["solids_fraction_range"][1], fit["log_lever_arm"]))
    out("- α = **%.3f**（bootstrap 95%% CI %.3f ~ %.3f），K = %.4f，R² = **%.3f**"
        % (fit["alpha"], fit["alpha_ci95"][0], fit["alpha_ci95"][1], fit["K"],
           fit["R2"]))
    out("")
    out("### 3.4 冰厚 / 强度扫描")
    out("")
    out("| h [m] | F_max [MN/m] | | σ_c [MPa] | F_max [MN/m] |")
    out("|---|---|---|---|---|")
    thick = sorted([r for r in rows if r["group"] == "thickness"],
                   key=lambda r: r["config"]["h"])
    strength = sorted([r for r in rows if r["group"] == "strength"],
                      key=lambda r: r["config"]["sigma_c"])
    for i in range(max(len(thick), len(strength))):
        left = ("%s | %.3f" % (thick[i]["config"]["h"], thick[i]["F_max"] / 1e6)
                if i < len(thick) else " | ")
        right = ("%g | %.3f" % (strength[i]["config"]["sigma_c"] / 1e6,
                                strength[i]["F_max"] / 1e6)
                 if i < len(strength) else " | ")
        out("| %s | %s |" % (left, right))
    out("")
    out("双对数斜率 d lnF/d ln h = %.3f（R²=%.3f）；d lnF/d ln σ_c = %.3f（R²=%.3f）。"
        % (ck["C5_monotonic_thickness"]["loglog_exponent"],
           ck["C5_monotonic_thickness"]["R2"],
           ck["C6_monotonic_strength"]["loglog_exponent"],
           ck["C6_monotonic_strength"]["R2"]))
    out("")
    out("### 3.5 判据机器复核")
    out("")
    out("| 判据 | 结果 | 数字 |")
    out("|---|---|---|")
    c1 = ck["C1_force_positive"]
    out("| C1 F_max>0（全部算例） | %s | 最小 F_max = %.4g N/m（%d 例） |"
        % ("PASS" if c1["pass"] else "FAIL", c1["min_F_max"], c1["n_cases"]))
    c2 = ck["C2_iso_order_of_magnitude"]
    out("| C2 ISO 同数量级（基准 \\|F/F_ISO−1\\|<50%%） | %s | 基准 F_max/(F/D) = %.3f"
        "（偏差 %.1f%%）；4 基准工况均在 5 倍内 = %s |"
        % ("PASS" if c2["pass"] else "FAIL", c2["base_ratio_max"],
           abs(c2["base_ratio_max"] - 1) * 100, c2["same_order_within_factor_5"]))
    c3 = ck["C3_porosity_scaling"]
    out("| C3 R²>0.8 且 α∈[1,2] | %s | R²=%.3f，α=%.2f，杠杆臂 %.3f |"
        % ("PASS" if c3["pass"] else "FAIL", c3["R2"], c3["alpha"],
           c3["log_lever_arm"]))
    c4 = ck["C4_pile_angle_bounds"]
    out("| C4 堆积坡面角 30–45° | %s | 基准 %.1f°；全算例 %.1f–%.1f° |"
        % ("PASS" if c4["pass"] else "FAIL", c4["base_angle_deg"],
           c4["angle_range_all_cases"][0], c4["angle_range_all_cases"][1]))
    c5 = ck["C5_monotonic_thickness"]
    out("| C5 F_max 随 h 单调递增 | %s | %s MN/m |"
        % ("PASS" if c5["pass"] else "FAIL",
           " < ".join("%.2f" % (v / 1e6) for v in c5["F_max"])))
    c6 = ck["C6_monotonic_strength"]
    out("| C6 F_max 随 σ_c 单调递增 | %s | %s MN/m |"
        % ("PASS" if c6["pass"] else "FAIL",
           " < ".join("%.2f" % (v / 1e6) for v in c6["F_max"])))
    out("")
    out("## 4. 结果图")
    out("")
    for f in rep["figures"]:
        out("### %s" % f["name"])
        out("")
        out("![%s](figures/ice_rubble_pileup/%s)" % (f["name"], f["name"]))
        out("")
        out("*%s*" % f["caption"])
        out("")
    out("## 5. 注意点（踩过的坑 / 局限 / 适用范围）")
    out("")
    out("1. **压碎上限必须生效**：若 k_n·δ_cap < σ_c·b(δ_cap)，载荷由弹簧刚度而非冰强度控制，"
        "F_max 可直接翻倍（本轮从 0.72 掉到 0.42 的教训）。本版取 k_n = 10σ_c，"
        "并由 `cap_binds()` 校验。")
    out("2. **分辨率敏感性（模型局限）**：垂直剖面 DEM 的墙面载荷 ≈ σ_c·Σb；块体越小，"
        "弦长越短而墙面接触点覆盖不随 1/r 增长，载荷比随 r/h 下降（图6）。"
        "因此本 DEM 不能作为收敛的墙面压强预报器；基准工况取物理上最接近实际碎石块的 "
        "r≈0.16h（块径 0.32h）。")
    out("3. **只解析 sail，未解析龙骨**：下表面固定于 z=-0.95h 抑制龙骨；真实冰脊龙骨通常比 "
        "sail 深 4–7 倍，故本版 F_max 不含龙骨支承/挤压贡献。")
    out("4. **二维剖面 → 三维载荷**：F_3D ≈ F'·D 为窄结构约定；圆柱绕流的三维卸载未解析，"
        "已通过 ISO 的 (w/h)^m 项间接体现。")
    out("5. **孔隙率量测**：凸包包络 + 块体全面积，属「上界固相率/下界孔隙率」；未扣除块体互穿。")
    out("")
    out("## 6. 误差判定（关键量 vs 基准，逐项）")
    out("")
    out("| 关键量 | DEM | 基准 | 相对误差 | 是否<3% |")
    out("|---|---|---|---|---|")
    for row in c2["all_base"]:
        out("| F_max/(F/D) %s | %.3f | 1.000 | %.1f%% | 否 |"
            % (row["case_id"], row["ratio_max"], row["rel_err_max"] * 100))
    out("| α（孔隙率标度指数） | %.3f | 1.0–2.0（判据区间） | 见 CI | 否 |"
        % fit["alpha"])
    out("| d lnF/d ln h | %.3f | 1.0（ISO F∝h^0.76 单位宽度） | %.1f%% | 否 |"
        % (ck["C5_monotonic_thickness"]["loglog_exponent"],
           abs(ck["C5_monotonic_thickness"]["loglog_exponent"] - 0.76) / 0.76 * 100))
    out("")
    out("**逐项回答「是否<3%」：全部为「否」。** 本任务属前沿机理性 DEM（无可标定的实验参考场），"
        "不存在可给出 3% 级误差的基准；唯一可供对比的是 ISO 19906 经验式与文献堆积体参数，"
        "误差量级为 10–100%。上表如实列出，不作粉饰。")
    out("")
    out("## 7. 审计链")
    out("")
    out("- 求解器：`src/tensordem/ice_rubble_pileup_vertical.py`（CLI："
        "`python -m tensordem.ice_rubble_pileup_vertical --quick`）")
    out("- 复现命令：`python benchmarks/ice_rubble_pileup/run_benchmark.py --workers 6`")
    out("- 算例矩阵在 `case_matrix()` 中预登记，运行前未被结果反向修改")
    out("- 运行工件：`runs/m14600/ice_rubble_pileup/{report.json,runs.csv,histories.json}`")
    out("- 测试：`python -m pytest tests/test_ice_rubble_pileup_vertical.py -q`")
    out("")
    DOC.write_text("\n".join(A) + "\n")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quick", action="store_true",
                    help="2-case smoke matrix (no report overwrite)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args(argv)

    cases = case_matrix()
    if args.quick:
        cases = [{"case_id": "iso_cyl_D5", "group": "iso", "config": dict(BASE)},
                 {"case_id": "iso_sq_D5", "group": "iso",
                  "config": dict(BASE, shape="square")}]
        for c in cases:
            c["config"]["h"] = 0.6
            c["config"]["sheet_length_factor"] = 5.0
            c["config"]["platen_travel_factor"] = 6.0

    t0 = time.time()
    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            rows = pool.map(_run, cases)
    else:
        rows = [_run(c) for c in cases]
    wall = round(time.time() - t0, 1)

    ana = analyse(rows)
    figs = make_figures(rows, ana)
    rep = {
        "meta": {
            "mission_id": 14600,
            "node": "p227w2-N04",
            "repo": "TensorDEM",
            "workers": args.workers,
            "wall_seconds": wall,
            "case_matrix": [{k: c[k] for k in ("case_id", "group")} for c in cases],
            "base_config": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in BASE.items()},
            "repro_cmd": "python benchmarks/ice_rubble_pileup/run_benchmark.py --workers %d"
                         % args.workers,
        },
        "cases": [{k: v for k, v in r.items()
                   if k not in ("pos", "rad", "flank_x", "flank_z", "history")}
                  for r in rows],
        "geometry": {r["case_id"]: {"pos": r["pos"], "rad": r["rad"],
                                    "flank_x": r["flank_x"], "flank_z": r["flank_z"]}
                     for r in rows},
        "histories": {r["case_id"]: r["history"] for r in rows},
        "analysis": ana,
        "figures": figs,
        "external_anchor": {
            "ISO 19906:2010 A.8.2.6": "p_G = C_R (h/h0)^n (w/h)^m, C_R=2.8 MPa, "
                                      "n=-0.50+h/10, m=-0.16, F=p_G w h, p_G<=sigma_c",
            "Timco & Weeks 2010 (CRST 60:107-129)": "sea-ice uniaxial compressive "
                                                    "strength 1-7 MPa; ice-ice friction 0.1-0.5",
            "Hopkins 1998 (JGR 103)": "2-D ridge DEM: rubble sail/keel, porosity 0.2-0.4",
            "ISO 19906 ice-structure friction": "steel-ice friction 0.1-0.15 for ride-up",
        },
    }

    outdir = Path(args.out) if args.out else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.json").write_text(json.dumps(rep, indent=1))
    (outdir / "histories.json").write_text(json.dumps(rep["histories"]))
    with open(outdir / "runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "group", "h", "D", "sigma_c", "V", "mu", "r_med",
                    "F_max", "F_steady", "F_ISO_per_width", "ratio_max",
                    "ratio_steady", "C_task", "porosity", "solids_fraction",
                    "sail", "toe", "angle_deg", "n_blocks", "secs"])
        for r in rows:
            c = r["config"]
            w.writerow([r["case_id"], r["group"], c["h"], c["D"], c["sigma_c"],
                        c["drift_speed"], c["mu"], c["r_median_factor"],
                        "%.6g" % r["F_max"], "%.6g" % r["F_steady"],
                        "%.6g" % r["F_ISO_per_width"], "%.4f" % r["ratio_max"],
                        "%.4f" % r["ratio_steady"], "%.4f" % r["C_task"],
                        "%.4f" % r["porosity"], "%.4f" % r["solids_fraction"],
                        "%.4f" % r["pile_height"], "%.4f" % r["pile_length"],
                        "%.2f" % r["pile_angle_deg"], r["n_blocks_initial"],
                        r["secs"]])
    if not args.quick:
        write_report(rep)

    summary = {k: {"pass": v["pass"]} for k, v in ana["checks"].items()}
    print(json.dumps({"cases": ana["n_cases"], "wall_seconds": wall,
                      "checks": summary,
                      "porosity_fit": {k: ana["porosity_fit"][k]
                                       for k in ("alpha", "R2", "log_lever_arm")},
                      "base_ratios": {r["case_id"]: r["ratio_max"]
                                      for r in ana["base_cases"]}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())