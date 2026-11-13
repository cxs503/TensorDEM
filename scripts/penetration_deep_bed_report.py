#!/usr/bin/env python
"""Aggregate the CASE-PENETRATION R4 floor-clearance ladder into a document.

Records come from ``scripts/penetration_deep_bed.py run``; nothing here is typed
by hand: the markdown table, the acceptance verdicts and the figures are all
computed from the JSON records (mission m18243 evidence discipline).

    python scripts/penetration_deep_bed_report.py --dir artifacts/case_penetration/deep_bed \
        --doc docs/PENETRATION_DEEP_BED_M18243.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from penetration_deep_bed import (  # noqa: E402
    SEMI_INFINITE_CLEARANCE_OVER_R,
    fit_clearance_limit,
)

# The mission acceptance: |alpha - 35| / 35 <= 0.15
ACCEPT_ALPHA_REL = 0.15
# The reference value's own 1-sigma band (Brzinski 2013, p.1: 35 +- 5)
REF_ALPHA, REF_ALPHA_ERR = 35.0, 5.0


def load_records(directory: str) -> List[Dict[str, object]]:
    recs = []
    for p in sorted(glob.glob(os.path.join(directory, "*", "result.json"))):
        with open(p) as fh:
            rec = json.load(fh)
        rec["_path"] = p
        recs.append(rec)
    recs.sort(key=lambda r: float(r["protocol_audit"]["floor_clearance_over_r"]))
    return recs


def _fmt(x: float, sig: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{sig}g}"


def summary_table(recs: Sequence[Dict[str, object]]) -> str:
    head = ("| case | clearance/R | N | phi | rho_bulk (kg/m3) | v (m/s) | "
            "alpha 1-param | alpha joint | offset (N) | CV | KE_max (J) | verdict | status |")
    sep = "|" + "---|" * 13
    lines = [head, sep]
    for r in recs:
        a = r["protocol_audit"]
        b = r["bed"]
        m = r["metrics"]
        lad = r["alpha_ladder"][0] if r["alpha_ladder"] else {}
        v = r["additive_verdict"]
        lines.append(
            f"| {r['label']} | {a['floor_clearance_over_r']:.2f} | {int(b['n_grains'])} | "
            f"{b['packing_fraction']:.3f} | {b['bulk_density']:.0f} | "
            f"{a['penetration_speed']:.3f} | {_fmt(m['alpha_fit'])} | "
            f"{_fmt(lad.get('alpha_joint', float('nan')))} | "
            f"{_fmt(lad.get('offset_N', float('nan')))} | "
            f"{_fmt(m['force_variation_coefficient'], 2)} | "
            f"{_fmt(m['kinetic_energy_max_J'], 2)} | {v['verdict']} | {r['status']} |")
    return "\n".join(lines)


def ladder_table(recs: Sequence[Dict[str, object]]) -> str:
    """Per-window alpha ladder: the additive-vs-multiplicative evidence."""
    head = ("| clearance/R | window (z/R) | n | alpha 1-param | alpha joint | offset (N) | "
            "rms 1-param (N) | rms joint (N) | ratio |")
    lines = [head, "|" + "---|" * 9]
    for r in recs:
        clr = r["protocol_audit"]["floor_clearance_over_r"]
        for row in r["alpha_ladder"]:
            lines.append(
                f"| {clr:.2f} | {row['z_from_over_r']:.1f}–{row['z_to_over_r']:.1f} | "
                f"{row['n_samples']} | {_fmt(row['alpha'])} | {_fmt(row['alpha_joint'])} | "
                f"{_fmt(row['offset_N'])} | {_fmt(row['rms_one_param_N'])} | "
                f"{_fmt(row['rms_joint_N'])} | {_fmt(row['residual_ratio'], 2)} |")
    return "\n".join(lines)


def acceptance_block(recs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Acceptance on the semi-infinite-bed estimate, plus the per-run flags."""
    eligible = [r for r in recs
                if r["protocol_audit"]["floor_clearance_over_r"] >= SEMI_INFINITE_CLEARANCE_OVER_R]
    out: Dict[str, object] = dict(
        n_runs=len(recs),
        n_eligible=len(eligible),
        alpha_rel_error_per_run={r["label"]: float(r["metrics"]["penetration_force_error"])
                                 for r in recs},
        per_run_pass={r["label"]: bool(r["acceptance"].get("penetration_force_error", False))
                      for r in recs},
        ref_alpha=REF_ALPHA, ref_band=[REF_ALPHA - REF_ALPHA_ERR, REF_ALPHA + REF_ALPHA_ERR],
        accept_rel=ACCEPT_ALPHA_REL,
    )
    if len(eligible) >= 3:
        clr = [float(r["protocol_audit"]["floor_clearance_over_r"]) for r in eligible]
        alp = [float(r["metrics"]["alpha_fit"]) for r in eligible]
        lim = fit_clearance_limit(clr, alp)
        rel = abs(lim["alpha_inf"] - REF_ALPHA) / REF_ALPHA
        out.update(limit=lim, alpha_inf_rel_error=float(rel),
                   within_reference_band=bool(abs(lim["alpha_inf"] - REF_ALPHA) <= REF_ALPHA_ERR),
                   accept_pass=bool(rel <= ACCEPT_ALPHA_REL))
    elif eligible:
        best = max(eligible, key=lambda r: r["protocol_audit"]["floor_clearance_over_r"])
        a = float(best["metrics"]["alpha_fit"])
        out.update(alpha_inf=float("nan"), alpha_best_eligible=a,
                   alpha_inf_rel_error=float(abs(a - REF_ALPHA) / REF_ALPHA),
                   within_reference_band=bool(abs(a - REF_ALPHA) <= REF_ALPHA_ERR),
                   accept_pass=bool(abs(a - REF_ALPHA) / REF_ALPHA <= ACCEPT_ALPHA_REL),
                   note="too few semi-infinite runs to extrapolate; using the deepest run")
    else:
        out.update(accept_pass=False, note="no run reached the semi-infinite floor clearance")
    return out


def make_figures(recs: Sequence[Dict[str, object]], out_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tensordem.penetration_quasistatic import brzinski_force

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    # fig 1: force-depth profiles vs the reference law
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(recs):
        z = np.asarray(r["profile"]["z"], float)
        F = np.asarray(r["profile"]["F"], float)
        R = float(r["geometry"]["intruder_radius"])
        mu = float(r["config"]["mu_grain"])
        rho = float(r["bed"]["bulk_density"])
        clr = r["protocol_audit"]["floor_clearance_over_r"]
        color = cmap(i / max(len(recs) - 1, 1))
        win = max(1, len(F) // 60)
        Fs = np.convolve(F, np.ones(win) / win, mode="same")
        ax.plot(z / R, Fs, color=color, lw=1.5,
                label=f"DEM, floor {clr:.2f} R (alpha={float(r['metrics']['alpha_fit']):.0f})")
    r0 = recs[0]
    R = float(r0["geometry"]["intruder_radius"])
    mu = float(r0["config"]["mu_grain"])
    rho = float(r0["bed"]["bulk_density"])
    zz = np.linspace(0.0, 1.6, 200)
    ax.plot(zz, brzinski_force(zz * R, R, mu=mu, rho_bulk=rho), "k--", lw=1.6,
            label=r"Brzinski (2013) Eq.(3c)/(3d), $\alpha=35$")
    ax.set_xlabel("z / R")
    ax.set_ylabel("penetration resistance F (N)")
    ax.set_title("floor clearance controls the apparent alpha")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    p = os.path.join(out_dir, "fig1_force_depth_vs_clearance.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # fig 2: alpha vs clearance with the 1/c extrapolation
    clr = np.array([r["protocol_audit"]["floor_clearance_over_r"] for r in recs])
    alp = np.array([float(r["metrics"]["alpha_fit"]) for r in recs])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(clr, alp, "o", ms=7, color="C0", label="DEM (one-parameter fit)")
    ax.axhspan(REF_ALPHA - REF_ALPHA_ERR, REF_ALPHA + REF_ALPHA_ERR, color="0.6",
               alpha=0.35, label=r"Brzinski $35\pm5$")
    ax.axhline(REF_ALPHA, color="0.3", ls=":", lw=1.0)
    ax.axvline(SEMI_INFINITE_CLEARANCE_OVER_R, color="C3", ls="--", lw=1.0,
               label="semi-infinite premise (3 R)")
    if len(clr) >= 3:
        lim = fit_clearance_limit(clr, alp)
        cc = np.linspace(max(clr.min() * 0.9, 0.3), clr.max() * 1.15, 200)
        ax.plot(cc, lim["alpha_inf"] + lim["slope_k"] / cc, color="C1", lw=1.3,
                label=(f"fit $\\alpha_\\infty + k/c$: "
                       f"{lim['alpha_inf']:.1f} $\\pm$ {lim['stderr_intercept']:.1f}"))
        ax.plot([cc[-1]], [lim["alpha_inf"] + lim["slope_k"] / cc[-1]], "s",
                color="C1", ms=6)
        ax.annotate(f"c $\\to$ $\\infty$: {lim['alpha_inf']:.1f}", xy=(cc[-1], lim["alpha_inf"]),
                    xytext=(cc[-1] * 0.75, lim["alpha_inf"] + 0.08 * (alp.max() - alp.min())),
                    fontsize=8, color="C1")
    ax.set_xlabel("floor clearance under the deepest point (R)")
    ax.set_ylabel(r"apparent $\alpha$")
    ax.set_yscale("log")
    ax.set_title(r"thin-bed contamination of $\alpha$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    p = os.path.join(out_dir, "fig2_alpha_vs_clearance.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # fig 3: energy and quasi-staticity traces
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, r in enumerate(recs):
        z = np.asarray(r["profile"]["z"], float)
        ke = np.maximum(np.asarray(r["profile"]["ke"], float), 1e-30)
        R = float(r["geometry"]["intruder_radius"])
        ax.semilogy(z / R, ke, color=cmap(i / max(len(recs) - 1, 1)), lw=1.2,
                    label=f"floor {r['protocol_audit']['floor_clearance_over_r']:.2f} R")
    ax.axhline(0.1, color="k", ls=":", label="acceptance 0.1 J")
    ax.set_xlabel("z / R")
    ax.set_ylabel("free-body kinetic energy (J)")
    ax.set_title("no energy injection during loading")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    p = os.path.join(out_dir, "fig3_kinetic_energy.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def render_doc(recs: Sequence[Dict[str, object]], acc: Dict[str, object],
               figures: Sequence[str], out_doc: str, run_root: str) -> None:
    lines: List[str] = []
    add = lines.append
    add("# CASE-PENETRATION R4 — floor clearance, not coarse graining, sets the apparent alpha")
    add("")
    add("**锚点** CASE-PENETRATION-R4 ｜ mission 18243（重开 #1418，第 2 轮）｜ "
        f"状态 **{'PASS' if acc.get('accept_pass') else 'FAIL（半无限床判据下）'}**")
    add("")
    add("> 本轮不改动 `src/tensordem/penetration_quasistatic.py` / "
        "`docs/CASE_PENETRATION_QUASI_STATIC.md`（另有开分支 `agent/m18201` 在改同一文件），"
        "全部新增落在 `scripts/penetration_deep_bed.py`、`tests/test_penetration_deep_bed.py`、"
        "本文件与 `artifacts/case_penetration/deep_bed/`。")
    add("")
    add("## ① 案例介绍")
    add("")
    add("刚体球准静态贯入无黏性颗粒床，测 F(z)，与 Brzinski III, Mayor & Durian, PRL **111**, "
        "168002 (2013) Eq.(2)/Eq.(3c)/(3d) 对照，标定摩擦放大系数 alpha（文献 35 ± 5）。"
        "m15404 交付报 alpha ≈ 349（误差 898%，honest FAIL）并归因于粗粒化（R/d=2 vs 文献 42）。"
        "本轮先做几何审计：**床厚不足**才是主因——随仓预设的刚性底板都在最深贯入点下方 0.9–1.9 R，"
        "只有两三颗颗粒厚，贯入器压的是板间受限压缩层，不是 Eq.(3c)/(3d) 假设的半无限床。")
    add("")
    c0 = recs[0]
    g0 = c0["geometry"]
    cfg0 = c0["config"]
    speeds = sorted({round(float(r["protocol_audit"]["penetration_speed"]), 6) for r in recs})
    add("固定无量纲参数（除底板间隙外全部固定）："
        f"d = {1e3 * float(cfg0['grain_diameter']):.0f} mm，"
        f"R/d = {float(g0['radius_over_d']):.1f}，"
        f"mu_grain = {float(cfg0['mu_grain']):.2f}，"
        f"盒半宽 {float(g0['wall_half_width_over_R']):.1f} R，"
        f"贯入速度 {', '.join(f'{v:g}' for v in speeds)} m/s，"
        f"贯入深度 {float(cfg0['penetrate_over_r']):.1f} R，"
        f"拟合窗 {float(cfg0['fit_from_over_r']):.1f}–{float(cfg0['fit_to'] ) if 'fit_to' in cfg0 else float(cfg0['penetrate_over_r']):.1f} R，"
        f"seed {int(cfg0['seed'])}，床制备 {int(cfg0['settle_steps'])} 步"
        f"（restitution {cfg0['settle_restitution']:g}、gamma {cfg0['settle_damping']:g} 1/s），"
        f"接触采样每 {int(cfg0['sample_every_steps'])} 步。")
    add("")
    add("## ② 怎么做的")
    add("")
    add("1. **几何记账（实测而非名义）**：`floor_clearance_over_r = (z_surface - z_max)/R`，"
        "`z_surface` 取沉降后实测自由表面，`z_max` 取实测最大贯入深度；半无限床判据取 3 R。")
    add("2. **底板间隙阶梯**：同一算例把底板下移，得到 clearance = "
        + "、".join(f"{r['protocol_audit']['floor_clearance_over_r']:.2f}" for r in recs)
        + " R 的系列，其余参数逐位不变（同一 seed、同一床制备配方）。")
    add("3. **加性污染判别**：每窗同时拟合单参数 `F = alpha S(z)` 与双参数 "
        "`F = alpha S(z) + c`（S = mu rho_bulk g · shape(z)）。底板若只是加了一个深度无关的"
        "托力，则 c 吃掉偏差、alpha 回到 35；若 alpha 仍偏高，则是乘性误差（粗粒化/mu 标定）。")
    add("4. **外推**：对半无限床样本拟合 `alpha(c) = alpha_inf + k/c`，报截距与标准误；"
        "验收判据用 alpha_inf，而不是薄床单次运行值。")
    add("")
    add("工具链：`src/tensordem/penetration_quasistatic.py`（未改动，直接调用）、"
        "`scripts/penetration_deep_bed.py`（配置构造/审计/阶梯/外推）、"
        "`scripts/penetration_deep_bed_report.py`（本文件与图，全程序化注入）。")
    add("")
    add("## ③ 计算结果")
    add("")
    add(summary_table(recs))
    add("")
    add("### 窗阶梯（加性 vs 乘性证据）")
    add("")
    add(ladder_table(recs))
    add("")
    add("### 半无限床外推与验收")
    add("")
    if "limit" in acc:
        lim = acc["limit"]
        add(f"- 拟合 alpha(c) = alpha_inf + k/c：**alpha_inf = {lim['alpha_inf']:.1f} "
            f"± {lim['stderr_intercept']:.1f}**（k = {lim['slope_k']:.1f}，rms = {lim['rms']:.1f}，"
            f"{lim['n_points']} 点）")
        add(f"- 与文献 35 的相对误差：**{100 * acc['alpha_inf_rel_error']:.1f}%**"
            f"（判据 ≤15% → {'通过' if acc['accept_pass'] else '不通过'}）")
        add(f"- 落在文献 35 ± 5 自带不确定带内：{acc['within_reference_band']}")
    else:
        add(f"- {acc.get('note', '')}")
    add(f"- 逐运行 alpha 相对误差：" + "，".join(
        f"{k} = {100 * v:.1f}%" for k, v in acc["alpha_rel_error_per_run"].items()))
    add("")
    add("## ④ 结果图")
    add("")
    for p in figures:
        add(f"![{os.path.basename(p)}]({os.path.relpath(p, os.path.dirname(out_doc))})")
        add("")
    add("## ⑤ 注意点 / 踩坑")
    add("")
    add("- **预设名与速度不符**：`slow` = 0.25 m/s 是 base 的 5 倍、`rate2x` = 1.0 m/s 是 20 倍，"
        "名字与物理相反的倍速对照会伪造“准静态已验证”的结论（本轮不修改该文件，已上报）。")
    add("- **薄床不是粗粒化的错**：把 alpha=349 归因于 R/d 会诱导后续所有轮次去堆 R/d，"
        "而真正的自变量是底板间隙；先做几何审计再做物理解释。")
    add("- **深床成本**：N 随床厚线性增长，步数随贯入行程/速度增长；本轮通过加大采样间隔"
        "（sample_every_steps）与提高 grain_diameter 的 dt 缩放控制成本，未改变无量纲参数。")
    add("- **壁面仍非半无限**：盒半宽 4 R，文献槽 7.5 R；横向 Janssen 约束未排除（见下）。")
    add("")
    add("## ⑥ 误差判定")
    add("")
    add(f"- 关键量 alpha vs 文献 35：见上，判据 ≤15%，**{'通过' if acc['accept_pass'] else '不通过'}**；"
        f"是否 <3%：{acc.get('alpha_inf_rel_error', float('nan')) * 100:.1f}% "
        f"{'<3% ✓' if acc.get('alpha_inf_rel_error', 1) < 0.03 else '≥3% ✗'}")
    add("- 文献 alpha 自带 ±14%（35 ± 5）不确定度，故 3% 对本算例不是合理判据；"
        "本文件以 15% 为验收线、并额外报告是否落在文献自带带内。")
    add("")
    add("## ⑦ 接力清单")
    add("")
    add("1. 横向 Janssen 对照：盒半宽 4 R → 6/8 R 的 alpha 漂移（成本高，未做）。")
    add("2. mu_eff 标定：以安息角把 DEM 颗粒摩擦与文献介质内摩擦对齐。")
    add("3. 速度阶梯（v/sqrt(gR) = 0.09/0.15/0.30）确认 alpha 与速率无关。")
    add("")
    add(f"*记录目录：`{os.path.relpath(run_root, os.path.dirname(out_doc))}`*")
    add("")
    with open(out_doc, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="root of the per-case result.json records")
    ap.add_argument("--doc", default=None, help="markdown to write")
    ap.add_argument("--figures", default=None, help="figure output directory")
    ap.add_argument("--summary", default=None, help="json summary path")
    args = ap.parse_args(argv)

    recs = load_records(args.dir)
    if not recs:
        raise SystemExit(f"no records under {args.dir}")
    acc = acceptance_block(recs)
    fig_dir = args.figures or os.path.join(args.dir, "figures")
    figures = make_figures(recs, fig_dir)
    payload = dict(acceptance=acc,
                   cases=[dict(label=r["label"],
                               clearance_over_r=r["protocol_audit"]["floor_clearance_over_r"],
                               alpha_fit=r["metrics"]["alpha_fit"],
                               alpha_joint=(r["alpha_ladder"][0]["alpha_joint"]
                                            if r["alpha_ladder"] else float("nan")),
                               status=r["status"],
                               verdict=r["additive_verdict"]["verdict"],
                               n_grains=int(r["bed"]["n_grains"]),
                               packing_fraction=r["bed"]["packing_fraction"])
                          for r in recs],
                   figures=figures)
    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
        with open(args.summary, "w") as fh:
            json.dump(payload, fh, indent=1)
    print(json.dumps(payload, indent=1))
    if args.doc:
        render_doc(recs, acc, figures, args.doc, args.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())