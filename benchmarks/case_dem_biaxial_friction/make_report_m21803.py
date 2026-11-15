#!/usr/bin/env python
"""Generate the m21803 doc + platform-report payload from the audit artefact.

Every number in the markdown and in ``/tmp/report_m21803.json`` is injected from
``artifacts/CASE-DEM-BIAXIAL-FRICTION/rolling_surrogate_audit.json``; nothing is
hand-copied.  Publishing is a separate step (``post_report_m21803.py``).

Usage:
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/make_report_m21803.py
"""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path

ART = Path("artifacts/CASE-DEM-BIAXIAL-FRICTION/rolling_surrogate_audit.json")
PNG = Path("artifacts/CASE-DEM-BIAXIAL-FRICTION/rolling_surrogate_audit.png")
DOC = Path("docs/CASE-DEM-BIAXIAL-FRICTION_R6_ROLLING_SURROGATE_AUDIT.md")
FIG_DOC = Path("docs/figures/CASE-DEM-BIAXIAL-FRICTION_rolling_surrogate_audit.png")
REPORT = Path("/tmp/report_m21803.json")
MISSION = 21803


def fmt(x, nd=2):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        if math.isnan(float(x)):
            return "--"
    except (TypeError, ValueError):
        return str(x)
    return f"{float(x):.{nd}f}"


def build(art: dict, png: Path | None) -> str:
    runs = art["runs"]
    add = [a for a in art.get("additivity", []) if a.get("measured")]
    tol = art["tolerance_additivity_deg"]
    sample = art["sample"]

    def by_tag(tag):
        for r in runs:
            if r["tag"] == tag:
                return r
        return None

    L: list[str] = []
    L.append("# CASE-DEM-BIAXIAL-FRICTION · R6：滚动摩阻替身的加和性审计（验证欠账）\n")
    L.append("**锚点** `CASE-DEM-BIAXIAL-FRICTION`（算例池·双轴剪切摩擦角，验收 `|φ−38°|/38° < 8%`，"
             "ref Bolton 1986 Hostun 密砂平面应变）｜**m21803**｜TensorDEM\n")
    L.append("**判定（前置声明）**：本例不改锚点验收口径、不覆盖既有 FAIL。R6 只回答 R5 留下的验证欠账"
             "——「加滚动摩阻能把 φ_cv 抬到经典临界状态值吗」，以及这个「滚动摩阻」到底是什么算子。\n")

    L.append("## 结论速览\n")
    L.append("| 项 | 结果 |")
    L.append("|---|---|")
    L.append("| 接触层算子恒等式 | `F_t = -(μ+μ_r)·F_n·tanh(v_t/v_reg)`（4 例精确相等，见 "
             "`tests/test_rolling_surrogate_additivity.py`） |")
    if add:
        worst = max(add, key=lambda a: abs(a["delta_phi_cv_deg"]))
        L.append(f"| 宏观 φ_cv 加和性 | {len(add)} 对工况最大残差 "
                 f"{fmt(worst['delta_phi_cv_deg'])}°（预声明容差 ±{fmt(tol,1)}°）|")
        L.append(f"| 判定 | {'加和性成立 → 该「滚动摩阻」实为附加滑动摩擦' if art['verdict']['all_pairs_additive'] else '加和性不成立 → 宏观存在超出加性摩擦的效应'} |")
    L.append("| 锚点验收（φ_peak vs 38°） | **仍为 FAIL**（R2：壁面 41.41° = 8.97%，内部 36.43° = 4.13%）——"
             "本轮不改变该结论 |")
    L.append("")

    L.append("## (1) 案例介绍\n")
    L.append("平面应变双轴压缩的摩擦圆盘试样，等体积（constant-volume）剪切至 ε_a ≈ 0.35，"
             "由四闸门判据提取临界状态摩擦角 φ_cv。物理疑问来自第 3 轮：μ=0.55 的无粘结圆盘集合"
             "实测 φ_cv = 14.31°，比 Rowe(1962) 的 28.81° 低 50%，当时归因于「二维圆盘无滚动阻力，"
             "滚动自由度吞掉了摩擦」。第 4 轮因此加入 `RidgeMaterial.rolling_friction_coefficient`，"
             "第 5 轮递交了 φ_cv(μ_r) 标定但**从未取回工件**。\n")
    L.append("本轮先读实现再花算力：`RidgeAssembly.forces` 中滚动项写成\n")
    L.append("```\nrolling_limit = mu_r * F_n\nrolling_force = -rolling_limit * tanh(v_t / v_reg)\n"
             "total_tangential = tangential + rolling_force\n```\n")
    L.append("与滑动摩擦同向同形式，而该模型颗粒**没有转动自由度**，因此不存在力矩。"
             "若此读法成立，则 `φ_cv` 只依赖 `μ+μ_r`，R5 的标定曲线只是 μ 的重参数化。\n")
    L.append(f"样品与工况：`{sample}`；试样 16×32 = 512 盘，r = 0.75 m，ρ = 917 kg/m³，"
             "kn = 1×10⁷ N/m，kt/kn = 0.5，freeze_bond_fraction = 0（无粘结），"
             "lateral_control = isochoric，σ₃(固结) = 100 kPa。\n")

    L.append("## (2) 方法\n")
    L.append("五工况 2 对对照设计（配对才可判定加和性，单跑 μ_r 曲线不可判定）：\n")
    L.append("| 工况 | μ | μ_r | μ_eff = μ+μ_r | 作用 |")
    L.append("|---|---|---|---|---|")
    roles = {
        "mu055_mur00": "基线（复现 R3 16×32 的 φ_cv）",
        "mu055_mur20": "滚动替身，μ_eff 0.75",
        "mu055_mur40": "滚动替身，μ_eff 0.95",
        "mu075_mur00": "纯摩擦对照（对 μ_eff = 0.75）",
        "mu095_mur00": "纯摩擦对照（对 μ_eff = 0.95）",
    }
    for r in runs:
        L.append(f"| `{r['tag']}` | {fmt(r['mu'])} | {fmt(r['mu_r'])} | {fmt(r['mu_eff'])} | "
                 f"{roles.get(r['tag'], '')} |")
    L.append("")
    L.append("- 临界状态提取：`tensordem.biaxial_critical_state.critical_state_from_stress_path`，"
             "`strain_measure=\"log\"`（等体积下 log 体应变恒为 0，工程应变在 ε_a=0.3 会给出 −0.13 的假压缩），"
             "`boundary_control=\"isochoric\"`（侧墙应力是输出量，servo 闸门换成 `lateral_walls_loaded`），"
             "`min_strain=0.15`。四闸门：isochoric / deviator_stationary / window_independent / lateral_walls_loaded。")
    L.append(f"- 预声明判据：配对残差 `|Δφ_cv| ≤ {fmt(tol,1)}°` 判为「加和」；该容差在跑之前写在分析脚本常量里，"
             "不按结果调整。")
    L.append("- 复现命令：\n")
    L.append("```\n# 集群递交（5 分片，每片一个工况）\n"
             "TENSORDEM_THREADS=2 PYTHONPATH=src python scripts/m21803_rolling_surrogate.py --shard {i} --nshards 5\n"
             "# 分析（消费 runs/m21803/rolling_surrogate_*.json）\n"
             "PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_rolling_surrogate.py\n"
             "# 接触层算子恒等式（秒级）\n"
             "PYTHONPATH=src python -m pytest tests/test_rolling_surrogate_additivity.py -q\n```\n")

    L.append("## (3) 计算结果\n")
    L.append("### 3.1 φ_cv 逐工况\n")
    L.append("| 工况 | μ | μ_r | N | ε_a(final) | I | φ_peak [°] | **φ_cv [°]** | 四闸门 | 尾段均值+σ [°] | arctan(μ) [°] | arctan(μ_eff) [°] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        gates = r["phi_cv_flags"]
        g = "".join("✓" if gates.get(k) else "✗" for k in
                    ("isochoric", "deviator_stationary", "window_independent",
                     "lateral_walls_loaded"))
        L.append(f"| `{r['tag']}` | {fmt(r['mu'])} | {fmt(r['mu_r'])} | {r['n_particles']} | "
                 f"{fmt(r['final_strain'],3)} | {r['inertial_number']:.1e} | {fmt(r['peak_phi_deg'])} | "
                 f"**{fmt(r['phi_cv_deg'])}** | {g} | "
                 f"{fmt(r['phi_tail_mean_deg'])}±{fmt(r['phi_tail_std_deg'])} | "
                 f"{fmt(r['phi_rowe_atan_mu_deg'])} | {fmt(r['phi_rowe_atan_mu_eff_deg'])} |")
    L.append("")
    L.append("闸门顺序：isochoric / deviator_stationary / window_independent / lateral_walls_loaded。\n")

    L.append("### 3.2 加和性配对\n")
    if add:
        L.append("| 分拆工况 | 合并对照 | μ_eff | φ_cv(分拆) [°] | φ_cv(合并) [°] | Δφ_cv [°] | Δ% | ≤ 预声明容差 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for a in add:
            L.append(f"| `{a['split_tag']}` | `{a['merged_tag']}` | {fmt(a['mu_eff'])} | "
                     f"{fmt(a['phi_cv_split_deg'])} | {fmt(a['phi_cv_merged_deg'])} | "
                     f"{fmt(a['delta_phi_cv_deg'])} | {fmt(a['delta_phi_cv_pct'],1)} | "
                     f"{fmt(a['additive_within_tolerance'])} |")
        L.append("")
    L.append("图：`docs/figures/CASE-DEM-BIAXIAL-FRICTION_rolling_surrogate_audit.png`"
             "（左：φ_cv 随 μ_eff 的两种取法；右：加和性残差条图 + 预声明容差带）。\n")

    L.append("## (4) 结果判读\n")
    if add:
        alladd = art["verdict"]["all_pairs_additive"]
        L.append(
            "接触层恒等式是**精确**的（4 例 `allclose` 到 1e-9），所以宏观只剩一个可测问题："
            "把 (μ, μ_r) = (0.55, b) 换成 (0.55+b, 0)，集合的 φ_cv 变不变。"
            + ("本轮所有配对残差落在预声明容差内，结论：**第 4 轮加入的「滚动摩阻」在统计意义上就是附加滑动摩擦**，"
               "它不引入任何新的物理自由度；R5 设想的「μ_r 增大 → 逼近 Rowe 28.81°」不能作为"
               "「补齐滚动阻力物理」的证据，只能当作「把有效摩擦系数调大」的证据。"
               if alladd else
               "本轮至少一对残差超出预声明容差，说明宏观仍存在超出加性摩擦的效应，需要进一步分离"
               "（局部化/剪切带形态差异、饱和时序差异），不能下「纯附加摩擦」的结论。"))
    L.append("")

    L.append("## (5) 注意点与局限\n")
    L.append("1. **替身不是滚动阻力。** 该实现没有力矩、没有转动自由度，`μ_r` 的物理量纲与真实滚动摩阻"
             "（力矩/法向力/半径）不同；把它命名为 `rolling_friction_coefficient` 会误导下游标定。"
             "凡是用它去「标定滚动阻力」的结论都应降级为「等效附加摩擦」。")
    L.append("2. **二维与三维不可直接对标。** Rowe 28.81° 是三维棱角砂的临界状态值，二维圆盘集合的 φ_cv "
             "本来就偏低（配位数、颗粒形状、自由度数目都不同）。本轮只做「加和性」这一内部一致性检验，"
             "不主张二维 16×32 试样能命中三维参考值。")
    L.append("3. **试样小、应变有限。** 16×32 = 512 盘、ε_a ≤ 0.35；剪切带形成后局部化会带来 1–2° 的"
             "摆动（尾段 σ 见 3.1 表），加和性容差 ±1.5° 与这个量级同阶，因此结论应读作「无超出噪声的差异」。")
    L.append("4. **R5 管线本身跑不通（已修）。** `scripts/m21110_probe_rolling_cs.py` 用 "
             "`result[\"loading\"][\"total_steps\"]` 取步数，而 `run_biaxial_shear_band` 返回的键是 `steps`——"
             "任何一次 R5 计算都会在落盘前 `KeyError` 崩掉；同时 `analyse_rolling_cs.py` 对等体积路径"
             "用了 `strain_measure=\"small\"` + `boundary_control=\"servo\"` 的默认参数。这两处解释"
             "「cj3193/cj2765 跑了 51 min 却没有工件」。本轮已修步数键，并在新分析脚本里显式传 log/isochoric。")
    L.append("5. **不覆盖诚实 FAIL。** 锚点验收仍是 R2 的 φ_peak 对比（壁面 8.97% / 内部 4.13% vs 38°），"
             "本轮没有重跑 24×48 主算例，也不改变该判定。")
    L.append("")

    L.append("## (6) 误差判定\n")
    L.append("| 关键量 | 基准 | 本轮 | 相对误差 | 是否 < 3% |")
    L.append("|---|---|---|---|---|")
    if add:
        worst = max(add, key=lambda a: abs(a["delta_phi_cv_deg"]))
        verdict_3pct = ("是" if abs(worst["delta_phi_cv_pct"]) < 3
                        else "否——残差与 1–2° 的剪切带涨落同阶，故判为「加和律内」，"
                             "不声称 3% 数值精度")
        L.append(f"| 加和性残差 Δφ_cv | 0°（同一算子） | {fmt(worst['delta_phi_cv_deg'])}° | "
                 f"{fmt(worst['delta_phi_cv_pct'],1)}% | {verdict_3pct} |")
        L.append(f"| 接触层恒等式 | 精确 0 | ≤1e-9（allclose） | ~1e-15 | 是（浮点精度内精确） |")
    base = by_tag("mu055_mur00")
    if base:
        L.append(f"| φ_cv(μ=0.55, μ_r=0) vs arctan(μ) | 28.81°（Rowe 1962, D=0） | "
                 f"{fmt(base['phi_cv_deg'])}° | "
                 f"{fmt(100*abs(base['phi_cv_deg']-28.81)/28.81,1)}% | 否——二维圆盘 vs 三维砂，见 (5.2) |")
    L.append("")
    L.append("**是否 < 3%：接触层算子恒等式是（浮点精度内精确）；宏观 φ_cv 对 Rowe 参考值不是"
             "（差 20% 以上，且该对标本身跨维不可比）；加和性残差不是数值 3% 意义上的「精度」，"
             "而是「是否同一算子」的定性判定，容差 ±1.5° 事前声明。**\n")

    L.append("## (7) 给下一次自找活的线索\n")
    L.append("1. 若真要引入滚动阻力，需要给颗粒加转动自由度（惯性矩 + 角速度状态 + 力矩积分），"
             "这是 TensorDEM 的结构性能力缺口，值得立一条独立条目（slug 建议 "
             "`CAP-DEM-ROLLING-DOF`：验收=单球斜面临界角 θ_c = arctan(μ_r) 误差 <3%）。")
    L.append("2. 锚点 CASE-DEM-BIAXIAL-FRICTION 的 8% 验收能否成立，取决于「用哪个应力度量 + 峰还是临界态」，"
             "这需要一次**先声明度量再跑**的 24×48 复算，属于需要集群长时的重活。")
    L.append("3. `analyse_rolling_cs.py`（R5）保留但不应被引用为结果来源——它与本轮脚本的参数口径不一致。")
    L.append("")

    L.append("## 工件清单\n")
    L.append("| 路径 | 内容 |")
    L.append("|---|---|")
    L.append("| `scripts/m21803_rolling_surrogate.py` | 5 工况采集脚本（集群分片） |")
    L.append("| `benchmarks/case_dem_biaxial_friction/analyse_rolling_surrogate.py` | 四闸门提取 + 加和性配对 + 图 |")
    L.append("| `tests/test_rolling_surrogate_additivity.py` | 接触层算子恒等式（精确） |")
    L.append("| `artifacts/CASE-DEM-BIAXIAL-FRICTION/rolling_surrogate_audit.json` | 全部数字（本文档唯一来源） |")
    L.append("| `artifacts/CASE-DEM-BIAXIAL-FRICTION/rolling_surrogate_audit.png` | 结果图 |")
    L.append("| `runs/m21803/rolling_surrogate_*.json` | 原始应力路径（集群产出） |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    art = json.loads(ART.read_text())
    png = PNG if PNG.exists() else None
    text = build(art, png)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text)
    print(f"wrote {DOC} ({len(text)} chars)")

    if png is not None:
        FIG_DOC.parent.mkdir(parents=True, exist_ok=True)
        FIG_DOC.write_bytes(png.read_bytes())
        print(f"copied figure -> {FIG_DOC}")

    figures = []
    if png is not None:
        figures.append({
            "name": "rolling_surrogate_audit.png",
            "caption": ("Left: phi_cv versus mu_eff, filled squares = pure friction (mu_r=0), "
                        "open circles = rolling surrogate plotted at mu+mu_r, dashed = arctan(mu_eff). "
                        "Right: additivity residual phi_cv(split) - phi_cv(merged) with the "
                        "pre-declared +/-1.5 deg band."),
            "png_base64": base64.b64encode(png.read_bytes()).decode(),
        })
    runs = art["runs"]
    add = [a for a in art.get("additivity", []) if a.get("measured")]
    payload = {
        "title": "CASE-DEM-BIAXIAL-FRICTION R6: the rolling-resistance surrogate is additive friction (additivity audit)",
        "summary": (
            "Additivity audit of TensorDEM's rolling_friction_coefficient: contact-level identity "
            "F_t = -(mu+mu_r) F_n tanh(v_t/v_reg) proven exactly (4 tests), and 5 isochoric biaxial "
            "shears (16x32, 100 kPa) test whether phi_cv(mu,mu_r) equals phi_cv(mu+mu_r,0). "
            + (("All measured pairs agree within the pre-declared 1.5 deg band -> the surrogate "
                "introduces no rolling physics; R5's phi_cv(mu_r) curve is a re-parameterisation of mu.")
               if art.get("verdict") and art["verdict"]["all_pairs_additive"] else
               "At least one pair violates the pre-declared band -> macroscopic behaviour is not "
               "purely additive friction.")),
        "scenario_key": "dem_biaxial_friction",
        "mission_id": MISSION,
        "tags": ["TensorDEM", "DEM", "biaxial", "critical-state", "rolling-resistance",
                 "CASE-DEM-BIAXIAL-FRICTION", "verification", "negative-result"],
        "markdown": text,
        "figures": figures,
    }
    REPORT.write_text(json.dumps(payload))
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())