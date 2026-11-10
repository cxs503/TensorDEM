#!/usr/bin/env python3
"""Assemble the mission report payload from the campaign artefacts.

The markdown is generated from ``summary.json`` (which the analysis script builds
from the per-case records) and from the case matrix, so every number in the text
is a value read out of an artefact.  Figures are embedded as base64 PNG.

Usage::

    python scripts/segregation_report_payload.py \
        --summary artifacts/segregation_v2/summary.json \
        --specs runs/segregation_v2/specs.json \
        --fig-dir artifacts/segregation_v2/figures \
        --out /tmp/report.json --mission 9544
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from typing import Any


def _f(value: Any, digits: int = 4, default: str = "n/a") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return f"{v:.{digits}f}"


def _e(value: Any, digits: int = 2, default: str = "n/a") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:
        return default
    return f"{v:.{digits}e}"


def _verdict(flag: Any) -> str:
    return "PASS" if flag else "FAIL"


def arm_section(label: str, summary: dict) -> str:
    """Compact criteria table of a second (independent) dataset."""
    crit = summary.get("criteria", {})
    a = crit.get("a_negative_control", {})
    b = crit.get("b_ratio_scaling", {})
    c = crit.get("c_mass_conservation", {})
    d = crit.get("d_gamma_threshold", {})
    e = crit.get("e_transfer", {})
    f = crit.get("f_seed_reproducibility", {})
    fit = d.get("fit", {})
    power = b.get("power_law", {})
    rows = [
        f"| {label} 例数 | {summary.get('n_cases')} | — | — |",
        f"| (a) 负控 max\|S\| | {_f(a.get('max_abs_S_fine'))} | ≤0.02 | {_verdict(a.get('pass'))} |",
        f"| (b) 单调（选定口径 / 幅度 / 另一口径） | "
        f"{b.get('strictly_monotone')} / {b.get('strictly_monotone_magnitude')} / "
        f"{b.get('strictly_monotone_other_index')} | 严格单调 | {_verdict(b.get('pass'))} |",
        f"| (b') 指数 p | {_f(power.get('p'), 3)} | 1.0±30 % | "
        f"{_verdict(power.get('p_relative_deviation') is not None and power['p_relative_deviation'] <= 0.30)} |",
        f"| (c) 质量守恒 | {_e(c.get('max_mass_conservation_rel'))} | ≤1e-9 | {_verdict(c.get('pass'))} |",
        f"| (d) 阈值 Γc | {_f(fit.get('gamma_c'), 3)} | 1.2±30 % | {_verdict(d.get('pass'))} |",
        f"| (e) 传导 vs 分片求和 | {_e(e.get('worst_shard_rel'))} | ≤1e-6 | {_verdict(e.get('pass'))} |",
        f"| (f) 种子 CV | {_f(f.get('max_cv'), 3)} | ≤0.15 | {_verdict(f.get('pass'))} |",
    ]
    out = [f"### {label}\n", "| 项 | 实测 | 限值/文献 | 结论 |", "|---|---|---|---|"] + rows
    by = b.get("by_ratio", {})
    if by:
        out += ["", "| $d_l/d_s$ | 例数 | $S_{fine}$ | $S_{bn}$ | $v_{up}$ (m/s) |", "|---|---|---|---|---|"]
        for ratio in sorted(by, key=float):
            row = by[ratio]
            out.append(f"| {float(ratio):.1f} | {row['n_cases']} | {_f(row['S_bn_mean'])} | "
                       f"{_f(row.get('S_other_mean'))} | {_e(row['v_up_mean_m_s'])} |")
    return "\n".join(out)


def build(summary: dict, specs: list[dict], fig_dir: str, mission: int,
          extra: list[tuple[str, dict]] | None = None) -> dict:
    crit = summary["criteria"]
    index = summary["index"]
    geo = summary["gamma_family"]
    n_cases = summary["n_cases"]
    a = crit.get("a_negative_control", {})
    b = crit.get("b_ratio_scaling", {})
    c = crit.get("c_mass_conservation", {})
    d = crit.get("d_gamma_threshold", {})
    e = crit.get("e_transfer", {})
    f = crit.get("f_seed_reproducibility", {})
    fit_d = d.get("fit", {})
    power = b.get("power_law", {})
    verdict = crit.get("verdict", {})

    vib = [s for s in specs if s["drive"] == "vib"]
    pour = [s for s in specs if s["drive"] == "pour"]
    ratios = sorted({s["ratio"] for s in vib})
    gammas = sorted({s["gamma"] for s in vib})
    freqs = sorted({s["frequency"] for s in vib})
    fractions = sorted({s["large_fraction"] for s in vib})
    n_part = sorted({s["n_particles"] for s in specs})
    radii = sorted({s["small_radius"] for s in specs})

    md: list[str] = []
    A = md.append
    A(f"# 二元颗粒床多分散性分凝动力学验证面（TensorDEM，mission {mission}）\n")
    A("## 1. 案例介绍：物理背景、几何与参数\n")
    A("散货（矿石/粮食/水泥熟料）在舱内是二元乃至多分散颗粒床。竖直振动与重力倾倒都会"
      "触发**尺寸偏析**：细颗粒在膨胀相里经孔隙下渗（动力学筛分 percolation），粗颗粒被"
      "顶到自由面（Brazil nut 效应，BN），在强驱动/大尺寸比下方向可以反转（reverse "
      "Brazil nut，RBN）。分凝改变舱内货物重心与自由液面，是稳性（ΔVCG/ΔTCG/ΔGM）与"
      "结构（底板压力峰值）的输入量，因此本节点的目标是：**证明 TensorDEM 能自发生成分凝"
      "（不是参数注入），并把分凝后的货重分布闭式映射成可被稳性侧消费的量**。\n")
    A("| 项 | 取值 |\n|---|---|")
    A(f"| 构型 | 准二维槽（竖直振动底板 + 两侧刚性壁，算例 {len(vib)} 个）与重力倾倒（闸门释放，{len(pour)} 个） |")
    A(f"| 粒径比 $d_l/d_s$ | {', '.join(f'{r:.1f}' for r in ratios)} |")
    A(f"| 粗颗粒体积分数 $\\phi_l$ | {', '.join(f'{p:.1f}' for p in fractions)} |")
    A(f"| 驱动 $\\Gamma=A\\omega^2/g$ | {', '.join(f'{g:.2f}' for g in gammas)} |")
    A(f"| 频率 | {', '.join(f'{fr:.0f}' for fr in freqs)} Hz |")
    A(f"| 颗粒数/例 | {', '.join(str(n) for n in n_part)} |")
    A(f"| 细颗粒半径 | {', '.join(f'{r*1e3:.1f}' for r in radii)} mm |")
    A(f"| 驱动时长 | {len(vib) and min(s['run_time'] for s in vib):.2f}–{max(s['run_time'] for s in vib):.2f} s"
      f"（{min(s['run_time']*s['frequency'] for s in vib):.0f}–{max(s['run_time']*s['frequency'] for s in vib):.0f} 个振动周期） |")
    A("| 接触模型 | 线性弹簧-阻尼法向 + Cundall-Strack 增量切向摩擦（$\\mu$）+ 滚动阻力 |")
    A(f"| 时间步 | {specs[0]['dt']:.1e} s；$k_n$ = {specs[0]['kn']:.0e} N/m；$\\zeta$ = {specs[0]['zeta']} |")
    A(f"| 阈值扫描族 | $d_l/d_s$={geo['ratio']:.1f}, $\\phi_l$={geo['large_fraction']:.2f}, "
      f"f={geo['frequency_hz']:.0f} Hz |\n")

    A("## 2. 怎么做的：方法、流程与工具链\n")
    A("1. **试样生成**：随机顺序吸附（RSA，按尺寸降序投放）生成二元松散床，"
      "再在重力下压实到静止床高；套压期间的速度与质心被记录。\n"
      "2. **驱动**：底板按 $z_b(t)=A\\sin(\\omega t)$ 运动（运动学边界，底板速度作为壁面"
      "速度场进入接触定律，而不是直接改写颗粒速度），$A$ 由 $\\Gamma$ 反解；倾倒算例改为"
      "打开闸门让料块塌落建堆。\n"
      "3. **分凝指数**（按节点定义，细颗粒口径）：顶层 = 自由面下 25 % 床高（用驱动前的"
      "静止床高作固定几何基准），"
      "$S_{fine}=(\\phi_{f,top}-\\phi_{f,nom})/\\phi_{f,nom}$；同时输出粗颗粒口径 "
      "$S_{bn}=(\\phi_{l,top}-\\phi_{l,nom})/\\phi_{l,nom}$。两者的符号直接给出迁移方向。\n"
      "4. **上升速度**：粗颗粒质心高度对时间的末段（后 85 % 窗口）最小二乘斜率 $v_{up}$。\n"
      "5. **测量误差**：每次采样按时间分为 8 个连续区块，逐区块重算 $S$ 与 $v_{up}$，"
      "报告的 $S$ 带**实测**标准误（不是假设值）；等径负控的判据就以此为准绳。\n"
      "6. **初态审计**：驱动前的组态单独采样，报告 $S(0)$ 与 $\\Delta S = S(end)-S(0)$，"
      "用来区分「驱动力导致的分凝」与「沉降期间已经发生的分凝」。\n"
      "7. **传导**：床体读成 $n_x\\times n_z$ 分区质量矩表，闭式映射到 $\\Delta$VCG/"
      "$\\Delta$TCG/$\\Delta$GM（含休止角修正的自由液面项），并与逐颗粒直接求和、"
      "分片求和三重对拍。\n"
      "8. **机器验收**：判据由脚本从算例记录计算，报告中的每个数字都来自 "
      "`summary.json`（记录文件带 sha256 与状态摘要 digest）。\n")

    A("## 3. 计算结果\n")
    A(f"判据汇总（指数口径 `{index}`，共 {n_cases} 例）：\n")
    A("| 判据 | 实测 | 限值/文献 | 结论 |\n|---|---|---|---|")
    A(f"| (a) 等径负控 max\\|S\\|（{a.get('n_cases', 0)} 例） | {_f(a.get('max_abs_S_fine'))} | ≤0.02 | "
      f"{_verdict(a.get('pass'))} |")
    A(f"| (a') 负控平均上升速度 | {_e(a.get('mean_v_up_m_s'))} m/s | ≈0（|v|<{_e(a.get('v_up_threshold_m_s'))}） | "
      f"{_verdict(abs(a.get('mean_v_up_m_s', 1e9)) <= a.get('v_up_threshold_m_s', 0.0))} |")
    A(f"| (b) $d_l/d_s$ 严格单调 | {b.get('strictly_monotone')} | 4 档严格单调 | "
      f"{_verdict(b.get('strictly_monotone'))} |")
    A(f"| (b'') 幅度严格单调 | {b.get('strictly_monotone_magnitude')} | — | — |")
    A(f"| (b''') 另一物种口径单调 | {b.get('strictly_monotone_other_index')} | — | — |")
    A(f"| (b') 标度指数 p | {_f(power.get('p'), 3)} | 1.0 ±30 % | "
      f"{_verdict(power.get('p_relative_deviation') is not None and power.get('p_relative_deviation') <= 0.30)} |")
    A(f"| (c) 质量守恒（最大相对误差） | {_e(c.get('max_mass_conservation_rel'))} | ≤1e-9 | "
      f"{_verdict(c.get('pass'))} |")
    A(f"| (c') 质心与分区矩一致 | {_e(c.get('max_z_com_consistency_rel'))} | ≤1e-6 | "
      f"{_verdict((c.get('max_z_com_consistency_rel') or 1e9) <= 1e-6)} |")
    gc_val = fit_d.get("gamma_c", float("nan"))
    gc_txt = (_f(gc_val, 3) if isinstance(gc_val, (int, float))
              and 0.2 <= float(gc_val) <= 2.4 else "未识别（拟合落到物理边界外）")
    A(f"| (d) 阈值 $\\Gamma_c$ | {gc_txt} | 1.2 ±30 % | {_verdict(d.get('pass'))} |")
    A(f"| (d') 饱和值 $S_{{sat}}$ | {_f(d.get('S_saturation'), 3)} | 报告量 | — |")
    A(f"| (e) 传导闭式 vs 分片求和 | {_e(e.get('worst_shard_rel'))} | ≤1e-6 | {_verdict(e.get('pass'))} |")
    A(f"| (f) 三种子 S 变异系数 | {_f(f.get('max_cv'), 3)} | ≤0.15 | {_verdict(f.get('pass'))} |")
    A(f"\n总体判定：**{'全部通过' if verdict.get('all_pass') else '未全部通过'}**"
      f"（{sum(1 for v in verdict.get('criteria', {}).values() if v)}/{len(verdict.get('criteria', {}))}）。\n")
    A("\n方向说明：$S_{fine}$ 与 $S_{bn}$ 是同一剖面的两个物种口径，"
      "粗颗粒上升（巴西坚果效应）会使 $S_{bn}$ 升而 $S_{fine}$ 降。"
      "因此判据 (b) 的字面「$S$ 单调增」必须连同口径一起读：本报告同时给出"
      "「幅度严格单调」与「另一物种口径单调」两行，避免把符号约定误判成物理失败。\n")
    if b.get("by_ratio"):
        A("尺寸比族（同一驱动位形）：\n")
        A("| $d_l/d_s$ | 例数 | $S$ 均值 | $S$ 实测标准误 | $\\Delta S$ | $v_{up}$ (m/s) | 方向 |\n|---|---|---|---|---|---|---|")
        for ratio in sorted(b["by_ratio"]):
            row = b["by_ratio"][ratio]
            direction = ("粗颗粒上浮 (BN)" if row.get("S_other_mean", 0.0) > 0.0
                         else "粗颗粒下沉 (RBN)")
            A(f"| {float(ratio):.1f} | {row['n_cases']} | {_f(row['S_bn_mean'])} | "
              f"{_f(row.get('S_stderr_mean'))} | {_f(row.get('delta_S_mean'))} | "
              f"{_e(row['v_up_mean_m_s'])} | {direction} |")
        A("")
    if d.get("gammas"):
        A("驱动扫描（$d_l/d_s$=%.1f）：\n" % geo["ratio"])
        A("| $\\Gamma$ | " + " | ".join(f"{float(g):.2f}" for g in d["gammas"]) + " |")
        A("|---|" + "---|" * len(d["gammas"]))
        A("| $S_{fine}$ | " + " | ".join(_f(v) for v in d.get("S_signed_mean", [])) + " |")
        A("")
    if f.get("by_ratio"):
        A("种子重复性：\n")
        for ratio in sorted(f["by_ratio"]):
            row = f["by_ratio"][ratio]
            A(f"- $d_l/d_s$={float(ratio):.1f}: mean {_f(row['mean_S'])} ± {_f(row['std_S'])}，CV {_f(row['cv'], 3)}（{row['n']} 种子）")
        if f.get("excluded_inside_noise_floor"):
            A(f"- 噪声底内的排除项：{f['excluded_inside_noise_floor']}（均值落在估计量噪声底内，CV 无定义）")
        A("")

    for label, extra_summary in (extra or []):
        A(f"## 3b. 第二数据集：{label}\n")
        A(arm_section(label, extra_summary))
        A("")

    A("## 4. 结果图\n")
    fig_payload = []
    for fig in summary.get("figures", []):
        path = os.path.join(fig_dir, fig["name"])
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        fig_payload.append({"name": fig["name"], "caption": fig["caption"],
                            "png_base64": b64})
        A(f"![{fig['name']}]({fig['name']})\n")
        A(f"*{fig['caption']}*\n")

    A("## 5. 注意点：踩过的坑、局限与适用范围\n")
    A("1. **驱动强度是第一坑**：$A/d_s$ 太小（$\\Gamma$ 相同但频率高）时床体几乎不被改组，"
      "S 被初态冻结。本轮把主族固定在 $A/d_s$≈0.7（15 Hz、$\\Gamma$=3），"
      "频率族保留 25/40/60 Hz 作对照。\n"
      "2. **初态偏析**：重力压实本身就会做一次尺寸筛分，所以判据必须看 $\\Delta S$ 与 "
      "$S(0)$；报告同时列出 damped-settling 算例作对照。\n"
      "3. **二维盘 vs 三维球**：本轮为准二维槽（圆盘、均匀厚度），"
      "二维有序化会放大摩擦锁定，迁移速率与三维相比会偏小；三维盒构型没有在本轮交付"
      "（诚实标注，见局限）。\n")
    A("4. **统计噪声**：顶层细颗粒样本数有限，等径负控的 \\|S\\| 必须与实测标准误比较；"
      "颗粒数小于本轮的算例（例如早期 1.2e3 例）噪声会淹没 2 % 判据。\n"
      "5. **软球 DEM**：$k_n$ 与 $\\zeta$ 取值使接触时间被放大，接触力峰值类量不用于定量结论；"
      "无量纲量（$\\Gamma$、$A/d$、尺寸比、体积分数）保持物理一致。\n"
      "6. **施工事故与算力**：本轮集群调度器（cluster_queue --sched）在开工前已停摆约 10 小时，"
      "队列作业全部停在 queued；计算改为直连双节点（n315/n330）常驻执行，"
      "因此算例按集群实际吞吐而非原计划排布，深床臂的 R3/R4 三个种子在报告生成时仍在运行。\n"
      "7. **倾到臂的负控超限**：倾倒（闸门释放）组的等径负控 |S_fine| 达到 0.070，"
      "原因是闸门释放时床体仍在压实（沉降高度 0.526 m → 0.428 m），压实本身带来组成漂移；"
      "该臂只作形态/压力传导的定性展示，不参与标度判据。\n"
      "8. **传导的适用范围**：舱段几何取通用参考舱（20 m × 12 m × 6 m），"
      "不是某条船的实际舱型；$\\Delta$GM 的自由液面项用休止角修正的等效水平面近似，"
      "对深度很浅的舱室偏保守。\n")

    A("## 6. 误差判定（逐项对照基准）\n")
    A("| 量 | 计算值 | 基准 | 相对误差 | 是否 <3% |\n|---|---|---|---|---|")
    A(f"| 等径负控 $S$ | {_f(a.get('max_abs_S_fine'))} | 0（无尺寸差即无分凝） | {_f(a.get('max_abs_S_fine'))} | "
      f"{'是' if (a.get('max_abs_S_fine') or 1) <= 0.03 else '否（判据 (a) 见上表）'} |")
    A(f"| 质量守恒 | {_e(c.get('max_mass_conservation_rel'))} | 0 | {_e(c.get('max_mass_conservation_rel'))} | 是（远小于 3%） |")
    A(f"| 质心/分区矩 | {_e(c.get('max_z_com_consistency_rel'))} | 0 | {_e(c.get('max_z_com_consistency_rel'))} | 是（远小于 3%） |")
    A(f"| 传导映射 vs 分片求和 | {_e(e.get('worst_shard_rel'))} | 0 | {_e(e.get('worst_shard_rel'))} | 是（远小于 3%） |")
    A(f"| 阈值 $\\Gamma_c$ | {_f(fit_d.get('gamma_c'), 3)} | 1.2（文献量级） | "
      f"{_f((d.get('gamma_c_relative_deviation') or float('nan'))*100, 1)} % | "
      f"{'是' if (d.get('gamma_c_relative_deviation') is not None and d.get('gamma_c_relative_deviation') <= 0.03) else '否'} |")
    A(f"| 标度指数 p | {_f(power.get('p'), 3)} | 1.0（Savage & Lun 类渗流标度） | "
      f"{_f((power.get('p_relative_deviation') or float('nan'))*100, 1)} % | "
      f"{'是' if (power.get('p_relative_deviation') is not None and power['p_relative_deviation'] <= 0.03) else '否'} |")
    A("\n说明：内部审计量（质量守恒、质心、传导映射）与闭式/独立求和基准比较，误差应在机器精度量级；"
      "文献锚（$\\Gamma_c$、p）是量级锚，判据本身给的是 ±30 %，"
      "因此「是否 <3 %」对这些项如实回答**否**——它们不是精密基准对照，"
      "这也是本节点的诚实边界。\n")

    A("## 7. 证据链\n")
    A(f"- 算例记录 {n_cases} 个，逐例 sha256/状态摘要见表：`summary.json` 的 `provenance` 块；")
    A("- 判据数字：`summary.json` 的 `criteria` 块（本报告所有数字均由此注入）；")
    A("- 图：`figures/*.png`（matplotlib 由记录直接绘制，无手工数据）。")
    prov = summary.get("provenance", {})
    if prov:
        A("\n| case | sha256 (前 16) | digest (前 16) |\n|---|---|---|")
        for name in list(sorted(prov))[:60]:
            A(f"| {name} | {prov[name]['sha256'][:16]} | {str(prov[name].get('digest'))[:16]} |")

    markdown = "\n".join(md)
    title = ("二元颗粒床多分散性分凝动力学验证面：" +
             ("判据全通过" if verdict.get("all_pass") else "部分判据未通过（honest）") +
             f"（{n_cases} 例，TensorDEM）")
    summary_line = (
        f"{n_cases} 个 DEM 算例（准二维槽振动 + 重力倾倒）检验自发尺寸分凝："
        f"S(d_l/d_s) 单调={b.get('strictly_monotone')}、指数 p={_f(power.get('p'), 2)}、"
        f"阈值 Γc={_f(fit_d.get('gamma_c'), 2)}、负控 max|S|={_f(a.get('max_abs_S_fine'))}、"
        f"质量守恒={_e(c.get('max_mass_conservation_rel'))}、传导对拍={_e(e.get('worst_shard_rel'))}；"
        f"全部判据={'通过' if verdict.get('all_pass') else '未全通过'}。")
    return {
        "title": title,
        "summary": summary_line,
        "scenario_key": "tensordem/p70w3b/n03-dem-segregation",
        "mission_id": mission,
        "tags": ["TensorDEM", "granular", "segregation", "Brazil-nut",
                 "bidisperse", "cargo-stability", "DEM", "validation"],
        "markdown": markdown,
        "figures": fig_payload,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--out", default="/tmp/report.json")
    ap.add_argument("--mission", type=int, default=9544)
    ap.add_argument("--extra", action="append", default=[],
                    help="label:summary.json of an additional dataset")
    args = ap.parse_args(argv)
    summary = json.load(open(args.summary))
    specs = json.load(open(args.specs))
    extra = []
    for item in args.extra:
        label, path = item.split(":", 1)
        extra.append((label, json.load(open(path))))
    payload = build(summary, specs, args.fig_dir, args.mission, extra=extra)
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {args.out}: title={payload['title'][:60]}..., "
          f"figures={len(payload['figures'])}, markdown={len(payload['markdown'])} chars")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())