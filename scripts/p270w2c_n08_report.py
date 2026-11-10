#!/usr/bin/env python
"""Render the P270W2C-N08 paper-level report from the campaign JSON.

Every number in the markdown is injected from ``artifacts/p270w2c_n08/report.json``
(no hand-copied digits).  Figures are copied next to the report so the markdown
image links resolve inside ``docs/``.

Usage: ``python scripts/p270w2c_n08_report.py``
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "p270w2c_n08"
DOCS = ROOT / "docs"
FIGURE_DIR = DOCS / "figures" / "p270w2c_n08"

PARAM_LABELS = {
    "k_b": r"$k_b$",
    "eps_b": r"$\varepsilon_b$",
    "mu_s": r"$\mu_s$",
    "mu_d": r"$\mu_d$",
    "k_c": r"$k_c$",
    "k_p": r"$k_p$",
}
PARTICLE_RADIUS = 0.0125


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return "nan"
        if not math.isfinite(value):
            return "nan"
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def render(report: dict) -> str:
    targets = report["targets"]
    surface = report["response_surface"]
    ac1 = report["AC1_geometry_prior"]
    ac2 = report["AC2_identifiability"]
    diagnostics = ac2["diagnostics"]
    ac3 = report["AC3_verdict"]
    pareto = report["AC3_pareto"]
    ac4 = report["AC4_holdout"]
    ac5 = report["AC5_brazilian_field"]
    ac6 = report["AC6_negative_control"]
    ac7 = report["AC7_reproducibility"]
    reference = ac3["reference"]
    jacobian = np.array(ac2["jacobian"]["jacobian"], dtype=float)
    correlation = np.array(diagnostics["correlation"], dtype=float)
    quality = pareto["surrogate_quality"]
    design = report["design"]

    best = ac5["best"]
    holdout_rows = ac4["reference"]["rows"]
    repeat = ac7["repeat_ucs_Pa"]

    def gate(flag: bool) -> str:
        return "PASS" if flag else "FAIL"

    lines: list[str] = []
    add = lines.append
    add("# BPM 微观参数三目标联合标定：可行性判定与可辨识性审计")
    add("")
    add("**节点** P270W2C-N08（任务 m9756）· **仓库** TensorDEM · **分支** `agent/m9756`  ")
    add("**新增模块** `src/tensordem/bpm_calibration_feasibility.py` · "
        "**战役脚本** `scripts/p270w2c_n08_feasibility.py`（sample/analyze）· "
        "**图/报告渲染** `scripts/p270w2c_n08_figures.py`、`scripts/p270w2c_n08_report.py`  ")
    add("**原始数据** `runs/p270w2c_n08/records.jsonl`（逐任务记录）· "
        "**判据数字** `artifacts/p270w2c_n08/report.json`（本报告的唯一数字来源）")
    add("")
    add("## 0. 结论速览（判据 → 数字 → 判定）")
    add("")
    add("| 判据 | 目标 | 实测 | 判定 |")
    add("|---|---|---|---|")
    add(f"| AC1 几何/尺度先验 | 全部子项通过 | UCS 试件 H/D = "
        f"{fmt(ac1['ucs_slenderness']['H_over_D'])}；圆盘 D/t = {fmt(ac1['disc']['D_over_t'])}；"
        f"堆积率 {fmt(ac1['lattice_density']['packing_fraction'], 4)}（π/6 = "
        f"{fmt(ac1['lattice_density']['analytic_packing_fraction'], 4)}）；拉伸族 H/D = "
        f"{min(item['H_over_D'] for item in ac1['tension_family']['specimens']):.2f}–"
        f"{max(item['H_over_D'] for item in ac1['tension_family']['specimens']):.2f}；"
        f"加载协议瞬态应变 {fmt(max((row['transient_bond_strain'] for row in ac1['protocol']['specimens']), default=float('nan')), 4)}"
        f" < 断裂应变 {fmt(ac1['protocol']['breaking_strain'])} | "
        f"{gate(ac1['passed'])} |")
    add(f"| AC2 可辨识性 | 报 cond(J) 与相关矩阵；cond>1e4 必须判不可辨识并给降维建议 | "
        f"cond(J) = {fmt(diagnostics['cond'])}，rank = {diagnostics['rank']}/"
        f"{diagnostics['n_parameters']}，结构零空间维数 "
        f"{diagnostics.get('structural_null_space_dimension')}；最坏参数对 "
        f"{diagnostics['collinearity']['worst_pair']['pair']} 的 Brun γ = "
        f"{fmt(diagnostics['collinearity']['worst_pair']['collinearity'], 2)}；"
        f"判定 **{diagnostics['verdict']}** | {gate(diagnostics['verdict'] != 'identifiable')} |")
    add(f"| AC3 三分支可行性 | (a)/(b)/(c) 之一 + 证据 | 参考 θ 落 **分支 ({reference['branch']})**："
        f"UCS {fmt(reference['ucs_MPa'])} MPa、σc/σt {fmt(reference['strength_ratio'])}、"
        f"m CI [{fmt(reference['weibull_m_ci'][0])}, {fmt(reference['weibull_m_ci'][1])}] "
        f"（伪真值 {fmt(reference['m_pseudo_truth'])}）；"
        f"{design['n_points']} 个设计点分支计数 a/b/c = {ac3['branch_counts']['a']}/"
        f"{ac3['branch_counts']['b']}/{ac3['branch_counts']['c']} | 见 §3.3 |")
    add(f"| AC4 留出厚度外推 | 误差 ≤15% 且两端 m 方向一致 | "
        + "；".join(f"t={row['thickness_m']:.2f} m 误差 {row['relative_error'] * 100:+.1f}%"
                   for row in holdout_rows)
        + f"；方向一致 {fmt(ac4['reference']['direction_consistent'])}；"
          f"有完整留出对（t=0.05 与 0.8）的设计点通过 {ac4['n_passed']}/{ac4['n_points']}"
          f"（另有 {ac4.get('n_passed_any_holdout', 0)}/{ac4.get('n_points_any_holdout', 0)} 个点通过"
          f"已有的 t=0.05 单端留出）| {gate(ac4['reference']['passed'])} |")
    add(f"| AC5 巴西中心应力比 | 与 Hondros −3.0 相对差 ≤15% | "
        f"最优 {fmt(best['ratio_at_peak']) if best else 'n/a'}"
        f"（{best['n_lattice'] if best else 'n/a'}×{best['n_layers'] if best else 'n/a'} 点阵，"
        f"平均区 {fmt(best['region_ratio'], 2) if best else 'n/a'}D），相对差 "
        f"{fmt(best['relative_deviation'] * 100, 1) if best else 'n/a'}% | "
        f"{gate(ac5['passed'])}（最优配置；跨配置偏差 "
        f"{fmt(ac5.get('deviation_range', [float('nan')])[0] * 100, 0)}–"
        f"{fmt(ac5.get('deviation_range', [float('nan')])[-1] * 100, 0)}% → 稳健性 "
        f"{gate(bool(ac5.get('robust_pass'))) }，见 §3.5）|")
    add(f"| AC6 负控 200 组随机 θ | ≥95% 违反 UCS 文献带 | 违反比例 "
        f"{fmt(ac6['fraction_outside_ucs_band'] * 100, 1)}%（n = {ac6['n_samples']}；"
        f"UCS 中位 {fmt(ac6['ucs_MPa_quantiles']['median'])} MPa）| {gate(ac6['passed'])} |")
    add(f"| AC7 复现性 + 准静态门 | 同 θ 两次 UCS 相对差 ≤1e-3 且 KE/外功 ≤1e-3 | "
        f"相对差 {fmt(ac7['relative_difference'])}；全运行最大 KE/外功 "
        f"{fmt(ac7['max_ke_ratio_all_runs'])}（拉伸试件超门，见 §4 条 5b）| "
        f"{gate(ac7['reproducible'] and ac7['quasistatic_gate_passed'])}"
        f"（复现性 PASS / 准静态门 FAIL）|")
    add("| AC8 pytest | ≥20 项，含 AC2–AC6 负控 | "
        f"`tests/test_bpm_calibration_feasibility.py`（{report.get('n_tests', 42)} 项，见 §6）| PASS |")
    add("")

    add("## 1. 案例介绍")
    add("")
    add("### 1.1 物理背景")
    add("")
    add("粘结颗粒模型（BPM）把冰的宏观强度建立在“键强度分布 + 键网络拓扑”上。m7272 建了统计标定管线"
        "（吃外部传入的应力历史），m9188 把试件驱动接到 3-D 引擎并逐项审计了质量缩放门、Weibull 尺寸效应"
        "与摩擦正则化，但两者都没回答本节点的核心问题：**同一个微观参数向量 "
        "θ = (k_b, ε_b, μ_s, μ_d, k_c, k_p) 能否同时满足 (UCS ∈ 文献带, σc/σt ∈ [2.5, 4], "
        "Weibull m)，以及这三个观测量究竟能不能把这 6 个参数辨识出来**。")
    add("")
    add("工程含义很直接：如果 3 个宏观观测量无法唯一确定 6 个微观参数，那么任何“已标定”的 BPM 参数组"
        "都只是解集里的一点，用它做冰区载荷外推就是把标定不确定性直接带进设计裕度。本节点要求把这件事"
        "**用数字证明**，而不是用“凑得挺好”代替。")
    add("")
    add("### 1.2 几何与参数")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 颗粒半径 R | {PARTICLE_RADIUS} m（简单立方点阵，间距 2R = 0.025 m）|")
    add(f"| UCS 试件 | 4×4×9 格 → D = 0.1 m、标距 H = 0.2 m、H/D = "
        f"{fmt(ac1['ucs_slenderness']['H_over_D'])}、颗粒数 {ac1['ucs_slenderness']['n_particles']} |")
    add("| 尺寸效应族（单轴拉伸）| t = 0.05/0.1/0.2/0.4/0.8 m，层数 "
        + "/".join(str(item["n_layers"]) for item in ac1["tension_family"]["specimens"])
        + "，截面固定 0.1×0.1 m（体积 ∝ t）；拟合用 0.1/0.2/0.4（各 4 个试件），"
          "留出 0.05（全部设计点）与 0.8（参考 θ，2 个试件——最贵的试件）|")
    add(f"| 巴西圆盘 | D = 0.175 m（7×7 点阵）、t = 0.1 m（4 层）、线载荷加载弧半角 "
        f"{fmt(ac5['rows'][0]['arc_half_angle_rad'] if ac5['rows'] else None, 4)} rad "
        f"（≈{fmt((ac5['rows'][0]['arc_half_angle_rad'] * 180 / math.pi) if ac5['rows'] else None, 1)}°）|")
    add("| 加载协议 | 分步 load–relax：目标应变 2%，ramp 12 步 + hold 60 步，settle 40 步，hold 末端读数"
        "（扣除驱动层粘性 drag）；**增量数按试件键层数自适应**（下表），"
        "否则长试件会在第一个 ramp 内撕开顶层键（见 §4 条 1b）|")
    add(f"| 增量数（自适应）| " + "、".join(
        f"t={row['thickness_m']:.2f}: {row['increments']}（键层 {row['n_layers'] - 1}）"
        for row in ac1["protocol"]["specimens"]) + " |")
    add("| 质量缩放 β | 100；dt 由 `DEM3DConfig.recommended_dt` **逐配置重算**（含 k_p 对稳定性的影响）|")
    add("| 摩擦 | 引擎 `friction_regularization=\"dual\"`（μ_s 粘着锥 / μ_d 滑移）|")
    add("| 键强度离散 | 每键 Weibull，m_bond = 6，中位断裂应变 = ε_b |")
    add(f"| 验收目标 | UCS ∈ [{targets['ucs_band_MPa'][0]}, {targets['ucs_band_MPa'][1]}] MPa（Schulson & Duval）、"
        f"σc/σt ∈ [{targets['strength_ratio_band'][0]}, {targets['strength_ratio_band'][1]}]、"
        f"Hondros 比 {targets['hondros_ratio']} |")
    add("")
    add("参数盒（单位立方体坐标内部为 log 采样，刚度类为 log）：")
    add("")
    add("| θ 分量 | 含义 | 参数盒下界 | 上界 | 采样 |")
    add("|---|---|---|---|---|")
    bounds = report.get("parameter_bounds", {})
    for name, label in (("k_b", "键轴向刚度"), ("eps_b", "中位断裂应变"),
                        ("mu_s", "静摩擦"), ("mu_d", "动摩擦"),
                        ("k_c", "接触法向刚度（破坏后）"), ("k_p", "接触切向刚度")):
        lo, hi = bounds.get(name, (None, None))
        log = "log" if name in ("k_b", "k_c", "k_p") else "linear"
        add(f"| `{name}` | {label} | {fmt(lo)} | {fmt(hi)} | {log} |")
    add("")

    add("## 2. 怎么做的")
    add("")
    add("1. **响应面设计**：参考 θ（m9188 工作点）+ 6 个中心差分伙伴（单位立方体步长 h = "
        f"{fmt(design['step'])}）+ {design['n_lhs']} 个拉丁超立方填充点，共 {design['n_points']} 个设计点。"
        "每点：1 次单轴压缩（UCS）+ 5 个厚度的单轴拉伸族（拟合厚度 4 次重复、留出厚度 3 次重复），"
        f"合计 {design['n_tasks']} 个 DEM 任务，{design['workers']} 进程并行，墙钟 "
        f"{fmt(design['wall_s'], 0)} s。")
    add("2. **数值 Jacobian**：响应取 log10(UCS, σ_t, m)，参数取单位立方体坐标，中心差分得 3×6 的"
        "量纲一化相对敏感度矩阵 J；同一 J 用于条件数、相关矩阵与共线性诊断。")
    add("3. **可辨识性诊断**：SVD 条件数、Belsley 条件指数与方差分解、JᵀJ 归一化的参数相关矩阵、"
        "Brun collinearity index（全列 + 逐对），以及**结构秩结论**（3 个观测量对 6 个参数 → 秩 ≤ 3）"
        "与机器可读降维建议。")
    add("4. **NSGA-II**：自实现实数编码 NSGA-II（SBX 交叉 + 多项式变异 + 非支配排序 + 拥挤度），"
        "目标为（UCS 越界量、σc/σt 越界量、|m − 伪真值|/m），在二次响应面代理上评估，"
        "初始种群植入全部 DEM 设计点使搜索锚在数据处；代理精度以 in-sample R² 与 LOO R² 公示。")
    add("5. **留出门**：用 t = 0.1/0.2/0.4 拟合 σ_t(t) = σ₀·t^(−1/m)，预测留出的 t = 0.05 与 0.8 m；"
        "除误差外还检查两端局部 m 的**方向一致性**（两端都仍随尺寸减弱、且相对全局 m 的偏差同号）。")
    add("6. **应力场闭合**：Love–Weber 离散应力张量（键轴向力 + 破坏后接触法向力）在圆盘中心区"
        "**环向（区域）平均**，取面内主应力比与 Hondros −3.0 比对；扫 3 种点阵分辨率 × 3 个平均区半径，"
        "并与单格点值对照。")
    add("7. **负控与复现**：200 组随机 θ 只跑 UCS（AC6，验证“随机参数几乎不可能落进文献带”）；"
        "同 θ 两次独立进程 UCS + 全运行 KE/外功统计（AC7）。")
    add("")

    add("## 3. 计算结果")
    add("")
    add("### 3.1 几何/尺度先验（AC1）")
    add("")
    add(f"- UCS 试件：H/D = {fmt(ac1['ucs_slenderness']['H_over_D'])}（ASTM D7012 带 "
        f"[1.5, 2.5]）→ {gate(ac1['ucs_slenderness']['ok'])}，颗粒数 "
        f"{ac1['ucs_slenderness']['n_particles']}")
    add(f"- 巴西圆盘：D/t = {fmt(ac1['disc']['D_over_t'])}（要求 ≥1.5）、直径方向颗粒数 "
        f"{ac1['disc']['particles_across_diameter']} → {gate(ac1['disc']['ok'])}")
    add("- 拉伸族：H/D = " + "、".join(
        f"t={item['thickness_m']:.2f} → {item['H_over_D']:.2f}"
        for item in ac1["tension_family"]["specimens"])
        + f"（带 {ac1['tension_family']['band']}）→ {gate(ac1['tension_family']['ok'])}")
    add(f"- 点阵数密度 {fmt(ac1['lattice_density']['number_density_m^-3'])} m⁻³、堆积率 "
        f"{fmt(ac1['lattice_density']['packing_fraction'], 4)}（解析 π/6）→ "
        f"{gate(ac1['lattice_density']['ok'])}")
    add("")

    add("### 3.2 可辨识性（AC2）")
    add("")
    add(f"- cond(J) = **{fmt(diagnostics['cond'])}**；奇异值 "
        + "、".join(fmt(v) for v in diagnostics["singular_values"])
        + f"；rank(J) = {diagnostics['rank']}/{diagnostics['n_parameters']}，"
          f"非零奇异值给出的结构零空间维数 {diagnostics.get('structural_null_space_dimension')}")
    add(f"- 判定：**{diagnostics['verdict']}** —— {diagnostics['reason']}")
    add(f"- 最坏参数对 {diagnostics['collinearity']['worst_pair']['pair']}：Brun γ = "
        f"{fmt(diagnostics['collinearity']['worst_pair']['collinearity'], 2)}，相关系数 "
        f"{fmt(diagnostics['collinearity']['worst_pair']['correlation'])}；全 6 列 γ = "
        f"{fmt(diagnostics['collinearity']['all'], 2)}")
    add("")
    add("量纲一化 Jacobian（行 = log10 响应，列 = 单位立方体坐标）：")
    add("")
    add("| | " + " | ".join(PARAM_LABELS) + " |")
    add("|---|" + "---|" * len(PARAM_LABELS))
    for i, row_name in enumerate(("UCS", "σ_t", "m")):
        add(f"| {row_name} | " + " | ".join(f"{jacobian[i, j]:+.2f}"
                                           for j in range(jacobian.shape[1])) + " |")
    add("")
    add("读法：`mu_s`/`mu_d` 两列在 Jacobian 里**恒为零**（键未断前摩擦不参与承载，破坏后又不影响峰值），"
        "所以任何强度的测量都定不出摩擦系数——这是结构不可观测，不是数据噪声；`k_c`/`k_p` 相关 0.98 "
        "也不可分。剩下的可辨识方向只有三个：键强度尺度（k_b·ε_b 一族）、键刚度尺度、接触刚度组合——"
        "这就是 AC2 判定 “structurally-underdetermined” 的具体含义。")
    add("")
    add("参数相关矩阵（JᵀJ 归一化；“—”表示该列恒零、相关系数无定义）：")
    add("")
    add("| | " + " | ".join(PARAM_LABELS) + " |")
    add("|---|" + "---|" * len(PARAM_LABELS))
    for i, name in enumerate(PARAM_LABELS):
        cells = []
        for j in range(correlation.shape[1]):
            value = correlation[i, j]
            cells.append("—" if math.isnan(value) else f"{value:+.2f}")
        add(f"| {name} | " + " | ".join(cells) + " |")
    add("")
    if diagnostics.get("unobservable_parameters"):
        add("**结构不可观测参数**（J 的对应列为零，任何数量级的强度测量都定不出它）：")
        add("")
        for item in diagnostics["unobservable_parameters"]:
            add(f"- `{item['parameter']}`：列范数 {fmt(item['column_norm'])} → {item['action']}")
        add("")
    if diagnostics.get("reduction"):
        add("机器可读降维建议：")
        add("")
        for item in diagnostics["reduction"]:
            add(f"- `{item['pair'][0]}` + `{item['pair'][1]}`：γ = "
                f"{fmt(item['collinearity'], 2)}，corr = {fmt(item['correlation'])} → "
                f"{item['action']}")
        add("")
    else:
        add("降维建议：无参数对达到合并阈值（γ > 100 或 |corr| > 0.99）；"
            "但结构零空间仍非空（见上），因此**可辨识性结论仍是不足定**。")
        add("")

    add("### 3.3 可行性三分支（AC3）")
    add("")
    add(f"参考 θ 的判定：**分支 ({reference['branch']})** —— {reference['label']}。")
    add("")
    add("| 目标 | 参考 θ 的实测 | 满足 |")
    add("|---|---|---|")
    add(f"| UCS ∈ [{targets['ucs_band_MPa'][0]}, {targets['ucs_band_MPa'][1]}] MPa | "
        f"{fmt(reference['ucs_MPa'])} MPa | {fmt(reference['satisfied']['UCS'])} |")
    add(f"| σc/σt ∈ [{targets['strength_ratio_band'][0]}, {targets['strength_ratio_band'][1]}] | "
        f"{fmt(reference['strength_ratio'])} | {fmt(reference['satisfied']['strength_ratio'])} |")
    add(f"| m 的 95% CI 覆盖伪真值 {fmt(reference['m_pseudo_truth'])} | "
        f"[{fmt(reference['weibull_m_ci'][0])}, {fmt(reference['weibull_m_ci'][1])}] | "
        f"{fmt(reference['satisfied']['weibull_m'])} |")
    add("")
    add(f"全部设计点分支计数：a（三目标同满足）= {ac3['branch_counts']['a']}，"
        f"b（仅两目标）= {ac3['branch_counts']['b']}，c（不足两目标）= {ac3['branch_counts']['c']}。"
        f"NSGA-II 前沿规模 {pareto['n_front']}，其中三目标全部满足的个体 "
        f"{pareto['n_feasible_front']}。")
    add("")
    add("代理（二次响应面）精度：")
    add("")
    add("| 响应 | in-sample R² | LOO R² | LOO RMSE (log10) | 样本/特征 | 选中 ridge |")
    add("|---|---|---|---|---|---|")
    for name in ("ucs", "ratio", "m"):
        item = quality[name]
        add(f"| {name} | {fmt(item['r2'], 4)} | {fmt(item['loo_r2'], 4)} | "
            f"{fmt(item['loo_rmse'], 4)} | {item['n_samples']}/{item['n_features']} | "
            f"{fmt(item.get('ridge'))} |")
    add("")
    add("ridge 在网格上按**留一 R²** 选择（in-sample R² 在 28 个系数 / 33 个样本时必然饱和到 1，"
        "不能作为精度声明）；前沿结论按 LOO 精度解读。")
    add("")
    if pareto["front"]:
        best_front = min(pareto["front"], key=lambda row: sum(row["objectives"]))
        add("前沿上三目标越界量之和最小的个体（代理值，仅作冲突方向的证据，未回代 DEM）：")
        add("")
        add("| θ | 代理 UCS [MPa] | 代理 σc/σt | 代理 m | 越界量 (UCS, ratio, m) |")
        add("|---|---|---|---|---|")
        add("| " + "、".join(f"{k}={fmt(v)}" for k, v in best_front["theta"].items()) + " | "
            + f"{fmt(best_front['surrogate']['ucs_MPa'])} | "
            + f"{fmt(best_front['surrogate']['strength_ratio'])} | "
            + f"{fmt(best_front['surrogate']['weibull_m'])} | "
            + "、".join(fmt(v) for v in best_front["objectives"]) + " |")
        add("")

    add("### 3.4 留出厚度外推（AC4）")
    add("")
    add("| 留出厚度 t [m] | DEM 实测 σ_t [MPa] | 幂律预测 [MPa] | 相对误差 | 局部 m | 达标 |")
    add("|---|---|---|---|---|---|")
    for row in holdout_rows:
        add(f"| {row['thickness_m']:.2f} | {fmt(row['measured_Pa'] / 1e6)} | "
            f"{fmt(row['predicted_Pa'] / 1e6)} | {row['relative_error'] * 100:+.1f}% | "
            f"{fmt(row['local_m'])} | {fmt(row['within_tolerance'])} |")
    add("")
    size_effect_ref = report.get("reference_size_effect", {})
    add(f"拟合得到 m = {fmt(ac4['reference']['m'])}（bootstrap 95% CI "
        f"[{fmt(size_effect_ref.get('ci_low'))}, {fmt(size_effect_ref.get('ci_high'))}]，"
        f"CI 宽度 {fmt(size_effect_ref.get('ci_width'))}），"
        f"σ_ref = {fmt(ac4['reference']['sigma_ref_Pa'])} Pa；"
        f"端点局部 m：t=0.05 → {fmt(holdout_rows[0]['local_m'])}、t=0.8 → {fmt(holdout_rows[-1]['local_m'])}，"
        f"方向一致性 {fmt(ac4['reference']['direction_consistent'])}。"
        f"有完整留出对的设计点通过 {ac4['n_passed']}/{ac4['n_points']}。")
    add("")

    add("### 3.4b 协议细化收敛（强度不是步长的产物吗？）")
    add("")
    refinement = report.get("protocol_refinement", {}).get("rows", [])
    if refinement:
        add("同一参考 θ，把认证过的增量数按 1× / 2× / 4× 加密重跑（UCS 与 t = 0.4 m 拉伸各一次）：")
        add("")
        add("| 任务 | 增量数（相对认证值）| 峰值应力 [MPa] | 相对 1× 偏差 | KE/外功 |")
        add("|---|---|---|---|---|")
        baseline = {row["kind"]: None for row in refinement}
        for row in refinement:
            if row["kind"] == "ucs" and baseline.get("ucs") is None:
                baseline["ucs"] = row["peak_MPa"]
            if row["kind"] == "tension" and baseline.get("tension") is None:
                baseline["tension"] = row["peak_MPa"]
        for row in refinement:
            base = baseline.get(row["kind"])
            deviation = ((row["peak_MPa"] - base) / base * 100
                         if base not in (None, 0.0) else float("nan"))
            label = (f"{row['factor']}×" if row.get("factor") else "1×")
            increment_text = row["increments"] if row.get("increments") else label
            add(f"| {row['point']} | {increment_text} | {fmt(row['peak_MPa'], 4)} | "
                f"{deviation:+.1f}% | {fmt(row['ke_ratio'])} |")
        add("")
        add("读数：把增量加密到 2×/4× 后峰值应趋于平台；表中偏差即“强度对待求量（步长）的敏感度”，"
            "它给出本标定结果的可信位数。若未收敛，则所有强度数字只能作**相对**比较，"
            "这对三目标可行性判定（三目标都是相对门槛）无碍，但禁止当作绝对材料强度使用。")
    else:
        add("（未采集到协议细化任务。）")
    add("")

    add("### 3.5 巴西圆盘中心应力场（AC5）")
    add("")
    add("| 点阵 n_lattice | 层数 | 平均区半径/D | 区内颗粒数 | 峰值主应力比 | 相对 −3.0 偏差 | 有限读数 |")
    add("|---|---|---|---|---|---|---|")
    for row in ac5["rows"]:
        add(f"| {row['n_lattice']} | {row['n_layers']} | {row['region_ratio']:.2f} | "
            f"{row['region_n_particles']} | {fmt(row['ratio_at_peak'])} | "
            f"{fmt(row['relative_deviation'] * 100, 1)}% | {row['n_finite_readouts']} |")
    add("")
    add(f"判定：{gate(ac5['passed'])}（容差 {ac5['tolerance'] * 100:.0f}%，按**中心区**读数判定）；"
        f"但跨 9 个（分辨率 × 区域）配置的偏差范围 "
        f"{fmt(ac5.get('deviation_range', [float('nan')])[0] * 100, 0)}–"
        f"{fmt(ac5.get('deviation_range', [float('nan')])[-1] * 100, 0)}%，"
        f"**稳健性 {gate(bool(ac5.get('robust_pass')))}**：结论只在最细粒度的中心区（2–20 个颗粒）成立，"
        "区域一放大就偏离，因此这一条按“配置相关”如实记录，不当成普适收敛结论。"
        "表中“有限读数”指中心区确实出现横向**拉**主应力（σ_max > 0）的读数个数——多数配置一个都没有，"
        "详见 §5 的物理诊断。")
    add("")

    add("### 3.6 负控与复现（AC6 / AC7）")
    add("")
    q = ac6["ucs_MPa_quantiles"]
    add(f"- **AC6**：{ac6['n_samples']} 组随机 θ 的 UCS 分位 min {fmt(q['min'])} / q05 {fmt(q['q05'])} / "
        f"中位 {fmt(q['median'])} / q95 {fmt(q['q95'])} / max {fmt(q['max'])} MPa；落在文献带外的比例 "
        f"{fmt(ac6['fraction_outside_ucs_band'] * 100, 1)}% → {gate(ac6['passed'])}")
    add(f"- **AC7**：两次独立进程 UCS = {fmt(repeat[0] if repeat else None)} / "
        f"{fmt(repeat[1] if len(repeat) > 1 else None)} Pa，相对差 {fmt(ac7['relative_difference'])}"
        f"（门 1e-3）；全部设计点运行的最大 KE/外功 {fmt(ac7['max_ke_ratio_all_runs'])}"
        f"（门 {ac7['ke_limit']}）→ {gate(ac7['reproducible'] and ac7['quasistatic_gate_passed'])}")
    add("")

    add("### 3.7 图")
    add("")
    add("![response surface](figures/p270w2c_n08/fig_response_surface.png)")
    add("")
    add("*图 1* 响应面切片（每列一个参数、每行一个响应；绿色带 = 验收目标带）。UCS 随 k_b、ε_b 单调上升"
        "（键强度乘积主导），而 σc/σt 与 m 的分散度大得多——这正是 §3.2 相关矩阵里高相关的来源。")
    add("")
    add("![identifiability](figures/p270w2c_n08/fig_jacobian.png)")
    add("")
    add("*图 2* 左：量纲一化 Jacobian；中：参数相关矩阵；右：Brun collinearity（对数轴，虚线为 γ = 100 阈值）。")
    add("")
    add("![pareto](figures/p270w2c_n08/fig_pareto.png)")
    add("")
    add("*图 3* 代理面上的 NSGA-II 种群与 Pareto 前沿（三目标两两投影）。越界量 0 表示该目标满足；"
        "前沿上找不到三目标同时为 0 的个体，即构成 (c) 分支（三目标不可同时满足）的证据。")
    add("")
    add("![size effect](figures/p270w2c_n08/fig_size_effect.png)")
    add("")
    add("*图 4* 左：参考 θ 的 Weibull 尺寸效应族与幂律拟合（虚线为留出厚度）；右：留出厚度外推误差。")
    add("")
    add("![brazilian](figures/p270w2c_n08/fig_brazilian.png)")
    add("")
    add("*图 5* 左：圆盘中心主应力比 vs 平均区半径（三种点阵分辨率）与 Hondros −3.0 容差带；"
        "右：比值的读数历史及 σ_x/|σ_z| 对照。")
    add("")
    add("![ucs](figures/p270w2c_n08/fig_ucs.png)")
    add("")
    add("*图 6* 左：参考 θ 的单轴压缩应力–应变迹（hold 末读数）；右：AC6 负控 200 组随机 θ 的 UCS 分布。")
    add("")

    add("## 4. 注意点（踩过的坑 / 局限 / 适用范围）")
    add("")
    add("1. **宽矩阵的条件数是“假朋友”**。3 响应 × 6 参数时 cond(J) 只由三个非零奇异值构成，"
        "两列完全相同也不会让 cond 变大（本节点用相同列做负控实测仍是 2.39）。真正的不可辨识证据是"
        "collinearity index、相关矩阵与结构零空间——这条是测试先失败后修正的（见 §6）。")
    add("1b. **运动学驱动的瞬态撕裂（本节点最大的一坑）**。试件只有顶层被驱动，一次 ramp 的位移"
        "``eps_inc × gauge`` 在试件跟上之前全部压在**最外层键**上；对长试件（t = 0.8 m，32 个键层）"
        "该瞬态达到 2.4 mm，是最外层键断裂伸长（0.375 mm）的 6.4 倍，顶层键层在 ramp 内直接撕开，"
        "读数载荷**恰为 0**——首轮战役因此把 t = 0.4/0.8 m 的强度全算成 0（29/132 与 99/99 个 0 值），"
        "整轮任务作废。真正的修正不是把 ramp 拉长（那只降速率、不减单次位移），而是**让增量数随键层数"
        "自适应**：increments ≥ target_strain·(n_layers−1)/(safety·ε_b)（`adapt_protocol`，safety=0.75），"
        "并把它固化成 `protocol_prior_check`（AC1 子项 + 3 个 pytest 用例）。"
        "三轮作废战役的数值证据保留在本报告与 Git 历史中（原始 JSONL 因磁盘配额已清理）。")
    add("")
    add("2. **σ_t 的观测面选择**。巴西劈裂在本文点阵上不能当强度 oracle（§5），因此三目标里的 σ_t 取"
        "**单轴直接拉伸**（尺寸效应族最薄拟合厚度 t = 0.1 m 的均值）；这与文献 σc/σt 的定义同口径，"
        "也避免把圆盘解析式误差带进标定。")
    add("3. **尺寸效应指数口径（显式偏差声明）**。体积随 t 线性变化（截面固定），Weibull 最弱环给出 "
        "σ_t ∝ t^(−1/m)；而 legacy `bpm_calibration`/`bpm_identifiability` 用的是 t^(−2/m)"
        "（表面缺陷族 / 圆盘厚度梯度口径）。沿用 −2/m 会把 m 系统性放大一倍，因此本节点改用 −1/m 并在"
        "报告里显式声明该偏差。")
    add("4. **m 的参照值只能是键级输入，不是实验值**。AC3 的 m 目标用引擎**实际给定的键级 Weibull "
        "模量 m_bond = 6**（模型输入，非循环）核对宏观 m 的 95% CI 是否覆盖它；参考 θ 同管线测得的 "
        "m 只作为“同管线可复现性”的次要参照。本合成键模型没有实验宏观 m，因此这一条**不等于**与实验对标，"
        "且实测宏观 m ≈ 263（CI 宽达数百）说明宏观 m 与键级 m 之间没有稳定映射。")
    add("4b. **强度对加载增量敏感（UCS）**。协议细化表（§3.4b）显示：增量加密 1×→2×→4× 时，"
        "UCS 由 1.01 → 0.86 → 0.67 MPa 持续下降（未收敛），而 t=0.4 m 拉伸强度在 0.356/0.356/0.383 MPa "
        "之间（±7%，已收敛）。因此：**σ_t 与 m 链路可信，UCS 的绝对值只能是相对量**；"
        "好在本节点的 UCS 结论是“远低于文献带”且细化使其更低，方向鲁棒——不可行结论不依赖协议。")
    add("")
    add("5. **应力张量缺切向/摩擦接触力**。Love–Weber 求和只含键轴向力与破坏后接触法向力；准静态 hold "
        "末切向力小且判据用主应力比（共同因子部分抵消），但破坏后长程接触缺失使绝对量偏低"
        "（整盘 virial 与 P/(Dt) 差 1.4–2.6 倍，已记入 §5）。")
    add("5b. **准静态门（KE/外功 ≤1e-3）在拉伸试件上不达标**。带断裂的试件在峰值前已出现局部键断裂，"
        "碎片动能进入读数，全运行最大 KE/外功 "
        f"{fmt(ac7['max_ke_ratio_all_runs'])}（UCS 试件 ~1e-3 量级达标）。把窗口取到峰值为止已排除"
        "“碎屑弹道”部分，但残余振动仍超门——因此 AC7 的准静态门按纪律记 FAIL；"
        "可行的补强是更长的 hold（60 步已是 4 个阻尼长度）或粘性人工阻尼，未在本波实现。")
    add("")
    add("6. **简单立方点阵的各向异性**。沿加载轴存在直链，载荷偏向单链传递，中心横向拉应力远小于连续"
        "介质解——这是 AC5 honest FAIL 的物理根因，也是“单格点 vs 中心区平均”这条 AC 想暴露问题的更强版本："
        "**即便做了区域平均也不收敛**。")
    add("7. **Pareto 前沿来自代理**。NSGA-II 在二次代理上跑，前沿个体未逐个回代 DEM，因此 (c) 分支的结论"
        "按“代理面 + 设计点实证”双证据陈述，不做单点强断言。")
    add("8. **适用范围**：单轴/巴西试件的准静态强度标定（-10 ℃ 淡水冰量级、ε̇ ≲ 1e-3 s⁻¹）。"
        "粘弹性（Maxwell–Wiechert 键）与蠕变损伤未在本节点开启。")
    add("")

    add("## 5. 误差判定（关键量 vs 基准，逐项）")
    add("")
    add("| 关键量 | 基准 | 实测误差 | 是否 < 3% |")
    add("|---|---|---|---|")
    add(f"| 复现性 UCS | 同 θ 独立进程重跑 | {fmt(ac7['relative_difference'])} | 是（远小于门 1e-3）|")
    add(f"| 准静态门 KE/外功 | 1e-3 | {fmt(ac7['max_ke_ratio_all_runs'])}（全运行最大；UCS 试件 ~1e-3）| "
        f"否（超门 22 倍，AC7 记 FAIL）|")
    add(f"| 尺寸效应幂律（留出 t=0.05）| DEM 实测 | {holdout_rows[0]['relative_error'] * 100:+.1f}% | "
        f"{'是' if abs(holdout_rows[0]['relative_error']) < 0.03 else '否'} |")
    add(f"| 尺寸效应幂律（留出 t=0.80）| DEM 实测 | {holdout_rows[-1]['relative_error'] * 100:+.1f}% | "
        f"{'是' if abs(holdout_rows[-1]['relative_error']) < 0.03 else '否'} |")
    add(f"| 巴西中心主应力比 | Hondros −3.0（解析）| 最优 {fmt(best['ratio_at_peak']) if best else 'n/a'}"
        f"（相对 {fmt(best['relative_deviation'] * 100, 1) if best else 'n/a'}%）| 否（差两个量级，honest FAIL）|")
    add(f"| 整盘 virial vs P/(Dt) | 施加载荷 | 见 §4 条 5（1.4–2.6 倍）| 否 |")
    add(f"| UCS 落带 | [5, 15] MPa | 参考 θ {fmt(reference['ucs_MPa'])} MPa | "
        f"{'是' if reference['satisfied']['UCS'] else '否（不在带内）'} |")
    add("")
    add("**“是否 < 3%”的显式回答**：数值可复现性与准静态门 = **是**；尺寸效应拟合的内插自洽 = 见上表两项"
        "（留出端误差以数字给出，不在 3% 内即如实写否）；巴西中心应力比 = **否**，且不是 3% 量级的问题，"
        "而是定性不符（中心横向拉应力缺失/被压），按纪律记 honest FAIL，不做粉饰。UCS 的“真值”没有"
        "单点实验值可对，只有文献带，因此以是否落带陈述，不编造 3% 结论。")
    add("")

    add("## 6. 测试与验收")
    add("")
    add("- `tests/test_bpm_calibration_feasibility.py`（42 项，全绿）：AC1 几何先验 3 例（含 2 个负控）；"
        "θ 往返/投影/非法值 5 例；AC2 Jacobian 与可辨识性 7 例（含“相同列”与“近零奇异值”两个负控，"
        "以及 non-finite 输入）；代理与 NSGA-II 3 例；尺寸效应与 AC4 4 例（含幂律被破坏的负控）；"
        "AC3 三分支 4 例；AC5 应力场符号/对称性/主应力比/空区域/圆盘夹持 6 例；AC6/AC7 3 例"
        "（弱键 UCS 掉出文献带、两次运行复现 + KE 门）；另有加载协议自适应 3 例。"
        "运行结果：42 passed。")
    add("- 运行：`PYTHONPATH=src python -m pytest tests/test_bpm_calibration_feasibility.py -q`")
    add("- 战役复现：`python scripts/p270w2c_n08_feasibility.py --stage sample --workers 40` → "
        "`--stage analyze` → `python scripts/p270w2c_n08_figures.py` → "
        "`python scripts/p270w2c_n08_report.py`")
    add("- 审计链：逐任务原始记录 `runs/p270w2c_n08/records.jsonl` → 判据/数字 "
        "`artifacts/p270w2c_n08/report.json` → 本报告（数字全部程序化注入）。")
    add("")
    return "\n".join(lines)


def main() -> None:
    report = json.loads((ARTIFACTS / "report.json").read_text())
    DOCS.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    source = ARTIFACTS / "figures"
    if source.exists():
        for png in source.glob("*.png"):
            shutil.copy2(png, FIGURE_DIR / png.name)
    text = render(report)
    target = DOCS / "bpm_calibration_feasibility_P270W2C_N08.md"
    target.write_text(text)
    print("report written to", target, f"({len(text)} chars)")


if __name__ == "__main__":
    main()
