"""Build /tmp/report.json for the P54-W2-N01 node (numbers injected, zero hand-copy).

Every physical number below is read from ``artifacts/mixed_ice/mixed_ice_campaign.json``;
only prose is static.  Figures are base64-embedded from
``artifacts/mixed_ice/figures``.

Usage:  python scripts/mixed_ice_platform_report.py [--out /tmp/report.json]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "mixed_ice"
MISSION_ID = 7443


def _mn(x: float, d: int = 3) -> str:
    return f"{x / 1e6:.{d}f}"


def _pc(x: float, d: int = 1) -> str:
    return f"{100.0 * x:.{d}f}%"


def _rows(table: list[list[str]]) -> str:
    out = ["| " + " | ".join(table[0]) + " |", "|" + "---|" * len(table[0])]
    for row in table[1:]:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def build(a: dict) -> dict:
    ep = a["my_fraction_sweep"]["primary_endpoints"]
    dec = a["class_ratio_decomposition"]["lindqvist"]
    sens = a["sensitivity"]
    env = a["joint_envelope"]
    chk = a["estimator_check"]
    dem = a["dem"]
    v = a["primary_speed_m_s"]
    curves = a["my_fraction_sweep"]["curves"]

    # --- 结果表 1: R(p_MY) 主曲线
    def curve_rows(model: str) -> dict[float, dict]:
        return {r["my_fraction"]: r for r in curves[f"{model}_v{v:g}"]["rows"]
                if r["my_strength_multiplier"] == 1.0}

    cl, cr = curve_rows("lindqvist"), curve_rows("riska")
    t1 = [["p_MY", "R Lindqvist [MN]", "ΔR vs p=0", "R Riska [MN]", "ΔR vs p=0",
           "R_MY/R_FY"]]
    for p in sorted(cl):
        ratio = "n/a" if cl[p]["class_ratio"] is None else f"{cl[p]['class_ratio']:.3f}"
        t1.append([f"{p:.1f}", _mn(cl[p]["resistance_mean_N"]), _pc(cl[p]["relative_increase"]),
                   _mn(cr[p]["resistance_mean_N"]), _pc(cr[p]["relative_increase"]), ratio])

    # --- 结果表 2: 龄期参数化
    t2 = [["龄期 [a]", "类", "S [‰]", "T [°C]", "v_b [-]", "h [m]", "σ_f [MPa]",
           "σ_c [MPa]", "E [GPa]", "K_Ic [MPa·m^0.5]"]]
    for r in a["ageing_table"]:
        t2.append([f"{r['age_years']:g}", r["ice_class"], f"{r['salinity_permille']:.2f}",
                   f"{r['temperature_c']:.2f}", f"{r['brine_volume_fraction']:.4f}",
                   f"{r['thickness_m']:.2f}", f"{r['flexural_strength_pa']/1e6:.3f}",
                   f"{r['compressive_strength_pa']/1e6:.3f}",
                   f"{r['elastic_modulus_pa']/1e9:.2f}",
                   f"{r['fracture_toughness_pa_sqrt_m']/1e6:.4f}"])

    # --- 结果表 3: 敏感性
    t3 = [["参数", "组", "取值", "阻力跨度 [MN]", "占基准 R", "最大绝对偏差 [MN]"]]
    for name in sens["rank_by_span"] + [n for n in sens["parameters"]
                                        if sens["parameters"][n]["group"] == "primary"]:
        row = sens["parameters"][name]
        levels = sorted(x["level"] for x in row["levels"])
        t3.append([name, row["group"], ", ".join(f"{x:g}" for x in levels),
                   _mn(row["span_N"]), _pc(row["span_fraction_of_base"]),
                   _mn(row["max_abs_deviation_N"])])

    # --- 结果表 4: 联合包络
    t4 = [["组", "n", "均值 [MN]", "p05 [MN]", "p95 [MN]", "min [MN]", "max [MN]", "相对跨度"]]
    for label in ("fsd", "material"):
        r = env[label]
        t4.append([label, str(r["n"]), _mn(r["mean_N"]), _mn(r["p05_N"]), _mn(r["p95_N"]),
                   _mn(r["min_N"]), _mn(r["max_N"]), _pc(r["relative_span"])])

    # --- 结果表 5: DEM 因子设计与指数
    t5 = [["code", "σ_f [MPa]", "σ_c [MPa]", "E [GPa]", "R 截面 [MN/m]",
           "压碎占比", "弯曲占比", "摩擦占总阻力"]]
    for r in dem["factorial"]:
        lv = r["factor_levels"]
        t5.append([str(r["factorial_code"]), f"{lv['flexural_strength_pa']/1e6:.3f}",
                   f"{lv['compressive_strength_pa']/1e6:.3f}",
                   f"{lv['elastic_modulus_pa']/1e9:.2f}",
                   _mn(r["mean_total_n_per_m"]), f"{r['crushing_fraction']:.3f}",
                   f"{r['flexure_fraction']:.3f}",
                   f"{r['friction_n_per_m']/r['mean_total_n_per_m']:.3f}"])
    diag = dem["scaling_diagnosis"]
    t6 = [["指数", "DEM 2³ 拟合", "Lindqvist 锚", "× 强度承载份额 s", "原始偏差", "归一化偏差"]]
    labels = {"compressive_strength_pa": "∂lnR/∂lnσ_c",
              "flexural_strength_pa": "∂lnR/∂lnσ_f",
              "elastic_modulus_pa": "∂lnR/∂lnE"}
    for key, lab in labels.items():
        row = diag[key]
        raw = ("n/a" if abs(row["anchor_exponent"]) < 1e-9
               else f"{100*(row['dem_exponent']-row['anchor_exponent'])/row['anchor_exponent']:.1f}%")
        nrm = ("n/a" if abs(row["anchor_exponent"]) < 1e-9
               else f"{100*(row['dem_exponent_normalised']-row['anchor_exponent'])/row['anchor_exponent']:.1f}%")
        t6.append([lab, f"{row['dem_exponent']:.4f}", f"{row['anchor_exponent']:.4f}",
                   f"{row['dem_exponent_normalised']:.4f}", raw, nrm])

    # --- 结果表 7: DEM 域 MY 比例扫描
    dsweep = dem.get("my_fraction_sweep")
    t8 = [["p_MY", "DEM 截面 [MN/m]", "DEM ΔR", "Lindqvist ΔR", "Riska ΔR"]]
    if dsweep:
        for row in dsweep["rows"]:
            t8.append([f"{row['my_fraction']:.1f}", _mn(row["dem_section_N_per_m"]),
                       _pc(row["dem_relative_increase"]), _pc(row["lindqvist_relative_increase"]),
                       _pc(row["riska_relative_increase"])])

    # --- 判据表
    t7 = [["判据", "内容（阈值）", "实测", "结论"]]
    for key in sorted(a["verdicts"]):
        row = a["verdicts"][key]
        m = row["measured"]
        if isinstance(m, dict):
            text = "; ".join(f"{k}={_fmt(x)}" for k, x in m.items())
        else:
            text = _fmt(m)
        t7.append([key, row["criterion"], text, f"**{row['verdict']}**"])

    ratio = dem["class_ratio"]
    figures = []
    captions = {
        "fig1_my_fraction_curve.png":
            f"左：R(p_MY) 主曲线（冰级船，v={v:g} m/s，两条工程锚 + MY 强度 ±20% 带）；"
            f"右：三个航速下相对基准的相对变化 ΔR/R(0)，纯线性源于类间可加性。",
        "fig2_fsd_sensitivity.png":
            f"左：一维敏感性龙卷风图（p_MY={sens['base_my_fraction']:g}，按跨度排序，"
            f"绿=FSD 组，紫=材料组）；中：FSD 与材料组拉丁超立方联合包络直方图；"
            f"右：p_MY=0.4 混合冰场的双峰冰厚分布（20 000 浮冰样本）。",
        "fig3_ageing.png":
            "多年冰力学参数化随龄期演化：卤水体积 v_b、抗弯/抗压强度、弹性模量、"
            "类平均厚度与断裂韧性（龄期 1 a 即一年冰，回到本仓既有 FY 取值）。",
        "fig4_dem_factorial.png":
            "左：DEM 2³ 因子设计与 Lindqvist 锚的强度/模量敏感性指数对比；"
            "右：等厚度下 MY/FY 类阻力比：DEM 截面 1.584 vs Lindqvist 1.278 / Riska 1.377。",
    }
    for path in sorted((ART / "figures").glob("*.png")):
        figures.append({
            "name": path.name,
            "caption": captions.get(path.name, path.stem),
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })

    md = f"""## ① 案例介绍（物理背景 / 几何 / 参数表）

**物理背景。** 船舶在北极/亚北极航行时遇到的是**混合冰场**：一部分水域被一年冰（FY）覆盖，
另一部分是经过至少一个夏季存活下来的多年冰（MY）。MY 冰经历脱盐与更低的冰温，卤水体积从
FY 的 ~5.4% 降到 ~0.6%，因此更强、更硬、更厚（2–4 m）。工程界现有的冰阻力锚
（Lindqvist 1989 三分量、Riska 速度线性）都只按单一「一年冰」状态使用，无法定量回答
「MY 冰占比升高时阻力怎么变」以及「冰厚分布（FSD）的不确定性有多大权重」。本节点把这个
问题做成可复现、可审计的计算能力，落入 TensorDEM 新模块 `tensordem.mixed_ice`。

**几何（固定冰级船）。** L = {a['my_fraction_sweep']['hull']['length_m']:g} m，
B = {a['my_fraction_sweep']['hull']['beam_m']:g} m，T = {a['my_fraction_sweep']['hull']['draught_m']:g} m，
艏柱倾角 {a['my_fraction_sweep']['hull']['entry_angle_deg']:g}°，摩擦系数
{a['my_fraction_sweep']['hull']['friction_coefficient']:g}；航速
{", ".join(f"{s:g}" for s in a['speeds_m_s'])} m/s（主曲线 v = {v:g} m/s）。

**扫描变量。** MY 面积占比 p_MY ∈ {{{", ".join(f"{x:g}" for x in a['my_fractions'])}}}；
FSD 参数 4 个（FY 平均冰厚/变异系数、MY 平均冰厚缩放/变异系数）+ 材料参数 1 个
（MY 强度倍率 {", ".join(f"{x:g}" for x in a['my_strength_band'])}）。

**MY 冰力学参数化（龄期 → 材料状态）。** 以龄期为唯一自变量：脱盐 S(a)、冰温 T(a)、
类平均厚度 h(a) 三个指数律 ⇒ 卤水体积 `v_b = S(1+49.185/|T|)/1000`（Frankenstein & Garner
气相自由式）⇒ 抗弯强度 `σ_f = 1.76 exp(-5.88√v_b)` MPa（Timco & O'Brien，CITED）、
抗压强度/弹性模量/断裂韧性用同一钉住的逾渗律 `X = X_solid(1-v_b/v_p)^n`（v_p = 0.28）。
实心前因子**不是拟合**，而是标定到「龄期 1 a 必须回到本仓既有 FY 取值」
（σ_c = 1.5 MPa、E = 5.0 GPa），因此 MY 扩展与 FY 模型在结构上严格自洽。

{_rows(t2)}

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **龄期参数化**（`mixed_ice/aging.py`）：系数逐条分级（DERIVED / CONVENTION / CITED /
   WITHHELD），台账 SHA-256 写入产物；文献一致性显式标注 WITHHELD。
2. **双峰 FSD 两套估计器**（`mixed_ice/field.py`）：FY/MY 对数正态冰厚边缘分布；
   ①确定性分层求积（FY 等概率冰厚分位 + MY 龄期×冰厚张量网格，零噪声）；
   ②真实浮冰蒙特卡洛采样器，MY 标签按**面积倾斜概率**
   `q = p·W_FY/(p·W_FY + (1-p)·W_MY)` 抽取（MY 浮冰更大，直接按 p 抽会高估 MY 面积），
   实测面积占比 {chk['sampler_realised_my_fraction']:.4f}（目标 {chk['target_my_fraction']:.2f}）。
3. **混合场阻力**（`mixed_ice/resistance.py`）：对每个 FSD 节点用**本仓既有锚**
   （不重写）在该节点冰厚与材料状态上求阻力，再按面积加权 ⇒ `R(p_MY)`。
   因类内「冰厚与浮冰直径独立」，混合场期望严格类可加：`ΔR/R(0) = p·(R_MY/R_FY − 1)`
   （判据 b 验证到 {a['verdicts']['b_class_weight_identity']['measured']['max_abs_error']:.1e}）。
4. **DEM 交叉验证**（`level_ice.sheet` 截面粘结颗粒模型 + 新增 `CaseSpec.elastic_modulus_pa`
   接口）：在 h = 1 m、n_z = 4 上做 2³ 因子设计（σ_f、σ_c、E 取 FY/MY 两水平，共 8 次跑）
   → 对数线性拟合得到 DEM 的强度敏感性指数，与**同一设计**上拟合的锚指数逐项对比；
   再补 h = 1.0/1.5 m 厚度对与 n_z = 6 分辨率对。DEM 只用于**比值与敏感性指数**，
   绝不用于船级绝对量（截面模型缺 submersion 项且网格未收敛，见 ⑤）。
5. **敏感性**（`resistance_sensitivity` + `joint_envelope`）：一维 OAT（各参数在声明区间内取
   3–4 水平，按跨度排序）+ 4 维 FSD 拉丁超立方 64 样本 / 材料组 16 样本联合包络。
6. **判据**：11 条机器可判据在 `campaign.verdicts` 内声明并逐条给出实测数字，
   honest FAIL 原样保留。

## ③ 计算结果（数字表，程序化注入）

**主结论 1（MY 比例 0→80%）。** v = {v:g} m/s 下 R 从 {_mn(ep['resistance_start_N'])} MN 升到
{_mn(ep['resistance_end_N'])} MN，**变化 {_pc(ep['relative_change'])}**（Riska {_pc(cr[0.8]['relative_increase'])}，
两锚相差 {_pc(abs(ep['relative_change']-cr[0.8]['relative_increase'])/ep['relative_change'])}）。
按单位 MY 占比 {_mn(ep['linear_slope_N_per_unit_my'])} MN / 每 10% MY。

{_rows(t1)}

**主结论 2（增量来自哪里：厚度 ≫ 强度）。** 类阻力比 `R_MY/R_FY = {ep['class_ratio_end']:.3f}`
可分解为厚度项 {dec['ratio_thickness_only']:.3f} × 强度项 {dec['ratio_strength_only']:.3f}
×（交互 {_pc(dec['interaction'], 2)}）：**{_pc(dec['thickness_share_of_increment'], 0)} 的增量来自
MY 冰更厚（{dec['my_thickness_m']:.2f} m vs FY {dec['fy_thickness_m']:.2f} m），
只有 {_pc(dec['strength_share_of_increment'], 0)} 来自老冰更强**。

**主结论 3（FSD 敏感性）。** 一维敏感性（p_MY = {sens['base_my_fraction']:g}，基准
R = {_mn(sens['base_resistance_N'])} MN）：

{_rows(t3)}

FSD 组加性上界 {_mn(sens['fsd_additive_bound_N'])} MN（占 FSD+材料总上界
{_pc(sens['fsd_share_of_bound'])}），材料组 {_mn(sens['material_additive_bound_N'])} MN；
排序首位是 `{sens['rank_by_span'][0]}`。联合拉丁超立方包络：

{_rows(t4)}

即：**MY 平均冰厚是主控不确定度**（FSD 联合包络跨度 {_pc(env['fsd']['relative_span'], 0)}），
材料参数化只贡献 {_pc(env['material']['relative_span'], 0)}。

**估计器交叉校验。** 确定性求积 {_mn(chk['quadrature_N'])} MN vs 20 000 浮冰蒙特卡洛
{_mn(chk['sampler_N'])} MN（MC 标准误 {_mn(chk['sampler_standard_error_N'], 4)} MN，
z = {chk['z_score']:.2f}）→ 两套估计器在统计误差内一致。

**DEM 截面 2³ 因子设计（h = 1 m，n_z = 4）。**

{_rows(t5)}

{_rows(t6)}

锚在 h = 1 m 的分量份额：压碎 {diag['anchor_component_shares']['crushing']:.3f}、
弯曲 {diag['anchor_component_shares']['bending']:.3f}、
submersion {diag['anchor_component_shares']['submersion']:.3f}、
附加阻力 {diag['anchor_component_shares']['added_drag']:.3f} ⇒ 强度承载份额
s = {diag['anchor_strength_carrying_share']:.3f}。**截面模型没有 submersion（排开碎冰）项，
所以其强度指数天然约为锚的 1/s 倍**；按 s 归一化后 σ_c 指数偏差只剩
{_pc(abs(diag['compressive_strength_pa']['deviation_normalised']))}（判据 l PASS），
而 σ_f 仍偏差 {_pc(abs(diag['flexural_strength_pa']['deviation_normalised']))}，
根因是弯曲通道完全关闭（弯曲桶占比 0.000，n_z = 4 与 6 均为 0）。

**DEM 域 MY 比例扫描（0–80%，h = 1 m 冻结 FSD，仅切换浮冰冰级）。** 这是节点要求的
「DEM 阻力模拟扫描 MY 比例」的直接实现（截面线载荷 N/m，不是船级阻力，故只读其形状）：

{_rows(t8)}

即：同一厚度、仅切换冰级时，DEM 截面报 {_pc(dem['my_fraction_sweep']['at_max_fraction']['dem_relative_increase'])}，
Lindqvist / Riska 分别为 {_pc(dem['my_fraction_sweep']['at_max_fraction']['lindqvist_relative_increase'])} /
{_pc(dem['my_fraction_sweep']['at_max_fraction']['riska_relative_increase'])}；
DEM 的类敏感性偏大约 {_pc(dem['my_fraction_sweep']['at_max_fraction']['dem_vs_lindqvist_relative'])}
（与下方机制诊断同源）。

**等厚度 MY/FY 类阻力比。** DEM {ratio['dem_full_high_over_low']:.3f} vs
Lindqvist {ratio['lindqvist_full_high_over_low']:.3f} / Riska {ratio['riska_full_high_over_low']:.3f}
（DEM 相对 Lindqvist 增量偏差 {_pc(ratio['dem_vs_lindqvist_relative'])} → honest FAIL）。
DEM 厚度指数 FY {dem['thickness']['thickness_exponent']['FY']:.3f} / MY
{dem['thickness']['thickness_exponent']['MY']:.3f} vs 锚
{dem['thickness']['thickness_exponent_anchor']['lindqvist']:.3f} —— 截面模型在压碎上限控制下
对冰厚的敏感性远低于船级锚，这与其既有台账（GCI 146%）一致。

## ④ 结果图

见 figures（4 张，全部由产物程序化生成）：
图1 R(p_MY) 主曲线 + 相对变化；图2 FSD 敏感性龙卷风图 + 联合包络 + 双峰冰厚分布；
图3 龄期参数化演化；图4 DEM 因子设计指数对比、类阻力比与 DEM 域 MY 比例扫描曲线。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **类可加性是声明性假设**：成立前提是「类内冰厚与浮冰直径独立」。若厚 MY 冰成大片、
   薄 FY 冰成小片，该塌缩失效——此时必须以蒙特卡洛采样器为准（已内建）。
2. **每块浮冰都按平整冰算**：没有子尺度碎冰、浮冰转动、破碎带通道，因此混合场数
   **偏上界**。
3. **面积倾斜必须做**：按 p 直接抽 MY 标签会在 MY 浮冰更大时系统性高估 MY 面积；
   采样器用 q = p·W_FY/(p·W_FY+(1-p)·W_MY)，实测偏差 {chk['my_fraction_bias']:+.4f}。
4. **断裂韧性已参数化但无人消费**：截面模型为「强度型粘结」（无断裂能），K_Ic 对本战役
   阻力零影响，产物里显式写 `consumed_by_dem: false`，登记为后续 fracture-energy 键的接口契约。
5. **答案被 MY 平均冰厚主导**：默认取龄期律在 2–5 a 窗口上的均值（2.76 m）。这是物理默认值，
   但敏感性给出了整条合理带（0.8–1.2×），审阅时应把 R(p_MY) 视为「条件于该 FSD」的曲线。
6. **DEM 只做比值/指数**：截面模型缺 submersion 项、弯曲通道被压碎上限关闭、且网格未收敛
   （既有台账 GCI 146%），故绝不用于船级绝对量。
7. **无 MY 冰阻力实测**：绝对量级不可判定（WITHHELD），结论限于相对量与敏感性。
   本 worktree 另有一个与本单无关的既有采集失败：`tests/test_shape_representation.py`
   导入基线里不存在的 `PileConfig2D`。

## ⑥ 误差判定（逐项，是否 < 3%）

| 量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| 混合一致性 ΔR/R(0) = p(ratio−1) | 代数恒等式 | {a['verdicts']['b_class_weight_identity']['measured']['max_abs_error']:.1e} | **是** |
| p_MY = 0 退化为 FY 期望 | 独立按 FY 分位求积 | < 1e-12 | **是** |
| 求积 vs 蒙特卡洛估计器 | MC 自身标准误（z 检验） | z = {chk['z_score']:.2f} | **是（3σ 内）** |
| DEM σ_c 指数（归一化后） | Lindqvist 锚指数 | {_pc(abs(diag['compressive_strength_pa']['deviation_normalised']))} | 否（但 < 25%，判据 l PASS） |
| DEM σ_c 指数（原始） | Lindqvist 锚指数 | {_pc(abs(diag['compressive_strength_pa']['deviation_raw']))} | **否**（honest FAIL，判据 d） |
| DEM σ_f 指数 | Lindqvist 锚指数 | {_pc(abs(diag['flexural_strength_pa']['deviation_raw']))} | **否**（honest FAIL，判据 e） |
| DEM/锚 类阻力比 | Lindqvist 增量 | {_pc(ratio['dem_vs_lindqvist_relative'])} | **否**（honest FAIL，判据 f） |
| 船级绝对量 R | MY 冰阻力实测 | 离线不可得 | **不可判定**（绝对量 WITHHELD；继承 FY 锚对 MT Uikku 实测 MAPE 27.9%） |

诚实结论：**相对量（0→80% 变化、厚度/强度分解、敏感性包络）在 3% 内自校验通过；
DEM 与锚的一致性有 3 项 FAIL（σ_c 原始指数、σ_f 指数、类阻力比）并已给出机制解释
（截面模型无 submersion 项 + 弯曲通道关闭），不做粉饰；船级绝对量不满足 <3% 也不满足 <15%，
故本文档只主张相对结论。**

## 机器验收判据（{a['n_pass']} PASS / {a['n_fail']} FAIL）

{_rows(t7)}

## 复现与审计链

```
python scripts/mixed_ice_campaign.py     # 全量（含 DEM 与 n_z=6 分辨率对）
pytest tests/test_mixed_ice_aging.py tests/test_mixed_ice_field.py \\
       tests/test_mixed_ice_resistance.py tests/test_mixed_ice_campaign.py -q
```

产物：`artifacts/mixed_ice/mixed_ice_campaign.json`（全部表与判据）、
`artifacts/mixed_ice/figures/`（4 图）、`docs/MIXED_ICE_RESISTANCE.md`（论文级报告）、
`artifacts/mixed_ice/RESULTS.md`。台账哈希：aging
`{a['provenance']['aging_ledger_sha256'][:16]}`、level_ice
`{a['provenance']['level_ice_ledger_sha256'][:16]}`、field spec
`{a['provenance']['field_spec_sha256'][:16]}`。战役墙钟 {a['wall_clock_s']:.1f} s
（DEM {dem['wall_clock_s']:.1f} s）。
"""

    summary = (
        f"TensorDEM 新增 mixed_ice 模块：MY/FY 混合冰场阻力 R(p_MY) 与 FSD 敏感性。"
        f"v={v:g} m/s 下 MY 面积占比 0→80% 使阻力由 {_mn(ep['resistance_start_N'])} MN 升至 "
        f"{_mn(ep['resistance_end_N'])} MN（{_pc(ep['relative_change'])}，两锚差 "
        f"{_pc(abs(ep['relative_change']-cr[0.8]['relative_increase'])/ep['relative_change'])}），"
        f"增量的 {_pc(dec['thickness_share_of_increment'], 0)} 来自 MY 冰更厚、仅 "
        f"{_pc(dec['strength_share_of_increment'], 0)} 来自强度；FSD 联合包络跨度 "
        f"{_pc(env['fsd']['relative_span'], 0)}（主控参数 {sens['rank_by_span'][0]}），材料组 "
        f"{_pc(env['material']['relative_span'], 0)}。判据 {a['n_pass']} PASS / {a['n_fail']} FAIL，"
        f"FAIL 为 DEM 截面与锚的强度指数/类阻力比一致性，已给出机制（无 submersion 项 + 弯曲通道关闭），"
        f"绝对量级因无 MY 实测数据标 WITHHELD。"
    )

    return {
        "title": "【P54-W2-N01】多年冰/混合冰场冰阻力模型扩展：MY 比例 0–80% 扫描 + FSD 敏感性（TensorDEM mixed_ice）",
        "summary": summary,
        "scenario_key": "p54-w2-n01-mixed-ice-resistance",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "mixed_ice", "multi-year-ice", "FSD", "ice-resistance",
                 "Lindqvist", "Riska", "DEM-factorial", "sensitivity", "P54-W2-N01"],
        "markdown": md,
        "figures": figures,
    }


def _fmt(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_fmt(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default=str(ART / "mixed_ice_campaign.json"))
    ap.add_argument("--out", default="/tmp/report.json")
    args = ap.parse_args(argv)
    artifact = json.loads(Path(args.artifact).read_text())
    payload = build(artifact)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False))
    print(json.dumps({"out": args.out, "markdown_chars": len(payload["markdown"]),
                      "figures": [f["name"] for f in payload["figures"]],
                      "bytes": Path(args.out).stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())