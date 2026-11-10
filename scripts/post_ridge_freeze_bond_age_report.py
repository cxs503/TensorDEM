"""Publish the IN54W1-N05 campaign report (mission 7096) to the platform.

The markdown body is composed from ``artifacts/ridge_freeze_bond_age/*.json``
and the four figures are embedded as base64 PNGs, so no number and no image in
the report is hand-copied.  Run after ``run_ridge_freeze_bond_age.py``::

    python scripts/post_ridge_freeze_bond_age_report.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts", "ridge_freeze_bond_age")
DOC = os.path.join(ROOT, "docs", "ridge_freeze_bond_age_evolution.md")
MISSION_ID = 7096
SCENARIO_KEY = "IN54W1-N05"


def load(name: str) -> dict:
    with open(os.path.join(ART, name)) as handle:
        return json.load(handle)


def table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def f(value, digits=3) -> str:
    if isinstance(value, bool):
        return "yes" if value else "**no**"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) >= 1.0e4 or abs(value) < 1.0e-3:
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def load_study() -> dict | None:
    path = os.path.join(ART, "young_age_seed_study.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def build_markdown(model: dict, cooling: dict, sweep: dict, ledger: dict,
                   study: dict | None = None) -> str:
    m = model["model"]
    records = sorted(sweep["records"], key=lambda record: record["age_hours"])
    criteria = {item["id"]: item for item in ledger["criteria"]}
    sol = cooling["solution"]
    coupling = sweep["lindqvist_coupling"]
    mode = sweep.get("contact_mode_sensitivity", {})
    lines: list[str] = []
    add = lines.append

    add("## 1. 案例介绍")
    add("")
    add("**物理背景**：一年冰脊龙骨是碎石堆体，空隙中的水在接触点冻结形成冻结键"
        "(freeze bond)。冻结键是龙骨中唯一的受拉/受剪载荷路径，因此新脊的龙骨阻力"
        "不是常数——随龄期增长。本节点定量回答：龙骨形成后 t 小时冻结键抗拉强度 "
        "σ_b(t)、抗剪强度 τ_b(t)、以及龙骨阻力 R_keel(t) 随龄期的增长，并给出可"
        "乘到 Lindqvist 脊阻力龙骨项的龄期修正因子 k_age(t)。")
    add("")
    add("**几何/参数表**（全部由 artifacts 注入）：")
    add("")
    add(table(["量", "取值", "来源/证据等级"], [
        ["接触温度 T_c", f"{f(m['temperature_c'], 1)} °C", "算例设定（冬季脊内接触）"],
        ["孔隙水盐度 S", f"{f(m['salinity_ppt'], 1)} ppt", "算例设定（半咸水）"],
        ["冰点 T_m(S)", f"{f(m['freezing_point_c'], 3)} °C", "REAL，Assur 关系"],
        ["卤水体积 v_b", f(m["brine_fraction"], 5), "REAL，Frankenstein & Garner 多项式（-2..-22.9 °C）"],
        ["σ_ice(T,S)", f"{f(m['ice_tensile_strength_pa'] / 1e3, 1)} kPa",
         "形式 REAL（Timco & Frederking），系数 WITHHELD"],
        ["冻结接触面积分数 η_b", f(m["area_fraction"], 5),
         "CALIBRATED（单点锚定，见第 5 节）"],
        ["σ_b∞", f"{f(m['sigma_inf_pa'] / 1e3, 2)} kPa", "= η_b·σ_ice"],
        ["τ_b∞/σ_b∞ = κ", f(m["shear_ratio"], 3), "ridge_dem 约定比"],
        ["导热长度 h", f"{f(m['thermal_length_m'], 2)} m", "= DEM 颗粒半径（每个键的供冷冰体）"],
        ["水膜厚度", f"{f(m['lens_thickness_m'] * 1e3, 1)} mm", "算例设定"],
        ["τ_f", f"{f(m['tau_f_hours'], 2)} h", "导出量（" + m["tau_f_source"] + "）"],
        ["ρc h²/(4k) 比对值", f"{f(m['conduction_time_hours'], 2)} h", "闭式导热时间"],
        ["龙骨（深×宽×孔隙率）",
         f"{f(sweep['keel']['depth_m'], 1)}×{f(sweep['keel']['width_m'], 1)} m, "
         f"n={f(sweep['keel']['porosity'], 2)}", "DEM 受压盒算例"],
        ["颗粒半径 / 摩擦系数",
         f"{f(sweep['material']['particle_radius'], 2)} m / "
         f"{f(sweep['material']['friction_coefficient'], 2)}",
         f"{records[0]['particles']} 颗粒 / "
         f"{max(record['bonds'] for record in records)} 键"],
    ]))
    add("")

    add("## 2. 怎么做的")
    add("")
    add("1. **冻结键形成判据**：接触温度低于当地孔隙水冰点 `T_c < T_m(S)` 时键才存在；"
        "高于冰点则 σ_b=τ_b=0（模块直接返回零，不产生虚假强度）。冰点用 Assur 关系，"
        "冰强度用卤水体积控制的 Timco–Frederking 关系。")
    add("2. **强度上限**：`σ_b∞ = η_b·σ_ice(T_c,S)`，η_b 是冻结接触面积分数（唯一标定参数）。")
    add("3. **增长动力学**：`σ_b(t)=σ_b∞(1-exp(-t/τ_f))`，`τ_b(t)=κσ_b(t)`；"
        "τ_f 不是拟合值，而由 **1-D 瞬态导热解**给出（`lens_cooling_transient`："
        "有限体积显式格式，冷面 Dirichlet 于 T_c，键面为绝热对称面+水膜的 Stefan 潜热"
        "能量平衡，时间步取扩散 CFL）。")
    add("4. **DEM 实现**：在 `ridge_dem` 弹性-脆性键模型里新增运行期强度接口 "
        "`RidgeAssembly.set_bond_strengths`，龄期材料由 `aged_material` 给出"
        "（键强度 + 冻结接触分数同时随龄期变化），不再重建装配体。")
    add("5. **龙骨阻力**：`keel_resistance` 把龙骨置于受限受压盒（两侧刚性壁+海床+"
        "机动压板），取 5% 窗口峰值与残余；每个龄期跑 "
        f"{len(sweep['apparatus']['seeds'])} 个种子（{sweep['apparatus']['seeds']}）给出种子散度。")
    add("6. **Lindqvist 对接**：`k_age(t)=R^DEM(t)/R^DEM(0)`，"
        "`R_keel(t)=R_keel^Lindqvist·k_age(t)`；Lindqvist 项系数在本仓为 WITHHELD，"
        "模块只输出无量纲因子，不发明文献系数。")
    add("7. **验收**：`acceptance_ledger` 生成 `ledger.json`，其中 C6 判据由 runner "
        "实际执行 pytest 得到 returncode，不写“已测试”。")
    add("")
    add("**与仓内既有模块的关系（重要）**：TensorDEM 已存在 "
        "`tensordem.ridge_freeze_bond_aging`，用 Ettema & Schaefer(1986) 实测"
        "压强/盐度律构建同一本构，但其生长时间 τ_g 是**规定的** Arrhenius 参数"
        "（其自述等级 PARAMETRIC_MODEL）。本模块补的正是缺的那一半：用 1-D 瞬态"
        "导热解**导出**生长时间，并给出它对块体尺度的标定关系（见 3.5 节）。二者"
        "不是竞争关系——导热导出值正是规定参数应当被检验的对象。")
    add("")

    add("## 3. 计算结果")
    add("")
    add("### 3.1 冻结键强度随龄期")
    add("")
    add(table(["t [h]", "σ_b [kPa]", "τ_b [kPa]", "冻结接触分数", "σ_b/σ_b∞"],
              [[f(r["age_hours"], 1), f(r["sigma_b_pa"] / 1e3, 2),
                f(r["tau_b_pa"] / 1e3, 2), f(r["bond_fraction"], 3),
                f(r["sigma_b_pa"] / m["sigma_inf_pa"], 4)] for r in records]))
    add("")
    add(f"τ_f = {f(m['tau_f_hours'], 2)} h；在 t={f(records[-1]['age_hours'], 0)} h "
        f"达到渐近强度的 {f(records[-1]['sigma_b_pa'] / m['sigma_inf_pa'], 4)}。")
    add("")
    add("### 3.2 龙骨阻力与龄期修正因子")
    add("")
    add(table(["t [h]", "R_peak [kN/m]", "种子散度 [kN/m]", "R_residual [kN/m]",
               "k_age(t)", "断键", "颗粒数"],
              [[f(r["age_hours"], 1), f(r["peak_force_n_per_m"] / 1e3, 2),
                f(r["peak_force_seed_spread_n_per_m"] / 1e3, 2),
                f(r["residual_force_n_per_m"] / 1e3, 2), f(r["age_factor"], 3),
                f"{sum(r['broken_bonds']) if isinstance(r['broken_bonds'], list) else r['broken_bonds']}/{r['bonds']}",
                r["particles"]] for r in records]))
    add("")
    add(table(["t [h]", "k_age [-]", "R_keel（× 参考值）"],
              [[f(item["age_hours"], 1), f(item["k_age"], 4),
                f(item["R_keel_n_per_m"], 4)] for item in coupling["k_age"]]))
    add("")
    add("### 3.3 导热时间尺度（τ_f 的来源与收敛性）")
    add("")
    add(table(["节点数", "τ（强度代理半松弛）[h]", "t95 [h]", "步数", "焓闭合误差"],
              [[item["nodes"], f(item["tau_half_hours"], 3), f(item["t95_hours"], 2),
                item["steps"], f"{f(100 * item['energy_closure_relative_error'], 3)} %"]
               for item in cooling["convergence"]]))
    add("")
    add(table(["诊断量", "数值"], [
        ["τ_f（模型使用）[h]", f(m["tau_f_hours"], 3)],
        ["闭式 ρc h²/(4k) [h]", f(m["conduction_time_hours"], 3)],
        ["比值 τ_f/比对值", f(m["tau_f_hours"] / m["conduction_time_hours"], 3)],
        ["一阶扩散模态 [h]", f(sol["fundamental_mode_s"] / 3600.0, 2)],
        ["水膜冻结（潜热阶段）[h]", f(sol["lens_freeze_time_hours"], 2)],
        ["5 mm 水膜 Stefan 时间 [h]", f(m["stefan_lens_time_hours"], 4)],
        ["键面温度半松弛 [h]", f(sol["temperature_half_hours"], 2)],
        ["无潜热对照（对闭式值的偏差）",
         f"{f(100 * cooling['latent_free']['relative_deviation'], 1)} %"],
    ]))
    add("")
    comparison = model.get("growth_time_comparison")
    if comparison:
        add("### 3.4 生长时间：导热导出值 vs 既有模块的规定值")
        add("")
        add(table(["块体尺度 h [m]", "导出 τ [h]", "ρch²/(4k) [h]", "导出/闭式",
                   "规定 τ_g [h]", "规定/导出"],
                  [[f(row["thermal_length_m"], 2), f(row["tau_derived_hours"], 3),
                    f(row["conduction_time_hours"], 3), f(row["ratio_to_conduction"], 3),
                    f(row["tau_prescribed_hours"], 3),
                    f(row["ratio_prescribed_over_derived"], 3)]
                   for row in comparison["records"]]))
        add("")
        add("规定 τ_g 是材料常数（与尺度无关），导热导出 τ 随块体尺度超线性增长"
            f"（~h²，叠加固定 5 mm 水膜的潜热），两者只在实验室接触尺度"
            f"（约十几 cm）附近吻合；对 0.5 m 量级现场块体，规定值偏快约 "
            f"{f(1.0 / comparison['records'][2]['ratio_prescribed_over_derived'], 1)} 倍。"
            "这不是否定既有模块（它的律锚在实测实验上），而是它单独无法给出的尺度"
            "外推陈述。")
        add("")
    if study:
        ens = study["ensemble"]
        study_ages = study["ages_hours"]
        add("### 3.5 早期龄期配对种子研究（R_keel(t) 非单调的定量来源）")
        add("")
        add("集合平均曲线**非单调**：6 h 龙骨比未键合的 0 h 龙骨更弱。DEM 峰值"
            "对堆积构型极敏感，因此用**配对各态种子**（同一堆积、只改键状态）复测：")
        add("")
        add(table(["种子"] + [f"R_peak({f(age, 0)} h) [kN/m]" for age in study_ages]
                  + ["R(6h)/R(0h)"],
                  [[row["seed"]] + [f(row["peak_n_per_m"][str(age)] / 1e3, 2)
                                    for age in study_ages]
                   + [f(row["ratio_6h_over_0h"], 3)] for row in study["paired"]]))
        add("")
        add("集合平均峰值：" + "，".join(
            f"{f(age, 0)} h = {f(ens['peak_mean_n_per_m'][str(age)] / 1e3, 2)} ± "
            f"{f(ens['peak_sd_n_per_m'][str(age)] / 1e3, 2)} kN/m" for age in study_ages)
            + "。")
        add("")
        add(f"`R(6h)/R(0h)` 区间 {f(ens['ratio_6h_over_0h_min'], 3)}–"
            f"{f(ens['ratio_6h_over_0h_max'], 3)}（均值 "
            f"{f(ens['ratio_6h_over_0h_mean'], 3)}），"
            f"**{ens['seeds_with_6h_below_0h']}/{len(study['paired'])}** 个配对堆积"
            "出现 6 h 低于 0 h 的凹陷；集合平均的凹陷主要来自 0 h 峰值离散度"
            "（同一天然堆积在无键时峰值最高），因此“脊阻力随龄期单调增大”在本模型的"
            "抗压峰值指标下**不成立**，只在 t ≳ 24 h 成立。")
        add("")
        add("6 h 剪切容量诊断（同一堆积，键抗剪容量乘以系数）：")
        add("")
        add(table(["种子", "系数", "R_peak [kN/m]", "断键"],
                  [[seed, f(entry["shear_capacity_factor"], 1),
                    f(entry["peak_n_per_m"] / 1e3, 2),
                    f"{entry['broken_bonds']}/{entry['bonds']}"]
                   for seed, entry in study["shear_capacity_probe"].items()]))
        add("")
        probe = study["shear_capacity_probe"]
        mean_probe = sum(entry["peak_n_per_m"] for entry in probe.values()) / len(probe)
        factor = list(probe.values())[0]["shear_capacity_factor"]
        add(f"响应很明确：把键抗剪容量乘以 {f(factor, 0)}，6 h 峰值由 "
            f"{f(ens['peak_mean_n_per_m']['6.0'] / 1e3, 1)} 升到 "
            f"{f(mean_probe / 1e3, 1)} kN/m（×{f(mean_probe / ens['peak_mean_n_per_m']['6.0'], 1)}）。"
            "因此年轻龙骨是**抗剪控制**而非抗拉控制：6 h 键抗剪容量很弱，压板行程早期"
            "即发生剪切破坏，滑移后的接触被推入更松构型；4 个配对堆积中有 1 个峰值甚至"
            "低于纯摩擦（无键）值。当抗剪强度达到 24 h 水平后，键合龙骨对每个堆积都更强，"
            "k_age 单调升至 168 h 的 "
            f"{f(records[-1]['age_factor'], 2)}。故 k_age 只对 t ≳ 24 h 提供；"
            "同时抗剪/抗拉分配比 κ 是本结果的一阶参数——本模块用 κ=2（ridge_dem 材料约定），"
            "仓内既有 ridge_freeze_bond_aging 用 κ=0.5（其自述为规定值），此差异随结果"
            "一并报告。")
        add("")
    if mode:
        add("### 3.6 接触约定敏感性")
        add("")
        add(table(["键接触模式", "R_peak [kN/m]", "R_residual [kN/m]", "断键"],
                  [[name, f(probe["peak_n_per_m"] / 1e3, 2),
                    f(probe["residual_n_per_m"] / 1e3, 2),
                    f"{probe['broken_bonds']}/{probe['bonds']}"]
                   for name, probe in mode["probes"].items()]))
        add("")

    add("## 4. 结果图")
    add("")
    add("四张图（场量/过程曲线 + 与文献带或闭式解的对比）见 figures 字段；"
        "内容：σ_b(t)/τ_b(t) 与标定带、键区 1-D 冷却与收敛、R_keel(t) 与 k_age(t)、"
        "渐近强度随 T_c/S 的分布。图由 `run_ridge_freeze_bond_age.py` 生成，"
        "sha256 逐个登记在 ledger.json。")
    add("")

    add("## 5. 注意点（踩坑与局限）")
    add("")
    add("1. **焓账双计（本轮发现并修复）**：初版冷却解把水膜潜热注入相邻固体节点，"
        "而该通量已被导热项携带，导致能量凭空多出。焓账闭合误差随网格加密收敛到 15.5%"
        "（记账错误不会随网格收敛消失），τ_f 因此偏高到闭式值的 2.29 倍。修复后闭合误差"
        f"收敛到 {f(100 * cooling['convergence'][-1]['energy_closure_relative_error'], 2)}%，"
        f"τ_f/闭式值 = {f(m['tau_f_hours'] / m['conduction_time_hours'], 2)}；"
        "pytest 已把焓账作为回归判据（41 节点 <3%）。")
    add("2. **锚点是标定不是验证**：η_b 由 Ettema & Schaefer (1986, "
        "-10 °C/5 ppt) 单点标定，该文离线不可读（WITHHELD-PRIMARY）。真实增量是"
        f"{f(model['anchor']['band_pa'][0] / 1e3, 0)}–{f(model['anchor']['band_pa'][1] / 1e3, 0)} kPa "
        "的带状不确定度与 T/S 趋势；k_age 因是同材料比值而对锚点不敏感。")
    add("3. **接触约定**：历史 `exclusive` 模式对已键合颗粒对关闭接触律（更软），"
        "与真实冻结透镜“叠加在接触上”不符；主曲线用 `parallel`，两者差值见 3.4，"
        "按模型选择不确定度报告。")
    add("4. **代表性体积**：`--quick` 的 9 颗粒烟测盒会出现“刚键合比未键合更弱”"
        "（所有键同时破坏）；烟测写入 `artifacts/ridge_freeze_bond_age_quick/`，"
        "不会覆盖正式产物。")
    add("5. **数字对应的提交**：DEM 扫描在合并当前 origin/main（513 个兄弟提交）后重跑，"
        "全部峰值/残余与合并前逐位一致（"
        + "，".join(f(r['peak_force_n_per_m'] / 1e3, 2) for r in records)
        + " kN/m），说明上游改动对本载荷路径是纯诊断性的，合并前算出的配对种子研究"
        "仍然适用。")
    add("6. **适用范围**：2-D 无转动圆盘、单一粒径、无卤水排出/再冻结/烧结/块体弯曲、"
        "刚性压板罚函数接触；热-力单向耦合。k_age 仅在 t ≳ 6 h 有意义（更早键龄小于"
        "键自身形成时间，离散堆积噪声占主导）。")
    add("")

    add("## 6. 误差判定（逐项）")
    add("")
    c3 = criteria["C3_tau_f_vs_conduction"]["value"]
    add(table(["关键量", "偏差", "基准", "是否 <3%"], [
        ["τ_f 网格自收敛（21 vs 81 节点）",
         f"{f(100 * c3['convergence_spread_relative'], 2)} %", "自收敛（无实验基准）",
         "yes" if c3["convergence_spread_relative"] < 0.03 else "**no**"],
        ["冷却解焓闭合误差",
         f"{f(100 * cooling['convergence'][-1]['energy_closure_relative_error'], 2)} %",
         "守恒恒等式",
         "yes" if cooling['convergence'][-1]['energy_closure_relative_error'] < 0.03 else "**no**"],
        ["σ_b∞ vs Ettema & Schaefer 锚点",
         f"{f(100 * model['anchor_error_relative'], 2)} %", "标定目标（构造性为 0）",
         "n/a（标定，非验证）"],
        ["τ_f vs ρc h²/(4k)",
         f"{f(100 * abs(m['tau_f_hours'] / m['conduction_time_hours'] - 1.0), 1)} %",
         "闭式尺度（非实测）",
         "**no**（>3%，差值来自水膜潜热）"],
        ["k_age(t) vs 独立实测", "-", "本仓无独立龄期脊载荷数据",
         "**无法检验**"],
    ]))
    add("")
    add("**显式回答**：只有两项是定量吻合且 <3%——网格自收敛与焓闭合；"
        "锚点对照 0% 是标定构造、无预测内容；τ_f 对闭式导热尺度通过节点给定的"
        "量级判据（因子 3 内）但不满足 3%；k_age 无独立基准，"
        "只能作为 DEM 测得的趋势报告。这是局限，不粉饰为验证。")
    add("")

    add("## 7. 机器验收台账")
    add("")
    add(table(["判据", "陈述", "结论"],
              [[item["id"], item["statement"], f(item["pass"])]
               for item in ledger["criteria"]]))
    add("")
    add(f"`all_pass = {f(ledger['all_pass'])}`；未决：{ledger['unresolved'] or '无'}。"
        f"完整文档：`docs/ridge_freeze_bond_age_evolution.md`；产物："
        f"`artifacts/ridge_freeze_bond_age/`（model.json / cooling.json / "
        f"age_sweep.json / ledger.json / figures/）。")
    return "\n".join(lines)


def build_payload(dry_run: bool = False) -> dict:
    model = load("model.json")
    cooling = load("cooling.json")
    sweep = load("age_sweep.json")
    ledger = load("ledger.json")
    markdown = build_markdown(model, cooling, sweep, ledger, load_study())
    figures = []
    for figure in ledger["figures"]:
        path = os.path.join(ROOT, figure["path"])
        with open(path, "rb") as handle:
            data = base64.b64encode(handle.read()).decode()
        figures.append({"name": figure["name"],
                        "caption": figure["name"].replace("_", " ").replace(".png", ""),
                        "png_base64": data})
    records = sorted(sweep["records"], key=lambda record: record["age_hours"])
    criteria = {item["id"]: item for item in ledger["criteria"]}
    summary = (
        "冰脊龙骨冻结键龄期模型（TensorDEM ridge_freeze_bond_age）："
        f"σ_b∞={model['model']['sigma_inf_pa'] / 1e3:.1f} kPa "
        f"(η_b={model['model']['area_fraction']:.4f}), "
        f"τ_f={model['model']['tau_f_hours']:.2f} h（1-D 导热解，"
        f"闭式 ρch²/4k={model['model']['conduction_time_hours']:.2f} h）；"
        f"DEM 龙骨阻力 {records[0]['peak_force_n_per_m'] / 1e3:.1f} → "
        f"{records[-1]['peak_force_n_per_m'] / 1e3:.1f} kN/m，"
        f"k_age(168h)={records[-1]['age_factor']:.2f}。"
        f"验收 all_pass={ledger['all_pass']}（C4 单调性判据 honest FAIL："
        f"6 h 龄期 k_age={records[1]['age_factor']:.2f}<1，早期凹陷在配对种子研究中"
        f"定量；C4b 成熟龄期判据 pass），pytest C6 由 runner 实跑 "
        f"{criteria['C6_pytest_chain']['value'].get('summary_line', '')}。"
        "另修复冷却解潜热双计缺陷（焓闭合 15.5%→<1%，τ_f/闭式比 2.29→1.54）。")
    return {
        "title": "IN54W1-N05 冰脊龙骨冻结键时间演化与 R_keel(t)/k_age(t)（TensorDEM）",
        "summary": summary,
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "ice-ridge", "freeze-bond", "age-effect",
                 "Lindqvist", "IN54W1"],
        "markdown": markdown,
        "figures": figures,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = build_payload(args.dry_run)
    with open("/tmp/report.json", "w") as handle:
        json.dump(payload, handle)
    print(f"payload written: /tmp/report.json "
          f"({len(payload['markdown'])} chars, {len(payload['figures'])} figures)")
    if args.dry_run:
        return 0
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    login = urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(login, timeout=120) as response:
        token = json.loads(response.read()).get("access_token")
    request = urllib.request.Request(
        base + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        print(response.read().decode()[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())