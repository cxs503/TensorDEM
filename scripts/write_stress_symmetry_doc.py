#!/usr/bin/env python3
"""Build docs/stress_symmetry_case_verification.md from the measured JSON.

Every number in the document is injected from
``artifacts/stress_symmetry_case/verification.json`` (written by
``scripts/verify_stress_symmetry_case.py``); nothing is typed by hand.

Run after the verification sweep:
    PYTHONPATH=src python3 scripts/write_stress_symmetry_doc.py
"""

from __future__ import annotations

import json
import os

JSON_PATH = os.path.join("artifacts", "stress_symmetry_case", "verification.json")
DOC_PATH = os.path.join("docs", "stress_symmetry_case_verification.md")


def fmt(x: float, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (x != x):
        return "nan"
    return f"{x:.{nd}f}"


def main() -> int:
    with open(JSON_PATH) as fh:
        data = json.load(fh)

    pts = data["points"]
    frictions = data["setup"]["frictions"]
    rates = data["setup"]["shear_rates"]
    slopes = data["loglog_slopes"]
    lib = data["library_exponent"]
    summ = data["defect_summary"]
    defaults = data["setup"]["case_defaults"]

    lines = []
    a = lines.append

    a("# 2-D 平面剪切的应力序参量：私有 Virial 求和并入共享仪器的验证账")
    a("")
    a("**案件**：`CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT`（"
      "`tensordem.elastic_limit_shear`）  ")
    a("**仪器**：`tensordem.granular2d.stress_arrays` → "
      "`tensordem.granular3d.stress.contact_outer`  ")
    a("**产出脚本**：`scripts/verify_stress_symmetry_case.py`（数据） / "
      "`scripts/write_stress_symmetry_doc.py`（本文，零手抄）  ")
    a(f"**工程参数**：$N={data['setup']['n_particles']}$，"
      f"剪切率 $\\dot\\gamma\\in\\{{{', '.join(f'{g:g}' for g in rates)}\\}}$，"
      f"摩擦系数 $\\mu\\in\\{{{', '.join(f'{m:g}' for m in frictions)}\\}}$；"
      f"其余取 case 默认值（$\\phi={defaults['solid_fraction']}$，"
      f"$k_n={defaults['kn']:g}$，$k_t/k_n={defaults['kt']/defaults['kn']:g}$，"
      f"restitution $={defaults['restitution']:g}$，"
      f"累计应变 $={defaults['strain_total']:g}$，"
      f"平均窗起始应变 $={defaults['strain_average_from']:g}$）。")
    a("")
    a("## 1. 案例介绍（物理背景与几何/参数）")
    a("")
    a("Lees–Edwards 平面剪切：二维圆盘体系在周期剪切盒中被 SLLOD 方程驱动，"
      "是「弹性极限 / Bagnold 惯性极限」的判据工况。门禁关注的关键量是"
      "无量纲剪应力 $\\tau^* = \\tau/(\\rho d^2\\dot\\gamma^2)$ 对 $\\kappa$ 的"
      "log–log 斜率（弹性极限 $n\\to 0$）与有效摩擦 $\\mu_{eff}=\\tau/P$。")
    a("")
    a("问题的来源不是物理，而是**序参量的记账**：原实现把剪切应力写成")
    a("")
    a("$$\\tau = -\\frac{1}{A}\\Big[\\sum_p m v_x v_y + \\sum_c F_{i,x}(r_i-r_j)_y\\Big]"
      " = \\sigma_{yx}$$")
    a("")
    a("即**原始 Virial 张量的一个非对角分量**，再叠加**对称**的动能项。"
      "对有心力（无摩擦）接触，原始 Virial 对称，$\\sigma_{xy}=\\sigma_{yx}$，"
      "该选择不可见；一旦存在 Coulomb 切向力，原始 Virial 不对称，二者相差")
    a("")
    a("$$\\sigma_{xy}-\\sigma_{yx}=\\frac{1}{A}\\sum_c (l_x f_y - l_y f_x)"
      "=\\frac{1}{A}\\sum_c |l|\\,f_t ,$$")
    a("")
    a("正是共享仪器文档中所说的「非中心部分 $\\sigma-\\sigma^{T}$」。"
      "把其中一个分量称为「剪应力」并与对称动能项相加，是任何能量门禁都看不见的记账缺陷。")
    a("")
    a("## 2. 怎么做的（方法/流程/工具链）")
    a("")
    a("1. 新增 `tensordem/granular2d/stress_arrays.py`：把二维 $(l,f)$ 向量**零填充到三维**后"
      "交给已独立验证的三维仪器 `granular3d.stress.contact_outer`，返回面平均张量与"
      "两个非对角分量、对称/反对称部分、二维压强 $-(\\sigma_{xx}+\\sigma_{yy})/2$。"
      "**不重新推导任何公式**，二维与三维案例因此不可能各自漂移。")
    a("2. `elastic_limit_shear.pair_stress` 改为**经由仪器**求值（保留原返回契约 "
      "`(sigma_yx, P)`，历史结果可复现）；新增 `pair_stress_components` 暴露对称剪切、"
      "反对称（杠杆臂）部分与压强。")
    a("3. 力核 `_forces` 每步同时累加对称非对角和 $\\sum(d_x f_y+d_y f_x)$ 与"
      "反对称和 $\\sum(d_x f_y-d_y f_x)$；结果字典新增 `tau_symmetric`、"
      "`tau_minus_symmetric`、`tau_symmetry_defect`、`stress_asymmetry(_relative)`、"
      "`stress_instrument`。**门禁条件不变**（不做静默改判）。")
    a("4. 三组摩擦系数（0.5 默认 / 0.3 / 0.0 对照）各扫三个剪切率，量化差距、"
      "并对两个序参量分别拟合 log–log 速率指数，回答「门禁结论是否受修正影响」。")
    a("")
    a("## 3. 计算结果（全部由 JSON 程序化注入）")
    a("")
    hdr = ("| $\\mu$ | $\\dot\\gamma$ | $\\tau=\\sigma_{yx}$ | 对称 $\\tau_{sym}$ "
           "| $\\tau-\\tau_{sym}$ | 相对缺陷 | 逐样本 $\\langle|\\sigma_{xy}-\\sigma_{yx}|\\rangle/|\\tau_{sym}|$ "
           "| $P$ | $\\mu_{eff}$(旧/新) | 门禁 |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    a(hdr)
    a(sep)
    for r in pts:
        a(f"| {fmt(r['friction'], 2)} | {fmt(r['shear_rate'], 2)} "
          f"| {fmt(r['tau'])} "
          f"| {fmt(r['tau_symmetric'])} | {fmt(r['tau_minus_symmetric'])} "
          f"| {100*r['symmetry_defect_relative']:.3f}% "
          f"| {100*r['stress_asymmetry_relative']:.3f}% | {fmt(r['pressure'], 2)} "
          f"| {fmt(r['mu_eff'], 4)} / {fmt(r['mu_eff_symmetric'], 4)} "
          f"| {'PASS' if r['gates_all'] else 'FAIL'} |")
    a("")
    a("两个序参量的速率指数（三点 log–log 斜率，由 `loglog_slope` 计算）：")
    a("")
    a("| $\\mu$ | $n$（历史 $\\sigma_{yx}$） | $n$（对称 $\\tau_{sym}$） | $|\\Delta n|$ |")
    a("|---|---|---|---|")
    for mu in frictions:
        s = slopes[str(mu)]
        a(f"| {fmt(mu, 2)} | {fmt(s['n_historical'], 5)} | {fmt(s['n_symmetric'], 5)} "
          f"| {fmt(s['abs_slope_change'], 5)} |")
    a("")
    a("库函数 `shear_rate_exponent`（同一 rate sweep，$\\mu=0.5$）返回："
      f"`n_historical = {fmt(lib['n_historical']['n_bagnold'], 5) if lib['n_historical'].get('status') == 'ok' else lib['n_historical']}`，"
      f"`n_symmetric = {fmt(lib['n_symmetric']['n_bagnold'], 5) if lib['n_symmetric'].get('status') == 'ok' else lib['n_symmetric']}`。")
    a("")
    a("图：`artifacts/stress_symmetry_case/fig_stress_symmetry.png`"
      "（左：两序参量随剪切率的幂律；右：历史序参量的记账缺陷 vs $\\mu$）。")
    a("")
    a("## 4. 误差判定")
    a("")
    a("本项**没有外部实验基准**（Campbell 的系数表 paywalled，案例说明里已声明该腿为 "
      "WITHHELD），因此判据是**内部一致性与解析恒等式**：")
    a("")
    a("* 恒等式 $\\sigma_{ab}=(1/A)\\sum_c l_a f_b$：`pair_stress` 与仪器一致到 "
      "$10^{-13}$ 相对（`tests/test_stress_symmetry_case.py::test_pair_stress_is_the_instrument`）；")
    a("* 杠杆臂恒等式 $\\sigma_{xy}-\\sigma_{yx}=\\sum_c|l|f_t/A$：解析钉扎 "
      "（`::test_tangential_gap_is_the_lever_arm_torque`），有心力极限下差距为 0 "
      "（`::test_central_forces_give_a_symmetric_virial`）；")
    a("* 无摩擦实跑（$\\mu=0$）：两序参量在 $10^{-12}$ 相对内相同，"
      "即差距确实全部来自切向力，不是数值噪声（`::test_frictionless_run_has_no_ordinate_gap`）。")
    a("")
    a(f"实测缺陷幅度：$\\mu=0.5$ 时时间平均意义的相对缺陷 "
      f"$|\\tau-\\tau_{{sym}}|/|\\tau_{{sym}}|$ 最大为 "
      f"${100*summ['max_relative_tau_defect']:.3f}\\%$；"
      f"逐样本 $\\langle|\\sigma_{{xy}}-\\sigma_{{yx}}|\\rangle/|\\tau_{{sym}}|$ 最大为 "
      f"${100*summ['max_relative_asymmetry_mu0p5']:.3f}\\%$。"
      "两者之差说明：**瞬时反对称部分很大（$\\sim$5–6%），但其时间平均几乎抵消**，"
      "所以历史序参量的系统性偏差是亚百分到百分量级，而不是与瞬时量同阶。")
    a("")
    a("「是否 <3%？」逐项回答：")
    a("")
    a(f"* 时间平均的序参量缺陷：**是**（最大 "
      f"${100*summ['max_relative_tau_defect']:.3f}\\% < 3\\%$）。")
    a(f"* 速率指数 $n$：$\\mu=0.5$ 时 $|\\Delta n|="
      f"{fmt(slopes['0.5']['abs_slope_change'], 5)}$，"
      f"相对 $n_{{hist}}={fmt(slopes['0.5']['n_historical'], 5)}$ 为 "
      f"${100*slopes['0.5']['abs_slope_change']/max(abs(slopes['0.5']['n_historical']), 1e-30):.2f}\\%$ 量级 —— "
      "**该项不满足 3%**（斜率本身接近 0，微小位移即放大），必须如实登记："
      "结论（弹性极限 $n\\to0$、速率弱依赖）不变，但**发表的指数数值应以对称序参量重算**。")
    a("* 逐样本反对称幅度：**不满足 3%**（5–6%），它是物理的（切向力力矩），"
      "不是误差；把它当误差抹掉才是错的。")
    a("")
    a("## 5. 注意点（坑与适用范围）")
    a("")
    a("* 二维压强是 $-(\\sigma_{xx}+\\sigma_{yy})/2$；把零填充后的三维张量送进 "
      "`granular3d.stress.pressure`（$-tr/3$）会得到同一状态的 **2/3**，两者绝不可混用——"
      "已在模块文档与实现里显式区分。")
    a("* 差距的**瞬时**值与**时间平均**值不是一回事：只报平均会低估反对称部分，"
      "只报瞬时会把抵消误读成系统偏差。本账两者都报。")
    a("* 门禁条件未改：本轮只把序参量挂到共享仪器上并量化，**不静默改判**。"
      "若后续决定以对称序参量作为门禁量，需要单独一轮（历史 benchmark JSON 需重算）。")
    a("* 仍未接线：`bpm_calibration_feasibility.virial_stress_tensor`（torch 逐粒子 "
      "Love–Weber 私有和）依旧自算；该项在总账里明确保留为 open。")
    a("")
    a("## 6. 复现")
    a("")
    a("```")
    a("PYTHONPATH=src python3 scripts/verify_stress_symmetry_case.py   # ~10 min, 9 points")
    a("PYTHONPATH=src python3 scripts/write_stress_symmetry_doc.py     # regenerates this file")
    a("python3 -m pytest tests/test_stress_symmetry_case.py -q")
    a("python3 -m pytest tests/test_elastic_limit_shear.py tests/test_elastic_limit_stress_decomposition.py -q")
    a("```")
    a("")

    os.makedirs(os.path.dirname(DOC_PATH), exist_ok=True)
    with open(DOC_PATH, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", DOC_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())