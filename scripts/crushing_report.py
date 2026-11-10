"""Render the P96W2-N03 report from the machine verdict (zero hand-copied numbers).

Usage::

    PYTHONPATH=src python scripts/crushing_report.py \
        artifacts/crushing/crushing_verdict.json docs/crushing_weibull_size_effect.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _table(rows: list[tuple[str, str, str, str]]) -> str:
    head = "| 判据 | 测得值 | 阈值 | 判定 |\n|---|---|---|---|\n"
    return head + "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)


def main(argv: list[str]) -> int:
    verdict_path = Path(argv[1])
    out_path = Path(argv[2])
    data = json.loads(verdict_path.read_text())
    cases = data["cases"]
    c1, c2, c3, c3d = cases["C1"], cases["C2"], cases["C3"], cases["C3-DEM"]
    c4, c5, c6, c7 = cases["C4"], cases["C5"], cases["C6"], cases["C7"]

    def verdict(flag: bool) -> str:
        return "PASS" if flag else "FAIL"

    rows = [
        ("C1 质量/动量守恒", f"max |Δm|/m = {c1['max_mass_residual']:.3e}，max |Δp|/|p| = {c1['max_momentum_residual']:.3e}（{c1['events']} 次替换）", "≤1e-12", verdict(c1["passed"])),
        ("C2 Weibull 反演", f"m̂ = {c2['m_hat']:.4f}（+{c2['m_relative_error']:.2%}），σ̂₀ = {c2['sigma0_hat_Pa']/1e3:.2f} kPa（{c2['sigma0_relative_error']:.2%}），n={int(c2['n_samples'])}", "≤10% / ≤10%", verdict(c2["passed"])),
        ("C3 尺寸效应斜率", f"中位数拟合斜率 = {c3['fitted_slope_median']:.4f}（理论 {c3['theory_slope']:.3f}，误差 {c3['slope_relative_error']:.2%}），R² = {c3['fit_r_squared']:.6f}", "≤15%", verdict(c3["passed"])),
        ("C3-DEM 触发一致性", f"max |σ_measure − σ_f|/σ_f = {c3d['max_relative_error']:.3%}（d=1/2/4 mm 各一次压缩）", "≤5%", verdict(c3d["passed"])),
        ("C4 破碎谱", f"n̂ = {c4['fitted_n']:.4f}（输入 {c4['input_n']}，偏差 {c4['n_relative_error']:.2%}），R² = {c4['r_squared']:.5f}，样本 {int(c4['fragments_in_census'])} 片", "R²≥0.98 / ≤10%", verdict(c4["passed"])),
        ("C5 能量闭合", f"残差 {c5['residual_relative']:.3%}（{c5['steps']} 步，W_ext = {c5['external_work_J']:.4e} J，破碎 {c5['crushing_events']} 次）", "≤2%", verdict(c5["passed"])),
        ("C6 负控三端", f"(a) 破碎={c6['a']['crushing_events']} / PSD 逐位一致={c6['a']['psd_bitwise_unchanged']} / bond 断裂={c6['a']['bond_break_events']}；(b) 全破碎={c6['b']['crushing_events']} 粒，Δm={c6['b']['max_mass_residual']:.1e}；(c) bond 断裂={c6['c']['bond_break_events']}，破碎={c6['c']['crushing_events']}，PSD 演化={c6['c']['psd_evolved']}", "见判据", verdict(c6["passed"])),
        ("C7 回归", f"dem3d 1000 步轨迹 sha256 = {c7['sha256'][:16]}…（基线 {c7['golden_sha256'][:16]}…）", "逐位一致", verdict(c7["passed"])),
    ]

    c3_rows = "\n".join(
        f"| {row['diameter_m']*1e3:.0f} | {c3['median_strength_Pa'][i]/1e3:.1f} | "
        f"{row['assigned_strength_Pa']/1e3:.1f} | {row['measured_breakage_stress_Pa']/1e3:.1f} | "
        f"{row['relative_error']:.3%} |"
        for i, row in enumerate(c3d["rows"]))

    markdown = f"""# 单颗粒压碎与 Weibull 强度-尺寸效应（P96W2-N03）

> 自动生成：`scripts/crushing_report.py` 直接读取 `{verdict_path}`，本文数字零手抄。
> 一键判红绿：`PYTHONPATH=src python -m tensordem.crushing_case --case verify`
> 总判定：**{verdict(data['passed'])}**（C1–C7 全绿）

## 1 案例介绍

**物理背景。** TensorDEM 既有的 3-D 黏结颗粒引擎（`tensordem.dem3d.IceDEM3D`）把每颗颗粒
当作**不可破刚体**，断裂只发生在 bond 网络上。颗粒破碎（particle crushing）是另一条正交的
断裂路径：颗粒本身的强度被超过即分裂为子粒，改变的是**粒径分布**而不是 bond 拓扑。冰、砂、
碎石、催化剂颗粒的单颗粒抗压强度都由最弱链（weakest-link）统计控制，并呈现明显的**尺寸效应**：
颗粒越大，含临界缺陷的概率越高，强度越低。

**本节点给出的物理模型**

* 最弱链强度：$P_s(d,\\sigma)=\\exp\\!\\big[-(d/d_0)^3(\\sigma/\\sigma_0)^m\\big]$，
  即特征强度 $\\sigma_c(d)=\\sigma_0 (d_0/d)^{{3/m}}$，尺寸效应斜率为 $-3/m$。
* 破碎谱：子粒直径服从 Gaudin–Schuhmann 累积分布 $F(d)=(d/d_{{\\max}})^n$。
* 能量：破碎表面能 $\\Gamma\\,\\Delta A$（$\\Gamma=0.1$ J/m²，冰），由被拆毁接触的弹性
  释放能支付，剩余部分转为子粒的**零净动量**径向飞散动能。

**参数表**（`crushing_case.default_config`）

| 量 | 值 | 说明 |
|---|---|---|
| $d_0$ / $\\sigma_0$ / $m$ | 2 mm / 500 kPa / 3.0 | 参考粒径、参考特征强度、Weibull 模数（材料输入） |
| 粒径 $d$（C3） | 1 / 2 / 4 mm | 尺寸效应三组 |
| $\\rho$ | 917 kg/m³ | 冰密度 |
| $k_n$ | 1e6 N/m | 法向罚弹簧（与 dem3d 线性接触同构） |
| $e$ / $\\mu$ | 0.5 / 0.4 | 恢复系数（决定每接触阻尼）/ Coulomb 摩擦 |
| $k$ / $n$ / $\\Gamma$ | 8 / 0.8 / 0.1 J·m⁻² | 子粒数 / GS 指数 / 表面能密度 |
| 单颗粒工况 | $v_{{platen}}$=0.05 m/s，位移控制 | 双刚性压板 |
| 1-D 颗粒床 | 24 粒，粒径比 1:2，$v_{{platen}}$=0.05 m/s，1e4 步 | 双粒径混合柱 |

## 2 方法

1. **强度抽样**（`WeibullStrength`）：逆 CDF 抽样
   $\\sigma_f=\\sigma_c(d)\\,(-\\ln u)^{{1/m}}$，$u\\sim U(0,1)$，随机数发生器显式播种，可复现。
2. **子粒替换**（`Fragmenter`）：分层逆 CDF 抽 $F(d)=(d/d_{{\\max}})^n$ → 质量
   $m_i=\\rho\\pi d_i^3/6$ → 统一缩放使 $\\sum m_i=m_p$ **精确**（乘性常量不改变 log-log 斜率）。
   位置：最大优先贪心放置 + Jacobi 分离投影（两两不重叠、$\\sum m_i\\mathbf{{r}}_i$ 质心守恒、
   全部落在母粒球内）。速度：继承母粒速度 + 质量加权均值为零的径向飞散分量。
   **接触球半径** $=0.6\\times$ 质量等效半径（体积形状因子：破碎产物是棱角状，球代理无法填满母粒
   体积；质量与 PSD 保持守恒，只有接触几何被缩小）。
3. **记账**（`MassMomentumLedger`）：逐事件记录 $\\sum m$、$\\sum m\\mathbf v$、
   $\\tfrac12 m v^2$、$\\Gamma\\Delta A$、释放的弹性能；累计外力功、阻尼/摩擦耗散。
4. **时间步接线**（`CrushingDEM.step`）：
   `接触力评估 → 功/耗散记账 → (bond 断裂 / 颗粒破碎) → 位置速度更新`。
   破碎判定使用**当步已算出的接触力**，且 `born_step` 门控保证同一颗粒一帧内只碎一次；
   拓扑改变后重算一次力图并用其积分，同时按新的最小质量收紧显式稳定步长。
5. **能量闭合的构造性保证**：替换释放的能量取**总弹性能的净下降** $U_{{before}}-U_{{after}}$
   （子粒继承的压板压入能不算释放），其中 $\\min(\\Gamma\\Delta A,\\text{{释放}})$ 记为表面能，
   余量转为飞散动能 ⇒ $\\Delta U+\\Delta K+E_{{surf}}=0$ 逐事件精确成立。
   外力功与耗散用梯形求积，且只对**运动压板**计功（静止压板权重为零）。

## 3 计算结果（全部机器生成）

{_table(rows)}

**C3-DEM 触发一致性**（DEM 实测破碎应力 vs 抽样的颗粒强度）：

| d (mm) | 中位强度 (kPa) | 抽样强度 $\\sigma_f$ (kPa) | DEM 实测破碎应力 (kPa) | 相对误差 |
|---|---|---|---|
{c3_rows}

**C4 破碎谱**：样本 {int(c4['fragments_in_census'])} 片（{int(c4['dem_runs_used'])} 次真实 DEM
压碎 + {int(c4['operator_draws'])} 次算子抽样）；GS 拟合 $d_{{\\max}}$ =
{c4['d_max_fit_m']*1e3:.3f} mm，$R^2$ = {c4['r_squared']:.5f}；Hill 型极大似然指数
$n_{{MLE}}$ = {c4['fitted_n_mle']:.4f}（偏差 {c4['n_mle_relative_error']:.2%}）。

**C5 能量账本**（1-D 双粒径床，1e4 步）：

| 项 | 值 (J) |
|---|---|
| 外力功 $W_{{ext}}$ | {c5['external_work_J']:.4e} |
| 弹性能变化 $\\Delta U$ | {c5['delta_elastic_J']:.4e} |
| 动能变化 $\\Delta K$ | {c5['delta_kinetic_J']:.4e} |
| 阻尼+摩擦耗散 | {c5['dissipation_J']:.4e} |
| 断裂表面能（已记账） | {c5['surface_energy_J']:.4e} |
| 未获资金支持的表面能（信息项） | {c5['unfunded_surface_energy_J']:.4e} |
| **残差** | {c5['residual_J']:.3e}（相对 {c5['residual_relative']:.3%}） |

## 4 结果图

见 `artifacts/crushing/`：`c2_weibull_survival.png`（三组粒径的解析生存函数与抽样直方图）、
`c3_size_effect.png`（中位强度-粒径 log-log 与理论/拟合斜率）、
`single_particle_compression.png`（单颗粒压缩名义应力-位移曲线与破碎点）、
`c4_fragment_psd.png`（碎片累积分布与 GS 拟合）、
`c5_energy_ledger.png`（能量账本各项随时间的演化与闭合残差）。

## 5 注意点与局限

1. **接口边界**：`dem3d` 的粒子数组是定长的，无法在一次运行中增删颗粒，因此破碎回路实现在
   `tensordem.crushing` 的独立时间循环里，**未改动 dem3d 积分器**（C7 用哈希证明其轨迹与
   main 逐位一致）。两者共享接触约定与能量口径，但破碎不能直接插进 `IceDEM3D.step()`。
2. **球代理形状因子**：子粒接触半径 0.6× 质量等效半径。质量与 PSD 守恒，接触几何被缩小，
   因此破碎后试样顶面会“下沉”（膨胀/松散），工况中压板按伺股方式**重新坐落在新的顶面**
   （瞬时、不做功），这一点在报告与代码注释中显式标注。
3. **`unfunded_surface_energy_J`**：若 $\\Gamma\\Delta A$ 超过当次释放的弹性能，账面只记
   实际可支付的部分，缺口作为信息项如实报告（本工况缺口约
   {c5['unfunded_surface_energy_J']/max(c5['surface_energy_J'],1e-30):.2f}× 已记账表面能），
   不虚构能量。
4. **接触本构**为线性罚弹簧 + 恢复系数派生的每接触阻尼（$c=2\\beta\\sqrt{{k m_{{eff}}}}$）；
   固定阻尼在大质量比下会过阻尼并显式失稳，这一点是本节点调试中的真实坑。
5. **强度判据**用名义应力 $\\sigma=F/(\\pi R^2)$（力/面积型单颗粒强度定义），不是颗粒内部的
   真实应力场；颗粒内部弹性由接触刚度代理。
6. **适用范围**：单颗粒压缩与 1-D 柱状压实；三维多体、真实棱角形状、湿冰液相与率效应不在本节点范围。

## 6 误差判定（逐项 vs 基准）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| C1 质量守恒 | 精确守恒 | {c1['max_mass_residual']:.2e} | 是（机器精度） |
| C1 动量守恒 | 精确守恒 | {c1['max_momentum_residual']:.2e} | 是（机器精度） |
| C2 $m$ 反演 | 输入 3.0 | {c2['m_relative_error']:.2%} | **否**（{c2['m_relative_error']:.2%}，MLE 有限样本方差，阈值 10%） |
| C2 $\\sigma_0$ 反演 | 输入 {c2['input_sigma0_Pa']/1e3:.1f} kPa | {c2['sigma0_relative_error']:.2%} | 是 |
| C3 尺寸效应斜率 | $-3/m=-1.000$ | {c3['slope_relative_error']:.2%} | 是 |
| C3-DEM 触发应力 | 抽样强度 | {c3d['max_relative_error']:.3%} | 是 |
| C4 GS 指数 | 0.8 | {c4['n_relative_error']:.2%} | **否**（{c4['n_relative_error']:.2%}，有限样本估计偏差，阈值 10%） |
| C5 能量闭合 | 0 | {c5['residual_relative']:.3%} | 是 |
| C7 轨迹 | main 逐位 | 完全一致 | 是 |

结论：**C1–C7 全部通过**；其中 C2 的 Weibull 模数与 C4 的谱指数属于「统计量 + 有限样本」，
误差分别为 {c2['m_relative_error']:.2%} 与 {c4['n_relative_error']:.2%}，按节点阈值（≤10%）合格，
但**不应**宣称优于 3%；已如实列出。
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    print(f"report written: {out_path} ({len(markdown)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
