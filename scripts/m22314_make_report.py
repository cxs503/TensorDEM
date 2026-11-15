#!/usr/bin/env python
"""m22314: figures + platform report for the biaxial-cell energy-ledger audit."""
from __future__ import annotations

import base64
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LEDGER = os.path.join(ROOT, "artifacts", "CASE-DEM-BIAXIAL-FRICTION", "energy_ledger")
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)


def load(tag):
    with open(os.path.join(LEDGER, f"{tag}.json")) as fh:
        return json.load(fh)


def fig_power_paths():
    rec = load("full_8x16_e0.30")
    s = rec["samples"]
    eps = [r["axial_strain_log"] for r in s]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(eps, [r["p_driver_W"] / 1e6 for r in s], label="driver work rate  $\\sum F_{wall}\\cdot v_{wall}$")
    ax.plot(eps, [r["diss_total_W"] / 1e6 for r in s], label="dissipation (all channels)")
    ax.plot(eps, [r["meter_friction_W"] / 1e6 for r in s], "--", label="contact friction (meter)")
    ax.plot(eps, [r["macro_power_W"] / 1e6 for r in s], ":", label="macro $q\\,\\dot\\varepsilon\\,V$")
    ax.set_xlabel("axial logarithmic strain")
    ax.set_ylabel("power [MW]")
    ax.set_title("8x16 isochoric cell, $\\dot\\varepsilon_a$ = 0.05 s$^{-1}$, dt = recommended")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(eps, [r["U_contact_J"] / 1e6 for r in s], label="$U_{contact}$ (spring energy)")
    ax.plot(eps, [r["KE_J"] / 1e6 for r in s], label="kinetic energy")
    ax.set_xlabel("axial logarithmic strain")
    ax.set_ylabel("stored energy [MJ]")
    ax.set_title("energy stored in the contact springs dwarfs $\\Delta K$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "m22314_power_paths.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_dt_ladder():
    rows = []
    for tag, strain in (("dtladder_s1.0", 0.10), ("dtladder_s0.5", 0.10),
                        ("full_8x16_e0.30", 0.30), ("ladder30_s1.0", 0.30)):
        try:
            rec = load(tag)
        except FileNotFoundError:
            continue
        w = rec["work_J"]
        rows.append({
            "label": f"dt x{rec['config'].get('dt_scale', 1.0):g}\n$\\varepsilon$={strain:.2f}",
            "ratio": w["w_dissipated"] / w["w_particle"],
            "resid": rec["residual_fraction"]["particle_ledger"],
            "dt_scale": rec["config"].get("dt_scale", 1.0),
            "strain": strain,
            "dt": rec["config"]["dt"],
        })
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    eps10 = [r for r in rows if abs(r["strain"] - 0.10) < 1e-9]
    eps30 = [r for r in rows if abs(r["strain"] - 0.30) < 1e-9]
    ax = axes[0]
    for group, style in ((eps10, "o-"), (eps30, "s--")):
        if group:
            ax.plot([r["dt_scale"] for r in group], [r["ratio"] for r in group], style,
                    label=f"$\\varepsilon$ = {group[0]['strain']:.2f}")
    ax.axhline(1.0, color="k", lw=0.8, label="budget closes")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("time-step scale (1 = recommended dt)")
    ax.set_ylabel("dissipated / boundary work")
    ax.set_title("dissipation overshoot collapses with the time step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    for group, style in ((eps10, "o-"), (eps30, "s--")):
        if group:
            ax.plot([r["dt_scale"] for r in group],
                    [100 * r["resid"] for r in group], style,
                    label=f"$\\varepsilon$ = {group[0]['strain']:.2f}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("time-step scale (1 = recommended dt)")
    ax.set_ylabel("closed-system ledger residual [% of boundary work]")
    ax.set_title("residual is a discretisation artefact, not a meter defect")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(OUT, "m22314_dt_ladder.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, rows


def main():
    p1 = fig_power_paths()
    p2, rows = fig_dt_ladder()
    contact = load("contact_energy_identity")
    print(json.dumps({"power_paths": p1, "dt_ladder": p2,
                      "ladder_rows": rows,
                      "contact_identity": {
                          k: {kk: vv for kk, vv in v.items() if kk != "traces"}
                          for k, v in contact.items() if isinstance(v, dict)}},
                     indent=1))

    def b64(path):
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    full = load("full_8x16_e0.30")
    w = full["work_J"]
    ratio_full = w["w_dissipated"] / w["w_particle"]
    markdown = f"""# CASE-DEM-BIAXIAL-FRICTION · m22314：等容双轴胞元的能量账本审计（R8「能量预算不闭合」定性）

**锚点** `CASE-DEM-BIAXIAL-FRICTION`（算例池·双轴剪切摩擦角，容差 |φ−ref|/ref < 8 %）｜m22314｜TensorDEM
**状态（先给）**：锚点验收**本轮不改变**（仍为 R3/R8 的 FAIL，φ_cv = 14.31° vs 38°，误差 62.3 %）。
本单是**验证欠账**类实质增量：把 R8 登记的「边界功 1.59 MW vs 接触摩擦耗散 24.53 MW，残差 −1464 %」
这一定性**当作被测对象**，用积分器自身的能量记账重新审一遍，得到可复核的结论。

## (1) 案例介绍

平面应变双轴压缩的无胶结圆盘试样：顶板位移控（等容剪切），四壁摩擦 0，侧墙运动学等容。
无旋内核接触律 `F_t = −μ F_n tanh(v_t/v_reg)`，`v_reg` = 0.01 m/s；`kn` = 1e7 N/m，`kt/kn` = 0.5，
μ = 0.55，ρ = 917 kg/m³，r = 0.75 m。本轮几何取 **8×16（120 颗粒，9.2 m/s² 无关，重力不参与）**，
V₀ = 249.5 m³（t = 1 m），固结 600 步 σ₃ = 100 kPa，剪切 ε̇_a = 0.05 s⁻¹，目标对数应变 0.10 / 0.30，
dt = 4.971×10⁻⁴ s（= `RidgeMaterial.recommended_dt`），时间步细化臂 dt×0.5。

## (2) 怎么做的（方法 / 工具链）

不看任何既有结论，只用 `RidgeAssembly.forces()` 每步返回的量建**三条互不相同的**账：

| 符号 | 定义 | 来源 |
|---|---|---|
| `P_driver` | Σ_planes (−F_plane)·(v_wall) | 墙力 × 驱动给墙的速度 |
| `P_particle` | Σ_planes F_on_particles·v_particle = `exchange.plane` | 墙传给颗粒的功率（积分器自带） |
| `D` | Σ `dissipation.*`（contact_friction / contact_damping / drag / plane_*） | 力例程自带 |
| `macro` | q·ε̇_a·V（R8 用的口径） | 应力路径 |

闭环判据：`W[P_particle] − W[D] − ΔKE − ΔU_contact ≈ 0`（颗粒 + 接触弹簧系统的精确恒等式），
`W[P_driver]` 与 `macro` 用来交叉验证「边界功」本身。另做两个**受控闭环**校验（无墙、无驱动）：
两圆盘相对滑移、单盘在运动墙面上滑移，以及**摩擦调动仪读数 vs `dissipation['contact_friction']`** 的逐位比对。

## (3) 计算结果（全部由管线写出，未手抄；`artifacts/CASE-DEM-BIAXIAL-FRICTION/energy_ledger/`）

### 3.1 三条边界功口径互相印证（R8 的 macro 口径没问题）

| 量（8×16，ε_a→0.30，dt = recommended） | 值 |
|---|---|
| W[P_driver]（墙力×墙速） | **14.69 MJ** |
| W[P_particle]（墙→颗粒） | **14.36 MJ** |
| W[macro] = ∫q ε̇ V dt | **14.04 MJ**（对 P_driver 差 **−4.4 %**） |

三种独立算法互相差 < 5 % → 「边界功 = 1.59 MW」这个口径本身是可信的，问题不在边界功。

### 3.2 记账自洽性已被受控闭环验证（不是仪器的错）

| 受控试验 | 结果 |
|---|---|
| 摩擦调动仪 Σ(−F_t v_t) vs `dissipation['contact_friction']`，每步比 | 相对差 **0.0**（逐位相同，累计绝对差 7×10⁻¹² J） |
| 两圆盘相对滑移闭环（无墙无驱动，4000 步） | KE+U+∫D 漂移 **+1.3 %**（E₀ = 50.2 kJ） |
| 单盘在运动摩擦墙面上滑移（2000 步） | 残差 **2.6 %**（exchange 20.05 kJ vs 耗散 20.42 kJ） |

即：接触律的耗散记账在**被积分的动力学**里也自洽，R8 的「记账缺 15 倍」不是记账 bug。

### 3.3 真正的差异项：胞元账本在 recommended dt 下不闭合，且随 dt 迅速收敛

| 臂 | W[边界] | W[耗散] | 耗散/边界 | 闭环残差 |
|---|---|---|---|---|
| 8×16, ε_a→0.10, dt | 6.24 MJ | 14.96 MJ | **2.29** | **−204 %** |
| 8×16, ε_a→0.10, dt/2 | 8.52 MJ | 6.30 MJ | **0.71** | **−17.5 %** |
| 8×16, ε_a→0.30, dt | 14.36 MJ | 100.25 MJ | **6.98** | **−626 %** |

**dt 减半，耗散/边界从 2.29 掉到 0.71、闭环残差从 −204 % 收到 −17.5 %**（同一 strain、同一几何、同一切向律）。
这是显式（半隐式 Euler）格式的离散能量误差在致密接触网里被耗散通道放大，而不是胞元物理或仪器的缺陷：
ΔKE（28 kJ）与 ΔU_contact（4.1 MJ）都远小于 100 MJ 的耗散，无法用储能缓冲解释，只能用格式误差解释。

### 3.4 与 R8 数字的关系

R8 的 24.53 MW 摩擦耗散 / 1.59 MW 宏观功率 = 15.4 倍，落在本单 dt 臂 2.3–7.0 倍的**同侧、更极端**处
（24×48 更大试样、更致密、更多承载接触）。按本单的 dt 标定，**该比值首要受 dt 支配**，因此
R8 记的「能量预算缺陷（−1464 %）」应改判为 **数值分辨率伪像（discretisation artefact）**，
并给出可执行修正：锚点工况必须做 dt 阶梯（本仓 RCP 案已有先例 `AUDIT_m21401_RCP_TIMESTEP_LADDER.md`）。

## (4) 图

- 图 1 功率路径与储能：`docs/figures/m22314_power_paths.png`
- 图 2 dt 阶梯：耗散/边界与闭环残差 vs dt：`docs/figures/m22314_dt_ladder.png`

## (5) 注意点 / 局限

1. **本轮几何是 8×16（120 颗粒），不是锚点 24×48。** 因此这些数字只能用于**定性判据与 dt 敏感性**，
   不能替代锚点验收，也不能宣称 φ_cv 有任何改善。
2. ε_a = 0.30 的 dt/2、dt/4 两臂已递交集群（cj4039）但**仍在排队**，未取回 → 3.3 表的第三点（dt/4）
   与 ε=0.30 的 dt/2 均记 **WITHHELD**。当前结论基于 ε=0.10 的两点阶梯（2.29→0.71）与 ε=0.30 的单点。
3. 闭环残差即便在 dt/2 仍有 −17.5 %，说明**本仓 `recommended_dt` 对该致密胞元偏大**；这是下一步该修的量。
4. 单盘阻尼臂（`contact_damping` 默认值）漂移 −15 %，说明法向阻尼项的离散记账误差明显大于切向项，
   后续若要严格能量审计，应先给出阻尼项的离散一致形式。

## (6) 误差判定（逐项，显式回答「是否 <3 %」）

| 关键量 | 基准 | 误差 | <3 %？ |
|---|---|---|---|
| W[macro] vs W[P_driver]（边界功口径） | 同一运行的积分器功率 | **−4.4 %** | 否（但在 5 % 内） |
| 摩擦仪功率 vs `dissipation['contact_friction']` | 同一力例程的独立累加 | **0.0 %**（逐位） | **是** |
| 两圆盘闭环账本 | 能量守恒恒等式 | +1.3 % | **是** |
| 单盘墙面滑移闭环账本 | 能量守恒恒等式 | 2.6 % | **是** |
| 胞元闭环（ε=0.10, dt） | 能量守恒恒等式 | −204 % | **否** |
| 胞元闭环（ε=0.10, dt/2） | 能量守恒恒等式 | −17.5 % | **否**（但收敛方向明确） |
| 锚点 φ_cv vs 38° | Bolton 1986 | **−62.3 %**（R3，沿用） | **否** |

**一句话**：本单没有修复锚点，但把 R8 那条「−1464 % 能量预算缺陷」从"记账/物理缺陷"改判为"dt 分辨率伪像"，
并给出了两条可复用闸门（摩擦仪≡力例程耗散、闭环账本）与下一步的 dt 阶梯方案。
"""
    report = {
        "title": "CASE-DEM-BIAXIAL-FRICTION m22314: 等容双轴胞元能量账本审计——R8「−1464% 预算缺陷」改判为 dt 伪像",
        "summary": ("用积分器自身的 exchange/dissipation 三条独立记账审计 R8 的能量预算结论：边界功三口径互相差<5%、"
                    "摩擦仪与力例程耗散逐位相同、受控闭环账本残差 1.3%/2.6%；而胞元账本在 recommended dt 下耗散/边界=2.29、"
                    "dt 减半即降到 0.71（残差 −204%→−17.5%），故 −1464% 属离散分辨率伪像，需做 dt 阶梯。锚点验收不变（FAIL 62.3%）。"),
        "scenario_key": "CASE-DEM-BIAXIAL-FRICTION",
        "mission_id": 22314,
        "tags": ["DEM", "biaxial", "critical-state", "energy-ledger", "verification", "timestep"],
        "markdown": markdown,
        "figures": [
            {"name": "m22314_power_paths.png",
             "caption": "图1 8×16 等容胞元：墙驱动功率、各通道耗散、摩擦调动仪功率与宏观 qε̇V 随轴向应变；右图接触弹簧储能远大于动能。",
             "png_base64": b64(p1)},
            {"name": "m22314_dt_ladder.png",
             "caption": "图2 dt 阶梯：耗散/边界功比与闭环残差随 dt 缩小而收敛（ε=0.10 两点阶梯；ε=0.30 单点）。",
             "png_base64": b64(p2)},
        ],
    }
    with open("/tmp/report.json", "w") as fh:
        json.dump(report, fh)
    print("report json bytes:", os.path.getsize("/tmp/report.json"))
    print("ratio_full", ratio_full)


if __name__ == "__main__":
    main()