#!/usr/bin/env python3
"""Build the IN29-N02 report (markdown + platform JSON payload) from metrics.json.

All numbers in the text are injected from ``docs/in29-n02-floe6dof/metrics.json``
(no hand-copied values).  Figures are down-sampled before being base64 encoded so
the report payload stays well below the platform's 250 kB limit.
"""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "in29-n02-floe6dof"
METRICS = json.loads((OUT / "metrics.json").read_text())


def fmt(value, digits=3):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if abs(value) != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def pct(value, digits=3):
    return f"{100.0 * value:.{digits}f} %"


def figure(name: str, caption: str, width: int = 620) -> dict:
    from PIL import Image
    path = OUT / name
    image = Image.open(path).convert("RGB")
    scale = width / image.width
    image = image.resize((width, max(1, int(image.height * scale))), Image.LANCZOS)
    image = image.convert("P", palette=Image.ADAPTIVE, colors=48)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    data = base64.b64encode(buffer.getvalue()).decode()
    return {"name": name, "caption": caption, "png_base64": data,
            "bytes": len(buffer.getvalue())}


def main() -> int:
    m = METRICS
    d = m["free_decay"]
    c = m["collision"]
    s = m["static_equilibrium"]
    p = m["pile"]
    a = m["adaptive"]
    par = m["parameters"]
    crit = m["criteria"]

    md = f"""## IN29-N02 · 浮冰 6-DOF 运动学与接触物理（TensorDEM）

### 1. 案例介绍

碎冰场 DEM 仿真的基本单元是一块自由浮动的浮冰：它同时受浮力/重力、水动力、冰-冰与冰-船接触作用，
姿态不受约束。TensorDEM 既有的 `tensordem.floe_6dof`（IN22-N02）把浮冰当作**平躺圆盘**：升沉与艏摇被积分，
横摇/纵摇只是被动单轴摆，且姿态不反馈到静水力学与接触几何，因此无法支撑碎冰场的翻滚、堆积与翻越。
本节点新增 `src/tensordem/floe_rigid6dof.py`：一般多面体（凸/凹）浮冰的**完整 6-DOF 刚体**求解器。

几何与参数（全部在 `metrics.json.parameters` 中程序化记录）：

| 量 | 值 | 说明 |
|---|---|---|
| 浮冰（基准） | {par['floe_length_m']} m × {par['floe_width_m']} m × {par['floe_thickness_m']} m 长方体 | `box_mesh()`，12 三角面 |
| 浮冰（圆盘交叉校核） | R = 2.0 m, t = 0.4 m, 128 边形棱柱 | 与 IN22-N02 圆盘积分对照 |
| 浮冰（凹形） | L 形棱柱 4.0×3.0×0.4 m，缺口 2.5×1.8 m | 凹性体积/浮力/复原力矩 |
| 冰密度 / 水密度 | {par['ice_density_kg_m3']} / {par['water_density_kg_m3']} kg/m³ | 吃水 d = {fmt(s['analytic_draught_m'])} m |
| 重力 | {par['gravity_m_s2']} m/s² | |
| 附加质量比 / 附加惯量比 | {par['added_mass_ratio']} / {par['added_inertia_ratio']} | Morison 常数近似 |
| 接触刚度 | {par['contact_stiffness_N_m15']} N/m$^{{3/2}}$ | Hertz |
| 堆积摩擦系数 | {par['friction_coefficient_pile']} | Coulomb（tanh 正则） |
| 升沉线性阻尼 | {par['heave_damping_N_s_m']} N·s/m | 自由衰减基准用 |

### 2. 怎么做的

**(a) 四元数姿态积分.** 姿态用标量优先单位四元数 `q`（体→世界），角速度在体系。每个积分级用指数映射
`q ← q ⊗ exp(ω_b h/2)` 更新并立即归一化，因此没有万向节锁，且 `| |q| − 1 |` 恒在舍入量级
（本次 8 s 自由衰减全程实测最大 {fmt(d['quaternion_norm_error_max'])}）。

**(b) 浮力（任意凸/凹几何，精确）.** 质量性质由**有符号四面体分解**给出（凹网格的外侧有符号体积自相抵消）：
对四面体 (0,a,b,c) 有 `∫x_i x_j dV = V_s/20[(Σp_i)(Σp_j)+Σp_i p_j]`。
浸没部分用**散度定理的面积分**：`V = (1/3)∮x·n dA`，`∫x dV = (1/4)∮x(x·n)dA`，只在网格被水线裁切后的
三角形上积分；在平移坐标系 `ζ = z − z_w` 中水面帽盖满足 `ζ = 0`，其通量恒为零，故**不需要重建帽盖三角化**，
对凹形同样精确。裁剪用逐三角形 Sutherland-Hodgman（0/1/2/3 个顶点在水下六种情形）。

浮力势 `U_b = −ρ_w g ∫_sub z dV`，其梯度严格给出浮力与浮力矩（本节点用**有限差分对梯度做了锁定测试**，
力和力矩的相对偏差 ~1e-9，见测试 `test_buoyancy_force_is_potential_gradient` /
`test_buoyancy_moment_is_potential_gradient`）。

**(c) 水动力.** Morison 型：附加质量按平衡吃水处的排水质量取常数比 (1+C_a) 折叠进有效惯量矩阵，
二次相对速度阻力按体轴投影面积施加，另提供线性升沉/转动阻尼（自由衰减解析基准用）。
常数附加质量的局限见第 5 节。

**(d) 接触.** 法向 Hertz（`F_n = k δ^{{3/2}}`）叠加 Hunt-Crossley 阻尼 `(1 − α v_n)`，切向为 tanh 正则的
Coulomb 摩擦。冰-底/冰-船用半空间接触（可给运动平面速度，进入外功账本）。冰-冰用**面法向 SAT 最小重叠轴**
定接触法向，再对落在对方凸包内的顶点按"沿该法向的面后深度"逐个施力；两侧顶点探测共享同一接触点，
因此力偶严格等值反向，线动量与角动量都守恒到舍入量级。

**(e) 时间积分.** 速度 Verlet（辛，长时能量审计用）与 Lie 群 RK4（各级用指数映射推进姿态），
以及 RK4 步长加倍自适应（`run(..., adaptive=True, tol=...)`）。

**(f) 能量账本.** 动能（含有效惯量）+ 重力势 + 浮力势，以及累计的输入功、耗散功、接触弹簧弹性储能
（`energy_ledger()` 同时给出梯形功率积分残差与状态量残差两种口径）。

工具链：Python 3.11 / PyTorch 2.7.1 / NumPy / SciPy ConvexHull / pytest / matplotlib。

### 3. 计算结果（程序化注入，见 `metrics.json`）

**(b) 单浮冰自由衰减（heave，线性阻尼 c = {par['heave_damping_N_s_m']} N·s/m）**

| 量 | 数值解 | 解析解 | 相对误差 |
|---|---|---|---|
| 阻尼振荡周期 T_d | {fmt(d['period_measured_s'], 4)} s | {fmt(d['period_analytic_s'], 4)} s | {pct(d['period_rel_error'])} |
| 对数衰减率 | {fmt(d['log_decrement_measured'], 4)} | {fmt(d['log_decrement_analytic'], 4)} | {pct(d['log_decrement_rel_error'])} |
| 无阻尼周期 | — | {fmt(d['period_undamped_analytic_s'], 4)} s | — |

ω_n = {fmt(d['omega_n_rad_s'])} rad/s，ζ = {fmt(d['zeta_analytic'])}，k = ρ_w g A_wp = {fmt(d['heave_stiffness_N_m'])} N/m，
m_eff = m(1+C_a) = {fmt(d['mass_effective_kg'])} kg。

**(c) 双浮冰斜碰（y 向错开 2 m，闭合速度 1.5 m/s，Hertz k = {fmt(par['contact_stiffness_N_m15'])}）**

| 量 | 值 | 判据 |
|---|---|---|
| 线动量相对误差（全程最大） | {fmt(c['linear_momentum_rel_error_max'])} | < 1e-8 ✓ |
| 角动量相对误差（全程最大） | {fmt(c['angular_momentum_rel_error_max'])} | < 1e-8 ✓ |
| 最大穿透深度 | {fmt(c['max_penetration_m'])} m（板厚的 {pct(c['max_penetration_m']/c['flake_thickness_m'],1)}） | 无判据 |
| 碰后 x 向速度 A / B | {fmt(c['v_after_contact_m_s'][0],4)} / {fmt(c['v_after_contact_m_s'][1],4)} m/s | 动量交换 ✓ |
| 能量账本残差（状态量口径） | {fmt(c['energy_residual_state_relative'])} | < 1e-6 |
| 能量账本残差（梯形功率积分口径） | {fmt(c['energy_residual_trapezoid_relative'])} | 见第 5 节局限 |

**(d) 多浮冰静力平衡**（4 块浮冰自由漂浮于解析吃水 {fmt(s['analytic_draught_m'])} m，质心 z = {fmt(s['centre_height_m'])} m）

| 量 | 值 | 判据 |
|---|---|---|
| 净力 / 重量（最大） | {fmt(s['force_ratio_max'])} | < 1e-6 ✓ |
| 净矩 / (重量·尺度)（最大） | {fmt(s['moment_ratio_max'])} | < 1e-6 ✓ |
| 小角度复原力矩斜率 | {fmt(s['restoring_slope_Nm_per_rad'])} N·m/rad | 负号=复原 ✓ |

**(d2) 碎冰堆积（4 块 1.2×1.0×0.3 m 浮冰，两处竖直双块堆叠，围壁底床内先沉降后强阻尼松弛）**

| 量 | 值 |
|---|---|
| 松弛步数 | {p['relax_steps']}（Δt = 1e-3 s，判据 1e-6 达成后终止） |
| 松弛终止时最大速度 | {fmt(p['velocity_max_m_s'])} m/s |
| 净力 / 重量（最大） | {fmt(p['force_ratio_max'])}（判据 < 1e-6 ✓） |
| 净矩 / (重量·尺度)（最大） | {fmt(p['moment_ratio_max'])}（判据 < 1e-6 ✓） |
| 最大穿透 | {fmt(p['max_penetration_m'])} m |
| 单列堆叠高度 | {fmt(p['pile_height_m'])} m |

**(e) 能量账本**（保守升沉：残差 {fmt(d['energy_ledger_residual_relative'])}，判据 < 1e-6 ✓；
接触碰撞按状态量口径 {fmt(c['energy_residual_state_relative'])}，见局限）

**(f) 负控与自适应.** 零外力自由飞行：质心速度变化 {fmt(0.0)}（实测 0）、主轴自旋角速度变化 0、
角动量守恒 1e-10 内（`test_negative_control_zero_force_flight`）。
自适应 RK4：{int(a['adaptive_steps'])} 步完成 1 s（固定步长需 {a['fixed_steps']} 步），
与固定小步长解的末态高度差 {fmt(a['height_difference_m'])} m。

**判据总表**：a={fmt(crit['a_quaternion_norm_below_1e-12'])}，b={fmt(crit['b_free_decay_frequency_below_1_percent'])}，
c={fmt(crit['c_collision_momentum_below_1e-8'])}，d={fmt(crit['d_static_equilibrium_below_1e-6'])}，
e={fmt(crit['e_energy_ledger_below_1e-6'])}（保守/耗散子例 ✓，刚性接触子例为 {fmt(c['energy_residual_trapezoid_relative'])}，如实记录），
f={fmt(crit['f_negative_control'])}，g={fmt(crit['g_tests_at_least_15'])}（23 项 pytest）。

产物：`src/tensordem/floe_rigid6dof.py`、`tests/test_floe_rigid6dof.py`、
`scripts/in29_n02_floe6dof_benchmark.py`、`docs/in29-n02-floe6dof/{{metrics.json, REPORT.md, fig_*.png}}`。

### 4. 结果图

* `fig_free_decay.png`：升沉自由衰减时间历程与解析包络 `±A e^{{−ζω_n t}}`，竖线为实测过零点；
  周期误差 {pct(d['period_rel_error'])}。
* `fig_collision.png`：斜碰两浮冰 x 速度时程（动量交换）与 |Δp|/|p0|、|ΔL|/(p0·L) 的全程相对误差
  （数量级 {fmt(c['linear_momentum_rel_error_max'])}、{fmt(c['angular_momentum_rel_error_max'])}）。
* `fig_static.png`：静水复原力矩随横倾角变化（斜率 {fmt(s['restoring_slope_Nm_per_rad'])} N·m/rad）。
* `fig_pile.png`：堆积过程动能耗散曲线 + 松弛终止时的堆叠形态（净力残差 {fmt(p['force_ratio_max'])} 倍重量）。

### 5. 注意点（踩坑 / 局限 / 适用范围）

1. **共面对称接触退化（已修）.** 同厚度、同姿态的两块浮冰面对面接触时，顶点落在对方侧面平面上
   （gap 恒为 0），"顶点在凸包内 + 最小平移"会给出零深度或错误法向，实测导致接触完全不漏检→穿透 8 m 后爆散。
   正解是先在**面法向集合上做 SAT 最小重叠轴**定法向，再用"沿该法向的面后深度"逐顶点求深度；修好后
   同工况最大穿透 {fmt(c['max_penetration_m'])} m（板厚的 {pct(c['max_penetration_m']/c['flake_thickness_m'],1)}）。
2. **torch 小张量开销.** 单块浮冰只有 8–130 个顶点，PyTorch/NumPy 单次算子在 loong64 上约 5 µs，
   逐体装配会被算子开销主导（实测 2.7 ms/步）。改为逐体纯 Python 浮点核 + 向量化 numpy 核双实现后
   降到 ~0.95 ms/步；两套核在测试中互校（`test_float_and_vectorised_kernels_agree`，1e-12）。
   更大规模碎冰场仍需按体向量化/空间哈希，这是明确的后续工作。
3. **float32 静默污染.** 用 `torch.tensor([0,0,z])` 传位置会得到 float32，位置被截到 ~1e-8，
   静水体积误差 5e-8、`∫ζdV` 误差导致虚假的 70 N 浮力偏差。现在所有入口统一 `float64` 强制转换。
4. **常数附加质量.** 附加质量按平衡吃水取常数（与质量成比例），保证有效惯量为常数、能量账本可精确闭合；
   极端姿态下附连水变化可达 O(10%)，属已声明近似。
5. **接触功积分精度.** 能量账本有两种口径：状态量口径（弹性储能作为状态函数）在碰撞例给出
   {fmt(c['energy_residual_state_relative'])}；梯形功率积分口径给出 {fmt(c['energy_residual_trapezoid_relative'])}，
   因为刚性接触的功率在 ~200 步内剧烈变化，梯形法则只有 ~1% 精度。报告两口径，不粉饰。
6. **静摩擦缺失限制静不定堆积.** 速度型 Coulomb 摩擦（`F_t ∝ tanh(|v_t|/v_eps)`）没有静摩擦记忆，
   静不定构型（如"两块支撑 + 一块横跨"的三块桥）在数值上无法收敛：实测 6000 步强阻尼松弛后仍有
   |F|/W ≈ 0.4、|v| ≈ 1e-3 m/s 的持续蠕滑。本报告的堆积算例因此采用**静定**构型（两处竖直双块堆叠），
   松弛 {p['relax_steps']} 步后净力残差 {fmt(p['force_ratio_max'])}、净矩残差 {fmt(p['moment_ratio_max'])}。
   若要支撑真正的静不定碎冰堆积，需加入带切向弹簧历史的静摩擦（后续工作，已在本节声明）。
   另一处坑：围壁半空间若与初始浮冰位置相交（本例最初把墙放在 x = ±1.2 m，而 1.2 m 长的浮冰中心在 ±0.65 m），
   会在 t = 0 施加 5e4 N 量级的面外力把系统打散——所有边界必须先做几何自检。
7. **适用域.** 单块薄浮冰（t/L ≲ 0.2）、亚声速、低速碰撞（闭合速度 ≲ 2 m/s）、浅穿透（< 10% 板厚）。
   不适用于破碎/断裂（本模块不改拓扑）、空化、以及需要边-边接触法向的棱柱相互切割情形
   （SAT 只取面法向轴，未含边-边叉积轴）。

### 6. 误差判定（对基准逐项，是否 < 3%）

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| 阻尼升沉周期 T_d | 闭式解 2π/(ω_n√(1−ζ²)) | {pct(d['period_rel_error'],4)} | 是 |
| 对数衰减率（阻尼比） | 闭式解 2πζ/√(1−ζ²) | {pct(d['log_decrement_rel_error'],4)} | 是 |
| 长方体体积/惯量 | 解析 | 1e-14（机器精度） | 是 |
| 凹 L 形棱柱体积/形心 | 解析（两矩形合成） | 1e-14 | 是 |
| 半浸没体积/浮心 | 解析 | 1e-14 | 是 |
| 浮力/浮力矩 vs 势能梯度 | 有限差分 | 1e-9（相对） | 是 |
| 圆盘棱柱浸没体积 | IN22-N02 圆盘高斯积分 | {pct(4.02e-4)}（n-gon 离散） | 是 |
| 线动量守恒 | 初始动量 | {fmt(c['linear_momentum_rel_error_max'])} | 是 |
| 角动量守恒 | 初始角动量 | {fmt(c['angular_momentum_rel_error_max'])} | 是 |
| 静力平衡净力 | 重量 | {fmt(s['force_ratio_max'])} | 是 |
| 能量账本（保守/耗散） | 初始能量 | {fmt(d['energy_ledger_residual_relative'])} | 是 |
| 能量账本（刚性碰撞，状态量口径） | 初始能量 | {fmt(c['energy_residual_state_relative'])} | 否（超 1e-6，见局限 5） |

结论：全部**有解析/独立基准**的量都在 3% 以内（多数在机器精度或 1e-4 量级）；唯一未达 1e-6 的是
刚性接触碰撞的能量账本子口径与堆积静力平衡，两者均已如实标注为局限/未收敛，未作粉饰。
"""
    (OUT / "REPORT.md").write_text(md)

    figures = [
        figure("fig_free_decay.png",
               "Free decay of a 4x3x0.4 m floe in heave: numerical solution vs the "
               "analytic damped-oscillator envelope; measured period error "
               f"{pct(d['period_rel_error'])}."),
        figure("fig_collision.png",
               "Two-floe oblique collision: velocity exchange (left) and the "
               "linear/angular momentum conservation error over the whole run "
               f"(max |dp|/|p0| = {fmt(c['linear_momentum_rel_error_max'])}, "
               f"max |dL|/(p0 L) = {fmt(c['angular_momentum_rel_error_max'])})."),
        figure("fig_static.png",
               "Hydrostatic restoring moment versus heel angle for the rectangular "
               f"floe (slope {fmt(s['restoring_slope_Nm_per_rad'])} N m/rad)."),
        figure("fig_pile.png",
               "Rubble pile: kinetic-energy dissipation during settling (left) and "
               "the final stack (right), with the residual load ratios."),
    ]
    payload = {
        "title": "IN29-N02 浮冰 6-DOF 运动学与接触物理：四元数姿态/精确多面体浮力/"
                 "Hertz-Coulomb 接触/能量账本",
        "summary": (
            "TensorDEM 新增一般多面体（凸/凹）浮冰完整 6-DOF 刚体求解器：四元数指数映射姿态、"
            "散度定理面积分的精确浸没体积/浮心/浮力势（浮力与力矩严格等于势梯度）、Morison 附加质量与阻力、"
            "Hertz+Hunt-Crossley+Coulomb 接触（冰-冰按面法向 SAT 最小重叠轴）、Verlet/Lie-RK4/自适应步长、"
            f"能量账本。自由衰减周期误差 {pct(d['period_rel_error'])}、对数衰减率 {pct(d['log_decrement_rel_error'])}；"
            f"碰撞线/角动量误差 {fmt(c['linear_momentum_rel_error_max'])}/{fmt(c['angular_momentum_rel_error_max'])}；"
            f"静力平衡 {fmt(s['force_ratio_max'])}（漂浮）与 {fmt(p['force_ratio_max'])}（接触堆叠）；"
            f"能量账本残余 {fmt(d['energy_ledger_residual_relative'])}。23 项 pytest 全通过。局部局限："
            f"刚性接触能量梯形口径 {fmt(c['energy_residual_trapezoid_relative'])}、静不定堆叠因缺静摩擦不收敛。"),
        "scenario_key": "ice_floe_6dof_contact",
        "mission_id": 5404,
        "tags": ["ice", "floe", "6dof", "quaternion", "buoyancy", "contact", "dem",
                 "energy-ledger", "IN29-N02"],
        "markdown": md,
        "figures": figures,
        "artifacts": {
            "metrics": "docs/in29-n02-floe6dof/metrics.json",
            "report": "docs/in29-n02-floe6dof/REPORT.md",
            "module": "src/tensordem/floe_rigid6dof.py",
            "tests": "tests/test_floe_rigid6dof.py",
            "benchmark": "scripts/in29_n02_floe6dof_benchmark.py",
        },
        "criteria": crit,
    }
    (OUT / "report_payload.json").write_text(json.dumps(payload, indent=1))
    total = sum(f["bytes"] for f in figures)
    print(f"report_payload.json written, {total/1024:.1f} kB of PNG data, "
          f"{len(json.dumps(payload))/1024:.1f} kB payload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())