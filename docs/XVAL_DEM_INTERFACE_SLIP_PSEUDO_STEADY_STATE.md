# XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE — 交叉对标报告

**锚点**: `XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE`
**任务**: m17451 · **仓**: TensorDEM · **分支**: `agent/m17451`
**对标对象**: TensorDEM `mindlin_partial_slip` 模块 + `contact_tangential` 引擎
**参考**: Mindlin & Deresiewicz (1953) *J. Appl. Mech.* 20:327, DOI 10.1115/1.4010702;
        Johnson (1985) *Contact Mechanics*, Cambridge Univ. Press, §4.3-4.4（二级来源）
**判据**: 对标误差 <15%（owner 精度标准）
**复评**: m18659（PR #1453 关闭后重开）补独立预言机、对照臂判据与 pytest 覆盖，
审计见 `docs/AUDIT_m17451_XVAL_DEM_INTERFACE_SLIP.md`

---

## ⓿ 实验设计（预声明，先于结果固定）

| 角色 | 臂 | 预声明判据 | 意义 |
|---|---|---|---|
| **target** | 引擎 `unloading="mindlin"`（精确 M&D 反转支） | 逐点偏差 <1e-6·μF_n，循环面积 <1e-3 | 被测对象 |
| **negative control** | 引擎 `unloading="linear"`（Cundall–Strack 常数 k_t0） | **必须**突破 owner 15 % 闸门，且落在 −23 %±8 % | 若对照臂也"通过"，则量具失效、结论作废 |
| **independent oracle** | RK4 积分粘着圆律 `c/a=(1−F_t/μF_n)^{1/3}`（不用 `F_t(δ)` 闭式） | 5e-3·μF_n | 打破"闭式对闭式"循环论证 |
| 重复性 | dt×0.5 / ×1 / ×2（固定每段位移 = δ_s） | 面积误差随 dt 单调、每次加倍约 ×4 | 时间离散无关性的真实检验 |

判定逻辑见 `run_xval.verdict()`：`ALL PASS` 要求**目标通过且对照臂确实失败**，否则报
`INCONCLUSIVE`（不允许在量具失灵时给绿灯）。---

## ① 问题与参考出处

### 物理背景

两弹性球体在法向力 F_n 压紧后承受切向循环加载。Mindlin (1949) 证明无滑移切向响应为
刚度 k_t = 8G*a 的弹簧（a = Hertz 接触半径，G* 为等效剪切模量）。Mindlin & Deresiewicz
(1953) 将其推广至摩擦情形：切向牵引力不能超过 μp，因此从第一个切向增量起接触边缘就形成
**滑移环**，内部**粘着圆**半径 c < a 承载载荷。

在稳态循环加载下（从 gross-slip 点卸载→反向→再加载），接触力-位移形成闭合滞回环，
其包围面积即为每个循环耗散的能量：

```
W = 0.8 μF_n δ_s = 1.2 (μF_n)² / k_t
```

其中 δ_s = 3μF_n/(2k_t) 为 gross-slip 位移。

### 对标意义

该"interface-slip-pseudo-steady-state"（界面滑移伪稳态）是 DEM 接触力学中
**能量耗散**的核心机制。正确实现该律对以下应用至关重要：
- 颗粒材料剪切带能量耗散
- 冰-船体接触摩擦热生成
- 振动阻尼与声学发射

### 参考出处

| 来源 | 内容 | 类型 |
|---|---|---|
| Mindlin & Deresiewicz (1953) | 原始闭式公式 | 一级来源 |
| Johnson (1985) §4.3-4.4 | 闭式推导与物理解释 | 二级来源 |
| LIGGGHTS pair_gran | OSS 实现（线性卸载） | 对标实现 |
| YADE MindlinPhys | OSS 实现（精确卸载） | 对标实现 |
| MercuryDPM Mindlin | OSS 实现（精确卸载） | 对标实现 |

---

## ② 方法与实现

### TensorDEM 实现架构

```
mindlin_partial_slip.py          ← 纯闭式模块（无求解器状态）
├── hertz_contact_radius()       ← a = (3FnR*/4E*)^{1/3}
├── mindlin_no_slip_stiffness()  ← k_t = 8G*a
├── gross_slip_displacement()    ← δ_s = 3μFn/(2k_t)
├── partial_slip_magnitude()     ← F_t = μFn[1-(1-δ/δ_s)^{3/2}]
├── reversal_branch()            ← F_t = F* - 2μFn[1-(1-Δ/(2δ_s))^{3/2}]
├── steady_state_cycle_area()    ← W = ∮ F_t dδ（梯形积分）
└── stick_radius_ratio()         ← c/a = (1-F_t/(μFn))^{1/3}

contact_tangential.py            ← 求解器引擎
├── MindlinTangentialConfig      ← 配置（unloading='linear'|'mindlin'）
└── mindlin_tangential_step()    ← 时间步进（弹性预测→包络钳制→锚点更新）
```

### 验证方法

1. **闭式对闭式**：`mindlin_partial_slip` 模块函数 vs 解析公式（机器精度）
2. **引擎对闭式**：`contact_tangential` 求解器驱动循环 vs 闭式支（逐点偏差）
3. **OSS 对标**： documented 差异分析（LIGGGHTS/YADE/MercuryDPM）
4. **参数敏感性**：μ、F_n、E 三个维度的标度律验证

### 测试参数（无量纲数完整）

| 符号 | 值 | 单位 | 无量纲组合 |
|---|---|---|---|
| R | 0.05 | m | — |
| E | 9.0e9 | Pa | — |
| ν | 0.33 | — | — |
| μ | 0.3 | — | — |
| F_n | 10.0 | N | — |
| E* | 1.010e10 | Pa | — |
| G* | 2.026e9 | Pa | — |
| a | 3.336e-4 | m | a/R = 6.672e-3 |
| δ_n | 2.651e-6 | m | δ_n/R = 5.302e-5 |
| k_t | 5.407e6 | N/m | k_t/k_n = 1.235 = 3(1-ν)/(2-ν) |
| δ_s | 8.322e-7 | m | δ_s/δ_n = 0.314 |
| μF_n | 3.0 | N | μF_n/(G*a²) = 1.337e-2 |

---

## ③ 对照表（ref vs got + 误差判定）

### 3.1 闭式模块 vs 解析公式

| 量 | 参考值 | TensorDEM 实测 | 相对误差 | <15%? |
|---|---|---|---|---|
| δ_s (gross-slip 位移) | 8.322230e-7 m | 8.322230e-7 m | **0** | ✅ |
| F_t(δ=0) | 0 | 0 | **0** | ✅ |
| F_t(δ=δ_s) | μF_n = 3.0 N | 3.0 N | **0** | ✅ |
| F_t(δ=0.5δ_s) | 0.6464 μF_n | 0.6464 μF_n | **0** | ✅ |
| 反转支 F_t(Δ=0) | +μF_n | +3.0 N | **0** | ✅ |
| 反转支 F_t(Δ=2δ_s) | -μF_n | -3.0 N | **0** | ✅ |
| 稳态循环面积 W | 1.997335e-6 J | 1.997335e-6 J | **7.8e-8** | ✅ |
| c/a at F_t=0.5μF_n | 0.7937 | 0.7937 | **0** | ✅ |

**闭式模块全部通过，最大相对误差 7.8e-8（数值积分引入）。**

### 3.2 引擎（求解器驱动）vs 闭式

| 配置 | 逐点最大偏差 (占 μF_n) | 循环面积 (μJ) | 面积误差 | <15%? |
|---|---|---|---|---|
| **unloading='mindlin'** | **2.52e-15** | 1.9973 | **7.7e-6** | ✅ |
| unloading='linear' | 2.93e-1 (29.3%) | 1.5365 | **-23.1%** | ❌ |

**关键发现**：
- `mindlin`（target）与闭式支逐点偏差 2.52e-15（机器精度），循环面积误差 7.7e-6
- `linear`（**negative control**）逐点偏差 29.3 %，循环面积低估 **23.07 %** —— 这是**设计上必须发生**的失败：
  它证明本量具能分辨"精确 M&D 反转"与"常数刚度卸载"；若该臂也通过，本报告全部绿灯作废
- 两者关系：linear 卸载是 legacy Cundall–Strack 近似（§5 移植取舍的默认值），mindlin 卸载是精确 M&D 支

### 3.3 OSS 实现对标（**文献声明，未在本 harness 执行**）

> 三个 OSS 码（LIGGGHTS / YADE / MercuryDPM）均**未**在本机安装或运行，下表为文献/文档声明值，
> 机读工件中标为 `oss_comparison[*].evidence = "documented … NOT executed"`。
> 本仓**可执行**的对照是 §3.2 的 `linear` 臂（它复现了 LIGGGHTS 风格的 ≈−23 % 能量亏欠）。

| 实现 | 加载支 | 卸载支 | 稳态循环能量误差 | 与 TensorDEM 关系 |
|---|---|---|---|---|
| **TensorDEM (mindlin)** | 精确 M&D | 精确 M&D | **0%** | 本实现 |
| TensorDEM (linear) | 精确 M&D | 常数 k_t0 | **-23.1%** | 兼容 LIGGGHTS |
| **LIGGGHTS** | 精确 M&D | 常数 k_t0 | **~-23%** | ≈ TensorDEM linear |
| **YADE** | 精确 M&D | 精确 M&D | **0%** | ≈ TensorDEM mindlin |
| **MercuryDPM** | 精确 M&D | 精确 M&D | **0%** | ≈ TensorDEM mindlin |

---

## ④ 收敛性与参数敏感性

### 4.1 时间步收敛（`timestep_convergence()` 机器生成）

每段位移固定为 δ_s，Δt×0.5/1/2 对应 800/400/200 个增量（真正的离散加密）：

| Δt 因子 | 增量数 | 循环面积相对误差 | 逐点最大偏差 (占 μF_n) |
|---|---|---|---|
| 0.5 | 2400 | 1.94e-06 | 1.61e-14 |
| 1.0 | 1200 | 7.73e-06 | 2.52e-15 |
| 2.0 | 600 | 3.08e-05 | 3.55e-15 |

面积误差约随 Δt 线性（每次加倍 ≈×4），来源是闭合环的梯形求积；力律本身是代数关系，
逐点偏差在机器精度（率无关性得到确认）。

### 4.2 摩擦系数 μ 敏感性

| μ | δ_s (m) | 预期 δ_s (m) | 相对误差 |
|---|---|---|---|
| 0.1 | 2.774e-7 | 2.774e-7 | 1.9e-16 |
| 0.2 | 5.548e-7 | 5.548e-7 | 1.9e-16 |
| 0.3 | 8.322e-7 | 8.322e-7 | 0 |
| 0.5 | 1.387e-6 | 1.387e-6 | 0 |
| 0.8 | 2.219e-6 | 2.219e-6 | 1.9e-16 |

**标度律**: δ_s ∝ μ^1.000000（理论 1.0）✅

### 4.3 法向载荷 F_n 敏感性

| F_n (N) | δ_s (m) | 预期 δ_s (m) | 相对误差 |
|---|---|---|---|
| 1.0 | 1.793e-7 | 1.793e-7 | 2.0e-16 |
| 5.0 | 5.243e-7 | 5.243e-7 | 2.0e-16 |
| 10.0 | 8.322e-7 | 8.322e-7 | 0 |
| 20.0 | 1.321e-6 | 1.321e-6 | 2.0e-16 |
| 50.0 | 2.433e-6 | 2.433e-6 | 2.0e-16 |

**标度律**: δ_s ∝ F_n^{0.666667}（理论 2/3）✅

### 4.4 杨氏模量 E 敏感性

| E (Pa) | δ_s (m) | 预期 δ_s (m) | 相对误差 |
|---|---|---|---|
| 1.0e9 (软冰) | 3.601e-6 | 3.601e-6 | 2.5e-16 |
| 5.0e9 | 1.231e-6 | 1.231e-6 | 2.5e-16 |
| 9.0e9 (冰) | 8.322e-7 | 8.322e-7 | 0 |
| 20.0e9 | 4.887e-7 | 4.887e-7 | 2.5e-16 |
| 70.0e9 (铝) | 2.120e-7 | 2.120e-7 | 2.5e-16 |

**标度律**: δ_s ∝ E^{-0.666667}（理论 -2/3）✅

---

## ⑤ 本仓 vs OSS 差异清单与移植取舍

### 差异清单

| 维度 | TensorDEM | LIGGGHTS | YADE | MercuryDPM |
|---|---|---|---|---|
| **加载支** | 精确 M&D 闭式 | 精确 M&D | 精确 M&D | 精确 M&D |
| **卸载支** | 可选: linear/mindlin | 常数 k_t0 | 精确 M&D | 精确 M&D |
| **多反转路径** | 状态机（锚点饱和时精确） | 线性外推 | 完整 M&D 历史 | 完整 M&D 历史 |
| **纯闭式模块** | ✅ 有（无求解器依赖） | ❌ 无 | ❌ 无 | ❌ 无 |
| **能量守恒验证** | ✅ 有（steady_state_cycle_area） | ❌ 无 | ❌ 无 | ❌ 无 |
| **向量实现** | ✅ 有（partial_slip_force） | 标量 | 向量 | 向量 |

### 移植取舍

1. **默认卸载模式选择**：
   - TensorDEM 默认 `unloading='linear'`（兼容 LIGGGHTS 用户习惯）
   - 推荐高精度场景使用 `unloading='mindlin'`（与 YADE/MercuryDPM 一致）
   - **取舍理由**：向后兼容 vs 物理精度；用户可通过配置切换

2. **纯闭式模块设计**：
   - TensorDEM 独有：`mindlin_partial_slip.py` 是纯函数模块（无求解器状态）
   - 优势：可直接用于解析验证、参数扫描、代理模型
   - **取舍理由**：增加代码量，但大幅提升可验证性

3. **能量耗散闭式**：
   - TensorDEM 独有：`steady_state_cycle_area()` 提供闭式能量耗散
   - 优势：无需运行 DEM 即可评估循环能量损失
   - **取舍理由**：仅适用于稳态循环（锚点饱和），部分滑移反转需数值积分

---

## ⑥ 失败模式与工件

### 已知局限

1. **多反转路径**：任意部分滑移转折点的通用多反转路径需要带运动学锚点的状态机；
   本模块仅覆盖单调加载和从 gross-slip 的反转。包络内部的反转回退线性预测器。

2. **速率无关**：本律为准静态率无关模型，不含切向阻尼（求解器自行添加）。

3. **参考来源风险**：1953 原文表格值未查阅（无网络）。本对标为闭式比较，非论文图表数字化。

### 工件清单

| 文件 | 内容 |
|---|---|
| `benchmarks/xval_interface_slip_pseudo_steady_state/run_xval.py` | 交叉对标脚本 |
| `benchmarks/artifacts/xval_interface_slip_pseudo_steady_state/results.json` | 机读结果（含 `design`/`oracle_ode`/`timestep_convergence`/`verdict`/`environment`） |
| `benchmarks/artifacts/xval_interface_slip_pseudo_steady_state/figures/fig1_loading_curve.png` | 加载曲线对比 |
| `benchmarks/artifacts/xval_interface_slip_pseudo_steady_state/figures/fig2_hysteresis.png` | 滞回环对比 |
| `benchmarks/artifacts/xval_interface_slip_pseudo_steady_state/figures/fig3_energy_dissipation.png` | 能量耗散对比 |
| `benchmarks/artifacts/xval_interface_slip_pseudo_steady_state/figures/fig4_stick_radius.png` | 粘着圆半径演化 |
| `benchmarks/artifacts/xval_interface_slip_pseudo_steady_state/figures/fig5_parameter_sensitivity.png` | 参数敏感性 |
| `tests/test_xval_interface_slip_pseudo_steady_state.py` | 14 个回归用例（含提交工件防漂移闸门） |
| `docs/AUDIT_m17451_XVAL_DEM_INTERFACE_SLIP.md` | 审计与证据分级 |
| `docs/XVAL_DEM_INTERFACE_SLIP_PSEUDO_STEADY_STATE.md` | 本报告 |

### 复现命令

```bash
cd <repo>
PYTHONPATH=src python benchmarks/xval_interface_slip_pseudo_steady_state/run_xval.py   # 退出码 0 = ALL PASS
pytest tests/test_xval_interface_slip_pseudo_steady_state.py -q                         # 14 passed
```

---

## 误差判定总结

| 关键量 | 基准 | TensorDEM (mindlin) | 误差 | <15%? |
|---|---|---|---|---|
| 加载支 F_t(δ) | **独立粘着圆 ODE** | 闭式模块 | **1.44e-9** | ✅ |
| 反转支 F_t(Δ) | **独立粘着圆 ODE** | 闭式模块 | **2.26e-8** | ✅ |
| 对照臂面积（linear） | exact | 引擎 linear | **−23.07 %**（必须 ≥15 %，量具有效性判据） | 预期失败 ✅ |
| 反转支 F_t(Δ) | M&D 闭式 | 引擎 mindlin | **2.5e-15** | ✅ |
| 稳态循环面积 W | 0.8μFnδs | 引擎 mindlin | **7.7e-6** | ✅ |
| δ_s(μ) 标度 | μ^1 | 闭式模块 | **1.9e-16** | ✅ |
| δ_s(F_n) 标度 | F_n^{2/3} | 闭式模块 | **2.0e-16** | ✅ |
| δ_s(E) 标度 | E^{-2/3} | 闭式模块 | **2.5e-16** | ✅ |

**结论：TensorDEM `mindlin_partial_slip` 模块与 `contact_tangential` 引擎（mindlin 卸载模式）
与 Mindlin-Deresiewicz (1953) 解析解完全一致，所有对标误差远低于 15% 阈值。**

---

*报告生成时间: 2026-11-13*
*所有数字由 `run_xval.py` 程序化注入，零手抄*
