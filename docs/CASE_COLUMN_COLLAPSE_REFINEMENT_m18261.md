# CASE-COLUMN-COLLAPSE · m18261 — grid refinement at fixed dimensionless problem + pre-registered arrival definitions

锚点 **CASE-COLUMN-COLLAPSE**（算例池·颗粒柱坍塌，ref Lube, Huppert, Sparks & Freundt 2004 JFM **508**:175-199，doi:10.1017/S0022112004009036，容差 <15 %）。前轮：round 1–5（`docs/case_column_collapse_*`）、只读审计 `docs/AUDIT_COLUMN_COLLAPSE_TIME_LAW_m17534.md`。

本单回答审计留下的两个未决问题：**到达定义未预登记**（审计建议改速度阈值定义，但明写「需重跑」）与 **跑距闭合只在一个分辨率上验证过**（且历史上所谓「12 粒/半径收敛控制例」从未证明是细化）。

## ① 案例介绍

轴对称颗粒柱：半径 $r_i=0.48$ m、高 $h_i=a r_i$ 的 HCP 圆柱在 $t=0$ 释放，重力驱动径向流动后沉积。冻结的「闭合臂」参数（round 5 原样不动）：$\mu=0.5$、$\mu_r=0.5$、$\mu_{\rm wall}=0.8$、$e=0.5$、$k_t/k_n=0.5$、$\rho=2500$ kg/m³、$g=9.81$ m/s²、HCP 格隙 2 %、seed 12345。参考式在运行时由 `tensordem.granular_column.axisym.lube_2004_*` 求值（零手抄）：$r_\infty=r_i(1+1.24a)$（$a<1.7$）、$r_i(1+1.6a^{1/2})$（$a>1.7$）、$t_\infty=3(h_i/g)^{1/2}$。

## ② 怎么做的

* **细化口径（本单核心纠正）**：网格细化必须冻结无量纲问题。本单冻结 $r_i$ 与无量纲接触刚度（静载沉降比 $\delta/d=\pi\rho g d^2/(6k_n)$），因此 `radius = 0.24/gpr`、`kn = 6e4 (d/0.03)^2` —— 见 `benchmarks/case_column_collapse_time_m18261.py::case_radius/case_stiffness`，测试断言三档 $\delta/d$ 相等（相对 1e-12）。

* **到达定义预登记**：新增 `src/tensordem/granular_column/arrival.py`，三族定义全部报告（fraction 0.5/0.75/0.90/0.95/0.99；velocity 阈值 10/5/2/1/0.5 % 峰值前缘速度；length 容差 0.5d/1d/2d），主判据 = fraction 0.90 / velocity 1 % / length 1d，平滑窗口固定 10 ms（按**时间**而非采样点数，故对采样步长不变）。

* **混淆轴分离**：初值格隙 `packing_gap` 0.02 vs 0.0（2 % 格隙给实验没有的自由落体/密实瞬态）与 $\Delta t/2$。

* 14 例预登记算例，本次实跑 12 例（缺失项在 ⑦ 如实申报；集群队列 cj1853–cj1858 排到 313 位后改本地 6 并行）。

## ③ 计算结果

**表 1 — 算例集主判据（$d/r_i=1/16$，即 round 5 的冻结分辨率）**

| a | $a_{\rm meas}$ | $r_\infty/r_i$ | ref | err | 判 | $t_{90}$ err | 判 | $t_{v<1\%}$ err | 判 | $t_{1d}$ err | 判 | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.550 | 1.573 | 1.620 | -2.9 % | ✅ | +27.5 % | ❌ | +67.6 % | ❌ | +11.1 % | ✅ | at_rest |
| 1 | 1.078 | 2.106 | 2.240 | -6.0 % | ✅ | +0.7 % | ✅ | +25.4 % | ❌ | -1.5 % | ✅ | at_rest |
| 1.5 | 1.501 | 2.697 | 2.860 | -5.7 % | ✅ | -0.3 % | ✅ | +32.1 % | ❌ | +4.2 % | ✅ | at_rest |
| 2 | 2.030 | 3.412 | 3.263 | +4.6 % | ✅ | +3.3 % | ✅ | +26.2 % | ❌ | +12.3 % | ✅ | at_rest |
| 3 | 3.087 | 3.984 | 3.771 | +5.6 % | ✅ | -6.6 % | ✅ | +15.6 % | ❌ | +3.5 % | ✅ | at_rest |

**表 2 — 细化轴（$r_i$ 与 $\delta/d$ 冻结，只缩 $d/r_i$）**

| a | gpr | $d/r_i$ | N | dt [s] | $r_\infty/r_i$ | err runout | err $t_{90}$ | err $t_{v<1\%}$ | err $t_{1d}$ |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 8 | 1/16 | 981 | 4.822e-04 | 1.573 | -2.9 % | +27.5 % | +67.6 % | +11.1 % |
| 0.5 | 12 | 1/24 | 3199 | 3.937e-04 | 1.427 | -11.9 % | +1.2 % | +42.7 % | -11.8 % |
| 0.5 | 16 | 1/32 | 8435 | 3.410e-04 | 1.560 | -3.7 % | +9.4 % | +42.5 % | +7.6 % |
| 1 | 8 | 1/16 | 1955 | 4.822e-04 | 2.106 | -6.0 % | +0.7 % | +25.4 % | -1.5 % |
| 1 | 12 | 1/24 | 6412 | 3.937e-04 | 2.079 | -7.2 % | +1.8 % | +31.8 % | +5.4 % |
| 3 | 8 | 1/16 | 5673 | 4.822e-04 | 3.984 | +5.6 % | -6.6 % | +15.6 % | +3.5 % |

**表 3 — 混淆轴（对同 a、同 gpr=8 基准的百分点差）**

| 算例 | 轴 | $\Delta$ runout [pp] | $\Delta t_{90}$ [pp] | $\Delta t_{v<1\%}$ [pp] | $\Delta t_{1d}$ [pp] | stop |
|---|---|---|---|---|---|---|
| G_a0.5_g8_gap0 | G | -3.6 | +30.8 | +22.1 | +27.7 | time_limit |
| G_a1_g8_gap0 | G | -2.0 | +11.6 | +6.5 | +11.3 | at_rest |
| T_a0.5_g8_dt0.5 | T | -6.3 | +2.3 | +1.5 | -17.3 | at_rest |
| T_a1_g8_dt0.5 | T | +3.6 | +5.8 | +14.0 | +7.0 | at_rest |

## ④ 结果图

![runout_law_refined.png](../artifacts/case_column_collapse_time_m18261/runout_law_refined.png)

**runout_law_refined.png** — Runout ratio vs measured aspect ratio: Lube et al. (2004) closed form and the DEM points at 8/12/16 grains per column radius.

![velocity_definition_family.png](../artifacts/case_column_collapse_time_m18261/velocity_definition_family.png)

**velocity_definition_family.png** — Emplacement-time error vs the velocity-threshold definition: the whole family sits above the reference, so it is not a definition artefact.

![front_trajectories.png](../artifacts/case_column_collapse_time_m18261/front_trajectories.png)

**front_trajectories.png** — Dimensionless front trajectories: the shallow cases keep creeping past the reference dimensionless emplacement time.

![refinement_convergence.png](../artifacts/case_column_collapse_time_m18261/refinement_convergence.png)

**refinement_convergence.png** — Errors vs the resolution parameter d/r_i at fixed r_i and fixed dimensionless stiffness; shaded band = ±15 % acceptance.

## ⑤ 注意点

1. **历史「12 粒/半径收敛控制例」不是细化**（H6，从归档 JSON 复核）：w5c_gpr12_a1: r_i 0.473 -> 0.718 m (+51.9 %), grain radius unchanged (0.0300 m) -> NOT a refinement；w5c_gpr12_a3: r_i 0.473 -> 0.718 m (+51.9 %), grain radius unchanged (0.0300 m) -> NOT a refinement。它们把 $r_i$ 从 0.473 m 换成 0.718 m（+51.9 %）、$a_{\rm meas}$ 从 1.078 变 0.988 —— **尺度**变了而非 $d/r_i$ 变细，故不能作为收敛证据；本单的细化轴重做并冻结 $r_i$。

2. **到达定义的族内平移仍主导判决**：velocity 阈值族（10 %→0.5 %）在同一算例上把 $t_\infty$ 拉晚 0.03–0.19 s，fraction 族 90 %→99 % 拉晚 0.12–0.14 s。因此任何时间判决必须连同定义一起报，不能只报一个数。

3. **爬行尾不是「p995 抖动」造成的**：前缘单调性需要容差（`MONOTONE_TOLERANCE=1 %` 的行程）才通得过校验 —— 真实 99.5 % 分位前缘会因边缘颗粒重排而微退，故平滑与容差都由本模块显式登记，而非在分析脚本里临时处理。

4. **单种子、单一初始构型**（HCP + gap，jitter=0），无系综散布；**未做**多分散/形状/$\phi_0$ 敏感性（如实申报）。

5. **算例缺口如实申报**：`R_a1_g16`（$a=1$ 的第三档细化，$N\approx1.6$ 万，超单次预算）, `R_a3_g12`（深端细化） 未完成；故 H2 在 $a=1$ 只有两档、$a=3$ 只有一档，对应裁决记 `incomplete` 而非通过。

## ⑥ 误差判定（逐项，含「是否 <3 %」）

| 关键量 | 基准 | 误差范围 | <15 % 判据 | <3 % ? |
|---|---|---|---|---|
| 跑距 $r_\infty/r_i$（5 例） | Lube 2004 闭式律 | -6.0…+5.6 % | **5/5 通过** | 否 |
| 埋置时间 $t_90$（5 例） | $3\sqrt{h_i/g}$ | -6.6…+27.5 % | **4/5** | 否 |
| 埋置时间 $t_{v<1\%}$（5 例，审计建议定义） | 同上 | +15.6…+67.6 % | **0/5** | 否 |

**显式回答「是否 <3 %」：否。** 跑距最好点误差约 −0.5 %（$a=1.5$，$t_{90}$ $a=1.5$ 亦在 1 % 内），但作为一组判据无一项达 3 %。

## ⑦ 裁决（预登记假设逐条）

| 假设 | 内容 | 裁决 | 证据 |
|---|---|---|---|
| H1 | 速度阈值定义（$v<1\%v_{\rm peak}$）能让 5 个 a 全进 15 % | **falsified** | 表 1 第 4 列：0/5；中位误差 +26.2 % |
| H2 | 跑距闭合在细化下存活（a=0.5/1，误差 ≤15 % 且 8→16 档漂移 ≤5 pp） | **a=0.5: supported；a=1: supported** | a=0.5 最差 +11.9 %，漂移 0.9 pp；a=1 最差 +7.2 % |
| H3 | $a=0.5$ 的时间欠账是模型而非离散化（8→16 档变化 ≤5 pp） | **falsified** | 变化 {"fraction_0.9": 18.1, "velocity_0.01": 25.1, "length_1d": 3.5} pp；最细档误差 {"fraction_0.9": "+9.4", "velocity_0.01": "+42.5", "length_1d": "+7.6"} % |
| H4 | 2 % 初始格隙不是原因（0 vs 0.02 变化 ≤5 pp） | **falsified** | 最差 Δrunout 3.6 pp、Δ$t$ 30.8 pp |
| H5 | $\Delta t/2$ 充分（变化 ≤2 pp） | **falsified** | 最差 Δrunout 6.3 pp、Δ$t$ 17.3 pp |
| H6 | 历史「12 粒/半径」控制例不是细化 | **supported** | $r_i$ 0.473→0.718 m（+51.9 %） |

判读 — 跑距（第一预言）在冻结分辨率上独立复现 5/5，并在 $a=0.5$ 顶部细化档保持；时间（第二预言）在任何预登记定义下都不成立，且**不是**定义造成的（H1 被否证）、也不是离散化造成的（见 H3）。

## 复现

```bash
PYTHONPATH=src python3 benchmarks/case_column_collapse_time_m18261.py --list
PYTHONPATH=src python3 benchmarks/case_column_collapse_time_m18261.py --case R_a0.5_g16
PYTHONPATH=src python3 scripts/analyze_column_collapse_time_m18261.py
PYTHONPATH=src python3 -m pytest tests/test_column_collapse_arrival_m18261.py tests/test_column_collapse_refinement_m18261.py -q
```
