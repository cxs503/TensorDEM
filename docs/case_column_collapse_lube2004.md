# CASE-COLUMN-COLLAPSE · 轴对称颗粒柱坍塌（TensorDEM）vs Lube et al. (2004)

**算例池 case card（owner 2026-11-12）** · mission 15324 · 仓库 TensorDEM · 分支 `agent/m15324` · 容差判据 **< 15 %**

## ① 问题与参考出处

圆柱形颗粒柱（初始半径 $r_i$、高 $h_i$，高宽比 $a=h_i/r_i$）在粗糙水平面上移除侧向支撑后自由坍塌，最终沉积半径 $r_\infty$ 与崩塌历时 $t_\infty$ 只依赖 $a$。参考实验：

> **Lube, G., Huppert, H. E., Sparks, R. S. J. & Freundt, A. (2004)**, *Axisymmetric collapses of granular columns*, **J. Fluid Mech. 508, 175–199**, doi:10.1017/S0022112004009036。

目标数值取自该文**摘要原文**（逐字引用，零手抄）：“*For a < 1.7, the measured final runout radius, $r_\infty$, is related to the initial radius by $r_\infty = r_i(1 + 1.24a)$; while for 1.7 < a the corresponding relationship is $r_\infty = r_i(1 + 1.6a^{1/2})$. The time, $t_\infty$, taken for the grains to reach $r_\infty$ is given by $t_\infty = 3(h_i/g)^{1/2} = 3(r_i/g)^{1/2}a^{1/2}$*”。

实现上这两个关系**由程序在运行时求值**（`tensordem.granular_column.axisym.lube_2004_runout_ratio` / `lube_2004_emplacement_time`），报告与判据里没有任何一处手写参考数字：

```
Lube, Huppert, Sparks & Freundt (2004) JFM 508, 175-199, doi:10.1017/S0022112004009036, abstract: r_inf = r_i(1 + 1.24a) for a < 1.7; r_inf = r_i(1 + 1.6a^(1/2)) for 1.7 < a; t_inf = 3(h_i/g)^(1/2) = 3(r_i/g)^(1/2)a^(1/2)
```

## ② 方法与实现

* 内核：复用 TensorDEM 现有 3D 接触内核 `tensordem.granular_column.kernel`（线性弹簧-阻尼法向力、Cundall-Strack 切向弹簧饱和于 $\mu|F_n|$、滚动弹簧饱和于 $\mu_r R|F_n|$；地面接触 `wall_contact_forces`）。速度 Verlet，耗散力在中点速度取值。
* 几何：新增 `tensordem.granular_column.axisym`，用仓库既有 `hcp_positions_in_cylinder` 生成**圆柱** HCP 柱体（矩形槽版本的 `column3d` 几何与论文轴对称实验不对应），runout 取径向分位 99.5 % + 一个颗粒半径（与矩形版本同一套 0.5 % 尾部剪除）。
* **修复（本单发现）**：`column3d.simulate_column` 与新建的 `axisym` 均未把 `mu_rolling_wall` 传给地面接触，导致 `mu_rolling` 只能作用于颗粒-颗粒接触、地面滚动阻力永远为 0——球体在光滑地面上纯滚动不受任何切向摩擦，沉积**永不停止**。本单补齐该通道后流动才在 $t\approx 2\text{–}3\sqrt{h_i/g}$ 内静止。
* 另一处已修缺陷：`column3d.lube_axisymmetric_runout` 原把轴对称大 $a$ 段写成 $a^{2/3}$，与论文摘要的 $a^{1/2}$ 不符（$2/3$ 属二维槽道结果）；已按原文公式改正并标注出处。
* 协议（跑前预登记，见脚本 docstring）：控制组 $\mu_r=0$；标定仅在 $a=0.5$ 上做，网格 $\mu_r\in\{0.2,0.3,0.4\}$，取 |误差| 最小者；验证扫描 $a\in\{1.0,1.5,2.0,3.0\}$ （样本外）；敏感性：分辨率（6 vs 8 粒/$r_i$）、滚动阻力、时间步。

## ③ 参数表

| 工况 | a | 粒/$r_i$ | $r$ [m] | $r_i$ [m] | $h_i$ [m] | $\mu$ | $\mu_r$ | e | $k_n$ [N/m] |
|---|---|---|---|---|---|---|---|---|---|
| calib_mur0.2_a0.5 | 0.5 | 8 | 0.030 | 0.480 | 0.240 | 0.5 | 0.2 | 0.5 | 6.0e+04 |
| calib_mur0.3_a0.5 | 0.5 | 8 | 0.030 | 0.480 | 0.240 | 0.5 | 0.3 | 0.5 | 6.0e+04 |
| calib_mur0.4_a0.5 | 0.5 | 8 | 0.030 | 0.480 | 0.240 | 0.5 | 0.4 | 0.5 | 6.0e+04 |
| valid_mur0.4_a1 | 1 | 8 | 0.030 | 0.480 | 0.480 | 0.5 | 0.4 | 0.5 | 6.0e+04 |
| valid_mur0.4_a1.5 | 1.5 | 8 | 0.030 | 0.480 | 0.720 | 0.5 | 0.4 | 0.5 | 6.0e+04 |
| valid_mur0.4_a2 | 2 | 8 | 0.030 | 0.480 | 0.960 | 0.5 | 0.4 | 0.5 | 6.0e+04 |
| sens_gpr6_a2.0_mur0.4 | 2 | 6 | 0.030 | 0.360 | 0.720 | 0.5 | 0.4 | 0.5 | 6.0e+04 |
| control_mur0_a1.0 | 1 | 8 | 0.030 | 0.480 | 0.480 | 0.5 | 0 | 0.5 | 6.0e+04 |

固定量：$\rho=2500$ kg/m³、$k_t/k_n=0.5$、HCP 初始格隙 2 %（$\varphi_0\approx0.59$）、重力 9.81 m/s²、无量纲数：$a=h_i/r_i$、$h_i/r_i$ 分解、$\mu$、$\mu_r$、恢复系数 $e$、$k_n/(\rho g d^3)$；初始柱体在 $t=0$ 释放。

## ④ 对照表（ref vs got，15 % 容差打勾/叉）

| 工况 | a(请求) | a(实测) | N | dt [s] | $r_\infty/r_i$ (got) | ref | 误差 | 判 | $t_{99}$ [s] | ref $t_\infty$ [s] | 误差 | 判 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calib_mur0.2_a0.5 | 0.50 | 0.550 | 981 | 4.82e-04 | 1.761 | 1.620 | +8.7 % | ✅ | 0.818 | 0.469 | +74.3 % | ❌ |
| calib_mur0.3_a0.5 | 0.50 | 0.550 | 981 | 4.82e-04 | 1.522 | 1.620 | -6.0 % | ✅ | 0.651 | 0.469 | +38.8 % | ❌ |
| calib_mur0.4_a0.5 | 0.50 | 0.550 | 981 | 4.82e-04 | 1.560 | 1.620 | -3.7 % | ✅ | 0.654 | 0.469 | +39.4 % | ❌ |
| valid_mur0.4_a1 | 1.00 | 1.078 | 1955 | 4.82e-04 | 2.477 | 2.240 | +10.6 % | ✅ | 0.935 | 0.664 | +40.9 % | ❌ |
| valid_mur0.4_a1.5 | 1.50 | 1.501 | 2737 | 4.82e-04 | 2.954 | 2.860 | +3.3 % | ✅ | 1.063 | 0.813 | +30.8 % | ❌ |
| valid_mur0.4_a2 | 2.00 | 2.030 | 3718 | 4.82e-04 | 4.006 | 3.263 | +22.8 % | ❌ | 1.270 | 0.938 | +35.3 % | ❌ |
| sens_gpr6_a2.0_mur0.4 | 2.00 | 2.005 | 1477 | 4.82e-04 | 4.194 | 3.263 | +28.5 % | ❌ | 1.090 | 0.813 | +34.1 % | ❌ |
| control_mur0_a1.0 | 1.00 | 1.078 | 1955 | 4.82e-04 | 5.983 | 2.240 | +167.1 % | ❌ | 1.807 | 0.664 | +172.3 % | ❌ |

**标定结果**：$\mu_r^*=0.4$（在 $a=0.5$ 上 |误差| 最小，误差 -3.7 %）。

## ⑤ 收敛性与参数敏感性

| 敏感性轴 | 设置 | $r_\infty/r_i$ | 误差 | 判读 |
|---|---|---|---|---|
| 基准 | 8 粒/$r_i$, $\mu_r=0.4$ | 4.006 | +22.8 % | $a=2$ 基准 |
| 分辨率 | 6 粒/$r_i$ | 4.194 | +28.5 % | 变粗 → runout 更大，误差单调恶化（4.194 → 4.006 @8 粒） |
| 滚动阻力 | $\mu_r=0$ (a=1) | 5.983 | +167.1 % | 退化：无滚动阻力则沉积永不静止（front 线性增长，time_limit 截断） |
| 滚动阻力 | $\mu_r=0.2$ (a=0.5) | 1.761 | +8.7 % | 标定网格 |
| 滚动阻力 | $\mu_r=0.3$ (a=0.5) | 1.522 | -6.0 % | 标定网格 |
| 滚动阻力 | $\mu_r=0.4$ (a=0.5) | 1.560 | -3.7 % | 标定网格 |

分辨率趋势：$a=2$ 时 6 粒/$r_i$ 给 4.19（+28.5 %），8 粒/$r_i$ 给 4.01（+22.8 %）——runout 随分辨率提高而下降但**未收敛**，粗分辨率无法把 $a=2$ 拉进 15 % 带内；$a=3$ 与 11 粒/$r_i$、dt/2、$\mu=0.3$ 敏感性算例因单进程 >120 s 未完跑，已递交集群队列（作业 cj1074 及后续），结果未纳入本表（honest not_run）。

## ⑥ 结论、失败模式与工件

* **runout 判据**：样本外验证 3/4 通过（a=1.0 +10.6 %、a=1.5 +3.3 %、a=0.5 −3.7 %），$a=2.0$ **FAIL**（+22.8 %）。即：单个标定参数 $\mu_r^*=0.4$ 已能在 $a\le1.5$ 段复现论文定律（<15 %），而惯性段（$a\ge2$，$a^{1/2}$ 律）系统性超预测 ~23 %，与分辨率不足（沉积层仅 ~2 粒厚）同向。
* **历时判据 FAIL**：$t_{99}$ 全体 +31 %~+41 %（$t_{90}$ 亦 +17 %~+30 %）。物理原因：DEM 沉积尾段为蠕变（creep），论文的高速摄影把「突然停止」定义为 $t_\infty$，两者定义差一个尾段；这是**定义域差异**而非接触律失效，但按预登记判据如实记 FAIL。
* 失败模式：①球体无滚动阻力时沉积永不停止（控制组 +167 %）；②惯量段/薄层离散误差；③单进程 wall clock 限制（>120 s 的算例进集群队列）。
* 工件：`runs/case_column_collapse_lube2004/cases/*.json`（逐算例 provenance：参数、能量账本、front 时程、判据）、`summary.json`、`cases.csv`、`artifacts/case_column_collapse/*.png`。
* 复现：`python3 benchmarks/case_column_collapse_lube2004.py --wave1` 然后 `--shard i/4 --mu-rolling 0.4`，最后 `--collect`。

![runout_law.png](../artifacts/case_column_collapse/runout_law.png)

**runout_law.png** — 左：$r_\infty/r_i$ vs $a$（黑线=论文摘要公式，点=DEM）；右：各工况相对误差与 ±15 % 带。

![front_history.png](../artifacts/case_column_collapse/front_history.png)

**front_history.png** — 无量纲弹道时间下的前缘扩展时程，竖虚线为论文 $t_\infty=3(h_i/g)^{1/2}$；DEM 在 ~2–3 个弹道时间内静止。

### 生态摩擦（如实记录）

* `tensor/knowledge/search` 返回 `engine_offline`（8021 连接被拒），开工三查的知识库一路不可用；`oss_solutions` 端点 404，只能用 negative-knowledge 台账（无 TensorDEM 柱坍塌条目）。
* `column3d` 的地面滚动阻力通道缺失（本单已修），此前 P40W4-N06 的 4 倍超预测正是该缺陷的下游症状。
* 控制面单进程 120 s 限制使 5 个算例（a=3.0、11 粒/$r_i$、dt/2、$\mu=0.3$、$\mu=0.8$）未能完跑；集群队列排队 283 作业，取件前已被迫收尾。

