# CASE-COLUMN-COLLAPSE 第 5 轮（m17111）— 速率相关 mu(I) 与粗糙底板：能否用**一条**接触律闭合 Lube (2004) 全曲线？

锚点：**CASE-COLUMN-COLLAPSE**（判据：[算例池|COLUMN-COLLAPSE｜ref: Lube et al. 2004 实验｜容差: <15 %]）。
前轮：轮 1 `docs/case_column_collapse_lube2004.md`、轮 2 `docs/case_column_collapse_round2_m15996.md`、
轮 3 `docs/case_column_collapse_round3_m16534.md`、轮 4 `docs/case_column_collapse_round4_m16862.md`；
m17025（轮 4 的接棒单）以 `fail_stage=未标 / failure_reason=未留` 结束，**没有产出**，本轮从头做调研再推进。
判据、容差、参考式一字未改：参考值运行时由 `tensordem.granular_column.axisym.lube_2004_*` 求值，报告里每个数字都由
`scripts/analyze_column_collapse_m17111.py` 从逐算例 JSON 注入（零手抄）。

## ① 问题设定与基准出处

轴对称颗粒柱：半径 $r_i$、高 $h_i=a r_i$ 的 HCP 圆柱在 $t=0$ 释放，重力驱动径向流动后沉积。参考实验（逐字引用其摘要，DOI 可查）：

> Lube, Huppert, Sparks & Freundt (2004) JFM **508**, 175-199, doi:10.1017/S0022112004009036：
> "For a < 1.7, the measured final runout radius, r_inf, is related to the initial radius by
> r_inf = r_i(1 + 1.24a); while for 1.7 < a the corresponding relationship is
> r_inf = r_i(1 + 1.6a^(1/2)). The time, t_inf, taken for the grains to reach r_inf is given by
> t_inf = 3(h_i/g)^(1/2) = 3(r_i/g)^(1/2) a^(1/2)."

**轮 4 留下的问题（本轮的出发点）**：在共同滚动阻力下，线性支（$a\le1.5$）要求 $\mu\approx0.51$，平方根支（$a\ge2$）要求
$\mu\approx0.80$（`docs/case_column_collapse_round4_m16862.md` §3.2）——需求随 $a$ 增长。$a$ 增长意味着流动**更惯性**，
而干颗粒摩擦恰恰随惯性数增长：

$$\mu(I)=\mu_s+\frac{\mu_2-\mu_s}{I_0/I+1},\qquad I=\dot\gamma d/\sqrt{P/\rho}$$

（Jop, Forterre & Pouliquen 2006, *Nature* **441**, 727-730, doi:10.1038/nature04801；da Cruz et al. 2005, *PRE* **72**, 011305,
doi:10.1103/PhysRevE.72.011305）。于是本轮两个**先于跑数登记**的假设：

* **H5**：用接触局部 $\mu(I_c)$ 闭合 + Jop 等人 (2006) 参数 $(\mu_s,\mu_2,I_0)=(0.36,0.84,0.30)$（**不针对 runout 拟合任何参数**），
  能否在一个固定滚动阻力下让 $a\in\{0.5,1,1.5,2,3\}$ 全部进入 15 %？
* **H6**：参考实验的底板是**胶砂粗糙底**，而我们的底板是摩擦等于颗粒摩擦的平面墙（等效更滑）。把 $\mu_{wall}$ 提到
  {0.8, 1.0}（固定 $\mu=0.5,\mu_r=0.5$）是否**在平方根支上削弱 runout 的幅度大于线性支**，即底板是不是缺失的"形状"杠杆？

裁决规则（跑数前写死在代码常量里）：H5 = 五个 a 全过 15 % 才算成立；H6 先在 $a\in\{1,3\}$ 两支代表点探针，
只有当 $a=3$ 的改善量大于 $a=1$、且 {0.8,1.0} 中存在一点让**两支代表点同时**进入 15 % 时，才做全 a 确认网格（$a=0.5,1.5,2$ 为样本外）。

## ② 调研与选型（第 2 轮法的"先调研"部分）

**①开源横向（机器可查的负结论）**：生态 OSS 台账 `oss_inventory` 共 33 行，按仓库名逐条核对，
**没有任何 DEM/颗粒流实现**（33 行全是 LBM/SPH/CFD 家族；本地缓存 `~/oss-cache` 只有 `pylbm`、`hschwane`、`Maarten-vd-Sande_lbm` 三个 LBM 仓）。
本例的可迁移实现只能来自文献侧；该空集结论已与既有的 `GAP-DEM-FRICTION / no-dem-repo-in-oss-inventory` 条目一致，并登记进
`oss_solutions (feature_slug='CASE-COLUMN-COLLAPSE')`。

**②方法谱系（选型依据）**：

| 环节 | 文献标准做法 | 本轮采用 |
|---|---|---|
| 参考 runout 律 | Lube et al. 2004 分段律（上式） | 运行时求值，不改 |
| 速率相关摩擦 | $\mu(I)$ 闭合（Jop 2006；da Cruz 2005；Forterre & Pouliquen 2008 综述） | 接触局部 $I_c=|v_t|/\sqrt{f_n/(\rho d^2)}$ + Jop 参数 |
| 底板边界 | 实验底为胶砂（rough base），DEM 需 $\mu_{wall}\ge\mu_{grain}$（Lube et al. 2005 JFM 544:1-14；Lajeunesse et al. 2005 PoF 17:103302） | 平面墙 + 可调 $\mu_{wall}$ |
| 球体滚动抑制 | 滚动阻力 $\mu_r$ 作为角性代理（Utili & Nova 2008, Granular Matter 10:207） | Cundall-Strack 滚动弹簧（既有） |
| 力矩/耗散账 | 分通道耗散记账（碰撞/摩擦/滚动） | 沿用仓内能量账本 |

**③选型沉淀**：见 `oss_solutions`（tag = `round2-method-selection`、`no-dem-repo-in-oss-inventory`、`mu-i-contact-proxy-negative`）。

## ③ 方法与实现

* **接触律**：`src/tensordem/granular_column/inertial_friction.py`（新增）
  `µ(I_c) = µ_s + (µ_2−µ_s)·I_c/(I_0+I_c)`，$I_c=|v_t|/\sqrt{f_n/(\rho d^2)}$，逐个接触求值；
  `pair_contact_forces(..., mu_i=...)` / `wall_contact_forces(..., mu_i=...)`
  （`src/tensordem/granular_column/kernel.py`）把库仑帽从常数 $\mu|F_n|$ 换成 $\mu(I_c)|F_n|$，
  **默认 `mu_i=None` 保持历史行为逐位不变**（测试 `tests/test_column_collapse_inertial_friction.py` 用 $\mu_s=\mu_2$ 退化情形做 bit-for-bit 对照）。
* **底板**：`AxisymCase.mu_wall`（新增，`None` → 沿用颗粒摩擦），墙面 $\mu$ 帽同样可走 $\mu(I_c)$。
* **算例编排**：`benchmarks/case_column_collapse_lube2004.py` 的 `wave5_*` / `wave5b_*` / `wave5c_*`
  （控制臂、$\mu(I)$ 臂、$\mu_r$ 与离散化敏感性、底板探针、全 a 确认、网格/dt 收敛控制）。
* **归约与作图**：`scripts/analyze_column_collapse_m17111.py`（只读逐算例 JSON → 表/图/summary.json），
  `scripts/render_column_collapse_m17111.py`（本文件由它渲染）。

## ④ 参数表（无量纲/量纲齐全）

| case | r [m] | d [m] | N | m [kg] | kn [N/m] | e | dt [s] | dt*omega_0 | h_i/d | phi_0 | v_g/g^(1/2)h_i |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w5_const_mur0.5_a0.5 | 0.03 | 0.06 | 981 | 0.2827 | 6e+04 | 0.5 | 4.822e-04 | 0.222 | 4.00 | 0.608 | 1.000 |
| w5_const_mur0.5_a1.5 | 0.03 | 0.06 | 2737 | 0.2827 | 6e+04 | 0.5 | 4.822e-04 | 0.222 | 12.00 | 0.621 | 1.000 |
| w5_const_mur0.5_a1 | 0.03 | 0.06 | 1955 | 0.2827 | 6e+04 | 0.5 | 4.822e-04 | 0.222 | 8.00 | 0.618 | 1.000 |
| w5_const_mur0.5_a2 | 0.03 | 0.06 | 3718 | 0.2827 | 6e+04 | 0.5 | 4.822e-04 | 0.222 | 16.00 | 0.624 | 1.000 |
| w5_const_mur0.5_a3 | 0.03 | 0.06 | 5673 | 0.2827 | 6e+04 | 0.5 | 4.822e-04 | 0.222 | 24.00 | 0.626 | 1.000 |
| w5_law_gpr12_a1 | 0.03 | 0.06 | 6412 | 0.2827 | 6e+04 | 0.5 | 4.822e-04 | 0.222 | 12.00 | 0.631 | 1.000 |

关键无量纲量：$a=h_i/r_i$；惯性数 $I_c$（接触局部代理，见 ⑤）；$\mu(I)$ 参数 $\mu_s=0.36,\mu_2=0.84,I_0=0.30$；
$\mu\in\{0.5\}$、$\mu_r\in\{0.4,0.5\}$、$\mu_{wall}\in\{0.5,0.8,1.0\}$；$e=0.5$；$k_t/k_n=0.5$；
时间步稳定性 $\Delta t\,\omega_0\le$ 表中值（$\omega_0=\sqrt{k_n/m}$）；初始固含 $\phi_0$ 由上表给出（HCP + gap 0.02）。

## ⑤ 计算结果（逐项对照）

| case | a | mu | mu_r | mu_wall | gpr | runout | ref | err [%] | 判定 | t_inf [s] | ref t_inf | mu_eff | I_c | E_res [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w5_const_mur0.5_a0.5 | 0.5 | 0.5 | 0.5 | - | 8 | 1.550 | 1.620 | -4.3 | ✓ | 0.730 | 0.469 | 0.500 | 0.0000 | +1.56 |
| w5_const_mur0.5_a1 | 1 | 0.5 | 0.5 | - | 8 | 2.229 | 2.240 | -0.5 | ✓ | 0.825 | 0.664 | 0.500 | 0.0000 | +5.95 |
| w5_const_mur0.5_a1.5 | 1.5 | 0.5 | 0.5 | - | 8 | 2.877 | 2.860 | +0.6 | ✓ | 1.046 | 0.813 | 0.500 | 0.0000 | +8.51 |
| w5_const_mur0.5_a2 | 2 | 0.5 | 0.5 | - | 8 | 3.599 | 3.263 | +10.3 | ✓ | 1.183 | 0.938 | 0.500 | 0.0000 | +10.44 |
| w5_const_mur0.5_a3 | 3 | 0.5 | 0.5 | - | 8 | 4.460 | 3.771 | +18.3 | ✗ | 1.385 | 1.149 | 0.500 | 0.0000 | +11.74 |

判定栏：**A 臂 4/5** 通过，最大绝对误差 18.3 %。

| case | a | mu | mu_r | mu_wall | gpr | runout | ref | err [%] | 判定 | t_inf [s] | ref t_inf | mu_eff | I_c | E_res [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w5_law_mur0.5_a0.5 | 0.5 | 0.36 | 0.5 | - | 8 | 1.531 | 1.620 | -5.5 | ✓ | 0.591 | 0.469 | 0.381 | 0.0217 | +1.74 |
| w5_law_mur0.5_a1 | 1 | 0.36 | 0.5 | - | 8 | 2.638 | 2.240 | +17.8 | ✗ | 0.887 | 0.664 | 0.391 | 0.0330 | +6.10 |
| w5_law_mur0.5_a1.5 | 1.5 | 0.36 | 0.5 | - | 8 | 2.982 | 2.860 | +4.3 | ✓ | 0.974 | 0.813 | 0.398 | 0.0414 | +8.71 |
| w5_law_mur0.5_a2 | 2 | 0.36 | 0.5 | - | 8 | 3.497 | 3.263 | +7.2 | ✓ | 1.170 | 0.938 | 0.401 | 0.0460 | +10.75 |
| w5_law_mur0.5_a3 | 3 | 0.36 | 0.5 | - | 8 | 4.193 | 3.771 | +11.2 | ✓ | 1.209 | 1.149 | 0.409 | 0.0598 | +10.73 |

**B 臂 4/5** 通过，最大绝对误差 17.8 %。

| case | a | mu | mu_r | mu_wall | gpr | runout | ref | err [%] | 判定 | t_inf [s] | ref t_inf | mu_eff | I_c | E_res [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w5b_wall0.8_a1 | 1 | 0.5 | 0.5 | 0.8 | 8 | 2.106 | 2.240 | -6.0 | ✓ | 0.784 | 0.664 | 0.532 | 0.0000 | +5.97 |
| w5b_wall1_a1 | 1 | 0.5 | 0.5 | 1 | 8 | 2.286 | 2.240 | +2.1 | ✓ | 0.912 | 0.664 | 0.553 | 0.0000 | +6.13 |
| w5b_wall0.8_a3 | 3 | 0.5 | 0.5 | 0.8 | 8 | 3.984 | 3.771 | +5.6 | ✓ | 1.318 | 1.149 | 0.529 | 0.0000 | +12.33 |
| w5b_wall1_a3 | 3 | 0.5 | 0.5 | 1 | 8 | 4.605 | 3.771 | +22.1 | ✗ | 1.589 | 1.149 | 0.549 | 0.0000 | +12.78 |

| case | a | mu | mu_r | mu_wall | gpr | runout | ref | err [%] | 判定 | t_inf [s] | ref t_inf | mu_eff | I_c | E_res [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w5b_confirm_wall0.8_a0.5 | 0.5 | 0.5 | 0.5 | 0.8 | 8 | 1.573 | 1.620 | -2.9 | ✓ | 0.721 | 0.469 | 0.538 | 0.0000 | +1.78 |
| w5b_confirm_wall0.8_a1.5 | 1.5 | 0.5 | 0.5 | 0.8 | 8 | 2.697 | 2.860 | -5.7 | ✓ | 1.005 | 0.813 | 0.531 | 0.0000 | +8.61 |
| w5b_confirm_wall0.8_a2 | 2 | 0.5 | 0.5 | 0.8 | 8 | 3.412 | 3.263 | +4.6 | ✓ | 1.140 | 0.938 | 0.530 | 0.0000 | +10.73 |

**F 臂（存活的 $(\mu,\mu_r,\mu_{wall})=(0.5,0.5,0.8)$ 全 a 确认）3/3** 通过，
最大绝对误差 5.7 %。

## ⑥ 收敛性与参数敏感性

| case | a | mu | mu_r | mu_wall | gpr | runout | ref | err [%] | 判定 | t_inf [s] | ref t_inf | mu_eff | I_c | E_res [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w5_law_gpr12_a1 | 1 | 0.36 | 0.5 | - | 12 | 2.238 | 2.240 | -0.1 | ✓ | 1.163 | 0.813 | 0.385 | 0.0266 | +5.42 |

| case | a | mu | mu_r | mu_wall | gpr | runout | ref | err [%] | 判定 | t_inf [s] | ref t_inf | mu_eff | I_c | E_res [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w5_law_mur0.4_a1 | 1 | 0.36 | 0.4 | - | 8 | 2.728 | 2.240 | +21.8 | ✗ | 0.984 | 0.664 | 0.386 | 0.0275 | +5.59 |
| w5_law_mur0.4_a3 | 3 | 0.36 | 0.4 | - | 8 | 5.315 | 3.771 | +40.9 | ✗ | 1.535 | 1.149 | 0.404 | 0.0538 | +11.48 |

| case | a | mu | mu_r | mu_wall | gpr | runout | ref | err [%] | 判定 | t_inf [s] | ref t_inf | mu_eff | I_c | E_res [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w5c_dt0.5_a1 | 1 | 0.5 | 0.5 | 0.8 | 8 | 2.187 | 2.240 | -2.4 | ✓ | 0.872 | 0.664 | 0.531 | 0.0000 | +5.05 |
| w5c_gpr12_a1 | 1 | 0.5 | 0.5 | 0.8 | 12 | 2.079 | 2.240 | -7.2 | ✓ | 1.005 | 0.813 | 0.521 | 0.0000 | +5.82 |
| w5c_dt0.5_a3 | 3 | 0.5 | 0.5 | 0.8 | 8 | 4.196 | 3.771 | +11.3 | ✓ | 1.273 | 1.149 | 0.529 | 0.0000 | +9.30 |


* **离散化**：$\mu(I)$ 臂在 $a=1$ 上 8→12 粒/柱半径使 runout 比从 2.638 变到 2.238（约 15 %）——该臂在 $a=1$ **未收敛**，
  这也是它唯一失败点落在 $a=1$ 的原因之一（诚实标注）。
* **时间步**：$\Delta t\to\Delta t/2$ 的 runout 变化见上表 G 臂（a=1 与 a=3）。
* **滚动阻力**：$\mu_r$ 0.5→0.4 的响应见 C 臂（$a=3$ 明显变差）。
* **底板摩擦**：见 ⑤ E 臂——非单调（0.8 好于 1.0），量级与上面各项敏感性同阶。

## ⑦ 失败模式、工件与复现

* **H5 裁决**：不成立（证伪） —— law arm 4/5 within 15 %, control arm 4/5。机理（可复核）：柱体大部分质量处于低速密集接触态，
  实测接触局部惯性数均值仅 I_c ~ 0.022（远小于 I_0 = 0.30），故 mu(I_c) 趋近 mu_s = 0.36 < 0.5：
  **闭合律在大部分接触上把摩擦调低**，在 a = 1 上把 runout 抬高到 +17.8 %；只有在 a = 3 的高速层才把有效摩擦抬到 0.5-0.6。
  即：接触局部滑移速度不是柱坍塌的合适应变率代理（这是可复用的负面知识）。
* **H6 裁决**：方向成立（底板是形状杠杆）。|err| 的改善量（百分点，正 = 变好）：{"1.0": {"mu_wall_0.8": -0.05496947001508136, "mu_wall_1": -0.01584752214762433}, "3.0": {"mu_wall_0.8": 0.1262151309840323, "mu_wall_1": -0.038484539987126615}}。
  在 mu_wall = 0.8 上平方根支改善远大于线性支（形状杠杆成立），但 mu_wall = 1.0 反而变差（非单调），
  说明该杠杆的量级与算例间散布同阶，应作为待标定量而非"默认全粗糙"。
* **能量账**：全部轮 5 算例最大 |能量残差| = 12.78 %（占释放势能，逐例见 ⑤ 表末列）；该残差随 a 增长，
  且**不是本轮引入**：常数摩擦控制臂 a = 3 同阶（11.74 %）。高 a 时拉力截断/释放损失通道
  （`release_loss`）记账不完整是既有 kernel 的欠账，因此本轮的 runout 判定**不宣称能量闭合**。终止原因集合 = ['at_rest']。
* **工件**：`runs/case_column_collapse_lube2004/cases/w5*.json`（逐算例参数+时程+能量账）、
  `artifacts/column_collapse_m17111/summary.json`、`artifacts/column_collapse_m17111/*.png`、`logs/w5/*.log`。

```bash
# 复现（8-12 路并行即可；本单全网格墙钟约 25 min）
PYTHONPATH=src python3 benchmarks/case_column_collapse_lube2004.py --wave5          # A+B 臂
PYTHONPATH=src python3 benchmarks/case_column_collapse_lube2004.py --wave5b         # E 臂探针
PYTHONPATH=src python3 benchmarks/case_column_collapse_lube2004.py --wave5b-confirm 0.8
PYTHONPATH=src python3 benchmarks/case_column_collapse_lube2004.py --wave5c         # 收敛控制
PYTHONPATH=src python3 scripts/analyze_column_collapse_m17111.py                    # 表/图/summary
PYTHONPATH=src python3 scripts/render_column_collapse_m17111.py                     # 本报告
PYTHONPATH=src python3 -m pytest tests/test_column_collapse_inertial_friction.py -q
```

## ⑧ 判读（本轮证明了什么、没证明什么）

* **锚点裁决（本轮主结论）**：三参数构型 $(\mu,\mu_r,\mu_{wall})=(0.5,0.5,0.8)$ 在**全部 5 个长宽比**上通过 15 % 判据：
  **5/5 通过，最大绝对误差 6.0 %**（逐点：a=0.5: -2.9 %，a=1: -6.0 %，a=1.5: -5.7 %，a=2: +4.6 %，a=3: +5.6 %）。
  其中 $a\in\{0.5,1.5,2\}$ 是对探针网格的**样本外确认**（F 臂），$a\in\{1,3\}$ 来自预先登记的探针（E 臂 $\mu_{wall}=0.8$ 列）。
  **诚实限定**：① 通过裕度不宽（最紧点在 $a=1$，6.0 %）；
  ② 12 粒/柱半径下 $a=1$ 为 -7.2 %、$\Delta t/2$ 下 $a=3$ 为 +11.3 %，两者仍在带内但说明**离散化不确定度约 5 个百分点**，
  故"闭合"应读作"在 8 粒/柱半径、$\Delta t_{ref}$ 的离散化下闭合"，不是连续极限下的证明；③ $a=3$ 的 12 粒/柱半径控制**未跑完**（欠账，见下）。
* 本轮**证伪了**"接触局部 $\mu(I_c)$ 是柱坍塌的正确闭合"（H5），并给出机理与可复核数字（$I_c\ll I_0$，$\mu_{eff}\to\mu_s$）。
  这是可复用知识：下一轮应改用**体相惯性数**（由深度压力 $\rho g(H-z)$ 与邻域剪切率估计的 $I$），而不是接触滑移速度。
* 本轮**给出**底板边界是本例的形状杠杆（H6 方向成立，$\mu_{wall}=0.8$ 使 $a=3$ 从 +18.3 % 降到 +5.6 %），但 1.0 非单调，
  故"粗糙底"应作为**待标定量**而非默认全粗糙。
* 未完成/未覆盖（如实申报）：① 只用单一种子/单一初始构型（HCP+gap 0.02，无 jitter 散布），算例间散布未量化；
  ② F 臂在 12 粒/柱半径下的 $a=3$ 收敛控制与 $\Delta t/2$ 控制见 G 臂，未做 $\mu(I)$ 臂的 $a=3$ dt 收敛；
  ③ t_inf 判据（$t_99$ vs $3\sqrt{h_i/g}$）在 ⑤ 表中给出但未作为主判据裁决。
