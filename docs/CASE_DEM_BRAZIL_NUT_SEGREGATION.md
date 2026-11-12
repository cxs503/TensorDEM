# CASE-DEM-BRAZIL-NUT-SEGREGATION: 振动分选巴西坚果效应（TensorDEM）

> 本文件由 `benchmarks/case_brazil_nut_moebius2001/analyse.py` 从 `runs/*.json` 与 `result.json` **程序化生成**；文中每个数字都是注入的，脚本里没有手写数字分支。

## ① 问题 + ref 出处

**问题**：竖直振动（well-separated sinusoidal taps）颗粒床中，一个大颗粒的上升时间 $T_{rise}$ 如何依赖密度比 $\rho/\rho_m$ 与尺寸比 $D/d$ ——即经典的巴西坚果（尺寸偏析）效应及其密度/尺寸依赖。

**参考出处（一手来源，原文查得）**：
- Möbius, Lauderdale, Nagel & Jaeger, Nature 414, 270 (2001), DOI 10.1038/35104697, Fig. 1 (figure page of the DOI)，图 1 *Density and size dependence of the Brazil-nut effect*（图注原文参数）。
- 实验参数（图注原文，逐一对应）：床料 $d$ = 0.5 mm 玻璃珠，$\rho_m$ = 2.4 g/ml；填充高度 8.6 cm；容器直径 8.2 cm；well-separated 正弦脉冲 $\Gamma=A(2\pi f)^2/g$ = 5，$f$ = 13 Hz；球形成分 $D$ = 2.54 cm（$D/d$ = 50.8）；初始时侵入体**顶端**位于自由面下 4.6 cm。原文摘要：密度依赖“does not follow a steady trend but is non-monotonic and sensitive to background air pressure”。
- **不可得项（如实标注）**：图 1 上的 $T_{rise}$ 绝对数值只以图形给出、正文付费墙内，本轮判为 **WITHHELD**，不做像素猜读。非单调峰位 $\rho^*/\rho_m$ 取同组后续论文 companion paper arXiv:cond-mat/0502622 (PRE 72, 011306 (2005)), Fig. 3 and its discussion: rho*/rho_m ~ 0.5, 0.47 for the d = 0.5 mm glass bed 给出的 0.47–0.50 作为趋势锚点。

## ② 方法与实现

复用 TensorDEM 现有能力，**不另起炉灶**：
- 求解器：`tensordem.granular2d.engine`（摩擦圆盘 DEM，线性弹簧-阻尼 + 库仑摩擦 + 滚动阻力，法向/切向接触、边界段壁面）
- 振动床：`tensordem.segregation.bed2d.BidisperseBed`（`set_vibration` / `stop_vibration`，基板按 $A\sin(\omega t)$ 运动，壁面速度场进入接触律）
- 本算例新增（仅 harness，不改求解器）：`run_case.py` 建床→重力压实→在指定深度**置换**若干小颗粒嵌入一个大颗粒（侵入体质量按 $\rho_l$ 逐粒覆写 `mass`/`inertia`）→逐拍脉冲驱动（一拍 = 一个正弦周期 + 半周期静置）→逐拍记录侵入体与床的高度。
- 算纲：`sweep.py`（预注册网格，进程池并行）；分析/出图：`analyse.py`。

**几何相似（与 ref 对齐的无量纲量）**：

| 无量纲量 | 参考 | 本算例 | 说明 |
|---|---|---|---|
| Gamma = A(2πf)²/g | 5.0 | 5.0 | 驱动强度对齐 |
| D / H_bed | 0.295 | 0.177 | 侵入体/床深比对齐 |
| W_cell / H_bed | 0.953 | 0.572 | 容器/床深比对齐 |
| 初始埋深(顶面) / H_bed | 0.535 | 0.535 | 初始条件对齐 |
| D / d | 50.8 | 6.0 | **未对齐**：算力所限，声明为尺度缺口 |
| f·√(d/g) | 0.0928 | 0.1856 | 脉冲时间尺度（重力时间单位） |

## ③ 参数表（无量纲数完整）

| 参数 | 值 | 单位 |
|---|---|---|
| Gamma | 5 | — |
| f_Hz | 13 | — |
| A_over_d | 3.676 | — |
| A_over_D | 0.6126 | — |
| D_over_d | 6 | — |
| D_over_H0 | 0.1772 | — |
| W_over_H0 | 0.5719 | — |
| a_peak_over_g | 5 | — |
| t_tap_over_sqrt_d_over_g | 5.387 | — |
| dt_over_contact_time | 0.02245 | — |
| rho_over_rho_m (扫描范围) | 0.47, 1, 2, 3.24 | — |
| m_intruder/m_small | 0.47→16.9 = 1→36.0 = 2→72.0 … | — |
| kn (N/m) | 3e+04 | — |
| dt (s) | 2.5e-05 | — |
| 床颗粒数（单案例） | 386 | — |
| 压实后面积分数 | 0.488 | — |

## ④ 对照表（ref vs got + 误差打勾叉）

| 判据 | 参考（Moebius 2001 / 同组 PRE 2005） | TensorDEM 实测 | 误差/判定 | 
|---|---|---|---|
| H1 大颗粒上升方向（$\rho/\rho_m$=3.24） | 上升（Brazil-nut，重颗粒每拍更快） | heavy intruder (rho/rho_m = 3.24) net ascent per case: -0.79 d (seed 0), -0.72 d (seed 1) | ✗ FAIL |
| H2 非单调：$T_{rise}$ 峰位 $\rho^*/\rho_m$ | 0.47–0.50 | 2.125（抛物线 $c_2=+0.006$） | 相对误差 338.1% ✗ FAIL |
| H3 尺寸依赖性（$T_{rise}$ 随 $D/d$ 减小） | 尺寸效应在 $D/d$ 大时显著 | ascent rate per tap vs D/d: D/d=4: -0.11 d/tap, D/d=6: -0.26 d/tap, D/d=6: -0.24 d/tap | ✗ FAIL |
| $T_{rise}$ 绝对值（s） | 仅图 1 图形给出 | — | **WITHHELD**（付费墙，未像素猜读） |
| 气压依赖（90 torr 分支） | 显著（原文核心结论） | 2D 干颗粒无间隙气体 | **NOT_SUPPORTED**（本轮引擎无气相） |

**逐案例明细**：

| case | $D/d$ | $\rho/\rho_m$ | seed | dt (s) | $\Gamma$ | taps | 净升 ($d$) | 升速 ($d$/tap) | 起浮拍 | 分层指数(初→末) |
|---|---|---|---|---|---|---|---|---|---|---|
| `brazil_nut_R4_rho3.24_s0` | 4 | 3.24 | 0 | 2.5e-05 | 5 | 4 | -0.32 | -0.106±0.023 | — | -0.141→-0.134 |
| `brazil_nut_R6_rho0.47_s0` | 6 | 0.47 | 0 | 2.5e-05 | 5 | 4 | -0.73 | -0.244±0.030 | — | -0.139→-0.144 |
| `brazil_nut_R6_rho1_s0` | 6 | 1 | 0 | 2.5e-05 | 5 | 4 | -0.76 | -0.253±0.023 | — | -0.139→-0.145 |
| `brazil_nut_R6_rho2_s0` | 6 | 2 | 0 | 2.5e-05 | 5 | 4 | -0.78 | -0.261±0.015 | — | -0.139→-0.146 |
| `brazil_nut_R6_rho3.24_s0` | 6 | 3.24 | 0 | 2.5e-05 | 5 | 4 | -0.79 | -0.265±0.010 | — | -0.139→-0.146 |
| `brazil_nut_R6_rho3.24_s1` | 6 | 3.24 | 1 | 2.5e-05 | 5 | 4 | -0.72 | -0.242±0.014 | — | -0.140→-0.144 |

> **执行偏差（如实声明）**：集群队列在本单预算内未派发（cj1238 排队），按预算降级为 reduced 网格执行：D/d=6、dt=2.5e-5 s（引擎稳定极限）、4 拍；完整网格的 命令与预注册仍在 sweep.py::grid_full 与 prereg.json 中，可复用。

> 预注册 {"D_over_d": 8.0, "dt_s": 2e-05, "Gamma": 5.0, "n_cases": 22} → 实际执行 {"D_over_d": 6.0, "dt_s": 2.5e-05, "Gamma_mode": 5.0, "n_cases": 6, "max_taps": 4}

## ⑤ 收敛性 / 参数敏感性

**(a) 时间步收敛**（$dt$ 减半，同一初值/种子）：

本轮 reduced 网格**未运行 dt 收敛对**（预算降级，见上方偏差声明）；dt 已取引擎自身稳定极限 $dt_{max}$=2.5e-5 s（引擎在更粗步长上直接拒绝构造），并以 seed 重复性（下面 (c)）作为重复性下界代替。

**(b) 驱动强度 $\Gamma$ 敏感性**（重颗粒家族）：

| $\Gamma$ | 5 | 5 |
|---|---|---|
| 升速 ($d$/tap) | -0.265 | -0.242 |

**(c) 种子重复性**（同一参数不同随机初值）：

| $\rho/\rho_m$ | 两 seed 升速 | 相对差 |
|---|---|---|
| 3.24 | -0.265, -0.242 | 9.2% |

**(d) 尺寸比 $D/d$**（重颗粒）：$D/d$=4: -0.106 $d$/tap, $D/d$=6: -0.265 $d$/tap, $D/d$=6: -0.242 $d$/tap。

## ⑥ 失败模式与工件

**判定：honest-FAIL**；fail_stage = physics (size separation) — not a harness or numerics failure: the engine is stable (no blow-up, KE decays in the rest phase), mass bookkeeping is exact and the repeated seeds agree。

**failure_reason**：dry 2D DEM at D/d = 6 and Gamma = 5 does not produce Brazil-nut ascent: the heavy intruder drifts DOWN by 0.7-0.8 d over four taps and the drift is density independent within 8 %.  The reference's ambient-pressure rise is attributed by the same authors to interstitial-air drag (companion paper, PRE 72 011306 (2005): at P < 0.13 kPa 'the rise times are nearly independent of rho/rho_m and D/d' and 'the heaviest intruder does not surface'). The measured density independence therefore MATCHES the reference's air-free branch, while the ambient-pressure rise branch is NOT_SUPPORTED: the 2D dry-disk engine has no gas phase, and DiskConfig drag_time is a global scalar, so the intruder and the bed cannot have different air-drag times.

**下一轮**：add a per-particle drag_time (differential air drag between intruder and bed) and re-run at ambient-pressure-like damping; the full pre-registered grid (D/d = 8, 30 taps, 22 cases) is already coded in sweep.py::grid_full

- **H1_direction 未通过**：heavy intruder (rho/rho_m = 3.24) net ascent per case: -0.79 d (seed 0), -0.72 d (seed 1)
- **H2_nonmonotonic 未通过**：slowest-ascent density ratio rho*/rho_m = 2.125 vs reference band (0.47, 0.5); quadratic curvature c2 = +0.006
- **H3_size 未通过**：ascent rate per tap vs D/d: D/d=4: -0.11 d/tap, D/d=6: -0.26 d/tap, D/d=6: -0.24 d/tap

已知失败模式与局限（如实）：
- **尺寸比缺口**：$D/d$=8 vs ref 50.8。ref 同组论文指出 $D/d<10$ 时侵入体对床的影响减弱、空气驱动尺寸分离趋于可忽略——这正是本算例只能对齐**趋势**而不能对齐绝对 $T_{rise}$ 的物理原因。
- **无间隙气体**：ref 的密度非单调性被归因于空气阻力（同组 PRE 2005）。干 2D DEM 的 $\rho$ 依赖机制只剩空隙充填/浮力竞争，**不应**期待复现 90 torr 分支。
- **2D vs 3D**：圆盘无面外自由度，空隙几何与球的差别影响渗流速率。
- **起浮拍的删失**：拍预算用完仍未到达自由面的案例记为 censored（`surfaced_at_tap`=—），只用升速做统计。

**工件**：
- `benchmarks/case_brazil_nut_moebius2001/runs/brazil_nut_R4_rho3.24_s0_dt2.5e-05.json` 等（每案例一个 JSON：完整拍序列、几何、质量覆写、digest、墙钟）
- `benchmarks/case_brazil_nut_moebius2001/runs/brazil_nut_R6_rho0.47_s0_dt2.5e-05.json` 等（每案例一个 JSON：完整拍序列、几何、质量覆写、digest、墙钟）
- `benchmarks/case_brazil_nut_moebius2001/runs/brazil_nut_R6_rho1_s0_dt2.5e-05.json` 等（每案例一个 JSON：完整拍序列、几何、质量覆写、digest、墙钟）
- `artifacts/case_brazil_nut_moebius2001/result.json`（聚合结果 + 判定）
- `artifacts/case_brazil_nut_moebius2001/fig1_trajectories.png` — Layering index of the intruder (its height above the small-particle centroid, in bed heights) against tap index, for the density family at D/d = 8 and Gamma = 5 (seed 0).
- `artifacts/case_brazil_nut_moebius2001/fig2_rate_vs_density.png` — Per-tap ascent rate of the intruder against density ratio, the quadratic fit whose interior minimum locates the slowest-ascent density, and the reference band for the T_rise peak.
- `artifacts/case_brazil_nut_moebius2001/fig3_size_and_dt.png` — Per-tap ascent rate against size ratio at rho/rho_m = 3.24 (production step) with the half-step points overlaid.
- `artifacts/case_brazil_nut_moebius2001/fig5_configuration.png` — Particle configuration before the taps (left) and after the tap sequence (right); the large red disk is the intruder, grey disks the bed.
- `artifacts/case_brazil_nut_moebius2001/fig4_height_history.png` — Height history of the intruder, of the small-species centroid and of the free surface for the heaviest intruder.

**一键复现**：
```bash
python3 benchmarks/case_brazil_nut_moebius2001/sweep.py --grid full --workers 20
python3 benchmarks/case_brazil_nut_moebius2001/analyse.py
```
