# CASE-PENETRATION R2 — 球体准静态贯入阻力（TensorDEM）

**锚点** CASE-PENETRATION-R2 ｜ mission 15404 ｜ 分支 `agent/m15404` ｜ 状态 **FAIL**

## ① 案例介绍

刚性球（半径 R）以准静态方式垂直贯入无黏性颗粒床，测深度相关阻力 F(z)，与 Brzinski III, Mayor & Durian, PRL **111**, 168002 (2013)（arXiv:1307.4638v2） Eq.(2) `dF = alpha mu (rho_bulk g z) dA` 及其积分 Eq.(3c)/(3d) 对照，标定无量纲摩擦放大系数 alpha（文献 35 ± 5，p.1）。

| 参数 | 符号 | 本次 DEM | 文献 |
|---|---|---|---|
| 颗粒直径 | d | 0.004 m | 3.0e-4 m |
| 贯入球半径 | R | 0.008 m (R/d = 2) | 1.27/2.54/5.05e-2 m (R/d~42) |
| 容器宽度 | L | 0.048 m（半宽 3 R） | 190 mm 直径 ≈ 3.7 R |
| 颗粒数 | N | 810 | — |
| 实测装填率 | phi | 0.6035 | 0.59 |
| 实测体密度 | rho_bulk | 1508.7 kg/m3 | 1480 kg/m3 |
| 颗粒摩擦 | mu | 0.4 | 0.40 (tan22°) |
| 贯入速度 | v | 0.6 m/s | ~1e-5 m/s |
| 时间步 | dt | 1.44e-06 s | — |

## ② 怎么做的

前轮 m15367 的失败阶段是 `simulation_protocol_energy_injection`：每亚步把贯入球瞬移 0.125 d = 0.5 mm，而该载荷下的弹性接触变形只有约 20 µm，即每步把颗粒床抛出平衡 25 个接触变形长度，弹性能（~0.5 J/接触）远大于床自重势能（2e-3 J 量级）→ 颗粒被弹射，记录的"平衡力"是冲击力。本轮改为文献/教科书标准的准静态 DEM 三条：

1. **速度（运动学）边界条件**：贯入球是 `fixed` 刚体，速度设为 -v，位置由积分器每步推进 v*dt = 8.63e-07 m，小于该载荷下的弹性接触重叠尺度；接触力在漂移后位置评估（KDK 半步修正）。
2. **接触本构与协议分离**：床制备用带接触阻尼的材料（restitution=0.4）压实；加载阶段的速度与阻尼固定，并用倍速对照（0.05 vs 0.10 m/s）实证准静态性，而不是声明准静态性。
3. **床必须先自支撑**：HCP 点阵 2% 表面间隙 + 仅靠背景黏滞拖曳时，15000 步内颗粒仍在下落（接触数 0），贯入器推的是"悬浮点阵"（实测接触数 1→1586、KE 60 J）。本轮用 `artifacts/case_penetration/tools/bed_recipes.py` 扫出配方：packing_gap=0.08、jitter=0.05 d、15000 步重力沉积（restitution=0.4, gamma=50 1/s）→ phi=0.6035、rho_bulk=1508.7 kg/m3、静止接触 230 个。

工具链：`src/tensordem/penetration_quasistatic.py`（协议+分析）、`src/tensordem/sphere_pack_dem.py`（两处引擎修复，见⑤）、`artifacts/case_penetration/tools/{bed_recipes,run_case,post_report}.py`、`tests/test_penetration_quasistatic.py`。

## ③ 计算结果

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 贯入阶段最大系统动能 kinetic_energy_max | ≤ 0.1 J | 0.365 J | ✗ |
| 力抖动系数 force_variation_coefficient | ≤ 0.1 | 0.476 | ✗ |
| alpha 相对误差 penetration_force_error | ≤ 0.15 | 8.98 | ✗ |
| 最大接触重叠比 | ≤ 0.05 | 0.014 | ✓ |

关键量：alpha_fit = **349.3**（约束估计，F = A(z-R/3)，A = alpha mu rho g pi R^2），自由线性拟合 alpha_fit_free = 90.91，文献带 [30.0, 40.0]，落在文献自带不确定带内: False。窗口内平均力 2.53 N，拟合残差 rms 1.2 N，对 Eq.(3c)/(3d)（alpha=35）逐点平均相对偏差 10.2。

**前轮 teleport 路径复现**（同一张床）：KE 最大值 5.7 J、力峰值 42.6 N —— 复现了 m15367 的 honest-FAIL 现象，证明能量注入来自加载路径而非床初始条件。

## ④ 结果图

见 fig1（F-z 与动能）、fig2（倍速对照）、fig3（三种加载路径动能对比）。

## ⑤ 注意点 / 踩坑

- **引擎修复 1（邻接表截断）**：`SpherePack._build_neighbour_list` 原用全局 `2*r_max` 截断，贯入球 R ≫ 颗粒半径时把整套床都放进候选表（N=1260 时 0.014 s/步）；改为自由粒子用颗粒截断、每个固定刚体单独 `r_body + r_grain` 截断，候选集仍是真接触集的超集，`tests/test_sphere_pack_dem.py` 全绿。
- **引擎修复 2（壁面切向历史）**：壁面接触的 Mindlin 历史原来存在 python dict 里，每步两次力评估各做 O(1e3) 次解释器级 get/set（实测占 90% 步时）；改为每个壁面一张 (N,3) 数组，算术不变、7 个引擎测试全绿。
- **本仓纯 numpy 内核**：R/d=42（文献）不可达；本轮 R/d=2 粗粒化，解理上 alpha 对 R/d 的敏感性是未完成项（见⑦）。
- **mu 的语义**：文献 mu 是介质内摩擦系数（tan 安息角），DEM 输入的是颗粒-颗粒摩擦系数；本轮未做安息角标定（已知缺口）。

## ⑥ 误差判定

- alpha vs 文献 35：相对误差 898% —— **>15%，未通过**；是否 <3%：898% ≥3% ✗（文献 alpha 自带 ±14% 不确定度，3% 不是本算例的合理判据）。
- 力曲线逐点 vs Eq.(3c)/(3d)（alpha=35）：平均相对偏差 1.02e+03%。
- 床装填率 vs 文献：0.6035 vs 0.59（相对 2.3%）；体密度 1508.7 vs 1480 kg/m3。

## ⑦ 接力清单（未完成项）

1. R/d = 3/4 的 alpha 敏感性（本单只跑通 R/d=2；成本随 R/d 三次方增长）。
2. 安息角标定 mu_eff，把 DEM 颗粒摩擦与文献介质内摩擦对齐。
3. 壁面半宽 4R → 5R 的 Janssen 对照。

![fig1](fig1_force_depth_energy.png)
![fig2](fig2_rate_check.png)
![fig3](fig3_protocol_energy.png)
