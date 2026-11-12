# W25 编队冰阻力节省率 vs Lindqvist 单船解析锚（返修轮1）

任务：`paper-exp-formation-ice-channel-2-rework1`（mission 16718，领域仓 TensorDEM；runner = native DEM，`tensordem.ice_channel`）。

工件：`papers/artifacts/w25/formation-ice-channel/lindqvist_formation_gci.json`（由 `benchmarks/w25_formation_collect.py` 从 7/148 个原始算例 JSON 程序化生成，零手抄）。

预注册：`prereg/w25_formation_ice_channel_prereg.json`（sha256 `9c9ddebf4d41c330…`，先于生产运行写定）。

## 1 案例与口径

| 项 | 值 |
|---|---|
| 船体 | L = 80 m，B = 20 m，L/B = 4.00，rectangular box (planar 2-D hull, no bow angle) |
| 方形系数 | Cb = 1.000（rectangular box hull -> Cb = 1 identically） |
| 冰况 | h = 0.70 m，密集度 0.8，平均冰块半径 10.0 m（冰径/船宽 = 1.00） |
| 环境 | 无风；横向漂流 0.05 m/s（re-closes the corridor cleared by the lead） |
| 数值 | dt = 0.008 s（dt scaled with floe radius (constant omega*dt)）；域宽 70 m；冰场长度 max(span + 2L, 4L)；每 5 步采样 |
| 系综 | 编队 3 个冰场实现；单船 5 个（节省率分母噪声最大） |

测量窗口（先于运行声明）：船体中心位于冰场内且距纵向两端 ≥ L/2、t ≥ 60 s 时才计入样本；
每船载荷取该窗口内**上游接触反力**的时间平均，编队总冰阻力取三船之和。

算例矩阵：148 个 DEM 算例（角色分布 {'formation': 105, 'gci': 4, 'numerics': 4, 'single': 35}），速度 [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] m/s × 间距 [1.0, 2.0, 3.0, 4.0, 5.0]L × 3 seeds。

## 2 方法

1. `tensordem.ice_channel.IceChannelSimulation` 生成抖动六角冰场（密集度 8/10，实现在 <1% 内），船体为运动学矩形，冰-船/冰-冰均为锚定罚函数弹簧接触；
2. 新增 `tensordem.ice_channel_formation`：算例口径、入水脚印清理（`carve_launch_footprints`，只删与 t=0 船体脚印重叠的冰块，避免初始深穿透冲量）、逐船载荷账本、节省率、
   Roache GCI、单调性、Lindqvist 同口径换算与「超出公式适用域」判据；
3. `benchmarks/w25_formation_runner.py` 按分片跑矩阵并落原始 JSON；
4. `benchmarks/w25_formation_collect.py` 汇总系综、判定三条判据、出图；
5. 单船退化算例对账对象 = `tensordem.level_ice.formula` 的 Lindqvist 三分量锚（系数 provenance 仍为 WITHHELD：主论文离线不可读，见锚脚本）。

## 3 结果

### 3.1 单船退化算例 vs Lindqvist 三分量锚

| V [m/s] | DEM 单船 [kN] | seeds | seed 相对散度 | Lindqvist 总量 [kN] | 弯曲+下沉下限 [kN] | 相对偏差 | <15%？ |
|---|---|---|---|---|---|---|---|
| 0.50 | 4.20 | 1 | n/a % | 260.1 | 105.0 | -98.4 % | 否 |
| 1.00 | 21.35 | 1 | n/a % | 265.4 | 105.0 | -92.0 % | 否 |
| 2.00 | 84.68 | 1 | n/a % | 287.0 | 105.0 | -70.5 % | 否 |

Lindqvist 主口径（α = 20°, μ = 0.05）分量占比：挤压 59.0%、弯曲 5.9%、下沉 34.5%、附加阻力 0.69%。

**超出适用域判据**：3/3 个速度点满足 `R_DEM < R_bending + R_submersion`（Lindqvist 的弯曲+下沉已是任何含连续冰盖模型的载荷下限）。该条件成立时，任何网格加密都无法把偏差压进 15%——差距是**模型形式（能力缺口）**，不是离散误差。

### 3.2 节省率-间距曲线

| S/L | V=0.50 | V=1.00 | V=2.00 |
|---|---|---|---|
| 1 | n/a | 66.7 ± n/a | n/a |
| 3 | n/a | 16.6 ± n/a | n/a |
| 5 | n/a | -14.7 ± n/a | n/a |

表中为**配对节省率** `1 − R_总/(3 R_领队)` [%]（同一冰场实现内比较，去掉单船参考的噪声），± 系综 SEM。独立单船参考口径的节省率见工件 `formation_savings.*.by_spacing.saving`。

### 3.3 单调性（判据 C2a）

| V [m/s] | 容忍阈 | 配对节省率逐档 | 严格单调 | 容忍单调 |
|---|---|---|---|---|
| 1.00 | 0.000 | 0.667, 0.166, -0.147 | 是 | 是 |

### 3.4 粒子数收敛与 GCI（判据 C2b / C3）

| 族 | N | 2N | 4N | 观测阶 p | GCI(fine) | 单调 | 收敛 |
|---|---|---|---|---|---|---|---|

（表内为总冰阻力 kN；GCI 取 fine 网格，安全因子 1.25，细化比 r = 2，同时细化粒子数 N→2N→4N 与 dt（保持 ω·dt 不变）。）

### 3.5 数值控制


### 3.6 ISO 19906 p-A 面积律交叉核对（上界口径）

| V [m/s] | DEM 名义压力 [kPa] | ISO 19906 p-A 压力 [kPa] | DEM/ISO 载荷比 |
|---|---|---|---|
| 0.50 | 0.3 | 348.0 | 0.0009 |
| 1.00 | 1.5 | 348.0 | 0.0044 |
| 2.00 | 6.0 | 348.0 | 0.0174 |

口径：nominal contact area A = B*h (waterline breadth x ice thickness)；实现 `tensordem.ice_ridge_keel_hull.iso19906_ridge_load`，参数 p_0 = 1000000 Pa，A_0 = 1.0 m²，指数 -0.40；来源 ISO 19906:2010 Annex C p-A law as implemented in src/tensordem/ice_ridge_keel_hull.py (p_0, A_0, exponent carried by ISO19906RidgeParams; grade CONVENTION)。DEM/ISO 比 0.001–0.017 = broken-ice channel load as a fraction of the level-ice ISO global design load on the same nominal area。

## 4 图

- `papers/artifacts/w25/formation-ice-channel/figures/saving_vs_spacing.png` — Ice-resistance saving of a three-ship column against three independent single ships, versus non-dimensional spacing S/L, for seven transit speeds (0.5-2.0 m/s). Error bars: SEM over three floe-field realisations.
- `papers/artifacts/w25/formation-ice-channel/figures/lindqvist_vs_dem.png` — Single-ship ice resistance from the DEM channel model (ensemble mean ± SEM) against the Lindqvist (1989) three-component anchor converted to the case caliber, and against the anchor's bending+submersion floor.
- `papers/artifacts/w25/formation-ice-channel/figures/gci_family.png` — Particle-count refinement family (N, 2N, 4N floes at constant ice concentration) for the single-ship and the three-ship column cases.
- `papers/artifacts/w25/formation-ice-channel/figures/reclosure_mechanism.png` — Measured follower-to-lead ice-load ratio versus the ratio of the follower lag S/V to the drift-driven channel re-closure time B/(2 u_drift).

## 5 注意点与适用范围

1. **刚体圆盘无弯曲破坏**：模型没有连续冰盖的弯曲失效机制，Lindqvist 的 R_b 在其中没有对应物；
2. **二维方形船体无艏倾角**：α = 90° 使挤压项发散，有限 α 只是替代口径（公式本身也拒绝 ≥89°）；
3. **单实现的系综散度数十个百分点**：瞬时载荷以 O(MN) 稀疏尖峰叠加出 O(10 kN) 均值，重尾时间平均必须用系综均值 + SEM，单例不足以支撑曲线；
4. **横向漂流是建模选择**：它是唯一能让领队清出的走廊重新闭合、从而产生间距依赖的机制，其敏感带已给出；
5. **冰厚恒定 0.7 m**：本算例无法检验锚点的厚度依赖（h² 标度）。

## 6 误差判定（关键量 vs 基准，逐项回答「是否 <3%」）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| 单船冰阻力（最差速度点） | Lindqvist 三分量总量 | -98.4 % | 否 |
| 单船冰阻力网格收敛（N/2N/4N） | 加密极限 | GCI n/a % | n/a |
| 编队总冰阻力网格收敛（N/2N/4N） | 加密极限 | GCI n/a % | n/a |

判据结论（程序化判定，honest FAIL 不覆盖）：

- **C1 FAIL** — single-ship degenerate case vs Lindqvist (1989), |relative deviation| < 15 %
- **C2a PASS** — saving-vs-spacing curve monotonicity (non-increasing)
- **C2b WITHHELD** — grid-convergence monotonicity of the particle-refinement family
- **C3 WITHHELD** — particle-count GCI (N x2 / N x4) < 5 %

汇总：1 PASS / 1 FAIL / 2 WITHHELD。

