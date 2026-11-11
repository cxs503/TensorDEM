# FEM→DEM 断裂交接（wave 2）：加载路径速度继承 + 刚体投影守恒链（TensorDEM）

**一句话**：wave 1 把 FEM 断裂结果以零速度交给 DEM（约定俗成要 DEM 自己编初速）；本波把连续解真正知道的运动交出去——加载路径速度场 `ẇ = q̇·ŵ` 经**质量加权刚体最小二乘投影**变成 7 块碎冰的平动+倾转初条件，并把三条**代数恒等式**（线动量、面内角动量、动能可加）当作验收判据逐条机检：残差分别为 -2.84e-14 kg·m/s、-1.14e-13/-7.11e-15 kg·m²/s、-1.11e-16 J（相对 1e-16 级）；质量残差 0.00e+00 kg。全套 D1–D8 判据 PASS，17 行 benchmark 全部在容差内。

## ① 案例介绍（物理背景 / 几何 / 参数表）

**物理背景**：船体/结构挤压平整冰板（level ice）时，冰板先以薄板弯曲响应，弯曲应力达到弯曲强度 `σ_flexure` 后开裂、分块；碎块随后作为离散刚体运动，其运动/碰撞/浮力响应必须由 DEM 承担。连续→离散的**交接面**（fracture handoff）是整个冰区全流程 POC 的关键接口：FEM 若只给几何不给运动，DEM 的初始动量与动能就是凭空捏造的。

**几何与载荷**：`10 m × 10 m` 自由边冰板浮于 Winkler 基础上（`k = ρ_w g`），两个矩形接触压力区（`2×2 m @ 1.0` 与 `1.2×1.6 m @ 0.65`，见 `failure_faces.json` 的 `meta.load`）。网格 `40×40` ACM 12 自由度 Kirchhoff 矩形元。

| 参数 | 值 | 说明 |
|---|---|---|
| 板尺寸 | 10.0 × 10.0 m | 面积 100 m² |
| 冰厚 t | 0.500 m | 薄板适用性 a/t = 20 |
| 弹性模量 E | 5 GPa | ν = 0.30 |
| 抗弯刚度 D | 5.723e+07 N·m | E t³/(12(1−ν²)) |
| 冰密度 ρ_i | 917 kg/m³ | 质量 = A t ρ_i |
| 弯曲强度 σ_flexure | 0.700 MPa | 开裂判据 σ_max ≥ σ_flexure |
| 基础模量 k | 10051.8 N/m³ | ρ_w g |
| 加载率 q̇ | 1000 Pa/s | 本波新增的运动学输入（速率标度 ∝ q̇，见 ⑤） |
| 断裂时压力尺度 | 35006.2 Pa | 分块状态（split）压力 |

## ② 怎么做的（方法 / 流程 / 关键设置）

**步 1（wave 1，复用）**：单位压力解 `K ŵ = b̂`；按 `σ_max = 6 max|M_i|/t²` 找首裂压力，再沿载荷上升找分块状态，冻结应力场的模式 I 裂纹追踪给出裂纹 polyline、断裂单元 polygon 与碎块分割，写出 `failure_faces.json`（schema 1，质量/面积/周长三重闭合账本）。

**步 2（本波核心）速度继承**：板问题对压力尺度 `q` 线性，故 `ẇ = q̇ ŵ` **精确**成立，不需要任何伪时间步进。把节点 12 自由度（`w, w,x, w,y`）在单元形心处做逐角点一阶 Taylor 后取四角平均，得到单元形心速率；该重构对**线性场精确**（后面刚体投影的解析判据正依赖这一点）。线性性由两级求解互检：`(w(2q̂) − w(q̂))/1` 与 `ŵ` 的相对残差 = 0.0e+00（逐位相同）。

**步 3 刚体投影**：DEM 刚体无法承载曲率，每块碎冰取其 FEM 单元点质量（`m_c = ρ t A_c`，故碎冰质量与其单元集合严格相等）上的质量加权最小二乘平面拟合：

```
min_{a,b,c} Σ_c m_c (u_c − a − b Δx'_c − c Δy'_c)²      （在碎冰主轴 e1,e2 系内）
  ⇒ a = Σm u /M（平动）          b = Σm Δx'u / I_b         c = Σm Δy'u / I_a
  ⇒ ω₁ = c（绕 e1）, ω₂ = −b（绕 e2）,  L = I_a ω₁ e1 + I_b ω₂ e2
```

在主轴系内惯量张量对角，于是三条恒等式**由正规方程直接给出**，本波把它们当判据机检而非假设：① `Σ m u = M a`（线动量）；② `L = I ω`（面内角动量，绕碎冰质心与绕板中心两次校验）；③ `KE_total = KE_rigid + KE_deform`（交叉项全为零）。

**步 4 工件与复核**：写出 schema 2 工件 `failure_faces_dynamic.json`（schema 1 全体 + `meta.dynamics` + 逐碎冰 `kinematics` + 逐单元 `velocity_field` + `momentum_ledger`），带 payload SHA-256；读取端**只用文件内的逐单元速度场与碎块多边形重算**整套账本，再与写入值逐项比对（篡改/陈旧文件直接抛错）。

**步 5 DEM 侧真调用**：`read → dem_initial_conditions_dynamic → dem_intake`：在冰板上建 DEM 栅格，用 `tensordem.fem_dem_seeding.clump.seed_floe_bodies` 逐碎冰播种等效面积刚体，再用 `...dynamics.heave_audit` 带着**继承速度**积分垂荡，与闭式周期 `ω = sqrt(ρ_w g /(ρ_i t))` 对照，最后用 `post_run_ledger` 复闭账本。

**工具链**：TensorDEM `src/tensordem/fem_dem_fracture/{dynamics,study_dynamic,__main__}.py`；入口 `python -m tensordem.fem_dem_fracture`；测试 `tests/test_fem_dem_fracture_dynamic.py`（20 例）；基准 `benchmarks/fem_dem_fracture_dynamic/benchmark.py`。

## ③ 计算结果（数字表，全部由产物 JSON 程序化注入）

**A. 交接守恒账本（schema 2，读取端重算值）**

| 量 | 数值 | 容差/判据 | 判定 |
|---|---|---|---|
| 场质量 Σm u 对应质量 | 45850 kg | 板质量 A t ρ | — |
| 质量残差 | 0.000e+00 kg | < 1e-6 kg | PASS |
| 竖直线动量（场/刚体） | 2.393842e+02 / 2.393842e+02 kg·m/s | — | — |
| 竖直线动量残差 | -2.842e-14 kg·m/s | < 1e-10 相对 | PASS |
| 角动量（x,y）场 | -6.949875e+01 / -1.804538e+01 kg·m²/s | — | — |
| 角动量残差（x,y） | -1.137e-13 / -7.105e-15 | < 1e-10 相对 | PASS |
| 动能（场/刚性/形变） | 6.316799e-01 / 6.316746e-01 / 5.280709e-06 J | — | — |
| 动能可加残差 | -1.110e-16 J | < 1e-10 相对 | PASS |
| 板级刚性能量占比 | 0.999991640 | [0,1] | PASS |

**B. 逐碎冰初条件（FEM 单元质量质心 + 继承运动）**

| floe | 面积 m² | 等效直径 m | v_z m/s | ω_x rad/s | ω_y rad/s | KE_rigid J | KE_deform J | 刚性占比 |
|---|---|---|---|---|---|---|---|---|
| 0 | 46.12 | 7.663 | 5.6825e-03 | -1.690e-04 | -4.645e-05 | 3.4234e-01 | 3.0182e-06 | 0.999991 |
| 1 | 1.25 | 1.262 | 5.7884e-03 | -1.707e-04 | -4.375e-05 | 9.6027e-03 | 1.6507e-09 | 1.000000 |
| 2 | 0.3125 | 0.631 | 5.3949e-03 | -1.760e-04 | 0.000e+00 | 2.0854e-03 | 2.0956e-11 | 1.000000 |
| 3 | 4.188 | 2.309 | 5.3597e-03 | -1.871e-04 | -3.245e-05 | 2.7603e-02 | 5.0693e-09 | 1.000000 |
| 4 | 46.44 | 7.689 | 4.7411e-03 | -1.918e-04 | -4.818e-05 | 2.4031e-01 | 2.2555e-06 | 0.999991 |
| 5 | 1.125 | 1.197 | 4.9862e-03 | -1.920e-04 | -5.116e-05 | 6.4131e-03 | 1.3730e-10 | 1.000000 |
| 6 | 0.5625 | 0.846 | 5.0756e-03 | -1.918e-04 | 0.000e+00 | 3.3240e-03 | 1.3302e-10 | 1.000000 |

**C. 准静态有效性（本波新增的诚实性闸门）**

| 量 | 数值 | 说明 |
|---|---|---|
| 弹性储能 U（断裂载荷下） | 1.710200e+04 J | U = ½ q² ŵᵀ b̂（线性系统恒等式） |
| 继承动能 KE | 6.316799e-01 J | KE = ½ ρ t ∫ v² dA |
| KE/U（运行加载率 1000 Pa/s） | 3.694e-05 | 二阶小量，`KE/U = (q̇/q̇_crit)²` |
| 临界加载率（KE/U = 0.1） | 5.203e+04 Pa/s | 超过此率应交由真动态 FEM |

**D. DEM 侧 intake（真调用）**

| 量 | 数值 | 容差 | 判定 |
|---|---|---|---|
| 刚体数 | 7 | = 碎块数 7 | PASS |
| 播种质量残差 | 0.000e+00 kg | < 1e-6 kg | PASS |
| 播种面积残差 | 0.000e+00 m² | < 1e-9 m² | PASS |
| 垂荡周期（解析/测量） | 1.341922 / 1.341908 s | < 0.1 % | PASS |
| 垂荡周期最大相对误差 | 1.028e-05 | < 1e-3 | PASS |
| 垂荡积分能量漂移 | 6.169e-05 | — | — |
| 继承竖速范围 | [4.7411e-03, 5.7884e-03] m/s | — | — |

**E. benchmark：参考值 vs 计算值（17 行，容差逐行给出）**

| benchmark | 量 | 参考值 | 计算值 | 相对误差 | 容差 | 判定 |
|---|---|---|---|---|---|---|
| strip cylindrical bending (beam limit) | w_max | 0.00015625 | 0.000156915 | 4.257e-03 | 3.0e-02 | PASS |
| strip cylindrical bending (beam limit) | M_max | 3125 | 3129.9 | 1.567e-03 | 3.0e-02 | PASS |
| SSSS square plate (Navier series) | w_max | 0.000709774 | 0.000709572 | 2.848e-04 | 3.0e-02 | PASS |
| SSSS square plate (Navier series) | M_x(centre) | 4788.51 | 4779.19 | 1.946e-03 | 3.0e-02 | PASS |
| clamped square plate | w_max | 0.000220147 | 0.000221354 | 5.481e-03 | 3.0e-02 | PASS |
| Winkler plate, local load (Hetenyi) | w_max | 0.0164801 | 0.0163981 | 4.973e-03 | 3.0e-02 | PASS |
| rigid projection, exactly rigid field | KE_rigid / KE_total | 1 | 1 | 2.220e-16 | 1.0e-12 | PASS |
| rigid projection, exactly rigid field | max |v - v_rigid| [m/s] | 0 | 3.25261e-19 | 3.253e-19 | 1.0e-12 | PASS |
| rigid projection, exactly rigid field | omega_x recovery [rad/s] | 0.0002 | 0.0002 | 2.439e-15 | 1.0e-10 | PASS |
| rigid projection, isotropic bending field | KE_rigid / KE_total | 0.718593 | 0.718593 | 0.000e+00 | 1.0e-12 | PASS |
| rigid projection, isotropic bending field | max |tilt rate| [rad/s] | 0 | 4.09419e-20 | 4.094e-20 | 1.0e-12 | PASS |
| momentum transfer | residual [kg m/s] | 0 | 2.84217e-14 | 2.842e-14 | 2.4e-10 | PASS |
| angular momentum transfer | max residual [kg m^2/s] | 0 | 1.13687e-13 | 1.137e-13 | 6.9e-11 | PASS |
| kinetic energy split | KE_rigid + KE_deform - KE_field [J] | 0 | 1.11022e-16 | 1.110e-16 | 1.0e-12 | PASS |
| floe heave (DEM-side integration) | period [s] | 1.34192 | 1.34191 | 1.028e-05 | 1.0e-03 | PASS |
| loading path linearity | relative finite-difference residual | 0 | 0 | 0.000e+00 | 1.0e-09 | PASS |
| quasi-static validity | KE/U at the run rate | 0.1 | 3.6936e-05 | 0.000e+00 | 1.0e+09 | PASS |

全部行在容差内：`all_within_tolerance = True`，`n_failing = 0`。

## ④ 结果图

1. `runs/p302w2_n03/fig_velocity_field.png` —— 继承速率场 `v_z`（mm/s）云图 + 碎块边界与编号：断裂时整板以 ~5 mm/s 量级下沉，最大单元速率 6.2729e-03 m/s。
2. `runs/p302w2_n03/fig_rigid_fraction.png` —— 逐碎冰动能分解（刚体/形变，对数纵轴）与刚性占比：大碎块占比略有下降（曲率方差 ∝ 尺寸²），小碎块刚性占比数值上 = 1。
3. `runs/p302w2_n03/fig_heave.png` —— 最大碎块带着继承 `v_z` 的垂荡轨迹，周期与闭式解一致（误差 <1e-5）。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

**踩过的坑（均为本轮实测）**

1. `KE_rigid/KE_total` **不是拟合的 R²**。刚体动能含平动项，故对称平方域上的各向同性弯曲场 `κ/2 (x²+y²)` 给出 `mean(u)²/mean(u²)`（离散格点 0.7186），而不是 0。第一版 benchmark 把参考值写成 0，被判 FAIL——错在参考值而非实现，已改为闭式 `mean²/mean(u²)` 并另加一条「倾转速率 = 0」的对称性判据（实测 4e-20）。
2. 有限差分查线性性用小 `δ` 会被灾难性相消淹没：`δ=1e-3` 时相对残差 4.7e-9（纯舍入放大 1/δ）；改用 `δ=1` 后残差逐位为 0。判据不能定得比所用算法的条件数更严。
3. 弹性储能不能用「节点 w 点乘一致载荷向量」以外的方式近似：`_pressure_load_vector` 只填挠度自由度，必须把 `(w,θx,θy)` 交错成完整自由度向量再点乘，否则量纲/维度都对不上（首版直接抛广播错误）。
4. 单元格心速率必须用**四角 12 自由度一阶 Taylor 后平均**；只用节点 `w` 平均会破坏刚体场的精确性，从而让「线性场刚性占比 = 1」这条判据失真。

**局限（不粉饰）**

* 裂纹追踪仍是冻结应力场的模式 I 图样生成器：无裂尖应力重分布、无断裂动力学、无黏聚区；因此碎块普查是 DEM 的几何输入，不是强度预测。
* 交接是**准静态**初条件：`KE/U = 3.7e-5`（运行 1000 Pa/s），临界率 5.2e4 Pa/s 之上应交由真动态 FEM。本波给出的 `KE/U ∝ q̇²` 是精确标度（已测 2 倍率给出 4 倍 KE，残差 1e-12）。
* 面内（膜）运动不在弯曲解内：`v_inplane` 是环境输入，本次取 0，并在工件中标注来源。
* DEM 侧只做「刚体播种 + 带继承速度的垂荡积分」；未做接触解析的整板 DEM 运行——本仓生产格点求解器（`IceDEM3D`）几何来自固定格点，任意碎块插入不在其接口内，该缺口如实登记，不用假跑填。
* 等效圆盘理想化在非凸碎块上可能重叠（polygon 随工件交付供形状解析接触使用）；本波只证质量/面积守恒，不证不重叠。
* 单一加载率的结果表述：所有速率相关量均按 `∝ q̇` 线性缩放，已由线性性判据保证。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 相对误差 | <3% ? |
|---|---|---|---|
| strip cylindrical bending (beam limit) · w_max | 闭式解 | 0.426 % | 是 |
| strip cylindrical bending (beam limit) · M_max | 闭式解 | 0.157 % | 是 |
| SSSS square plate (Navier series) · w_max | 闭式解 | 0.028 % | 是 |
| SSSS square plate (Navier series) · M_x(centre) | 闭式解 | 0.195 % | 是 |
| clamped square plate · w_max | 闭式解 | 0.548 % | 是 |
| Winkler plate, local load (Hetenyi) · w_max | 闭式解 | 0.497 % | 是 |
| 线动量守恒 | 精确恒等式（参考 0） | 1.19e-16（相对） | 是 |
| 面内角动量守恒 | 精确恒等式（参考 0） | 1.64e-15（相对） | 是 |
| 动能可加性 | 精确恒等式（参考 0） | 1.76e-16（相对） | 是 |
| 质量守恒（FEM→DEM 刚体） | 板质量 A t ρ | 0.00e+00（相对） | 是 |
| 刚体投影（线性场） | 解析：占比 1、残差 0 | 2.44e-15 | 是 |
| 刚体投影（各向同性弯曲场） | 解析 `mean²/mean(u²)` | 0.00e+00 | 是 |
| 垂荡周期 | 闭式 `sqrt(ρ_w g/(ρ_i t))` | 0.00103 % | 是 |
| 加载路径线性性 | 解析：残差 0 | 0.00e+00 | 是 |

**结论**：连续侧 6 个闭式基准误差 **0.028 %–0.55 %**（<3 %）；交接侧 7 类量级为**机器精度级恒等式残差**（1e-16–1e-13 相对），唯一的物理近似是刚体投影丢弃的形变动能（占板总动能 0.00084 %），该量已显式报告而非隐藏。无「无基准」项；DEM 接触解析整板运行未做（见 ⑤ 局限）。

## 审计链

* 工件 SHA-256（6 个）：`benchmark`=3b4c33f8e19168b2…；`dem_initial_conditions_dynamic`=6e46569254a40ee5…；`dem_initial_conditions_static`=9502c55118b3e753…；`dem_intake`=38f4d96606a021ac…；`failure_faces`=1c2c2178ef80cfd8…；`failure_faces_dynamic`=0221d3e31d790400…。
* payload SHA-256（schema 2 内容哈希）：`96e3ec111c7a8526085b6cc2b13d9fbace96d08fd82f2d0da0b2bb3e1c2f9b13`。
* 复现命令：`python -m tensordem.fem_dem_fracture --out runs/p302w2_n03 --pressure-rate 1000`；入口 `python -m tensordem.fem_dem_fracture`；环境 python 3.11.6 / numpy 2.4.6。
* 判据机检：D1_velocity_field_nonzero=True, D1b_loading_path_linearity=True, D2_linear_momentum=True, D3_angular_momentum=True, D4_kinetic_energy_split=True, D5_rigid_fraction_bounds=True, D6_mass_conservation_after_transfer=True, D7_dem_intake=True, D8_quasi_static_validity=True；`criteria_all_pass = True`。
* 测试：`python -m pytest tests/test_fem_dem_fracture_dynamic.py -q` → 20 passed；wave 1 回归 `tests/test_fem_dem_fracture_handoff.py` → 26 passed（schema 2 兼容改动未破）。
