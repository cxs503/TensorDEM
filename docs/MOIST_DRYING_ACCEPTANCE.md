# 含湿颗粒蒸发-传质耦合的滚筒干燥：湿球温度钉扎 + Ranz-Marshall 传质 + 薄层干燥动力学

*Mission m14592 / campaign node p380w2-n05 (TensorDEM, plan 380 wave 2) — 证据包自动生成于 2026-11-11T09:14:46.238410+00:00，git HEAD `6bfda4dd97513e23bfacbd26a66814f02ce2e248`，Python 3.11.6 / numpy 2.4.6，profile `full`。*

**总判定：全部 7 条机器判据通过**（criterion_pass = {"a": true, "b": true, "c": true, "d": true, "e": true, "f": true, "g": true}）；benchmark 校验行 存在未通过行：X_out cell-count convergence (GCI proxy)。

## 1 案例介绍

**物理背景。** 直接加热式顺流回转滚筒干燥机（direct-heat co-current rotary drum）。plan 380 n02 已把干颗粒的热-力耦合做通；本节点把颗粒从“干颗粒”扩展为“含湿颗粒”：每颗粒携带干基含水率 X 与温度 T_p，气相以进气温度 T_g、湿度 Y_g、气速 u_g 作边界剖面，颗粒表面的蒸发消耗显热、带走水分，进入恒速期后颗粒温度被湿球温度钉扎，越过临界含水率 X_c 后按薄层（Page/Lewis）动力学降速干燥。

**几何与运行点（全部为程序注入的算例配置，见 JSON `case_config`）。**

| 参数 | 值 | 单位 |
|---|---|---|
| 滚筒直径 D | 1.0000 | - |
| 滚筒长度 L | 5.0000 | - |
| 颗粒直径 d_p | 0.0040 | - |
| 颗粒密度 rho_p | 1250.0000 | - |
| 颗粒比热 c_ps | 1500.0000 | - |
| 干固体进料 mdot_s | 0.2500 | - |
| 停留时间 tau | 600.0000 | - |
| 轴向网格数 | 24.0000 | - |
| 进气温度 T_g,in | 400.0000 | - |
| 进气湿度 Y_g,in | 0.0100 | - |
| 表观气速 u_g | 3.0000 | - |
| 初始含水率 X_0（干基） | 0.3500 | - |
| 临界含水率 X_c | 0.1500 | - |
| 平衡含水率 X_e | 0.0200 | - |
| 吸湿活度指数 kappa | 6.0000 | - |
| 时间步长 dt | 0.1000 | - |

## 2 方法与实现

**状态量。** 颗粒相：干基含水率 X、温度 T_p（集总热容，Bi = h d_p/(2 k_p) ≈ 0.2–0.4）；气相：Y_g、h_g（焓）沿轴推进，进气为边界剖面。

**传质闭合。** Ranz-Marshall 双类比 `Sh = 2 + 0.6 Re^0.5 Sc^(1/3)`，`Nu = 2 + 0.6 Re^0.5 Pr^(1/3)`，物性取膜温 T_f = (T_p+T_g)/2；蒸发通量 `mdot = h_m A_p [a_w(X) rho_v,sat(T_p) - rho_v,g]`，恒速期 a_w = 1、降速期 a_w = exp[-kappa (X - X_c)]。

**能量方程。** `(m_s c_ps + m_w c_pl) dT_p/dt = h A_p (T_g - T_p) - mdot h_fg(T_p)`，潜热汇显式进入颗粒能量方程；气相能量与湿量以焓/含湿量通量形式推进（准稳态，因 L/u_g 比干燥时间小两个数量级）。

**湿球钉扎。** 恒速期不动点解 `h (T_g - T*) = h_m h_fg(T*) [rho_v,sat(T*) - rho_v,g]`。这是带 Lewis 因子的湿球闭式解；当 Le = 1（把 D_AB 强制为热扩散率 alpha）时它逼近经典湿球图表值（Carrier），两者之差由膜物性约定解释（见 §5）。

**账本。** 水量与焓的全时程 in/out/storage 三项记账，残差逐时步累加，另有“关掉潜热汇”的负控（negative control）证明账本检查有牙。

**工具链。** 纯 Python + numpy/scipy（曲线拟合）+ matplotlib（图）；计算入口 `python -m tensordem.rotary_dryer.moist_drying_case`；库内集成入口 `tensordem.rotary_dryer.calculate_rotary_dryer_with_drying`。

## 3 计算结果

### 3.1 七条机器判据

| criterion | measured | verdict |
|---|---|---|
| (a) wet-bulb pinning | max |T_p - T_wb| = 3.63e-06 K (< 1.0 K); max |dT_p/dt| = 1.00e-06 K/s over 30 steps | PASS |
| (b) Ranz-Marshall | |Sh(Re->0) - 2|/2 = 2.53e-04; |Sh(Re=100) - formula|/formula = 0.003 (Re_probe = 100.00) | PASS |
| (c) water ledger | relative residual 4.29e-14 (< 1.000e-09); evaporated 40.61 kg | PASS |
| (d) energy ledger | relative residual 6.88e-14 (< 0.0100) | PASS |
| (e) thin-layer kinetics | Page R^2(X) = 0.99757, R^2(dX/dt) = 0.96641 (strict shared-parameter variant 0.89752) | PASS |
| (f) degenerate dry limit | max delta vs independent dry baseline = 2.15e-11 K (< 1.000e-09) | PASS |
| (g) monotonicity | 9 points per parameter, 0 direction flips (T_g down, u_g down, feed up) | PASS |

### 3.2 参数单调性（判据 g 明细）

| sweep | parameter | points | X_out range | flips | verdict |
|---|---|---|---|---|---|
| gas_temperature | gas_inlet_temperature | 9 | 0.0612 -> 0.2535 | 0 | PASS |
| gas_velocity | gas_inlet_velocity | 9 | 0.0607 -> 0.2727 | 0 | PASS |
| solid_feed | solid_feed_dry | 9 | 0.0340 -> 0.2510 | 0 | PASS |

### 3.3 benchmark：参考值 vs 计算值（程序注入，零手抄）

| # | quantity | reference | computed | unit | error | tolerance | class | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | p_sat(0 C) | 611.2000 | 611.2129 | Pa | 2.105e-05 (relative) | 0.0020 | validation | PASS |
| 2 | p_sat(20 C) | 2339.0000 | 2338.8037 | Pa | 8.392e-05 (relative) | 0.0020 | validation | PASS |
| 3 | p_sat(50 C) | 12352.0000 | 12349.8565 | Pa | 1.735e-04 (relative) | 0.0020 | validation | PASS |
| 4 | p_sat(100 C) | 1.013e+05 | 1.014e+05 | Pa | 9.249e-04 (relative) | 0.0020 | validation | PASS |
| 5 | h_fg(100 C) | 2.257e+06 | 2.268e+06 | J/kg | 0.0050 (relative) | 0.0100 | validation | PASS |
| 6 | h_fg(20 C) | 2.454e+06 | 2.454e+06 | J/kg | 1.548e-04 (relative) | 0.0100 | validation | PASS |
| 7 | Sh(Re->0) | 2.0000 | 2.0005 | - | 2.530e-04 (relative) | 0.0200 | validation | PASS |
| 8 | Sh(Re=100, Sc=0.6) | 7.0606 | 7.0376 | - | 0.0033 (relative) | 0.0500 | validation | PASS |
| 9 | Sh(Re=100) vs Whitaker 1972 | 6.2910 | 7.0376 | - | 0.1187 (relative) | 0.1500 | diagnostic | PASS |
| 10 | Sh(Re=100) vs Froessling 1938 | 6.6346 | 7.0376 | - | 0.0607 (relative) | 0.1500 | diagnostic | PASS |
| 11 | T_wb Le=1 closure vs chart: original <0.05 K claim | 312.5377 | 310.9223 | K | 1.6154 (absolute) | 1.0000 | falsified_claim | **REFUTED CLAIM** |
| 12 | X_out (transient cells vs RK4 steady) | 0.1110 | 0.1051 | kg/kg_da | 0.0530 (relative) | 0.1500 | validation | PASS |
| 13 | T_p_out (transient cells vs RK4 steady) | 318.4986 | 320.5227 | K | 0.0064 (relative) | 0.0500 | validation | PASS |
| 14 | T_g_out (transient cells vs RK4 steady) | 318.4987 | 320.5228 | K | 0.0064 (relative) | 0.0500 | validation | PASS |
| 15 | Y_g_out (transient cells vs RK4 steady) | 0.0392 | 0.0399 | kg/kg_da | 0.0183 (relative) | 0.1000 | validation | PASS |
| 16 | water ledger residual | 0.0000 | -4.289e-14 | - | 4.289e-14 (absolute) | 1.000e-09 | validation | PASS |
| 17 | energy ledger residual | 0.0000 | -6.885e-14 | - | 6.885e-14 (absolute) | 0.0100 | validation | PASS |
| 18 | degenerate limit vs dry baseline | 0.0000 | 2.149e-11 | K | 2.149e-11 (absolute) | 1.000e-09 | validation | PASS |
| 19 | Page fit R^2 | 1.0000 | 0.9976 | - | 0.0024 (relative) | 0.0500 | validation | PASS |
| 20 | Page dX/dt R^2 | 1.0000 | 0.9664 | - | 0.0336 (relative) | 0.0500 | validation | PASS |
| 21 | X_out cell-count convergence (GCI proxy) | 0.1157 | 0.1218 | kg/kg_da | 0.0523 (relative) | 0.0500 | validation | **FAIL** |

计算出口状态（profile `full`，单程）：含水率 X_out = 0.1277 kg/kg（干基，进口 0.3500）、T_p,out = 315.2209 K、T_g,out = 317.7789 K、Y_g,out = 0.0410 kg/kg_da；全时程蒸发水量 40.61 kg。

## 4 结果图

![fig1_wet_bulb_pinning.png](artifacts/moist_drying/figures/fig1_wet_bulb_pinning.png)

*Fig 1 - single-particle constant-rate drying: T_p converges to the wet-bulb fixed point of the Ranz-Marshall pair (309.37 K, deviation 0.000 K); the classical chart wet bulb (312.54 K) sits -3.16 K away because Le = 1.178 != 1.*

![fig2_axial_profiles.png](artifacts/moist_drying/figures/fig2_axial_profiles.png)

*Fig 2 - axial moisture, particle/gas temperature and humidity profiles; two independent discretisations of the same two-phase plug-flow ODEs agree to 5.30 % on the outlet moisture.*

![fig3_thin_layer_kinetics.png](artifacts/moist_drying/figures/fig3_thin_layer_kinetics.png)

*Fig 3 - falling-rate kinetics: 4-parameter Page fit to the simulated curve (R^2 = 0.9976) and the drying rate against the Page derivative (R^2 = 0.9664).*

![fig4_sherwood.png](artifacts/moist_drying/figures/fig4_sherwood.png)

*Fig 4 - Sh(Re) with the Re->0 Maxwell limit Sh = 2 (deviation 0.025 %), the Re=100 probe against Ranz-Marshall (0.326 %) and two independent literature correlations showing the correlation spread band.*

![fig5_monotonicity.png](artifacts/moist_drying/figures/fig5_monotonicity.png)

*Fig 5 - parametric monotonicity of the outlet moisture over 9 points per parameter: 0 direction flips total.*

![fig6_ledgers.png](artifacts/moist_drying/figures/fig6_ledgers.png)

*Fig 6 - gas-solid water and enthalpy ledgers over the whole time history: relative residuals -4.29e-14 (water) and -6.88e-14 (energy), i.e. round-off level.*

## 5 注意点（踩过的坑 / 局限 / 适用域）

1. **文档里的旧声明被本节点的测量推翻（已修）。** `moist_transfer.wet_bulb_from_closures` 原 docstring 声称 Le=1 时与湿球图表值相差 <0.05 K；实测差 -3.163 K（400 K / Y=0.01）。原因是闭式比值为 `h/h_m = rho_f cp_f`（膜温物性）＝1001.3，而 Carrier 方程用气态湿比热 `cp_H` = 1024.6（−2.3%），正好对应约 1.6 K。docstring 已改为如实表述，benchmark 表保留该行作为审计线（class=falsified_claim）。**判据 (a) 用的不是这个近似值**，而是同一套 (Nu,Sh) 闭合的自洽固定点，实测偏差 3.63e-06 K。

2. **轴向离散误差是本报告的主要数值不确定度。** 一阶（全混cell）轴向离散：网格 [8, 16, 24, 32, 48] 上 X_out = [0.16427, 0.13856, 0.12766, 0.1218, 0.11574]，最细两格相对差 0.0523（容差 5%，**未通过**）。 独立 RK4 轴向推进（plug-flow，无单元混合）与 24 单元瞬态级联对比 X_out 相对差 0.0530、T_p,out 相对差 0.0064——差异方向为级联更“干”，即数值扩散类的系统偏差；工程使用应取 n_cells >= 48 或直接用 RK4 入口。

3. **降速期 dX/dt 的 Page 拟合**：对 dX/dt 直接拟合给 R² = 0.9664（判据 (e) 口径）；若强制沿用 X(t) 拟合的同一组 Page 参数，则 R² = 0.8975（< 0.95，如实列出）。也就是说机制模型（指数吸湿活度 + 传质驱动）并非严格 Page 曲线，Page 是拟合近似而非模型内核。

4. **模型边界**：颗粒为集总热容、定径无收缩/破碎；无机械 DEM 接触网络（机械流动在 `tensordem.drum_flow`）；壁面接触/辐射热源默认关闭（绝热壁直热算例）；气相准稳态。判据 (f) 的退化对照给出 X→0、Y_g→0 时与干颗粒 n02 基线的逐位可比重合（Δ = 2.15e-11 K），说明含湿闭合在干极限下不引入伪偏差。

5. **运行成本**：单元级联 24 单元 × 6000 步 ≈ 15 s/算例；全判据套件（含 27 个参数扫描点与网格收敛）在单核约 8.7 min。

## 6 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| p_sat 曲线 | IAPWS-95 表 | 9.249e-04 | 是 |
| 潜热 h_fg | ASHRAE 表 | 0.0050 | 是 |
| Sh(Re=100) | Ranz-Marshall 公式 | 0.0033 | 是 |
| Sh(Re->0) | Maxwell 精确解 2 | 2.530e-04 | 是 |
| 湿球钉扎偏差 | (Nu,Sh) 闭式固定点 | 1.174e-08 | 是 |
| X_out 瞬态级联 | 独立 RK4 稳态推进 | 0.0530 | **否** |
| T_p,out 瞬态级联 | 独立 RK4 稳态推进 | 0.0064 | 是 |
| 水量账本 | 闭合控制体（0） | 4.289e-14 | 是 |
| 能量账本 | 闭合控制体（0） | 6.885e-14 | 是 |
| 退化极限 | 干颗粒基线（0） | 2.149e-11 | 是 |

**结论：上表 10 项关键量中 9 项误差 <3%；未达 3% 的项：X_out 瞬态级联。** 需要说明：(i) 物料/传质关联式本身的文献散布（Ranz-Marshall vs Frössling 0.061）远大于 3%，因此“<3%”只能作为*与解析锚点/独立求解器*的对照口径，不能声称传质系数本身有 3% 精度；(ii) 轴向网格收敛未达 5% 容差（见 §5.2），故出口含水率的工程不确定度按 0.052 量级对待。

## 7 复现命令

```bash
python -m tensordem.rotary_dryer.moist_drying_case --profile full \
    --out runs/m14592_moist_drying/acceptance.json \
    --figdir artifacts/moist_drying/figures
python tools/moist_drying_report.py          # 生成本报告与 benchmark.json
python -m pytest tests/test_moist_drying_acceptance.py -q
```
