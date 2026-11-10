# 落锚拖锚贯入凹陷损伤链 —— DNV-RP-F107 闭式能量法对账

战役节点 `p22w2-n08` · 仓库 TensorDEM · 物理域：结构 / 岩土耦合 · 锚害

## 1. 案例介绍（物理背景 / 几何 / 参数表）

工程场景：船舶走锚或拖锚时，锚（3000 kg 量级）自水面以上自由下落，穿透海底
覆盖土后撞击埋设管道，造成管壁环向压扁（凹陷 dent）。判别量是最大贯入深度
`z`、管道凹陷率 `δ/D` 与规范允许值（DNV-RP-F107 无修复凹陷限值 5 % D）。

损伤链五环，每环自带解析锚：

1. **水中自由下落**：刚性锚（球 / 锥两种几何），附加质量系数 `C_a`，
   净加速度 `a_net = W' / (m (1+C_a))`，`v0 = sqrt(2 a_net h)`；
2. **土中贯入**：不排水黏土 Mohr-Coulomb 承载力 `Q_u(z) = (N_c c_u + γ' z) A(z)`，
   `A(z)` 为锚嵌入部分的最大水平截面（球：`π(2Rz-z²)` 至 `πR²` 饱和；
   锥：`π(z tanα)²` 至 `π(H tanα)²` 饱和）；
3. **管道环向压扁**：四铰机构塑性极限载荷 `P_c = 4 M_pl / R`，
   `M_pl = σ_y t² / 4`（单位轴向长度）；接触斑长度 `L_c(δ) = 2 sqrt(2 R_eq δ)`，
   故 `F_pipe = P_c L_c`、`E_dent = (4/3) P_c sqrt(2 R_eq) δ^{1.5}`（经典
   `E ∝ δ^{1.5}` 凹陷律）；
4. **动力链（冲量/动量时间积分）**：`m_eff z'' = W' − Q_u(z) − F_pipe(z − z_cov)`，
   速度 Verlet + 自适应步长，输出碰撞力时程与能量分配；
5. **DNV-RP-F107 闭式能量法**：同一工作-能量平衡的静力解
   `E_imp + W'(z_cov+δ) = E_soil + E_dent`，配无修复凹陷限值 5 % D。

| 参数 | 符号 | 取值 |
|---|---|---|
| 管外径 | D | 0.508 m |
| 壁厚 | t | 0.0206 m |
| 屈服强度 | σ_y | 448 MPa |
| 弹性模量 | E | 210 GPa |
| 塑性弯矩（单位长） | M_pl = σ_y t²/4 | 47528 N·m/m |
| 环向极限线载 | P_c = 4M_pl/R | 748 kN/m |
| 土不排水强度 | c_u | 60 kPa |
| 承载力系数 | N_c | 7.5 |
| 土浮重度 | γ' | 8.0 kN/m³ |
| 锚质量 / 等效半径 | m / R | 3000 kg / 0.450 m |
| 附加质量系数 | C_a | 0.50 |

## 2. 怎么做的（方法 / 流程 / 关键设置）

* 动力链：单自由度速度 Verlet，步长按局部切线刚度自适应（`dt ≤ 0.25 sqrt(m/k_loc)`），
  土阻力与凹陷阻力均按率无关塑性功处理（撞击后不卸载，锚静止于最大贯深处）；
  梯形速度更新使 `∫F dt = m Δv + m a_net T` 成为恒等式（E1 用它做纯积分器检验）。
* 环向压扁：以 64 边形离散圆环（节点质量、轴向罚弹簧、弹塑性转动铰 `M_pl`），
  位移控制施加对径压扁 δ，对总势能作 L-BFGS 极小化，反力 `P = k_flat(δ − flatten)`
  即对径点载荷；`P(δ)` 的最大值即塑性极限载荷。与闭式 `4M_pl/R` 对比。
* 闭式能量法：对 δ 作单调二分求解工作-能量平衡，两种口径：
  (i) 计入自重功 `W'z`（完整工作-能量平衡）；(ii) 仅计自由落体动能
  （设计常规的保守简化）。
* 情景矩阵：锚型 2 × 质量 3 × 落高 3 × 覆土 3 × 土强度 2 = 108 算例。
* 工具链：`src/tensordem/anchor_pipe_impact.py`（模型） + 本目录 `run.py`（批量、图、
  `results.json`） + `tests/test_anchor_pipe_impact.py`（23 个 pytest）。

## 3. 计算结果

### 3.1 验收判据逐项

| 判据 | 阈值 | 实算 | 结论 |
|---|---|---|---|
| E1_rebound_le_2pct | free-fall rebound height e^2 h, relative error <= 2 % | 3.04e-04 % | PASS |
| E1_momentum_le_1e-6 | momentum change vs int F dt <= 1e-6 (relative) | 2.8e-15 | PASS |
| E2_penetration_le_3pct | settled penetration depth vs analytic root <= 3 % | 0.078 % | PASS |
| E3_ring_squash_le_5pct | ring-squash limit load vs 4 M_pl / R <= 5 % | 3.29 % | PASS |
| E4_scaling_le_10pct | dent-energy scaling exponent vs plastic-work model <= 10 % | 1.67e-14 % | PASS |
| E5_agreement_ge_90pct | allow/not-allow agreement with RP-F107 closed form >= 90 % | 100.0 % | PASS |
| E6_anchor_arrested_with_soil | with soil cover the anchor is arrested before the pipe | delta/D = 0.0 % | PASS |
| E6_no_soil_penetrates | without soil the anchor crushes the pipe (delta/D > 15 %) | delta/D = 62.0 % | PASS |
| E6_no_soil_amplifies_dent | without soil the dent is larger than with soil | 62.0 % > 0.0 % | PASS |
| E6_mass_scale_ge_5pct | anchor mass x1.5 changes the penetration depth by >= 5 % | +17.2 % | PASS |
| E7_withheld_registered | no full-scale anchor-impact data -> WITHHELD registered | WITHHELD | PASS |

### 3.2 基准算例（球锚 3000 kg，落高 5 m，覆土 0.5 m，c_u = 60 kPa）

| 量 | 数值 |
|---|---|
| 到达海底顶面落速 v0 | 7.541 m/s |
| 到海底顶面动能 | 127.9 kJ |
| 最终贯入深度 z | 0.5761 m |
| 管顶凹陷 δ | 76.1 mm |
| 凹陷率 δ/D | 15.0 % |
| 土吸能 / 凹陷吸能 / 自重功 | 122.8 / 19.9 / 14.7 kJ |
| 能量账闭合残差 | -1.64e-08 |
| 峰值碰撞力 | 681 kN |
| 本链判定（δ/D ≤ 5 %） | 不允许 |
| 闭式能量法判定 | 不允许 |

覆土敏感性（同一锚 / 落高 / 土）：

| 覆土 [m] | 贯入 z [m] | δ [mm] | δ/D | 土吸能 [kJ] | 凹陷吸能 [kJ] | 判定 |
|---|---|---|---|---|---|---|
| 0.0 | 0.232 | 231.6 | 45.6 % | 28.4 | 105.5 | 不允许 |
| 0.5 | 0.576 | 76.1 | 15.0 % | 122.8 | 19.9 | 不允许 |
| 1.0 | 0.651 | 0.0 | 0.0 % | 144.6 | 0.0 | 允许 |

### 3.3 逐判据数字

* **E1**（4 组 h/e）：最大反弹高度相对误差 3.04e-04 %（阈值 2 %）；动量差 vs `∫F dt` 最大相对误差 2.8e-15（阈值 1e-6）。
* **E2**（球/锥 × 2 土强度）：静平衡贯入深度 vs 解析根 `W' = Q_u(z)` 最大相对误差 0.078 %（阈值 3 %）。
* **E3**：环向压扁极限载荷 `P_num = 723.9 kN/m` vs `4M_pl/R = 748.5 kN/m`，相对误差 -3.29 %（阈值 5 %）；96 段加密后 -3.29 %，网格敏感性 0.11 %。
* **E4**：`δ ∝ E^n`，n = 0.6667 vs 解析塑性功 0.6667，相对误差 1.67e-14 %（4 个倍增区间，阈值 10 %）。大变形区（接触斑饱和）n 升至 0.706，属模型预测而非误差。
* **E5**：108 算例，本链与闭式能量法布尔判定一致率 100.0 %（阈值 90 %）；只用落体动能的保守设计口径一致率 95.4 %。
* **E6**：去土（负控）⇒ 锚命中并贯穿管道，δ/D = 62.0 %（> 15 %），同算例带覆土时锚被土止住、管道未被触及（δ/D = 0.0 %）；锚质量 ×1.5 ⇒ 贯入深度变化 +17.2 %（解析标度 +17.2 %，阈值 ≥ 5 %）。
* **E7**：WITHHELD —— No full-scale or flume anchor-drop-on-pipeline test data exist in this worktree; DNV-RP-F107 itself is a licensed document whose tables were not verified verbatim here. Only analytic anchors (rebound, bearing capacity, ring-squash limit load, dent-energy scaling, energy closure) are claimed.

### 3.4 环向铰模型与碰撞力时程的两向耦合

把刚塑性平台换成弹塑性环向铰载荷曲线（`P(δ)` 由能量极小化环给出）后，基准算例凹陷由 76.1 mm 变为 79.4 mm（比值 1.045），δ/D = 15.6 %，能量账闭合残差 -7.3e-08（按链内实际吸收功记账）。

该算例链内实际凹陷吸能 19.0 kJ，而刚塑性平台律给出的解析凹陷功为 21.2 kJ（高 11.8 %）：耦合铰曲线在小接触斑阶段抗力偏低，故同样凹陷对应的塑性功偏小。这条差异说明用恒平台律估算凹陷功会系统偏保守，属本工作的一手发现。

## 4. 结果图

**fig1_dynamics.png** —— 贯入与碰撞力时程：覆土 0/0.5/1.0 m 三种情形（0 m 直接命中，凹陷最大；1.0 m 锚被土止住，管道未触及）。

![fig1_dynamics.png](benchmarks/anchor_pipe_impact/figs/fig1_dynamics.png)

**fig2_dent_energy.png** —— 凹陷深度 vs 凹陷功（双对数）：刚塑性闭合与解析塑性功斜率 2/3 重合；方点为耦合环向铰曲线（早期抗力低 ⇒ 凹陷更深）。

![fig2_dent_energy.png](benchmarks/anchor_pipe_impact/figs/fig2_dent_energy.png)

**fig3_verdict_map.png** —— 判定图：δ/D 随落高与锚质量变化，点线为 RP-F107 无修复限值 5 % D。

![fig3_verdict_map.png](benchmarks/anchor_pipe_impact/figs/fig3_verdict_map.png)

**fig4_verification.png** —— 四联验证图：E1 反弹高度、E2 承载力-平衡深度、E3 环向压扁载荷曲线（对经典 4M_pl/R 归一）、E6 质量标度（数值 vs 解析）。

![fig4_verification.png](benchmarks/anchor_pipe_impact/figs/fig4_verification.png)

## 5. 注意点（踩坑 / 局限 / 适用范围）

1. **动态链的五环串联是按能量守恒的准静态分配实现的**：锚在最大贯深处静止后
   不再回弹，土与凹陷均为率无关塑性。因此 E5 的“时间积分 vs 静力能量法”
   一致率检验的是**求解路径**（时程积分与静力根求解）与能量记账，而不是
   本构律本身；两条路径共用同一组 `Q_u` / `F_pipe` 闭式律。
2. **两项建模简化有明确偏差**：(i) 未计管道整体弯曲与土对管的支撑（局部环向
   压扁为主）；(ii) 未计锚的转动/拖曳方向（纯竖向）。二者都会使本链偏于保守地
   把能量集中到局部凹陷。
3. **五环中唯一的经验参数是 `N_c`、`C_a` 与 5 % D 允许值**：均按文献/规范量级
   取值并在 `results.json` 中冻结；改变 `N_c` 只改变土吸能比例，不改变判据结论
   的趋势（覆土 1.0 m 时锚被土止住）。
4. **E3 的 3 % 系统偏差来自多边形离散**：64/96/192 段结果一致（网格敏感性
   < 1 %），说明该偏差是离散铰位置（多边形弦）而非未收敛，属于本节模型的可报告
   系统偏差，未通过调参消除。
5. **DNV-RP-F107 为授权规范**：本目录实现的是其“能量法 + 5 % D 无修复凹陷限值”
   口径，公式来源为公开文献转述与本工作自洽推导，未逐字核对规范原文表格 ⇒
   见 E7 WITHHELD。

## 6. 误差判定（逐项显式回答“是否 < 3 %”）

| 关键量 | 基准 | 误差 | < 3 % ? |
|---|---|---|---|
| 反弹高度 | `e²h`（解析） | 3.04e-04 % | 是 |
| 动量-冲量恒等式 | `∫F dt`（梯形积分） | 2.8e-15（相对） | 是 |
| 静平衡贯入深度 | `W' = Q_u(z)` 解析根 | 0.078 % | 是 |
| 环向塑性极限载荷 | `4M_pl/R` 经典解 | 3.29 % | 否（阈值为 5 %，离散系统偏差） |
| 凹陷-能量标度指数 | 解析塑性功模型 | 1.67e-14 % | 是 |
| 判定一致率（108 算例） | 闭式能量法 | 不一致 0.0 % | 是（阈值 10 %） |
| 质量标度（m×1.5） | 解析动量/自重平衡 | 相对差 0.00 % | 是 |
| 与全尺度落锚试验对拍 | 实测试验数据 | —— | **无法判定（WITHHELD）** |

结论：解析锚定项（E1/E2/E4/E6）误差均在 3 % 内；环向压扁极限载荷相对经典
`4M_pl/R` 有 3.3 % 的可解释离散偏差（阈值 5 %，PASS）；与实测试验的对拍因无数据
登记 WITHHELD，未作任何“已验证”声称。

*数据来源：`benchmarks/anchor_pipe_impact/results.json`（本文件由 `run.py` 从该 JSON
程序化生成，时间戳 2026-11-10T10:22:38+00:00，代码 rev `04daaddba438`）。*
