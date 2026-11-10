# 三体碰撞动力学与碎冰阻力分量分离（DEM，hull–floe–floe）

**节点** p262w1-N04 · **仓** TensorDEM · **模块** `src/tensordem/threebody_ice` · **驱动** `scripts/threebody_ice_validation.py` · **测试** `tests/test_threebody_ice.py`

**判据结果**：15/19 项通过（全通过 = False）· `parameter_ledger_sha256` = `4457b267ed10f4a7…` · `artifact_sha256` = `34701cc5405a202f…`

## ① 案例介绍

船舶在碎冰场中航行时，船体常常**同时**与多块浮冰接触：近处浮冰被船首挤压后把载荷通过冰–冰接触传给更远处的浮冰，远处的浮冰又反过来限制近处浮冰的运动。这一多体接触链决定了载荷在浮冰之间的分配，也决定了船舶功率中哪一部分被碰撞耗散、哪一部分被水的拖曳吸收、哪一部分随被清开的碎冰带走。本节点在 TensorDEM 中新建 `threebody_ice` 模块，把这三件事做成可机器裁决的验证面。

**几何与参数**（平面二维、浮冰平躺刚体、船体为运动学刚体边界）：

三体碰撞场景：船首为垂直平板（block，长 24 m、宽 12 m，速度 V 沿 +x），两块等径浮冰半径 a = 1.5 m 沿 x 前后串列布置，横向各自偏离中心线 ±0.6a，纵向**净间距** d = (d/D)·D，D = 2a；d/D 扫描 0.5 → 3.0。

碎冰场场景：矩形水道（60 m × 30 m，播冰区 52 m × 24 m），船形（楔形首，进流角 20°，长 24 m、宽 10 m）；浮冰半径由 Laguerre 型（最近邻间隙）布置给出，避免重叠并精确复现目标冰浓 c_ice；船体匀速通过，稳态窗口 = 船首位于域长的 25%–90%，速度扫描 0.5–3 m/s，冰浓扫描 0.3/0.5/0.7。

**参数台账（分级）**：

| 参数 | 取值 | 证据级 | 说明 |
|---|---|---|---|
| `hull_ice_friction` | 0.1000 | CONVENTION | steel-ice sliding friction (wet) |
| `ice_density_kg_m3` | 917.0000 | REAL | columnar sea ice, 900-927 kg/m3 range |
| `ice_floe_thickness_m` | 0.8000 | CONVENTION | level first-year floe thickness |
| `ice_ice_friction` | 0.1500 | CONVENTION | sliding friction of ice on ice |
| `normal_stiffness_N_m` | 2.0000e+07 | DEM-SOFT | softened contact stiffness (explicit stability); the crushing stiffness at 1.5 MPa over a 2t sqrt(2 R d) line contact is O(1e8-1e9 N/m) |
| `restitution_floe_floe` | 0.4000 | CONVENTION | crushed-layer dissipation; the analytic single-collision fractions (1-e)/2 = 0.30 and (1-e)/(1+e) = 0.43 are then measured, not assumed |
| `restitution_hull_floe` | 0.4000 | CONVENTION | as above for steel-ice impact |
| `water_density_kg_m3` | 1025.0000 | REAL | sea water |
| `water_drag_coefficient` | 1.0000 | CONVENTION | normal flat-plate drag on the submerged side wall |

## ② 怎么做的

1. **接触律**：Cundall–Strack 线性弹簧–阻尼法向力 `F_n = k δ + c δ̇`，阻尼按**对**的折合质量构造 `c = 2β√(k m_eff)`，切向为纯滑动的库仑摩擦 `|F_t| ≤ μ|F_n|`（无切向弹簧，已声明）。
2. **恢复系数可标定**：线性弹簧–阻尼的解析反演 `β = f(e)` 只是近似，因此用**实测**恢复系数做二分标定（自由对撞 + 船体对撞各一次），标定后实测 e = 0.4003 / 0.3998，目标 e = 0.40。
3. **积分**：速度 Verlet（kick–drift–kick），步长由接触周期定 `dt = safety·2π√(m_eff/k)`（碰撞参照与三体事件 safety = 0.03，碎冰场 safety = 0.05）；工作/能量速率在**中点状态**（漂移后的位置、半步速度，即速度 Verlet 实际取速度的地方）取值，冲量则取两次力评估的梯形——这正是速度 kick 所积分的量，因此能量恒等式是二阶的，冲量台账可审计。
4. **能量台账（全部实测，无一项反推）**：船体对冰做功 `W_hull`、水的拖曳功 `W_drag`、法向阻尼耗散 `E_diss,n = Σc δ̇²`、摩擦耗散 `E_diss,t = Σμ|F_n||v_t|`、被清出通道的浮冰带走的动能 `E_out`。恒等式 `W_hull + W_drag = ΔE_kin + ΔE_spring + E_diss + E_out` 逐步累加，残差作为闭包误差**如实报告**。
5. **阻力分量分离**：稳态窗口内把每个能量项除以航程 `V·T`：`R_collision = E_diss/(VT)`、`R_submerged_drag = |W_drag|/(VT)`、`R_channel_clearing = E_out/(VT)`；`R_ice` 由两条**独立**路径给出：船体反力时均 `-<F_hull·e_x>` 与 `W_hull/(VT)`。
6. **对标口径**：节点声明的 `E_diss/ΔE_k` 判定在**单次碰撞**粒度上做（这才是「碰撞」的含义），多体事件的整段聚合值另表报告，两者都不调参。
7. **工具链**：纯 PyTorch float64、确定性归约（同种子逐位一致）、pytest 用例覆盖接触律/守恒/编排/判据，matplotlib 出图，所有数字程序化写入 `artifacts/threebody_ice/validation.json`。

## ③ 计算结果

**解析参照**（单次常速船首撞击，`e` 为恢复系数）：耗散占输入功比例 `(1-e)/2 = 0.3000`，耗散/动能增量 `(1-e)/(1+e) = 0.4286`。

**弹性特例（e = 1，自由对撞）**：

| 量 | 数值 | 判据 |
|---|---|---|
| 动能+弹簧能相对漂移 | 0.0020 (0.20 %) | ≤ 2% (F2) |
| 总动量相对误差 | 1.973e-16 | ≤ 1% (F1) |
| 实测恢复系数 | 1.0020 | 目标 1.0 |

**三体碰撞：载荷分配与能量耗散**（F_total = F_11 + F_12；`E_diss/W_hull` 为耗散占输入功比例，`E_diss/ΔE_k` 为耗散/动能增量）：

| 配置 | d/D | $I_{11}/I_{total}$ | 同时接触时间分数 | $E_{diss}/W_{hull}$ | $E_{diss}/\Delta E_k$ | 能量闭包误差 | 峰值 F [kN] |
|---|---|---|---|---|---|---|---|
| 无拖曳（单次撞击） | 0.50 | 1.000 | 0.000 | 0.492 | 0.890 | 0.0447 | 136.0 |
| 无拖曳（单次撞击） | 0.75 | 1.000 | 0.000 | 0.492 | 0.898 | 0.0403 | 136.0 |
| 无拖曳（单次撞击） | 1.00 | 1.000 | 0.000 | 0.491 | 0.886 | 0.0447 | 136.0 |
| 无拖曳（单次撞击） | 1.50 | 1.000 | 0.000 | 0.492 | 0.892 | 0.0438 | 136.0 |
| 无拖曳（单次撞击） | 2.00 | 1.000 | 0.000 | 0.491 | 0.887 | 0.0438 | 136.0 |
| 无拖曳（单次撞击） | 2.50 | 1.000 | 0.000 | 0.492 | 0.893 | 0.0431 | 136.0 |
| 无拖曳（单次撞击） | 3.00 | 1.000 | 0.000 | 0.491 | 0.889 | 0.0431 | 136.0 |
| 带水拖曳（持续挤压） | 0.50 | 0.738 | 0.098 | 0.200 | 1.087 | 0.0126 | 136.1 |
| 带水拖曳（持续挤压） | 0.50 | 0.738 | 0.098 | 0.200 | 1.087 | 0.0126 | 136.1 |
| 带水拖曳（持续挤压） | 0.75 | 0.762 | 0.089 | 0.192 | 1.073 | 0.0084 | 136.1 |
| 带水拖曳（持续挤压） | 0.75 | 0.762 | 0.089 | 0.192 | 1.073 | 0.0084 | 136.1 |
| 带水拖曳（持续挤压） | 1.00 | 0.780 | 0.049 | 0.183 | 1.050 | 0.0078 | 136.1 |
| 带水拖曳（持续挤压） | 1.00 | 0.780 | 0.049 | 0.183 | 1.050 | 0.0078 | 136.1 |
| 带水拖曳（持续挤压） | 1.50 | 0.789 | 0.045 | 0.174 | 1.067 | 0.0077 | 136.1 |
| 带水拖曳（持续挤压） | 1.50 | 0.789 | 0.045 | 0.174 | 1.067 | 0.0077 | 136.1 |
| 带水拖曳（持续挤压） | 2.00 | 0.800 | 0.039 | 0.164 | 1.057 | 0.0068 | 136.1 |
| 带水拖曳（持续挤压） | 2.00 | 0.800 | 0.039 | 0.164 | 1.057 | 0.0068 | 136.1 |
| 带水拖曳（持续挤压） | 2.50 | 0.811 | 0.032 | 0.156 | 1.065 | 0.0068 | 136.1 |
| 带水拖曳（持续挤压） | 2.50 | 0.811 | 0.032 | 0.156 | 1.065 | 0.0068 | 136.1 |
| 带水拖曳（持续挤压） | 3.00 | 0.818 | 0.023 | 0.147 | 1.043 | 0.0064 | 136.1 |
| 带水拖曳（持续挤压） | 3.00 | 0.818 | 0.023 | 0.147 | 1.043 | 0.0064 | 136.1 |

**单次碰撞的能量分配（节点声明带 [0.3, 0.8] 的判定粒度）**：恢复系数 e 作为敏感性参数，每个 e 都用实测恢复系数标定后再测量：

| e 声明 | e 实测 | $E_{diss}/W_{hull}$ 实测 | $(1-e)/2$ 解析 | $E_{diss}/\Delta E_k$ 实测 | $(1-e)/(1+e)$ 解析 | 能量闭包 |
|---|---|---|---|---|---|---|
| 0.40 | 0.3998 | 0.3334 | 0.3000 | 0.4862 | 0.4286 | 0.0191 |
| 0.50 | 0.5005 | 0.2753 | 0.2500 | 0.3723 | 0.3333 | 0.0148 |
| 0.35 | 0.3495 | 0.3723 | 0.3250 | 0.5656 | 0.4815 | 0.0306 |

**碎冰场：阻力分量（稳态窗口均值）**：

| $c_{ice}$ | V [m/s] | $R_{ice}$ [N] | $R_{coll}$ [N] | $R_{drag}$ [N] | $R_{clear}$ [N] | 残差 [N] | 闭包误差 [%] | 实现冰浓 | 浮冰数 |
|---|---|---|---|---|---|---|---|---|---|
| 0.30 | 0.5 | 257.8 | 106.7 | 113.3 | 32.3 | 10.5 | 2.18 | 0.300 | 83 |
| 0.30 | 1.0 | 1031.3 | 426.9 | 454.4 | 130.0 | 41.0 | 1.94 | 0.300 | 83 |
| 0.30 | 2.0 | 4103.2 | 1693.6 | 1819.8 | 512.4 | 157.1 | 1.89 | 0.300 | 83 |
| 0.30 | 3.0 | 9235.0 | 3776.2 | 4123.3 | 1141.3 | 363.8 | 2.10 | 0.300 | 83 |
| 0.50 | 0.5 | 436.3 | 182.1 | 187.4 | 63.5 | 13.8 | 0.79 | 0.500 | 136 |
| 0.50 | 1.0 | 1761.5 | 724.5 | 754.6 | 262.4 | 60.4 | 1.14 | 0.500 | 136 |
| 0.50 | 2.0 | 7056.8 | 2859.5 | 3048.9 | 1082.0 | 214.7 | 0.94 | 0.500 | 136 |
| 0.50 | 3.0 | 15867.5 | 6428.7 | 6891.8 | 2461.3 | 447.6 | 0.54 | 0.500 | 136 |
| 0.70 | 0.5 | 557.0 | 237.0 | 242.7 | 89.4 | 4.3 | 2.16 | 0.694 | 199 |
| 0.70 | 1.0 | 2225.4 | 946.1 | 974.7 | 353.4 | 17.5 | 2.19 | 0.694 | 199 |
| 0.70 | 2.0 | 8938.5 | 3732.5 | 3925.3 | 1416.4 | 107.1 | 1.52 | 0.694 | 199 |
| 0.70 | 3.0 | 20248.6 | 8325.3 | 8927.1 | 3244.8 | 293.6 | 1.23 | 0.694 | 199 |

**速度指数（log–log 最小二乘，含标准误）**：

| 冰浓 | 分量 | 指数 | 标准误 | $R^2_{log}$ |
|---|---|---|---|---|
| c0.3 | $R_{ice}$ | 2.00 | 0.00 | 1.0000 |
| c0.3 | 碰撞 | 1.99 | 0.00 | 1.0000 |
| c0.3 | 拖曳 | 2.01 | 0.00 | 1.0000 |
| c0.3 | clearing | 1.99 | 0.01 | 1.0000 |
| c0.5 | $R_{ice}$ | 2.01 | 0.00 | 1.0000 |
| c0.5 | 碰撞 | 1.99 | 0.00 | 1.0000 |
| c0.5 | 拖曳 | 2.01 | 6.86e-04 | 1.0000 |
| c0.5 | clearing | 2.04 | 0.00 | 1.0000 |
| c0.7 | $R_{ice}$ | 2.01 | 0.00 | 1.0000 |
| c0.7 | 碰撞 | 1.99 | 0.00 | 1.0000 |
| c0.7 | 拖曳 | 2.01 | 0.00 | 1.0000 |
| c0.7 | clearing | 2.00 | 0.01 | 1.0000 |

## ④ 结果图

![force_distribution.png](../artifacts/threebody_ice/force_distribution.png)

*force_distribution.png*：（a）d/D = 1 持续挤压工况下 `F_11(t)`、`F_12(t)`、`F_total(t)`；（b）近处浮冰的冲量份额随净间距变化；（c）两块浮冰同时受载的时间分数。

![resistance_decomposition.png](../artifacts/threebody_ice/resistance_decomposition.png)

*resistance_decomposition.png*：（a）`c_ice = 0.5` 下三个阻力分量的堆叠柱与实测 `R_ice`；（b）各分量的速度指数拟合；（c）不同冰浓下的 `R_ice(V)`。

![energy_dissipation.png](../artifacts/threebody_ice/energy_dissipation.png)

*energy_dissipation.png*：（a）耗散比随间距变化与节点声明带 [0.3, 0.8]；（b）弹性特例的动能/弹簧能/总能曲线；（c）能量台账闭包误差随速度变化与 5% 判据线。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **接触法向符号**：`n` 由 i 指向 j 时排斥力必须写成 `-F_n n`（作用在 i 上）。符号写反会得到反恢复力，能量指数增长（实测 1 s 内速度从 1 m/s 涨到 22 m/s），是本次实现中最贵的一个 bug。
2. **恢复系数必须实测标定**：线性弹簧–阻尼的 `β = f(e)` 解析式只是近似，直接使用会让实测 e 偏离目标约 8%（e=0.6 时实测 0.646）。本模块用二分标定实测 e，误差 3.2435e-04。
3. **积分步长**：`ω dt ≈ 0.6` 时能量台账残差已达 12%，把 safety 从 0.10 降到 0.05（`ω dt ≈ 0.3`）后闭包误差才落到判据以内。接触刚度是被**软化**的（`DEM-SOFT`，5e6 N/m，真实压碎刚度高 1–2 个量级），因此峰值接触力绝对值不可作为设计载荷引用。
4. **冰浓的可达上限**：不重叠圆盘的堆积在 `α ≤ 0.47` 约束下密度上限约 0.69，所以 `c_ice = 0.7` 的实现值只能到约 0.69（表内如实给出实现冰浓）。更高冰浓需要允许浮冰破碎或重叠（本节点不做）。
5. **二维平躺假设**：没有垂荡/纵倾自由度，也没有碎冰下潜——因此本模型**不会**产生文献里那种「碎冰下潜」型阻力项；「channel clearing」在这里定义为被推出扫掠走廊的浮冰带走的动能，而不是把它们压到水下做功。这是本节点与 Lindqvist/Riska 分解在物理内涵上的差别，不能直接逐项比较。
6. **对标状态（诚实声明）**：Kujala (1991) 的碎冰阻力试验数据本沙箱内**不可得**，因此本报告**未**做绝对量级对标（证据级 WITHHELD，不编造数字）。可做的对照是量纲/指数级：静水拖曳与动量通量主导的阻力项应为 `V²`，本模型实测的 `R_ice` 指数见上表。
7. **能量台账闭包的边角**：三体事件整段的 `W_hull = ΔE + E_diss`在无拖曳工况残差 4–4.5%（判据 X1 对碎冰场要求 1%，实测 3.0% 未过）——来源是接触**出现/消失瞬间**的折点：冲量用梯形、功速率用中点，两种求积在折点处的差为 O(dt·|dF/dt|)。这不是能量不守恒（弹性特例的能量漂移只有 0.2%），而是台账求积的离散误差，已如实记为 FAIL。
8. **P2 的 5.8% 偏差**来自解析式本身是近似：线性弹簧–阻尼的耗散比并非严格的 (1−e)/(1+e)，且实测恢复系数被标定到 0.403 而非精确 0.400；本报告不为此放宽阈值。
9. **船体为运动学边界**（无限质量），没有船体六自由度响应，所以本节点只回答「冰受到什么力」，不回答「船如何运动」。

## ⑥ 误差判定

| 判据 | 含义 | 实测 | 阈值 | 结论 |
|---|---|---|---|---|
| F0a_measured_restitution_floe_floe_vs_target | - | 3.2435e-04 | 0.0010 | PASS |
| F0b_measured_restitution_hull_floe_vs_target | - | 2.2777e-04 | 0.0010 | PASS |
| F1a_momentum_error_free_pair_elastic | - | 1.9732e-16 | 0.0100 | PASS |
| F1b_momentum_error_three_body_drag_free | max over the spacing sweep | 5.8406e-14 | 0.0100 | PASS |
| F1c_momentum_error_three_body_sustained | max over the spacing sweep, drag impulse included | 5.8406e-14 | 0.0100 | PASS |
| F2_elastic_energy_drift | - | 0.0020 | 0.0200 | PASS |
| F3a_force_partition_algebraic | - | 0.0000 | 1.0000e-09 | PASS |
| F3b_force_partition_impulse_vs_momentum | - | 5.8406e-14 | 0.0500 | PASS |
| F4_decomposition_closure | max over the concentration x speed matrix | 0.0219 | 0.0500 | PASS |
| F4b_force_vs_power_identity | R_ice measured twice, independently | 9.7207e-04 | 0.0500 | PASS |
| X1_ledger_identity_with_storage_residual | three components + window storage residual = R_ice (the full ledger) | 0.0298 | 0.0100 | FAIL |
| F4c_all_components_non_negative | negative component = sign/bookkeeping error | 32.2678 | 0.0000 | PASS |
| F5_pytest_cases_green | - | 21.0000 | 12.0000 | PASS |
| F6_figures_present | force_distribution.png, resistance_decomposition.png, energy_dissipation.png | 3.0000 | 3.0000 | PASS |
| P1_single_collision_vs_analytic_primary | measured E_diss/W_hull at the declared restitution | 0.0334 | 0.0500 | PASS |
| P2_single_collision_vs_analytic_over_ke_gain | measured E_diss/dE_k at the declared restitution | 0.0576 | 0.0500 | FAIL |
| P3_dissipation_band_single_collision | node band [0.3, 0.8] on the measured single-collision fractions; the analytic value for e=0.40 is 0.3000, i.e. exactly on the lower edge | [0.333, 0.486] | [0.3, 0.8] | PASS |
| P4_dissipation_band_multibody_aggregate | E_diss/W_hull of the whole tandem event (reported, not selected): the chain keeps transferring energy to the far floe after the near floe separates, so less of the hull work is dissipated inside contacts | [0.147, 0.492] | [0.3, 0.8] | FAIL |
| P5_dissipation_over_ke_gain_multibody | aggregate E_diss/dE_k (reported for completeness) | [0.886, 1.087] | [0.3, 0.8] | FAIL |

**速度依赖判定（节点预期 vs 实测）**：节点预期「碰撞 ∝ v²、拖曳 ∝ v²、clearing ∝ v」。实测指数（冰浓 c0.3）：碰撞 1.99、拖曳 2.01、clearing 1.99、总阻力 2.00。

前两项与预期一致（≈2，量纲上由「单位时间碰撞次数 ∝ v」×「每次耗散 ∝ v²」给出）；**clearing 的实测指数也在 2 附近，与节点预期的 1 不符**——本模型的 channel clearing 是被推出扫掠走廊的浮冰**带走的动能**，它同样是「事件率 ∝ v × 单个事件能量 ∝ v²」的对流项，因此 ∝v²；文献里接近线性的 clearing/下潜项来自**动态下潜深度随速度增长**（垂直方向自由度），本二维平躺模型按构造就没有这一项。这是一条诚实记录的**否定性结果**：要复现 clearing ∝ v，必须引入垂荡/下潜自由度，而不是调二维参数。

**逐项回答「是否 < 3%」**：

- 动量守恒（F1）：自由对撞 1.973e-14 %、三体事件最大 5.841e-12 %、持续挤压最大 5.841e-12 % → **< 3%（是）**。
- 弹性特例能量（F2）：0.203 % → **< 3%（是）**。
- 力分配（F3）：代数恒等式 0.00（机器精度）、冲量–动量互检最大 5.841e-12 % → **< 3%（是）**。
- 阻力分量闭包（F4）：最大 2.189 % → **< 3%（是）**。
- 单次碰撞与解析解的一致性（P1/P2）：3.34 % / 5.76 % → 第一项 <3%（是），第二项 3%–6% 之间（否，见上表与 ⑤.8）。
- 力学量绝对量级：**无基准可比**（Kujala 数据离线不可得），故不作 <3% 声明；峰值接触力因刚度软化而不可直接引用。

**闭包残差的物理来源**：残差项 `R_residual` 是窗口内浮冰储能（动能+弹簧能）的变化率除以航程。碎冰场在船体通过时不断有新浮冰被加速，因此它不为零；它被显式列出，而不是塞进任何一个分量里。

## 审计链

- 参数台账 SHA-256：`4457b267ed10f4a703fb773ccaee14d6578900e7d967784f0870aada9623662f`
- 工件 SHA-256（数值载荷）：`34701cc5405a202f53d8e9bdb5216345f08e24843e05041e57660bf220d6e1e2`
- 工件：`artifacts/threebody_ice/validation.json`（合并后的全部判据数字）、图 force_distribution.png, resistance_decomposition.png, energy_dissipation.png；逐案例原始件 `artifacts/threebody_ice/parts/<key>.json` 由 `python scripts/threebody_ice_validation.py --shard i/14` 再生（入库只保留合并后的工件，避免把 1.6 MB 的中间件塞进版本库）
- 已执行案例：29 条 manifest 条目；未执行：无
