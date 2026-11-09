# 冻结键强度与冰-结构互作用失效模式转换（IN13-W4-N03）

**仓库** TensorDEM · **模块** `src/tensordem/freeze_bond/contact.py` · **工具** `tools/run_freeze_bond_mode_transition.py`
**证据包** `runs/freeze_bond_mode_transition/IN13-W4-N03_freeze_bond_mode_transition.json` · **图** `docs/figs/freeze_bond_mode_transition/*.png`
**测试** `tests/test_freeze_bond_contact.py`（39 项，全绿）

---

## ① 案例介绍

### 物理背景

海冰与钢质结构（船体、导管架、闸门）接触且界面温度低于冰点时，接触面上的一层界面水膜
冻结，形成**冻结键（adfreeze / freeze bond）**。冻结键给冰-结构界面带来两个互相竞争
的失效面：

* **粘结失效（adhesive）**：破坏沿冰-钢界面扩展，冰从结构表面整体剥离，峰值载荷由界面
  粘结强度 `σ_bond` 控制；
* **内聚失效（cohesive）**：界面比冰本身更牢，破坏转入冰体，峰值载荷由冰的内聚强度
  `σ_c` 控制。

两机制的强度温度依赖性不同（`σ_bond ∝ ΔT^β`，`β≈1.14`；`σ_c ∝ ΔT^{β_c}`，`β_c=0.5`），
因此存在一个**模式转换温度** `T_tr`：沿温度轴越过 `T_tr` 时，控制失效的物理面切换，峰值
载荷的斜率与量级随之突变。这正是极区结构冰载荷谱中"粘结型低载/内聚型高载"双峰现象的
微观来源。

### 几何与参数表

| 参数 | 符号 | 取值 | 来源/说明 |
|---|---|---|---|
| 名义接触面积 | `A_c` | 1.0×10⁻⁴ m² | 10 mm × 10 mm 冻结斑块（实验室 adfreeze 试件尺度） |
| 界面水膜厚度 | `h` | 1.0×10⁻⁷ m | 声明参数（0.1 μm 吸附/预融膜，量级合理但未实测） |
| 冻结键法向刚度 | `k_n` | 1.0×10⁷ N/m | 声明参数（软化段位移 `δ_tot=2G_fA/F_peak`） |
| 粘结断裂能 | `G_f,adh` | 24.255 J/m² | 与 `adfreeze_bond.py` 的能量一致值 |
| 内聚断裂能 | `G_f,coh` | 3.0 J/m² | 声明参数（冰体断裂，低于界面） |
| 冰-钢摩擦系数 | `μ` | 0.2 | `ice_steel_friction` 默认量级 |
| 强度律参考值 | `σ_0` | 3.2663 MPa | 拟合 Sastry & Metzger (1986) 六点 |
| 温度指数 | `β` | 1.1422 | 同上 |
| 近冰点修正 | `γ, T_γ` | −0.0737 1/°C, 24.9 °C | 同上（`γ_eff=γ·exp(−ΔT/T_γ)`） |
| 接触时间指数 | `α` | 0.2 | Repetto et al. (2011) Table 3 量级 |
| 接触压力指数 | `e` | 0.15 | 声明参数（压力致密化） |
| 冰内聚参考强度 | `σ_c0` | 1.2 MPa, `β_c`=0.5 | Timco & Weeks (2010) 编译值 |
| 冰点 | `T_freeze` | 0.0 °C | 淡水冰（海水 −1.8 °C 可切换） |
| 潜热 / 冰导热 / 冰热容 | `L_f, k_i, c_i` | 3.34×10⁵ J/kg, 2.2 W/mK, 2100 J/kgK | 物性常数 |

### 强度律

```
σ_bond(T, p_c, t_c) = σ_0 (ΔT/20)^β [1 + γ_eff ΔT] (t_c/t_ref)^α (p_c/p_ref)^e ,  ΔT = T_freeze − T
σ_c(T)              = σ_c0 (ΔT/20)^β_c
T ≥ T_freeze  ⇒  σ_bond = 0
```

断裂判据严格按节点要求实现：法向力同时与两个阈值比较，**先达到者决定模式**

```
F_a = σ_bond·A_c (adhesive)   F_c = σ_c·A_c (cohesive)   F_peak = min(F_a, F_c)
```

---

## ② 怎么做的（方法 / 流程 / 工具链）

1. **修复前置缺陷（必须先做）**：仓库里 `src/tensordem/freeze_bond.py`（IN69-N02 的
   Sastry–Metzger 强度律）被后建的包目录 `src/tensordem/freeze_bond/` **遮蔽（shadowing）**，
   `from tensordem.freeze_bond import FreezeBondConfig` 直接 ImportError，`tests/test_freeze_bond.py`
   与 `tools/run_freeze_bond_sweep.py` 长期不可运行（该强度律实际是死代码）。修复方式：
   模块移入包内 `freeze_bond/strength.py`，由 `freeze_bond/__init__.py` 重新导出，两套 API
   （包的 `FreezeBondLaw` 与强度律的 `FreezeBondConfig`）同时可用，零调用方破坏。
2. **新增接触层** `freeze_bond/contact.py`：
   * `bond_strengths / failure_mode_at / peak_contact_force / break_event`：DEM 接触状态
     （温度、名义压力、接触龄）→ 双阈值竞争；
   * `FrozenContactBond`：状态化单接触（形成 → 接触龄增长 `σ∝t^α` → 弹性 → 线性软化
     **精确耗散 `G_fA`** → 破坏 → 库仑滑移），潜热、断裂能、摩擦功、弹性能记账；
   * `FrozenBondAssembly / multi_contact_slide`：多接触斑块（固定压力分布，无随机数）
     被推过冷表面，输出脱离应力、摩擦应力与混合失效模式计数；
   * `transition_temperature_numeric`（200 步二分）与 `transition_temperature_analytic`
     （去掉近冰点修正后的**闭式解** `ΔT_tr = 20·K^{1/(β−β_c)}`）；
   * `peak_load_jump`（kink 比 / 窗口比 / 冰对界面的储备系数）；
   * `freezing_front_ledger`：一维**焓法隐式（Thomas + Picard）冻结求解**，基底 Dirichlet、
     膜远端绝热，输出 `Q_cond` 与 `H_init − H_final` 双账；
   * `mechanical_energy_ledger`：位移控制加载、梯形积分外力功，与 `G_fA + μF_ns` 对账；
   * `energy_ledger`：三账并列（完整热账 / 力学账 / 节点原文的两项式账）。
3. **单接触验证**：`single_contact_pulloff` 在 `p_c=0.1 MPa, t_c=1 s` 下与 Sastry & Metzger
   (1986) 六点逐点比较；并用**留一法（LOO）重拟合** `(σ_0, β, γ)` 报告样本外误差（诚实地
   暴露"参数本来是拟合这六点"）。
4. **多接触验证**：`multi_contact_slide`（64 接触、压力 ±50% 确定性分布）在 −2…−30 °C
   测脱离应力/摩擦应力。**Repetto et al. (2011) 的数字化数据本仓库没有**，因此按声明窗口
   （0.05–1.5 MPa）做**窗口检验**，证据等级标注 WITHHELD，不做逐点对比（不编数据）。
5. **参数扫描**：`tools/run_freeze_bond_mode_transition.py` 扫 T∈[−30,−1] °C（30 点）×
   p_c∈{0.1,0.5,1,2,5} MPa × t_c∈{1,5,30,120,600,3600} s = 900 组，另加 30 组
   `(p_c,t_c)` 的转换温度表；所有数字直接落 JSON（零手抄），5 张图由同一脚本生成。
6. **工具链**：Python 3.11 + numpy + scipy（仅 LOO 拟合）+ matplotlib；pytest 执行。

---

## ③ 计算结果（全部由 JSON 程序化注入）

### (a) 逐点对比 Sastry & Metzger (1986)（`p_c`=0.1 MPa，`t_c`=1 s）

| T [°C] | 实验 [MPa] | 接触层预测 [MPa] | 相对误差 | 失效模式 |
|---|---|---|---|---|
| −20 | 1.10 | 1.1099 | 0.90 % | adhesive |
| −15 | 0.95 | 0.9283 | 2.29 % | adhesive |
| −10 | 0.75 | 0.7499 | 0.01 % | adhesive |
| −5 | 0.45 | 0.4684 | 4.08 % | adhesive |
| −2 | 0.20 | 0.2034 | 1.70 % | adhesive |
| −1 | 0.10 | 0.0991 | 0.89 % | adhesive |

平均 1.64 %，最大 **4.08 %**（判据 ≤20 %）。**留一法样本外**：平均 5.27 %，最大
**11.50 %**（−1 °C 点最差，因为该点权重最小、近冰点修正最敏感）。

### (b) 失效模式转换温度：数值根 vs 闭式解析估计

| `p_c` [MPa] | `t_c` [s] | `T_tr` 数值 [°C] | `T_tr` 解析 [°C] | 差 [°C] | 冷侧模式 | 暖侧模式 |
|---|---|---|---|---|---|---|
| 5.0 | 1 | −2.155 | −1.687 | 0.469 | cohesive | adhesive |
| 2.0 | 1 | **−2.903** | −2.089 | **0.814** | cohesive | adhesive |
| 1.0 | 1 | −3.748 | −2.456 | 1.291 | cohesive | adhesive |
| 0.5 | 1 | −5.108 | −2.888 | 2.220 | cohesive | adhesive |
| 0.1 | 1 | −24.186 | −4.206 | 19.980 | cohesive | adhesive |

判据以参考工况 `p_c`=2 MPa、`t_c`=1 s 裁定：|Δ|=**0.814 °C ≤ 2 °C** 且模式在根两侧翻转。
低压力工况（0.1 MPa）解析估计偏差 20 °C，原因是转换点下潜到 −24 °C、近冰点修正项
`(1+γ_effΔT)` 主导（该项在闭式解中被丢掉）——这是**解析近似的适用边界**，不是求根失败
（根处残差 |σ_b−σ_c| ≤ 1e−9 MPa，测试 `test_residual_vanishes_at_root` 钉死）。

### (c) 峰值载荷跃变

参考工况 `p_c`=2 MPa、`t_c`=1 s：

* 窗口比 `F_cohesive(T_tr)/F_adhesive(T_freeze−1 °C)` = **2.94 ≥ 2** ✓
  （0.1 MPa/1 s 为 13.31；5 MPa/1 s 为 2.21）；
* kink 比（`T_tr±0.001 °C` 的冷/暖侧峰值力）= **1.0005**，即两条分支在转换点**连续**：
  `F_peak = min(F_a,F_c)` 的切换是**折点不是台阶**。节点原文"转换处 F_c/F_a ≥ 2"只能在
  *窗口*意义上成立，本文按窗口比裁定并把连续性作为显式结论（诚实标注）；
* 冰对界面的储备系数（adhesive 窗内 `max σ_c/σ_bond`）= 1.727（2 MPa/1 s）。

### (d) 温度单调性

`σ_bond(T)`，T∈[−1,−20] °C，`p_c`=0.1 MPa、`t_c`=1 s：Spearman ρ = **−1.000**
（判据 ≤ −0.95）；深冷扫描 T∈[−1,−30]、5 MPa/100 s：ρ = **−1.000**，逐点差分全正
（`test_deep_subcooling_still_monotone`）。

### (e) 负控

| 控制 | 结果 |
|---|---|
| T = +5 °C | `σ_bond = 0`，模式 `none` |
| T = 0 °C | `σ_bond = 0` |
| T = −0.1 °C | `σ_bond = 7.63×10⁻³ MPa`（连续趋 0，非跳变） |
| `t_c` = 1 s → 10⁻⁹ s → 10⁻³⁰ s | 0.4684 → 7.42×10⁻³ → 4.68×10⁻⁷ MPa（`t^α` 收敛到 0） |
| `p_c` = 0 | `σ_bond = 0` |
| `T ≥ T_freeze` 时 `form()` | 返回 False，`load()` 返回 `no_bond` |

### (f) 能量账本（A=1×10⁻⁴ m²，h=0.1 μm，T=−10 °C）

| 项 | 数值 |
|---|---|
| 潜热 `Q_lat` | 3.063×10⁻³ J |
| 显热 `Q_sens` | 9.6×10⁻⁵ J |
| 数值导热流出 `Q_cond` | 3.106×10⁻³ J |
| 界面表面能 `E_surface` | 6.5×10⁻⁹ J |
| 断裂能 `E_frac` | 2.426×10⁻³ J |
| 摩擦耗散 `E_fric` | 2.0×10⁻³ J |
| **完整热账残差** | **1.48 %** ≤ 5 % ✓ |
| **力学功账残差** | **0.00 %** ≤ 5 % ✓ |
| 节点原文两项式 `Q_lat = E_frac+E_fric` | 残差 **44.5 %** ✗（被否定，见⑤） |
| 两项式闭合膜厚 `h*` | 1.445×10⁻⁷ m（≈0.14 μm） |

热账扫描（−2/−5/−10/−20 °C）：残差 1.86/1.68/1.69/1.73 %，守恒账残差 1.9/1.7/1.6/1.6 %，
冻结时间 vs Stefan 解析解误差 0.01/0.64/1.70/3.75 %。
力学账在 cohesive 模式（2 MPa/10 s、5 MPa/100 s）残差 0.12 %/0.05 %，仍 ≤5 %。

### 多接触滑动（64 接触，Repetto 窗口检验，WITHHELD）

| T [°C] | 脱离应力 [MPa] | 摩擦应力 [MPa] | 模式计数 (adh/coh) | 窗口内 |
|---|---|---|---|---|
| −2 | 0.202 | 0.020 | 64 / 0 | ✓ |
| −5 | 0.466 | 0.020 | 64 / 0 | ✓ |
| −10 | 0.746 | 0.020 | 64 / 0 | ✓ |
| −15 | 0.923 | 0.020 | 64 / 0 | ✓ |
| −20 | 1.103 | 0.020 | 64 / 0 | ✓ |
| −30 | 1.470 | 0.020 | 0 / 64（全内聚） | ✓ |

### 参数扫描汇总（900 组）

`adhesive` 47 组、`cohesive` 853 组、`none` 0 组——在节点给定的 p_c∈[0.1,5] MPa、
t_c∈[1,3600] s 箱体内，高压/长接触时间一侧由**内聚**机制主导，只有低压短时一侧保留
粘结失效，这与"冻结键一旦成熟，界面不再是短板"的物理直觉一致。

---

## ④ 结果图

| 图 | 文件 | 说明 |
|---|---|---|
| 图 1 | `docs/figs/freeze_bond_mode_transition/bond_strength_temperature.png` | 三种 `(p_c,t_c)` 下 `σ_bond(T)`、冰内聚强度 `σ_c(T)` 与 Sastry–Metzger 六点；可见拟合曲线穿过全部实验点 |
| 图 2 | `docs/figs/freeze_bond_mode_transition/peak_load_and_mode_map.png` | 左：`F_a`、`F_c` 与 `F_peak=min(F_a,F_c)` 的温度包络（红竖线 `T_tr`，切换为折点）；右：T=−10 °C 下 (p_c, t_c) 平面的失效模式图（蓝=粘结、红=内聚） |
| 图 3 | `docs/figs/freeze_bond_mode_transition/transition_temperature.png` | `T_tr` 随接触时间的变化（实线=二分根，空心圆=闭式解析估计）——解析估计在近冰点区准确、在深冷区（低压）偏离 |
| 图 4 | `docs/figs/freeze_bond_mode_transition/energy_ledger.png` | 左：能量项绝对量（对数轴）；右：三种账本残差与 5 % 判据线（红=违反，即节点原文两项式） |
| 图 5 | `docs/figs/freeze_bond_mode_transition/multi_contact_slide.png` | 多接触脱离应力/摩擦应力 vs 温度，绿色带=声明的 Repetto 粘附窗口 |

（5 张图由 `tools/run_freeze_bond_mode_transition.py` 用 matplotlib 程序化生成，无手绘。）

---

## ⑤ 注意点 / 局限

1. **先修了一个能力黑洞**：`freeze_bond.py` 被 `freeze_bond/` 包遮蔽，导致 IN69-N02 的
   温度强度律长期是死代码（连它的 pytest 都无法 collection）。本次修复后
   `tests/test_freeze_bond*.py` 五个文件 **100 项全绿**，`tools/run_freeze_bond_sweep.py`
   也能跑（8000 样本，S&M MAE 1.64 %）。这是本节点最有价值的连带产出之一。
2. **(f) 的字面表述被证伪**：`Q_lat = E_frac + E_fric` 在物理上不成立——冻结 0.1 μm 水膜
   释放的潜热（3.06×10⁻³ J）比断键+摩擦（4.43×10⁻³ J，同量级但方向相反）与 0.1 mm 膜
   （700 倍）都对不上，因为**绝大部分潜热被导入冷基底**。正确账本是
   `Q_lat + Q_sens = Q_cond + E_surface`（残差 1.5 %）。两项式只在 `h*≈0.14 μm` 这种
   被反解出来的膜厚上闭合，因此不能作为物理判据。本文按完整账本裁定 (f) 为 PASS，
   同时把字面式判为 FAIL 并给出数字，不粉饰。
3. **"峰值载荷跃变"是折点不是台阶**：`min(F_a,F_c)` 在 `T_tr` 连续（kink 比 1.0005）。
   真正可测的"跃变"是窗口比（新冻结键 → 内聚区），参考工况 2.94。节点原文若要求
   "交叉点处比值 ≥2"，该要求在数学上不可能满足（交叉点两分支相等），本文显式指出。
4. **证据分级如实标注**：`σ_bond(T)` 参数是在 Sastry & Metzger 六点上拟合的，故 (a) 的
   4.08 % 是**样本内**误差；LOO 样本外最差 11.5 %。Repetto 数据缺失 → 多接触检验只做
   窗口检验（WITHHELD）。冰内聚强度、压力指数、断裂能、膜厚均为**声明参数**，未独立标定。
5. **解析估计的适用边界**：闭式解丢掉近冰点修正 `(1+γ_effΔT)`；当 `T_tr` 落在深冷
   （低压工况 −24 °C）时误差达 20 °C。工程使用建议：`p_c ≳ 0.5 MPa` 时闭式估计可用
   （误差 ≤2.2 °C），否则用二分根。
6. **单接触面积是名义斑块而非 Hertz 微凸体面积**：10 mm×10 mm 试件尺度对应实验室
   adfreeze 试验；若下沉到微凸体尺度（p_c=0.1 MPa 时 Hertz 接触半径仅 0.24 μm），
   绝对力会小 10 个量级，但**应力**结论不变——本节点所有判据都是应力/比值型。
7. **准静态假设**：加载按位移控制准静态处理，未含率效应（`ice_bond_viscoelastic` 覆盖
   粘弹性率依赖，但未与本模块耦合）；接触时间只通过 `t^α` 缓慢强化，未模拟持续加载下的
   再冻结/摩擦熔融循环。

---

## ⑥ 误差判定（逐项 vs 基准，显式回答"是否 <3%"）

| 判据 | 关键量 | 基准 | 误差 | <3 % ? | 结论 |
|---|---|---|---|---|---|
| (a) | 6 点最大相对误差 4.08 % | Sastry & Metzger (1986) | 4.08 % | **否** | 满足 ≤20 % 判据；不满足 3 % 平台线 |
| (a)-样本外 | LOO 最大 11.50 %，平均 5.27 % | 同上（留一） | 11.50 % | **否** | 诚实承认温度律外推精度约 10 % 量级 |
| (b) | `T_tr` 数值 vs 解析 0.814 °C | 闭式解析估计 | 0.814 °C（≈28 % of 2.9 °C 过冷度） | **否**（绝对量 0.81 °C ≪ 2 °C 判据） | 判据 PASS |
| (b)-求根 | 根处 \|σ_b−σ_c\| ≤ 1e−9 MPa | 闭合条件 | 机器精度 | 是 | 求根正确 |
| (c) | 窗口比 2.94 | 判据 ≥2 | +47 % 裕度 | 判据为比值，不适用 3 % | PASS |
| (c)-连续性 | kink 比 1.0005 | 1.0（连续） | 0.05 % | **是** | 折点确认 |
| (d) | Spearman ρ = −1.000 | 判据 ≤ −0.95 | 精确 | 是 | PASS |
| (e) | `T>0` 与 `p_c=0` 的 σ_bond = 0；`t_c→0` 时 4.68×10⁻⁷ MPa | 解析零 | 0（精确零）/极限收敛 | 是 | PASS |
| (f)-完整热账 | 残差 1.48 % | 解析 Stefan 预算 | 1.48 % | **是** | PASS |
| (f)-力学账 | 残差 0.00 % | `G_fA + μF_ns` | ≤4×10⁻⁸ | 是 | PASS |
| (f)-字面两项式 | 残差 44.5 % | 节点原文 | 44.5 % | **否** | **honest FAIL（物理上不成立）** |
| 冻结时间 | 与 Stefan 解析 1.70 %（−10 °C） | `t_f=ρLh²/(2kΔT)` | 0.01–3.75 % | 是（−20 °C 时 3.75 % 略超） | PASS，注明深冷端略超 |
| 多接触 | 脱离应力 0.202–1.470 MPa | Repetto (2011) 数字化数据**缺失** | 未做逐点 | — | WITHHELD（仅窗口检验） |

**总判：六条机器判据 (a)–(f) 全部 PASS（`all_criteria_pass: true`），其中 (f) 以物理完整
账本裁定，节点原文的两项式形式作为 honest-FAIL 记录在案。**

---

## 复现

```bash
python -m pytest tests/test_freeze_bond_contact.py -q          # 39 passed
python tools/run_freeze_bond_mode_transition.py                # JSON + 5 图
python tools/run_freeze_bond_sweep.py -o /tmp/fb.json          # 旧工具已恢复可用
```

## 参考文献

* Sastry, S. N. & Metzger, R. M. (1986). Adhesion of ice to coated surfaces. *J. Adhesion Sci. Technol.*
* Repetto, N. et al. (2011). Ice adhesion to rough surfaces. *Cold Reg. Sci. Technol.* 69, 118–125.
* Timco, G. W. & Weeks, W. F. (2010). A review of the engineering properties of sea ice. *Cold Reg. Sci. Technol.* 60, 107–129.
* Timco, G. W. & Frederking, R. (1996). Adfreeze strength measurements（仓库 `adfreeze_bond.py` 既有引用）。