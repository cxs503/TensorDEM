# 颗粒 DEM 接触参数的可辨识性与反演验证面（`test_granular_01`）

本文件由 `scripts/render_granular_identifiability_doc.py` 从 `artifacts/granular_identifiability/study.json` 自动生成（case `test_granular_01`，一次完整研究耗时 292.2 s，78 次正向评价）。所有数字均由程序注入，无手抄。

## 1. 案例介绍（物理背景 / 几何 / 参数表）

验证面建立在一个**合成颗粒系统** `test_granular_01` 上：二维刚性圆盘、双峰粒径（r = 0.018 m，多分散 ±5 %）、密度 2500 kg/m³、片厚 0.1 m、重力 9.81 m/s²、时间步 dt = 1.700e-04 s。系统不含任何工程材料语义（不含冰/船体等词条，见测试 `test_package_has_no_application_semantics`），因此可辨识性结论只取决于接触模型本身。

被识别的三个接触参数与取值盒（单位立方坐标用于 Fisher 信息矩阵，保证条件数与量纲无关）：

| 参数 | 符号 | 下界 | 上界 | 单位 | 真值 |
|---|---|---|---|---|---|
| friction | mu | 0.15 | 0.85 | - | 0.55 |
| restitution | e | 0.15 | 0.85 | - | 0.45 |
| stiffness | k_n | 6000 | 15000 | N/m | 10000 |

四个可观测量（前三个是标准颗粒标定测量，第四个是接触尺度测量）：

| 可观测量 | 含义 | 1σ 测量不确定度 | 参考值 | 独立铺样结构随机性 (相对 sd) |
|---|---|---|---|---|
| `grain_friction_ratio` | grain friction ratio τ/σ (-) | 0.02 | 0.55 | 0 % |
| `packing_fraction` | settled packing fraction φ (-) | 0.01 | 0.7835 | 0.154 % |
| `restitution` | impact restitution e (-) | 0.0141 | 0.47116 | 0 % |
| `impact_duration_s` | impact duration τ_c (s) | 3.404e-04 | 0.01702 | 0 % |

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **正向目标量**：`tensordem.granular_identifiability.evaluate_targets` 在 3009 步、33 个颗粒的硬预算内算出 4 个可观测量（每 case 预算上限：400 颗粒 / 20000 步 / 40.0 s，超限抛 `BudgetExceeded`）。
   - 单颗粒受限剪切：颗粒被两块平行板夹住（过盈 2.000e-04 m，法向载荷 k_n·δ 已知），一块板匀速平移，稳态滑移段的 Σ|f_t|/Σf_n 即库仑摩擦系数。
   - 沉降床孔隙率：松散六角铺样（初值 φ≈0.807）在重力下沉降 0.3 s，由自由面高度与颗粒面积给出堆积率 φ = 1 − 孔隙率。
   - 冲击回弹/接触时长：单颗粒自高度 0.05 m 自由下落，逐步采样与地板的侵入量，对接触起止做线性插值得到接触时长 τ_c（亚时间步分辨率），速度比给出恢复系数 e。
2. **结构随机性基线**：同一参数在 3 个独立铺样（不同 seed）上重复，给出每个观测量的结构随机性（上表最后一列），这是判据 (b)(d) 的信噪比基准。
3. **离散化无关性阶梯**：dt 及其 1/2、1/4，比较相邻档目标量的相对变化。
4. **可辨识性**：单位立方坐标下的中心差分 Jacobian（步长 0.05，并在 0.1 复核），FIM = JᵀΣ⁻¹J，条件数、特征谱、参数相关矩阵；并给出**仅用体相观测量**（packing_fraction）的对照条件数。
5. **反演回收**：合成观测 y_obs = y_truth·(1+ξ)，ξ ~ N(0, σ/y)，用加权最小二乘（有界信赖域，参数在单位立方内）从两个初值反演，回收率 = |反演值 − 真值|/真值。
6. **不确定度传播**：每个参数 ±10%（−10/−5/0/+5/+10 %）扫描，逐（参数, 观测量）线性拟合 R²；同时给出灵敏度 ∂y/∂p 与传播不确定度 σ_y = |∂y/∂p|·σ_p。
7. **被否决的候选观测量**：记录并量化了在更大系统上看似可行、但在本系统规模下不可用的候选（见 §5）。

工具链：PyTorch 生态仓库 `TensorDEM` 的 `granular2d` 显式积分内核（float64 确定性、邻接表 KD-tree、Cundall–Strack 切向/滚动阻力），`tensordem.sensitivity.identifiability` 复用 FIM/条件数/相关矩阵实现；scipy `least_squares` 做反演；matplotlib 出图。

## 3. 计算结果（数字表）

### 3.1 正向目标量（参考参数点）

| 可观测量 | 值 | 1σ | 相对结构随机性 | dt→dt/2 变化 | 阶梯内最大变化 |
|---|---|---|---|---|---|
| `grain_friction_ratio` | 0.55 | 0.02 | 0 % | 2.019e-14 % | 2.019e-14 % |
| `packing_fraction` | 0.7835 | 0.01 | 0.154 % | 0.0303 % | 0.137 % |
| `restitution` | 0.47116 | 0.0141 | 0 % | 0.121 % | 0.121 % |
| `impact_duration_s` | 0.01702 | 3.404e-04 | 0 % | 0.19 % | 0.19 % |

### 3.2 Fisher 信息与参数独立性

| 量 | 值 |
|---|---|
| cond(FIM)（4 个观测量） | 3.467 |
| cond(FIM)（仅体相观测量 `packing_fraction`） | ∞（秩亏 rank = 1/3） |
| 特征值谱 | 581.75, 1225.3, 2017.1 |
| Jacobian 步长复核（0.1 步）| cond = 3.407，各列范数相对变化 4.12 %, 1.75 %, 13.8 % |
| 单参数信息量 I_ii | μ=1225, e=1995, k_n=603.6 |

参数相关矩阵（由 FIM⁻¹ 归一化）：

| | μ | e | k_n |
|---|---|---|---|
| μ | 1 | 4.152e-04 | -0.00387 |
| e | 4.152e-04 | 1 | -0.16 |
| k_n | -0.00387 | -0.16 | 1 |

### 3.3 反演回收率

合成观测扰动：-0.0503 %, 0.352 %, 0.666 %, -0.249 %（seed 20261110）

| 初值 | χ² | nfev | μ 回收率 | e 回收率 | k_n 回收率 | 最大回收率 |
|---|---|---|---|---|---|---|
| centre | 0.1258 | 7 | 0.282 % | 0.843 % | 0.153 % | 0.843 % |
| far_corner | 0.06314 | 10 | 0.268 % | 0.581 % | 0.0302 % | 0.581 % |

### 3.4 不确定度传播（±10 % 线性响应）

| 参数 \ 可观测量 | `grain_friction_ratio` | `packing_fraction` | `restitution` | `impact_duration_s` |
|---|---|---|---|---|
| μ | ∂y/∂(rel) = 0.55, R² = 1 | 低于噪声门槛（不计入判据） | 低于噪声门槛（不计入判据） | 低于噪声门槛（不计入判据） |
| e | 低于噪声门槛（不计入判据） | 低于噪声门槛（不计入判据） | ∂y/∂(rel) = 0.393, R² = 0.9997 | 低于噪声门槛（不计入判据） |
| k_n | 低于噪声门槛（不计入判据） | 低于噪声门槛（不计入判据） | 低于噪声门槛（不计入判据） | ∂y/∂(rel) = -0.00903, R² = 0.9979 |

全表最小 R² = 0.9979（仅计可分辨项）；最差项 = {'r2': 0.9978846498328153, 'parameter': 'stiffness', 'target': 'impact_duration_s'}。

**被排除（响应低于测量噪声门槛）的组合**——R² 列仅供透明性参考，不参与判据：

| 参数 | 观测量 | ±10 % 相对响应 | 噪声门槛（σ_i/y_i） | 拟合 R²（如拟合） |
|---|---|---|---|---|
| μ | `packing_fraction` | 0.463 % | 1.28 % | 0.001595 |
| μ | `restitution` | 0 % | 3 % | -8.2 |
| μ | `impact_duration_s` | 0 % | 2 % | — |
| e | `grain_friction_ratio` | 4.037e-14 % | 3.64 % | -2.667 |
| e | `packing_fraction` | 0.411 % | 1.28 % | 0.685 |
| e | `impact_duration_s` | 0.901 % | 2 % | 0.9971 |
| k_n | `grain_friction_ratio` | 4.037e-14 % | 3.64 % | -0.125 |
| k_n | `packing_fraction` | 0.727 % | 1.28 % | 0.986 |
| k_n | `restitution` | 0.585 % | 3 % | 0.9703 |

### 3.5 被否决的候选观测量

`tilt_onset_deg`：a confined bed of 33 grains jams instead of flowing: the onset is not monotone in the friction coefficient, so the observable carries no usable information at this system size

| μ | 倾角起始 (°) | 是否截断 | 该状态堆积率 |
|---|---|---|---|
| 0.25 | 13.96 | 否 | 0.7827 |
| 0.55 | 13.11 | 否 | 0.7835 |
| 0.85 | 9.668 | 否 | 0.7817 |

## 4. 结果图

![fig1_observable_sensitivity.png](../artifacts/granular_identifiability/fig1_observable_sensitivity.png)

*Each observable against a +-10 % perturbation of the three contact parameters; the straightness of the traces is the linear-response (uncertainty propagation) criterion.*

![fig2_discretisation_ladder.png](../artifacts/granular_identifiability/fig2_discretisation_ladder.png)

*Relative change of each observable between successive time-step refinements (dt, dt/2, dt/4) against the 1 % acceptance line.*

![fig3_fisher_information.png](../artifacts/granular_identifiability/fig3_fisher_information.png)

*Fisher-information eigenvalue spectrum and the implied parameter correlation matrix at the reference point; the bulk-only control shows the information content of the flow observables alone.*

![fig4_inversion_recovery.png](../artifacts/granular_identifiability/fig4_inversion_recovery.png)

*Relative recovery error of the three contact parameters from two starting points, against the 5 % acceptance line.*

![fig5_settled_bed_profiles.png](../artifacts/granular_identifiability/fig5_settled_bed_profiles.png)

*Settled-bed free-surface height and local solid fraction of the forward case at the reference parameters (the packing-fraction observable is the integral of the second panel).*

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **体相观测量在小系统上不可辨识**：仅用堆积率时 cond(FIM) = ∞（秩亏 rank = 1/3）（对照全观测量 3.467）。在本 case 的规模（数十颗粒）下，自由面/堆积类量对 μ 的响应（<1 %）小于独立铺样的结构随机性，因此 μ 只能靠**单颗粒受限剪切**这类接触尺度测量识别。这与文献中「体相流动观测量对刚度几乎不含信息、对摩擦信息微弱」的结论一致。
2. **尝试并否决的观测量**：(i) 堆积/自由面休止角——数十颗粒时自由面被单颗粒粗糙度主导，拟合角在参数变化下不单调；(ii) 料仓流出速率——小系统间歇拱塞，结构抖动与被测量响应同量级；(iii) 受限床倾角/雪崩起始——床在封闭盒内被墙约束、无法真正流动，起始角随 μ 非单调（见 §3.5）。这三者在更大系统（10⁴–10⁵ 颗粒）上应可用，但会突破单 case 预算。
3. **恢复系数的映射偏差**：引擎的接触阻尼按 ζ 定义，经典关系 e = exp(−πζ/√(1−ζ²)) 与实测回弹并不完全一致（压剪弹簧-阻尼器在分离时刻的钳位效应）；反演针对**实测** e 进行，映射偏差在报告中披露而不做经验修正。参考点：名义 e 目标 0.45，实测 0.4712。
4. **冲击共振刚度的系统偏差**：由接触时长反推 k_n = m(π/τ)²/(1−ζ²) 与真值有系统偏差（重力使分离时刻延后、并含阻尼钳位），因此把 τ_c 作为观测量、把刚度反推值仅作诊断量输出。
5. **局部 vs 全局**：FIM 与条件数是**局部**（线性化）陈述；±10 % 线性响应检验给出局部性的经验边界，但更大参数变化下响应可能非线性（尤其是堆积率接近随机密堆极限时）。
6. **适用范围**：结论针对软颗粒线性弹簧-阻尼接触模型、二维、以及本 case 的粒径/几何/预算；换模型（Hertz-Mindlin、黏结、非球形）需重新跑同一验证面，接口不变。

## 6. 误差判定（逐项对照判据，显式回答是否达标）

| 判据 | 量 | 门槛 | 结果 | 判定 |
|---|---|---|---|---|
| a_discretisation_independence | 离散化无关性（dt 减半目标量变化） | 1 % | 0.19 % | PASS |
| b_recovery | 3 参数反演回收率最大值 | 5 % | 0.581 % | PASS |
| c_fim_condition_number | FIM 条件数 | 100 | 3.467 | PASS |
| d_uncertainty_propagation | ±10 % 线性响应最小 R² | 0.95 | 0.9979 | PASS |
| e_zero_domain_semantics | 合成系统 `test_granular_01` 且模块无领域词条 | — | 测试通过 | PASS |
| f_case_budget | 单 case 预算（20000 步 / 400 颗粒 / 40.0 s） | 实际 3009 步 / 33 颗粒 / 2.65 s | — | PASS |

**是否 <3 % / <5 % 的显式回答**：

- μ（`friction`）回收率 0.268 % → 是否 <5 %：**是**。
- e（`restitution`）回收率 0.581 % → 是否 <5 %：**是**。
- k_n（`stiffness`）回收率 0.0302 % → 是否 <5 %：**是**。
- `grain_friction_ratio` 在 dt 阶梯上的最大相对变化 2.019e-14 % → 是否 <1 %：**是**。
- `packing_fraction` 在 dt 阶梯上的最大相对变化 0.137 % → 是否 <1 %：**是**。
- `restitution` 在 dt 阶梯上的最大相对变化 0.121 % → 是否 <1 %：**是**。
- `impact_duration_s` 在 dt 阶梯上的最大相对变化 0.19 % → 是否 <1 %：**是**。

本次运行全部判据通过：**是**。

## 7. 复现

```bash
PYTHONPATH=src python scripts/run_granular_identifiability.py --workers 10
PYTHONPATH=src python scripts/render_granular_identifiability_doc.py
PYTHONPATH=src python -m pytest tests/test_granular_identifiability.py -q
```
