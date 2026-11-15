# Pre-registration — m21958 · 接触摩擦正则化速度 `v_reg` 是否是 φ_cv 的隐性自变量

**锚点** `CASE-DEM-BIAXIAL-FRICTION`（算例池·双轴剪切摩擦角，验收 |φ−38°|/38° < 8%，ref Bolton 1986）
**仓** TensorDEM ｜ **mission** 21958 ｜ 本文件在任何算例启动之前写入并 commit。

## 0 为什么做这一件

第 3 轮（m20537）测得 φ_cv = 14.31°（μ = 0.55，等体积剪切），比 Rowe(1962) 的 arctan μ = 28.81°
低 **50.3 %**。第 4/5 轮把差额归给「二维圆盘无滚动摩阻」并加了 `rolling_friction_coefficient`；
第 6 轮（m21803）审计出该旋钮在无转动自由度的内核里等于**附加滑动摩擦**（`F_t = -(μ+μ_r)F_n tanh(v_t/v_reg)`，
加和性成立），因此它不可能是缺口的补丁。

读代码（`src/tensordem/ridge_dem.py`）暴露出一件从未被验证的事：**无粘结接触的切向律不是库仑律**，
而是速率正则化律

```
F_t = -mu * F_n * tanh(v_t / v_reg)          v_reg = material.velocity_regularisation (默认 0.01 m/s)   (1)
```

它是**速率比**的函数，不是材料属性。经验库仑摩擦的必要条件是**与速率无关**：`v_reg` 应当只是数值
光滑参数，任何量都不应依赖它。这一条从来没有被检验过，而它恰好是锚点 50 % 缺口的候选根因。

## 1 假设（先写判据，后跑算例）

| 编号 | 假设 | 判据（预声明） | 否定条件 |
|---|---|---|---|
| **H1** | φ_cv 对数值参数 `v_reg` 不敏感（式 (1) 只在数值上光滑） | 四档 `v_reg` ∈ {0.0025, 0.01, 0.04, 0.1} 的 φ_cv 极差 **≤ 1.0°** | 极差 > 1.0° → φ_cv 不是模型材料属性，是正则化参数的函数 |
| **H2** | φ_cv 对剪切速率不敏感（准静态，I ≪ 0.01） | ε̇_a = 0.01 与 0.05 两档 φ_cv 之差 **≤ 1.0°** | 差 > 1.0° → 模型在准静态区仍有速率依赖 |
| **H3** | 缺口可用接触层算子定量解释：⟨\|tanh(v_t/v_reg)\|⟩ 是接触摩擦「实际调动比例」 | 实测 φ_cv 与 `arctan(μ·⟨\|tanh\|⟩_Fn加权)` 的差 **≤ 2.0°** | 差 > 2.0° → 缺口不在接触算子的调动率（转而在 fabric / 二维平面应变关系） |

容差 1.0° 的取法：R3/R6 的窗口离散 ≤ 1.5°，1.0° 略严于既有噪声，且 ≈ 锚点 8 % 容差（14.3°×8 % = 1.1°）
量级；2.0° 是给 H3 这个**跨界映射**（微观算子 → 宏观角）留的一档。

**不做的事**：不改锚点验收口径，不选窗（沿用 R3 的规则：`min_strain=0.15`、末 20 % 窗、四闸门），
不为了贴近 28.81° 调任何参数；`v_reg` 的各臂都是「诊断扫描」，锚点工况仍是默认值。

## 2 算例（全部固定，只动 `v_reg` / `ε̇_a`）

| 量 | 值 |
|---|---|
| 试样 | 24 × 48 抖动三角格（jitter 3 %），1152 盘，r = 0.75 m，ρ = 917 kg/m³，厚度 1 |
| 接触 | kn = 1×10⁷ N/m，kt/kn = 0.5，**μ = 0.55**，`freeze_bond_fraction = 0`（无胶结） |
| 墙 | `wall_friction = 0`（侧向读数为纯法向应力） |
| 固结 | 侧墙伺服 σ₃ = 100 kPa，6000 步，`axial_servo = False`（沿用 R3） |
| 剪切 | **等体积**（`lateral_control="isochoric"`），顶板位移控，ε_a 至 **0.45**，Δt = 4.97×10⁻⁴ s |
| 计量 | `mobilisation_every = 500` 步采样接触摩擦调动仪（本轮新增，见 §3） |

| 臂 | ε̇_a [1/s] | `v_reg` [m/s] | 作用 |
|---|---|---|---|
| `A_base` | 0.01 | 1×10⁻²（默认） | **复现 R3 基线**（期望 φ_cv ≈ 14.31°） |
| `B_vreg_lo` | 0.01 | 2.5×10⁻³ | H1（更「黏」的正则化） |
| `C_vreg_hi` | 0.01 | 4×10⁻² | H1 |
| `D_vreg_x10` | 0.01 | 1×10⁻¹ | H1（近乎全库仑：tanh 基本饱和） |
| `E_rate_x5` | 0.05 | 1×10⁻² | H2（I = 7.2×10⁻³ < 0.01，仍准静态） |

## 3 新增的可观测量（引擎内，纯诊断）

`RidgeAssembly.enable_friction_meter()` 在每次 `forces()` 里对**承载中的无粘结 disk–disk 接触**累加
`Σ F_n`、`Σ F_n|tanh(v_t/v_reg)|`、`Σ|tanh|`、饱和接触数（|tanh| ≥ 0.99）、`Σ|v_t|`；
`read_friction_meter(reset=True)` 返回窗口均值 `mean_abs_tanh`、`fn_weighted_mean_abs_tanh`、
`saturated_fraction`、`mean_abs_vt`。**它不改变任何一个力**（默认关闭；`tests/test_friction_mobilisation_meter.py`
断言开关两种情况下应力路径逐位相同）。

## 4 判定的写法（先写，后填数）

1. 每臂用 `biaxial_critical_state.critical_state_from_stress_path(strain_measure="log",
   boundary_control="isochoric")` 取 φ_cv，报四闸门（isochoric / deviator_stationary /
   window_independent / lateral_walls_loaded）——闸门不通过就照实写「临界状态未到达」，不换窗。
2. H1：`max(φ_cv) − min(φ_cv)`（四臂）对照 1.0°；H2：`|φ_cv(E) − φ_cv(A)|` 对照 1.0°。
3. H3：每臂窗口内 `fn_weighted_mean_abs_tanh` 与 φ_cv 配对，报 `arctan(μ⟨|tanh|⟩)` 与 φ_cv 的差。
4. 锚点验收（<8 % vs 38°）照旧判定，**本轮预计仍 FAIL**；本单不宣称修好锚点，只宣称缺口归因是否成立。
5. honest FAIL 不粉饰：若三假设全被否定，就写「式 (1) 的速率正则化不是缺口根因」，并把 φ_cv 的
   其它候选原因列成下一单线索。

## 5 复现

```bash
# 集群递交（5 分片，每片一臂）
TENSORDEM_THREADS=3 PYTHONPATH=src python scripts/m21958_friction_regularisation_probe.py --shard {i} --nshards 5
# 分析 + 图 + 工件
PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_regularisation.py
# 计量仪不变量测试
PYTHONPATH=src python -m pytest tests/test_friction_mobilisation_meter.py -q
```