# CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT — round 2: the *concentration* route

**锚点**：`CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT`（status 仍 open）
**任务**：m16587（tp-infra-1 自找活）｜**仓库**：TensorDEM｜**分支** `agent/m16587`
**接棒对象**：round 1（`docs/CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT.md`，m16136/m16243/m16350）
其结论是"弹性支再现（`n=0.26`），惯性支未辨（`k_n=1e5/1e6` 预算内 not_run），
系数 WITHHELD（付费墙）"。

## ① 案例介绍

物理背景与装置同 round 1：无壁 Lees–Edwards（SLLOD）均匀剪切盒，
接触律=线性弹簧-阻尼法向 + Coulomb 上限切向弹簧（带历史），
单分散比 1.4 的双分散盘，颗粒摩擦 μ=0.5、恢复系数 e=0.7、面密度 ρ=1。
**本单唯一改变的自变量是浓度**：`k_n=1e4`、`γ̇=1` 固定不变。

| 参数 | 值 |
|---|---|
| `k_n` | 1.0e4（γ̇=1 时 κ=5787） |
| `N`（盘数） | 400（50/50 双分散，d=1.00/1.40，d̄=1.20） |
| 面积分数 φ | 0.80 → 0.64，步长 0.01（17 点） |
| 剪切率 γ̇ | 速率梯：0.5 / 1 / 2（φ=0.80, 0.76, 0.70；扩展 φ=0.66） |
| 时间步 | dt = 0.004 t_c（round 1 已证 τ* 对 halving 仅 1.2%） |
| 累积应变 | 1.4，取平均窗 0.7–1.4 |
| 对照 | 种子孪生（seed+1）、应变孪生（2.8，窗 1.4–2.8）、5 个 φ × 3 种子散度 |
| 单点成本 | 12–54 s（loongarch64, numpy；与 γ̇ 成反比） |

**为什么要换轴**：round 1 只在 `φ=0.80` 上扫 κ，因此在 (φ, κ) 平面上只有一条竖线。
可及的 Campbell (2002) 摘要有两句可机器裁的断言：

> "Reducing the concentration slightly causes a transition from an elastic to a
> purely inertial behaviour. This transition occurs so abruptly that a 2%
> concentration reduction can be accompanied by nearly three orders of
> magnitude of stress reduction. […] there is no path between inertial (rapid)
> flow and quasi-static flow by varying the shear rate at a fixed concentration;
> only by reducing the concentration can one cause a transition from
> quasi-static to inertial flow."

出处锚：`benchmarks/case_dem_shear_campbell_elastic_limit/reference_anchors.json`
（出版方落地页摘要，sha256 `bc6d07a3…`，全文 CLOSED → 系数仍 WITHHELD）。

## ② 怎么做的

1. **预注册**（先于任何一次跑）：`phi_transition.py --prereg` 把 3 条预测、
   稠密指标与判定阈值写进 `runs/phi_transition/prereg.json`（含锚文件 sha256）。
2. **装置级新增（本单唯一代码改动）**：`tensordem.elastic_limit_shear`
   现在在同一批采样上把应力**可加分解**为
   `τ = τ_kinetic + τ_contact`、`P = P_kinetic + P_contact`
   （peculiar-velocity 项 vs 接触 virial 项），并给出
   `kinetic_stress_share = ⟨|τ_kin|⟩/(⟨|τ_kin|⟩+⟨|τ_contact|⟩)`。
   这是"elastic（接触承载）→ purely inertial（碰撞承载）"的可测判据，
   round 1 只有一层未定义的 `kinetic_stress_fraction` 比值。
   **gates 集合与判定语义完全未改**。
3. φ 梯（17 点）→ 速率梯（φ=0.80/0.76/0.70，各 3 个 γ̇）→
   对照（种子孪生 2 + 应变孪生 2）→ 种子散度（5 φ × 3 种子）。
4. **事后追加（如实披露）**：种子散度阶段与 φ=0.66 的速率梯是在看到
   round-1 式偏差（种子孪生 +21%）之后加的；写入 `analysis.json.amendments`，
   **不改任何阈值**，P2 判定仍只用预注册的三个浓度。
5. `--analyse` 重读 prereg，逐条套用判定规则，输出
   `runs/phi_transition/{tables.md, analysis.json, fig_phi_transition.png}`。

## ③ 计算结果（程序化注入，全部来自 `runs/phi_transition/*.json`）

### 3.1 φ 梯（γ̇=1，`k_n=1e4`，N=400）

| φ | τ | P | τ* | μ=τ/P | I | Z | kin share | max ov/d | E-defect | gates |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 122.9 | 249.1 | 85.35 | 0.4934 | 0.0760 | 2.93 | 0.006 | 0.159 | 0.248 | ok |
| 0.78 | 91.39 | 166.4 | 63.46 | 0.5492 | 0.0930 | 2.66 | 0.011 | 0.118 | 0.217 | ok |
| 0.76 | 59.91 | 109.7 | 41.60 | 0.5460 | 0.1146 | 2.30 | 0.014 | 0.113 | 0.207 | ok |
| 0.74 | 44.16 | 73.91 | 30.67 | 0.5975 | 0.1396 | 2.00 | 0.019 | 0.120 | 0.184 | ok |
| 0.72 | 32.02 | 50.76 | 22.24 | 0.6309 | 0.1684 | 1.64 | 0.021 | 0.084 | 0.187 | ok |
| 0.70 | 23.31 | 33.91 | 16.19 | 0.6875 | 0.2061 | 1.33 | 0.040 | 0.074 | 0.166 | ✗ steady,coord |
| 0.68 | 17.17 | 25.61 | 11.93 | 0.6705 | 0.2371 | 1.14 | 0.045 | 0.072 | 0.139 | ✗ coord |
| 0.66 | 8.736 | 12.64 | 6.066 | 0.6910 | 0.3375 | 0.78 | 0.058 | 0.052 | 0.146 | ✗ coord |
| 0.64 | 6.886 | 9.529 | 4.782 | 0.7226 | 0.3887 | 0.68 | 0.066 | 0.038 | 0.128 | ✗ coord |

全程（φ 0.80→0.64，共 16 个百分点）应力只降 **17.8 倍**；接触配位数 Z 在
φ≈0.70 跌破 1.5（协调门），即**接触网络在 φ≈0.70 断开**。

### 3.2 预注册指标：2 % 浓度窗（仅双端过门的窗）

| φ_hi → φ_lo | τ_hi | τ_lo | 比值 | d ln τ/d ln φ |
|---|---|---|---|---|
| 0.79 → 0.77 | 116.6 | 70.01 | **1.67（最大）** | 19.9 |
| 0.78 → 0.76 | 91.39 | 59.91 | 1.53 | 16.3 |
| 0.80 → 0.78 | 122.9 | 91.39 | 1.34 | 11.7 |
| 0.76 → 0.74 | 59.91 | 44.16 | 1.36 | 11.4 |
| 0.74 → 0.72 | 44.16 | 32.02 | 1.38 | 11.7 |
| 0.73 → 0.71 | 38.22 | 30.26 | 1.26 | 8.4 |

含过门失败点的全点窗（只描述形状，不参与判定）最大为 φ 0.68→0.66 的 **1.97**。
**R2 = 1.67**，对"nearly three orders of magnitude"（1e3）而言差 600 倍。

### 3.3 速率梯：剪切率指数 n(φ)

| φ | τ(0.5) | τ(1) | τ(2) | **n** | 是否预注册 |
|---|---|---|---|---|---|
| 0.80 | 114.0 | 122.9 | 144.8 | **0.173** | 是 |
| 0.76 | 38.64 | 59.91 | 89.86 | **0.609** | 是 |
| 0.70 | 10.88 | 23.31 | 52.33 | **1.133** | 是 |
| 0.66 | 3.755 | 8.736 | 27.51 | **1.437** | 否（事后扩展） |

n 随浓度下降**单调**从 0.17（弹性，n→0）升到 1.44（Bagnold 惯性，n→2）。

### 3.4 种子散度与对照

| φ | 种子数 | τ 均值 | 相对标准差 |
|---|---|---|---|
| 0.80 | 3 | 115.5 | 6.25% |
| 0.76 | 3 | 60.68 | 2.29% |
| 0.72 | 3 | 27.32 | 22.3% |
| 0.68 | 3 | 12.88 | 29.6% |
| 0.64 | 3 | 6.115 | 13.3% |

| 对照 | φ | 相对差（vs 主梯） | 过门 |
|---|---|---|---|
| 种子孪生 | 0.77 | **+21.4%** | 是 |
| 种子孪生 | 0.79 | −7.8% | 是 |
| 应变孪生（2.8） | 0.77 | +15.0% | 是 |
| 应变孪生（2.8） | 0.79 | −8.0% | 否（steady 门） |

### 3.5 副产物：与 `mu(I)` 律的跨装置一致性

用仓内 mu(I) 常数（mu_s=0.36, mu_2=0.84, I_0=0.30；出处见
`docs/CASE_DEM_PLANE_SHEAR_MUIDI_R3.md`，该处标注 PROVENANCE-UNVERIFIED）：
本装置实测 μ 在过门段（I=0.076–0.175）**系统性高于** mu(I) 律 **+7.9% … +20.0%**
（均值 +15.1%）。

### 3.6 图

`runs/phi_transition/fig_phi_transition.png`：
(a) τ(φ) 带种子散度误差棒（过门/被拒分色）；(b) 三个浓度下的速率梯 + n=0 / n=2 参考线；
(c) n(φ) 单调曲线；(d) kinetic share(φ)。

## ④ 注意点（诚实挂账）

1. **控制变量决定结论**：本盒是**定容定剪切率**（SLLOD）装置，γ̇ 是外加的，
   所以浓度降低时应力**不可能塌到 0**——它转成率相关（黏性/惯性）响应。
   Campbell 的"2 % 塌三个量级"更可能在**定压**装置上显现（低于 jamming 时材料流动、
   P 无法维持）。本单因此**不能**判定 Campbell 错，只能判定
   "在定容定 γ̇ 的 2-D 摩擦盘中，该突发性不出现"。
2. **低 φ 段过门失败是物理而不是 bug**：Z<1.5（协调门）在 φ≤0.70 触发，
   正是接触网络断开的指纹；这些点照实标 ✗ 并排除出拟合。
3. **种子散度在 φ≤0.72 达 22–30%**：此时单点 τ 不可当精确量用；
   2 % 窗比值（1.26–1.67）与散度同量级，**任何小于 30% 的"突降"都无法与散度区分**。
4. 2-D 与 3-D 的偏差、线性弹簧接触、e=0.7 都与 round 1 相同，未改。
5. round 1 的 `k_n=1e5/1e6` rung 本单**没有补**（κ 高端不改变本单结论）；
   它们仍记 not_run。

## ⑤ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 结果 | 相对误差 | <3%？ |
|---|---|---|---|---|
| P1 2 % 窗应力比 | "nearly 1e3"（摘要，非精确数） | 1.67 | −99.8% | **否** |
| P2 剪切率指数（预注册三浓度最大） | Bagnold n=2 | 1.133 | −43% | **否** |
| P2 剪切率指数（含事后 φ=0.66） | Bagnold n=2 | 1.437 | −28% | **否** |
| P2 高浓度端 | 弹性 n=0 | 0.173（φ=0.80） | 绝对 0.173 | 否（基准为 0，绝对判据 ≤0.30 由 round 1 预注册，本单沿用） |
| P3 kinetic share 最大 | 惯性支 ≥0.5 | 0.066（φ=0.64） | −87% | **否** |
| μ vs mu(I) 律 | 仓内常数 | 过门段均值 +15.1% | 15.1% | 否（但 <15% 的边界内，最大 +20%） |
| 分解可加性不变量 | τ = τ_kin + τ_cont | 残差 ≤1.4e-13 | ≈1e-15 相对 | **是** |
| dt 收敛（round 1 继承） | 0.004 vs 0.002 t_c | τ* 差 1.2% | 1.2% | **是** |

**显式回答**：本单**没有任何前向物理量 <3%**；<3% 的只有两个数值不变量
（分解可加性、时间步收敛）。物理层最好的腿是"n(φ) 单调趋向 Bagnold"
（28–43% 差距），最差的腿是"2 % 突发塌陷"（差 600 倍）。
因此本锚点**不关闭**，后续判定应为：
`abrupt_drop_NOT_SUPPORTED_in_constant_volume_2D`。

## ⑥ 判定（预注册规则逐条套用）

| 预测 | 规则 | 结果 | 判定 |
|---|---|---|---|
| P1 2 % 浓度→近 3 量级降 | R2≥100 PASS / ≥10 PARTIAL / <10 NOT_SUPPORTED | R2=1.67 | **NOT_SUPPORTED** |
| P2 定浓度下无速率路径到惯性 | max n≤0.6 PASS / ≥1.2 NOT_SUPPORTED / 中间 PARTIAL | max n=1.133（预注册） | **PARTIAL** |
| P3 弹性→纯惯性是接触→碰撞切换 | Spearman≤−0.8 且最低 φ 处 share≥0.5 | ρ=−0.998，share=0.066 | **PARTIAL** |

三条里最硬的一条是 P2 的**方向**：n 随 φ 下降单调上升（0.17→0.61→1.13→1.44），
即"惯性支只能靠降浓度进入"与本装置数据**方向一致**；但突变性和量级都不成立。

## ⑦ 复现

```bash
cd <repo>
PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/phi_transition.py --prereg
PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/phi_transition.py --stage map --jobs 6
PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/phi_transition.py --stage rates --jobs 6
PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/phi_transition.py --stage controls --jobs 4
PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/phi_transition.py --stage scatter --jobs 6
PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/phi_transition.py --analyse
PYTHONPATH=src python3 -m pytest tests/test_elastic_limit_stress_decomposition.py -q   # 9 passed
PYTHONPATH=src python3 -m pytest tests/ -q -k elastic_limit                            # 35 passed, 2 skipped
```

工件：`benchmarks/case_dem_shear_campbell_elastic_limit/runs/phi_transition/`
（`prereg.json`、26 个 `*.json` 单点、`analysis.json`、`tables.md`、`fig_phi_transition.png`）。
环境：loongarch64 / numpy 2.x。

## ⑧ 给下一轮的线索

1. **换控制变量**：在定压伺服盒（`tensordem.mu_i_rheology`）上从低位原密度起做 P 伺服固结，
   看 jamming 密度 φ_c 是否表现为"应力塌陷"——这是 Campbell 突变性的正确探针。
2. **把 n(φ) 推到极限**：φ=0.60/0.56 处速率梯，看 n 是否收敛到 2（Bagnold 系数可测）。
3. **种子数换精度**：φ≤0.72 需要 ≥8 个种子才能把 2 % 窗判定分辨率压到 10% 以内。
4. 跨装置一致性差异（本盒 μ 高 mu(I) 律 8–20%）值得在 mu(I) 车道单独归因
   （壁面/周期性 vs 伺服盒几何）。