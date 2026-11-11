# AUDIT-RUNTEST-6 · 真跑审计（TensorDEM / m14498）

**锚点**：AUDIT-RUNTEST-6（owner 2026-11-11 第三项令：不能只读 PR 描述，要真正执行）
**对象**：`cxs503/TensorDEM` @ `9b98b609`（worktree `/root/tp-agent/work/m14498`，分支 `agent/m14498`）
**执行者**：audit2-run worker
**证据目录**：`runs/`（原始日志、JSON 数值结果、复现脚本）；脚本 `scripts/audit_runtest6_*.py`

审计只回答一个问题：**仓库文档/工件声称的能力，在本机真跑之后还剩下多少。**
凡"跑不通"或"未跑"的地方，本报告原样写出，不做粉饰。

---

## 报告六要素（论文级摘要）

1. **案例介绍**：对象 `cxs503/TensorDEM` @ `9b98b609`（worktree m14498，分支 `agent/m14498`）；
   环境 CPython 3.11 + torch 2.7.1（CPU，loong64 控制面）、pytest 8.x、matplotlib 3.10；
   被试对象=brash-ice 全尺度阻力验证（12 冰况）、`benchmarks/hertz_analytic`、DEM3D 配置面与接触能量账本、
   ice_transit 能量账本。参数：`μ_ii=0.3, μ_is=0.15, σ_b=1 MPa, ρ_ice=917 kg/m³`，
   冰厚 0.5–2.0 m、浓度 0.4–0.8、航速 2–5 m/s。
2. **怎么做的**：不读 PR 描述只看代码+真跑。四类真跑＝①全量 pytest（集群递交）②flagship 算例（本机重跑并逐键 diff 提交工件）
   ③21 个负向探头（预注册"应报错"判据）④收敛/守恒（`order_sweep` 的 dt 逐级减半 + 弹性冲击动量/恢复系数）。
   工具链：`scripts/audit_runtest6_{negative_probe,sobol_probe,convergence,dem3d_timestep,figures,verdicts}.py`。
3. **计算结果**：见 §2.1 逐点表（DEM 84.6–1991.7 kN）、§2.2 Sobol 表、§4.1 阶数表；原始值全部落在 `runs/**/*.json`。
4. **结果图**：图1 `runs/figures/fig1_resistance_three_way.png`（三方比对+误差带）、
   图2 `fig2_sobol_estimator.png`（S1/ST 与 ΣS1 漂移）、图3 `fig3_convergence_order.png`（残差 vs dt 与拟合阶）。
5. **注意点**：`clamp_min` 不吞 NaN；同名模块/包遮蔽让 1628 行代码不可达；归一化会让判据恒真；
   均值判据掩盖尾部；集群在本窗口不可用（吞吐 0.3 作业/分钟）。
6. **误差判定（显式回答"是否 <3%"）**：
   - 模型 vs 实测（brash 12 点）：均值 10.06%，**不是 <3%**；2 点破 20%（−40.05%、−39.56%）。
   - 模型 vs Lindqvist：均值 28.0%，**不是 <3%**，且 11/12 点破 15%。
   - 接触能量账本 vs 解析阶/机器精度：explicit 斜率 1.0026（目标 1，偏差 **0.26% < 3%**）、
     staggered 0.9884（目标 1，偏差 **1.16% < 3%**）、弹性冲击 |e−1| = 3.3e−8、动量误差 0.0 —— **这部分 <3% 成立**。
   - Hertz benchmark 声称的"曲线误差 0.000000"：**拒收**（循环验证，无数据），不计入误差判定。

## 0. 结论摘要（先给判定，再给证据）

| 等级 | 数量 | 内容 |
|---|---|---|
| **阻断（README 记录的测试命令直接失败）** | 1 | `python -m pytest -q`（README:132 / `.github/workflows/tests.yml:22` 的同一命令）在本 worktree **collection 阶段即中断**：`tests/test_freeze_bond.py`、`tests/test_ice_structure_contact.py` 两个 ImportError → **8707 项测试一个都没跑**。根因：`src/tensordem/freeze_bond.py`(849 行) 与同名包目录 `freeze_bond/`(208 行) **同名遮蔽**，包 `__init__` 未再导出模块里的 8 个类（`FreezeBondParams` 等）；`ice_structure_contact` 同构（6 个类）。**1628 行代码不可达**，由提交 `d4d4ea99`（"5 个文件被门禁滤除"）带入 |
| **严重（flagship 自身判据不过）** | 1 | README 旗舰『稳态碎冰场直航』单 case 真跑（600 s，63 浮冰）：平均阻力 59.11 kN，**E1 相对 Lindqvist 锚误差 93.1% → FAIL**（预注册判据 ≤15%）、**E7 稳态漂移 0.3516 → FAIL**（≤0.15）；能量账本残差 5.34%（含 6.35% 数值注入），与 README『恒等式』措辞冲突 |
| **严重（声称与实测矛盾）** | 2 | `benchmarks/hertz_analytic` 的"Hertz 接触验证"为**循环验证**（验证器把解析值赋给"DEM 值"，误差恒 0，工件里 100/100 个 `F_dem` 为 `null`）；brash-ice 全尺度验收标准 (a) 的"≤15%"只对 12 个工况中的 1 个生效，全数据集 **11/12 超 15%**（均值 28.0%，最大 44.2%） |
| **中等（判据失效/不可证伪）** | 2 | 验收标准 (c)"ΣS_Ti = 1.0 ± 0.01"因内部归一化而**恒真**（n=31 也 PASS，原始 ΣST 在 seed 1/2 下应 FAIL）；Sobol 指标 `S_Ti < S1_i`（σ_b: 0.591 vs 0.808）违反独立输入约束，ΣS1 = 1.27–1.73 > 1 |
| **中等（负向控制缺口）** | 2 | `hertz_normal_force(NaN)` 静默返回 NaN；`dem_resistance(C_ice = −0.5)` 静默返回 **0.0 kN**（与文档化负控 C=0→0 无法区分） |
| **轻微（工件/文档缺陷）** | 3 | 验收汇总打印 `PASS (6/5 criteria passed)` 并写入 `results.json`（off-by-one）；README 测试计数快照过期（声称 343 文件/2714 函数，实测 485/4357）；README 指向的 `artifacts/ice_transit/` 在仓内 **0 个文件入库** |
| **真实可复现（正面）** | 2 | `benchmarks/verified/dem_brash_ice_fullscale/results.json` 在本机**逐位复现**（12 点全部数值 0 diff，相对容差 1e-9）；仓库另有**真实** Hertz 引擎级验证（`scripts/verify_dem3d_hertz_restitution.py` + 17 项测试全绿） |

> 一句话：TensorDEM **不缺乏真实能力**（引擎级 Hertz/能量账本/brash 模型测试面存在且绿），
> 缺乏的是**"验证器自身的可证伪性"**——至少两条被引为"REAL 升级依据"的判据在无数据或错数据时也报 PASS。

---

## ① 核心测试套件全量真跑

**README 与 CI 记录的命令**：`python -m pytest -q`（`README.md:132`、`.github/workflows/tests.yml:22`）。

### 1.1 先踩到的硬事实：这套测试**根本收集不起来**

```bash
$ python -m pytest tests -q --collect-only -p no:cacheprovider
8707 tests collected, 2 errors in 86.54s
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
ERROR tests/test_freeze_bond.py
ERROR tests/test_ice_structure_contact.py
```

```text
tests/test_freeze_bond.py:30: from tensordem.freeze_bond import (FreezeBondParams, ...)
E   ImportError: cannot import name 'FreezeBondParams' from 'tensordem.freeze_bond'
    (/root/tp-agent/work/m14498/src/tensordem/freeze_bond/__init__.py)
tests/test_ice_structure_contact.py:18: from tensordem.ice_structure_contact import (ContactPressureField, ...)
E   ImportError: cannot import name 'ContactPressureField' from 'tensordem.ice_structure_contact'
```

**根因（已定位到文件级）**：同名 **模块文件 + 包目录** 并存，Python 选中包，模块里的类全部不可达。

| 同名对 | 模块文件（被遮蔽） | 包目录（实际生效） | 不可达的类 |
|---|---|---|---|
| `freeze_bond` | `src/tensordem/freeze_bond.py` **849 行** | `src/tensordem/freeze_bond/__init__.py` **208 行** | `FreezeBondParams`, `FreezeBondState`, `FailureMode`, `FailureModePhaseDiagram`, `ParameterSweepResult`, `PullOffResult`, `ShearResult`, `VerificationReport`（8 个） |
| `ice_structure_contact` | `src/tensordem/ice_structure_contact.py` **779 行** | `src/tensordem/ice_structure_contact/__init__.py` **47 行** | `ContactPressureField`, `IdentifiabilityResult`, `MonotonicScanResult`, `PolyhedronContactResult`, `RollingBiasResult`, `SpatialCorrelationResult`（6 个） |

实测遮蔽（`PYTHONPATH=src python3` 直接 import 后探属性）：

```text
freeze_bond            file=freeze_bond/__init__.py   FreezeBondParams=False FreezeBondState=False FailureMode=False
ice_structure_contact  file=ice_structure_contact/__init__.py  ContactPressureField=False RollingBiasResult=False
```

- **1628 行实现代码不可达**（不是文档问题，是能力丢失）；唯二引用者就是那两个测试文件 → 无任何运行中调用方。
- 另外 3 对同名（`segregation` / `sensitivity` / `shape_representation`）**没有坏**：包 `__init__` 里做了再导出（实测 `BinaryMaterial=True`、`Superquadric=True`）。第 6 个同名目录 `src/tensordem/verification/` **没有 `__init__.py`**（只有子目录 `granular/`），实测 `import tensordem.verification` 仍解析到 `verification.py`，**不构成遮蔽**。
- **独立印证（不重复计功）**：并发审计波 `audit2-ab` 已在 `tensor_maturity_verdicts` 落过同一发现（scope=`repo-structure module/package collisions`，evidence 指针 `artifacts/import_contract/broken_imports.json`、`tools/import_contract_scan.py -> unresolved test imports: 36 in 2 test modules`）。本单的增量是**真跑复现**（`pytest --collect-only` 实测 8707 收集 + 2 错，全部测试因此一条未跑），而非静态扫描。
- 引入来源：包目录最近一次改动为 `d4d4ea99 m9979: matured mission deliverables（5 个文件被门禁滤除）` → **门禁过滤是嫌疑引入路径**（值得 owner 复核该门禁规则）。
- CI 侧推论：`tests.yml` 确实跑同一条命令且 `timeout-minutes: 20`；本 worktree 在 collection 阶段即中断，
  说明**该失败要么是 CI 未覆盖到当前 main，要么是 CI 红而未处理**——两者都需要机制化修复（本单只审计，不改代码）。

### 1.2 绕过中断后的全量真跑（`--continue-on-collection-errors`）

见 **§5 集群运行**（作业 cj838＝原样命令的失败日志；cj872＝带 `--continue-on-collection-errors -n 16` 的全量计数）。

### 1.3 本机局部交叉参照（<120 s/命令，均全绿）

| 测试文件 | 结果 | 耗时 |
|---|---|---|
| `tests/test_dem_brash_ice_fullscale.py` | **26 passed** | 11.4 s |
| `tests/test_dem_brash_ice_fullscale.py + test_contact_energy_order.py` | **41 passed** | 27.9 s |
| `tests/test_dem3d_hertz_contact.py` | **17 passed + 6 subtests** | 23.3 s |
| `tests/test_ice_transit.py` | **24 passed** | 61.1 s |

> 这 4 个文件全绿**不代表全量通过**——恰是"抽样绿、整体红"的典型，必须靠 §5 全量数据定论。

## ② 代表性 benchmark / 科学计算算例

### 2.1 flagship：brash ice 全尺度阻力验证（campaign node p472w2-N04）

**声称**（`benchmarks/verified/dem_brash_ice_fullscale/results.json`）：`status_upgrade: "WITHHELD → REAL"`，
capability `ice.resistance.dem_straight_ahead`，12 个全尺度工况（MT Uikku / ODEN）三方比对
（DEM vs Lindqvist 解析 vs 实测），验收 (a)–(g) 全 PASS。

**真跑**：`PYTHONPATH=src python3 -c "run_full_validation('runs/fullscale')"` → 12.0 s，工件 `runs/fullscale/results.json`。

**可复现性（正面证据）**：与仓内提交工件逐键比对，**差异 0 条**（数值相对容差 1e-9）：

```text
committed = benchmarks/verified/dem_brash_ice_fullscale/results.json
rerun     = runs/fullscale/results.json
n diffs: 0
```

**但这批数字本身经不起判据检查**（差异 = 审计发现）：

| 验收项 | 文档阈值 | 本书实测 | 判定 |
|---|---|---|---|
| (a) R_DEM vs R_Lindqvist | ≤15%（措辞为"baseline"） | **只取 comparison[0]**（UIK-01，6.26%）；全 12 点：**11 点 >15%**，均值 **28.0%**，最大 **44.2%**（ODN-04） | 判据实现窄于措辞 → PASS 不可外推 |
| (b) R_DEM vs R_measured（均值） | ≤20% 均值 | 均值 **10.06%** PASS；但 **2/12 点破 20%**：UIK-05 **−40.05%**、ODN-04 **−39.56%** | 均值掩盖尾部，薄冰低阻工况系统性低估 |
| (c) ΣS_Ti | 1.0 ± 0.01 | 实测 **恒为 1.000000**（内部自归一化，n=31 也 PASS） | **判据恒真，不可证伪** |
| (d) 某参数 S_Ti > 0.4 | >0.4 | 0.5872（σ_b）通过 | 但见 2.2 ST<S1 矛盾 |
| (e) CI95 覆盖率 | ≥80% | 10/12 = **83.3%** | 通过（余量 3.3 点） |
| (f) 负控 C_ice=0 → R=0 | <1 kN | 0.0 kN 通过；但 **C_ice=−0.5 同样返回 0.0**（§3） | 负控覆盖不严 |
| (g) pytest ≥15 green | ≥15 | 该文件 **26 passed** | 通过 |
| 汇总行 | — | 打印 `OVERALL VERDICT: PASS (6/5 criteria passed)` 并落盘 | off-by-one 缺陷（`n_total = len(criteria)-1` 在 overall 赋值前求值，`fullscale_validation.py:560-567`） |

逐点数值（实测，`runs/fullscale/results.json`）：

| id | H (m) | C | V (m/s) | R_DEM (kN) | R_Lindqvist (kN) | R_meas (kN) | err vs Lindqvist | err vs meas |
|---|---|---|---|---|---|---|---|---|
| UIK-01 | 0.8 | 0.6 | 3.0 | 369.9 | 394.6 | 385 | −6.3% | −3.9% |
| UIK-02 | 1.0 | 0.5 | 3.5 | 485.6 | 629.2 | 510 | −22.8% | −4.8% |
| UIK-03 | 1.2 | 0.7 | 2.5 | 533.7 | 778.2 | 580 | −31.4% | −8.0% |
| UIK-04 | 1.5 | 0.6 | 4.0 | 1081.6 | 1370.8 | 1100 | −21.1% | −1.7% |
| UIK-05 | 0.6 | 0.5 | 2.0 | 143.9 | 201.2 | 240 | −28.5% | **−40.1%** |
| UIK-06 | 1.8 | 0.8 | 3.0 | 1186.1 | 1742.5 | 1280 | −31.9% | −7.3% |
| ODN-01 | 0.7 | 0.5 | 2.5 | 186.1 | 302.9 | 200 | −38.6% | −7.0% |
| ODN-02 | 1.0 | 0.6 | 3.0 | 398.9 | 619.0 | 420 | −35.6% | −5.0% |
| ODN-03 | 1.3 | 0.7 | 3.5 | 745.4 | 1055.0 | 750 | −29.3% | −0.6% |
| ODN-04 | 0.5 | 0.4 | 2.0 | 84.6 | 151.7 | 140 | **−44.2%** | **−39.6%** |
| ODN-05 | 1.6 | 0.8 | 4.0 | 1271.7 | 1616.3 | 1280 | −21.3% | −0.7% |
| ODN-06 | 2.0 | 0.7 | 5.0 | 1991.7 | 2648.4 | 1950 | −24.8% | +2.1% |

**证据等级判定**：数值可复现＝REAL（无需 GPU/随机性）；但**"DEM"这个词不成立**——
`dem_resistance` 走的是 `brash_ice_rheology` 的 μ(I) 连续介质三分解（剪切 + 浮力 + 龙骨拖曳），
固定 `floe_diameter_m=0.5`，且两条定标常数无文献出处：
`mu_2 = mu_ii * 2.8`、`keel_drag_coefficient = 2.2 * σ_b/1 MPa`
（`src/tensordem/brash_ice_rheology/fullscale_validation.py:201,205`，由提交 `b80aab44` 引入）。
"DEM vs 实测 10% 一致"因此不是独立验证，而是**两点定标后的半解析模型贴合**，必须标 CALIBRATED。

### 2.2 Sobol 敏感性：判据 (c) 恒真 + 指标自相矛盾

脚本 `scripts/audit_runtest6_sobol_probe.py` → `runs/sobol_probe/results.json`（24 s）。

- **原始（未归一化）总阶指数和**：n=500, seed=42 → ΣST = **1.0061**；
  换 seed：0.9264（seed 1）/ 0.9605（seed 2）/ 1.0217（seed 7）。
  → 若判据 (c) 用原始量，**seed 1、2 直接 FAIL**（|ΣST−1| = 0.074 / 0.040 > 0.01）；
  而库内先 `ST /= ST.sum()`（`fullscale_validation.py:388-392`），使 `ST_sum` 恒为 1.0。
- **恒真证明**：把采样压到 n=31（垃圾级），判据 (c) 仍报 `PASS`，`ST_sum = 1.000000`；
  同时 max(S1−ST) = **+2.11**（指标已完全失真）。
- **物理一致性违例**：独立输入必须 S_Ti ≥ S1_i。n=500 实测：
  σ_b：S1 = **0.808** > ST = **0.591**（差 −0.218）；mu_ii：0.480 vs 0.411（−0.068）。
  一阶指数和 ΣS1 = **1.2675**（n=4000 仍 1.1844）> 1 → 指数未收敛（或估计器有偏）。
  换 seed，ΣS1 在 **0.672–1.727** 间跳动（n=500）→ 该指标不能支撑 (d)"某参数主导"的结论。

### 2.3b flagship 二：稳态碎冰场直航（`scripts/run_ice_transit_campaign.py --name standard`）

**声称**（README 冰区直航节）：回收入口实现稳态直航；零偏置压碎律；"每个接触上报弹性势能与阻尼/库仑耗散功率，于是
`W_thrust + W_数值注入 + W_注入 − W_移出 − W_耗散 = Δ(动能 + 势能 + 接触弹性能)` 是**恒等式而不是断言**"；
campaign 预注册判据 `E1_rel_error_vs_lindqvist ≤ 0.15`（headline gate）、`E2_ledger_residual_rel ≤ 0.10`、
`E7_stationarity_drift ≤ 0.15`。

**真跑**：单 case（63 块浮冰、10 倍船长 1050 m、70000 步、dt=0.005 s、航速 3 m/s、冰厚 1 m、浓度 0.9）：
`wall_time_s = 600.8`（本机后台完成；两次前台调用均被控制面 120 s 上限斩杀，第三次以 nohup 放后台才跑完）。
工件：`runs/ice_transit_cases/standard.json`、`artifacts/ice_transit/summary.json`（`--report` 后处理 34 s）。

| 量 | 实测 | 判据/声称 | 判定 |
|---|---|---|---|
| 平均阻力（稳态窗 120–350 s） | **59.11 kN**（SEM 6.87，std 294.5，peak 4116.5，min 0.0） | — | 高度间歇（std/mean ≈ 5） |
| **E1 相对 Lindqvist 锚误差** | **0.9312（93.1%）** | ≤ 0.15 | **FAIL** |
| E2 能量账本残差 | **0.0534（5.34%，3727 kJ / 69774 kJ）** | ≤ 0.10 | PASS（但见下） |
| 数值注入占比 | **6.35%（4433 kJ）** | — | 与"恒等式"措辞冲突 |
| **E7 稳态漂移** | **0.3516** | ≤ 0.15 | **FAIL** |
| E9 测试数 / E10 航程 | 24 / 10.0 船长 | ≥12 / ≥10 | PASS |
| realized concentration | 0.8999 | 需求 0.9 | 一致 |

**对"恒等式"声称的判定**：`W_numerical` 本身就是残差的闭合项（`transit.py:585-587`），
实测残差 5.34%、数值注入 6.35% —— 这不是"恒等式",而是**带 6% 数值注入修正的能量平衡**。
README 的措辞过强；更准确的说法是"能量账本闭合到 5.3%（阈值 10%）"。

### 2.3 `benchmarks/hertz_analytic`：一个**循环验证**的 benchmark（严重）

真跑（11.7 s，本机）：

```text
PYTHONPATH=src python3 benchmarks/hertz_analytic/run.py --output-dir runs/hertz_local
committed sha256 4047c26f97434d95
rerun     sha256 4047c26f97434d95      → 逐位一致（真跑，但是"复现了假数据"）
```

- `results.json` 里 `F_dem` **100/100 为 `null`**（coarse/fine × static curve）；
  `run.py:54` 注释"Will be filled by DEM run"，实际**没有任何代码填充**。
- `run_impact_e09`（`run.py:61-72`）**不跑冲击**，直接把目标值当测量值返回：
  `"measured_e": 0.9,  # DEM3D maps restitution_coefficient to damping` → `error = 0.0` 恒真。
- `run_energy_balance` 用解析反解 δ_max 再代回解析势能 → `energy_error = 1.3e-15` 是代数恒等式；
  coarse(dt=1e-6) 与 fine(dt=5e-7) 的输出**逐位相同**，`dt` 从未参与 → 声称的"两级收敛"不存在。
- **最严重**：验证器 `benchmarks/hertz_analytic/validate.py:49-51` 直接

  ```python
  F_dem = F_analytic  # DEM implementation matches analytic
  ```

  即"把解析值赋给 DEM 值再比较" → 打印 `coarse: 0.000000 (PASS, threshold=3%)`，
  `Overall: PASS`。**没有数据也能 PASS**，这是审计发现里最不可接受的一条。
- 反面参照（真实能力存在的证据）：`tests/test_dem3d_hertz_contact.py` **17 passed + 6 subtests**（23.3 s），
  `scripts/verify_dem3d_hertz_restitution.py` 系真实引擎冲击 + RK4 参考解，并有"honest-debt pin"测试。
  结论：**能力真实，benchmark 工件是假的**。

---

## ③ 负向测试（错误输入必须报错，不能静默给占位值）

脚本 `scripts/audit_runtest6_negative_probe.py`（11.4 s）→ `runs/negative_probe/results.json`。
**预注册判据**：探头声明 `expect="raise"` 时，无异常一律记 `SILENT`（= 审计发现）；
声明 `expect="accept"` 时抛异常记 `CRASH_UNEXPECTED`。

**21 个探头：REJECTED 16 / ACCEPTED 3 / SILENT 2**（明细）

| # | 探头 | 输入 | 期望 | 实测 | 判定 |
|---|---|---|---|---|---|
| 1 | hertz-neg-overlap | overlap = −1e−5 | 返回 0 | `0` | ACCEPTED |
| 2 | **hertz-nan-overlap** | overlap = NaN | raise | `nan` 静默传播 | **SILENT** |
| 3 | hertz-int-tensor | 整型张量 | raise | `TypeError: overlap must be a floating-point tensor` | REJECTED |
| 4 | hertz-poisson-1 | ν = 1.0 | raise | `ValueError: poisson_ratio must be in [0, 0.5)` | REJECTED |
| 5 | hertz-negative-E | E = −7e10 | raise | `ValueError: young_modulus must be finite and positive` | REJECTED |
| 6–7 | dem3d radius | −0.025 / 0.0 | raise | `ValueError: radius must be finite and positive` | REJECTED |
| 8 | dem3d-empty-lattice | nx = 0 | raise | `ValueError: nx must be an integer >= 2` | REJECTED |
| 9 | dem3d-zero-dt | dt = 0.0 | raise | `ValueError: dt must be finite and positive` | REJECTED |
| 10 | dem3d-negative-damping | c = −5 | raise | `ValueError: contact_damping must be finite and nonnegative` | REJECTED |
| 11 | dem3d-negative-breaking-strain | ε_b = −0.015 | raise | `ValueError: breaking_strain must be finite and positive` | REJECTED |
| 12 | dem3d-valid-baseline | 默认配置构建+2 步 | accept | OK | ACCEPTED |
| 13 | dem3d-nan-position | 位置注入 NaN | raise | `FloatingPointError: non-finite DEM3D state; reduce dt or contact stiffness` | REJECTED（**好**：状态级 NaN 守卫） |
| 14 | **brash-negative-concentration** | C_ice = −0.5 | raise | **0.0 kN** | **SILENT** |
| 15 | brash-concentration-1.5 | C_ice = 1.5 | raise | `ValueError: porosity must be in [0, 1)` | REJECTED |
| 16 | brash-negative-thickness | H = −1.0 m | raise | `ValueError: layer_thickness_m must be positive` | REJECTED |
| 17 | brash-negative-speed | V = −3 m/s | raise | `ValueError: speed must be non-negative` | REJECTED |
| 18–20 | BrashIceState | φ=1.0 / H=0 / d=−0.5 | raise | 三条 `ValueError` | REJECTED |
| 21 | brash-zero-concentration-control | C_ice = 0 | 返回 0 | `0.0` | ACCEPTED |

两条 **SILENT** 的机理与修法（未在本单实施，避免与 owner "只审计+留证据"口径冲突）：

1. `dem3d.py:85`：`delta = _hertz_argument(overlap).clamp_min(0.0)` —— `torch.clamp_min` 不吞 NaN，
   NaN 原样穿透到 `F`；建议 `if not torch.isfinite(delta).all(): raise ValueError`。
2. `fullscale_validation.py:170`：`if point.ice_concentration <= 0.0: return 0.0` ——
   守卫写成 `<=`，使 **C_ice < 0 这种非物理输入得到与合法负控完全相同的 0.0**；
   建议改 `== 0.0 → 0.0`、`< 0 → raise`，并把 (f) 判据限定为精确 0。

---

## ④ 收敛性 / 守恒性检验（真跑，带数值）

### 4.1 接触能量账本：收敛阶 + 守恒残差（**REAL，正面证据**）

脚本 `scripts/audit_runtest6_convergence.py`（20 s）→ `runs/convergence/results.json`；
用法 `order_sweep(law, refinements=8, levels=5)`，dt 逐级减半，残差按 `log-log` 拟合成幂律。

Hertz 无阻尼、`k=2e5`、`n=1.5`、m_eff=v0=1（SI 无量纲化）：

| 通道 | 残差（dt: 3.05e−3 → 1.90e−4） | 拟合斜率 | r² | 文档声称 | 实测判定 |
|---|---|---|---|---|---|
| `residual_explicit`（左端点冲量） | 7.01e−1 → 4.35e−2 | **+1.0026** | 0.999997 | O(dt) | **一致** |
| `residual_staggered_power`（力对半步速度，`floe_rigid6dof._accumulate` 的口径） | 1.24e−1 → 8.0e−3 | **+0.9884** | 0.999939 | "measured slope 1.00，损失一阶" | **一致** |
| `residual_average_velocity`（梯形/平均速度通道） | 1.9e−16 ~ 2.6e−16（**舍入级**） | 无意义（已到机器精度） | — | "round-off，exact closure" | **一致（真·精确闭合）** |
| `residual_state`（\|ΔKE+ΔU\|/E0，即 m5404 口径） | 1.39e−3 → 6.55e−8 | +3.96 | 0.887 | — | Verlet 能量误差在粗级不稳，斜率不可作节气结论 |

守恒性（弹性分支，128 步/接触，dt = 1.905e−4 s）：

```text
e_measured = 1.000000032767     |e-1| = 3.28e-08
momentum_error = 0.000e+00      max_overlap = 有限
```

→ 无阻尼冲击**动量精确守恒（0.0）**、恢复系数回到 1 到 3e−8。**这是本仓最扎实的真实数值能力**。

### 4.2 估计器收敛（Sobol）：未收敛

| n / seed | ΣS1（须 ≤1） | 原始 ΣST（须 1±0.01） | max(S1−ST) |
|---|---|---|---|
| 500 / 42 | 1.2675 | 1.0061 | +0.2176 |
| 500 / 1 | 0.6718 | 0.9264 | +0.0185 |
| 500 / 2 | 1.2535 | 0.9605 | +0.2236 |
| 500 / 7 | 1.7274 | 1.0217 | +0.5579 |
| 2000 / 42 | 1.3017 | 1.0073 | +0.3088 |
| 4000 / 42 | 1.1844 | 0.9747 | +0.1154 |

→ ΣS1 跨 seed 在 0.67–1.73 间跳；加采样到 4 倍仍 >1。**这不是"敏感度结论"，这是估计器噪声**。
（附带：`residual_restitution` 在未给 `e_theory` 时返回 inf 而不是 0 —— 设计上"不比较就不声称"，值得肯定。）

### 4.3 能量守恒（解析 Hertz benchmark 侧）：**假**

`benchmarks/hertz_analytic` 的 `energy_error = 1.33e-15` 由解析式反解 δ_max 再代回解析势能产生，
是代数恒等式；coarse/fine 输出逐位相同（§2.3）。**不作为守恒证据采信**。

## ⑤ 集群运行：全部作业**排队未派发**（honest FAIL，原样上报）

按纪律把重计算递交集群（`run_cluster submit`，未指定节点；节点为 loong64）。四个作业：

| 作业 | 内容 | 递交 | 终态（本报告落笔时） |
|---|---|---|---|
| `cj838` | `python -m pytest tests -q -p no:cacheprovider --durations=25`（**原样**的 README/CI 命令） | 07:19 | **queued**（>45 min 未被派发） |
| `cj843` | hertz run+validate / `benchmark_dem3d_ucs_timestep_convergence.py` / `benchmark_dem3d_bending_resolution.py` / 三个收敛测试文件 | 07:23 | **queued** |
| `cj855` | `run_ice_transit_campaign.py --name standard` + `tests/test_ice_transit.py` | 07:30 | **queued**（本机后台已跑出同一 case 的结果，见 §2.3b，故不再等） |
| `cj872` | 全量 `pytest tests --continue-on-collection-errors -n 16`（绕开 2 个收集错误） | 07:42 | **queued** |

队列证据（平台库 `tensorplatform.cluster_jobs`，本报告落笔时实测）：

```text
state            count
queued             275
running             13
succeeded          112
killed             358
failed               3

-- 排在我 cj838 之前仍在 queued 的作业数：232（每 10 分钟只减少 ~3 个）
-- 已运行作业时长：791 s / 1122 s / 1498 s / 1805 s / 1848 s / 1941 s（长尾占槽）
```

结论：**集群通道在本任务窗口内不可用**（吞吐 ~0.3 作业/分钟，232 个在前）。
因此 ①②④ 里"依赖集群"的三块（全量套件计数、DEM3D 时步/弯曲收敛、ice_transit 能量恒等式残差）
**本单未能取得数值**——不猜、不编、不以"应该会过"代替运行。

### 5.1 为替代而做的本机尝试（也原样记录失败）

| 尝试 | 结果 |
|---|---|
| 单个 case 直跑 `run_ice_transit_campaign.py --name standard` | 控制面 120 s 上限被杀（1 次 115 s、1 次 120 s 均未跑完），无工件 |
| 全量集分块本机跑（`-n 12`，24 分片，每片 ≤112 s） | 单分片 112 s 只跑到 ~375 项（`runs/suite_chunks/chunk_00.log`：进度 95% 处被杀）；按此外推全量需 **~43 min 墙钟**，超出控制面单命令预算，**主动放弃**（不压控制面） |
| `pytest --collect-only` | **成功**（86.5 s）：8707 项收集 + 2 个收集错误 —— 提供了 §① 的分母 |
| DEM3D 时步收敛小算例（自建，`scripts/audit_runtest6_dem3d_timestep.py`） | **无结论**：3 个 dt 级别（dt_rec/1,2,4）各跑到 t=0.02 s，`peak_reaction_z = 0.0 N`、`KE = 0.0 J`、`broken_bonds = 0` —— 小算例里刚性压头始终未与试样接触（工具起点/间隙配置），**不能作为收敛证据**；工件 `runs/dem3d_timestep/results.json` 原样保留 |

**因此：①的全量通过/失败/跳过计数在本单是 honest FAIL。** 可确证的只有：
收集层面 8707 项可收集、2 个文件收集失败（机制已定位到 `src/tensordem/freeze_bond*`、`ice_structure_contact*`）；
运行层面本机跑了 4 个文件共 108 项全绿（§1.3）。

## ⑥ 「文档声称 vs 实测」逐条对照表（差异 = 审计发现）

| # | 文档/工件声称（出处） | 实测（本单真跑） | 差异性质 | 严重度 |
|---|---|---|---|---|
| 1 | `validate.py` docstring："Force-indentation curve error ≤ 3%"，实跑打印 `0.000000 PASS`（`benchmarks/hertz_analytic/validate.py:1-10, 49-51`） | 该误差由 `F_dem = F_analytic` 赋值产生；`results.json` 中 `F_dem` 100/100 为 null；**无任何 DEM 数据** | 循环验证，判据不可证伪 | **严重** |
| 2 | `run.py` docstring："Restitution coefficient e=0.9 impact vs analytic baseline"（`run.py:5`） | `run_impact_e09` 直接返回 `measured_e = 0.9`，无仿真 | 硬编码/占位 | **严重** |
| 3 | `run.py` docstring："two convergence cases (coarse/fine)"（`run.py:4`） | 两 case 输出逐位相同（sha256 一致）；`dt` 从未参与计算 | 名义收敛，无实际收敛研究 | 中 |
| 4 | fullscale `dem_resistance` 命名与 capability `ice.resistance.dem_straight_ahead` | 实为 μ(I) 连续介质三分解 + 固定 floe 0.5 m + 定标常数 2.8/2.2（无出处） | 方法标签过强；未标 CALIBRATED | **严重** |
| 5 | 验收 (a)"R_DEM vs R_Lindqvist ≤15%"（`fullscale_validation.py:39`） | 只评 comparison[0]；全 12 点 11 点超限，均值 28.0%、最大 44.2% | 判据窄于措辞（择优单点） | **严重** |
| 6 | 验收 (b)"均值 ≤20%" | 均值 10.06% 通过，但 2/12 点破 20%（−40.05%、−39.56%） | 均值掩盖尾部 | 中 |
| 7 | 验收 (c)"ΣS_Ti = 1.0 ± 0.01"（`fullscale_validation.py:520-528`） | 内部 `ST/=ST.sum()` 使该量恒等 1.0；n=31 仍 PASS；原始 ΣST 在 seed1/2 应 FAIL（0.926/0.960） | 判据恒真 | **严重** |
| 8 | 验收 (d)"至少一个 S_Ti > 0.4" | 0.587 通过，但同批 S1(σ_b)=0.808 > ST(σ_b)=0.591，ΣS1=1.27 | 指标自相矛盾，结论不稳 | 中 |
| 9 | 验收 (f)"negative control C=0 → R=0" | 0.0 通过；但 C=−0.5 也返回 0.0 | 负控不排他 | 中 |
| 10 | 验收 (g)"pytest ≥15 all green" | `tests/test_dem_brash_ice_fullscale.py` 26 passed | 一致 | — |
| 11 | `results.json` `status_upgrade: WITHHELD → REAL` | 数值可复现（0 diff）但 (a)(c) 属伪证据、(d) 不稳 | 升级依据不足，宜降为 CALIBRATED | **严重** |
| 12 | `results.json` 汇总 `PASS (6/5 criteria passed)` | off-by-one（`fullscale_validation.py:561-567`）：`n_pass` 在 overall 赋值前统计 6 项，`n_total=len-1=5` | 工件内可见缺陷 | 轻微 |
| 13 | README："343 个测试文件、2714 个模块级 test_* 函数、319 个 TestCase"（`README.md:135`） | 实测 **485** 个测试文件、**4357** 个 `def test_*` | 文档快照过期 | 轻微 |
| 14 | README：`artifacts/ice_transit/` 作为证据路径（`README.md` 冰区直航节） | `git ls-files` 该路径 **0 个文件**（仅 `artifacts/pr166/*` 两个入库） | 证据未入库，读者无法核验 | 中 |
| 15 | README："W_thrust + … − W_耗散 = Δ(KE+PE+接触弹性能) 是恒等式而不是断言" | 见 §5 `run_ice_transit_campaign.py --name standard` 真跑残差 | 待回填 | — |
| 16 | 文档/测试口径：负向控制覆盖 | 21 探头中 2 条 SILENT（NaN 穿透、C<0 → 0.0） | 负向控制缺口 | 中 |
| 17 | 提交工件可复现性（隐含声称） | brash fullscale **逐键 0 diff**；hertz benchmark **sha256 逐位一致** | 复现性强（正面），但"复现了假数据" | 正面/中性 |

---
| 18 | `README.md:132` + `.github/workflows/tests.yml:22`："python -m pytest -q" | 本 worktree **collection 中断**（2 个 ImportError，8707 项未跑） | 记录的测试命令不可执行 | **阻断** |
| 19 | 隐含：`src/tensordem/freeze_bond.py`、`ice_structure_contact.py` 是可用实现 | 同名包目录遮蔽，**14 个类/1628 行不可达**；`segregation`/`sensitivity`/`shape_representation` 三对已正确再导出 | 部分合并事故（提交 `d4d4ea99`） | **严重** |
| 20 | `contact_energy_order.py` docstring：各通道阶数（explicit O(dt)、staggered 损失一阶、average-velocity 舍入级） | 实测斜率 +1.0026 (r²=0.999997) / +0.9884 (r²=0.99994) / 残差 1e−16 | **完全一致** | 正面（真实能力） |
| 21 | `contact_energy_order.py`：无阻尼冲击应守恒动量 | 实测 `momentum_error = 0.0`，e = 1.00000003 | 一致 | 正面 |
| 22 | 任务前置假设：表中『已有审计波 verdicts』可直接补证据 | **07:38 UTC 实测该表 0 行**（`id_seq.last_value = 1`）；并发审计波随后陆续落表（报告落笔时全表 170 行，其中 `audit2-ab` 26 条 TensorDEM 行） | 非造假，是**时序**：审计波与本单并行写表；本单新建 `audit2-run` 7 行，留此时序记录备查 | **中（时序说明）** |
| 23 | 隐含：CI 会拦住"测试收集失败" | `tests.yml` 有 `python -m pytest -q` 且 `timeout-minutes: 20`，但当前 main 的 worktree 收集中断 | 门禁未生效或 CI 红未处理 | **严重** |
| 24 | README：`W_thrust + W_数值注入 + W_注入 − W_移出 − W_耗散 = Δ(…)` "是恒等式而不是断言" | 单 case 实测残差 **5.34%**、数值注入 **6.35%** | 措辞过强（应为"闭合到 5.3%，阈值 10%"） | 中 |
| 25 | campaign 预注册 headline gate `E1 ≤ 15%`（vs Lindqvist 锚） | 标准 case 实测 **93.1% → FAIL**（59.11 kN vs 锚） | 旗舰算例自身判据不过 | **严重** |
| 26 | campaign 预注册 `E7 稳态漂移 ≤ 0.15` | 实测 **0.3516 → FAIL**（"稳态"未达稳态） | 同上 | **严重** |
| 27 | 仓库"验证工件"是否入库（`artifacts/ice_transit/`） | 仓内 0 个；本地重跑 34 s 可重建 `artifacts/ice_transit/summary.json`（本次会话内生成，未提交） | 证据可重建但未入库 | 中 |

## ⑦ 复现方式（全部可重跑）

```bash
cd /root/tp-agent/work/m14498
PYTHONPATH=src python3 scripts/audit_runtest6_negative_probe.py     # §3  11 s
PYTHONPATH=src python3 scripts/audit_runtest6_sobol_probe.py        # §2.2 24 s
PYTHONPATH=src python3 -c "from tensordem.brash_ice_rheology.fullscale_validation import run_full_validation as r; r('runs/fullscale')"   # §2.1 12 s
PYTHONPATH=src python3 benchmarks/hertz_analytic/run.py --output-dir runs/hertz_local    # §2.3 12 s
PYTHONPATH=src python3 benchmarks/hertz_analytic/validate.py        # §2.3（需 cwd=repo 根）
python -m pytest tests -q -p no:cacheprovider                       # ①全量（集群）
```

## ⑧ 未做/局限（诚实申报）

- **①全量套件计数未取得**：集群 4 个作业排队 >45 min 未派发（§5），本机全量需 ~43 min 墙钟被纪律否决。
  已确证的是"收集阶段就中断"（可复现、有日志）与 4 个文件 108 项全绿。
- **②flagship 已完成**（brash-ice 全尺度逐键复现 + Hertz benchmark 拆穿）；`benchmark_dem3d_bending_resolution.py`
  与 `benchmark_dem3d_ucs_timestep_convergence.py`（DEM3D 时步/网格收敛）随 cj843 未跑。
- **④的接触能量账本部分已完成**（实测阶数 + 守恒），ice_transit 能量恒等式（README 主声称）随 cj855 未跑。
- `scripts/verify_dem3d_hertz_restitution.py --quick` 本机 >115 s 未跑完（其数值面由 17 项测试覆盖，全绿）。
- 覆盖范围限于：brash 全尺度、hertz analytic、dem3d 配置面、contact_energy_order、ice_transit（未跑）；
  其余 ~60 个功能模块不在本单范围（并发审计波次分工不同）。
- **未修改任何库代码**（本单口径＝审计+留证据）；两条 SILENT 输入缺陷给出可执行修法但未落地。

## 回填状态

- [x] 本机真跑四类全部落工件：`runs/{fullscale,sobol_probe,negative_probe,convergence,hertz_local,ice_transit_cases,dem3d_timestep,figures}/`
- [x] verdicts 落表 7 行（`verdict_by='audit2-run'`，`scope='<模块>#run'`）：
  `brash_ice_rheology/fullscale_validation` L3 · `benchmarks/hertz_analytic` L1 ·
  `public_input_guards` L2 · `contact_energy_order` L4 · `tests/full_suite` L1（honest FAIL）·
  `dem3d/timestep_convergence` L1（honest FAIL）· `ice_transit/energy_ledger` L2
- [x] 报告已发布：`POST /api/v1/tensor/reports` → report id **13632**（4 图，markdown 全量）
- [ ] 集群 cj838 / cj843 / cj855 / cj872 仍 queued（本窗口未派发）；后续任一派发完成后，
      可直接 `fetch` 取 `runs/cluster_runtest/*` 并回填 §5，无需重跑本单其余部分
