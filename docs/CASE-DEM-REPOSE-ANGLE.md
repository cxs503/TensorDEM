# CASE-DEM-REPOSE-ANGLE — 玻璃珠休止角 θ_r = 22°（TensorDEM）

*mission 19975 · branch `agent/m19975` · 案例池第 1 轮 · 总判定 **PASS**（route: calibrated）*

## ① 问题与参考文献出处

**问题**：TensorDEM 已有的提升圆筒（lifting-cylinder）休止角协议，在其已有的 Cundall–Strack 切向 + EPSD 滚阻接触模型下，能否复现玻璃珠实验休止角 **22°**（判据 ±3°）？

**参考文献**：B. T. Brzinski III, M. Mayor, D. J. Durian, Phys. Rev. Lett. 111, 168002 (2013); arXiv:1307.4638v2；本仓取文版本 arXiv:1307.4638v2。

逐条引用（原文位置：**arXiv:1307.4638v2 p. 1, col. 2 (medium description, one line)**）：

| 量 | 原文数值 | 本算例用途 |
|---|---|---|
| 介质休止角 | **22°** | 对照目标 θ_ref |
| 内摩擦系数 | **μ = 0.40** | 与 tan(22°) 互校 |
| 松装密度 | **ρ_bulk = 1.48 g/cm³** | 派生颗粒密度 |
| 装填率 | **φ = 0.59** | 派生颗粒密度 |
| 玻璃珠粒径 | **d = 300 ± 50 µm** | 粗粒化基准 |

**零手抄派生（代码计算，非人工抄写）**：ρ_p = ρ_bulk/φ = **2508.5 kg/m³**（玻璃，合理）；μ = tan(22°) = **0.4040**，与原文 0.40 相对差 **1.01%**（原文自身两个数字自洽）。

**预注册的两个备择假设**（`docs/CASE-DEM-REPOSE-ANGLE_prereg.md`，跑前冻结）：

* **H1 预测型**：坡度只由滑动摩擦控制 → μ = tan22° = 0.404 加玻璃滚阻先验 μ_r = 0.05 即可得 22°±3°；
* **H0 滚阻受限**：单分散光滑球的 θ_r ≈ arctan(min(μ, μ_r))，必须有 μ_r ≳ tan22° 才能撑住 22° 坡 → 该算例**可标定但不可预测**。

## ② 方法与实现（复用现有能力，不另起炉灶）

* 求解器：`tensordem.repose_material_calibration.simulate`（提升圆筒柱体坍塌；线性弹簧-阻尼法向 + Cundall–Strack 切向弹簧饱和于 μF_n + EPSD 滚阻力矩 capped μ_r·R_eff·F_n，粗糙壁与粗糙底）。
* 宏观量估计器：`tensordem.repose_calibration.repose_angle`（环带×扇区自由面高度，肩-趾 15–90% 区间最小二乘拟合 + 单边 2σ 剔除）——**未重写**，因此本算例的差异不可能来自估计器。
* 本单只新增：参考文献与派生、预注册 legs 编排、并行执行、反演、判据与图表（`benchmarks/case_dem_repose_angle/run_case.py`）。
* 协议时序：hold 0.04 s / lift 1.20 s / settle 0.60 s；dt = T_stiff/20；每个 realisation 单线程（位级可复现），8 个 worker 并行。
* 环境：python 3.11.6，torch 2.7.1 CPU，47 个 realisation，墙钟 784 s（集群作业 cj2409）。
* 一键复现：`python3 benchmarks/case_dem_repose_angle/run_case.py --out artifacts/CASE-DEM-REPOSE-ANGLE --jobs 8`

**legs 编排**（N=负对照，P=H1 预测，S=μ_r 扫描，C=标定后留出验证，M/E/W/L/R=五项敏感性）：

| leg | 含义 | 工况 | seeds |
|---|---|---|---|
| N | 无摩擦负对照 μ=μ_r=0 | `N` | 19981 |
| P | H1 预测：μ=tan22°, μ_r=0.05 | `P` | 19981, 19982, 19983 |
| S | μ_r 扫描 0.05…0.50（9 节点） | `S` | 19975, 19976 |
| C | μ_r* 留出种子验证 | `C` | 19981, 19982, 19983 |
| M | μ = 0.30 / μ_ref / 0.50 | `M` | 19981, 19982 |
| E | 粒径 r = 3 / 6 mm（粗粒化） | `E` | 19981, 19982 |
| W | 容器 R_c/r = 5 / 8 | `W` | 19981, 19982 |
| L | 提升时间 0.90 / 1.50 s | `L` | 19981, 19982 |
| R | 恢复系数 e = 0.30 / 0.90 | `R` | 19981, 19982 |

## ③ 参数表（含无量纲数）

| 参数 | 值 | 说明 |
|---|---|---|
| R_c / r | 5.00 | container confinement (lifting cylinder radius / grain radius) |
| H / R_c | 9.00 | column aspect ratio (fill height / cylinder radius) |
| k_n / (m g) | 3.032e+05 | stiffness-to-weight (contact stiffness number) |
| dt / T_stiff | 0.05 | timestep / Rayleigh-like contact period (fixed by the solver) |
| mu | 0.4040 | Cundall-Strack sliding friction (from the reference friction angle) |
| mu_r | 0.2882 | EPSD rolling-resistance coefficient |
| mu_r / mu | 0.713 | rolling/sliding ratio (the identifiability axis) |
| e | 0.70 | normal restitution |
| d_DEM / d_exp | 26.7 | coarsening factor vs the 300 um laboratory beads |
| U_lift / sqrt(g d) | 0.535 | Froude number of the wall withdrawal |
| N_grains | 338 | particle count |
| n_steps | 14285 | timesteps per realisation |
| theta_r | 19.392 deg | measured repose angle (this run) |

## ④ 对照表（ref vs got，逐项误差）

参考 θ_ref = **22°**，容差 **±3°**（任务判据）。

| leg | 需求 | 工况 | 有效种子 | θ 均值 [deg] | σ [deg] | Δ = got−ref [deg] | |Δ| ≤ 3° ? | 相对 |Δ|/ref | 质量标记 |
|---|---|---|---|---|---|---|---|---|---|
| N | frictionless control | `frictionless control` | 1/1 | -0.00 | 0.00 | -22.00 | FAIL ❌ | 100.00% | 1 |
| P | H1 prediction (mu_r=0.05) | `prediction mu_r=0.05` | 3/3 | 7.79 | 0.53 | -14.21 | FAIL ❌ | 64.61% | 0 |
| S | mu_r scan | `mu_r=0.05` | 2/2 | 8.61 | 0.00 | -13.39 | FAIL ❌ | 60.84% | 0 |
| S | mu_r scan | `mu_r=0.1` | 2/2 | 12.22 | 2.41 | -9.78 | FAIL ❌ | 44.44% | 0 |
| S | mu_r scan | `mu_r=0.15` | 2/2 | 13.18 | 0.01 | -8.82 | FAIL ❌ | 40.10% | 0 |
| S | mu_r scan | `mu_r=0.2` | 2/2 | 21.43 | 1.35 | -0.57 | PASS ✅ | 2.59% | 0 |
| S | mu_r scan | `mu_r=0.25` | 2/2 | 15.07 | 3.85 | -6.93 | FAIL ❌ | 31.48% | 0 |
| S | mu_r scan | `mu_r=0.3` | 2/2 | 24.14 | 2.35 | +2.14 | PASS ✅ | 9.73% | 0 |
| S | mu_r scan | `mu_r=0.35` | 2/2 | 30.04 | 1.67 | +8.04 | FAIL ❌ | 36.56% | 0 |
| S | mu_r scan | `mu_r=0.404` | 1/2 | 23.10 | 0.00 | +1.10 | PASS ✅ | 5.00% | 1 |
| S | mu_r scan | `mu_r=0.5` | 2/2 | 29.12 | 2.13 | +7.12 | FAIL ❌ | 32.37% | 0 |
| C | calibrated holdout | `holdout at mu_r*` | 3/3 | 21.27 | 3.56 | -0.73 | PASS ✅ | 3.33% | 0 |
| M | sliding-mu sweep | `mu=0.3` | 2/2 | 13.76 | 1.56 | -8.24 | FAIL ❌ | 37.45% | 0 |
| M | sliding-mu sweep | `mu=0.404` | 2/2 | 20.01 | 5.81 | -1.99 | PASS ✅ | 9.05% | 0 |
| M | sliding-mu sweep | `mu=0.5` | 2/2 | 16.80 | 4.57 | -5.20 | FAIL ❌ | 23.64% | 0 |
| E | grain radius | `r=3mm` | 2/2 | 23.11 | 6.19 | +1.11 | PASS ✅ | 5.03% | 0 |
| E | grain radius | `r=6mm` | 2/2 | 19.23 | 0.04 | -2.77 | PASS ✅ | 12.60% | 0 |
| W | container | `Rc/r=5` | 2/2 | 22.38 | 4.23 | +0.38 | PASS ✅ | 1.73% | 0 |
| W | container | `Rc/r=8` | 2/2 | 17.10 | 1.33 | -4.90 | FAIL ❌ | 22.29% | 0 |
| L | lift rate | `lift=0.9s` | 2/2 | 20.93 | 1.62 | -1.07 | PASS ✅ | 4.86% | 0 |
| L | lift rate | `lift=1.5s` | 2/2 | 19.11 | 5.15 | -2.89 | PASS ✅ | 13.15% | 0 |
| R | restitution | `e=0.3` | 2/2 | 19.08 | 3.54 | -2.92 | PASS ✅ | 13.27% | 0 |
| R | restitution | `e=0.9` | 1/2 | 17.76 | 0.00 | -4.24 | FAIL ❌ | 19.27% | 1 |

**判据逐条**：

| 判据 | 内容 | 数值 | 判定 |
|---|---|---|---|
| AC-1 | predictive leg (mu=tan22, mu_r=0.05, 3 held-out seeds) within 3 deg | θ = 7.79 deg，Δ = -14.21 deg | **FAIL** |
| AC-2 | calibrated mu_r* on 3 held-out seeds within 3 deg | θ = 21.27 deg，Δ = -0.73 deg | **PASS** |
| AC-3 | frictionless control flattens (theta < 5 deg) | θ = -0.00 deg | **PASS** |
| AC-4 | theta(mu_r) invertible on [0.05, 0.35] | monotone-within-2σ = True | **PASS** |
| AC-5 | radius/container sensitivity <= 3 deg (informational) | worst |Δ| = 4.17 deg | **FAIL** |
| CASE | AC-3 AND (AC-1 OR AC-2) | rule: AC-3 AND (AC-1 OR AC-2) → route **calibrated** | **PASS** |

**μ_r 反演**：目标 22° 在扫描曲线上 bracketed，μ_r* = **0.2882**（括号区间 0.250–0.300，对应 θ 15.07°–24.14°，插值 1σ ≈ ±2.01°）。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

**（a）种子离散（重复性）**：S 节点 2 seeds、C 腿 3 seeds；各腿 σ 见 ④ 表。σ 与 ±3° 判据的比值即为该算例的分辨率上限。

**（b）参数敏感性**（相对标定基线 θ = 21.27°）：

| 轴 | 工况 | θ [deg] | Δ vs 基线 [deg] | ≤3° ? |
|---|---|---|---|---|
| 滑动 μ | `mu=0.3` | 13.76 | -7.51 | FAIL ❌ |
| 滑动 μ | `mu=0.404` | 20.01 | -1.26 | PASS ✅ |
| 滑动 μ | `mu=0.5` | 16.80 | -4.47 | FAIL ❌ |
| 粒径 r（粗粒化） | `r=3mm` | 23.11 | +1.84 | PASS ✅ |
| 粒径 r（粗粒化） | `r=6mm` | 19.23 | -2.04 | PASS ✅ |
| 容器 R_c/r | `Rc/r=5` | 22.38 | +1.11 | PASS ✅ |
| 容器 R_c/r | `Rc/r=8` | 17.10 | -4.17 | FAIL ❌ |
| 提升速率 | `lift=0.9s` | 20.93 | -0.34 | PASS ✅ |
| 提升速率 | `lift=1.5s` | 19.11 | -2.16 | PASS ✅ |
| 恢复系数 e | `e=0.3` | 19.08 | -2.19 | PASS ✅ |
| 恢复系数 e | `e=0.9` | 17.76 | -3.51 | FAIL ❌ |

最大偏离来自 AC-5 记录：4.17°（判据 ≤ 3°，FAIL）。

**（c）收敛性（时间步/刚度）**：k_n 沿用已验证的刚度过收敛值 2000 N/m（`docs/repose_material_calibration.md` §3.6：k_n 2000→8000 只移动1.4°，小于本算例容差），dt = T_stiff/20 为求解器固有权值；本单未重复刚度扫描，属已知局限，见 ⑥。

## ⑥ 失败模式、工件与生态摩擦

**失败模式（本算例实测/可预期）**

* 估计器拒算：未坍塌的柱体没有坡面，`repose_angle` 抛`no measurable flank`——本单把这类 run 记为 `error` 并计入 `n_error`，不用 0° 冒充结果（quick 冒烟测试即命中该路径）。
* 未静止即测量会系统性抬高角度（前序 m17903/GAP-DEM-CLUMP 已量化 6.8°），故本单沿用固定 settle 0.60 s 并检查 `max_speed` 标记。
* 平坡时 `r2` 天然低（负对照），因此质量标记只作诊断、判定只用角度。

**工件**

* `artifacts/CASE-DEM-REPOSE-ANGLE/results.json`（每个 realisation 的全部诊断 + 反演 + 判据）
* `artifacts/CASE-DEM-REPOSE-ANGLE/criteria.json`、`artifacts/CASE-DEM-REPOSE-ANGLE/TABLES.md`（本文表格源）
* `benchmarks/case_dem_repose_angle/run_case.py`（一键复现）
* `docs/CASE-DEM-REPOSE-ANGLE_prereg.md`（预注册）、`tests/test_case_dem_repose_angle.py`（11 项）

**生态摩擦（如实记）**

* 开工三查第 1 查（knowledge_base 同域 capability）：`/tensor/knowledge/search` 返回 `engine_offline`（127.0.0.1:8021 拒绝连接）；`/knowledge/search` 可用但沉淀库 `knowledge_sediment/knowledge.jsonl` 为**0 条**，`q='repose angle glass beads'` 命中 0 → 同域能力只能从仓库文档（`docs/repose_material_calibration.md` 等）反查。
* 第 2 查（negative_knowledge_ledger keywords='TensorDEM'）：台账 1028 条 active，但条目 `keywords`/`domain_tags` 多为空，`q='TensorDEM'`、`q='repose'` 均 **0 命中**；模糊检索只返回 char-overlap 的无关 TensorLBM 条目（score 0.09）。
* 第 3 查（oss_solutions 同域）：PG 中确有 `oss_solutions` 表，但 agent 角色 **permission denied**，平台也无对应 REST 端点（候选路径全 404）。
* 本机计算门把 60 s 级探针（后台 `nohup`）一并拦截，只能整批走 compute gate；对本算例是合理成本，但对“只想测一次运行时间”的探查不友好。

**局限/适用范围**

* 单分散光滑球：真实玻璃珠有轻微不规则度，标定出的 μ_r* 是**数值等效量**，不是测得的材料参数。
* 一个休止角无法分离 (μ, μ_r)（前序否定性知识已记），扫描只能给一条 locus。
* 实验室粒径 300 µm、DEM 粒径 8 mm（粗粒化 26.7×），E 腿只界定其影响，不能消除。
