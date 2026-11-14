# CASE-DEM-BIAXIAL-FRICTION · 第 2 轮调研：成熟方法 → 选型 → 迁移方案

**任务**：m20304｜仓库 TensorDEM｜锚点 `CASE-DEM-BIAXIAL-FRICTION`
**算例池判据**：`ref: 经典临界状态`，容差 `<8%`
**前轮**：m20191 honest-FAIL（`fail_stage=未标`，`failure_reason=未留`）；更早 m20052 以「峰值 40.36° vs 38° = 6.20% PASS」结单，但该 PASS 有可复核的方法学缺陷（见 §0）。

---

## §0 前轮死因复盘（先查尸检，避免重蹈）

对 `docs/CASE-DEM-BIAXIAL-FRICTION.md`（m20052 产物）+ `src/tensordem/biaxial_friction_case.py` 的独立复核结论：

| # | 缺陷 | 证据 | 后果 |
|---|---|---|---|
| D1 | **判据锚点与模型不匹配**：ref 声明为"Bolton 1986 Table 2 密集 Hostun 砂平面应变**峰值** 38°(D_r=80%, σ3=100 kPa)"，但 Bolton 自己的关系式给出 φ_ps = φ_crit + 5·I_R = 33 + 5·[0.8·(10−ln100)−1] = **49.6°**；反解 38° 对应 D_r ≈ 0.37，与声明 D_r=80% 自相矛盾。 | `biaxial_friction_case.py:43-49`；Bolton (1986) Eq.(4) 与其 Q=10/R=1 参数 | 锚点数字不可溯源 → 数字"命中"无物理含义 |
| D2 | **用胶结样去对"无胶结砂"的强度**：`freeze_bond_fraction=0.35` 给颗粒间加 freeze bond（抗拉 50 kPa / 抗剪 100 kPa），峰值强度由**胶结**而非摩擦-咬合决定。 | `biaxial_friction_case.py:67`；`ridge_dem.py:115-119` | 峰值角可被胶结强度"调"到任意目标 —— 本质是拟合 |
| D3 | **峰值 ≠ 临界状态**：算例池要的是"经典**临界状态**"，m20052 报的是 σ_d 峰值（ε_a≈6.3%）处的角。峰值依赖初始密度/胶结/尺寸，不可作为临界状态判据。 | `biaxial_shear_band.py:521-527`（argmax 平滑剪应力） | 与算例池 ref 语义不符 |
| D4 | **物理自检项非法而未被拦截**：报出 dilation angle ψ = 71.81°，且 σ_3@peak = 112.9 kPa（壁）/125.2 kPa（内）vs 设定 100 kPa（偏 12.9% / 25.2%）。ψ > φ 且伺服未闭环的工况被判 PASS。 | 同 doc §3-§4 表 | 缺少"伺服闭环 + 剪胀率 → 0"闸门 |
| D5 | **工件被门禁滤除后不可复现**：`artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json` 不在库内，m20191 无留痕。 | `git log -- src/…/biaxial_friction_case.py` = 35b64a47（merge #1524） | 数字零复现 → 下一轮只能从零开始（本单要消除这一点） |

**结论**：前轮的死法不是"算错了"，而是**判据与模型错配 + 缺少收敛闸门 + 工件不留痕**。第 2 轮必须先做方法调研，把判据锚到"临界状态"自身的经典定义上。

---

## §1 调研（一）：开源横向扫描

按任务指定的 `oss_inventory` / `oss_solutions` 通道查询（`psql $TP_PG_DB`，`-U postgres`）：

| 查询 | 结果 |
|---|---|
| `SELECT * FROM oss_inventory WHERE repo ILIKE '%fric%' OR methods ILIKE '%friction%' OR methods ILIKE '%biaxial%' OR focus ILIKE '%friction%'` | **0 行**（schema 无 `feature_slug` 列；`repo/focus/methods/summary` 中无 DEM 摩擦/双轴条目） |
| `… WHERE repo ILIKE '%yade%' OR '%liggghts%' OR '%mercurydpm%' OR '%chrono%' OR focus ILIKE '%DEM%' OR '%granular%' OR '%triaxial%'` | 仅 1 行 `espressomd/espresso`（tier 3，MD–LBM 软物质耦合；**无颗粒双轴/临界状态**；GPL 仅可参照） |
| `SELECT * FROM oss_solutions WHERE feature_slug ILIKE '%BIAXIAL%' OR '%FRICTION%'` | **0 行**（本单是本 feature_slug 的首个解决方案条目） |
| 节点 `~/oss-cache`（本机可见副本） | `Maarten-vd-Sande_lbm`、`hschwane`、`pylbm` —— **全部 LBM**，无 DEM 仓 |

**横向调研的诚实结论**：OSS 缓存/清单里**没有**可"照着学结构"的 DEM 双轴实现；本生态的 DEM 清单被 LBM 语料占满。因此方法来源改为两条真实可得通道：
（a）**本仓内已验收实现的同族方法**（不重复造轮子，只补缺口）；（b）经典文献原文。

### §1.1 本仓已有实现清单（禁重复实现核对表）

| 文件 | 能力 | 与本锚点关系 |
|---|---|---|
| `src/tensordem/biaxial_shear_band.py` (611 行) | 平面应变双轴主机：抖动三角格点制样 → 等向固结 → 顶板位移加载 + 侧墙常压伺服；同时记壁面牵引应力与 Love–Weber 内应力；输出完整 `stress_path` | **复用**（引擎，不改） |
| `src/tensordem/triaxial_cd.py` (≈500 行) | 3D 三轴：**周期边界 + 侧向 barostat**（drained gate）、滚动阻力 μ_r、峰值 + 临界状态（尾部窗均值）`phi_crit` | **复用其临界状态方法学**（窗均值 + 伺服漂移闸门），不复制代码 |
| `src/tensordem/biaxial_anchor_case.py` / `biaxial_xval.py` | 已知倾角标定、交叉校验（CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE，L4 已验收） | 参照（不再动） |
| `src/tensordem/shape_error_budget.py` | 形状误差预算：`mu_eff = mu + tan i`（Rowe 1962 / Newland & Allely 1957 咬合项），量化"圆颗粒 vs 角形砂"的强度差 | **复用**：解释/预测球-砂缺口 |
| `src/tensordem/biaxial_friction_case.py` | 本锚点旧包装（bonded + 峰值 38°） | **缺口所在**：无临界状态、无闸门、锚点不可溯源 |

**明确缺口（本单只补这些）**：① 平面应变路径上"临界状态"的定位与判据化（同构于 `triaxial_cd` 的尾部窗均值，但 2D 壁面伺服域）；② 无胶结样（bond 关闭）到大应变（ε_a ≳ 0.4）的路径；③ 可溯源、与模型形状匹配的 ref；④ 收敛/敏感性与工件留痕。

---

## §2 调研（二）：方法谱系与标准算例

| 要素 | 标准做法 | 出处 |
|---|---|---|
| 临界状态定义 | 应力比（q/p）与体积**同时**稳定：dε_v/dε_1 → 0 且 dq/dε_1 → 0 | Schofield & Wroth (1968) *Critical State Soil Mechanics*, McGraw-Hill, Ch.1-2 |
| 应力-剪胀关系 | σ_1/σ_3 = tan²(45° + φ_μ/2)·(1 − dε_v/dε_1)；**临界状态 D=0 时 φ_cv = arctan μ**（摩擦型散体的经典极限） | Rowe (1962) *Proc. R. Soc. Lond. A* **269**: 500-527 |
| 角形砂临界状态值 | 石英砂 φ_crit ≈ 33°；峰值 φ_peak = φ_crit + A·I_R（A=3 三轴 / 5 平面应变） | Bolton (1986) *Géotechnique* **36**(1): 65-78 |
| 双轴/三轴边界条件 | 侧向常压伺服，**侧墙摩擦取 0**（否则侧墙承担竖向剪应力，顶板 σ_1 被系统性低估）；或用周期边界 + barostat | 本仓 `biaxial_shear_band.py:305-320`（已注明）；`triaxial_cd.py` barostat |
| 临界状态取样窗 | 取剪切段**尾部**固定比例窗（不看结果选窗），并做 10/20/30% 窗敏感性 | `triaxial_cd.py:395-400`（`critical_fraction` 尾部均值）；本单推广为三窗一致性 |
| 球 vs 角形缺口 | 圆颗粒无咬合项，φ_cv 系统性低于角形砂；咬合可用 μ_eff = μ + tan i 量化 | `shape_error_budget.py:33-37, 484`（Rowe 1962 / Newland & Allely 1957） |
| 数值必需的自检 | 惯性数 I < 0.01（准静态）、伺服漂移、两套应力定义（壁 vs Love–Weber）一致性 | 本仓既有 doc/AUDIT 惯例 |

---

## §3 选型沉淀（本单采用的方案 + 竞争假设）

### 3.1 决策与竞争假设（先声明，后跑数）

| 项 | 内容 |
|---|---|
| **决策** | 用**临界状态**（等体积 + 应力比平稳）而非峰值作为本锚点的判据量 |
| **假设 H1（摩擦型极限）** | 无胶结、无滚动阻力的圆盘组件在临界状态满足 Rowe 的 D=0 极限：**φ_cv = arctan μ = 28.81°**（μ=0.55）。若成立，8% 容差带 = [26.51, 31.11]° |
| **假设 H2（咬合型砂）** | 若把 ref 取为角形石英砂的经典临界状态值 **φ_cv = 33°**（Bolton 1986），则圆盘模型应**系统性偏低**，缺口量级由咬合项 tan i 解释（`shape_error_budget`） |
| **判别指标** | φ_cv（尾部 20% 窗均值，壁面牵引应力），及其三条不变量：\|dε_v/dε_a\| ≤ 0.02、q 窗内相对变化 ≤ 5%、伺服漂移 ≤ 10% |
| **预声明阈值** | ref 误差 < 8%（算例池）；三窗（10/20/30%）离散 ≤ 1.5° |
| **停止规则** | 若三窗离散 > 1.5° 或伺服漂移 > 10%，判"临界状态未达"，不得改窗迁就数字，如实记 WITHHELD/FAIL |
| **混杂因素** | 胶结（本单置 0）、侧墙摩擦（置 0）、初始密度（抖动 + 固结固定）、尺寸（16×32 vs 24×48）、围压（100 vs 200 kPa）、μ（0.55 vs 0.30 敏感性） |
| **不可做的事** | 不为命中数字调 μ 或胶结比例；不选窗；不改 ref |

选 H1 为**主判据**、H2 为**对照**的理由：本模型（无滚动阻力、无角形、无胶结的圆盘）的经典理论预测就是 Rowe 极限；用角形砂的 33° 去判圆盘模型，等于要求模型给出它结构上不具备的咬合强度——这正是前轮"用胶结强度去补 5°"的翻版。

### 3.2 迁移方案（结构 → 本仓）

```
run_biaxial_shear_band(…)            ← 复用（不改引擎）
    └─ stress_path: eps, σ_v, σ_h, q, ε_v
         └─ biaxial_critical_state.critical_state_from_stress_path()   ← 新增（本单）
              • 尾部窗规则（min_strain, window_fraction 预声明）
              • 三不变量闸门 + 10/20/30% 窗敏感性
              • Rowe (1962) 应力-剪胀交叉校验：由实测 D 反推 φ_cv,pred
         └─ judge_against_reference()  ← 8% 相对误差判据（两 ref 各判一次）
```

**落表**：本方案已写入 `oss_solutions`（`repo=TensorDEM`, `feature_slug='CASE-DEM-BIAXIAL-FRICTION'`, `tag='critical-state-rowe-2026-11-14'`），供后续轮次直接复用，避免重复实现。

---

## §4 实施与工件

| 物料 | 路径 |
|---|---|
| 临界状态检测模块（新增，131 行函数级） | `src/tensordem/biaxial_critical_state.py` |
| 不变量测试（合成路径，解析值对照） | `tests/test_biaxial_critical_state.py`（12 passed） |
| 探索记录 | `runs/m20304/probe/probe.json`（`scripts/m20304_probe_cs.py`，集群 cj2626） |
| 基准算例（正式） | `benchmarks/case_dem_biaxial_friction/run_critical_state.py` + `artifacts/CASE-DEM-BIAXIAL-FRICTION/` |
| 结果报告 | `docs/CASE-DEM-BIAXIAL-FRICTION_R2_RESULTS.md` |

---

## §5 适用范围与局限（如实声明）

1. 2D 圆盘、无滚动阻力、无胶结：**只能**代表摩擦型圆颗粒介质，不能代表角形砂（咬合项缺位）。因此 H2 的 33° 比较预期为**偏低**，这是物理事实而非实现缺陷。
2. 壁面伺服 vs 周期 barostat：壁面方案在小样上有边界层效应（σ_h 内/外应力差 10-25% 为已知），本单以壁面牵引为准并用伺服漂移闸门兜底；周期 barostat 的等价算例留作后续（`triaxial_cd` 已在 3D 提供）。
3. 无重力、位移控顶板、恒定应变率 0.05 s⁻¹（I ≈ 0.007 ≪ 0.01，准静态）。
4. `oss_inventory`/`oss_solutions` 表在 `root` 会话下无 SELECT 权限，需 `psql -U postgres`；`oss_inventory` 无 `feature_slug` 列（任务书中的示例查询按 schema 不可执行）—— 已按实际列查询，见 §1。