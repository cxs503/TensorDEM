# 交差核验 m16136 — CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT（TensorDEM）

审计单：m16194（board-audit）｜被审单：m16136（算例池·第1轮，DB 状态 `succeeded`）｜
被审提交：`00e704d9`（工作树 `/root/tp-agent/work/m16136`，「3 个文件被门禁滤除」式的最终成果提交）
平台裁判表：`tensor_benchmark_cases`（anchor `CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT`）
机器证据：`docs/audit-m16136/evidence_verification.json`（由 `collect_evidence.py` 生成，本报告所有数字均取自该文件）
裁定落表：`tensor_maturity_verdicts` **id=438**，level=**2**，gate_state=`pending`

> 独立性声明：本次是「交差核验」而非重做。审计者未参与 m16136；执行路径 = 独立重跑被审
> 工作树的测试与整条 kappa 梯子并逐字段比对已提交 JSON；未使用被审报告的自述数字作为判据。
> 局限：只有单机单次复算（该算例确定性、无随机并发面；`np.random` 种子在 case 内固定），
> 未审计 `sweep_kn.json`（N=200/400 探索性跑，报告已声明不作为收敛研究使用）。

---

## 1. 对象与契约

物理背景：密集颗粒剪切流的**弹限区**（颗粒持续接触、弹性成为流变参数）。Campbell (2002,
JFM 465:261-291, DOI 10.1017/S002211200200109X) 用无量纲刚度群
`kappa = k_n/(rho d^3 gdot^2)` 组织相图，被测量为 `tau* = tau/(rho d^2 gdot^2)`；
标度指数 `beta = d ln tau*/d ln kappa`，等效剪切率指数 `n = 2(1-beta)`。
惯性（Bagnold）极限 `n=2`（`tau*`=const），弹性准静态极限 `n→0`（`tau* ~ kappa`）。

**案内先验判据（动手前声明，见 `docs/CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT.md:60-69`）**

| id | 量 | 参考 | 容差 |
|---|---|---|---|
| R1 | 惯性极限 Bagnold 指数 `n` | 2.0 | 15 % rel. |
| R2 | 弹性-惯性→惯性边界 `kappa_c` | 1e4 | 数量级（因子 3） |
| R3 | 弹性准静态指数 `n`（`tau*~kappa`） | 0.0 | 绝对 0.3 |
| R4 | Bagnold 系数 `tau*` 平台值 | **WITHHELD** | — |

## 2. 怎么做的（审计执行路径，可复现）

1. 读被审工作树的模块/驱动/测试/文档，逐行确认「真代码」而非桩：
   `src/tensordem/elastic_limit_shear.py`（620 行；`:331` 积分循环、`:364` 颗粒对接触、
   `:424` 壁面接触、`:491-496` 壁面动量通量测应力、`:522` 三门判据、`:574` 标度拟合器）。
2. **重跑验收测试**：`python -m pytest -q tests/test_elastic_limit_shear.py` → **6 passed（5.6 s）**。
3. **重跑整条梯子**（文档给出的 repro_cmd，5 档，N=100，strain 1.5）到
   `runs/audit_m16136/repro/`，与已提交 `artifacts/.../point_kn*.json` **逐字段比对**
   （tau / pressure / tau_star / steps / coordination / I / gate / gate_pass / kappa）→
   **5/5 全部 bit-identical**（`repro.all_five_rungs_bit_identical = true`）。
4. **重跑 `--aggregate`**：拟合拒绝载荷与已提交 `scaling.json` **完全一致**
   （`n_points=5, n_gated=0`，`error="fewer than 3 points passed the physics gates"`）。
5. **查基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
   WHERE anchor_slug='CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT'` → **0 行（审计前无任何裁判行）**。
   本次由 `land_verdict.py` 按案内先验判据补登记 R1-R4 四行，`status=pending`（不宣称通过）。
6. **查报告**：`tensor_reports` id=15268，六段标志 `truetruetruetruetruetrue`，正文 3998 字符，
   figures 数组 1 项（仅 name+caption；经同侪样本对照确认为平台存储行为，见 §5 F4 修正）。

## 3. 三件套核验（逐项）

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts | ✅ **真存在且已入库** 9 个文件，全部在 `00e704d9` 的 git 树内 | `evidence_verification.json:run_artifacts`（每文件 sha256 + `git_tracked_in_audited_commit=true`） |
| benchmark ref vs got | ❌ **无任何可用 ref–got 对比**：0/3 可测判据成立，1 项（R4）合法 WITHHELD | §4 表；`scaling.json` 显式拒绝拟合 |
| repro_cmd | ✅ **可执行且确定性复现**（5 档逐字段 bit-identical；aggregate 拒绝载荷一致） | `runs/audit_m16136/repro/point_kn*.json`、`repro/scaling.json` |

工件清点（sha256 前 16 位，`evidence_verification.json:run_artifacts.files`）：
`point_kn1e2/1e3/1e4/1e5/1e6.json`、`scaling.json`、`sweep_kn.json`、
`reference_evidence.json`、`fig_tau_star_kappa.png`。

## 4. 基准裁判逐项（唯一裁判 = `tensor_benchmark_cases` + 案内先验判据）

| id | reference | tolerance | got（本轮实测） | 判定 |
|---|---|---|---|---|
| R1 | `n = 2.0` | 15 % rel. | **不可辨识**：`n_gated=0`，`tau*` 非单调、`tau` 变号 | NOT_MEASURED（不满足判据） |
| R2 | `kappa_c ≈ 1e4` | 因子 3 | **不可辨识**（需 ≥3 个过门点） | NOT_MEASURED |
| R3 | `n = 0.0` | 绝对 0.3 | **不可辨识** | NOT_MEASURED |
| R4 | Campbell 表（pp.261-291） | — | **WITHHELD**：原文 CLOSED（Unpaywall `best_oa_location=null`），不手抄 | WITHHELD（合法） |

五档门限诊断（`evidence_verification.json:benchmark.gate_diagnostics_all_five_rungs`）：

| tag | kappa | tau | tau* | Z | max ovl/d | 能量平衡误差 | 稳态漂移 | 门 |
|---|---|---|---|---|---|---|---|---|
| kn1e2 | 2.31e2 | 1.311 | 3.643 | 1.12 | 0.951 | 1.63e2 | nan | FAIL |
| kn1e3 | 2.31e3 | -4.055 | 11.26 | 0.469 | 0.950 | 9.01e2 | 0.245 | FAIL |
| kn1e4 | 2.31e4 | 0.4488 | 1.247 | 0.092 | 0.675 | 3.77e4 | 0.204 | FAIL |
| kn1e5 | 2.31e5 | -6.773 | 18.81 | 0.042 | 0.459 | 2.63e4 | 0.378 | FAIL |
| kn1e6 | 2.31e6 | -132.4 | 366.7 | 0.026 | 0.389 | 2.11e4 | 0.065 | FAIL |

自门限拒绝的理由是物理性的：能量平衡差 **2–5 个数量级**（壁做功 ≠ 接触耗散 ⇒ `tau` 的符号/归一化
不可信）、深穿透 `max_overlap/d` 0.39–0.95（线性弹簧失效区）、配位数从起点 1.12 塌到 0.026
（元胞被抽空到壁面，与「弹限密实剪切层」相反）。因此在**当前实现下不存在可与文献对比的数值**。
「`n_gated=0` 时显式拒绝拟合」这一条设计本身是正确的（未把噪声当结果做拟合）。

## 5. 判级（L0–L5）与发现

**判级 = L2（独立功能实现）**，逐级理由（`evidence_verification.json:level_decision`）：

- **L1 ✅**：无量纲群代数、起点无重叠、确定性与拟合器短梯子拒绝均有单元测试钉住
  （`tests/test_elastic_limit_shear.py:23/41/53/77`）。
- **L2 ✅ 达成**：真代码、输入→计算→输出（DEM 积分 + 接触律 + 壁面动量通量应力 + 三门 + 标度拟合器），
  620 行模块 + 130 行驱动；确定性、可复现（5/5 bit-identical）。
- **L3 ❌ 未达成**：无集成调用方——`grep -rn elastic_limit_shear src/ tests/ examples/ benchmarks/`
  在模块/测试/算例驱动之外**零命中**，`src/tensordem/__init__.py` 亦未导出；且未复用
  `tensordem.shear_cell` 的接触律/伺服压板（案内 §6.4 已自述为「需在第2轮复用」）。独立算例模块。
- **L4 ❌ 未达成**：完整算例+误差分析+可复现三项中前两项的**判据侧**不成立——0/3 可测判据有 got 值，
  1 项合法 WITHHELD，无任何 ref–got 误差可算，拟合被显式拒绝。
- **L5 ❌ 未达成**：无科研结论（第1轮 honest FAIL）。
- 附百页报告不改变级别：报告完整、诚实，但**级别=功能实质**。

**发现（按严重度）**

- **F1（blocking，虚标风险已消除）**：`tensor_benchmark_cases` 中该 anchor **无任何裁判行**——
  即平台侧根本没有机器可裁的判据，任何「基准已验证」的说法都不可机器复核。本次补登记 4 行
  （`status=pending`，无 pass 宣称）。复现：`runs/audit_m16136/register_case.sql`（幂等，可重放）。
- **F2（major，诚实申报，非缺陷）**：核心物理未成立（能量门差 2–5 个数量级、深穿透、
  `Z` 塌缩）。任务单状态为 `succeeded`，这与产出的真实内容一致（交付=模块+诚实 FAIL+第2轮路线图），
  **但不可读作「算例通过」**。第2轮修复路线（载荷伺服压板 / SLLOD + 奇异速度应力估计器 + `Z_init≥3` 起点门 +
  滑移门）已写在 `docs/CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT.md:§7`，具体可执行。
- **F3（minor，测试语义）**：6/6 绿包含一条**反向钉**（`:66`
  `test_current_integrator_is_rejected_by_the_physics_gates` 断言缺陷**必须存在**）。
  这是合格的诚实工程做法，但意味着「测试绿 ≠ 物理正确」；第2轮修好后必须翻转该测试。
  复现：`pytest -q tests/test_elastic_limit_shear.py`。
- **F4（minor，平台侧，非 m16136 缺陷）**：图在报告中的 `png_base64` **未被持久化**——`SELECT jsonb_object_keys(figures->0) FROM tensor_reports WHERE id=15268` 只返回 `name,caption`；但 15300–15305 等**其他**报告同样只有 `name,caption`，故此现象归因于平台报告存储丢弃图片字段，**不构成对被审单的指控**（初判为作者偏差，经同侪样本对照后修正）。图本体已入库且可由 `--aggregate` 重生成；本单报告同样受此平台限制，故审计图另存 `runs/audit_m16136/`。
- **F5（minor，数值卫生）**：`kn1e2` 的 `steady_drift = NaN`（`_relative_drift` 在 |tau| 样本 <8 时返回
  NaN，最短一档 251 步、平均窗样本不足）。NaN 比较为 False 故门限仍**保守拒绝**，但诊断信息无意义、
  且 NaN 会污染 JSON（本审计证据须做非有限值清洗才能入库）。建议第2轮把样本不足显式记为
  `"insufficient_samples"` 而非 NaN（`src/tensordem/elastic_limit_shear.py:559`）。

**未跑/未核（如实标注）**

- `sweep_kn.json`（N=200/400 探索性跑）未复算：报告已声明它**不是**收敛研究，且不进入判据；
- `--quick`/集群长跑未做：梯子单机 ≤90 s，无需上集群（与被审单判断一致）；
- 参考文献原文可达性未重查（沙箱无机构访问路径；`reference_evidence.json` 内引文+sha256 已入库，
  审计只核「有没有把不可读的东西当读过了」——结论：R4 确实声明 WITHHELD，未手抄）。

## 6. 误差判定与处置

| 量 | 基准 | 实测 | 误差 | 是否 <3% |
|---|---|---|---|---|
| R1 `n`（惯性极限） | 2.0 | — | 无定义（不可辨识） | **否/不可判** |
| R2 `kappa_c` | ~1e4 | — | 无定义 | **否/不可判** |
| R3 `n`（准静态） | 0.0 | — | 无定义 | **否/不可判** |
| R4 `tau*` 系数 | 文献表 | WITHHELD | 无 | **否（WITHHELD）** |

**显式回答**：本算例**没有任何量进入 15 % 判据，更没有 <3%**；这不是粉饰，是被审报告 §6 原文的立场。
可复现性与工件完整性则**通过**（5/5 bit-identical、9/9 工件已入库、repro_cmd 可执行）。

**处置（disposition）= approve with notes（级别确认 L2，能力未验证 → 返修第 2 轮）**：
交付物本身合规（诚实、可复现、工件齐），无需返工文档；但**能力等级停在 L2**，L3/L4 缺口属于第 2 轮工作。
`maturity_gate` 若判 `reopened`，返修清单按 F2 的路线执行，并通过 F2/F5 的验收（`n_gated≥3`、
能量门 ≤0.30、`Z≥3` 起点门、滑移门、翻转 `:66` 反向钉）。

**落表**：
```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorDEM', 'm16136 CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT', 16194, 2,
        '<evidence JSON: impl_files/entry/run_artifacts/tests/integration_callers/
          benchmark/repro_cmd/external_anchor>', 'board-audit');
```
→ 已执行，**id=438**，`gate_state=pending`（由 maturity_gate 自动裁定 confirmed/reopened）。

## 7. 图

`runs/audit_m16136/fig_gate_diagnostics.png`（由 `make_figure.py` 从 evidence JSON 程序化生成）：
左=五档 `tau*(kappa)` 全被门限拒绝（Bagnold 平台作参考虚线）；右=三门诊断 vs 阈值的双对数对比
（能量平衡差 2–5 个数量级）。图已 base64 内嵌到本单报告。

## 8. 复现命令

```bash
# 审计证据（重跑测试 + 整条梯子 + 逐字段比对 + DB 裁判/报告检查）
python3 docs/audit-m16136/collect_evidence.py
# 图
python3 docs/audit-m16136/make_figure.py
# 落表（幂等：ON CONFLICT 更新 + 裁判行 NOT EXISTS 去重）
python3 docs/audit-m16136/land_verdict.py
```