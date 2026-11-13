# 交差审计 · m17643 `XVAL-DEM-ROUND2-METHOD-SELECTION`（TensorDEM 滚动阻力律 round-2 选型）

- 审计任务：m17671（board-audit，非重做）
- 被审对象：m17643，commit `e1cdcab1`（分支 `agent/m17643`），状态 succeeded，报告 `tensor_reports#16606`
- 裁决落表：`tensor_maturity_verdicts` id=614，`level=4`，`verdict_by=board-audit`
- 机器证据：`artifacts/xval_m17643_board_audit/evidence.json`（本审计脚本产出，零手抄）
- 复算脚本：`scripts/board_audit_m17643.py`（`git cat-file` 直读 commit，不依赖被审 worktree 存活）

## 1. 审计口径与独立性限制

判据只有一条：`tensor_benchmark_cases` 逐项 ref vs got 是否在容差内 + 报告六段是否齐全。
本单的**第一个硬事实**：

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-DEM-ROUND2-METHOD-SELECTION';   -- → 0 rows
```

该锚点**没有任何注册基准行**，因此"逐项核 got 是否在容差内"这一步无机裁对象。为不把
「没登记」错判成「没做」，审计改用可核查的替代裁判（并在 §6 记为待补项）：

1. 解析裁判：`t_stop = v0/((5/7) mu_r g)`、等球对 `t_half = 0.25 w0 I/(mu_r reff Fn)`
   —— 由审计**独立反解**（见 §3 的 implied w0），不是接受被审方给的 theory 列；
2. 外部裁判：LIGGGHTS-PUBLIC `@3d5c00f2…` 源码快照，审计**重新算了 SHA-256**；
3. 容差：`results.json.tolerance = 0.15`，与 prereg 预声明一致，非事后回填。

独立性限制（如实记）：本审计**没有执行 LIGGGHTS 二进制**（环境无该软件栈），0.00% 的一致
是"本仓核 vs OSS 律的转写"之间的一致；转写正确性由审计人工核对 4 处源码（`kr=kt*reff^2`、
`cap=|Fn|*reff*rmu`、扭转分量剥离、CDT `rmu*Fn*reff*w/|w|`）确认，属人工复核而非机器验证。
本审计也未重算 DEM 时间积分的物理正确性，只核"两码是否同一律 + 是否合解析解"。

## 2. 三件套核验（run_artifacts / ref vs got / repro_cmd）

| 件 | 声称 | 审计核验 | 结论 |
|---|---|---|---|
| run_artifacts | `artifacts/xval_dem_round2_method_selection/{results.json,table_main.md,traces.json}` + 5 图 | commit 内全部存在（10158 B / 2351 B / 14344 B；5 张 png 45–57 kB） | ✅ 真存在 |
| ref vs got | E1–E7 七臂 | 审计从 `results.json` 逐项重算（§3），与被审 `table_main.md` 逐位吻合 | ✅ 数字对得上 |
| repro_cmd | `PYTHONPATH=src python3 benchmarks/xval_dem_round2_method_selection.py` | 在 `git archive e1cdcab1` 的独立检出内实跑：**46 s**，重生成的 `results.json`/`table_main.md`/`traces.json` 与 commit 内文件**逐字节相同**（`diff` → IDENTICAL） | ✅ 可复现且确定性 |
| 回归 | 「默认经 13 passed」 | 独立复跑 `tests/test_axisym_column_lube2004.py` → **13 passed (10.3 s)**；`tests/test_xval_oss_rolling.py` → **10 passed (11.9 s)** | ✅ |

`table_main.md` 的 `mu_r` 扫描（E2）里的"+0.79%""+17.63%"看似手写风险点，实为程序化注入：
`results.json.spin_down.rows[*].err_vs_theory` 与审计重算值一致到 1e-12。

## 3. 审计独立复算（不是读被审方的结论）

**E2 单球滚停（5 点，`v0=0.30 m/s`，`dt=2e-6 s`）**

| mu_r | kernel [ms] | 解析 [ms] | kernel vs 解析 | kernel vs OSS EPSD2 | 15% 内？ |
|---|---|---|---|---|---|
| 0.1 | 428.10 | 428.13 | −0.01% | +0.00% | ✅ |
| 0.2 | 212.80 | 214.07 | −0.57% | +0.00% | ✅ |
| 0.3 | 141.20 | 142.71 | −1.06% | +0.00% | ✅ |
| 0.5 | 86.30 | 85.63 | +0.79% | +0.00% | ✅ |
| 0.8 | 62.95 | 53.52 | **+17.63%** | +0.00% | ❌（见 F4） |

**E3 等球对自旋松弛 + 换算因子（审计反解，关键）**
以 `t_half = 0.25 w0 I/(mu_r reff Fn)` 反解每臂隐含的 `w0`：四臂全为 **20.000 rad/s**
（离散度 0.00%）⇒ 被审方的 theory 列是**同一个闭式**在不同 `(mu_r, reff)` 下的取值，不是逐臂拟合。
`t_half`：kernel(reff=R, mu=0.5)=33.95 ms；OSS(reff=R/2)=67.90 ms；kernel(mu/2)=67.90 ms。
比例恰为 **2.000000**，故换算律 `mu_r,OSS = 2·mu_r,TD`（等球）与数据自洽。
> 审计曾按"杠杆减半应使 mu_r 减半才等效"怀疑文档写反了换算方向；用数据核过后**否证**该怀疑：
> `t_half ∝ 1/(mu_r·reff)`，reff 减半 ⇒ OSS 同 mu_r 下慢一倍 ⇒ 要让本仓（reff=R）复现同一物理
> 需 mu_r/2，等价于 `mu_r,OSS = 2 mu_r,TD`。文档表述正确，不记缺陷。

**其余臂**：E1 墙接触 max|Δω| = **7.283e-14 rad/s**（<1e-12 ✅，两侧 reff 同为 R，属恒等检查）；
E4 静持：5/15/20/25° 全部停留在 1e-4…1e-5 m/s，30°（> arctan 0.5=26.57°）滚到 1.4e-1 m/s ✅；
E5 钻削自旋：本仓 w_end = −0.410 rad/s（失去 102.05%）vs OSS 默认 20.000 rad/s，且**生产内核
`wall_contact_forces` 与精简核逐位一致**（差 3e-13）⇒ 该缺陷是生产路径的真实缺陷，不是玩具核的问题；
E6 `dt/dt_R = 0.598`；E7 dt 扫描 `t_stop` 散布：kernel 0.116%、epsd2 0.116%、cdt 0.093% ✅。

## 4. 判级（L0–L5）

| 级 | 判据 | 本单实证 | 判 |
|---|---|---|---|
| L1 | 数学/单元验证 | 解析闭式 + 恒等检查（E1 7.3e-14） | ✅ |
| L2 | 独立功能实现 | `src/tensordem/xval/oss_rolling.py:97,120,173,178,183,204`、`scenes.py:35,40,112`（输入→计算→输出），`benchmarks/…:578 main()` | ✅ |
| L3 | 系统集成 | `granular3d_core.py:50,54,93,98` 新约定字段；`granular_column/kernel.py:224,283,390,454` 生产 `pair/wall_contact_forces` 读取；`axisym.py:145,208` 透传；默认值=历史行为（`test_defaults_are_legacy`） | ✅（默认保持型集成） |
| L4 | 基准验证（完整算例+误差分析+可复现） | 七臂闭环算例 + 逐项误差 + 预声明 15% 容差 + 审计实跑 46 s 逐字节复现 + 外部锚 SHA-256 复核 + 三条 honest ❌ 如实披露 | ✅ |
| L5 | 科研工程成果 | 未达：系统级算例（柱坍塌 runout 用换算后 mu_r 复现 2.106）本单明确推迟 | ❌ |

**裁决 = L4。** 与榜上同类先例一致（m17633 亦 0 注册行、以解析裁判 + 预声明门获 L4）；
本单额外有哈希外部锚与审计复现，证据强于该先例。

## 5. 发现（按严重度）

| ID | 严重度 | 发现 | 证据 / 复现 |
|---|---|---|---|
| F1 | 中（流程，阻断机裁） | 锚点 0 条 `tensor_benchmark_cases` 行 ⇒ 下一轮复核无 ref/got 机裁对象；平台内其他 XVAL 锚点已有注册行（`referee_rows_other_xval_anchors=2`），说明"注册"是既有可行做法，非苛求 | `evidence.json:referee_row_count=0`；SQL 见 §1 |
| F2 | 中（交付未落主线） | `git merge-base --is-ancestor e1cdcab1 origin/main` → **false**；该 commit 只在 `agent/m17643` 分支 ⇒ 能力尚未进入产品主线，worktree 回收后仅剩分支 | `evidence.json:merged_into_main=false, branch=['agent/m17643']` |
| F3 | 低（覆盖边界） | 新开关 `rolling_lever="reduced_radius"` / `rolling_torsion=False` 无端到端使用者（`docs` §5 自认推迟柱坍塌 runout 臂）⇒ 集成只到接触核/API 层 | `docs/XVAL_DEM_ROUND2_METHOD_SELECTION.md` §5；`grep rolling_lever` 无 case 侧调用 |
| F4 | 低（已披露） | E2 在 `mu_r=0.8` 对解析式 +17.63%（>15%）：机制为 `mu_r>mu_s` 时切向先饱和，三律同值 0.00% 互证，属解析式有效域边界而非两码分歧；报告 ⑥ 已打 ❌ | §3 表；`results.json.spin_down.rows[4]` |
| F5 | 低（已披露） | E6 `dt/dt_R = 0.598`，落在 OSS 惯例带 `[0.1,0.2]` **之外**（约 3–6×）；`results.json.timestep` 已注明该比例是用户输入而非源常数，且 E7 实测 0.116% 收敛 | `evidence.json:E6_timestep` |

**未发现虚标**：报告 ⑥ 的三行 ❌（E2@0.8 / E3 映射前 50% / E5 102%）都是对自己不利的真实数字，
且与审计重算一致；headline「墙接触上逐位同一」有"墙接触"限定词，与 E3 的 2× 杠杆差并存自洽。

## 6. 未做的检查（不得记作通过）

- 未执行 LIGGGHTS 二进制；OSS 一致性只在"转写"层面成立（人工核 4 处公式，非机器）。
- 未审 DEM 时间积分器本身的守恒/能量账（本单不涉及，但 E5 的"钻削吃掉"缺陷说明积分层仍值得单独立项）。
- 未跑全量 `pytest tests/`（只跑了本单声明相关的 2 个测试文件，均通过）。
- 未复核 `traces.json` 的逐步轨迹与 `results.json` 聚合值的一致性（只核了重跑逐字节相同 ⇒ 聚合与轨迹同源）。

## 7. 处置（disposition）

**L4 确认成立，但返修两项（不降级，属流程/交付面）：**

1. **补注册基准行**（F1）：把 E1/E2/E3/E6 的 metric + 容差 + 当前 got 值写入
   `tensor_benchmark_cases`（anchor=`XVAL-DEM-ROUND2-METHOD-SELECTION`），使后续复核可机裁；
2. **合分支**（F2）：`agent/m17643` → main，否则能力不在产品内；
3. 建议（非阻断）：在 `benchmarks/case_column_collapse_lube2004.py` 增一臂用
   `rolling_lever="reduced_radius"` 或换算后 `mu_r` 跑柱坍塌 runout，闭合 F3；
4. 建议：prereg 里把 E2@`mu_r=0.8` 的裁判从解析式改为 OSS 参照（解析式在该点超出有效域），闭合 F4。

---

**审计产出**：`artifacts/xval_m17643_board_audit/evidence.json`（机器证据）、
`fig1_audit_recompute.png`（审计复算图：E2 三律对解析 + 15% 带；E3 四臂杠杆/换算）、
`scripts/board_audit_m17643.py`（复算，可重跑）、`scripts/record_m17643_verdict.py`（落表）。