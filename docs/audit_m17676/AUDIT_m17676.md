# AUDIT m17676 — XVAL-DEM-SANDPILE-REPOSE-CALIBRATION-ANCHOR（交差核验，非重做）

- 被审单：`m17676`（TensorDEM，status=succeeded，decided_by=eco_admin，engine 自报 `fail_stage=budget_exhausted+forced_finish`，`accepted=true` 走的是 `TP_AUTO_APPROVE_DEV_ACCEPTANCE` 自动验收）
- 审计单：`m17745`（board-audit）／审计工作树 `/root/tp-agent/work/m17745`，被审工作树 `/root/tp-agent/work/m17676`
- 审计时刻：2026-11-13（UTC）；被审 deliverable **全部未提交**（`head_commit=null`，`git status` 为 `M src/…/sandpile.py` + 4 个 `??`）
- 证据快照：本目录（与原件逐字节相同，见 `SHA256SUMS`）。所有结论只引用快照内文件与快照中的 sha256，不引用报告叙述。

## 0 独立性限制（如实声明）

1. 我不是被审单的执行者；但两者共用同一 checkout 与同一 Python 环境，**无法排除环境层面的共模误差**。
2. Zhou et al. (1999) *Physica A* 269:536–553 原文不在本机，`oss_solutions`/知识引擎本单内未取得读权限。参考带只能采信**仓内既有文献记录**（`docs/GAP_DEM_CLUMP_ROUND2.md`），因此本次核验的是「本仓实现 vs 本仓登记的锚点值」，**不是**「本仓实现 vs 论文原表」。这一条是 WITHHELD，任何级别的判定都不含「与文献原文一致」的结论。
3. 未重跑 8 个算例的全量时程（总 wall time ≈ 2.5 h CPU）。审计重跑的是**聚合段**（`--aggregate`，秒级、确定性）与**定向量测探针**（秒级）；逐算例时程的真实性由工件内部自洽性 + 交叉路线 + 物性控制来旁证（见 §3.4）。
4. 未做非球形（clump/角砾）分支核验——本单未交付该分支（ρ 报告 §局限 已自认）。

## 1 三件套取证（逐项，机器可核）

| 项 | 要求 | 实际 | 判定 |
|---|---|---|---|
| run_artifacts | 真实运行工件 | `artifacts/XVAL-SANDPILE-ZHOU1999/cases/*.json` 8 个（每例含 `surface_profile`/`packing`/`fabric`/`energy`/`feed`/`env`），`verdict.json`，`figures/*.png` 3 张，7 个运行日志 | ✅ 存在且非空（快照 `artifacts/`） |
| benchmark ref vs got | 参考值 vs 实测 + 误差 | ref 带 `[22.0, 25.0]°`（中心 23.5°）运行时解析自 `docs/GAP_DEM_CLUMP_ROUND2.md`（sha256 `47675131…072b`，本审计独立重算一致）；got `θ=19.0527°`（`gentle_anchor`）；`rel_err=0.189247` | ✅ 数值在案；❌ **超 15% 容差** |
| repro_cmd | 可执行复现 | `python benchmarks/xval_sandpile_zhou1999/xval_zhou1999.py --aggregate` 实跑 exit=0，重生成 `verdict.json` sha256 与原件**逐字节相同**（`d541d7af…c4ac`） | ✅ |

零手抄复核：本审计独立执行了参考值解析链——`docs/GAP_DEM_CLUMP_ROUND2.md` 正则命中 `[("22","25"),("35","45"),…]`；`src/tensordem/protocols/sandpile.py:80-82` 的 Zhou 书目行在 `re.S` 下命中。参考数字确实**不在**基准脚本里硬编码（`xval_zhou1999.py` 无 `23.5`/`22.0`/`25.0` 字面量），与 deliverable 声明一致。

## 2 基准裁判：登记的算例**不存在**

任务规定的唯一裁判：

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-DEM-SANDPILE-REPOSE-CALIBRATION-ANCHOR';
-- (0 rows)
```

补充检索（`repo ILIKE '%DEM%'` 全表）后本锚点最近的两条是：`id=1 GAP-DEM-CLUMP`（休止角 vs 文献带，**±20%**，pending）、`id=2 GAP-DEM-FRICTION`（μ 标定，<10%，pending）。**没有任何一行把本锚点的 15% 判据登记进机器裁判表。**

后果有二，必须在判级里体现：

1. 没有任何机制能在本单结账时自动裁「过/不过」——本单 `succeeded` 是**自动验收**给的（`decision_note=TP_AUTO_APPROVE_DEV_ACCEPTANCE`），不是基准裁判给的。
2. 若按 ±20%（台账里现存的休止角容差）判，18.92% 会**通过**；按本单预登记的 15%（owner 标准）判，**不通过**。判据口径差异直接改变结论——这正是必须把算例登记进裁判表的原因。本审计按任务给定的 15% 判。

## 3 独立复现与对抗性探针

### 3.1 聚合段确定性
在 `/tmp` 的独立副本中 `--aggregate`：`verdict.json` sha256 **与交付件完全相同**，`checks`/`decision`/`anchor` 三段逐字段相等（`decision.xval_within_15pct=false`）。可复现性成立。

### 3.2 逐项判据（机器值，非报告文字）

| id | 判据 | 实测 | 判定 |
|---|---|---|---|
| D1 | `|θ−23.5|/23.5 ≤ 15%` | 18.925% | ❌ |
| D2 | `θ ∈ [22,25]°` | 19.05° | ❌ |
| D3 | `θ(baseline)−θ(μ_r=0) ≥ 3°` 且 `θ(μ_r=0) ≤ 20°` | 19.05°；4.99e-07° | ✅ |
| D4 | `|θ(baseline)−θ(μ_s=0)| ≤ 3°` | 19.05° | ❌（预登记假设被证伪） |
| D5 | 系统尺寸/静置时间散布 `≤ 3°` | 3.1755° | ❌ |
| D6 | `r²≥0.95`、`stderr≤1°`、`Z∈[3,7]` | 0.9938 / 0.4659° / 3.856 | ✅ |

`all_checks_pass=false`。**honest-FAIL 的记账是对的**：报告 §⑧ 明写「18.9% > 15% 不通过」，未粉饰；失败机理（撒料边界注入 ≈16R 落高动能）标注为相关性而非因果，措辞克制。

### 3.3 交付的回归测试是**红的**（可复现的测试缺陷）
```
$ python -m pytest tests/test_sandpile_degenerate_measure_m17676.py -x -q
FAILED …::test_flat_deposit_keeps_the_angle_and_reports_packing_gap
E  TypeError: argument of type 'float' is not iterable   (line 62)
```
原因在**测试自己**：它假定 `record["packing"]["packing_fraction"]` 是 dict，而 `probe_packing_fraction()` 返回的是 `{"packing_fraction": <float>, "n_probe": …, …}`。因此该断言在**任何**代码版本下都必然 `TypeError`——这条测试不可能绿。

### 3.4 被改的代码本身是有效的（审计独立探针）
审计自行构造退化沉积（N=400 浅锥、`control=mu_r_zero`）调用 `run.measure()`：
```
packing types: {'packing_fraction': 'float', 'n_probe': 'int', …, 'bulk': 'dict', 'core': 'dict', 'voronoi': 'dict'}
repose_deg = 13.3802679906812  (有限)
packing_fraction = 0.0
```
即被审的 `_safe()` 降级补丁**确实**把「packing 探针抛异常 → 整条记录（含已算出的休止角）丢失」改成了「NaN + 原因、角度保留」。两例控制组重跑（`ctrl_mu_r_zero`/`ctrl_mu_s_zero` 于 08:16 重生成，θ≈5e-07°/3e-17°，`poured=2000`）与补丁落盘时刻（`sandpile.py` mtime 08:12:58）自洽，可作旁证。**但**：`run_all.log`（08:11，补丁前）仍记 `ctrl_* rc=1`，`run_controls.log` 只记 `[launch]` 不记 rc——「修复后 rc=0」是推断，不是日志直证。

## 4 缺陷清单（按严重度）

| # | 严重度 | 事实 | 证据/复现 | 影响 |
|---|---|---|---|---|
| F1 | **阻断 L4** | 锚点算例从未登记进 `tensor_benchmark_cases` | §2 的 SQL 返回 0 行 | 判据无机器裁判；`succeeded` 由自动验收产生 |
| F2 | 高 | 基准超差：18.92% > 15%（D1），D2/D4/D5 同败 | `verdict.json` → `checks[0].metric.rel_err`；本审计 sha256 复现一致 | 校准锚点未闭合；本单本质是 honest-FAIL |
| F3 | 高 | 交付的回归测试结构性错误，恒红 | §3.3 traceback，`test_…:62` | 补丁只能靠人工旁证，L1「测试」证据为空转 |
| F4 | 中 | 交付物全部未提交/未合并（`head_commit=null`；ECO main 无 `benchmarks/xval_sandpile_zhou1999/`、无 `docs/XVAL_SANDPILE_ZHOU1999.md`、`src/…/sandpile.py` 无 `_safe()`） | `/root/tensor-ecosystem/TensorDEM` 对照 grep/diff | 工件只活在一个 `worktree_recyclable=false` 但仍可被回收的工作树里；本审计已快照入本仓 `docs/audit_m17676/` |
| F5 | 中 | `tensor_reports` 无 `mission_id=17676` 的报告行（0 rows），但 `report_owed=f` | `SELECT … WHERE mission_id=17676` | 平台侧发文缺失（仓内 doc 存在）；本审计报告补的是审计结论，不代原单发文 |
| F6 | 低 | `as_delivered` 被描述为 canonical baseline，实为“散喷”（Z=2.08，θ=9.51°） | `cases/as_delivered.json` + 报告 §④ | 已被报告自己标明不可用于对标，不构成虚标 |
| F7 | 低 | 路线 B（提筒）同参数 θ=8.08° vs 路线 A 19.05°（差 13.9°），文档节点 μ_r=0.2 的 12.54° 重跑 8.08°（差 35.5%），r²=0.63 | `cross_route.json`；报告 §⑦ | 交付件自己判「B 不具锚定资格」，判定只落 A 线——处理正确，但说明本仓休止角**驱动相关**，锚定面很窄 |
| F8 | 低 | `verdict.json.cases` 无 `valid` 字段，而脚本 docstring 声明「无效算例记 invalid 不静默丢弃」 | 8 例均 `poured==n_particles` 实为有效，字段缺失未造成错误结论 | 模式漂移，建议补字段 |

## 5 级别判定：**L3**

| 级 | 判据 | 依据 | 结论 |
|---|---|---|---|
| L0 | 文档成果 | — | 超过 |
| L1 | 数学/单元验证 | 交付的测试**恒红**（F3），无有效单测；补丁由审计探针（§3.4）与两组控制旁证 | 实质满足但证据形式不合格 |
| L2 | 独立功能实现 | `xval_zhou1999.py`（真代码：配置→时程→自由面→环带拟合→判据→工件）、`cross_route.py`、`sandpile.py` 的 `_safe()` 补丁；0 个 placeholder；8 例真跑出数 | ✅ |
| L3 | 系统集成 | 补丁落在**主协议** `src/tensordem/protocols/sandpile.py:1250 measure()`，生产入口 `scripts/run_dem_macro_anchors.py:85` 与基准脚本 `xval_zhou1999.py:246` 均实例化 `SandpileRun` 走同一路径 | ✅ |
| L4 | 基准验证 | 完整算例 ✅、误差分析 ✅、可复现 ✅ —— 但**基准不过**（F2 18.92%>15%）且**算例未登记**（F1）。L4 的语义是“验证通过并闭环”，不是“跑了并做了误差分析” | ❌ |
| L5 | 科研工程成果 | 需 L4 + 外部锚成立；外部锚为**带级 + WITHHELD（原文不可得）** | ❌ |

「附百页报告不改变级别」在此单的表现：报告 8 节、3 图、零手抄、honest-FAIL 都做得比多数单好，但**功能实质**是「一个诚实的负结果 + 一个有效的内核补丁 + 一条坏测试 + 未合并的工件」，基准闭环未达成 → L3，不是 L4。

与同类先例一致：`m15187` 审 `GAP-DEM-CLUMP`（休止角 ±20% 未过）同样判 **L3**（「内核真且已集成，但基准未通过」）。

## 6 未做的检查
- 8 例全时程重跑（成本 ≈2.5 h CPU；审计取聚合段确定性 + 控制组 + 探针替代）。
- 论文原表核对（原文/知识库不可得，WITHHELD）。
- 非球形（clump）分支、μ_r 扫描（`assert_priors` 防标定，交付件已声明此限制）。
- 跨机器/跨架构重跑（未上集群；loong64 与 x86 的浮点一致性未验）。

## 7 处置（disposition）

**request changes（返修项，不降级重开——L3 已由机制确认）**。返修清单按性价比排序：

1. 把本锚点算例登记进 `tensor_benchmark_cases`（`anchor_slug` 上述值，`metric='theta_deg vs [22,25] band centre'`，`tolerance='<15%'`），让后续轮次有机器裁判（**须由平台侧落表，不属本审计权限**）。
2. 修 `tests/test_sandpile_degenerate_measure_m17676.py:62`：`packing["packing_fraction"]` 是 float，断言改为 `isinstance(…, float)` + `math.isnan` 分支；并把 §3.4 的探针固化成第二个用例。
3. 合并交付物到 `main`（F4）；合并前用修好的测试跑绿。
4. 补 `tensor_reports` 原单报告（F5）。
5. 若要冲 L4：解决撒料边界注入（改 lifted-box 边界或降落到堆面），把 θ 拉进 `23.5°±15%`（≥19.975°），并在 3 个以上 N/dt/settle 点上证明 D5 散布 ≤3°。这是**新实验**，须重新预登记，不得回头改本单参数。

审计判级：**L3，evidence 见 DB `tensor_maturity_verdicts`（verdict_by=board-audit）**。