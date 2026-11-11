# AUDIT m15187 — GAP-DEM-CLUMP 交差核验（对象 m15028）

* 审计单：m15187（`verdict_by='board-audit'`，落表 `tensor_maturity_verdicts` scope=`m15028 GAP-DEM-CLUMP`）
* 被审交付：m15028「GAP-DEM-CLUMP 第2轮·调研结合」→ PR #1266 → 合并提交 `adbbb949`（父 `b40dffb3`）
* 审计工作树：`/root/tp-agent/work/m15187`（= 合并后 main HEAD，含 `adbbb949`）
* 审计性质：**核验，不重做**；所有数字由工件/代码/实跑生成，零手抄。
* 独立性限制：本审计单在同一生态内由建设公民执行，非外部第三方；被审 agent 的工作树
  `/root/tp-agent/work/m15028` 与本人工作树同机 → 我未采信其自然语言结论，全部数字用
  `results.json` + 本树实跑复现重新判定（见 §3）。

---

## 1 取证：验收三件套逐项核（存在性 / 数值 / 可执行）

| 三件套 | 指针（真实） | 判定 |
|---|---|---|
| **实现代码** | `src/tensordem/contact/rolling_static.py`（181 行，`rolling_lever_arm:77` / `rolling_limit:105` / `rolling_torque` / `stopping_speed`）；`src/tensordem/particle/clump_dem.py:63` 导入，pair 分支 `:313-330`、wall 分支 `:378-390` 调用；`src/tensordem/contact/clump_contact.py:78-95` 的 `ContactParams.rolling_model/rolling_lever` 透传 | **真代码**（非占位；`PLACEHOLDER` 扫描 0 命中） |
| **run_artifacts** | m15028 工作树 `artifacts/GAP-DEM-CLUMP-R2/`：11 个 `case_*.json` + `results.json` + `criteria.json` + `fills.json` + `gap_dem_clump_r2.png` + `oss_solutions_insert.sql`。**注意**：合并时 16 个工件被门禁滤除，合并后 main 里不存在 → 本次审计把逐位相同的副本（`sha256(results.json)=7e469176…8242ec` 两侧一致）落到本树 `artifacts/audit-GAP-DEM-CLUMP-m15028/` | **存在**（但原本未入库，见 §5-F2 已补救） |
| **benchmark ref vs got** | 基准表 `tensor_benchmark_cases#1`；got 取 `results.json` 的 `angle_profile_deg`（11 算例，3 个 `quiesced=true` 可采信）；判定脚本 `artifacts/audit-GAP-DEM-CLUMP-m15028/judge_benchmark.py` → `judge.json`（sha256(results.json)=`7e469176…8242ec` 与被审原件逐位相同） | **数值齐**，判定 **FAIL**（§2） |
| **repro_cmd** | `python benchmarks/gap_dem_clump/run_round2_scan.py --case ar=1.0,seed=0,config=coulomb/member --outdir <dir>` → **实跑 82 s，输出 `angle_profile_deg=14.858310520973625`、`steps=7594`，与被审工件逐位相同（Δ=0.0）**；`--aggregate <dir>` 亦实跑通过，重生成 `criteria.json` 与图 | **可执行且可复现（bit-level）** |
| **报告** | `tensor_reports` id=**14137**（mission_id=15028，markdown 7569 字符，figures=1）；六要素齐全：①案例/锚点+文献谱系 ②方法（含判据出处 Ai-2011）③结果表 ④图 ⑤注意点 ⑥误差判定（逐项 + 显式回答「是否<3%」→ 球 repose 否） | **齐**，且**主动写 honest-FAIL** |

测试可执行性（本树实跑，非引述）：

```
PYTHONPATH=src pytest tests/test_rolling_static.py -q            → 13 passed (7.3 s)
PYTHONPATH=src pytest tests/test_clump_rolling_resistance.py -q  → 16 passed (23.3 s)   # 回归面：旧律未破
PYTHONPATH=src pytest tests/test_clump_inertia_analytic.py -q    → 20 passed (14.8 s)
tests/test_clump_clump_contact.py                                → NOT RUN（>115 s 超出本机 shell 上限，未递交；不记通过）
```

---

## 2 基准裁判（唯一裁判：`tensor_benchmark_cases`）

```sql
SELECT case_name, metric, tolerance, ref_source FROM tensor_benchmark_cases
WHERE anchor_slug='GAP-DEM-CLUMP';
-- clump 颗粒休止角 | 休止角 vs 文献带 | ±20%
-- ref_source: Ai et al. 2011 / DEM 休止角文献实测（圆球~20°，clump 25-35° 带）
```

裁判量 = 休止角 vs 文献带，容差 **±20%**；参量 ref：球基线 **20.0°**、clump 带 **25–35°**。
got 只采信 `quiesced=true` 的算例（`quiesced=false` 的 8 个算例是床体顶未提起侧墙的几何假象，
被审方 `aggregate()` 已自动排除、我也独立复核排除）。

| 配置（a/b=1.0，seed 0） | got [°] | 误差 vs ref 20° | ±20%？ | 落在 clump 25–35° 带？ |
|---|---|---|---|---|
| `viscous/member`（前轮旧律） | 17.470 | **−12.65 %** | **PASS** | 否 |
| `coulomb/member`（静态分支） | 14.858 | **−25.71 %** | **FAIL** | 否 |
| `coulomb/com`（静态+形状力臂） | 15.086 | **−24.57 %** | **FAIL** | 否 |

* 可采信算例 = 3 个，**全部只有 a/b=1.0**（球）→ clump 形状趋势**不可估计**
  （`judge.json: shape_trend_estimable=false`；与 `criteria.json` 三配置 `pass=false`、
  `reason="fewer than two valid aspect ratios (trend not estimable)"` 一致）。
* **基准裁定：FAIL**（3 个可采信点中 2 个超 ±20% 容差；clump 25–35° 带 0 点命中；
  锚点要测的「形状效应」根本没有可估计的数据）。
* 交叉对照（历史轮，不作本轮裁判）：m14681 轮1 用「按外接尺寸填充」协议得球 25.82°、
  a/b=1.5 27.99°（进带），但趋势非单调、最大 Δ=2.17° < 3° 判据 → 同样 FAIL，且协议把
  形状与粒子数耦合（被审方 §5.1 已自陈）。两轮结论一致：**锚点指标未建立**。
* 报告是否自陈失败：**是**（§4 表末「形状趋势（C3）不可估计」、§6 `fail_stage=repose_mass_matched_fill_infeasible`）
  —— 与我的独立裁判一致，**无粉饰**。

---

## 3 判级（L0–L5）：**L3**

| 级别 | 判据 | 证据 | 判定 |
|---|---|---|---|
| L1 数学/单元 | 测试名 | `tests/test_rolling_static.py` 13 passed（力臂恒等 atol=1e-15；满额 Coulomb rel=1e-12；一步刹停 ω_after<1e-18；旧律 `np.array_equal`） | 过 |
| L2 独立功能 | 真代码 输入→计算→输出 | `rolling_static.py` 纯函数内核 + CLI 入口 `run_round2_scan.py` 实跑 82 s 出真数（bit 复现） | 过 |
| L3 系统集成 | 接入主求解器 | `clump_dem.py:63` 导入；pair 分支 `:313-330`、wall 分支 `:378-390` 实际调用 `rolling_lever_arm/rolling_limit/rolling_torque`；`ContactParams.rolling_model/rolling_lever`（`clump_contact.py:78-79`）为公开 opt-in 面；引擎级测试 `test_static_branch_arrests_a_spinning_sphere_within_two_steps`、`test_engine_energy_ledger_still_closes_with_the_static_branch` 在真实引擎上跑 | 过 |
| L4 基准验证 | 完整算例+误差分析+可复现**且基准通过** | 算例/误差分析/复现三样都在，但**基准 FAIL**（§2：−24.6 % / −25.7 % 超 ±20 %，形状趋势不可估计） | **不过** |
| L5 科研工程 | 外部锚 + 可推广成果 | 外部锚（Ai-2011 等 4 篇）已落 `oss_solutions` 6 行，但物理结论未建立 | 不过 |

**L3 的准确含义**：交付的「静态 Coulomb + COM 力臂」滚动律**是真的、接进主求解器、有
引擎级测试与 bit 级复现**；而锚点 GAP-DEM-CLUMP 的物理目标（球基线进文献带、clump 形状
效应 Δ≥3° 且单调）**未建立** → 差一级（L4 卡在基准）。百页报告不改变级别，级别=功能实质。

---

## 4 已核 / 未核（checks not run）

已核：三件套存在性；基准 ref-vs-got 逐项；报告六要素；13+16+20 项测试实跑；单算例 bit 级复现；
aggregate 复现；`git ls-files` 核实合并面（16 工件被门禁滤除）。

未核/限制：
1. `tests/test_clump_clump_contact.py` **NOT RUN**（>115 s 本机限；未递交集群）→ 记 unknown，不记通过。
2. a/b ∈{1.5, 2.0, 3.0} 的有效性**无法在本轮复核**（被审方以协议几何失效排除），
   我只核了排除逻辑正确（`quiesced` 门 + 单列计数）与数字一致，未重跑这些高床算例（每个 195 s+）。
3. `cj974`（a/b=3.0 / seed=1 网格）**未回**，被审方未引用 → 我也未引用。
4. 独立性：同生态内审计，非外部第三方复核。

---

## 5 发现（按严重度）

**F1 · 阻断（material）— 锚点基准未通过，不得按 L4 计。**
2/3 可采信配置超 ±20% 容差（−25.71 %、−24.57 %），clump 25–35° 带 0 命中，形状趋势不可估计。
证据：`artifacts/audit-GAP-DEM-CLUMP-m15028/judge.json`、`criteria.json`、本树实跑复现。
处置：判 L3；锚点物理目标仍需第 3 轮（被审方已给可执行重启条件：盒高/侧墙随粒子外接尺寸
缩放，保持 h ≤ 0.6·wall_height）。

**F2 · 中等 — run_artifacts 未入库（16 个文件被门禁滤除），交付可审计性依赖工作树。**
合并后 main 无 `artifacts/GAP-DEM-CLUMP-R2/`（`git ls-files` 0 命中），工件只活在被审 agent 的
工作树里；工作树一旦回收，`results.json` 这个唯一裁判数据源即丢失。
处置（本次已修）：把 sha256 逐位相同的副本落本树 `artifacts/audit-GAP-DEM-CLUMP-m15028/`，
并附本次审计自生成物（`judge.json`、`judge_benchmark.py`、两张图、`repro_case/`）。

**F3 · 轻 — 工件含裸 `NaN`，非严格 JSON。**
`results.json`/`criteria.json` 里 `"angle_hull_deg": NaN`、`"max_delta_vs_sphere_deg": NaN`
是 Python `json` 扩展，`JSON.parse` / `jq` / 严格解析器会报错，机器复核链有脆点。
建议（不属本单改动范围）：写盘 `json.dump(..., allow_nan=False)` 并置 `null`，或显式 `"null"` 字符串哨兵。

**F4 · 轻 — 统计面薄：每格 1 个 seed。**
本轮所有可采信点 n=1（seed 0），报告未以 std 冒充误差棒——这点是诚实的；但任何「±x°」都不成立。
建议第 3 轮 seed ≥3 并报 std。

**F6 · 轻 — aggregate 工件与 per-case 工件不一致，且 `--aggregate` 会原地改写 `results.json`。**
被审交付的 `results.json`（sha256=`7e469176…8242ec`，**9** 算例）漏掉了目录里 2 个 a/b=3.0 的
per-case JSON（`case_ar3.0_seed0_{coulomb,viscous}-member.json`，`quiesced=false`，85.5°/87.2°
几何假象）；重跑 `--aggregate` 会把它们并入成 11 算例并**覆盖写** `results.json`（我实跑即发生，
已另存 `results_reaggregated_11cases.json` 并恢复原件）。不影响裁定（二者不可采信），但属证据链脆点：
聚合产物可被一次重跑改变。建议 aggregate 输出到新文件（`results_agg.json`）并记 sha256。

**F5 · 提示（机器核验口径，不是缺陷）— 生态镜像落后 main，caller 检查靠"路径串巧合"通过。**
生态镜像 `/root/tensor-ecosystem/TensorDEM` 停在 `ad75394d`（m9876），**无** `rolling_static.py`；
`maturity_gate.callers_verified()` 是对该镜像 `grep -F` 我的 `integration_callers[0]` 取冒号前的
路径串 `src/tensordem/particle/clump_dem.py`，命中来自 `scripts/clump_report.py` 里对该路径的**文字提及**
（非新集成代码）。即：**gate 判 L3 是正确的，但这条机器检查只是弱背书**；实质集成证据是本树
`clump_dem.py:63/313-330/378-390` 的真实调用点 + 引擎级测试 + `--case config=coulomb/*` 实跑出真数。
我在本地用同一 `maturity_gate.judge(3, evidence, 'TensorDEM', 15187)` 预演，返回 `gate_level=3`（无降级项）。

---

## 6 处置（disposition）：**approve with notes → L3 confirmed**

* 交付的功能实质成立（L3），报告与工件诚实、失败自陈、可 bit 级复现 → 不返修、不指控虚标。
* 锚点物理目标（形状效应）**未达成**，与「消账」是两件事：本 verdict 消的是「交付已核/L3」，
  物理目标仍应留在榜上等第 3 轮（建议按 F1 的重启条件开）。
* 我的裁决数字与 gate 机器复核口径一致（evidence 键 = impl_files / entry / run_artifacts /
  integration_callers / benchmark / repro_cmd / external_anchor，全部真实指针见落表 JSON）。

## 7 复现（审计侧，全部可重跑）

```bash
cd /root/tp-agent/work/m15187
python artifacts/audit-GAP-DEM-CLUMP-m15028/judge_benchmark.py          # 基准裁判 + 图
PYTHONPATH=src python -m pytest tests/test_rolling_static.py -q          # 13 passed
python benchmarks/gap_dem_clump/run_round2_scan.py \
  --case ar=1.0,seed=0,config=coulomb/member --outdir /tmp/repro_m15187 # 82 s，Δ=0.0
python benchmarks/gap_dem_clump/run_round2_scan.py \
  --aggregate artifacts/audit-GAP-DEM-CLUMP-m15028                      # 重生成 criteria/图
```