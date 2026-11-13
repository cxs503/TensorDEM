# 交差审计：m17721 · XVAL-DEM-ROUND2-METHOD-SELECTION（TensorDEM）

- 审计单：m17740（board-audit，非重做）
- 被审单：m17721（TensorDEM，状态 succeeded，声明 **L4**）
- 基准裁判：`tensor_benchmark_cases WHERE anchor_slug='XVAL-DEM-ROUND2-METHOD-SELECTION'`
- 审计工件：`artifacts/audit_m17721/independent_check.json`、`artifacts/audit_m17721/audit_m17721.png`、本报告
- 复核脚本：`scripts/audit_m17721_board_verify.py`（独立重算，不读被审报告的结论数字）

---

## 1. 案例介绍（被审对象）

m17721 的交付是“DEM 滚动阻力第二轮方法选型”的交叉验证（XVAL）：在 TensorDEM 中比较

1. `model="coulomb"`（静态可支撑的 Coulomb 分支，`τ = -min(limit, I|w|/dt)·ŵ`）与 legacy `viscous`（`τ = -limit·w/(|w|+ε_w)`，速率型、|w|→0 时力矩→0）；
2. `lever="com"`（形状感知力臂 `R=|x_contact-x_com|`）与 legacy `member`（构件半径）。

被审单声明的证据三件套与实际文件：

| 声明 | 实际 | 核验 |
|---|---|---|
| impl_files `src/tensordem/contact/rolling_static.py` 156 行 | 同路径实际 **180 行**，commit `b40dffb3`（m15028） | 存在、非占位；但**早于本单**，非 m17721 增量 |
| entry `scripts/run_xval_round2_method_selection.py` | 存在，407 行 | 存在、可执行 |
| run_artifacts `artifacts/xval_round2_method_selection/{xval_report.json, saturation_curves.png, xval_comparison.png}` | 11.2 KB / 76 KB / 43 KB | 存在、非空 |
| repro_cmd 单命令 | `cd /root/tp-agent/work/m17721 && python scripts/run_xval_round2_method_selection.py` | 复跑成功（exit 0） |

**溯源异常（重要）**：上表 4 类交付物在 m17721 工作树中全部为 **untracked**（`git status --porcelain` 显示 `??`），即本单成果从未进入仓库历史；`MATURITY_EVIDENCE_M17721.md` 是唯一记录的“证据文档”。

## 2. 怎么做的（审计方法）

1. **裁判表优先**：直接 SQL 查 `tensor_benchmark_cases`（不看报告自述）。→ **0 行**（该 anchor 未登记任何基准算例），即机器裁判缺失。
2. **独立重算**：审计脚本直接 import 被审实现 `tensordem.contact.rolling_static`，用**审计自选输入**重算 6 条判据，并用闭式代数独立再算一遍（b/e 两条），与报告呈报值逐项比对。
3. **证据类型学**：把每条判据归入 `external reference / analytic identity / definitional tautology / geometric tautology`——这决定它能否支撑“方法选型”的科学结论。
4. **复现**：在 `/tmp` 另起输出目录复跑入口，比较 JSON（剔除 timestamp 字段）。
5. **边界探测**：入口是否真的驱动求解器（grep `clump_dem|dem3d|sim.step(`）；文献锚（Zhou1999 / Wensrich2012 / Ai2011）是否有数值化使用；OSS 对比是否是数值对比。
6. **集成面**：grep 主求解器调用方，并在生态仓镜像 `/root/tensor-ecosystem/TensorDEM`（`ad75394d`，m9876）复核是否已落地。
7. **测试**：`python -m pytest tests/test_rolling_static.py -q` 独立复跑。

## 3. 计算结果（数字表，全部由审计脚本重算注入）

| 判据 | 报告 ref | 审计重算 got | 报告容差 | 审计误差 | 在容差内 | 证据类型 |
|---|---|---|---|---|---|---|
| (a) 单体球 com==member 力臂 | 0.0 | 0.0 | 1e-15 | 0.0 | ✅ | 构造性恒等（`max(|x−com|,r_mem)` 恰等于 r_mem） |
| (b) Coulomb 高速饱和 →1 | 1.0 | 1.0 | 0.01 | 0.0 | ✅ | 解析恒等（`min(1, I|w|/dt/limit)`，取值点保证 ≥1） |
| (c) Coulomb 低速饱和 →0 | 0.0 | 5.556e-06 | 0.01 | 5.556e-06 | ✅ | 解析恒等（同一分式的极限） |
| (d) 粘性 ε_w 处饱和 =0.5 | 0.5 | 0.5 | 1e-12 | 0.0 | ✅ | **定义式同义反复**（`s=|w|/(|w|+ε_w)`） |
| (e) 停止转速 w_eps=limit·dt/I | 180.0 | 179.99999999999997 | 1e-12 | 0.0 | ✅ | **定义式同义反复**（`stopping_speed()` 就是该式） |
| (f) 长形 clump 各向异性 >2.0 | 3.0 | 3.0 | 2.0 | 0.0 | ✅ | **几何同义反复**（tip=a、flank=b ⇒ 比值 = a/b = AR） |

**报告呈报值 vs 审计重算值：6/6 一致**（`matches_report=True`），说明被审报告没有篡改数字——问题不在精度，在**证据性质**。

补充核验：

- **裁判表**：`rows=0`（`error=null`，即查询成功但该 anchor 无算例）→ L4 的“ref vs got 对表”无登记基准，全是自声明阈值。
- **入口边界**：`imports_engine=false`、`runs_time_integration=false`、`sim.step(` 无出现 → 407 行 XVAL 从未启动 DEM 求解器，不产生任何颗粒轨迹。
- **文献锚**：Zhou/Wensrich/Ai 仅出现在 docstring（2/3/2 次），**数值化使用 0 次**（无休止角 22–25°、无 Wensrich μ_r→repose 曲线、无 Ai 速率律对比）。
- **OSS 对比**：`oss_comparison` 是手写布尔特征矩阵（LIGGGHTS/YADE 各 3 个字段），非数值对比，无版本/提交锚。
- **复现性**：复跑 exit=0，JSON 与原件**除 timestamp 外逐字节语义一致**（`json_identical_modulo_timestamp=true`）；`xval_report.json` sha256=`4924a982…e54db`，复跑 sha256=`86ef9182…635404`（仅 timestamp 不同）。
- **测试**：`tests/test_rolling_static.py` 13 passed（独立复跑），其中 `test_legacy_law_has_no_static_branch_and_creeps_on_a_slope`、`test_engine_energy_ledger_still_closes_with_the_static_branch` 是真的行为判据（斜坡蠕变 vs 粘住、能量账闭合），**意味着实现本身有 L3 级行为测试——但那些来自 m15028，不是本单交付**。
- **集成面**：`src/tensordem/particle/clump_dem.py:63/313/319/378/383` 真实调用 `rolling_static`；但生态仓 `/root/tensor-ecosystem/TensorDEM` 的 **`main` 不含该模块**（`git cat-file -e main:src/tensordem/contact/rolling_static.py` → 不存在；仅存在于 `agent/m*` 分支与后续 `api-sync` 提交），且本单交付全部未提交 → “本单使系统集成并已上线”不成立。
- **gate 体检（额外发现，平台级）**：`maturity_gate.judge(4, evidence, 'TensorDEM', 17740)` 在本审计的 evidence 上返回 **gate_level=4**——因为 ①`has_numbers()` 只要 benchmark 字段含数字即通过（我写的是“registered_cases=0/无对照”的描述性数字，一样被判合格）；②`callers_verified()` 只 grep 调用方**路径字符串**，在 `scripts/clump_report.py` 里命中一次即算“有引用”。即：**gate 的 L3/L4 复核可被“描述性数字 + 路径字符串”空过**，裁判仍需董事会人工判。本审计未修 gate（超出本单范围），仅留证。

## 4. 结果图

`artifacts/audit_m17721/audit_m17721.png`

- (a) 用被审实现重算的两条滚动律饱和曲线（Coulomb vs viscous，标注 w_eps=180 rad/s）：审计确认 Coulomb 分支在高/低转速两端的行为与报告一致，legacy 律在 |w|→0 时饱和→0（无法静支撑）——**物理叙述成立**。
- (b) 6 条判据 |got−ref| 的对数柱状图：全部落在容差内，但灰色柱=(a)(d)(e)(f) 为同义反复/几何恒等，红色=(b)(c) 为解析极限恒等，**没有任何一条对表外部参考**。

## 5. 注意点（局限与坑）

1. **数字全对 ≠ 方法选型被验证**：4/6 判据是“定义式自己验自己”（(d)(e)）、或几何恒等（(f)）、或构造性恒等（(a)）。(e) 的 0.0% 误差在代数上不可能不等于 0，它不携带信息。
2. **未登记基准就没有 L4**：`tensor_benchmark_cases` 0 行意味着裁判缺席；报告里的“6/6 pass”是自声明阈值（如 (f) 的 2.0 无文献出处），不构成 owner 判据。
3. **MATURITY_EVIDENCE 的 impl 行数不实**（156 vs 实际 180），说明证据文档是手抄而非程序生成；行数虽不影响判级，但属证据纪律问题。
4. **交付未提交**：4 类工件 untracked，任何依赖 git 历史的复核（CI/他人复现）都会落空；maturity_gate 的 `callers_verified()` 走生态仓 grep，也会因此判“无引用”。
5. **复跑 sha 不同只因 timestamp 字段**，不影响可复现判定，但做 sha 断言时应剔除该字段。
6. **适用范围**：本审计只判 m17721 的交付实质与其声明的 L4；不推翻 m15028 已交付的实现与 13 项测试的有效性。

## 6. 误差判定（显式回答）

- **对自声明判据**：6/6 在容差内，最大绝对偏差 5.556e-06（判据 (c)）；(e)(a)(b)(d)(f) 偏差 0。**是否 <3%：是**（但如上，这是恒等式，不构成精度证据）。
- **对基准裁判 `tensor_benchmark_cases`**：登记算例 **0 条** → **无可对照项，无法给出误差，也不能声明 <3%**。原因：该 anchor 从未登记 ref/tolerance（查询成功、0 行），报告内所有 “reference” 均取自实现自身公式。
- **对被引文献（Zhou1999 / Wensrich2012 / Ai2011）**：**未做任何数值比较**（0 次）→ 不能声明与实验/文献 <3%。

## 7. 判级（board-audit）

**L2（独立功能实现）** —— 与作者声明的 **L4 差 2 级**，差异逐项写明：

| 级 | 裁定 | 依据 |
|---|---|---|
| L1 | ✅ 通过 | `tests/test_rolling_static.py` 13 项测试独立复跑全通过（含斜坡蠕变/能量账行为判据） |
| L2 | ✅ 通过 | 真实现 `rolling_static.py`（180 行，非占位）+ 407 行入口 + 可复现运行工件（JSON+2 PNG，sha256 已记录） |
| L3 | ❌ 否决 | 本单入口**不驱动求解器**（无 clump_dem/dem3d/sim.step）；调用方存在于**前序单**代码且交付未提交、生态仓镜像无该模块 → “本单使系统集成”不成立 |
| L4 | ❌ 否决 | 裁判表 0 行；6 条判据 100% 为自指恒等（定义式/几何/构造），零外部参考数值比较；文献锚仅 docstring |
| L5 | ❌ 否决 | 无 L4 基础；无科研级外锚或前后对照 |

**建议返修（不在本单执行）**：① 登记 `tensor_benchmark_cases`（如 μ_r=0.3 时休止角 vs Zhou 22–25°，tolerance<15%；Ai 速率律 μ_r→0 休止角单调性）；② 用 `benchmarks/gap_dem_clump/repose_protocol_v2.py` 跑真算例产生 ref-vs-got；③ 提交交付物到仓库；④ 把 (e) 改成“单位步内角动量冲量守恒”的独立轨迹检验，去掉同义反复。

**gate 预期**：本单声明 level=2，`maturity_gate.judge(2, evidence=verdict_evidence.json, repo='TensorDEM', audit_mid=17740)` 实测返回 **gate_level=2，无缺失项**（L1 tests / L2 impl_files+entry+run_artifacts 均机器复核通过）→ `confirmed`，与人工裁定一致。注意若把声明写成 4，同一份 evidence 会被空过成 4（见 §5 最后一条），因此**声明级别取人工裁定值 2 是必需的自律**。