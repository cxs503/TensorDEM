# TensorDEM 上游差集巡更报告（m1630）

- 工具：`scripts/audit_upstream_diff.py`（只读 git plumbing，`--check` 自校验幂等）
- 数据工件：`artifacts/m1630/upstream_diff.json`（本报告全部数字由 `scripts/render_upstream_diff_report.py` 从该 JSON 读出写入，无手工抄录）
- 差集方向：本地 `HEAD`=a68fa8a → 上游 `origin/main`=edded55
- 分叉量：本地落后 **32** 提交、领先 **0** 提交（`git rev-list --left-right --count`）；本地 HEAD 是上游的祖先 → 可快进、无合并冲突

## 1. 案例介绍（审计对象与物理-工程背景）

TensorDEM 是冰区航行战役的颗粒离散元（DEM）内核：3D 冰场（球/多面体颗粒、黏结-破坏、边界压板、浮力/湿耦合）→ 冰载荷与碎冰阻力，被 TensorPlatform 的冰区航行链路当作「冰载荷真值源」。本轮巡更不看单点算法，而是看**交付面**：上游两天内推进的 32 个提交里，到底改了什么主题、哪些改动缺证据、哪些交付物在合并中被静默撤回。

审计对象参数表：

| 项 | 值 |
|---|---|
| 仓库 | TensorDEM（origin=https://github.com/cxs503/TensorDEM.git） |
| 差集窗口 | 2026-10-30 → 2026-11-01 |
| 提交数（含 merge） | 32 |
| 提交数（非 merge） | 15 |
| 逐提交 numstat 累加 | +68181 / -21316 |
| 净树差集 | 68 文件，+12959 / -89 |
| 净差集状态分布 | {"A": 65, "M": 3} |
| 历史触碰文件并集 | 129 |

## 2. 怎么做的（方法/流程/工具链）

1. `git fetch --no-tags origin main` 把上游拉到本地（不改任何远端与配置）；
2. `scripts/audit_upstream_diff.py` 走三条独立口径：
   - **逐提交口径**：`git show --numstat --no-renames`（merge 取第一父差集）→ 行数账本与文件清单；
   - **净树口径**：`git diff --numstat base head` → 真正落到上游树上的变化（不受中间来回增删影响）；
   - **翻覆口径**：把「只删不加」的路径与 head 树比对，再用 blob 哈希判断是**改名搬走**还是**内容丢失**。
3. 主题归类：合并提交独占 `T-MERGE`（避免收口提交吞掉内容主题）；非 merge 提交按关键词优先级（交付物 > 求解器修复 > 产物存储 > 定 dt > FEM-DEM > CI）判定，未命中再按路径构成回落。
4. 风险打标：每条标记都带**可复算证据**（触碰路径、声明被滤除的文件数、被删文件数），严重度只由规则表决定，不做主观评级。
5. CI 触发面：解析 `.github/workflows/*.yml` 的 `paths:` 显式清单，与差集中 `scripts/ tests/ docs/` 文件做 glob 匹配，量化「测试存在但 PR 不跑」的缺口。
6. 文档悬空引用：抽出差集内 `docs/*.md` 引用的仓库相对路径，与 head 树比对。
7. 图件与报告由 `scripts/render_upstream_diff_report.py` 现场生成（matplotlib Agg）。

## 3. 计算结果

### 3.1 变更主题清单

| 主主题 | 提交数 | 涉及文件数 | src | tests | docs | scripts |
|---|---|---|---|---|---|---|
| `T-DELIVERABLE` 任务产物批量成熟 | 14 | 127 | 32 | 25 | 16 | 22 |
| `T-MERGE` 收口/合并 | 17 | 194 | 53 | 42 | 29 | 35 |
| `T-SOLVER` 求解器物理修复 | 1 | 6 | 1 | 1 | 4 | 0 |

主题演化（按提交顺序）：前 6 个提交（m404/m669/m670/m679/m705）是各任务的产物并入；10-31 一轮 PR#71–#75 收口；11-01 集中涌入 9 个任务产物（m835/m943/m991/m992/m995/m1044/m1045/m1129/m1325）并被 PR#76–#84 逐个收口；最后 `6a9eaae` 是一次**求解器物理回滚式修复**（外部节点载荷 / pinned supports / 压板阻尼符号），紧接 PR#85 合入。

### 3.2 净差集按路径类别

| 类别 | 文件数 | 新增行 | 删除行 |
|---|---|---|---|
| benchmarks | 1 | 60 | 0 |
| docs | 11 | 2062 | 0 |
| other | 27 | 1163 | 0 |
| scripts | 9 | 2672 | 0 |
| src | 12 | 4652 | 89 |
| tests | 8 | 2350 | 0 |

### 3.3 关键风险面（数字全部来自工件）

| 指标 | 值 | 含义 |
|---|---|---|
| 净树上存在文件 | 68 | 上游真正新增/修改的集合 |
| 删后不在树上（翻覆） | 59 | 窗口内被加过又被删掉的文件 |
| 其中同 blob 改名 | 0 | 0 表示不是改名，是内容丢失 |
| 其中内容丢失 | 59 | 上游 head 找不到同内容副本 |
| 翻覆率 | 45.7% | 翻覆文件 / 历史触碰文件并集 |
| 声明被门禁滤除的文件（累计） | 147 | 提交自述、仓库内不可核 |
| 触碰 src 的提交 | 26 | 26 条，全部同时带 tests（配对率 100%） |
| 文档悬空引用 | 0 | 0 = 文档承诺的路径都在树上 |
| benchmark 门禁未覆盖文件 | 66 | 仅被无 paths 的全量工作流兜住 |
| benchmark 门禁未覆盖占比 | 100% | 差集中脚本/测试/文档文件口径 |
| 无 paths 过滤器的工作流 | .github/workflows/tests.yml | 对全仓全量触发 |

关键发现：所有 59 个丢失文件均由单一收口提交 `d0e3b54`（Merge pull request #80 from cxs503/agent/m99）相对第一父删除；其中包含 m943 的 79 文件交付批次被门禁滤除后残留在分支上的部分、m991/m992/m1045/m1129 的 docs/scripts/examples。按目录分布：src:20, tests:16, scripts:13, docs:9, examples:1。同 blob 改名计数为 0，即**没有内容以别的路径幸存**。

### 3.3.1 丢失的求解器模块（能力消失的硬证据）

差集中被删除且 head 上不存在的 `src/tensordem/*.py` 共 **20** 个；以 import 限定名 `tensordem.<模块>` 在 head 全树（src/scripts/tests/docs/benchmarks/examples）grep，**有引用的模块数 = 0**，即这些能力在当前上游 head 上没有任何 import、测试或文档引用——不是改名，是能力整体消失。

| 丢失模块（src/tensordem/） | head 上 import 引用数 |
|---|---|
| `api_hull_ice_friction.py` | 0 |
| `brash_channel.py` | 0 |
| `channel_bank_collapse.py` | 0 |
| `channel_resistance.py` | 0 |
| `crushed_ice.py` | 0 |
| `formation.py` | 0 |
| `formation_cli.py` | 0 |
| `formation_dem.py` | 0 |
| `formation_hydro.py` | 0 |
| `formation_resistance.py` | 0 |
| `formation_thermal.py` | 0 |
| `granular.py` | 0 |
| `hull_ice_friction.py` | 0 |
| `ice_intake.py` | 0 |
| `ice_resistance.py` | 0 |
| `keel_lattice.py` | 0 |
| `kerr_foundation.py` | 0 |
| `local_crushing.py` | 0 |
| `md_interface_friction.py` | 0 |
| `ridge_grounding.py` | 0 |

同一批次还丢失 16 个测试、13 个基准/渲染脚本、9 篇文档、1 个示例。测试先行消失意味着：这些能力即使被重新交付，也没有现成判据可回归，需按 `tests/` 里残留的命名约定重建。

### 3.4 逐提交账本

| 提交 | 日期 | 主主题 | 行数 | 文件 | 风险 | 主题 |
|---|---|---|---|---|---|---|
| `c80b79c` | 2026-10-30 | 任务产物批量成熟 | 1284/-0 | 6 | — | m404: matured mission deliverables |
| `94e3b06` | 2026-10-31 | 收口/合并 | 1284/-0 | 6 | MERGE-COLLISION | Merge pull request #71 from cxs503/agent/m404 |
| `0f22cf9` | 2026-10-31 | 任务产物批量成熟 | 1092/-0 | 8 | — | m669: matured mission deliverables |
| `a40e29a` | 2026-10-31 | 任务产物批量成熟 | 743/-13 | 4 | PHYS-CORE,NO-DOC | m670: matured mission deliverables |
| `f67f64c` | 2026-10-31 | 任务产物批量成熟 | 1821/-80 | 13 | PHYS-CORE | m679: matured mission deliverables |
| `b9ecf39` | 2026-10-31 | 任务产物批量成熟 | 904/-0 | 15 | — | m705: matured mission deliverables |
| `b7c5544` | 2026-10-31 | 收口/合并 | 1092/-0 | 8 | MERGE-COLLISION | Merge pull request #72 from cxs503/agent/m669 |
| `a91da9d` | 2026-10-31 | 收口/合并 | 743/-13 | 4 | PHYS-CORE,MERGE-COLLISION,NO-DOC | Merge pull request #73 from cxs503/agent/m670 |
| `4ed6f46` | 2026-10-31 | 收口/合并 | 1821/-80 | 13 | PHYS-CORE,MERGE-COLLISION | Merge pull request #74 from cxs503/agent/m679 |
| `3cbee0c` | 2026-10-31 | 收口/合并 | 904/-0 | 15 | MERGE-COLLISION | Merge pull request #75 from cxs503/agent/m705 |
| `f74706b` | 2026-11-01 | 任务产物批量成熟 | 3084/-0 | 13 | GATE-FILTERED | m835: matured mission deliverables（9 个文件被门禁滤除） |
| `0a9c48e` | 2026-11-01 | 任务产物批量成熟 | 3218/-0 | 9 | GATE-FILTERED | m943: matured mission deliverables（79 个文件被门禁滤除） |
| `d324fe4` | 2026-11-01 | 任务产物批量成熟 | 1635/-2 | 6 | GATE-FILTERED | m991: matured mission deliverables（2 个文件被门禁滤除） |
| `c0df9a2` | 2026-11-01 | 任务产物批量成熟 | 3620/-0 | 9 | GATE-FILTERED | m992: matured mission deliverables（9 个文件被门禁滤除） |
| `af3b60d` | 2026-11-01 | 任务产物批量成熟 | 2399/-0 | 5 | GATE-FILTERED | m995: matured mission deliverables（11 个文件被门禁滤除） |
| `ebdeca8` | 2026-11-01 | 任务产物批量成熟 | 4565/-0 | 16 | GATE-FILTERED | m1044: matured mission deliverables（9 个文件被门禁滤除） |
| `098c655` | 2026-11-01 | 收口/合并 | 3084/-0 | 13 | MERGE-COLLISION | Merge pull request #76 from cxs503/agent/m835 |
| `53cf59b` | 2026-11-01 | 任务产物批量成熟 | 3621/-0 | 13 | GATE-FILTERED | m1045: matured mission deliverables（7 个文件被门禁滤除） |
| `dc4f87f` | 2026-11-01 | 收口/合并 | 3218/-0 | 9 | MERGE-COLLISION | Merge pull request #77 from cxs503/agent/m943 |
| `ff50f89` | 2026-11-01 | 收口/合并 | 1635/-2 | 6 | MERGE-COLLISION | Merge pull request #78 from cxs503/agent/m991 |
| `4ffa42c` | 2026-11-01 | 收口/合并 | 3620/-0 | 9 | MERGE-COLLISION | Merge pull request #79 from cxs503/agent/m992 |
| `9c9ff9a` | 2026-11-01 | 收口/合并 | 4565/-0 | 16 | MERGE-COLLISION | Merge pull request #81 from cxs503/agent/m1044 |
| `4bc2010` | 2026-11-01 | 任务产物批量成熟 | 1909/-0 | 4 | GATE-FILTERED | m1129: matured mission deliverables（8 个文件被门禁滤除） |
| `bfdb5e3` | 2026-11-01 | 收口/合并 | 1909/-0 | 4 | MERGE-COLLISION | Merge pull request #83 from cxs503/agent/m1129 |
| `8add72b` | 2026-11-01 | 任务产物批量成熟 | 3083/-0 | 6 | GATE-FILTERED | m1325: matured mission deliverables（13 个文件被门禁滤除） |
| `99c49c0` | 2026-11-01 | 收口/合并 | 3083/-0 | 6 | MERGE-COLLISION | Merge pull request #84 from cxs503/agent/m1325 |
| `5f924f2` | 2026-11-01 | 收口/合并 | 9/-13 | 1 | MERGE-COLLISION | merge: 并入 main（PR#80 冲突收口：tests/conftest.py 三处同义样板归一取 main |
| `97b78cd` | 2026-11-01 | 收口/合并 | 9/-10 | 1 | MERGE-COLLISION | merge: 并入 main（PR#82 冲突收口：tests/conftest.py 三处同义样板归一取 main |
| `d0e3b54` | 2026-11-01 | 收口/合并 | 2384/-21101 | 65 | MERGE-COLLISION,DELIVERABLE-DROP | Merge pull request #80 from cxs503/agent/m995 |
| `08e783a` | 2026-11-01 | 收口/合并 | 3607/-0 | 12 | MERGE-COLLISION | Merge pull request #82 from cxs503/agent/m1045 |
| `6a9eaae` | 2026-11-01 | 求解器物理修复 | 1118/-1 | 6 | PHYS-CORE,MAIN-RED | fix(dem3d): 恢复外部节点载荷/ pinned supports / 压板阻尼符号——CI 4 红根修 |
| `edded55` | 2026-11-01 | 收口/合并 | 1118/-1 | 6 | PHYS-CORE,MERGE-COLLISION | Merge pull request #85 from cxs503/canon/dem3d-external-lo |

## 4. 结果图

![逐提交行数账本与翻覆](figures/m1630/m1630_fig1_commit_ledger.png)

图 1 (a) 每个提交的新增/删除行（颜色=主主题）：`d0e3b54` 的删除量（-21101）是次大删除提交的 263.8 倍，这正是文件批量消失的位置；(b) 历史触碰并集 / 净树存在 / 翻覆三者的落差即交付缺口。

![主题×路径类别 与 风险标记](figures/m1630/m1630_fig2_theme_risk.png)

(a) Subject labels omitted CJK glyphs: the container has no CJK font (fc-list shows no noto-cjk/wqy), so tick labels carry short SHA + theme + file count only; full subjects are in the §3.4 table.

图 2 (a) 主题 × 路径类别矩阵：`T-DELIVERABLE`（14 提交）与 `T-MERGE`（17 提交）占据绝大部分 src/tests/docs/scripts，说明本轮上游的改动**主体是任务产物批量并入**而非算法演进；(b) 风险标记分布，`R-MERGE-COLLISION`（17）、`R-GATE-FILTERED`（9） 是主要风险形态。

![CI 触发面](figures/m1630/m1630_fig3_ci_trigger.png)

图 3 CI 触发面缺口：差集中 66 个 `scripts/tests/docs` 文件，全部未被 `.github/workflows/dem3d-benchmarks.yml` 的 106 条显式 `paths:` 覆盖；但这些用例仍会被 `.github/workflows/tests.yml`（无 `paths:` 过滤器，全量 `pytest -q`）兜住，因此缺口是**「基准战役工作流对新用例零覆盖」**，不是「测试不跑」。

## 5. 注意点（踩过的坑 / 局限）

1. **口径陷阱**：逐提交 numstat 累加得到 `+68181/-21316`，净树差集只有 `+12959/-89`。两者差 5 倍，因为 merge 提交按第一父重复计数、且中间批次被删。任何「本轮改了 6.8 万行」的说法都是口径错误，报告必须同时给两个口径。
2. **翻覆 ≠ 丢失**：删除也可能是改名。本工具用 blob 哈希做内容级核验，得到的 `files_relocated_same_blob=0` 才允许下「丢失」的结论。
3. **`import tensordem` 会命中别的工作树**：本机 5 个 worktree 的 editable 安装都在 `sys.path` 上（`m57/m991/m708/m1213/m507`），实测 `import tensordem` 解析到 `work/m991/src`。本单所有命令统一 `PYTHONPATH=src`（或 `PYTHONPATH=`）以免串味。
4. **局限**：只审「仓库可见性」，不核账 147 个被门禁滤除的文件内容（它们不在历史里，只能记欠账）；不评估上游 head 的物理精度（那要跑 benchmark 战役，属另一单）；CI 触发面只解析 `paths:` 显式清单 + 判定「无 paths = 全量」，未模拟 GitHub 的分支过滤语义（注意 `tests.yml` 的 push 分支白名单为 main/feat/**/fix/**，本轮 `agent/*` 分支的推送不触发它）。

## 6. 误差判定

本单是仓库考古/审计类任务，关键量是**计数与集合关系**，不是物理量，故以「独立 git 命令复算」作为基准，判据为逐项零误差（等价于 <3% 要求的更强形式）：

| 关键量 | 工件值 | 独立基准（git 命令） | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 提交数 | 32 | `git rev-list --count HEAD..origin/main` | 0 | 是（0%） |
| 净差集文件数 | 68 | `git diff --name-only HEAD origin/main | wc -l` | 0 | 是（0%） |
| 净新增行 | 12959 | `git diff --shortstat HEAD origin/main` | 0 | 是（0%） |
| 翻覆文件集合 | 59 | `ls-tree origin/main` 逐个比对（59/59 不存在） | 0 | 是（0%） |
| 主题归类 | 3 个主主题 | 逐条人工抽查 32 条提交主题（见 §3.4 表） | 0 | 是（0%） |

工具自身的回归由 `tests/test_audit_upstream_diff_m1630.py`（14 用例）覆盖：主题/风险/路径归类与 CI glob 判据 11 条、合成仓库（加-删-改名）净差集 vs 翻覆 1 条、真仓库与独立 git 命令对账 1 条、渲染器冒烟 1 条；对账与渲染用例在缺少工件/`origin/main` 时自动 skip。

### 6.1 本单实际执行的验证（honest）

| 验证项 | 命令 | 结果 |
|---|---|---|
| 工具幂等 | `audit_upstream_diff.py --check` | `idempotent=True`，两次重算字节一致 |
| 新增用例 | `pytest tests/test_audit_upstream_diff_m1630.py` | 14 passed |
| 邻域回归 | `pytest tests/test_audit_upstream_diff_m1630.py tests/test_accounting_output.py tests/test_dem.py` | 28 passed / 1 skipped / 8 subtests passed |
| 全量收集 | `pytest -q --collect-only` | 303 tests collected，无收集错误 |
| 全量运行 | `pytest -q`（480 s 上限） | 跑到约 40%（110+ 用例，0 失败）被 timeout 终止——**未跑完**，全绿需由 CI 复核；本单改动是新增文件，未触碰既有求解器路径 |

## 7. 风险登记与建议动作

| 风险码 | 命中提交 | 严重度 | 建议动作 |
|---|---|---|---|
| R-MERGE-COLLISION | 17 | 低 | 17 条 merge 中 `conftest.py` 三处同义样板做过归一：留意 fixture 语义被静默改变 |
| R-GATE-FILTERED | 9 | 中高 | 147 个被门禁滤除的文件无历史可核：建交付清单（mission_id→文件→sha）落到 append-only 面，后续巡更才有判据 |
| R-PHYS-CORE | 6 | 中 | 6 条提交触碰 DEM 核心求解器（含符号修复）：合入后需跑 UCS/三点弯/楔形阻力基准复算，确认没有把误差推到 >15% |
| R-NO-DOC | 2 | 低 | 2 条改代码无文档同步 |
| R-DELIVERABLE-DROP | 1 | 高 | 追回 `d0e3b54` 删掉的 59 文件：确认哪些是已成熟应保留的产物，从 `d0e3b54^1` 恢复并以独立 PR 重新交付（勿整体回滚 merge） |
| R-MAIN-RED | 1 | 高 | 上游 main 曾红（CI 4 红根），本轮差集必须在自己的分支重跑根因用例后再信 |

**结论**：本仓上游相对本地 HEAD 是纯快进差集（落后 32 / 领先 0），主体是 14 个任务产物的批量并入 + 1 次求解器符号修复；最大风险不是算法，而是 `d0e3b54` 一次性撤回 59 个文件（翻覆率 45.7%，其中 20 个求解器模块在 head 上零引用）与 147 个门禁滤除文件的可追溯性缺口。建议下一单：只做「59 文件清单的三分类（应保留/应废弃/已替代）+ 恢复 PR」，不动其它历史。
