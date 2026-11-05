# PR #200（分支 `agent/m3296`）— main 变脏的三方合并与逐文件解冲

**日期**：2026-11-05
**任务**：TensorDEM #200（治愈单 m4012）
**仓库**：`TensorDEM`
**旧 SHA**：`agent/m3296 = 4fe28d65a3f90b43ea334ce625db7b26c98bdcca`
**main（并发基线）**：`6c6f98b7ac8fbd889db459e06329901869b89dbb`
**merge-base**：`f5de971c03f6363afa701063a833a3766c874eca`
**治愈提交**：本文件所在提交（父提交 = `4fe28d6` 旧 SHA + `6c6f98b` main），内容对 main 干净可合（`git merge-tree` 无冲突）
**回流情况**：手工回流到 `agent/m3296` 的操作被平台安全策略拦截（REJECTED），按纪律交平台验收后自动发布；
本地 `agent/m4012` 已快进到该治愈提交，包含 PR #200 全部交付内容

## 1. 为什么会脏

`git merge-tree --write-tree --name-only origin/main origin/agent/m3296` 报 `mergeable=false`，
冲突面只有一个文件：

```
src/tensordem/dem3d_wedge.py
Auto-merging src/tensordem/dem3d_wedge.py
CONFLICT (content): Merge conflict in src/tensordem/dem3d_wedge.py
```

两侧自 merge-base 起的改动规模极不对称（`git diff --stat f5de971 <side>`）：

| 侧 | 文件数 | 增删 | 内容 |
|---|---|---|---|
| `main` | 20 | +2553 / −413 | PR #166 治愈（`fracture_gen.py` 语义并集 + mass footprint 审计）、m3317 门禁清理（`benchmarks/ice_floe_physics.py` 锚点实现恢复）、接触阻尼约定、wake-LBM 解析验证、`docs/doc_evidence_ledger.json` 冻结的**证据指针账本** |
| `agent/m3296` | 3 | +223 / −1 | `MovingWedgeBowIceDEM3D.forces` 接受并转发 `update_history`（探针不得推进 Cundall–Strack 滑移史）+ 契约回归测试 + 契约文档 |

**冲突的实质**：两侧是**同一意图的重复引入**，不是互斥改动。
main 侧在别处（m2942 治愈波次）已经把 `IceDEM3D.forces` 的 `update_history` 关键字传进
bow 子类，写成两行调用；m3296 侧独立做了同一件事，并在同一处附上了 9 行说明性注释。
功能语义相同 → 这不是取舍，而是**并集**。

## 2. 逐文件解法

### 2.1 `src/tensordem/dem3d_wedge.py`（唯一真冲突，`UU`）

冲突区（`git checkout --conflict=merge` 原样）：

```python
<<<<<<< HEAD   (agent/m3296)
        # ``update_history`` is part of the ``IceDEM3D.forces`` contract: probes
        # (diagnostics, energy audits, invariance checks) call with
        # ``update_history=False`` so that merely monitoring a run does not
        # advance the Cundall-Strack slip history.  It has to be accepted and
        # forwarded here as well, otherwise the bow subclass silently narrows
        # the base-class signature and every probe raises TypeError.
        force, _ = super().forces(
            external_forces,
            update_fracture=update_fracture,
            update_history=update_history,
        )
=======
        force, _ = super().forces(external_forces, update_fracture=update_fracture,
                                  update_history=update_history)
>>>>>>> github/main
```

| 冲突点 | main | m3296（HEAD） | 取用 | 理由 |
|---|---|---|---|---|
| `forces` 的 `update_history` 参数 | 已在签名中（`keyword-only`，默认 `True`） | 同（两侧签名逐字相同 → 自动合并，无冲突） | 单一份 | 自动合并已一致，无需决策 |
| `super().forces(...)` 调用 | 两行硬换行，转发 `update_history` | 多行展开 + 9 行契约注释 | **m3296 侧整块** | 两侧是**同一次调用**，语义完全等价（关键字实参、同一顺序）；m3296 侧是严格超集（含解释「为什么探针必须能传这个 flag」与失败模式 `TypeError`）。取并集=保留注释 + 保留转发，**main 侧没有一行内容被丢弃**（它的两行调用是同一调用的另一种排版） |

解冲后（`src/tensordem/dem3d_wedge.py:112-127`）：

```python
    @torch.no_grad()
    def forces(
        self,
        external_forces: torch.Tensor | None = None,
        *,
        update_fracture: bool = True,
        update_history: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``update_history`` is part of the ``IceDEM3D.forces`` contract: probes
        # ... (9 行契约注释)
        force, _ = super().forces(
            external_forces,
            update_fracture=update_fracture,
            update_history=update_history,
        )
```

**验证无静默丢失**：AST 顶层重复定义扫描 `{}`（见 §3.3）；两份转发调用的等价性由运行期证据背书
（§3.4 探针在 `update_history=False` 下不推进滑移史，而默认路径仍推进）——
若转发被丢了任何一个关键字，`IceDEM3D.forces` 会重新落回默认 `update_history=True`，探针就会污染史。

### 2.2 两处「合并后才暴露」的集成修理（非冲突标记，但属同一治愈范围）

机械 overlay 的另一种形态：**冲突标记为零，但语义上仍然坏**。m3296 的三份交付物写在旧树上，
与 main 的并发重构（m3317 的脚本归位、`doc_evidence_ledger.json` 的账本棘轮）撞车。
原始合并结果（未做 §2.2 修理）实测 **3 红**，全部落在 `tests/test_doc_evidence_pointers.py`：

| 门禁测试 | 症状 | 根因 | 修法 |
|---|---|---|---|
| `test_cited_source_docs_and_tests_are_tracked` | 契约文档第 34 行引用的脚本路径（旧树 `scripts/` 下）为 `MISSING` | m3296 在旧树上写文档时脚本位于 `scripts/` 目录；main 的 m3317 恢复把锚点实现落在 `benchmarks/ice_floe_physics.py` | 指针改指 `benchmarks/ice_floe_physics.py`（**不删这句话**——它记录的是一段真实历史缺陷，只把路径对齐到合并后树中实际存在的文件） |
| `test_committed_ledger_is_current` | `ledger is stale`（+6 条新指针未入账） | `docs/doc_evidence_ledger.json` 是 main 引入的**冻结账本**：任何新文档的指针集必须先入账，否则闸门视为「不可复算的断言」 | `python scripts/audit_doc_evidence.py --json docs/doc_evidence_ledger.json` 重新生成：107 docs / 527 pointers / 0 orphan |
| `test_uncommitted_artifacts_have_a_producer` | 同上一条的派生失败（该脚本路径被归为无 producer 的未提交物） | 同上 | 同上 |

修后 `tests/test_doc_evidence_pointers.py` = **14 passed**，`audit_doc_evidence.py --check` = `ledger up to date`。

### 2.3 合并结果的文件面（20 + 3 侧文件，无删减）

* main 侧 20 个文件：全部以 **纯 main 版本**进入合并结果（无冲突，逐字节一致，见 §3.2）；
* m3296 侧 3 个文件：`src/tensordem/dem3d_wedge.py`（手工并集）、
  `tests/test_dem3d_wedge_bow_forces_contract.py`（6 个契约测试，纯 m3296 版本）、
  `docs/dem3d_wedge_bow_probe_contract.md`（纯 m3296 版本 + §2.2 的一处指针修复）；
* **无一侧文件被丢弃**（`dropped_files = []`，§3.1）。

## 3. 无静默丢失：程序化验证

工具：`scripts/verify_pr200_merge.py`（本 PR 新增，可复跑），记录：`artifacts/pr200/verify_pr200.json`，
对照图：`artifacts/pr200/pr200_merge_evidence.png`（左：合并面文件账目；右：main 基线 vs 治愈后全量套件）。

```bash
python scripts/verify_pr200_merge.py \
    --base github/main --head github/agent/m3296 --merge HEAD \
    --tests artifacts/pr200/suite_results.json
```

### 3.1 文件面账目（逐类可数）

唯一侧文件 = `main 侧 20 ∪ m3296 侧 3`（其中 `src/tensordem/dem3d_wedge.py` 双侧重合）→ 22 个侧文件；
合并结果共 27 个条目，分解后账目闭合：

| 类别 | 计数 | 判据 |
|---|---|---|
| 仅 main 改动、合并后与 main 逐字节相同 | 19 | 无冲突、无干涉 |
| 仅 m3296 改动、合并后与 m3296 逐字节相同 | 1（`tests/test_dem3d_wedge_bow_forces_contract.py`） | 同上 |
| 双侧共改、手工并集 | 1（`src/tensordem/dem3d_wedge.py`） | §2.1 逐条决策表 |
| **已声明**的偏离（各有理由，不允许静默） | 2 | 见下表 |
| 本次治愈新增（两个分支都没有） | 5 | 文档/审计脚本/证据产物，非「丢内容」 |
| **丢弃** | **0** | `(main_side ∪ branch_side) − merged = ∅` |
| **未声明的偏离** | **0** | `undeclared_amendments = []` |

已声明的两处偏离（`DECLARED_AMENDMENTS`，写死在审计脚本里，不是口头承诺）：

| 文件 | 侧 | 理由 |
|---|---|---|
| `docs/doc_evidence_ledger.json` | main | 用 `scripts/audit_doc_evidence.py` 重新生成，把新契约文档的指针纳入账本（账本是棘轮，只能重生成不能手改） |
| `docs/dem3d_wedge_bow_probe_contract.md` | m3296 | 一处指针 `scripts/…` → `benchmarks/ice_floe_physics.py`（main 的 m3317 恢复把脚本搬到了 `benchmarks/`） |

### 3.2 单侧文件逐字节一致（不信文本合并，只信字节）

对**仅一侧改动**的每个文件，比较 `git show <merge>:<path>` 与 `git show <side>:<path>`：

```
only-one-side files byte-identical to that side : 20 / 20
declared amendments                              : 2
undeclared amendments (silent loss)              : 0
```

即：除两个双侧文件（1 手工并集 + 2 已声明偏离）外，合并结果与「改动方」的版本**逐字节相同**——
没有缩进漂移、没有 hunk 丢失。

### 3.3 结构性卫生

| 检查 | 结果 |
|---|---|
| 冲突标记残留（`.py/.md/.json/.toml/.yml` 全扫，markdown 围栏内的引用除外） | `{}` |
| `src/tensordem/dem3d_wedge.py` AST 解析 | OK |
| 顶层重复定义（并集解冲的经典后遗症） | `{}`（无重复的 `forces` / `_wedge_contact`） |

### 3.4 行为证据：契约真的成立，且不是空转

`inspect` + 运行期探针（`artifacts/pr200/verify_pr200.json → contract_audit`）。
**关键设置**：新造的格点相邻球心距恰好 `2r`（相切而不重叠）→ 没有活跃接触对，
「探针不动史」会 `{} == {}` 空转通过。因此审计先把格点向心压缩 3 %（`_compress`），
强制产生真实接触：实测 **33 个活跃接触对**。

| 量 | 期望 | 实测 |
|---|---|---|
| 签名含 `update_history` 且 keyword-only | True | True |
| 活跃接触对数（反空转前提） | > 0 | **33** |
| 两次 `forces(update_fracture=False, update_history=False)` 力场逐元素相等 | True | True |
| 两次探针后 `_tangential_state` 指纹不变 | True | True |
| **正控①**：`update_history=True` 调用必须推进滑移史（否则「无副作用」可由空实现骗取） | True | True |
| **正控②**：默认步之后再调探针，史仍不变 | True | True |
| 探针力范数（压缩格点，单次评估） | — | 59.24 N |

## 4. 测试证据

命令（本地 99 控制面，Python 3.11.6 / torch 2.7.1，`-p no:cacheprovider`）：

```bash
python -m pytest -q
```

| 树 | passed | failed | skipped | subtests | 耗时 |
|---|---|---|---|---|---|
| `main` 基线（`6c6f98b`） | 1232 | 0 | 2 | 152 | 546 s |
| 原始合并结果（未做 §2.2 修理） | 1234 | **3** | 3 | 152 | 508 s |
| **最终合并结果（本分支）** | **1237** | **0** | 3 | 152 | 548 s |

* 相对基线：**+5 passed / +1 skipped / 0 failed**；
  m3296 的契约文件共 6 个用例（`tests/test_dem3d_wedge_bow_forces_contract.py`），其中
  `tests/test_dem3d_wedge_bow_forces_contract.py:109` 自带条件跳过
  （`no active contact pair in this lattice, nothing to advance`）→ 5 passed + 1 skipped，
  账目闭合：`1232 + 5 = 1237`，`2 + 1 = 3`。
* **回归数 = 0**，即「不得低于 main 基线」判据满足（见 §5 误差判定）。
* 诚实说明：全量套件测于 §2.2 修理完成后的合并树（1237 passed）；其后仅追加本文件与
  `artifacts/pr200/` 证据产物（纯文档/数据，不改任何 `src`/`tests`）。这些追加物在**最终内容**上
  单独复核：`tests/test_doc_evidence_pointers.py` = **14 passed**、`audit_doc_evidence.py --check` = up to date。
* 定向复核（冲突文件相关 + 门禁）：

```bash
python -m pytest -q tests/test_dem3d_wedge_bow_forces_contract.py \
    tests/test_dem3d_moving_wedge_bow.py tests/test_dem3d_wedge_resistance_sensitivity.py \
    tests/test_threefield.py
# 38 passed, 1 skipped

python -m pytest -q tests/test_doc_evidence_pointers.py      # 14 passed
python scripts/audit_doc_evidence.py --check                 # ledger up to date
```

## 5. 误差判定（vs baseline，逐项）

关键量不是物理量而是**门禁量**，基准 = main 基线（`6c6f98b`）同一命令的计数：

| 关键量 | 基准（main） | 合并后 | 绝对差 | 相对误差 | <3%？ |
|---|---|---|---|---|---|
| 失败用例数 | 0 | 0 | 0 | 0 % | **是** |
| 通过用例数 | 1232 | 1237 | +5 | +0.41 % | **是**（且为正向新增，非漂移） |
| subtests 通过数 | 152 | 152 | 0 | 0 % | **是** |
| 文档证据闸门（指针未解析数） | 0 orphan | 0 orphan | 0 | 0 % | **是** |

无物理基准可对（本任务零物理模型改动：唯一源码改动是关键字转发的重排版，动力学方程未触及），
因此不涉及解析解/实验值误差。**结论：全部关键量误差 < 3%，判据满足。**

## 6. 注意点与局限

1. **「零冲突标记」≠「合并干净」**。本例真正的副作用是 m3296 的文档撞上 main 新引入的
   **证据指针账本**：机械合并后 3 个门禁测试变红，而这段内容在 diff 里毫不起眼（一行路径 + 一个 JSON 账本）。
   教训：治愈单必须**跑全量套件**，不能只看冲突文件的测试。
2. **账本是有意为之的棘轮**，不要为了过闸门而删指针或放宽判据；正确做法是
   `scripts/audit_doc_evidence.py --json docs/doc_evidence_ledger.json` 重新入账 + 把失效路径改到真实文件。
3. **路径归位是并发的常态**：m3317 把 `ice_floe_physics.py` 从 `scripts/` 恢复到 `benchmarks/`。
   文档里的路径断言会随重构腐烂，跨分支写文档时优先引用**测试**（不易搬家）而非脚本路径。
4. **机械 overlay 禁令**：本仓历史上有过「按一侧整文件覆盖」导致另一侧 hunk 静默消失的前科
   （见 `docs/pr166_merge_conflict_resolution.md`）。本次对 22 个单侧文件做**逐字节相等**审计、
   对双侧文件逐条列决策表，就是为了让「没有丢内容」成为可复跑的断言而不是声明。
5. **纪律**：本单没有手工改动远端（平台安全策略拒绝）。治愈提交落在本地分支并快进到
   `agent/m4012`，由平台自动发布；PR #200 是否被 `merge_governor` 记账/合并属平台侧周期复验，
   不由本分支决定（本分支只保证「内容可干净合并 + 门禁全绿」）。
6. **复跑的注意点**：`artifacts/pr200/verify_pr200.json` 里的 `merge_ref` 是提交**前**用
   `git write-tree` + `git commit-tree` 构造的等价提交对象（tree 与最终提交逐字节相同），
   这样审计能在落 commit 之前跑、避免「先提交再验证」的自证循环；用 `--merge HEAD` 可在提交后复跑。
7. **局限**：本单只覆盖 PR #200 的合并治愈面，未改动任何数值算法；
   合并后 PR 是否被 `merge_governor` 记账、何时进 main 由平台侧周期复验决定（不由本分支决定）。
   若 main 在本分支推回后继续前进，需要按同一流程复验（本脚本可复跑）。