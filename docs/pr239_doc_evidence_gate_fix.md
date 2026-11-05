# PR #239 修红：`docs/` 证据链门禁（main 漂移，非分支缺陷）

Status: 修好并本地复现验证（`tests/test_doc_evidence_pointers.py` 14 passed，0 orphan）。
Reproduce:

```bash
python scripts/audit_doc_evidence.py --check   # 之前: exit 1 (ledger stale)
python scripts/audit_doc_evidence.py           # 重写 docs/doc_evidence_ledger.json
python -m pytest tests/test_doc_evidence_pointers.py -q
```

## 1. 结论先说

分支 `agent/m3813` 的 PR #239 相对 main 只带 1 个提交、且只加一份 `docs/` 报告，
它自己不可能让实现类测试变红。CI 红是**主干漂移**：main 上 `docs/` 的引用与
`docs/doc_evidence_ledger.json` 已经不同步，门禁
`tests/test_doc_evidence_pointers.py` 在 PR 的 merge tree 上直接红——任何从 main
切出的 PR 都会红，与 PR #239 内容无关。

复现（PR merge tree = main + 本分支文档，等价于 `origin/main` 内容）：

```
$ python -m pytest tests/test_doc_evidence_pointers.py -q
4 failed, 10 passed          # test_cited_source_docs_and_tests_are_tracked /
                             # test_committed_ledger_is_current /
                             # test_no_orphan_pointers /
                             # test_uncommitted_artifacts_have_a_producer
```

把本分支新增的那份 doc 临时移走再跑，仍是同样 4 红 → 与本分支改动无关，纯主干欠账。

## 2. 逐条根因与修法

门禁规则（`scripts/audit_doc_evidence.py`）：`docs/**.md` 里出现的每个仓库相对路径
都是「证据指针」，必须三项同时成立——(a) git 有该文件（TRACKED）；(b) `code/doc/test`
类指针不得悬空；(c) 未入库的数据/图必须有 `scripts|benchmarks|tests|src` 里的生产者
（REGENERABLE），否则判 ORPHAN。

| # | 指针（文档:行） | 状态 | 根因 | 修法 |
|---|---|---|---|---|
| 1 | `docs/pr_red_rca_m3184_m3201_m3207_m3267.md:13,105-107` 引三个会话临时脚本 | ORPHAN ×3 | 那三个脚本只存在于会话工作树，从未入库，树内无人能重建 | 改为等价 `curl` 复现命令（URL 被审计器按 EXTERNAL 跳过），产物表写明「临时脚本未入库」 |
| 2 | `docs/dem2d_energy_ledger_analytic_verification.md:113` 引 `lattice_elasticity2d.md`（docs 下不存在此名） | MISSING (doc) | 该文档名不存在，真正讨论 `bond_stiffness`→模量的文档是 `docs/dem2d_lattice_elasticity.md` | 改正引用名 |
| 3 | `docs/dem3d_bounce_chain_verification.md:16` 引 `tests/test_dem3d_restitution.py` 里的 `EngineClosureTests` 类（已改名） | MISSING (test node) | 类已改名，树内只有 `RestitutionAnalyticsTests` / `RestitutionEngineTests` | 改引 `RestitutionEngineTests`（单对/单压板撞击那组引擎用例） |
| 4 | `docs/dem3d_engine_threshold_crossing.md:130` 的 把两个文件名用 `/` 串成的伪指针 | MISSING (data) | 两个文件名被 `/` 串成一个伪指针 | 改写成「metrics.json 与 summary.md 及两图」 |
| 5 | `docs/dem3d_execution_evidence_ledger.md:8,125` 引静态账本脚本 `audit_verification_evidence.py` 与契约说明 `verification-evidence/README`（后者是 docs 子目录下的 README） | MISSING ×2 | 这两个文件的交付提交（m2853）从未并入 main，树内不存在 | 改为如实声明「静态账本脚本/契约说明未随交付入库，树内不可重跑」 |
| 6 | `docs/doc_evidence_ledger.json` 与当前 `docs/` 不同步 | stale | 上述文档与近期交付改过 docs 却未重跑审计器 | `python scripts/audit_doc_evidence.py` 重写账本（123 条未入库产物全部 REGENERABLE，0 ORPHAN） |

原则：**不删除引用、不伪造证据**。能改正名的改正名；只有「原件确实不在树内」时才如实
写明缺席，并给出可执行的等价复现路径。

## 3. 修后状态（程序化读出，未手抄）

```
$ python scripts/audit_doc_evidence.py
documents scanned : 137
pointers          : 790
status counts     : {'code:TRACKED': 259, 'data:MISSING': 41, 'data:TRACKED': 38,
                     'doc:TRACKED': 112, 'figure:MISSING': 82, 'figure:TRACKED': 20,
                     'test:TRACKED': 238}
remedy            : {'REGENERABLE': 123, 'TRACKED': 667}
unresolved        : 123 (0 orphan)

$ python -m pytest tests/test_doc_evidence_pointers.py -q
14 passed
```

未入库的 123 条（122 data/figure + 1 其余类别）全部带生产者脚本，属**允许缺席但可重建**，
是仓库既定口径（`REGENERABLE`），不是本次新增欠账。

## 4. 注意点 / 局限

* 只改文档与账本，未动 `src/`、`tests/`、`scripts/`——门禁红是文档-账本漂移，代码无缺陷。
* `audit:historical` 引号块是禁止滥用的豁免（门禁要求只允许出现在
  `docs/dem3d_doc_evidence_pointer_audit.md`），本次修复没有用它绕过任何一条。
* 宿主 load average 一度 >2000（多智能体共用），全量 1505 用例的整套绿票在本机难以跑完；
  本轮以「门禁文件全绿 + 红/绿对照（移走本分支文档仍红）」定因，全量套件交 PR CI 复核。

## 5. 误差判定

本任务无物理量、无基准数值，判据是**布尔门禁**，逐项列出：

| 关键量 | 修前 | 修后 | 判据 | 是否达标 |
|---|---|---|---|---|
| `test_committed_ledger_is_current` | FAIL | PASS | ledger 与扫描结果集合相等 | 是 |
| `test_cited_source_docs_and_tests_are_tracked` | FAIL（4 条悬空） | PASS（0 条） | code/doc/test 指针 100% TRACKED | 是 |
| `test_no_orphan_pointers` | FAIL（4 ORPHAN） | PASS（0 ORPHAN） | ORPHAN 数 = 0 | 是 |
| `test_uncommitted_artifacts_have_a_producer` | FAIL | PASS | 未入库指针 100% REGENERABLE | 是 |
| 门禁文件通过数 | 10 passed / 4 failed | 14 passed / 0 failed | 0 failed | 是 |

无基准的数值误差判定不适用（本文不含测量值），故声明「无基准」。