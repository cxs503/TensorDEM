# TensorDEM main CI 清红（m4829）：runner 路径断言 + 证据账本过期

范围：`main@524821fd`（Merge #343）上 `python -m pytest -q` 的三条红项。本增量不改任何物理模型、
不改数值内核，只修两处**门禁**本身：一条把 runner 挂载布局当成不变量来断言的溯源测试，
一条过期并含失效指针的文档证据账本。每条都给出可复算的前后证据（脚本 + 产物）。

| # | 红项 | 现象 | 归因 | 处置 |
|---|---|---|---|---|
| 1 | `tests/test_dem2d_fracture_ledger.py::Dem2DFractureLedgerTests::test_provenance` | `self.assertIn(os.sep + "work" + os.sep, check_provenance())` 在 CI 上失败 | 测试断言的是 **runner 的挂载路径**，不是仓库不变量：GitHub 托管 runner 的目录是 `/home/runner/work/<repo>/<repo>`，含 `/work/`；自托管 runner（`runs-on: ci-loong64`）检出在 `.../_work/<repo>/<repo>`，只有 `_work/`，于是 `"/work/"` 断言为假——而 `tensordem` 确实来自本树，溯源本身是干净的 | 把门禁锚定到**仓库根**（`check_provenance()` 的返回值解析后必须落在本树的 `src/tensordem/`），并把 `check_provenance` 改成可传入被测路径，从而补一条**负控**：外部 checkout 的引擎必须报 `RuntimeError` |
| 2 | `tests/test_doc_evidence_pointers.py::DocumentedEvidenceGateTests::test_cited_source_docs_and_tests_are_tracked` | `docs/threefield_fragmentation_analytic_verification.md:11` 处引用的子包路径被判 `code:MISSING` | 文档把子包路径写成 `fragmentation.py`（漏掉 `src/tensordem/threefield/` 前缀），git 里的真实路径是 `src/tensordem/threefield/fragmentation.py`（同文档第 164 行本来就是对的） | 文档两处指针改为真实路径 |
| 3 | `tests/test_doc_evidence_pointers.py::DocumentedEvidenceGateTests::test_committed_ledger_is_current` | 账本与树不一致 | `docs/doc_evidence_ledger.json` 冻结在 145 篇文档 / 831 条指针；m4819 合并后树里已有 151 篇文档，且 `artifacts/m4812_ledger_negative_controls/` 的 3 个产物被门禁滤除后账本仍记 `TRACKED` | 重生成账本：152 篇 / 879 条指针，3 条由 `TRACKED` 改判 `MISSING + REGENERABLE`（producer 为 `scripts/plot_m4812_ledger_negative_controls.py`），新增 51 条指针全部有据 |

## 1. 溯源门禁：为什么红在 CI 而在工作树是绿的

`check_provenance()` 的语义是「导入的 `tensordem` 必须来自本树」，实现是拿
`tensordem.__file__` 与 `scripts/../` 比对——这部分是对的，工作树/托管 runner 都通过。
错的是测试额外加的那一条 `"/work/" in path`：它把「工作目录叫 work」当成了证据。
同一棵干净的树，只改挂载前缀就会红/绿翻转，这样的门禁既不能防回归（换 runner 就误报），
也不能提供证据（托管 runner 上它恒真）。

复算（`scripts/m4829_ci_gate_evidence.py`，产物 `artifacts/m4829_ci_gate/gate_evidence.json`
与 `artifacts/m4829_ci_gate/provenance_gate_matrix.png`）：

| 检出布局 | `main` 的门禁（`"/work/" in path`） | 修复后的门禁（解析后须落在本树 `src/tensordem/`） |
|---|---|---|
| `/home/runner/work/TensorDEM/TensorDEM`（托管 runner） | PASS | PASS |
| `/data/ci_env/_work/TensorDEM/TensorDEM`（自托管 loong64） | **FAIL（假红）** | PASS |
| `/tmp/_work/TensorDEM/TensorDEM`（沙箱复现） | **FAIL（假红）** | PASS |
| `/opt/other-checkout/...`（负控：外部 checkout） | FAIL | **FAIL（正确拒绝，`RuntimeError`）** |

端到端复现：把工作树整份复制到 `/tmp/_work/TensorDEM/TensorDEM`，用 `main` 上的测试文件跑

```
python -m pytest -q tests/test_dem2d_fracture_ledger.py    # 该文件先替换成 HEAD 版本再跑
# FAILED ...::Dem2DFractureLedgerTests::test_provenance
```

同一目录改用修复后的测试：

```
python -m pytest -q tests/test_dem2d_fracture_ledger.py tests/test_doc_evidence_pointers.py
# 20 passed
```

即：CI 的红被本地按同形布局复现，修复后在同形布局下通过——不是把断言放松，而是把断言换成
真正的不变量，并加负控保证它不会退化成恒真。

代码侧改动（`scripts/verify_dem2d_fracture_ledger.py`）：

* `check_provenance(module_path=None)`：默认仍取 `tensordem.__file__`，但允许被测路径注入，
  使拒绝分支可在不污染 `sys.modules` 的前提下被测试；
* 通过时仍返回源码路径，`main()` 的 `provenance` 字段语义不变。

测试侧改动（`tests/test_dem2d_fracture_ledger.py`）：`test_provenance` 断言
`Path(path).resolve().parent.parent == <repo>/src` 且文件名是 `__init__.py`；
新增 `test_provenance_rejects_a_foreign_tree`。顺手清掉该文件里 3 个从未使用的导入
（`EPS_B`、`L0`、`R`，`ruff check` 在该文件上的 3 条 `F401` 随之清零）。

## 2. 证据账本：过期 + 一条失效指针

`artifacts/m4829_ci_gate/ledger_regeneration.png` 是重生成前后的逐状态计数
（数字由脚本从 `git show HEAD:docs/doc_evidence_ledger.json` 与工作树的账本程序化读出，不手抄）：

| 量 | 提交态（过期） | 重生成后 |
|---|---|---|
| 扫描文档数 | 145 | 152 |
| 指针数 | 831 | 879 |
| `code:TRACKED` | 275 | 290 |
| `test:TRACKED` | 251 | 263 |
| `data:MISSING`（可重建） | 44 | 47 |
| 判 `TRACKED` 的占比 | 0.8472 | 0.8476 |
| 失效指针（ORPHAN） | 0 | 0 |

判读：账本过期来自**文档侧先合并、账本没跟上**：51 条新指针里 36 条来自 m4819 合并的文档，
15 条来自本清红文档及其产物，全部指向已提交的代码/文档/测试。判 `TRACKED` 的占比几乎不动
（0.8472 → 0.8476）：新增的 3 条是「产物被门禁滤除但 producer 仍在」的可重建条目，不是无主指针。`test_audit_covers_the_documentation` 的非空洞阈值（>60% 判 `TRACKED`）仍有余量。

重生成命令：`python scripts/audit_doc_evidence.py --json docs/doc_evidence_ledger.json`，
`--check` 复核通过（`ledger up to date`）。

## 3. 验证清单（本增量实际跑过的）

| 命令 | 结果 |
|---|---|
| `python -m pytest -q tests/test_dem2d_fracture_ledger.py tests/test_doc_evidence_pointers.py` | 20 passed |
| 同上，但整棵树复制到 `/tmp/_work/TensorDEM/TensorDEM`（自托管布局同形） | 20 passed |
| `main` 版测试文件在 `/tmp/_work/...` 布局下 | 1 failed（红项复现） |
| `python scripts/audit_doc_evidence.py --check` | `ledger up to date` |
| `python -m ruff check <改动的 3 个 py>` | All checks passed |
| `python -m pytest -q`（全量，loongarch64） | 见下方结论 |

## 4. 注意点

* 溯源门禁**不能**用「路径里有没有某个目录名」做判据：这只测 runner，不测仓库。
  同类风险：任何依赖 `cwd`、`$GITHUB_WORKSPACE`、`/home/runner` 的断言在自托管 runner 上都会假红。
* 账本重生成会**如实记录**被门禁滤除的产物（`MISSING + REGENERABLE`）：它们是可重建的
  （producer 在树内），不是无主指针；不要把这类条目手改成 `TRACKED` 来「清红」。
* 本增量不触碰任何物理/数值内核，也不动 m4812 / m4819 的成果文件；账本里那 3 条 `MISSING`
  是门禁滤除的事实，重跑 `scripts/plot_m4812_ledger_negative_controls.py` 才会把它们变回可提交产物。
* 适用范围：`tests.yml` 的 `unit-tests` 作业（`python -m pytest -q`）。本仓没有独立的 lint 作业，
  `ruff` 只在 `[test]` extra 里，故未把 lint 纳入验收判据。