# PR #317 CI 修红（unit-tests=failure）

分支 `agent/m4337`，PR 内容 = 三维直剪试件的静态横向周期闭合面
（`src/tensordem/shear_periodic.py` + `tests/test_dem3d_shear_periodic_closure.py`）。
PR 自身的 13 条用例在本地全绿，失败发生在**仓库级证据台账闸门**
`tests/test_doc_evidence_pointers.py`（4 failed），其中三条是 main 上已存在的
红（合并树继承），一条由本 PR 新增文档触发。诊断与修法如下。

## 1. 复现（本地，合并树）

```
$ git merge origin/main            # 把 main 的台账更新合入分支
$ python -m pytest tests/test_doc_evidence_pointers.py -q
4 failed, 10 passed
```

同一套断言在 `origin/main`（1a5a9c9）的干净工作树上**同样 4 failed**，即
`test_cited_source_docs_and_tests_are_tracked` / `test_no_orphan_pointers` /
`test_uncommitted_artifacts_have_a_producer` / `test_committed_ledger_is_current`
四条是主干继承红，PR 只要合 main 就必然一起红；只有
`test_committed_ledger_is_current` 额外受本 PR 新增文档影响。

## 2. 根因与修法（逐条）

| # | 失败断言 | 根因 | 修法 |
|---|---|---|---|
| R1 | `test_cited_source_docs_and_tests_are_tracked` | `docs/dem2d_energy_ledger_analytic_verification.md:113` 引用的 一个旧文档名（2D 点阵弹性，已被 `docs/dem2d_lattice_elasticity.md` 取代）在树里不存在 | 改指正确文档名（1 行），指针转 TRACKED |
| R2 | `test_no_orphan_pointers` + `test_uncommitted_artifacts_have_a_producer` | `docs/pr_red_rca_m3184_m3201_m3207_m3267.md` 把当时工作区里的一次性脚本写成仓内路径（runs 目录下三个 py 文件），这些文件从未入库、也无生产者 → ORPHAN | 把三处改为「工作区临时脚本（未入库）」的散文描述，保留原结论与日志摘录，不再制造悬空指针 |
| R3 | `test_committed_ledger_is_current` | 台账 `docs/doc_evidence_ledger.json` 落后于当前树：main 的台账由 #311/#318 重生成，本 PR 又新增 `docs/dem3d_shear_periodic_closure.md` 与 `docs/dem3d-shear-periodic-closure/` 产物 | 机生成重跑 `scripts/audit_doc_evidence.py`（零手抄），台账与扫描结果逐字节一致 |
| R4 | （证据质量，非失败） | 本 PR 文档把场图写成文档相对路径 `figures/field_map.png`，只能记为 REGENERABLE | 改为仓根相对 `docs/dem3d-shear-periodic-closure/figures/field_map.png`，转 TRACKED |

## 3. 修复后状态（机生成，`scripts/audit_doc_evidence.py` 输出）

```
documents scanned : 143
pointers          : 815
status counts     : {'code:TRACKED': 268, 'data:MISSING': 42, 'data:TRACKED': 39,
                     'doc:TRACKED': 119, 'figure:MISSING': 83, 'figure:TRACKED': 22,
                     'test:TRACKED': 247}
remedy            : {'REGENERABLE': 125, 'TRACKED': 691}
unresolved        : 125 (0 orphan)
```

即 code/doc/test 三类指针**无 MISSING**（干净 clone 里全部可解析），
未入库的 125 条 data/figure 全部 REGENERABLE（有生产者脚本），ORPHAN = 0。

## 4. 验证

```
$ python -m pytest tests/test_doc_evidence_pointers.py -q          # 14 passed
$ python scripts/audit_doc_evidence.py --check                     # ledger up to date
$ python -m pytest tests/test_dem3d_shear_periodic_closure.py -q   # 13 passed
```

## 5. 改动面

只动 4 个文件 + 1 个台账：两个文档的悬空指针（各 1–5 行）、本 PR 自己的场图路径、
台账重生成；`src/`、`tests/` 的算法与用例**未改动**（PR 的物理内容原样保留）。

## 6. 注意点

* 本机 `import tensordem` 会通过遗留的 editable 安装解析到**兄弟工作树**
  （`work/m4337/src`），全量套件必须带 `PYTHONPATH=$PWD/src` 跑，否则测的是别人的代码
  ——这是同一台账文档 §4 已经记录过的 provenance 陷阱。
* 台账是 ratchet：任何新增文档/产物指针都要重跑 `scripts/audit_doc_evidence.py`，
  否则 `test_committed_ledger_is_current` 必红。