# PR #258 `unit-tests` 红 → 绿的修复记录（m4332）

分支 `agent/m3897`（PR #258，提交 `m3897: matured mission deliverables`）的 `unit-tests` job 红。
本文件记录复现、根因、最小修复与验证证据。

## 1. 复现（本地，等价于 CI 的 merge ref）

CI 的 `pull_request` 触发检出的是 `refs/pull/258/merge`，即「分支 ⋈ 当前 main」，而不是分支本身。
本地等价重建：

```bash
git checkout -B agent/m3897 8c2b8e8          # PR head
git merge origin/main --no-edit              # = PR merge tree（本次 main = 4a08a6d）
PYTHONPATH=src python -m pytest -q tests/test_doc_evidence_pointers.py
```

结果：`4 failed, 10 passed`。红在文档证据指针门（`tests/test_doc_evidence_pointers.py`，
由 `scripts/audit_doc_evidence.py` 驱动、以 `docs/doc_evidence_ledger.json` 为棘轮）。

## 2. 根因（两类，均为「文档引用了树上不存在的路径」）

扫描 `python scripts/audit_doc_evidence.py` 得到 9 条悬空指针（`status=MISSING`，其中 4 条 `ORPHAN`）：

| # | 文档:行 | 指针 | 引入方 | 处置 |
|---|---|---|---|---|
| 1 | `docs/dem3d_wedge_bow_probe_contract.md`:34 | `scripts/ice_floe_physics.py` | **本 PR（m3897）** | 该实现实际在 `benchmarks/ice_floe_physics.py` → 改指向 |
| 2 | `docs/dem2d_energy_ledger_analytic_verification.md`:113 | `docs/lattice_elasticity2d.md` | main（已合入的历史 PR） | 同目录下真实文件是 `docs/dem2d_lattice_elasticity.md` → 改指向 |
| 3 | `docs/dem3d_bounce_chain_verification.md`:16 | `tests/test_dem3d_restitution.py::EngineClosureTests` | main | 该类现居 `tests/test_restitution.py`（`class EngineClosureTests`） → 改指向 |
| 4 | `docs/dem3d_engine_threshold_crossing.md`:130 | `metrics.json/summary.md` | main | 正则把「metrics.json / summary.md」吞成一个 token（假阳性） → 拆成两个行内 token |
| 5 | `docs/dem3d_execution_evidence_ledger.md`:8 | `scripts/audit_verification_evidence.py` | main | 静态账本脚本随交付门被滤除、从未入库 → 如实注明并指向现存契约文档 |
| 6 | `docs/dem3d_execution_evidence_ledger.md`:125 | `docs/verification-evidence/README.md` | main | 该 README 未随文档入库 → 改指 `docs/dem_analytic_verification_debt_restoration.md` 与本文件 §5 |
| 7–9 | `docs/pr_red_rca_m3184_m3201_m3207_m3267.md`:13/16/105/107 | `runs/ci_logs.py`, `runs/ci_probe.py`, `runs/apply_fixes.py` | main | 一次性驱动脚本从未入库（ORPHAN，无 producer） → 去掉路径 token，如实写明「临时脚本，未入库」 |

要点：第 1 条是本 PR 自身引入的（文档把 `benchmarks/ice_floe_physics.py` 误写成 `scripts/`）；
第 2–9 条在纯 main（`6fc0a10`、`4a08a6d`）上同样复现，属于 main 已红的存量欠账。
由于 CI 检查的是 merge tree，只修第 1 条仍然是红，因此本轮把 9 条一并按「引用的路径必须存在」的
原意修到可解析，并重生成棘轮账本（`docs/doc_evidence_ledger.json`）。

## 3. 修复原则

* 只改「指针文本」，不改任何数值、断言、阈值或引擎代码；`src/`、`tests/` 零改动。
* 缺文件就指向树上真实存在的等价文件；确实未曾入库的一次性脚本，如实降级为「未入库」表述，
  而不是伪造一个空文件来骗过门（那会把不可复现的声明洗成 TRACKED）。

## 4. 验证

```bash
python scripts/audit_doc_evidence.py                        # 0 orphan
python -m pytest -q tests/test_doc_evidence_pointers.py      # 14 passed
python -m pytest -q tests/test_dem3d_wedge_bow_forces_contract.py \
                   tests/test_dem3d_moving_wedge_bow.py \
                   tests/test_threefield.py                  # 33 passed, 1 skipped
```

## 5. 局限（honest）

* 本轮只保证「文档证据链可解析 + 分支原有修复不回退」，没有替 main 修其余功能债务
  （`test_fracture_gen_audit.py`、`test_ice_floe_physics.py` 等 `src`/`scripts` 被滤除的历史欠账）。
* `REGENERABLE` 的 data/figure 指针按门的设计允许缺席（产物不入库、有 producer 即可），
  它们的数值仍未被本轮复算。
* 账本改为程序化重生成，diff 较大（棘轮被刷新到当前文档集），属门设计内的预期更新。