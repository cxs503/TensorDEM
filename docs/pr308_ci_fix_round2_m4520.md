# PR #308 CI 修复（第 2 轮，mission m4520）

## 1. 案例介绍
- 对象：TensorDEM PR #308（分支 `agent/m4220`，head `aea434e`，基于 main `6fc0a10`，仅含
  `docs/pr235_ci_fix_summary.md` 一个文档提交）。
- 现象：CI `unit-tests` 作业红。第 1 轮（m4220）曾修复
  `tests/test_dem3d_contact_invariants.py::test_mixed_topology_all_invariants_pass`
  的 8 个 subtest 失败，但分支后来在更新的 main 上重建（`aea434e`），**修复丢失**，红测复现。

## 2. 怎么做的
1. 本地复现：在 PR 合并态（= main `4a08a6d` + 该分支文档）跑全量 pytest（1508 用例）。
   基线：`8 failed, 276 passed`（-x 首停），8 个失败全部为
   `test_mixed_topology_all_invariants_pass` 的 subtest（4 不变量 × 2 种子）。
2. 根因：测试用 `getattr(module, 短标签)` 动态取函数，而
   `tensordem.contact_invariants` 暴露的是长函数名
   （`translation_invariance`/`rotation_equivariance`/`brute_force_agreement`/
   `determinism_repeatability`），短标签 getattr 抛 `AttributeError`。
3. 修复（最小改动，仅 1 个文件）：把 getattr 换成显式 标签→callable 映射字典；
   translation 用例沿用审计的刚性偏移 `(1.5, -7.25, 3.5)`。
4. 验证：`pytest tests/test_dem3d_contact_invariants.py` → **15 passed, 56 subtests passed**
   （修复前 8 个 subtest 失败）。全量套件（xdist -n 32）：`24 failed, 1472 passed, 9 skipped`，
   其中 contact_invariants 文件已全绿。

## 3. 计算结果（程序化注入）
| 项 | 修复前 | 修复后 |
|---|---|---|
| contact_invariants 文件 | 8 subtest 失败 | 15 passed / 56 subtests passed |
| 全量套件（1508 用例） | 32 失败（8 目标 + 24 存量） | 24 失败（全部为 main 存量，见 §5） |

## 4. 结果图
无图：本单为测试修复/后端任务，无场量场量云图可言；判据为测试通过与否（二值），
以 pytest 输出为证（上表数字直接取自 pytest 汇总行）。

## 5. 注意点（重要：main 存量红测清单）
全量复跑确认以下 24 个失败**在干净 main 上同样失败**（与本 PR 的改动无关，
本单按「不得改动无关文件」纪律未动）：
- `tests/test_dem3d_energy_friction_rolling.py` ×11（`build_subsystem` 路径
  `ValueError: bond_stiffness must be finite and positive` / `_bond_keys` IndexError）
- `tests/test_dem3d_hertz_stability.py::TestHertzStabilityMargin` ×6
  （测试传 `bond_stiffness=0.0`，而 `DEM3DConfig.__post_init__` 自初版起要求 >0）
- `tests/test_doc_evidence_pointers.py::DocumentedEvidenceGateTests` ×4
  （文档引用 `docs/lattice_elasticity2d.md` 在提交中缺失；ledger 过期需重跑
  `scripts/audit_doc_evidence.py`）
- `tests/test_ridge_dem_core_physics.py::test_recommended_dt_elastic_limit` ×1
- `tests/test_threefield_lbm_transport.py::test_streaming_is_mass_conserving` ×1
- `tests/test_verification_execution_ledger.py::test_timeout_is_contained_and_labelled` ×1

**结论**：PR #308 的 CI 要全绿，还需单独任务修掉 main 上这 24 个存量失败
（涉及 5 个测试文件 + 文档 ledger 再生成），超出本单范围，如实上报。

## 6. 误差判定
CI 修复类任务无连续量基准，判据为二值（测试通过/失败）：目标文件 56/56 subtest 通过，
显式回答「是否 <3%」：不适用（无连续误差量）；目标红测 100% 转绿。
honest 状态：**部分完成**——PR #308 的已知红测（8 个）已修并验证；PR 整体 CI
仍会因 §5 的 main 存量失败而红，需后续任务处理。
