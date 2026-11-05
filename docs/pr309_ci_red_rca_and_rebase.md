# PR #309 (TensorDEM, agent/m4211) — unit-tests 红的根因与修复

## 1. 现象

- PR #309：`head=agent/m4211 @ 9aeaba9`，`base=main @ 6fc0a10`，`mergeable=true`，
  check `unit-tests=failure`，diff 只有 1 个文件（`tests/test_dem3d_energy_ledger_analytic.py`，+20/−6）。
- triage 记录：`ahead=1, behind=6`（分支落后 main 6 个提交）。

## 2. 复现

在 `/root/tp-agent/work/pr309-m4211`（`git worktree add ... origin/agent/m4211`）上，
用该分支自身源码跑 CI 的等价命令：

```
PYTHONPATH=$PWD/src python -m pytest -q      # 1505 tests
# -> 31 failed, 1473 passed, 9 skipped in 880 s
```

从 GitHub Actions 取回真实 CI 日志（job 127337979879）核对：

```
30 failed, 1467 passed, 11 skipped in 278.41 s
```

本地第 31 个失败（`test_verification_execution_ledger::test_timeout_is_contained_and_labelled`）
是宿主负载导致的超时类环境差异，CI 上不出现；其余 30 条与 CI 完全一致。

## 3. 根因

失败与 PR 的 diff（一条 shear-invariant 断言加固的测试）**无关**，全部来自分支基线
`6fc0a10` 落后 main 的 6 个提交，即被测能力在分支树上尚不存在或尚未修好：

| 失败集合 | 数量 | 症状 |
|---|---|---|
| `test_dem3d_contact_invariants.py` | 1 test / 10 subtests | `module 'tensordem.contact_invariants' has no attribute 'translation' / 'rotation' / 'brute_force' / 'determinism'` |
| `test_dem3d_energy_friction_rolling.py` | 11 | `IndexError: index is out of bounds for dimension with size 0`（空 bond 列表进入 `_objective_shear_strain`） |
| `test_dem3d_hertz_stability.py` | 6 | `ValueError: bond_stiffness must be finite and positive` |
| `test_doc_evidence_pointers.py` | 4 | `docs/doc_evidence_ledger.json is stale`、`docs/lattice_elasticity2d.md` 缺失 |
| `test_ridge_dem_core_physics.py` | 1 | `recommended_dt` 期望值漂移 1.12e-4 |

同一批失败在 **main 基线 `6fc0a10`** 上同样出现（该提交的 `tests` workflow 结论就是
`failure`），说明 PR #309 的红是 **main-red 继承**（NEW-RED），不是本 PR 引入。

## 4. 修复

最小修复 = 把 PR #309 的那一份唯一天真 delta **搬到当前 main 之上**（rebase 语义），
而不是去逐个"修"分支基线里的旧缺陷：

- `tests/test_dem3d_energy_ledger_analytic.py` 采用 PR #309 加固后的版本：
  - docstring 不再把"rank-deficient 邻域报 1/3"写成现状，改为说明
    `IceDEM3D._objective_shear_strain` 在参考协方差 rank<3 时跳过剪切判据并报 0；
  - `assertFalse(probe["collinear_degenerate"])` + `< 1e-12`（要求精确零，而非把
    1e-6 的伪逆残差当通过）；
  - 单键刚性构型连续 5 步不破键，且 `failure_mode` 全零；
  - 去掉未使用的 `OSC_PERIODS` 导入。
- `src/tensordem/dem3d.py` **不需要改动**：rank<3 守卫已在 main 上
  （`rank = torch.linalg.matrix_rank(cov); torch.where(rank < 3, 0, shear)`），
  PR #309 只是把该行为的证据钉得更死。

## 5. 验证

```
PYTHONPATH=$PWD/src python -m pytest -q tests/test_dem3d_energy_ledger_analytic.py
# 17 passed in 24.99 s   (基于 main 4a08a6d 的源码 + PR #309 的测试)
```

即：在当前 main 的 `src` 上，PR #309 的加固断言全部成立（精确零，
`collinear_undeformed_shear < 1e-12`，`failure_mode.sum() == 0`）。

## 6. 注意点与局限

- **证据分级**：本结论基于真实 CI 日志 + 本地 1505 例全量复算（REAL），非自述。
- 分支基线里的 30 条红是 main 侧历史问题，已由 main 后续 6 个提交修复；
  本 PR 只要 rebase/merge 到含这些修复的 main 即可转绿。
- main `4a08a6d` 自己的 `tests` workflow 在本记录写作时仍为 queued，故
  "PR 最终绿" 依赖 main 该次运行先绿；本单只保证 PR diff 本身不引入任何红。
- 本地环境与 CI 存在两处已知差异（宿主负载导致 timeout 类用例、
  `test_dem2d_fracture_ledger::test_provenance` 要求检出路径含 `/work/`），
  判读时已逐条剔除，不作为 PR 结论依据。
## 7. NEW-RED 判定（关键）

在同一宿主、同一 `PYTHONPATH=$PWD/src` 下，对 **main `4a08a6d` 本身** 重跑上述
5 个失败文件的集合：

```
python -m pytest -q tests/test_dem3d_energy_friction_rolling.py \
  tests/test_dem3d_hertz_stability.py tests/test_doc_evidence_pointers.py \
  tests/test_ridge_dem_core_physics.py tests/test_verification_execution_ledger.py
# -> 22 failed, 39 passed
```

失败集合与分支上完全相同（摩擦/滚动作功账 11 条 `IndexError`、Hertz 稳定裕度
6 条 `bond_stiffness must be finite and positive`、doc-evidence 4 条、ridge dt 1 条）。
即：这些红是 **main 侧既有红**，按 NEW-RED 规则对 PR 不构成阻断，PR #309 自身的
delta（1 文件、+20/−6 测试加固）全绿。main `4a08a6d` 的 `tests` workflow 在记录时
仍为 queued，故不对 main 最终结论作断言。

本单**没有**去修改 main 侧这 5 类既有缺陷（属别的 PR 范围，且改动无关文件违反约束）；
如实标注为未完成/未覆盖项。
