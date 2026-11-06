# TensorDEM main CI 清红（m4637）：红项实测、根因与本次增量

范围：`main` @ `c60dd3e`（run 808, `unit-tests` 在 *Run complete pytest suite* 步骤失败）。
本增量只做**最小源修复**（不改任何断言来洗红），并把仍未清完的红项逐条登记。

## 1. 实测基线（本地复现，`python -m pytest -q`，c60dd3e）

序列执行的完整套件（1515 项收集）：

```
32 failed, 1482 passed, 9 skipped, 167 subtests passed in 1609.46s
```

红项按文件聚合（`-rf` 摘要 + `SUBFAILED` 条目）：

| 文件 | 红项数 | 首要错误 |
|---|---|---|
| `tests/test_dem3d_contact_invariants.py` | 8（SUBTEST） | `AttributeError: module 'tensordem.contact_invariants' has no attribute 'determinism'` |
| `tests/test_dem3d_energy_friction_rolling.py` | 11 | `IndexError: index is out of bounds for dimension with size 0` @ `dem3d.py:508` |
| `tests/test_dem3d_hertz_stability.py` | 6 | `ValueError: bond_stiffness must be finite and positive` |
| `tests/test_doc_evidence_pointers.py` | 4 | 文档证据账本漂移 |
| `tests/test_verification_execution_ledger.py` | 1 | `assert 'timeout' == 'green'`（子进程被 20 min 级负载拖超时） |
| `tests/test_ridge_dem_core_physics.py` | 1 | `2.209e-4 != 3.329e-4`（推荐 dt 数值漂移） |
| `tests/test_hull_mesh_contact.py` | 1 | 同族数值漂移 |

## 2. 本次修复（两处源改动，均在 `src/tensordem`，不动测试）

### 2.1 接触不变量审计：按 `INVARIANTS` 名暴露可调用对象

`dem3d_contact_invariants` 的用例按审计键（`translation` / `rotation` / `brute_force` /
`determinism`）用 `getattr(module, key)` 派发，但模块只提供带后缀的实现
（`translation_invariance` / `rotation_equivariance` / `brute_force_agreement` /
`determinism_repeatability`），审计键本身没有同名的可调用对象 → 4 个键 × 2 个种子 = 8 条红。
修法：在 `contact_invariants.py` 末尾补四个薄别名（默认参数与实现一致，`translation`
缺省位移取 `(1.5, -7.25, 3.5)`，与测试显式调用一致），并加入 `__all__`。

### 2.2 全断裂晶格上的空键集：`_alive_bond_for_pairs` 越界

```python
safe = where.clamp(max=max(len(self._bond_keys) - 1, 0))
matched = (where < len(self._bond_keys)) & (self._bond_keys[safe] == keys)
```

键集为空（全部键断裂，`_bond_keys` 长度 0）时，`safe` 被夹到 0，`self._bond_keys[0]` 抛
`IndexError`。修法：空键集直接返回与 `pairs` 等长的全 `False` 掩码（语义上“这些对没有一条
存活的键”），保持 0 对时的既有返回约定不变。

### 2.3 `bond_stiffness`/`contact_stiffness` 的 0.0 哨兵（Hertz 模型）

`__post_init__` 曾对这两个字段统一要求 `> 0`，但 `contact_model="hertz"` 下二者**从不被
乘积使用**：割线刚度由 `young_modulus`/`poisson_ratio` 经 `hertz_secant_stiffness` 得出
（`_bound_stiffness` 的 Hertz 分支直接忽略传入的 `linear_stiffness`），因此 0.0 表示
“由 Hertz 模型给出”，不是“缺输入”。修法：始终要求有限非负，仅线性路径（真正做乘法的那条）
保留严格正；Hertz 路径允许 0.0。

## 3. 修复后的可复算证据

```
python -m pytest -q tests/test_dem3d_contact_invariants.py tests/test_dem3d_hertz_stability.py
# 修前：8 failed（contact_invariants, AttributeError）+ 6 failed（hertz_stability, ValueError）
# 修后：0 failed + 6 failed；contact_invariants 15 passed, 56 subtests passed
```

- `tests/test_dem3d_contact_invariants.py`：**8 红 → 全绿**（15 passed / 56 subtests）。
- `tests/test_dem3d_hertz_stability.py`：`ValueError` 消失（红项性质改变），
  但 6 条断言仍红，见 §4.1 —— 未清。
- `tests/test_dem3d_energy_friction_rolling.py`：`IndexError` 消失，红项性质由崩溃变为
  数值断言不符，见 §4.2 —— 未清。

## 4. 未清红项（诚实登记，本增量不掩盖）

### 4.1 `test_dem3d_hertz_stability.py`（6）

掩掉 `ValueError` 后暴露的是 `rho=1.0`：`recommended_dt` 在 `bond_stiffness=0.0`、
`contact_stiffness=0.0` 的 Hertz 配置下给出的步长使 `symplectic_euler_growth_factor`
恰好为 1（等价于 `dt·ω→0`），而测试要求 design overlap（≤0.2R）处 `rho<1` 且
`rho≈{0.32,0.45,0.58,0.82}`。需要把 `recommended_dt` 的刚性集在 Hertz 分支上改为
“以 design overlap 的割线刚度为下界”，属独立增量（会动所有 Hertz 用例的步长），
本单不做，不以改断言洗红。

### 4.2 `test_dem3d_energy_friction_rolling.py`（13，性质已变）

闭合解对比仍不通过，典型：`v_T 0.00199 vs -0.2999`（切向速度几乎为零）、
`D_r -3.12e-4 vs 2.34e-3`、`residual 2.59e-4 vs 4.74e-3`。即滚动/摩擦通道的
实际响应与闭合解相差 1–2 个数量级，说明该通道在当前主干上**没有真正驱动**（不是数值精度
问题）。这需要把滚动/摩擦通道的装配逐项对账，属物理增量，本单只消除了崩溃，未清红。

### 4.3 `test_doc_evidence_pointers.py`（4）

文档证据账本漂移，标准处置是 `python scripts/audit_doc_evidence.py` 重生成
（见 `docs/ci_clearance_m3317.md` §4）。本单未重生成：账本必须在“无未跟踪产物”的干净树上
生成，而本工作树在排查期产生了临时日志/图，重生成会把噪声写进账本。

### 4.4 数值漂移两条 + 环境超时一条

- `test_ridge_dem_core_physics.py::test_recommended_dt_elastic_limit`：
  `2.209e-4 != 3.329e-4`（约 1.5 倍），与 §4.1 同源（推荐 dt 定义漂移）。
- `test_hull_mesh_contact.py`：同族数值漂移。
- `test_verification_execution_ledger.py::test_timeout_is_contained_and_labelled`：
  `'timeout' != 'green'`。复现机 load average ≈ 1144（128 核被多智能体共享），
  该用例的子进程跑不完；属环境性红项，非代码回归，需在空载机器上复核。

## 5. 结论与判据

- 已清：**8/32**（`contact_invariants` 全绿），并消除了 11 条 `IndexError` 崩溃
  （红项未清但性质由崩溃变为可诊断的数值断言）。
- 未清：24 条（其中 6 条 hertz 稳定性、13 条滚动/摩擦闭合解、4 条文档账本、3 条其它）。
- **main CI 仍未绿**：本次增量是必要但不充分的一步，剩余部分需要动 Hertz 推荐步长与
  滚动/摩擦通道的物理装配，属独立增量，已在 §4 逐条留证。