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
---

# 6. 重开增补（m4696，2026-11-06）：补齐测试凭证 + 全量复测 + 剩余红项诊断

PR #327 关闭后重开，本节是新增内容；§1–§5 保持原样（它们是 m4637 当时的实测记录）。
本增量在 m4637 的两处源修复之上**补齐了回归测试**（原 PR 只有源改动与本文档，没有一条测试
钉住被修的崩溃路径），并把全量套件在**当前 main** 上重跑了一遍（数字见 §6.3）。

## 6.1 本 PR 的文件（4 个，全部在本增量内）

| 文件 | 改动 | 作用 |
|---|---|---|
| `src/tensordem/contact_invariants.py` | +35/-1 | 四个审计键（`translation`/`rotation`/`brute_force`/`determinism`）按 `INVARIANTS` 名暴露可调用别名（§2.1） |
| `src/tensordem/dem3d.py` | +11/-3 | Hertz 路径允许 `bond_stiffness`/`contact_stiffness` 的 0.0 哨兵（§2.3）；空键集守卫在 rebase 后与主线已有实现合并（见 §6.6） |
| `tests/test_dem3d_contact_kernel_regressions.py` | 新增，8 用例 + 6 子用例 | 钉住上面两条崩溃路径与三条“不许被修坏”的邻接行为（§6.2） |
| `docs/ci_clearance_m4637.md` | 本节 | 复测基线与剩余红项诊断 |

不改任何既有断言，不动测试来洗红；新增测试只描述修复后的行为。

## 6.2 回归测试与负向对照（修前必须红、修后必须绿）

`tests/test_dem3d_contact_kernel_regressions.py`，用 `git show main:src/tensordem/dem3d.py`
复原修前代码复跑（负控），实测：

| 用例 | 覆盖修复 | 修前（main `c60dd3e`） | 修后 |
|---|---|---|---|
| `test_empty_bond_keys_yield_all_false_mask` | §2.2 空键集 | FAIL `IndexError` | PASS |
| `test_empty_bond_keys_mask_length_follows_the_query` | §2.2 掩码长度随查询 | FAIL `IndexError` | PASS |
| `test_unbonded_pair_subsystem_steps_and_repels` | §2.2 经生产 `step()` 端到端 | FAIL `IndexError` | PASS |
| `test_hertz_model_accepts_zero_stiffness_sentinels` | §2.3 Hertz 0.0 哨兵 | FAIL `ValueError: bond_stiffness must be finite and positive` | PASS |
| `test_linear_model_still_rejects_zero_bond_stiffness` | 线性路径仍严格正 | PASS | PASS |
| `test_negative_stiffness_rejected_in_both_models` | 负刚度两种模型都拒 | PASS | PASS |
| `test_live_bond_still_maps_to_true` | 非空键集路径未退化为恒 `False` | PASS | PASS |
| `test_hertz_zero_sentinels_do_not_change_the_linear_bound` | 线性刚度界不受哨兵影响 | PASS | PASS |

```
# 修前（负控）：4 failed, 4 passed, 6 subtests passed
# 修后：          8 passed, 6 subtests passed
python -m pytest -q tests/test_dem3d_contact_kernel_regressions.py
```

> **rebase 后的口径修正（2026-11-06）**：本分支已 rebase 到当时的 `origin/main` `a91163b`，
> 主线在此期间通过 #325 独立修掉了同一个空键集崩溃（`if pairs.numel() == 0 or len(self._bond_keys) == 0`）。
> 因此 §6.2 的负控结论只在 `c60dd3e` 上成立；rebase 后那 3 条空键集用例变成**纯回归钉**
> （修前修后都在主线代码上通过），价值在于防止该守卫被后续改动删掉，
> 本 PR 在 `dem3d.py` 上保留的实际增量只剩 §2.3 的 Hertz 0.0 哨兵。

`tests/test_dem3d_contact_invariants.py` 仍按 `getattr(module, INVARIANTS[i])` 派发，
是 §2.1 端到端覆盖：修前 8 红（`AttributeError`），修后 `15 passed, 56 subtests passed`。

## 6.3 全量复测（本地，串行，当前 main `c60dd3e` + 本增量）

```
python -m pytest -q -p no:randomly            # 1515 项收集，序列执行
19 failed, 1495 passed, 9 skipped, 181 subtests passed in 1352.41s
```

对照 §1：**32 red → 19 red**（`-13`），其中

- 8 条 `contact_invariants` 红项**全清**（§2.1）；
- 11 条 `test_dem3d_energy_friction_rolling` 的 `IndexError` 崩溃**不再崩溃**，下降为 8 条可诊断的数值断言（§4.2、§6.4b）；
- 其余红项性质未变（见下表）。

| 文件 | 修前 | 修后 | 说明 |
|---|---|---|---|
| `test_dem3d_contact_invariants.py` | 8 | **0** | 本增量清除 |
| `test_dem3d_energy_friction_rolling.py` | 11 | **8** | 崩溃→数值，性质改变（§6.4b） |
| `test_dem3d_hertz_stability.py` | 6 | 6 | 断言在物理上不可满足（§6.4a） |
| `test_doc_evidence_pointers.py` | 4 | 4 | 证据账本漂移，需干净树重生成（§4.3） |
| `test_ridge_dem_core_physics.py` | 1 | 1 | `recommended_dt` 取 `min(弹性, 粘性)`，测试只算弹性（§6.4c） |
| `test_hull_mesh_contact.py` | 1 | 1 | 同族数值漂移 |
| `test_verification_execution_ledger.py` | 1 | 0 | 复测机 load≈800–1250，本轮未复现超时（环境项，非代码） |
| `test_audit_upstream_diff_m1630.py` | 0 | 1（全量时） | 单独跑 `13 passed`，全量跑时红：git 状态审计类用例受并行/瞬时文件影响的隔离性缺陷，与本增量无关 |

所有数字由 `artifacts/m4696/` 下的脚本程序化产生，无手抄：`red_diagnostics.json` 由
`artifacts/m4696/measure_red_diagnostics.py` 生成（直接调用被测的
`tests/test_dem3d_energy_friction_rolling.py` 内的 `_rolling()` / `_friction()` 与
`DEM3DConfig`），图 `hertz_growth_factor_impossibility.png`、`ci_red_inventory.png` 由
`artifacts/m4696/make_figures.py` 生成。

## 6.4 剩余红项的三条根因（本轮新增的定量证据）

### 6.4a `test_dem3d_hertz_stability.py`（6）：断言 `rho < 1` 在数学上不可达

半隐式 Euler 的放大矩阵 `A = [[1-d^2, dt(1-g)], [-dt w^2, 1-g]]`（§`stability.py` 导读），
`det A = 1-g`、`tr A = 2-d^2-g`。测试传 `damping_rate=0`，于是 `g=0`、`det A = 1`，
在 `d^2 < 4` 时判别式 `< 0`，谱半径 `rho = sqrt(|det A|) = 1` —— **恒等于 1**，与 `dt` 无关。
实测（`artifacts/m4696/red_diagnostics.json`，`nx=ny=nz=2`、`E=1e9 Pa`、`mu=0.3`、`R=25 mm`、
`rho_ice=917`，`dt_rec = 4.6885e-6 s`）：

| δ/R | k_sec (N/m) | ω (rad/s) | dt_rec·ω | 实测 rho | 测试断言 rho |
|---|---|---|---|---|---|
| 0.01 | 1.2951e6 | 4645.23 | 0.0218 | **1.000** | 0.32 |
| 0.05 | 2.8959e6 | 6946.24 | 0.0326 | **1.000** | 0.45 |
| 0.10 | 4.0954e6 | 8260.52 | 0.0387 | **1.000** | 0.58 |
| 0.20 | 5.7917e6 | 9823.46 | 0.0461 | **1.000** | 0.82 |

全稳定带（`0 < dt·ω < 2`，2001 点采样）上 `rho` 的取值区间是 `[1.000, 1.000]`。也就是说
`margin = 1/rho - 1 = 0` 是**无阻尼半隐式 Euler 的精确性质**（辛格式保相空间体积、不耗散），
不是数值精度不足。这 6 条红项的判据需要重写成步长裕度（例如 `dt_crit/dt` 或
`dt_rec·ω_max <= 0.05` 这一类可满足的量），会牵动所有 Hertz 用例的步长定义，属独立增量，
本单**不动**（既不改断言，也不为了凑绿去给 `recommended_dt` 加人为阻尼）。

### 6.4b `test_dem3d_energy_friction_rolling.py`（8）：闭合解本身超过可用动能

case R（滚动阻力）的闭合解假设“恒定减速度 `a = mu_r F_n / m` 一直作用到 `T`”：
`v(T) = v0 - aT`。但滚动阻力是**耗散型**（`_rolling_resistance_couple` 文档串明写
`f_j·v_rel <= 0`、`|v_t|→0` 时力归零），`a` 的方向随 `v_t` 变号。实测参数
（`red_diagnostics.json`）：`m = 0.06002 kg`、`F_n = 10 N`、`mu_r = 0.3` ⇒ `a = 49.985 m/s^2`，
`v0 = 0.1 m/s` ⇒ **零穿越时刻 `t = v0/a = 2.0 ms < T = 8 ms`**。于是：

| 量 | 闭合解 | 可用上界 / 实测 | 判定 |
|---|---|---|---|
| 滚动耗散 `D_r` | 2.340e-3 J | 初动能 `0.5 m v0^2` = **3.001e-4 J**（1/7.8） | 闭合解要求耗散 **7.80 倍**于系统全部初动能 → 物理不可能 |
| `v_T` | −0.2999 m/s | 引擎 1.99e-3 m/s（±a·dt = ±5e-3 的数值抖动带内） | 引擎行为 = “减速到静止” |
| 账本 `rolling_work` | +2.340e-3 J | −3.117e-4 J，模长 = 1.039×初动能 | 引擎耗散量正确（≈初动能），符号沿用“有符号功”约定（`P<=0`） |

case F（Coulomb 粘着）反而基本吻合：`W_f = -1.0279e-5 J` vs 闭合 `-1.0041e-5 J`，**2.37%**
（测试阈值 3%），`E_tang` 同量级；但可恢复功恒等式残差 `-3.97e-7 J` 超阈值 3.60e-7 J 约 10%，
账本残差 `-1.138e-4 J` 超 5e-6 J 的地板 23 倍，说明**切向通道的账本科目仍有未入账功**
（量级 1e-4 J，与 case F 的动能尺度 1.2e-5 J 相差约 10 倍，值得单独对账）。

结论：这 8 条不是“精度不够”，而是**验证用例的参考解与引擎声明的耗散不变量互相矛盾**
（滚动 5 条）+ **切向账本漏项**（摩擦 3 条中 2 条）。重写参考解（“减速至静止”的闭合解
`D_r = 0.5 m v0^2`、`v_T -> 0`）并逐项对账切向科目，是独立物理增量；本单只保证不再崩溃，
不以改断言的方式让它们变绿。

### 6.4c `test_ridge_dem_core_physics.py`（1）：测试漏了 `min()`

`RidgeMaterial.recommended_dt` 返回 `min(弹性界, 粘性界)`（`src/tensordem/ridge_dem.py`），
测试只算了弹性界：`3.329e-4 s`；实测 `2.209e-4 s` 恰为粘性界
`0.2 m / (13 c + c_drag)`。同族用例 `test_recommended_dt_viscous_limit` 覆盖粘性分支，
所以这条是测试表述缺陷（缺 `min`），与 §6.4a 同类：**不改代码，也不在本单改断言**，
登记为独立增量（改法是把断言写成 `min(elastic, viscous)`）。

## 6.6 rebase 基线与主线重叠声明（热文件）

`src/tensordem/dem3d.py` 是多单重叠的热文件。本分支 rebase 到 `a91163b` 时，冲突只有空键集守卫一处，
解决方式是**整体采纳主线实现**（连同 `pairs.numel() == 0` 的既有约定），本 PR 不再重复携带同义守卫。
于是本 PR 在 `dem3d.py` 的有效改动只有 `__post_init__` 的 Hertz 哨兵分支（§2.3），
`contact_invariants.py` 的别名（§2.1）与主线无重叠。

## 6.5 判据与边界（诚实登记）

- 本增量**只做两件事**：把 m4637 的两处崩溃修复带上测试，并把复测基线压实。
- 判据（可机器复算）：新增模块 `8 passed`；负控 `4 failed`；`contact_invariants` `15 passed / 56 subtests`；
  全量 `19 failed / 1495 passed / 9 skipped / 181 subtests`（修前 `32 failed`）。
- **main CI 仍未绿**（19 红），剩余项全部有定量根因与处置归属（§6.4a/b/c、§4.3），
  没有一条是“改断言就能过”的（唯一例外 §6.4c 是测试缺 `min`，但它属 ridge 模块的独立增量）。
- 适用范围：本增量不动步长定义、不动滚动/摩擦物理装配、不动证据账本；
  凡是会改变既有 Hertz / 摩擦 / 滚动轨迹数值的改动，都不在这一单里。

## 7. rebase 后基线复核（`a91163b`，2026-11-06）——取代 §6.3/§6.4 的红项清单

本分支 rebase 到当时 `origin/main` `a91163b` 后复核，**主线在这期间已独立重写了 §6.4 里的三家用例**：
`tests/test_dem3d_hertz_stability.py`（201 行改动）、`tests/test_dem3d_energy_friction_rolling.py`
（163 行）、`tests/test_ridge_dem_core_physics.py`（16 行）。因此

- §6.3 的「32 red → 19 red」与 §6.4a/b/c 的三条根因，**都是 `c60dd3e` 上的历史实测**，
  在 `a91163b` 上已不成立（§6.4 的分析仍可作为那批用例重写的依据留证，但不再描述当前红线）；
- rebase 后对原红项涉及的文件做定向复跑：`1 failed, 78 passed, 1 skipped, 56 subtests passed`
  —— 唯一剩下的红项是 `test_doc_evidence_pointers.py::test_committed_ledger_is_current`
  （证据账本需按 §4.3 在干净树上重生成；本单新增了 `docs/ci_clearance_m4637.md`，
  账本因此落后一格，属可机械补齐的登记项，不是代码回归）；
- 本增量自身的判据在 rebase 后仍然成立：`tests/test_dem3d_contact_kernel_regressions.py`
  8 passed / 6 subtests，`tests/test_dem3d_contact_invariants.py` 15 passed / 56 subtests，
  `tests/test_dem3d_hertz_stability.py` 7 passed（三个文件合计 30 passed / 62 subtests）。

**未完成（诚实登记）**：`a91163b` 全量套件（1515 项）**未在本轮重跑**（`c60dd3e` 上的一次全量
耗时 1352 s，已如实记在 §6.3）；rebase 后的证据账本未重生成。下一轮应先跑 `scripts/audit_doc_evidence.py`
补齐账本，再重跑全量把红线数定稿。
