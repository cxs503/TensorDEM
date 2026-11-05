# PR #204 (branch `agent/m3362`) — 三方合并与逐文件解冲（含静默丢失审计）

**日期**：2026-11-03（治愈单 m4013，第 1 轮）
**仓库**：`TensorDEM`
**分支旧 SHA**：`3fcc6e58770726ee21f1bea916cdfcd5a40f61f8`（`agent/m3362`，合并前）
**main 基线 SHA**：`6c6f98b7ac8fbd889db459e06329901869b89dbb`
**merge-base**：`f5de971c03f6363afa701063a833a3766c874eca`

> 本文件的全部数字由 `scripts/verify_pr204_merge_semantics.py` 生成后程序化注入
> （`python3 scripts/verify_pr204_merge_semantics.py --doc docs/pr204_merge_conflict_resolution.md`），
> 机器判据 JSON 在 `artifacts/pr204/verify_pr204.json`，图在 `fig_pr204_norm_convention.png`。

## 1. 为什么会脏

`agent/m3362` 自 merge-base 起只有 1 个 commit（`m3362: matured mission deliverables`，6 文件
+483/−25），而 `main` 同时前进了 6 个 commit（20 文件 +2553/−413：PR #166 的 fracture_gen
语义并集、m3317 的 ice_floe 恢复等）。两侧**同时改到的文件只有两个**：

| 文件 | main 侧 | `agent/m3362` 侧 | 冲突 |
|---|---|---|---|
| `scripts/verify_wake_lbm_analytic.py` | 修 `_manual_pi_norm` 的 Frobenius 权重（离对角权重 2）＋闭包代数补 `/rho` | 重写同函数：`off_diagonal_weight` 参数化（默认 2.0）＋C5b 解析锚点＋legacy 约定上报 | **内容冲突（UU）** |
| `docs/wake_lbm_analytic_verification.md` | 路径更正：耦合模块引用补上 `src/tensordem/` 前缀 | 追加 §6（C5b 锚点与约定修复） | 自动合并（需复核） |

其余 19 个 main 侧文件（`fracture_gen.py`、`ice_floe_physics.py`、`doc_evidence_ledger.json` 等）
与 3 个 3362 侧文件（两张 PNG、`wake_lbm_analytic.json`、tests）互不重叠，合并时整份取入。

**冲突的实质**：不是互斥改动，而是**同一处物理约定的两次引入**——main 把权重硬编码成
`(1.0 if a == b else 2.0)`，3362 把同一个权重提成参数（默认 2.0）并保留 weight=1 以便
把旧约定的漂移**显式报出来**。因此语义并集 = 3362 的参数化实现（main 的 2.0 即其默认值），
且必须**数值证明**默认值与 main 逐位相同（见 §3）。

## 2. 逐文件解法

### 2.1 `scripts/verify_wake_lbm_analytic.py`（唯一真冲突）

冲突块共 2 处，逐处取用如下（均为语义并集，无一侧内容被丢弃）：

| 冲突点 | main | `agent/m3362` | 取用 | 理由 |
|---|---|---|---|---|
| `_manual_pi_norm` 的范数求和 | `total += (1.0 if a == b else 2.0) * sab * sab` | `total += sab * sab * (off_diagonal_weight if a != b else 1.0)`，默认 2.0 | **3362 的参数化版**，并把 main 的 m2949 约定说明（`WakeLBM._effective_tau` / `_outer_norm_weight` 指针）并入注释 | 默认值 2.0 与 main 逐位相同（§3 A 项 `max|Δ|=0.0`）；参数化是严格超集，额外提供 legacy 报告能力 |
| 闭包代数 | 内联 `tau = 0.5*(tau0+sqrt(tau0²+18√2 Cs² · \|Π\|/rho))` 并新增 `rho_node` 取值 | 抽出 `_tau_from_pi(tau0, cs_sgs, pi_norm, rho)`＋保留 main 的 `rho_node` 变量，另加 `pi_legacy`/`legacy_*` 上报 | **保留 main 的 `rho_node` 提升与两行代数注释**，调用改为 `_tau_from_pi(...)` | `_tau_from_pi` 与 main 内联式逐位等价（§3 B 项 `max|Δ|=0.0`）；3362 的 legacy 上报需要 `pi_legacy`，不能删 |

**复核自动合并的文件**（机械 overlay 的典型失败点是「合并不报冲突但丢内容」）：

- `docs/wake_lbm_analytic_verification.md`：`git diff origin/main -- <doc>` 只剩 3362 的
  §6/测试计数改动；`git diff origin/agent/m3362 -- <doc>` 只剩 main 的路径更正——两侧内容
  同时在位（§4 审计：main 1/1 行、branch 58/58 行 verbatim 在位）。
- 顺带修掉 3362 侧文件末「No newline at end of file」，否则新 §6 与末段会被拼成一行。

### 2.2 无冲突但被整份取入的文件

`fracture_gen.py`、`ice_floe_physics.py`、`doc_evidence_ledger.json`＋PR #166 的 docs/scripts/tests
全部来自 main；`wake_lbm_analytic.json`＋两张 PNG＋`tests/test_wake_lbm_analytic_verification.py`
全部来自 3362。两类互不覆盖，合并后 `git diff origin/main --stat` 只显示 3362 的增量。

## 3. 数值等价证明（不是文字比对）

两个实现都是从 git blob 里用 `ast` 抠出函数体后**直接执行**，不靠抄写：

**A. `_manual_pi_norm`：main 硬编码权重 vs 合并版默认参数**（24 组确定性伪随机非平衡态，
`random.Random(20261103)`）

| 比较 | 结果 |
|---|---|
| main 版 vs 合并版（默认 `off_diagonal_weight=2.0`）最大绝对差 | `0.0e+00` |
| 3362 版 vs 合并版最大绝对差 | `0.0e+00` |

**B. 闭包代数：main 内联式 vs 合并版 `_tau_from_pi`**

| tau0 | Cs | \|Π\| | rho | \|Δtau\| | 相对差 |
|---|---|---|---|---|---|
| 0.5006 | 0.160 | 9.747373e-04 | 1.000000e+00 | 0.0e+00 | 0.0e+00 |
| 0.5006 | 0.160 | 9.748657e-03 | 9.993000e-01 | 0.0e+00 | 0.0e+00 |
| 0.5050 | 0.200 | 2.637731e-02 | 1.000700e+00 | 0.0e+00 | 0.0e+00 |
| 0.5013 | 0.100 | 1.000000e-06 | 1.000000e+00 | 0.0e+00 | 0.0e+00 |

**C. 两种范数约定差 √2（纯剪切解析构造 `neq_i = 18σ w_i c_i0 c_i1`，即 Π_01=Π_10=2σ）**

| σ | \|Π\| weight=1 | \|Π\| weight=2 | 解析 2σ | 比值 | 相对 √2 偏差 |
|---|---|---|---|---|---|
| 1e-04 | 2.000000000e-04 | 2.828427125e-04 | 2.000000000e-04 | 1.414213562 | 1.6e-16 |
| 1e-03 | 2.000000000e-03 | 2.828427125e-03 | 2.000000000e-03 | 1.414213562 | 0.0e+00 |
| 1e-02 | 2.000000000e-02 | 2.828427125e-02 | 2.000000000e-02 | 1.414213562 | 0.0e+00 |

=> 离对角权重 2 就是 main 修完之后的 shipped 约定；权重 1（3362 保留的 legacy 通道）
在纯剪切下确实低估 √2 倍，与 `docs/wake_lbm_analytic_verification.md` §6 的
`legacy_pi_ratio = 1.414212` 一致。

## 4. 静默丢失审计（逐行）

对两侧自 merge-base 起**新增**的非注释行，逐条在合并结果里查找；找不到的必须命中
`SUPERSEDED` 白名单（写明替代构造），否则判 FAIL。

| 文件 | 侧 | 新增行 | verbatim 在位 | 显式被替代 | 未解释缺失 |
|---|---|---|---|---|---|
| `scripts/verify_wake_lbm_analytic.py` | main | 5 | 2 | 3 | 0 |
| `scripts/verify_wake_lbm_analytic.py` | branch | 201 | 201 | 0 | 0 |
| `docs/wake_lbm_analytic_verification.md` | main | 1 | 1 | 0 | 0 |
| `docs/wake_lbm_analytic_verification.md` | branch | 58 | 58 | 0 | 0 |
| `docs/artifacts/wake_lbm_analytic/wake_lbm_analytic.json` | main | 0 | 0 | 0 | 0 |
| `docs/artifacts/wake_lbm_analytic/wake_lbm_analytic.json` | branch | 135 | 135 | 0 | 0 |

未解释缺失合计：**0**。

## 5. 测试证据

**5.1 受影响模块用例集**（`tests/test_wake_lbm_analytic_verification.py`，即 PR #204 改动的模块）

| 树 | passed | failed |
|---|---|---|
| 合并树（`agent/m3362` + main） | 10 | 0 |
| main 基线（`origin/main` 独立 worktree） | 7 | 0 |

合并树比 main 基线**多 3 个通过**（3362 新增的 C5b 用例），无失败、无回归。
原始日志：`runs/pr204/wake_merged.log`、`runs/pr204/wake_baseline.log`。

**5.2 全库回归**（`pytest tests`，同一台机器、同一环境）

| 树 | passed | failed | skipped |
|---|---|---|---|
| 合并树（首轮） | 1234 | 1（1 个＝本条欠账，同提交内修复并定点复跑为绿） | 2 |
| main 基线 | 1232 | 0 | 2 |

原始日志：`runs/pr204/full_merged.log`、`runs/pr204/full_baseline.log`。

> 逐项判定：合并树**有效失败数 = 0**（原始失败
> 1 减去已修复并定点复跑为绿的 1 个），main 基线失败数 =
> 0；合并树 passed 亦不低于基线。
>
> 全库首轮（合并后、未刷新证据台账时）在 `tests/test_doc_evidence_pointers.py::
> test_committed_ledger_is_current` 上 1 failed：3362 的 §6 新增引用
> `docs/artifacts/wake_lbm_analytic/fig4_sgs_strain_anchor.png`，但 3362 没有同步刷新
> `docs/doc_evidence_ledger.json`（该台账是「指针棘轮」，新引用必须入账）。**这不是合并冲突
> 造成的**，而是分支自带的欠账；已用 `scripts/audit_doc_evidence.py` 刷新台账并定点复跑该
> 用例集：`runs/pr204/doc_evidence_after_fix.log`（{'passed': 14, 'warning': 1}，全绿）。
> 整库未在本机二次重跑：机器被其他任务的 `pytest -n 24` 抢占，单次整库约 10 分钟，故合并树整库
> 数字取首轮实测、欠账项以定点复跑补齐——如实记录，不粉饰。

## 6. 机器判据汇总

| 判据 | 结论 |
|---|---|
| 冲突标记零残留 | PASS |
| 顶层定义无重复（overlay 典型事故） | PASS |
| 两侧语义并集（数值等价） | PASS |
| 逐行内容保全 | PASS |
| 测试不低于 main 基线（受影响模块） | PASS |
| 全库回归不低于 main 基线 | PASS |

## 7. 误差判定（是否 <3%）

- 关键量 1：`_manual_pi_norm` 默认构造 vs main 实现的相对偏差 = **0.0 %**（逐位相同）→ **<3% ✔**
- 关键量 2：`_tau_from_pi` vs main 内联闭包式的相对偏差 = **0.0 %** → **<3% ✔**
- 关键量 3：weight=2 / weight=1 比值 vs √2 的偏差 ≤ 1.6e-16 相对量 → **<3% ✔**
- 测试层面：受影响模块合并树 10 passed / 0 failed ≥ main 基线 7 passed / 0 failed → **无回归 ✔**
- 全库回归：合并树 1234 passed / 1 failed vs main 基线 1232 passed / 0 failed → **无回归 ✔**

以上三项均是「实现等价性」判据（应当精确相等），不是物理精度判据；本单不含新的物理算例，
因此没有对实验/文献值的误差可比。

## 8. 注意点 / 局限

1. **禁止机械 overlay**：本次两侧改的是同一个函数的同一行表达式，`git checkout --ours/--theirs`
   任一方向都会静默丢掉另一侧的物理修复（main 的权重 2 或 3362 的 legacy 上报）。所以必须
   参数化取并集，并用**数值执行**证明等价，而不是看 diff 眼判。
2. **文档自动合并 ≠ 无风险**：`docs/wake_lbm_analytic_verification.md` 没报冲突，但若不复核，
   main 的路径更正会以「无冲突」名义被 3362 的整节追加掩盖掉。审计脚本对 docs 也做了逐行保全。
3. **`report.json` 是 main 侧产物**：合并把 main 的 `report.json` 带入，工作树的同名文件被覆盖，
   这与代码无关，不参与判据。
4. 本审计只覆盖**两侧共同改写**的文件；单侧改动的文件由 git 整份取入，不构成静默丢失风险面。
5. `_manual_pi_norm` 的 `nx, ny, nz` 形参在函数体内未使用（历史签名），本次保持原样以免
   扩大改动面。
