# PR #204 (`agent/m3362`) 三方合并治愈 — 第 2 轮复验与入库证据

**任务**：m4031【治愈·TensorDEM#204】  |  **仓库**：`TensorDEM`
**结论**：`agent/m3362` 已含 main 全部历史，PR 由 `dirty` 转为可合并；冲突仅 1 个文件，语义并集解冲，零静默丢失，测试不低于 main 基线。

## 1. SHA 谱系

| 角色 | SHA | 说明 |
|---|---|---|
| main 基线 | `6c6f98b` | 合并目标（merge-base 之外 main 前进了 6 个 commit） |
| PR 旧头 | `3fcc6e5` | 本轮推送前的 `agent/m3362`（对 main `mergeable=false`/`dirty`） |
| 合并提交 | `9c959bf` | `merge main into agent/m3362`，父提交 = PR 旧头 + main 基线 |
| 本轮证据头 | 见 `git log -1 -- docs/pr204_round2_verification.md` 的完整 SHA | 第 2 轮复验证据提交，父提交 = 合并提交 `9c959bf`（不改合并树，只加证据） |

merge-base = `f5de971c`（PR 分支唯一提交的起点）。合并后 `git merge-base --is-ancestor main agent/m3362` → 真，这正是 pr-gate 判 `mergeable=true` 的充要条件。

## 2. 冲突文件清单与逐文件解法

两侧同时改到 **2 个文件**，其中 **1 个真冲突**：

| 文件 | main 侧改动 | `agent/m3362` 侧改动 | 解法 |
|---|---|---|---|
| `scripts/verify_wake_lbm_analytic.py` | `_manual_pi_norm` 离对角权重硬编码 `(1.0 if a == b else 2.0)`；闭包代数补 `/rho` 归一 | 同函数重写为 `off_diagonal_weight` 关键字参数（默认 2.0）、`_tau_from_pi` 抽取、legacy 约定上报 | **取分支参数化实现**（main 的 2.0 = 默认值，严格超集），并把 main 的 m2949 约定注释并入 |
| `docs/wake_lbm_analytic_verification.md` | 耦合模块路径更正补 `src/tensordem/` 前缀 | 追加 §6（C5b 解析锚点与约定修复） | 自动合并，两侧行均在位（见 §3） |

合并树相对 main 共 16 个新增/修改文件：PR 侧的 3 个文件（2 张 PNG + `wake_lbm_analytic.json`）整份保留，main 侧文件整份取入，其余为本单审计产物。

**为何不是机械 overlay**：机械 overlay（整文件取一侧）的失败模式是「不报冲突但丢内容」。本轮对两侧自 merge-base 起的**每一次新增行**逐行判定：

| 文件 | 侧 | 新增行 | 逐字在位 | 显式取代（有记录） | 未解释缺失 |
|---|---|---|---|---|---|
| `scripts/verify_wake_lbm_analytic.py` | main | 5 | 2 | 3 | **0** |
| `scripts/verify_wake_lbm_analytic.py` | branch | 201 | 201 | 0 | **0** |
| `docs/wake_lbm_analytic_verification.md` | main | 1 | 1 | 0 | **0** |
| `docs/wake_lbm_analytic_verification.md` | branch | 58 | 58 | 0 | **0** |

「显式取代」指 main 的硬编码权重行被分支的等义参数化行替换，其等价性由 §4.1 的数值执行证明，不是文字比对。

## 3. 不改冲突的路径也做了丢失审计

| 检查 | 结果 |
|---|---|
| 全树残留冲突标记 | 无 |
| 合并文件重复顶层定义（overlay 典型病灶） | 无 |
| main 单侧改动文件与 main 逐字节一致 | 全部一致（`git diff main merge -- <file>` 为空） |
| PR 单侧改动文件与 PR 头逐字节一致 | 全部一致 |
| `docs/doc_evidence_ledger.json` 条目 | main 521 → 合并 538，**删除 0**，新增 17 |

ledger 是唯一「main 单侧改过但合并树与其不同」的文件：差异全部来自**新增**证据条目（新增文档 + 本轮审计文档），无任何条目前被删改。

## 4. 测试证据（本轮独立重跑，非引用旧日志）

### 4.1 语义等价性（数值执行，非文字比对）

| 比较对象 | 判据 | 结果 |
|---|---|---|
| main 硬编码权重 vs 合并版默认参 | 24 组确定性非平衡态上的 `|Π|` 最大绝对差 | `0.0e+00` |
| 分支版 vs 合并版 | 同上 | `0.0e+00` |
| main 内联闭包式 vs `_tau_from_pi` | 输入扫描上的 `|Δτ|` 最大相对差 | `0.0e+00` |

### 4.2 测试套件（`python3 -m pytest tests -q`，两棵树同一环境）

| 套件 | main 基线 `6c6f98b` | 合并树 `9c959bf` | 判据 |
|---|---|---|---|
| 全量 `tests/` | 1232 passed / 2 skipped / **0 failed** | 1235 passed / 2 skipped / **0 failed** | 合并失败数不高于基线，通过数不低于基线 → **满足** |
| `tests/test_wake_lbm_analytic_verification.py`（PR 标的） | 7 passed | 10 passed | 分支新增 3 个测试，全部通过 → **满足** |

全量耗时约 8.5 min/树（`runs/pr204/full_*_r4031.log` 原始日志入库）。

## 5. 复现命令

```bash
# 合并结果侧
python3 -m pytest tests -q > runs/pr204/full_merged_r4031.log
python3 -m pytest tests/test_wake_lbm_analytic_verification.py -q \
    > runs/pr204/wake_merged_r4031.log
# main 基线侧（worktree @ 6c6f98b）
git worktree add /tmp/base204 6c6f98b && cd /tmp/base204 && python3 -m pytest tests -q \
    > runs/pr204/full_baseline_r4031.log
# 机器判据（本文件与图由它生成）
python3 scripts/verify_pr204_merge_semantics.py --mission 4031 \
    --merged-log runs/pr204/wake_merged_r4031.log --baseline-log runs/pr204/wake_baseline_r4031.log \
    --merged-full-log runs/pr204/full_merged_r4031.log \
    --baseline-full-log runs/pr204/full_baseline_r4031.log \
    --out artifacts/pr204/verify_r4031.json --figure artifacts/pr204/fig_r4031_evidence.png
```

## 6. 注意点与局限

- **冲突的物理实质**不是互斥：main 与分支各自把同一个 Frobenius 权重约定引入了一次（main 硬编码 2.0，分支提成参数并保留 weight=1 以暴露旧约定漂移）。因此「并集」=参数化实现 + main 的约定注释，而不是两段代码拼接。
- **测得的不是物理精度**：本单是 VCS 合并治愈，判据是「内容不丢 + 测试不降」。底层物理精度见 PR #204 自身的 `docs/wake_lbm_analytic_verification.md` §6（解析锚点）。
- **合并提交由上一轮（m4013）产出，本轮独立复验**：本轮不重做合并，而是用自己跑的日志重跑全部机器判据（结果全绿），并补齐本轮证据头与 SHA 谱系。
- **不自行合并进 main**：按治理链，`agent/m3362` 推送后由 `merge_governor` 在下一周期按 `mergeable=true` 执行合并；本单不改 main。
- 合并树里 `docs/wake_lbm_analytic_verification.md` 末行换行修复（避免新 §6 与末段拼接）属必要的最小修复，已计入上面的逐行审计。

## 7. 误差判定（逐项，显式回答「是否 <3%」）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 合并版 `|Π|`（默认权重） | main 硬编码实现 | `0.0e+00`（绝对） | **是**（严格 0） |
| 合并版 `τ`（`_tau_from_pi`） | main 内联闭包式 | `0.0e+00`（相对） | **是**（严格 0） |
| 合并树全量测试通过数 | main 基线 1232 | +3（1235 vs 1232），失败 0 vs 0 | **是** |
| PR 内容是否丢失 | PR 头逐字节 | 0 行未解释缺失（branch 侧 201/201 逐字在位） | **是** |

无基准项的声明：本单不产生新的物理量，故不涉及其它参照量；上述四项即全部判据，均无「无基准」情形。
