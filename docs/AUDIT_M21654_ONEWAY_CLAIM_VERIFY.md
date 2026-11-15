# 对抗式复核：GAP-ICE-ONEWAY-CLAIM（m21654，verifier 席 tp-ver-3）

**复核对象**：`examples/dem_fem_oneway/`（DEM 单颗粒冲击简支梁，单向耦合）在 TensorDEM 树上的
**在册判据改名 + 换基准**工作，交付单 m21584（commit `12506e70`，平台报告 id 19499）。
**复核结论（disposition）**：**approve with notes（有保留通过）+ 2 处已修缺陷**。
交付的 5 项验收在机检下全绿，但本轮查出 **1 个高危「自证式闸门」缺陷**（`validate.py` 有三个数值门
只读 `run.py` 自己写的布尔标志，篡改后的 `results.json` 仍判 PASS）与 **1 处未同步的旧判据声明**
（`run.py` 模块 docstring 仍写着已退役的「peak ratio ≥0.9」门）。两处均已修并加锁测试；
另有 2 条**未消账的诚实偏差**（注册数字 0.908% 与四档链字面值不可复现；1% 偏转门是单向的）。

| 项 | 值 |
|---|---|
| 锚点 | `GAP-ICE-ONEWAY-CLAIM`（feature_ledger，in_progress） |
| 复核单 | m21654（本单，verifier 席） |
| 被复核交付 | m21584 / commit `12506e70` / 报告 id 19499 |
| 本轮改动 | `examples/dem_fem_oneway/validate.py`（数值门改为「数 + 标志双源」，见 F1）、`examples/dem_fem_oneway/run.py`（docstring 第 5 条，见 F2） |
| 新增 | `tools/verify_oneway_claim_gates_m21654.py`（独立复核探针 P0–P5）、`tests/test_dem_fem_oneway_gate_bite_m21654.py`（13 条咬合测试）、`scripts/make_oneway_claim_verify_figure_m21654.py`、本文件、图 `docs/figures/oneway_claim_verify_m21654.png`、探针原始输出 `runs/oneway_claim_verify_m21654/probe.json` |
| 平台报告 | id **19552**（`tools/publish_report_m21654.py`，数字由 `probe.json` 注入） |
| 机裁台账 | `tensor_maturity_verdicts` repo=TensorDEM scope=GAP-ICE-ONEWAY-CLAIM level=3（verifier=m21654-verifier，deviations 已列） |
| 复现 | `PYTHONPATH=src python examples/dem_fem_oneway/run.py` → `validate.py`（10/10 PASS）→ `pytest tests/test_dem_fem_oneway_gate_bite_m21654.py tests/test_dem_fem_oneway_claim_review.py tests/test_dem_fem_independent_gates.py -q`（32 passed） |

---

## 1. 复核范围与独立性限制

**做了什么**

| 探针 | 内容 | 用时 |
|---|---|---|
| P0 | 重跑 DEM+FEM（40 s），把提交的 `results.json` 四个加密档峰值逐位比对 | 20 s |
| P1 | 用**独立代码路径**（直接调 `tensordem.dem_fem_gates.modal_response` + 自写网格插值）重算 4 档模态锚峰值 | 2 s |
| P2 | **自写** Newmark 积分器（线性加速度 β=1/6，另同格式 β=1/4 对照组）重算 FEM 峰值 | 15 s |
| P3 | 闸门咬合/灵敏度：E 缩放 9 档、单元数 4 档、模态阶数 6 档、力幅 3 档 | 20 s |
| P4 | 注册链字面值（0.908/0.102/0.041/0.043 %）可复现性检索：18 个变体 | 15 s |
| P5 | `validate.py` 咬合：11 个篡改版 `results.json`（含「错数 + 对标志」组合） | 2 s |

**独立性限制（如实声明）**

1. DEM 侧**未独立复核**：接触律（`platen_stiffness=2.0e4`）、恢复系数、底板反力仍是 m21584 的
   同一实现，本轮只做确定性复核（P0 rel_diff 0.0），不做物理复核。
2. P2 的自写积分器**共用** `EulerBernoulliBeam` 装配出的 `K_free/M_free`，因此它检验的是
   **时间积分器与自由自由度处理**，不是单元刚度/质量矩阵本身；后者由模态锚（完全独立的连续体
   路径）覆盖。
3. 界面守恒**仍未取得外部证据**：port parity 结构性恒 0（P3 量化确认），work-energy 是自比较。
   本单不改变 m21584 的定级结论，只确认定级没被偷偷抬回「守恒证据」。
4. 未重做双向耦合（本算例未建模，超范围）。

## 2. 分级发现（severity-ranked）

### F1 —【High，已修】`validate.py` 三个数值门是「自证」：只读 `run.py` 写的布尔标志

**证据**（P5，探针前）：`validate.py` 的 1/2/3 号门分别提交
`v["force_history_port_parity_pass"]`、`v["deflection_pass"]`、`v["energy_pass"]` 作为 PASS 判据，
`error_relative` 只当展示值打印，**从不与阈值比较**。篡改 `results.json` 后：

| 篡改 | 修复前 | 修复后 |
|---|---|---|
| `deflection_error_relative = 0.02`（2%，门是 1%） | **PASS**（漏判） | FAIL |
| `deflection_error_relative=0.02` 且 `deflection_pass=True` | **PASS**（漏判） | FAIL |
| `force_history_port_parity_error_relative = 0.5` 且 pass=True | **PASS**（漏判） | FAIL |
| `work_energy_error_relative = 1e-2` 且 energy_pass=True | **PASS**（漏判） | FAIL |
| `deflection_pass=False`（数与标志打架） | FAIL | FAIL |
| 链缩短 / 末档 5e-3 / 去 scope / 接口行退回 impulse / 去 inline ledger | FAIL | FAIL |

**为什么是高危**：这正是锚点标题要禁的形态——「用同义反复充抵证据」。闸门此时等价于「相信生成器
自报通过」，任何手改/回退的 `results.json` 都能拿到 10/10 PASS，而验收的机裁口径恰恰是
「验脚本全绿」。修复前 `validate.py` 在「偏转」这条主判据上**没有咬合力**。

**修复**（`examples/dem_fem_oneway/validate.py`）：新增 `_numeric_gate(error, declared, tol)`——
① 实测误差必须 ≤ 阈值，② 生成器布尔必须为 `True` 与数一致，**两者任一不满足即 FAIL**；
阈值集中为 `PARITY_TOL=1e-10 / DEFLECTION_TOL=1e-2 / ENERGY_TOL=1e-8`；dt 链门补上「细档必须低于
粗档」的重算（原来只信 `dt_convergence_pass`）。修复后：提交的 `results.json` 仍 **10/10 PASS**，
11 个篡改版全红。测试：`tests/test_dem_fem_oneway_gate_bite_m21654.py`（13 条，含基线必须绿）。

### F2 —【Medium，已修】`run.py` 模块 docstring 仍宣称已退役的「peak ratio ≥0.9」门

**证据**：m21584 把该比值门降级为 diagnostic（`dt_convergence_peak_ratio_is_diagnostic_only=true`），
但模块 docstring 第 5 条仍写 “with peak-deflection convergence ratio ≥0.9”，与验收项 ③ 直接冲突。
这是「在册判据改名」类工作最容易漏的一处：**代码改了、账改了、文档没改**。
**修复**：docstring 第 5 条改为四档「逐档 vs 独立模态锚、error(dt/T1=0.14) ≤1%、error(0.0175) ≤0.1%、
细档低于粗档」，并明写比值恒 ≈1.000025、任何「接近 1」的门都恒真、本处不作判据；测试
`test_run_docstring_does_not_advertise_a_ratio_gate` 加锁。

### F3 —【Medium，未消账·局限】1% 偏转门是**单向**的：把 FEM 刚度调软 1% 反而「更准」

P3 刚度缩放（只改 FEM 的 E，锚不动）：

| E 倍数 | 0.95 | 0.98 | 0.985 | 0.99 | 0.995 | 1.0（交付） | 1.005 | 1.01 | 1.02 |
|---|---|---|---|---|---|---|---|---|---|
| 偏转误差 | 4.472% | 1.277% | 0.744% | **0.189%** | 0.388% | **0.932%** | 1.283% | 1.655% | 2.461% |
| 1% 门 | RED | RED | green | green | green | green | **RED** | RED | RED |

即：门在「+0.5% 偏硬」就红，却对「−1% 偏软」给出**比正确刚度更好看**的读数（0.189%）——因为
交付值 0.932% 本身是**时间离散**误差（见 F7：收敛后 FEM 与锚只差 0.0303%），刚度偏软恰好把
峰值抬向锚。**含义**：验收项 ② 的「≤1%」不能被读成「FEM 刚度被验证到 1%」；本算例 1% 门的
真实判别力约为「+0.5% / −2%」区间外才算错。**建议**（未在本单改判据，属口径决策）：把头条
精度改报**收敛档读数 0.0303%**（dt/8 + 独立积分器），把 0.932% 明确标为 dt/T1=0.14 的
**积分误差**；并在 results 里带上误差符号（现在只有绝对值，掩盖了符号抵消）。

### F4 —【Low，未消账·声明过度】注册链字面值不可复现（`0.908/0.102/0.041/0.043 %`）

- 交付实测：`0.9324 / 0.1268 / 0.0162 / 0.0063 %`（dt/T1 = 0.1399/0.0699/0.0350/0.0175，
  与注册值前四位一致；误差**单调下降**）。
- 注册值**非单调**（0.041% → 0.043%），且 0.908% 与交付 0.9324% 差 0.024 pp。
- P4 扫了 18 个变体（n_modes ∈ {1,3,5,6,7,9,12,15,29} × {强制窗, 含 5T1 余振窗} +
  「四档共用基网格锚」），最接近的是 `n_modes=7, window=forced → [0.9181, 0.1124, 0.0306, 0.0207]`，
  最大偏差 0.0223 pp；**没有任何变体复现注册链**（图 b）。
- **判定**：判据（≤1% / ≤0.1%）满足，**字面数字不满足**。m21584 的 §5.1「模态截断敏感带
  0.879–0.947%」经本单独立复算**逐位吻合**（5/9/15/29 阶 → 0.879/0.932/0.944/0.947%），
  所以 0.908% 只能来自**未被记录的第三种口径**（另有网格/单元数/窗长），不粉饰。
  建议把 feature_ledger 的验收文字改为可复现数字（0.9324% 与单调四档链），而不是反过来调参去凑。

### F5 —【Info，已复现】交付的确定性

同一棵树重跑 DEM+FEM（P0），四个加密档峰值与提交的 `results.json` **rel_diff = 0.0**（逐位相同），
`git status` 只有未跟踪的 `artifacts/dem_fem_coupling_ledger.json`。即交付可复现，不是抽签结果。

### F6 —【Info，已复现】m21584 的模态截断敏感度表

P3 `n_modes` 扫描（1/3/5/9/15/29 → 0.446/0.730/0.879/0.932/0.944/0.947%）与 m21584 §5.1
表中 5/9/15/29 四列逐位一致；同时显示 1–3 阶时锚自身偏差 0.45–0.73%，故「9 阶够用」成立。

### F7 —【Info，已验证】独立积分器确认 FEM 瞬态峰值不是 Newmark 平均加速度的产物

自写 Newmark（线性加速度 β=1/6，另一格式）在**稳定档**（dt/1600 与 dt/3200，ω_max·dt = 2.5 / 1.26）：
峰值 4.547208e-8 m，两档互差 3e-9（相对 6e-10）→ 收敛；对交付 dt/8 的 4.546682e-8 相对差 **1.157e-4**。
对照：自写 β=1/4 同格式对交付实现 rel_diff **8.95e-13**（实现无笔误）。
收敛峰值 4.547208e-8 vs 模态锚 4.545831e-8 → **0.0303%**：**偏转-锚一致性本身极好，交付的
0.932% 全部是 dt/T1=0.14 的时间离散误差**，不是物理/装配误差。
（同一条探针也记录了 β=1/6 在 dt/8 下必发散：ω_max·dt = 503，远超条件稳定限 ~3.4——
这是参考格式的稳定性性质，不是 FEM 缺陷。）

## 3. 五条验收的逐项机检结论

| 验收项 | 机检 | 结论 |
|---|---|---|
| ① impulse → `force_history_port_parity` + `scope=numerical_verification_only` | `results.json`、`ledger.interface`、`validate.py` 三处命名/作用域一致；P3 力幅 ×1.5 / ×0.5 时 parity 恒 0.0 而偏转误差 48.6% / 50.5% → **结构性盲**，定级正确 | **PASS**（且 F1 修复后不再可被标志冒名通过） |
| ② 偏转判据 = 动态 FEM vs 模态叠加 ≤1% | 0.9324% ≤1%（P1 独立重算锚 = 4.545831072e-8，逐位一致）；**注册的 0.908% 未复现**（F4）；门是单向的（F3） | **PASS（判据）/ 偏差记 F3–F4** |
| ③ dt 四档链 | dt/T1 = 0.1399/0.0699/0.0350/0.0175（与注册前四位一致）；误差 0.9324/0.1268/0.0162/0.0063%（单调）；**t 比值门已降级为 diagnostic 且有测试锁定** | **PASS（结构）/ 字面值偏差记 F4** |
| ④ docstring `k=200` → `platen_stiffness=2.0e4` | 现 docstring 写明 2.0e4 N/m、2.0e2 是该算例惰性的颗粒-颗粒律、t_c=5.44 ms（实测 5.367 ms）；**但同一 docstring 第 5 条仍是旧判据**（F2，已修） | **PASS（修复后）** |
| ⑤ ledger 入库（净检出可验） | `results["ledger"]` 内联，`validate.py` 优先读内联；把 `artifacts/` 副本移走后仍 **10/10 PASS**（本单复测） | **PASS** |
| 附：验脚本咬合 | 修复前 11 个篡改版有 **4 个漏判**；修复后**全红**，基线仍全绿（F1） | **修复后 PASS** |

## 4. 未执行的检查（marked not_run，不得读成通过）

- `not_run`：DEM 接触律/恢复系数的独立物理复核（无独立接触实验基准）；界面的**外部**守恒证据
  （动量/能量两侧独立测量）——port parity 与 work-energy 都仍是自比较/一致性；
- `not_run`：双向耦合反馈误差（本算例未建模，属适用范围外）；
- `not_run`：`artifacts/dem_fem_coupling_ledger.json` 是否入库（仍是未跟踪文件；因已内联，净检出
  可验，故不阻塞，但账面上「artifacts 入库或内联」应记为**内联**路径）；
- `not_run`：0.908% 的**原始口径**溯源（未找到生成脚本；P4 18 变体均不符）。

## 5. 结论与建议

**Disposition：approve with notes（机裁 level 3，附两条未消账偏差 F3/F4）。** 5 项验收在机检下全绿，核心诉求（改名 + 换基准 + 去掉恒真门 +
ledger 内联 + docstring 更正）确实落地；但复核同时发现并修复了 1 个高危自证式闸门（F1）与 1 处
旧判据残留（F2），且有 2 条偏差未消账（F3 单向门、F4 字面值）。

**给账务的三条建议**
1. feature_ledger 的 `GAP-ICE-ONEWAY-CLAIM` 验收文字把 `0.908/0.102/0.041/0.043 %` 换成可复现的
   `0.9324/0.1268/0.0162/0.0063 %`（不要反向调参凑旧数字）；并补一句「1% 门单向，判别力
   ≈ +0.5%/−2%」（F3）。
2. 新条目建议（slug / 量化验收 / 域）：`GAP-ICE-ONEWAY-CONVERGED-HEADLINE`——
   把结果 headline 精度改为「收敛峰值 vs 模态锚 ≤0.1%（实测 0.0303%）且带符号」，并把
   dt/T1=0.14 的 0.9324% 明确标为积分误差；机检：`results.json` 新增
   `deflection_error_signed` 与 `deflection_error_converged` 两键 + `validate.py` 两条数值门 → 域：ice / dem-fem coupling。
3. **同类缺陷已全仓只读扫过（本单）**：`benchmarks/{case_mindlin_tangential,case_pair_collision_hertz,hertz_analytic}/validate.py`
   ＋ `scripts/validate_*.py`（6 个）均**不含** `*_pass"]` 自证键 → 该缺陷形态在 TensorDEM 树内仅此一处，
   已修；建议把「数值门必须由数重算」写进验脚本模板，防回归。

## 6. 复现

```bash
PYTHONPATH=src python examples/dem_fem_oneway/run.py                     # 20 s，重写 results.json
PYTHONPATH=src python examples/dem_fem_oneway/validate.py                # 10/10 PASS（无需 artifacts/）
PYTHONPATH=src python tools/verify_oneway_claim_gates_m21654.py          # P0–P5，47 s，写 runs/oneway_claim_verify_m21654/probe.json
PYTHONPATH=src python -m pytest tests/test_dem_fem_oneway_gate_bite_m21654.py \
    tests/test_dem_fem_oneway_claim_review.py tests/test_dem_fem_independent_gates.py -q   # 32 passed
PYTHONPATH=src python scripts/make_oneway_claim_verify_figure_m21654.py   # 重画图
```