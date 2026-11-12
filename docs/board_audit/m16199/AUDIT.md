# Board audit — m16142 `CASE-DEM-ROLLING-SINGLE-PARTICLE`

* audit mission: **m16199** (交差核验单，非重做)
* audited mission: **m16142** (TensorDEM 算例池，status=succeeded)
* case pool row: `tensor_case_pool.id=90`, `case_code=DEM-ROLLING-SINGLE-PARTICLE`, `tol_policy=<2% | 球心平动加速度与转动角加速度 vs 解析 (theta=15deg, 无滑移)`
* verdict: **L4 — confirmed**（evidence: `docs/board_audit/m16199/verdict_evidence.json`, row 435 in `tensor_maturity_verdicts`）
* independence limitation: the auditor did not participate in m16142; the audit ran in a **separate worktree** (`/root/tp-agent/work/m16199`) on the merged commit and re-derived every acceptance number itself. The only shared object is the repository code under test.

## 1. Scope / what was checked

| 检查项 | 结果 | 证据 |
|---|---|---|
| run_artifacts 真存在 | ✔ | `artifacts/case_rolling_single_particle/result.json` (91 737 B), `*.png` (205 513 B), `logs/*.log` in the m16142 worktree |
| benchmark ref vs got 在容差内 | ✔ | a: 1.8135820231826634 vs 1.8135801271553287 → **1.05e-6**; alpha: 2.61e-6; no-slip closure 3.66e-6; 判据 2% |
| repro_cmd 可执行 | ✔ | 审计在自己工作树重跑 → `a_got` **逐位相同** (1.8135801271553287)，`a_rel_error=1.0454599298265815e-06` |
| 报告六段齐全 | ✔ | `tensor_reports` 15262：①问题/参考出处 ②方法 ③参数表(含无量纲) ④ref vs got 对照 + 图 ⑤收敛/敏感性 ⑥失败模式/局限/工件 + 显式回答「是否<3%」 |
| 负控判别力 | ✔ | μ=0.01 → 34.78% 偏差被拒；无惯性伪参考 g sinθ → 28.57% 被拒 |
| 预注册阈值先于生产运行 | ✔ | `prereg/CASE-DEM-ROLLING-SINGLE-PARTICLE.json`（2% 门、负控、stop rule） |

复现命令（审计实跑）：

```bash
cd /root/tp-agent/work/m16199
PYTHONPATH=src python3 benchmarks/case_rolling_single_particle/run_case.py \
    --out docs/board_audit/m16199/repro_result.json      # 全部点位，<1 min
PYTHONPATH=src python3 -m pytest tests/test_case_rolling_single_particle.py -q   # 6 passed in 29.82 s
```

## 2. Independent (non-reused) evidence

`docs/board_audit/m16199/probe_independent.py` 用与算例不同的估计量再测一遍：

| 探针 | 方法 | 结果 |
|---|---|---|
| P1 | 位移二次拟合 `x = x0 + v0 t + a t²/2`（非 v 斜率） | θ=10/15/25°，误差 **2.47e-10**；与 v 斜率差 8.15e-08·a_ref |
| P2 | a/(g sinθ) 三角验证 | 0.7142857727 vs 1/1.4=0.7142857143 → 实测惯性因子 0.4（非假定） |
| P3 | μ_r=0.1 分支位移 | 实测 −0.3732069956 vs 预测 −μ_r cotθ = −0.3732050808（2.97e-6 相对）→ **CDT 力矩真的进了方程，不是闭式套壳** |

图：`docs/board_audit/m16199/figures/audit_rolling_single_particle.png`（全部数字来自审计自己的复现结果，脚本 `make_audit_figure.py`，零手抄）。

## 3. Findings (severity ranked)

* **F1 informational** — `tensor_benchmark_cases` 对该 anchor **0 行**（表共 25 行，2026-11-12 查）。机器基准裁判缺席，且 `tensor_case_pool` 该行仍为 `status='dispatched'`。本审计以 case_pool 的 tol_policy（<2%）+ 仓内闭式解析解裁定。属登记缺口，非 m16142 缺陷。
* **F2 minor** — 合入门禁滤除了 3 个文件（`artifacts/result.json`、`artifacts/*.png`、`logs/*.log`），故 main 树上只有 harness 没有工件；因 repro_cmd 逐位可复现，**非阻断**，但 gate 复核时须重跑 repro_cmd 而非期待仓内已存工件。
* **F3 minor（读法警告）** — 验收量是 v(t) 在 ~195 个切向振荡周期上的最小二乘均值，离散误差被平均掉，因此任何自洽积分器都落在 1e-6；**不得**读作「求解器精度 1e-6」。真正有判别力的是振荡幅值（基准点 0.94%，随 Δt 一阶收敛 3.93%→0.94%→0.23%→0.058%）与 μ_r 分支（相对依赖于力矩的参考 4e-6）。两者都在案且通过。
* **F4 informational** — 外部锚仅闭式刚体动力学 + Ai et al. 2011 的**模型关系式**；原文 PDF 与表格数值在本环境不可得，报告已在 `provenance.reference.access_note` 与正文 ① 明示不宣称其表值 → **L5 拒绝**，此为诚实降级而非缺陷。

## 4. Checks NOT run (marked unknown, not "passed")

* 未与 3-D 求解器 `IceDEM3D` 交叉验证——本算例的仪器是专用 1-DOF 斜面 harness（报告 ⑥ 已如实声明），3-D 对偶滚动由别的锚覆盖。
* 无实验/材料标定对照——参考值是刚体运动学恒等式，不适用。
* 多颗粒/clump 斜面滚动不在本算例范围。

## 5. Level justification (L0–L5)

* **L1** 解析/单元验证：`tests/test_case_rolling_single_particle.py` 6 项含两个负控。
* **L2** 真实现：`src/tensordem/rolling_resistance.py:443` 是真实积分循环（接触力→加速度→半隐式 Euler），非硬编码。
* **L3** 系统集成：滚动阻力律是仓内唯一实现，被 `contact_tangential.py:642`、`ball_ball_rolling.py:404`、`clump_rolling.py` 消费；但斜面 harness 本身是 1-DOF 仪器。
* **L4** 基准验证：完整算例 + 逐项误差分析（全项 <1%，门 2%）+ 收敛/刚度/窗口敏感性 + 两负控 + 逐位可复现 → **取 L4**。
* **L5** 拒绝：无外部文献/实验锚（F4）。

## 6. Disposition

**Approve (L4)**。建议（不阻断）：① 给该 anchor 补 `tensor_benchmark_cases` 行（metric = mean a、alpha、振荡幅值、μ_r 分支，tolerance 2%），并把 case_pool 状态从 `dispatched` 推进；② 让合入门对 `artifacts/` 下小体积 JSON 放行，或在 gate 复核流程里显式调用 repro_cmd。