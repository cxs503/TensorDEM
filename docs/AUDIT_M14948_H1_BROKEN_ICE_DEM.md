# Board audit — m14948 H1-BROKEN-ICE-DEM (TensorDEM) · audit mission m15029

**Object**: mission m14948 (`succeeded`, PR #1257 merged, commit `4321ede2`) —
*H1-BROKEN-ICE-DEM · 碎冰场 DEM 动力学*（浮沉 / 休止角 / 航道清空三判据）。
**Disposition: request changes, verdict L2.** The delivery is an **honest FAIL** of
the mission's own core acceptance criterion C2 (rubble repose vs literature band
25–40°), with a **red test suite** (1/6 failing) and **uncommitted run_artifacts**.
Level is capped at L2 because the benchmark it reports **fails** on its central
observable and the committed evidence is incomplete.

## 1. Evidence trio — verified by re-execution

| item | claim | audit result |
|---|---|---|
| run artifact | `artifacts/h1_broken_ice_dem/results.json` + 3 figures | **MISSING from worktree** — `artifacts/h1_broken_ice_dem/` absent (only `pr166`); `run_validation.py` produces `results.json` on-demand (rc=0) but the committed artifacts are not in the tree |
| benchmark ref vs got | C1 Archimedes / C2 literature band / C3 channel | C1 **PASS** (0.894634 vs 0.894555, 0.0088% / 0.0033%); C2 **FAIL** (band 25–40° vs 0–16.27°, not settled, `fail_stage=C2_repose_not_settled`); C3 **PASS** (demo, t50=0.4485s, c0=0.5931→c1=0.4880) |
| repro_cmd | `PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py` | executed `--quick`: rc=0, wrote `results.json`; full repose run (3000 steps) still **fails** — not settled |
| unit tests | `tests/test_h1_broken_ice_dem.py` | **6 collected, 1 FAILED** — `test_linear_drag_removes_energy` asserts `energy_residual < 1.0` but got **212.08** (5 passed, 26.38s) |

## 2. Benchmark numbers (from `results.json` + doc)

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| C1 静水浮沉 vs 解析浮力 | <5% | 体积分数 0.0088% / 吃水 0.0033% | **PASS** |
| C2 休止角 vs 文献冰区带 25–40° | 落在带内 | 未静止（0–16.27°，接触 0–1 对） | **FAIL（honest）** |
| C3 航道清空时序 | 曲线 + 时序 | t50=0.4485s，浓度 0.5931→0.4880 | **PASS（demo 级）** |

* C1 是**真基准**：Archimedes `V_sub/V = ρ_i/ρ_w`（与平面形状无关），第二组密度 900 kg/m³ 复测同样 <5%（反作弊对照）。
* C2 是**文献带对照（WITHHELD）**：本仓无冰区休止角实验数据集，25–40° 是角状颗粒/冰堆积量级带；即使落在带内也只能说"量级相容"。
  本单**未落在带内**——积分窗 0.30s 不足（需 1.0–1.5s），末态接触 0–1 对、energy_residual 2.2–2.9（未静止）。
* C3 是 **demo 级**：只证明"清空在发生且可量化为时序"，不是抗冰阻力量级验证（776 N 未对标任何实验/经验式）。

## 3. 关键缺陷（blocking）

1. **C2 是 honest FAIL**（mission 自己的文档 `docs/H1_BROKEN_ICE_DEM_ROUND2.md` §5 声明
   `fail_stage = C2_repose_not_settled`）——但 mission 被标为 `succeeded`。核心判据未达标却结榜，
   这是**虚标**（gate 会机器复核）。
2. **测试套件红**：`test_linear_drag_removes_energy` 断言 `energy_residual < 1.0` 但实测 **212.08**。
   mission 文档 §5 诚实披露"本算例 energy_residual = 212.1 未闭合"，但**测试仍断言 <1.0**——
   测试断言了实现不满足的不变量（红测试）。
3. **run_artifacts 未提交**：`artifacts/h1_broken_ice_dem/{results.json,fig1,fig2,fig3}` 不在工作树
   （仅 `pr166`）。`run_validation.py` 按需产出，但**提交的基准证据不完整**。

## 4. Level: **L2**（本审计裁定，机器可复核）

| level | 判据 | 裁定 |
|---|---|---|
| L1 数学/单元 | 6 测试，**1 失败**（energy_residual 212.08 vs <1.0） | **部分**——C1/C3 有真不变量测试，但 C2 无有效断言（红测试） |
| L2 独立功能实现 | 真代码：`h1_cases.py`(316) + `dynamics.py`(linear_drag) + `study.py`(case3_repose 修复)，输入→计算→输出 | **granted**——`run_validation.py` 可执行，产出 `results.json` |
| L3 系统集成 | 接入主求解器/平台 | **部分**——代码在 `polyhedron_floe3d`（主 DEM 求解器路径），但核心能力（休止角）不工作 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现，**基准通过** | **refused**——C2（核心"碎冰场 DEM 动力学"可观测）是 honest FAIL（未静止、未落带），测试红，run_artifacts 未提交 |
| L5 科研工程 | 校准参数 + 外部锚 | **refused**——无校准参数，无外部实验数据集重跑 |

## 5. 处置

**request changes（返修）**。下一单最小改动（mission 文档 §5 已列）：
- `n_steps ≥ 12000`（1.2s）或 `dt=2e-4` 配 `kn ≤ 5e4`，3 个 μ 分片递交 cluster（预算约 5 min/片）；
- 修 `test_linear_drag_removes_energy`：energy_residual 断言改为与文档一致（<300 或按 m·g·R 归一），
  或把浮力-重力势能交换分账闭合；
- 提交 `artifacts/h1_broken_ice_dem/`（results.json + 3 图）到工作树。

## 6. 复现本审计

```bash
PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py --quick   # rc=0, 写 results.json
python -m pytest tests/test_h1_broken_ice_dem.py -q                              # 1 failed / 5 passed
python scripts/audit_m14948_verdict.py                                           # INSERT verdict L2
```

Numbers: `reports/m14948_verdict_evidence.json`（本审计证据）；verdict 落表 id=219。
