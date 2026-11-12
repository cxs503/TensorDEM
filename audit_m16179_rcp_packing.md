# 交差审计报告：m16179 CASE-DEM-RCP-PACKING-FRACTION

**审计任务**: m16201 (board-audit)
**审计对象**: m16179 CASE-DEM-RCP-PACKING-FRACTION · TensorDEM
**审计日期**: 2026-11-12
**审计人**: board-audit (AI agent)

---

## ① 取证核验

### 1.1 run_artifacts 文件存在性 ✓

| 文件 | 大小 | 状态 |
|---|---|---|
| `artifacts/case_rcp_packing/result.json` | 9.0 KB | ✓ 存在 |
| `artifacts/case_rcp_packing/gates.json` | 594 B | ✓ 存在 |
| `artifacts/case_rcp_packing/fig_rcp_comparison.png` | 57 KB | ✓ 存在 |
| `artifacts/case_rcp_packing/fig_jamming_scan.png` | 66 KB | ✓ 存在 |
| `artifacts/case_rcp_packing/fig_friction_comparison.png` | 46 KB | ✓ 存在 |
| `artifacts/case_rcp_packing/jamming_scan.json` | 4.7 KB | ✓ 存在 |
| `artifacts/case_rcp_packing/friction_comparison.json` | 1.1 KB | ✓ 存在 |

### 1.2 benchmark ref vs got 数值核验 ✓

| 门 | 要求 | 实际值 | 状态 |
|---|---|---|---|
| φ ∈ [0.62, 0.66] | 0.62 ≤ φ ≤ 0.66 | φ = 0.6400 | ✓ PASS |
| \|φ - φ_ref\| < 0.02 | 误差 < 0.02 | 0.0034 | ✓ PASS |
| overlap/d < 0.05 | < 0.05 | 0.0352 | ✓ PASS |
| local ε std < 0.02 | < 0.02 | 0.0049 | ✓ PASS |

**参考值来源**: Bernal & Mason (1960) Nature 188:910, φ_ref = 0.6366
**相对误差**: 0.53%（远小于 3% 精度标准）

### 1.3 repro_cmd 可执行性 ✓

```bash
PYTHONPATH=src python benchmarks/case_rcp_packing_fraction.py --n-particles 1000 --seed 42 --diameter 0.001 --out /tmp/rcp_verify --skip-jamming-scan --skip-friction-cmp
```

**验证结果**: 成功执行，输出 φ=0.6400，误差 0.0034，所有门通过。

### 1.4 报告六段完整性 ✓

`docs/CASE_RCP_PACKING.md` 包含：
1. ✓ Problem Statement & Reference
2. ✓ Method & Implementation
3. ✓ Parameter Table
4. ✓ Results: Reference vs Achieved
5. ✓ Convergence & Sensitivity
6. ✓ Failure Modes & Artifacts

---

## ② 判级（L0-L5）

| 级别 | 要求 | 满足？ | 证据 |
|---|---|---|---|
| L0 | 文档成果 | ✓ | docs/CASE_RCP_PACKING.md |
| L1 | 数学/单元验证 | ✓ | 8 tests pass (geometry, MC, analytic, periodic, determinism) |
| L2 | 独立功能实现 | ✓ | packing.py 347 行真实代码，输入→计算→输出 |
| L3 | 系统集成 | ✓ | 被 7+ 模块调用（bed.py, fluidization.py, study.py 等） |
| L4 | 基准验证 | ✓ | 完整算例 + 误差分析 + 可复现 + 文献对标 |
| L5 | 科研工程成果 | ✗ | 此为基准验证算例，非完整科研工程产品 |

**判定等级: L4**

---

## ③ 落表

已执行：
```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorDEM', 'm16179 CASE-DEM-RCP-PACKING-FRACTION', 16201, 4, '<JSON>', 'board-audit')
```

Row ID: 32

---

## ④ 独立复核发现

### 无阻塞性问题

1. **overlap/d 门放宽说明**：门限 0.05 对 RCP 是合理的——在堵塞转变附近，几何压缩算法内禀产生小重叠，这是物理上粒子接触的数值表现。
2. **N=10000 算例**：报告提到 cluster job cj1261 pending，不影响当前 L4 判定（N=3000 已充分验证）。

### 建议（非阻塞）

1. 可考虑增加 N=10000 的结果以展示有限尺寸效应收敛性
2. 可考虑与物理 DEM 沉降结果做对比（当前为几何压缩算法）

---

## ⑤ 处置

**Disposition: APPROVE** — m16179 交差核验通过，L4 等级确认，消账。
