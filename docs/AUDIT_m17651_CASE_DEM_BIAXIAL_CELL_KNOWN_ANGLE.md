# 交差审计 m17651 · CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE（TensorDEM）

**审计者**: board-audit（本单 m17733）｜**对象**: m17651（已 succeeded）｜**锚点**: CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE

## ① 取证（三件套核验）

| 验收项 | 证据 | 核验 |
|---|---|---|
| run_artifacts | `artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE/run.json`（88510 B）+ `shear_band_case.png`（254819 B） | ✅ 存在非空 |
| benchmark ref vs got | ref 60.0°（Desrues & Viggiani 2004 中点）vs got 55.50° = 7.50% 误差 | ✅ 在 15% 容差内 |
| repro_cmd | `TENSORDEM_THREADS=4 PYTHONPATH=src python benchmarks/biaxial_shear_band/run_anchor_case.py --out-dir ...` | ✅ 已复跑，复现 55.50° |

**基准裁判**（tensor_benchmark_cases）：`anchor_slug='CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE'` → case_name=`TensorDEM plane-strain biaxial cell (24x48, p=100 kPa, wall_friction 0.55, 4000 consolidation steps) shear-band inclination at the resolved late window`，metric=`shear band inclination vs anchor (deg)`，tolerance=`<15% (owner standard)`。got 55.50° vs ref 60° = 7.50% < 15% ✅。

**报告六段**：tensor_reports id=16631 六段齐全（①案例介绍 ②方法与实现 ③计算结果 ④结果图 ⑤注意点 ⑥误差判定），误差判定显式回答「7.50% 未达 3% 但在 15% 验收线内 PASS」。

## ② 判级

- **L2 独立功能**：`src/tensordem/biaxial_anchor_case.py`（473 行）+ `biaxial_shear_band.py`（611 行）真代码，输入→计算→输出；入口 `run_anchor_case.py`；run_artifacts 真实。✅
- **L3 系统集成**：`src/tensordem/integration_callers.py`（run_biaxial_shear_band_case L421 / load_biaxial_case_result L480 / check_biaxial_maturity_gate L535），平台侧真调用。✅
- **L4 基准验证**：完整算例（24×48, 100 kPa, mu_wall 0.55, 4000 步）+ 误差分析（7.50%）+ 可复现（已复跑）。✅
- **L5 科研工程**：external_anchor（Desrues & Viggiani 2004 文献）存在，但本单为基准验证（L4 实质），不虚标 L5。

**裁定：L4**（基准验证完整算例+误差分析+可复现）。

## ③ 落表

已 INSERT `tensor_maturity_verdicts`（id=636, mission_id=17733, level=4, verdict_by=board-audit, gate_state=pending）。gate 机器复核将 confirmed（消账）。

## ④ 差异与诚实声明

- 倾角对文献锚点 7.50% 未达 3%（文献本身给 55–60° 区间，取中点 60°），落在 15% 验收线内——如实报，不粉饰。
- 已知角算子误差 0.00°（30/45/60/70 全精确复现）达 3% 标准。
- 全局 psi（41.51°）> phi（35.02°）非物理，Roscoe/Arthur 行不作为证据（已声明 WITHHELD）。
