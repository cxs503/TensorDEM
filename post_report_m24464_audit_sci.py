#!/usr/bin/env python3
"""Post the paper-grade report for AUDIT-SCI (README claim vs code truth reconciliation)."""
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- login ---------------------------------------------------------------
req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/auth/login",
    data=json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode(),
    headers={"Content-Type": "application/json"},
)
body = json.loads(urllib.request.urlopen(req).read())
token = body.get("access_token") or body.get("token")
assert token, f"login did not return a token: {list(body.keys())}"

markdown = r"""# AUDIT-SCI：求解器 README 声称 vs 代码真相对账（TensorDEM + TensorSPH）

## ① 案例介绍（物理背景/几何/参数表）

本单是**验证欠账类审计**（feature_ledger 锚点 `AUDIT-SCI`，status=in_progress）：把两仓
（TensorDEM、TensorSPH）已合并功能的 README 能力声称逐条对到「代码文件 + 测试 + 基准数值证据」，
重点核查三处：
- **TensorDEM**：冰脊触底（ridge keel grounding）、海底管道地基（pipeline foundation）、
  船体冰磨损（ice-hull abrasion）是否**真有实现与算例**；
- **TensorSPH**：3-D 算子验证**真伪**（2.7% 剪切衰减/静水底压声称是否硬编码），以及
  自由液面/表面张力/多相**未完成项是否如实标注**。

## ② 怎么做的（方法/流程/工具链）

1. 读两仓 README，抽取全部能力声称（TensorDEM §14-330、TensorSPH §19-152）；
2. 每条声称 → `grep`/`find` 定位源码模块 → 定位对应 pytest 测试 → 定位文档/脚本/工件；
3. **跑测试**（本机轻量，load15≈46）：TensorDEM 关键测试（ridge_grounding 24、ice_abrasion 39、
   pipeline_dem 9）与 TensorSPH 关键测试（threed 29、csf_mixture 9、surface_tension_dispersion 19）全绿；
4. 对「证据门」类测试做**独立复现**：发现 TensorSPH `test_surface_tension_dispersion.py` 的
   `test_committed_campaign_artifacts_satisfy_the_reported_invariants` 在 HEAD 上**红** ——
   提交的 `meta.json`/`report.json` 在 `benchmarks/surface_tension_dispersion/`，但测试与
   provenance 记录引用 `runs/surface_tension_dispersion/`（`git ls-files runs/` 为空，从未提交）；
5. **修复**：按 run 脚本 `--out-dir` 约定重新生成全部工件到正确路径，证据门测试转绿。

## ③ 计算结果（数字表，程序化注入）

| 声称 | 仓 | 证据门测试 | 判定 |
|---|---|---|---|
| 冰脊触底+管道地基（D1/D2） | TensorDEM | `test_ridge_grounding.py` 24 passed；`test_p54w3_n06_ridge_pipeline_dem.py` 9 passed | **REAL** |
| 船体冰磨损（D3） | TensorDEM | `test_ice_abrasion.py` 39 passed | **REAL**（CALIBRATED 如实标） |
| 3-D 算子（S1） | TensorSPH | `test_threed_*` 29 passed；剪切衰减 n=24 -2.68%、静水底压 +1.81% | **REAL**（2.7% 有解析基准） |
| 混合表面张力（S4） | TensorSPH | `test_csf_mixture.py` 9 passed；Young-Laplace 1.4% | **REAL** |
| 自由液面/多相未完成项（S6） | TensorSPH | —（scope 声明） | **诚实标注**（非虚标） |
| **验证欠账 V1** | TensorSPH | `test_committed_campaign_artifacts...` 修复前**红** → 修复后 **1 passed** | **已修复** |

## ④ 结果图

**无图**（理由）：本单是**文档/审计类**任务 —— 产出是「声称-证据对照表」与「证据门复现/修复」，
不是场量云图或收敛曲线。关键数值证据（剪切衰减 -2.68%、静水底压 +1.81%、Young-Laplace 1.4%）
已在 §③ 数字表与对应 `docs/` 工件中给出，且全部由 pytest 程序化断言而非手抄。

## ⑤ 注意点（踩过的坑/局限）

- **工件路径漂移是审计高发坑**：`meta.json` 的 `artifacts` 列表引用 `runs/...`，但实际提交在
  `benchmarks/...` —— 证据门测试用 `REPO / "runs/..."` 硬编码路径，一旦路径漂移就红。修法是
  按 run 脚本 `--out-dir` 约定重新生成，而不是改测试去迁就漂移路径。
- **诚实 FAIL 记录**：重新生成的 `report.json` 是 status=FAIL（C1/C2/C3 未达标、C4 通过）——
  与测试断言一致，证明该能力**如实报告未达标项**而非粉饰。
- **无基准声明**：TensorDEM 冰脊/管道/磨损是 DEM 微参数集数值预测（非实测），README 已明确
  「not validated field measurements」，故不设 <3% 门 —— 这是 honest 的 scope 边界。

## ⑥ 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| TensorSPH 3-D 剪切衰减率 | unsteady-Stokes `2νk²` | -2.68%（n=24） | **是** |
| TensorSPH 3-D 静水底压 | 静水柱 | +1.81%（c0=40） | **是** |
| TensorSPH 混合表面张力 | Young-Laplace `Δp=σ/R` | 1.4%（R/dx=32） | **是** |
| TensorDEM 冰脊/管道/磨损 | 无单一解析基准 | 测试全绿 | **无基准**（已声明原因） |

**结论**：两仓核心能力全部 REAL、无虚标；1 处验证欠账（TensorSPH surface-tension-dispersion
证据门红）已修复并转绿。给下次自找活的建议：① TensorSPH `verification_threed.md` 链接漂移（5 分钟级）；
② surface-tension-dispersion 的 honest-FAIL 可做「换估计量」后续改进；③ `ice_abrasion_calculation`
API 合约与 ship-platform 侧注册的跨仓对齐核对。
"""

report = {
    "title": "AUDIT-SCI：求解器README声称vs代码真相对账（TensorDEM+TensorSPH）——两仓核心能力全部REAL，修复1处证据门红",
    "summary": "把 TensorDEM+TensorSPH 已合并功能的 README 能力声称逐条对到代码+测试+基准证据：冰脊触底/管道地基/船体冰磨损与 3-D 算子/表面张力/多相全部 REAL 无虚标；发现并修复 TensorSPH surface-tension-dispersion 证据门红（工件路径漂移，重新生成到 runs/ 后转绿）。",
    "scenario_key": "AUDIT-SCI",
    "mission_id": "24464",
    "tags": ["audit", "verification-debt", "tensordem", "tensorsph", "claim-evidence"],
    "markdown": markdown,
    "figures": [],
}

req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/tensor/reports",
    data=json.dumps(report).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
)
resp = json.loads(urllib.request.urlopen(req).read())
print("REPORT POSTED:", json.dumps(resp, ensure_ascii=False)[:500])
