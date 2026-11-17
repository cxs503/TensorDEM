#!/usr/bin/env python3
"""Post the paper-grade report for CASE-BIAXIAL-SHEAR-BAND machine-referee registration."""
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

# ---- figure --------------------------------------------------------------
fig_path = ROOT / "artifacts" / "biaxial_shear_band" / "r10_literature_verification" / "r10_literature_verification.png"
fig_b64 = base64.b64encode(fig_path.read_bytes()).decode()

markdown = r"""# CASE-BIAXIAL-SHEAR-BAND — 机器裁判登记（tensor_benchmark_cases id=179）

## ① 案例介绍（物理背景/几何/参数表）

平面应变双轴试验：矩形冻结黏结圆盘试样（24×48 抖动三角格，35% 冻结键，
壁面摩擦 0.55，围压 σ₃=100 kPa 伺服侧壁，顶板位移加载），峰后材料局部化为
一条剪切带，带相对 σ₃ 轴（水平）的倾角 θ 是观测量。经典理论给出三条只依赖
φ、ψ 的关系（Mohr–Coulomb 45+φ/2；Roscoe 45+ψ/2；Arthur 45+(φ+ψ)/4）。

本案例验收锚 = 「文献剪切带倾角，容差 <15% + 定性」，锚值 60°（Desrues &
Viggiani 2004 密实 Hostun 砂 55–60° 带中点）。

| 参数 | 值 |
|---|---|
| 试样 | 24×48 抖动三角格，35% 冻结键 |
| 壁面摩擦 | 0.55 |
| 围压 σ₃ | 100 kPa |
| 带倾角 θ（R7 累计晚窗，n=8） | 55.00° ± 2.07° |
| 实测 φ（峰） | 35.45° |
| Rowe ψ（φ_cv=32°） | 4.14° |

## ② 怎么做的（方法/流程/工具链）

m15315 板审计（`docs/board_audit/m15315/AUDIT.md`）发现本锚在平台机器裁判表
`tensor_benchmark_cases` **0 行**——<15% 容差只能从仓内 docstring 判，无法机器
对表。R10（`docs/CASE_BIAXIAL_SHEAR_BAND_R10.md`）已独立核验文献带并重判
（θ=55.00 ∈ [55,60]，8.33% < 15% = PASS）。本单**清偿登记欠账**：

1. `scripts/register_biaxial_shear_band_benchmark.py`：从 R10 验证模块
   （`verify_literature_anchor`）程序化重导出判定（零手抄），幂等插入
   `tensor_benchmark_cases` 裁判行（status=passed）；
2. 写 `artifacts/biaxial_shear_band/benchmark_registration/benchmark_registration.json`
   （程序化注入）；
3. `tests/test_biaxial_shear_band_benchmark_registration.py`：10 个自洽性测试，
   断言登记工件与案例判定一致（防陈旧/手改登记静默通过）。

## ③ 计算结果（数字表，程序化注入）

| 量 | 数值 |
|---|---|
| 带倾角 θ | 55.00° |
| 主文献带（Desrues & Viggiani 2004） | 55–60° |
| 相对锚点中点 60° 误差 | 8.33% |
| <15% 容差 | **True（PASS）** |
| <3% 精度 | **False**（8.33% > 3%，honest） |
| 落在文献带内 | True（55.00 ∈ [55,60]） |
| 三定律理论窗 | [47.07°, 62.73°] |
| 机器裁判行 | tensor_benchmark_cases **id=179**，status=passed |

## ④ 结果图

图：R10 文献核验对比——实测 θ=55.00° 放回 Desrues & Viggiani 2004 55–60° 带
与三定律理论窗（MC 62.73 / Roscoe 47.07 / Arthur 54.90），相对中点 60° 误差
8.33% < 15% 容差。

## ⑤ 注意点（踩过的坑/局限/适用范围）

- **坑**：`RETURNING id` 的 psql 输出含 `INSERT 0 1` 状态行，需取首行解析；
  已修（幂等重跑）。
- **局限**：本单不重跑 DEM，所有数字来自已合并 R7/R9 证据（θ=55.00、φ=35.45、
  Rowe ψ@32°=4.14）；2D 圆盘限制（非 3D 球）与单围压（100 kPa）未覆盖。
- **适用范围**：机器裁判登记使平台门可对表裁决本锚；3% 精度条未达（8.33%），
  属 honest 记录，不粉饰。

## ⑥ 误差判定

关键量 θ vs 基准（文献带中点 60°）：**8.33%**，显式回答「是否<3%」= **否**
（8.33% > 3%）；「是否<15%」= **是（PASS）**。无基准须声明——本锚有明确
文献基准（Desrues & Viggiani 2004 55–60° 带），故不适用「无基准」豁免。
"""

report = {
    "title": "CASE-BIAXIAL-SHEAR-BAND — 机器裁判登记（tensor_benchmark_cases id=179，PASS）",
    "summary": "清偿 m15315 板审计验证欠账：为 CASE-BIAXIAL-SHEAR-BAND 在 tensor_benchmark_cases 登记机器裁判行（id=179, status=passed）。R10 已独立核验文献带（θ=55.00° ∈ [55,60]，8.33% < 15% = PASS），本单程序化注入登记并加 10 个自洽性测试（全绿）。",
    "scenario_key": "CASE-BIAXIAL-SHEAR-BAND",
    "mission_id": "24414",
    "tags": ["TensorDEM", "biaxial", "shear-band", "benchmark-registration", "verification-debt"],
    "markdown": markdown,
    "figures": [
        {
            "name": "r10_literature_verification.png",
            "caption": "R10 文献核验：实测 θ=55.00° 放回 Desrues & Viggiani 2004 55–60° 带与三定律理论窗，相对中点 60° 误差 8.33% < 15% 容差 = PASS。",
            "png_base64": fig_b64,
        }
    ],
}

req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/tensor/reports",
    data=json.dumps(report).encode(),
    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
)
resp = json.loads(urllib.request.urlopen(req).read())
print("report_id:", resp.get("id") or resp.get("report_id") or resp)
