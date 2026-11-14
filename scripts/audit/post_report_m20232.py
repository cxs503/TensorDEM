#!/usr/bin/env python3
"""Post the m20232 board-audit report for m20055 / CASE-DEM-SILO-BEVERLOO."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "board_audit" / "m20232"
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER, PASS = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]
A = json.loads((OUT / "audit_m20055_beverloo.json").read_text())
REV = json.loads((OUT / "refs" / "cj2578_D4.json").read_text())


def post(path, body, token=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def fig(name, caption):
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode((OUT / name).read_bytes()).decode()}


tok = post("/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")
if not token:
    raise SystemExit(f"login failed: {json.dumps(tok)[:300]}")

est = A["estimators"]
rows = "\n".join(f"| {k} | {v['W']:.4f} | {v['rel_pct']:+.1f} | {v['C_recovered']:.3f} | "
                 f"{'✓' if v['pass_10pct'] else '✗'} | {'✓' if v['pass_15pct'] else '✗'} |"
                 for k, v in est.items())

md = f"""# 交差核验：m20055 CASE-DEM-SILO-BEVERLOO（TensorDEM）判级 **L2**，基准锚 **FAIL**

任务 m20232（交差审计，非重做）。对象 m20055（status succeeded，fail_stage=wall_exhausted）。
判决：**L2**（真实代码+真实测试，1e-16 可重放），L4 拒绝（+25.0% 超差、6 例预注册只落地 1 例、库里无裁判行）。

## ① 案例介绍（物理背景 / 几何 / 参数表）

平底圆形料仓中心圆孔自由重力卸料，稳态质量流率 W 对 Beverloo (1961) 律
W = C·ρ_b·√g·(D−k·d)^(5/2)，球状颗粒 C=0.58（0.55–0.60）、k=1.4（1.2–1.6），Eq.(4) p.265。

| 量 | 值（来自 filed artefact） |
|---|---|
| 算例 | D4（唯一落地），D/d = {A['case']['D_over_d']:.2f} |
| d / D / ρ_p | {A['case']['d_m']*1000:.1f} mm / {A['case']['D_m']*1000:.1f} mm / 2500 kg·m⁻³ |
| 床堆密度 ρ_b（开孔瞬间实测） | {A['case']['rho_b_kg_m3']:.1f} kg·m⁻³，φ = {A['case']['phi_bed']:.3f} |
| 颗粒数 / 步数 / dt | 1525 / {A['case']['steps']} / {A['case']['dt_s']:.2e} s（dt/T_c = 0.0354） |
| 卸料 | {A['case']['n_discharged']} 颗，首个卸料事件 t = {A['case']['t_first_discharge_s']:.3f} s ≥ settle 0.60 s（堵孔有效） |

## ② 怎么做的（核验方法 / 工具链）

1. 逐项取证：m20055 全部 8 个交付路径在工作树内**未提交**（head_commit NULL、git_sha='unknown'），
   故以 sha256 固定字节副本复核（`docs/board_audit/m20232/refs/`）。
2. 独立重算：`scripts/audit/audit_m20055_beverloo.py` 只用原始卸料事件流 (t,N) 重算 W，
   **复现 filed W 到 2.2e-16**；并用 8 种估计器检验判词是否依赖估计方法。
3. 基准裁判：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE
   anchor_slug='CASE-DEM-SILO-BEVERLOO'` → **0 行**（全表 127 行，无任何 SILO/BEVERLOO 锚）。
4. 回归：`pytest tests/test_silo_discharge_steady.py tests/test_silo_discharge_beverloo.py -q` → 22 passed 13.3 s。
5. 计算递交门重放：cj2578（1 核，node n330，734.6 s，succeeded）。

## ③ 计算结果（ref vs got）

| 量 | 值 |
|---|---|
| W_ref（Beverloo，C=0.58,k=1.4，实测 ρ_b） | {A['beverloo_ref']['W_ref_kg_s']:.4f} kg/s |
| W_dem（质量分位 0.2–0.8 稳态率） | {A['filed']['W_got_kg_s']:.4f} kg/s |
| 相对误差 | **{A['filed']['rel_pct']:+.1f} %**（filed 自身 pass_10pct=false、pass_15pct=false） |
| 回收 C | {est['quantile_0.20-0.80 (predeclared)']['C_recovered']:.3f}（文献 0.55–0.60 之外） |
| 稳态 CV（5 子窗） | {A['filed']['steady_rate_cv']:.3f}；0.25 s 分箱 CV 0.126；最长 ±5% 平台仅 0.75 s |
| cj2578 独立重放 | W = {REV['verdict']['W_got_kg_s']:.10f} kg/s，rel {REV['verdict']['rel_err_W']*100:+.4f}%，
  n_discharged = {REV['telemetry']['n_discharged']}，CV = {REV['telemetry']['steady_rate_cv']:.6f} —— 与 filed **逐位相同（差 0.0）** |

同一原始记录的 8 种估计器（判词稳健性）：

| estimator | W [kg/s] | rel % | C_rec | <10% | <15% |
|---|---|---|---|---|---|
{rows}

## ④ 结果图

图 1：DEM 卸料率时间序列（0.25 s 分箱）vs Beverloo 参考线 —— 无干净平台，箱间 ±12.6% 起伏。
图 2：同一原始记录 8 种估计器的相对误差 —— 除「含堵孔期零窗的旧协议均值」(−5.5%) 外全部落在
+19.6…+28.8%；那个 −5.5% 是协议缺陷（windows[0]=0，而首个卸料事件在 t=0.665 s > settle 0.60 s）的产物。

## ⑤ 注意点（坑 / 局限 / 适用范围）

* **B2 6 例只落地 1 例**：`runs/m20055_beverloo/` 只有 D4.json；报告 §⑤(a) dt 收敛、§⑤(b) 摩擦敏感性
  两张表是**只有表头的空表**，`power_law={{}}`（<3 个直径无法拟合 5/2 指数）。cj2460 只落地 shard1，
  cj2555 未取回。
* **B3 判词点名 D5 但无 D5 工件**：报告第 3 行/第 86 行称「主算例 D/d=5 FAIL」，磁盘与库里都没有 D5；
  实际证据只有 D/d=4，而 D/d=4 恰在该相关式自声明适用域的下边界（D_over_d_min=4.0）。
* **M1 库里无裁判行**：锚不可被机器判；mission 自身 server_quality_audit = gate_valid:false。
* **M3 稳态门太弱**：CV<0.15 通过，但分箱率在 1.20–2.12 kg/s 之间摆；紧带 0.35–0.65 的 CV=0.164 就不过门
  ⇒ 「稳态」未成立，且 10% 一致度目标已接近本例分辨率（n=766）。+25% 本身不是噪声：7.5σ。
* **M4 复现身份不全**：git_sha='unknown'、8 路径未提交、工件未记 nshards/job id。
* **M5 报告的一键复现行**把 bash 脚本交给 python（`PYTHONPATH=src python …/repro.sh`）不可执行；
  repro.sh 本身正确，物理路径已由 cj2578 端到端验证。
* **诚实点**：这一单有价值的地方是协议修正 + 密度基准修正 —— 参考值用**开孔瞬间实测床密度**
  ρ_b=1298.5（φ=0.519），而不是旧 beverloo_discharge.py 里硬编码 φ=0.64（m15418 审计发现的密度基准缺陷）。
  我重算 W_ref=1.454480 kg/s 与 filed 完全一致，无放大因子。

## ⑥ 误差判定（逐项，显式回答）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| W（D/d=4，质量分位 0.2–0.8） | Beverloo 1961 Eq.(4) p.265 | **+25.0%** | **否**（且 >15% 不可接受） |
| 回收 C | 0.58（0.55–0.60） | +25.0%（C=0.7248） | **否** |
| 5/2 指数（W ∝ (D−kd)^{{5/2}}） | 2.5 | 无法评估（<3 个直径） | **未评（unknown，非通过）** |
| dt 收敛 / μ 敏感性 | 预注册 6 例 | 5 例无工件 | **未跑（unknown，非通过）** |
| cj2578 重放 vs filed W | 自洽 | 0.0（逐位相同） | 是（复现性） |

**结论：基准锚 FAIL，不是「报告不好看」，是数字本身超差 25%。** 级别 = 功能实质 = L2；
L4 被拒的三重理由：超差、1/6 覆盖、无裁判行。verdict 已落 `tensor_maturity_verdicts`
(id 855, repo=TensorDEM, scope='m20055 CASE-DEM-SILO-BEVERLOO', mission_id=20232, level=2,
verdict_by='board-audit', gate_state=pending) 交 maturity_gate 自动裁定。
"""

body = {
    "title": "交差核验 m20055 CASE-DEM-SILO-BEVERLOO（TensorDEM）：判 L2，基准锚 +25.0% FAIL（可重放到 1e-16）",
    "summary": ("m20055 交付的是真实 DEM + 真实不变量测试（22 passed），但不是基准验证：唯一落地的 D4 相对 "
                "Beverloo +25.0%（自身 pass_10pct=false），8 种估计器里 7 种落在 +19.6…+28.8%，唯一「通过」的 "
                "-5.5% 是含堵孔期零窗的旧协议产物；预注册 6 例只落地 1 例（§⑤a/b 空表、power_law={}），"
                "库里该锚 0 行裁判、报告点名的 D5 无工件。cj2578 重放逐位复现 filed W。判 L2，L4 拒绝。"),
    "scenario_key": "CASE-DEM-SILO-BEVERLOO",
    "mission_id": 20232,
    "tags": ["board-audit", "TensorDEM", "DEM", "silo", "Beverloo", "maturity-L2", "honest-FAIL",
             "cross-check", "m20055"],
    "markdown": md,
    "figures": [
        fig("fig_m20055_rate_vs_time.png",
            "m20055 D4（D/d=4）：DEM 卸料率 0.25 s 分箱 vs Beverloo 参考 1.4545 kg/s。箱间起伏 ±12.6%，"
            "最长 ±5% 平台 0.75 s —— 「稳态」并不干净，但 +25% 偏离 = 7.5σ，不是噪声。"),
        fig("fig_m20055_estimators.png",
            "同一原始卸料记录、8 种估计器的相对误差。7 种（5 个分位带 + 全流均值 + 最长平台）落在 "
            "+19.6…+28.8%；唯一在容差内的 −5.5% 是「含堵孔期零窗」的旧固定窗均值，属协议缺陷产物。"),
    ],
}
res = post("/tensor/reports", body, token)
print(json.dumps(res, ensure_ascii=False)[:400])