#!/usr/bin/env python3
"""Post the m15418 board-audit report (CASE-HOPPER-BEVERLOO / m15381) to the platform.

Every number in the markdown comes from ``docs/board_audit/m15418/audit_m15381.json``
(rendered by ``render_m15418_md.py``); figures are read from disk and base64-embedded.

Usage:  python scripts/audit/post_report_m15418.py
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "board_audit" / "m15418"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

A = json.loads((OUT / "audit_m15381.json").read_text())
med, qck = A["C3_arithmetic_replay"][1], A["C3_arithmetic_replay"][0]
cl, tq = A["C6_claims"], A["C5_test_quality"]
rp = cl["repro"]

markdown = (OUT / "AUDIT.md").read_text() + f"""

---

### 附：本审计的可复现命令（全部在 worktree 内）

```
python scripts/audit/m15381_board_audit.py --audit-mission 15418 --insert-verdict
python scripts/audit/check_beverloo_density_basis.py \\
    docs/board_audit/m15418/refs/beverloo_medium_benchmark_seed42.json \\
    docs/board_audit/m15418/refs/beverloo_quick_validation_seed42.json
python scripts/audit/render_m15418_md.py
```

- 独立复现：`{rp.get('runner','')}`
  → 流率 `{rp.get('reproduced_flow_rate_kg_s', float('nan')):.10f}` kg/s vs 工件
  `{rp.get('filed_flow_rate_kg_s', float('nan')):.10f}` kg/s（绝对差 {rp.get('abs_diff_kg_s', float('nan')):.1e}），
  逐窗序列 `{[round(w,6) for w in rp.get('reproduced_windows', [])]}` 与工件完全相同，
  质量账残差 {rp.get('reproduced_mass_residual_kg', float('nan')):.2e} kg。
- 测试回放：`{tq['pytest_replay']['cmd']}` → rc={tq['pytest_replay']['rc']}，
  {tq['pytest_replay']['tail'][-1] if tq['pytest_replay']['tail'] else ''}；
  其中重算关联式的不变式断言 {tq['n_assert_that_recompute_the_correlation']} 条 /
  读工件字段 {tq['n_assert_on_stored_artifact_fields']} 条 / 总 {tq['n_assert_statements']} 条。
- 集群：`cluster_jobs` id=1100 (`--case full --seed 1`) 仍 `{cl['cluster_job_cj1100'].split('|')[1] if '|' in cl['cluster_job_cj1100'] else 'queued'}`
  → full 算例与收敛性论断记 **not_run**。
- 判级：L{A['level']}；anchor = **{A['anchor_conclusion']}**（一致密度基准 +{med['replay_consistent_density']['rel_err_pct']:.2f}%，
  recovered C={med['replay_consistent_density']['recovered_C']:.4f}）。
- verdict 行：`tensor_maturity_verdicts` id=303（repo=TensorDEM, scope='m15381 CASE-HOPPER-BEVERLOO',
  verdict_by='board-audit'），evidence 经 `maturity_gate.judge()` 自查 = L2 无缺项。
"""


def fig(name: str, caption: str) -> dict:
    with open(OUT / name, "rb") as fh:
        return {"name": name, "caption": caption,
                "png_base64": base64.b64encode(fh.read()).decode()}


figures = [
    fig("fig_anchor_check.png",
        "左：m15381 两个算例在『工件填报 ρ_b=0.64ρ_p』与『一致基准=实测床层 ρ_b』下的相对误差"
        "（绿带=±15% 容差）；右：medium 逐窗流率与两条参考线，第一窗 0.134 kg/s 决定 headline。"),
]
for src, cap in (
    ("refs/m15381_beverloo_comparison.png",
     "被审方原图（m15381 工件，逐字节副本）：柱状对比显示 −14.86% PASS——该 PASS 仅在 0.64ρ_p 基准下成立。"),
    ("refs/m15381_window_series.png", "被审方原图：逐窗流率序列（工件副本）。"),
):
    p = OUT / src
    if p.exists():
        with open(p, "rb") as fh:
            figures.append({"name": p.name, "caption": cap,
                            "png_base64": base64.b64encode(fh.read()).decode()})

payload = {
    "title": "交差审计：CASE-HOPPER-BEVERLOO / m15381（TensorDEM）— L2, anchor FAIL（密度基准错配）",
    "summary": (
        "m15381 的 DEM 卸料数字真实且逐位可复现（独立重跑流率 1.9704069123 kg/s 与工件相同），"
        f"但其 −14.86% PASS 源于参考值用 ρ_b=0.64ρ_p=1600 kg/m³ 而模拟床层仅 {med['packing']['rho_b_measured_kg_m3']:.1f} kg/m³"
        f"（φ={med['packing']['phi_measured']:.3f}，膨胀 ×{med['packing']['density_inflation_factor']:.2f}）；"
        f"一致基准下 rel_err=+{med['replay_consistent_density']['rel_err_pct']:.2f}%、recovered C={med['replay_consistent_density']['recovered_C']:.4f}"
        "（+34%，文献带 0.55–0.60 之外）→ anchor FAIL。判级 L2（无集成调用方、交付 0 commits）。"),
    "scenario_key": "dem_granular_silo_discharge_benchmark",
    "mission_id": 15418,
    "tags": ["board-audit", "TensorDEM", "beverloo", "silo-discharge",
             "benchmark-referee", "density-basis", "maturity-L2"],
    "markdown": markdown,
    "figures": figures,
}


def api(method, path, body=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


def main():
    st, d = api("POST", "/auth/login", {"username": USER, "password": PASS})
    tok = d.get("access_token") or d.get("token")
    if not tok:
        raise SystemExit(f"login failed: {st} {d}")
    st, d = api("POST", "/tensor/reports", payload, tok)
    print(st, json.dumps(d)[:400])
    (OUT / "report_post.json").write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()