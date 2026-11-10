"""Assemble and POST the P82W5-N07 progress report to the platform.

    python scripts/post_mts_speedup_gate_report.py

Reads ``artifacts/P82W5-N07/acceptance.json`` + the generated markdown + the PNG
figures and posts the six-element report (title/summary/scenario_key/mission_id/
tags/markdown/figures).  Nothing is retyped: the summary numbers come from the
JSON.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MISSION = "P82W5_N07"
MISSION_ID = 8125
ART = ROOT / "artifacts" / MISSION
DOC = ROOT / "docs" / f"{MISSION}_mts_speedup_gate.md"

FIGURES = [
    ("fig1_restitution_ladder.png", "判据 (a)：restitution 误差阶梯与时间步门的三个区间"),
    ("fig2_energy_ledger.png", "判据 (b)：1e5 步能量漂移与活跃接触比例"),
    ("fig3_speedup_ablation.png", "判据 (c) 成本半边：四变体墙钟与加速比"),
    ("fig4_jamming_density.png", "判据 (c) 物理半边：φ(P) 阶梯与 φ_J 对比"),
    ("fig5_cost_contract.png", "成本契约：加速比 = n_sub / 粗步成本比"),
]


def _post(path: str, payload: dict, token: str | None = None) -> dict:
    base = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
    url = f"{base}{path}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def login() -> str:
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    out = _post("/auth/login", {"username": user, "password": password})
    return out.get("access_token") or out.get("token") or out["data"]["access_token"]


def build_payload() -> dict:
    data = json.loads((ART / "acceptance.json").read_text())
    markdown = DOC.read_text()
    v = data.get("verdict", {})
    c1 = data.get("c1_speed", {})
    c2 = data.get("c2_jamming", {})
    a = data.get("a_restitution_ladder", {})
    b = data.get("b_energy_ledger", {})
    summary = (
        f"MTS 成熟化：摩擦 jamming(N=256, μ=0.5) 同物理时段墙钟 "
        f"{c1.get('speedup', float('nan')):.2f}×（改造前 {c1.get('speedup_legacy', float('nan')):.2f}×）；"
        f"φ_J 相对同引擎单速率 {100 * c2.get('phi_j_rel_diff', float('nan')):.3f}%；"
        f"restitution 最差 {100 * a.get('worst_error', float('nan')):.2f}%；"
        f"1e5 步能量漂移 {100 * (b.get('results', {}).get('mts', {}) or {}).get('energy_drift_max', float('nan')):.2f}%。"
        f"c-φ_J vs W4-P1 锚 honest FAIL({100 * (c2.get('phi_j_vs_anchor_rel_diff') or float('nan')):.2f}%, 引擎级偏移)。"
    )
    figures = []
    for name, caption in FIGURES:
        path = ART / name
        if not path.exists():
            continue
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(path.read_bytes()).decode()})
    return {
        "title": "P82W5-N07 多速率接触积分（MTS）时间步门：摩擦 jamming 2× 墙钟而 φ_J 不变（含 1 条 honest FAIL）",
        "summary": summary,
        "scenario_key": "tensordem.mts_speedup_gate",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "MTS", "DEM", "multi-rate", "jamming", "timestep-gate",
                 "P82W5-N07", "honest-fail"],
        "markdown": markdown,
        "figures": figures,
        "verdict": v,
        "artifacts": [str((ART / "acceptance.json").relative_to(ROOT))],
    }


def main() -> int:
    payload = build_payload()
    token = login()
    out = _post("/tensor/reports", payload, token=token)
    print(json.dumps(out)[:800])
    Path("/tmp/report_p82w5_n07.json").write_text(json.dumps(payload))
    print(f"figures: {len(payload['figures'])}; markdown chars: {len(payload['markdown'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())