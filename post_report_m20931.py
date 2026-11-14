"""Publish the m20931 progress report for CASE-COLUMN-COLLAPSE (a=3 third rung)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "docs" / "artifacts" / "column_collapse_sqrt_branch_m20931"
DOC = HERE / "docs" / "CASE_COLUMN_COLLAPSE_SQRT_BRANCH_m20931.md"
API = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
FIGURES = [
    ("fig_a3_ladder.png",
     "a=3 square-root branch refinement ladder (8/12/16 grains per column radius): "
     "runout ratio vs resolution with the Lube (2004) law evaluated at the measured aspect "
     "ratio (left) and the relative error against the +/-15 % acceptance band (right)."),
    ("fig_branch_comparison.png",
     "Branch asymmetry: error trajectories of the linear (a=1) and square-root (a=3) ladders "
     "on the same refinement axis."),
    ("fig_increments.png",
     "Step-to-step increments of the runout ratio (contraction test); |delta_23|/|delta_12| = 0.62."),
]


def token() -> str:
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/auth/login",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"username": USER, "password": PASS})],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return data.get("access_token") or data.get("token") or data["data"]["access_token"]


def main() -> int:
    verdict = json.loads((ART / "verdict.json").read_text())
    v = verdict["verdict"]
    figs = [{"name": n, "caption": c,
             "png_base64": base64.b64encode((ART / n).read_bytes()).decode()}
            for n, c in FIGURES]
    body = {
        "title": "CASE-COLUMN-COLLAPSE · 平方根支 a=3 补齐第三细化档：15.40 % 超线系离散误差（m20931）",
        "summary": (f"补 m20873 申报未做的 a=3 第三档（gpr=16, N=48923, 1787 s, cj3056）："
                    f"跑距误差 +11.06 % 回到 15 % 带内，增量收缩比 {v['abs_delta_ratio']:.2f}"
                    f"（H1 → pass）；Richardson 括号 p=1 → -0.59 % / p=2 → +6.07 %（仅作趋势）。"
                    f"t90 误差 -6.38 %。诚实限定：三点阶梯非单调，不声称已收敛。"),
        "scenario_key": "CASE-COLUMN-COLLAPSE",
        "mission_id": 20931,
        "tags": ["TensorDEM", "granular-column-collapse", "Lube2004", "refinement",
                 "verification-debt", "CASE-COLUMN-COLLAPSE", "adversarial-review"],
        "markdown": DOC.read_text(),
        "figures": figs,
    }
    tok = token()
    payload = json.dumps(body)
    with open("/tmp/report_m20931.json", "w") as fh:
        fh.write(payload)
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/tensor/reports",
         "-H", f"Authorization: Bearer {tok}",
         "-H", "Content-Type: application/json", "-d", "@/tmp/report_m20931.json"],
        capture_output=True, text=True).stdout
    print(out[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
