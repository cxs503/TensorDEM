"""Publish the m20873 progress report for CASE-COLUMN-COLLAPSE (deep refinement)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "docs" / "artifacts" / "column_collapse_deep_refinement_m20873"
DOC = HERE / "docs" / "CASE_COLUMN_COLLAPSE_DEEP_REFINEMENT_m20873.md"

API = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

FIGURES = [
    ("fig_refinement_ladders.png",
     "Runout-law error on the two refinement ladders (a=1: 8/12/16 grains per column radius; "
     "a=3: 8/12). Grey band = +/-15 % acceptance. The a=3, 12-grains point sits 0.40 pp "
     "outside the band; the archived a=2 deep points are drawn dashed."),
    ("fig_anchor_coverage.png",
     "Anchor coverage: runout error vs measured aspect ratio over every resolution the anchor "
     "has run. Crosses = archived m20337 ladder, circles = re-integrated here. The single "
     "out-of-band point is the deep, high-resolution one."),
    ("fig_time_vs_resolution.png",
     "Emplacement time t90 error on the same ladders with the frozen 90 % arrival definition: "
     "all five points lie within 3 % of 3*sqrt(h_i/g). The archived a=2 points use the 0.95 "
     "definition and are deliberately not mixed in."),
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
    md = DOC.read_text()
    figures = []
    for name, caption in FIGURES:
        png = base64.b64encode((ART / name).read_bytes()).decode()
        figures.append({"name": name, "caption": caption, "png_base64": png})
    body = {
        "title": "CASE-COLUMN-COLLAPSE · 两条细化缺口闭合并给出深端反例（m20873）",
        "summary": ("补齐 m18261 申报的两格细化（a=1 第三档、a=3 深端）：a=1 三档全在 15 % 带内"
                    "（漂移 2.40 pp），a=3 的 12 粒/半径点 +15.40 % 首次出带（P3/P6 fail）；"
                    "埋置时间律在 a>=1 上全部 <3 %。"),
        "scenario_key": "CASE-COLUMN-COLLAPSE",
        "mission_id": 20873,
        "tags": ["TensorDEM", "granular-column-collapse", "Lube2004", "refinement",
                 "verification-debt", "CASE-COLUMN-COLLAPSE", "honest-fail"],
        "markdown": md,
        "figures": figures,
    }
    payload = json.dumps(body)
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/tensor/reports",
         "-H", f"Authorization: Bearer {token()}",
         "-H", "Content-Type: application/json", "-d", payload],
        capture_output=True, text=True, check=True).stdout
    print(out[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())