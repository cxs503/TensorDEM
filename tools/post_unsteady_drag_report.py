"""Publish the IN71-N11 progress report to the platform report API.

Builds the payload programmatically from ``docs/unsteady_drag_n11_report.md`` and
the PNG figures in ``artifacts/unsteady_drag_n11/`` (base64 inline), then posts to
``$TENSOR_API_BASE/api/v1/tensor/reports`` with a freshly minted token.

Usage: ``PYTHONPATH=src python tools/post_unsteady_drag_report.py``
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_MD = ROOT / "docs" / "unsteady_drag_n11_report.md"
FIG_DIR = ROOT / "artifacts" / "unsteady_drag_n11"
VALIDATION = FIG_DIR / "validation.json"
MISSION_ID = 9272


def main() -> None:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    if not base.endswith("/api/v1"):
        base = base.rstrip("/") + "/api/v1"

    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    markdown = REPORT_MD.read_text(encoding="utf-8")

    captions = {f["name"]: f["caption"] for f in validation["figure_captions"]}
    figures = []
    for name, caption in captions.items():
        path = FIG_DIR / name
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })

    payload = {
        "title": "IN71-N11 非定常曳力最小闭合面成熟化：附加质量 + Basset 历史力解析锚与 O(N) 递推",
        "summary": (
            "TensorDEM 交付四项可分离非定常曳力账本（准稳态/附加质量/Basset/压力梯度）+ "
            "O(N) 指数核递推 + POST /api/v1/dem/unsteady_drag；判据全绿："
            f"附加质量高频渐近偏差 {validation['a']['asymptote_rel_error']*100:.2f}%，"
            f"振荡流幅值误差 max {max(r['amplitude_rel_error'] for r in validation['c']['rows'])*100:.2f}%、"
            f"相位误差 max {max(r['phase_error_deg'] for r in validation['c']['rows']):.2f}°，"
            f"递推 vs 直接卷积 L2 {validation['d']['terms'][2]['l2_rel_error']*100:.4f}%、"
            f"加速 {validation['d']['terms'][2]['speedup_cpu_vs_direct']:.1f}×，"
            f"多球稀释极限外推 {validation['f']['extrapolation_error']*100:.3f}%。"
        ),
        "scenario_key": "dem.unsteady_drag",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "unsteady-drag", "Basset", "added-mass",
                 "O(N)-recursion", "Landau-Lifshitz", "IN71-N11"],
        "markdown": markdown,
        "figures": figures,
    }

    tmp = Path("/tmp/report_m9272.json")
    tmp.write_text(json.dumps(payload), encoding="utf-8")

    login = json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    req = urllib.request.Request(base + "/auth/login", data=login,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit(f"login did not return a token: {body}")

    req = urllib.request.Request(
        base + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        print(resp.read().decode())


if __name__ == "__main__":
    main()