#!/usr/bin/env python
"""Post the ridge-mobility report (markdown + embedded figures) to TensorPlatform."""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "ridge_mobility"
DOC = ROOT / "docs" / "ridge_mobility_criterion.md"
MISSION_ID = 9430

FIGURES = [
    ("fig1_dem_keel_state.png",
     "图1 左：搁浅冰脊 DEM 截面（灰线=冻结键骨架，点=颗粒，蓝线=水线，棕线=海床）；"
     "右：自由漂浮平衡，自相似棱柱 φ=(t/h_k)^2 与阿基米德线 ρi/ρw 的交点给出闭式吃水比，"
     "DEM 自由漂浮点（面积/几何两种度量）落于其上。"),
    ("fig2_normal_force_vs_depth.png",
     "图2 触底法向力 N 随相对水深 H/h_k 的变化：闭式曲线在 H=t_ff 处归零，DEM 点落在曲线上。"),
    ("fig3_mu_star_boundary.png",
     "图3 临界海床摩擦 μ*(h_k/d)：黑线=闭式力平衡，蓝点=DEM 边界（标注相对差），绿带=文献 4-6 带。"),
    ("fig4_mobility_map.png",
     "图4 (h_k/d, μ_b) 可移动地图：绿区可被流/船推走，红区不可移动，圆点=DEM 扫掠分类（h_k/d=5）。"),
]


def main() -> int:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    if not base.endswith("/api/v1"):
        base = base.rstrip("/") + "/api/v1"
    login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=login,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    tok = body.get("access_token") or body.get("token")
    if not tok:
        raise SystemExit(f"login failed: {body}")

    metrics = json.loads((ART / "metrics.json").read_text())
    crit = metrics["criteria"]
    figures = []
    for name, caption in FIGURES:
        png = (ART / name).read_bytes()
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(png).decode()})
    markdown = DOC.read_text()
    payload = {
        "title": "搁浅冰脊可移动性判据：DEM 龙骨触底静力平衡与 (h_k/d, μ_b) 可移动边界",
        "summary": (
            "TensorDEM 新增 ridge_mobility：三态判据（漂浮/触底/可移动）+ 可承载极限 "
            "R_max=μ_b N+c_a b + 临界摩擦 μ*(h_k/d)，闭式与 2-D 颗粒 DEM 双实现。"
            f"机器判据 C1={crit['C1']} C2={crit['C2']} C3={crit['C3']} "
            f"C4={crit['C4']} C5={crit['C5']}。"),
        "scenario_key": "ice_ridge_grounded_mobility",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "ice ridge", "grounding", "mobility", "DEM",
                 "seabed friction", "p273w3-n08"],
        "markdown": markdown,
        "figures": figures,
    }
    path = Path("/tmp/report_m9430.json")
    path.write_text(json.dumps(payload))
    req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        print(resp.read().decode()[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())