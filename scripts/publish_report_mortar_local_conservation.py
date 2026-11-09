#!/usr/bin/env python
"""Publish the p423w2-n07 report to the platform (TensorPlatform /tensor/reports).

Reads the generated markdown and the PNG figures, base64-embeds them, and posts
with the agent credentials from the environment.  Nothing is typed by hand: the
summary numbers are taken from ``acceptance.json``.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACC = ROOT / "artifacts" / "P423W2-N07_mortar_local_conservation" / "acceptance.json"
DOC = ROOT / "docs" / "P423W2-N07_mortar_local_conservation.md"

CAPTIONS = {
    "per_patch_conservation.png":
        "逐面片守恒缺陷 vs h：真交集（左）恒在 1e-12 门之下；面片投影（右）在 4× 错配下 O(1e-1)。",
    "fold_geometry_and_defect.png":
        "30° 折线界面（m6114 几何）：非匹配两侧与 2× 错配下的逐面片缺陷（真交集 vs 面片投影）。",
    "energy_account.png":
        "时域界面功率 |P_int|（左，理想为 0）与总能量历史（右）：破度量/破局部守恒的负控清晰可分。",
    "gap_penetration_balance.png":
        "间隙/渗透体积记账（左）与逐面片恒等式残差（右）：关掉间隙项后残差回到 1e-1 量级（负控）。",
    "transpose_consistency.png":
        "曲面度量下转置（伴随）输运缺陷 vs h：二阶重构 p≈2，中点转置（负控）明显更差。",
}


def main() -> int:
    acc = json.loads(ACC.read_text())
    c = acc["criteria"]
    A, B, C, D, E, F = (c["A_per_facet_conservation"], c["B_true_intersection_vs_projection"],
                        c["C_gap_penetration_terms"], c["D_time_domain_energy_account"],
                        c["E_transpose_consistency"], c["F_transfer_order_regression"])
    figdir = ACC.parent / "figures"
    figures = []
    for png in sorted(figdir.glob("*.png")):
        figures.append({"name": png.name,
                        "caption": CAPTIONS.get(png.name, png.stem.replace("_", " ")),
                        "png_base64": base64.b64encode(png.read_bytes()).decode()})
    summary = (
        f"逐面片局部守恒 max={A['max_per_facet']:.2e}（{A['n_facets_gated']} 面片，含 30° 折线与 κL=2.4 高曲率）；"
        f"4× 错配真交集 {B['true_max_4x']:.2e} vs 面片投影 {B['plane_per_facet_min_4x']:.2e}..{B['plane_per_facet_max_4x']:.2e}；"
        f"间隙/渗透恒等式 {C['max_identity_balance_rel']:.2e}、伪体积 {C['conforming_spurious_volume_rel']:.2e}、关项负控 {C['control_min']:.2e}；"
        f"时域界面功率 {D['max_interface_power_rel']:.2e}（负控 {D['control_min_interface_power_rel']:.2e}）；"
        f"转置对称 {E['symmetry_rel_max']:.2e}、伴随输运阶 {E['min_rate_transport_curved']:.2f}..{E['max_rate_transport_curved']:.2f}；"
        f"m6114 D3 阶 {F['field_order']:.2f}/误差 {F['field_accuracy']:.2e}；确定性逐位一致={c['G_determinism']['bitwise_identical']}；"
        f"all_pass={acc['all_pass']}，git_head={acc['provenance']['git_head'][:8]}"
    )
    payload = {
        "title": "P423W2-N07 曲面非匹配界面的逐面片局部守恒：真交集、30° 折线界面与时域耦合能量账",
        "summary": summary,
        "scenario_key": "mortar_local_conservation_curved",
        "mission_id": "8403",
        "tags": ["TensorDEM", "mortar", "non-matching interface", "per-facet conservation",
                 "true intersection", "crease interface", "energy account",
                 "transpose consistency", "P423W2-N07"],
        "markdown": DOC.read_text(),
        "figures": figures,
    }
    Path("/tmp/report.json").write_text(json.dumps(payload))
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
    req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        print(resp.read().decode()[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())