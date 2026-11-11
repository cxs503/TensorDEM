"""Post the p227w2-N04 report (mission 11181) to the TensorPlatform report API.

Payload is written to a mission-unique path on purpose: a fixed ``/tmp/report.json``
is shared by every concurrent agent on this host and silently posts somebody
else's report.

    python post_report_m11181.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "benchmarks" / "ice_pileup_iso19906" / "results"
DOC = ROOT / "docs" / "p227w2_N04_vertical_structure_pileup.md"
PAYLOAD = Path("/tmp/report_m11181.json")

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
if not base.endswith("/api/v1"):
    base += "/api/v1"


def login() -> str:
    data = json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    req = urllib.request.Request(base + "/auth/login", data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    tok = body.get("access_token") or body.get("token")
    if not tok:
        raise SystemExit(f"login did not return a token: {body}")
    return tok


def build_payload() -> dict:
    rep = json.loads((RES / "report.json").read_text())
    val = rep["validation"]
    por = rep.get("porosity_scaling") or {}
    sig = rep.get("sigma_scaling") or {}
    base_rows = rep["base_cases"]["iso_rows"]

    figures = []
    for fig in rep["figures"]:
        png = RES / fig["name"]
        if png.exists():
            figures.append({
                "name": fig["name"],
                "caption": fig["caption"],
                "png_base64": base64.b64encode(png.read_bytes()).decode(),
            })

    summary = (
        "2-D 平面视图 DEM 模拟冰排向垂直结构漂移挤压堆积，输出载荷时历："
        + "; ".join(f"{r['case_id']} F_max/F_ISO={r['ratio_max']:.3f}, "
                    f"F_steady/F_ISO={r['ratio_steady']:.3f}" for r in base_rows)
        + f"; 孔隙率-阻力标度律 α={por.get('alpha', float('nan')):.3f}, "
          f"R²={por.get('R2', float('nan')):.3f}（{por.get('n_points', 0)} 算例）"
        + f"; σ_c 标度指数 {sig.get('exponent', float('nan')):.2f}"
    )
    return {
        "title": "冰体挤压堆积对垂直结构载荷（p227w2-N04）：DEM 载荷时历 + ISO 19906 对标 + "
                 "孔隙率-阻力标度律",
        "summary": summary,
        "scenario_key": "ice_vertical_structure_pileup",
        "mission_id": 11181,
        "tags": ["TensorDEM", "p227w2-N04", "ISO19906", "ice_load", "DEM",
                 "porosity_scaling", "ice_structure_interaction"],
        "markdown": DOC.read_text(),
        "figures": figures,
        "validation": {
            "C1_iso_same_order_of_magnitude": val["C1_iso_same_order_of_magnitude"],
            "C1b_iso_within_factor_3": val["C1b_iso_within_factor_3"],
            "C2_porosity_scaling": val["C2_porosity_scaling"],
            "C3_sigma_c_linearity": val["C3_sigma_c_linearity"],
        },
    }


def main() -> int:
    payload = build_payload()
    PAYLOAD.write_text(json.dumps(payload))
    tok = login()
    req = urllib.request.Request(base + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        print(resp.read().decode()[:2000])
    print(f"posted mission 11181 report ({len(payload['figures'])} figures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())