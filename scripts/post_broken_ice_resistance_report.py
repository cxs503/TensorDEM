#!/usr/bin/env python
"""Post the p247w1-N07 progress report from the campaign artifacts.

Every number in the payload is read from
``artifacts/broken_ice_resistance_validation/`` (cases.json / criteria.json),
never typed by hand; the figures are the PNGs written by the same campaign and
are embedded as base64.

    TENSOR_API_BASE=... TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... \
        python scripts/post_broken_ice_resistance_report.py [payload.json]
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "broken_ice_resistance_validation"
DOC = ROOT / "docs" / "BROKEN_ICE_RESISTANCE_VALIDATION.md"
MISSION_ID = 9135

FIGURE_CAPTIONS = {
    "fig_time_history.png": "Resistance time history of the standard case "
                            "(V = 3 m/s, h = 1.0 m, C = 0.6) against the contact-law "
                            "and hull-form baselines, with the Lindqvist level-ice value.",
    "fig_speed_sweep.png": "Speed sweep 2-5 m/s against the Lindqvist and broken-ice "
                           "closed-form references.",
    "fig_thickness_sweep.png": "Thickness sweep 0.5-1.5 m against Lindqvist.",
    "fig_criteria.png": "Criterion values with verdicts (green PASS, red FAIL, grey WITHHELD).",
    "fig_sensitivity.png": "Contact law / hull form / floe size sensitivity.",
}


def build_payload() -> dict:
    cases = json.loads((ART / "cases.json").read_text())
    criteria = json.loads((ART / "criteria.json").read_text())
    standard = cases["cases"]["standard"]
    c1 = criteria["criteria"]["C1"]
    verdicts = {k: v["verdict"] for k, v in criteria["criteria"].items()}
    figures = []
    for path in sorted(ART.glob("*.png")):
        figures.append({
            "name": path.name,
            "caption": FIGURE_CAPTIONS.get(path.name, path.name),
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })
    summary = (
        "DEM 碎冰场直航冰阻力验证面建成：plan-view 浮冰 DEM + 船形首轮廓 + 压碎限幅"
        f"接触律，标准工况 R_ice = {standard['mean_kN']:.1f} kN vs Lindqvist "
        f"{c1['lindqvist_kN']:.1f} kN（{100 * c1['value']:+.1f}%），C1 诚实 FAIL；"
        f"趋势判据 C2/C3/C5/C6/C7 = {sum(1 for v in verdicts.values() if v == 'PASS')} PASS，"
        "C4 因仓内无碎冰实验数据 WITHHELD。"
    )
    markdown = DOC.read_text() if DOC.exists() else summary
    return {
        "title": "p247w1-N07 DEM broken-ice direct-navigation resistance: validation surface "
                 "(MT Uikku anchor + Lindqvist cross-check)",
        "summary": summary,
        "scenario_key": "dem_broken_ice_resistance_validation",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "broken ice", "DEM", "ice resistance", "Lindqvist",
                 "MT Uikku", "validation surface", "honest FAIL"],
        "markdown": markdown,
        "figures": figures,
        "verdicts": verdicts,
        "standard_case_kN": standard["mean_kN"],
        "lindqvist_kN": c1["lindqvist_kN"],
        "ledger_sha256": cases["ledger_sha256"],
    }


def post(payload: dict) -> dict:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    if not base.endswith("/api/v1"):
        base = base.rstrip("/") + "/api/v1"
    login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    request = urllib.request.Request(base + "/auth/login", data=login,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit(f"login did not return a token: {body}")
    request = urllib.request.Request(
        base + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    payload = build_payload()
    if payload["mission_id"] != MISSION_ID:  # pragma: no cover - defensive
        raise SystemExit("payload mission mismatch")
    if argv:
        Path(argv[0]).write_text(json.dumps(payload, indent=1))
        print(f"payload written to {argv[0]}")
        if len(argv) == 1:
            return 0
    print(json.dumps(post(payload))[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())