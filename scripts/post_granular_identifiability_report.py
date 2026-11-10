#!/usr/bin/env python
"""Build and post the paper-grade progress report for this mission (9063).

The payload is generated from ``artifacts/granular_identifiability/study.json``
(no hand-copied numbers) and from ``docs/granular_parameter_identifiability.md``
(markdown body); every figure is embedded as base64 PNG.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

STUDY = "artifacts/granular_identifiability/study.json"
DOC = "docs/granular_parameter_identifiability.md"
MISSION_ID = 9063


def _login(base: str) -> str:
    payload = json.dumps(
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }
    ).encode()
    req = urllib.request.Request(
        base + "/auth/login", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit(f"login did not return a token: {body}")
    return token


def build_payload() -> dict:
    with open(STUDY) as handle:
        report = json.load(handle)
    with open(DOC) as handle:
        markdown = handle.read()
    figures = []
    for fig in report.get("figures", []):
        with open(fig["path"], "rb") as handle:
            png = base64.b64encode(handle.read()).decode()
        figures.append({"name": fig["name"], "caption": fig["caption"], "png_base64": png})
    verdicts = report["verdicts"]
    passed = report["all_criteria_pass"]
    summary = (
        "颗粒 DEM 接触参数（μ/e/k_n）可辨识性与反演验证面："
        f"dt 减半目标量最大变化 {max(v['max_relative_change'] for v in report['discretisation']['changes'].values()) * 100:.3f}%，"
        f"反演最大回收率 {report['inversion']['max_recovery_error'] * 100:.2f}%，"
        f"FIM 条件数 {report['fim']['condition_number']:.2f}"
        f"（仅体相观测量时 rank={report['fim']['bulk_only']['rank']}/3，秩亏），"
        f"±10% 线性响应最小 R² = {report['propagation']['min_r2']:.4f}；"
        f"六项判据{'全部通过' if passed else '未全部通过（见诚实 FAIL 表）'}。"
    )
    return {
        "title": "P189-N02: 颗粒 DEM 接触参数可辨识性与反演验证面（test_granular_01）",
        "summary": summary,
        "scenario_key": "granular_contact_parameter_identifiability",
        "mission_id": MISSION_ID,
        "tags": [
            "TensorDEM", "DEM", "identifiability", "inverse_problem",
            "fisher_information", "uncertainty_propagation", "P189-N02",
        ],
        "markdown": markdown,
        "figures": figures,
        "artifacts": [
            STUDY, DOC,
            "src/tensordem/granular_identifiability/",
            "scripts/run_granular_identifiability.py",
            "scripts/render_granular_identifiability_doc.py",
            "tests/test_granular_identifiability.py",
        ],
        "verdicts": {
            key: {"pass": value["pass"], "value": value.get("value"),
                  "threshold": value.get("threshold")}
            for key, value in verdicts.items()
        },
    }


def main() -> int:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    if not base.endswith("/api/v1"):
        base = base.rstrip("/") + "/api/v1"
    payload = build_payload()
    out = f"/tmp/report_{MISSION_ID}.json"
    with open(out, "w") as handle:
        json.dump(payload, handle)
    assert payload["mission_id"] == MISSION_ID
    token = _login(base)
    req = urllib.request.Request(
        base + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        print(resp.read().decode())
    print(f"payload: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())