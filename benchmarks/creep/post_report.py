"""Publish the P46W3-N02 report to the platform API.

Builds the payload from ``artifacts/ice_creep_p46w3n02/results.json`` (captions)
and the generated Markdown report, then POSTs it with the six figures embedded
as base64 PNGs.  A unique payload path is used because /tmp/report.json is
shared by every concurrent agent on this host.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = REPO_ROOT / "artifacts" / "ice_creep_p46w3n02"
DOC = REPO_ROOT / "docs" / "P46W3_N02_ice_creep_andrade_fractional.md"
PAYLOAD = Path("/tmp/report_p46w3n02_m9150.json")

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"


def build_payload() -> dict:
    results = json.loads((ART / "results.json").read_text(encoding="utf-8"))
    figures = []
    for entry in results["figures"]:
        path = ART / entry["file"]
        figures.append({
            "name": entry["file"],
            "caption": entry["caption"],
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })
    summary = (
        "P46W3-N02 ice creep (TensorDEM): new fractional springpot Andrade "
        "primary-creep element + Mittag-Leffler relaxation. Glen n=2.9134 "
        "(2.89% vs 3); Andrade curve 5.8e-3 max rel.err; tau ~ A^-1 (slope "
        "-1.0000, r=-1.000) and ~ eta^(1/alpha) (3.000000, r=1.000); punch "
        "-1.0063, r=-0.999996. Honest FAILs: statistical-CI item (lattice bias) "
        "and default fluidity 10x Barnes 1971; Cole/Schulson overlay WITHHELD."
    )
    assert len(summary) <= 500, len(summary)
    return {
        "title": "TensorDEM 冰蠕变本构 P46W3-N02：Andrade 分数阶初级蠕变、Mittag-Leffler 松弛与 Glen 指数验证面",
        "summary": summary,
        "scenario_key": "p46w3-n02-ice-creep-andrade",
        "mission_id": 9150,
        "tags": ["TensorDEM", "ice-creep", "Glen", "Norton", "Andrade",
                 "fractional-calculus", "Mittag-Leffler", "stress-relaxation",
                 "literature-anchor", "honest-FAIL"],
        "markdown": DOC.read_text(encoding="utf-8"),
        "figures": figures,
    }


def main() -> None:
    login = json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    request = urllib.request.Request(
        BASE + "/auth/login", data=login,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit(f"login did not return a token: {body}")

    payload = build_payload()
    PAYLOAD.write_text(json.dumps(payload), encoding="utf-8")
    request = urllib.request.Request(
        BASE + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token}, method="POST")
    with urllib.request.urlopen(request, timeout=300) as response:
        print(response.read().decode())


if __name__ == "__main__":
    sys.exit(main())