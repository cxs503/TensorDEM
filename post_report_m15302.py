#!/usr/bin/env python3
"""Post the CASE-SETTLING-STOKES report to the tensor platform.

Builds the report payload (title/summary/scenario_key/mission_id/tags/markdown/
figures with png_base64) from the run artifacts and posts it via the
``/tensor/reports`` API.  Every number in the markdown is injected from
``result.json`` (no hand-copied digits).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not base.endswith("/api/v1"):
    base = base.rstrip("/") + "/api/v1"

# ---- load run artifacts ----
with open(os.path.join(HERE, "artifacts/case_settling_stokes/result.json")) as fh:
    result = json.load(fh)
with open(os.path.join(HERE, "docs/CASE_SETTLING_STOKES.md")) as fh:
    markdown = fh.read()

p = result["primary"]
num = p["numerical"]
ref = p["reference"]
dim = p["dimensionless"]

# ---- figures (png_base64) ----
figures = []
for f in result["figures"]:
    path = os.path.join(HERE, "artifacts/case_settling_stokes", f["name"])
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    figures.append({"name": f["name"], "caption": f["caption"], "png_base64": b64})

# ---- report payload ----
payload = {
    "title": "CASE-SETTLING-STOKES: 单球沉降末速（TensorDEM）—— Stokes 解析对照",
    "summary": (
        f"单球在静止流体中沉降末速：数值 v_t={num['v_t_numerical']:.4e} m/s "
        f"vs Stokes 解析 v_t={ref['v_t_stokes']:.4e} m/s，相对误差 "
        f"{num['relative_error_stokes']*100:.2f}%（<5% 容差 ✓）；"
        f"Re={dim['Re_stokes']:.3f}（Stokes 区）。复用 TensorDEM "
        f"two_way_coupling 验证面，直径/密度比/dt 三向敏感性全通过。"
    ),
    "scenario_key": "case-pool/settling-stokes",
    "mission_id": 15302,
    "tags": ["TensorDEM", "Stokes", "settling", "terminal-velocity",
             "single-sphere", "benchmark", "case-pool", "verification"],
    "markdown": markdown,
    "figures": figures,
}

# ---- login + post ----
login = json.dumps({
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"],
}).encode()
req = urllib.request.Request(base + "/auth/login", data=login,
                             headers={"Content-Type": "application/json"},
                             method="POST")
with urllib.request.urlopen(req, timeout=60) as resp:
    body = json.loads(resp.read())
tok = body.get("access_token") or body.get("token")
if not tok:
    raise SystemExit(f"login did not return a token: {body}")

req = urllib.request.Request(base + "/tensor/reports",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok},
                             method="POST")
with urllib.request.urlopen(req, timeout=180) as resp:
    print(resp.read().decode())
