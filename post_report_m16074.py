#!/usr/bin/env python3
"""Post the CASE-DEM-BRAZIL-NUT-SEGREGATION report to the tensor platform.

Everything in the payload is read from the generated artefacts
(``artifacts/case_brazil_nut_moebius2001/result.json`` and ``report.md``); there
is no hand-written number in this file.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts", "case_brazil_nut_moebius2001")

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not base.endswith("/api/v1"):
    base = base.rstrip("/") + "/api/v1"

with open(os.path.join(OUT, "result.json")) as fh:
    res = json.load(fh)
with open(os.path.join(OUT, "report.md")) as fh:
    markdown = fh.read()

figures = []
for f in res["figures"]:
    path = os.path.join(OUT, f["name"])
    if not os.path.exists(path):
        continue
    with open(path, "rb") as fh:
        figures.append({"name": f["name"], "caption": f["caption"],
                        "png_base64": base64.b64encode(fh.read()).decode()})

v = res["verdicts"]
fit = res["density_fit"]
summary = (
    f"巴西坚果尺寸偏析（Moebius 2001 Nature 414:270, Fig.1）：以 D/d=8、Γ=5 的几何相似 2D DEM "
    f"跑 {res['n_cases']} 个案例。H1 重颗粒上升 "
    f"{'通过' if v['H1_direction']['pass'] else '未通过'}；"
    f"H2 非单调峰位 rho*/rho_m="
    + (("%.3f" % fit["rho_star"]) if fit.get("rho_star") is not None else "n/a")
    + " vs ref 0.47-0.50 "
    f"{'通过' if v['H2_nonmonotonic']['pass'] else '未通过'}；"
    f"H3 尺寸依赖 {'通过' if v['H3_size']['pass'] else '未通过'}。"
    f"升速量级 {res['cases'][0]['rise_per_tap_over_d']:+.3f} d/tap（首例）。"
    "绝对值 T_rise 与 90 torr 气压分支判 WITHHELD / NOT_SUPPORTED 并如实记录。"
)

payload = {
    "title": "CASE-DEM-BRAZIL-NUT-SEGREGATION: 振动分选巴西坚果效应（TensorDEM）—— Moebius 2001 对照",
    "summary": summary,
    "scenario_key": "case-pool/dem-brazil-nut-segregation",
    "mission_id": 16074,
    "tags": ["TensorDEM", "Brazil-nut", "segregation", "granular", "vibration",
             "Moebius2001", "benchmark", "case-pool", "DEM"],
    "markdown": markdown,
    "figures": figures,
}

login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                    "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
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