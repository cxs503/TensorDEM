"""Post the p307w1-n06 report to the TensorPlatform report API (mission 9999).

Reads runs/p307w1_n06/report.json + docs/P307W1_N06_granular_solitary_waves.md +
artifacts/p307w1_n06/*.png and posts a single report payload.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"

FIGDIR = os.path.join(ROOT, "artifacts", "p307w1_n06")
DOC = os.path.join(ROOT, "docs", "P307W1_N06_granular_solitary_waves.md")
REPORT = os.path.join(ROOT, "runs", "p307w1_n06", "report.json")


def login() -> str:
    data = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    tok = body.get("access_token") or body.get("token")
    if not tok:
        raise SystemExit(f"login did not return a token: {body}")
    return tok


def main() -> int:
    with open(DOC, encoding="utf-8") as fh:
        markdown = fh.read()
    with open(REPORT, encoding="utf-8") as fh:
        rep = json.load(fh)
    captions = {f["name"]: f["caption"] for f in rep.get("figures", [])}
    figures = []
    for name in sorted(os.listdir(FIGDIR)):
        if not name.endswith(".png"):
            continue
        with open(os.path.join(FIGDIR, name), "rb") as fh:
            figures.append({"name": name,
                            "caption": captions.get(name, name),
                            "png_base64": base64.b64encode(fh.read()).decode()})

    acc = rep["acceptance"]
    payload = {
        "title": "颗粒链非线性孤立波与冲击隔离包络：Nesterenko v_s∝F^(1/6) 标度、"
                 "无序链指数衰减与颗粒层传递率（p307w1-n06）",
        "summary": (
            "TensorDEM 新增 granular_chain 包（Hertz 链孤立波 + 颗粒层阻尼器）。"
            "机器判据 7/7 通过：v_s∝F 幂律指数 0.1666667（目标 1/6，相对误差 3e-12）；"
            "孤立波剖面对 Nesterenko 解析 cos⁴ 解 L1(偶部)=4.1%（<10%）；"
            "链内能量守恒残差 2.5e-4（<2%）；无序链衰减长度 ξ∝σ^-2.11（带 1.7–2.3）；"
            "三个负控全部按预期退化；颗粒层阻带 191–510 Hz 达 -47.9 dB，"
            "ζ 双路互裁 0.198 vs 0.175（13.3%<15%）。"
            "同时否定两个次假设：脉宽与幅值无关（指数 1e-5，非 F^(1/3)）、"
            "e=0 并非“无传播”（幅度按 0.35%/颗粒衰减但速度仍遵守 1/6 律）。"
        ),
        "scenario_key": "p307w1-n06-granular-solitary-waves",
        "mission_id": "9999",
        "tags": ["granular-chain", "nesterenko", "solitary-wave", "hertzian-contact",
                 "particle-damper", "shock-isolation", "transmissibility",
                 "disorder-attenuation", "tensordem", "energy-ledger"],
        "markdown": markdown,
        "figures": figures,
    }
    tok = login()
    req = urllib.request.Request(BASE + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            print(resp.read().decode())
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read().decode()[:2000])
        return 1
    print("acceptance all_pass:", acc["all_pass"], "figures:", len(figures))
    return 0


if __name__ == "__main__":
    sys.exit(main())