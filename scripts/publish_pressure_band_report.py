"""Publish the P204-N08 report to the TensorPlatform reports API.

Builds the payload from the campaign artifacts (numbers injected from JSON, no
hand-typing), downsamples the figures so the request stays well below the
250 kB body limit, logs in with the agent credentials and POSTs the report.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

from PIL import Image

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
MISSION = 6926
OUT = Path("artifacts/pressure_band_lambda")
MAX_W = 760
MAX_BYTES = 22000


def login() -> str:
    body = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["access_token"]


def shrink(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    scale = min(1.0, MAX_W / img.width)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    for colors in (128, 96, 64, 48, 32):
        buf = io.BytesIO()
        img.quantize(colors=colors, method=Image.MEDIANCUT).save(
            buf, format="PNG", optimize=True)
        if buf.tell() <= MAX_BYTES:
            break
    return base64.b64encode(buf.getvalue()).decode()


def build() -> dict:
    report = json.loads((OUT / "report.json").read_text())
    dem = json.loads((OUT / "dem_runs.json").read_text())
    md = (OUT / "report.md").read_text()
    figs = []
    for item in report["figures"]:
        p = OUT / "figs" / item["name"]
        figs.append({"name": item["name"], "caption": item["caption"],
                     "png_base64": shrink(p)})
    c1 = report["criteria"]["C1_bias_model_rms_le_0.03"]
    c3 = report["criteria"]["C3_validation_3seeds_3L_within_5pct"]
    c5 = report["criteria"].get("C5_dem_lambda_over_h", {})
    summary = (
        f"λ_x 标定：偏差模型 RMS {max(c1['value'][m] for m in ('integral','expfit')):.3f}"
        f"（integral/expfit 可标定，spectral 不可）；外样本门槛 K*="
        f"{report['recommended_k_star']['spectral']:.0f}，独立确认 pooled 偏差 "
        f"{c3['value']['integral']['pooled_mean_bias']:+.3f}/{c3['value']['expfit']['pooled_mean_bias']:+.3f}"
        f"（均 <5%）；长域 DEM λ_x/h={c5['value']['mean']:.3f}，不在文献带 1.5–3（honest FAIL，"
        f"粗化阶梯证明是白场/沉积核尺度）；方差标度 Var∝λ/L 未被确认（honest FAIL）。"
    )
    summary = summary[:490]
    return {
        "title": "P204-N08 λ_x 原生标定：有限窗偏差修正、记录长度门槛与长域 DEM 压力带（TensorDEM）",
        "summary": summary,
        "scenario_key": "P204-N08",
        "mission_id": MISSION,
        "tags": ["ice-pressure", "correlation-length", "lambda_x", "DEM", "bias-correction",
                 "TensorDEM", "P204-N08"],
        "markdown": md,
        "figures": figs,
        "metadata": {
            "criteria": report["criteria"],
            "recommended_k_star": report["recommended_k_star"],
            "artifacts": report["artifacts"],
        },
    }


def main() -> None:
    token = login()
    payload = build()
    body = json.dumps(payload).encode()
    print("payload bytes", len(body))
    req = urllib.request.Request(
        f"{BASE}/tensor/reports?mission_id={MISSION}", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            print(r.status, r.read()[:400].decode(errors="replace"))
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read()[:600].decode(errors="replace"))
        sys.exit(1)


if __name__ == "__main__":
    main()