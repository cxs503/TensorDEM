"""Publish the m17789 progress report for XVAL-DEM-RADON-PROJECTION-MEASUREMENT.

Reads the merged xval JSON + figure, renders the paper-style markdown (numbers
injected from the JSON, never typed), authenticates with the agent credentials
from the environment and POSTs to ``/tensor/reports``.

Usage::

    PYTHONPATH=src python publish_report_m17789.py
"""

from __future__ import annotations

import base64
import json
import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "benchmarks", "biaxial_shear_band"))
from render_radon_xval_doc import render                # noqa: E402

API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "agent")
PASSWORD = os.environ.get("TENSOR_AGENT_PASS", "")
MISSION_ID = int(os.environ.get("MISSION_ID", "17789"))

JSON_PATH = "artifacts/biaxial_shear_band/radon_projection_xval.json"
FIG_PATH = "artifacts/biaxial_shear_band/radon_projection_xval.png"


def token() -> str:
    resp = requests.post(f"{API}/auth/login",
                         json={"username": USER, "password": PASSWORD}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("access_token") or data.get("token")


def main() -> int:
    with open(JSON_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    body = render(payload)
    summary = payload["summary"]["radon_production"]
    verdict = payload["verdict"]
    markdown = body.split("## ① Case introduction", 1)
    markdown = "## ① 案例介绍" + (markdown[1] if len(markdown) > 1 else body)
    with open(FIG_PATH, "rb") as fh:
        png = base64.b64encode(fh.read()).decode()

    report = {
        "title": "XVAL-DEM-RADON-PROJECTION-MEASUREMENT: Radon 投影剪切带测量算子"
                 "已知角度交叉验证（TensorDEM m17789）",
        "summary": (
            f"{summary['n']} 个已知角度注入算例"
            f"（{len(payload['design']['lattices'])} lattice × "
            f"{len(payload['design']['widths_d'])} band width × "
            f"{len(payload['design']['angles'])} angle × "
            f"{len(payload['design']['noise_fractions'])} noise）"
            f"复核 measure_band 的 Radon 投影算子：无噪声偏差 "
            f"{summary['zero_noise']['bias_deg']:+.2f}°、最大绝对误差 "
            f"{summary['zero_noise']['max_abs_deg']:.2f}°；测量噪声包络 "
            f"{100 * summary['noise_envelope_fraction']:g}% d 内满足 15% 判据："
            f"{verdict['pass_precision_15pct_within_envelope']}；"
            f"修复前逐轴归一化基线在 24×48 算例上偏差 "
            f"{payload['summary']['radon_legacy_peraxis']['per_lattice']['24x48']['bias_deg']:+.1f}°。"
        ),
        "scenario_key": "tensordem.shear_band_measurement_xval",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "XVAL-DEM-RADON-PROJECTION-MEASUREMENT",
                 "shear-band", "Radon-projection", "cross-validation",
                 "measurement-error", "DEM"],
        "markdown": markdown,
        "figures": [{
            "name": "radon_projection_xval.png",
            "caption": "左：24×48 算例格点上测得 vs 真实倾角（带宽 1.5 d，无噪声），"
                       "生产算子与修复前基线对照 ±15% 导引线；右：各倾角区间的平均绝对误差"
                       "（全设计格），修复前算子的 45° 吸附即逐轴归一化偏置的指纹。",
            "png_base64": png,
        }],
    }
    tok = token()
    resp = requests.post(f"{API}/tensor/reports", json=report, timeout=120,
                         headers={"Authorization": f"Bearer {tok}"})
    print(resp.status_code, resp.text[:800])
    return 0 if resp.status_code < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())