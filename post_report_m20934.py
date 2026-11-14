"""Publish the m20934 progress report (CASE-DEM-SILO-BEVERLOO kn-sweep)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "runs" / "m20934_beverloo_kn"
DOC = HERE / "docs" / "CASE-DEM-SILO-BEVERLOO_m20934.md"

API = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

FIGURES = [
    ("fig_w_vs_dd.png",
     "Steady mass flow rate W vs D/d for each contact stiffness kn (markers = DEM, "
     "dashed = Beverloo 1961 curve with C=0.58, k=1.4 and the measured bed density). "
     "Flat-bottom silo, mu=0.30, mu_wall=0, settled disordered bed, seed 20055."),
    ("fig_err_vs_kn.png",
     "Relative error vs Beverloo as a function of kn (log axis). Dotted lines mark the "
     "+/-10 % ledger acceptance band. The kn=1e5 arm reproduces the m20055 FAIL; the "
     "kn=1e6 arm tests the m20585 hypothesis."),
    ("fig_cumulative_D5.png",
     "D5 (D/d=5) discharge kinetics: cumulative discharged particles vs time since "
     "orifice opening, kn=1e5 (blue) vs kn=1e6 (red)."),
]


def token() -> str:
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/auth/login",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"username": USER, "password": PASS})],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return data.get("access_token") or data.get("token") or data["data"]["access_token"]


def main() -> int:
    md = DOC.read_text()
    figures = []
    for name, caption in FIGURES:
        p = ART / name
        if not p.exists():
            print(f"missing figure {p}", flush=True)
            return 1
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(p.read_bytes()).decode()})
    body = {
        "title": "CASE-DEM-SILO-BEVERLOO · 平底料仓 Beverloo 律 kn 敏感性验证（m20934）",
        "summary": "（发布前由 render 步骤按 summary.json 回填）",
        "scenario_key": "CASE-DEM-SILO-BEVERLOO",
        "mission_id": 20934,
        "tags": ["TensorDEM", "silo-discharge", "beverloo-1961", "contact-stiffness",
                  "verification-debt", "CASE-DEM-SILO-BEVERLOO"],
        "markdown": md,
        "figures": figures,
    }
    tok = token()
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/tensor/reports",
         "-H", f"Authorization: Bearer {tok}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(body)],
        capture_output=True, text=True, check=True).stdout
    print(out[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
