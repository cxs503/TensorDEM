"""Publish the m23318 reopen-verification report for TensorDEM#1657 (m20791)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "reports" / "report_m23318_verification.json"
FIG = HERE / "benchmarks" / "plane_shear_mu_i" / "runs" / "phi_audit" / "fig_phi_definition_audit.png"

API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def token() -> str:
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/auth/login",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"username": USER, "password": PASS})],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return data.get("access_token") or data.get("token") or data["data"]["access_token"]


def main() -> None:
    payload = json.loads(REPORT.read_text())
    figures = []
    if FIG.exists():
        figures.append({
            "name": FIG.name,
            "caption": ("Left: archived phi under both definitions vs the reference "
                        "band. Middle: measured definitional gap vs "
                        "plate_z/(plate_z-2r) (points) with synthetic ghost packings. "
                        "Right: mu(I) unchanged."),
            "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
        })
    payload["figures"] = figures
    tok = token()
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/tensor/reports",
         "-H", f"Authorization: Bearer {tok}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True)
    print("REPORT_PUBLISH:", out.stdout[:800])
    print("STDERR:", out.stderr[:400])


if __name__ == "__main__":
    main()
