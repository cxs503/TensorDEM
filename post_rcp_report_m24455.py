#!/usr/bin/env python3
"""Post the CASE-DEM-RCP-PACKING-FRACTION verification report (m24455)."""
import json, os, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- login ---------------------------------------------------------------
req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/auth/login",
    data=json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode(),
    headers={"Content-Type": "application/json"},
)
body = json.loads(urllib.request.urlopen(req).read())
token = body.get("access_token") or body.get("token")
assert token, f"login did not return a token: {list(body.keys())}"

# ---- load report ---------------------------------------------------------
report = json.loads((ROOT / "reports" / "rcp_report_m24455.json").read_text())

req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/tensor/reports",
    data=json.dumps(report).encode(),
    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
)
resp = json.loads(urllib.request.urlopen(req).read())
print("report_id:", resp.get("id") or resp.get("report_id") or resp)
