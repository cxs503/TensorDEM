#!/usr/bin/env python3
"""Publish the m22824 audit report (payload rendered by m22824_doc.py)."""
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
if not BASE.endswith("/api/v1"):
    BASE += "/api/v1"

payload_path = Path(os.environ.get("M22824_REPORT", "/tmp/report_m22824.json"))
report = json.loads(payload_path.read_text())
print({k: (len(v) if isinstance(v, (list, str)) else v)
       for k, v in report.items() if k != "figures"},
      "figures=", len(report.get("figures", [])))

creds = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                    "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(BASE + "/auth/login", data=creds,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=180) as resp:
    tok = json.loads(resp.read())
token = tok.get("access_token") or tok.get("token")
print("token chars:", len(token))

req = urllib.request.Request(BASE + "/tensor/reports",
                             data=json.dumps(report).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"},
                             method="POST")
with urllib.request.urlopen(req, timeout=600) as resp:
    print("posted:", json.loads(resp.read()))