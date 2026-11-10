"""Post the ice-lubrication regime report (node p9050-n09) to the platform API."""
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "ice_lubrication_regime"
MISSION_ID = 9897

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not base.endswith("/api/v1"):
    base = base.rstrip("/") + "/api/v1"

payload = json.loads((OUT / "report.json").read_text())
assert str(payload["mission_id"]) == str(MISSION_ID), payload["mission_id"]

login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                    "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(base + "/auth/login", data=login,
                             headers={"Content-Type": "application/json"},
                             method="POST")
with urllib.request.urlopen(req, timeout=180) as resp:
    body = json.loads(resp.read())
tok = body.get("access_token") or body.get("token")
if not tok:
    raise SystemExit(f"login did not return a token: {body}")

req = urllib.request.Request(base + "/tensor/reports",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok},
                             method="POST")
with urllib.request.urlopen(req, timeout=300) as resp:
    print(resp.status, resp.read()[:500].decode("utf-8", "replace"))
