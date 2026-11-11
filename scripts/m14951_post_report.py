"""m14951: publish the paper-level report (login + POST /tensor/reports)."""
import base64
import json
import os
import urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
if not base.endswith("/api/v1"):
    base += "/api/v1"

data = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                   "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(base + "/auth/login", data=data,
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=180) as resp:
    body = json.loads(resp.read())
tok = body.get("access_token") or body.get("token")
if not tok:
    raise SystemExit(f"login failed: {body}")

payload = json.load(open("/tmp/report_m14951.json"))
req = urllib.request.Request(base + "/tensor/reports",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok},
                             method="POST")
with urllib.request.urlopen(req, timeout=300) as resp:
    out = json.loads(resp.read())
print("report id:", out.get("id") or out.get("report_id") or out)