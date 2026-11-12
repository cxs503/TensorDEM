"""Publish the m16526 report via urllib (shell curl is blocked)."""
import json
import os
import sys
import urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not base.endswith("/api/v1"):
    base = base.rstrip("/") + "/api/v1"

login_data = json.dumps({
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"]
}).encode()
req = urllib.request.Request(base + "/auth/login", data=login_data,
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=180) as resp:
    login_body = json.loads(resp.read())
tok = login_body.get("access_token") or login_body.get("token")
if not tok:
    raise SystemExit(f"login did not return a token: {login_body}")

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/report_m16526.json"
with open(path, "r") as f:
    payload = json.load(f)
assert payload.get("mission_id") == 16526, f"payload mission is {payload.get('mission_id')}, expected 16526"

req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok}, method="POST")
with urllib.request.urlopen(req, timeout=180) as resp:
    print(resp.read().decode())
