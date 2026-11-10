import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(url, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, resp.read().decode()


status, body = post(f"{BASE}/auth/login", {"username": USER, "password": PASS})
login = json.loads(body if status != 200 else body)
token = None
try:
    token = json.loads(body).get("access_token") or json.loads(body).get("token")
except Exception:
    token = None
print("login status", status, "token", bool(token))

payload = json.load(open("/tmp/report.json"))
status, body = post(f"{BASE}/tensor/reports", payload, token)
print("report status", status)
print(body[:400])