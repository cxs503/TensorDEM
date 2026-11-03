"""Publish the m2862 progress report (login -> POST /tensor/reports)."""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, body, token=None):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read().decode()


status, text = post("/auth/login", {"username": USER, "password": PASS})
payload = json.loads(text)
token = payload.get("access_token") or payload.get("token") or payload.get("data", {}).get("access_token")
print("login:", status, "token_len:", len(token or ""))

with open("/tmp/report.json") as fh:
    report = json.load(fh)
status, text = post("/tensor/reports", report, token)
print("report:", status, text[:400])