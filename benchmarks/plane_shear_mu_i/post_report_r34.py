"""POST /tmp/report.json to the platform with a fresh agent token (urllib only)."""
from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
print("base:", BASE, "user:", USER, "pass_len:", len(PASS))


def post(url: str, payload: dict, token: str | None = None) -> str:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read().decode()


login = json.loads(post(f"{BASE}/auth/login", {"username": USER, "password": PASS}))
tok = login.get("access_token") or login.get("token")
print("login keys:", sorted(login), "token_len:", len(tok or ""))

report = json.load(open("/tmp/report.json"))
out = json.loads(post(f"{BASE}/tensor/reports", report, tok))
print("report id:", out.get("id"), "| title:", str(out.get("title"))[:80],
      "| status:", out.get("status"))