#!/usr/bin/env python3
"""Publish the PR #216 cure report to the platform (mission 4034).

Credentials come from the environment, never from the command line.
"""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def get(path: str, token: str) -> dict:
    req = urllib.request.Request(f"{BASE}{path}", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


auth = post("/auth/login", {"username": USER, "password": PASS})
token = auth.get("access_token") or auth.get("token")
print("login ok, token length", len(token or ""))

report = json.load(open("/tmp/report.json"))
print("payload KB", round(os.path.getsize("/tmp/report.json") / 1024, 1))
resp = post("/tensor/reports", report, token)
print("post resp:", json.dumps(resp)[:300])

back = get("/tensor/reports?mission_id=4034", token)
items = back.get("reports", back if isinstance(back, list) else [])
titles = [i.get("title", "")[:60] for i in items][:3]
print("stored reports for mission 4034:", len(items), titles)