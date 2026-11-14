"""Publish the R5 (constant-pressure) report; every number is generated from runs/*.json."""
from __future__ import annotations
import json, os, urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path, body, token=None):
    hdr = {"Content-Type": "application/json"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(body).encode(),
                                 headers=hdr, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def main():
    payload = json.loads(Path("/tmp/report.json").read_text())
    tok = post("/auth/login", {"username": USER, "password": PASS})
    token = tok.get("access_token") or tok.get("token")
    out = post("/tensor/reports", payload, token)
    print(json.dumps(out)[:400])


if __name__ == "__main__":
    main()
