#!/usr/bin/env python3
"""Post the mission report payload to the TensorPlatform reports API."""
import json
import os
import sys
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def _post(path: str, body: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main(path: str = "/tmp/report.json") -> int:
    payload = json.load(open(path))
    login = _post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                  "password": os.environ["TENSOR_AGENT_PASS"]})
    token = login.get("access_token") or login.get("token")
    if not token:
        print("login failed:", login)
        return 1
    out = _post("/tensor/reports", payload, token)
    print(json.dumps(out)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))