"""Post the P82W2-N09 report to the platform (`POST /tensor/reports`).

Reads ``runs/P82W2_N09/report_payload.json`` (built by
``scripts/p82w2_n09_report.py``), authenticates with the agent credentials from the
environment and posts the paper-grade report with inlined figures.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = ROOT / "runs" / "P82W2_N09" / "report_payload.json"


def _post(url: str, body: dict, token: str | None = None, timeout: int = 300):
    data = json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        return exc.code, {"error": exc.read().decode()[:2000]}


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    status, login = _post(f"{base}/auth/login", {"username": user, "password": password})
    if status != 200:
        print(f"login failed: {status} {login}")
        return 1
    token = login.get("access_token") or login.get("token")
    payload = json.loads(PAYLOAD.read_text())
    status, body = _post(f"{base}/tensor/reports", payload, token=token)
    print(f"report post: HTTP {status}")
    print(json.dumps(body, ensure_ascii=False)[:800])
    return 0 if status in (200, 201) else 1


if __name__ == "__main__":
    sys.exit(main())