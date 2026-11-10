"""Publish the P40W4-N06 report to the platform report API.

Credentials come from TENSOR_API_BASE / TENSOR_AGENT_USER / TENSOR_AGENT_PASS.
The report payload is produced by ``tensordem.granular_column.report``
(``/tmp/report_p40w4_n06.json`` locally, or ``--payload <path>``).
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as fh:
        return json.loads(fh.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", default="/tmp/report_p40w4_n06.json")
    args = parser.parse_args()
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    login = _post(base + "/auth/login",
                  {"username": os.environ["TENSOR_AGENT_USER"],
                   "password": os.environ["TENSOR_AGENT_PASS"]})
    token = login.get("token") or login.get("access_token")
    print("login ok:", bool(token))
    payload = json.load(open(args.payload))
    resp = _post(base + "/tensor/reports", payload, token)
    print("report:", json.dumps(resp)[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())