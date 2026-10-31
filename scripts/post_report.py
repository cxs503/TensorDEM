"""Post a rendered report payload to the TensorPlatform report endpoint.

Reads the login response and the report payload from files (so no secret ever
appears on a command line) and posts it:

    curl -s -X POST "$TENSOR_API_BASE/auth/login" -H 'Content-Type: application/json' \\
         -d "{\\"username\\":\\"$TENSOR_AGENT_USER\\",\\"password\\":\\"$TENSOR_AGENT_PASS\\"}" \\
         -o /tmp/login.json
    python scripts/post_report.py /tmp/login.json /tmp/w5_n03_report.json
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    login_path, payload_path = argv[1], argv[2]
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    with open(login_path) as handle:
        token = json.load(handle)["access_token"]
    with open(payload_path, "rb") as handle:
        payload = handle.read()
    request = urllib.request.Request(
        f"{base}/tensor/reports", data=payload, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8", "replace")
        print(response.status, body[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))