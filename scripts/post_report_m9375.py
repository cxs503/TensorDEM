"""Post the P202W1-N10 report to the platform (mission 9375).

Payload path is a parameter on purpose: ``/tmp/report.json`` is shared by every
concurrent agent on this host, so a fixed path silently posts somebody else's
report.  Usage::

    python scripts/post_report_m9375.py /tmp/report_m9375.json
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

MISSION_ID = 9375

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not base.endswith("/api/v1"):
    base = base.rstrip("/") + "/api/v1"

login_data = json.dumps({
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"],
}).encode()
req = urllib.request.Request(base + "/auth/login", data=login_data,
                             headers={"Content-Type": "application/json"},
                             method="POST")
with urllib.request.urlopen(req, timeout=180) as resp:
    body = json.loads(resp.read())
tok = body.get("access_token") or body.get("token")
if not tok:
    raise SystemExit(f"login did not return a token: {body}")

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/report_m9375.json"
with open(path) as fh:
    payload = json.load(fh)
assert payload.get("mission_id") == MISSION_ID, \
    f"payload mission is {payload.get('mission_id')}, expected {MISSION_ID}"

req = urllib.request.Request(base + "/tensor/reports",
                            data=json.dumps(payload).encode(),
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + tok},
                            method="POST")
with urllib.request.urlopen(req, timeout=180) as resp:
    print(resp.read().decode())