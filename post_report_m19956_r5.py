#!/usr/bin/env python3
"""Publish the R5 platform report (payload written by report_r5.py to /tmp/report.json)."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as fh:
        return json.load(fh)


auth = post("/auth/login", {"username": USER, "password": PASS})
token = auth.get("access_token") or auth.get("token")
if not token:
    raise SystemExit("no token: %s" % list(auth))
with open("/tmp/report.json") as fh:
    payload = json.load(fh)
res = post("/tensor/reports", payload, token)
print(json.dumps(res, ensure_ascii=False)[:800])