#!/usr/bin/env python3
"""Publish the m17063 report through the tensor API (login + POST)."""
import json
import os
import urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


auth = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                            "password": os.environ["TENSOR_AGENT_PASS"]})
token = auth.get("access_token") or auth.get("token")
payload = json.load(open("/tmp/report.json"))
out = post("/tensor/reports", payload, token)
print(json.dumps({k: out[k] for k in out if k in ("id", "report_id", "status",
                                                  "title")}))