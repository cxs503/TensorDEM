#!/usr/bin/env python3
"""m21119: login + publish /tmp/report_m21119.json to the TensorPlatform API."""
import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

def post(path, body, tok=None, trunc=None):
    h = {"Content-Type": "application/json"}
    if tok:
        h["Authorization"] = "Bearer " + tok
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode()
        return r.status, (raw[:trunc] if trunc else raw)

st, out = post("/auth/login", {"username": user, "password": pw})
tok = json.loads(out).get("access_token", "")
print("login", st, "token_len", len(tok))
st, out = post("/tensor/reports", json.load(open("/tmp/report_m21119.json")), tok, trunc=500)
print("publish", st, out)
