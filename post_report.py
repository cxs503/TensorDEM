#!/usr/bin/env python3
"""Post report for m3999"""
import json
import os
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")

def call(path, data=None, tok=None):
    r = urllib.request.Request(BASE + path,
                               data=json.dumps(data).encode() if data is not None else None,
                               method="POST" if data is not None else "GET")
    r.add_header("Content-Type", "application/json")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=90) as f:
            raw = f.read().decode()
            return f.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]

# Login
st, res = call("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                               "password": os.environ["TENSOR_AGENT_PASS"]})
if st != 200:
    print(f"Login failed: {st} {res}")
    exit(1)
    
tok = res["access_token"]
print("Login OK, token obtained")

# Load report
with open("/root/tp-agent/work/m3999/report_m3999.json") as f:
    report = json.load(f)

# Post report
st, res = call("/tensor/reports", report, tok)
print(f"Report POST: status={st}")
if st == 200 or st == 201:
    print(f"Report ID: {res.get('id', 'N/A')}")
    # Save report ID
    with open("/root/tp-agent/work/m3999/report_id.txt", "w") as f:
        f.write(str(res.get('id', '')))
else:
    print(f"Response: {res}")
