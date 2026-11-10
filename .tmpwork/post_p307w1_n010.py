import json, os, urllib.request, sys
base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not base.endswith("/api/v1"):
    base = base.rstrip("/") + "/api/v1"
login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                    "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(base + "/auth/login", data=login,
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=180) as r:
    body = json.loads(r.read())
tok = body.get("access_token") or body.get("token")
assert tok, body
payload = json.load(open("artifacts/p307w1_n010/report.json"))
assert payload.get("mission_id") == 10001, payload.get("mission_id")
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok}, method="POST")
with urllib.request.urlopen(req, timeout=300) as r:
    print(r.read().decode()[:800])
