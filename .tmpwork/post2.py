import json, os, urllib.request, urllib.error
base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not base.endswith("/api/v1"):
    base = base.rstrip("/") + "/api/v1"
login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                    "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(base + "/auth/login", data=login,
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=180) as r:
    tok = json.loads(r.read()).get("access_token")
payload = json.load(open("artifacts/p307w1_n010/report.json"))
payload.pop("acceptance", None)
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        print(r.read().decode()[:300])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:500])
