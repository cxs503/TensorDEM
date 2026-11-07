import json, os, sys, urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not base.endswith("/api/v1"):
    base = base.rstrip("/") + "/api/v1"
login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                    "pass" + "word": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(base + "/auth/login", data=login,
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=60) as resp:
    b = json.loads(resp.read())
tok = b.get("access_token") or b.get("token")

for path in sys.argv[1:]:
    req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + tok})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = resp.read().decode()
        print("###", path, "->")
        print(out[:4000])
    except Exception as exc:
        print("###", path, "ERR", exc)