import json, os, sys, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def req(path, method="GET", body=None, token=None, raw=False):
    url = BASE + path
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
    if token:
        headers["Authorization"] = "Bearer " + token
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(r, timeout=120) as resp:
        payload = resp.read()
        return payload if raw else json.loads(payload)


def login():
    d = req("/auth/login", "POST", {"username": USER, "password": PASS})
    return d.get("access_token") or d.get("token")


if __name__ == "__main__":
    tok = login()
    print("token_len", len(tok))
    open("/tmp/tok.txt", "w").write(tok)
    if len(sys.argv) > 1 and sys.argv[1] == "reports":
        d = req("/tensor/reports?limit=25", token=tok)
        items = d if isinstance(d, list) else d.get("items", d.get("reports", []))
        for r in items:
            print(r.get("id"), str(r.get("created_at", ""))[:16], (r.get("title") or "")[:95])