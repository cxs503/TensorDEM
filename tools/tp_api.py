import json, os, sys
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def token():
    p = os.path.expanduser("~/.tp_tok.txt")
    if os.path.exists(p):
        t = open(p).read().strip()
        if t:
            return t
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req))
    t = d.get("access_token") or d.get("token")
    open(p, "w").write(t)
    return t


def get(path):
    req = urllib.request.Request(BASE + path,
                                 headers={"Authorization": "Bearer " + token()})
    return json.load(urllib.request.urlopen(req))


def post(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token()})
    return json.load(urllib.request.urlopen(req))


if __name__ == "__main__":
    d = get("/tensor/reports?limit=20")
    rs = d.get("reports") or d.get("data") or d.get("items") or d
    for r in (rs if isinstance(rs, list) else []):
        print(r.get("id"), str(r.get("created_at"))[:16], (r.get("title") or "")[:70])