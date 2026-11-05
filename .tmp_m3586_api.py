import json, os, urllib.request, urllib.error

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
tok = open("/tmp/m3586_token").read().strip()

def req(method, path, body=None, token=None):
    url = base + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}

for rid in (3705, 3694):
    st, out = req("GET", f"/tensor/reports/{rid}", token=tok)
    md = out.get("markdown") or ""
    print("=" * 20, rid, out.get("title"))
    # print sections that look like limitations / todos
    lines = md.splitlines()
    keep = False
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(k in low for k in ("局限", "待办", "注意", "limitation", "todo", "debt", "欠账", "未", "open")):
            keep = True
        if ln.startswith("#") and keep and i > 3:
            keep = any(k in low for k in ("局限", "待办", "注意", "limitation", "todo", "debt", "欠账"))
        if keep:
            print(ln)
    print()
