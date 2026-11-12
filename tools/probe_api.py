import json, os, requests
BASE = os.environ["TENSOR_API_BASE"]
u, p = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]
s = requests.Session()
d = s.post(f"{BASE}/auth/login", json={"username": u, "password": p}).json()
H = {"Authorization": f"Bearer {d['access_token']}"}
for rid in (15188, 15187):
    r = s.get(f"{BASE}/tensor/reports/{rid}", headers=H)
    print("=== rid", rid, r.status_code, len(r.text))
    if r.status_code == 200:
        j = r.json()
        json.dump(j, open(f"/tmp/report_{rid}.json", "w"), ensure_ascii=False, indent=1)
        md = j.get("markdown") or ""
        print("keys:", list(j), "md_len:", len(md), "figs:", [(f.get("name"), len(f.get("png_base64") or "")) for f in (j.get("figures") or [])])
        print(md[:1500])