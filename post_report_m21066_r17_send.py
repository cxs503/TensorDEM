import json, os, subprocess

API = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1").rstrip("/")
U, P = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]


def curl(args):
    return subprocess.run(["curl", "-s", *args], capture_output=True, text=True,
                          timeout=180).stdout


def token():
    for body in (f'{{"username":"{U}","password":"{P}"}}',
                 f"username={U}&password={P}"):
        for ct in ("application/json", "application/x-www-form-urlencoded"):
            raw = curl(["-X", "POST", f"{API}/auth/login",
                        "-H", f"Content-Type: {ct}", "-d", body])
            try:
                d = json.loads(raw)
            except Exception:
                continue
            t = d.get("access_token") or d.get("token") or \
                (d.get("data") or {}).get("access_token")
            if t:
                print("login ok via", ct, "keys:", list(d)[:5])
                return t
    print("login failed, last raw:", raw[:300])
    return None


tok = token()
assert tok, "no token"
payload = open("/tmp/report_m21066.json").read()
out = curl(["-X", "POST", f"{API}/tensor/reports",
            "-H", f"Authorization: Bearer {tok}",
            "-H", "Content-Type: application/json", "-d", "@-"]
           ) if False else subprocess.run(
    ["curl", "-s", "-X", "POST", f"{API}/tensor/reports",
     "-H", f"Authorization: Bearer {tok}", "-H", "Content-Type: application/json",
     "-d", "@/tmp/report_m21066.json"],
    capture_output=True, text=True, timeout=300).stdout
print("response:", out[:600])
try:
    d = json.loads(out)
    print("REPORT_ID:", d.get("id") or (d.get("data") or {}).get("id"),
          "| status:", d.get("status") or (d.get("data") or {}).get("status"))
except Exception:
    pass