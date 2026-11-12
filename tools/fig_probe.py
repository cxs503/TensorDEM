import hashlib, os, requests
BASE = os.environ["TENSOR_API_BASE"]
s = requests.Session()
d = s.post(f"{BASE}/auth/login", json={"username": os.environ["TENSOR_AGENT_USER"],
                                       "password": os.environ["TENSOR_AGENT_PASS"]}).json()
H = {"Authorization": f"Bearer {d['access_token']}"}
for name in ("fig1_ft_delta_curve", "fig2_microslip_error", "fig3_dt_convergence",
             "fig4_traction_field", "fig5_reversal"):
    r = s.get(f"{BASE}/tensor/reports/15188/figures/{name}.png", headers=H, timeout=60)
    print(name, r.status_code, r.headers.get("content-type"), len(r.content),
          hashlib.sha256(r.content).hexdigest() if r.ok else r.text[:80])