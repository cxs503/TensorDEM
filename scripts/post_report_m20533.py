import base64, json, os, urllib.request
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8000/api/v1").rstrip("/")
USER, PASS = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]
ART = "docs/artifacts/column_collapse_completion_m20533"
DOC = "docs/AUDIT_COLUMN_COLLAPSE_ASPECT_BAND_m20533.md"
def post(url, data, token=None):
    h = {"Content-Type": "application/json"}
    if token: h["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=300) as fh: return json.load(fh)
r = post(f"{BASE}/auth/login", {"username": USER, "password": PASS})
token = r.get("access_token") or r.get("token") or r["data"]["access_token"]
figs = []
for name, cap in (("fig_runout_vs_aspect.png", "Runout ratio vs measured aspect ratio: Lube (2004) two-branch law, deterministic lattice control and the four admissible ensemble members; a=2.0 exercises the a^(1/2) branch, a=0.5/1.0 the linear branch."),
                  ("fig_band_margin.png", "Per-case runout error against the +-15 % anchor band for both reference variants (law at measured a = paper-consistent vs law at nominal a = archived convention).")):
    figs.append({"name": name, "caption": cap, "png_base64": base64.b64encode(open(os.path.join(ART, name), "rb").read()).decode()})
payload = {
 "title": "CASE-COLUMN-COLLAPSE：可容许系综三长宽比收官 + 长宽比记账审计（m20533）",
 "summary": ("补齐 m20259 自报的 incomplete_aspects=[1.0,2.0]：同成员同参数新增 10 个 DEM 算例，"
             "三长宽比 15 行走同一归约路径。15 % 锚点在 a=1.0（最差 -8.1 %）与 a=2.0（+5.5 %）对全部可容许成员成立；"
             "a=0.5 在论文口径（律在实测 a 求值）下最差 -16.1 % 越带、名义口径 -12.8 %，m20259 的 A_BAND"
             "「supported」判为口径依赖；A_SCALE 在 a=0.5（10.3 %）与 a=1.0（7.6 %）被否证；"
             "σ/mean 三档 1.2–2.4 % 恒在 5 % 内。"),
 "scenario_key": "CASE-COLUMN-COLLAPSE", "mission_id": 20533,
 "tags": ["TensorDEM", "granular-column-collapse", "Lube2004", "ensemble", "admissibility", "aspect-bookkeeping", "m20533"],
 "markdown": open(DOC).read(), "figures": figs}
print(json.dumps(post(f"{BASE}/tensor/reports", payload, token))[:700])
