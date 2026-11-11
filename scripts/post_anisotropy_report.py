"""Post the node report to the platform (mission 13868).

Reads credentials from the environment only; nothing is written to disk except
the composed report payload under /tmp/report.json.
"""

import base64
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
ROOT = "/root/tp-agent/work/m13868"


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def login():
    data = post("/auth/login", {"username": USER, "password": PASS})
    for key in ("access_token", "token", "accessToken"):
        if isinstance(data, dict) and data.get(key):
            return data[key]
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        for key in ("access_token", "token"):
            if data["data"].get(key):
                return data["data"][key]
    raise SystemExit(f"login response had no token: {list(data)[:6]}")


FIGURES = [
    ("pole_figures", "图1 c 轴极图（等面积投影）：左=S2 柱状组构（c 轴水平面内随机，g=1.000），右=均匀负控（g=0.000）",
     "docs/figures/ice_anisotropy_pole_figures.png"),
    ("strength_vs_angle", "图2 单轴强度 vs 加载方向（0°=沿柱轴）：Hill 连续面 / 滑移系 mean-field / Sachs 下界；45° 处误差 +2.67%",
     "docs/figures/ice_anisotropy_strength_vs_angle.png"),
    ("yield_loci", "图3 平面应力屈服轨迹：柱状（σ∥/σ⊥=3.5）vs 各向同性负控（von Mises 型）",
     "docs/figures/ice_anisotropy_yield_loci.png"),
    ("dem_saturation", "图4 离散键合晶格：微观键强对比度 vs 宏观各向异性（各向同性律→1.000；冗余导致饱和，无法复现 3-5）",
     "docs/figures/ice_anisotropy_dem_saturation.png"),
]

MARKDOWN = open(os.path.join(ROOT, "docs/ICE_ANISOTROPIC_STRENGTH_COLUMNAR.md"),
                encoding="utf-8").read()

payload = {
    "title": "柱状晶海冰横观各向同性强度：Hill 屈服面 + 组构账本 + 各向同性退化负控（p474w1-n08）",
    "summary": ("TensorDEM 新增 ice_anisotropy 能力包：c 轴 ODF 组构账本 → 滑移系(Schmid)参考模型 → "
                "横观各向同性 Hill 屈服面；σ∥/σ⊥=3.5000（文献带 3–5 通过），未参与定标的 45° 方向 "
                "连续面 0.8944σ⊥ vs 滑移系参考 0.8712σ⊥（误差 +2.67% ≤10%），均匀组构退化强度比 1.000000"
                "（误差 0.00% ≤5%），离散 DEM 各向同性微观律负控精确为 1；45 例 pytest 全绿，"
                "特性关联 H1-BROKEN-ICE-DEM。肩部 10° 处 −21.59% 与 τ_t/σ⊥ 偏离 von Mises +3.76% 已如实登记。"),
    "scenario_key": "p474w1-n08-dem-anisotropic",
    "mission_id": 13868,
    "tags": ["TensorDEM", "sea-ice", "columnar-S2", "transverse-isotropy",
             "Hill-1948", "fabric-tensor", "ODF", "slip-systems", "DEM",
             "anisotropy-negative-control"],
    "markdown": MARKDOWN,
    "figures": [],
}

for name, caption, rel in FIGURES:
    with open(os.path.join(ROOT, rel), "rb") as fh:
        payload["figures"].append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(fh.read()).decode(),
        })

payload["summary"] = payload["summary"][:500]
assert len(payload["tags"]) <= 10
with open("/tmp/report.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh)

token = login()
res = post("/tensor/reports", payload, token=token)
print(json.dumps(res)[:600])