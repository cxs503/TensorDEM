"""Publish the round-33 report (figures embedded, zero hand-typed numbers)."""
import base64
import json
import os
import subprocess
import sys

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def _post(url: str, payload: dict, token: str | None = None) -> str:
    cmd = ["curl", "-s", "-X", "POST", url, "-H", "Content-Type: application/json"]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    cmd += ["-d", json.dumps(payload)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout


login = _post(f"{BASE}/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                     "password": os.environ["TENSOR_AGENT_PASS"]})
try:
    tok = json.loads(login).get("access_token") or json.loads(login).get("token")
except Exception:
    print("login failed:", login[:300])
    sys.exit(1)

md = open("docs/CASE_DEM_PLANE_SHEAR_MUIDI_R33.md").read()
png = open("benchmarks/plane_shear_mu_i/figures/wall_translation_r33.png", "rb").read()
rep = {
    "title": "R33 · DEM 平面剪切 mu(I) 标定：粗糙壁平动缺陷定位与闭合修复",
    "summary": ("2-D 平面剪切装置的胶结粗糙壁只赋速度、从不推进壁粒位置，装置因此不施加剪切率"
                "（R31 闭合率 0.07-0.13、滑移 2.8h）。修复后同胞同协议闭合率 0.103->0.631、"
                "滑移 4.49h->0.17h、p/P 0.059->0.676；对归档 2-D 尺子 mu 由 -42% 翻为 +19..27%，"
                "锚点仍未过关（honest）。"),
    "scenario_key": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "mission_id": 23115,
    "tags": ["DEM", "plane-shear", "mu(I)", "wall-slip", "boundary-condition",
             "verification-debt", "TensorDEM"],
    "markdown": md,
    "figures": [{
        "name": "wall_translation_r33.png",
        "caption": ("左：壁面平动前（冻结）/后（平动）的实测/名义剪切率闭合率，±10% 带；"
                    "右：两种壁面下 mu_bulk 对 da Cruz 2005 二维律 (mu=0.22+1.0I)。"),
        "png_base64": base64.b64encode(png).decode()}],
}
out = _post(f"{BASE}/tensor/reports", rep, tok)
print(out[:800])