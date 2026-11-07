"""Post the ice-beam anchor report to the platform report API."""
import base64, json, os, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ["TENSOR_API_BASE"]
figures_json = ROOT / "artifacts" / "ice-beam-bending" / "figures.json"
report_json = ROOT / "artifacts" / "ice-beam-bending" / "ice_beam_report.json"
doc = ROOT / "docs" / "ice_beam_bending_anchor.md"

report = json.loads(report_json.read_text(encoding="utf-8"))
figures = json.loads(figures_json.read_text(encoding="utf-8")) if figures_json.exists() else []
anchor = report["primary"]["anchor"]
crit = report["criteria"]

body = doc.read_text(encoding="utf-8")
body += "\n## 7. Figures\n\n"
for fig in figures:
    body += f"**{fig['name']}** — {fig['caption']}\n\n"

def _shrink(path: Path, limit: int = 40_000) -> bytes:
    """Downscale a PNG until it fits the report payload budget."""
    import io
    from PIL import Image
    data = path.read_bytes()
    if len(data) <= limit:
        return data
    image = Image.open(io.BytesIO(data))
    scale = 0.85
    while True:
        width = max(320, int(image.width * scale))
        height = max(240, int(image.height * scale))
        buffer = io.BytesIO()
        image.resize((width, height)).convert("P", palette=Image.ADAPTIVE,
                                              colors=64).save(buffer, format="PNG",
                                                              optimize=True)
        blob = buffer.getvalue()
        if len(blob) <= limit or scale < 0.4:
            return blob
        scale *= 0.85


payload = {
    "title": "DEM 冰梁弯曲破坏解析锚: Timoshenko/EB vs 键合粒子格点平衡 + 脆性断裂级联",
    "summary": (
        "TensorDEM 键合粒子模型首次对冰梁三点弯曲做解析锚定：以格点 Born 常数反演 bond_stiffness 与 "
        f"breaking_strain，在任务几何（L=1 m, b=0.2 m, h=0.1 m, d=5 mm, E=9 GPa, σ_b=1.5 MPa）上求 DEM "
        f"键网络的无速率平衡，与 Timoshenko/EB 闭式解三方比对。判定 {report['verdict']}："
        f"{crit['passed']}/{crit['total']} 项判据通过。中跨挠度 9.84%（plane-stress E）/ 2.55%"
        "（圆柱弯曲板模量）偏差；截面弯矩、应力梯度、起裂位置（跨中底部）与 σ_b±20%→P_ult±20% 通过；"
        "格点弯曲刚度存在约 2.4·(d/h) 的离散偏差，5 mm 仍未收敛，是挠度判据未达 5% 的根因；"
        "强度型键准则在 h/d=7 时不能驱动穿厚裂纹（沿受拉纤维扩展），如实记为未通过。"
    ),
    "scenario_key": "tensordem-ice-beam-bending",
    "mission_id": 5174,
    "tags": ["TensorDEM", "ice", "flexural", "Timoshenko", "DEM", "fracture", "IN12-S3"],
    "markdown": body,
    "figures": [
        {"name": f["name"], "caption": f["caption"][:295],
         "png_base64": base64.b64encode(_shrink(ROOT / f["path"])).decode()}
        for f in figures
    ],
}


def post(path, obj, token=None):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req) as response:
        return json.load(response)


token = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                             "password": os.environ["TENSOR_AGENT_PASS"]})
token = token.get("access_token") or token.get("token")
size = len(json.dumps(payload))
print("payload bytes", size, "figures", len(payload["figures"]))
if size > 250_000:
    raise SystemExit("payload too large; shrink the figures first")
print(post("/tensor/reports", payload, token))