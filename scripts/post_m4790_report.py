"""Post the m4790 paper-level progress report to the Tensor platform."""
import base64
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
ROOT = "/root/tp-agent/work/m4790"


def req(path, method="GET", body=None, token=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read())


def main():
    tok = req("/auth/login", "POST", {"username": USER, "password": PASS})
    tok = tok.get("access_token") or tok.get("token")
    print("token_len", len(tok))

    md = open(f"{ROOT}/artifacts/m4790/report.md", encoding="utf-8").read()
    png = open(f"{ROOT}/artifacts/m4790/ledger_corruption_sensitivity.png", "rb").read()
    b64 = base64.b64encode(png).decode()
    print("png_bytes", len(png), "b64_len", len(b64))

    report = {
        "title": "TensorDEM 2-D 能量/断裂台账负控修复与 main CI 转绿 (m4790)",
        "summary": (
            "修复 main(3ede547) 两处红测：(1) 负控测试断言方向写反（断言篡改后台账仍闭合，"
            "必红）→ 翻转为断言检查能检出篡改(残差>=0.5%)，并对本场景恒为0的 external_work "
            "通道 skip；(2) m4730 合入新文档后未重生成证据台账 → 重生成 doc_evidence_ledger.json。"
            "全量套件 1532 passed, 10 skipped, 0 failed (RC=0)。未篡改台账相对残差 0.28% < 3%。"
        ),
        "scenario_key": "tensordem_2d_energy_fracture_ledger_negative",
        "mission_id": 4790,
        "tags": ["tensordem", "dem", "energy-ledger", "negative-control", "ci-fix", "ice"],
        "markdown": md,
        "figures": [
            {
                "name": "ledger_corruption_sensitivity.png",
                "caption": (
                    "2-D DEM 能量/断裂台账残差敏感性：未篡改闭合于 0.28%(<0.5% 容差)；"
                    "6 个非零通道篡改后残差 4.0%–398.8%，全部越过 0.5% 容差被检出。"
                    "本单为后端/测试修复，无物理场量，故以残差敏感性为证据图。"
                ),
                "png_base64": b64,
            }
        ],
    }
    d = req("/tensor/reports", "POST", report, token=tok)
    print("POST status", d.get("status"), "id", d.get("id") or d.get("data", {}).get("id"))
    print("resp_keys", list(d.keys()))


if __name__ == "__main__":
    main()
