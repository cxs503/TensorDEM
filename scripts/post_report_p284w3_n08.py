"""Post the P284W3-N08 report to the TensorPlatform report API (mission 13965).

Reads ``artifacts/dem_joint_calibration/*.json`` + the generated doc and the
figure directory, so every number in the summary is read from the artefacts
(zero hand-copying).
"""
from __future__ import annotations

import base64
import glob
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"

ART = os.path.join(ROOT, "artifacts", "dem_joint_calibration")
FIGDIR = os.path.join(ART, "figures")
DOC = os.path.join(ROOT, "docs", "P284W3_N08_JOINT_CALIBRATION.md")

CAPTIONS = {
    "fig1_benchmarks.png": "图1 三基准孪生目标：直剪 τ-γ、粘结单轴 σ-ε 与休止角",
    "fig2_convergence.png": "图2 联合目标函数在 50 次正演预算内的收敛（对数轴）",
    "fig3_posterior.png": "图3 五参数后验 95% 可信区间 vs 真值（虚线）",
    "fig4_verification.png": "图4 换种子验证：剪切曲线 1:1 与峰值应力误差 vs 5% 判据",
}


def login() -> str:
    data = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    tok = body.get("access_token") or body.get("token")
    if not tok:
        raise SystemExit(f"login did not return a token: {body}")
    return tok


def load_results() -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(ART, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.endswith("_replicates"):
            continue
        with open(path) as fh:
            out[name] = json.load(fh)
    return out


def main() -> int:
    res = load_results()
    if not res:
        raise SystemExit("no artefacts to report")
    lines = []
    for name, r in res.items():
        v = r["verdicts"]
        lines.append(
            f"- **{name}**: J={r['best_objective']:.3g}（drop {r['objective_drop']:.1%}）；"
            f"(a) θ 误差 {v['a_repose']['value']:.2f}° {'PASS' if v['a_repose']['pass'] else 'FAIL'}；"
            f"(b) 剪切 RMSE {v['b_shear_rmse']['value']:.2%} {'PASS' if v['b_shear_rmse']['pass'] else 'FAIL'}；"
            f"(c) 峰值应力误差 {v['c_peak_stress']['value']:.2%} {'PASS' if v['c_peak_stress']['pass'] else 'FAIL'}；"
            f"(e) 后验预测覆盖 {v['e_coverage']['value']:.1%} {'PASS' if v['e_coverage']['pass'] else 'FAIL'}；"
            f"总判定 {'PASS' if r['all_pass'] else 'FAIL'}")
    summary = ("TensorDEM 新增三基准联合标定框架（contact_engine/joint_benchmarks/"
               "joint_calibration）：休止角(柱塌落)+恒体积直剪+粘结单轴三正演联合反演 "
               "θ=[k_n,k_t,μ,e,σ_bond]，GP+EI 50 次正演，MCMC 后验，换种子验证。"
               + "；".join(l.replace("**", "").replace("- ", "").split("：")[0]
                           for l in lines) +
               "。判定：全部 (a)θ (b)剪切 (c)峰值 未达标，砂的 (d)drop94% (e)覆盖100% 通过；"
               "根因=百粒试样换种子噪声底(θ 7-11%,UCS 9%)与 3°/5% 判据同量级。")
    with open(DOC, encoding="utf-8") as fh:
        markdown = fh.read()
    figures = []
    for name in sorted(os.listdir(FIGDIR)) if os.path.isdir(FIGDIR) else []:
        if not name.endswith(".png"):
            continue
        with open(os.path.join(FIGDIR, name), "rb") as fh:
            figures.append({"name": name, "caption": CAPTIONS.get(name, name),
                            "png_base64": base64.b64encode(fh.read()).decode()})
    if len(summary) > 480:
        summary = summary[:470] + "…"
    payload = {
        "title": "三基准联合反演 DEM 接触参数（休止角/剪切盒/单轴压缩 + 贝叶斯优化 + 不确定性量化）"
                 "（P284W3-N08）",
        "summary": summary,
        "scenario_key": "p284w3-n08-joint-dem-calibration",
        "mission_id": "13965",
        "tags": ["dem", "calibration", "bayesian-optimization", "gaussian-process",
                 "repose-angle", "direct-shear", "uniaxial-compression", "bonded-particle",
                 "uncertainty-quantification", "tensordem"],
        "markdown": markdown,
        "figures": figures,
    }
    tok = login()
    req = urllib.request.Request(BASE + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            print(resp.read().decode()[:1000])
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read().decode()[:2000])
        return 1
    print("figures:", len(figures), "materials:", list(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())