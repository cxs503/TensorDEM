"""Publish the round-23 progress report to the platform (numbers read from
``runs/seed_ensemble/analysis_r23.json``, zero hand-copying).

    python benchmarks/plane_shear_mu_i/post_report_r23.py

Needs ``TENSOR_API_BASE`` / ``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS`` in the
environment.  The markdown body is the same programmatic text as
``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R23.md`` with a Chinese summary in front, so the
report and the document cannot diverge.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import write_r23_doc as DOCGEN  # noqa: E402

RUNS = HERE / "runs" / "seed_ensemble"
MISSION_ID = 22015
API = os.environ["TENSOR_API_BASE"].rstrip("/")


def fig_b64(name: str) -> str:
    return base64.b64encode((RUNS / name).read_bytes()).decode()


def login() -> str:
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(f"{API}/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return d.get("access_token") or d["token"]


def main() -> int:
    a = json.loads((RUNS / "analysis_r23.json").read_text())
    v = a["verdict"]
    d = a["dip"]
    lad = a["ladder"]
    pts = a["points"]
    judged = v["judged_points"]

    def pct(x):
        return f"{100 * x:+.1f}"

    zh = [
        "# CASE-DEM-PLANE-SHEAR-MUIDI · R23 — 种子集合统计：10% 验收带到底可不可分辨（honest FAIL，带误差棒）",
        "",
        "## 结论摘要",
        "",
        f"- 设备固定（与 R14 归档点同参数），只动随机种子：{len(pts)} 个惯性数 × "
        f"{len(pts['0.01']['seeds'])} 个种子 = "
        f"{len(pts) * len(pts['0.01']['seeds'])} 个算例，另加 {len(lad['points'])} 点接触摩擦梯度；集群作业 `cj3846`。",
        f"- 归档种子（{pts['0.01']['seeds'][0]}）重跑与 R14 归档点**逐位一致**"
        f"（最大相对差 {v['parity_max_rel_diff']:.1e}）：设备确定性、无协议漂移。",
        f"- 种子间离散（CV，`mu_full_zz_final`）在有效区各点为 "
        + "、".join(f"I={k}: {pct(pts[k]['mu']['mu_full_zz_final']['cv'])}%"
                   for k in judged)
        + f"；预登记 P1（CV<10%）{'成立' if v['P1_spread_holds'] else '被否证'}。",
        f"- `I=0.03 → 0.1` 的 μ 下降配对检验：{d['n_pairs']} 对中 {d['n_negative']} 对为负，"
        f"平均 {pct(d['rel_mean']['mean'])}%（95% CI [{d['ci95_delta'][0]:+.4f}, "
        f"{d['ci95_delta'][1]:+.4f}]）→ 系统性 = **{d['systematic']}**。",
        f"- 集合均值对归档 da Cruz 律（μ=0.22+1.00I，I≤0.2）逐点误差："
        + "、".join(f"I={k}: {pct(pts[k]['mu']['mu_full_zz_final']['rel_err_mean_vs_law'])}%"
                   for k in judged)
        + f"；验收判定 **{'PASS' if v['passes_acceptance'] else 'FAIL'}**"
        f"（{len(v['points_all_seeds_within_tol'])}/{len(judged)} 点全部种子入带）。",
        f"- 接触摩擦梯度（μ_p=0.2…0.7，I=0.03）：单调 = {lad['monotone_in_mu_contact']}，"
        f"最小误差 {pct(lad['min_rel_err_vs_law'])}%，是否入带 = {v['P4_ladder_reaches_band']}。",
        "- φ 仍 WITHHELD（定义不可比，仅报离散度）。",
        "",
        "---",
        "",
    ]
    body = DOCGEN.build(a)
    md = "\n".join(zh) + body

    payload = {
        "title": "CASE-DEM-PLANE-SHEAR-MUIDI R23: seed ensemble — is the 10% "
                 "mu(I) acceptance resolvable? (honest FAIL with error bars)",
        "summary": (
            f"{len(pts)}x{len(pts['0.01']['seeds'])} seed ensemble on the archived "
            f"plane-shear device + {len(lad['points'])}-point contact-friction ladder "
            f"(job cj3846). Archive seed reproduces round 14 bit-for-bit "
            f"(max rel {v['parity_max_rel_diff']:.1e}). Seed CV of mu: "
            + ", ".join(f"I={k} {100 * pts[k]['mu']['mu_full_zz_final']['cv']:.1f}%"
                        for k in judged)
            + f". Paired dip I=0.03->0.1: {100 * d['rel_mean']['mean']:+.1f}% "
              f"(systematic={d['systematic']}). Ensemble-mean mu still outside the "
              f"archived da Cruz law by "
            + ", ".join(f"{100 * pts[k]['mu']['mu_full_zz_final']['rel_err_mean_vs_law']:+.1f}%"
                        for k in judged)
            + f" -> acceptance {'PASS' if v['passes_acceptance'] else 'FAIL'}."),
        "scenario_key": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "mission_id": MISSION_ID,
        "tags": ["DEM", "plane-shear", "mu(I)", "rheology", "uncertainty",
                 "seed-ensemble", "TensorDEM", "honest-FAIL"],
        "markdown": md,
        "figures": [
            {"name": "fig_r23_muI_ensemble.png",
             "caption": "mu(I): per-seed points, ensemble mean +-95% CI, archived "
                        "da Cruz law +-10% band (dotted = outside validity)",
             "png_base64": fig_b64("fig_r23_muI_ensemble.png")},
            {"name": "fig_r23_cv.png",
             "caption": "seed CV of all five stress ordinates vs the 10% band",
             "png_base64": fig_b64("fig_r23_cv.png")},
            {"name": "fig_r23_dip_paired.png",
             "caption": "paired seed dip I=0.03 -> 0.1",
             "png_base64": fig_b64("fig_r23_dip_paired.png")},
            {"name": "fig_r23_mup_ladder.png",
             "caption": "contact-friction ladder at I=0.03 vs the archived law band",
             "png_base64": fig_b64("fig_r23_mup_ladder.png")},
        ],
    }
    token = login()
    req = urllib.request.Request(
        f"{API}/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    print(json.dumps(out, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())