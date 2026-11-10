#!/usr/bin/env python
"""Assemble and post the P270W2C-N08 progress report.

Builds the ``/tensor/reports`` payload from the campaign artefacts (title, summary,
markdown report and the base64 PNG figures) and posts it with the agent token.

Usage: ``python scripts/p270w2c_n08_post.py [--dry-run]``
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "p270w2c_n08"
DOCS = ROOT / "docs"
MISSION_ID = 9756

FIGURES = {
    "fig_response_surface.png": "响应面切片：每列一个微观参数（单位立方体坐标），每行一个宏观响应"
                                "（UCS / σc/σt / Weibull m）；绿色带为验收目标带。",
    "fig_jacobian.png": "可辨识性审计：量纲一化 Jacobian dlog10(y)/du、参数相关矩阵（JᵀJ 归一化）"
                        "与 Brun collinearity index。",
    "fig_pareto.png": "代理响应面上的 NSGA-II 种群与 Pareto 前沿（三目标两两投影）；越界量 0 表示目标满足。",
    "fig_size_effect.png": "Weibull 尺寸效应：参考 θ 的五厚度族与幂律拟合（虚线为留出厚度），"
                           "右图为留出厚度外推误差。",
    "fig_brazilian.png": "巴西圆盘中心主应力比 vs 应力平均区半径（三种点阵分辨率）与 Hondros −3.0 容差带；"
                         "右图为比值读数历史。",
    "fig_ucs.png": "参考 θ 的单轴压缩应力–应变迹（hold 末读数）与 AC6 负控 200 组随机 θ 的 UCS 分布。",
}


def build_payload() -> dict:
    report = json.loads((ARTIFACTS / "report.json").read_text())
    markdown_path = DOCS / "bpm_calibration_feasibility_P270W2C_N08.md"
    markdown = markdown_path.read_text()

    ac2 = report["AC2_identifiability"]["diagnostics"]
    ac3 = report["AC3_verdict"]
    ac4 = report["AC4_holdout"]
    ac5 = report["AC5_brazilian_field"]
    ac6 = report["AC6_negative_control"]
    ac7 = report["AC7_reproducibility"]
    reference = ac3["reference"]
    best = ac5["best"]

    summary = (
        "BPM 三目标联合标定可行性判定 + 可辨识性审计：33 点响应面（{n_tasks} 个 DEM 任务）→ "
        "cond(J)={cond:.2f}、rank=3/6（结构零空间维数 {null_dim}，结论 {verdict}，最坏参数对 {worst} "
        "γ={gamma:.1f}）；三分支判定落分支 ({branch})（UCS {ucs:.2f} MPa、σc/σt {ratio:.2f}、"
        "m CI [{m_lo:.2f}, {m_hi:.2f}]）；留出厚度外推 {hold_passed}/{hold_n} 通过；"
        "巴西中心主应力比最优 {hondros:.2f}（相对 −3.0 偏差 {hondros_dev:.0f}%，honest FAIL）；"
        "负控 {neg:.0f}% 落带外；复现性相对差 {rep:.1e}、最大 KE/外功 {ke:.2e}。"
    ).format(
        n_tasks=report["design"]["n_tasks"],
        cond=ac2["cond"], null_dim=ac2.get("structural_null_space_dimension"),
        verdict=ac2["verdict"],
        worst="-".join(ac2["collinearity"]["worst_pair"]["pair"]),
        gamma=ac2["collinearity"]["worst_pair"]["collinearity"],
        branch=reference["branch"], ucs=reference["ucs_MPa"],
        ratio=reference["strength_ratio"],
        m_lo=reference["weibull_m_ci"][0], m_hi=reference["weibull_m_ci"][1],
        hold_passed=ac4["n_passed"], hold_n=ac4["n_points"],
        hondros=best["ratio_at_peak"] if best else float("nan"),
        hondros_dev=(best["relative_deviation"] * 100) if best else float("nan"),
        neg=ac6["fraction_outside_ucs_band"] * 100,
        rep=ac7["relative_difference"], ke=ac7["max_ke_ratio_all_runs"],
    )

    figures = []
    for name, caption in FIGURES.items():
        path = ARTIFACTS / "figures" / name
        if not path.exists():
            continue
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })

    return {
        "title": "BPM 微观参数三目标联合标定的可行性判定与可辨识性审计（P270W2C-N08）",
        "summary": summary,
        "scenario_key": "p270w2c-n08-bpm-calibration-feasibility",
        "mission_id": MISSION_ID,
        "tags": ["DEM", "BPM", "calibration", "identifiability", "Weibull", "Brazilian-test",
                 "NSGA-II", "TensorDEM", "P270W2C-N08"],
        "markdown": markdown,
        "figures": figures,
    }


def post(payload: dict) -> None:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    if not base.endswith("/api/v1"):
        base = base.rstrip("/") + "/api/v1"
    login = json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    request = urllib.request.Request(base + "/auth/login", data=login,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit(f"login did not return a token: {body}")
    assert payload["mission_id"] == MISSION_ID
    request = urllib.request.Request(
        base + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST")
    with urllib.request.urlopen(request, timeout=300) as response:
        print(response.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="/tmp/report_m9756.json")
    args = parser.parse_args()
    payload = build_payload()
    Path(args.out).write_text(json.dumps(payload))
    print(f"payload written to {args.out} "
          f"(markdown {len(payload['markdown'])} chars, {len(payload['figures'])} figures)")
    if not args.dry_run:
        post(payload)


if __name__ == "__main__":
    main()