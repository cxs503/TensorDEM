#!/usr/bin/env python
"""Post the P295W1-N07 paper-grade report to the platform ``/tensor/reports`` API.

Payload is built from ``docs/P295W1_N07_random_ice_strength.md`` (itself generated
from ``runs/P295W1_N07/study.json``) and embeds the four figures as base64.  The
payload path is unique per mission because ``/tmp/report.json`` is shared by every
concurrent agent on this host.

    PYTHONPATH=src python scripts/p295w1_n07_post_report.py
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

DOC = Path("docs/P295W1_N07_random_ice_strength.md")
STUDY = Path("runs/P295W1_N07/study.json")
ART = Path("artifacts/p295w1_n07")
PAYLOAD = Path("/tmp/report_m9906_p295w1_n07.json")

FIGURES = [
    ("fig1_qq_weibull.png",
     "图 1 (a) Weibull 边际标定：左为 20000 个梁条强度的 QQ 图（R²=0.99993，"
     "k̂=5.957 vs 输入 6.0）；右为 lognormal 负对照（R²=0.903）被判不合格，"
     "证明判据有分辨力。"),
    ("fig2_correlation.png",
     "图 2 (b) 相关结构：左为 L=5/10/20 m 的 2000 次实现相关函数与高斯核"
     "（最差 L2 相对误差 3.16%）；右为实测 1/e 相关长度 vs 指定 L"
     "（实心=高斯场，空心=copula 秩相关，最差 1.34%）。"),
    ("fig3_ensemble.png",
     "图 3 (c) 破冰力 ensemble（B_ref=50 m，N=50）：左为 F/B 直方图与 F_5%、μ_F、"
     "确定性公式三条竖线；右为 running mean 收敛（s.e./μ=3.27%，drift=1.32%）。"),
    ("fig4_size_effect.png",
     "图 4 (c) 尺寸效应：左为四种跑法的 F_5%/B vs B=10…100 m，虚线为 bundle 对照"
     "模型（符号相反，随 B 上升）；右为 log-log 斜率与 Weibull 1/k=0.167 解析锚对比。"),
]


def main() -> int:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    if not base.endswith("/api/v1"):
        base = base.rstrip("/") + "/api/v1"
    login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=login,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit(f"login did not return a token: {body}")

    study = json.loads(STUDY.read_text())
    checks = study["checks"]
    assert study["all_pass"], "refusing to post a report for a failing study"
    figs = []
    for name, caption in FIGURES:
        png = (ART / name).read_bytes()
        figs.append({"name": name, "caption": caption,
                     "png_base64": base64.b64encode(png).decode()})

    payload = {
        "title": "P295W1-N07 随机冰板强度场：Weibull 梁单元 + 空间相关长度 + 破冰力 ensemble 统计",
        "summary": (
            "TensorDEM 新增 tensordem.random_ice_strength：精确 KL 高斯场 + 高斯 copula "
            "Weibull 强度场 + 单位宽度梁条 weakest-link 破冰力模型。判据全部通过（11/11）："
            f"QQ-plot R²={checks['criterion_a_qq_r2']['value']:.5f}（lognormal 负对照 "
            f"{checks['criterion_a_negative_control_below']['value']:.4f} 被判不合格）；"
            f"相关函数 L2 误差 {checks['criterion_b_correlation_l2']['value']*100:.2f}%，1/e 相关长度"
            f"误差 {checks['criterion_b_correlation_length']['value']*100:.2f}%；N=50 下 "
            f"σ_F/μ_F={checks['criterion_c_cov_band']['value']:.4f}∈[0.1,0.3]、"
            f"s.e./μ={checks['criterion_c_mu_convergence']['value']*100:.2f}%、F_5%<μ_F；"
            f"尺寸效应 B=10→100 m 的 F_5%/B 严格单调下降（幂指数 "
            f"{study['stage_d_size_effect']['power_law_q05']['nested_n2000']['exponent']:.4f} vs "
            f"Weibull 锚 1/k={study['stage_d_size_effect']['weibull_1_over_k']:.4f}）。"
            "18 个 pytest 用例、4 张图、study.json 全链工件入库。"
        ),
        "scenario_key": "ice_random_strength_field_breaking_force",
        "mission_id": 9906,
        "tags": ["TensorDEM", "冰区航行", "随机场", "Weibull", "尺寸效应",
                 "weakest-link", "P295W1-N07"],
        "markdown": DOC.read_text(),
        "figures": figs,
    }
    PAYLOAD.write_text(json.dumps(payload))
    print(f"payload written to {PAYLOAD} ({PAYLOAD.stat().st_size/1e6:.2f} MB), "
          f"{len(figs)} figures")

    req = urllib.request.Request(base + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        print(resp.read().decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())