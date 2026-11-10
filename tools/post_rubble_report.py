#!/usr/bin/env python
"""Publish the IN69-N06 report to the platform (paper-level, figures embedded)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
DOC = "docs/IN69_N06_rubble_accumulation.md"
ART = "artifacts/rubble_accumulation"
CAPTIONS = {
    "fig1_growth_laws.png":
        "堆积几何 L_p/B_p/H_p 随时间演化与 y=y_inf(1-exp(-t/tau)) 拟合（D=3 m, h_i=0.25 m, v_i=0.3 m/s，三浓度）",
    "fig2_shielding_timeseries.png":
        "迎冰面压力时程：有堆积 vs 清除型基准 p_0（同冰场同驱动的孪生算例）",
    "fig3_shielding_vs_fr.png":
        "屏蔽因子 R_shield 随 Froude 数变化，左：自由流动量通量基准；右：清除型孪生基准（实线参考组，虚线 20 kPa 板压处理组）",
    "fig4_scaling.png": "稳态几何标度 B_inf/D、H_inf/h_i、L_inf/D 与 Fr 的双对数关系",
    "fig5_geometry_families.png": "不同碎冰浓度下堆积几何量的演化族",
    "fig6_thrust_decision.png":
        "修订 1 决策图：20 kPa 冰盖推力边界的配对比——堆积高度未达决策阈值（H_p/(2r)>1.5），载荷上升但堆积体变薄，故板压不作为 sail 机制",
}


def login() -> str:
    req = urllib.request.Request(
        f"{BASE}/auth/login",
        data=json.dumps({"username": USER, "password": PASS}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as fh:
        return json.load(fh)["access_token"]


def main() -> int:
    token = login()
    markdown = open(DOC).read()
    # publish stamp: keeps the report body identical to the committed document
    # while giving the platform row a stable, distinct identity on republication
    import hashlib
    stamp = hashlib.sha256(markdown.encode()).hexdigest()[:12]
    markdown = markdown + (f"\n> 发布戳：正文 sha256[:12] = {stamp}；"
                           "工件目录 artifacts/rubble_accumulation/；"
                           "本报告由 tools/post_rubble_report.py 自动发布。\n")
    figures = []
    for name, cap in CAPTIONS.items():
        path = os.path.join(ART, name)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            figures.append({"name": name, "caption": cap,
                            "png_base64": base64.b64encode(fh.read()).decode()})
    payload = {
        "title": "IN69-N06 固定圆柱周围碎冰堆积：DEM 几何增长律、双基准屏蔽因子与稳态标度（honest FAIL 记录）",
        "summary": (
            "三维软球 DEM（入口质量通量进料 + 水拖曳，G7 组加 20 kPa 冰盖推力压力边界）扫描 43 个算例，"
            "研究碎冰场绕固定圆柱的堆积：给出 L_p/B_p/H_p 增长律拟合、双基准屏蔽因子"
            "（自由流动量通量 C_d=2 与清除型孪生）、稳态几何标度指数、质量账本与两项负控。"
            "诚实判定：质量闭合（最大 1.8e-3）与负控通过；增长律 4/57 拟合达 R²>0.95 判 FAIL；"
            "稳态标度无权威文献对照判 WITHHELD，且堆积高度仅 H_p/(2r)=0.15–0.44（球元无互锁，"
            "不形成 sail，20 kPa 板压仅使 H 提高 1.2–1.6 倍，按预注册规则 REJECT）；"
            "屏蔽假设被否定——所有有载算例 R>1（动量基准 2.0–163，孪生基准 1.44–12.7），"
            "即堆积体只会放大而非屏蔽迎冰载荷。同时报告两处修复：清除基准 p_0≡0 缺陷、"
            "板压边界无速度上限导致的能量泵入缺陷。"),
        "scenario_key": "rubble_accumulation_cylinder",
        "mission_id": 7234,
        # the platform accepts at most 10 tags
        "tags": ["TensorDEM", "ice_rubble", "DEM", "shield_factor", "Froude",
                 "IN69-N06", "cylinder", "honest_FAIL", "mass_ledger",
                 "prereg"],
        "markdown": markdown,
        "figures": figures,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE}/tensor/reports", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as fh:
        print(fh.status, fh.read()[:400].decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())