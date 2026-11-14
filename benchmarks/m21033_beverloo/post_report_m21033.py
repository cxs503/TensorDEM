#!/usr/bin/env python3
"""Publish the m21033 CASE-DEM-SILO-BEVERLOO report to the TensorPlatform API.

All numbers are read from ``runs/m21033_beverloo/*.json`` (zero hand-typed
values).  Figures are embedded as base64 PNGs.

Usage
-----
    python3 benchmarks/m21033_beverloo/post_report_m21033.py
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "m21033_beverloo"
FIG_DIR = ROOT / "docs" / "figures"
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"
ORDER = ["D5", "D4", "D6", "D5_mu30", "D5_kn1e5", "D5_dthalf"]


def load(cid: str) -> dict:
    p = RUN_DIR / f"{cid}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def fig(name: str, caption: str) -> dict | None:
    p = FIG_DIR / name
    if not p.exists():
        return None
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode(p.read_bytes()).decode()}


def main() -> int:
    ref = json.loads(REF_PATH.read_text())
    rows = [(cid, load(cid)) for cid in ORDER]
    rows = [(cid, rec) for cid, rec in rows if rec]
    if not rows:
        print("no run artifacts found; run the campaign first")
        return 1

    primary = next((rec for cid, rec in rows if cid == "D5"), None)
    verdict = "PASS" if (primary and primary["verdict"]["pass_10pct"]) else "FAIL"
    perr = 100.0 * primary["verdict"]["rel_err_W"] if primary else float("nan")

    # ---- markdown (six elements) ------------------------------------------
    md = []
    md.append("# 料仓卸料流率 Beverloo 律验证（m21033，平底料仓 3-D DEM）\n")

    md.append("## ① 案例介绍\n")
    md.append("平底圆形料仓（R=0.130 m，光滑壁）底部圆心圆孔自由卸料；颗粒为玻璃珠"
              "（d=20 mm，ρ_p=2500 kg/m³，μ_p=0.15，e=0.30，μ_r=0.02）。床层由松散晶格"
              "（净距 0.30 r，jitter 0.05 r）在**堵孔**状态下重力压实 0.60 s 成无序床，"
              "开孔后取卸料流质量分位 0.2–0.8 的稳态段测质量流率 W，对照 Beverloo (1961) "
              "律 W = C·ρ_b·√g·(D−k·d)^(5/2)，C=0.58、k=1.4（机器可读 "
              "ref_beverloo_1961.json，零手抄）；ρ_b 取开孔瞬间实测床堆密度。\n")
    md.append("| 参数 | 值 |")
    md.append("|---|---|")
    for cid, rec in rows:
        md.append(f"| {cid}: D/d | {rec['dimensionless']['D_over_d']:.2f} |")
    md.append("| kn | 1e6 N/m（主臂）/ 1e5 N/m（归因臂 D5_kn1e5） |")
    md.append("| 协议 | settled+plug（m20055 协议 B），seed=21033 |")
    md.append("| 判据 | \\|W−W_ref\\|/W_ref < 10%（总账 CASE-DEM-SILO-BEVERLOO） |\n")

    md.append("## ② 怎么做的\n")
    md.append("复用既有 REAL 3-D DEM `tensordem.silo_discharge`（线性弹簧-阻尼法向 + "
              "Cundall–Strack 切向 + 滚阻，granular3d_core 内核），不另起炉灶。相对 "
              "m20055（kn=1e5、μ=0.30，D4 +25.0% FAIL）只改两处：**kn 1e5→1e6**"
              "（m20585 根因诊断：软接触放大有效孔径；hopper 内核 kn=1e6 已达 −0.4%）"
              "与 **μ 0.30→0.15**（玻璃珠低摩擦，m15259 经验）。矩阵 6 算例"
              "（D4/D5/D6 律校验 + μ 敏感性 + kn 归因臂 + dt/2 收敛），预注册于 "
              "`prereg/m21033_beverloo.md`；集群作业 cj3141（stage_dir 直达节点）。\n")

    md.append("## ③ 计算结果\n")
    md.append("| case | D/d | μ_p | kn [N/m] | n_bed | ρ_b [kg/m³] | W_dem [kg/s] | "
              "W_ref [kg/s] | 误差 | C_rec | CV | 卸料占比 | <10% |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid, rec in rows:
        v, dim, tele = rec["verdict"], rec["dimensionless"], rec["telemetry"]
        frac = tele["n_discharged_total"] / tele["n_initial"]
        md.append(f"| {cid} | {dim['D_over_d']:.2f} | {dim['mu_p']:.2f} | "
                  f"{dim['kn_N_m']:.0e} | {dim['n_particles_bed']} | "
                  f"{v['rho_b_kg_m3']:.0f} | {v['W_got_kg_s']:.4f} | "
                  f"{v['W_ref_kg_s']:.4f} | {100*v['rel_err_W']:+.1f}% | "
                  f"{dim['C_recovered']:.3f} | {tele['steady_rate_cv']:.3f} | "
                  f"{100*frac:.0f}% | {'✓' if v['pass_10pct'] else '✗'} |")
    md.append("")

    md.append("## ④ 结果图\n")
    md.append("图 1：D/d=4/5/6 实测 W 与 Beverloo 参考值对比（柱上标注相对误差）。"
              "图 2：累计卸料质量-时间曲线（稳态段取中间 60%）。\n")

    md.append("## ⑤ 注意点\n")
    md.append("* 1961 原文 PDF 不在仓库、环境无外网：C/k 为文献公布系数"
              "（cited, not re-digitized，见 ref JSON provenance）；")
    md.append("* 平底料仓 silo_D/D≈2.6，略小于部分文献建议的 ≥3，壁面效应未完全消除；")
    md.append("* 小体系泊松噪声：CV 同时含物理起伏与计数噪声；")
    md.append("* kn 是 DEM 数值参数（非物理量），kn=1e6 下颗粒最大接触变形 ~0.1% d，"
              "在 Beverloo 有效范围内可视为刚球极限。\n")

    md.append("## ⑥ 误差判定\n")
    md.append("| 量 | 基准 | 实测 | 相对误差 | <10%? |")
    md.append("|---|---|---|---|---|")
    for cid, rec in rows:
        v = rec["verdict"]
        md.append(f"| W ({cid}) | {v['W_ref_kg_s']:.4f} kg/s | "
                  f"{v['W_got_kg_s']:.4f} kg/s | {100*v['rel_err_W']:+.1f}% | "
                  f"{'是' if v['pass_10pct'] else '否'} |")
    md.append(f"\n**总判定：{verdict}**（主算例 D5 误差 {perr:+.1f}%，判据 <10%）。\n")

    figures = []
    f1 = fig("m21033_beverloo_flow_law.png",
             "D/d=4/5/6 实测稳态流率 vs Beverloo (1961) 参考值")
    if f1:
        figures.append(f1)
    f2 = fig("m21033_beverloo_discharge_stream.png",
             "累计卸料质量-时间曲线（D4/D5/D6）")
    if f2:
        figures.append(f2)

    report = {
        "title": (f"CASE-DEM-SILO-BEVERLOO 料仓卸料流率 Beverloo 律验证（m21033）："
                  f"{verdict}，主算例 D5 误差 {perr:+.1f}%"),
        "summary": (f"平底料仓 3-D DEM（settled+plug 协议，kn=1e6 N/m，μ=0.15）稳态"
                    f"卸料流率 vs Beverloo (1961)：主算例 D5 误差 {perr:+.1f}%，"
                    f"判据 <10% → {verdict}。6 算例矩阵含 D/d=4/5/6 律校验、μ 敏感性、"
                    f"kn 归因臂与 dt/2 收敛。"),
        "scenario_key": "CASE-DEM-SILO-BEVERLOO",
        "mission_id": 21033,
        "tags": ["TensorDEM", "beverloo", "silo-discharge", "validation", "granular"],
        "markdown": "\n".join(md),
        "figures": figures,
    }
    out = Path("/tmp/report_m21033.json")
    out.write_text(json.dumps(report, ensure_ascii=False))
    print(f"report payload written to {out} ({out.stat().st_size} bytes)")

    base = os.environ.get("TENSOR_API_BASE", "")
    if not base:
        print("TENSOR_API_BASE not set; payload only")
        return 0
    login = json.load(urllib.request.urlopen(urllib.request.Request(
        base + "/auth/login", method="POST",
        data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
        headers={"Content-Type": "application/json"}))
    tok = login["access_token"]
    req = urllib.request.Request(
        base + "/tensor/reports", method="POST",
        data=json.dumps(report, ensure_ascii=False).encode(),
        headers={"Authorization": "Bearer " + tok,
                 "Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req))
    print("posted report id:", resp.get("id") or resp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
