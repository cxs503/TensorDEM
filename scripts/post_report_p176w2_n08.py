#!/usr/bin/env python
"""Publish the p176w2-n08 report to the platform report API.

Builds the six-element paper-grade markdown *programmatically* from
``artifacts/p176w2_n08/validation_report.json`` + the case records, embeds the
figures as base64 and POSTs to ``$TENSOR_API_BASE/tensor/reports``.

    PYTHONPATH=src python scripts/post_report_p176w2_n08.py
"""
from __future__ import annotations

import base64
import glob
import json
import os
import sys

import requests

MISSION_ID = 11326
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
ART = "artifacts/p176w2_n08"
CASES = "runs/p176w2_n08/cases"

CAPTIONS = {
    "fig1_radial_profiles.png": "Radial coarse-species area fraction profiles at the end of each run "
                                "(16 revolutions, rolling regime Fr=0.127; monodisperse control in grey).",
    "fig2_scaling.png": "Measured segregation amplitude vs (d_l/d_s - 1) with the fitted power law; "
                        "literature exponent band 0.5-1.0 and the pre-registered +-50% window.",
    "fig4_mode_map.png": "Segregation mode map in the size-ratio x density-ratio plane "
                         "(periphery = Brazil-nut, core = reverse Brazil-nut, mixed = no structure).",
    "fig5_time_series.png": "Core-shell excess growth per drum revolution (measured every 0.5 rev).",
    "fig6_negative_controls.png": "Three negative controls: size ratio -> 1, drum speed -> 0, fill -> 1.",
    "fig7_axial_bands.png": "Reduced-order axial banding field and spectrum; red line = analytic k_max.",
}


def main():
    with open(os.path.join(ART, "validation_report.json")) as fh:
        rep = json.load(fh)
    cases = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(CASES, "*.json")))]
    cases = [c for c in cases if "null" in c]

    token = requests.post(f"{API}/auth/login",
                          json={"username": USER, "password": PASS}, timeout=30).json()
    token = token.get("access_token") or token.get("token")
    headers = {"Authorization": f"Bearer {token}"}

    crit = rep["criteria"]
    c1, c2, c3, c4, c5, c6 = (crit[k] for k in ("C1", "C2", "C3", "C4", "C5", "C6"))
    ax = rep["axial_checks"]

    L = []
    A = L.append
    A("## 1. 案例介绍")
    A("")
    A("旋转圆筒内双分散颗粒床在渗滤（percolation/sieving）作用下发生径向尺寸离析：粗颗粒富集于筒壁、"
      "细颗粒形成核心——旋转筒内的 Brazil-nut 效应；密度差异大时反向（reverse Brazil-nut，粗颗粒入核心）。"
      "船舶语境：散货（矿砂/粮食/水泥）在货舱与转运环节的尺寸离析改变货堆重心与休止角，"
      "是「散货/颗粒货物」域喂给稳性输入的上游物理。几何/参数：二维圆盘筒 $R$=0.20 m，"
      "填充率 0.30，$d_s$=8 mm，$d_l/d_s\\in\\{1,1.05,1.1,1.2,1.5,2,3\\}$，$\\omega$=2.5 rad/s"
      "（Fr=0.127，滚动区），线性弹簧-阻尼接触 + 罚化库仑摩擦 + 滚动阻力，$dt$=8e-5 s，16 转，N≈581。")
    A("")
    A("## 2. 怎么做的")
    A("")
    A("自建二维软球 DEM（`src/tensordem/drum_segregation/`）：法向弹簧-阻尼、切向罚化库仑摩擦"
      "（带自旋自由度，力作用于接触点产生力矩）、滚动阻力矩、旋转壁面 $u=\\omega\\times r$、"
      "半隐式欧拉、cKDTree 邻接表；初始床由 RSA 生成后在静止筒内**动能淬火至静止**（rms<5e-3 m/s）"
      "再驱动，初始物种与位置无关联。指标两把尺子：无量纲**粒径-径向位置相关系数** $\\rho_{d,r}$"
      "（单分散$\\equiv$0）与环带浓度剖面（$S$、核-壳差），显著性用**位置置换零模型**（200 次）给 z。"
      "判据 C1–C6 由 `verification.py` 按记录数值裁决；轴向成带用降阶守恒 Cahn-Hilliard 模型并对其"
      "解析色散关系自校验。")
    A("")
    A("## 3. 计算结果")
    A("")
    A("| 案例 | 模式 | 核-壳差 | z | ρ(d,r) | z(ρ) | S | N |")
    A("|---|---|---|---|---|---|---|---|")
    for rec in sorted(cases, key=lambda r: r["name"]):
        n = rec["null"]
        A(f"| {rec['name']} | {rec['mode']} | {n['excess_observed']:+.4f} | {n['excess_z']:+.1f} | "
          f"{n['corr_observed']:+.4f} | {n['corr_z']:+.1f} | {n['S_observed']:.4f} | {rec['n_particles']} |")
    A("")
    A("| 判据 | 实测 | 阈值 | 结论 |")
    A("|---|---|---|---|")
    A(f"| C1 方向 vs 文献（≥3 粒径比） | {c1['n_compliant']}/{c1['n_ratios']} | ≥3 | {'PASS' if c1['pass'] else 'FAIL'} |")
    A(f"| C2 标度 α / R² | {c2['fit']['alpha']:.3f} / {c2['fit']['R2']:.3f} | α∈{list(c2['alpha_window'])}, R²≥0.8 | "
      f"{'PASS' if c2['alpha_pass'] and c2['R2_pass'] else 'FAIL'} |")
    A(f"| C2 幅度 ratio 3 | {c2['amplitude_at_ratio3']:+.4f} | {list(c2['amplitude_window'])} | "
      f"{'PASS' if c2['amplitude_pass'] else 'FAIL'} |")
    A(f"| C3 模式数 | {c3['n_modes']} ({', '.join(sorted(c3['modes']))}) | ≥3 | {'PASS' if c3['pass'] else 'FAIL'} |")
    A(f"| C4 单分散 ρ(d,r) | {c4['rows'][0]['corr']:.2e} | ≤1e-6 | {'PASS' if c4['pass'] else 'FAIL'} |")
    A(f"| C5 负控 (a)(b)(c) | {'/'.join('P' if c5[k]['pass'] else 'F' for k in ('a_ratio_to_one','b_omega_to_zero','c_fill_to_one'))} | 三条全过 | "
      f"{'PASS' if c5['pass'] else 'FAIL'} |")
    A(f"| C6 pytest | {c6['n_pytest']} | ≥15 | {'PASS' if c6['pass'] else 'FAIL'} |")
    A("")
    A(f"**总判定 all_pass = {rep['all_pass']}**（honest verdict，未过项如实列示）。")
    A("")
    A("轴向降阶模型解析校验：")
    for mode_row in ax["A1_growth_rate"].values():
        A(f"- A1 增长率 k={mode_row['k']:.2f}: 实测 {mode_row['measured']:.1f} vs 解析 {mode_row['theory']:.1f} 1/s"
          f"（{mode_row['rel_error']*100:.2f}%，{'PASS' if mode_row['pass'] else 'FAIL'}）")
    a2 = ax["A2_wavelength"]
    A(f"- A2 带波长: {a2['measured']:.4f} vs 解析 {a2['theory']:.4f}（{a2['rel_error']*100:.2f}%，"
      f"{'PASS' if a2['pass'] else 'FAIL'}）；A3 守恒 {ax['A3_conservation']['value']:.2e}；"
      f"A4 spinodal 负控 {'PASS' if ax['A4_spinodal_negative_control']['pass'] else 'FAIL'}；"
      f"A5 κ→0 极限 {'PASS' if ax['A5_no_kappa_limit']['pass'] else 'FAIL'}")
    A("")
    A("## 4. 结果图")
    A("")
    A("（图见 figures 字段，全部由 `scripts/drum_seg_report.py` 从记录程序化生成；"
      "图 1 说明中的转数笔误（20 转）在本版修正为实际运行的 16 转，判据数字与结论不变。）")
    A("")
    A("## 5. 注意点")
    A("")
    A("二维切片几何使绝对离析幅度不可移植到三维筒；颗粒无自旋-滚动阻力以外的形状效应；"
      "滚动阻力 $\\mu_r$=0.10 是静止床收敛的必要条件（0.02 时自旋自由度欠耗散、床呈数值沸腾态）；"
      "填充率上限控制点取 0.65 而非 0.75（RSA 无法在该密度下无重叠建床，属预注册修正）；"
      "轴向成带是降阶机制模型、系数未标定三维 DEM，故只作机制与实现验证（定量预测 WITHHELD）。")
    A("")
    A("## 6. 误差判定")
    A("")
    A("存在解析基准的量才做 3% 级判定：轴向增长率、质量守恒、单分散零指标均通过（见上）；"
      "粒径离析的方向/标度类判据的基准是**文献定性方向与量级带**，无可数字化 3% 基准，"
      "因此不宣称 3%，只按预注册的 ±50% 量级窗口裁决，并如实标注。")

    figures = []
    for name, rel in rep["figures"].items():
        path = os.path.join(ART, "figures", os.path.basename(rel))
        if not os.path.exists(path):
            continue
        figures.append({
            "name": os.path.basename(path),
            "caption": CAPTIONS.get(os.path.basename(path), ""),
            "png_base64": base64.b64encode(open(path, "rb").read()).decode(),
        })

    payload = {
        "title": "p176w2-n08 · 旋转筒双分散颗粒离析（Brazil-nut）验证面：方向判据/标度律/模式图/三负控",
        "summary": ("（本报告替代 #12984：仅修正图 1 说明文字中的转数笔误，判据数字与结论不变）"
                    f"二维软球 DEM 旋转筒离析验证面：C1 方向 {c1['n_compliant']}/{c1['n_ratios']}、"
                    f"C2 标度 α={c2['fit']['alpha']:.2f}、C3 模式 {c3['n_modes']} 类、"
                    f"C4 单分散零指标、C5 三负控、C6 {c6['n_pytest']} 例；all_pass={rep['all_pass']}。"),
        "scenario_key": "dem.rotating_drum.segregation",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "granular", "segregation", "brazil-nut", "rotating-drum",
                 "p176w2-n08", "verification"],
        "markdown": "\n".join(L),
        "figures": figures,
    }
    r = requests.post(f"{API}/tensor/reports", headers=headers, json=payload, timeout=120)
    print(r.status_code, r.text[:500])
    if r.status_code >= 300:
        sys.exit(1)


if __name__ == "__main__":
    main()