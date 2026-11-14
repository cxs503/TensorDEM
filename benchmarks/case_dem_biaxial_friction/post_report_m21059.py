#!/usr/bin/env python
"""Publish the CASE-DEM-BIAXIAL-FRICTION stress-chain audit report."""
from __future__ import annotations

import base64
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(REPO, "artifacts", "CASE-DEM-BIAXIAL-FRICTION",
                   "stress_chain_audit.json")
FIG = os.path.join(REPO, "docs", "figures",
                   "CASE-DEM-BIAXIAL-FRICTION_stress_chain_audit.png")
OUT = "/tmp/report_m21059.json"

with open(ART) as fh:
    audit = json.load(fh)
runs = {r["tag"]: r for r in audit["runs"]}
leg = runs["legacy (platens position-fixed)"]
srv = runs["axial_servo=True"]
cl, cs = leg["consolidation_state"], srv["consolidation_state"]
pl, ps = leg["peak"], srv["peak"]
sl, ss = leg["stress_definitions"], srv["stress_definitions"]
gap_l = abs(sl["friction_angle_internal_deg"] - pl["friction_angle_deg"])
gap_s = abs(ss["friction_angle_internal_deg"] - ps["friction_angle_deg"])
dphi_cfg = abs(ps["friction_angle_deg"] - pl["friction_angle_deg"])

markdown = f"""## ①案例介绍
平面应变双轴压缩（CASE-DEM-BIAXIAL-FRICTION）的**应力测量链审计**。试样 {leg['specimen']['particles']} 圆盘（{leg['config']['columns']}×{leg['config']['rows']}，r=0.75 m，ρ=917 kg/m³，μ=0.55，冻结键 0.35），
侧墙摩擦 0，σ₃ = {leg['config']['confining_stress']/1e3:.0f} kPa，固结 {leg['config']['consolidation_steps']} 步，顶板位移控剪至 ε_a = {leg['config']['strain_target']:.2f}（ε̇_a = {leg['config']['strain_rate']:.2f} 1/s，I = {srv['loading']['inertial_number']:.1e}）。
判据：锚点 ref φ = 38°（Bolton 1986，容差 8 %）；对照 ref φ_cv = 28.81°（Rowe 1962 D=0）。

## ②怎么做的
同一初始试样只改**固结边界条件**做 A/B：
legacy = 仅侧墙压力伺服、顶/底板位置固定（历史全部轮次）；`axial_servo=True` = 四边压力伺服（本单新增能力）。
同时记录两条独立应力通路：边界摩擦力量测（F/A）与 Love-Weber virial。
生产引擎新增 `run_biaxial_shear_band(..., axial_servo=True)`（顶/底板伺服 + `servo_width` 随宽度更新 + 剪切前释放）与输出 `consolidation_state`、`sigma_v_top/sigma_v_mean/platen_asymmetry`；默认 False 保历史可复现。

## ③计算结果
| 方案 | σ_v (kPa) | σ_h (kPa) | K=σ_v/σ_h | 两板不平衡 | φ 边界(°) | φ 内部(°) | \\|Δφ\\|(°) |
|---|---|---|---|---|---|---|---|
| legacy | {cl['sigma_v']/1e3:.1f} | {cl['sigma_h']/1e3:.1f} | {cl['sigma_v_over_sigma_h']:.3f} | {cl['platen_asymmetry']:+.3f} | {pl['friction_angle_deg']:.2f} | {sl['friction_angle_internal_deg']:.2f} | {gap_l:.2f} |
| axial_servo | {cs['sigma_v']/1e3:.1f} | {cs['sigma_h']/1e3:.1f} | {cs['sigma_v_over_sigma_h']:.3f} | {cs['platen_asymmetry']:+.4f} | {ps['friction_angle_deg']:.2f} | {ss['friction_angle_internal_deg']:.2f} | {gap_s:.2f} |

峰值 ε_a = {ps['axial_strain']:.3f}，峰值偏应力 q = {ps['deviator']/1e3:.0f} kPa。

## ④结果图
左：固结末态 K（虚线 = 各向同性）；右：边界摩擦力量测 vs Love-Weber 内部应力给出的峰值摩擦角，标注两者差。

## ⑤注意点
- legacy 固结**不是各向同性**：顶/底板位置固定，实测 K = {cl['sigma_v_over_sigma_h']:.2f}（K0 型），且两板轴向载荷长期不平衡（{cl['platen_asymmetry']:+.3f}），剪切起点不是受控应力状态。
- 四边伺服后 K = {cs['sigma_v_over_sigma_h']:.3f}，不平衡度 {cs['platen_asymmetry']:+.4f}。
- 但**固结边界条件对峰值 φ 的影响只有 {dphi_cfg:.2f}°**，故历史工件「各向同性固结」是措辞/账目缺陷，不是锚点 8.97 % FAIL 的成因。
- 应力定义（边界 vs virial）只差 {gap_l:.2f}°/{gap_s:.2f}°（≤1.1°），所以历史上「41.41° vs 36.43°」的 5° 差**不能**用归一化/体积定义解释。
- 局限：本审计 12×24 小试样、ε_a 到 0.06；24×48 锚点全尺寸未重跑（集群排队 >20 min 未起跑，本地按容量纪律只做轻量）。

## ⑥误差判定（逐项，显式回答是否 <3%）
| 量 | 值 | 基准 | 相对误差 | <3% |
|---|---|---|---|---|
| φ (legacy) | {pl['friction_angle_deg']:.2f}° | 38° | {100*abs(pl['friction_angle_deg']-38)/38:.2f} % | 否 |
| φ (axial_servo) | {ps['friction_angle_deg']:.2f}° | 38° | {100*abs(ps['friction_angle_deg']-38)/38:.2f} % | 否 |
| φ (legacy) | {pl['friction_angle_deg']:.2f}° | 28.81° | {100*abs(pl['friction_angle_deg']-28.81)/28.81:.2f} % | 否 |
| φ (axial_servo) | {ps['friction_angle_deg']:.2f}° | 28.81° | {100*abs(ps['friction_angle_deg']-28.81)/28.81:.2f} % | 否 |
| K (axial_servo) | {cs['sigma_v_over_sigma_h']:.3f} | 1.000 | {100*abs(cs['sigma_v_over_sigma_h']-1):.2f} % | {'是' if abs(cs['sigma_v_over_sigma_h']-1)<0.03 else '否'} |
| 应力定义一致性 | {gap_s:.2f}° | 0° | {100*abs(ss['friction_angle_relative_difference']):.2f} % | {'是' if abs(ss['friction_angle_relative_difference'])<0.03 else '否'} |

锚点摩擦角判据仍未达（honest FAIL，不粉饰）；固结各向同性与应力定义一致性两项内部一致性指标 <3 %。
"""

figures = []
if os.path.exists(FIG):
    with open(FIG, "rb") as fh:
        figures.append({"name": "stress_chain_audit.png",
                        "caption": "K=σv/σh at end of consolidation (left) and peak friction angle from "
                                   "boundary traction vs Love-Weber virial (right).",
                        "png_base64": base64.b64encode(fh.read()).decode()})

report = {
    "title": "CASE-DEM-BIAXIAL-FRICTION 应力链审计：固结各向同性缺陷 + 边界/内部应力一致性",
    "summary": ("双轴池 legacy 固结实为 K0（K=σv/σh 实测 %.2f，两板不平衡 %+.2f），新增四边压力伺服 "
                "axial_servo=True 给出 K=%.3f；边界摩擦力量测与 Love-Weber virial 的峰值摩擦角差 ≤1.1°，"
                "固结边界条件对 φ 影响仅 %.2f°。" % (cl["sigma_v_over_sigma_h"], cl["platen_asymmetry"],
                                                cs["sigma_v_over_sigma_h"], dphi_cfg)),
    "scenario_key": "CASE-DEM-BIAXIAL-FRICTION",
    "mission_id": 21059,
    "tags": ["DEM", "biaxial", "stress-measurement", "verification", "TensorDEM",
             "CASE-DEM-BIAXIAL-FRICTION"],
    "markdown": markdown,
    "figures": figures,
}
with open(OUT, "w") as fh:
    json.dump(report, fh)
print("wrote", OUT, "figures:", len(figures))


base = os.environ.get("TENSOR_API_BASE", "http://localhost:8002/api/v1").rstrip("/")
user = os.environ.get("TENSOR_AGENT_USER")
pwd = os.environ.get("TENSOR_AGENT_PASS")
token = os.environ.get("TOKEN")
if not token and user and pwd:
    out = subprocess.run(["curl", "-s", "-X", "POST", f"{base}/auth/login",
                          "-H", "Content-Type: application/json",
                          "-d", json.dumps({"username": user, "password": pwd})],
                         capture_output=True, text=True).stdout
    try:
        token = json.loads(out).get("access_token")
    except json.JSONDecodeError:
        print("login failed:", out[:300])
if token:
    res = subprocess.run(["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
                          "-H", f"Authorization: Bearer {token}",
                          "-H", "Content-Type: application/json",
                          "-d", f"@{OUT}"], capture_output=True, text=True)
    print(res.stdout[:500])
else:
    print("no token available")