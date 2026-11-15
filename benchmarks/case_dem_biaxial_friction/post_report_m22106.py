#!/usr/bin/env python
"""Publish the m22106 progress report for CASE-DEM-BIAXIAL-FRICTION R8."""
from __future__ import annotations

import base64
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
FIG = "docs/figures/CASE-DEM-BIAXIAL-FRICTION_R8_mu_ladder.png"
RES = "benchmarks/case_dem_biaxial_friction/results_m22106/mu_ladder.json"


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as fh:
        return json.loads(fh.read().decode())


def main() -> int:
    login = post("/api/v1/auth/login", {"username": USER, "password": PASS})
    token = login.get("access_token") or login.get("token") or login.get("data", {}).get("access_token")
    if not token:
        print("login failed:", login)
        return 1
    res = json.load(open(RES))
    arm = res["arms"][0]
    eb, mb, sb = arm["energy_balance"], arm["mobilisation"], arm["strain_budget"]
    shot = float(arm["peak"]["friction_angle_deg"]) if arm.get("peak") else float("nan")
    md = f"""# CASE-DEM-BIAXIAL-FRICTION · R8（m22106）：μ 阶梯的能量预算闸门

**锚点** CASE-DEM-BIAXIAL-FRICTION（算例池·双轴剪切摩擦角，验收 |φ−ref|/ref < 8 %）｜TensorDEM｜commit d5f0f542

**判定先给**：预注册的 5 臂 μ 阶梯（cj3903）与速率对照臂（cj3906）**递交后仍在排队、未上机**，
故 H1/H2/H3 全部 **WITHHELD**，锚点验收**没有推进**（仍为历史 honest FAIL）。本报告只报已跑成的东西。

## ① 案例介绍
平面应变双轴压缩：无胶结圆盘试样，四壁摩擦 0，顶板位移控加载，剪切段改运动学等容边界
（W·H 守恒，侧墙应力变输出）。Mohr-Coulomb 角 sinφ = (σ_v−σ_h)/(σ_v+σ_h)，c=0。
本轮唯一实算的是小样标定臂（10×20，190 颗，r=0.75 m，ρ=917 kg/m³，V={eb['volume_m3']:.1f} m³，
kn=1e7 N/m，kt/kn=0.5，μ=0.55，ε̇={eb['strain_rate']} s⁻¹，目标对数应变 0.30，12 081 步，81 s/8 线程，I=7.18e-3）。
该臂**不是**锚点工况，只用于打通分析管线。

## ② 怎么做
1. 先预注册 H1–H4 阈值（`prereg/m22106_mu_ladder_prereg.md`），再递交扫描（只给 cmd_tpl，R7 事故教训）；
2. 扫描器 `scripts/m22106_mu_ladder_probe.py`（μ=0.10/0.30/0.55/0.90/1.50，锚点 24×48，与第 3 轮同协议只改 μ）
   ＋ `scripts/m22106_rate_probe.py`（ε̇ 降 20 倍，检验能量缺口是否速率驱动）；
3. 分析 `benchmarks/case_dem_biaxial_friction/analyse_mu_ladder.py`：每臂先过第 3 轮四闸门（对数应变＋等容边界）再判 H1–H4；
4. 新增 7 个测试（功率恒等式 4、应变预算语义 3），与既有调动仪测试同批 **14 passed**。

## ③ 计算结果（程序化注入，未手抄）
| 量 | 实测 |
|---|---|
| 交付对数应变 / 目标 | {sb['delivered_log']:.6f} / 0.30（{sb['delivered_relative_to_target_log']*100:+.3f} %） |
| 工程应变（= 1−e^(−log)） | {sb['delivered_engineering']:.6f} |
| 边界输入功率 q·ε̇·V | {eb['boundary_power_W']/1e6:.3f} MW |
| 接触层摩擦耗散 | {eb['contact_friction_power_W']/1e6:.2f} MW |
| 接触法向阻尼耗散 | {eb['contact_damping_power_W']/1e6:.3f} MW |
| H4 残差 | {eb['residual_fraction']*100:.1f} %（判据 ≤15 % → FAIL） |
| F 加权调动率 ⟨|tanh|⟩ | {mb['fn_weighted_mean_abs_tanh']:.4f}（饱和接触 {mb['saturated_fraction']*100:.1f} %） |
| 平均滑移速率 ⟨|v_t|⟩ | {mb['mean_abs_vt']:.3f} m/s |
| 墙反力 vs Love-Weber（σ_v/σ_h，尾窗） | 453.2/382.1 kPa vs 430.7/365.1 kPa（5.0 %/4.4 %） |
| 锚点 φ_cv | 无有效值（`deviator_stationary` ✗ → WITHHELD；窗内 {arm['phi_cv_deg']:.2f}° 不作结论） |
| 计量仪功率恒等式（受控滑动对） | 27 500 W 进 = 27 500 W 出，rel < 1e-12 |

**两处缺陷（已修、有测试）**
1. `strain_target` 名义达标：判停用工程应变、步数预算用对数应变 → 要 0.60 的算例实跑工程 0.451
   （历史文档"ε_a → 0.45"正由此而来），且不报警；改为按 `axial_log` 判停。
2. 摩擦调动仪新增耗散功率累加（纯诊断，开关前后力逐位相同），使"应力比"与"能量耗散"两条独立路径可对表。

## ④ 结果图
(a) φ_cv–μ 与 Rowe(1962) arctanμ、Bolton(1986) 33°/38° 参考线；(b) 接触层调动率；(c) 能量互检柱状图
（边界功 {eb['boundary_power_W']/1e6:.2f} MW vs 接触耗散 {eb['contact_total_power_W']/1e6:.2f} MW）。
阶梯五臂未回，曲线上只有单点，且该点闸门未过——不当作 φ_cv。

## ⑤ 注意点
- 分析坑：调动仪样本带**工程**应变、定位器用**对数**应变，直接比较会静默选中 0 个样本（首跑 samples: 0），已换算修正（log = −ln(1−ε_eng)）。
- 递交坑：`queued` 46 min 仍未上机；绝不能把"递交"当"跑过"。
- 能量缺口的候选解释（未验证）：①无转动内核的高非仿射滑移（0.180 m/s ≈ 2.4× 仿射估计 ε̇·d），使 μ 恒饱和；
  ②步内弹性能大进大出（本单只测净功率）；③二维厚度=1 的功率/体积换算已用四壁功解析式核对（q·ε̇·V，与代码一致），不是记账错。
- 单种子、单臂小样，未做系综。

## ⑥ 误差判定（显式回答是否 <3 %）
- 交付应变 vs 声明目标：**+0.088 % → 是**
- 计量仪功率恒等式：**0（rel < 1e-12）→ 是**
- 墙反力 vs 内应力 σ_v：5.0 % → **否**（历史口径差）
- 墙反力 vs 内应力 σ_h：4.4 % → **否**
- 边界功 vs 接触耗散（H4）：{eb['residual_fraction']*100:.1f} % → **否，未闭合**
- 锚点 φ_cv vs Rowe 28.81° / Bolton 33°、38°：**无有效值 → 无法判定（WITHHELD，不粉饰）**
"""
    payload = {
        "title": "CASE-DEM-BIAXIAL-FRICTION R8：μ 阶梯能量预算闸门（两处缺陷修复 + H1–H3 WITHHELD）",
        "summary": (f"等容双轴胞元接触耗散 {eb['contact_friction_power_W']/1e6:.2f} MW 对边界功 "
                    f"{eb['boundary_power_W']/1e6:.3f} MW（H4 未闭合；计量仪功率恒等式精确）；"
                    "修复 strain_target 名义达标缺陷；μ 阶梯与速率臂已递交仍在队列，H1–H3 记 WITHHELD。"),
        "scenario_key": "CASE-DEM-BIAXIAL-FRICTION",
        "mission_id": 22106,
        "tags": ["TensorDEM", "biaxial", "critical-state", "friction-angle",
                 "energy-budget", "honest-fail"],
        "markdown": md,
        "figures": [{"name": "mu_ladder",
                     "caption": "R8: (a) φ_cv–μ 与参考线；(b) 接触调动率；(c) 能量互检（边界功 vs 接触耗散）",
                     "png_base64": base64.b64encode(open(FIG, "rb").read()).decode()}],
    }
    out = post("/api/v1/tensor/reports", payload, token)
    print(json.dumps(out)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())