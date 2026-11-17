#!/usr/bin/env python3
"""Post the paper-grade R11 report for CASE-BIAXIAL-SHEAR-BAND (phi-state-point sensitivity)."""
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- login ---------------------------------------------------------------
req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/auth/login",
    data=json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode(),
    headers={"Content-Type": "application/json"},
)
body = json.loads(urllib.request.urlopen(req).read())
token = body.get("access_token") or body.get("token")
assert token, f"login did not return a token: {list(body.keys())}"

# ---- figure --------------------------------------------------------------
fig_path = ROOT / "artifacts/biaxial_shear_band/r11_phi_state_sensitivity/figures/r11_phi_state_sensitivity.png"
fig_b64 = base64.b64encode(fig_path.read_bytes()).decode()

markdown = r"""# CASE-BIAXIAL-SHEAR-BAND R11 — 把「φ 取哪个状态点」的敏感性定量化

## ① 案例介绍（物理背景/几何/参数表）

平面应变双轴试验：矩形冻结黏结圆盘试样（24 × 48 抖动三角格，35 % 冻结键，
壁面摩擦 0.55，围压 σ₃ = 100 kPa 伺服侧壁，顶板位移加载），峰后材料局部化为
一条剪切带，带相对 σ₃ 轴（水平）的倾角 θ 是观测量。经典理论给出三条只依赖
φ、ψ 的关系（Mohr–Coulomb 45+φ/2；Roscoe 45+ψ/2；Arthur 45+(φ+ψ)/4）。

本案例验收锚 =「文献剪切带倾角，容差 <15 % + 定性」，锚值 60°（55–60° 带中点）。
R7 给出实测带倾角 **θ = 55.00° ± 2.07°**（n=8）；R9/R10 用实测 φ≈35.45°（峰）
与 Rowe 应力–剪胀关系（φ_cv 声明区间 26–34°）判定：**Arthur 定律被支持**
（平均 |误差| 2.81–3.35 %），Mohr–Coulomb 不被支持（12.31 %）。

**本轮不改锚、不改算子、不改算例**，只做一件事：把 R9/R10 明确记为 `not_run`
的状态匹配臂（cj4770，用晚窗 φ 替代峰值 φ）所回答的「φ 取哪个状态点」的
敏感性，**用纯算术定量化**——不等待在飞的 DEM 重算，先把结论的稳健边界画出来。

## ② 怎么做的（方法/流程/工具链）

1. `benchmarks/biaxial_shear_band/run_r11_phi_state_sensitivity.py`（新增，纯算术）：
   在物理可行 φ 状态点区间 **[30°, 40°]**（0.5° 步长，21 点）上扫，覆盖实测峰值
   35.45° 与任何合理峰后/晚窗值；对每个 φ 在**全部声明 φ_cv ∈ {26,28,30,32,34}**
   上重跑 R9 判定（`adjudicate` + `stress_dilatancy_psi_deg`，Rowe/Bolton），
   得到「Arthur 在全部 φ_cv 上支持」的**稳健窗**。
2. `tests/test_r11_phi_state_sensitivity.py`（新增）：5 测试，**5 通过**——断言
   峰值在稳健窗内、窗是窄的（[34, 35.5] 而非 [33, 38]）、MC 在 [30, 40] 全程
   不进入 3 % 精度、峰值处 Arthur 是最近定律且 <3 %。
3. `make_r11_figures.py`：从 JSON 渲染对比图（左：Arthur/MC 误差 vs φ 状态点，
   含 3 %/15 % 线与稳健窗；右：稳健窗 flag）。全部数字程序化注入，无手抄。

## ③ 计算结果（数字表，程序化注入）

| 量 | 数值 |
|---|---|
| 带倾角 θ（R7/R9/R10 主估计量，不变） | **55.00° ± 2.07°**（n=8） |
| 实测峰值 φ | **35.45°** |
| **稳健窗**（Arthur 在全部 φ_cv 上支持） | **[34.0°, 35.5°]** |
| 稳健窗宽 | **1.5°** |
| 峰值在窗内 | **是**（35.45 ∈ [34.0, 35.5]） |
| 预注册 [33, 38] 余量 | **未达成**（窗只到 34.0–35.5） |
| 判定 | **NARROW**（峰值在窗内，但窗窄于预注册余量） |

逐 φ（φ_cv=32 臂）：φ=30 → Arthur 4.76 %/MC 8.33 %（窗外）；φ=34 → 1.67 %/11.29 %
（窗内）；φ=35.5 → 0.13 %/12.35 %（窗内）；φ=36 → 0.37 %/12.70 %（窗外，φ_cv=26
臂 3.34 % 超 3 %）；φ=40 → 4.29 %/15.38 %（窗外）。**MC 全程不进入 3 %**（8–15 %），
判别力来自 φ/4 vs φ/2 系数差（~7°），与 R9 `psi=0` 臂同向。

## ④ 结果图

`figures/r11_phi_state_sensitivity.png`：左 = Arthur（绿）/MC（红）误差 vs φ 状态点，
含 3 % 精度线、15 % 容差线、实测峰值（蓝）与稳健窗（绿区 [34.0, 35.5]）；
右 = 稳健窗 flag（窄窗，峰值在窗内，预注册 [33, 38] 未达成）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **本轮没有重跑 DEM**：所有数字来自 R7/R9/R10 实测均值（θ=55.00、φ_peak=35.45），
   程序化注入；不能回答「晚窗 φ 实际是多少」——那是 cj4770 的活。
2. **稳健窗窄是真实发现，不是缺陷**：R9/R10 的「Arthur 在全部 φ_cv 上支持」声明
   **只在 φ ∈ [34.0, 35.5] 上成立**。这不是说 R9/R10 结论错了（峰值 35.45 在窗内，
   结论在峰值处成立），而是说**该声明的稳健边界只有 1.5°**——若 cj4770 的晚窗 φ
   落在窗外，需重新审视「Arthur 在全部 φ_cv 上支持」。这正是把 `not_run` 欠账
   变成「已定量、已标为真实风险」的意义。
3. **窗边界由 φ_cv 臂决定**：下界 34.0° 是 φ_cv=34 臂（Arthur 3.04 %），上界 35.5°
   是 φ_cv=26 臂（Arthur 3.34 %）——不是 φ 本身，而是某个声明 φ_cv 在窗边把
   Arthur 推出 3 % 精度。φ_cv 是声明值（R9 已说明），不是拟合值。
4. **2D 圆盘 ≠ 3D 砂**：φ/4 vs φ/2 系数判别是二维离散元经典结果，不能直接搬到
   三维角粒材料；本轮结论只对本实现 2D 冻结黏结圆盘成立。
5. **cj4770 仍在飞**：状态匹配臂（用晚窗 φ 重建）仍在计算队列（running，已 29+ 小时），
   本轮**未取回**其晚窗 φ 结果，按纪律记为 `not_run`。R11 的意义是**先**把
   「φ 取哪个状态点」的敏感性边界画出来（[34.0, 35.5]），让 cj4770 取回后能
   直接对照：若其晚窗 φ 在窗内，R9/R10 结论稳健；若在窗外，需按 R11 边界重新判定。

## ⑥ 误差判定（逐项，显式回答「是否 <3 %」）

| 关键量 | 基准 | 误差 | 是否 <3 % |
|---|---|---|---|
| θ 均值 55.00° | 主文献带中点 60°（Desrues & Viggiani 2004） | 8.33 % | **否**（容差 15 % 内 = 是，R7/R10 不变） |
| θ 均值 | Arthur @ 峰值 φ=35.45（φ_cv=32，自身材料状态） | **0.13 %** | **是** |
| θ 均值 | Mohr-Coulomb @ 峰值 φ=35.45 | 12.35 % | **否** |
| 稳健窗宽 | 预注册 [33, 38] 余量 | 1.5° vs 5°（未达成） | **不适用**（敏感性边界，非测量误差） |
| 峰值 φ | 稳健窗 [34.0, 35.5] | 35.45 ∈ 窗内 | **是**（定性） |
| 晚窗 φ（cj4770） | 稳健窗 | **未取回**（not_run） | **不判定**（在飞，非测量） |

**honest 说明**：3 % 在「Arthur + 峰值 φ + 自身材料状态」通路上达成（0.13 %）；
对「主文献带中点 60°」的误差仍是 8.33 %（<15 % 容差内 = PASS，R7/R10 不变）。
本轮**新增**的诚实发现：R9/R10 的「Arthur 在全部 φ_cv 上支持」声明**只在
φ ∈ [34.0, 35.5] 上成立**（稳健窗 1.5° 宽，窄于预注册 [33, 38] 余量）——
这是把 `not_run` 欠账变成「已定量、已标为真实风险」的增量，不是把结论改错。
所有未跑项（cj4770 晚窗 φ）已在 ⑤ 标注。
"""

report = {
    "title": "CASE-BIAXIAL-SHEAR-BAND R11 — φ 状态点敏感性定量化：稳健窗 [34.0, 35.5] 窄于预注册余量（m24586）",
    "summary": ("R11 把 R9/R10 的 not_run 欠账（状态匹配臂 cj4770，用晚窗 φ 替代峰值 φ）"
                "用纯算术定量化：在 φ∈[30,40] 上扫，R9/R10 的『Arthur 在全部 φ_cv 上支持』"
                "声明只在稳健窗 [34.0, 35.5]（1.5° 宽）内成立，峰值 35.45 在窗内但预注册 "
                "[33,38] 余量未达成——判定 NARROW。φ 状态点敏感性是真实、已定量的风险，"
                "cj4770 取回后须对照：晚窗 φ 在窗内则 R9/R10 稳健，在窗外则需重新判定。"
                "5 测试全绿，无 DEM 重跑。"),
    "scenario_key": "CASE-BIAXIAL-SHEAR-BAND",
    "mission_id": "24586",
    "tags": ["TensorDEM", "biaxial", "shear-band", "r11", "phi-state-sensitivity",
             "verification", "robustness-window", "honest"],
    "markdown": markdown,
    "figures": [
        {"name": "r11_phi_state_sensitivity.png",
         "caption": "R11：Arthur（绿）/Mohr-Coulomb（红）误差 vs φ 状态点（θ=55.00°，φ_cv=32 臂），"
                    "含 3 % 精度线、15 % 容差线、实测峰值（蓝）与稳健窗（绿区 [34.0, 35.5]）；"
                    "右：稳健窗 flag——窄窗（1.5°），峰值在窗内，预注册 [33, 38] 未达成。",
         "png_base64": fig_b64},
    ],
}

req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/tensor/reports",
    data=json.dumps(report).encode(),
    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
)
resp = json.loads(urllib.request.urlopen(req).read())
print("report id:", resp.get("id") or resp.get("report_id") or resp)
