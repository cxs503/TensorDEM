#!/usr/bin/env python3
"""m16127 board-audit report: figure + paper-style six-section markdown + platform post."""
from __future__ import annotations

import base64
import io
import json
import math
import os
import pathlib
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

AUDIT = pathlib.Path(__file__).resolve().parents[2]
ART = AUDIT / "artifacts" / "audit_m16065"
SRC = pathlib.Path("/root/tp-agent/work/m16065/benchmarks/case_mindlin_tangential")
CHK = json.loads((ART / "independent_check.json").read_text())
R = json.loads((SRC / "results.json").read_text())
C = np.load(SRC / "curves.npz")

P = R["parameters"]["normal_load_N"]
MU = R["parameters"]["friction_coefficient"]
DS = CHK["hertz"]["delta_star"]
G = CHK["hertz"]["g_star"]
A = CHK["hertz"]["a"]
E = CHK["hertz"]["e_star"]
RS = CHK["hertz"]["r_star"]

art = SRC / "benchmarks/case_mindlin_tangential" if False else SRC  # keep pathlib simple

# independent references (no TensorDEM import)
vv = lambda d: MU * P * (1.0 - (1.0 - min(d / DS, 1.0)) ** 1.5)


def rev_ref(amp):
    p = lambda x: E * (A * A - x * x) / RS
    xr = math.sqrt(max(0.0, A * A - 2.0 * G * amp * RS / (MU * E)))
    st, _ = quad(lambda x: MU * p(x) - 4.0 * G * amp, 0.0, xr, limit=200)
    sl, _ = quad(lambda x: MU * p(x), xr, A, limit=200)
    return -2.0 * (st - sl)


fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
dl, fl = C["delta_law"], C["force_law"]
ax[0].plot(dl / DS, fl / (MU * P), "-", lw=2, label="TensorDEM G1 (delivered)")
grid = np.linspace(0, 1.4, 300)
ax[0].plot(grid, [vv(x * DS) / (MU * P) for x in grid], "--", lw=1.4,
           label="independent Mindlin ref (audit)")
lin = C["delta_lin"], C["force_lin"]
ax[0].plot(lin[0] / DS, lin[1] / (MU * P), ":", lw=1.2, label="linear Cundall–Strack (ALT)")
ax[0].axhline(1.0, color="grey", lw=0.6)
ax[0].axvline(1.0, color="grey", lw=0.6)
ax[0].set_xlabel(r"$\delta/\delta^*$")
ax[0].set_ylabel(r"$F_t/(\mu P)$")
ax[0].set_title("virgin loading: max dev "
                f"{CHK['g1_error_metrics']['micro_max_dev_over_muP']:.2e}" + r"$\mu P$")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

n_fwd = int(C["n_forward"])
dmax = float(C["delta_rev"][n_fwd - 1])
amp = dmax - C["delta_rev"][n_fwd - 1:]
ax[1].plot(amp / DS, C["force_rev"][n_fwd - 1:] / (MU * P), "-", lw=2, label="delivered law (envelope)")
ag = np.linspace(0, 2.0, 120)
ax[1].plot(ag, [rev_ref(x * DS) / (MU * P) for x in ag], "--", lw=1.4,
           label="independent reversal ref (2$\\delta^*$)")
ax[1].plot([0, 2.0], [-1, 1], ":", color="grey", lw=0.8)
ax[1].set_xlabel(r"reversed amplitude $\Delta/\delta^*$")
ax[1].set_ylabel(r"$F_t/(\mu P)$")
ax[1].set_title("reversal branch F1: dev "
                f"{CHK['reversal']['max_dev_over_muP']:.3f}" + r"$\mu P$; "
                f"full reversal at {CHK['reversal']['delivered_law_full_reversal_amplitude_over_delta_star']:.3f}"
                r"$\delta^*$ vs 2$\delta^*$")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
fig.suptitle("m16127 board audit of m16065 (CASE-DEM-MINDLIN-TANGENTIAL): independent recomputation", fontsize=10)
fig.tight_layout()
png = ART / "fig1_virgin_and_reversal.png"
ART.mkdir(parents=True, exist_ok=True)
fig.savefig(png, dpi=140)
print("figure ->", png, png.stat().st_size)

md = f"""# 交差核验 m16065（CASE-DEM-MINDLIN-TANGENTIAL，TensorDEM）— 独立重算与判级

## ① 案例介绍（物理背景 / 几何 / 参数表）

两个相同弹性球（R = {R['parameters']['radius_m']:.3e} m，E = {R['parameters']['energy_model_Pa'] if False else R['parameters']['young_modulus_Pa']:.3e} Pa，
ν = {R['parameters']['poisson_ratio']}，ρ = {R['parameters']['density_kg_m3']:.0f} kg/m³）在恒定法向力 P = {P:.1f} N 下压紧，
再施加单调切向位移 δ；Mindlin & Deresiewicz 1953（J. Appl. Mech. 20:327–344, DOI 10.1115/1.4010702，部分滑移构型 p.331）
的部分滑移解给出 F_t(δ) = μP[1−(1−δ/δ*)^{{3/2}}]（δ≤δ*），δ≥δ* 后饱和于 μP，δ* = 3μP/(16G*a)。

派生量（本审计**独立**由 Hertz 闭式重算，与交付件 rel_diff 全部为 0）：
R* = {RS:.3e} m，E* = {E:.6e} Pa，G* = {G:.6e} Pa，a = {A:.6e} m，
p0 = {CHK['hertz']['p0']:.6e} Pa，k_t0 = 8G*a = {CHK['hertz']['k_t0']:.6e} N/m，δ* = {DS:.6e} m，μ = {MU}。

## ② 怎么做的（方法 / 流程 / 工具链）

审计不重做算例，只做三件事（脚本 `scripts/audit/m16065_independent_check.py`，**不 import TensorDEM**）：

1. **取证**：`git show 14470b0e --stat` 确认 results.json / curves.npz / 5 图 / REPORT.md 均入库；
   平台 `tensor_reports id=15188` 六段齐全；平台存图 `backend/uploads/tensor_reports/15188/*.png`
   的 sha256 与仓内图逐张相同。
2. **独立重算**：Hertz 闭式重建 E*/G*/a/p0/k_t0/δ*；独立 scipy 自适应求积重建
   `F = 4G*∫₀ᵃ min(δ, μp(x)/(4G*))dx`（3/2 指数与 δ* 的 1/2 均为输出）；独立反演支
   `F(Δ) = 2[∫₀^{{x_r}}(μp−4G*Δ)dx − ∫_{{x_r}}ᵃ μp dx]`，x_r² = a²−2G*ΔR*/(μE*)，全反向解析 = 2δ*。
3. **复现**：干净 /tmp 副本（去 .git）跑 `run.py` + `validate.py` + pytest，逐叶比对 results.json。

工具链：python3 / numpy / scipy.quad / matplotlib；判据来源：`tensor_benchmark_cases` 该 anchor **0 行**（审计时），
故采用使命 goal 文本 <5 % 与 `validate.py:21 TOL=0.05`（预登记、只读 results.json）两条一致口径。

## ③ 计算结果（数字表，程序化注入自 artifacts/audit_m16065/independent_check.json）

| 量 | 参考 | 交付 got | 审计独立重算 | 判 |
|---|---|---|---|---|
| E* / R* / a / p0 / k_t0 / δ* | Hertz 闭式 | 3.846e10 / 2.5e-3 / 7.870e-5 / 7.708e8 / 4.986e6 / 9.026e-7 | 同（rel_diff 0.0） | ✅ |
| 闭式 vs 独立求积（参考自洽） | 闭式 | R1-vs-R2 9.0e-12 | {CHK['r1_vs_independent_quadrature']['max_abs_dev_over_muP']:.2e}·μP（受 1e-6 求积容差限制） | ✅ |
| 微滑移段最大偏差 G1 | Mindlin 闭式 | {CHK['g1_error_metrics']['claimed_in_results_json']:.4e}·μP | {CHK['g1_error_metrics']['micro_max_dev_over_muP']:.4e}·μP（真相对误差 {CHK['g1_error_metrics']['micro_max_true_relative_error']:.3e}） | ✅ <5 % |
| 宏观滑移平台 F_t(1.6δ*) | μP = 3.000000 N | {CHK['g1_error_metrics']['plateau_got_N']:.6f} N | rel 0.0 | ✅ |
| 过渡点 δ(0.99μP) | s99·δ* | 0.2533 % | 同 | ✅ |
| 动力学 G2 微滑移 | 闭式 | {R['primary']['dynamic']['micro_max_rel_err']:.3e} | Δt 一阶收敛 2.63e-4→6.90e-5 | ✅ |
| 线性 Cundall–Strack（ALT） | — | {R['alternatives']['linear_backbone']['micro_max_rel_err']:.4f} | >5 % → 算例有判别力 | ✅（预期出局） |
| 反演支 G3（能力边界 F1） | 独立反演支（2δ* 全反向） | {CHK['reversal']['claimed_reverse_max_rel_err']:.4f}·μP | {CHK['reversal']['max_dev_over_muP']:.4f}·μP；全反向 {CHK['reversal']['delivered_law_full_reversal_amplitude_over_delta_star']:.4f}δ* vs 2δ* | ❌ 未复现（如实申报为边界） |

复现证据：`run.py` exit 0 / 56.3 s；`validate.py` exit 0，`VERDICT: PASS (8/8 checks)`；
pytest 7 passed / 8.49 s；干净副本 results.json 与交付件 **0 个叶子差**（跳过 wall_clock 类键）。

## ④ 结果图

图 1（左）virgin loading：交付 G1（实线）vs 本审计独立闭式（虚线）vs 线性 Cundall–Strack（点线），
微滑移最大偏差 {CHK['g1_error_metrics']['micro_max_dev_over_muP']:.2e}·μP；
图 1（右）完全滑移后反演支：交付律（实线、包络形式）vs 独立反演支（虚线，2δ* 全反向），
偏差 {CHK['reversal']['max_dev_over_muP']:.3f}·μP，交付律在 {CHK['reversal']['delivered_law_full_reversal_amplitude_over_delta_star']:.3f}δ* 即完成反向
（解析 2δ*）——即 F1 能力边界。场量云图由被审方 fig4（接触盘内 q(r)/[μp(r)]）交付并经 sha256 核验，
本审计不重复渲染。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **口径（F2）**：交付件 `*_rel_err` 字段实为 |ΔF|/(μP) 的归一化偏差（run.py:324,:356），不是相对误差；
   换算后微滑移真相对误差 {CHK['g1_error_metrics']['micro_max_true_relative_error']:.3e}（结论不变），
   而反演支"117.13 %"若读作相对误差会**低估**（真相对误差量级 ~2.7e2）。另 commit message "0.55δ*" 与 results.json
   的 {CHK['reversal']['delivered_law_full_reversal_amplitude_over_delta_star']:.4f}δ* 不一致，取后者。
2. **参考非抄表**：1953 原文本环境 403 不可达，参考为闭式重构 + 独立降维交叉验证；使命要求的"引用原表原页"未满足（交付方已声明）。
3. **G1 是代数一致性检查**：库内包络与 R1 同式；非循环定量门是 G2 动力学与过渡点反解，故 L4 的含义是"与规范闭式一致 + 集成正确 + 可复现 + 有判别力"，非两套物理模型对照。
4. **裁判登记缺口（F3）**：`tensor_benchmark_cases` 该 anchor 0 行，机器裁判无处可依；本审计不代填（避免自证）。
5. **平台侧**：图端点 404（字节在 backend/uploads/...，服务 cwd 解析到另一路径，F5）；遗留占位报告 id=15187 仍 published（F4，交付方已披露）。
6. 适用范围：恒定法向力、virgin loading、无粘附/塑性/滚动；变法向力与反演支属未实现分支。

## ⑥ 误差判定

关键量逐项：微滑移段最大偏差 {CHK['g1_error_metrics']['micro_max_dev_over_muP']:.4e}·μP（真相对误差 {CHK['g1_error_metrics']['micro_max_true_relative_error']:.3e}）；
宏观平台 0.0 %；过渡点 0.2533 %；参考自洽 R1-vs-R2 9.0e-12（审计独立求积 {CHK['r1_vs_independent_quadrature']['max_abs_dev_over_muP']:.2e}·μP）。
**是否 <3 %：是**（最大 0.2533 %，远小于 3 % 与算例 5 % 门）。
反演支 **不满足**（偏差 {CHK['reversal']['max_dev_over_muP']:.3f}·μP、全反向幅值 −69.67 %），如实登记为能力边界 F1，
不计入算例门、也不粉饰为通过。

**判级：L4**（完整算例 + 逐项误差 + 收敛/敏感性阶梯 + 判别力 + 逐比特可复现 + 接入 protocols.sandpile 默认路径）；
不给 L5（无新物理）。Disposition：approve with notes / 消账，挂 F1/F2/F3 返修子项。
"""

payload = {
    "title": "交差核验 m16065 CASE-DEM-MINDLIN-TANGENTIAL：三件套取证 + 独立重算 + L4 判级",
    "summary": (f"独立重算（不 import TensorDEM）复现 m16065 关键数字：微滑移 {CHK['g1_error_metrics']['micro_max_dev_over_muP']:.4e}·μP、"
                f"平台 0.0 %、过渡点 0.2533 %（门 5 %）；反演支 {CHK['reversal']['max_dev_over_muP']:.3f}·μP、"
                f"全反向 {CHK['reversal']['delivered_law_full_reversal_amplitude_over_delta_star']:.3f}δ* vs 2δ* 确认为能力边界 F1；"
                "干净副本重跑 results.json 逐比特一致、validate 8/8、pytest 7 passed；"
                "判 L4（消账），挂 F1/F2/F3 返修子项与 F4/F5 平台侧观察（tensor_benchmark_cases 该 anchor 0 行）。"),
    "scenario_key": "case-dem-mindlin-tangential-audit",
    "mission_id": 16127,
    "stage": "S1",
    "tags": ["TensorDEM", "Mindlin-Deresiewicz", "board-audit", "case-pool", "maturity-verdict",
             "L4", "independent-recompute"],
    "markdown": md,
    "figures": [{"name": "fig1_virgin_and_reversal",
                 "caption": ("左：virgin loading 交付 G1 vs 审计独立闭式 vs 线性 Cundall–Strack；"
                             "右：完全滑移后反演支，交付包络律 vs 独立反演支（2δ* 全反向），F1 能力边界"),
                 "png_base64": base64.b64encode(png.read_bytes()).decode()}],
}

base = os.environ["TENSOR_API_BASE"]
s_req = urllib.request.Request(
    base + "/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"})
tok = json.load(urllib.request.urlopen(s_req, timeout=30))["access_token"]
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok})
try:
    resp = json.load(urllib.request.urlopen(req, timeout=120))
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code, exc.read().decode()[:1500])
    raise
print("report posted:", json.dumps(resp)[:300])
(ART / "report_payload_meta.json").write_text(json.dumps(
    {k: v for k, v in payload.items() if k != "figures"} |
    {"figures": [{"name": f["name"], "caption": f["caption"], "bytes": len(f["png_base64"])}
                 for f in payload["figures"]],
     "posted": resp}, ensure_ascii=False, indent=1))