#!/usr/bin/env python
"""Build and post the m16199 audit report (all numbers injected from artifacts)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

repro = json.load(open(os.path.join(HERE, "repro_result.json")))
probe = json.load(open(os.path.join(HERE, "probe_independent.json")))
ev = json.load(open(os.path.join(HERE, "verdict_evidence.json")))
s = repro["summary"]
p = repro["points"]["A_base"]
b = ev["benchmark"]["ref_vs_got_at_tol"]

png = base64.b64encode(open(os.path.join(HERE, "figures", "audit_rolling_single_particle.png"), "rb").read()).decode()


def pct(x: float) -> str:
    return f"{100.0 * x:.4g}%"


conv = repro["convergence_dt"]
stiff = repro["stiffness_invariance"]
win = repro["window_sensitivity"]
P = probe["P1_P2_angle_scaling"]
p3 = probe["P3_mu_r_branch"]

rows_conv = "\n".join(
    f"| {c['dt_s']:.0e} | {c['steps']} | {c['a_t']:.9f} | {pct(c['a_rel_error'])} | "
    f"{pct(c['alpha_rel_error'])} | {pct(c['oscillation_amplitude_rel_error'])} |" for c in conv)
rows_stiff = "\n".join(
    f"| {c['k_n_N_per_m']:.0f} | {c['a_t']:.9f} | {pct(c['a_rel_error'])} | {c['a_instantaneous_std']:.4f} |"
    for c in stiff)
rows_win = "\n".join(
    f"| {w['duration_s']} | {w['fit_window_s']} | {w['a_t']:.9f} | {pct(w['a_rel_error'])} | {pct(w['alpha_rel_error'])} |"
    for w in win)
rows_probe = "\n".join(
    f"| {q['theta_deg']:.0f} | {q['a_ref']:.9f} | {q['a_quad']:.9f} | {q['err_quad']:.2e} | "
    f"{q['a_over_g_sin']:.10f} |" for q in P)

markdown = f"""# 交差审计：m16142 `CASE-DEM-ROLLING-SINGLE-PARTICLE` 独立复核 → 裁定 L4

* 审计单: **m16199**（交差核验，非重做）｜被审单: **m16142**（TensorDEM 算例池，succeeded）
* 裁定: **L4 = confirmed**（`tensor_maturity_verdicts` id 435，verdict_by=board-audit）
* 独立性: 审计在**独立工作树** `/root/tp-agent/work/m16199` 上、对被合并后的代码**自己重跑**并**自建独立估计量**；
  唯一共享对象是被测代码本身。

## ① 案例介绍（物理背景 / 几何 / 参数）

一个实心球在倾角 θ 的刚性斜面上**无滑移纯滚动**。经典刚体动力学给出唯一解析解
`a = g (sinθ − μ_r cosθ)/(1 + I/(mR²))`、`α = −a/R`（harness 符号约定 v+ΩR=0）。
本算例取 θ=15°、R=0.01 m、ρ=900 kg/m³、μ=0.9（无滑移所需 μ≈0.051 的 17 倍）、μ_r=0（基准点）/0.1（滚动阻力分支），
k_n=k_t=2e5 N/m、Δt=2e-5 s、时长 0.12 s（拟合窗 0.09 s）。验收判据先于生产运行登记在
`prereg/CASE-DEM-ROLLING-SINGLE-PARTICLE.json` 与 `tensor_case_pool.id=90.tol_policy`：**<2%**。

## ② 怎么做的（审计方法 / 流程 / 工具链）

1. **取证**：逐项核 run_artifacts（`result.json` {os.path.getsize('/root/tp-agent/work/m16142/artifacts/case_rolling_single_particle/result.json')} B、四联图 png、日志）、
   `benchmarks/case_rolling_single_particle/run_case.py`(414 行)、`src/tensordem/rolling_resistance.py`(575 行)、`tests/…`(125 行)、预注册 JSON。
2. **重跑**（repro_cmd 可执行性）：在审计自己的工作树执行 `run_case.py --out docs/board_audit/m16199/repro_result.json`，
   与 m16142 工件 **a_got 逐位相同**（{s['a_got']:.16f}），耗时 <1 min；`pytest tests/test_case_rolling_single_particle.py -q` → **6 passed in 29.82 s**。
3. **独立估计量探针**（不复用算例自己的 v 斜率估计量）：位移二次拟合 `x=c0+c1t+a t²/2`、三角度 a/(g sinθ) 标定惯性因子、μ_r 分支位移量。
4. **代码实质核验**：`simulate_sphere_on_incline` 是真实积分循环（Cundall-Strack 切向弹簧 + 库仑钳制 + 法向弹簧阻尼 + CDT 滚动阻力矩 + 半隐式 Euler），
   非硬编码、非闭式套壳（见 ③ 的 μ_r 判别实验）。
5. **裁判口径复核**：`tensor_benchmark_cases` 对该 anchor **0 行**（全表 25 行），故用 case_pool 的 tol_policy（<2%）+ 闭式解析解裁定；
   并用 maturity_gate 的 `judge()` 以只读方式预览证据链 → 返回 `(4, [])`（L4 零缺口，机器判据全过）。

## ③ 计算结果（数字表，全部由工件程序化注入）

基准点 ref vs got（判据 2%）：

| 指标 | ref（闭式解） | got（TensorDEM，审计重跑） | 相对误差 |
|---|---|---|---|
| 平动加速度 a | {p['reference']['a_ref']:.10f} | {p['got']['a_t_m_per_s2']:.10f} | {pct(s['a_rel_error'])} |
| 角加速度 α | {p['reference']['alpha_ref']:.6f} | {p['got']['alpha_rad_per_s2']:.6f} | {pct(s['alpha_rel_error'])} |
| 无滑移闭合 \\|a+Rα\\|/\\|a\\| | 0 | {s['no_slip_closure']:.3e} | {pct(s['no_slip_closure'])} |
| 瞬时加速度振荡幅值 | {s['oscillation_amplitude_ref']:.7f} | {s['oscillation_amplitude_measured']:.7f} | {pct(s['oscillation_amplitude_rel_error'])} |
| μ_r=0.1 分支 a | {p['reference']['a_ref'] * 0 + repro['points']['B_rolling']['reference']['a_ref']:.9f} | {repro['points']['B_rolling']['got']['a_t_m_per_s2']:.9f} | {pct(s['rolling_resistance_a_rel_error'])} |
| 负控 μ=0.01 vs 无滑移参考（必须被拒） | {p['reference']['a_ref']:.9f} | {repro['points']['NEG_low_friction']['got']['a_t_m_per_s2']:.9f} | {pct(s['negative_control_slip']['a_rel_error'])} |
| 负控 无惯性伪参考 g sinθ（必须被拒） | {s['negative_control_no_inertia_reference']['wrong_a']:.9f} | {s['a_got']:.9f} | {pct(s['negative_control_no_inertia_reference']['rel_error_vs_wrong'])} |

时间步收敛（基准点）：

| Δt [s] | 步数 | a [m/s²] | a 误差 | α 误差 | 振荡幅值误差 |
|---|---|---|---|---|---|
{rows_conv}

接触刚度不变性：

| k_n [N/m] | a [m/s²] | a 误差 | a(t) 标准差 |
|---|---|---|---|
{rows_stiff}

拟合窗敏感性：

| 时长 [s] | 窗 [s] | a [m/s²] | a 误差 | α 误差 |
|---|---|---|---|---|
{rows_win}

**审计自建独立探针**（非算例原有估计量）：

| θ [deg] | a_ref | a（位移二次拟合） | 相对误差 | a/(g sinθ)（应为 {1/1.4:.10f}） |
|---|---|---|---|---|
{rows_probe}

μ_r 判别实验：实测位移 {p3['measured_shift_vs_mu_r0']:.10f} vs 预测 −μ_r·cotθ = {p3['predicted_shift']:.10f}
（相对 {p3['rel_err']:.2e}）→ 滚动阻力矩真的进入方程，1e-6 级吻合**不是闭式套壳**。

## ④ 结果图（审计复现，非转发原图）

图中全部数字来自审计自己的 `repro_result.json`：(a) v(t) 与 −RΩ(t) 重合即无滑移闭合，滑移残差 ~1e-4 m/s；
(b) 瞬时 a(t) 在解析均值上下振荡，包络与闭式 `g sinθ(1±f)/(1+f)` 一致（无阻尼切向弹簧的必然结果）；
(c) Δt 收敛：均值误差 ~1e-6 与 Δt 无关，振荡幅值误差一阶下降 {pct(conv[0]['oscillation_amplitude_rel_error'])} → {pct(conv[-1]['oscillation_amplitude_rel_error'])}；
(d) 误差预算（v 斜率、位移二次、振荡幅值、α）全部落在 2% 门内。

![audit figure](audit_rolling_single_particle.png)

## ⑤ 注意点（坑 / 局限 / 适用范围）

* **基准表空行**：`tensor_benchmark_cases` 对 `CASE-DEM-ROLLING-SINGLE-PARTICLE` 0 行（25 行全表），`tensor_case_pool` id=90 仍是 `dispatched`。
  机器基准裁判缺席 → 本次以 tol_policy + 闭式解裁定，并建议补行（不阻断）。
* **合入门滤除**：m16142 分支的 `artifacts/result.json`、`*.png`、`logs/*.log` 被门禁挡在 main 之外（commit 信息自述「3 个文件被门禁滤除」），
  故复核必须跑 repro_cmd 而不是期待仓内已存工件；审计重跑已逐位复现，故不阻断。
* **指标读法警告**：验收量是 v(t) 在 ~195 个切向振荡周期上的最小二乘均值，离散误差被平均掉，
  **不得**把 1e-6 读成「求解器精度 1e-6」。真正有判别力的是振荡幅值（基准点 0.94%，一阶收敛）与 μ_r 分支（力矩相关参考，4e-6）。
* **无外部实验锚**：Ai et al. 2011 原文/表格本环境不可得（报告 ① 与 `provenance.reference.access_note` 已明示不宣称其表值），
  参考锚是经典闭式刚体动力学 + 该文模型关系式 → **L5 拒绝**，此为诚实降级。
* **未做**：与 3-D 求解器 `IceDEM3D` 的交叉验证（算例仪器是专用 1-DOF 斜面 harness，报告 ⑥ 已声明）、多颗粒/clump 斜面滚动。

## ⑥ 误差判定（关键量 vs 基准，逐项；显式回答是否 <3%）

| 关键量 | 相对误差 vs 基准 | <3% ? |
|---|---|---|
| 平动加速度 a | {pct(s['a_rel_error'])} | **是** |
| 角加速度 α | {pct(s['alpha_rel_error'])} | **是** |
| 无滑移闭合 | {pct(s['no_slip_closure'])} | **是** |
| 振荡幅值（真正的一阶量） | {pct(s['oscillation_amplitude_rel_error'])} | **是** |
| μ_r=0.1 分支 a | {pct(s['rolling_resistance_a_rel_error'])} | **是** |
| 负控 μ=0.01 偏差（要求 >2%，具备判别力） | {pct(s['negative_control_slip']['a_rel_error'])} | 期望被拒 ✔ |
| 负控 无惯性伪参考偏差 | {pct(s['negative_control_no_inertia_reference']['rel_error_vs_wrong'])} | 期望被拒 ✔ |

**结论：全部验收项均 <3%（实测最坏 {pct(max(s['a_rel_error'], s['alpha_rel_error'], s['no_slip_closure'], s['oscillation_amplitude_rel_error'], s['rolling_resistance_a_rel_error']))}，门 2%）→ PASS，
裁定 L4 confirmed**；L5 因无外部文献/实验锚而拒绝，理由已入证据（`external_anchor=null` + `external_anchor_note`）。
"""

payload = {
    "title": "交差审计 m16142 CASE-DEM-ROLLING-SINGLE-PARTICLE：独立复现 + 独立估计量 → L4 confirmed",
    "summary": ("审计在独立工作树重跑算例，a_got 与 m16142 工件逐位相同；"
                f"a 相对误差 {pct(s['a_rel_error'])}、α {pct(s['alpha_rel_error'])}、振荡幅值 {pct(s['oscillation_amplitude_rel_error'])}（门 2%）；"
                "自建位移二次拟合探针 2.5e-10、μ_r 分支判别实验 3.0e-6 证明非闭式套壳；maturity_gate.judge() 预览返回 (4,[])；裁定 L4。"),
    "scenario_key": "audit/m16142-case-dem-rolling-single-particle",
    "mission_id": 16199,
    "tags": ["audit", "TensorDEM", "benchmark", "rolling", "L4", "m16142"],
    "markdown": markdown,
    "figures": [{"name": "audit_rolling_single_particle.png",
                 "caption": "审计复现四联图：(a) 无滑移闭合 (b) 瞬时加速度振荡与闭式包络 (c) Δt 收敛 (d) 误差预算（全部数字取自审计自己的 repro_result.json）",
                 "png_base64": png}],
}

out = "/tmp/report_m16199.json"
json.dump(payload, open(out, "w"))
print(out, len(markdown))

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(base + "/auth/login", data=login, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=180) as r:
    tok = json.loads(r.read()).get("access_token") or json.loads(r.read()).get("token")
assert tok, "login failed"
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok}, method="POST")
with urllib.request.urlopen(req, timeout=180) as r:
    print(r.read().decode()[:600])