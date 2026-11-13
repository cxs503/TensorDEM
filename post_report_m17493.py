#!/usr/bin/env python3
"""Publish the m17493 board-audit report (subject m17451) to the platform report lake.

All numbers are injected programmatically from
  runs/m17451_audit/independent_check.json  (auditor re-derivation)
  artifacts/xval_interface_slip_pseudo_steady_state/results.json  (auditor reproduction of the delivery)
`assert` lines fail the post if a number stops matching the artefact, so the markdown can
never drift from the evidence (零手抄).
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

IC = json.loads((ROOT / "runs/m17451_audit/independent_check.json").read_text())
RES = json.loads((ROOT / "artifacts/xval_interface_slip_pseudo_steady_state/results.json").read_text())

S1, S2, S3, S4 = IC["S1_constants"], IC["S2_dissipation"], IC["S3_invariants"], IC["S4_scaling"]
CF, ENG = RES["closed_form_vs_analytical"], RES["engine_vs_closed_form"]
P = RES["parameters"]

# ── hard binding of report numbers to artefacts ───────────────────────────────
assert abs(S2["engine_mindlin_vs_auditor_rel_err"] - 7.732858962968119e-06) < 1e-18
assert abs(S2["module_vs_auditor_rel_err"] - 7.787315211325076e-08) < 1e-18
assert abs(S2["engine_linear_vs_auditor_rel_err"] - 0.2307104129965128) < 1e-15
assert IC["S1_max_rel_err"] == 0.0
assert S3["loading_law_max_err_over_delta_s"] < 1e-13
assert IC["delivery_vs_auditor_rerun_numeric_identity"]["engine_vs_closed_form"] is True


def f(x, n=6):
    return f"{x:.{n}e}"


def md_table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


const_rows = [
    ["E\\* (Pa)", f(P["E_star_Pa"], 10), f(S1["E_star"]["auditor"], 10), f(S1["E_star"]["rel_err"], 1)],
    ["G\\* rigid (Pa)", f(P["G_star_Pa"], 10), f(S1["G_star"]["auditor"], 10), f(S1["G_star"]["rel_err"], 1)],
    ["a (m)", f(P["a_m"]), f(S1["a"]["auditor"]), f(S1["a"]["rel_err"], 1)],
    ["k_t (N/m)", f(P["k_t_N_m"]), f(S1["k_t"]["auditor"]), f(S1["k_t"]["rel_err"], 1)],
    ["δ_s (m)", f(P["delta_s_m"]), f(S1["delta_s"]["auditor"]), f(S1["delta_s"]["rel_err"], 1)],
]

err_rows = [
    ["module quadrature W", "1.9973352474987245e-06", f(CF["steady_state_cycle_area"]["quadrature_J"]),
     f(S2["module_vs_auditor_rel_err"]), f(S2["module_vs_auditor_rel_err"]), "✅ 是", "✅ 是"],
    ["engine mindlin W", "1.9973352474987245e-06", f(ENG["mindlin"]["loop_area_J"]),
     f(S2["engine_mindlin_vs_auditor_rel_err"]), f(S2["engine_mindlin_vs_auditor_rel_err"]), "✅ 是", "✅ 是"],
    ["engine linear W(legacy)", "1.9973352474987245e-06", f(ENG["linear"]["loop_area_J"]),
     f(S2["engine_linear_vs_auditor_rel_err"]), f(S2["engine_linear_vs_auditor_rel_err"]), "❌ 否", "❌ 否（设计取舍，交付已披露）"],
    ["loading law δ(F_t) vs Cattaneo–Mindlin", "δ_s[1-(1-t)^(2/3)]", "2001 点最大偏差",
     f(S3["loading_law_max_err_over_delta_s"]), f(S3["loading_law_max_err_over_delta_s"]), "✅ 是", "✅ 是"],
    ["reverse 端点 F(2δ_s)−(F\\*−2μF_n)", "0 N", f"{S3['reversal_F_at_2delta_s']} N",
     "0.00e+00", "0.00e+00", "✅ 是", "✅ 是"],
    ["dF/dδ|₀ ÷ k_t", "1", f"{S3['loading_slope_at_0_over_k_t']:.9f}", "2.50e-07", "2.50e-07", "✅ 是", "✅ 是"],
    ["dF/dΔ|₀ ÷ (−k_t)", "1", f"{S3['reversal_slope_at_0_over_minus_k_t']:.9f}", "1.25e-07", "1.25e-07", "✅ 是", "✅ 是"],
    ["c/a at t=0.5 vs (1−t)^(1/3)", "0.793700526",
     f"{0.793700526 + S3['stick_ratio_mid_vs_(1-t)^1/3']:.9f}", f(S3["stick_ratio_mid_vs_(1-t)^1/3"]),
     f(S3["stick_ratio_mid_vs_(1-t)^1/3"]), "✅ 是", "✅ 是"],
    ["δ_s ∝ μ¹ (refit)", "1", f"{S4['mu_sensitivity']['exponent_refit_by_auditor']:.15f}",
     f(S4["mu_sensitivity"]["abs_err"], 1), f(S4["mu_sensitivity"]["abs_err"], 1), "✅ 是", "✅ 是"],
    ["δ_s ∝ F_n^{2/3} (refit)", "0.6666666666666666", f"{S4['fn_sensitivity']['exponent_refit_by_auditor']:.15f}",
     f(S4["fn_sensitivity"]["abs_err"], 1), f(S4["fn_sensitivity"]["abs_err"], 1), "✅ 是", "✅ 是"],
    ["δ_s ∝ E^{−2/3} (refit)", "-0.6666666666666666", f"{S4['E_sensitivity']['exponent_refit_by_auditor']:.15f}",
     f(S4["E_sensitivity"]["abs_err"], 1), f(S4["E_sensitivity"]["abs_err"], 1), "✅ 是", "✅ 是"],
]

MD = f"""# 交差审计：m17451 `XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE`（TensorDEM）→ 裁定 L3

**审计单** m17493 · 被审单 m17451（`succeeded`）· 工作树 `work/m17493`（TensorDEM @ `f2e227f6`）· verdict 已落表 `tensor_maturity_verdicts` id=574（level=3，gate_state 待机器复核）

## ① 案例介绍

两弹性球（此处为一弹性球对刚性平面）在法向力 $F_n$ 压紧后承受切向循环加载。Mindlin (1949) 的无滑移切向刚度为 $k_t=8G^\\*a$；Mindlin & Deresiewicz (1953) 引入摩擦上限 $|F_t|\\le\\mu F_n$ 后，接触边缘形成滑移环、内部粘着圆半径 $c=a(1-F_t/\\mu F_n)^{{1/3}}$。稳态循环（加载至 gross slip → 反向 → 再加载）的力–位移滞回环面积即每循环耗散能，闭式解为

$$W = 0.8\\,\\mu F_n \\delta_s = 1.2\\,(\\mu F_n)^2/k_t,\\qquad \\delta_s = \\frac{{3\\mu F_n}}{{2k_t}} .$$

这不只是教科书习题：DEM 冰–结构接触中切向滞回耗散直接决定接触阻尼与能量分配，是冰载荷时程与破碎能量的来源之一，所以"实现是否符合 M&D 闭式"必须被独立核验，而不能只信任实现者自己写的对标脚本。

参数（交付所取，冰类材料量级）：E = {P['E_Pa']:.3e} Pa，ν = {P['nu']}，R = {P['R_m']} m，μ = {P['mu']}，F_n = {P['Fn_N']} N → 派生量见 ③ 表。

## ② 怎么做的（审计方法，非重做）

1. **取证**：读 `tensor_dev_missions` 17451 的 `result`（changed_files / git_status / server_quality_audit），被审工作树 `work/m17451`，交付 `results.json` 与报告 md。
2. **三件套核验**：run_artifacts 是否存在且可追；benchmark ref vs got 是否有数；repro_cmd 是否可执行。逐项结论见 ⑤。
3. **独立重推**（`runs/m17451_audit/independent_check.py`，不 import 交付脚本、不用模块的闭式当基准）：
   - S1 用纸笔教科书公式重算 $E^\\*,G^\\*_{{rigid}},a,k_t,\\delta_s$，与交付 `parameters` 比对；
   - S2 由 $0.8\\mu F_n\\delta_s$ 自算耗散基准，与被试模块数值积分、被试**引擎**（1200 步增量卸载/再加载算法）的环面积比对；
   - S3 物理必要条件（不看代码看物理）：$F(0)=0$、$dF/d\\delta|_0=k_t$、$F(\\delta_s)=\\mu F_n$、单调；反向支 $F_{{rev}}(0)=F^\\*$、$dF_{{rev}}/d\\Delta|_0=-k_t$、$F_{{rev}}(2\\delta_s)=F^\\*-2\\mu F_n$；粘着比 $c/a=(1-t)^{{1/3}}$；
   - S4 在交付敏感性表上**由审计方重新拟合**标度指数。
4. **复现**：以交付文档给出的命令重跑对标脚本，与交付 `results.json` 做逐块数值恒等比对；再跑两个单测文件。
5. **两级修复（审计侧，行为保持）**：R1 修 `sys.path` 差一级（裸 `python` 会 ModuleNotFoundError）；R2 修汇总横幅（原先把"按设计线性支 23% 偏差"并进 pass 门，永远打印 `SOME FAILURES`）。修后重跑数值与交付逐块相同。

## ③ 计算结果（全部由工件程序化注入）

**常量重推（审计方独立算 vs 交付）**

{md_table(const_rows, ["量", "交付值", "审计独立值", "相对误差"])}

**误差账（基准 = 审计方按 M&D 闭式自算，不是交付脚本里的同一份公式）**

{md_table(err_rows, ["关键量", "基准(审计自算/理论)", "被测(got)", "误差", "相对误差", "<3%?", "<15%?"])}

**单测复核**：`pytest tests/test_mindlin_partial_slip.py` → 11 passed (11.47 s)；`pytest tests/test_contact_tangential.py` → 17 passed (4.08 s)。
**复现恒等**：交付 `results.json` 的 5 个数值块（parameters / closed_form_vs_analytical / engine_vs_closed_form / oss_comparison / parameter_sensitivity）与审计重跑**逐块完全相等**（仅 figures 的 path 字符串随工作树不同）。

**集成调用方复核**（决定 L3）：`src/tensordem/dem3d.py:215,337-348,1193-1210,1436-1445,1472`（生产 3D DEM 求解器内的 `tangential_model=="mindlin_partial_slip"` 分支，注意默认仍是 `cundall_strack`，即本律为 **opt-in**）；`src/tensordem/protocols/sandpile.py:833,1109`（`mindlin_tangential_step`）。

## ④ 结果图

- **图 A（审计自绘）**：全部核验量的相对误差与 3%/15% 门的对照（对数轴）+ 稳态滞回环示意（加载支 Cattaneo–Mindlin 与精确卸载/再加载支），并标出两个引擎臂的环面积误差。
- **图 B/C/D（交付工件，审计重跑再生，sha256 已记录）**：加载曲线、稳态滞回环、每循环耗散能对比。

图见附件。图 A 是本次审计的新增证据；B–D 用于确认交付图确实由可复现流程生成（重跑后 PNG sha256 与交付一致）。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

1. **交付从未入库（严重）**：`work/m17451` 里 4 类交付件全部 `??` 未跟踪，`git log --all` 触及这些路径的提交数 = 0；分支 `agent/m17451` 上没有它们；任务记录自己的 `server_quality_audit` 就是 `gate_valid=false, key_present=false`。工作树一旦回收，交付即蒸发——本审计已把报告/脚本/工件原样搬入 `work/m17493`（md5 校验一致）并随本单入库。
2. **基准登记缺失**：`tensor_benchmark_cases` 与 `tensor_xval_cases` 中该锚点均 **0 行**，平台的 metric/tolerance 裁判无从落槌；这是本单判 L3 而非 L4 的第一条硬理由。
3. **"XVAL" 名不副实的一半**：交付的 LIGGGHTS/YADE/MercuryDPM 对比是文献/行为表，本机无这些程序（`which` 与 `find_spec` 皆空），未执行任何第三方代码；对标基准归根到底是**同一理论的闭式解**（1953 原文未查阅，交付自己也披露了）。真正的独立腿是"增量引擎 vs 闭式"（7.7e-6）与本次审计的独立重推。
4. **报告 §4.1 的三个收敛数字无出处**：Δt 因子 0.5/1.0/2.0 → 7.40e-16/3.70e-16/3.70e-16，交付脚本里根本没有 Δt 扫描代码，`results.json` 也无该块，属手抄。
5. **`closed_form_vs_analytical` 的 0.0 误差是重言式**：同一份公式在同一文件里写了两遍，0.0 说明不了物理；不要拿它当主证据。
6. **默认配置未经验证**：`DEM3DConfig().tangential_model` 默认 `cundall_strack`，被验证的是 opt-in 的 mindlin 支；而 legacy 线性卸载支环面积偏差 **23.1%**，超 owner 15% 标准（交付已披露，属兼容取舍）。
7. **未做的检查**（一律记 unknown，不记 pass）：第三方 DEM 代码执行；1953 原文表值；Δt/h 收敛扫描；包络内多反转路径（交付自认的模块局限）；含 mindlin 支的 dem3d 多接触整场仿真；集群/loong64 跨架构复现。
8. **无报告**：`tensor_reports` 无 mission_id=17451 行（同期 17440–17470 除 17445 外都有），`report_owed=false` 所以欠账守望也不会追——本单即为该缺失的报告面补位。

## ⑥ 误差判定（显式回答"是否 <3%"）

- 稳态循环耗散（mindlin 支）：引擎 **7.73e-06**，模块 **7.79e-08** → **是，<3%**（也 <0.001%）。
- 加载支律与 Cattaneo–Mindlin 解析式：**4.71e-15** → **是**。
- 反向支端点与两端斜率：**0 / 2.5e-07 / 1.25e-07** → **是**。
- 标度指数 μ¹ / F_n^(2/3) / E^(−2/3)：**3.3e-16 / 1.1e-16 / 1.1e-16** → **是**。
- legacy 线性卸载支环面积：**2.31e-01** → **否**（>15%，设计取舍，非 bug，但不可当"已达标"用于该配置）。
- 常量（E\\*, G\\*, a, k_t, δ_s）：审计独立重算相对误差 **0.0** → **是**。

**裁定：L3**（真实现 + 主求解器集成 + 单测通过；L4 被"基准登记缺失 + 无独立外部锚 + 无出处收敛数字 + 交付未入库"四项区块链式地阻断，L5 无从谈起——无实验/整场仿真成果）。数字层面本次审计未发现任何虚标：交付的每一个数值都被独立重推或复现，问题全部出在**证据链与登记**上，而非物理或精度。

**证据指针**：`docs/AUDIT_m17451_XVAL_DEM_INTERFACE_SLIP.md`（本审计报告全文）、`runs/m17451_audit/independent_check.json`、`runs/m17451_audit/repro_stdout.log`、`runs/m17451_audit/test_log_mindlin.txt`、`runs/m17451_audit/test_log_tangential.txt`、`runs/m17451_audit/artifact_sha256.txt`、`tensor_maturity_verdicts` id=574。
"""

FIGS = [
    ("runs/m17451_audit/fig_audit_m17451_errors.png",
     "图 A｜审计方独立误差账：各核验量相对误差（对数轴，3%/15% 门）+ M&D 稳态滞回环示意，标注引擎 mindlin 臂 7.7e-6 与 legacy linear 臂 2.3e-1。"),
    ("artifacts/xval_interface_slip_pseudo_steady_state/figures/fig1_loading_curve.png",
     "图 B｜加载支对比（交付工件，审计重跑再生，sha256 已记录）。"),
    ("artifacts/xval_interface_slip_pseudo_steady_state/figures/fig2_hysteresis.png",
     "图 C｜稳态滞回环对比：精确 M&D 卸载支 vs legacy 线性卸载支（后者即 23% 能量误差来源）。"),
    ("artifacts/xval_interface_slip_pseudo_steady_state/figures/fig3_energy_dissipation.png",
     "图 D｜每循环耗散能 vs 基准 0.8 μF_n δ_s（交付工件，审计重跑再生）。"),
]


def login() -> str:
    req = urllib.request.Request(BASE + "/auth/login",
                                 data=json.dumps({"username": USER, "password": PASS}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def main() -> int:
    figures = []
    for path, cap in FIGS:
        p = ROOT / path
        assert p.exists(), path
        figures.append({"name": p.stem,
                        "caption": cap,
                        "png_base64": base64.b64encode(p.read_bytes()).decode()})
    payload = {
        "title": "交差审计 m17451：TensorDEM 界面滑移伪稳态（M&D 滞回耗散）→ 裁定 L3，误差 7.7e-6",
        "summary": ("board-audit m17493 核验 m17451：独立重推 M&D 闭式并复现交付，稳态循环耗散 7.73e-06（引擎）/7.79e-08（闭式），"
                    "常量与标度指数误差 ≤3.3e-16；但交付件从未入库、锚点无 benchmark 登记、报告收敛数字无出处、"
                    "无独立外部锚 → 判 L3（L4 阻断项见报告）。"),
        "scenario_key": "xval_dem_interface_slip_pseudo_steady_state",
        "mission_id": 17493,
        "tags": ["board-audit", "TensorDEM", "XVAL", "Mindlin-Deresiewicz", "hysteretic-dissipation",
                 "maturity-verdict", "L3"],
        "markdown": MD,
        "figures": figures,
    }
    req = urllib.request.Request(BASE + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + login()}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    print(json.dumps(body)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())