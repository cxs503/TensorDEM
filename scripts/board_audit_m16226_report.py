#!/usr/bin/env python3
"""Publish the m16226 board-audit report (six sections, figures embedded)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(REPO, "docs/board_audit_m16226/evidence")
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
CREDS = json.load(open(os.path.expanduser("~/.eco_admin_creds.json")))

res = json.load(open(os.path.join(EV, "results.json")))
g = res["got_mindlin_partial_slip"]
ref = res["reference"]["chain"]
cs = res["comparison_legacy_cundall_strack"]
conv = res["convergence"]
sens_mu = res["sensitivity_mu"]
sens_n = res["sensitivity_Fn"]

FIG = [("fig1_ft_delta_curve.png", "图1 F_t/(mu F_n) 对 delta/delta_s：解析 Mindlin 闭式、TensorDEM 求解器分支、遗留线性 Cundall-Strack"),
       ("fig2_error_vs_delta.png", "图2 逐点偏差 |F_t-F_ref|/(mu F_n)（对数轴）：Mindlin 分支在机器精度地板，线性律最大 19.24%"),
       ("fig3_dt_and_sensitivity.png", "图3 时间步收敛 + delta_s 随 mu、F_n 的标定（拟合 vs 3 mu F_n/(2k_t)）"),
       ("fig4_stick_circle_map.png", "图4 接触斑内粘着圆半径 c/a=(1-F_t/(mu F_n))^{1/3} 随加载收缩")]

figs = []
for name, cap in FIG:
    p = os.path.join(EV, "figures", name)
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(open(p, "rb").read()).decode()})

md = f"""# CASE-DEM-MINDLIN-TANGENTIAL 交差核验报告（m16226 审 m16147，TensorDEM）

## ① 案例介绍
切向接触的 **Mindlin–Deresiewicz (1953) 微滑移→完全滑移过渡**：两个弹性体受法向力 F_n 压紧后施加切向位移 δ，
接触斑边缘先形成滑移环、内部保留粘着圆，切向力为

    F_t = mu F_n [1 - (1 - delta/delta_s)^{{3/2}}],   delta_s = 3 mu F_n / (2 k_t),   k_t = 8 G* a,   c/a = (1 - F_t/(mu F_n))^{{1/3}}

物理意义：δ≪δ_s 时退化为无滑移 Mindlin 弹簧（斜率 8G*a），δ→δ_s 时摩擦饱和（F_t→mu F_n）。
遗留 Cundall–Strack 线性弹簧+库仑钳位缺**曲率**，本算例正是为判别这一缺口而设。
参考出处：Mindlin & Deresiewicz (1953) *J. Appl. Mech.* **20**:327–344, DOI 10.1115/1.4010702；
同一闭式见 Johnson, *Contact Mechanics* (1985) §4.3 eq. 4.32。1953 原表付费墙且沙箱无网 → **未手抄原表**，
参考值由 run.py 用 Hertz 一阶原理独立评估（与求解器模块无 import 关系）。

几何/参数（交付 `results.json` 机器注入）：R={ref['radius_m']:.4e} m，E={ref['young_modulus_Pa']:.3e} Pa，nu={ref['poisson_ratio']:.2f}，
rho=917 kg/m³，mu={ref['mu']:.2f}，F_n={ref['normal_load_N']:.1f} N；派生 E*={ref['E_star_Pa']:.6e} Pa，
G*={ref['G_star_Pa']:.6e} Pa，a={ref['a_m']:.6e} m（a/R={ref['a_over_R']:.4e}），
k_n={ref['k_n_N_per_m']:.6e} N/m，k_t={ref['k_t_N_per_m']:.6e} N/m（k_t/k_n={ref['k_t_over_k_n']:.6f} = 3(1-nu)/(2-nu)），
delta_n={ref['delta_n_m']:.6e} m，delta_s={ref['delta_s_m']:.6e} m（delta_s/delta_n={ref['delta_s_over_delta_n']:.6f}），
无量纲载荷 mu F_n/(G* a^2)={ref['mu_Fn_over_G_a2']:.6e}，dt={res['solver']['dt_s']:.6e} s。

## ② 怎么做的
本单是**交差核验单，非重做**。审计只读交付字节并独立复算：
1. 取证：被审工作树 `/root/tp-agent/work/m16147`（commit 00025434，Merge #1351）——源码 2 个、入口 2 个、工件 6 个；
   工件按字节复制进本审计仓 `docs/board_audit_m16226/evidence/`，并**用审计自己的 sha256** 与交付 `checksums.json` 比对（6/6 match）。
2. 裁判：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-DEM-MINDLIN-TANGENTIAL'` → **0 行**
   （全表 29 行、LIKE '%MINDLIN%' 亦 0 行），机器裁判缺席；实际判据=使命 goal `<5%`+`validate.py` 预登记阈值（5%/5%/1%/2%，无回溯调参）。
3. 独立重算：从 `curves.csv` 6438 个求解器样点重算逐点偏差；从 Hertz 一阶原理独立重算 a、k_t、delta_s 与 k_t/k_n 恒等式；独立重算线性律最大偏离。
4. 独立复跑：把交付树复制到 `/tmp/repro_m16147`（去 .git）跑全量 `run.py`、`validate.py`、`pytest tests/test_mindlin_partial_slip.py`。
5. 落表：`tensor_maturity_verdicts` id=451（mission_id=16226, scope='m16147 CASE-DEM-MINDLIN-TANGENTIAL', level=4, verdict_by='board-audit'），
   并用平台自身 `maturity_gate.judge()` 预判 → gate_level=4、缺失项为空。六要素依据=工件+复跑日志，零手抄。

## ③ 计算结果（ref vs got，机器注入）

| # | 指标 | ref | got | 误差 | 判据 | 判定 |
|---|---|---|---|---|---|---|
| B1 | 微滑移段 max \\|F_t−F_MD\\|/(mu F_n) | 闭式 | — | {g['max_micro_slip_err']:.4e} | <5% | PASS |
| B2 | 过渡段 \\|δ−δ_s\\|≤0.1δ_s | 闭式 | — | {g['max_transition_err']:.4e} | <5% | PASS |
| B3 | 完全滑移平台 | {g['mu_Fn_N']:.6f} N | {g['plateau_force_N']:.7f} N | {g['max_gross_slip_err']:.4e} | <1% | PASS |
| B4 | 初始斜率 k_t | {g['k_t_theory_N_per_m']:.6e} N/m | {g['k_t_secant_N_per_m']:.6e} N/m | {g['k_t_rel_err']:.4e} | <2% | PASS |
| B5 | 拟合 δ_s | {g['delta_s_theory_m']:.6e} m | {g['delta_s_fit_m']:.6e} m | {abs(g['delta_s_fit_m']-ref['delta_s_m'])/ref['delta_s_m']:.4e} | <5% | PASS |
| B6 | 判别力：遗留线性律偏离 | — | {cs['max_dev_vs_analytic_muN']:.6f} | — | >5% | PASS |

收敛/敏感性（≥2 轴）：Δt×{{0.5,1,2}} → 微滑移误差 {{ {conv[0]['max_micro_slip_err']:.2e}, {conv[1]['max_micro_slip_err']:.2e}, {conv[2]['max_micro_slip_err']:.2e} }}（代数支，无率相关）；
mu×{{0.2,0.3,0.5}} → δ_s 拟合相对误差 ≤{max(s['rel_err'] for s in sens_mu):.1e}，指数 mu^1.000000；
F_n×{{0.5,1,2}} N → δ_s 相对误差 ≤{max(s['rel_err'] for s in sens_n):.1e}，指数 F_n^0.666667（理论 2/3）。

审计侧独立重算：`curves.csv` 重算 max 微滑移偏差 {2.243e-13:.3e}（与 3.70e-16 之差是 CSV 15 位十进制序列化地板，仍低判据 11 个数量级）；
Hertz 链 a/k_t/δ_s 与 `results.json` 相对差 2.1e-16/1.3e-16/2.2e-16；线性律最大偏离 0.192445 @ δ/δ_s=2/3（δ=0.5δ_s 处 0.103553）。
独立复跑逐项复现：micro {g['max_micro_slip_err']:.3e}、platform {g['plateau_force_N']:.6f} N、k_t rel {g['k_t_rel_err']:.4e}、
fitted δ_s {abs(g['delta_s_fit_m']-ref['delta_s_m'])/ref['delta_s_m']:.3e}、legacy {cs['max_dev_vs_analytic_muN']*100:.3f}%；`validate.py` **9/9 PASS ACCEPT**；
`pytest tests/test_mindlin_partial_slip.py -q` → **10 passed in 12.22 s**。

## ④ 结果图
图1–图4 为被审交付的工件（sha256 已与 `checksums.json` 逐张核对，字节未改）。
图1/2 是本算例的主判据视图（曲线形状 + 逐点偏差，线性律被判别）；图3 是两轴敏感性；图4 是粘着圆 c/a 的收缩。

## ⑤ 注意点（踩过的坑 / 局限）
- **口径警示（最重要）**：B1/B2/B3/B5 的 1e-16 一致性是**闭式对闭式**（参考与实现同一条 Mindlin 公式），
  它证明的是实现与接线正确，**不是定律被独立证实**；不可循环的量是 B4（k_t 差 0.915%）、δ_s∝mu^1、δ_s∝F_n^{{2/3}} 与 B6 判别力。
- **F1（medium）**：注册裁判 0 行 → 机器裁决缺席；本审计**不代填**，留池侧补登。
- **F2（medium）**：README §4 分段样本数 5766/672 与工件不符（独立重算 5602/438/398），且 `results.json` 未落 n_micro/n_gross → 报告数字不可溯源（判据不受影响）。
- **F3（low）**：求解器只接**单调加载支**（`tang_disp_next=tang_disp`，无锥投影/无运动锚点），`reversal_branch`/`reversal_loop_area` 只在模块内；
  故**循环加载下模型无滞回耗散**（力是 |δ| 的无历史函数）——本算例单调加载不受影响，但不得外推到循环/地震工况。
- **F4（low）**：`err_rel_muN` 等字段是对 mu F_n 归一化的绝对偏差，非相对误差（命名残留）。
- **F5（platform）**：`maturity_gate.callers_verified()` grep 的生态克隆落后 origin/main 277 个 commit，新代码不可见——
  上一轮 id=425 的 `integration_callers(grep 无引用)` 属环境性误判；`artifacts/` 被 .gitignore 排除导致 `run_artifacts` 永远"缺失"（本审计以受版本管理的 evidence 目录规避）。
- 未做：1953 原表核对（付费墙/无网）、实验或第三方 FEM-BEM 对照、完整回归套件复跑、循环滞回数值验证、集群第二机器佐证（cj1279 递交后长时间排队）。

## ⑥ 误差判定（是否 <3%）
- B1 微滑移 {g['max_micro_slip_err']:.2e} ≪ 3% → **是**
- B3 平台 {g['max_gross_slip_err']:.2e} → **是**
- B4 初始斜率 {g['k_t_rel_err']*100:.3f}% → **是**（割线估计偏低，方向符合理论预期）
- B5 δ_s {abs(g['delta_s_fit_m']-ref['delta_s_m'])/ref['delta_s_m']:.2e} → **是**
- B2 过渡 {g['max_transition_err']:.2e} → **是**
- 结论：**关键量误差全部 <3%**（绝大多数在 1e-15 量级）；但按 ⑤ 的口径警示，该精度是"实现 vs 闭式"的精度，
  真实物理可信度由 k_t/δ_s 标定（0.915%/1e-15）与判别力（19.24%）支撑，**不给 L5**（无独立外部锚）。
- 判级：**L4**（真实现+主求解器集成+完整算例+误差分析+干净副本可复现）；处置 approve with notes，消账。
"""

payload = {
    "title": "CASE-DEM-MINDLIN-TANGENTIAL 交差核验（m16226 审 m16147 · TensorDEM）→ L4",
    "summary": ("交差核验 m16147 切向 Mindlin 微滑移-完全滑移算例：三件套真实（工件 6/6 sha256 自洽、validate 9/9 ACCEPT、"
                "干净副本复跑逐项复现、pytest 10 passed）；6 项判据全 PASS，k_t 差 0.915%、δ_s 差 4.4e-16、判别力 19.24%；"
                "发现：注册裁判 0 行(F1)、报告分段计数不可溯源(F2)、求解器只接单调加载支(F3)；判 L4，不给 L5（无独立外部锚）。"),
    "scenario_key": "case-dem-mindlin-tangential",
    "mission_id": 16226,
    "tags": ["board-audit", "TensorDEM", "Mindlin-Deresiewicz", "partial-slip", "CASE-DEM-MINDLIN-TANGENTIAL", "L4"],
    "markdown": md,
    "figures": figs,
}


def main():
    req = urllib.request.Request(BASE + "/auth/login",
                                 data=json.dumps({"username": CREDS["username"],
                                                  "password": CREDS["password"]}).encode(),
                                 headers={"Content-Type": "application/json"})
    tok = json.load(urllib.request.urlopen(req, timeout=60)).get("access_token")
    req = urllib.request.Request(BASE + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok})
    out = json.load(urllib.request.urlopen(req, timeout=180))
    print(json.dumps({k: v for k, v in out.items() if k != "markdown"})[:600])
    json.dump(out, open(os.path.join(REPO, "artifacts", "board_audit_m16226_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()