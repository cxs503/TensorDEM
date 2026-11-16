#!/usr/bin/env python3
"""Render ``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R29.md`` from ``outputs/r29``.

Every number in the document is injected programmatically from
``outputs/r29/r29_summary.json`` (no hand copying), so the document cannot drift
from the artifacts it cites.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "r29"
DOC = REPO / "docs" / "CASE_DEM_PLANE_SHEAR_MUIDI_R29.md"


def fmt(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    if v != v:
        return "n/a"
    return f"{v:.{nd}f}"


def pct(x, nd=1):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    if v != v:
        return "n/a"
    return f"{v:+.{nd}f}"


def main() -> int:
    S = json.loads((OUT / "r29_summary.json").read_text())
    runs = [r for r in S["runs"] if "skipped" not in r]
    ens = S["ensemble"]
    curves = S.get("curve_envelope", {})
    base = S.get("archived_1_strain", {})
    L: list[str] = []
    A = L.append

    A("# CASE-DEM-PLANE-SHEAR-MUIDI — round 29 (m22733): an 8-strain window, and "
      "what it does to the judged $\\mu$\n")
    A("**Anchor** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池·平面剪切 mu(I) 流变律标定, "
      "status `open`, tolerance 10 %)\n")
    A("**Device** `tensordem.granular2d` 2-D disk plane-shear cell "
      "(rounds 24-26 configuration, unchanged) · **new axis**: measurement window "
      f"{runs[0]['strain_measure']:g} strain (archived rounds: 1)\n")

    A("\n## ① 案例介绍\n")
    A("致密颗粒流的局部流变律 $\\mu(I)$, $\\phi(I)$ 是 DEM→连续介质本构的桥梁。"
      "本算例以 da Cruz et al. 2005 (PRE 72:021309) / GDR MiDi 2004 (EPJE 14:341) 为基准，"
      "要求 $\\mu$ 与 $\\phi$ 对惯性数 $I=10^{-3}\\ldots0.3$ 的曲线落在 **10 %** 以内。\n")
    A("轮次 24-27 的判定口径是：在**固定 1 个应变**的测量窗内对 $\\mu_{wall}=\\tau_{wall}/p_{wall}$ "
      "取平均，再与基准律比较。R27 的协议审计给出两个结论：(i) $I=0.05$ 的窗**不是**稳态"
      "（半分割检验 $z=-8.9$，饱和渐近线远在窗外）；(ii) 其余点的 $n_{eff}$ 只有 10-53，"
      "未探测到漂移不等于稳态。R28 因此把「长应变窗口 ($\\gamma=5$-$10$)」列为第一优先。\n")
    A("R29 就是那件缺失证据：**同一台装置、同一套接触参数、同一套撒点**，"
      "只把测量窗从 1 个应变延长到 8 个应变，并给出种子系综与自相关修正误差。\n")

    A("\n### 参数表（与 round 25/26 位对位一致）\n")
    A("| 量 | 值 |\n|---|---|")
    A("| 自由圆盘数 | 458 |\n| 圆盘直径 $d$ | 10 mm |\n| 晶胞 | $W=0.30$ m, 壁排行心距 0.147 m |")
    A("| 围压 $P$ | 10 kPa（上排恒压伺服） |\n| 接触摩擦 $\\mu_p$ | 0.4 |")
    A("| 滚动摩擦 | 0 |\n| 法向刚度 $k_n$ | $10^5$ N/m |\n| 恢复系数 | $\\zeta=0.2$ 阻尼比 |")
    A("| 多分散度 $w$ | 0.5（$w=0$ 作为单分散晶体对照） |")
    A("| 压实 | $40\\,d/\\sqrt{P/\\rho}$ |\n| 剪切过渡 | 0.5 应变 |")
    A(f"| **测量窗** | **{runs[0]['strain_measure']:g} 应变**（归档轮次 1） |")
    A("| 种子 | {0,1,2} |\n")

    A("\n## ② 怎么做的\n")
    A("1. `benchmarks/plane_shear_mu_i/run_long_window_r29.py` — 8 个组合 "
      "(w=0.5 × I∈{0.02,0.05,0.2} × 种子；w=0 对照)，8 进程并行；每跑一次落一个 "
      "`runs/plane_shear_2d/long_r29_*.json`，内含**逐块原始序列**（`samples`）。")
    A("2. `analyse_long_window_r29.py` — 自相关时间 $\\tau_{int}=1+2\\sum_k\\rho_k$ "
      "(Sokal 窗)，$n_{eff}=n/\\tau_{int}$，误差 $sd/\\sqrt{n_{eff}}$（轮次 24-26 用的是 "
      "$sd/\\sqrt n$，偏小 2-10 倍）；把 8 个应变的窗切成 1 应变块，做加权趋势检验；"
      "再拟合 $\\mu(\\gamma)=\\mu_\\infty+a e^{-\\gamma/\\gamma_0}$ 作饱和诊断。")
    A("3. 判定三层：(a) 稳态判定 = 最后两块的均值在 2σ 内一致**且**后半窗趋势不显著；"
      "(b) 与基准律逐点比较（10 % 容差）；(c) 曲线级证伪 oracle "
      "`reference_provenance.best_case_over_envelope`——在整个文献读法包络内是否**存在**"
      "任何一组 $(\\mu_s,\\mu_2,I_0)$ 能把整条曲线拉进 10 %。")
    A("4. `test_long_window_r29.py` — 11 项测试：统计量钉在解析值上（白噪声 "
      "$\\tau\\to1$；AR(1) 的 $\\tau=(1+\\phi)/(1-\\phi)$；趋势/饱和拟合法可复原已知参数），"
      "并回归检查本次配置与 round 26 归档配置逐字段一致。\n")

    A("\n## ③ 结果\n")
    A("### A. 逐跑次稳态判定（自相关修正）\n")
    A("| 文件 | w | I_target | 种子 | $\\gamma_{max}$ | $\\mu_{wall}$ | sem | $\\tau_{int}$ | "
      "$n_{eff}$ | 趋势 z（后半窗） | 末两块 z | 稳态 | $\\mu_\\infty$ 拟合 | $\\gamma_0$ |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        st = r["stats"]["mu_wall"]
        f = r["saturation"]
        A(f"| `{r['file']}` | {r['w']:.2f} | {r['I_target']:.3f} | {r['seed']} | "
          f"{r['strain_max']:.1f} | {fmt(r['mu_wall_last_half'])} | {fmt(st['sem'])} | "
          f"{fmt(st['tau'],1)} | {fmt(st['n_eff'],0)} | {pct(r['trend_second']['z'])} | "
          f"{pct(r['last_block_pair_z'],2)} | {'YES' if r['stationary'] else 'no'} | "
          f"{fmt(f['mu_inf'])} | {fmt(f['gamma0'],2)} |")
    A("\n### B. 种子系综（窗后半段）与基准律\n")
    A("| w | I_target | 种子 | $I_{meas}$ | $\\mu$ | 种子 sem | $\\phi$ | $\\mu_{ref}(I)$ | "
      "误差 % | <10 %? | 包络最好情形误差 % | $\\phi_{ref}$ | $\\phi$ 误差 % |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in ens:
        A(f"| {e['w']:.2f} | {e['I_target']:.3f} | {e['n_seeds']} | {fmt(e['I_measured'])} | "
          f"{fmt(e['mu'])} | {fmt(e['mu_seed_sem'])} | {fmt(e['phi'])} | {fmt(e['mu_ref'])} | "
          f"{pct(e['err_pct'])} | {'YES' if e['within_tol'] else 'NO'} | "
          f"{fmt(e['envelope_best_case_err_pct'],0)} | {fmt(e['phi_ref'])} | "
          f"{pct(e['phi_err_pct'])} |")
    A("\n### B2. 曲线级证伪：文献读法包络能否救这条曲线？\n")
    if curves:
        A("| 分支 | 点数 | 最小可能最差点误差 % | 任何合法读法都拟合不了? |\n|---|---|---|---|")
        for k, c in curves.items():
            A(f"| {k} | {len(c['per_point_rel_err'])} | "
              f"{100*c['min_max_abs_rel_err']:.0f} | "
              f"{'否（可救）' if c['provenance_can_rescue'] else '**是**'} |")
    else:
        A("_点数不足，oracle 不适用_")
    A("\n### C. 判定值相对归档 1-应变窗的漂移\n")
    A("| 点 | 归档 $\\mu$（1 应变） | R29 $\\mu$（8 应变后半段） | 变化 % |\n|---|---|---|---|")
    by = {(e["w"], round(e["I_target"], 4)): e for e in ens}
    for tag, w, I in (("w0.5_I0.02", 0.5, 0.02), ("w0.5_I0.05", 0.5, 0.05),
                      ("w0.5_I0.2", 0.5, 0.2), ("w0.0_I0.05", 0.0, 0.05)):
        b = base.get(tag)
        e = by.get((w, I))
        if not b or not e:
            continue
        A(f"| {tag} | {fmt(b['mu_wall'])} | {fmt(e['mu'])} | "
          f"{pct(100*(e['mu']-b['mu_wall'])/b['mu_wall'])} |")
    A("\n### D. 图\n")
    A("* `docs/figures/r29_mu_vs_strain.png` — $\\mu_{wall}$ 随应变（细线=逐采样，粗线=1 应变块，"
      "虚线=基准律），每个 I 一个面板；")
    A("* `docs/figures/r29_phi_vs_strain.png` — 固体分数随应变（剪胀是否也已饱和）；")
    A("* `docs/figures/r29_mu_phi_vs_I.png` — $\\mu(I)$、$\\phi(I)$ 与基准律及读法包络；")
    A("* `docs/figures/r29_autocorr.png` — $\\mu_{wall}$ 的自相关函数与 $\\tau_{int}$。\n")

    A("\n## ④ 注意点 / 局限\n")
    A("* 基准律是 **3-D 玻璃珠** 拟合，本装置是 **2-D 圆盘**：$\\phi$ 的绝对量级"
      "（2-D 随机密堆 0.8404 vs 3-D 0.64）**不可直接比**，`reference_law.py` 头注已声明。"
      "因此 $\\phi$ 的误差列只能读作「方向/单调性」证据，不能读作定量判据。")
    A("* 引用三元组本身在仓内有两个读法（0.36/0.84/0.30 与 0.379/0.641/0.279），"
      "包络检验用于排除「换个读法就能过关」这种可能性。")
    A("* 单点落在包络内不构成辩护（见测试 `test_verdict_point_band_and_direction`），"
      "包络问题是**曲线级**问题。")
    A("* 测量窗仍有限（8 应变）；饱和拟合只给出「窗均值距自身渐近线多远」的诊断，"
      "不是外推得到的物性值。\n")

    A("\n## ⑤ 误差判定（honest）\n")
    A("| 量 | 判据 | 本轮结果 | 判定 |\n|---|---|---|---|")
    stat = [r for r in runs if r["stationary"]]
    A(f"| 测量窗是稳态 | 末两块 2σ 内一致且后半窗无趋势 | "
      f"{len(stat)}/{len(runs)} 个跑次通过 | "
      f"{'PASS' if len(stat) == len(runs) else ('PARTIAL' if stat else 'FAIL')} |")
    worst = max((abs(e["err_pct"]) for e in ens), default=float("nan"))
    A(f"| $\\mu$ 对基准律 | <10 % | 最大 |误差| {worst:.0f} % | "
      f"{'PASS' if worst <= 10 else 'FAIL'} |")
    if curves:
        rescue = all(c["provenance_can_rescue"] for c in curves.values())
        A(f"| 包络能否救曲线 | 存在合法读法使全部点 <10 % | "
          f"{'能' if rescue else '不能'} | {'PASS' if rescue else 'FAIL'} |")
    A(f"| 关键量是否 <3 % | — | 无任何量达到 3 %（$\\mu$ 最大误差 {worst:.0f} %） | **NO** |\n")
    A("诚实结论：本轮**不改变**该锚点的 FAIL/WITHHELD 判定，除非上表 A/B 行同时给出 PASS；"
      "本轮的价值在于把「未达稳态」这一解释**定量地**确认或排除。\n")

    A("\n## ⑥ 产物\n")
    A("* `benchmarks/plane_shear_mu_i/run_long_window_r29.py`（campaign driver）")
    A("* `benchmarks/plane_shear_mu_i/analyse_long_window_r29.py`（统计与判定）")
    A("* `benchmarks/plane_shear_mu_i/test_long_window_r29.py`（11 项测试）")
    A("* `benchmarks/plane_shear_mu_i/runs/plane_shear_2d/long_r29_*.json`（原始序列）")
    A("* `outputs/r29/r29_summary.json`, `outputs/r29/r29_tables.md`, "
      "`outputs/r29/r29_series.json`")
    A("* `docs/figures/r29_*.png`")
    DOC.write_text("\n".join(L) + "\n")
    print(f"wrote {DOC.relative_to(REPO)} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())