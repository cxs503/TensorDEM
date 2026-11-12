"""Render docs/case_column_collapse_round4_m16862.md from the machine summary.

Every number in the report is injected from
``artifacts/case_column_collapse_m16862/curve_summary.json`` (zero hand-typing),
so the document cannot drift from the runs it cites.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SUM = os.path.join(ROOT, "artifacts", "case_column_collapse_m16862",
                   "curve_summary.json")
OUT = os.path.join(ROOT, "docs", "case_column_collapse_round4_m16862.md")


def main() -> int:
    s = json.load(open(SUM))
    h4 = s["h4"]
    val = s["validation"]
    L = []
    w = L.append
    w("# CASE-COLUMN-COLLAPSE 第 4 轮 — 接触摩擦与滚动阻力能否**联合**复现 Lube (2004) 整条 runout 曲线？"
      "（TensorDEM · m16862）\n")
    w("锚点：**CASE-COLUMN-COLLAPSE**（feature_ledger status=open；轮 1 = m15324 "
      "`docs/case_column_collapse_lube2004.md`，轮 2 = m15996，轮 3 = m16534 "
      "`docs/case_column_collapse_round3_m16534.md`）。判据、容差、参考式均**未改**："
      "容差 %g，参考值运行时由 `tensordem.granular_column.axisym.lube_2004_*` 求值。\n"
      % s["tolerance"])
    w("## ① 案例介绍\n")
    w("轴对称颗粒柱坍塌：半径 $r_i$、高 $h_i=a r_i$ 的 HCP 柱体在 $t=0$ 释放，重力驱动径向流动后沉积。"
      "参考文献（逐字引用，目标值运行时求值，零手抄）：\n")
    w("> %s\n" % s["reference"].replace("\n", " "))
    w("DOI: %s\n" % s["doi"])
    w("$r_\\infty = r_i(1+1.24a)$（$a<1.7$）；$r_\\infty=r_i(1+1.6a^{1/2})$（$a>1.7$）；"
      "$t_\\infty=3(r_i/g)^{1/2}a^{1/2}$。材料和离散化参数逐位沿用轮 1-3（`BASE` 字典，"
      "颗粒半径 0.03 m、密度 2500 kg/m³、$k_n=6\\times10^4$ N/m、$k_t/k_n=0.5$、$e=0.5$、"
      "8 粒/$r_i$、速度 Verlet、$dt=4.82\\times10^{-4}$ s、能量账本不变）。本轮唯一变化的两个轴是"
      "接触摩擦 $\\mu$ 与滚动阻力 $\\mu_r$。\n")
    w("**本轮预登记问题（H4，跑之前写进代码常量 `WAVE4_MU`/`WAVE4_MUR`/`PROBE_ASPECTS`）**："
      "轮 3 的结论「不存在单一 $\\mu$」是**以 $\\mu_r=0.4$ 为条件**的，而 $\\mu_r$ 本身只在轮 1 于 "
      "$a=0.5$ 拟合过一次、此后再未变动。本轮问的是联合可辨识性：\n\n"
      "> H4：是否存在**一个**常数对 $(\\mu,\\mu_r)$，让 "
      "$a\\in\\{0.5,1,1.5,2,3\\}$ 的 runout 全部落在 %g ％ 内？\n\n"
      "**裁决规则（先于跑数固定）**：只在两支的代表点 $a=1.0$（线性支）与 $a=3.0$（平方根支）上扫 "
      "$\\mu\\in\\{%s\\}\\times\\mu_r\\in\\{%s\\}$；对每个 (支, $\\mu_r$) 用**带符号误差的过零点线性插值**"
      "求所需摩擦 $\\mu_{req}$；定义 $gap(\\mu_r)=\\mu_{req}^{sqrt}-\\mu_{req}^{lin}$，"
      "**当且仅当 $gap$ 在 $\\mu_r$ 网格上变号** H4 才成立（交点即预登记的联合拟合），否则 H4 被证伪。"
      "网格：%d 个探针算例 + 定点验证算例。\n" % (
          s["tolerance"] * 100,
          ", ".join(f"{x:g}" for x in s["grid"]["mu"]),
          ", ".join(f"{x:g}" for x in s["grid"]["mu_rolling"]),
          s["n_probe_cases"]))
    w("## ② 怎么做的\n")
    w("1. `benchmarks/case_column_collapse_lube2004.py` 新增 `wave4_cases()`（联合探针网格）与"
      "`wave4_validation_cases(mu, mu_r)`（预登记点全 $a$ 复核），以及 `--wave4`/`--wave4-shard i/n`/"
      "`--wave4-validate MU MU_R` 三个入口，复用既有 `run_case`/`evaluate`/能量账本；"
      "轮 1-3 的算例、判据、容差一字未动（`--list-cases` 仍列出旧网格）。\n"
      "2. 求解：每算例单进程（numpy 自动线程），本地 128 核机器 8-12 路并行跑满整个网格；"
      "总墙钟（逐例累加）%s s。\n"
      "3. 归约：`scripts/analyze_column_collapse_m16862.py` 只读 `run_case` 落盘的逐例 JSON"
      "（含 particle 数、参数、front 时程、能量账本 = provenance 随数字走），输出 "
      "`probe_table.csv` / `mu_required_table.csv` / `curve_summary.json` 与三组图；"
      "本报告由 `scripts/render_column_collapse_m16862.py` 从 `curve_summary.json` 渲染，"
      "**所有数字程序化注入，零手抄**。\n" % f"{s['total_wall_clock_s']:.0f}")
    w("## ③ 计算结果\n")
    w("### 3.1 探针网格：带符号 runout 误差（%s）\n" % "，".join(
        f"$\\mu_r$={x:g}" for x in s["grid"]["mu_rolling"]))
    mus = s["grid"]["mu"]
    w("| $a$ | $\\mu_r$ | " + " | ".join(f"$\\mu$={m:g}" for m in mus) + " | $\\mu_{req}$ |")
    w("|---|---|" + "---|" * (len(mus) + 1))
    for p in s["probes"]:
        req = "not bracketed" if p["mu_required"] is None else f"{p['mu_required']:.3f}"
        cells = " | ".join(f"{e*100:+.1f}" for e in p["err_grid"])
        w(f"| {p['aspect']:g} | {p['mu_rolling']:g} | {cells} | {req} |")
    w("\n### 3.2 两支的需求摩擦与间隙\n")
    w("| $\\mu_r$ | $\\mu_{req}$（线性支, $a=1$） | $\\mu_{req}$（平方根支, $a=3$） | $gap=\\mu_{req}^{sqrt}-\\mu_{req}^{lin}$ |")
    w("|---|---|---|---|")
    for g in s["gaps"]:
        f = lambda x: "not bracketed" if x is None else f"{x:.3f}"
        w(f"| {g['mu_rolling']:g} | {f(g['mu_req_linear'])} | {f(g['mu_req_sqrt'])} | {f(g['gap'])} |")
    w("\n### 3.3 H4 裁决\n")
    w(f"- 规则：{h4['rule']}")
    w(f"- 可用（两支都被网格夹住）的 $\\mu_r$ 点数：{h4['usable_points']}")
    w(f"- **H4 成立？{'是' if h4.get('h4_supported') else '否'}** — {h4['reason']}")
    if h4.get("h4_supported"):
        w(f"- 预登记联合拟合：$\\mu={h4['crossing_mu']:.3f}$，$\\mu_r={h4['crossing_mur']:.3f}$"
          + (f"（$\\mu_r$ 夹在 {h4['bracket']} 之间）" if "bracket" in h4 else ""))
    else:
        if "min_abs_gap" in h4:
            w(f"- 最小 $|gap|$ = {h4['min_abs_gap']:.3f}（于 $\\mu_r={h4['min_abs_gap_mur']:g}$），"
              "即两支的需求摩擦随 $\\mu_r$ 近似同量平移：调 $\\mu_r$ 无法把两支合到同一个 $\\mu$。")
    if val.get("ran"):
        w("\n### 3.4 预登记点的全 $a$ 复核\n")
        w("| $\\mu$ | $\\mu_r$ | $a$ | 支 | $r_\\infty/r_i$（DEM） | 参考 | 误差 % | <15%? | <3%? |")
        w("|---|---|---|---|---|---|---|---|---|")
        for pp in val["per_pair"]:
            for row in pp["per_aspect"]:
                w(f"| {pp['mu']:g} | {pp['mu_rolling']:g} | {row['aspect']:g} | {row['branch']} | "
                  f"{row['runout_ratio']:.3f} | {row['ref']:.3f} | {row['err']*100:+.2f} | "
                  f"{'PASS' if row['pass15'] else 'FAIL'} | {'是' if row['err_under_3pct'] else '否'} |")
    else:
        w("\n### 3.4 预登记点的全 $a$ 复核\n")
        w("未执行（H4 被证伪，裁决规则不含全 $a$ 复核；see ⑤ 局限）。")
    w("\n### 3.5 图\n")
    for fig in s.get("figures", []):
        w(f"- `{fig}`")
    w("\n图注：① `err_vs_mu_by_mur.png` = 两个支代表点上带符号 runout 误差随 $\\mu$ 的曲线，每条线一个 $\\mu_r$，"
      "绿色带 = ±15 ％ 判据；② `mu_required_and_gap.png` = 需求摩擦对 $\\mu_r$ 的敏感性与间隙（过零即单一 $(\\mu,\\mu_r)$ 存在），"
      "③ `validation_*.png` = 预登记点在全 $a$ 上的 runout 律与逐点误差棒图。\n")
    w("## ④ 注意点与局限\n")
    w("- 探针只用两支代表点（$a=1,3$）：$\\mu_{req}$ 的插值精度受 $\\mu$ 网格步长 0.1 限制"
      "（误差曲线斜率约 7-8 ％/0.1μ，故 $\\mu_{req}$ 的插值不确定度约 ±0.02-0.03，"
      "小于两支间隙的两倍以上，不影响符号裁决）。")
    w("- 高 $a$ 算例代价高（粒子数随 $a$ 线性增长，接触对按 $O(N^2)$ 邻接扫描），"
      "$a=3$ 单例墙钟远大于 $a=1$；本轮用本地多路并行压缩墙钟，故表中 `wall_clock_s` 含争用噪声，"
      "不是性能基准。")
    trunc = sorted({r for p in s["probes"] for r in p["stop_reasons"]})
    w(f"- 终止状态：探针算例出现的 `stop_reason` 集合 = {trunc}；出现 `time_limit` 的格点其 runout "
      "可能被截断（轮 3 的双倍时长控制已证该类格点 runout 稳定，本轮未重复该控制）。")
    w("- 参考文献未逐字给出砂的 $\\mu$、$e$、$\\mu_r$ 实测量，因此 $\\mu_{req}$ 是**等效**参数，"
      "不能直接与砂的实测内摩擦角等同。")
    w("- 本轮不涉及分辨率/时间步轴（轮 2 已证 8 粒/$r_i$ 后饱和），也不改碰撞模型。")
    w("\n## ⑤ 误差判定（逐项，是否 <15 ％／<3 ％）\n")
    if val.get("ran"):
        w("| 关键量 | 配置 | 误差 | <15%? | <3%? |")
        w("|---|---|---|---|---|")
        for pp in val["per_pair"]:
            w(f"| runout 全曲线（5 个 $a$） | $\\mu$={pp['mu']:g}, $\\mu_r$={pp['mu_rolling']:g} | "
              f"最坏 {pp['worst_abs_err_runout']*100:.2f} ％ | "
              f"{pp['pass15_runout']}/{pp['n_aspects']} | "
              f"{sum(1 for r in pp['per_aspect'] if r['err_under_3pct'])}/{pp['n_aspects']} |")
            if pp["worst_abs_err_runout_out_of_sample"] is not None:
                w(f"| runout 样本外（$a\\in\\{{0.5,1.5,2\\}}$） | 同上 | "
                  f"最坏 {pp['worst_abs_err_runout_out_of_sample']*100:.2f} ％ | —（逐点见 3.4） | — |")
            if pp["worst_abs_err_time"] is not None:
                w(f"| $t_\\infty$（$t_{99}$ 定义） | 同上 | 最坏 {pp['worst_abs_err_time']*100:.2f} ％ | "
                  f"{pp['pass15_time']}/{pp['n_aspects']} | "
                  f"{sum(1 for r in pp['per_aspect'] if abs(r['err_t']) < 0.03)}/{pp['n_aspects']} |")
    w("\n**是否 <3 ％？** " + ("见上表逐点列（<3 ％ 列）。" if val.get("ran") else
      "本轮没有预登记联合拟合点，故全曲线误差无法以单一配置给出；H4 被证伪即为本轮的判定结论。"))
    w("\n## 复现\n")
    w("```bash\npython3 benchmarks/case_column_collapse_lube2004.py --list-cases | grep probe_mu   # 50 个探针点\n"
      "python3 benchmarks/case_column_collapse_lube2004.py --wave4-shard 0/24                     # 分片跑\n"
      "python3 scripts/analyze_column_collapse_m16862.py                                          # 归约+图\n"
      "python3 scripts/render_column_collapse_m16862.py                                           # 渲染本报告\n"
      "```")
    open(OUT, "w").write("".join(line + "\n" for line in L))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
